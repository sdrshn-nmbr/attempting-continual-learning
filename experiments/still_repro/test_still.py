import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout
from still import StillCompactor, continue_from, prefill, streaming_pairs, support_kl


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    model = Qwen3ForCausalLM(config).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def teacher_student_logits(model, pairs, compact, continuation, logical):
    with torch.no_grad():
        teacher = continue_from(model, pairs, continuation, logical).logits
    student = continue_from(model, compact, continuation, logical).logits
    return teacher, student


def test_cached_continuation_matches_uninterrupted_forward():
    model = tiny_model()
    ids = torch.randint(0, 97, (2, 47))
    prefix, continuation = ids[:, :40], ids[:, 40:]
    with torch.no_grad():
        whole = model(input_ids=ids).logits[:, 40:]
        cached = continue_from(model, prefill(model, prefix), continuation, 40).logits
    assert torch.allclose(whole, cached, atol=1e-4)


def test_identity_initialization_passes_cache_through_at_full_length():
    model = tiny_model()
    ids = torch.randint(0, 97, (1, 48))
    prefix, continuation = ids[:, :40], ids[:, 40:]
    pairs = prefill(model, prefix)
    compactor = StillCompactor(model.config, slots=40)
    with torch.no_grad():
        compact = compactor(model, pairs)
        full = continue_from(model, pairs, continuation, 40).logits
        through = continue_from(model, compact, continuation, 40).logits
        empty = model(input_ids=continuation).logits
    for (keys, values), (ck, cv) in zip(pairs, compact):
        assert torch.nn.functional.cosine_similarity(keys.flatten(), ck.flatten(), dim=0) > 0.9
        assert torch.nn.functional.cosine_similarity(values.flatten(), cv.flatten(), dim=0) > 0.9
    assert (through - full).abs().mean() < 0.25 * (empty - full).abs().mean()


def test_gradient_reaches_compactor_and_not_base_model():
    model = tiny_model()
    ids = torch.randint(0, 97, (2, 72))
    prefix, continuation = ids[:, :64], ids[:, 64:]
    pairs = prefill(model, prefix)
    compactor = StillCompactor(model.config, slots=8)
    teacher, student = teacher_student_logits(model, pairs, compactor(model, pairs), continuation, 64)
    loss = support_kl(teacher[:, :-1].reshape(-1, 97), student[:, :-1].reshape(-1, 97),
                      continuation[:, 1:].reshape(-1), top_k=20).mean()
    loss.backward()
    grads = [p.grad for p in compactor.parameters() if p.grad is not None]
    assert grads and sum(g.norm() for g in grads) > 0
    assert all(p.grad is None for p in model.parameters())


def test_support_kl_is_zero_for_identical_and_includes_gold():
    logits = torch.randn(5, 97)
    gold = torch.full((5,), 3)
    assert support_kl(logits, logits.clone(), gold, top_k=10).abs().max() < 1e-5
    shifted = logits.clone()
    shifted[:, 3] -= 20
    teacher = logits.clone()
    teacher[:, 3] = logits.min(dim=-1).values - 50
    assert (support_kl(logits, shifted, gold, top_k=10) > 0).all()
    assert torch.isfinite(support_kl(teacher, shifted, gold, top_k=10)).all()


def test_streaming_keeps_sinks_and_most_recent():
    keys = torch.arange(20.0).view(1, 1, 20, 1)
    kept, _ = streaming_pairs([(keys, keys)], budget=8)[0]
    assert kept.flatten().tolist() == [0, 1, 2, 3, 16, 17, 18, 19]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")


def test_prefix_is_exactly_budget_and_header_uses_special_tokens(tokenizer):
    header = layout.header_ids(tokenizer)
    assert header[0] == tokenizer.convert_tokens_to_ids("<|im_start|>")
    document = list(range(1000, 1000 + 9000))
    assert len(layout.prefix_ids(tokenizer, document)) == layout.PREFIX_TOKENS
    with pytest.raises(ValueError, match="DOCUMENT_TOO_SHORT"):
        layout.prefix_ids(tokenizer, document[:100])
    tail = layout.encode(tokenizer, layout.question_text(
        {"question": "Q?", "options": ["w", "x", "y", "z"]}))
    assert tail[-3:] == layout.encode(tokenizer, "<|im_start|>assistant\n")


def test_answer_letters_are_balanced_and_parsed():
    letters = Counter(layout.arrange_options("right", ["a", "b", "c"], index, seed=7)[1] for index in range(400))
    assert set(letters.values()) == {100}
    options, letter = layout.arrange_options("right", ["a", "b", "c"], 6, seed=7)
    assert options["ABCD".index(letter)] == "right"
    assert layout.parse_letter("The filing says X.\nAnswer: C") == "C"
    assert layout.parse_letter("Answer: **B**") == "B"
    assert layout.parse_letter("I think A. Answer: (D)") == "D"
    assert layout.parse_letter("no final line") is None


def test_batched_left_padded_generation_matches_single_item_generation(tokenizer):
    from evaluate import generate
    torch.manual_seed(1)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    model = Qwen3ForCausalLM(config).float().eval()
    items = [{"question": "Which year?", "options": ["1990", "1991", "1992", "1993"]},
             {"question": "Which company filed the longest annual report in the sample?",
              "options": ["Acme Corp", "Beta Industries", "Gamma Holdings", "Delta Partners"]}]
    prefix = torch.randint(0, len(tokenizer), (2, 30))
    batched = generate(model, tokenizer, prefill(model, prefix), 30, items, torch.device("cpu"), 6)
    single = [generate(model, tokenizer, prefill(model, prefix[i:i + 1]), 30, [items[i]], torch.device("cpu"), 6)[0]
              for i in range(2)]
    assert batched == single


def test_training_targets_align_logits_with_next_token():
    from train import collate
    examples = [([1] * 5, [10, 11, 12], [20, 21]), ([2] * 5, [13], [22, 23, 24])]
    prefixes, ids, (rows, columns), gold = collate(examples, 0, torch.device("cpu"))
    assert prefixes.shape == (2, 5)
    assert torch.equal(ids[rows, columns + 1], gold)
    assert gold.tolist() == [20, 21, 22, 23, 24]
