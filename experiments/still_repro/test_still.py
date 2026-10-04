import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout
from still import BudgetedStillCompactor, StillCompactor, continue_from, prefill, streaming_pairs, support_kl


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


def test_written_questions_are_validated():
    from write_questions import extract_json, validate
    good = {"type": "connect", "sections": [2, 6], "question": "Which firm paid Acme in 1998?",
            "correct": "Beta Corp", "distractors": ["Gamma LLC", "Delta Inc", "Omega Co"]}
    assert validate(good)["sections"] == [2, 6]
    assert validate({**good, "sections": [2, 3]}) is None
    assert validate({**good, "question": "What did S4 say Acme paid?"}) is None
    assert validate({**good, "question": "What does the document say Acme paid?"}) is None
    assert validate({**good, "distractors": ["Beta Corp", "Delta Inc", "Omega Co"]}) is None
    assert validate({**good, "type": "detail", "sections": [9]}) is None
    assert extract_json('```json\n{"questions": []}\n```') == {"questions": []}
    assert extract_json("no json here") is None


def test_candidate_items_balance_letters_and_keep_provenance(tmp_path):
    import json
    from questions import balance, candidate_items
    rows = [{"domain": domain, "row": row, "writer": "gpt-6-luna",
             "questions": [{"type": "detail", "sections": [1], "question": f"Q{row}-{n}?", "correct": "yes",
                            "distractors": ["a", "b", "c"]} for n in range(5)]}
            for domain in ["financial", "gutenberg", "legal", "code"] for row in range(8)]
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items = candidate_items(path, "eval", 17)
    assert len(items) == 160 and len({i["id"] for i in items}) == 160
    assert all(i["options"]["ABCD".index(i["gold"])] == "yes" and i["writer"] == "gpt-6-luna" for i in items)
    chosen = balance(items, 20)
    assert len(chosen) == 80
    assert Counter(i["gold"] for i in chosen) == Counter({letter: 20 for letter in "ABCD"})


def test_bias_attention_matches_sdpa_with_and_without_cache():
    import attention_matching
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    reference = Qwen3ForCausalLM(config).float().eval()
    custom = Qwen3ForCausalLM(config).float().eval()
    custom.load_state_dict(reference.state_dict())
    reference.set_attn_implementation("sdpa")
    custom.set_attn_implementation(attention_matching.NAME)
    ids = torch.randint(0, 97, (2, 30))
    with torch.no_grad():
        assert torch.allclose(reference(input_ids=ids).logits, custom(input_ids=ids).logits, atol=1e-5)
        base = continue_from(reference, prefill(reference, ids[:, :20]), ids[:, 20:], 20).logits
        mine = continue_from(custom, prefill(custom, ids[:, :20]), ids[:, 20:], 20).logits
    assert torch.allclose(base, mine, atol=1e-5)


def test_bias_is_added_to_compact_prefix_columns_only():
    import attention_matching

    class Module:
        layer_idx = 0

    torch.manual_seed(0)
    query, key, value = torch.randn(1, 2, 3, 8), torch.randn(1, 1, 7, 8), torch.randn(1, 1, 7, 8)
    beta = torch.tensor([[[0.5, -1.0]]])
    attention_matching.STATE.bias = [beta]
    try:
        output, _ = attention_matching.attention(Module(), query, key, value, None, scaling=0.3)
    finally:
        attention_matching.STATE.bias = None
    logits = query @ key.transpose(-1, -2) * 0.3
    logits[..., :2] += beta.view(1, 1, 1, 2)
    logits = logits + attention_matching.causal_bias(3, 7, logits.device, logits.dtype)
    expected = torch.softmax(logits, -1) @ value
    assert torch.allclose(output.transpose(1, 2), expected, atol=1e-5)


def test_saved_official_caches_batch_with_padding_and_reproduce_full_cache(tmp_path):
    import attention_matching
    import cache_format
    model = tiny_model()
    model.set_attn_implementation(attention_matching.NAME)
    ids = torch.randint(0, 97, (2, 26))
    prefix, continuation = ids[:, :20], ids[:, 20:]
    full = prefill(model, prefix)
    for row in range(2):
        layers = []
        for keys, values in full:
            keys, values, beta = keys[row], values[row], torch.zeros(keys.shape[1:3])
            if row == 1:
                keys = torch.cat((keys, torch.randn(keys.shape[0], 3, keys.shape[-1])), dim=1)
                values = torch.cat((values, torch.randn(values.shape[0], 3, values.shape[-1])), dim=1)
                beta = torch.cat((beta, torch.full((beta.shape[0], 3), float("-inf"))), dim=1)
            layers.append((keys, beta, values))
        torch.save(cache_format.pack(layers), attention_matching.cache_path(tmp_path, "code", row))
    compacted = [attention_matching.load(tmp_path, "code", row, "cpu") for row in (0, 1)]
    assert all(keys.shape[2] == 20 for keys, _, _ in compacted)
    keys, beta, values = compacted[1]
    compacted[1] = (torch.cat((keys, torch.randn(*keys.shape[:2], 3, keys.shape[-1])), dim=2), torch.cat(
        (beta, torch.full((*beta.shape[:2], 3), float("-inf"))), dim=2),
        torch.cat((values, torch.randn(*values.shape[:2], 3, values.shape[-1])), dim=2))
    pairs, betas = attention_matching.batch(compacted, torch.float32)
    assert pairs[0][0].shape[2] == 23 and torch.isinf(betas[0][0, :, 20:]).all()
    attention_matching.STATE.bias = betas
    try:
        with torch.no_grad():
            loaded = continue_from(model, pairs, continuation, 20).logits
    finally:
        attention_matching.STATE.bias = None
    with torch.no_grad():
        expected = continue_from(model, full, continuation, 20).logits
    assert torch.allclose(loaded, expected, atol=1e-5)
    with pytest.raises(FileNotFoundError, match="AM_CACHE_MISSING"):
        attention_matching.load(tmp_path, "code", 5, "cpu")


def budgeted_from(still, counts, config):
    budgeted = BudgetedStillCompactor(config, counts)
    state = {}
    for name, tensor in still.state_dict().items():
        if name.endswith(".latents"):
            for head in range(tensor.shape[0]):
                state[f"{name}.{head}"] = tensor[head]
        else:
            state[name] = tensor
    budgeted.load_state_dict(state)
    return budgeted


def test_still_compactor_keeps_its_parameter_names_and_initialization():
    model = tiny_model()
    torch.manual_seed(5)
    compactor = StillCompactor(model.config, slots=12)
    names = sorted(compactor.state_dict())
    assert names[:4] == ["layers.0.blocks.0.key.bias", "layers.0.blocks.0.key.weight", "layers.0.blocks.0.output.weight",
                         "layers.0.blocks.0.query.bias"]
    assert "layers.0.key_output.weight" in names and "layers.1.value_output.weight" in names
    torch.manual_seed(5)
    again = StillCompactor(model.config, slots=12)
    assert all(torch.equal(a, b) for a, b in zip(compactor.state_dict().values(), again.state_dict().values()))


def test_budgeted_compactor_with_equal_counts_computes_still_compactor():
    model = tiny_model()
    torch.manual_seed(3)
    still = StillCompactor(model.config, slots=12)
    for parameter in still.parameters():
        parameter.data.add_(0.01 * torch.randn_like(parameter))
    counts = [[12] * model.config.num_key_value_heads for _ in range(model.config.num_hidden_layers)]
    budgeted = budgeted_from(still, counts, model.config)
    pairs = prefill(model, torch.randint(0, 97, (2, 40)))
    with torch.no_grad():
        expected = still(model, pairs)
        output, betas = budgeted(model, pairs)
    for (keys, values), (budget_keys, budget_values), beta in zip(expected, output, betas, strict=True):
        assert torch.allclose(keys, budget_keys, atol=1e-5) and torch.allclose(values, budget_values, atol=1e-5)
        assert beta.shape == (2, model.config.num_key_value_heads, 12) and (beta == 0).all()


def test_budgeted_padding_receives_no_attention_and_every_head_learns():
    import attention_matching
    model = tiny_model()
    model.set_attn_implementation(attention_matching.NAME)
    torch.manual_seed(4)
    counts = [[5, 17], [9, 3]]
    compactor = BudgetedStillCompactor(model.config, counts)
    assert compactor.width == 17 and compactor.mean_slots() == 8.5
    ids = torch.randint(0, 97, (2, 46))
    prefix, continuation = ids[:, :40], ids[:, 40:]
    output, betas = compactor(model, prefill(model, prefix))
    for layer, row in enumerate(counts):
        for head, count in enumerate(row):
            assert (betas[layer][:, head, :count] == 0).all() and torch.isinf(betas[layer][:, head, count:]).all()
            assert (output[layer][0][:, head, count:] == 0).all()
    garbage = [(keys + torch.randn_like(keys) * torch.isinf(beta)[..., None],
                values + torch.randn_like(values) * torch.isinf(beta)[..., None])
               for (keys, values), beta in zip(output, betas, strict=True)]
    attention_matching.STATE.bias = betas
    try:
        logits = continue_from(model, output, continuation, 40).logits
        with torch.no_grad():
            perturbed = continue_from(model, garbage, continuation, 40).logits
    finally:
        attention_matching.STATE.bias = None
    assert torch.allclose(logits.detach(), perturbed, atol=1e-5)
    logits.square().mean().backward()
    for layer in compactor.layers:
        assert all(latents.grad is not None and latents.grad.abs().sum() > 0 for latents in layer.latents)


def test_training_loss_with_equal_budgets_matches_still_and_clears_the_bias():
    import attention_matching
    import train
    model = tiny_model()
    model.set_attn_implementation(attention_matching.NAME)
    torch.manual_seed(6)
    still = StillCompactor(model.config, slots=12)
    for parameter in still.parameters():
        parameter.data.add_(0.01 * torch.randn_like(parameter))
    budgeted = budgeted_from(still, [[12, 12], [12, 12]], model.config)
    ids = torch.randint(0, 97, (2, 8))
    batch = (torch.randint(0, 97, (2, 40)), ids, (torch.tensor([0, 0, 1]), torch.tensor([3, 4, 6])),
             torch.tensor([ids[0, 4].item(), ids[0, 5].item(), ids[1, 7].item()]))
    expected = train.step_loss(model, still, batch, top_k=5)
    assert torch.allclose(train.step_loss(model, budgeted, batch, top_k=5), expected, atol=1e-5)
    uneven = train.step_loss(model, BudgetedStillCompactor(model.config, [[5, 17], [9, 3]]), batch, top_k=5)
    assert torch.isfinite(uneven) and attention_matching.STATE.bias is None
