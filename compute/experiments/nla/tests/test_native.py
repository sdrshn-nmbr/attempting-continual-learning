from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import GenerationConfig, Qwen2Model

from native import (
    Sidecar,
    chat_query,
    completion_ids,
    extract_activation,
    generation_diagnostics,
    inject_activation,
    injection_position,
    norm_matched_random,
    reconstruction_metrics,
    score_behavior,
)
from protocol import Record, make_calibration, make_records


def test_native_sidecar_tokenization_and_ar_suffix(native_paths, native_tokenizers):
    av = Sidecar.load(native_paths["av"], "av")
    ar = Sidecar.load(native_paths["ar"], "ar")
    tokenizer = native_tokenizers["av"]
    assert (av.layer, av.width) == (20, 3584)
    assert av.injection_scale == 150
    assert tokenizer.encode(av.injection_char, add_special_tokens=False) == [av.injection_token_id]
    content = av.av_template.format(injection_char=av.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, return_dict=False, add_generation_prompt=True
    )
    position = injection_position(ids, av)
    assert tokenizer.decode([ids[position]]) == av.injection_char
    for explanation in ("A blue object.", "A red entity named Talin0."):
        ids = native_tokenizers["ar"](ar.ar_template.format(explanation=explanation), add_special_tokens=True)[
            "input_ids"
        ]
        assert tuple(ids[-len(ar.suffix_ids) :]) == ar.suffix_ids


def test_native_queries_stop_before_answer_and_have_stable_prefixes(native_tokenizers, config):
    tokenizer = native_tokenizers["base"]
    for record in make_records(config):
        query = chat_query(tokenizer, record.query, config["max_sequence_length"])
        replacement = replace(record, answer="injected gold must not affect source")
        assert query == chat_query(tokenizer, replacement.query, config["max_sequence_length"])
        assert query["source_token_index"] < len(query["input_ids"]) - 1
        for candidate in record.candidates:
            full, suffix = completion_ids(tokenizer, query, candidate, config["max_sequence_length"])
            assert full[: len(query["input_ids"])] == query["input_ids"]
            assert len(suffix) > 0
    for item in make_calibration(config):
        chat_query(tokenizer, item["content"], config["max_sequence_length"])


def test_injection_changes_one_embedding_and_matches_released_norm():
    embeddings = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    vector = torch.randn(32)
    result = inject_activation(embeddings, vector, 4, 150.0)
    torch.testing.assert_close(result[:, :4], embeddings[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 5:], embeddings[:, 5:], rtol=0, atol=0)
    assert abs(result[0, 4].float().norm().item() - 150) < 0.5
    assert result.data_ptr() != embeddings.data_ptr()
    empty = inject_activation(embeddings, torch.zeros(32), 4, 150.0)
    assert torch.count_nonzero(empty[0, 4]) == 0
    with pytest.raises(ValueError, match="INJECTION_NONFINITE"):
        inject_activation(embeddings, vector * float("nan"), 4, 150.0)


def test_injection_rejects_wrong_neighbors_and_duplicate_markers(native_paths, native_tokenizers):
    av = Sidecar.load(native_paths["av"], "av")
    with pytest.raises(ValueError, match="INJECTION_NEIGHBORS"):
        injection_position([1, av.injection_token_id, av.right_id], av)
    with pytest.raises(ValueError, match="INJECTION_COUNT"):
        injection_position([av.left_id, av.injection_token_id, av.right_id, av.injection_token_id], av)


def test_random_control_matches_raw_and_injected_norm_without_global_rng_changes():
    source = torch.linspace(-4, 8, 3584)
    rng_before = torch.get_rng_state().clone()
    vector, provenance = norm_matched_random(source, 2718, "topic_baking")
    assert torch.equal(rng_before, torch.get_rng_state())
    assert vector.norm().item() == pytest.approx(source.norm().item(), rel=2e-7)
    assert abs(float(F.cosine_similarity(source, vector, dim=0))) < 0.1
    repeated, repeated_provenance = norm_matched_random(source, 2718, "topic_baking")
    assert torch.equal(vector, repeated)
    assert provenance == repeated_provenance
    for seed, identifier in ((2719, "topic_baking"), (2718, "topic_ocean")):
        different, _ = norm_matched_random(source, seed, identifier)
        assert not torch.equal(vector, different)
    norm_matched_random(source, 2718, "other_processing_order")
    assert torch.equal(vector, norm_matched_random(source, 2718, "topic_baking")[0])
    assert torch.equal(rng_before, torch.get_rng_state())
    embeddings = torch.ones(1, 5, 3584, dtype=torch.bfloat16)
    for control in (source, vector):
        injected = inject_activation(embeddings, control, 2, 150.0)
        assert injected[0, 2].float().norm().item() == pytest.approx(150, abs=0.5)
    assert inject_activation(embeddings, torch.zeros_like(source), 2, 150.0)[0, 2].count_nonzero() == 0


@pytest.mark.parametrize(
    "vector", [torch.zeros(4), torch.ones(4) * float("nan"), torch.ones(4) * 1e38, torch.ones(2, 2)]
)
def test_random_control_rejects_invalid_source(vector):
    with pytest.raises(ValueError, match="RANDOM_CONTROL_VECTOR"):
        norm_matched_random(vector, 2718, "test")


@pytest.mark.parametrize(
    "text,add_eos,at_cap,valid,exact,stop,early_eos",
    [
        ("<explanation>bread</explanation>", True, False, True, True, "eos", False),
        ("<explanation>bread", True, False, False, False, "eos", True),
        ("<explanation>bread", False, True, False, False, "length", False),
        ("<explanation>bread</explanation>", True, True, True, True, "eos", False),
        ("<explanation> </explanation>", True, False, False, False, "eos", False),
        ("<explanation>a</explanation><explanation>b</explanation>", True, False, False, False, "eos", False),
        ("<explanation><explanation>bread</explanation>", True, False, True, False, "eos", False),
        ("extra <explanation>bread</explanation>", True, False, True, False, "eos", False),
    ],
)
def test_exact_format_and_eos_diagnostics_preserve_original_parser(
    native_tokenizers, text, add_eos, at_cap, valid, exact, stop, early_eos
):
    tokenizer = native_tokenizers["av"]
    ids = tokenizer.encode(text, add_special_tokens=False) + ([tokenizer.eos_token_id] if add_eos else [])
    result = generation_diagnostics(tokenizer, ids, len(ids) if at_cap else len(ids) + 20)
    assert result["format_valid"] is valid
    assert result["exact_format_valid"] is exact
    assert result["stop_reason"] == stop
    assert result["eos_before_closing_tag"] is early_eos
    assert result["hit_token_cap"] is at_cap
    assert result["ended_on_eos"] is add_eos
    assert result["generated_token_ids"] == ids
    assert result["final_token_id"] == ids[-1]
    assert result["eos_token_positions"] == ([len(ids) - 1] if add_eos else [])
    assert result["raw"] == text + (tokenizer.eos_token if add_eos else "")


def test_real_hf_embeds_generation_returns_only_new_tokens(tiny_model, tokenizer):
    ids = torch.tensor([[3, 4, 11, 12, 13, 14, 5, 6]])
    mask = torch.ones_like(ids)
    generation = GenerationConfig(
        do_sample=False, max_new_tokens=3, eos_token_id=None, pad_token_id=tokenizer.pad_token_id
    )
    by_ids = tiny_model.generate(input_ids=ids, attention_mask=mask, generation_config=generation)
    by_embeds = tiny_model.generate(
        inputs_embeds=tiny_model.get_input_embeddings()(ids), attention_mask=mask, generation_config=generation
    )
    assert by_embeds.shape == (1, 3)
    torch.testing.assert_close(by_embeds, by_ids[:, ids.shape[1] :])


def test_hidden_state_index_is_previous_block_output(tiny_model, tokenizer):
    query = chat_query(tokenizer, " ".join(f"word{i}" for i in range(16)), 100)
    captures = []

    def capture(module, inputs, output):
        captures.append(output.detach().clone())

    hook = tiny_model.model.layers[1].register_forward_hook(capture)
    try:
        activation = extract_activation(tiny_model, query, 2)
    finally:
        hook.remove()
    torch.testing.assert_close(activation, captures[0][0, query["source_token_index"]])


def test_candidate_logprob_positions_match_full_forward(tiny_model, tokenizer):
    query = chat_query(tokenizer, " ".join(f"word{i}" for i in range(16)), 100)
    record = Record("r", "e", "native_fact", "heldout", query["rendered"], ("red", "blue green"), "blue green")
    metrics = score_behavior(tiny_model, tokenizer, query, record, 100)
    for i, candidate in enumerate(record.candidates):
        full, suffix = completion_ids(tokenizer, query, candidate, 100)
        ids = torch.tensor([full])
        logits = tiny_model(input_ids=ids, use_cache=False).logits[0]
        start = len(query["input_ids"]) - 1
        expected = -F.cross_entropy(logits[start : start + len(suffix)], torch.tensor(suffix), reduction="sum")
        assert metrics["candidate_logprobs"][i] == pytest.approx(float(expected.detach()), abs=1e-5)


def test_released_ar_prefix_layout_loads_only_missing_final_norm(tmp_path, tiny_model):
    tiny_model.config.save_pretrained(tmp_path)
    state = {
        key: value
        for key, value in tiny_model.state_dict().items()
        if key.startswith("model.") and key != "model.norm.weight"
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    model, loading = Qwen2Model.from_pretrained(tmp_path, output_loading_info=True, local_files_only=True)
    assert set(loading["missing_keys"]) == {"norm.weight"}
    assert not loading["unexpected_keys"]
    model.norm = torch.nn.Identity()
    ids = torch.tensor([[1, 2, 3]])
    output = model(ids).last_hidden_state
    assert output.shape == (1, 3, 32)


def test_direction_metric_has_correct_scale_and_ignores_source_norm():
    a = torch.tensor([1.0, 0, 0, 0])
    b = torch.tensor([0.0, 1, 0, 0])
    assert reconstruction_metrics(a, b, 2.0)["direction_mse"] == pytest.approx(2.0)
    assert reconstruction_metrics(a * 30, b * 90, 2.0)["direction_mse"] == pytest.approx(2.0)
    assert reconstruction_metrics(a, a, 2.0)["cosine"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="COSINE_ZERO"):
        reconstruction_metrics(a, torch.zeros(4), 2.0)
