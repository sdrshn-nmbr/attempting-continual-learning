import json
import sys

import pytest
import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from choice_contract import (
    ROOT,
    TASKS,
    digest,
    file_hash,
    load_corpus,
    prediction,
    write_json,
)
from generation_teacher import (
    eos_encoded,
    forced_diagnostics,
    generate,
    generation_gate,
    generation_settings,
    grade,
    main,
    select_checkpoint,
    source_hashes,
    train_eos,
    validate_design,
)
from test_choice_consolidation import CharacterTokenizer, sample_row


class GenerativeTokenizer(CharacterTokenizer):
    all_special_ids = [0, 1, 2]

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join({0: "<pad>", 1: "<bos>", 2: "<eos>"}.get(i, chr(max(0, i - 3))) for i in ids)


class ScriptedGenerator(torch.nn.Module):
    def __init__(self, response):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.response = response
        self.generation_config = GenerationConfig(forced_eos_token_id=7, max_new_tokens=1)

    def generate(self, input_ids, attention_mask, generation_config):
        assert generation_config is self.generation_config
        assert generation_config.forced_eos_token_id is None
        assert generation_config.eos_token_id == 2
        assert generation_config.max_new_tokens == 16
        assert not generation_config.do_sample
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        return torch.cat([input_ids, torch.tensor([self.response])], -1)


@pytest.fixture
def design():
    return json.loads((ROOT / "configs/generation_teacher_protocol.json").read_text())


def generated_record(row, body=None, eos=True):
    tokenizer = GenerativeTokenizer()
    body = row["choices"][row["gold_idx"]] if body is None else body
    tokens = tokenizer(body, add_special_tokens=False).input_ids + ([2] if eos else [])
    return {
        "id": row["id"], "task": row["task"], "group": row["group"], "row_sha256": digest(row),
        "generation": {"body_text": body, "token_ids": tokens, **grade(row, tokens, body, 2, [0, 1, 2], 16)},
    }


def panels_for(corpus, eos=True):
    return {
        split: {
            "generated": [generated_record(row, eos=eos) for row in corpus[split]],
            "choice": [prediction(row, [1.0 if i == row["gold_idx"] else -1.0 for i in range(4)]) for row in corpus[split]],
        }
        for split in ("train", "validation")
    }


def test_strict_three_digit_contract_does_not_accept_prefix_or_normalize():
    row = sample_row()
    valid = generated_record(row)["generation"]
    assert valid["correct"]
    for body in ("1 2 3", " 1 2 3 ", " 1 2 3\n", " 1 2 3 4", " 1 2 4", " 1 2 3 1 2 3"):
        assert not generated_record(row, body)["generation"]["correct"]
    missing = generated_record(row, eos=False)["generation"]
    assert not missing["correct"]
    assert missing["exact_body_without_valid_stop"]
    assert generated_record(row, " 1 2 3 x")["generation"]["correct_prefix_with_extra_text"]
    bad_special = grade(row, [0, 35, 52, 2], " 1 2 3", 2, [0, 1, 2], 16)
    assert not bad_special["format_valid"]


def test_raw_generation_overrides_inherited_forced_eos(design):
    tokenizer = GenerativeTokenizer()
    row = {**sample_row(), "prompt": "Output:"}
    tokens = tokenizer(row["choices"][0], add_special_tokens=False).input_ids + [2]
    model = ScriptedGenerator(tokens)
    result = generate(model, tokenizer, row, design)
    assert result["correct"]
    assert result["body_text"] == " 1 2 3"
    assert model.generation_config.forced_eos_token_id is None
    assert generation_settings(tokenizer, design).forced_eos_token_id is None


def test_teacher_forced_probability_includes_eos(design):
    validate_design(design)
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, use_cache=False,
    )).requires_grad_(False).eval()
    tokenizer = GenerativeTokenizer()
    row = {**sample_row(), "prompt": "Output:"}
    encoded = eos_encoded(tokenizer, row)
    assert encoded[1] == 7
    assert encoded[0][-1] == tokenizer.eos_token_id
    diagnostic = forced_diagnostics(model, tokenizer, row)
    assert len(diagnostic["gold_token_log_probabilities"]) == 7
    assert diagnostic["target_token_ids"][-1] == 2
    expected_logp = sum(diagnostic["gold_token_log_probabilities"])
    assert diagnostic["gold_sequence_log_probability_including_eos"] == expected_logp
    assert 0 < diagnostic["gold_sequence_probability_including_eos"] < 1
    assert expected_logp < diagnostic["gold_digit_body_log_probability"]


def test_choice_competence_cannot_substitute_for_whole_generation(design):
    protocol = validate_design(design)
    corpus, _ = load_corpus(protocol)
    good = panels_for(corpus)
    assert generation_gate(design, corpus, good, 2, [0, 1, 2])["passed"]
    without_eos = panels_for(corpus, eos=False)
    assert not generation_gate(design, corpus, without_eos, 2, [0, 1, 2])["passed"]
    without_eos["train"]["generated"][0]["generation"]["correct"] = True
    with pytest.raises(ValueError, match="GRADING_CHANGED"):
        generation_gate(design, corpus, without_eos, 2, [0, 1, 2])


def test_training_only_checkpoint_selection(design):
    checkpoints = {
        str(step): {"train_probe_summary": {task: {"correct": accuracy} for task in TASKS}}
        for step, accuracy in ((32, 0.9), (128, 0.96875), (384, 1.0))
    }
    assert select_checkpoint(checkpoints, design) == 128
    checkpoints["128"]["train_probe_summary"]["sequence_b"]["correct"] = 0.9
    assert select_checkpoint(checkpoints, design) == 384
    checkpoints["32"]["train_probe_summary"] = checkpoints["384"]["train_probe_summary"]
    assert select_checkpoint(checkpoints, design) == 32


def test_already_qualified_source_skips_training_before_gpu_load(design, tmp_path, monkeypatch):
    protocol = validate_design(design)
    corpus, audit = load_corpus(protocol)
    panels = panels_for(corpus)
    write_json(tmp_path / "qualification_panels.json", panels)
    write_json(tmp_path / "qualification.json", {
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "teacher_tensor_sha256": protocol["teacher"]["tensor_sha256"],
        "panels_sha256": file_hash(tmp_path / "qualification_panels.json"),
        "gate": generation_gate(design, corpus, panels, 2, [0, 1, 2]), "status": "qualified",
        "eos_token_id": 2, "special_token_ids": [0, 1, 2],
    })
    calls = []
    monkeypatch.setattr("generation_teacher.load_teacher", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="ALREADY_QUALIFIED_NO_TRAINING_NEEDED"):
        train_eos(protocol, design, {"diagnosis_dir": str(tmp_path)}, corpus, audit, tmp_path / "train")
    assert not calls


def test_generation_dispatch_cpu_validation(tmp_path, monkeypatch):
    dispatch = json.loads((ROOT / "configs/generation_teacher_diagnose.json").read_text())
    write_json(tmp_path / "config.json", dispatch)
    monkeypatch.setattr(sys, "argv", ["generation_teacher.py", "--config", str(tmp_path / "config.json"), "--output-dir", str(tmp_path), "--validate-only"])
    main()
    assert json.loads((tmp_path / "validation.json").read_text())["gpu_qualified"] is False
    assert not (tmp_path / "qualification.json").exists()
