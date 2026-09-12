import copy
import json
import sys

import pytest
import torch
from peft import PeftModel
from portallib import ChoiceExample, PortalBase, PortalEvaluator
from transformers import Qwen3Config, Qwen3ForCausalLM

from choice_consolidation import (
    adapter_state,
    base_tensors,
    candidate_logps,
    check_base,
    choice_kl,
    encode_choices,
    evaluate,
    main,
    make_learner,
    prepare,
    tensor_hash,
    train,
    train_arm,
)
from choice_contract import (
    ROOT,
    digest,
    file_hash,
    load_corpus,
    prediction,
    qualification_gate,
    source_hashes,
    training_schedule,
    validate_protocol,
    verify_qualification,
    write_json,
)


class CharacterTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens):
        return Tokenization(([self.bos_token_id] if add_special_tokens else []) + [ord(char) + 3 for char in text])


class Tokenization:
    def __init__(self, ids):
        self.input_ids = ids


@pytest.fixture
def protocol():
    return json.loads((ROOT / "configs/choice_consolidation_protocol.json").read_text())


@pytest.fixture
def tiny_base():
    torch.manual_seed(3)
    config = Qwen3Config(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, use_cache=False,
    )
    return Qwen3ForCausalLM(config).requires_grad_(False).eval()


def sample_row():
    return {"id": "sample", "task": "sequence_a", "group": "0 1 2", "prompt": "Code:\nOutput: ", "choices": [" 1 2 3", " 3 2 1", " 0 0 0", " 7 6 5"], "gold_idx": 0}


def perfect_panels(corpus, errors=0):
    result = {}
    for split in ("train", "validation"):
        teachers = []
        for i, row in enumerate(corpus[split]):
            chosen = (row["gold_idx"] + (i < errors)) % 4
            teachers.append(prediction(row, [2.0 if j == chosen else -2.0 for j in range(4)]))
        result[split] = {"teacher": teachers, "untouched": [prediction(row, [0.0, 0.0, 0.0, 0.0]) for row in corpus[split]]}
    return result


def save_qualification(path, protocol, corpus, audit, panels):
    write_json(path / "qualification_panels.json", panels)
    gate = qualification_gate(protocol, corpus, panels)
    write_json(path / "qualification.json", {
        "protocol_sha256": digest(protocol), "source_sha256": source_hashes(), "dataset": audit,
        "teacher_tensor_sha256": protocol["teacher"]["tensor_sha256"],
        "base_tensor_sha256": protocol["source_base_tensor_sha256"],
        "optimizer_updates": 0, "gate": gate, "status": "qualified" if gate["passed"] else "rejected",
        "panels_sha256": file_hash(path / "qualification_panels.json"),
    })


def test_fixture_groups_oracle_and_budget(protocol):
    validate_protocol(protocol)
    corpus, audit = load_corpus(protocol)
    assert [len(corpus[s]) for s in ("train", "validation", "test", "unused288")] == [384, 96, 192, 864]
    assert sum(audit["unique_inputs_per_split"].values()) == 512
    schedule = training_schedule(protocol, corpus["train"])
    assert len(schedule) == 384
    assert sum(map(len, schedule)) == 1536
    assert all(sum(i in batch for batch in schedule) == 4 for i in range(384))
    assert schedule == training_schedule(protocol, corpus["train"])
    assert schedule[:96] != schedule[96:192]


def test_qualification_gate_and_tamper_detection(protocol, tmp_path):
    corpus, audit = load_corpus(protocol)
    panels = perfect_panels(corpus)
    save_qualification(tmp_path, protocol, corpus, audit, panels)
    receipt, targets = verify_qualification(tmp_path, protocol, corpus, audit)
    assert receipt["gate"]["passed"]
    assert len(targets) == 384
    panels["train"]["teacher"][0]["correct"] = False
    with pytest.raises(ValueError, match="BINDING"):
        qualification_gate(protocol, corpus, panels)
    assert not qualification_gate(protocol, corpus, perfect_panels(corpus, errors=4))["passed"]


def test_failed_eligibility_precedes_model_loading(protocol, tmp_path, monkeypatch):
    corpus, audit = load_corpus(protocol)
    save_qualification(tmp_path / "rejected", protocol, corpus, audit, perfect_panels(corpus, errors=4))
    calls = []
    monkeypatch.setattr("choice_consolidation.load_base", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="TEACHER_NOT_QUALIFIED_NO_LEARNER_UPDATES"):
        train(protocol, {"qualification_dir": str(tmp_path / "rejected")}, corpus, audit, tmp_path / "training")
    assert not calls
    assert not (tmp_path / "training").exists()


def test_choice_kernel_matches_original_evaluator(tiny_base):
    tokenizer = CharacterTokenizer()
    rows = [sample_row(), {**sample_row(), "id": "short", "prompt": "Q:", "choices": [" 3 2", " 1 2 3", " 7", " 0 0 0"]}]
    base = PortalBase("tiny", tiny_base, tokenizer)
    expected, _, _ = PortalEvaluator(max_prompt=768, batch_size=8)._score_rows(
        base, [ChoiceExample(r["task"], r["prompt"], tuple(r["choices"]), r["gold_idx"]) for r in rows]
    )
    actual = evaluate(tiny_base, tokenizer, rows)
    assert [r["scores"] for r in actual] == expected
    boundary = encode_choices(tokenizer, rows[0])[0]
    assert boundary[1] == 7
    assert boundary[2] == 7
    with torch.inference_mode():
        logp = candidate_logps(tiny_base, tokenizer, [boundary])[0]
    assert actual[0]["scores"][0] == float(logp) / 7


def test_kl_math_and_teacher_stop_gradient():
    student = torch.tensor([[0.5, -0.2, 0.0, 0.8]], requires_grad=True)
    teacher = torch.tensor([[3.0, 0.5, -0.7, 0.1]], requires_grad=True)
    temperature = 2.0
    expected_p = torch.softmax(teacher.detach() / temperature, -1)
    expected = temperature**2 * (expected_p * (expected_p.log() - torch.log_softmax(student / temperature, -1))).sum()
    loss = choice_kl(student, teacher, temperature)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(student.grad, temperature * (torch.softmax(student.detach() / temperature, -1) - expected_p))
    assert teacher.grad is None
    assert abs(float(choice_kl(teacher.detach(), teacher.detach(), 1))) < 1e-7


def test_independent_learner_updates_and_saved_reload(protocol, tiny_base, tmp_path):
    config = copy.deepcopy(protocol)
    config["learner"]["expected_trainable_parameters"] = 1792
    base_weights = copy.deepcopy(tiny_base.state_dict())
    base_hash = tensor_hash(base_tensors(tiny_base))
    tokenizer = CharacterTokenizer()
    baseline = evaluate(tiny_base, tokenizer, [sample_row()])
    model = make_learner(tiny_base, config)
    assert evaluate(model, tokenizer, [sample_row()]) == baseline
    initial_hash = tensor_hash(adapter_state(model))
    targets = [prediction(sample_row(), [4.0, -3.0, -3.0, -3.0])]
    result = train_arm(model, tokenizer, [sample_row()], targets, [[0], [0]], "teacher_choice_kl", config, tmp_path)
    assert result["updates"] == result["example_exposures"] == 2
    assert tensor_hash(adapter_state(model)) != initial_hash
    check_base(model, base_hash)
    second_base = Qwen3ForCausalLM(model.get_base_model().config).requires_grad_(False).eval()
    second_base.load_state_dict(base_weights)
    restored = PeftModel.from_pretrained(second_base, result["checkpoint"]["path"], adapter_name="learner", is_trainable=False)
    assert tensor_hash(adapter_state(restored)) == result["checkpoint"]["tensor_sha256"]
    assert evaluate(restored, tokenizer, [sample_row()]) == evaluate(model, tokenizer, [sample_row()])
    check_base(restored, base_hash)
    with torch.no_grad():
        next(iter(base_tensors(restored).values())).view(-1)[0].add_(1)
    with pytest.raises(ValueError, match="FROZEN_BASE_MUTATED"):
        check_base(restored, base_hash)


def test_dispatch_cpu_validation_and_immutable_output(protocol, tmp_path, monkeypatch):
    dispatch = json.loads((ROOT / "configs/choice_consolidation_qualify.json").read_text())
    write_json(tmp_path / "config.json", dispatch)
    (tmp_path / "run.log").write_text("supervisor\n")
    monkeypatch.setattr(sys, "argv", ["choice_consolidation.py", "--config", str(tmp_path / "config.json"), "--output-dir", str(tmp_path), "--validate-only"])
    main()
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert not validation["gpu_qualified"]
    assert not (tmp_path / "qualification.json").exists()
    _, audit = load_corpus(protocol)
    with pytest.raises(FileExistsError, match="OUTPUT_ALREADY_USED"):
        prepare(tmp_path, dispatch, protocol, audit)
