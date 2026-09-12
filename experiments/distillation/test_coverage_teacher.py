import copy
import json
import os
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import coverage_teacher as lane
from choice_consolidation import base_tensors, make_learner, save_adapter, tensor_hash
from choice_contract import digest, file_hash, load_corpus, write_json
from generation_teacher import generation_gate
from test_generation_teacher import GenerativeTokenizer, generated_record, panels_for


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/coverage_teacher_protocol.json").read_text())


def perfect_records(rows):
    return [generated_record(row, row["choices"][row["gold_idx"]], eos=True) for row in rows]


def saved_screen(tmp_path, design, teacher_design, corpus, audit, pass128=True):
    original = perfect_records(corpus["train"])
    original[0] = generated_record(corpus["train"][0], " 7 7 7", eos=True)
    checkpoints, candidates = {}, {}
    proof = {"eos_token_id": 2, "special_token_ids": [0, 1, 2], "test_fixture": True}
    for step in (32, 128, 384):
        path = tmp_path / f"adapter{step}"
        path.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (path / name).write_bytes(f"fixture-{step}-{name}".encode())
        checkpoint = {"path": str(path), "tensor_sha256": f"fixture-{step}", "files": {name: file_hash(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")}}
        checkpoints[str(step)] = checkpoint
        records = original if step == 32 or (step == 128 and not pass128) else perfect_records(corpus["train"])
        write_json(tmp_path / f"train{step}.json", records)
        candidates[str(step)] = {"coverage": lane.coverage_gate(corpus["train"], records, teacher_design, 2, [0, 1, 2]), "records_sha256": file_hash(tmp_path / f"train{step}.json"), "checkpoint": checkpoint}
    selected = lane.select_checkpoint(candidates)
    receipt = {
        "status": "selected", "pid": 111, "design_sha256": digest(design), "source_sha256": lane.source_hashes(),
        "dataset": audit, "teacher_training_sha256": design["teacher_training_sha256"], "original32_proof": proof,
        "teacher_optimizer_updates": 0, "learner_updates": 0, "new_validation_predictions": 0, "test_predictions": 0,
        "candidates": candidates, "selected_checkpoint": selected, "eos_token_id": 2, "special_token_ids": [0, 1, 2],
    }
    write_json(tmp_path / "coverage.json", receipt)
    archive = {"training": {"checkpoints": checkpoints}, "original32_proof": proof, "original32_train_records": original}
    return receipt, archive


def test_sealed_protocol_preserves_original32_and_dispatched_device4(design):
    previous, teacher, choice = lane.validate_design(design)
    assert previous["training"]["updates_per_arm"] == 384
    assert teacher["training"]["selection"] == "earliest_checkpoint_with_train_probe95_all_tasks_else384"
    assert choice["learner"]["rank"] == 8
    frozen = json.loads((lane.ROOT / "device4_curriculum_handoff.json").read_text())["staging_files_sha256"]
    assert all(file_hash(lane.ROOT / name) == checksum for name, checksum in frozen.items())
    for stage in ("run", "screen", "qualify"):
        cfg = json.loads((lane.ROOT / f"configs/coverage_teacher_{stage}.json").read_text())
        assert cfg["protocol_sha256"] == digest(design)
    changed = copy.deepcopy(design)
    changed["coverage"]["minimum_position_accuracy"] = 0.9
    with pytest.raises(ValueError, match="FIXED_TRAIN_RULE_CHANGED"):
        lane.validate_design(changed)


def test_real_original32_full_train_binding_is7_of15(design):
    _, teacher, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    folder = lane.ROOT.parent.parent / "outputs/portfolio/followthrough-20260912/runs/consolidation-source303-generation-evaluate-20260912"
    receipt = json.loads((folder / "qualification.json").read_text())
    assert file_hash(folder / "qualification.json") == design["original32_qualification_sha256"]
    assert receipt["dataset"] == audit and receipt["status"] == "qualified"
    assert file_hash(folder / "qualification_panels.json") == receipt["panels_sha256"]
    panels = json.loads((folder / "qualification_panels.json").read_text())
    gate = lane.coverage_gate(corpus["train"], panels["train"]["generated"], teacher, receipt["eos_token_id"], receipt["special_token_ids"])
    cell = gate["cells"]["sequence_a/position3/input5"]
    assert (cell["n"], cell["correct"], cell["expected_digits"]) == (15, 7, [3])
    assert len(cell["wrong_ids"]) == 8
    assert not gate["passed"] and gate["passed_cells"] == 71
    validation = lane.coverage_gate(corpus["validation"], panels["validation"]["generated"], teacher, receipt["eos_token_id"], receipt["special_token_ids"])
    assert (validation["cells"]["sequence_a/position3/input5"]["n"], validation["cells"]["sequence_a/position3/input5"]["correct"]) == (3, 2)


def test_empty_cells_and_invalid_stop_never_count_as_coverage(design):
    _, teacher, choice = lane.validate_design(design)
    corpus, _ = load_corpus(choice)
    rows = corpus["train"]
    complete = lane.coverage_gate(rows, perfect_records(rows), teacher, 2, [0, 1, 2])
    assert complete["passed"] and complete["present_cells"] == complete["passed_cells"] == 72
    subset = [row for row in rows if not (row["task"] == "sequence_a" and row["group"].split()[2] == "5")]
    missing = lane.coverage_gate(subset, perfect_records(subset), teacher, 2, [0, 1, 2])
    assert not missing["passed"] and missing["cells"]["sequence_a/position3/input5"]["n"] == 0
    records = perfect_records(rows)
    records[0] = generated_record(rows[0], rows[0]["choices"][rows[0]["gold_idx"]], eos=False)
    assert not lane.coverage_gate(rows, records, teacher, 2, [0, 1, 2])["passed"]
    records[0] = generated_record(rows[0], rows[0]["choices"][rows[0]["gold_idx"]] + " extra", eos=True)
    assert not lane.coverage_gate(rows, records, teacher, 2, [0, 1, 2])["passed"]


def test_earliest_complete_train_checkpoint_and_tampering(design, tmp_path):
    _, teacher, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    receipt, archive = saved_screen(tmp_path, design, teacher, corpus, audit)
    selected, records = lane.verify_screen(tmp_path, design, teacher, corpus, audit, archive)
    assert selected["selected_checkpoint"] == 128 and len(records) == 384
    receipt["selected_checkpoint"] = 384
    (tmp_path / "coverage.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="TRAIN_ONLY_SELECTION_CHANGED"):
        lane.verify_screen(tmp_path, design, teacher, corpus, audit, archive)


def test_new_teacher_receipt_requires_old_validation_gate_too(design, tmp_path, monkeypatch):
    previous, teacher, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    selection, archive = saved_screen(tmp_path, design, teacher, corpus, audit)
    root = tmp_path / "qualification"
    root.mkdir()
    panels = panels_for(corpus, eos=True)
    panels["train"]["generated"] = json.loads((tmp_path / "train128.json").read_text())
    gate = generation_gate(teacher, corpus, panels, 2, [0, 1, 2])
    assert gate["passed"]
    write_json(root / "qualification_panels.json", panels)
    candidate = selection["candidates"]["128"]
    receipt = {
        "status": "qualified", "pid": 222, "screen_pid": 111, "design_sha256": digest(design), "source_sha256": lane.source_hashes(),
        "dataset": audit, "screen_dir": str(tmp_path), "screen_sha256": file_hash(tmp_path / "coverage.json"),
        "panels_sha256": file_hash(root / "qualification_panels.json"), "selected_checkpoint": 128,
        "checkpoint": candidate["checkpoint"], "teacher_tensor_sha256": candidate["checkpoint"]["tensor_sha256"],
        "eos_token_id": 2, "special_token_ids": [0, 1, 2], "gate": gate, "train_coverage": candidate["coverage"],
        "validation_coverage_diagnostic_only": lane.coverage_gate(corpus["validation"], panels["validation"]["generated"], teacher, 2, [0, 1, 2]),
        "teacher_optimizer_updates": 0, "learner_updates": 0, "test_predictions": 0,
    }
    write_json(root / "qualification.json", receipt)
    monkeypatch.setattr(lane, "verify_archive", lambda *args: archive)
    qualified, records = lane.verify_qualified(root, design, previous, teacher, choice, corpus, audit)
    assert qualified["selected_checkpoint"] == 128 and len(records) == 384
    for row, record in zip(corpus["validation"], panels["validation"]["generated"], strict=True):
        record.update(generated_record(row, " 7 7 7", eos=False))
    (root / "qualification_panels.json").write_text(json.dumps(panels))
    receipt["panels_sha256"] = file_hash(root / "qualification_panels.json")
    (root / "qualification.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="QUALIFIED_GATE_OR_TRAIN_TARGETS_CHANGED"):
        lane.verify_qualified(root, design, previous, teacher, choice, corpus, audit)


def test_rejected_coverage_precedes_new_teacher_load(design, tmp_path, monkeypatch):
    _, teacher, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    _, archive = saved_screen(tmp_path, design, teacher, corpus, audit)
    receipt = json.loads((tmp_path / "coverage.json").read_text())
    for step in (128, 384):
        (tmp_path / f"train{step}.json").write_text(json.dumps(archive["original32_train_records"]))
        receipt["candidates"][str(step)]["records_sha256"] = file_hash(tmp_path / f"train{step}.json")
        receipt["candidates"][str(step)]["coverage"] = receipt["candidates"]["32"]["coverage"]
    receipt.update(status="rejected", selected_checkpoint=None)
    (tmp_path / "coverage.json").write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(lane, "load_teacher", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="NO_FULLY_COVERED_TEACHER"):
        lane.qualify(design, teacher, choice, corpus, audit, archive, tmp_path, tmp_path / "qualification")
    assert not calls


def test_coverage_run_driver_uses_fresh_pid_and_no_optimizer(tmp_path, monkeypatch):
    calls = []

    def child(command, cwd, check):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-project", "--python"]
        write_json(tmp_path / "qualification/qualification.json", {"status": "qualified", "selected_checkpoint": 128, "pid": os.getpid() + 1, "screen_pid": os.getpid()})

    monkeypatch.setattr(lane.subprocess, "run", child)
    lane.qualify_child(tmp_path, {"protocol": "configs/coverage_teacher_protocol.json", "protocol_sha256": "fixture"})
    assert len(calls) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["teacher_optimizer_updates"] == result["learner_updates"] == 0


def test_cpu_checkpoint_loading_swapping_and_frozen_greedy_screen(design, tmp_path, monkeypatch):
    _, teacher, choice = lane.validate_design(design)
    corpus, _ = load_corpus(choice)
    corpus = {"train": [next(row for row in corpus["train"] if row["task"] == task) for task in lane.TASKS]}
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=256, use_cache=False)
    torch.manual_seed(17)
    base = Qwen3ForCausalLM(config).requires_grad_(False).eval()
    weights = copy.deepcopy(base.state_dict())
    choice["source_base_tensor_sha256"] = tensor_hash(base_tensors(base))
    choice["learner"]["expected_trainable_parameters"] = 1792
    model = make_learner(base, choice)
    checkpoints = {}
    for step in (32, 128, 384):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.0001 * step)
        checkpoints[str(step)] = save_adapter(model, tmp_path / f"checkpoint{step}")

    def load_tiny(protocol, output):
        fresh = Qwen3ForCausalLM(config).requires_grad_(False).eval()
        fresh.load_state_dict(weights)
        return fresh, GenerativeTokenizer()

    monkeypatch.setattr(lane, "load_base", load_tiny)
    archive = {
        "training": {"checkpoints": checkpoints},
        "original32_proof": {"eos_token_id": 2, "special_token_ids": [0, 1, 2]},
        "original32_train_records": perfect_records(corpus["train"]),
    }
    output = tmp_path / "screen"
    output.mkdir()
    result = lane.screen(design, teacher, choice, corpus, {"cpu_only": True}, archive, output)
    assert result["status"] == "rejected" and result["selected_checkpoint"] is None
    assert result["teacher_optimizer_updates"] == result["learner_updates"] == 0
    assert result["new_validation_predictions"] == result["test_predictions"] == 0
    assert result["candidates"]["128"]["checkpoint"]["tensor_sha256"] != result["candidates"]["384"]["checkpoint"]["tensor_sha256"]
    assert all(len(json.loads((output / f"train{step}.json").read_text())) == 3 for step in (32, 128, 384))


def test_cpu_config_validation_without_gpu_or_history_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["coverage_teacher.py", "--config", str(lane.ROOT / "configs/coverage_teacher_run.json"), "--output-dir", str(tmp_path), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validation.json").read_text())["gpu_qualified"]
