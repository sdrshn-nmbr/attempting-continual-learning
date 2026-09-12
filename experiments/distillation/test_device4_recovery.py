import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import device4_recovery as lane
import pytest
import torch
from generated_contract import digest, file_digest, make_corpus
from test_qualified import tiny_runner

from run import adapter_parameters, tensor_digest
from tasks import FAMILIES, to_records


def cpu_runner(protocol, output):
    output.mkdir(parents=True, exist_ok=True)
    return tiny_runner(protocol, output)


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/device4_recovery_protocol.json").read_text())


def test_real_interrupted_source_and_all_saved_optimizers(design):
    upstream, original, snapshot = lane.validate_design(design)
    corpus = make_corpus(original, lane.original_lane.SPLITS)
    schedules = lane.original_lane.make_schedules(corpus, upstream)
    folder = lane.ROOT.parent.parent / "outputs/portfolio/followthrough-20260912/runs/distillation-device4-curriculum-20260912"
    partial = lane.inspect_partial(folder, snapshot, upstream, original, corpus, schedules)
    assert {arm: len(rows) for arm, rows in partial["ledgers"].items()} == {"flat_mixed": 384, "primitive_first": 384, "legacy96_budget": 223}
    assert set(partial["checkpoints"]["legacy96_budget"]) == {"24", "96"}
    assert snapshot["discarded_completed_legacy_updates"] == 127
    assert partial["initial"]["tensor_sha256"] == "c564aed5ded07d2b70e61420ae60e2b92c759ab6a0031b541e1356c63c40a9cf"
    assert not (folder / "training.json").exists()
    for filename in ("device4_curriculum_handoff.json", "device4_depth34_handoff.json", "coverage_learner_handoff.json"):
        handoff = json.loads((lane.ROOT / filename).read_text())
        assert all(file_digest(lane.ROOT / name) == checksum for name, checksum in handoff["staging_files_sha256"].items())
    for stage in ("run", "resume", "evaluate"):
        config = json.loads((lane.ROOT / f"configs/device4_recovery_{stage}.json").read_text())
        assert config["protocol_sha256"] == digest(design)


@pytest.fixture
def cpu_run(design, tmp_path):
    upstream, original, _ = lane.validate_design(design)
    full = make_corpus(original, lane.original_lane.SPLITS)
    corpus = {"demonstration": full["demonstration"], **{split: [task for family in FAMILIES for depth in (1, 2) for task in [row for row in full[split] if row.family == family and len(row.program) == depth][:3]] for split in ("train", "fallback_validation")}}
    original["lora_rank"], original["lora_alpha"] = 2, 4
    original["qualification"]["train_per_family"] = original["qualification"]["validation_per_family"] = 6
    recipe = upstream["training"]
    recipe.update(updates_per_arm=4, examples_per_update=1, example_exposures_per_arm=4, checkpoints=[2, 4])
    recovery = {**design["resume"], "checkpoint": 2, "last_logged_update": 3, "final_update": 4, "new_updates": 2, "new_exposures": 2}
    schedule = [[corpus["train"][i].uid] for i in range(4)]
    runner = cpu_runner(original, tmp_path / "uninterrupted")
    runner.corpus = corpus
    recipe["expected_trainable_parameters"] = sum(p.numel() for p in runner.student_parameters)
    reference = lane.original_lane.train_arm(runner, upstream, "legacy96_budget", schedule, tmp_path / "uninterrupted")
    ledger = json.loads((tmp_path / "uninterrupted/legacy96_budget/ledger.json").read_text())
    checkpoint = {**reference["checkpoints"]["2"], "optimizer_path": str(tmp_path / "uninterrupted/legacy96_budget/checkpoint2/optimizer.pt"), "train_diagnostics_path": str(tmp_path / "uninterrupted/legacy96_budget/checkpoint2/train_diagnostics.json")}
    return {"design": design, "upstream": upstream, "original": original, "corpus": corpus, "recovery": recovery, "schedule": schedule, "ledger": ledger, "checkpoint": checkpoint, "reference": reference}


def test_real_cpu_optimizer_resume_matches_uninterrupted_weights_and_replay(cpu_run, tmp_path):
    data = cpu_run
    runner = cpu_runner(data["original"], tmp_path / "resumed")
    runner.corpus = data["corpus"]
    before = {name: file_digest(Path(data["checkpoint"]["adapter"]["path"]) / name) for name in data["checkpoint"]["adapter"]["files_sha256"]}
    parity = lane.reload_parity(runner, data["checkpoint"], tmp_path / "resumed", "cpu/source2")
    assert parity["rows"] == 18 and parity["exact_generation_parity"]
    checkpoints, ledger, replay = lane.resume_legacy(runner, data["upstream"], data["recovery"], data["checkpoint"], data["ledger"][:3], data["schedule"], tmp_path / "resumed")
    assert ledger == data["ledger"]
    assert replay == [{"step": 3, "exact": True, "within_predeclared_tolerance": True, "loss_difference": 0.0, "gradient_norm_difference": 0.0}]
    assert checkpoints["4"]["adapter"]["tensor_sha256"] == data["reference"]["checkpoints"]["4"]["adapter"]["tensor_sha256"]
    optimizer = torch.load(checkpoints["4"]["optimizer_path"], map_location="cpu", weights_only=True)
    assert {int(row["step"]) for row in optimizer["state"].values()} == {4}
    assert all(file_digest(Path(data["checkpoint"]["adapter"]["path"]) / name) == checksum for name, checksum in before.items())


def test_bad_optimizer_moments_and_steps_are_rejected(cpu_run, tmp_path):
    data = cpu_run
    parameters = lane.saved_parameters(Path(data["checkpoint"]["adapter"]["path"]) / "adapter_model.safetensors")
    runner = cpu_runner(data["original"], tmp_path / "optimizer-audit")
    names = list(adapter_parameters(runner.model, "teacher"))
    state = torch.load(data["checkpoint"]["optimizer_path"], map_location="cpu", weights_only=True)
    key = next(iter(state["state"]))
    bad = copy.deepcopy(state)
    bad["state"][key]["step"].fill_(1)
    path = tmp_path / "bad_step.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="OPTIMIZER_STEP_MISMATCH"):
        lane.inspect_optimizer(path, 2, names, parameters, data["upstream"]["training"])
    bad = copy.deepcopy(state)
    bad["state"][key]["exp_avg"].fill_(float("nan"))
    path = tmp_path / "bad_moment.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="MOMENTS_INVALID"):
        lane.inspect_optimizer(path, 2, names, parameters, data["upstream"]["training"])


def test_numerical_replay_failure_stops_before_parameter_update(cpu_run, tmp_path):
    data = cpu_run
    runner = cpu_runner(data["original"], tmp_path / "mismatch")
    runner.corpus = data["corpus"]
    ledger = copy.deepcopy(data["ledger"][:3])
    ledger[2]["loss"] += 1
    with pytest.raises(ValueError, match="NUMERICAL_REPLAY_MISMATCH_BEFORE_UPDATE: 3"):
        lane.resume_legacy(runner, data["upstream"], data["recovery"], data["checkpoint"], ledger, data["schedule"], tmp_path / "mismatch")
    assert tensor_digest(adapter_parameters(runner.model, "teacher")) == data["checkpoint"]["adapter"]["tensor_sha256"]
    assert not (tmp_path / "mismatch/legacy96_budget/ledger.json").exists()


def test_partial_or_reordered_prefix_cannot_be_reused(cpu_run):
    data = cpu_run
    with pytest.raises(ValueError, match="EXACT_SCHEDULE_PREFIX"):
        lane.verify_ledger(data["ledger"][:2], data["schedule"], 3)
    with pytest.raises(ValueError, match="EXACT_SCHEDULE_PREFIX"):
        lane.verify_ledger(list(reversed(data["ledger"])), data["schedule"], 4)


def test_actual_new_process_qualification_keeps_original_gates(cpu_run, tmp_path):
    data = cpu_run
    checkpoint = {**data["reference"]["checkpoints"]["4"], "train_diagnostics_path": str(tmp_path / "uninterrupted/legacy96_budget/checkpoint4/train_diagnostics.json")}
    runner = cpu_runner(data["original"], tmp_path / "qualify-reference")
    training = {"pid": os.getpid(), "snapshot_sha256": runner.snapshot, "eos_token_id": runner.eos, "special_token_ids": runner.special_ids, "base_after_sha256": lane.original_lane.frozen_digest(runner), "new_optimizer_updates": 2, "arms": {arm: {"ledger_path": str(tmp_path / "uninterrupted/legacy96_budget/ledger.json"), "selected_checkpoint": 4, "origin": "cpu_fixture", "checkpoints": {"4": checkpoint}} for arm in lane.original_lane.ARMS}}
    data["training"] = training
    data["snapshot"] = {"execution_sha256": "cpu_interruption_proof"}
    data["corpus"] = to_records(data["corpus"])
    lane.write_new(tmp_path / "fixture.json", data)
    lane.write_new(tmp_path / "training/training.json", training)
    script = tmp_path / "qualify_cpu.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import device4_recovery as lane\nfrom test_qualified import tiny_runner\nfrom tasks import Task\n"
        "root=Path(sys.argv[1])\nf=json.loads((root/'fixture.json').read_text())\n"
        "corpus={split:[Task(r['uid'],r['family'],r['split'],tuple(r['initial']),tuple(r['program']),tuple(r['answer'])) for r in rows] for split,rows in f['corpus'].items()}\n"
        "def load(original,upstream,rows,history,output):\n    runner=tiny_runner(original,output)\n    runner.corpus=rows\n    return runner\n"
        "lane.make_corpus=lambda *args:corpus\nlane.original_lane.make_schedules=lambda *args:{}\n"
        "lane.original_lane.verify_history=lambda *args:{}\nlane.verify_recovered_training=lambda *args:f['training']\n"
        "lane.original_lane.load_runner=load\n(root/'qualification').mkdir()\n"
        "result=lane.evaluate(f['design'],{'training_dir':str(root/'training')},f['upstream'],f['original'],f['snapshot'],root/'qualification')\n"
        "assert result['status']=='completed' and result['pid']!=result['training_pid']\n"
        "assert result['learner_updates']==result['test_predictions']==result['composition_predictions']==0\n"
        "assert all(len(arm['gate']['panels'])==6 for arm in result['arms'].values())\n"
    )
    child = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(script), str(tmp_path)], cwd=lane.ROOT, env={**os.environ, "PYTHONPATH": str(lane.ROOT)}, capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "qualification/qualification.json").read_text())
    assert all(arm["status"] == "rejected" for arm in result["arms"].values())


def test_driver_requires_fresh_completed_qualification(tmp_path, monkeypatch):
    calls = []

    def child(command, cwd, check):
        calls.append(command)
        lane.write_new(tmp_path / "evaluation/qualification.json", {"status": "completed", "pid": os.getpid() + 1, "training_pid": os.getpid(), "new_optimizer_updates": 288, "arms": {}})

    monkeypatch.setattr(lane.subprocess, "run", child)
    lane.evaluate_child(tmp_path, {"protocol": "configs/device4_recovery_protocol.json", "protocol_sha256": "cpu_fixture"})
    assert calls[0][:4] == ["uv", "run", "--no-project", "--python"]
    assert json.loads((tmp_path / "result.json").read_text())["new_optimizer_updates"] == 288


def test_validate_only_without_gpu_or_provider_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["device4_recovery.py", "--config", str(lane.ROOT / "configs/device4_recovery_run.json"), "--output-dir", str(tmp_path), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validation.json").read_text())["gpu_qualified"]
