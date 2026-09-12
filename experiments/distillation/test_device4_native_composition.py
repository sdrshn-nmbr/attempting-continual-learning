import copy
import json
import os
import subprocess
import sys
from collections import Counter

import device4_native_composition as lane
import pytest
import torch
from generated_contract import digest, file_digest
from test_device4_depth34 import cpu_fixture, model_records
from test_qualified import tiny_runner

from tasks import to_records


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/device4_native_composition_protocol.json").read_text())


def test_original_scientific_recipe_and_frozen_sources_unchanged(design):
    _, original = lane.validate_design(design)
    corpus, audit = lane.load_data(design, original)
    assert audit["rows_per_split"] == {"demonstration": 18, "train": 384, "validation": 144, "test": 288}
    schedule = lane.train_schedule(corpus, design)
    assert len(schedule) == 384 and all(len(batch) == 4 for batch in schedule)
    assert Counter(uid for batch in schedule for uid in batch) == {task.uid: 4 for task in corpus["train"]}
    assert lane.train_method is lane.native.train_method
    assert lane.verify_training_tokens is lane.native.verify_training_tokens
    for name in ("device4_recovery_handoff.json", "device4_depth34_handoff.json", "device4_chain_handoff.json"):
        hashes = json.loads((lane.ROOT / name).read_text())["staging_files_sha256"]
        assert all(file_digest(lane.ROOT / path) == value for path, value in hashes.items())
    changed = copy.deepcopy(design)
    changed["training"]["updates_per_method"] = 96
    with pytest.raises(ValueError, match="SCIENTIFIC_RECIPE_MUST_REMAIN_UNCHANGED"):
        lane.validate_design(changed)
    for stage in ("run", "qualify", "learn", "evaluate"):
        config = json.loads((lane.ROOT / f"configs/device4_native_composition_{stage}.json").read_text())
        assert config["protocol_sha256"] == digest(design)


def test_recovered_depth12_failure_stops_before_model_load(design, tmp_path, monkeypatch):
    upstream, original = lane.validate_design(design)
    corpus, audit = lane.load_data(design, original)
    calls = []
    monkeypatch.setattr(lane.recovered, "verify_source", lambda *args: ({"eligible": []}, {}, {}))
    monkeypatch.setattr(lane, "load_runner", lambda *args: calls.append(args))
    result = lane.qualify(design, upstream, original, corpus, audit, tmp_path)
    assert result["status"] == "rejected_no_eligible_depth12_teacher" and result["learner_updates"] == 0
    assert not calls


@pytest.fixture
def fixture(design, tmp_path):
    old = json.loads((lane.ROOT / design["native_design"]["path"]).read_text())
    f = cpu_fixture(old, tmp_path)
    design["training"], design["qualification"] = old["training"], old["qualification"]
    f["design"] = design
    path = tmp_path / "qualified/qualification.json"
    receipt = json.loads(path.read_text())
    receipt["source_sha256"], receipt["design_sha256"] = lane.source_hashes(), digest(design)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return f


def test_deep_qualification_carries_all_passing_arms_under_new_source(fixture, tmp_path, monkeypatch):
    f = fixture
    old_corpus = {"demonstration": f["corpus"]["demonstration"]}
    monkeypatch.setattr(lane.recovered, "verify_source", lambda *args: (f["proof"], old_corpus, {}))
    def load(original, upstream, rows, history, output):
        output.mkdir()
        runner = tiny_runner(original, output)
        runner.corpus = rows
        return runner
    monkeypatch.setattr(lane, "load_runner", load)
    monkeypatch.setattr(lane, "measure", lambda runner, rows, output, label, privileged=True, forced=True: model_records(runner, rows, runner.corpus, privileged, privileged))
    root = tmp_path / "new-qualification"
    receipt = lane.qualify(f["design"], f["upstream"], f["original"], f["corpus"], f["audit"], root)
    assert receipt["eligible"] == ["flat_mixed", "primitive_first"]
    assert receipt["source_sha256"] == lane.source_hashes()
    assert receipt["teacher_optimizer_updates"] == receipt["learner_updates"] == receipt["test_predictions"] == 0
    assert lane.read_certificate(root, f["design"], f["original"], f["corpus"], f["audit"]) == receipt
    receipt["eligible"] = ["flat_mixed"]
    (root / "qualification.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="ALL_PASSING_TEACHERS_MUST_BE_CARRIED"):
        lane.read_certificate(root, f["design"], f["original"], f["corpus"], f["audit"])


def test_rejected_deep_source_opens_no_learner_and_old_receipt_is_rejected(fixture, tmp_path, monkeypatch):
    f = fixture
    monkeypatch.setattr(lane.recovered, "verify_source", lambda *args: (f["proof"], {}, {}))
    def forbidden(*args):
        raise AssertionError("Learner loaded after prerequisite rejected")
    monkeypatch.setattr(lane, "load_runner", forbidden)
    path = tmp_path / "qualified/qualification.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "rejected"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="QUALIFIED_TEACHER_REQUIRED_BEFORE_LEARNER"):
        lane.train(f["design"], {"qualification_dir": str(tmp_path / "qualified")}, f["upstream"], f["original"], f["corpus"], f["audit"], tmp_path / "learning")
    receipt["status"], receipt["source_sha256"] = "qualified", lane.native.source_hashes()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="QUALIFIED_TEACHER_REQUIRED_BEFORE_LEARNER"):
        lane.read_certificate(tmp_path / "qualified", f["design"], f["original"], f["corpus"], f["audit"])


def test_actual_context_updates_and_fresh_pid_evaluation_without_teachers(fixture, tmp_path, monkeypatch):
    f = fixture
    monkeypatch.setattr(lane.recovered, "verify_source", lambda *args: (f["proof"], {}, {}))
    monkeypatch.setattr(lane, "load_runner", lambda original, upstream, rows, history, output: tiny_runner(original, output))
    root = tmp_path / "learning"
    root.mkdir()
    training = lane.train(f["design"], {"qualification_dir": str(tmp_path / "qualified")}, f["upstream"], f["original"], f["corpus"], f["audit"], root)
    assert training["total_learner_updates"] == 6 and len(training["methods"]) == 3
    assert training["teacher_optimizer_updates"] == training["test_predictions_during_learning"] == 0
    for method, arm in training["methods"].items():
        assert arm["updates"] == 2 and arm["example_exposures"] == 6
        assert arm["checkpoints"]["2"]["adapter"]["tensor_sha256"] != training["initial"]["tensor_sha256"]
        optimizer = torch.load(root / method / "checkpoint2/optimizer.pt", map_location="cpu", weights_only=True)
        assert {int(row["step"]) for row in optimizer["state"].values()} == {2}
        if method != "oracle_sft":
            rows = json.loads((root / method / "rows.json").read_text())
            assert all(row["diagnostics"]["support"] == "full_vocabulary" for row in rows)
            assert arm["teacher_tokens"] > arm["rollout_tokens"] > 0
    with pytest.raises(ValueError, match="NOT_FRESH_PROCESS"):
        lane.verify_training(root, f["design"], f["original"], f["corpus"], f["audit"])
    for spec in f["proof"]["checkpoints"].values():
        lane.native.verify_adapter(spec)
    payload = {name: f[name] for name in ("design", "original", "audit")}
    payload["corpus"] = to_records(f["corpus"])
    lane.write_new(tmp_path / "fixture.json", payload)
    (tmp_path / "teachers").rename(tmp_path / "teachers-unavailable")
    (tmp_path / "qualified").rename(tmp_path / "qualification-unavailable")
    script = tmp_path / "evaluate_cpu.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import device4_native_composition as lane\nfrom test_qualified import tiny_runner\nfrom tasks import Task\n"
        "root=Path(sys.argv[1])\nf=json.loads((root/'fixture.json').read_text())\n"
        "corpus={split:[Task(r['uid'],r['family'],r['split'],tuple(r['initial']),tuple(r['program']),tuple(r['answer'])) for r in rows] for split,rows in f['corpus'].items()}\n"
        "def load(original,output):\n    runner=tiny_runner(original,output)\n    runner.model.set_adapter('student',inference_mode=True)\n    runner.model.delete_adapter('teacher')\n    return runner\n"
        "def forbidden(*args,**kwargs):\n    raise AssertionError('External teacher/source accessed during standalone evaluation')\n"
        "lane.load_evaluator=load\nlane.recovered.verify_source=forbidden\nlane.native.verify_upstream=forbidden\n"
        "(root/'evaluation').mkdir()\n"
        "result=lane.evaluate(f['design'],{'training_dir':str(root/'learning')},f['original'],corpus,f['audit'],root/'evaluation')\n"
        "assert result['teacher_weights_loaded'] is False and result['external_teacher_artifacts_read'] is False\n"
        "training=json.loads((root/'learning/training.json').read_text())\ntraining['methods']['oracle_sft']['selected_checkpoint']=1\n(root/'learning/training.json').write_text(json.dumps(training))\n"
        "try:\n    lane.verify_training(root/'learning',f['design'],f['original'],corpus,f['audit'])\n"
        "except ValueError as error:\n    assert 'FIXED_FINAL_CHANGED' in str(error)\n"
        "else:\n    raise AssertionError('Earlier checkpoint accepted')\n"
    )
    child = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(script), str(tmp_path)], cwd=lane.ROOT, env={**os.environ, "PYTHONPATH": str(lane.ROOT)}, capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "evaluation/result.json").read_text())
    assert result["pid"] != training["pid"] and result["resident_adapters"] == ["student"]
    assert result["qualified_teachers"] == ["flat_mixed", "primitive_first"]
    assert result["fresh_process_reload_generation_parity"]


def test_automatic_new_process_stage_completion(design, tmp_path, monkeypatch):
    calls = []
    def child(command, cwd, check):
        calls.append(command)
        lane.write_new(tmp_path / "evaluation/result.json", {"status": "completed", "pid": os.getpid() + 1, "training_pid": os.getpid(), "comparisons": {}, "qualified_teachers": ["flat_mixed"], "learner_updates": 768, "claim_boundary": "cpu fixture"})
    monkeypatch.setattr(lane.subprocess, "run", child)
    dispatch = {"protocol": "configs/device4_native_composition_protocol.json", "protocol_sha256": digest(design)}
    lane.child_stage("evaluate", tmp_path, dispatch)
    assert calls[0][:3] == ["uv", "run", "--no-project"]
    assert "device4_native_composition.py" in calls[0][6]
    assert json.loads((tmp_path / "result.json").read_text())["learner_updates"] == 768


def test_run_rejection_does_not_launch_learner_and_validate_only(design, tmp_path, monkeypatch):
    def rejected(design, upstream, original, corpus, audit, output):
        receipt = {"status": "rejected", "eligible": [], "learner_updates": 0}
        lane.write_new(output / "qualification.json", receipt)
        return receipt
    def forbidden(*args):
        raise AssertionError("Child launched after prerequisite rejection")
    monkeypatch.setattr(lane, "qualify", rejected)
    monkeypatch.setattr(lane, "child_stage", forbidden)
    command = ["device4_native_composition.py", "--config", str(lane.ROOT / "configs/device4_native_composition_run.json"), "--output-dir"]
    monkeypatch.setattr(sys, "argv", [*command, str(tmp_path / "run")])
    lane.main()
    assert json.loads((tmp_path / "run/result.json").read_text())["learner_updates"] == 0
    monkeypatch.setattr(sys, "argv", [*command, str(tmp_path / "validate"), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validate/validation.json").read_text())["gpu_qualified"]
