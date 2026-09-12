import copy
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

import device4_curriculum as lane
from generated_contract import (
    digest,
    file_digest,
    grade_generation,
    make_corpus,
    qualification_rows,
    teacher_prompt,
)
from run import adapter_parameters, tensor_digest
from tasks import FAMILIES, digits, to_records
from test_qualified import tiny_runner


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/device4_curriculum_protocol.json").read_text())


def local_history(design):
    for spec in design["history"].values():
        spec["path"] = str(lane.ROOT.parent.parent / "outputs/portfolio/resume-20260911/runs" / Path(spec["path"]).name)


def test_sealed_contract_and_all_previously_dispatched_sources_unchanged(design):
    original = lane.validate_design(design)
    assert original["teacher_training"]["updates"] == 24
    assert original["model_id"] == "Qwen/Qwen3.5-4B"
    for stage in ("run", "train", "evaluate"):
        config = json.loads((lane.ROOT / f"configs/device4_curriculum_{stage}.json").read_text())
        assert config["stage"] == stage
        assert config["protocol_sha256"] == digest(design)
    frozen = json.loads((lane.ROOT / "generation_consolidation_handoff.json").read_text())["code_source_files_sha256"]
    assert {name: file_digest(lane.ROOT / name) for name in frozen} == frozen
    changed = copy.deepcopy(design)
    changed["training"]["updates_per_arm"] = 96
    with pytest.raises(ValueError, match="FIXED_TRAINING_CONTRACT"):
        lane.validate_design(changed)


def test_curriculum_changes_only_order_and_retains_later_primitives(design):
    original = lane.validate_design(design)
    corpus = make_corpus(original, lane.SPLITS)
    schedules = lane.make_schedules(corpus, design)
    assert schedules == lane.make_schedules(corpus, design)
    flat = [uid for batch in schedules["flat_mixed"] for uid in batch]
    curriculum = [uid for batch in schedules["primitive_first"] for uid in batch]
    old = [task.uid for task in lane.legacy_pool(corpus)]
    expected = Counter({task.uid: 6 if task.uid in old else 5 for task in corpus["train"]})
    assert Counter(flat) == Counter(curriculum) == expected
    assert flat != curriculum
    lookup = {task.uid: task for task in corpus["train"]}
    assert all(len(lookup[uid].program) == 1 for uid in curriculum[:192])
    assert Counter(lookup[uid].family for uid in curriculum[:192]) == {family: 64 for family in FAMILIES}
    assert Counter(len(lookup[uid].program) for uid in curriculum[192:]) == {1: 192, 2: 1152}
    assert {len(lookup[uid].program) for uid in curriculum[192:384]} == {1, 2}
    assert len(flat) == 1536 and all(len(batch) == 4 for schedule in schedules.values() for batch in schedule)
    legacy = [uid for batch in schedules["legacy96_budget"] for uid in batch]
    assert Counter(legacy) == {uid: 16 for uid in old}
    audit = lane.schedule_audit(corpus, schedules, design)
    assert audit["flat_mixed"]["family_depth_exposures"] == audit["primitive_first"]["family_depth_exposures"]
    assert audit["legacy96_budget"]["family_depth_exposures"] != audit["flat_mixed"]["family_depth_exposures"]
    assert not set(flat) & {task.uid for task in corpus["fallback_validation"]}


def test_real_historical_failure_and_exact_first24_exposures(design):
    original = lane.validate_design(design)
    corpus = make_corpus(original, lane.SPLITS)
    schedules = lane.make_schedules(corpus, design)
    local_history(design)
    history = lane.verify_history(design, original, corpus, schedules)
    assert all(panel["passed"] for panel in history["feasibility"].values())
    assert history["fallback24"]["status"] == "rejected"
    changed = copy.deepcopy(schedules)
    changed["legacy96_budget"][0].reverse()
    with pytest.raises(ValueError, match="HISTORICAL_BUDGET"):
        lane.verify_history(design, original, corpus, changed)


def test_infeasible_original_gain_gate_stops_before_updates(design):
    original = lane.validate_design(design)
    corpus = make_corpus(original, lane.SPLITS)
    pairs = []
    for task in qualification_rows(corpus, original, "fallback_validation"):
        pairs.append({"uid": task.uid, "learner": {
            "prompt": lane.learner_prompt(task), "body_text": digits(task.answer),
            "token_ids": [ord(c) for c in digits(task.answer)] + [255],
        }})
    with pytest.raises(ValueError, match="GAIN_GATE_INFEASIBLE_NO_UPDATES"):
        lane.gate_feasibility(pairs, corpus, original, 255, [255])


@pytest.mark.parametrize("suffix,eos", [(" extra", True), ("", False), ("\n", True)])
def test_diagnostics_never_rescue_complete_prefix(design, suffix, eos):
    original = lane.validate_design(design)
    corpus = make_corpus(original, lane.SPLITS)
    task = corpus["train"][0]
    body = digits(task.answer) + suffix
    tokens = [ord(c) for c in body] + ([255] if eos else [])

    def generated(task, prompt):
        return {"prompt": prompt, "body_text": body, "token_ids": tokens, **grade_generation(task, tokens, body, 255, [255], 16)}

    record = lane.diagnostic_record(SimpleNamespace(generated_record=generated), task, corpus["demonstration"], forced=False)
    assert all(record["literal_position_correct"])
    assert not record["generation"]["correct"]
    lane.verify_records([task], [record], corpus, original, 255, [255])


def test_whole_sequence_probability_and_digit_offsets_use_original_targets(design, tmp_path):
    original = lane.validate_design(design)
    runner = tiny_runner(original, tmp_path)
    task = runner.corpus["train"][0]
    prompt = teacher_prompt(task, runner.corpus["demonstration"])
    result = lane.forced_diagnostics(runner, task, prompt)
    response = runner.target_tokens(task)
    with torch.no_grad():
        logits = runner.logits(runner.prompt_ids(prompt), response).float()
        expected = functional.log_softmax(logits, -1).gather(-1, response[:, None]).sum()
    assert result["gold_sequence_log_probability_including_eos"] == float(expected)
    assert 0 <= result["gold_sequence_probability_including_eos"] <= 1
    assert len(result["teacher_forced_position_correct"]) == 4
    assert response[-1] == runner.eos


def test_cli_failure_stops_before_loading_and_preserves_seal(design, tmp_path, monkeypatch):
    called = []

    def failed(*args):
        raise ValueError("DEVICE4_HISTORICAL_EVIDENCE_CHANGED")

    monkeypatch.setattr(lane, "verify_history", failed)
    monkeypatch.setattr(lane, "load_runner", lambda *args: called.append(args))
    config = lane.ROOT / "configs/device4_curriculum_run.json"
    monkeypatch.setattr(sys, "argv", ["device4_curriculum.py", "--config", str(config), "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="HISTORICAL_EVIDENCE_CHANGED"):
        lane.main()
    assert not called
    assert (tmp_path / "seal.json").exists()
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "training.json").exists()


def test_actual_cpu_updates_all_arms_and_fresh_process_final_qualification(design, tmp_path):
    original = lane.validate_design(design)
    full = make_corpus(original, lane.SPLITS)
    corpus = {"demonstration": full["demonstration"]}
    for split in ("train", "fallback_validation"):
        corpus[split] = [next(t for t in full[split] if t.family == family and len(t.program) == depth) for family in FAMILIES for depth in (1, 2)]
    original["qualification"].update(train_per_family=2, validation_per_family=2, maximum_p=1.0)
    recipe = design["training"]
    recipe.update(updates_per_arm=2, example_exposures_per_arm=8, checkpoints=[1, 2])
    output = tmp_path / "run"
    output.mkdir()
    runner = tiny_runner(original, output)
    runner.corpus = corpus
    recipe["expected_trainable_parameters"] = sum(p.numel() for p in runner.student_parameters)
    initial_hash = tensor_digest(runner.initial_adapter)
    initial_base = lane.frozen_digest(runner)
    ids = [task.uid for task in corpus["train"]]
    schedules = {
        "flat_mixed": [ids[:4], ids[2:6]],
        "primitive_first": [[ids[i] for i in (0, 2, 4, 2)], [ids[i] for i in (1, 3, 3, 5)]],
        "legacy96_budget": [ids[:4], ids[:4]],
    }
    lane.reset_teacher(runner)
    initial = lane.save_checkpoint(runner, output / "initial")
    lane.write_new(output / "seal.json", {"cpu_scaled_protocol": True})
    lane.write_new(output / "schedule.json", schedules)
    lane.write_new(output / "schedule_audit.json", lane.schedule_audit(corpus, schedules, design))
    baseline = lane.measure(runner, corpus["train"], output, "cpu/raw")
    lane.write_new(output / "raw_instruction_demo_train.json", baseline)
    arms = {arm: lane.train_arm(runner, design, arm, schedules[arm], output) for arm in lane.ARMS}
    assert all(arm["updates"] == 2 and arm["example_exposures"] == 8 for arm in arms.values())
    assert arms["flat_mixed"]["loss_token_exposures"] == arms["primitive_first"]["loss_token_exposures"] == 64
    assert all(arm["checkpoints"]["2"]["adapter"]["tensor_sha256"] != initial_hash for arm in arms.values())
    assert arms["flat_mixed"]["checkpoints"]["2"]["adapter"]["tensor_sha256"] != arms["primitive_first"]["checkpoints"]["2"]["adapter"]["tensor_sha256"]
    assert lane.frozen_digest(runner) == initial_base
    assert tensor_digest(adapter_parameters(runner.model, "student")) == initial_hash
    training = {
        "status": "trained_evaluation_pending", "pid": os.getpid(), "source_sha256": lane.source_hashes(), "design_sha256": digest(design),
        "original_protocol_sha256": digest(original), "dataset_sha256": digest(to_records(corpus)), "arms": arms,
        "snapshot_sha256": runner.snapshot, "eos_token_id": runner.eos, "special_token_ids": runner.special_ids,
        "initial": initial, "learner_updates": 0, "new_validation_predictions_observed_during_training": False,
        "test_or_composition_predictions_observed": False, "trainable_parameters": recipe["expected_trainable_parameters"],
        "files_sha256": {name: file_digest(output / name) for name in ("seal.json", "schedule.json", "schedule_audit.json", "raw_instruction_demo_train.json")},
    }
    lane.write_new(output / "training.json", training)
    with pytest.raises(ValueError, match="NOT_FRESH_PROCESS"):
        lane.verify_training(output, design, original, corpus, schedules)
    lane.write_new(tmp_path / "fixture.json", {"design": design, "original": original, "corpus": to_records(corpus), "schedules": schedules})
    script = tmp_path / "evaluate_tiny.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import device4_curriculum as lane\nfrom tasks import make_task\nfrom test_qualified import tiny_runner\n"
        "root=Path(sys.argv[1])\nfixture=json.loads((root/'fixture.json').read_text())\n"
        "corpus={split:[make_task(r['family'],r['split'],r['initial'],r['program']) for r in rows] for split,rows in fixture['corpus'].items()}\n"
        "def load_tiny(original,design,corpus,history,output):\n"
        "    runner=tiny_runner(original,output)\n    runner.corpus=corpus\n    return runner\n"
        "lane.load_runner=load_tiny\n(root/'evaluation').mkdir()\n"
        "lane.evaluate(fixture['design'],fixture['original'],corpus,fixture['schedules'],{},root/'run',root/'evaluation')\n"
    )
    child = subprocess.run(
        ["uv", "run", "--no-project", "--python", sys.executable, "python", str(script), str(tmp_path)],
        cwd=lane.ROOT, env={**os.environ, "PYTHONPATH": str(lane.ROOT)}, capture_output=True, text=True, check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "evaluation/qualification.json").read_text())
    assert result["pid"] != training["pid"]
    assert result["learner_updates"] == result["test_predictions"] == result["composition_predictions"] == 0
    assert all(arm["checkpoint_update"] == 2 and arm["reload_generation_parity"] for arm in result["arms"].values())
    assert all(arm["status"] == "rejected" and not arm["eligible_for_depth3_4_prerequisite"] for arm in result["arms"].values())
    training["pid"] = -1
    training["arms"]["flat_mixed"]["selected_checkpoint"] = 1
    (output / "training.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="ARM_BUDGET_OR_CHECKPOINT"):
        lane.verify_training(output, design, original, corpus, schedules)


def test_run_driver_completes_only_after_new_pid_qualification(tmp_path, monkeypatch):
    calls = []

    def child(command, check, cwd):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-project", "--python"]
        lane.write_new(tmp_path / "evaluation/qualification.json", {
            "status": "completed", "pid": os.getpid() + 1, "training_pid": os.getpid(),
            "arms": {}, "comparisons": {}, "claim_boundary": "cpu control flow",
        })

    monkeypatch.setattr(lane.subprocess, "run", child)
    lane.evaluate_child(tmp_path, {"protocol": "configs/device4_curriculum_protocol.json", "protocol_sha256": "fixture"})
    assert len(calls) == 1
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "completed"
