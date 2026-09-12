import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import device4_depth34 as lane
import pytest
import torch
from device4_depth34_contract import (
    DATA_SPEC,
    build_data,
    deep_gate,
    feasible_gate,
    load_data,
    train_schedule,
)
from generated_contract import (
    digest,
    file_digest,
    grade_generation,
    make_corpus,
    qualification_gate,
    qualification_rows,
    teacher_prompt,
)
from test_qualified import reference, tiny_runner

from run import adapter_parameters, tensor_digest
from tasks import FAMILIES, digits, to_records


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/device4_depth34_protocol.json").read_text())


def scripted_records(tasks, corpus, privileged, correct=True, eos=True):
    def generated(task, prompt):
        body = digits(task.answer) if correct else "incorrect"
        tokens = [ord(char) for char in body] + ([255] if eos else [])
        return {"prompt": prompt, "prompt_token_ids": [1], "body_text": body, "token_ids": tokens, **grade_generation(task, tokens, body, 255, [255], 16)}

    return [lane.upstream_lane.diagnostic_record(SimpleNamespace(generated_record=generated), task, corpus["demonstration"], privileged, forced=False) for task in tasks]


def test_sealed_data_unseen_programs_and_inputs_and_frozen_sources(design):
    _, original = lane.validate_design(design)
    corpus, audit = load_data(design, original)
    assert audit["rows_per_split"] == {"demonstration": 18, "train": 384, "validation": 144, "test": 288}
    assert audit["historical_unique_inputs_excluded"] == 790
    assert audit["historical_input_overlap"] == audit["cross_split_input_overlap"] == audit["cross_split_program_overlap"] == 0
    regenerated, _ = build_data(original, json.loads(json.dumps(DATA_SPEC, sort_keys=True)))
    assert regenerated["audit"] == audit
    for split, rows in corpus.items():
        for task in rows:
            assert task.answer == reference(task.family, task.initial, task.program)
        if split != "demonstration":
            for family in FAMILIES:
                for depth in (3, 4):
                    programs = {task.program for task in rows if task.family == family and len(task.program) == depth}
                    assert len(programs) == DATA_SPEC["program_counts"][str(depth)][split]
    poisoned = replace(corpus["train"][0], answer=(99, 99, 99, 99))
    assert teacher_prompt(poisoned, corpus["demonstration"]) == teacher_prompt(corpus["train"][0], corpus["demonstration"])
    schedule = train_schedule(corpus, design)
    assert len(schedule) == 384 and all(len(batch) == 4 for batch in schedule)
    assert Counter(uid for batch in schedule for uid in batch) == {task.uid: 4 for task in corpus["train"]}
    assert len(lane.train_probes(corpus, design)) == 24
    for handoff_name in ("device4_curriculum_handoff.json", "coverage_teacher_handoff.json", "coverage_learner_handoff.json"):
        frozen = json.loads((lane.ROOT / handoff_name).read_text())["staging_files_sha256"]
        assert all(file_digest(lane.ROOT / name) == expected for name, expected in frozen.items())
    for stage in ("run", "qualify", "learn", "evaluate"):
        config = json.loads((lane.ROOT / f"configs/device4_depth34_{stage}.json").read_text())
        assert config["protocol_sha256"] == digest(design)


def test_every_family_depth_gate_is_required_and_no_prefix_rescue(design):
    _, original = lane.validate_design(design)
    corpus, _ = load_data(design, original)
    raw = {split: scripted_records(corpus[split], corpus, False, correct=False) for split in ("train", "validation")}
    teacher = {split: scripted_records(corpus[split], corpus, True) for split in raw}
    assert feasible_gate(corpus, original, raw, 255, [255], design["qualification"])["passed"]
    result = deep_gate(corpus, original, raw, teacher, 255, [255], design["qualification"])
    assert result["passed"] and len(result["panels"]) == 12
    for index in range(4):
        task = corpus["validation"][index]
        teacher["validation"][index] = scripted_records([task], corpus, True, eos=False)[0]
    failed = deep_gate(corpus, original, raw, teacher, 255, [255], design["qualification"])
    assert not failed["passed"] and failed["panels"]["A/depth3/validation"]["teacher_accuracy"] == 20 / 24
    all_correct_raw = {split: scripted_records(corpus[split], corpus, False) for split in raw}
    assert not feasible_gate(corpus, original, all_correct_raw, 255, [255], design["qualification"])["passed"]


def test_unqualified_upstream_stops_before_model_loading(design, tmp_path, monkeypatch):
    upstream, original = lane.validate_design(design)
    corpus, audit = load_data(design, original)
    calls = []
    monkeypatch.setattr(lane, "verify_upstream", lambda *args: ({"eligible": []}, {}, {}))
    monkeypatch.setattr(lane, "load_runner", lambda *args: calls.append(args))
    result = lane.qualify(design, upstream, original, corpus, audit, tmp_path)
    assert result["status"] == "rejected_no_eligible_depth12_teacher" and result["learner_updates"] == 0
    assert not calls


def test_upstream_receipt_recomputes_original_gates_and_selection(design, tmp_path, monkeypatch):
    upstream, original = lane.validate_design(design)
    corpus = make_corpus(original, lane.upstream_lane.SPLITS)
    tasks = qualification_rows(corpus, original, "fallback_validation")
    raw = scripted_records(tasks, corpus, False, correct=False)
    measured = scripted_records(tasks, corpus, True)
    training_root, root = tmp_path / "training", tmp_path / "qualification"
    design["upstream_training_dir"], design["upstream_qualification_dir"] = str(training_root), str(root)
    training = {"pid": 101, "eos_token_id": 255, "special_token_ids": [255], "initial": {"tensor_sha256": "raw"}, "snapshot_sha256": {"cpu": "fixture"}, "arms": {}}
    panels, arms = {}, {}
    for arm in lane.upstream_lane.ARMS:
        folder = training_root / arm / "checkpoint384"
        adapter = folder / "teacher"
        adapter.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (adapter / name).write_bytes(f"cpu-receipt-{arm}-{name}".encode())
        checkpoint = {"path": str(adapter), "tensor_sha256": arm, "files_sha256": {name: file_digest(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")}}
        lane.write_new(folder / "train_diagnostics.json", scripted_records(corpus["train"], corpus, True))
        training["arms"][arm] = {"base_before_sha256": "base", "checkpoints": {"384": {"adapter": checkpoint}}}
        pairs = [{"uid": task.uid, "family": task.family, "split": task.split, "learner": baseline["generation"], "teacher": trained["generation"]} for task, baseline, trained in zip(tasks, raw, measured, strict=True)]
        gate = qualification_gate(pairs, corpus, original, 255, [255], "fallback_validation")
        assert gate["passed"]
        panels[arm] = {"pairs": pairs, "diagnostics": measured}
        arms[arm] = {"gate": gate, "summary": {split: lane.summarize([row for row in measured if row["split"] == split]) for split in ("train", "fallback_validation")}, "status": "qualified", "depth_accuracy_check_passed": True, "eligible_for_depth3_4_prerequisite": True, "checkpoint": checkpoint, "checkpoint_update": 384, "reload_generation_parity": True, "learner_updates": 0}
    lane.write_new(training_root / "training.json", training)
    lane.write_new(root / "panels.json", panels)
    qualified = {"status": "completed", "source_sha256": lane.upstream_lane.source_hashes(), "design_sha256": digest(upstream), "training_sha256": file_digest(training_root / "training.json"), "training_pid": 101, "pid": 202, "learner_updates": 0, "test_predictions": 0, "composition_predictions": 0, "arms": arms, "panels_sha256": file_digest(root / "panels.json")}
    lane.write_new(root / "qualification.json", qualified)
    monkeypatch.setattr(lane.upstream_lane, "verify_history", lambda *args: {})
    monkeypatch.setattr(lane.upstream_lane, "verify_training", lambda *args: training)
    proof, _, _ = lane.verify_upstream(design, upstream, original)
    assert proof["eligible"] == list(lane.upstream_lane.ARMS)
    qualified["arms"]["flat_mixed"]["eligible_for_depth3_4_prerequisite"] = False
    (root / "qualification.json").write_text(json.dumps(qualified))
    with pytest.raises(ValueError, match="UPSTREAM_ELIGIBILITY_RECOMPUTATION_FAILED"):
        lane.verify_upstream(design, upstream, original)


def model_records(runner, tasks, corpus, privileged, correct):
    def generated(task, prompt):
        body = digits(task.answer) if correct else "incorrect"
        tokens = runner.tokenizer.encode(body, add_special_tokens=False) + [runner.eos]
        return {"prompt": prompt, "prompt_token_ids": runner.prompt_ids(prompt)[0].tolist(), "token_ids": tokens, "body_text": body, "raw_text": runner.tokenizer.decode(tokens, clean_up_tokenization_spaces=False), **grade_generation(task, tokens, body, runner.eos, runner.special_ids, runner.config["max_new_tokens"])}

    return [lane.upstream_lane.diagnostic_record(SimpleNamespace(generated_record=generated), task, corpus["demonstration"], privileged, forced=False) for task in tasks]


def cpu_fixture(design, tmp_path):
    upstream, original = lane.validate_design(design)
    full, _ = load_data(design, original)
    corpus = {"demonstration": full["demonstration"], **{split: [next(task for task in full[split] if task.family == family and len(task.program) == depth) for family in FAMILIES for depth in (3, 4)] for split in ("train", "validation", "test")}}
    design["qualification"]["maximum_p"] = 1.0
    recipe = design["training"]
    recipe.update(updates_per_method=2, examples_per_update=3, example_exposures_per_method=6, exposures_per_family_depth=1, checkpoint_updates=[1, 2], probe_rows_per_family_depth=1, rank=2, alpha=4)
    original["lora_rank"], original["lora_alpha"] = 2, 4
    runner = tiny_runner(original, tmp_path / "source")
    recipe["expected_trainable_parameters"] = sum(p.numel() for p in runner.student_parameters)
    audit = {"cpu_fixture_only": True, "dataset_sha256": digest(to_records(corpus))}
    proof = {
        "qualification_sha256": "cpu-original-qualification", "training_sha256": "cpu-original-training",
        "eligible": ["flat_mixed", "primitive_first"], "checkpoints": {},
        "initial_tensor_sha256": tensor_digest(adapter_parameters(runner.model, "student")),
        "snapshot_sha256": runner.snapshot, "eos_token_id": runner.eos, "special_token_ids": runner.special_ids,
        "base_tensor_sha256": lane.frozen_digest(runner),
    }
    for arm, amount in (("flat_mixed", 0.01), ("primitive_first", 0.02)):
        with torch.no_grad():
            for name, value in adapter_parameters(runner.model, "teacher").items():
                value.copy_(runner.initial_adapter[name])
                if "lora_B" in name:
                    value.add_(amount)
        proof["checkpoints"][arm] = lane.upstream_lane.save_checkpoint(runner, tmp_path / "teachers" / arm)
    raw = {split: model_records(runner, corpus[split], corpus, False, False) for split in ("train", "validation")}
    panels = {"raw": raw, "teachers": {}}
    results = {}
    for arm in proof["eligible"]:
        measured = {split: model_records(runner, corpus[split], corpus, True, True) for split in raw}
        panels["teachers"][arm] = measured
        gate = deep_gate(corpus, original, raw, measured, runner.eos, runner.special_ids, design["qualification"])
        assert gate["passed"]
        results[arm] = {"gate": gate, "status": "qualified", "checkpoint": proof["checkpoints"][arm], "summary": {split: lane.summarize(rows) for split, rows in measured.items()}}
    root = tmp_path / "qualified"
    lane.write_new(root / "qualification_panels.json", panels)
    qualified = {
        "status": "qualified", "pid": os.getpid() + 100, "design_sha256": digest(design), "source_sha256": lane.source_hashes(),
        "dataset": audit, "upstream": proof, "teacher_optimizer_updates": 0, "learner_updates": 0, "test_predictions": 0,
        "eligible": proof["eligible"], "arms": results, "feasibility": feasible_gate(corpus, original, raw, runner.eos, runner.special_ids, design["qualification"]),
        "panels_sha256": file_digest(root / "qualification_panels.json"),
    }
    lane.write_new(root / "qualification.json", qualified)
    return {"design": design, "upstream": upstream, "original": original, "corpus": corpus, "audit": audit, "proof": proof}


def test_cannot_drop_a_passing_teacher_or_select_earlier_checkpoint(design, tmp_path):
    fixture = cpu_fixture(design, tmp_path)
    args = [fixture[name] for name in ("design", "original", "corpus", "audit")]
    qualified = lane.read_certificate(tmp_path / "qualified", *args)
    assert len(lane.methods_for(qualified)) == 3
    qualified["eligible"] = ["flat_mixed"]
    (tmp_path / "qualified/qualification.json").write_text(json.dumps(qualified))
    with pytest.raises(ValueError, match="ALL_PASSING_TEACHERS_MUST_BE_CARRIED"):
        lane.read_certificate(tmp_path / "qualified", *args)


def test_rejected_deeper_teacher_precedes_learner_model_load(design, tmp_path, monkeypatch):
    fixture = cpu_fixture(design, tmp_path)
    path = tmp_path / "qualified/qualification.json"
    qualified = json.loads(path.read_text())
    qualified["status"] = "rejected"
    path.write_text(json.dumps(qualified))
    monkeypatch.setattr(lane, "verify_upstream", lambda *args: (fixture["proof"], {}, {}))
    calls = []
    monkeypatch.setattr(lane, "load_runner", lambda *args: calls.append(args))
    arguments = [fixture[name] for name in ("upstream", "original", "corpus", "audit")]
    with pytest.raises(ValueError, match="QUALIFIED_TEACHER_REQUIRED_BEFORE_LEARNER"):
        lane.train(fixture["design"], {"qualification_dir": str(tmp_path / "qualified")}, *arguments, tmp_path / "learning")
    assert not calls and not (tmp_path / "learning").exists()


def test_actual_cpu_context_updates_shared_oracle_and_fresh_teacher_absent_evaluation(design, tmp_path, monkeypatch):
    fixture = cpu_fixture(design, tmp_path)
    monkeypatch.setattr(lane, "verify_upstream", lambda *args: (fixture["proof"], {}, {}))
    monkeypatch.setattr(lane, "load_runner", lambda original, upstream, corpus, history, output: tiny_runner(original, output))
    output = tmp_path / "learning"
    output.mkdir()
    arguments = [fixture[name] for name in ("upstream", "original", "corpus", "audit")]
    training = lane.train(fixture["design"], {"qualification_dir": str(tmp_path / "qualified")}, *arguments, output)
    assert training["total_learner_updates"] == 6 and len(training["methods"]) == 3
    assert training["teacher_optimizer_updates"] == 0 and training["test_predictions_during_learning"] == 0
    assert all(method["updates"] == 2 and method["example_exposures"] == 6 for method in training["methods"].values())
    assert training["methods"]["oracle_sft"]["teacher_tokens"] == 0
    assert training["methods"]["oracle_sft"]["rollout_tokens"] == 0
    for method, result in training["methods"].items():
        assert result["checkpoints"]["2"]["adapter"]["tensor_sha256"] != training["initial"]["tensor_sha256"]
        if method != "oracle_sft":
            assert result["teacher_tokens"] > result["rollout_tokens"] > 0
            rows = json.loads((output / method / "rows.json").read_text())
            assert all(row["diagnostics"]["support"] == "full_vocabulary" for row in rows)
    for spec in fixture["proof"]["checkpoints"].values():
        lane.verify_adapter(spec)
    with pytest.raises(ValueError, match="NOT_FRESH_PROCESS"):
        lane.verify_training(output, fixture["design"], fixture["original"], fixture["corpus"], fixture["audit"])
    fixture["corpus"] = to_records(fixture["corpus"])
    lane.write_new(tmp_path / "fixture.json", fixture)
    (tmp_path / "teachers").rename(tmp_path / "teachers-unavailable")
    (tmp_path / "qualified").rename(tmp_path / "qualification-unavailable")
    script = tmp_path / "evaluate_cpu.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import device4_depth34 as lane\nfrom tasks import Task\nfrom test_qualified import tiny_runner\n"
        "root=Path(sys.argv[1])\nfixture=json.loads((root/'fixture.json').read_text())\n"
        "corpus={split:[Task(row['uid'],row['family'],row['split'],tuple(row['initial']),tuple(row['program']),tuple(row['answer'])) for row in rows] for split,rows in fixture['corpus'].items()}\n"
        "def load_tiny(original,output):\n"
        "    runner=tiny_runner(original,output)\n    runner.model.set_adapter('student',inference_mode=True)\n"
        "    runner.model.delete_adapter('teacher')\n    return runner\n"
        "def forbidden(*args,**kwargs):\n    raise AssertionError('External teacher prerequisite/artifact access during evaluation')\n"
        "lane.load_evaluator=load_tiny\nlane.verify_upstream=forbidden\n"
        "(root/'evaluation').mkdir()\n"
        "lane.evaluate(fixture['design'],{'training_dir':str(root/'learning')},fixture['original'],corpus,fixture['audit'],root/'evaluation')\n"
        "training=json.loads((root/'learning/training.json').read_text())\n"
        "training['methods']['oracle_sft']['selected_checkpoint']=1\n"
        "(root/'learning/training.json').write_text(json.dumps(training))\n"
        "try:\n    lane.verify_training(root/'learning',fixture['design'],fixture['original'],corpus,fixture['audit'])\n"
        "except ValueError as error:\n    assert 'FIXED_FINAL_CHANGED' in str(error)\n"
        "else:\n    raise AssertionError('Earlier checkpoint accepted')\n"
    )
    child = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", str(script), str(tmp_path)], cwd=lane.ROOT, env={**os.environ, "PYTHONPATH": str(lane.ROOT)}, capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "evaluation/result.json").read_text())
    assert result["pid"] != training["pid"] and result["training_pid"] == training["pid"]
    assert result["fresh_process_reload_generation_parity"] and result["resident_adapters"] == ["student"]
    assert not result["teacher_weights_loaded"] and not result["external_teacher_artifacts_read"]
    assert result["qualified_teachers"] == ["flat_mixed", "primitive_first"]


def test_conditional_default_driver_launches_learning_then_evaluation(tmp_path, monkeypatch):
    calls = []

    def child(command, cwd, check):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-project", "--python"]
        lane.write_new(tmp_path / "learning/result.json", {"status": "completed", "training_pid": os.getpid() + 1, "evaluation_pid": os.getpid() + 2, "comparisons": {}, "qualified_teachers": ["flat_mixed"], "learner_updates": 768, "claim_boundary": "CPU driver fixture"})

    monkeypatch.setattr(lane.subprocess, "run", child)
    lane.child_stage("learn", tmp_path, {"protocol": "configs/device4_depth34_protocol.json", "protocol_sha256": "fixture"})
    assert len(calls) == 1 and json.loads((tmp_path / "result.json").read_text())["learner_updates"] == 768


def test_validate_only_needs_no_upstream_gpu_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["device4_depth34.py", "--config", str(lane.ROOT / "configs/device4_depth34_run.json"), "--output-dir", str(tmp_path), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validation.json").read_text())["gpu_qualified"]
