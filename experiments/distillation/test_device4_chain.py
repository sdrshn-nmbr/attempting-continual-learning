import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import device4_chain as lane
import pytest
import torch
from device4_chain_contract import verify_chains
from generated_contract import (
    digest,
    file_digest,
    make_corpus,
    qualification_gate,
    qualification_rows,
)
from test_device4_depth34 import model_records, scripted_records
from test_qualified import reference, tiny_runner

from run import adapter_parameters, tensor_digest
from tasks import FAMILIES, digits, to_records


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/device4_chain_protocol.json").read_text())


def cpu_runner(original, output):
    output.mkdir(parents=True, exist_ok=True)
    return tiny_runner(original, output)


def emission(runner, prompt, state, eos=True):
    body = digits(state)
    ids = runner.tokenizer.encode(body, add_special_tokens=False) + ([runner.eos] if eos else [])
    return {"prompt": prompt, "prompt_token_ids": runner.prompt_ids(prompt)[0].tolist(), "token_ids": ids, "body_text": body, "raw_text": runner.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)}


def scripted_chain(runner, task, wrong_final=False, invalid_final=False):
    state, steps = task.initial, []
    for index, command in enumerate(task.program):
        prompt = lane.step_prompt(task, state, index, runner.corpus["demonstration"])
        assert prompt is not None
        following = reference(task.family, state, (command,))
        if wrong_final and index == len(task.program) - 1:
            following = ((following[0] + 1) % 10, *following[1:])
        generation = emission(runner, prompt, following, eos=not (invalid_final and index == len(task.program) - 1))
        steps.append({"index": index, "command": command, "input_state": list(state), "generation": generation})
        state = following
    return lane.chain_record(task, steps, runner.corpus["demonstration"], runner.config, runner.eos, runner.special_ids)


def test_sealed_source_data_budget_and_all_original_files_immutable(design):
    old, _, original = lane.validate_design(design)
    corpus, audit = lane.native.load_data(old, original)
    assert audit["rows_per_split"] == {"demonstration": 18, "train": 384, "validation": 144, "test": 288}
    assert audit["cross_split_program_overlap"] == audit["cross_split_input_overlap"] == 0
    for family in FAMILIES:
        schedule = lane.train_schedule(corpus, family, design["training"])
        assert len(schedule) == 128 and all(len(batch) == 4 for batch in schedule)
        assert Counter(uid for batch in schedule for uid in batch) == {task.uid: 4 for task in lane.family_rows(corpus, "train", family)}
        assert len(lane.family_rows(corpus, "train", family)) == 128
        assert len(lane.family_rows(corpus, "test", family)) == 96
    for name in ("device4_curriculum_handoff.json", "device4_recovery_handoff.json", "device4_depth34_handoff.json", "coverage_learner_handoff.json"):
        hashes = json.loads((lane.ROOT / name).read_text())["staging_files_sha256"]
        assert all(file_digest(lane.ROOT / path) == value for path, value in hashes.items())
    for path in (lane.ROOT / "configs").glob("device4_chain_*.json"):
        if path.name != "device4_chain_protocol.json":
            assert json.loads(path.read_text())["protocol_sha256"] == digest(design)
    changed = copy.deepcopy(design["teacher_source"])
    changed["training_dir"] = "/mnt/shared/cl-portfolio/runs/distillation-device4-curriculum-20260912"
    with pytest.raises(ValueError, match="CANNOT_IMPERSONATE_INTERRUPTED_RUN"):
        lane.recovered.validate_source(changed)


def test_chain_routes_emitted_state_without_gold_repair_and_aborts(design, tmp_path, monkeypatch):
    old, _, original = lane.validate_design(design)
    corpus, _ = lane.native.load_data(old, original)
    runner = cpu_runner(original, tmp_path)
    runner.corpus = corpus
    task = corpus["train"][0]
    wrong = (9, 9, 9, 9)
    calls = []

    def generated(model, prompt):
        calls.append(prompt)
        return emission(model, prompt, wrong, eos=len(calls) != 2)

    monkeypatch.setattr(lane, "emitted_generation", generated)
    record = lane.execute_chain(runner, replace(task, answer=(88, 88, 88, 88)))
    assert len(calls) == 2 and record["aborted"] == "malformed_or_missing_native_eos"
    assert record["steps"][1]["input_state"] == list(wrong)
    assert calls[1] == lane.step_prompt(task, wrong, 1, corpus["demonstration"])
    assert not record["correct"] and record["final_generation"] is None
    assert lane.step_prompt(task, task.initial, 0, corpus["demonstration"]) == lane.step_prompt(replace(task, answer=()), task.initial, 0, corpus["demonstration"])
    collision = next(row.initial for row in corpus["demonstration"] if row.family == task.family)
    monkeypatch.setattr(lane, "emitted_generation", lambda model, prompt: emission(model, prompt, collision))
    record = lane.execute_chain(runner, task)
    assert record["aborted"] == "demonstration_input_collision" and record["model_calls"] == 1
    assert record["steps"][1]["generation"] is None


def test_recovered_source_qualification_is_recomputed_under_new_contract(design, tmp_path, monkeypatch):
    spec = design["teacher_source"]
    spec["training_dir"], spec["qualification_dir"] = str(tmp_path / "recovery"), str(tmp_path / "recovery/evaluation")
    recovery_design, upstream, original, snapshot = lane.recovered.validate_source(spec)
    corpus = make_corpus(original, lane.native.upstream_lane.SPLITS)
    tasks = qualification_rows(corpus, original, "fallback_validation")
    raw, measured = scripted_records(tasks, corpus, False, correct=False), scripted_records(tasks, corpus, True)
    root = Path(spec["qualification_dir"])
    training = {"pid": 101, "eos_token_id": 255, "special_token_ids": [255], "initial": {"tensor_sha256": "cpu_initial"}, "snapshot_sha256": {"cpu": True}, "base_after_sha256": "cpu_base", "arms": {}}
    panels, arms = {}, {}
    for arm in lane.native.upstream_lane.ARMS:
        destination = tmp_path / ("original" if arm != "legacy96_budget" else "recovery") / arm
        path = destination / "train_diagnostics.json"
        lane.write_new(path, scripted_records(corpus["train"], corpus, True))
        checkpoint = {"path": str(destination / "teacher"), "tensor_sha256": f"cpu_{arm}"}
        origin = "retained_complete_original_arm" if arm != "legacy96_budget" else "resumed_original_adapter_and_optimizer"
        training["arms"][arm] = {"origin": origin, "selected_checkpoint": 384, "checkpoints": {"384": {"adapter": checkpoint, "train_diagnostics_path": str(path), "train_diagnostics_sha256": file_digest(path)}}}
        pairs = [{"uid": task.uid, "family": task.family, "split": task.split, "learner": baseline["generation"], "teacher": trained["generation"]} for task, baseline, trained in zip(tasks, raw, measured, strict=True)]
        gate = qualification_gate(pairs, corpus, original, 255, [255], "fallback_validation")
        assert gate["passed"]
        panels[arm] = {"pairs": pairs, "diagnostics": measured}
        arms[arm] = {"gate": gate, "summary": {split: lane.summarize([row for row in measured if row["split"] == split]) for split in ("train", "fallback_validation")}, "status": "qualified", "depth_accuracy_check_passed": True, "eligible_for_depth3_4_prerequisite": True, "checkpoint": checkpoint, "checkpoint_update": 384, "reload_generation_parity": True, "learner_updates": 0, "origin": origin}
    lane.write_new(Path(spec["training_dir"]) / "training.json", training)
    lane.write_new(root / "panels.json", panels)
    qualified = {"status": "completed", "source_sha256": lane.recovered.recovery.source_hashes(), "design_sha256": digest(recovery_design), "original_design_sha256": digest(upstream), "training_sha256": file_digest(Path(spec["training_dir"]) / "training.json"), "training_pid": 101, "pid": 202, "interrupted_execution_sha256": snapshot["execution_sha256"], "new_optimizer_updates": 288, "learner_updates": 0, "test_predictions": 0, "composition_predictions": 0, "arms": arms, "panels_sha256": file_digest(root / "panels.json")}
    lane.write_new(root / "qualification.json", qualified)
    monkeypatch.setattr(lane.recovered.recovery.original_lane, "verify_history", lambda *args: {})
    monkeypatch.setattr(lane.recovered.recovery, "verify_recovered_training", lambda *args: training)
    proof, _, _ = lane.recovered.verify_source(spec)
    assert proof["eligible"] == list(lane.native.upstream_lane.ARMS)
    assert proof["source_contract"] == lane.recovered.recovery.CONTRACT
    assert "/original/" in proof["train_diagnostics"]["flat_mixed"]["path"]
    assert "/recovery/" in proof["train_diagnostics"]["legacy96_budget"]["path"]
    qualified["arms"]["flat_mixed"]["eligible_for_depth3_4_prerequisite"] = False
    (root / "qualification.json").write_text(json.dumps(qualified))
    with pytest.raises(ValueError, match="ELIGIBILITY_RECOMPUTATION_FAILED"):
        lane.recovered.verify_source(spec)
    qualified["source_sha256"] = lane.native.upstream_lane.source_hashes()
    (root / "qualification.json").write_text(json.dumps(qualified))
    with pytest.raises(ValueError, match="QUALIFICATION_RECEIPT_CHANGED"):
        lane.recovered.verify_source(spec)


@pytest.fixture
def fixture(design, tmp_path):
    old, upstream, original = lane.validate_design(design)
    full, _ = lane.native.load_data(old, original)
    corpus = {"demonstration": full["demonstration"], **{split: [row for family in FAMILIES for depth in (3, 4) for row in [t for t in full[split] if t.family == family and len(t.program) == depth][:8 if split != "test" else 2]] for split in ("train", "validation", "test")}}
    original["lora_rank"], original["lora_alpha"] = 2, 4
    design["training"].update(updates_per_method=8, example_exposures_per_method=32, epochs=2, checkpoint_updates=[4, 8], probe_rows_per_depth=1, rank=2, alpha=4)
    runner = cpu_runner(original, tmp_path / "source")
    runner.corpus = corpus
    design["training"]["expected_trainable_parameters"] = sum(p.numel() for p in runner.student_parameters)
    audit = {"cpu_fixture": True, "dataset_sha256": digest(to_records(corpus))}
    old_corpus = make_corpus(original, lane.native.upstream_lane.SPLITS)
    primitives = lane.primitive_rows(old_corpus)
    raw = {split: model_records(runner, rows, old_corpus, False, False) for split, rows in primitives.items()}
    candidates = {}
    for candidate in lane.CANDIDATES:
        candidates[candidate] = {split: [model_records(runner, [task], old_corpus, True, task.family == "A" and candidate in {"raw_privileged", "flat_mixed"})[0] for task in rows] for split, rows in primitives.items()}
    primitive = lane.gates_by_family(primitives, raw, candidates, original, old_corpus, runner.eos, runner.special_ids, design["qualification"])
    proof = {"source_spec": design["teacher_source"], "initial_tensor_sha256": tensor_digest(adapter_parameters(runner.model, "student")), "snapshot_sha256": runner.snapshot, "eos_token_id": runner.eos, "special_token_ids": runner.special_ids, "base_tensor_sha256": lane.native.frozen_digest(runner)}
    panels = {"primitive_raw": raw, "primitive_candidates": candidates, "raw_full_program": {}, "chains": {family: {} for family in FAMILIES}}
    feasibility, pairs, eligible, caches = {}, {}, {}, {}
    root = tmp_path / "qualified"
    for family in FAMILIES:
        admitted = [candidate for candidate in lane.CANDIDATES if primitive[candidate][family]["passed"]]
        feasibility[family] = None
        pairs[family], eligible[family] = {}, []
        if admitted:
            panels["raw_full_program"][family] = {split: model_records(runner, lane.family_rows(corpus, split, family), corpus, False, False) for split in ("train", "validation")}
            feasibility[family] = lane.family_feasibility(corpus, family, panels["raw_full_program"][family], original, runner.eos, runner.special_ids, design["qualification"])
        for candidate in lane.CANDIDATES:
            gate = None
            if candidate in admitted:
                records = {split: [scripted_chain(runner, task, wrong_final=candidate == "flat_mixed" and split == "train" and index == 0) for index, task in enumerate(lane.family_rows(corpus, split, family))] for split in ("train", "validation")}
                panels["chains"][family][candidate] = records
                gate = lane.family_gate(corpus, family, panels["raw_full_program"][family], records, original, runner.eos, runner.special_ids, design["qualification"])
            result = lane.pair_result(candidate in admitted, feasibility[family], gate)
            pairs[family][candidate] = result
            if result["status"] == "qualified":
                eligible[family].append(candidate)
                path = f"caches/{family}/{candidate}.json"
                lane.write_new(root / path, lane.cache_outputs(lane.family_rows(corpus, "train", family), records["train"]))
                caches[path] = file_digest(root / path)
    lane.write_new(root / "qualification_panels.json", panels)
    receipt = {"status": "qualified", "pid": os.getpid() + 100, "design_sha256": digest(design), "source_sha256": lane.source_hashes(), "dataset": audit, "upstream": proof, "primitive": primitive, "feasibility": feasibility, "pairs": pairs, "eligible": eligible, "cache_files_sha256": caches, "panels_sha256": file_digest(root / "qualification_panels.json"), "teacher_optimizer_updates": 0, "learner_updates": 0, "test_predictions": 0}
    lane.write_new(root / "qualification.json", receipt)
    assert eligible == {"A": ["raw_privileged", "flat_mixed"], "B": [], "C": []}
    return {"design": design, "old": old, "upstream": upstream, "original": original, "corpus": corpus, "old_corpus": old_corpus, "audit": audit, "proof": proof, "candidates": candidates, "panels": panels}


def test_every_qualifier_retained_and_wrong_train_answers_not_substituted(fixture, tmp_path):
    f = fixture
    receipt, _ = lane.read_certificate(tmp_path / "qualified", f["design"], f["original"], f["corpus"], f["audit"])
    assert len(lane.methods_for(receipt, "A")) == 3 and lane.methods_for(receipt, "B") == {"oracle_sft": None}
    cache = json.loads((tmp_path / "qualified/caches/A/flat_mixed.json").read_text())
    assert sum(not row["teacher_correct"] for row in cache) == 1
    cache[0]["response_token_ids"] = cache[1]["response_token_ids"]
    (tmp_path / "qualified/caches/A/flat_mixed.json").write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="CACHED_EMISSIONS_CHANGED_OR_GOLD_SUBSTITUTED"):
        lane.read_certificate(tmp_path / "qualified", f["design"], f["original"], f["corpus"], f["audit"])


def test_cannot_drop_a_qualified_candidate(fixture, tmp_path):
    f = fixture
    path = tmp_path / "qualified/qualification.json"
    receipt = json.loads(path.read_text())
    receipt["eligible"]["A"] = ["flat_mixed"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="ALL_QUALIFIERS_AND_REJECTIONS"):
        lane.read_certificate(tmp_path / "qualified", f["design"], f["original"], f["corpus"], f["audit"])


def test_invalid_chain_trace_cannot_be_filtered_or_repaired(fixture, tmp_path):
    f = fixture
    runner = cpu_runner(f["original"], tmp_path / "trace")
    runner.corpus = f["corpus"]
    tasks = lane.family_rows(f["corpus"], "train", "A")
    records = copy.deepcopy(f["panels"]["chains"]["A"]["raw_privileged"]["train"])
    records[0] = scripted_chain(runner, tasks[0], invalid_final=True)
    with pytest.raises(ValueError, match="COMPLETE_TRAIN_OUTPUTS_REQUIRED_NO_FILTERING"):
        lane.cache_outputs(tasks, records)
    records = copy.deepcopy(f["panels"]["chains"]["A"]["raw_privileged"]["train"])
    records[0]["steps"][1]["input_state"][0] = (records[0]["steps"][1]["input_state"][0] + 1) % 10
    with pytest.raises(ValueError, match="STATE_MUST_EQUAL_PREVIOUS_EMISSION"):
        verify_chains(tasks, records, f["corpus"], f["original"], runner.eos, runner.special_ids)


def test_qualification_stage_caches_every_qualifier_and_reports_rejections(fixture, tmp_path, monkeypatch):
    f = fixture
    runner = cpu_runner(f["original"], tmp_path / "qualification-model")
    f["proof"]["checkpoints"] = {"flat_mixed": lane.native.upstream_lane.save_checkpoint(runner, tmp_path / "fixture-teacher")}
    monkeypatch.setattr(lane, "verify_source", lambda *args: (f["proof"], f["old_corpus"], {}, f["candidates"]))
    def load(original, upstream, rows, history, output):
        model = cpu_runner(original, output)
        model.corpus = rows
        return model
    def measure(model, rows, output, label, privileged=True, forced=True):
        return [model_records(model, [task], model.corpus, privileged, privileged and task.family == "A")[0] for task in rows]
    calls = []
    def chain(model, task):
        assert task.family == "A" and task.split in {"train", "validation"}
        calls.append(task.uid)
        return scripted_chain(model, task)
    monkeypatch.setattr(lane.native, "load_runner", load)
    monkeypatch.setattr(lane, "measure", measure)
    monkeypatch.setattr(lane, "execute_chain", chain)
    output = tmp_path / "actual-qualification-stage"
    result = lane.qualify(f["design"], f["old"], f["upstream"], f["original"], f["corpus"], f["audit"], output)
    assert result["eligible"] == {"A": ["raw_privileged", "flat_mixed"], "B": [], "C": []}
    assert len(calls) == 64 and result["teacher_optimizer_updates"] == result["learner_updates"] == result["test_predictions"] == 0
    assert all(item["status"] == "rejected_primitive_gate" for family in ("B", "C") for item in result["pairs"][family].values())
    verified, _ = lane.read_certificate(output, f["design"], f["original"], f["corpus"], f["audit"])
    assert verified == result and len(result["cache_files_sha256"]) == 2


def test_all_rejected_qualification_still_trains_only_oracle_reference(fixture, tmp_path, monkeypatch):
    f = fixture
    model = cpu_runner(f["original"], tmp_path / "rejected-fixture")
    primitives = lane.primitive_rows(f["old_corpus"])
    f["candidates"] = {name: {split: model_records(model, rows, f["old_corpus"], True, False) for split, rows in primitives.items()} for name in lane.CANDIDATES}
    panels = {"primitive_raw": f["panels"]["primitive_raw"], "primitive_candidates": f["candidates"], "raw_full_program": {}, "chains": {family: {} for family in FAMILIES}}
    source = tmp_path / "qualified"
    receipt = json.loads((source / "qualification.json").read_text())
    (source / "qualification_panels.json").write_text(json.dumps(panels))
    receipt.update(status="rejected", eligible={family: [] for family in FAMILIES}, feasibility={family: None for family in FAMILIES}, pairs={family: {name: lane.pair_result(False, None, None) for name in lane.CANDIDATES} for family in FAMILIES}, cache_files_sha256={}, panels_sha256=file_digest(source / "qualification_panels.json"))
    receipt["primitive"] = lane.gates_by_family(primitives, panels["primitive_raw"], f["candidates"], f["original"], f["old_corpus"], model.eos, model.special_ids, f["design"]["qualification"])
    (source / "qualification.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(lane, "verify_source", lambda *args: (f["proof"], f["old_corpus"], {}, f["candidates"]))
    def load(original, output):
        runner = cpu_runner(original, output)
        runner.model.set_adapter("student", inference_mode=True)
        runner.model.delete_adapter("teacher")
        return runner
    def forbidden(*args):
        raise AssertionError("Teacher model loaded or chain executed for oracle-only reference")
    monkeypatch.setattr(lane.native, "load_evaluator", load)
    monkeypatch.setattr(lane.native, "load_runner", forbidden)
    monkeypatch.setattr(lane, "execute_chain", forbidden)
    result = lane.train(f["design"], {"family": "C", "qualification_dir": str(source)}, f["old"], f["upstream"], f["original"], f["corpus"], f["audit"], tmp_path / "oracle-only")
    assert result["oracle_only"] and result["method_candidates"] == {"oracle_sft": None}
    assert result["total_learner_updates"] == result["oracle_reference_updates"] == 8
    assert result["teacher_output_updates"] == 0 and not result["teacher_model_loaded"]
    assert result["qualified_candidates"] == [] and result["target_equivalence"] == {}
    assert result["methods"]["oracle_sft"]["example_exposures"] == 32


def test_incomplete_qualification_blocks_even_oracle_reference(fixture, tmp_path, monkeypatch):
    f = fixture
    path = tmp_path / "qualified/qualification.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "running"
    path.write_text(json.dumps(receipt))
    def forbidden(*args):
        raise AssertionError("Source/model loaded before completed qualification")
    monkeypatch.setattr(lane, "verify_source", forbidden)
    monkeypatch.setattr(lane.native, "load_evaluator", forbidden)
    with pytest.raises(ValueError, match="QUALIFICATION_CERTIFICATE_CHANGED"):
        lane.train(f["design"], {"family": "B", "qualification_dir": str(tmp_path / "qualified")}, f["old"], f["upstream"], f["original"], f["corpus"], f["audit"], tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


@pytest.mark.parametrize("family", ["A", "B"])
def test_actual_updates_cached_errors_and_new_pid_teacher_absent_evaluation(fixture, tmp_path, monkeypatch, family):
    f = fixture
    monkeypatch.setattr(lane, "verify_source", lambda *args: (f["proof"], f["old_corpus"], {}, f["candidates"]))
    def load(original, output):
        runner = cpu_runner(original, output)
        runner.model.set_adapter("student", inference_mode=True)
        runner.model.delete_adapter("teacher")
        return runner
    monkeypatch.setattr(lane.native, "load_evaluator", load)
    root = tmp_path / "learning"
    root.mkdir()
    training = lane.train(f["design"], {"family": family, "qualification_dir": str(tmp_path / "qualified")}, f["old"], f["upstream"], f["original"], f["corpus"], f["audit"], root)
    assert training["total_learner_updates"] == (24 if family == "A" else 8)
    assert training["oracle_reference_updates"] == 8
    assert training["teacher_output_updates"] == (16 if family == "A" else 0)
    assert training["oracle_only"] == (family == "B")
    assert not training["teacher_model_loaded"] and training["external_chain_calls"] == 0
    hashes = {name: report["checkpoints"]["8"]["adapter"]["tensor_sha256"] for name, report in training["methods"].items()}
    if family == "A":
        assert training["target_equivalence"]["chain_output_flat_mixed"]["wrong_teacher_targets_retained"] == 1
        assert training["target_equivalence"]["chain_output_raw_privileged"]["token_identical_to_oracle"]
        assert hashes["oracle_sft"] == hashes["chain_output_raw_privileged"] != hashes["chain_output_flat_mixed"]
    else:
        assert training["method_candidates"] == {"oracle_sft": None} and training["target_equivalence"] == {}
    for method in training["methods"]:
        traces = json.loads((root / method / "rows.json").read_text())
        assert len(traces) == 32
        optimizer = torch.load(root / method / "checkpoint8/optimizer.pt", map_location="cpu", weights_only=True)
        assert {int(row["step"]) for row in optimizer["state"].values()} == {8}
    with pytest.raises(ValueError, match="TRAINING_METHOD_IDENTITY_OR_BUDGET"):
        lane.verify_training(root, f["design"], f["original"], f["corpus"], f["audit"], family)
    payload = {key: f[key] for key in ("design", "original", "audit")}
    payload["family"] = family
    payload["corpus"] = to_records(f["corpus"])
    lane.write_new(tmp_path / "fixture.json", payload)
    (tmp_path / "qualified").rename(tmp_path / "original-qualification-unavailable")
    script = tmp_path / "evaluate_cpu.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import device4_chain as lane\nfrom test_qualified import tiny_runner\nfrom tasks import Task\n"
        "root=Path(sys.argv[1])\nf=json.loads((root/'fixture.json').read_text())\n"
        "corpus={split:[Task(r['uid'],r['family'],r['split'],tuple(r['initial']),tuple(r['program']),tuple(r['answer'])) for r in rows] for split,rows in f['corpus'].items()}\n"
        "def load(original,output):\n    runner=tiny_runner(original,output)\n    runner.model.set_adapter('student',inference_mode=True)\n    runner.model.delete_adapter('teacher')\n    return runner\n"
        "def forbidden(*args,**kwargs):\n    raise AssertionError('External teacher/chain/qualification access during evaluation')\n"
        "lane.native.load_evaluator=load\nlane.verify_source=forbidden\nlane.recovered.verify_source=forbidden\nlane.execute_chain=forbidden\n"
        "(root/'evaluation').mkdir()\n"
        "result=lane.evaluate(f['design'],{'family':f['family'],'training_dir':str(root/'learning')},f['original'],corpus,f['audit'],root/'evaluation')\n"
        "assert result['teacher_weights_loaded'] is False and result['external_chain_calls']==0\n"
        "training=json.loads((root/'learning/training.json').read_text())\ntraining['methods']['oracle_sft']['selected_checkpoint']=4\n(root/'learning/training.json').write_text(json.dumps(training))\n"
        "try:\n    lane.verify_training(root/'learning',f['design'],f['original'],corpus,f['audit'],f['family'])\n"
        "except ValueError as error:\n    assert 'ARM_BUDGET_OR_FINAL_CHECKPOINT_CHANGED' in str(error)\n"
        "else:\n    raise AssertionError('Earlier checkpoint accepted')\n"
    )
    child = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(script), str(tmp_path)], cwd=lane.ROOT, env={**os.environ, "PYTHONPATH": str(lane.ROOT)}, capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "evaluation/result.json").read_text())
    assert result["pid"] != training["pid"] and result["resident_adapters"] == ["student"]
    assert result["qualified_candidates"] == (["raw_privileged", "flat_mixed"] if family == "A" else [])
    assert result["oracle_only"] == (family == "B") and result["oracle_reference_updates"] == 8
    assert result["fresh_process_reload_generation_parity"]


def test_auto_child_and_validate_only(design, tmp_path, monkeypatch):
    def child(command, cwd, check):
        assert command[:3] == ["uv", "run", "--no-project"]
        lane.write_new(tmp_path / "evaluation/result.json", {"status": "completed", "pid": os.getpid() + 1, "training_pid": os.getpid(), "family": "A", "comparisons": {}, "qualified_candidates": ["raw_privileged"], "learner_updates": 256, "oracle_reference_updates": 128, "teacher_output_updates": 128, "oracle_only": False, "claim_boundary": "cpu fixture"})
    monkeypatch.setattr(lane.subprocess, "run", child)
    lane.evaluate_child(tmp_path, {"protocol": "configs/device4_chain_protocol.json", "protocol_sha256": digest(design), "family": "A"})
    assert json.loads((tmp_path / "result.json").read_text())["learner_updates"] == 256
    monkeypatch.setattr(sys, "argv", ["device4_chain.py", "--config", str(lane.ROOT / "configs/device4_chain_qualify.json"), "--output-dir", str(tmp_path / "validate"), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validate/validation.json").read_text())["gpu_qualified"]
