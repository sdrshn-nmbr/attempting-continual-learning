import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import device4_sufficiency as lane
import pytest
import torch
from device4_sufficiency_contract import (
    acquisition,
    schedule_for,
    verify_cached_targets,
    verify_prefix,
)
from generated_contract import digest, file_digest
from test_qualified import tiny_runner

from run import adapter_parameters, tensor_digest
from tasks import digits, to_records


@pytest.fixture
def design():
    return json.loads((lane.ROOT / "configs/device4_sufficiency_protocol.json").read_text())


def cpu_runner(original, corpus, output):
    output.mkdir(parents=True, exist_ok=True)
    runner = tiny_runner(original, output)
    runner.model.delete_adapter("teacher")
    runner.corpus = corpus
    return runner


def test_real_all_four_origin_prefixes_and_frozen_sources(design):
    _, _, corpus, _ = lane.validate_design(design)
    root = lane.ROOT.parent.parent / "outputs/portfolio/followthrough-20260912/runs"
    for arm, spec in design["arms"].items():
        source = root / Path(spec["training_dir"]).name
        assert all(file_digest(source / name) == checksum for name, checksum in spec["receipt_files_sha256"].items())
        schedule = schedule_for(corpus["train"], spec["family"], design["training"])
        ledger = json.loads((source / spec["method"] / "ledger.json").read_text())
        prior = json.loads((source / "schedule.json").read_text())
        verify_prefix(schedule, prior, ledger, 128)
        assert len(schedule) == 1024 and sum(map(len, schedule[128:])) == 3584
        with pytest.raises(ValueError, match="PREFIX_NOT_EXACT"):
            verify_prefix(schedule, list(reversed(prior)), ledger, 128)
        for stage in ("learn", "evaluate"):
            config = json.loads((lane.ROOT / f"configs/device4_sufficiency_{stage}_{arm}.json").read_text())
            assert config["protocol_sha256"] == digest(design)
    for filename in ("device4_chain_handoff.json", "device4_recovery_handoff.json", "device4_native_composition_handoff.json"):
        handoff = json.loads((lane.ROOT / filename).read_text())
        assert all(file_digest(lane.ROOT / name) == value for name, value in handoff["staging_files_sha256"].items())


def test_negative_changed_source_gate_and_fixed_budget(design):
    bad = copy.deepcopy(design)
    bad["frozen_source_sha256"]["device4_chain.py"] = "changed"
    with pytest.raises(ValueError, match="FROZEN_SCIENTIFIC_CONTRACT_CHANGED"):
        lane.validate_design(bad)
    bad = copy.deepcopy(design)
    bad["training"]["final_update"] = 512
    with pytest.raises(ValueError, match="FROZEN_SCIENTIFIC_CONTRACT_CHANGED"):
        lane.validate_design(bad)
    bad = copy.deepcopy(design)
    bad["acquisition"]["selection"] = "early_stop"
    with pytest.raises(ValueError, match="ARMS_OR_MASTERY_GATE_CHANGED"):
        lane.validate_design(bad)


@pytest.fixture
def cpu_fixture(design, tmp_path):
    _, original, full, _ = lane.validate_design(design)
    original = copy.deepcopy(original)
    original.update(lora_rank=2, lora_alpha=4, max_new_tokens=8)
    corpus = {"demonstration": full["demonstration"], **{split: [row for depth in (3, 4) for row in [t for t in full[split] if t.family == "A" and len(t.program) == depth][:8 if split == "train" else 2]] for split in ("train", "test")}}
    recipe = {**design["training"], "start_update": 4, "final_update": 8, "checkpoints": [4, 6, 8], "train_rows": 16, "new_updates": 4, "new_example_exposures": 16}
    runner = cpu_runner(original, corpus, tmp_path / "initial")
    recipe["expected_trainable_parameters"] = sum(p.numel() for p in runner.student_parameters)
    proof = {"snapshot_sha256": runner.snapshot, "eos_token_id": runner.eos, "special_token_ids": runner.special_ids, "initial_tensor_sha256": tensor_digest(adapter_parameters(runner.model, "student")), "base_tensor_sha256": lane.chain.native.frozen_digest(runner)}
    return {"original": original, "corpus": corpus, "recipe": recipe, "proof": proof}


@pytest.mark.parametrize("cached", [False, True])
def test_actual_adam_continuation_equals_uninterrupted_and_keeps_wrong_targets(cpu_fixture, tmp_path, cached):
    f = cpu_fixture
    tasks, recipe = f["corpus"]["train"], f["recipe"]
    schedule = schedule_for(tasks, "A", recipe)
    lookup = {t.uid: t for t in tasks}
    baseline = cpu_runner(f["original"], f["corpus"], tmp_path / "uninterrupted")
    cache = None
    if cached:
        cache = {}
        for index, task in enumerate(tasks):
            answer = list(task.answer)
            if index == 0:
                answer[-1] = (answer[-1] + 1) % 10
            ids = baseline.tokenizer.encode(digits(answer), add_special_tokens=False) + [baseline.eos]
            cache[task.uid] = {"uid": task.uid, "response_token_ids": ids, "chain_sha256": digest([task.uid, ids])}
        with pytest.raises(ValueError, match="CACHED_TEACHER_TARGETS_CHANGED"):
            verify_cached_targets(tasks, list(cache.values()), "wrong")
    baseline.optimizer = torch.optim.AdamW(baseline.student_parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    baseline_ledger, baseline_rows = [], []
    names = list(adapter_parameters(baseline.model, "student"))
    for step, batch in enumerate(schedule, 1):
        update, examples = lane.update(baseline, lookup, batch, step, cache, recipe, None)
        baseline_ledger.append(update)
        baseline_rows.extend(examples)
        if step == 4:
            source = lane.save_point(baseline, tasks, cache, recipe, tmp_path / "uninterrupted", 4)
            probes = lane.chain.measure(baseline, tasks[:8], tmp_path / "uninterrupted", "source8", privileged=False, forced=False)
    resumed = cpu_runner(f["original"], f["corpus"], tmp_path / "resumed")
    lane.restore(resumed, source["adapter"], tmp_path / "uninterrupted/checkpoint4/optimizer.pt", 4, names, recipe)
    assert lane.probe_parity(resumed, tasks[:8], probes, tmp_path / "resumed") == probes
    resumed_ledger, resumed_rows = [], []
    for step in range(5, 9):
        update, examples = lane.update(resumed, lookup, schedule[step - 1], step, cache, recipe, None)
        resumed_ledger.append(update)
        resumed_rows.extend(examples)
    assert resumed_ledger == baseline_ledger[4:]
    assert resumed_rows == baseline_rows[16:]
    assert tensor_digest(adapter_parameters(resumed.model, "student")) == tensor_digest(adapter_parameters(baseline.model, "student"))
    left, right = resumed.optimizer.state_dict(), baseline.optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    for index in left["state"]:
        for key in left["state"][index]:
            assert torch.equal(left["state"][index][key], right["state"][index][key])
    assert {int(s["step"]) for s in left["state"].values()} == {8}
    if cached:
        wrong = tasks[0]
        assert [r["response_token_ids"] for r in resumed_rows if r["uid"] == wrong.uid] == [cache[wrong.uid]["response_token_ids"]]
        assert cache[wrong.uid]["response_token_ids"] != resumed.target_tokens(wrong).tolist()


def test_bad_optimizer_or_probe_stops_before_any_update(cpu_fixture, tmp_path):
    f = cpu_fixture
    runner = cpu_runner(f["original"], f["corpus"], tmp_path / "source")
    runner.optimizer = torch.optim.AdamW(runner.student_parameters, lr=f["recipe"]["learning_rate"], weight_decay=0.0)
    tasks = f["corpus"]["train"]
    lookup = {t.uid: t for t in tasks}
    schedule = schedule_for(tasks, "A", f["recipe"])
    for step in range(1, 5):
        lane.update(runner, lookup, schedule[step - 1], step, None, f["recipe"], None)
    point = lane.save_point(runner, tasks, None, f["recipe"], tmp_path / "source", 4)
    path = tmp_path / "source/checkpoint4/optimizer.pt"
    parameters = lane.recovery.saved_parameters(Path(point["adapter"]["path"]) / "adapter_model.safetensors")
    names = list(adapter_parameters(runner.model, "student"))
    optimizer = torch.load(path, map_location="cpu", weights_only=True)
    for field in ("step", "exp_avg"):
        corrupt = copy.deepcopy(optimizer)
        value = next(iter(corrupt["state"].values()))[field]
        value.fill_(3 if field == "step" else float("nan"))
        damaged = tmp_path / f"bad_{field}.pt"
        torch.save(corrupt, damaged)
        with pytest.raises(ValueError, match="OPTIMIZER_STEP_MISMATCH|MOMENTS_INVALID"):
            lane.recovery.inspect_optimizer(damaged, 4, names, parameters, f["recipe"])
    before = tensor_digest(adapter_parameters(runner.model, "student"))
    with pytest.raises(ValueError, match="PROBE_PARITY_BEFORE_UPDATES"):
        lane.probe_parity(runner, tasks[:8], [], tmp_path / "source")
    assert tensor_digest(adapter_parameters(runner.model, "student")) == before and runner.step == 4


def test_teacher_adapter_is_forbidden(cpu_fixture, tmp_path):
    f = cpu_fixture
    runner = tiny_runner(f["original"], tmp_path)
    with pytest.raises(ValueError, match="TEACHER_ADAPTER_PRESENT"):
        lane.student_only(runner)


def test_gold_mastery_and_cached_target_fit_stay_separate():
    records = [{"depth": 3 if i < 64 else 4, "generation": {"correct": i not in (0, 64), "format_valid": True, "terminated": True}, "supplied_target_match": True, "gold_teacher_forced_token_loss": 0.1, "supplied_teacher_forced_token_loss": 0.05} for i in range(128)]
    score = acquisition(records)
    assert score["passed"] and score["panels"]["all"]["gold_correct"] == 126
    assert score["panels"]["all"]["supplied_target_match"] == 128
    for row in records[:8]:
        row["generation"]["correct"] = False
    assert not acquisition(records)["passed"]


def test_origin_receipt_change_rejected_without_model_load(design, tmp_path, monkeypatch):
    copied = copy.deepcopy(design)
    copied["arms"]["B_oracle"]["training_dir"] = str(tmp_path)
    (tmp_path / "training.json").write_text("{}")
    def forbidden(*args):
        raise AssertionError("MODEL_OR_OLD_VALIDATION_REACHED")
    monkeypatch.setattr(lane.chain, "verify_training", forbidden)
    with pytest.raises(ValueError, match="PARENT_RECEIPT_CHANGED"):
        lane.origin_inputs(copied, "B_oracle", None, None, None, None)


def test_actual_new_pipeline_and_fresh_pid_evaluation_without_origin_or_teacher(design, cpu_fixture, tmp_path, monkeypatch):
    f = cpu_fixture
    source = tmp_path / "old-source"
    runner = cpu_runner(f["original"], f["corpus"], source)
    tasks, method = f["corpus"]["train"], "chain_output_flat_mixed"
    old = json.loads((lane.ROOT / "configs/device4_chain_protocol.json").read_text())
    old["training"].update(updates_per_method=4, examples_per_update=4, example_exposures_per_method=16, checkpoint_updates=[4], epochs=1, probe_rows_per_depth=4)
    old["training"]["expected_trainable_parameters"] = f["recipe"]["expected_trainable_parameters"]
    cache = []
    for index, task in enumerate(tasks):
        answer = list(task.answer)
        if index == 0:
            answer[-1] = (answer[-1] + 1) % 10
        ids = runner.tokenizer.encode(digits(answer), add_special_tokens=False) + [runner.eos]
        cache.append({"uid": task.uid, "response_token_ids": ids, "chain_sha256": digest([task.uid, ids]), "teacher_correct": index != 0})
    schedule = lane.chain.train_schedule(f["corpus"], "A", old["training"])
    prior = lane.chain.train_method(runner, old, "A", method, cache, schedule, source)
    checkpoint = prior["checkpoints"]["4"]
    q = {"upstream": f["proof"]}
    lane.write_new(source / "qualified/qualification.json", q)
    lane.write_new(source / "qualified/caches/A/flat_mixed.json", cache)
    lane.write_new(source / "schedule.json", schedule)
    lane.write_new(source / "runtime.json", {"targeted_parameter_names": list(adapter_parameters(runner.model, "student"))})
    origin = {"family": "A", "pid": os.getpid(), "methods": {method: prior}, "method_candidates": {method: "flat_mixed"}, "base_tensor_sha256": f["proof"]["base_tensor_sha256"], "qualification_files_sha256": {"qualification.json": file_digest(source / "qualified/qualification.json"), "caches/A/flat_mixed.json": file_digest(source / "qualified/caches/A/flat_mixed.json")}, "target_equivalence": {method: {"equal_oracle_count": 15, "rows": 16, "token_identical_to_oracle": False, "wrong_teacher_targets_retained": 1}}}
    lane.write_new(source / "training.json", origin)
    spec = {"family": "A", "method": method, "training_dir": str(source), "receipt_files_sha256": {p: file_digest(source / p) for p in ("training.json", "runtime.json", "schedule.json")}, "checkpoint_tensor_sha256": checkpoint["adapter"]["tensor_sha256"], "optimizer_sha256": checkpoint["optimizer_sha256"], "cache_canonical_sha256": digest(cache), "expected_wrong_cached_targets": 1}
    prepared = copy.deepcopy(design)
    prepared["training"] = f["recipe"]
    prepared["arms"]["A_chain"] = spec
    files = {**spec["receipt_files_sha256"], "qualified/qualification.json": file_digest(source / "qualified/qualification.json"), "qualified/caches/A/flat_mixed.json": file_digest(source / "qualified/caches/A/flat_mixed.json"), f"{method}/ledger.json": prior["ledger_sha256"], f"{method}/rows.json": prior["rows_sha256"], f"{method}/checkpoint4/train_probes.json": checkpoint["train_probes_sha256"]}
    monkeypatch.setattr(lane, "origin_inputs", lambda *args: (source, origin, q, checkpoint, files, cache))
    monkeypatch.setattr(lane, "load_student", lambda original, output: cpu_runner(original, f["corpus"], output))
    out = tmp_path / "continued"
    out.mkdir()
    audit = {"cpu_fixture": True, "sha256": digest(to_records(f["corpus"]))}
    trained = lane.train(prepared, {"arm": "A_chain"}, old, f["original"], f["corpus"], audit, out)
    assert trained["total_updates"] == 8 and trained["new_updates"] == 4
    assert set(trained["checkpoints"]) == {"4", "6", "8"}
    with pytest.raises(ValueError, match="TRAINING_IDENTITY_OR_FIXED_BUDGET_CHANGED"):
        lane.verify_training(out, prepared, "A_chain", f["corpus"], audit)
    payload = {"design": prepared, "original": f["original"], "corpus": to_records(f["corpus"]), "audit": audit}
    lane.write_new(tmp_path / "fixture.json", payload)
    source.rename(tmp_path / "old-source-unavailable")
    script = tmp_path / "child.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import device4_sufficiency as lane\nfrom device4_sufficiency_test import cpu_runner\nfrom tasks import Task\n"
        "root=Path(sys.argv[1])\nf=json.loads((root/'fixture.json').read_text())\n"
        "corpus={s:[Task(r['uid'],r['family'],r['split'],tuple(r['initial']),tuple(r['program']),tuple(r['answer'])) for r in rows] for s,rows in f['corpus'].items()}\n"
        "def load(original,output):\n    return cpu_runner(original,corpus,output)\n"
        "def forbidden(*args,**kwargs):\n    raise AssertionError('TEACHER_OR_EXTERNAL_ORIGIN_REACHED')\n"
        "lane.load_student=load\nlane.origin_inputs=forbidden\nlane.chain.execute_chain=forbidden\nlane.chain.native.load_runner=forbidden\n"
        "lane.evaluate(f['design'],{'arm':'A_chain','training_dir':str(root/'continued')},f['original'],corpus,f['audit'],root/'continued/evaluation')\n"
    )
    completed = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", str(script), str(tmp_path)], cwd=lane.ROOT, env={**os.environ, "PYTHONPATH": str(lane.ROOT)}, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evaluated = json.loads((out / "evaluation/result.json").read_text())
    assert evaluated["pid"] != os.getpid() and evaluated["training_pid"] == os.getpid()
    assert evaluated["train_reload_rows"] == 16 and evaluated["full_train_native_reload_exact"]
    assert evaluated["test"]["all"]["n"] == 4 and evaluated["post_observation_reused_test"]
    assert not evaluated["mastery"]["passed"] and not evaluated["generalization_after_mastery_interpretable"]
    assert not evaluated["teacher_weights_loaded"] and not evaluated["external_origin_artifacts_read"]
    assert evaluated["resident_adapters"] == ["student"]


def test_all_eight_configs_validate_and_no_early_selection(design, tmp_path):
    config = lane.ROOT / "configs/device4_sufficiency_learn_A_chain.json"
    output = tmp_path / "validated"
    result = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", str(lane.ROOT / "device4_sufficiency.py"), "--config", str(config), "--output-dir", str(output), "--validate-only"], cwd=lane.ROOT, check=True, capture_output=True, text=True)
    assert result.returncode == 0
    sealed = json.loads((output / "seal.json").read_text())
    assert sealed["design_sha256"] == digest(design)
    assert not (output / "training.json").exists()
