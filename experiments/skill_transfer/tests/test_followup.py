import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import audit_persistence
import compress_fusion
import followup
from data import audit_corpus, build_corpus, digest
from learning import (
    LowRankLinear,
    capture,
    frozen_hash,
    save_checkpoint,
    score,
    tree_record,
    write_json,
)
from sandbox import FAMILIES, SCHEMAS, grade_text

ROOT = Path(__file__).resolve().parents[1]


def execution_receipt(task, status="completed", attempt="a" * 32):
    receipt = {
        "task_id": task["id"],
        "attempt_id": attempt,
        "task": copy.deepcopy(task),
        "task_sha256": followup.execution_digest(task),
        "config_sha256": followup.execution_digest(task["config"]),
        "source_sha256": task["source_sha256"],
        "started_at": "2026-09-12T05:00:00+00:00" if status == "running" else "2026-09-12T04:00:00+00:00",
        "status": status,
        "pid": 55,
        "pod": "fixture-pod",
        "supervisor": {"id": "c" * 32, "pid": 11, "pod_uid": "fixture-pod-uid"},
    }
    if status == "completed":
        receipt.update(finished_at="2026-09-12T04:59:00+00:00", exit_code=0, timed_out=False)
    return receipt


def followup_transport(output, settings, monkeypatch, mode):
    output.mkdir()
    original = json.loads((ROOT / "followup_protocol.json").read_text())["frozen_source_files"]
    additions = [
        "followup.py",
        "audit_persistence.py",
        "compress_fusion.py",
        "followup_protocol.json",
        "followup_handoff.json",
    ]
    additions += [
        f"configs/{kind}_stream_{seed}.json" for kind in ("audit", "compress") for seed in (1103, 2207, 3301, 4409)
    ]
    checksum = hashlib.sha256()
    for name in sorted([*original, *additions]):
        checksum.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    entrypoint = "audit_persistence.py" if mode == "audit" else "compress_fusion.py"
    task = {
        "id": f"followup-{mode}",
        "entrypoint": entrypoint,
        "code_dir": str(ROOT),
        "source_sha256": checksum.hexdigest(),
        "config": settings,
    }
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (output / "execution.json").write_text(json.dumps(execution_receipt(task, "running", "b" * 32), indent=2) + "\n")
    monkeypatch.setattr(
        sys, "argv", [str(ROOT / entrypoint), "--config", str(output / "config.json"), "--output-dir", str(output)]
    )


def tiny_model(fused=False):
    model = torch.nn.Module()
    base = torch.nn.Linear(12, 12, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        base.weight.zero_()
    model.projection = LowRankLinear(base, 8, 16)
    if fused:
        model.projection.offset = torch.diag(torch.arange(12, 0, -1, dtype=torch.float32))
    return model


def response(row, conventions, correct):
    text = json.dumps(row.calls, separators=(",", ":")) if correct else "[]"
    return {
        **grade_text(text, row, conventions),
        "generated_ids": list(text.encode()) + [0],
        "raw_generation_ids": list(text.encode()) + [0],
    }


def simulated_generation(model, encoded, rows):
    ability = model.projection(torch.eye(12, dtype=torch.bfloat16)).diagonal().float().tolist()
    return [response(row, encoded.conventions, ability[int(row.id[:8], 16) % 12] > 0.5) for row in rows]


def source_fixture(tmp_path, qualified=(True, True), transfers=True):
    root = tmp_path / "source"
    root.mkdir()
    config = json.loads((ROOT / "configs/stream_1103.json").read_text())
    for kind in ("primitive", "workflow"):
        for split in ("train", "validation", "test"):
            config[f"{kind}_{split}_examples"] = 8
    protocol = json.loads((ROOT / "followup_protocol.json").read_text())
    code = tmp_path / "original-code"
    for relative in protocol["frozen_source_files"]:
        path = code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    settings = {
        "experiment": "skill-transfer-compress",
        "source_run_dir": str(root),
        "source_task_id": "fixture",
        "source_code_sha256": protocol["source_code_sha256"],
        "source_config_sha256": digest(config),
        "followup_protocol": "followup_protocol.json",
        "followup_protocol_sha256": followup.sha256_file(ROOT / "followup_protocol.json"),
    }
    spec, corpus = build_corpus(config)
    encoded = SimpleNamespace(conventions=spec["conventions"])
    model = tiny_model(fused=True)
    write_json(root / "config.json", config)
    task = {
        "id": "fixture",
        "code_dir": str(code),
        "entrypoint": "run.py",
        "source_sha256": settings["source_code_sha256"],
        "config": config,
    }
    (root / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (root / "execution.json").write_text(json.dumps(execution_receipt(task), indent=2) + "\n")
    (root / "protocol.json").write_bytes((ROOT / "protocol.json").read_bytes())
    write_json(root / "stream.json", spec)
    write_json(root / "dataset-audit.json", audit_corpus(spec, corpus))
    write_json(root / "runtime.json", {"cpu_controller_fixture": True})
    for family in FAMILIES:
        rows = corpus["primitive"][family]["validation"]
        records = [response(row, spec["conventions"], False) for row in rows]
        write_json(root / "baseline" / f"{family}.validation.json", {"metrics": score(records), "records": records})
        save_checkpoint(model, None, root / "qualification" / family / "128" / "checkpoint", {"updates": 128})
    for method, passed in zip(("arithmetic", "agent_dice"), qualified, strict=True):
        for family in FAMILIES:
            for split in ("validation", "test"):
                rows = corpus["primitive"][family][split]
                records = [response(row, spec["conventions"], passed) for row in rows]
                write_json(
                    root / "fusion" / method / f"{family}.{split}.json", {"metrics": score(records), "records": records}
                )
        save_checkpoint(
            model,
            None,
            root / "fusion" / method / "checkpoint",
            {
                "expert_archive_bytes_fp32": 4 * (12 * 8 + 8 * 12) * 4,
                "expert_training_updates": 512,
                "expert_training_example_exposures": 2048,
            },
        )
    result = {
        "status": "completed",
        "artifact_manifest": {},
        "config_sha256": digest(config),
        "protocol_sha256": protocol["source_protocol_sha256"],
        "frozen_base": {"before": frozen_hash(model)},
        "qualification": {"passed": True, "budget": 128},
        "transfer": {name: {} for name in protocol["audit"]["conditions"]} if transfers else {},
    }
    fake_source = SimpleNamespace(config=config, corpus=corpus)
    for condition in result["transfer"]:
        save_checkpoint(model, None, root / "transfer" / condition / "checkpoint", {"updates": 128})
        for reference, rows in audit_persistence.test_surfaces(fake_source, condition).values():
            records = simulated_generation(model, encoded, rows)
            write_json(root / reference, {"metrics": score(records), "records": records})
    result["artifact_manifest"] = {
        str(path.relative_to(root)): followup.sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"config.json", "task.json", "execution.json"}
    }
    write_json(root / "result.json", result)
    return settings, protocol, encoded


def cpu_runtime(source, output):
    return tiny_model(), SimpleNamespace(conventions=source.spec["conventions"])


def test_source_rebuild_and_hashes_reject_changed_saved_artifacts(tmp_path):
    settings, protocol, _ = source_fixture(tmp_path, transfers=False)
    source = followup.SourceRun(settings, protocol)
    path = source.root / "fusion/arithmetic/calendar.validation.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="SOURCE_ARTIFACT_HASH"):
        source.read_json("fusion/arithmetic/calendar.validation.json")
    with pytest.raises(RuntimeError, match="SOURCE_PATH_ESCAPE"):
        source.verified_path("../config.json")


def test_source_requires_completed_matching_run(tmp_path):
    settings, protocol, _ = source_fixture(tmp_path, transfers=False)
    settings["source_config_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="SOURCE_IDENTITY_MISMATCH"):
        followup.SourceRun(settings, protocol)
    result_path = Path(settings["source_run_dir"]) / "result.json"
    result = json.loads(result_path.read_text())
    result["status"] = "prerequisite_failed"
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="SOURCE_NOT_COMPLETED"):
        followup.SourceRun(settings, protocol)


def test_followup_cannot_write_into_source(tmp_path):
    settings, _, _ = source_fixture(tmp_path, transfers=False)
    with pytest.raises(RuntimeError, match="FOLLOWUP_OUTPUT_OVERLAPS_SOURCE"):
        followup.prepare_followup(settings, Path(settings["source_run_dir"]) / "audit", "compress")


def test_eligibility_reads_only_original_validation(tmp_path):
    settings, protocol, _ = source_fixture(tmp_path, transfers=False)
    source = followup.SourceRun(settings, protocol)
    _, _, gates = compress_fusion.saved_eligibility(source)
    assert all(gate["passed"] for gate in gates.values())
    assert not any(".test." in name or "checkpoint" in name for name in source.consumed)


def test_primitive_gate_requires_each_operation_and_original_gain():
    config = json.loads((ROOT / "configs/stream_1103.json").read_text())
    baseline = {
        family: {"n": 32, "accuracy": 0, "per_pattern": {name: 0 for name in SCHEMAS[family]}} for family in FAMILIES
    }
    fused = {
        family: {"n": 32, "accuracy": 1, "per_pattern": {name: 1 for name in SCHEMAS[family]}} for family in FAMILIES
    }
    assert compress_fusion.fusion_gate(baseline, fused, config)["passed"]
    fused["records"]["per_pattern"]["take"] = 0.5
    assert not compress_fusion.fusion_gate(baseline, fused, config)["passed"]
    with pytest.raises(ValueError, match="FUSION_GATE_FAMILY_COVERAGE"):
        compress_fusion.fusion_gate(baseline, {"records": fused["records"]}, config)


def test_zero_qualified_branches_perform_no_runtime_svd_or_test(tmp_path, monkeypatch):
    settings, _, _ = source_fixture(tmp_path, qualified=(False, False), transfers=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("failed gate must prevent intervention")

    monkeypatch.setattr(compress_fusion, "fresh_runtime", forbidden)
    monkeypatch.setattr(compress_fusion, "svd_rank8", forbidden)
    output = tmp_path / "compression"
    followup_transport(output, settings, monkeypatch, "compress")
    result = compress_fusion.compress(settings, output)
    assert all(row["status"] == "branch_gate_failed" and not row["svd_performed"] for row in result["methods"].values())
    consumed = json.loads((output / "consumed-source.json").read_text())
    assert not any(".test." in name or "checkpoint" in name for name in consumed)


def test_compression_commits_before_test_and_preserves_fixed_capacity(tmp_path, monkeypatch):
    settings, _, _ = source_fixture(tmp_path, qualified=(False, True), transfers=False)
    monkeypatch.setattr(compress_fusion, "fresh_runtime", cpu_runtime)
    output = tmp_path / "compression"
    followup_transport(output, settings, monkeypatch, "compress")

    def ordered_generation(model, encoded, rows):
        if rows[0].split in {"test", "novel_test"}:
            assert (output / "agent_dice/checkpoint/state.pt").is_file()
        return simulated_generation(model, encoded, rows)

    monkeypatch.setattr(followup, "generate", ordered_generation)
    result = compress_fusion.compress(settings, output)
    assert result["methods"]["arithmetic"]["status"] == "branch_gate_failed"
    branch = result["methods"]["agent_dice"]
    assert branch["svd_performed"] and not branch["teacher_qualified"]
    assert branch["memory"]["compressed"]["dense_offset_bytes"] == 0
    assert branch["memory"]["compressed"]["factor_bytes"] == branch["memory"]["original"]["factor_bytes"]
    assert branch["memory"]["source_expert_training_updates"] == 512
    assert branch["memory"]["retained_source_expert_archive_bytes_fp32"] > 0
    assert branch["memory"]["retained_source_expert_checkpoint_files"]["total_file_bytes"] > 0
    assert len(branch["memory"]["retained_source_expert_checkpoint_files"]["by_family"]) == 4
    assert branch["energy"]["energy_retained_fraction"] < 1
    assert any(row["accuracy_change"] < 0 for row in branch["behavior"].values())
    assert "workflow.novel-test" in branch["behavior"]
    saved = torch.load(output / "agent_dice/checkpoint/state.pt", weights_only=True)
    assert saved["optimizer"] is None and saved["adapter"]["projection"]["offset"] is None
    assert saved["adapter"]["projection"]["A"].shape == (8, 12)


def test_fresh_dev_mismatch_blocks_svd_and_test(tmp_path, monkeypatch):
    settings, _, _ = source_fixture(tmp_path, transfers=False)
    monkeypatch.setattr(compress_fusion, "fresh_runtime", cpu_runtime)

    def changed(model, encoded, rows):
        assert rows[0].split == "validation"
        records = simulated_generation(model, encoded, rows)
        records[0] = response(rows[0], encoded.conventions, False)
        return records

    def forbidden(*args, **kwargs):
        raise AssertionError("SVD must wait for exact fresh dev reload")

    monkeypatch.setattr(followup, "generate", changed)
    monkeypatch.setattr(compress_fusion, "svd_rank8", forbidden)
    output = tmp_path / "compression"
    followup_transport(output, settings, monkeypatch, "compress")
    result = compress_fusion.compress(settings, output)
    assert all(row["reason"] == "fresh_load_dev_mismatch_or_unqualified" for row in result["methods"].values())


@pytest.mark.parametrize("mutate", [False, True])
def test_audit_all_final_conditions_and_records_without_early_exit(tmp_path, monkeypatch, mutate):
    settings, _, _ = source_fixture(tmp_path)
    settings["experiment"] = "skill-transfer-audit"
    monkeypatch.setattr(audit_persistence, "fresh_runtime", cpu_runtime)
    measured = []

    def generation(model, encoded, rows):
        measured.append(rows[0].id)
        records = simulated_generation(model, encoded, rows)
        if mutate and len(measured) == 1:
            records[0]["raw_generation_ids"] += [0]
        return records

    monkeypatch.setattr(followup, "generate", generation)
    output = tmp_path / "audit"
    followup_transport(output, settings, monkeypatch, "audit")
    result = audit_persistence.audit(settings, output)
    assert len(measured) == 7 * 6
    assert len(result["conditions"]) == 7 and result["all_records_exact"] != mutate
    assert result["conditions"]["agent_dice"]["all_records_exact"]
    if mutate:
        changed = result["conditions"]["fresh"]["comparisons"]["workflow.test"]
        assert changed["all_semantic_outcomes_equal"] and not changed["all_raw_generation_ids_exact"]
    assert result["training_updates"] == 0 and result["source_unchanged"]
    assert result["process_proof"]["distinct_task_and_attempt_ids"]
    assert result["process_proof"]["source_launcher_pid"] == result["process_proof"]["followup_launcher_pid"]
    assert followup.sha256_file(output / "source-execution.json") == result["source_execution_sha256"]


@pytest.mark.parametrize("mutation", ["running", "exit", "timeout", "task", "task_hash", "config_hash", "code_hash"])
def test_source_execution_requires_success_and_verified_embedded_identity(tmp_path, mutation):
    settings, protocol, _ = source_fixture(tmp_path, transfers=False)
    path = Path(settings["source_run_dir"]) / "execution.json"
    receipt = json.loads(path.read_text())
    if mutation == "running":
        receipt["status"] = "running"
    elif mutation == "exit":
        receipt["exit_code"] = 1
    elif mutation == "timeout":
        receipt["timed_out"] = True
    elif mutation == "task":
        receipt["task"]["config"]["data_seed"] = 0
    elif mutation == "task_hash":
        receipt["task_sha256"] = "wrong"
    elif mutation == "config_hash":
        receipt["config_sha256"] = "wrong"
    else:
        receipt["source_sha256"] = "wrong"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="EXECUTION_"):
        followup.SourceRun(settings, protocol)


@pytest.mark.parametrize("mutation", ["changed", "added", "missing"])
def test_source_archive_content_must_match_actual_original_twenty_files(tmp_path, mutation):
    settings, protocol, _ = source_fixture(tmp_path, transfers=False)
    code = tmp_path / "original-code"
    if mutation == "changed":
        (code / "learning.py").write_text("changed")
    elif mutation == "added":
        (code / "injected.py").write_text("pass")
    else:
        (code / "learning.py").unlink()
    with pytest.raises(RuntimeError, match="CODE_BUNDLE_HASH"):
        followup.SourceRun(settings, protocol)


@pytest.mark.parametrize(
    "mutation", ["same_task", "same_attempt", "wrong_entrypoint", "not_cli", "before_source_completed", "config_hash"]
)
def test_followup_requires_own_task_attempt_and_bound_standalone_cli(tmp_path, monkeypatch, mutation):
    settings, protocol, _ = source_fixture(tmp_path, transfers=False)
    source = followup.SourceRun(settings, protocol)
    output = tmp_path / "followup"
    followup_transport(output, settings, monkeypatch, "compress")
    task = json.loads((output / "task.json").read_text())
    receipt = json.loads((output / "execution.json").read_text())
    if mutation == "same_task":
        task["id"] = source.task["id"]
        receipt = execution_receipt(task, "running", "b" * 32)
    elif mutation == "same_attempt":
        receipt["attempt_id"] = source.execution["attempt_id"]
    elif mutation == "wrong_entrypoint":
        task["entrypoint"] = "run.py"
        receipt = execution_receipt(task, "running", "b" * 32)
    elif mutation == "not_cli":
        monkeypatch.setattr(sys, "argv", ["pytest"])
    elif mutation == "before_source_completed":
        receipt["started_at"] = source.execution["started_at"]
    else:
        receipt["config_sha256"] = "wrong"
    (output / "task.json").write_text(json.dumps(task))
    (output / "execution.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="EXECUTION"):
        followup.verify_followup_process(settings, output, source, "compress")


def test_checkpoint_load_in_independent_uv_process(tmp_path):
    model = tiny_model(fused=True)
    with torch.no_grad():
        model.projection.B.fill_(0.01)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(model, None, checkpoint, {"test": "fresh-process"})
    expected = model.projection(torch.eye(12, dtype=torch.bfloat16)).float().tolist()
    code = """import json
import sys
import torch
from learning import LowRankLinear, load_checkpoint
model = torch.nn.Module()
base = torch.nn.Linear(12, 12, bias=False, dtype=torch.bfloat16)
with torch.no_grad():
    base.weight.zero_()
model.projection = LowRankLinear(base, 8, 16)
load_checkpoint(model, sys.argv[1])
print(json.dumps(model.projection(torch.eye(12, dtype=torch.bfloat16)).float().tolist()))
"""
    completed = subprocess.run(
        ["uv", "run", "--no-project", "--python", sys.executable, "python", "-c", code, str(checkpoint)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == expected


def test_svd_complete_effective_matrix_and_source_immutability():
    torch.manual_seed(59)
    state = capture(tiny_model(fused=True))
    state["projection"]["A"].normal_()
    state["projection"]["B"].normal_()
    before = tree_record(state)
    values = state["projection"]
    delta = values["offset"].double() + 2 * values["B"].double() @ values["A"].double()
    compressed, energy = compress_fusion.svd_rank8(state)
    factors = compressed["projection"]
    actual = 2 * factors["B"].double() @ factors["A"].double()
    singular = torch.linalg.svdvals(delta)
    assert energy["projections"]["projection"]["ideal_tail_energy"] == pytest.approx(float(singular[8:].square().sum()))
    assert float((delta - actual).square().sum()) == pytest.approx(float(singular[8:].square().sum()), rel=1e-5)
    assert tree_record(state) == before
    assert factors["offset"] is None and factors["A"].dtype == factors["B"].dtype == torch.float32


@pytest.mark.parametrize("rank", [0, 3, 8, 12])
def test_svd_known_spectrum_rank_energy_and_memory(rank):
    model = tiny_model(fused=True)
    model.projection.offset = torch.diag(torch.tensor([float(rank - i) if i < rank else 0.0 for i in range(12)]))
    state = capture(model)
    compressed, report = compress_fusion.svd_rank8(state)
    spectra = torch.arange(rank, 0, -1, dtype=torch.float64)
    total = float(spectra.square().sum())
    kept = float(spectra[:8].square().sum())
    assert report["energy_retained_fraction"] == pytest.approx(kept / total if total else 1)
    assert report["projections"]["projection"]["source_numerical_rank"] == rank
    assert (
        compress_fusion.state_memory(compressed)["factor_bytes"] == compress_fusion.state_memory(state)["factor_bytes"]
    )
    if rank <= 8:
        assert report["stored_factor_relative_frobenius_error"] < 1e-6
    if rank == 0:
        assert torch.count_nonzero(compressed["projection"]["A"]) > 0
        assert torch.count_nonzero(compressed["projection"]["B"]) == 0


@pytest.mark.parametrize("mutation", ["nan", "bf16", "rank", "offset_shape"])
def test_svd_rejects_invalid_state(mutation):
    state = capture(tiny_model(fused=True))
    values = state["projection"]
    if mutation == "nan":
        values["offset"][0, 0] = float("nan")
    elif mutation == "bf16":
        values["A"] = values["A"].to(torch.bfloat16)
    elif mutation == "rank":
        values["A"] = values["A"][:7]
    else:
        values["offset"] = values["offset"][:4]
    with pytest.raises(ValueError, match="SVD_"):
        compress_fusion.svd_rank8(state)


def test_full_record_comparison_distinguishes_semantics_from_tokens():
    record = {
        "id": "one",
        "generated_ids": [1],
        "raw_generation_ids": [1, 0],
        "correct": True,
        "format_valid": True,
        "executable": True,
    }
    expected = {"records": [record], "metrics": {"accuracy": 1}}
    actual = copy.deepcopy(expected)
    actual["records"][0]["raw_generation_ids"].append(0)
    result = followup.comparison(expected, actual)
    assert not result["all_records_exact"] and result["all_semantic_outcomes_equal"]
    assert result["differences"] == [{"id": "one", "fields": ["raw_generation_ids"]}]
