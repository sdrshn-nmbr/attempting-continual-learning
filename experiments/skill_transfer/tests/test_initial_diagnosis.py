import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_followup import cpu_runtime, execution_receipt, response, simulated_generation, source_fixture, tiny_model

import diagnose_initial
import followup
import learning
from data import build_corpus, make_example
from learning import score, write_json
from sandbox import grade_text

ROOT = Path(__file__).resolve().parents[1]


def source_for(seed):
    config = json.loads((ROOT / f"configs/stream_{seed}.json").read_text())
    spec, corpus = build_corpus(config)
    return SimpleNamespace(config=config, spec=spec, corpus=corpus)


def protocol():
    return json.loads((ROOT / "diagnosis_protocol.json").read_text())


def diagnosis_fixture(tmp_path):
    settings, _, encoded = source_fixture(tmp_path)
    settings["experiment"] = "skill-transfer-initial-diagnosis"
    settings["diagnosis_protocol"] = "diagnosis_protocol.json"
    settings["diagnosis_protocol_sha256"] = followup.sha256_file(ROOT / "diagnosis_protocol.json")
    root = Path(settings["source_run_dir"])
    config = json.loads((root / "config.json").read_text())
    spec, corpus = build_corpus(config)
    result = json.loads((root / "result.json").read_text())
    source = SimpleNamespace(config=config, spec=spec, corpus=corpus)
    model = tiny_model(fused=True)
    for condition, record in result["transfer"].items():
        record["initial_checkpoint"] = "fusion/agent_dice/checkpoint"
        for initial in (True, False):
            for reference, rows in diagnose_initial.validation_surfaces(source, condition, initial).values():
                records = simulated_generation(model, encoded, rows)
                write_json(root / reference, {"metrics": score(records), "records": records})
    result["artifact_manifest"] = {
        str(path.relative_to(root)): followup.sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"config.json", "task.json", "execution.json", "result.json"}
    }
    (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return settings


def transport(output, settings, monkeypatch):
    output.mkdir()
    checksum = hashlib.sha256()
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") or part in {"tests", "outputs", "runs", "__pycache__"}
            for part in path.relative_to(ROOT).parts
        )
        and (
            path.suffix in {".py", ".json", ".txt", ".yaml", ".yml", ".sha256"}
            or path.name.startswith(("LICENSE", "NOTICE"))
        )
    )
    for path in paths:
        checksum.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    task = {
        "id": "posthoc-fixture",
        "entrypoint": "diagnose_initial.py",
        "code_dir": str(ROOT),
        "source_sha256": checksum.hexdigest(),
        "config": settings,
    }
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (output / "execution.json").write_text(json.dumps(execution_receipt(task, "running", "d" * 32), indent=2) + "\n")
    monkeypatch.setattr(sys, "argv", [str(ROOT / "diagnose_initial.py")])


@pytest.mark.parametrize("seed", [1103, 2207, 3301, 4409])
def test_fresh_inputs_are_deterministic_disjoint_and_semantically_valid(seed):
    source = source_for(seed)
    first = diagnose_initial.fresh_panels(source, protocol())
    assert first == diagnose_initial.fresh_panels(source, protocol())
    assert {row.pattern for row in first["ordinary"]} == {"0", "1"}
    assert {row.pattern for row in first["unseen-combinations"]} == {"2", "3"}
    assert len({row.group_id for rows in first.values() for row in rows}) == 32
    assert all(row.split == "posthoc_diagnostic" for rows in first.values() for row in rows)


def test_fresh_overlap_is_rejected_before_runtime():
    config = protocol()
    config["fresh_inputs"]["seed_offset"] = 0
    with pytest.raises(RuntimeError, match="FRESH_INPUT_OVERLAP"):
        diagnose_initial.fresh_panels(source_for(2207), config)


@pytest.mark.parametrize("kind", ["training_plan", "argument_case"])
def test_text_failure_explanations_distinguish_plan_substitution_from_argument_case(kind):
    source = source_for(2207)
    row = make_example(source.spec, "text", "workflow", "novel_test", int(kind == "argument_case"))
    calls = copy.deepcopy(row.calls)
    if kind == "training_plan":
        calls[1] = {"tool": source.spec["conventions"]["text"]["lower"], "args": {}}
    else:
        calls[1]["args"]["contains"] = "ALERT"
        calls[2]["args"]["old"] = "ALERT"
    record = grade_text(json.dumps(calls), row, source.spec["conventions"])
    assert record["executable"] and not record["correct"]
    explanation = diagnose_initial.explain({"metrics": score([record]), "records": [record]}, [row], source, protocol())
    pattern = explanation["per_pattern"][row.pattern]
    detail = explanation["all_records"][0]
    assert detail["first_state_divergence"] == 2
    if kind == "training_plan":
        assert pattern["wrong_answers_using_training_plan"] == 1
        assert detail["emitted_operations"] == ["keep", "lower", "join"]
        assert not detail["same_operations"]
    else:
        assert detail["same_operations"] and len(detail["argument_differences"]) == 2
        assert pattern["wrong_answers_with_same_operations"] == 1


def test_semantically_equivalent_different_plan_is_still_correct():
    source = source_for(4409)
    row = source.corpus["workflow"]["calendar"]["validation"][0]
    calls = [row.calls[1], row.calls[0], row.calls[2]]
    record = grade_text(json.dumps(calls), row, source.spec["conventions"])
    assert record["correct"] and not record["canonical_calls_match"]
    explanation = diagnose_initial.explain({"metrics": score([record]), "records": [record]}, [row], source, protocol())
    assert explanation["per_pattern"][row.pattern]["examples"] == []
    assert explanation["all_records"][0]["correct"]


def test_paired_changes_track_exact_input_losses_and_gains():
    source = source_for(2207)
    rows = source.corpus["workflow"]["text"]["novel_test"][:4]
    before = [
        response(row, source.spec["conventions"], value)
        for row, value in zip(rows, [True, True, False, False], strict=True)
    ]
    after = [
        response(row, source.spec["conventions"], value)
        for row, value in zip(rows, [True, False, True, False], strict=True)
    ]
    result = diagnose_initial.paired_change(
        {"metrics": score(before), "records": before}, {"metrics": score(after), "records": after}, rows
    )
    assert result["paired"]["all"]["counts"] == {"retained": 1, "lost": 1, "gained": 1, "wrong_at_both": 1}
    assert result["paired"]["all"]["lost_ids"] == [rows[1].id]
    with pytest.raises(RuntimeError, match="PAIRED_ROW_ORDER"):
        diagnose_initial.paired_change({"records": before}, {"records": list(reversed(after))}, rows)


def test_all_seven_initial_checkpoints_original_panels_and_fresh_final_loads(tmp_path, monkeypatch):
    settings = diagnosis_fixture(tmp_path)
    output = tmp_path / "diagnosis"
    transport(output, settings, monkeypatch)
    monkeypatch.setattr(diagnose_initial, "fresh_runtime", cpu_runtime)
    monkeypatch.setattr(followup, "generate", simulated_generation)

    def forbidden(*args, **kwargs):
        raise AssertionError("posthoc diagnosis must not train")

    monkeypatch.setattr(learning, "new_optimizer", forbidden)
    monkeypatch.setattr(learning, "train_update", forbidden)
    result = diagnose_initial.diagnose(settings, output)
    assert result["training_updates"] == 0 and result["posthoc"] and not result["recipe_selection"]
    assert result["all_initial_dev_reloads_exact"] and len(result["conditions"]) == 7
    for condition, record in result["conditions"].items():
        assert len(record["original_panels"]) == 11 and len(record["fresh_diagnostic_panels"]) == 2
        assert record["initial_checkpoint"] == "fusion/agent_dice/checkpoint"
        assert record["final_checkpoint"] == f"transfer/{condition}/checkpoint"
        path = output / condition / "explanations" / "initial" / "workflow.novel-test.json"
        assert json.loads(path.read_text())["all_records"]
    assert result["source_unchanged"]
    assert followup.sha256_file(output / "source-execution.json") == result["source_execution_sha256"]


def test_initial_dev_reload_mismatch_skips_that_conditions_test_and_fresh_panels(tmp_path, monkeypatch):
    settings = diagnosis_fixture(tmp_path)
    output = tmp_path / "diagnosis"
    transport(output, settings, monkeypatch)
    monkeypatch.setattr(diagnose_initial, "fresh_runtime", cpu_runtime)
    generated = 0

    def changed(model, encoded, rows):
        nonlocal generated
        records = simulated_generation(model, encoded, rows)
        if generated == 0:
            records[0]["raw_generation_ids"].append(0)
        generated += 1
        return records

    monkeypatch.setattr(followup, "generate", changed)
    result = diagnose_initial.diagnose(settings, output)
    assert not result["all_initial_dev_reloads_exact"]
    assert result["conditions"]["fresh"]["status"] == "initial_reload_mismatch"
    assert result["conditions"]["relevant"]["status"] == "posthoc_diagnosed"
    assert not (output / "fresh/initial/workflow.novel-test.json").exists()
    assert not (output / "fresh/fresh-initial").exists()
    consumed = json.loads((output / "consumed-source.json").read_text())
    assert "transfer/fresh/novel-test.json" not in consumed


@pytest.mark.parametrize("mutation", ["same_attempt", "entrypoint", "receipt_hash"])
def test_diagnosis_requires_its_own_bound_execution(tmp_path, monkeypatch, mutation):
    settings = diagnosis_fixture(tmp_path)
    output = tmp_path / "diagnosis"
    transport(output, settings, monkeypatch)
    receipt = json.loads((output / "execution.json").read_text())
    if mutation == "same_attempt":
        receipt["attempt_id"] = "a" * 32
    elif mutation == "entrypoint":
        monkeypatch.setattr(sys, "argv", [str(ROOT / "audit_persistence.py")])
    else:
        receipt["config_sha256"] = "wrong"
    (output / "execution.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="EXECUTION"):
        diagnose_initial.prepare(settings, output)


def test_frozen_dependencies_are_verified_by_actual_bytes():
    config = protocol()
    diagnose_initial.verify_dependencies(config)
    config["dependency_files"]["learning.py"] = "wrong"
    with pytest.raises(RuntimeError, match="DEPENDENCY_HASH"):
        diagnose_initial.verify_dependencies(config)
