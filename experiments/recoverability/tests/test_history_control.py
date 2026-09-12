import copy
from pathlib import Path

import pytest
import test_prospective as fixtures

import history_control
import history_control_sources as sources
from budget_prediction_model import grid as original_grid
from budget_prediction_model import verify_grid
from history_control_analysis import (
    alternate_mapping,
    common_gate,
    curve,
    input_schedule,
    learning_gate,
    learning_schedules,
    paired_analysis,
    relabel,
)
from history_control_model import GradientWitness, learn_control, observe_target, repair_grid, verify_gradients
from prospective import run_learning
from prospective_data import balanced_batches, rows_for, seed_for
from protocol import ROOT, digest, file_hash, read_json, write_json

design = fixtures.design
tiny_model = fixtures.tiny_model


@pytest.fixture
def protocol():
    return read_json(ROOT / "history-control-protocol.json")


def native(rows, predictions):
    values = []
    for row, prediction in zip(rows, predictions, strict=True):
        logits = [-2.0] * 16
        logits[prediction - 32] = 2.0
        values.append(
            {key: row[key] for key in ("intent", "target", "text_sha256")}
            | {"output": {"prediction": prediction, "code_logits": logits}}
        )
    return values


def panel(stream, condition, old_correct=10, new_correct=10):
    original = rows_for(stream, stream["old"], "gate")
    own = (
        relabel(original, alternate_mapping(stream)[1], list(range(32, 48)))
        if condition == "alternate_binding"
        else original
    )
    new = rows_for(stream, stream["new"], "gate")
    old_predictions = [row["target"] if index % 10 < old_correct else 47 for index, row in enumerate(own)]
    new_predictions = [row["target"] if index % 10 < new_correct else 32 for index, row in enumerate(new)]
    return {
        "old_own": native(own, old_predictions),
        "old_original": native(original, old_predictions),
        "new": native(new, new_predictions),
    }


def observations(stream, condition, old_correct=10, new_correct=10, initial_correct=0):
    values = {"initial": panel(stream, condition, initial_correct, 0)}
    if condition == "alternate_binding":
        values["acquired"] = panel(stream, condition, old_correct, 0)
    values["new_midpoint"] = panel(stream, condition, 0, new_correct)
    values["post_stream"] = panel(stream, condition, 0, new_correct)
    return values


def all_controls(protocol):
    streams, historical, controls = {}, {}, {}
    for declaration in protocol["streams"]:
        name = declaration["id"]
        stream = fixtures.fake_stream(name, declaration["split"])
        streams[name] = stream
        historical[name] = {"eligible": sorted(stream["old"])}
        controls[name] = {
            "new_only": {"qualified": True, "alternate_acquired": []},
            "alternate_binding": {"qualified": True, "alternate_acquired": sorted(stream["old"])},
        }
    return streams, historical, controls


def test_parent_input_protocol_and_original_sources_bound():
    context = sources.load_context()
    assert (
        context["design"]["parent_decision"]["sha256"]
        == "9d7ae6269ba2a5c67a37f2f131dcce66e06cf1b67f2d92c1fb21eaba740ec16d"
    )
    assert (
        context["protocol"]["original_source_sha256"]
        == "99ac3030bb56b44579b07e0f672ce098b0831f9c871b350d087201ad546ba8d1"
    )
    assert len(context["protocol"]["original_file_sha256"]) == 47
    assert context["spec"]["recovered"] == context["protocol"]["repair"]["qualification"]
    assert "may occur after" in context["design"]["temporal_interpretation"]


def test_changed_frozen_source_rejected(monkeypatch):
    original = sources.file_hash
    monkeypatch.setattr(
        sources, "file_hash", lambda path: "changed" if Path(path).name == "prospective_model.py" else original(path)
    )
    with pytest.raises(ValueError, match="FROZEN_ORIGINAL_SOURCE_CHANGED"):
        sources.load_context()


def test_alternate_rotation_has_no_fixed_points_and_keeps_text_and_tokens():
    data = read_json(ROOT / "inputs/budget-prediction-cohort.json")
    for stream in data["streams"].values():
        original, alternate = alternate_mapping(stream)
        assert all(original[name] != alternate[name] for name in stream["old"])
        assert set(original.values()) == set(alternate.values())
        rows = rows_for(stream, stream["old"], "learn")
        shifted = relabel(rows, alternate, data["codes"])
        for left, right in zip(rows, shifted, strict=True):
            assert {key: value for key, value in left.items() if key not in {"code", "target"}} == {
                key: value for key, value in right.items() if key not in {"code", "target"}
            }
            assert right["target"] == data["codes"][right["code"]]
        new = copy.deepcopy(rows_for(stream, stream["new"], "learn"))
        assert new == rows_for(stream, stream["new"], "learn")


def test_rebalancing_rotated_codes_would_change_historical_inputs(design, tmp_path):
    config, spec = design
    stream = fixtures.fake_stream()
    rows = rows_for(stream, stream["old"], "learn")
    seed = seed_for(stream["seed"], "acquisition")
    batches = balanced_batches(rows, 128, 8, seed)
    rotated = relabel(rows, alternate_mapping(stream)[1], list(range(32, 48)))
    assert input_schedule(rows, batches) == input_schedule(rotated, batches)
    assert input_schedule(rows, batches) != input_schedule(rotated, balanced_batches(rotated, 128, 8, seed))
    for kind, filename, seed_name in (
        ("old", "acquisition-updates.json", "acquisition"),
        ("new", "forgetting-updates.json", "forgetting"),
    ):
        plan = balanced_batches(rows_for(stream, stream[kind], "learn"), 128, 8, seed_for(stream["seed"], seed_name))
        write_json(
            tmp_path / filename,
            [{"step": i + 1, "rows": batch, "loss": 1.0, "gradient_norm": 1.0} for i, batch in enumerate(plan)],
        )
    result = learning_schedules(tmp_path, config, spec, stream)
    assert result["old"]["batches"] == batches
    assert len(result["new"]["batches"]) == 128


@pytest.mark.parametrize(
    "old_correct,initial_correct,expected", [(8, 0, True), (7, 0, False), (10, 10, False), (8, 5, True)]
)
def test_alternate_own_acquisition_uses_accuracy_and_genuine_gain(
    design, protocol, old_correct, initial_correct, expected
):
    _, spec = design
    stream = fixtures.fake_stream()
    values = observations(stream, "alternate_binding", old_correct=old_correct, initial_correct=initial_correct)
    if not expected:
        del values["post_stream"], values["new_midpoint"]
    result = learning_gate(values, "alternate_binding", stream, spec, protocol, list(range(32, 48)))
    assert result["old_source_passed"] is expected
    assert result["qualified"] is expected
    if expected:
        assert len(result["alternate_acquired"]) == 8
        assert all(value == 0 for value in result["scores"]["acquired"]["old_original"].values())


@pytest.mark.parametrize("condition", ["new_only", "alternate_binding"])
def test_new_phase_gain_is_from_correct_before_new_checkpoint(design, protocol, condition):
    _, spec = design
    stream = fixtures.fake_stream()
    values = observations(stream, condition)
    before = "initial" if condition == "new_only" else "acquired"
    values[before]["new"] = values["post_stream"]["new"]
    result = learning_gate(values, condition, stream, spec, protocol, list(range(32, 48)))
    assert not result["qualified"] and result["new_acquired"] == []


def test_new_only_has_no_old_acquisition_or_forgetting_requirement(design, protocol):
    _, spec = design
    stream = fixtures.fake_stream()
    values = observations(stream, "new_only")
    result = learning_gate(values, "new_only", stream, spec, protocol, list(range(32, 48)))
    assert result["qualified"] and result["alternate_acquired"] == []
    assert not result["old_acquisition_required"]


def test_old_gate_failure_cannot_be_followed_by_new_updates(design, protocol):
    _, spec = design
    stream = fixtures.fake_stream()
    with pytest.raises(ValueError, match="NEW_UPDATES_AFTER_FAILED_OLD_GATE"):
        learning_gate(
            observations(stream, "alternate_binding", old_correct=7),
            "alternate_binding",
            stream,
            spec,
            protocol,
            list(range(32, 48)),
        )


def test_native_logit_contract_rejects_fake_correct_prediction(design, protocol):
    _, spec = design
    stream = fixtures.fake_stream()
    values = observations(stream, "new_only")
    values["initial"]["new"][0]["output"]["prediction"] = 5000
    with pytest.raises(ValueError, match="NATIVE_CODE_CONTRACT"):
        learning_gate(values, "new_only", stream, spec, protocol, list(range(32, 48)))


def test_common_gate_intersects_without_using_control_repair_or_original_baseline(protocol):
    streams, historical, controls = all_controls(protocol)
    name = next(iter(streams))
    controls[name]["alternate_binding"]["alternate_acquired"] = streams[name]["old"][2:]
    controls[name]["new_only"]["irrelevant_repair_outcome"] = 1.0
    gate = common_gate(historical, controls, streams, protocol)
    assert gate["qualified"]
    assert len(gate["streams"][name]["common"]) == 6
    assert len(gate["streams"][name]["exclusions"]) == 2
    assert gate["source_clusters"] == {"train": 2, "test": 2}


@pytest.mark.parametrize("failure", ["one_source", "three_common", "missing_condition", "missing_stream"])
def test_common_gate_is_all_sources_and_never_vacuous(protocol, failure):
    streams, historical, controls = all_controls(protocol)
    name = next(iter(streams))
    if failure == "one_source":
        controls[name]["new_only"]["qualified"] = False
    elif failure == "three_common":
        controls[name]["alternate_binding"]["alternate_acquired"] = streams[name]["old"][:3]
    elif failure == "missing_condition":
        del controls[name]["new_only"]
    else:
        del controls[name]
    if failure.startswith("missing"):
        with pytest.raises(ValueError, match="REQUIRED"):
            common_gate(historical, controls, streams, protocol)
    else:
        assert not common_gate(historical, controls, streams, protocol)["qualified"]


def test_failed_common_gate_stops_before_model_or_historical_actions(protocol, tmp_path, monkeypatch):
    context = {
        "config": {},
        "spec": {},
        "data": {},
        "protocol": protocol,
        "design": read_json(ROOT / "history-control-design.json"),
    }
    gate_path = tmp_path / "gate"
    gate_path.mkdir()
    write_json(gate_path / "receipt.json", {})
    monkeypatch.setattr(history_control, "historical_cohort", lambda context: ({}, {}))
    monkeypatch.setattr(history_control, "checked_common", lambda *args: {"qualified": False})

    def forbidden(*args):
        pytest.fail("No model, predictor, or historical outcome access after failed common gate")

    monkeypatch.setattr(history_control, "verify_model", forbidden)
    monkeypatch.setattr(history_control, "checked_historical_repairs", forbidden)
    result = history_control.run_stage({"stage": "repair", "gate": str(gate_path)}, context, tmp_path)
    assert result["model_loads"] == result["repair_updates"] == result["predictor_fits"] == 0


@pytest.mark.parametrize("condition", ["new_only", "alternate_binding"])
def test_actual_qwen_control_training_exact_schedules_and_optimizer_carry(
    tiny_model, protocol, tmp_path, condition, monkeypatch
):
    observer = tiny_model
    config = {"acquisition_updates": 4, "forgetting_updates": 4}
    spec = copy.deepcopy(observer.spec)
    spec["gates"].update(
        acquired_min=0,
        acquisition_gain_min=-1,
        minimum_acquired_per_stream=0,
        minimum_new_acquired_per_stream=0,
        minimum_forgotten_per_stream=0,
    )
    observer.spec = spec
    stream = fixtures.fake_stream()
    for roles in stream["units"].values():
        for role in roles:
            roles[role] = roles[role][:2]
    historical = tmp_path / "historical"
    historical.mkdir()
    run_learning(observer, config, spec, stream, historical)
    schedules = learning_schedules(historical, config, spec, stream)
    output = tmp_path / condition
    output.mkdir()
    result = learn_control(
        observer,
        config,
        spec,
        {"codes": list(range(32, 48))},
        stream,
        condition,
        protocol,
        historical,
        schedules,
        output,
    )
    assert result["qualified"]
    assert result["optimizer_steps"]["final"] == ([8] if condition == "alternate_binding" else [4])
    assert result["optimizer_steps"]["new_midpoint"] == ([6] if condition == "alternate_binding" else [2])
    assert result["test_rows_evaluated"] == result["repair_updates"] == 0
    assert result["old_updates"] == (4 if condition == "alternate_binding" else 0)
    assert file_hash(output / "initial/adapter_model.safetensors") == file_hash(
        historical / "initial/adapter_model.safetensors"
    )
    for kind in ["old", "new"] if condition == "alternate_binding" else ["new"]:
        trace = read_json(output / f"{kind}-updates.json")
        assert [entry["rows"] for entry in trace["updates"]] == schedules[kind]["batches"]
        assert (
            digest(input_schedule(trace["rows"], schedules[kind]["batches"]))
            == schedules[kind]["input_schedule_sha256"]
        )
    context = {
        "data": {"streams": {stream["id"]: stream}, "codes": list(range(32, 48))},
        "config": config,
        "spec": spec,
        "protocol": copy.deepcopy(protocol),
    }
    context["protocol"]["learning_sources"][stream["id"]] = {"source": str(historical)}
    monkeypatch.setattr(sources, "control_source", lambda *args: (result, {}, {}))
    gate = sources.checked_control_learning(output, stream["id"], condition, context, schedules)
    assert gate["qualified"]


def test_actual_qwen_failed_alternate_acquisition_preserves_receipt_without_new_updates(tiny_model, protocol, tmp_path):
    observer = tiny_model
    config = {"acquisition_updates": 2, "forgetting_updates": 2}
    spec = copy.deepcopy(observer.spec)
    spec["gates"].update(
        acquired_min=0,
        acquisition_gain_min=-1,
        minimum_acquired_per_stream=0,
        minimum_new_acquired_per_stream=0,
        minimum_forgotten_per_stream=0,
    )
    observer.spec = spec
    stream = fixtures.fake_stream()
    for roles in stream["units"].values():
        for role in roles:
            roles[role] = roles[role][:1]
    historical = tmp_path / "historical"
    historical.mkdir()
    run_learning(observer, config, spec, stream, historical)
    spec["gates"]["acquired_min"] = 1.1
    result = learn_control(
        observer,
        config,
        spec,
        {"codes": list(range(32, 48))},
        stream,
        "alternate_binding",
        protocol,
        historical,
        learning_schedules(historical, config, spec, stream),
        tmp_path / "control",
    )
    assert not result["qualified"] and result["old_updates"] == 2 and result["new_updates"] == 0
    assert not (tmp_path / "control/post_stream").exists()


def test_actual_prefix_gradients_weights_and_clocks_match_and_witness_does_not_change_updates(
    tiny_model, protocol, tmp_path
):
    observer = tiny_model
    stream = fixtures.fake_stream()
    for roles in stream["units"].values():
        for role in roles:
            roles[role] = roles[role][:1]
    observer.checkpoint(tmp_path / "learning/forgotten")
    baseline = observe_target(observer, stream, stream["old"][0])
    outcome = repair_grid(
        observer,
        observer.spec,
        protocol,
        stream,
        stream["old"][0],
        tmp_path / "learning/forgotten",
        baseline,
        tmp_path / "control",
    )
    unit = {"intent": stream["old"][0], "actions": outcome}
    verify_grid(tmp_path / "control", unit, stream, observer.spec, {"budgets": [2, 4, 8, 16]})
    verify_gradients(tmp_path / "control", unit, [2, 4, 8, 16])
    original = original_grid(
        observer,
        observer.spec,
        {"budgets": [2, 4, 8, 16]},
        stream,
        stream["old"][0],
        tmp_path / "learning",
        tmp_path / "unobserved",
    )
    for budget in ("2", "4", "8", "16"):
        for key in ("test", "guard", "trace_sha256", "adapter_sha256", "schedule_sha256"):
            assert outcome[budget][key] == original[budget][key]
        assert outcome[budget]["optimizer_steps"] == [int(budget)]
    assert len(read_json(tmp_path / "control/actions" / stream["old"][0] / "16-gradients.json")) == 16


def test_gradient_witness_requires_actual_nonempty_finite_gradients(tiny_model, tmp_path):
    optimizer = tiny_model.optimizer()
    witness = GradientWitness(tiny_model, optimizer, tmp_path, [2])
    with pytest.raises(ValueError, match="MISSING_OR_NONFINITE_GRADIENT"):
        witness.before_step(optimizer, (), {})


@pytest.mark.parametrize("missing", ["forecast", "reverse_time"])
def test_original_test_forecast_required_before_any_control_model(tmp_path, protocol, monkeypatch, missing):
    stream = fixtures.fake_stream("fresh-test-a", "test")
    path = tmp_path / "original"
    path.mkdir()
    write_json(path / "receipt.json", {})
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    write_json(forecast / "receipt.json", {"sealed": True})
    receipt = {
        "stream": stream["id"],
        "cohort_sha256": protocol["original_cohort"]["receipt_sha256"],
        "forecasts_sha256": file_hash(forecast / "receipt.json"),
        "readout_gate_sha256": "readout",
    }

    def source(path, stage, *args):
        return receipt, {}, {"started_at": "2026-09-12T07:00:00+00:00", "finished_at": "2026-09-12T08:00:00+00:00"}

    monkeypatch.setattr(sources, "original_source", source)
    with pytest.raises(ValueError, match="FORECAST_REQUIRED|TEMPORAL_ORDER"):
        sources.checked_historical_repairs(
            path, stream, {"protocol": protocol}, [], None if missing == "forecast" else forecast
        )


def test_time_requires_actual_order_and_manifest_requires_test_forecast():
    sources.time_before("2026-09-12T07:00:00+00:00", "2026-09-12T08:00:00+00:00", "okay")
    with pytest.raises(ValueError, match="TEMPORAL_ORDER"):
        sources.time_before("2026-09-12T09:00:00+00:00", "2026-09-12T08:00:00+00:00", "late")
    context = sources.load_context()
    manifest = {
        "stage": "repair",
        "run_id": "task",
        "stream": "fresh-test-a",
        "condition": "new_only",
        "gate": "gate",
        "historical_repair": "original",
    }
    with pytest.raises(ValueError, match="MANIFEST_FIELDS"):
        history_control.validate_manifest(manifest, context)
    history_control.validate_manifest(manifest | {"forecasts": "forecast"}, context)


def sample_unit(name, split, intent, accuracy_by_budget, guards_by_budget, base_accuracy=0, base_guard=10):
    rows = fixtures.rows(intent, 0, "test")
    guards = fixtures.rows("new", 8, "test")
    return {
        "id": f"{name}/{intent}",
        "intent": intent,
        "stream": name,
        "split": split,
        "baseline": {"test": fixtures.records(rows, base_accuracy), "guard": fixtures.records(guards, base_guard)},
        "actions": {
            str(b): {
                "test": fixtures.records(rows, accuracy_by_budget[i]),
                "guard": fixtures.records(guards, guards_by_budget[i]),
            }
            for i, b in enumerate((2, 4, 8, 16))
        },
    }


def test_guard_aware_target_qualification_reports_nonmonotonicity_and_never(protocol):
    unit = sample_unit("train-a", "train", "old", [10, 10, 10, 10], [10, 5, 10, 5])
    result = curve(unit, protocol["repair"]["qualification"], [2, 4, 8, 16])
    assert result["minimum_budget"] == "2" and result["later_qualification_loss"]
    assert result["qualified_budgets"] == [2, 8]
    assert result["curve"]["4"]["utility"] == 0.5
    unit = sample_unit("train-a", "train", "old", [10] * 4, [10] * 4, base_accuracy=10)
    result = curve(unit, protocol["repair"]["qualification"], [2, 4, 8, 16])
    assert result["minimum_budget"] == "never"
    assert all(value["accuracy_passed"] for value in result["curve"].values())


def test_paired_guard_excess_and_equal_source_aggregation(protocol):
    streams, historical, controls = all_controls(protocol)
    for index, name in enumerate(streams):
        historical[name]["eligible"] = historical[name]["eligible"][: 4 if index % 2 else 8]
    gate = common_gate(historical, controls, streams, protocol)
    units = {}
    for index, (name, stream) in enumerate(streams.items()):
        units[name] = {condition: [] for condition in ["original_history", *protocol["conditions"]]}
        for intent in gate["streams"][name]["common"]:
            units[name]["original_history"].append(sample_unit(name, stream["split"], intent, [10] * 4, [9] * 4))
            units[name]["new_only"].append(
                sample_unit(name, stream["split"], intent, [0 if index % 2 else 10] * 4, [2] * 4, base_guard=2)
            )
            units[name]["alternate_binding"].append(sample_unit(name, stream["split"], intent, [10] * 4, [7] * 4))
    result = paired_analysis(units, streams, gate, protocol)
    for split in ("train", "test"):
        summary = result["aggregate"][split]
        assert summary["source_clusters"] == 2 and summary["intents"] == 12
        assert summary["equal_source"]["means"]["new_only"]["2"]["accuracy"] == 0.5
        assert summary["pooled_intents_descriptive"]["means"]["new_only"]["2"]["accuracy"] == pytest.approx(2 / 3)
        excess = summary["equal_source"]["paired_historical_excess"]["new_only"]["2"]
        assert excess["guard"] == pytest.approx(0.7)
        assert excess["guard_loss"] == pytest.approx(0.1)
    assert "interference" in result["interpretation_limit"]


def test_dispatch_accepts_only_config_and_existing_run_dir_and_refuses_overwrite(tmp_path, monkeypatch):
    context = sources.load_context()
    manifest = {"stage": "learn", "run_id": "history-job", "stream": "fresh-train-a", "condition": "new_only"}
    config = tmp_path / "manifest.json"
    write_json(config, manifest)
    output = tmp_path / "history-job"
    output.mkdir()
    called = []

    def run(manifest, context, path):
        called.append(path)
        return {"qualified": False, "status": "cpu_dispatch_contract", "repair_updates": 0}

    monkeypatch.setattr(history_control, "run_stage", run)
    history_control.main(["--config", str(config), "--output-dir", str(output)])
    assert called == [output / "study"]
    seal = read_json(output / "study/seal.json")["payload"]
    assert seal["history_identity"] == context["identity"]
    assert seal["history_design"]["parent_decision"] == context["design"]["parent_decision"]
    with pytest.raises(FileExistsError):
        history_control.main(["--config", str(config), "--output-dir", str(output)])


def test_ready_graph_has_eight_sources_then_common_gate_and_forecast_before_test_repair():
    context = sources.load_context()
    jobs = read_json(ROOT / "history-control-dependencies.json")
    assert len(jobs) == 18
    learn_jobs = []
    for run_id, job in jobs.items():
        manifest = read_json(ROOT / job["config"])
        history_control.validate_manifest(manifest, context)
        assert manifest["run_id"] == run_id and job["gpus"] == 1
        assert job["entrypoint"] == "history_control.py"
        if manifest["stage"] == "learn":
            learn_jobs.append(run_id)
        if manifest["stage"] == "repair":
            assert "followthrough-20260912-recovery-history-gate" in job["depends_on"]
            assert manifest["historical_repair"].split("/")[-2] in job["depends_on"]
            if manifest["stream"].startswith("fresh-test-"):
                assert manifest["forecasts"].split("/")[-2] in job["depends_on"]
    assert len(learn_jobs) == 8
    assert set(jobs["followthrough-20260912-recovery-history-gate"]["depends_on"]) == set(learn_jobs)
    internal = set(jobs)
    emitted = set()
    while len(emitted) < len(jobs):
        ready = {
            run_id
            for run_id, job in jobs.items()
            if set(job["depends_on"]) & internal <= emitted and run_id not in emitted
        }
        assert ready
        emitted |= ready
