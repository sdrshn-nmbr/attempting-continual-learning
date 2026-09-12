import math
from datetime import datetime
from pathlib import Path

from budget_prediction import completed, load_study
from budget_prediction_model import verify_grid
from history_control_analysis import (
    alternate_mapping,
    common_gate,
    input_schedule,
    learning_gate,
    learning_schedules,
    relabel,
)
from history_control_model import verify_gradients
from minimum_budget import checked_records, schedule_for
from prospective import qualify_learning
from prospective_data import PROMPT, rows_for
from protocol import ROOT, digest, file_hash, read_json

SOURCE_FILES = (
    "history_control.py",
    "history_control_sources.py",
    "history_control_model.py",
    "history_control_analysis.py",
)


def load_context():
    documents = {}
    for name in ("protocol", "design"):
        path = ROOT / f"history-control-{name}.json"
        if file_hash(path) != path.with_suffix(".sha256").read_text().strip():
            raise ValueError(f"HISTORY_CONTROL_DOCUMENT_HASH {name}")
        documents[name] = read_json(path)
    protocol, design = documents["protocol"], documents["design"]
    parent = design["parent_decision"]
    if (
        digest(parent["payload"]) != parent["sha256"]
        or design["protocol_sha256"] != file_hash(ROOT / "history-control-protocol.json")
        or design["parent_decision_at"] != parent["payload"]["at"]
        or design["lane_protocol_at"] != protocol["sealed_at_utc"]
        or parent["payload"]["dataset_sha256"]
        != protocol["original_file_sha256"]["inputs/budget-prediction-cohort.json"]
        or parent["payload"]["repair_budgets"] != protocol["repair"]["budgets"]
        or datetime.fromisoformat(design["parent_decision_at"]) > datetime.fromisoformat(design["lane_protocol_at"])
    ):
        raise ValueError("HISTORY_CONTROL_PARENT_DECISION_BINDING")
    for name, expected in protocol["original_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"HISTORY_CONTROL_FROZEN_ORIGINAL_SOURCE_CHANGED {name}")
    original = load_study({"study_config": protocol["original_config"]})
    config, spec, data = original["config"], original["spec"], original["data"]
    if (
        original["identity"] != protocol["original_prediction_identity"]
        or spec["recovered"] != protocol["repair"]["qualification"]
        or PROMPT != protocol["prompt"]
        or config["acquisition_updates"] != protocol["training"]["alternate_binding"]["old_updates"]
        or config["forgetting_updates"] != protocol["training"]["new_only"]["new_updates"]
        or config["forgetting_updates"] != protocol["training"]["alternate_binding"]["new_updates"]
        or spec["optimizer"] != protocol["training"]["settings"]
        or spec["gates"]["acquired_min"] != protocol["source_qualification"]["acquired_min"]
        or spec["gates"]["acquisition_gain_min"] != protocol["source_qualification"]["acquisition_gain_min"]
    ):
        raise ValueError("HISTORY_CONTROL_ORIGINAL_LEARNING_CONTRACT")
    for name, stream in data["streams"].items():
        original_map, alternate = alternate_mapping(stream)
        declared = protocol["mappings"][name]
        if (
            declared["original"] != original_map
            or declared["alternate"] != alternate
            or declared["old_intent_order"] != stream["old"]
        ):
            raise ValueError("HISTORY_CONTROL_DECLARED_ALTERNATE_MAPPING")
    implementation = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    identity = digest(
        {
            "protocol": protocol,
            "design": design,
            "implementation": implementation,
            "original_prediction_identity": original["identity"],
        }
    )
    return {
        "protocol": protocol,
        "design": design,
        "original": original,
        "config": config,
        "spec": spec,
        "data": data,
        "identity": identity,
        "implementation": implementation,
    }


def original_source(path, stage, context, run_id, receipt_hash=None):
    path = Path(path)
    receipt, seal, execution = completed(path)
    if (
        receipt["pipeline_stage"] != stage
        or not receipt["qualified"]
        or seal["prediction_identity"] != context["protocol"]["original_prediction_identity"]
        or execution["task_id"] != run_id
        or execution["source_sha256"] != context["protocol"]["original_source_sha256"]
        or execution["task"]["config"] != seal["dispatch_manifest"]
        or receipt_hash is not None
        and file_hash(path / "receipt.json") != receipt_hash
    ):
        raise ValueError(f"HISTORY_CONTROL_ORIGINAL_SOURCE_BINDING stage={stage}")
    return receipt, seal, execution


def historical_cohort(context):
    protocol, spec, data = context["protocol"], context["spec"], context["data"]
    cohort_rule = protocol["original_cohort"]
    receipt, _, _ = original_source(
        cohort_rule["source"], "cohort", context, cohort_rule["run_id"], cohort_rule["receipt_sha256"]
    )
    paths, schedules = [], {}
    for name, rule in protocol["learning_sources"].items():
        path = Path(rule["source"])
        result, _, _ = original_source(path, "learn", context, rule["run_id"], rule["receipt_sha256"])
        stream = data["streams"][name]
        if (
            result["stream"] != name
            or result["eligible"] != rule["eligible"]
            or result["optimizer_steps"] != {"acquired": [128], "final": [256]}
        ):
            raise ValueError("HISTORY_CONTROL_ORIGINAL_STREAM_ELIGIBILITY_OR_CLOCK")
        for filename, key in (
            ("initial/adapter_model.safetensors", "initial_adapter_sha256"),
            ("acquisition-updates.json", "acquisition_trace_sha256"),
            ("forgetting-updates.json", "new_trace_sha256"),
        ):
            if file_hash(path / filename) != rule[key]:
                raise ValueError("HISTORY_CONTROL_ORIGINAL_LEARNING_ARTIFACT_CHANGED")
        for phase in read_json(path / "observations.json").values():
            for kind in ("old", "new"):
                checked_records(phase[kind + "_gate"], rows_for(stream, stream[kind], "gate"), data["codes"])
        schedules[name] = learning_schedules(path, context["config"], spec, stream)
        paths.append(path)
    recomputed = qualify_learning(paths, context["config"], spec, data)
    if not recomputed["qualified"] or any(receipt[key] != value for key, value in recomputed.items()):
        raise ValueError("HISTORY_CONTROL_ORIGINAL_COHORT_RECOMPUTATION")
    return receipt, schedules


def control_source(path, stage, context, stream=None, condition=None):
    path = Path(path)
    receipt, seal, execution = completed(path)
    manifest = seal["dispatch_manifest"]
    if (
        receipt["stage"] != "history-control-" + stage
        or seal["history_identity"] != context["identity"]
        or manifest["stage"] != stage
        or execution["task_id"] != manifest["run_id"]
        or execution["task"]["config"] != manifest
        or stream is not None
        and receipt["stream"] != stream
        or stream is not None
        and manifest["stream"] != stream
        or condition is not None
        and receipt["condition"] != condition
        or condition is not None
        and manifest["condition"] != condition
    ):
        raise ValueError(f"HISTORY_CONTROL_OWN_SOURCE_BINDING stage={stage}")
    return receipt, seal, execution


def checked_control_learning(path, name, condition, context, schedules):
    path = Path(path)
    receipt, _, _ = control_source(path, "learn", context, name, condition)
    stream, data, spec, protocol = (
        context["data"]["streams"][name],
        context["data"],
        context["spec"],
        context["protocol"],
    )
    observed = read_json(path / "observations.json")
    gate = learning_gate(observed, condition, stream, spec, protocol, data["codes"])
    if read_json(path / "learning-gate.json") != gate or any(receipt[key] != value for key, value in gate.items()):
        raise ValueError("HISTORY_CONTROL_SOURCE_GATE_RECOMPUTATION")
    initial = Path(protocol["learning_sources"][name]["source"]) / "initial/adapter_model.safetensors"
    original_path = Path(protocol["learning_sources"][name]["source"])
    original_initial = read_json(original_path / "observations.json")["initial"]
    if (
        file_hash(initial) != receipt["initial_adapter_sha256"]
        or file_hash(path / "initial/adapter_model.safetensors") != file_hash(initial)
        or read_json(path / "schedules-before-updates.json") != schedules
        or observed["initial"]["old_original"] != original_initial["old_gate"]
        or observed["initial"]["new"] != original_initial["new_gate"]
        or read_json(path / "frozen-backbone.json") != read_json(original_path / "frozen-backbone.json")
    ):
        raise ValueError("HISTORY_CONTROL_INITIAL_OR_SCHEDULE_BINDING")
    clocks = {"initial": []}
    old_updates = context["config"]["acquisition_updates"] if condition == "alternate_binding" else 0
    new_updates = context["config"]["forgetting_updates"] if gate["old_source_passed"] else 0
    for kind, count in (("old", old_updates), ("new", new_updates)):
        path_trace = path / f"{kind}-updates.json"
        if not count:
            if path_trace.exists():
                raise ValueError("HISTORY_CONTROL_FORBIDDEN_TRAINING_PHASE")
            continue
        trace = read_json(path_trace)
        rows = schedules[kind]["rows"]
        if kind == "old":
            rows = relabel(rows, alternate_mapping(stream)[1], data["codes"])
        if (
            trace["rows"] != rows
            or [entry["step"] for entry in trace["updates"]] != list(range(1, count + 1))
            or [entry["rows"] for entry in trace["updates"]] != schedules[kind]["batches"]
            or any(not math.isfinite(entry[key]) for entry in trace["updates"] for key in ("loss", "gradient_norm"))
            or digest(input_schedule(trace["rows"], schedules[kind]["batches"]))
            != schedules[kind]["input_schedule_sha256"]
        ):
            raise ValueError("HISTORY_CONTROL_ACTUAL_CONTROL_TRAINING_SCHEDULE")
    if old_updates:
        clocks["acquired"] = [old_updates]
    if new_updates:
        clocks["new_midpoint"] = [old_updates + new_updates // 2]
    clocks["final"] = [old_updates + new_updates]
    if (
        receipt["optimizer_steps"] != clocks
        or receipt["old_updates"] != old_updates
        or receipt["new_updates"] != new_updates
        or receipt["old_exposures"] != old_updates * spec["optimizer"]["batch_size"]
        or receipt["new_exposures"] != new_updates * spec["optimizer"]["batch_size"]
        or receipt["lens_updates"] != 0
        or receipt["predictor_fits"] != 0
    ):
        raise ValueError("HISTORY_CONTROL_CONTROL_SOURCE_UPDATE_ACCOUNTING")
    return gate


def recompute_common(paths, context, historical, schedules):
    if set(paths) != set(context["data"]["streams"]):
        raise ValueError("HISTORY_CONTROL_COMMON_ALL_STREAMS_REQUIRED")
    controls, hashes = {}, {}
    for name in context["data"]["streams"]:
        if set(paths[name]) != set(context["protocol"]["conditions"]):
            raise ValueError("HISTORY_CONTROL_COMMON_ALL_CONDITIONS_REQUIRED")
        controls[name], hashes[name] = {}, {}
        for condition, path in paths[name].items():
            controls[name][condition] = checked_control_learning(path, name, condition, context, schedules[name])
            hashes[name][condition] = file_hash(Path(path) / "receipt.json")
    gate = common_gate(historical["streams"], controls, context["data"]["streams"], context["protocol"])
    return gate | {
        "control_sources": paths,
        "control_receipts": hashes,
        "historical_cohort_sha256": context["protocol"]["original_cohort"]["receipt_sha256"],
    }


def checked_common(path, context, historical, schedules):
    receipt, seal, _ = control_source(path, "gate", context)
    gate = recompute_common(seal["dispatch_manifest"]["sources"], context, historical, schedules)
    if read_json(Path(path) / "common-gate.json") != gate or any(receipt[key] != value for key, value in gate.items()):
        raise ValueError("HISTORY_CONTROL_COMMON_GATE_RECOMPUTATION")
    return gate


def time_before(earlier, later, label):
    if datetime.fromisoformat(earlier) > datetime.fromisoformat(later):
        raise ValueError(f"HISTORY_CONTROL_TEMPORAL_ORDER {label}")


def checked_historical_repairs(path, stream, context, common, forecast_path=None):
    name = stream["id"]
    run_id = "followthrough-20260912-recovery-predict-repair-" + name.removeprefix("fresh-")
    receipt, _, execution = original_source(path, "repair", context, run_id)
    if (
        receipt["stream"] != name
        or receipt["cohort_sha256"] != context["protocol"]["original_cohort"]["receipt_sha256"]
    ):
        raise ValueError("HISTORY_CONTROL_HISTORICAL_REPAIR_COHORT")
    provenance = {
        "historical_repair_sha256": file_hash(Path(path) / "receipt.json"),
        "historical_repair_finished_at": execution["finished_at"],
    }
    if stream["split"] == "test":
        if forecast_path is None:
            raise ValueError("HISTORY_CONTROL_ORIGINAL_TEST_FORECAST_REQUIRED")
        forecast, _, forecast_execution = original_source(
            forecast_path, "forecast", context, context["protocol"]["test_ordering"]["require_forecast"]
        )
        forecast_hash = file_hash(Path(forecast_path) / "receipt.json")
        if (
            receipt["forecasts_sha256"] != forecast_hash
            or forecast["cohort_sha256"] != receipt["cohort_sha256"]
            or forecast["readout_gate_sha256"] != receipt["readout_gate_sha256"]
        ):
            raise ValueError("HISTORY_CONTROL_ORIGINAL_FORECAST_REPAIR_BINDING")
        time_before(
            forecast_execution["finished_at"], execution["started_at"], "original_forecast_before_original_TEST"
        )
        provenance.update(forecast_sha256=forecast_hash, forecast_finished_at=forecast_execution["finished_at"])
    original_units = read_json(Path(path) / "units.json")
    eligible = context["protocol"]["learning_sources"][name]["eligible"]
    if [unit["intent"] for unit in original_units] != eligible or receipt["unit_ids"] != [
        u["id"] for u in original_units
    ]:
        raise ValueError("HISTORY_CONTROL_ORIGINAL_REPAIR_ELIGIBLE_SET")
    spec, codes, budgets = context["spec"], context["data"]["codes"], context["protocol"]["repair"]["budgets"]
    units = []
    start_hash = file_hash(
        Path(context["protocol"]["learning_sources"][name]["source"]) / "forgotten/adapter_model.safetensors"
    )
    for unit in original_units:
        if unit["id"] != f"{name}/{unit['intent']}" or unit["stream"] != name or unit["split"] != stream["split"]:
            raise ValueError("HISTORY_CONTROL_HISTORICAL_UNIT_IDENTITY")
        verify_grid(Path(path), unit, stream, spec, {"budgets": budgets})
        if set(unit["actions"]) != {str(budget) for budget in budgets}:
            raise ValueError("HISTORY_CONTROL_HISTORICAL_BUDGET_SET")
        for panel in [*unit["baseline"].values(), *unit["actions"].values()]:
            checked_records(panel["test"], stream["units"][unit["intent"]]["test"], codes)
            checked_records(panel["guard"], rows_for(stream, stream["new"], "test"), codes)
        if any(
            action["start_adapter_sha256"] != start_hash or not action["fresh_optimizer"]
            for action in unit["actions"].values()
        ):
            raise ValueError("HISTORY_CONTROL_HISTORICAL_REPAIR_START")
        if unit["intent"] in common:
            units.append(
                {key: unit[key] for key in ("id", "intent", "stream", "split", "actions")}
                | {"baseline": unit["baseline"]["after"], "condition": "original_history"}
            )
    if [unit["intent"] for unit in units] != common:
        raise ValueError("HISTORY_CONTROL_HISTORICAL_COMMON_SELECTION")
    return units, provenance


def checked_control_repairs(path, name, condition, gate_path, context, common, provenance):
    path = Path(path)
    receipt, _, execution = control_source(path, "repair", context, name, condition)
    if (
        not receipt["qualified"]
        or receipt["common_gate_sha256"] != file_hash(Path(gate_path) / "receipt.json")
        or any(receipt[key] != value for key, value in provenance.items())
    ):
        raise ValueError("HISTORY_CONTROL_CONTROL_REPAIR_PREREQUISITES")
    stream = context["data"]["streams"][name]
    gate = read_json(Path(gate_path) / "common-gate.json")
    learning = Path(gate["control_sources"][name][condition])
    start_hash = file_hash(learning / "post_stream/adapter_model.safetensors")
    if receipt["control_learning_sha256"] != file_hash(learning / "receipt.json") or read_json(
        path / "frozen-backbone.json"
    ) != read_json(learning / "frozen-backbone.json"):
        raise ValueError("HISTORY_CONTROL_REPAIR_LEARNING_SOURCE_BINDING")
    baseline = read_json(path / "baselines-before-updates.json")
    units = read_json(path / "units.json")
    if (
        [unit["intent"] for unit in units] != common
        or receipt["unit_ids"] != [unit["id"] for unit in units]
        or baseline["units"] != [unit | {"actions": {}} for unit in units]
        or file_hash(path / "baselines-before-updates.json") != receipt["baselines_sha256"]
    ):
        raise ValueError("HISTORY_CONTROL_CONTROL_REPAIR_BASELINE_OR_UNIT_SET")
    budgets, spec, codes = context["protocol"]["repair"]["budgets"], context["spec"], context["data"]["codes"]
    if receipt["repair_updates"] != len(units) * sum(budgets) or receipt["predictor_fits"] or receipt["lens_updates"]:
        raise ValueError("HISTORY_CONTROL_CONTROL_REPAIR_UPDATE_ACCOUNTING")
    time_before(provenance["historical_repair_finished_at"], execution["started_at"], "historical_before_control")
    _, _, gate_execution = completed(gate_path)
    time_before(gate_execution["finished_at"], execution["started_at"], "common_gate_before_control")
    if stream["split"] == "test":
        time_before(provenance["forecast_finished_at"], execution["started_at"], "forecast_before_control_TEST")
    for unit in units:
        if (
            unit["id"] != f"{name}/{unit['intent']}"
            or unit["stream"] != name
            or unit["split"] != stream["split"]
            or unit["condition"] != condition
        ):
            raise ValueError("HISTORY_CONTROL_CONTROL_REPAIR_UNIT_IDENTITY")
        verify_grid(path, unit, stream, spec, {"budgets": budgets})
        verify_gradients(path, unit, budgets)
        for panel in [unit["baseline"], *unit["actions"].values()]:
            checked_records(panel["test"], stream["units"][unit["intent"]]["test"], codes)
            checked_records(panel["guard"], rows_for(stream, stream["new"], "test"), codes)
        rows, batches = schedule_for(stream, unit["intent"], spec, max(budgets))
        for budget in budgets:
            action = unit["actions"][str(budget)]
            if (
                not action["fresh_optimizer"]
                or action["optimizer_steps"] != [budget]
                or action["start_adapter_sha256"] != start_hash
                or not action["reload_exact"]
                or action["input_schedule_sha256"] != digest(input_schedule(rows, batches[:budget]))
            ):
                raise ValueError("HISTORY_CONTROL_CONTROL_REPAIR_CLOCK_OR_UID_SEQUENCE")
            times = read_json(path / "actions" / unit["intent"] / f"{budget}-timing.json")
            time_before(baseline["at"], times["started_at"], "baseline_before_any_repair")
            time_before(times["started_at"], times["finished_at"], "arm_timing")
    return units
