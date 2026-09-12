import argparse
import logging
import math
from pathlib import Path

from budget_prediction_analysis import evaluate, fit_models, forecasts, prediction_gate
from budget_prediction_model import readout, repair, verify_grid
from minimum_budget import (
    checked_learning_sources,
    checked_records,
    minimum_outcome,
    permutation_plan,
)
from minimum_budget import (
    load_manifest as load_pilot_manifest,
)
from minimum_budget import (
    qualify as recompute_pilot_gate,
)
from prospective import finish, make_seal, qualify_learning, run_learning, verify_model, verify_result
from prospective_analysis import feature_vector
from prospective_data import load_design, rows_for, validate_data
from prospective_model import ProspectiveModel
from protocol import ROOT, digest, file_hash, read_json, write_json

PROTOCOL = ROOT / "budget-prediction-protocol.json"
SOURCE_FILES = (
    "budget_prediction.py",
    "budget_prediction_analysis.py",
    "budget_prediction_model.py",
    "prepare_budget_prediction.py",
    "minimum_budget.py",
    "prospective.py",
    "prospective_data.py",
    "prospective_model.py",
    "prospective_analysis.py",
    "tuned_lens.py",
    "model.py",
    "protocol.py",
)


def load_study(manifest):
    if file_hash(PROTOCOL) != PROTOCOL.with_suffix(".sha256").read_text().strip():
        raise ValueError("BUDGET_PREDICTION_PROTOCOL_HASH")
    protocol = read_json(PROTOCOL)
    config, spec = load_design(ROOT / manifest["study_config"])
    if file_hash(ROOT / config["protocol"]) != protocol["learning"]["spec_sha256"]:
        raise ValueError("BUDGET_PREDICTION_LEARNING_SPEC_HASH")
    data = read_json(ROOT / config["dataset"])
    if file_hash(ROOT / config["dataset"]) != protocol["dataset_sha256"]:
        raise ValueError("BUDGET_PREDICTION_DATASET_HASH")
    validate_data(data, spec)
    prior = read_json(ROOT / "inputs/prospective-cohort.json")
    excluded = set(read_json(ROOT / "minimum-budget-protocol.json")["future_holdout"]["excluded_intents"])
    if (
        set(data["source"]["excluded_prior_intents"]) != excluded
        or data["source"]["prior_dataset_sha256"] != digest(prior)
        or data["source"]["preparation_implementation_sha256"] != file_hash(ROOT / "prepare_budget_prediction.py")
        or data["lens_fit"] != prior["lens_fit"]
        or data["lens_check"] != prior["lens_check"]
        or data["base_manifest"] != prior["base_manifest"]
        or data["codes"] != prior["codes"]
        or spec["gates"] != protocol["learning"]["gates"]
        or spec["recovered"] != protocol["recovered"]
    ):
        raise ValueError("BUDGET_PREDICTION_COHORT_OR_CONTRACT_BINDING")
    if any(excluded.intersection(stream["old"] + stream["new"]) for stream in data["streams"].values()):
        raise ValueError("BUDGET_PREDICTION_PREVIOUSLY_OBSERVED_INTENTS")
    prior_rows = [
        row
        for stream in prior["streams"].values()
        for roles in stream["units"].values()
        for rows in roles.values()
        for row in rows
    ]
    prior_text = {row["text_sha256"] for row in prior_rows}
    prior_tokens = {tuple(row["input_ids"]) for row in prior_rows}
    if any(
        row["text_sha256"] in prior_text or tuple(row["input_ids"]) in prior_tokens
        for stream in data["streams"].values()
        for roles in stream["units"].values()
        for rows in roles.values()
        for row in rows
    ):
        raise ValueError("BUDGET_PREDICTION_PREVIOUSLY_OBSERVED_ROWS")
    for split in ("train", "test"):
        if (
            sorted(name for name, stream in data["streams"].items() if stream["split"] == split)
            != protocol[f"{split}_streams"]
        ):
            raise ValueError("BUDGET_PREDICTION_DECLARED_SOURCE_SPLIT")
    implementation = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    identity = digest(
        {"config": config, "spec": spec, "data": digest(data), "protocol": protocol, "implementation": implementation}
    )
    return {
        "config": config,
        "spec": spec,
        "data": data,
        "protocol": protocol,
        "identity": identity,
        "implementation": implementation,
    }


def validate_manifest(manifest, context):
    common = {"stage", "study_config", "authorization"}
    required = {
        "authorize": {"stage", "study_config", "budget_gate"},
        "learn": common | {"stream"},
        "cohort": common | {"learning_sources"},
        "readout": common | {"cohort", "stream"},
        "readout-gate": common | {"cohort", "split", "sources"},
        "repair": common | {"cohort", "stream", "readout_gate"},
        "fit": common | {"cohort", "readout_gate", "sources"},
        "forecast": common | {"cohort", "readout_gate", "fitted"},
        "analyze": common | {"cohort", "readout_gate", "fitted", "forecasts", "sources"},
    }
    stage = manifest["stage"]
    if stage not in required:
        raise ValueError("BUDGET_PREDICTION_UNKNOWN_STAGE")
    if "stream" in manifest and manifest["stream"] not in context["data"]["streams"]:
        raise ValueError("BUDGET_PREDICTION_UNKNOWN_STREAM")
    if stage in {"readout", "repair"} and context["data"]["streams"][manifest["stream"]]["split"] == "test":
        required[stage] |= {"fitted"} if stage == "readout" else {"forecasts"}
    if stage == "readout-gate":
        if manifest["split"] not in {"train", "test"}:
            raise ValueError("BUDGET_PREDICTION_READOUT_SPLIT")
        if manifest["split"] == "test":
            required[stage].add("fitted")
    if set(manifest) != required[stage]:
        raise ValueError("BUDGET_PREDICTION_MANIFEST_FIELDS")


def completed(path):
    path = Path(path)
    execution_path = path.parent / "execution.json"
    if not execution_path.is_file():
        raise ValueError("BUDGET_PREDICTION_COMPLETED_EXECUTION_REQUIRED")
    execution = read_json(execution_path)
    if execution["status"] != "completed" or execution["exit_code"] != 0 or not execution.get("finished_at"):
        raise ValueError("BUDGET_PREDICTION_SOURCE_NOT_COMPLETED")
    receipt, sealed = verify_result(path)
    return receipt, sealed, execution


def authorize(path, context):
    protocol = context["protocol"]
    receipt, sealed, execution = completed(path)
    rule = protocol["minimum_budget_gate"]
    if (
        execution["task_id"] != rule["run_id"]
        or execution["source_sha256"] != rule["source_sha256"]
        or file_hash(ROOT / "minimum_budget.py") != rule["entrypoint_sha256"]
        or file_hash(ROOT / "minimum-budget-protocol.json") != rule["protocol_sha256"]
        or receipt["stage"] != "minimum-budget-qualify"
    ):
        raise ValueError("BUDGET_PREDICTION_MINIMUM_GATE_SOURCE_BINDING")
    base = {
        "minimum_gate_receipt_sha256": file_hash(Path(path) / "receipt.json"),
        "minimum_gate_execution_sha256": file_hash(Path(path).parent / "execution.json"),
        "minimum_gate_source_sha256": execution["source_sha256"],
        "lens_updates": 0,
        "repair_updates": 0,
        "predictor_fits": 0,
        "model_loads": 0,
    }
    if not receipt["qualified"]:
        return base | {
            "qualified": False,
            "status": "blocked_minimum_budget_gate",
            "decision": "no_fresh_learning_lens_predictor_or_TEST; another_TRAIN_protocol_required",
        }
    manifest, config, spec, data, pilot = load_pilot_manifest(ROOT / "configs/minimum-budget-gate.json")
    if sealed["dispatch_manifest"] != manifest:
        raise ValueError("BUDGET_PREDICTION_MINIMUM_GATE_MANIFEST")
    for source in manifest["sources"].values():
        _, _, upstream_execution = completed(source)
        if upstream_execution["source_sha256"] != rule["source_sha256"]:
            raise ValueError("BUDGET_PREDICTION_PILOT_SOURCE_CHANGED")
    prerequisites = checked_learning_sources(manifest["learning_sources"], config, spec, data, pilot)
    if not prerequisites["qualified"]:
        raise ValueError("BUDGET_PREDICTION_PILOT_LEARNING_PREREQUISITE")
    recomputed = recompute_pilot_gate(manifest["sources"], prerequisites, config, spec, data, pilot)
    if recomputed != {key: value for key, value in receipt.items() if key != "files"}:
        raise ValueError("BUDGET_PREDICTION_MINIMUM_GATE_RECOMPUTATION")
    return base | {
        "qualified": True,
        "status": "authorized",
        "pilot_class_counts": receipt["counts"],
        "decision": "fresh_learning_permitted; cohort_lens_and_TRAIN_label_gates_still_required",
    }


def source(path, stage, context, auth_hash=None, require_qualified=True):
    receipt, sealed, execution = completed(path)
    if (
        receipt["pipeline_stage"] != stage
        or sealed["prediction_identity"] != context["identity"]
        or auth_hash is not None
        and receipt["authorization_sha256"] != auth_hash
        or require_qualified
        and not receipt["qualified"]
    ):
        raise ValueError(f"BUDGET_PREDICTION_QUALIFIED_SOURCE_REQUIRED stage={stage}")
    return receipt, sealed, execution


def cohort_sources(paths, context, auth_hash):
    if set(paths) != set(context["data"]["streams"]):
        raise ValueError("BUDGET_PREDICTION_ALL_FRESH_SOURCES_REQUIRED")
    for name, path in paths.items():
        receipt, _, _ = source(path, "learn", context, auth_hash, require_qualified=False)
        if receipt["stream"] != name:
            raise ValueError("BUDGET_PREDICTION_LEARNING_STREAM_BINDING")
        stream = context["data"]["streams"][name]
        for phase in read_json(Path(path) / "observations.json").values():
            for kind in ("old", "new"):
                checked_records(phase[kind + "_gate"], rows_for(stream, stream[kind], "gate"), context["data"]["codes"])
    return qualify_learning(list(paths.values()), context["config"], context["spec"], context["data"])


def checked_cohort(path, context, auth_hash):
    receipt, sealed, _ = source(path, "cohort", context, auth_hash)
    recomputed = cohort_sources(sealed["dispatch_manifest"]["learning_sources"], context, auth_hash)
    if any(receipt[key] != value for key, value in recomputed.items()):
        raise ValueError("BUDGET_PREDICTION_COHORT_RECOMPUTATION")
    return receipt


def fresh_permutations(cohort, protocol):
    pairs = [
        (f"{name}/{intent}", name)
        for name in protocol["train_streams"]
        for intent in cohort["streams"][name]["eligible"]
    ]
    return permutation_plan(
        [key for key, _ in pairs],
        [name for _, name in pairs],
        protocol["label_permutations"]["seeds"],
        protocol["label_permutations"]["schemes"],
    )


def validate_units(units, stream, cohort, spec, codes, actions):
    if [unit["intent"] for unit in units] != cohort["streams"][stream["id"]]["eligible"]:
        raise ValueError("BUDGET_PREDICTION_ELIGIBLE_UNIT_SET")
    for unit in units:
        intent = unit["intent"]
        if (
            unit["id"] != f"{stream['id']}/{intent}"
            or unit["stream"] != stream["id"]
            or unit["split"] != stream["split"]
        ):
            raise ValueError("BUDGET_PREDICTION_UNIT_SOURCE_IDENTITY")
        for phase in ("before", "after"):
            checked_records(unit["baseline"][phase]["test"], stream["units"][intent]["test"], codes)
            checked_records(unit["baseline"][phase]["guard"], rows_for(stream, stream["new"], "test"), codes)
            checked_records(unit["probes"][phase], stream["units"][intent]["probe"], codes)
        if unit["features"] != feature_vector(unit["probes"]["before"], unit["probes"]["after"], spec["layers"]):
            raise ValueError("BUDGET_PREDICTION_READOUT_FEATURE_RECOMPUTATION")
        if actions:
            for panel in unit["actions"].values():
                checked_records(panel["test"], stream["units"][intent]["test"], codes)
                checked_records(panel["guard"], rows_for(stream, stream["new"], "test"), codes)
        elif unit["actions"]:
            raise ValueError("BUDGET_PREDICTION_READOUT_ACTION_LEAKAGE")


def lens_valid(path, receipt, data, spec):
    training = read_json(path / "lens-training.json")
    late = read_json(path / "lens-late-check.json")
    for key in ("fit", "check"):
        expected = {row["text_sha256"]: row for row in data["lens_" + key]}
        positions = training[key]["positions"]
        if (
            {value["text_sha256"] for value in positions} != set(expected)
            or training[key]["prompts"] != len(expected)
            or training[key]["tokens"] != len(positions)
            or len({(value["text_sha256"], value["position"]) for value in positions}) != len(positions)
            or any(not 0 <= value["position"] < len(expected[value["text_sha256"]]["input_ids"]) for value in positions)
            or any(row["source_split"] != "train" for row in expected.values())
        ):
            raise ValueError("BUDGET_PREDICTION_LENS_TRAIN_BACKGROUND_BINDING")
    if len(training["history"]) != spec["lens"]["updates"] or any(
        entry["step"] != index + 1
        or len(entry["positions"]) != spec["lens"]["token_batch_size"]
        or any(not 0 <= position < training["fit"]["tokens"] for position in entry["positions"])
        for index, entry in enumerate(training["history"])
    ):
        raise ValueError("BUDGET_PREDICTION_LENS_FIT_BUDGET")
    if any(late["metadata"][key] != training["check"][key] for key in ("positions", "prompts", "tokens")):
        raise ValueError("BUDGET_PREDICTION_LENS_CHECK_POSITIONS")
    for values, reported in (
        (training["final_heldout_kl"], receipt["acquired_kl_improvement"]),
        (late["heldout_kl"], receipt["forgotten_kl_improvement"]),
    ):
        if set(values) != {str(layer) for layer in spec["layers"]} or any(
            not math.isfinite(value) for layer in values.values() for value in layer.values()
        ):
            raise ValueError("BUDGET_PREDICTION_LENS_LAYER_OR_NONFINITE_KL")
        improvement = 1 - sum(value["tuned"] for value in values.values()) / max(
            sum(value["frozen"] for value in values.values()), 1e-12
        )
        if improvement != reported or improvement < spec["lens"]["min_relative_kl_improvement"]:
            raise ValueError("BUDGET_PREDICTION_LENS_KL_QUALIFICATION")
    if (
        receipt["lens_sha256"] != file_hash(path / "tuned-lens.safetensors")
        or receipt["lens_sha256"] != training["lens_sha256"]
        or not training["model_parameters_unchanged"]
        or not training["reload_exact"]
    ):
        raise ValueError("BUDGET_PREDICTION_LENS_FREEZE")


def readout_sources(paths, split, cohort, cohort_path, context, auth_hash):
    if set(paths) != set(context["protocol"][f"{split}_streams"]):
        raise ValueError("BUDGET_PREDICTION_ALL_READOUT_SOURCES_REQUIRED")
    collected, passed, receipts = [], True, {}
    for name in context["protocol"][f"{split}_streams"]:
        path = Path(paths[name])
        receipt, _, _ = source(path, "readout", context, auth_hash, require_qualified=False)
        if receipt["cohort_sha256"] != file_hash(cohort_path / "receipt.json") or receipt["stream"] != name:
            raise ValueError("BUDGET_PREDICTION_READOUT_COHORT_BINDING")
        receipts[name] = file_hash(path / "receipt.json")
        passed &= receipt["qualified"]
        if not receipt["qualified"]:
            continue
        lens_valid(path, receipt, context["data"], context["spec"])
        units = read_json(path / "readouts.json")
        validate_units(
            units, context["data"]["streams"][name], cohort, context["spec"], context["data"]["codes"], False
        )
        if receipt["readouts_sha256"] != file_hash(path / "readouts.json"):
            raise ValueError("BUDGET_PREDICTION_READOUT_HASH")
        collected.extend(units)
    return {
        "qualified": bool(passed),
        "split": split,
        "sources": paths,
        "source_receipts": receipts,
        "unit_ids": [unit["id"] for unit in collected],
        "repair_updates": 0,
        "lens_updates": 0,
    }, collected


def checked_readout_gate(path, split, cohort, cohort_path, context, auth_hash):
    receipt, sealed, _ = source(path, "readout-gate", context, auth_hash)
    recomputed, units = readout_sources(
        sealed["dispatch_manifest"]["sources"], split, cohort, cohort_path, context, auth_hash
    )
    if any(receipt[key] != value for key, value in recomputed.items()):
        raise ValueError("BUDGET_PREDICTION_READOUT_GATE_RECOMPUTATION")
    return receipt, units


def checked_fitted(path, context, auth_hash, cohort_hash):
    receipt, _, _ = source(path, "fit", context, auth_hash)
    fitted = read_json(Path(path) / "predictors.json")
    if (
        receipt["predictors_sha256"] != file_hash(Path(path) / "predictors.json")
        or receipt["cohort_sha256"] != cohort_hash
        or not fitted["gate"]["qualified"]
        or fitted["training_source_streams"] != context["protocol"]["train_streams"]
    ):
        raise ValueError("BUDGET_PREDICTION_FITTED_TRAIN_GATE")
    return fitted


def repaired_sources(paths, split, readouts, cohort, context, auth_hash, forecast_hash=None):
    if set(paths) != set(context["protocol"][f"{split}_streams"]):
        raise ValueError("BUDGET_PREDICTION_ALL_REPAIR_SOURCES_REQUIRED")
    baseline = {unit["id"]: unit for unit in readouts}
    units = []
    for name in context["protocol"][f"{split}_streams"]:
        path = Path(paths[name])
        receipt, _, _ = source(path, "repair", context, auth_hash)
        measured = read_json(path / "units.json")
        if receipt["stream"] != name or forecast_hash is not None and receipt["forecasts_sha256"] != forecast_hash:
            raise ValueError("BUDGET_PREDICTION_REPAIR_FORECAST_OR_SOURCE_BINDING")
        stream = context["data"]["streams"][name]
        validate_units(measured, stream, cohort, context["spec"], context["data"]["codes"], True)
        if (
            receipt["repair_updates"] != len(measured) * sum(context["protocol"]["budgets"])
            or receipt["features_sha256"] != file_hash(path / "features-frozen.json")
            or read_json(path / "features-frozen.json") != [baseline[unit["id"]] for unit in measured]
        ):
            raise ValueError("BUDGET_PREDICTION_PREUPDATE_FEATURES_OR_BUDGET")
        for unit in measured:
            if {**unit, "actions": {}} != baseline[unit["id"]]:
                raise ValueError("BUDGET_PREDICTION_POSTUPDATE_FEATURE_MUTATION")
            verify_grid(path, unit, stream, context["spec"], context["protocol"])
            expected = file_hash(Path(cohort["streams"][name]["source"]) / "forgotten/adapter_model.safetensors")
            if any(action["start_adapter_sha256"] != expected for action in unit["actions"].values()):
                raise ValueError("BUDGET_PREDICTION_FORGOTTEN_CHECKPOINT_BINDING")
        outcomes = [minimum_outcome(unit, context["spec"], context["protocol"]["budgets"]) for unit in measured]
        if outcomes != read_json(path / "outcomes.json"):
            raise ValueError("BUDGET_PREDICTION_REPAIR_OUTCOME_RECOMPUTATION")
        units.extend(measured)
    return units


def run_stage(manifest, context, output):
    stage = manifest["stage"]
    config, spec, data, protocol = [context[key] for key in ("config", "spec", "data", "protocol")]
    if stage == "authorize":
        return authorize(manifest["budget_gate"], context)
    source(manifest["authorization"], "authorize", context)
    auth_hash = file_hash(Path(manifest["authorization"]) / "receipt.json")
    provenance = {"authorization_sha256": auth_hash}
    if stage == "learn":
        stream = data["streams"][manifest["stream"]]
        verify_model(config, data)
        observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
        result = run_learning(observer, config, spec, stream, output)
        result.pop("next_config_if_failed")
        result["qualified"] = result["status"] == "qualified"
        result["decision"] = (
            "await_all_fresh_learning_sources" if result["qualified"] else "stop; new_TRAIN_protocol_required"
        )
        return result | provenance
    if stage == "cohort":
        result = cohort_sources(manifest["learning_sources"], context, auth_hash)
        write_json(output / "cohort.json", result)
        if result["qualified"]:
            write_json(output / "permutation-plan.json", fresh_permutations(result, protocol))
        return result | provenance
    cohort_path = Path(manifest["cohort"])
    cohort = checked_cohort(cohort_path, context, auth_hash)
    provenance["cohort_sha256"] = file_hash(cohort_path / "receipt.json")
    if stage in {"readout", "repair"}:
        stream = data["streams"][manifest["stream"]]
        learning = Path(cohort["streams"][stream["id"]]["source"])
        if stage == "readout":
            if stream["split"] == "test":
                checked_fitted(manifest["fitted"], context, auth_hash, provenance["cohort_sha256"])
                provenance["fitted_sha256"] = file_hash(Path(manifest["fitted"]) / "receipt.json")
        else:
            _, all_units = checked_readout_gate(
                Path(manifest["readout_gate"]), stream["split"], cohort, cohort_path, context, auth_hash
            )
            units = [unit for unit in all_units if unit["stream"] == stream["id"]]
            provenance["readout_gate_sha256"] = file_hash(Path(manifest["readout_gate"]) / "receipt.json")
            if stream["split"] == "test":
                forecast_receipt, _, _ = source(manifest["forecasts"], "forecast", context, auth_hash)
                provenance["forecasts_sha256"] = file_hash(Path(manifest["forecasts"]) / "receipt.json")
                if forecast_receipt["readout_gate_sha256"] != provenance["readout_gate_sha256"]:
                    raise ValueError("BUDGET_PREDICTION_FORECAST_READOUT_BINDING")
        verify_model(config, data)
        observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
        result = (
            readout(observer, spec, data, stream, learning, cohort["streams"][stream["id"]]["eligible"], output)
            if stage == "readout"
            else repair(observer, spec, protocol, stream, learning, units, output)
        )
        return result | provenance | {"stream": stream["id"], "split": stream["split"]}
    if stage == "readout-gate":
        if manifest["split"] == "test":
            checked_fitted(manifest["fitted"], context, auth_hash, provenance["cohort_sha256"])
            provenance["fitted_sha256"] = file_hash(Path(manifest["fitted"]) / "receipt.json")
        result, _ = readout_sources(manifest["sources"], manifest["split"], cohort, cohort_path, context, auth_hash)
        write_json(output / "readout-gate.json", result)
        return result | provenance
    split = "train" if stage == "fit" else "test"
    _, readouts = checked_readout_gate(Path(manifest["readout_gate"]), split, cohort, cohort_path, context, auth_hash)
    provenance["readout_gate_sha256"] = file_hash(Path(manifest["readout_gate"]) / "receipt.json")
    if stage == "fit":
        units = repaired_sources(manifest["sources"], "train", readouts, cohort, context, auth_hash)
        outcomes = [minimum_outcome(unit, spec, protocol["budgets"]) for unit in units]
        gate = prediction_gate(outcomes, protocol)
        write_json(output / "fresh-train-budget-gate.json", gate)
        result = gate | provenance | {"repair_updates": 0, "lens_updates": 0, "predictor_fits": 0}
        if gate["qualified"]:
            plan = read_json(cohort_path / "permutation-plan.json")
            if plan != fresh_permutations(cohort, protocol):
                raise ValueError("BUDGET_PREDICTION_PERMUTATION_PLAN_MUTATED")
            fitted = fit_models(units, spec, protocol, plan)
            write_json(output / "predictors.json", fitted)
            result.update(
                predictor_fits=sum(len(model.get("targets", {})) for model in fitted["models"].values()),
                predictors_sha256=file_hash(output / "predictors.json"),
            )
        return result
    fitted = checked_fitted(manifest["fitted"], context, auth_hash, provenance["cohort_sha256"])
    provenance["fitted_sha256"] = file_hash(Path(manifest["fitted"]) / "receipt.json")
    predicted = forecasts(fitted, readouts)
    if stage == "forecast":
        write_json(output / "forecasts.json", predicted)
        return provenance | {
            "qualified": True,
            "forecasts_file_sha256": file_hash(output / "forecasts.json"),
            "repair_updates": 0,
            "lens_updates": 0,
            "predictor_fits": 0,
        }
    forecast_receipt, _, _ = source(manifest["forecasts"], "forecast", context, auth_hash)
    if forecast_receipt["fitted_sha256"] != provenance["fitted_sha256"] or predicted != read_json(
        Path(manifest["forecasts"]) / "forecasts.json"
    ):
        raise ValueError("BUDGET_PREDICTION_SAVED_FORECAST_BINDING")
    forecast_hash = file_hash(Path(manifest["forecasts"]) / "receipt.json")
    units = repaired_sources(manifest["sources"], "test", readouts, cohort, context, auth_hash, forecast_hash)
    analysis = evaluate(fitted, predicted, units, spec, protocol)
    write_json(output / "analysis.json", analysis)
    return provenance | {
        "qualified": True,
        "forecasts_sha256": forecast_hash,
        "repair_updates": 0,
        "lens_updates": 0,
        "predictor_fits": 0,
        "independent_test_source_clusters": analysis["independent_test_source_clusters"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [budget-prediction] %(message)s")
    manifest = read_json(args.config)
    context = load_study(manifest)
    validate_manifest(manifest, context)
    output = args.output_dir / "study"
    output.mkdir(parents=True, exist_ok=False)
    try:
        sealed = make_seal(
            context["config"], context["spec"], context["data"], "budget-prediction-" + manifest["stage"]
        )["payload"]
        sealed.update(
            prediction_identity=context["identity"],
            prediction_protocol=context["protocol"],
            prediction_implementation=context["implementation"],
            dispatch_manifest=manifest,
        )
        write_json(output / "seal.json", {"payload": sealed, "sha256": digest(sealed)})
        result = run_stage(manifest, context, output)
        result.update(
            pipeline_stage=manifest["stage"],
            status=result.get("status", "qualified" if result["qualified"] else "blocked"),
        )
        result.setdefault("stage", "budget-prediction-" + manifest["stage"])
        finish(output, result)
        logging.info(
            "BUDGET_PREDICTION_STAGE stage=%s status=%s qualified=%s",
            manifest["stage"],
            result["status"],
            result["qualified"],
        )
    except Exception as error:
        logging.exception("BUDGET_PREDICTION_STAGE_FAILED")
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
