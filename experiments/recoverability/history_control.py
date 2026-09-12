import argparse
import logging
from pathlib import Path

from history_control_analysis import paired_analysis
from history_control_model import learn_control, now, repair_control
from history_control_sources import (
    checked_common,
    checked_control_repairs,
    checked_historical_repairs,
    historical_cohort,
    load_context,
    recompute_common,
    time_before,
)
from prospective import finish, make_seal, verify_model
from prospective_model import ProspectiveModel
from protocol import ROOT, digest, file_hash, read_json, write_json


def validate_manifest(manifest, context):
    required = {
        "learn": {"stage", "run_id", "stream", "condition"},
        "gate": {"stage", "run_id", "sources"},
        "repair": {"stage", "run_id", "stream", "condition", "gate", "historical_repair"},
        "analyze": {"stage", "run_id", "gate", "control_sources", "historical_sources", "forecasts"},
    }
    stage = manifest["stage"]
    if stage not in required:
        raise ValueError("HISTORY_CONTROL_UNKNOWN_STAGE")
    if stage in {"learn", "repair"}:
        if (
            manifest["stream"] not in context["data"]["streams"]
            or manifest["condition"] not in context["protocol"]["conditions"]
        ):
            raise ValueError("HISTORY_CONTROL_UNKNOWN_STREAM_OR_CONDITION")
        if stage == "repair" and context["data"]["streams"][manifest["stream"]]["split"] == "test":
            required[stage] = required[stage] | {"forecasts"}
    if set(manifest) != required[stage]:
        raise ValueError("HISTORY_CONTROL_MANIFEST_FIELDS")
    if stage == "analyze" and (
        set(manifest["control_sources"]) != set(context["data"]["streams"])
        or set(manifest["historical_sources"]) != set(context["data"]["streams"])
        or any(
            set(sources) != set(context["protocol"]["conditions"]) for sources in manifest["control_sources"].values()
        )
    ):
        raise ValueError("HISTORY_CONTROL_ANALYSIS_MANIFEST_SOURCES")


def run_stage(manifest, context, output):
    historical, schedules = historical_cohort(context)
    config, spec, data, protocol = context["config"], context["spec"], context["data"], context["protocol"]
    stage = manifest["stage"]
    design = {
        "design_sha256": file_hash(ROOT / "history-control-design.json"),
        "protocol_sha256": file_hash(ROOT / "history-control-protocol.json"),
        "parent_decision_sha256": context["design"]["parent_decision"]["sha256"],
        "parent_decision_at": context["design"]["parent_decision_at"],
        "lane_protocol_at": context["design"]["lane_protocol_at"],
        "historical_cohort_sha256": protocol["original_cohort"]["receipt_sha256"],
    }
    if stage == "learn":
        stream = data["streams"][manifest["stream"]]
        source = Path(protocol["learning_sources"][stream["id"]]["source"])
        write_json(
            output / "learning-prerequisites.json",
            design
            | {
                "at": now(),
                "historical_learning_sha256": file_hash(source / "receipt.json"),
                "qualified": historical["qualified"],
                "original_source_sha256": protocol["original_source_sha256"],
            },
        )
        verify_model(config, data)
        observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
        return (
            learn_control(
                observer,
                config,
                spec,
                data,
                stream,
                manifest["condition"],
                protocol,
                source,
                schedules[stream["id"]],
                output,
            )
            | design
        )
    if stage == "gate":
        result = recompute_common(manifest["sources"], context, historical, schedules)
        write_json(output / "common-gate.json", result)
        return result | design
    gate = checked_common(manifest["gate"], context, historical, schedules)
    gate_hash = file_hash(Path(manifest["gate"]) / "receipt.json")
    design["common_gate_sha256"] = gate_hash
    write_json(output / "eligibility-before-updates.json", {"at": now(), "gate_sha256": gate_hash, "gate": gate})
    if not gate["qualified"]:
        return design | {
            "qualified": False,
            "status": "blocked_common_eligibility",
            "repair_updates": 0,
            "lens_updates": 0,
            "predictor_fits": 0,
            "model_loads": 0,
            "decision": "stop_all_control_repairs; new_TRAIN_protocol_required",
        }
    if stage == "repair":
        stream, condition = data["streams"][manifest["stream"]], manifest["condition"]
        name = stream["id"]
        common = gate["streams"][name]["common"]
        _, provenance = checked_historical_repairs(
            Path(manifest["historical_repair"]), stream, context, common, manifest.get("forecasts")
        )
        checked_at = now()
        time_before(provenance["historical_repair_finished_at"], checked_at, "history_before_control_repairs")
        if stream["split"] == "test":
            time_before(provenance["forecast_finished_at"], checked_at, "forecast_before_control_TEST_repairs")
        learning = Path(gate["control_sources"][name][condition])
        write_json(
            output / "repair-prerequisites.json",
            design
            | provenance
            | {
                "at": checked_at,
                "common": common,
                "control_learning_sha256": file_hash(learning / "receipt.json"),
                "prediction_inputs_changed": False,
            },
        )
        verify_model(config, data)
        observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
        result = repair_control(observer, spec, protocol, stream, condition, learning, common, output)
        return result | design | provenance | {"control_learning_sha256": file_hash(learning / "receipt.json")}
    units = {}
    for name, stream in data["streams"].items():
        common = gate["streams"][name]["common"]
        original, provenance = checked_historical_repairs(
            Path(manifest["historical_sources"][name]), stream, context, common, manifest["forecasts"]
        )
        units[name] = {"original_history": original}
        for condition in protocol["conditions"]:
            units[name][condition] = checked_control_repairs(
                manifest["control_sources"][name][condition],
                name,
                condition,
                manifest["gate"],
                context,
                common,
                provenance,
            )
    analysis = paired_analysis(units, data["streams"], gate, protocol)
    analysis["design_inputs"] = design
    analysis["temporal_interpretation"] = context["design"]["temporal_interpretation"]
    write_json(output / "analysis.json", analysis)
    return design | {
        "qualified": True,
        "status": "completed",
        "repair_updates": 0,
        "predictor_fits": 0,
        "model_loads": 0,
        "lens_updates": 0,
        "source_clusters": gate["source_clusters"],
        "analysis_sha256": file_hash(output / "analysis.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [history-control] %(message)s")
    context = load_context()
    manifest = read_json(args.config)
    validate_manifest(manifest, context)
    if args.output_dir.name != manifest["run_id"]:
        raise ValueError("HISTORY_CONTROL_OUTPUT_RUN_ID_MISMATCH")
    output = args.output_dir / "study"
    output.mkdir(parents=True, exist_ok=False)
    try:
        sealed = make_seal(context["config"], context["spec"], context["data"], "history-control-" + manifest["stage"])[
            "payload"
        ]
        sealed.update(
            history_identity=context["identity"],
            history_protocol=context["protocol"],
            history_design=context["design"],
            history_implementation=context["implementation"],
            dispatch_manifest=manifest,
            stage_started_at=now(),
        )
        write_json(output / "seal.json", {"payload": sealed, "sha256": digest(sealed)})
        result = run_stage(manifest, context, output)
        result.update(
            stage="history-control-" + manifest["stage"], history_identity=context["identity"], completed_at=now()
        )
        finish(output, result)
        logging.info(
            "HISTORY_CONTROL_STAGE stage=%s qualified=%s status=%s",
            manifest["stage"],
            result["qualified"],
            result["status"],
        )
    except Exception as error:
        logging.exception("HISTORY_CONTROL_STAGE_FAILED")
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
