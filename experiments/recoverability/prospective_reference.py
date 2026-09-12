import argparse
import logging
from pathlib import Path

import numpy as np

from prospective import (
    assert_bound_records,
    checked_pilot_qualification,
    checked_qualification,
    checked_repair_qualification,
    finish,
    learning_identity,
    learning_implementation,
    make_seal,
    measurement_identity,
    perform_actions,
    verify_model,
    verify_result,
)
from prospective_analysis import accuracy, learned, per_intent
from prospective_data import balanced_batches, load_design, rows_for, seed_for, validate_data
from prospective_model import ProspectiveModel, parameter_hashes
from protocol import ROOT, digest, file_hash, read_json, write_json

REFERENCE_PROTOCOL = ROOT / "reference-protocol.json"
REFERENCE_ACTIONS = ("none", "replay_target", "replay_balanced", "sham")


def source_learning(path, stream, config, spec, data):
    result, sealed = verify_result(path)
    if (
        result["stage"] != "learn"
        or result["status"] != "qualified"
        or result["stream"] != stream["id"]
        or sealed["learning_identity"] != learning_identity(config, spec, data)
        or sealed["learning_implementation"] != learning_implementation()
    ):
        raise ValueError("PROSPECTIVE_REFERENCE_LEARNING_SOURCE")
    return result


def learn_reference(observer, config, spec, stream, source, output):
    observer.reset(source / "initial")
    frozen = parameter_hashes(observer.model, frozen_only=True)
    rows = rows_for(stream, stream["new"], "learn")
    gate = rows_for(stream, stream["new"], "gate")
    old_gate = rows_for(stream, stream["old"], "gate")
    initial = {"old_gate": observer.observe(old_gate), "new_gate": observer.observe(gate)}
    optimizer = observer.optimizer()
    schedule = balanced_batches(
        rows, config["forgetting_updates"], spec["optimizer"]["batch_size"], seed_for(stream["seed"], "forgetting")
    )
    if schedule != [step["rows"] for step in read_json(source / "forgetting-updates.json")]:
        raise ValueError("PROSPECTIVE_REFERENCE_NEW_TASK_SCHEDULE_MISMATCH")
    if any(row["intent"] in stream["old"] for row in rows):
        raise ValueError("PROSPECTIVE_REFERENCE_OLD_SKILL_EXPOSURE")
    trace = observer.update(rows, schedule, optimizer)
    observed = {"old_gate": observer.observe(old_gate), "new_gate": observer.observe(gate)}
    observer.checkpoint(output / "retained", optimizer)
    observer.reset(output / "retained")
    if observed["new_gate"] != observer.observe(gate):
        raise ValueError("PROSPECTIVE_REFERENCE_RELOAD")
    acquired = learned(per_intent(initial["new_gate"]), per_intent(observed["new_gate"]), spec)
    passed = len(acquired) >= spec["gates"]["minimum_new_acquired_per_stream"]
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("PROSPECTIVE_REFERENCE_CHANGED_BACKBONE")
    write_json(output / "observations.json", {"initial": initial, "retained": observed})
    write_json(output / "updates.json", {"rows": rows, "updates": trace})
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "stage": "reference-learn",
        "stream": stream["id"],
        "split": stream["split"],
        "qualified": passed,
        "acquired_new": acquired,
        "old_training_exposures": 0,
        "test_rows_evaluated": 0,
        "updates": len(trace),
        "schedule_sha256": digest({"rows": rows, "batches": schedule}),
        "initial_adapter_sha256": file_hash(source / "initial" / "adapter_model.safetensors"),
        "learning_source": str(source.resolve()),
        "learning_receipt_sha256": file_hash(source / "receipt.json"),
        "decision": "reference_actions_allowed_after_main_gates" if passed else "stop_reference_actions",
    }


def verify_reference_learning(path, config, spec, data, stream):
    result, sealed = verify_result(path)
    if (
        result["stage"] != "reference-learn"
        or not result["qualified"]
        or result["stream"] != stream["id"]
        or sealed["learning_identity"] != learning_identity(config, spec, data)
        or sealed["learning_implementation"] != learning_implementation()
        or sealed["implementation"]["prospective_reference.py"] != file_hash(ROOT / "prospective_reference.py")
        or result["old_training_exposures"] != 0
    ):
        raise ValueError("PROSPECTIVE_REFERENCE_ACQUISITION_REQUIRED")
    source_learning(result["learning_source"], stream, config, spec, data)
    if file_hash(Path(result["learning_source"]) / "receipt.json") != result["learning_receipt_sha256"]:
        raise ValueError("PROSPECTIVE_REFERENCE_SOURCE_CHANGED")
    observations = read_json(path / "observations.json")
    observed, initial = observations["retained"], observations["initial"]
    assert_bound_records(observed["new_gate"], rows_for(stream, stream["new"], "gate"))
    assert_bound_records(initial["new_gate"], rows_for(stream, stream["new"], "gate"))
    if learned(per_intent(initial["new_gate"]), per_intent(observed["new_gate"]), spec) != result["acquired_new"]:
        raise ValueError("PROSPECTIVE_REFERENCE_GATE_RECOMPUTATION")
    return result


def measure_reference(observer, config, spec, stream, reference, intents, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    observer.reset(reference / "retained")
    guards = observer.observe(rows_for(stream, stream["new"], "test"))
    baselines = {
        intent: {"after": {"test": observer.observe(stream["units"][intent]["test"]), "guard": guards}}
        for intent in intents
    }
    write_json(output / "baselines-frozen.json", baselines)
    units = []
    for intent in intents:
        actions = perform_actions(
            observer,
            config,
            spec,
            stream,
            intent,
            None,
            reference / "retained",
            baselines[intent],
            output,
            actions=REFERENCE_ACTIONS,
        )
        unit = {
            "id": f"{stream['id']}/{intent}",
            "stream": stream["id"],
            "split": stream["split"],
            "intent": intent,
            "baseline": baselines[intent],
            "actions": actions,
        }
        units.append(unit)
        write_json(output / "units" / f"{intent}.json", unit)
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("PROSPECTIVE_REFERENCE_REPAIR_CHANGED_BACKBONE")
    write_json(output / "units.json", units)
    return {
        "stage": "reference-measure",
        "stream": stream["id"],
        "split": stream["split"],
        "unit_ids": [unit["id"] for unit in units],
        "reference_source": str(reference.resolve()),
        "reference_receipt_sha256": file_hash(reference / "receipt.json"),
        "scope": "First acquisition of old intent-to-code bindings, after learning only the newer task; never called recovery.",
    }


def compare_reference(main_path, reference_path, config, spec, data):
    main, main_seal = verify_result(main_path)
    ref, ref_seal = verify_result(reference_path)
    expected = measurement_identity(config, spec, data)
    if (
        main["stage"] != "measure"
        or ref["stage"] != "reference-measure"
        or main["stream"] != ref["stream"]
        or main["unit_ids"] != ref["unit_ids"]
        or main_seal["measurement_identity"] != expected
        or ref_seal["measurement_identity"] != expected
        or main["qualification_sha256"] != ref["qualification_sha256"]
    ):
        raise ValueError("PROSPECTIVE_REFERENCE_COMPARISON_BINDING")
    a, b = read_json(main_path / "units.json"), read_json(reference_path / "units.json")
    paired = []
    for old, fresh in zip(a, b, strict=True):
        comparisons = {}
        for action in REFERENCE_ACTIONS:
            x, y = old["actions"][action], fresh["actions"][action]
            for key in ("updates", "total_exposures", "old_exposures", "new_exposures", "schedule_sha256"):
                if x[key] != y[key]:
                    raise ValueError(f"PROSPECTIVE_REFERENCE_UNMATCHED_REPAIR {key}")
            old_score, fresh_score = accuracy(x["test"]), accuracy(y["test"])
            old_gain = old_score - accuracy(old["baseline"]["after"]["test"])
            fresh_gain = fresh_score - accuracy(fresh["baseline"]["after"]["test"])
            comparisons[action] = {
                "recovery_accuracy": old_score,
                "fresh_acquisition_accuracy": fresh_score,
                "recovery_minus_fresh_accuracy": old_score - fresh_score,
                "recovery_minus_fresh_gain": old_gain - fresh_gain,
                "recovery_guard": accuracy(x["guard"]),
                "fresh_guard": accuracy(y["guard"]),
            }
        paired.append({"id": old["id"], "stream": old["stream"], "actions": comparisons})
    return {
        "stage": "reference-compare",
        "stream": main["stream"],
        "split": main["split"],
        "units": paired,
        "run_level": {
            action: {
                key: float(np.mean([unit["actions"][action][key] for unit in paired]))
                for key in paired[0]["actions"][action]
            }
            for action in REFERENCE_ACTIONS
        },
        "scope": "Paired common-seed/code/support/budget comparison of re-accessing learned bindings vs first learning them. The retain-only model omits old-task updates; equal total lifetime updates and equivalent optimizer history are not claimed. Neither result implies erasure of semantic intent concepts.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("learn", "measure", "compare"), required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-source", type=Path)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--main-source", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--repair-qualification", type=Path)
    parser.add_argument("--pilot-qualification", type=Path)
    args = parser.parse_args(argv)
    config, spec = load_design(args.config)
    data = read_json(ROOT / config["dataset"])
    validate_data(data, spec)
    stream = data["streams"][args.stream]
    if file_hash(REFERENCE_PROTOCOL) != REFERENCE_PROTOCOL.with_suffix(".sha256").read_text().strip():
        raise ValueError("PROSPECTIVE_REFERENCE_PROTOCOL_CHANGED")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [prospective-reference] %(message)s")
    try:
        write_json(args.output_dir / "seal.json", make_seal(config, spec, data, "reference-" + args.stage))
        write_json(args.output_dir / "reference-protocol.json", read_json(REFERENCE_PROTOCOL))
        if args.stage == "learn":
            if args.learning_source is None:
                raise ValueError("PROSPECTIVE_REFERENCE_SOURCE_REQUIRED")
            source_learning(args.learning_source, stream, config, spec, data)
            verify_model(config, data)
            observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
            result = learn_reference(observer, config, spec, stream, args.learning_source, args.output_dir)
        elif args.stage == "measure":
            if args.qualification is None or args.reference_source is None or args.pilot_qualification is None:
                raise ValueError("PROSPECTIVE_REFERENCE_ELIGIBILITY_REQUIRED")
            qualification = checked_qualification(args.qualification, config, spec, data)
            checked_pilot_qualification(args.pilot_qualification, config, spec, data, args.qualification)
            source = verify_reference_learning(args.reference_source, config, spec, data, stream)
            if source["learning_receipt_sha256"] != qualification["streams"][args.stream]["receipt_sha256"]:
                raise ValueError("PROSPECTIVE_REFERENCE_MATCHED_LEARNER_REQUIRED")
            if stream["split"] == "test":
                if args.repair_qualification is None:
                    raise ValueError("PROSPECTIVE_TRAIN_REPAIR_SIGNAL_REQUIRED")
                checked_repair_qualification(args.repair_qualification, config, spec, data, args.qualification)
            verify_model(config, data)
            observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
            result = measure_reference(
                observer,
                config,
                spec,
                stream,
                args.reference_source,
                qualification["streams"][args.stream]["eligible"],
                args.output_dir,
            )
            result["qualification_sha256"] = file_hash(args.qualification / "receipt.json")
            result["pilot_qualification_sha256"] = file_hash(args.pilot_qualification / "receipt.json")
        else:
            if args.main_source is None or args.reference_source is None:
                raise ValueError("PROSPECTIVE_REFERENCE_COMPARISON_SOURCES_REQUIRED")
            result = compare_reference(args.main_source, args.reference_source, config, spec, data)
            write_json(args.output_dir / "comparison.json", result)
        finish(args.output_dir, result)
    except Exception as error:
        logging.exception("PROSPECTIVE_REFERENCE_FAILED")
        write_json(args.output_dir / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
