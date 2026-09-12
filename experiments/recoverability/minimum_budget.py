import argparse
import copy
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np

from prospective import (
    LEARNING_SOURCE_FILES,
    assert_bound_records,
    finish,
    learning_identity,
    learning_implementation,
    make_seal,
    measurement_identity,
    verify_model,
    verify_result,
)
from prospective_analysis import accuracy, eligible, feature_vector, learned, per_intent, stream_summary, unit_outcomes
from prospective_data import load_design, repair_batches, rows_for, seed_for, validate_data
from prospective_model import ProspectiveModel, parameter_hashes
from protocol import ROOT, digest, file_hash, read_json, write_json

PROTOCOL = ROOT / "minimum-budget-protocol.json"
SOURCE_FILES = (*LEARNING_SOURCE_FILES, "tuned_lens.py", "minimum_budget.py")


def implementation():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def pilot_identity(config, spec, data, pilot):
    return digest(
        {
            "config": config,
            "protocol": spec,
            "dataset": digest(data),
            "pilot": pilot,
            "implementation": implementation(),
        }
    )


def load_manifest(path):
    manifest = read_json(path)
    common = {"study_config", "stage", "learning_sources"}
    required = common | ({"stream", "measurement_source"} if manifest.get("stage") == "measure" else {"sources"})
    if manifest.get("stage") not in {"measure", "qualify"} or set(manifest) != required:
        raise ValueError("MINIMUM_BUDGET_MANIFEST_FIELDS_OR_STAGE")
    if file_hash(PROTOCOL) != PROTOCOL.with_suffix(".sha256").read_text().strip():
        raise ValueError("MINIMUM_BUDGET_PROTOCOL_HASH")
    pilot = read_json(PROTOCOL)
    if set(manifest["learning_sources"]) != set(pilot["train_streams"]):
        raise ValueError("MINIMUM_BUDGET_ALL_TRAIN_SOURCES_REQUIRED")
    if manifest["stage"] == "measure" and manifest["stream"] not in pilot["train_streams"]:
        raise ValueError("MINIMUM_BUDGET_TRAIN_ONLY")
    if manifest["stage"] == "qualify" and set(manifest["sources"]) != set(pilot["train_streams"]):
        raise ValueError("MINIMUM_BUDGET_ALL_TRAIN_MEASUREMENTS_REQUIRED")
    config_path = ROOT / manifest["study_config"]
    if file_hash(config_path) != pilot["source_config_sha256"]:
        raise ValueError("MINIMUM_BUDGET_SOURCE_CONFIG_HASH")
    config, spec = load_design(config_path)
    data = read_json(ROOT / config["dataset"])
    validate_data(data, spec)
    if (
        file_hash(ROOT / config["protocol"]) != pilot["source_protocol_sha256"]
        or digest(data) != pilot["source_dataset_sha256"]
        or learning_implementation() != pilot["learning_implementation"]
        or file_hash(ROOT / "tuned_lens.py") != pilot["tuned_lens_implementation_sha256"]
    ):
        raise ValueError("MINIMUM_BUDGET_SOURCE_IMPLEMENTATION_OR_DATA_CHANGED")
    if (
        pilot["budgets"] != [2, 4, 8, 16]
        or config["repair_updates"] != 16
        or pilot["recovered"] != spec["recovered"]
        or pilot["learning_gates"] != spec["gates"]
    ):
        raise ValueError("MINIMUM_BUDGET_SCORING_OR_BUDGET_CHANGED")
    return manifest, config, spec, data, pilot


def checked_records(records, rows, codes):
    assert_bound_records(records, rows)
    for record in records:
        logits = record["output"]["code_logits"]
        if (
            len(logits) != 16
            or any(not math.isfinite(x) for x in logits)
            or record["target"] not in codes
            or codes[max(range(16), key=logits.__getitem__)] != record["output"]["prediction"]
        ):
            raise ValueError("MINIMUM_BUDGET_NATIVE_CODE_CONTRACT")


def learning_gate(observations, stream, spec, codes):
    for phase in observations.values():
        for key, intents in (("old_gate", stream["old"]), ("new_gate", stream["new"])):
            checked_records(phase[key], rows_for(stream, intents, "gate"), codes)
    initial = per_intent(observations["initial"]["old_gate"])
    before = per_intent(observations["acquired"]["old_gate"])
    acquired = learned(initial, before, spec)
    forgotten, new_acquired = [], []
    if "forgotten" in observations:
        forgotten = eligible(initial, before, per_intent(observations["forgotten"]["old_gate"]), spec)
        new_acquired = learned(
            per_intent(observations["acquired"]["new_gate"]), per_intent(observations["forgotten"]["new_gate"]), spec
        )
    passed = (
        len(acquired) >= spec["gates"]["minimum_acquired_per_stream"]
        and len(new_acquired) >= spec["gates"]["minimum_new_acquired_per_stream"]
        and len(forgotten) >= spec["gates"]["minimum_forgotten_per_stream"]
    )
    return {"acquired": acquired, "new_acquired": new_acquired, "eligible": forgotten, "passed": passed}


def checked_learning_sources(paths, config, spec, data, pilot):
    if set(paths) != set(pilot["train_streams"]):
        raise ValueError("MINIMUM_BUDGET_TRAIN_COHORT_SET")
    streams = {}
    for name in pilot["train_streams"]:
        path = Path(paths[name])
        stream = data["streams"][name]
        if stream["split"] != "train":
            raise ValueError("MINIMUM_BUDGET_TRAIN_SOURCE_SPLIT")
        if file_hash(path / "receipt.json") != pilot["source_receipts"][name]["learn"]:
            raise ValueError("MINIMUM_BUDGET_LEARNING_RECEIPT_HASH")
        result, sealed = verify_result(path)
        if (
            result["stage"] != "learn"
            or result["stream"] != name
            or result["split"] != "train"
            or sealed["learning_identity"] != learning_identity(config, spec, data)
            or sealed["learning_implementation"] != learning_implementation()
            or any(sealed["implementation"][key] != value for key, value in learning_implementation().items())
        ):
            raise ValueError("MINIMUM_BUDGET_LEARNING_BINDING")
        gate = learning_gate(read_json(path / "observations.json"), stream, spec, data["codes"])
        if (
            any(gate[key] != result[key] for key in ("acquired", "new_acquired", "eligible"))
            or result["repair_updates"] != 0
            or result["test_rows_evaluated"] != 0
            or gate["passed"] != (result["status"] == "qualified")
        ):
            raise ValueError("MINIMUM_BUDGET_LEARNING_GATE_RECOMPUTATION")
        streams[name] = gate | {"source": str(path.resolve()), "receipt_sha256": file_hash(path / "receipt.json")}
    count = sum(len(value["eligible"]) for value in streams.values())
    passed = all(value["passed"] for value in streams.values()) and count >= pilot["gate"]["minimum_train_skills"]
    return {
        "qualified": passed,
        "streams": streams,
        "eligible_train_skills": count,
        "source_clusters": len(streams),
        "new_repair_updates": 0,
        "heldout_outcomes_read": 0,
    }


def schedule_for(stream, intent, spec, maximum):
    repair_spec = spec | {"repair": spec["repair"] | {"updates": maximum}}
    return repair_batches(
        stream["units"][intent]["repair"],
        rows_for(stream, stream["new"], "repair"),
        repair_spec,
        seed_for(stream["seed"], intent, "repair"),
    )


def checked_measurement(path, stream, learning, config, spec, data, pilot):
    path = Path(path)
    if file_hash(path / "receipt.json") != pilot["source_receipts"][stream["id"]]["measure"]:
        raise ValueError("MINIMUM_BUDGET_MEASUREMENT_RECEIPT_HASH")
    result, sealed = verify_result(path)
    if (
        result["stage"] != "measure"
        or result["stream"] != stream["id"]
        or result["split"] != "train"
        or sealed["measurement_identity"] != measurement_identity(config, spec, data)
        or result["features_sha256"] != file_hash(path / "features-frozen.json")
        or any(
            sealed["implementation"][name] != file_hash(ROOT / name)
            for name in (*LEARNING_SOURCE_FILES, "tuned_lens.py")
        )
    ):
        raise ValueError("MINIMUM_BUDGET_MEASUREMENT_BINDING")
    units, frozen = read_json(path / "units.json"), read_json(path / "features-frozen.json")
    if (
        [unit["intent"] for unit in units] != learning["eligible"]
        or [unit["id"] for unit in units] != result["unit_ids"]
        or len(units) != len(frozen)
    ):
        raise ValueError("MINIMUM_BUDGET_ALL_ELIGIBLE_SKILLS_REQUIRED")
    learning_path = Path(learning["source"])
    for unit, prior in zip(units, frozen, strict=True):
        intent = unit["intent"]
        if (
            unit["id"] != f"{stream['id']}/{intent}"
            or unit["stream"] != stream["id"]
            or unit["split"] != "train"
            or any(
                unit[key] != prior[key] for key in ("id", "intent", "stream", "split", "baseline", "probes", "features")
            )
            or prior["actions"]
        ):
            raise ValueError("MINIMUM_BUDGET_FROZEN_FEATURE_BINDING")
        for phase in ("before", "after"):
            checked_records(unit["baseline"][phase]["test"], stream["units"][intent]["test"], data["codes"])
            checked_records(unit["baseline"][phase]["guard"], rows_for(stream, stream["new"], "test"), data["codes"])
            checked_records(unit["probes"][phase], stream["units"][intent]["probe"], data["codes"])
        if unit["features"] != feature_vector(unit["probes"]["before"], unit["probes"]["after"], spec["layers"]):
            raise ValueError("MINIMUM_BUDGET_FEATURE_RECOMPUTATION")
        for control, phase, checkpoint in (("none", "after", "forgotten"), ("restore", "before", "acquired")):
            action = unit["actions"][control]
            if (
                any(action[key] != unit["baseline"][phase][key] for key in ("test", "guard"))
                or action["adapter_sha256"] != file_hash(learning_path / checkpoint / "adapter_model.safetensors")
                or action["updates"] != 0
            ):
                raise ValueError("MINIMUM_BUDGET_SOURCE_CONTROL")
        anchor = unit["actions"]["replay_balanced"]
        rows, batches = schedule_for(stream, intent, spec, max(pilot["budgets"]))
        history = read_json(path / "actions" / intent / "replay_balanced-updates.json")
        if (
            anchor["updates"] != max(pilot["budgets"])
            or history["rows"] != rows
            or [entry["rows"] for entry in history["updates"]] != batches
            or anchor["schedule_sha256"] != digest({"rows": rows, "batches": batches})
            or anchor["adapter_sha256"]
            != file_hash(path / "actions" / intent / "replay_balanced" / "adapter_model.safetensors")
        ):
            raise ValueError("MINIMUM_BUDGET_EXISTING_SIXTEEN_UPDATE_ANCHOR")
        checked_records(anchor["test"], stream["units"][intent]["test"], data["codes"])
        checked_records(anchor["guard"], rows_for(stream, stream["new"], "test"), data["codes"])
    return units


def permutation_plan(unit_ids, streams, seeds, schemes):
    if len(set(unit_ids)) != len(unit_ids) or len(streams) != len(unit_ids) or len(set(seeds)) != len(seeds):
        raise ValueError("MINIMUM_BUDGET_PERMUTATION_IDENTITIES")
    result = []
    for scheme in schemes:
        if scheme not in {"within_source", "pooled_train"}:
            raise ValueError("MINIMUM_BUDGET_PERMUTATION_SCHEME")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            indices = list(range(len(unit_ids)))
            groups = (
                [indices]
                if scheme == "pooled_train"
                else [[i for i, source in enumerate(streams) if source == name] for name in sorted(set(streams))]
            )
            for group in groups:
                shuffled = rng.permutation(group).tolist()
                for index, value in zip(group, shuffled, strict=True):
                    indices[index] = value
            result.append(
                {
                    "scheme": scheme,
                    "seed": seed,
                    "indices": indices,
                    "fixed_indices": [i for i, value in enumerate(indices) if i == value],
                }
            )
    return {"unit_ids": unit_ids, "source_streams": streams, "draws": result, "redraws": 0}


def planned_permutations(prerequisites, pilot):
    pairs = [
        (f"{name}/{intent}", name)
        for name in pilot["train_streams"]
        for intent in prerequisites["streams"][name]["eligible"]
    ]
    return permutation_plan(
        [item[0] for item in pairs],
        [item[1] for item in pairs],
        pilot["label_permutations"]["seeds"],
        pilot["label_permutations"]["schemes"],
    )


def permutation_diagnostics(plan, outcomes, budgets):
    if [unit["id"] for unit in outcomes] != plan["unit_ids"]:
        raise ValueError("MINIMUM_BUDGET_PERMUTATION_LABEL_ORDER")
    labels = {"minimum_budget": [unit["minimum_budget"] for unit in outcomes]}
    labels.update(
        {
            f"{budget}/{target}": [unit["actions"][str(budget)][target] for unit in outcomes]
            for budget in budgets
            for target in ("recovered", "utility")
        }
    )
    diagnostics = []
    for draw in plan["draws"]:
        changed = {
            key: sum(value != values[draw["indices"][i]] for i, value in enumerate(values))
            for key, values in labels.items()
        }
        diagnostics.append(
            draw | {"changed_labels": changed, "inert_targets": [key for key, count in changed.items() if not count]}
        )
    return {
        "labels": labels,
        "draws": diagnostics,
        "redraws": 0,
        "predictor_fits": 0,
        "interpretation": "Descriptive TRAIN null plans; inert draws retained. No independent-model replication or significance claim.",
    }


def minimum_outcome(unit, spec, budgets):
    if set(unit["actions"]) != {str(budget) for budget in budgets}:
        raise ValueError("MINIMUM_BUDGET_ACTION_GRID_INCOMPLETE")
    outcomes = unit_outcomes(unit, spec)
    passed = [budget for budget in budgets if outcomes["actions"][str(budget)]["recovered"]]
    outcomes["minimum_budget"] = str(min(passed)) if passed else "never"
    outcomes["qualified_budgets"] = passed
    outcomes["later_qualification_loss"] = any(
        not outcomes["actions"][str(budget)]["recovered"] for budget in budgets if passed and budget > min(passed)
    )
    return outcomes


def distribution_gate(outcomes, pilot):
    if not outcomes or any(unit["split"] != "train" for unit in outcomes):
        raise ValueError("MINIMUM_BUDGET_GATE_TRAIN_ONLY")
    if len({unit["id"] for unit in outcomes}) != len(outcomes):
        raise ValueError("MINIMUM_BUDGET_GATE_DUPLICATE_UNIT")
    classes = [str(budget) for budget in pilot["budgets"]] + ["never"]
    if any(unit["minimum_budget"] not in classes for unit in outcomes):
        raise ValueError("MINIMUM_BUDGET_GATE_LABEL")
    counts = Counter(unit["minimum_budget"] for unit in outcomes)
    by_source = {
        name: dict(Counter(unit["minimum_budget"] for unit in outcomes if unit["stream"] == name))
        for name in sorted({unit["stream"] for unit in outcomes})
    }
    enough_classes = [label for label in classes if counts[label] >= pilot["gate"]["minimum_skills_per_class"]]
    passed = (
        len(enough_classes) >= pilot["gate"]["minimum_budget_classes"]
        and len(outcomes) >= pilot["gate"]["minimum_train_skills"]
        and set(by_source) == set(pilot["train_streams"])
    )
    return {
        "qualified": passed,
        "status": "informative_train_budget_labels" if passed else "no_identifiable_prediction_task",
        "counts": dict(counts),
        "counts_by_source": by_source,
        "supported_classes": enough_classes,
        "train_skills": len(outcomes),
        "source_clusters": len(by_source),
        "test_authorized": False,
        "decision": "prepare_new_disjoint_cohorts_and_seal_prediction_protocol"
        if passed
        else "stop_prediction_study; changes_require_another_new_TRAIN_protocol",
        "later_qualification_loss_ids": [unit["id"] for unit in outcomes if unit["later_qualification_loss"]],
    }


def run_budget_arms(observer, stream, intent, spec, budgets, checkpoint, baseline, output, anchor):
    rows, schedule = schedule_for(stream, intent, spec, max(budgets))
    actions, prefixes = {}, {}
    guard_rows = rows_for(stream, stream["new"], "test")
    for budget in budgets:
        observer.reset(checkpoint)
        optimizer = observer.optimizer()
        if observer.optimizer_steps(optimizer):
            raise ValueError("MINIMUM_BUDGET_OPTIMIZER_NOT_FRESH")
        trace, prefix_records, start = [], {}, 0
        for stop in [value for value in budgets if value <= budget]:
            segment = observer.update(rows, schedule[start:stop], optimizer)
            for entry in segment:
                entry["step"] += start
            trace.extend(segment)
            if observer.optimizer_steps(optimizer) != [stop]:
                raise ValueError("MINIMUM_BUDGET_ACTUAL_OPTIMIZER_CLOCK")
            prefix_dir = output / "actions" / intent / str(budget) / f"prefix-{stop}"
            observer.checkpoint(prefix_dir)
            proof = {
                "adapter_sha256": file_hash(prefix_dir / "adapter_model.safetensors"),
                "updates_sha256": digest(trace),
                "optimizer_steps": [stop],
            }
            if stop in prefixes and proof != prefixes[stop]:
                raise ValueError(f"MINIMUM_BUDGET_ACTUAL_PREFIX_MISMATCH budget={budget} prefix={stop}")
            prefixes[stop] = proof
            prefix_records[str(stop)] = proof
            start = stop
        observed = {"test": observer.observe(stream["units"][intent]["test"]), "guard": observer.observe(guard_rows)}
        observer.reset(prefix_dir)
        if observed != {
            "test": observer.observe(stream["units"][intent]["test"]),
            "guard": observer.observe(guard_rows),
        }:
            raise ValueError("MINIMUM_BUDGET_RELOAD_BEHAVIOR")
        action = observed | {
            "updates": budget,
            "total_exposures": budget * spec["repair"]["batch_size"],
            "old_exposures": sum(rows[i]["intent"] == intent for batch in schedule[:budget] for i in batch),
            "new_exposures": sum(rows[i]["intent"] != intent for batch in schedule[:budget] for i in batch),
            "schedule_sha256": digest({"rows": rows, "batches": schedule[:budget]}),
            "start_adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
            "adapter_sha256": proof["adapter_sha256"],
            "prefixes": prefix_records,
            "fresh_optimizer": True,
            "reload_exact": True,
        }
        write_json(output / "actions" / intent / f"{budget}-updates.json", {"rows": rows, "updates": trace})
        actions[str(budget)] = action
        write_json(output / "actions" / intent / f"{budget}-outcome.json", action)
        logging.info(
            "MINIMUM_BUDGET_ARM stream=%s intent=%s updates=%d accuracy=%.3f guard=%.3f",
            stream["id"],
            intent,
            budget,
            accuracy(observed["test"]),
            accuracy(observed["guard"]),
        )
        if budget == max(budgets) and any(
            action[key] != anchor[key] for key in ("test", "guard", "adapter_sha256", "schedule_sha256", "updates")
        ):
            raise ValueError("MINIMUM_BUDGET_SIXTEEN_UPDATE_REPRODUCTION_FAILED")
    observer.reset(checkpoint)
    if baseline != {"test": observer.observe(stream["units"][intent]["test"]), "guard": observer.observe(guard_rows)}:
        raise ValueError("MINIMUM_BUDGET_NONE_RESTORE_BASELINE")
    return actions


def measure(observer, stream, spec, pilot, learning_path, source_units, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    units = [
        {key: copy.deepcopy(unit[key]) for key in ("id", "stream", "split", "intent", "baseline", "probes", "features")}
        | {"actions": {}}
        for unit in source_units
    ]
    for phase, checkpoint in (("before", "acquired"), ("after", "forgotten")):
        observer.reset(learning_path / checkpoint)
        guard = observer.observe(rows_for(stream, stream["new"], "test"))
        for unit in units:
            observed = {"test": observer.observe(stream["units"][unit["intent"]]["test"]), "guard": guard}
            if observed != unit["baseline"][phase]:
                raise ValueError("MINIMUM_BUDGET_PREUPDATE_BASELINE_MISMATCH")
    write_json(output / "features-frozen.json", units)
    write_json(output / "source-outcomes-frozen.json", source_units)
    feature_hash = file_hash(output / "features-frozen.json")
    for unit, prior in zip(units, source_units, strict=True):
        unit["actions"] = run_budget_arms(
            observer,
            stream,
            unit["intent"],
            spec,
            pilot["budgets"],
            learning_path / "forgotten",
            unit["baseline"]["after"],
            output,
            prior["actions"]["replay_balanced"],
        )
        write_json(output / "units" / f"{unit['intent']}.json", unit)
    if feature_hash != file_hash(output / "features-frozen.json") or frozen != parameter_hashes(
        observer.model, frozen_only=True
    ):
        raise ValueError("MINIMUM_BUDGET_FROZEN_FEATURES_OR_BACKBONE_CHANGED")
    outcomes = [minimum_outcome(unit, spec, pilot["budgets"]) for unit in units]
    write_json(output / "units.json", units)
    write_json(output / "outcomes.json", outcomes)
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "stage": "minimum-budget-measure",
        "status": "completed",
        "stream": stream["id"],
        "split": "train",
        "unit_ids": [unit["id"] for unit in units],
        "features_sha256": feature_hash,
        "repair_updates": len(units) * sum(pilot["budgets"]),
        "lens_updates": 0,
        "predictor_fits": 0,
        "stream_summary": stream_summary(units, spec),
        "actual_prefixes_equal": True,
        "anchor_sixteen_reproduced": True,
        "test_authorized": False,
    }


def qualify(sources, prerequisites, config, spec, data, pilot):
    units, source_hashes = [], {}
    identity = pilot_identity(config, spec, data, pilot)
    for name in pilot["train_streams"]:
        path = Path(sources[name])
        result, sealed = verify_result(path)
        if (
            result["stage"] != "minimum-budget-measure"
            or result["status"] != "completed"
            or result["stream"] != name
            or result["split"] != "train"
            or sealed["minimum_budget_identity"] != identity
            or read_json(path / "prerequisites.json") != prerequisites
            or result["features_sha256"] != file_hash(path / "features-frozen.json")
            or read_json(path / "permutation-plan.json") != planned_permutations(prerequisites, pilot)
        ):
            raise ValueError("MINIMUM_BUDGET_GATE_SOURCE_BINDING")
        stream = data["streams"][name]
        provenance = read_json(path / "measurement-source.json")
        if (
            provenance["receipt_sha256"] != pilot["source_receipts"][name]["measure"]
            or provenance["path"] != sealed["dispatch_manifest"]["measurement_source"]
        ):
            raise ValueError("MINIMUM_BUDGET_GATE_MEASUREMENT_PROVENANCE")
        measured = read_json(path / "units.json")
        frozen_units = read_json(path / "features-frozen.json")
        anchors = read_json(path / "source-outcomes-frozen.json")
        if (
            [unit["intent"] for unit in measured] != prerequisites["streams"][name]["eligible"]
            or result["unit_ids"] != [unit["id"] for unit in measured]
            or result["repair_updates"] != len(measured) * sum(pilot["budgets"])
        ):
            raise ValueError("MINIMUM_BUDGET_GATE_ELIGIBLE_SET_OR_UPDATES")
        for unit, before, anchor in zip(measured, frozen_units, anchors, strict=True):
            intent = unit["intent"]
            if (
                unit["id"] != f"{name}/{intent}"
                or unit["stream"] != name
                or unit["split"] != "train"
                or before["actions"]
                or any(unit[key] != before[key] for key in before if key != "actions")
                or any(unit[key] != anchor[key] for key in before if key != "actions")
            ):
                raise ValueError("MINIMUM_BUDGET_GATE_FEATURES_CHANGED")
            rows, batches = schedule_for(stream, intent, spec, max(pilot["budgets"]))
            full_trace = read_json(path / "actions" / intent / f"{max(pilot['budgets'])}-updates.json")["updates"]
            for budget in pilot["budgets"]:
                action = unit["actions"][str(budget)]
                history = read_json(path / "actions" / intent / f"{budget}-updates.json")
                if (
                    history["rows"] != rows
                    or history["updates"] != full_trace[:budget]
                    or [entry["step"] for entry in history["updates"]] != list(range(1, budget + 1))
                    or [entry["rows"] for entry in history["updates"]] != batches[:budget]
                    or action["updates"] != budget
                    or action["total_exposures"] != budget * spec["repair"]["batch_size"]
                    or action["old_exposures"] != budget * spec["repair"]["batch_size"] // 2
                    or action["new_exposures"] != action["old_exposures"]
                    or action["schedule_sha256"] != digest({"rows": rows, "batches": batches[:budget]})
                    or action["start_adapter_sha256"] != anchor["actions"]["none"]["adapter_sha256"]
                ):
                    raise ValueError("MINIMUM_BUDGET_GATE_ACTUAL_UPDATE_PREFIX")
                for stop in [value for value in pilot["budgets"] if value <= budget]:
                    prefix = action["prefixes"][str(stop)]
                    if (
                        prefix != unit["actions"][str(stop)]["prefixes"][str(stop)]
                        or prefix["updates_sha256"] != digest(full_trace[:stop])
                        or prefix["optimizer_steps"] != [stop]
                        or prefix["adapter_sha256"]
                        != file_hash(
                            path / "actions" / intent / str(budget) / f"prefix-{stop}" / "adapter_model.safetensors"
                        )
                    ):
                        raise ValueError("MINIMUM_BUDGET_GATE_ACTUAL_WEIGHT_PREFIX")
                if action["adapter_sha256"] != action["prefixes"][str(budget)]["adapter_sha256"]:
                    raise ValueError("MINIMUM_BUDGET_GATE_FINAL_ADAPTER")
                checked_records(action["test"], stream["units"][intent]["test"], data["codes"])
                checked_records(action["guard"], rows_for(stream, stream["new"], "test"), data["codes"])
            if any(
                unit["actions"][str(max(pilot["budgets"]))][key] != anchor["actions"]["replay_balanced"][key]
                for key in ("test", "guard", "adapter_sha256", "schedule_sha256")
            ):
                raise ValueError("MINIMUM_BUDGET_GATE_ANCHOR")
        if [minimum_outcome(unit, spec, pilot["budgets"]) for unit in measured] != read_json(path / "outcomes.json"):
            raise ValueError("MINIMUM_BUDGET_GATE_OUTCOME_RECOMPUTATION")
        if stream_summary(measured, spec) != result["stream_summary"]:
            raise ValueError("MINIMUM_BUDGET_GATE_SOURCE_SUMMARY")
        units.extend(measured)
        source_hashes[name] = file_hash(path / "receipt.json")
    outcomes = [minimum_outcome(unit, spec, pilot["budgets"]) for unit in units]
    return distribution_gate(outcomes, pilot) | {
        "stage": "minimum-budget-qualify",
        "sources": source_hashes,
        "units": outcomes,
        "lens_updates": 0,
        "repair_updates": 0,
        "run_level": {
            name: stream_summary([unit for unit in units if unit["stream"] == name], spec)
            for name in pilot["train_streams"]
        },
        "predictor_fits": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [minimum-budget] %(message)s")
    manifest, config, spec, data, pilot = load_manifest(args.config)
    output = args.output_dir / "study"
    output.mkdir(parents=True, exist_ok=False)
    try:
        sealed = make_seal(config, spec, data, "minimum-budget-" + manifest["stage"])["payload"]
        sealed.update(
            minimum_budget_identity=pilot_identity(config, spec, data, pilot),
            minimum_budget_protocol=pilot,
            dispatch_manifest=manifest,
        )
        write_json(output / "seal.json", {"payload": sealed, "sha256": digest(sealed)})
        prerequisites = checked_learning_sources(manifest["learning_sources"], config, spec, data, pilot)
        write_json(output / "prerequisites.json", prerequisites)
        if not prerequisites["qualified"]:
            logging.warning("MINIMUM_BUDGET_PREREQUISITES_FAILED no model loading or repair")
            finish(
                output,
                {
                    "stage": "minimum-budget-" + manifest["stage"],
                    "status": "prerequisite_failed",
                    "repair_updates": 0,
                    "lens_updates": 0,
                    "predictor_fits": 0,
                    "test_authorized": False,
                },
            )
            return
        plan = planned_permutations(prerequisites, pilot)
        write_json(output / "permutation-plan.json", plan)
        if manifest["stage"] == "measure":
            name = manifest["stream"]
            stream = data["streams"][name]
            learning = prerequisites["streams"][name]
            source_units = checked_measurement(
                manifest["measurement_source"], stream, learning, config, spec, data, pilot
            )
            write_json(
                output / "measurement-source.json",
                {"path": manifest["measurement_source"], "receipt_sha256": pilot["source_receipts"][name]["measure"]},
            )
            verify_model(config, data)
            observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
            result = measure(observer, stream, spec, pilot, Path(learning["source"]), source_units, output)
        else:
            result = qualify(manifest["sources"], prerequisites, config, spec, data, pilot)
            write_json(output / "budget-qualification.json", result)
            write_json(
                output / "label-permutations.json", permutation_diagnostics(plan, result["units"], pilot["budgets"])
            )
            logging.info(
                "MINIMUM_BUDGET_GATE status=%s counts=%s TEST remains disabled", result["status"], result["counts"]
            )
        finish(output, result)
    except Exception as error:
        logging.exception("MINIMUM_BUDGET_FAILED")
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
