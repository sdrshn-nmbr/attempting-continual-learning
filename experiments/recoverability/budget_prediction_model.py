import copy
import logging

from minimum_budget import minimum_outcome, schedule_for
from prospective_analysis import accuracy, feature_vector, stream_summary
from prospective_data import rows_for, seed_for
from prospective_model import parameter_hashes
from protocol import digest, file_hash, read_json, write_json
from tuned_lens import cache_activations, evaluate_translations, train_lens


def readout(observer, spec, data, stream, learning, intents, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    observer.reset(learning / "acquired")
    lens, training = train_lens(observer, data, spec, output, seed_for(stream["seed"], "lens"))
    write_json(output / "lens-training.json", training)
    observer.reset(learning / "forgotten")
    cache, metadata = cache_activations(observer, data["lens_check"], spec["lens"]["positions_per_prompt"])
    late = evaluate_translations(lens, observer, cache, spec["lens"]["token_batch_size"])
    del cache
    improvement = 1 - sum(value["tuned"] for value in late.values()) / max(
        sum(value["frozen"] for value in late.values()), 1e-12
    )
    write_json(
        output / "lens-late-check.json",
        {"metadata": metadata, "heldout_kl": late, "relative_kl_improvement": improvement},
    )
    qualified = training["translator_qualified"] and improvement >= spec["lens"]["min_relative_kl_improvement"]
    if not qualified:
        return {
            "qualified": False,
            "status": "lens_calibration_failed",
            "repair_updates": 0,
            "lens_updates": spec["lens"]["updates"],
            "decision": "stop_before_repairs; new_protocol_required",
        }
    units = [
        {
            "id": f"{stream['id']}/{intent}",
            "intent": intent,
            "stream": stream["id"],
            "split": stream["split"],
            "baseline": {},
            "probes": {},
            "actions": {},
        }
        for intent in intents
    ]
    for phase, checkpoint in (("before", "acquired"), ("after", "forgotten")):
        observer.reset(learning / checkpoint)
        guard = observer.observe(rows_for(stream, stream["new"], "test"))
        for unit in units:
            rows = stream["units"][unit["intent"]]
            unit["baseline"][phase] = {"test": observer.observe(rows["test"]), "guard": guard}
            unit["probes"][phase] = observer.observe(rows["probe"], lens=lens)
    for unit in units:
        unit["features"] = feature_vector(unit["probes"]["before"], unit["probes"]["after"], spec["layers"])
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("BUDGET_PREDICTION_READOUT_CHANGED_BACKBONE")
    write_json(output / "readouts.json", units)
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "qualified": True,
        "status": "qualified",
        "stream": stream["id"],
        "split": stream["split"],
        "repair_updates": 0,
        "lens_updates": spec["lens"]["updates"],
        "unit_ids": [unit["id"] for unit in units],
        "readouts_sha256": file_hash(output / "readouts.json"),
        "lens_sha256": training["lens_sha256"],
        "acquired_kl_improvement": training["relative_kl_improvement"],
        "forgotten_kl_improvement": improvement,
    }


def grid(observer, spec, protocol, stream, intent, learning, output):
    rows, batches = schedule_for(stream, intent, spec, max(protocol["budgets"]))
    prefixes, actions = {}, {}
    order = [max(protocol["budgets"]), *protocol["budgets"][:-1]]
    for budget in order:
        observer.reset(learning / "forgotten")
        optimizer = observer.optimizer()
        if observer.optimizer_steps(optimizer):
            raise ValueError("BUDGET_PREDICTION_REPAIR_OPTIMIZER_NOT_FRESH")
        trace, start = [], 0
        for stop in [value for value in protocol["budgets"] if value <= budget]:
            segment = observer.update(rows, batches[start:stop], optimizer)
            for entry in segment:
                entry["step"] += start
            trace.extend(segment)
            if observer.optimizer_steps(optimizer) != [stop]:
                raise ValueError("BUDGET_PREDICTION_REPAIR_CLOCK")
            checkpoint = output / "actions" / intent / str(budget) / f"prefix-{stop}"
            observer.checkpoint(checkpoint)
            proof = {
                "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
                "trace_sha256": digest(trace),
            }
            if stop in prefixes and proof != prefixes[stop]:
                raise ValueError("BUDGET_PREDICTION_ACTUAL_PREFIX_MISMATCH")
            prefixes[stop] = proof
            start = stop
        observed = {
            "test": observer.observe(stream["units"][intent]["test"]),
            "guard": observer.observe(rows_for(stream, stream["new"], "test")),
        }
        observer.reset(checkpoint)
        if observed != {
            "test": observer.observe(stream["units"][intent]["test"]),
            "guard": observer.observe(rows_for(stream, stream["new"], "test")),
        }:
            raise ValueError("BUDGET_PREDICTION_REPAIR_RELOAD")
        actions[str(budget)] = observed | {
            "updates": budget,
            "old_exposures": budget * 4,
            "new_exposures": budget * 4,
            "start_adapter_sha256": file_hash(learning / "forgotten/adapter_model.safetensors"),
            "adapter_sha256": proof["adapter_sha256"],
            "trace_sha256": proof["trace_sha256"],
            "schedule_sha256": digest({"rows": rows, "batches": batches[:budget]}),
            "fresh_optimizer": True,
            "reload_exact": True,
        }
        write_json(output / "actions" / intent / f"{budget}-updates.json", {"rows": rows, "updates": trace})
        write_json(output / "actions" / intent / f"{budget}-outcome.json", actions[str(budget)])
        logging.info(
            "BUDGET_PREDICTION_REPAIR stream=%s intent=%s updates=%d accuracy=%.3f guard=%.3f",
            stream["id"],
            intent,
            budget,
            accuracy(observed["test"]),
            accuracy(observed["guard"]),
        )
    return actions


def repair(observer, spec, protocol, stream, learning, units, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    units = copy.deepcopy(units)
    if any(unit["actions"] for unit in units):
        raise ValueError("BUDGET_PREDICTION_READOUT_ALREADY_REPAIRED")
    for phase, checkpoint in (("before", "acquired"), ("after", "forgotten")):
        observer.reset(learning / checkpoint)
        guard = observer.observe(rows_for(stream, stream["new"], "test"))
        for unit in units:
            if unit["baseline"][phase] != {
                "test": observer.observe(stream["units"][unit["intent"]]["test"]),
                "guard": guard,
            }:
                raise ValueError("BUDGET_PREDICTION_PREUPDATE_BASELINE_PARITY")
    write_json(output / "features-frozen.json", units)
    feature_hash = file_hash(output / "features-frozen.json")
    for unit in units:
        unit["actions"] = grid(observer, spec, protocol, stream, unit["intent"], learning, output)
        observer.reset(learning / "forgotten")
        if unit["baseline"]["after"] != {
            "test": observer.observe(stream["units"][unit["intent"]]["test"]),
            "guard": observer.observe(rows_for(stream, stream["new"], "test")),
        }:
            raise ValueError("BUDGET_PREDICTION_NO_ACTION_CONTROL")
        write_json(output / "units" / f"{unit['intent']}.json", unit)
    if frozen != parameter_hashes(observer.model, frozen_only=True) or feature_hash != file_hash(
        output / "features-frozen.json"
    ):
        raise ValueError("BUDGET_PREDICTION_FROZEN_STATE_CHANGED")
    write_json(output / "units.json", units)
    write_json(output / "outcomes.json", [minimum_outcome(unit, spec, protocol["budgets"]) for unit in units])
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "qualified": True,
        "status": "completed",
        "stream": stream["id"],
        "split": stream["split"],
        "unit_ids": [unit["id"] for unit in units],
        "repair_updates": len(units) * sum(protocol["budgets"]),
        "lens_updates": 0,
        "features_sha256": feature_hash,
        "run_level": stream_summary(units, spec),
    }


def verify_grid(path, unit, stream, spec, protocol):
    rows, batches = schedule_for(stream, unit["intent"], spec, max(protocol["budgets"]))
    maximum = str(max(protocol["budgets"]))
    full = read_json(path / "actions" / unit["intent"] / f"{maximum}-updates.json")["updates"]
    for budget in protocol["budgets"]:
        action = unit["actions"][str(budget)]
        history = read_json(path / "actions" / unit["intent"] / f"{budget}-updates.json")
        checkpoint = path / "actions" / unit["intent"] / str(budget) / f"prefix-{budget}" / "adapter_model.safetensors"
        reference = path / "actions" / unit["intent"] / maximum / f"prefix-{budget}" / "adapter_model.safetensors"
        if (
            history["rows"] != rows
            or history["updates"] != full[:budget]
            or [entry["rows"] for entry in history["updates"]] != batches[:budget]
            or [entry["step"] for entry in history["updates"]] != list(range(1, budget + 1))
            or digest(history["updates"]) != action["trace_sha256"]
            or action["updates"] != budget
            or action["old_exposures"] != budget * 4
            or action["new_exposures"] != budget * 4
            or action["schedule_sha256"] != digest({"rows": rows, "batches": batches[:budget]})
            or action["adapter_sha256"] != file_hash(checkpoint)
            or file_hash(reference) != file_hash(checkpoint)
        ):
            raise ValueError("BUDGET_PREDICTION_GRID_RECOMPUTATION")
