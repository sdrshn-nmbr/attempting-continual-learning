import logging
from datetime import datetime, timezone

import torch
from safetensors.torch import load_file, save_file

from history_control_analysis import alternate_mapping, input_schedule, learning_gate, relabel
from minimum_budget import schedule_for
from prospective_analysis import accuracy
from prospective_data import rows_for
from prospective_model import parameter_hashes, tensor_hash
from protocol import digest, file_hash, read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def observe_gate(observer, stream, condition, codes):
    rows = rows_for(stream, stream["old"], "gate")
    own = relabel(rows, alternate_mapping(stream)[1], codes) if condition == "alternate_binding" else rows
    return {
        "old_own": observer.observe(own),
        "old_original": observer.observe(rows),
        "new": observer.observe(rows_for(stream, stream["new"], "gate")),
    }


def learn_control(observer, config, spec, data, stream, condition, protocol, historical, schedules, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    if frozen != read_json(historical / "frozen-backbone.json"):
        raise ValueError("HISTORY_CONTROL_INITIAL_BACKBONE_MISMATCH")
    observer.reset(historical / "initial")
    optimizer = observer.optimizer()
    if observer.optimizer_steps(optimizer):
        raise ValueError("HISTORY_CONTROL_INITIAL_OPTIMIZER_NOT_EMPTY")
    checkpoints = {"initial": observer.checkpoint(output / "initial", optimizer)}
    if file_hash(output / "initial/adapter_model.safetensors") != file_hash(
        historical / "initial/adapter_model.safetensors"
    ):
        raise ValueError("HISTORY_CONTROL_INITIAL_ADAPTER_MISMATCH")
    observations = {"initial": observe_gate(observer, stream, condition, data["codes"])}
    original_initial = read_json(historical / "observations.json")["initial"]
    if (
        observations["initial"]["old_original"] != original_initial["old_gate"]
        or observations["initial"]["new"] != original_initial["new_gate"]
    ):
        raise ValueError("HISTORY_CONTROL_INITIAL_NATIVE_OUTPUT_MISMATCH")
    write_json(output / "schedules-before-updates.json", schedules)
    old_updates, new_updates = 0, 0
    clocks = {"initial": []}
    if condition == "alternate_binding":
        rows = relabel(schedules["old"]["rows"], alternate_mapping(stream)[1], data["codes"])
        batches = schedules["old"]["batches"]
        if digest(input_schedule(rows, batches)) != schedules["old"]["input_schedule_sha256"]:
            raise ValueError("HISTORY_CONTROL_RELABEL_CHANGED_INPUT_SCHEDULE")
        trace = observer.update(rows, batches, optimizer)
        old_updates = len(trace)
        if [entry["rows"] for entry in trace] != batches or observer.optimizer_steps(optimizer) != [len(batches)]:
            raise ValueError("HISTORY_CONTROL_OLD_SCHEDULE_OR_CLOCK")
        write_json(output / "old-updates.json", {"rows": rows, "updates": trace})
        observations["acquired"] = observe_gate(observer, stream, condition, data["codes"])
        checkpoints["acquired"] = observer.checkpoint(output / "acquired", optimizer)
        observer.reset(output / "acquired")
        if observe_gate(observer, stream, condition, data["codes"]) != observations["acquired"]:
            raise ValueError("HISTORY_CONTROL_ACQUIRED_RELOAD")
        clocks["acquired"] = observer.optimizer_steps(optimizer)
    early = learning_gate(observations, condition, stream, spec, protocol, data["codes"])
    write_json(output / "old-acquisition-gate.json", early)
    if early["old_source_passed"]:
        rows, batches = schedules["new"]["rows"], schedules["new"]["batches"]
        midpoint = len(batches) // 2
        first = observer.update(rows, batches[:midpoint], optimizer)
        clocks["new_midpoint"] = observer.optimizer_steps(optimizer)
        if clocks["new_midpoint"] != [old_updates + midpoint]:
            raise ValueError("HISTORY_CONTROL_MIDPOINT_OPTIMIZER_CLOCK")
        observations["new_midpoint"] = observe_gate(observer, stream, condition, data["codes"])
        second = observer.update(rows, batches[midpoint:], optimizer)
        for entry in second:
            entry["step"] += midpoint
        trace = first + second
        new_updates = len(trace)
        if [entry["rows"] for entry in trace] != batches or observer.optimizer_steps(optimizer) != [
            old_updates + len(batches)
        ]:
            raise ValueError("HISTORY_CONTROL_NEW_SCHEDULE_OR_CARRIED_CLOCK")
        write_json(output / "new-updates.json", {"rows": rows, "updates": trace})
        observations["post_stream"] = observe_gate(observer, stream, condition, data["codes"])
        checkpoints["post_stream"] = observer.checkpoint(output / "post_stream", optimizer)
        observer.reset(output / "post_stream")
        if observe_gate(observer, stream, condition, data["codes"]) != observations["post_stream"]:
            raise ValueError("HISTORY_CONTROL_POST_STREAM_RELOAD")
    else:
        logging.warning("HISTORY_CONTROL_ACQUISITION_FAILED stream=%s; stop before new updates", stream["id"])
    clocks["final"] = observer.optimizer_steps(optimizer)
    gate = learning_gate(observations, condition, stream, spec, protocol, data["codes"])
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("HISTORY_CONTROL_LEARNING_CHANGED_BACKBONE")
    write_json(output / "observations.json", observations)
    write_json(output / "learning-gate.json", gate)
    write_json(output / "checkpoints.json", checkpoints)
    write_json(output / "frozen-backbone.json", frozen)
    return gate | {
        "stream": stream["id"],
        "split": stream["split"],
        "condition": condition,
        "old_updates": old_updates,
        "new_updates": new_updates,
        "old_exposures": old_updates * spec["optimizer"]["batch_size"],
        "new_exposures": new_updates * spec["optimizer"]["batch_size"],
        "optimizer_steps": clocks,
        "lens_updates": 0,
        "predictor_fits": 0,
        "initial_adapter_sha256": file_hash(output / "initial/adapter_model.safetensors"),
        "frozen_backbone_unchanged": True,
        "decision": "await_common_gate" if gate["qualified"] else "stop_all_control_repairs",
    }


class GradientWitness:
    def __init__(self, observer, optimizer, output, prefixes):
        self.observer, self.optimizer, self.output = observer, optimizer, output
        self.prefixes = set(prefixes)
        self.records = []

    def before_step(self, optimizer, args, kwargs):
        step = len(self.records) + 1
        clock = self.observer.optimizer_steps(optimizer)
        if optimizer is not self.optimizer or clock != ([] if step == 1 else [step - 1]):
            raise ValueError("HISTORY_CONTROL_GRADIENT_CLOCK_DISCONTINUITY")
        gradients = {}
        for name, parameter in self.observer.model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"HISTORY_CONTROL_MISSING_OR_NONFINITE_GRADIENT {name}")
                gradients[name] = parameter.grad.detach().cpu().contiguous().clone()
        if not gradients:
            raise ValueError("HISTORY_CONTROL_EMPTY_GRADIENT_WITNESS")
        tensors = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
            for name, value in gradients.items()
        }
        self.records.append({"step": step, "optimizer_before": clock, "tensors": tensors, "sha256": digest(tensors)})
        if step in self.prefixes:
            self.output.mkdir(parents=True, exist_ok=True)
            path = self.output / f"gradient-{step}.safetensors"
            if path.exists():
                raise ValueError("HISTORY_CONTROL_GRADIENT_ARTIFACT_EXISTS")
            save_file(gradients, path)


def observe_target(observer, stream, intent):
    return {
        "test": observer.observe(stream["units"][intent]["test"]),
        "guard": observer.observe(rows_for(stream, stream["new"], "test")),
    }


def repair_grid(observer, spec, protocol, stream, intent, checkpoint, baseline, output):
    budgets = protocol["repair"]["budgets"]
    rows, batches = schedule_for(stream, intent, spec, max(budgets))
    prefixes, actions = {}, {}
    initial_hash = file_hash(checkpoint / "adapter_model.safetensors")
    for budget in [max(budgets), *budgets[:-1]]:
        observer.reset(checkpoint)
        if observe_target(observer, stream, intent) != baseline:
            raise ValueError("HISTORY_CONTROL_REPAIR_START_PARITY")
        optimizer = observer.optimizer()
        if observer.optimizer_steps(optimizer):
            raise ValueError("HISTORY_CONTROL_REPAIR_OPTIMIZER_NOT_FRESH")
        arm = output / "actions" / intent / str(budget)
        witness = GradientWitness(observer, optimizer, arm, budgets)
        handle = optimizer.register_step_pre_hook(witness.before_step)
        started, trace, start, prefix_proofs = now(), [], 0, {}
        try:
            for stop in [value for value in budgets if value <= budget]:
                segment = observer.update(rows, batches[start:stop], optimizer)
                for entry in segment:
                    entry["step"] += start
                trace.extend(segment)
                if observer.optimizer_steps(optimizer) != [stop] or len(witness.records) != stop:
                    raise ValueError("HISTORY_CONTROL_REPAIR_PREFIX_CLOCK")
                prefix = arm / f"prefix-{stop}"
                observer.checkpoint(prefix)
                proof = {
                    "adapter_sha256": file_hash(prefix / "adapter_model.safetensors"),
                    "trace_sha256": digest(trace),
                    "gradients_sha256": digest(witness.records),
                    "gradient_tensors_sha256": witness.records[-1]["sha256"],
                    "optimizer_steps": observer.optimizer_steps(optimizer),
                }
                if stop in prefixes and proof != prefixes[stop]:
                    raise ValueError("HISTORY_CONTROL_ACTUAL_WEIGHT_TRACE_GRADIENT_PREFIX_MISMATCH")
                prefixes[stop] = prefix_proofs[str(stop)] = proof
                start = stop
        finally:
            handle.remove()
        observed = observe_target(observer, stream, intent)
        observer.reset(prefix)
        if observed != observe_target(observer, stream, intent):
            raise ValueError("HISTORY_CONTROL_REPAIR_PREFIX_RELOAD")
        actions[str(budget)] = observed | {
            "updates": budget,
            "old_exposures": budget * 4,
            "new_exposures": budget * 4,
            "start_adapter_sha256": initial_hash,
            "adapter_sha256": proof["adapter_sha256"],
            "trace_sha256": proof["trace_sha256"],
            "gradients_sha256": proof["gradients_sha256"],
            "schedule_sha256": digest({"rows": rows, "batches": batches[:budget]}),
            "input_schedule_sha256": digest(input_schedule(rows, batches[:budget])),
            "optimizer_steps": observer.optimizer_steps(optimizer),
            "fresh_optimizer": True,
            "reload_exact": True,
            "prefix_proofs": prefix_proofs,
        }
        write_json(output / "actions" / intent / f"{budget}-updates.json", {"rows": rows, "updates": trace})
        write_json(output / "actions" / intent / f"{budget}-gradients.json", witness.records)
        write_json(output / "actions" / intent / f"{budget}-outcome.json", actions[str(budget)])
        write_json(output / "actions" / intent / f"{budget}-timing.json", {"started_at": started, "finished_at": now()})
        observer.reset(checkpoint)
        if observe_target(observer, stream, intent) != baseline:
            raise ValueError("HISTORY_CONTROL_POST_ARM_RESTORE")
        logging.info(
            "HISTORY_CONTROL_REPAIR stream=%s intent=%s updates=%d accuracy=%.3f guard=%.3f",
            stream["id"],
            intent,
            budget,
            accuracy(observed["test"]),
            accuracy(observed["guard"]),
        )
    return actions


def repair_control(observer, spec, protocol, stream, condition, learning, common, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    checkpoint = learning / "post_stream"
    observer.reset(checkpoint)
    guard = observer.observe(rows_for(stream, stream["new"], "test"))
    units = [
        {
            "id": f"{stream['id']}/{intent}",
            "intent": intent,
            "stream": stream["id"],
            "split": stream["split"],
            "condition": condition,
            "baseline": {"test": observer.observe(stream["units"][intent]["test"]), "guard": guard},
            "actions": {},
        }
        for intent in common
    ]
    write_json(output / "baselines-before-updates.json", {"at": now(), "units": units})
    baseline_hash = file_hash(output / "baselines-before-updates.json")
    for unit in units:
        unit["actions"] = repair_grid(
            observer, spec, protocol, stream, unit["intent"], checkpoint, unit["baseline"], output
        )
        write_json(output / "units" / f"{unit['intent']}.json", unit)
    if frozen != parameter_hashes(observer.model, frozen_only=True) or baseline_hash != file_hash(
        output / "baselines-before-updates.json"
    ):
        raise ValueError("HISTORY_CONTROL_REPAIR_CHANGED_FROZEN_STATE")
    write_json(output / "units.json", units)
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "qualified": True,
        "status": "completed",
        "stream": stream["id"],
        "split": stream["split"],
        "condition": condition,
        "unit_ids": [unit["id"] for unit in units],
        "repair_updates": len(units) * sum(protocol["repair"]["budgets"]),
        "lens_updates": 0,
        "predictor_fits": 0,
        "baselines_sha256": baseline_hash,
        "frozen_backbone_unchanged": True,
    }


def verify_gradients(path, unit, budgets):
    base = path / "actions" / unit["intent"]
    maximum = max(budgets)
    full = read_json(base / f"{maximum}-gradients.json")
    for budget in budgets:
        trace = read_json(base / f"{budget}-gradients.json")
        if (
            trace != full[:budget]
            or len(trace) != budget
            or digest(trace) != unit["actions"][str(budget)]["gradients_sha256"]
        ):
            raise ValueError("HISTORY_CONTROL_GRADIENT_PREFIX_RECOMPUTATION")
        for index, entry in enumerate(trace):
            if (
                entry["step"] != index + 1
                or entry["optimizer_before"] != ([] if index == 0 else [index])
                or not entry["tensors"]
                or digest(entry["tensors"]) != entry["sha256"]
            ):
                raise ValueError("HISTORY_CONTROL_GRADIENT_TRACE_CLOCK_OR_DIGEST")
        for stop in [value for value in budgets if value <= budget]:
            tensors = load_file(base / str(budget) / f"gradient-{stop}.safetensors", device="cpu")
            observed = {
                name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
                for name, value in tensors.items()
            }
            proof = unit["actions"][str(budget)]["prefix_proofs"][str(stop)]
            updates = read_json(base / f"{budget}-updates.json")["updates"][:stop]
            if observed != trace[stop - 1]["tensors"] or proof != {
                "adapter_sha256": file_hash(base / str(budget) / f"prefix-{stop}" / "adapter_model.safetensors"),
                "trace_sha256": digest(updates),
                "gradients_sha256": digest(trace[:stop]),
                "gradient_tensors_sha256": digest(observed),
                "optimizer_steps": [stop],
            }:
                raise ValueError("HISTORY_CONTROL_SAVED_GRADIENT_OR_PREFIX_PROOF")
