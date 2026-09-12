import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import device4_curriculum as original_lane
import torch
from generated_contract import (
    digest,
    file_digest,
    make_corpus,
    qualification_gate,
    qualification_rows,
    teacher_prompt,
)
from safetensors.torch import load_file
from torch.nn import functional

from run import adapter_parameters, tensor_digest
from tasks import FAMILIES, to_records

ROOT = Path(__file__).resolve().parent
CONTRACT = "device4_interrupted_curriculum_recovery_20260912"
SOURCE_FILES = ("device4_recovery.py", *original_lane.SOURCE_FILES)
RECORD_KEYS = ("step", "ids", "response_sha256", "loss_tokens", "loss", "gradient_norm")
LOGGER = logging.getLogger("distillation.curriculum_recovery")
write_new = original_lane.write_new


def source_hashes():
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def checked_json(spec):
    path = ROOT / spec["path"]
    if file_digest(path) != spec["sha256"]:
        raise ValueError(f"RECOVERY_BOUND_INPUT_CHANGED: {path}")
    return json.loads(path.read_text())


def validate_design(design):
    upstream = checked_json(design["original_design"])
    original = original_lane.validate_design(upstream)
    snapshot = checked_json(design["interrupted_snapshot"])
    worker = checked_json(design["worker_failure"])
    if design["contract"] != CONTRACT or design["original_source_sha256"] != original_lane.source_hashes():
        raise ValueError("RECOVERY_ORIGINAL_SOURCE_CHANGED")
    if snapshot["tokenizer_identity_receipt_sha256"] != upstream["history"]["fallback24"]["files_sha256"]["qualification.json"]:
        raise ValueError("RECOVERY_TOKENIZER_PROVENANCE_CHANGED")
    if design["resume"] != {
        "arm": "legacy96_budget", "checkpoint": 96, "last_logged_update": 223, "final_update": 384,
        "new_updates": 288, "new_exposures": 1152, "reuse_arms": ["flat_mixed", "primitive_first"],
        "replay_loss_abs_tolerance": 1e-5, "replay_loss_rel_tolerance": 1e-4,
    }:
        raise ValueError("RECOVERY_SELECTED_BOUNDARY_CHANGED")
    execution = snapshot["execution"]
    if (
        execution["status"] != "interrupted" or execution["exit_code"] is not None
        or execution["interruption_review"]["child_exit_code_observed"]
        or execution["interruption_review"]["scientific_completion_claimed"]
        or execution["supervisor"]["pod_uid"] != worker["metadata"]["uid"]
        or worker["status"]["phase"] != "Failed"
        or execution["interruption_review"]["pod_proof_sha256"] != snapshot["worker_failure_sha256"]
        or file_digest(ROOT / design["worker_failure"]["path"]) != snapshot["worker_failure_sha256"]
    ):
        raise ValueError("RECOVERY_AUTHENTIC_INTERRUPTION_REQUIRED")
    terminated = next(row for row in worker["status"]["containerStatuses"] if row["name"] == "portfolio")["state"]["terminated"]
    if terminated["finishedAt"] != execution["finished_at"] or terminated["exitCode"] != execution["interruption_review"]["supervisor_exit_code"]:
        raise ValueError("RECOVERY_INTERRUPTION_TIME_OR_SUPERVISOR_EXIT_CHANGED")
    return upstream, original, snapshot


def saved_parameters(path):
    state = load_file(str(path), device="cpu")
    return {name.replace(".lora_A.weight", ".lora_A.ADAPTER.weight").replace(".lora_B.weight", ".lora_B.ADAPTER.weight"): value for name, value in state.items()}


def adapter_spec(path):
    return {
        "path": str(path.resolve()), "tensor_sha256": tensor_digest(saved_parameters(path / "adapter_model.safetensors")),
        "files_sha256": {name: file_digest(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
    }


def inspect_optimizer(path, step, parameter_names, parameters, recipe):
    optimizer = torch.load(path, map_location="cpu", weights_only=True)
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("RECOVERY_OPTIMIZER_GROUP_COUNT")
    group = optimizer["param_groups"][0]
    if (
        group["lr"] != recipe["learning_rate"] or group["weight_decay"] != recipe["weight_decay"]
        or tuple(group["betas"]) != (0.9, 0.999) or group["eps"] != 1e-8 or group["amsgrad"]
        or group["maximize"] or group["capturable"] or group["differentiable"]
        or group["foreach"] is not None or group["fused"] is not None
        or len(group["params"]) != len(parameter_names) or set(parameter_names) != set(parameters)
        or group["params"] != list(range(len(parameter_names)))
        or set(optimizer["state"]) != set(group["params"])
    ):
        raise ValueError("RECOVERY_OPTIMIZER_RECIPE_OR_PARAMETER_ORDER_CHANGED")
    for index, name in zip(group["params"], parameter_names, strict=True):
        state = optimizer["state"][index]
        if int(state["step"]) != step or float(state["step"]) != step:
            raise ValueError("RECOVERY_OPTIMIZER_STEP_MISMATCH")
        for field in ("exp_avg", "exp_avg_sq"):
            value = state[field]
            if value.shape != parameters[name].shape or value.dtype != parameters[name].dtype or not torch.isfinite(value).all():
                raise ValueError("RECOVERY_OPTIMIZER_MOMENTS_INVALID")
    return optimizer


def verify_ledger(ledger, schedule, count):
    if len(ledger) != count or [r["ids"] for r in ledger] != schedule[:count] or [r["step"] for r in ledger] != list(range(1, count + 1)):
        raise ValueError("RECOVERY_LOGGED_EXPOSURES_NOT_EXACT_SCHEDULE_PREFIX")
    for row in ledger:
        if set(row) != set(RECORD_KEYS) or len(row["response_sha256"]) != len(row["ids"]) or row["loss_tokens"] <= 0 or not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm"]) or row["gradient_norm"] <= 0:
            raise ValueError("RECOVERY_LOGGED_UPDATE_INVALID")


def inspect_partial(folder, snapshot, upstream, original, corpus, schedules):
    folder = Path(folder)
    for name, spec in snapshot["files"].items():
        path = folder / name
        if path.stat().st_size != spec["bytes"] or file_digest(path) != spec["sha256"]:
            raise ValueError(f"RECOVERY_INTERRUPTED_ARTIFACT_CHANGED: {name}")
    if (folder / "training.json").exists() or (folder / "evaluation/qualification.json").exists():
        raise ValueError("RECOVERY_SOURCE_WAS_NOT_AN_INCOMPLETE_RUN")
    seal = json.loads((folder / "seal.json").read_text())
    runtime = json.loads((folder / "runtime.json").read_text())
    if (
        seal["source_sha256"] != original_lane.source_hashes() or seal["design"] != upstream
        or seal["design_sha256"] != digest(upstream) or seal["dataset_sha256"] != digest(to_records(corpus))
        or json.loads((folder / "schedule.json").read_text()) != schedules
        or json.loads((folder / "execution.json").read_text()) != snapshot["execution"]
    ):
        raise ValueError("RECOVERY_SOURCE_SEAL_OR_SCHEDULE_CHANGED")
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    ledgers = {arm: [{key: row[key] for key in RECORD_KEYS} for row in events if row["event"] == "device4_teacher_update" and row["arm"] == arm] for arm in original_lane.ARMS}
    checkpoints = {}
    for arm, ledger in ledgers.items():
        count = snapshot["observed_updates"][arm]
        verify_ledger(ledger, schedules[arm], count)
        if arm != "legacy96_budget" and json.loads((folder / arm / "ledger.json").read_text()) != ledger:
            raise ValueError("RECOVERY_COMPLETED_ARM_LEDGER_EVENT_MISMATCH")
        checkpoints[arm] = {}
        for step in upstream["training"]["checkpoints"]:
            if step > count:
                continue
            destination = folder / arm / f"checkpoint{step}"
            spec = adapter_spec(destination / "teacher")
            parameters = saved_parameters(destination / "teacher/adapter_model.safetensors")
            inspect_optimizer(destination / "optimizer.pt", step, runtime["targeted_parameter_names"], parameters, upstream["training"])
            records = json.loads((destination / "train_diagnostics.json").read_text())
            original_lane.verify_records(corpus["train"], records, corpus, original, snapshot["eos_token_id"], snapshot["special_token_ids"])
            checkpoints[arm][str(step)] = {
                "adapter": spec, "optimizer_path": str((destination / "optimizer.pt").resolve()),
                "optimizer_sha256": file_digest(destination / "optimizer.pt"),
                "train_diagnostics_path": str((destination / "train_diagnostics.json").resolve()),
                "train_diagnostics_sha256": file_digest(destination / "train_diagnostics.json"), "train_summary": original_lane.summarize(records),
            }
    initial = adapter_spec(folder / "initial/teacher")
    if initial["tensor_sha256"] != runtime["adapter_initial_sha256"]:
        raise ValueError("RECOVERY_INITIAL_ADAPTER_IDENTITY_CHANGED")
    return {"source_run": str(folder.resolve()), "original_training_pid": seal["pid"], "runtime": runtime, "initial": initial, "checkpoints": checkpoints, "ledgers": ledgers, "snapshot_sha256": digest(snapshot)}


def verify_target_ledgers(runner, corpus, ledgers):
    lookup = {task.uid: task for task in corpus["train"]}
    for ledger in ledgers.values():
        for row in ledger:
            responses = [runner.target_tokens(lookup[uid]).tolist() for uid in row["ids"]]
            if [digest(r) for r in responses] != row["response_sha256"] or sum(map(len, responses)) != row["loss_tokens"]:
                raise ValueError("RECOVERY_ORIGINAL_GOLD_TOKEN_LEDGER_CHANGED")


def reload_parity(runner, checkpoint, output, label):
    original_lane.restore(runner, checkpoint["adapter"])
    measured = original_lane.measure(runner, runner.corpus["train"], output, label, forced=False)
    saved = json.loads(Path(checkpoint["train_diagnostics_path"]).read_text())
    if [r["generation"] for r in measured] != [r["generation"] for r in saved]:
        raise ValueError(f"RECOVERY_SOURCE_RELOAD_GENERATION_PARITY_FAILED: {label}")
    return {"rows": len(measured), "exact_generation_parity": True, "adapter_tensor_sha256": checkpoint["adapter"]["tensor_sha256"]}


def resume_legacy(runner, upstream, recovery, checkpoint, original_ledger, schedule, output):
    recipe = upstream["training"]
    start = recovery["checkpoint"]
    original_lane.restore(runner, checkpoint["adapter"])
    runner.model.set_adapter("teacher")
    runner.model.eval()
    parameters = adapter_parameters(runner.model, "teacher")
    optimizer_state = inspect_optimizer(Path(checkpoint["optimizer_path"]), start, list(parameters), parameters, recipe)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    optimizer.load_state_dict(optimizer_state)
    torch.manual_seed(recipe["order_seed"])
    initial_base = original_lane.frozen_digest(runner)
    initial_raw = tensor_digest(adapter_parameters(runner.model, "student"))
    tasks = {task.uid: task for task in runner.corpus["train"]}
    ledger, checkpoints, replay = list(original_ledger[:start]), {}, []
    for step, ids in enumerate(schedule[start:], start + 1):
        optimizer.zero_grad(set_to_none=True)
        losses, responses = [], []
        for uid in ids:
            task = tasks[uid]
            response = runner.target_tokens(task)
            logits = runner.logits(runner.prompt_ids(teacher_prompt(task, runner.corpus["demonstration"])), response)
            loss = functional.cross_entropy(logits.float(), response)
            if not torch.isfinite(loss):
                raise ValueError(f"RECOVERY_NONFINITE_LOSS: {step}")
            (loss / len(ids)).backward()
            losses.append(float(loss.detach()))
            responses.append(response.tolist())
            del logits, loss
        norm = float(torch.nn.utils.clip_grad_norm_(list(parameters.values()), recipe["max_grad_norm"]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"RECOVERY_INVALID_GRADIENT: {step}")
        record = {"step": step, "ids": ids, "response_sha256": [digest(row) for row in responses], "loss_tokens": sum(len(row) for row in responses), "loss": sum(losses) / len(losses), "gradient_norm": norm}
        if step <= len(original_ledger):
            previous = original_ledger[step - 1]
            exact = all(record[field] == previous[field] for field in RECORD_KEYS)
            close = all(math.isclose(record[field], previous[field], rel_tol=recovery["replay_loss_rel_tolerance"], abs_tol=recovery["replay_loss_abs_tolerance"]) for field in ("loss", "gradient_norm"))
            delta = {"step": step, "exact": exact, "within_predeclared_tolerance": close, "loss_difference": record["loss"] - previous["loss"], "gradient_norm_difference": norm - previous["gradient_norm"]}
            replay.append(delta)
            runner.event("recovery_replay_check", **delta)
            if not close or any(record[field] != previous[field] for field in ("ids", "response_sha256", "loss_tokens")):
                raise ValueError(f"RECOVERY_NUMERICAL_REPLAY_MISMATCH_BEFORE_UPDATE: {step}")
        optimizer.step()
        runner.assert_frozen()
        if any(p.grad is not None for p in runner.student_parameters):
            raise ValueError("RECOVERY_RAW_REFERENCE_RECEIVED_GRADIENT")
        ledger.append(record)
        runner.event("device4_recovered_teacher_update", arm="legacy96_budget", **record)
        if step in recipe["checkpoints"]:
            optimizer.zero_grad(set_to_none=True)
            destination = output / "legacy96_budget" / f"checkpoint{step}"
            adapter = original_lane.save_checkpoint(runner, destination)
            torch.save(optimizer.state_dict(), destination / "optimizer.pt")
            measured = original_lane.measure(runner, runner.corpus["train"], output, f"legacy96_budget/recovered/train/{step}")
            write_new(destination / "train_diagnostics.json", measured)
            checkpoints[str(step)] = {
                "adapter": adapter, "optimizer_path": str((destination / "optimizer.pt").resolve()), "optimizer_sha256": file_digest(destination / "optimizer.pt"),
                "train_diagnostics_path": str((destination / "train_diagnostics.json").resolve()), "train_diagnostics_sha256": file_digest(destination / "train_diagnostics.json"),
                "train_summary": original_lane.summarize(measured),
            }
    if original_lane.frozen_digest(runner) != initial_base or tensor_digest(adapter_parameters(runner.model, "student")) != initial_raw:
        raise ValueError("RECOVERY_MUTATED_BASE_OR_RAW_REFERENCE")
    verify_ledger(ledger, schedule, recipe["updates_per_arm"])
    write_new(output / "legacy96_budget/ledger.json", ledger)
    write_new(output / "replayed_discarded_updates.json", replay)
    return checkpoints, ledger, replay


def resume(design, upstream, original, snapshot, output):
    corpus = make_corpus(original, original_lane.SPLITS)
    schedules = original_lane.make_schedules(corpus, upstream)
    history = original_lane.verify_history(upstream, original, corpus, schedules)
    partial = inspect_partial(Path(snapshot["source_run"]), snapshot, upstream, original, corpus, schedules)
    runner = original_lane.load_runner(original, upstream, corpus, history, output)
    if runner.eos != snapshot["eos_token_id"] or runner.special_ids != snapshot["special_token_ids"]:
        raise ValueError("RECOVERY_NATIVE_TOKENIZER_IDENTITY_CHANGED")
    current_runtime = json.loads((output / "runtime.json").read_text())
    runtime_keys = ("packages", "rocm", "base_dtype", "adapter_dtypes", "trainable_parameter_count", "adapter_initial_sha256", "targeted_parameter_names")
    if any(current_runtime[key] != partial["runtime"][key] for key in runtime_keys):
        raise ValueError("RECOVERY_RUNTIME_MUST_MATCH_INTERRUPTED_RUN")
    verify_target_ledgers(runner, corpus, partial["ledgers"])
    before_base = original_lane.frozen_digest(runner)
    preflight = {}
    for arm in original_lane.ARMS:
        step = design["resume"]["checkpoint"] if arm == "legacy96_budget" else upstream["training"]["updates_per_arm"]
        preflight[arm] = reload_parity(runner, partial["checkpoints"][arm][str(step)], output, f"preflight/{arm}/{step}")
    write_new(output / "preflight.json", preflight)
    newer, legacy_ledger, replay = resume_legacy(runner, upstream, design["resume"], partial["checkpoints"]["legacy96_budget"][str(design["resume"]["checkpoint"])], partial["ledgers"]["legacy96_budget"], schedules["legacy96_budget"], output)
    arms = {}
    for arm in original_lane.ARMS:
        retained = arm != "legacy96_budget"
        ledger = partial["ledgers"][arm] if retained else legacy_ledger
        checkpoints = partial["checkpoints"][arm] if retained else {**partial["checkpoints"][arm], **newer}
        path = Path(snapshot["source_run"]) / arm / "ledger.json" if retained else output / arm / "ledger.json"
        arms[arm] = {
            "origin": "retained_complete_original_arm" if retained else "resumed_original_adapter_and_optimizer",
            "updates": len(ledger), "example_exposures": sum(len(row["ids"]) for row in ledger),
            "new_optimizer_updates": 0 if retained else len(ledger) - design["resume"]["checkpoint"],
            "retained_optimizer_updates": len(ledger) if retained else design["resume"]["checkpoint"],
            "loss_token_exposures": sum(row["loss_tokens"] for row in ledger),
            "ledger_path": str(path.resolve()), "ledger_sha256": file_digest(path), "checkpoints": checkpoints,
            "selected_checkpoint": upstream["training"]["updates_per_arm"], "learner_updates": 0,
        }
    result = {
        "status": "recovered_training_evaluation_pending", "pid": os.getpid(), "source_sha256": source_hashes(),
        "design_sha256": digest(design), "original_design_sha256": digest(upstream), "original_source_sha256": original_lane.source_hashes(),
        "interrupted_snapshot_sha256": digest(snapshot), "interrupted_execution_sha256": snapshot["execution_sha256"],
        "interrupted_training_pid": partial["original_training_pid"], "initial": partial["initial"],
        "snapshot_sha256": runner.snapshot, "dataset_sha256": digest(to_records(corpus)), "arms": arms,
        "base_before_sha256": before_base, "base_after_sha256": original_lane.frozen_digest(runner),
        "eos_token_id": runner.eos, "special_token_ids": runner.special_ids,
        "preflight_sha256": file_digest(output / "preflight.json"), "replay_sha256": file_digest(output / "replayed_discarded_updates.json"),
        "replayed_updates": len(replay), "replayed_updates_exact": sum(row["exact"] for row in replay),
        "scientific_retained_updates": sum(arm["updates"] for arm in arms.values()),
        "new_optimizer_updates": design["resume"]["new_updates"], "discarded_observed_updates": snapshot["discarded_completed_legacy_updates"],
        "confirmed_executed_updates_lower_bound": sum(snapshot["observed_updates"].values()) + design["resume"]["new_updates"],
        "unlogged_inflight_update_count": snapshot["unlogged_inflight_update_count"],
        "learner_updates": 0, "test_predictions": 0, "composition_predictions": 0,
    }
    write_new(output / "training.json", result)
    return result


def verify_recovered_training(folder, design, upstream, original, snapshot, corpus, schedules):
    folder = Path(folder)
    result = json.loads((folder / "training.json").read_text())
    partial = inspect_partial(Path(snapshot["source_run"]), snapshot, upstream, original, corpus, schedules)
    if (
        result["status"] != "recovered_training_evaluation_pending" or result["pid"] == os.getpid()
        or result["source_sha256"] != source_hashes() or result["design_sha256"] != digest(design)
        or result["original_design_sha256"] != digest(upstream) or result["original_source_sha256"] != original_lane.source_hashes()
        or result["interrupted_snapshot_sha256"] != digest(snapshot) or result["interrupted_execution_sha256"] != snapshot["execution_sha256"]
        or result["dataset_sha256"] != digest(to_records(corpus)) or result["initial"] != partial["initial"]
        or result["base_before_sha256"] != result["base_after_sha256"] or set(result["arms"]) != set(original_lane.ARMS)
        or result["new_optimizer_updates"] != design["resume"]["new_updates"]
        or any(result[key] != 0 for key in ("learner_updates", "test_predictions", "composition_predictions"))
    ):
        raise ValueError("RECOVERY_NEW_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
    for arm, report in result["arms"].items():
        retained = arm != "legacy96_budget"
        ledger = json.loads(Path(report["ledger_path"]).read_text())
        verify_ledger(ledger, schedules[arm], upstream["training"]["updates_per_arm"])
        if (
            file_digest(Path(report["ledger_path"])) != report["ledger_sha256"]
            or report["selected_checkpoint"] != upstream["training"]["updates_per_arm"] or report["updates"] != len(ledger)
            or report["example_exposures"] != sum(len(row["ids"]) for row in ledger)
            or report["loss_token_exposures"] != sum(row["loss_tokens"] for row in ledger)
            or report["new_optimizer_updates"] != (0 if retained else design["resume"]["new_updates"])
            or report["retained_optimizer_updates"] != (len(ledger) if retained else design["resume"]["checkpoint"])
            or report["origin"] != ("retained_complete_original_arm" if retained else "resumed_original_adapter_and_optimizer")
            or set(report["checkpoints"]) != {str(step) for step in upstream["training"]["checkpoints"]}
        ):
            raise ValueError("RECOVERY_ARM_ORIGIN_BUDGET_OR_LEDGER_CHANGED")
        if retained and (ledger != partial["ledgers"][arm] or report["checkpoints"] != partial["checkpoints"][arm]):
            raise ValueError("RECOVERY_COMPLETE_ORIGINAL_ARM_MUST_BE_REUSED_EXACTLY")
        if not retained and ledger[:design["resume"]["checkpoint"]] != partial["ledgers"][arm][:design["resume"]["checkpoint"]]:
            raise ValueError("RECOVERY_RESUMED_LEDGER_PREFIX_CHANGED")
        for step, checkpoint in report["checkpoints"].items():
            old = int(step) <= design["resume"]["checkpoint"] or retained
            if old and checkpoint != partial["checkpoints"][arm][step]:
                raise ValueError("RECOVERY_RETAINED_CHECKPOINT_CHANGED")
            if not old and Path(checkpoint["adapter"]["path"]).resolve() != (folder / arm / f"checkpoint{step}/teacher").resolve():
                raise ValueError("RECOVERY_NEW_CHECKPOINT_OUTSIDE_RECOVERY_RUN")
            if adapter_spec(Path(checkpoint["adapter"]["path"])) != checkpoint["adapter"] or file_digest(Path(checkpoint["optimizer_path"])) != checkpoint["optimizer_sha256"] or file_digest(Path(checkpoint["train_diagnostics_path"])) != checkpoint["train_diagnostics_sha256"]:
                raise ValueError("RECOVERY_CHECKPOINT_ARTIFACT_CHANGED")
            parameters = saved_parameters(Path(checkpoint["adapter"]["path"]) / "adapter_model.safetensors")
            inspect_optimizer(Path(checkpoint["optimizer_path"]), int(step), partial["runtime"]["targeted_parameter_names"], parameters, upstream["training"])
            records = json.loads(Path(checkpoint["train_diagnostics_path"]).read_text())
            original_lane.verify_records(corpus["train"], records, corpus, original, result["eos_token_id"], result["special_token_ids"])
            if original_lane.summarize(records) != checkpoint["train_summary"]:
                raise ValueError("RECOVERY_TRAIN_SUMMARY_CHANGED")
    preflight = json.loads((folder / "preflight.json").read_text())
    replay = json.loads((folder / "replayed_discarded_updates.json").read_text())
    if (
        file_digest(folder / "preflight.json") != result["preflight_sha256"] or set(preflight) != set(original_lane.ARMS)
        or any(not row["exact_generation_parity"] or row["rows"] != len(corpus["train"]) for row in preflight.values())
        or file_digest(folder / "replayed_discarded_updates.json") != result["replay_sha256"]
        or [row["step"] for row in replay] != list(range(design["resume"]["checkpoint"] + 1, design["resume"]["last_logged_update"] + 1))
        or any(not row["within_predeclared_tolerance"] for row in replay)
        or result["replayed_updates"] != len(replay) or result["replayed_updates_exact"] != sum(row["exact"] for row in replay)
        or result["scientific_retained_updates"] != sum(arm["updates"] for arm in result["arms"].values())
        or result["discarded_observed_updates"] != snapshot["discarded_completed_legacy_updates"]
        or result["confirmed_executed_updates_lower_bound"] != sum(snapshot["observed_updates"].values()) + design["resume"]["new_updates"]
    ):
        raise ValueError("RECOVERY_PARITY_OR_EXECUTION_ACCOUNTING_CHANGED")
    return result


def evaluate(design, dispatch, upstream, original, snapshot, output):
    folder = Path(dispatch["training_dir"])
    corpus = make_corpus(original, original_lane.SPLITS)
    schedules = original_lane.make_schedules(corpus, upstream)
    history = original_lane.verify_history(upstream, original, corpus, schedules)
    training = verify_recovered_training(folder, design, upstream, original, snapshot, corpus, schedules)
    runner = original_lane.load_runner(original, upstream, corpus, history, output)
    if runner.snapshot != training["snapshot_sha256"] or runner.eos != training["eos_token_id"] or runner.special_ids != training["special_token_ids"]:
        raise ValueError("RECOVERY_EVALUATION_RUNTIME_IDENTITY_CHANGED")
    tasks = qualification_rows(corpus, original, "fallback_validation")
    runner.model.set_adapter("student", inference_mode=True)
    raw = original_lane.measure(runner, tasks, output, "raw_learner/qualification", privileged=False, forced=False)
    original_lane.gate_feasibility([{"uid": row["uid"], "learner": row["generation"]} for row in raw], corpus, original, runner.eos, runner.special_ids)
    panels, results = {}, {}
    for arm, report in training["arms"].items():
        ledger = json.loads(Path(report["ledger_path"]).read_text())
        verify_target_ledgers(runner, corpus, {arm: ledger})
        checkpoint = report["checkpoints"][str(report["selected_checkpoint"])]
        original_lane.restore(runner, checkpoint["adapter"])
        measured = original_lane.measure(runner, tasks, output, f"{arm}/qualification")
        saved = {row["uid"]: row for row in json.loads(Path(checkpoint["train_diagnostics_path"]).read_text())}
        if any(row["generation"] != saved[row["uid"]]["generation"] for row in measured if row["split"] == "train"):
            raise ValueError(f"RECOVERY_FINAL_RELOAD_PARITY_FAILED: {arm}")
        pairs = [{"uid": task.uid, "family": task.family, "split": task.split, "learner": baseline["generation"], "teacher": teacher["generation"]} for task, baseline, teacher in zip(tasks, raw, measured, strict=True)]
        gate = qualification_gate(pairs, corpus, original, runner.eos, runner.special_ids, "fallback_validation")
        summaries = {split: original_lane.summarize([row for row in measured if row["split"] == split]) for split in ("train", "fallback_validation")}
        depth_ok = all(summaries[split][f"{family}/depth{depth}"]["accuracy"] >= upstream["evaluation"]["minimum_depth_accuracy_for_consolidation"] for split in summaries for family in FAMILIES for depth in (1, 2))
        panels[arm] = {"pairs": pairs, "diagnostics": measured}
        results[arm] = {
            "status": "qualified" if gate["passed"] else "rejected", "gate": gate, "summary": summaries,
            "checkpoint": checkpoint["adapter"], "checkpoint_update": report["selected_checkpoint"], "reload_generation_parity": True,
            "depth_accuracy_check_passed": depth_ok, "eligible_for_depth3_4_prerequisite": gate["passed"] and depth_ok,
            "origin": report["origin"], "learner_updates": 0,
        }
        runner.assert_frozen()
        if original_lane.frozen_digest(runner) != training["base_after_sha256"] or tensor_digest(adapter_parameters(runner.model, "teacher")) != checkpoint["adapter"]["tensor_sha256"]:
            raise ValueError("RECOVERY_EVALUATION_CHANGED_BASE_OR_TEACHER")
    write_new(output / "panels.json", panels)
    result = {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"], "source_sha256": source_hashes(),
        "design_sha256": digest(design), "original_design_sha256": digest(upstream), "training_sha256": file_digest(folder / "training.json"),
        "panels_sha256": file_digest(output / "panels.json"), "arms": results,
        "learner_updates": 0, "test_predictions": 0, "composition_predictions": 0,
        "interrupted_execution_sha256": snapshot["execution_sha256"], "new_optimizer_updates": training["new_optimizer_updates"],
        "claim_boundary": design["claim_boundary"],
    }
    write_new(output / "qualification.json", result)
    return result


def evaluate_child(output, dispatch):
    config = {"stage": "evaluate", "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "training_dir": str(output.resolve())}
    path = output / "evaluation_config.json"
    write_new(path, config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "device4_recovery.py"), "--config", str(path.resolve()), "--output-dir", str((output / "evaluation").resolve())], cwd=ROOT, check=True)
    result = json.loads((output / "evaluation/qualification.json").read_text())
    if result["status"] != "completed" or result["pid"] == os.getpid() or result["training_pid"] != os.getpid():
        raise ValueError("RECOVERY_NEW_PROCESS_QUALIFICATION_INCOMPLETE")
    write_new(output / "result.json", {"status": "completed", "training_pid": os.getpid(), "evaluation_pid": result["pid"], "qualification_sha256": file_digest(output / "evaluation/qualification.json"), "new_optimizer_updates": result["new_optimizer_updates"], "learner_updates": 0, "arms": result["arms"]})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [curriculum-recovery] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"run", "resume", "evaluate"}:
        raise ValueError("RECOVERY_DISPATCH_SEAL_CHANGED")
    upstream, original, snapshot = validate_design(design)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if output.resolve() == Path(snapshot["source_run"]).resolve() or any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("RECOVERY_NEW_UNUSED_OUTPUT_REQUIRED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("RECOVERY_OUTPUT_CONFIG_CHANGED")
    sources = source_hashes()
    write_new(output / "seal.json", {"design": design, "dispatch": dispatch, "source_sha256": sources, "pid": os.getpid()})
    if args.validate_only:
        write_new(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        if dispatch["stage"] == "evaluate":
            evaluate(design, dispatch, upstream, original, snapshot, output)
        else:
            resume(design, upstream, original, snapshot, output)
            if source_hashes() != sources:
                raise ValueError("RECOVERY_SOURCE_CHANGED_DURING_UPDATES")
            if dispatch["stage"] == "run":
                evaluate_child(output, dispatch)
        if source_hashes() != sources:
            raise ValueError("RECOVERY_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_new(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("RECOVERY_FAILED")
        raise


if __name__ == "__main__":
    main()
