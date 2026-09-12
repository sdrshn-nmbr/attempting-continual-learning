import hashlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from config import derived_seed, digest
from data import (
    collate,
    replay_buffer_record,
    retain_replay,
    schedule_budget,
    validate_replay_refs,
)
from learning import (
    acquisition_summary,
    activation_probe,
    apply_boundary,
    class_logits,
    evaluate,
    optimizer_for,
    restore,
    snapshot,
    trainables,
    update_diagnostics,
)

logger = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class StopRequest:
    def __init__(self):
        self.signal_number = None

    def handle(self, signum, frame):
        self.signal_number = signum


class Reporter:
    def __init__(self, output_dir, cfg):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attempt_id = uuid.uuid4().hex
        self.cfg = cfg
        self.started = time.perf_counter()

    def event(self, kind, logical_id=None, **values):
        event = {
            "event": kind,
            "at": utc_now(),
            "attempt_id": self.attempt_id,
            "run_id": self.output_dir.name,
            "logical_event_id": logical_id or kind,
            "attempt_elapsed_seconds": time.perf_counter() - self.started,
            **values,
        }
        with (self.output_dir / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            handle.flush()
        logger.info("%s %s", kind, logical_id or "")

    def preparing(self, stage, **values):
        progress = {
            "status": "preparing",
            "stage": stage,
            "updated_at": utc_now(),
            "attempt_id": self.attempt_id,
            "config": self.cfg.as_dict(),
            **values,
        }
        atomic_json(self.output_dir / "progress.json", progress)
        if not (self.output_dir / "metrics.json").exists():
            atomic_json(
                self.output_dir / "metrics.json",
                {**progress, "measured_results": False},
            )
        self.event("preparing", stage, stage=stage, **values)


class Experiment:
    def __init__(self, model, prepared, cfg, reporter, provenance, stop=None):
        self.model = model
        self.data = prepared
        self.cfg = cfg
        self.reporter = reporter
        self.stop = stop or StopRequest()
        self.initial = snapshot(model)
        self.optimizer = optimizer_for(model, cfg)
        self.signature = digest({"config": cfg.as_dict(), "provenance": provenance})
        self.provenance = provenance
        self.clock_started = time.perf_counter()
        self.prior_wall_seconds = 0.0
        self.state = {
            "method_index": 0,
            "task_index": 0,
            "step": 0,
            "task_started": False,
            "baseline_done": False,
            "total_updates": 0,
            "wall_seconds": 0.0,
            "timing_seconds": {
                "training": 0.0,
                "evaluation": 0.0,
                "diagnostics": 0.0,
                "boundary": 0.0,
            },
            "evaluation_input_tokens": 0,
            "diagnostic_input_tokens": 0,
            "methods": {},
            "baseline": {},
            "replay_buffer": [],
            "completed": False,
        }
        checkpoint = self.reporter.output_dir / "checkpoint.pt"
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["signature"] != self.signature:
                raise RuntimeError(
                    "RESUME_SIGNATURE_MISMATCH: config, code, model or data changed; use a separate output directory"
                )
            if payload["state"]["completed"]:
                raise RuntimeError("COMPLETED_RUN_IMMUTABLE: choose a unique run ID")
            self.initial = payload["initial"]
            restore(self.model, payload["adapter"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.state = payload["state"]
            self.prior_wall_seconds = self.state["wall_seconds"]
            restore_rng(payload["rng"])
            self.reporter.event(
                "resumed",
                signature=self.signature,
                cursor=self.cursor(),
                checkpoint_sha256=file_sha256(checkpoint),
            )
        else:
            self.reporter.event(
                "initialized",
                signature=self.signature,
                config=cfg.as_dict(),
                provenance=provenance,
            )

    def cursor(self):
        index = self.state["method_index"]
        return {
            "method_index": index,
            "method": self.cfg.methods[index]
            if index < len(self.cfg.methods)
            else None,
            "task_index": self.state["task_index"],
            "updates_in_task": self.state["step"],
            "total_updates": self.state["total_updates"],
            "task_started": self.state["task_started"],
        }

    def synchronize(self):
        if self.cfg.device == "cuda:0":
            torch.cuda.synchronize()

    def timed(self, category, function, *args):
        self.synchronize()
        start = time.perf_counter()
        result = function(*args)
        self.synchronize()
        self.state["timing_seconds"][category] += time.perf_counter() - start
        return result

    def evaluate(self, task, split):
        result = self.timed(
            "evaluation", evaluate, self.model, task, split, self.data, self.cfg
        )
        self.state["evaluation_input_tokens"] += result["input_tokens"]
        return result

    def probe(self, task):
        result = self.timed(
            "diagnostics", activation_probe, self.model, task, self.data, self.cfg
        )
        self.state["diagnostic_input_tokens"] += result["input_tokens"]
        return result

    def publish(self, status):
        self.state["wall_seconds"] = (
            self.prior_wall_seconds + time.perf_counter() - self.clock_started
        )
        progress = {
            "status": status,
            "updated_at": utc_now(),
            "attempt_id": self.reporter.attempt_id,
            "signature": self.signature,
            **self.cursor(),
            "wall_seconds": self.state["wall_seconds"],
        }
        metrics = {
            "status": status,
            "measured_results": bool(self.state["baseline_done"]),
            "updated_at": utc_now(),
            "config": self.cfg.as_dict(),
            "provenance": self.provenance,
            "signature": self.signature,
            **self.state,
            "comparison": self.comparison() if self.state["completed"] else None,
        }
        if self.cfg.device == "cuda:0":
            metrics["peak_gpu_memory_allocated_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            metrics["peak_gpu_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        atomic_json(self.reporter.output_dir / "metrics.json", metrics)
        atomic_json(self.reporter.output_dir / "progress.json", progress)

    def checkpoint(self, status="running"):
        self.synchronize()
        self.state["wall_seconds"] = (
            self.prior_wall_seconds + time.perf_counter() - self.clock_started
        )
        payload = {
            "signature": self.signature,
            "initial": self.initial,
            "adapter": snapshot(self.model),
            "optimizer": cpu_tree(self.optimizer.state_dict()),
            "rng": rng_state(),
            "state": self.state,
        }
        path = self.reporter.output_dir / "checkpoint.pt"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self.publish(status)
        self.reporter.event(
            "checkpoint",
            f"checkpoint:{self.state['total_updates']}",
            status=status,
            cursor=self.cursor(),
            checkpoint_sha256=file_sha256(path),
        )

    def stop_if_requested(self):
        if self.stop.signal_number is None:
            return False
        self.checkpoint("interrupted")
        self.reporter.event(
            "interrupted", signal=self.stop.signal_number, cursor=self.cursor()
        )
        return True

    def record_curve(self, record, task):
        evaluation = self.evaluate(task, "validation")
        evaluation.update(
            {
                "updates": self.state["step"],
                "training_input_tokens": record["training_input_tokens"],
                "current_examples": record["current_examples"],
                "replay_examples": record["replay_examples"],
                "training_seconds": record["training_seconds"],
            }
        )
        record["validation_curve"].append(evaluation)
        method = self.cursor()["method"]
        self.reporter.event(
            "acquisition",
            f"{method}:task{task.index}:validation:{self.state['step']}",
            **evaluation,
        )

    def start_task(self):
        task = self.data.tasks[self.state["task_index"]]
        method = self.cursor()["method"]
        if task.index == 0:
            self.state["replay_buffer"] = []
        expected_buffer = (
            self.data.replay_buffers[task.index - 1]
            if method == "balanced_replay" and task.index
            else []
        )
        if self.state["replay_buffer"] != expected_buffer:
            raise RuntimeError(f"REPLAY_BUFFER_CURSOR_MISMATCH task={task.index}")
        rng_seed = derived_seed(self.cfg.seed, "training", task.index)
        random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        before = snapshot(self.model, cpu=False)
        self.optimizer, changes = self.timed(
            "boundary",
            apply_boundary,
            self.model,
            self.optimizer,
            self.initial,
            method,
            task.index,
            self.cfg,
        )
        boundary_update = update_diagnostics(self.model, before)
        del before
        record = {
            "task": task.index,
            "training_input_tokens": 0,
            "training_padded_tokens": 0,
            "training_examples": 0,
            "current_examples": 0,
            "replay_examples": 0,
            "current_input_tokens": 0,
            "replay_input_tokens": 0,
            "planned_budget": schedule_budget(
                self.data.tasks,
                self.data.schedules[method][task.index],
                task.index,
                self.cfg,
            ),
            "training_seconds": 0.0,
            "updates": 0,
            "validation_curve": [],
            "training_curve": [],
            "boundary_changes": changes,
            "boundary_update": boundary_update,
            "post_boundary_old_test": [
                self.evaluate(old, "test") for old in self.data.tasks[: task.index]
            ],
            "initial_diagnostics": self.probe(task),
            "completed": False,
        }
        if method == "balanced_replay":
            record["replay_buffer_before"] = replay_buffer_record(
                self.data.tasks, self.state["replay_buffer"]
            )
        result = self.state["methods"].setdefault(
            method, {"tasks": [], "retention_comparable": method != "fresh_base"}
        )
        if len(result["tasks"]) != task.index:
            raise RuntimeError("TASK_CURSOR_MISMATCH")
        result["tasks"].append(record)
        self.record_curve(record, task)
        self.state["task_started"] = True
        self.reporter.event(
            "task_started",
            f"{method}:task{task.index}:start",
            task=task.index,
            method=method,
            boundary_update=boundary_update,
            replay_buffer=record.get("replay_buffer_before"),
        )
        self.checkpoint()

    def train_step(self):
        task = self.data.tasks[self.state["task_index"]]
        method = self.cursor()["method"]
        record = self.state["methods"][method]["tasks"][-1]
        refs = self.data.schedules[method][task.index][self.state["step"]]
        validate_replay_refs(refs, self.state["replay_buffer"], task.index)
        rows = [self.data.tasks[t].rows["train"][i] for t, i in refs]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        before = snapshot(self.model, cpu=False)
        self.synchronize()
        started = time.perf_counter()
        summed_loss, padded_tokens = 0.0, 0
        for start in range(0, len(rows), self.cfg.batch_size):
            batch = rows[start : start + self.cfg.batch_size]
            inputs, labels = collate(batch, self.data.pad_token_id, self.cfg.device)
            logits = class_logits(self.model, inputs, self.data.code_token_ids)
            loss = F.cross_entropy(logits, labels, reduction="sum") / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("NONFINITE_TRAINING_LOSS")
            loss.backward()
            summed_loss += float(loss.detach())
            padded_tokens += inputs["input_ids"].numel()
        parameters = list(trainables(self.model).values())
        missing = [name for name, p in trainables(self.model).items() if p.grad is None]
        if missing:
            raise RuntimeError(f"ADAPTER_GRADIENT_MISSING: {missing[:5]}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.cfg.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("NONFINITE_ADAPTER_AFTER_UPDATE")
        self.synchronize()
        elapsed = time.perf_counter() - started
        diagnostics = self.timed("diagnostics", update_diagnostics, self.model, before)
        del before
        if diagnostics["effective_adapter_update_frobenius"] == 0:
            raise RuntimeError(
                "ZERO_EFFECTIVE_ADAPTER_UPDATE: optimizer did not change the model function"
            )
        self.state["step"] += 1
        self.state["total_updates"] += 1
        self.state["timing_seconds"]["training"] += elapsed
        record["training_seconds"] += elapsed
        record["training_input_tokens"] += sum(len(row["input_ids"]) for row in rows)
        record["training_padded_tokens"] += padded_tokens
        record["training_examples"] += len(rows)
        exposure = {
            "current_examples": sum(t == task.index for t, _ in refs),
            "replay_examples": sum(t < task.index for t, _ in refs),
            "current_input_tokens": sum(
                len(row["input_ids"])
                for (t, _), row in zip(refs, rows, strict=True)
                if t == task.index
            ),
            "replay_input_tokens": sum(
                len(row["input_ids"])
                for (t, _), row in zip(refs, rows, strict=True)
                if t < task.index
            ),
        }
        for key, value in exposure.items():
            record[key] += value
        record["updates"] = self.state["step"]
        measured = {
            "updates": self.state["step"],
            "loss": summed_loss,
            "gradient_l2_before_clip": float(gradient_norm),
            "seconds": elapsed,
            "training_input_tokens": record["training_input_tokens"],
            "training_padded_tokens": record["training_padded_tokens"],
            "training_refs": refs,
            "batch_sha256": digest(refs),
            **exposure,
            **diagnostics,
        }
        record["training_curve"].append(measured)
        self.reporter.event(
            "update",
            f"{method}:task{task.index}:update{self.state['step']}",
            method=method,
            task=task.index,
            **measured,
        )

    def finish_task(self):
        method = self.cursor()["method"]
        task = self.data.tasks[self.state["task_index"]]
        results = self.state["methods"][method]
        record = results["tasks"][-1]
        if record["validation_curve"][-1]["updates"] != self.state["step"]:
            self.record_curve(record, task)
        record["test_after_task"] = [
            self.evaluate(old, "test") for old in self.data.tasks[: task.index + 1]
        ]
        record["final_diagnostics"] = self.probe(task)
        record["acquisition"] = acquisition_summary(
            record["validation_curve"], self.cfg.acquisition_threshold
        )
        record["retention"] = []
        if method != "fresh_base":
            for old_index in range(task.index):
                current = record["test_after_task"][old_index]
                acquired = results["tasks"][old_index]["test_after_task"][old_index]
                previous = results["tasks"][task.index - 1]["test_after_task"][
                    old_index
                ]
                after_boundary = record["post_boundary_old_test"][old_index]
                record["retention"].append(
                    {
                        "task": old_index,
                        "accuracy": current["accuracy"],
                        "loss": current["loss"],
                        "accuracy_at_acquisition": acquired["accuracy"],
                        "signed_forgetting": acquired["accuracy"] - current["accuracy"],
                        "forgetting_from_best_endpoint": max(
                            r["test_after_task"][old_index]["accuracy"]
                            for r in results["tasks"][old_index:-1]
                        )
                        - current["accuracy"],
                        "task_local_accuracy": current["task_local_accuracy"],
                        "task_local_signed_forgetting": acquired["task_local_accuracy"]
                        - current["task_local_accuracy"],
                        "immediate_boundary_accuracy_drop": previous["accuracy"]
                        - after_boundary["accuracy"],
                        "subsequent_training_accuracy_drop": after_boundary["accuracy"]
                        - current["accuracy"],
                    }
                )
        record["retention_summary"] = (
            {
                key: sum(item[key] for item in record["retention"])
                / len(record["retention"])
                for key in (
                    "accuracy",
                    "loss",
                    "signed_forgetting",
                    "task_local_accuracy",
                    "task_local_signed_forgetting",
                )
            }
            if record["retention"]
            else None
        )
        if method == "balanced_replay":
            retained = retain_replay(
                self.state["replay_buffer"], task, self.data.tasks, self.cfg
            )
            if retained != self.data.replay_buffers[task.index]:
                raise RuntimeError(f"REPLAY_BUFFER_PLAN_MISMATCH task={task.index}")
            self.state["replay_buffer"] = retained
            record["replay_buffer_after"] = replay_buffer_record(
                self.data.tasks, retained
            )
        record["completed"] = True
        adapter_path = (
            self.reporter.output_dir / "adapters" / method / f"task-{task.index}"
        )
        self.model.save_pretrained(
            adapter_path, safe_serialization=True, save_embedding_layers=False
        )
        record["adapter_path"] = str(adapter_path)
        record["adapter_files_sha256"] = {
            path.name: file_sha256(path)
            for path in sorted(adapter_path.iterdir())
            if path.is_file()
        }
        self.reporter.event(
            "task_completed",
            f"{method}:task{task.index}:complete",
            method=method,
            task=task.index,
            acquisition=record["acquisition"],
            retention=record["retention"],
            adapter_files_sha256=record["adapter_files_sha256"],
            replay_buffer=record.get("replay_buffer_after"),
        )
        self.state["task_index"] += 1
        self.state["step"] = 0
        self.state["task_started"] = False
        if self.state["task_index"] == self.cfg.tasks:
            self.state["method_index"] += 1
            self.state["task_index"] = 0
        self.checkpoint()

    def comparison(self):
        methods = self.state["methods"]
        budgets = {}
        for method, result in methods.items():
            budgets[method] = []
            for task in result["tasks"]:
                measured = {
                    key: task[key]
                    for key in (
                        "updates",
                        "training_examples",
                        "training_input_tokens",
                        "training_padded_tokens",
                        "current_examples",
                        "replay_examples",
                        "current_input_tokens",
                        "replay_input_tokens",
                    )
                }
                if any(
                    value != task["planned_budget"][key]
                    for key, value in measured.items()
                ):
                    raise RuntimeError(
                        f"EXPOSURE_RECEIPT_MISMATCH {method=} task={task['task']}"
                    )
                if (
                    measured["updates"] != self.cfg.updates_per_task
                    or measured["training_examples"]
                    != self.cfg.updates_per_task * self.cfg.effective_batch_size
                ):
                    raise RuntimeError(
                        f"MATCHED_BUDGET_INVARIANT_FAILED {method=} task={task['task']}"
                    )
                budgets[method].append({**task["planned_budget"], **measured})
        gaps = {}
        if "fresh_base" in methods:
            for method, result in methods.items():
                if method == "fresh_base":
                    continue
                gaps[method] = []
                for task_index, task in enumerate(result["tasks"]):
                    fresh = methods["fresh_base"]["tasks"][task_index]
                    gaps[method].append(
                        {
                            "task": task_index,
                            "fresh_minus_sequential_test_accuracy": fresh[
                                "test_after_task"
                            ][task_index]["accuracy"]
                            - task["test_after_task"][task_index]["accuracy"],
                            "fresh_minus_sequential_test_task_local_accuracy": fresh[
                                "test_after_task"
                            ][task_index]["task_local_accuracy"]
                            - task["test_after_task"][task_index][
                                "task_local_accuracy"
                            ],
                            "fresh_minus_sequential_validation_accuracy_auc": fresh[
                                "acquisition"
                            ]["accuracy_auc_per_update"]
                            - task["acquisition"]["accuracy_auc_per_update"],
                            "fresh_minus_sequential_validation_task_local_accuracy_auc": fresh[
                                "acquisition"
                            ]["task_local_accuracy_auc_per_update"]
                            - task["acquisition"]["task_local_accuracy_auc_per_update"],
                        }
                    )
        return {
            "matched_update_and_example_budgets_verified": True,
            "matching": "Shared initial LoRA, frozen base, optimizer settings, effective batch and evaluation data. Equal updates and total example exposures. Replay substitutes previous-task examples; current-task exposures and input/padded token counts are not matched.",
            "budgets": budgets,
            "calibration_gaps": gaps,
            "replay_screen": replay_screen(methods, self.cfg)
            if set(methods) == {"persistent", "balanced_replay"}
            else None,
        }

    def run(self):
        if self.state["completed"]:
            return 0
        if self.stop_if_requested():
            return 128 + self.stop.signal_number
        if not self.state["baseline_done"]:
            self.state["baseline"] = {
                "description": "Frozen pretrained model with zero-output LoRA and arbitrary unlearned label-code mapping",
                "test": [self.evaluate(task, "test") for task in self.data.tasks],
            }
            self.state["baseline_done"] = True
            self.checkpoint()
        while self.state["method_index"] < len(self.cfg.methods):
            if self.stop_if_requested():
                return 128 + self.stop.signal_number
            if not self.state["task_started"]:
                self.start_task()
            if self.stop_if_requested():
                return 128 + self.stop.signal_number
            task = self.data.tasks[self.state["task_index"]]
            record = self.state["methods"][self.cursor()["method"]]["tasks"][-1]
            while self.state["step"] < self.cfg.updates_per_task:
                self.train_step()
                if self.state["step"] % self.cfg.eval_every == 0:
                    self.record_curve(record, task)
                if self.stop_if_requested():
                    return 128 + self.stop.signal_number
                if self.state["step"] % self.cfg.checkpoint_every == 0:
                    self.checkpoint()
            self.finish_task()
        self.comparison()
        self.state["completed"] = True
        self.checkpoint("completed")
        self.reporter.event(
            "completed",
            signature=self.signature,
            total_updates=self.state["total_updates"],
        )
        return 0


def replay_screen(methods, cfg):
    endpoints = {
        method: [
            task["test_after_task"][i]["accuracy"]
            for i, task in enumerate(result["tasks"])
        ]
        for method, result in methods.items()
    }
    final = {
        method: result["tasks"][-1]["test_after_task"]
        for method, result in methods.items()
    }
    old = {
        method: {
            "correct": sum(task["correct"] for task in scores[:-1]),
            "examples": sum(task["examples"] for task in scores[:-1]),
        }
        for method, scores in final.items()
    }
    for result in old.values():
        result["global_accuracy"] = result["correct"] / result["examples"]
    gain = (
        old["balanced_replay"]["global_accuracy"] - old["persistent"]["global_accuracy"]
    )
    acquisition_drops = [
        a - b
        for a, b in zip(
            endpoints["persistent"], endpoints["balanced_replay"], strict=True
        )
    ]
    acquisition_ok = all(
        a >= cfg.acquisition_floor for values in endpoints.values() for a in values
    )
    noninferiority_ok = all(
        drop <= cfg.max_acquisition_drop + 1e-12 for drop in acquisition_drops
    )
    retention_ok = gain + 1e-12 >= cfg.min_retention_gain
    if acquisition_ok and noninferiority_ok and retention_ok:
        decision = "keep_for_confirmation"
        next_action = "Run the predeclared confirmation seed with the same settings; reassess acquisition and retention before further expansion."
    elif not acquisition_ok or not noninferiority_ok:
        decision = "redesign_acquisition_replay_tradeoff"
        next_action = "Inspect fixed acquisition curves and exposure counts; predeclare a new mixture or learning schedule, then screen it. Continue the campaign."
    else:
        decision = "redesign_retention_mechanism"
        next_action = "Replay did not clear the retention effect floor. Use errors and global-versus-oracle diagnostics to choose the next interference hypothesis; continue the campaign without reviving the reset matrix."
    return {
        "primary_scoring": "global unknown-task accuracy over all selected codes, including future codes; true-task-restricted scores are privileged diagnostics only",
        "acquisition_floor": cfg.acquisition_floor,
        "max_acquisition_drop": cfg.max_acquisition_drop,
        "min_retention_gain": cfg.min_retention_gain,
        "global_acquisition_by_task": endpoints,
        "persistent_minus_replay_acquisition_by_task": acquisition_drops,
        "final_old_test": old,
        "replay_minus_persistent_final_old_global_accuracy": gain,
        "acquisition_floor_passed": acquisition_ok,
        "acquisition_drop_passed": noninferiority_ok,
        "retention_gain_passed": retention_ok,
        "decision": decision,
        "next_action": next_action,
        "inference_limit": "A fixed-seed screen, not a significance or equivalence test. Example-level scores do not establish robustness across training seeds, task orders, or model families.",
    }
