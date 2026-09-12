import argparse
import fcntl
import hashlib
import json
import logging
import os
import signal
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from config import (
    CALIBRATION_PROTOCOL,
    ORACLE_CONTROL_PROTOCOL,
    PROTOCOL,
    TRAIN_ANSWER_CALIBRATION_PROTOCOL,
    read_config,
)
from model import ModelIO, StopFlag, StopRequested, load_model, policy_mode
from objectives import dense_reverse_kl, oracle_nll, response_logits
from tasks import (
    CALIBRATION_DEV_START,
    CALIBRATION_VARIANTS,
    TASKS,
    build_calibration_data,
    build_data,
    build_oracle_eval_data,
    build_oracle_train_data,
    build_train_answer_data,
    data_manifest,
    grade,
    oracle_batch,
    oracle_train_manifest,
    prompt,
    stable_seed,
)

LOGGER = logging.getLogger("onpolicy")
RUNTIME_OUTPUT_FILES = {
    "run.lock",
    "task.json",
    "execution.json",
    "attempts",
    "run.log",
    "packages.txt",
    "config.json",
}


class OutputConflict(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def source_hash():
    digest = hashlib.sha256()
    for name in ("run.py", "config.py", "model.py", "objectives.py", "tasks.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def adapter_state(model):
    return cpu_tree(get_peft_model_state_dict(model, save_embedding_layers=False))


def suites(data, split):
    result = {}
    for task in TASKS:
        result[f"{task}/inputs"] = data[task][split]
        result[f"{task}/compositions"] = data[task][f"{split}_compositions"]
    result["cross_task/compositions"] = data["cross_task"][split]
    return result


def generation_aggregate(rows):
    n = len(rows)
    return {
        "n": n,
        "correct": sum(x["correct"] for x in rows),
        "accuracy": sum(x["correct"] for x in rows) / n,
        "invalid_rate": sum(x["invalid"] for x in rows) / n,
        "refusal_rate": sum(x["refusal"] for x in rows) / n,
        "truncated_rate": sum(x["truncated"] for x in rows) / n,
    }


def aggregate(rows):
    n = len(rows)
    tokens = sum(x["gold_answer_tokens"] for x in rows)
    return {
        **generation_aggregate(rows),
        "mean_gold_answer_logprob": sum(x["gold_answer_logprob"] for x in rows) / n,
        "mean_gold_complete_logprob": sum(x["gold_complete_logprob"] for x in rows) / n,
        "mean_gold_token_logprob": sum(x["gold_answer_logprob"] for x in rows) / tokens,
    }


def select_calibration_variant(measurements, threshold):
    if set(measurements) != set(CALIBRATION_VARIANTS) or any(
        not rows for rows in measurements.values()
    ):
        raise ValueError("Calibration selection requires both fixed variants with nonempty suites")
    for variant in CALIBRATION_VARIANTS:
        if all(row["accuracy"] >= threshold for row in measurements[variant].values()):
            return variant
    return None


def qualifies_train_answer(measurements):
    names = {f"{task}/{kind}" for task in TASKS for kind in ("inputs", "compositions")}
    names.add("cross_task/compositions")
    return set(measurements) == names and all(
        row["n"] == 16 and row["accuracy"] == 1.0 and row["truncated_rate"] == 0
        for row in measurements.values()
    )


def oracle_control_gates(evaluations):
    acquisition = {}
    for task in TASKS:
        initial = evaluations["initial"][f"{task}/inputs"]["accuracy"]
        acquired = evaluations[f"after_{task}"][f"{task}/inputs"]["accuracy"]
        acquisition[task] = {
            "initial_accuracy": initial,
            "after_task_accuracy": acquired,
            "gain_over_initial": acquired - initial,
            "passed": acquired >= 0.75 and acquired - initial >= 0.25,
        }
    eligible = all(row["passed"] for row in acquisition.values())
    drop = (
        evaluations["after_permutation"]["permutation/inputs"]["accuracy"]
        - evaluations["after_symbol_map"]["permutation/inputs"]["accuracy"]
    )
    return {
        "acquisition": acquisition,
        "retention": {
            "eligible": eligible,
            "condition": "Both primitive acquisition gates pass",
            "accuracy_drop": drop,
            "maximum_drop": 0.10,
            "passed": drop <= 0.10 if eligible else None,
        },
        "feasible": eligible and drop <= 0.10,
    }


def oracle_training_allocation(updates, replay_examples):
    allocation = {}
    for task in TASKS:
        old_count = replay_examples if task == TASKS[1] else 0
        counts = Counter(
            example_id.split("/")[0]
            for row in updates if row["task"] == task
            for example_id in row["example_ids"]
        )
        expected = {task: 32 * (4 - old_count)}
        if old_count:
            expected[TASKS[0]] = 32 * old_count
        if counts != expected:
            raise RuntimeError(f"Oracle training allocation mismatch for {task}: {counts}; expected {expected}")
        current = counts[task]
        replay = sum(count for source, count in counts.items() if source != task)
        allocation[task] = {
            "current_complete_answers": current,
            "replay_complete_answers": replay,
            "current_supervised_tokens": current * 10,
            "replay_supervised_tokens": replay * 10,
            "current_exposure_fraction_of_no_replay": current / 128,
        }
    return allocation


def comparisons(evaluations):
    result = {}
    before, after_a, after_b = (
        evaluations[stage] for stage in ("before", "after_permutation", "after_symbol_map")
    )
    for key in before:
        a, b, final = before[key], after_a[key], after_b[key]
        row = {
            "final_minus_before_accuracy": final["accuracy"] - a["accuracy"],
            "final_minus_before_gold_logprob": final["mean_gold_answer_logprob"]
            - a["mean_gold_answer_logprob"],
        }
        if key.startswith("permutation/"):
            row["acquisition_accuracy_delta"] = b["accuracy"] - a["accuracy"]
            row["retention_accuracy_delta"] = final["accuracy"] - b["accuracy"]
            row["forgetting_accuracy_drop"] = b["accuracy"] - final["accuracy"]
            row["retention_gold_logprob_delta"] = (
                final["mean_gold_answer_logprob"] - b["mean_gold_answer_logprob"]
            )
        if key.startswith("symbol_map/"):
            row["forward_transfer_accuracy_delta"] = b["accuracy"] - a["accuracy"]
            row["acquisition_accuracy_delta"] = final["accuracy"] - b["accuracy"]
            row["forward_transfer_gold_logprob_delta"] = (
                b["mean_gold_answer_logprob"] - a["mean_gold_answer_logprob"]
            )
        result[key] = row
    return result


class Experiment:
    def __init__(self, config, output, stop):
        self.config, self.output, self.stop = config, output, stop
        self.calibration = config.mode in ("teacher_calibration", "train_answer_calibration")
        if config.mode == "oracle_sft_control":
            self.data = build_oracle_train_data(config.seed)
            protocol = ORACLE_CONTROL_PROTOCOL
        elif config.mode == "train_answer_calibration":
            self.data = build_train_answer_data(config.seed)
            protocol = TRAIN_ANSWER_CALIBRATION_PROTOCOL
        elif config.mode == "teacher_calibration":
            self.data = build_calibration_data(config.seed, config.gate_examples)
            protocol = CALIBRATION_PROTOCOL
        else:
            self.data = build_data(
                config.seed, config.train_examples, config.eval_examples, config.gate_examples
            )
            protocol = PROTOCOL
        if config.mode == "oracle_sft_control":
            self.manifest = oracle_train_manifest(self.data, config.seed, config.replay_examples_per_update)
        else:
            self.manifest = data_manifest(self.data, config.seed, config.steps_per_task)
        self.code_sha = source_hash()
        self.model = self.io = self.optimizer = self.initial_adapter = None
        self.current_arm = None
        self.next_step = 0
        self.metrics = {
            "schema_version": 1,
            "status": "starting",
            "created_at": now(),
            "config": config.to_dict(),
            "config_sha256": config.sha256,
            "source_sha256": self.code_sha,
            "data_sha256": self.manifest["sha256"],
            "protocol": protocol,
            "arms": {},
            "reference": {},
            "evidence": {"kind": config.experiment_kind, "gpu_training_observed": False},
        }
        if config.mode == "oracle_sft_control":
            self.metrics["oracle_sft_control"] = {
                "phase": "training", "updates": [], "saved_adapters": {}, "evaluations": {}
            }

    def event(self, event, **fields):
        record = {"at": now(), "event": event, **fields}
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
        if event != "evaluation_example":
            LOGGER.info("%s %s", event, json.dumps(fields, allow_nan=False))

    def progress(self, phase, **fields):
        atomic_json(
            self.output / "progress.json",
            {
                "at": now(),
                "status": self.metrics["status"],
                "phase": phase,
                "arm": self.current_arm,
                "completed_updates_in_arm": self.next_step,
                "total_updates_per_arm": len(TASKS) * self.config.steps_per_task,
                **fields,
            },
        )

    def write_metrics(self):
        self.metrics["updated_at"] = now()
        atomic_json(self.output / "metrics.json", self.metrics)

    def optimizer_for_model(self):
        if self.calibration:
            raise RuntimeError("Teacher calibration cannot construct an optimizer")
        if (
            self.config.mode == "oracle_sft_control"
            and self.metrics["oracle_sft_control"]["phase"] != "training"
        ):
            raise RuntimeError("Oracle control cannot construct an optimizer after training")
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.0,
            foreach=False,
        )

    def save_checkpoint(self):
        self.write_metrics()
        if self.model is None:
            return
        directory = self.output / "checkpoint"
        directory.mkdir(exist_ok=True)
        state = {
            "config_sha256": self.config.sha256,
            "source_sha256": self.code_sha,
            "data_sha256": self.manifest["sha256"],
            "metrics": self.metrics,
            "adapter": None if self.calibration else adapter_state(self.model),
            "initial_adapter": None if self.calibration else self.initial_adapter,
            "optimizer": cpu_tree(self.optimizer.state_dict())
            if self.optimizer is not None
            else None,
            "arm": self.current_arm,
            "next_step": self.next_step,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all()
            if self.config.device.startswith("cuda")
            else [],
        }
        temporary = directory / "state.pt.tmp"
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(directory / "state.pt")

    def save_adapter(self, label):
        directory = self.output / "adapters" / label
        directory.mkdir(parents=True, exist_ok=True)
        state = {name: tensor.contiguous() for name, tensor in adapter_state(self.model).items()}
        temporary = directory / "adapter_model.safetensors.tmp"
        save_file(state, str(temporary))
        temporary.replace(directory / "adapter_model.safetensors")
        self.model.peft_config["default"].save_pretrained(directory)
        atomic_json(
            directory / "provenance.json",
            {
                "model_id": self.config.model_id,
                "declared_revision": self.config.revision,
                "config_sha256": self.config.sha256,
                "data_sha256": self.manifest["sha256"],
                "arm": self.current_arm,
                "completed_updates": self.next_step,
            },
        )

    def prepare(self):
        existing = self.output / "metrics.json"
        checkpoint = self.output / "checkpoint" / "state.pt"
        state = None
        if existing.exists():
            try:
                previous = json.loads(existing.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise OutputConflict(f"Cannot read existing run metrics: {error}") from error
            for key in ("config_sha256", "source_sha256", "data_sha256"):
                if previous.get(key) != self.metrics[key]:
                    raise OutputConflict(f"Refusing to resume mismatched {key}")
            self.metrics = previous
            if previous["status"] in (
                "completed",
                "teacher_gate_failed",
                "teacher_calibration_failed",
                "train_answer_calibration_passed",
                "train_answer_calibration_failed",
            ):
                return False
            if not checkpoint.exists() and previous.get("arms"):
                raise OutputConflict("Nonempty training results have no recoverable checkpoint")
            if (
                self.config.mode == "oracle_sft_control"
                and not checkpoint.exists()
                and (
                    previous["oracle_sft_control"]["updates"]
                    or previous["oracle_sft_control"]["phase"] != "training"
                )
            ):
                raise OutputConflict("Oracle control results have no recoverable checkpoint")
        elif any(path.name not in RUNTIME_OUTPUT_FILES for path in self.output.iterdir()):
            raise OutputConflict(
                "Output directory contains artifacts without identifiable run metrics"
            )
        if checkpoint.exists():
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except Exception as error:
                raise OutputConflict(f"Cannot read checkpoint: {error}") from error
            for key in ("config_sha256", "source_sha256", "data_sha256"):
                if state.get(key) != self.metrics[key]:
                    raise OutputConflict(f"Checkpoint does not match {key}")
            if self.calibration and (
                state["arm"] is not None
                or state["next_step"] != 0
                or state["optimizer"] is not None
                or state["adapter"] is not None
                or state["initial_adapter"] is not None
                or state["metrics"]["arms"]
            ):
                raise OutputConflict(
                    "Teacher-calibration checkpoint contains forbidden training state"
                )
            if self.config.mode == "oracle_sft_control":
                self.check_oracle_checkpoint(state)
            self.metrics = state["metrics"]
        model_config_sha = hashlib.sha256(
            (Path(self.config.model_path) / "config.json").read_bytes()
        ).hexdigest()
        if (
            self.metrics.get("model_provenance", {}).get("config_sha256", model_config_sha)
            != model_config_sha
        ):
            raise OutputConflict("Cached model configuration changed since the checkpoint")
        self.metrics["status"] = "running"
        self.write_metrics()
        atomic_json(self.output / "data_manifest.json", self.manifest)
        self.progress("loading_model")
        self.event(
            "model_loading",
            model_id=self.config.model_id,
            revision=self.config.revision,
            path=self.config.model_path,
        )
        self.model, tokenizer, runtime = load_model(self.config)
        self.initial_adapter = adapter_state(self.model)
        self.io = ModelIO(self.model, tokenizer, self.config, self.data, self.stop)
        self.metrics["runtime"] = runtime
        self.metrics["runtime"]["termination"] = {
            "model_eos_token_ids": list(self.io.model_eos_ids),
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "effective_eos_token_ids": list(self.io.eos_ids),
            "pad_token_id": self.io.pad_id,
        }
        self.metrics["model_provenance"] = {
            "model_id": self.config.model_id,
            "declared_revision": self.config.revision,
            "model_path": self.config.model_path,
            "config_sha256": model_config_sha,
            "weight_revision_verification": "Uses caller-verified local cache; weights are not downloaded or rehashed by this runner.",
        }
        if state is not None:
            if not self.calibration:
                self.initial_adapter = state["initial_adapter"]
                set_peft_model_state_dict(self.model, state["adapter"])
            self.current_arm, self.next_step = state["arm"], state["next_step"]
            if state["optimizer"] is not None:
                self.optimizer = self.optimizer_for_model()
                self.optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng_cpu"])
            if state["rng_cuda"]:
                torch.cuda.set_rng_state_all(state["rng_cuda"])
            self.event("checkpoint_resumed", arm=self.current_arm, next_step=self.next_step)
        else:
            if not self.calibration:
                if self.config.mode == "oracle_sft_control":
                    self.save_oracle_adapter("initial")
                else:
                    self.save_adapter("initial")
            self.save_checkpoint()
        self.event("model_ready", **runtime)
        self.stop.check()
        return True

    def evaluate_example(
        self, example, phase, name, context, privileged, instruction_variant="current"
    ):
        self.stop.check()
        if self.config.mode == "oracle_sft_control" and (
            self.metrics["oracle_sft_control"]["phase"] != "post_training"
            or self.next_step != 64
            or self.optimizer is not None
            or privileged
            or context != "cue_only"
            or not phase.startswith("post_training/")
            or instruction_variant != "current"
            or example.split not in ("heldout", "heldout_compositions")
            or example not in self.data.get(example.task, {}).get(example.split, [])
        ):
            raise RuntimeError("Oracle control evaluation requires the sealed post-training phase")
        if self.config.mode == "train_answer_calibration" and (
            not privileged
            or context != "train_answer"
            or phase != "train_answer_calibration"
            or instruction_variant != "current"
            or example.split not in ("train", "train_compositions")
            or example not in self.data.get(example.task, {}).get(example.split, [])
        ):
            raise RuntimeError(
                "Train-answer calibration may generate only its fixed train schedule"
            )
        if self.config.mode == "teacher_calibration" and (
            not privileged
            or context != "examples"
            or example.split not in ("gate", "gate_compositions")
            or example not in self.data[example.task][example.split]
        ):
            raise RuntimeError("Teacher calibration may score only its frozen dev examples")
        prefix = self.io.prefix(example, context, privileged, instruction_variant)
        response, _ = self.io.generate(
            prefix,
            privileged,
            False,
            stable_seed(self.config.seed, "eval", example.key),
            self.config.max_new_tokens,
        )
        text = self.io.decode(response)
        row = {
            "phase": phase,
            "arm": self.current_arm,
            "suite": name,
            "context": context,
            "privileged": privileged,
            "instruction_variant": instruction_variant,
            "split": example.split,
            "example_id": example.key,
            "prompt_sha256": hashlib.sha256(
                prompt(example, self.data, context, privileged, instruction_variant).encode()
            ).hexdigest(),
            "output": text,
            "expected": example.answer,
            **grade(text, example),
            "generated_tokens": len(response),
            "generated_token_ids": response.tolist(),
            "truncated": int(response[-1]) not in self.io.eos_ids,
        }
        if self.config.mode == "train_answer_calibration":
            row["student_prompt_sha256"] = hashlib.sha256(
                prompt(example, self.data, "train_answer", False).encode()
            ).hexdigest()
        elif self.config.mode != "oracle_sft_control":
            row.update(self.io.gold_logprob(prefix, example, privileged))
        return row

    def evaluate(self, phase, privileged=False, split="heldout", contexts=("cue_only", "examples")):
        if self.calibration:
            raise RuntimeError("Calibration must use its dedicated fixed-schedule generation path")
        results = {}
        for context in contexts:
            for name, examples in suites(self.data, split).items():
                rows = []
                for index, example in enumerate(examples):
                    self.stop.check()
                    self.progress(
                        phase, suite=name, context=context, item=index, items=len(examples)
                    )
                    row = self.evaluate_example(example, phase, name, context, privileged)
                    rows.append(row)
                    self.event("evaluation_example", **row)
                results[f"{name}/{context}"] = aggregate(rows)
                self.event(
                    "evaluation_suite",
                    phase=phase,
                    suite=name,
                    context=context,
                    **results[f"{name}/{context}"],
                )
        return results

    def collect_generation_records(self, result, schedule, context, privileged, phase):
        records = result["records"]
        if len(records) > len(schedule):
            raise OutputConflict("Calibration checkpoint exceeds its fixed call budget")
        for index, (name, example, variant) in enumerate(schedule):
            self.stop.check()
            if index < len(records):
                saved = records[index]
                if (
                    saved["suite"], saved["example_id"], saved["instruction_variant"],
                    saved["context"], saved["privileged"], saved["phase"],
                ) != (name, example.key, variant, context, privileged, phase):
                    raise OutputConflict(
                        "Calibration checkpoint does not match the fixed generation schedule"
                    )
                continue
            self.progress(
                phase,
                suite=name,
                instruction_variant=variant,
                completed_generation_calls=index,
                expected_generation_calls=len(schedule),
            )
            row = self.evaluate_example(
                example,
                phase,
                name,
                context,
                privileged,
                variant,
            )
            records.append(row)
            self.event("evaluation_example", **row)
            self.save_checkpoint()
        return records

    def check_calibration_budget(self, records, expected_calls):
        if (
            self.optimizer is not None
            or self.next_step
            or self.current_arm is not None
            or self.metrics["arms"]
            or len(records) != expected_calls
        ):
            raise RuntimeError("Calibration violated its generation-only budget")
        if any(
            not torch.equal(value, self.initial_adapter[key])
            for key, value in adapter_state(self.model).items()
        ):
            raise RuntimeError("Calibration changed adapter parameters")
        self.metrics["budget_check"] = {
            "passed": True,
            "optimizer_updates": 0,
            "supervised_tokens": 0,
            "expected_generation_calls": expected_calls,
            "generation_calls": len(records),
            "primary_heldout_calls": 0,
            "adapter_parameters_unchanged": True,
        }

    def train_answer_calibration(self):
        result = self.metrics.setdefault(
            "train_answer_calibration",
            {
                "train_input_range": [4, 20],
                "threshold": 1.0,
                "records": [],
            },
        )
        train_suites = suites(self.data, "train")
        schedule = [
            (name, example, "current")
            for name, examples in train_suites.items()
            for example in examples
        ]
        if len(schedule) != 80:
            raise RuntimeError("Train-answer calibration requires exactly 80 teacher calls")
        records = self.collect_generation_records(
            result, schedule, "train_answer", True, self.config.mode
        )
        measurements = {
            name: generation_aggregate([row for row in records if row["suite"] == name])
            for name in train_suites
        }
        result["measurements"] = measurements
        result["qualified"] = qualifies_train_answer(measurements)
        self.check_calibration_budget(records, 80)
        self.metrics["budget_check"].update(dev_calls=0, student_generation_calls=0)
        self.metrics["status"] = (
            "train_answer_calibration_passed"
            if result["qualified"]
            else "train_answer_calibration_failed"
        )
        self.metrics["completed_at"] = now()
        self.save_checkpoint()
        self.progress(
            "train_answer_calibration_complete",
            qualified=result["qualified"],
            completed_generation_calls=len(records),
        )
        self.event(
            "train_answer_calibration_complete",
            qualified=result["qualified"],
            budget=self.metrics["budget_check"],
            measurements=measurements,
        )
        return 0 if result["qualified"] else 2

    def teacher_calibration(self):
        result = self.metrics.setdefault(
            "teacher_calibration",
            {
                "dev_input_start": CALIBRATION_DEV_START,
                "threshold": self.config.teacher_min_accuracy,
                "variant_order": list(CALIBRATION_VARIANTS),
                "records": [],
            },
        )
        schedule = [
            (name, example, variant)
            for name, examples in suites(self.data, "gate").items()
            for example in examples
            for variant in CALIBRATION_VARIANTS
        ]
        records = self.collect_generation_records(
            result, schedule, "examples", True, self.config.mode
        )
        measurements = {
            variant: {
                name: aggregate(
                    [
                        row
                        for row in records
                        if row["suite"] == name and row["instruction_variant"] == variant
                    ]
                )
                for name in suites(self.data, "gate")
            }
            for variant in CALIBRATION_VARIANTS
        }
        result["variant_results"] = {
            variant: {
                "qualified": all(
                    row["accuracy"] >= self.config.teacher_min_accuracy for row in rows.values()
                ),
                "measurements": rows,
            }
            for variant, rows in measurements.items()
        }
        result["paired"] = {}
        for name in suites(self.data, "gate"):
            current = [
                row
                for row in records
                if row["suite"] == name and row["instruction_variant"] == "current"
            ]
            numbered = [
                row
                for row in records
                if row["suite"] == name and row["instruction_variant"] == "numbered"
            ]
            pairs = list(zip(current, numbered, strict=True))
            result["paired"][name] = {
                "n": len(pairs),
                "current_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
                "numbered_only_correct": sum(b["correct"] and not a["correct"] for a, b in pairs),
                "both_correct": sum(a["correct"] and b["correct"] for a, b in pairs),
                "neither_correct": sum(not a["correct"] and not b["correct"] for a, b in pairs),
                "numbered_minus_current_accuracy": measurements["numbered"][name]["accuracy"]
                - measurements["current"][name]["accuracy"],
                "numbered_minus_current_gold_logprob": measurements["numbered"][name][
                    "mean_gold_answer_logprob"
                ]
                - measurements["current"][name]["mean_gold_answer_logprob"],
            }
        selected = select_calibration_variant(measurements, self.config.teacher_min_accuracy)
        result["selected_variant"] = selected
        self.check_calibration_budget(records, len(schedule))
        self.metrics["status"] = (
            "completed" if selected is not None else "teacher_calibration_failed"
        )
        self.metrics["completed_at"] = now()
        self.save_checkpoint()
        self.progress(
            "teacher_calibration_complete",
            selected_variant=selected,
            completed_generation_calls=len(records),
        )
        self.event(
            "teacher_calibration_complete",
            selected_variant=selected,
            budget=self.metrics["budget_check"],
            variant_results=result["variant_results"],
        )
        return 0 if selected is not None else 2

    def check_oracle_checkpoint(self, state):
        result = state["metrics"]["oracle_sft_control"]
        phase = result["phase"]
        if (
            phase not in ("training", "post_training", "completed")
            or not 0 <= state["next_step"] <= 64
            or len(result["updates"]) != state["next_step"]
            or [row["global_step"] for row in result["updates"]]
            != list(range(1, state["next_step"] + 1))
            or any(
                row["supervised_tokens"] != 40 or row["complete_answers"] != 4
                for row in result["updates"]
            )
        ):
            raise OutputConflict("Oracle checkpoint violates its fixed update schedule")
        schedule = [
            (task, step + 1, batch)
            for task in TASKS
            for step, batch in enumerate(self.manifest["training_schedule"][task])
        ]
        if any(
            (row["task"], row["step_in_task"], row["example_ids"]) != expected
            for row, expected in zip(result["updates"], schedule)
        ):
            raise OutputConflict("Oracle checkpoint violates its fixed train-only schedule")
        if phase == "training":
            if state["next_step"] and (
                state["optimizer"] is None or state["arm"] != "oracle_sft"
            ):
                raise OutputConflict("Oracle training checkpoint lost its persistent optimizer")
            if (
                result["evaluations"]
                or (self.output / "eval_manifest.json").exists()
                or self.metrics["oracle_sft_control"]["phase"] != "training"
            ):
                raise OutputConflict("Oracle control cannot resume training after heldout access")
        elif (
            state["next_step"] != 64
            or state["optimizer"] is not None
            or set(result["saved_adapters"])
            != {"initial", "after_permutation", "after_symbol_map"}
        ):
            raise OutputConflict("Oracle post-training checkpoint is not sealed")

    def save_oracle_adapter(self, label):
        self.save_adapter(label)
        path = self.output / "adapters" / label / "adapter_model.safetensors"
        self.metrics["oracle_sft_control"]["saved_adapters"][label] = {
            "path": str(path.relative_to(self.output)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "completed_updates": self.next_step,
        }

    def oracle_sft_update(self, task, step):
        if (
            self.metrics["oracle_sft_control"]["phase"] != "training"
            or self.optimizer is None
            or not 0 <= self.next_step < 64
            or (TASKS[self.next_step // 32], self.next_step % 32) != (task, step)
        ):
            raise RuntimeError("Oracle update requires its fixed unfinished training phase")
        examples = oracle_batch(self.data, task, step, self.config.replay_examples_per_update)
        record = {
            "task": task, "step_in_task": step + 1, "global_step": self.next_step + 1,
            "example_ids": [example.key for example in examples],
            "supervised_tokens": 0, "complete_answers": 0, "generated_tokens": 0,
            "teacher_forward_input_tokens": 0, "student_forward_input_tokens": 0,
            "loss_sum": 0.0,
        }
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for example in examples:
            self.stop.check()
            prefix = self.io.prefix(example, "cue_only", False)
            completion = self.io.oracle_tokens(example)
            if (
                len(completion) != 10
                or int(completion[-1]) != self.io.tokenizer.eos_token_id
                or any(token in self.io.eos_ids for token in completion[:-1].tolist())
            ):
                raise RuntimeError(f"Oracle answer must contain exactly 9 tokens plus EOS: {example.key}")
            logits = response_logits(self.model, prefix, completion)
            loss_sum = oracle_nll(logits, completion, torch.ones_like(completion, dtype=torch.bool))
            if not torch.isfinite(loss_sum):
                raise FloatingPointError(f"Non-finite oracle loss: {task}/{step}/{example.key}")
            (loss_sum / 40).backward()
            record["loss_sum"] += float(loss_sum.detach())
            record["supervised_tokens"] += len(completion)
            record["complete_answers"] += 1
            record["student_forward_input_tokens"] += len(prefix) + len(completion)
            self.event(
                "oracle_training_example",
                global_step=record["global_step"],
                example_id=example.key,
                context="cue_only",
                privileged=False,
                prompt_sha256=hashlib.sha256(prompt(example, self.data, "cue_only").encode()).hexdigest(),
                supervised_token_ids=completion.tolist(),
                complete_answer=True,
            )
            del logits, loss_sum
        self.stop.check()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.config.max_grad_norm, error_if_nonfinite=True
        )
        if not any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("Oracle adapter receives no gradient")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        record["grad_norm"] = float(norm)
        record["loss_per_token"] = record["loss_sum"] / 40
        return record

    def oracle_sft_control(self):
        result = self.metrics["oracle_sft_control"]
        if result["phase"] == "training":
            if self.optimizer is None:
                self.current_arm = "oracle_sft"
                self.optimizer = self.optimizer_for_model()
                self.save_checkpoint()
            while self.next_step < 64:
                self.stop.check()
                task_index, step = divmod(self.next_step, 32)
                task = TASKS[task_index]
                self.progress("oracle_training", task=task, step_in_task=step)
                row = self.oracle_sft_update(task, step)
                result["updates"].append(row)
                self.next_step += 1
                self.metrics["evidence"]["gpu_training_observed"] = self.config.device.startswith("cuda")
                self.event("optimizer_update", arm="oracle_sft", **row)
                if self.next_step % 32 == 0:
                    self.save_oracle_adapter(f"after_{task}")
                self.save_checkpoint()
            result["training_allocation"] = oracle_training_allocation(
                result["updates"], self.config.replay_examples_per_update
            )
            self.metrics["budget_check"] = {
                "passed": True,
                "optimizer_updates": len(result["updates"]),
                "supervised_tokens": sum(row["supervised_tokens"] for row in result["updates"]),
                "complete_answers": sum(row["complete_answers"] for row in result["updates"]),
                "training_generation_calls": 0,
                "teacher_calls": 0,
                "heldout_calls_before_training_complete": 0,
            }
            if self.metrics["budget_check"]["supervised_tokens"] != 2560:
                raise RuntimeError("Oracle control did not complete its exact 2560-token budget")
            result["phase"] = "post_training"
            self.optimizer = None
            self.save_checkpoint()
            self.event("oracle_training_sealed", updates=64, supervised_tokens=2560)
        return self.evaluate_saved_oracle_adapters()

    def evaluate_saved_oracle_adapters(self):
        self.stop.check()
        if (
            self.metrics["oracle_sft_control"]["phase"] != "post_training"
            or self.next_step != 64
            or self.optimizer is not None
        ):
            raise RuntimeError("Saved-adapter evaluation requires completed, sealed oracle training")
        sealed = torch.load(self.output / "checkpoint/state.pt", map_location="cpu", weights_only=True)
        self.check_oracle_checkpoint(sealed)
        if sealed["metrics"]["oracle_sft_control"]["phase"] != "post_training":
            raise OutputConflict("Oracle training seal is not committed on disk")
        result = self.metrics["oracle_sft_control"]
        self.data = build_oracle_eval_data(self.config.seed)
        self.io.data = self.data
        evaluation_manifest = data_manifest(self.data, self.config.seed, 0)
        atomic_json(self.output / "eval_manifest.json", evaluation_manifest)
        result["eval_data_sha256"] = evaluation_manifest["sha256"]
        heldout_suites = suites(self.data, "heldout")
        schedule = [
            (name, example, "current")
            for name, examples in heldout_suites.items()
            for example in examples
        ]
        for label, updates in (("initial", 0), ("after_permutation", 32), ("after_symbol_map", 64)):
            self.stop.check()
            saved = result["saved_adapters"][label]
            path = self.output / saved["path"]
            if saved["completed_updates"] != updates or hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
                raise OutputConflict(f"Saved oracle adapter changed: {label}")
            state = load_file(path, device="cpu")
            set_peft_model_state_dict(self.model, state)
            actual = adapter_state(self.model)
            if set(actual) != set(state) or any(not torch.equal(actual[key], value) for key, value in state.items()):
                raise RuntimeError(f"Oracle adapter reload did not reproduce saved tensors: {label}")
            evaluation = result["evaluations"].setdefault(
                label, {"adapter_sha256": saved["sha256"], "loaded_from_disk": True, "records": []}
            )
            if evaluation["adapter_sha256"] != saved["sha256"]:
                raise OutputConflict(f"Oracle evaluation used a different saved adapter: {label}")
            records = self.collect_generation_records(
                evaluation, schedule, "cue_only", False, f"post_training/{label}"
            )
            evaluation["measurements"] = {
                name: generation_aggregate([row for row in records if row["suite"] == name])
                for name in heldout_suites
            }
            self.save_checkpoint()
        measurements = {label: value["measurements"] for label, value in result["evaluations"].items()}
        result["gates"] = oracle_control_gates(measurements)
        result["composition_changes"] = {
            name: {
                "initial_accuracy": measurements["initial"][name]["accuracy"],
                "after_permutation_accuracy": measurements["after_permutation"][name]["accuracy"],
                "final_accuracy": measurements["after_symbol_map"][name]["accuracy"],
                "final_minus_initial": measurements["after_symbol_map"][name]["accuracy"]
                - measurements["initial"][name]["accuracy"],
            }
            for name in heldout_suites if name.endswith("/compositions")
        }
        evaluation_calls = sum(len(value["records"]) for value in result["evaluations"].values())
        if evaluation_calls != 480:
            raise RuntimeError("Oracle control violated its 480-call post-training evaluation budget")
        self.metrics["budget_check"]["post_training_evaluation_calls"] = evaluation_calls
        result["phase"] = "completed"
        self.metrics["status"] = "completed"
        self.metrics["completed_at"] = now()
        self.save_checkpoint()
        self.progress("oracle_control_completed", feasible=result["gates"]["feasible"])
        self.event("oracle_control_completed", gates=result["gates"], budget=self.metrics["budget_check"])
        return 0

    def teacher_gate(self):
        if "teacher_gate" not in self.metrics["reference"]:
            measurements = self.evaluate(
                "teacher_gate", privileged=True, split="gate", contexts=("examples",)
            )
            failed = {
                name: row["accuracy"]
                for name, row in measurements.items()
                if row["accuracy"] < self.config.teacher_min_accuracy
            }
            self.metrics["reference"]["teacher_gate"] = {
                "passed": not failed,
                "threshold": self.config.teacher_min_accuracy,
                "failed_suites": failed,
                "measurements": measurements,
            }
            if failed:
                self.metrics["status"] = "teacher_gate_failed"
                self.event(
                    "teacher_gate_failed",
                    failed_suites=failed,
                    threshold=self.config.teacher_min_accuracy,
                )
            self.save_checkpoint()
        return self.metrics["reference"]["teacher_gate"]["passed"]

    def update(self, arm, task, step):
        example = self.data[task]["train"][step % self.config.train_examples]
        student_prefix = self.io.prefix(example)
        teacher_prefix = self.io.prefix(example, privileged=True)
        oracle = self.io.oracle_tokens(example)
        remaining = self.config.tokens_per_update
        record = {
            "task": task,
            "step_in_task": step + 1,
            "global_step": self.next_step + 1,
            "example_id": example.key,
            "supervised_tokens": 0,
            "generated_tokens": 0,
            "student_forward_input_tokens": 0,
            "teacher_forward_input_tokens": 0,
            "generation_prompt_tokens": 0,
            "rollouts": 0,
            "loss_sum": 0.0,
        }
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while remaining:
            self.stop.check()
            limit = min(self.config.max_new_tokens, remaining)
            if arm == "oracle_sft":
                completion, generated = oracle[:limit], 0
            else:
                privileged = arm == "off_policy_dense"
                prefix = teacher_prefix if privileged else student_prefix
                completion, generated = self.io.generate(
                    prefix,
                    privileged,
                    True,
                    stable_seed(self.config.seed, "train", self.next_step, record["rollouts"]),
                    limit,
                )
                record["generation_prompt_tokens"] += len(prefix)
            n = len(completion)
            if not 0 < n <= remaining:
                raise RuntimeError("Continuation quota violated")
            mask = torch.ones(n, dtype=torch.bool, device=completion.device)
            if arm == "oracle_sft":
                student_logits = response_logits(self.model, student_prefix, completion)
                loss_sum = oracle_nll(student_logits, completion, mask)
            else:
                with torch.no_grad(), policy_mode(self.model, privileged=True):
                    teacher_logits = response_logits(self.model, teacher_prefix, completion)
                student_logits = response_logits(self.model, student_prefix, completion)
                loss_sum = dense_reverse_kl(student_logits, teacher_logits, mask)
                record["teacher_forward_input_tokens"] += len(teacher_prefix) + n
                del teacher_logits
            if not torch.isfinite(loss_sum):
                raise FloatingPointError(f"Non-finite training loss: {arm}/{task}/{step}")
            (loss_sum / self.config.tokens_per_update).backward()
            record["loss_sum"] += float(loss_sum.detach())
            record["supervised_tokens"] += n
            record["generated_tokens"] += generated
            record["student_forward_input_tokens"] += len(student_prefix) + n
            record["rollouts"] += 1
            self.event(
                "training_rollout",
                arm=arm,
                global_step=record["global_step"],
                example_id=example.key,
                source=arm,
                tokens=completion.tolist(),
                student_prefix_tokens=len(student_prefix),
                teacher_prefix_tokens=len(teacher_prefix),
                supervised_tokens=n,
                truncated=int(completion[-1]) not in self.io.eos_ids,
            )
            remaining -= n
            del student_logits, loss_sum, completion
        self.stop.check()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.config.max_grad_norm, error_if_nonfinite=True
        )
        if not any(p.grad is not None for p in parameters):
            raise RuntimeError("No adapter receives a gradient; the training graph is disconnected")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        record["grad_norm"] = float(norm)
        record["zero_gradient"] = float(norm) == 0
        record["loss_per_token"] = record["loss_sum"] / record["supervised_tokens"]
        return record

    def run_arm(self, arm):
        if self.metrics["arms"].get(arm, {}).get("completed"):
            return
        if self.current_arm != arm:
            set_peft_model_state_dict(self.model, self.initial_adapter)
            self.optimizer = self.optimizer_for_model()
            self.current_arm, self.next_step = arm, 0
            self.metrics["arms"][arm] = {
                "completed": False,
                "updates": [],
                "evaluations": {"before": self.metrics["baseline"]},
            }
            self.save_checkpoint()
        result = self.metrics["arms"][arm]
        total_steps = len(TASKS) * self.config.steps_per_task
        while self.next_step <= total_steps:
            self.stop.check()
            if self.next_step and self.next_step % self.config.steps_per_task == 0:
                finished_task = TASKS[self.next_step // self.config.steps_per_task - 1]
                label = f"after_{finished_task}"
                if label not in result["evaluations"]:
                    self.save_adapter(f"{arm}/{label}")
                    result["evaluations"][label] = self.evaluate(label)
                    self.save_checkpoint()
            if self.next_step == total_steps:
                break
            task_index, step = divmod(self.next_step, self.config.steps_per_task)
            self.progress("training", task=TASKS[task_index], step_in_task=step)
            record = self.update(arm, TASKS[task_index], step)
            result["updates"].append(record)
            self.next_step += 1
            self.metrics["evidence"]["gpu_training_observed"] = self.config.device.startswith(
                "cuda"
            )
            self.event("optimizer_update", arm=arm, **record)
            if self.next_step % self.config.checkpoint_every == 0 or self.stop.signum is not None:
                self.save_checkpoint()
        result["completed"] = True
        result["comparisons"] = comparisons(result["evaluations"])
        result["adapter_delta_l2"] = (
            sum(
                float((value.float() - self.initial_adapter[key].float()).square().sum())
                for key, value in adapter_state(self.model).items()
            )
            ** 0.5
        )
        result["budget"] = {
            "updates": len(result["updates"]),
            **{
                key: sum(row[key] for row in result["updates"])
                for key in (
                    "supervised_tokens",
                    "generated_tokens",
                    "rollouts",
                    "student_forward_input_tokens",
                    "teacher_forward_input_tokens",
                    "generation_prompt_tokens",
                )
            },
        }
        self.save_checkpoint()

    def run(self):
        if not self.prepare():
            return (
                0
                if self.metrics["status"] in ("completed", "train_answer_calibration_passed")
                else 2
            )
        if self.config.mode == "oracle_sft_control":
            return self.oracle_sft_control()
        if self.config.mode == "train_answer_calibration":
            return self.train_answer_calibration()
        if self.config.mode == "teacher_calibration":
            return self.teacher_calibration()
        if not self.teacher_gate():
            self.progress("teacher_gate_failed")
            return 2
        if "teacher_heldout" not in self.metrics["reference"]:
            self.metrics["reference"]["teacher_heldout"] = self.evaluate(
                "teacher_heldout", privileged=True
            )
            self.save_checkpoint()
        if "baseline" not in self.metrics:
            self.metrics["baseline"] = self.evaluate("before")
            self.save_checkpoint()
        for arm in self.config.arms:
            self.run_arm(arm)
        budgets = [result["budget"] for result in self.metrics["arms"].values()]
        expected_updates = len(TASKS) * self.config.steps_per_task
        if any(
            x["updates"] != expected_updates
            or x["supervised_tokens"] != expected_updates * self.config.tokens_per_update
            for x in budgets
        ):
            raise RuntimeError("Matched update/token budget verification failed")
        self.metrics["budget_check"] = {
            "passed": True,
            "updates_per_arm": expected_updates,
            "supervised_tokens_per_arm": expected_updates * self.config.tokens_per_update,
        }
        dense = self.metrics["arms"]
        if "on_policy_dense" in dense and "off_policy_dense" in dense:
            self.metrics["on_minus_off"] = {
                key: {
                    metric: dense["on_policy_dense"]["comparisons"][key][metric] - value
                    for metric, value in row.items()
                }
                for key, row in dense["off_policy_dense"]["comparisons"].items()
            }
        if self.config.device.startswith("cuda"):
            self.metrics["runtime"]["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
        self.metrics["status"] = "completed"
        self.metrics["completed_at"] = now()
        self.save_checkpoint()
        self.progress("completed")
        self.event(
            "experiment_completed", output_dir=str(self.output), budget=self.metrics["budget_check"]
        )
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Bounded teacher calibration or sequential privileged self-distillation pilot"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [onpolicy] %(message)s")
    config = read_config(args.config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop = StopFlag()
    signal.signal(signal.SIGTERM, stop.request)
    signal.signal(signal.SIGINT, stop.request)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment = Experiment(config, output, stop)
        try:
            return experiment.run()
        except OutputConflict as error:
            LOGGER.error("OUTPUT_CONFLICT: %s", error)
            return 1
        except StopRequested as error:
            experiment.metrics["status"] = "terminated"
            experiment.event("termination_requested", signum=stop.signum, reason=str(error))
            experiment.save_checkpoint()
            experiment.progress("terminated")
            return 128 + stop.signum
        except Exception as error:
            LOGGER.exception("EXPERIMENT_FAILURE: %s", error)
            experiment.metrics["status"] = "failed"
            experiment.metrics["error"] = {"type": type(error).__name__, "message": str(error)}
            experiment.event(
                "experiment_failed", **experiment.metrics["error"], traceback=traceback.format_exc()
            )
            experiment.write_metrics()
            experiment.progress("failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())
