import argparse
import fcntl
import hashlib
import json
import logging
import os
import random
import signal
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from config import RunConfig, digest
from data import make_dataset, step_examples
from evaluation import evaluate, summarize_method
from gradients import assign_gradient, choose_gradient
from modeling import (
    encode_examples,
    file_sha256,
    inspect_model_provenance,
    load_model,
    synchronize,
    training_gradient,
)

LOGGER = logging.getLogger("continual")
SCHEMA_VERSION = 1


class InterruptedRun(Exception):
    pass


class StopFlag:
    def __init__(self):
        self.signal = None

    def handle(self, signum, frame):
        self.signal = signum

    def raise_if_requested(self):
        if self.signal is not None:
            raise InterruptedRun(f"[stop] received signal {self.signal}")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Artifacts:
    def __init__(self, output, config):
        self.output = output
        self.config = config
        self.attempt_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.position = {
            "method": None,
            "task": None,
            "step": 0,
            "optimizer_updates": 0,
        }

    def emit(self, event, **fields):
        record = {
            "schema_version": SCHEMA_VERSION,
            "at": now(),
            "attempt_id": self.attempt_id,
            "event": event,
            "config_sha256": self.config.fingerprint(),
            "position": dict(self.position),
            **fields,
        }
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        self.progress("running", event=event, **fields)
        if event != "eval_batch":
            LOGGER.info(
                "[%s] method=%s task=%s step=%s",
                event,
                self.position["method"],
                self.position["task"],
                self.position["step"],
            )

    def progress(self, status, **fields):
        atomic_json(
            self.output / "progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": now(),
                "status": status,
                "attempt_id": self.attempt_id,
                "attempt_seconds": time.perf_counter() - self.started,
                "config_sha256": self.config.fingerprint(),
                **self.position,
                **fields,
            },
        )

    def metrics(self, metrics):
        metrics["updated_at"] = now()
        metrics["last_attempt"] = {
            "attempt_id": self.attempt_id,
            "seconds": time.perf_counter() - self.started,
        }
        atomic_json(self.output / "metrics.json", metrics)


def adapter_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def adapter_hash(state):
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        hasher.update(name.encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.contiguous().numpy().tobytes())
    return hasher.hexdigest()


@torch.no_grad()
def restore_adapter(model, saved):
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if parameters.keys() != saved.keys():
        raise ValueError("[checkpoint] trainable parameter names changed")
    for name, parameter in parameters.items():
        if parameter.shape != saved[name].shape or parameter.dtype != saved[name].dtype:
            raise ValueError(f"[checkpoint] parameter shape/dtype changed: {name}")
        parameter.copy_(saved[name].to(parameter.device))


def save_checkpoint(
    output, state, model, optimizer, fingerprints, initial_hash, artifacts
):
    state["checkpoint_generation"] += 1
    state["metrics"]["durable_optimizer_updates"] = state["optimizer_updates"]
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "fingerprints": fingerprints,
        "initial_adapter_sha256": initial_hash,
        "state": state,
        "adapter": adapter_state(model),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = output / "checkpoint.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "checkpoint.pt")
    artifacts.metrics(state["metrics"])
    artifacts.emit(
        "checkpoint_committed",
        generation=state["checkpoint_generation"],
        durable_optimizer_updates=state["optimizer_updates"],
    )


def restore_checkpoint(path, model, optimizer, fingerprints, initial_hash):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint["schema_version"] != SCHEMA_VERSION
        or checkpoint["fingerprints"] != fingerprints
    ):
        raise ValueError(
            "[checkpoint] config, data, source, model contents or library versions changed; use a new output directory"
        )
    if checkpoint["initial_adapter_sha256"] != initial_hash:
        raise ValueError(
            "[checkpoint] initial adapter changed under the same configured seed"
        )
    restore_adapter(model, checkpoint["adapter"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint["state"]


def empty_budgets():
    return {
        "optimizer_updates": 0,
        "forward_backward_passes": 0,
        "current_examples": 0,
        "current_input_tokens": 0,
        "current_supervised_tokens": 0,
        "current_padded_tokens": 0,
        "reference_examples": 0,
        "reference_input_tokens": 0,
        "reference_supervised_tokens": 0,
        "reference_padded_tokens": 0,
    }


def new_method():
    return {
        "status": "running",
        "boundaries": [],
        "budgets": empty_budgets(),
        "training_seconds": 0.0,
        "gradient_reference_steps": 0,
        "conflicting_steps": 0,
        "projected_steps": 0,
        "schedule_sha256": digest([]),
        "steps": [],
    }


def make_optimizer(parameters, config):
    return torch.optim.SGD(
        parameters,
        lr=config.optimization.learning_rate,
        momentum=0,
        weight_decay=0,
        foreach=False,
    )


def update_position(artifacts, state, config):
    index = state["method_index"]
    artifacts.position = {
        "method": config.methods[index] if index < len(config.methods) else None,
        "task": state["task"],
        "step": state["step"],
        "optimizer_updates": state["optimizer_updates"],
    }


def validate_budget_match(methods):
    completed = {
        name: value for name, value in methods.items() if value["status"] == "completed"
    }
    signatures = {
        digest([value["budgets"], value["schedule_sha256"]])
        for value in completed.values()
    }
    if len(signatures) > 1:
        raise RuntimeError(
            "[budget] methods did not process identical current/reference budgets and example schedules"
        )
    return {
        "checked_methods": list(completed),
        "passed": len(completed) > 1 and len(signatures) == 1,
        "contract": "identical current/reference rows, padded and supervised tokens, backward passes and SGD update counts",
        "objective_difference": "sequential and A-GEM optimize current loss; replay mixes current and reference loss; A-GEM constrains with references",
        "timing_matched": False,
    }


def execute(config, output, stop):
    output = Path(output)
    artifacts = Artifacts(output, config)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "created_at": now(),
        "config": config.as_dict(),
        "config_sha256": config.fingerprint(),
        "methods": {},
        "claims": "Measured pilot outcomes only. CPU fixtures and integration gates do not establish Qwen GPU learning.",
    }
    existing = output / "metrics.json"
    if (
        existing.exists()
        and json.loads(existing.read_text())["config_sha256"] != config.fingerprint()
    ):
        raise ValueError("[output] directory belongs to a different config")
    artifacts.metrics(metrics)
    artifacts.emit("initializing", purpose=config.purpose)
    state = None
    try:
        config.validate()
        torch.set_num_threads(config.runtime.cpu_threads)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(config.runtime.deterministic_algorithms)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        dataset = make_dataset(config.data, config.seed)
        atomic_json(output / "dataset.json", dataset.manifest())
        metrics["data_provenance"] = {
            "dataset_sha256": dataset.fingerprint(),
            "generator": dataset.manifest()["generator"],
            "manifest": "dataset.json",
            "split_policy": dataset.manifest()["split_policy"],
        }
        stop.raise_if_requested()
        artifacts.emit("model_provenance_started")
        provenance = inspect_model_provenance(config)
        metrics["model_provenance"] = provenance
        artifacts.metrics(metrics)
        artifacts.emit(
            "model_load_started", model_id=config.model_id, revision=config.revision
        )
        model, tokenizer, runtime = load_model(config)
        metrics["runtime"] = runtime
        artifacts.metrics(metrics)
        device = torch.device(config.runtime.device)
        encoded = encode_examples(
            dataset.examples,
            tokenizer,
            config.pilot.max_sequence_length,
            config.pilot.max_new_tokens,
        )
        initial = adapter_state(model)
        initial_hash = adapter_hash(initial)
        metrics["initial_adapter_sha256"] = initial_hash
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = make_optimizer(parameters, config)
        metrics["optimizer"] = {
            "name": "SGD",
            "learning_rate": config.optimization.learning_rate,
            "momentum": 0,
            "weight_decay": 0,
            "adapter_dtype": "float32",
            "projection_scope": "Euclidean adapter-gradient halfspace; first-order reference loss constraint only",
        }
        fingerprints = {
            "config": config.fingerprint(),
            "dataset": dataset.fingerprint(),
            "model": digest(provenance),
            "source": {
                file.name: file_sha256(file)
                for file in sorted(Path(__file__).parent.glob("*.py"))
            },
            "libraries": runtime["versions"],
        }
        metrics["fingerprints"] = fingerprints
        metrics["status"] = "running"
        state = {
            "method_index": 0,
            "task": 0,
            "step": 0,
            "before_done": False,
            "optimizer_updates": 0,
            "checkpoint_generation": 0,
            "metrics": metrics,
        }
        checkpoint_path = output / "checkpoint.pt"
        if checkpoint_path.exists():
            state = restore_checkpoint(
                checkpoint_path, model, optimizer, fingerprints, initial_hash
            )
            metrics = state["metrics"]
            metrics["status"] = "running"
            update_position(artifacts, state, config)
            artifacts.emit(
                "resumed",
                generation=state["checkpoint_generation"],
                durable_optimizer_updates=state["optimizer_updates"],
                event_policy="Prior attempt steps beyond this committed generation were rolled back; events remain append-only.",
            )
        elif "headroom" not in metrics:
            headroom = evaluate(
                model,
                tokenizer,
                encoded,
                dataset,
                config,
                device,
                stop,
                artifacts.emit,
                splits=("calibration",),
            )
            failing = [
                task
                for task, value in headroom["tasks"].items()
                if value["calibration"]["exact_match"]
                >= config.pilot.base_accuracy_ceiling
            ]
            metrics["headroom"] = {
                "evaluation": headroom,
                "passed": not failing,
                "failing_tasks": failing,
                "ceiling": config.pilot.base_accuracy_ceiling,
                "policy": "Fixed predeclared calibration only; no test-based selection or resampling",
            }
            if failing:
                metrics["status"] = "rejected_base_headroom"
                artifacts.emit("rejected_base_headroom", tasks=failing)
                artifacts.metrics(metrics)
                artifacts.progress(metrics["status"])
                return metrics
            save_checkpoint(
                output, state, model, optimizer, fingerprints, initial_hash, artifacts
            )
        update_position(artifacts, state, config)
        while state["method_index"] < len(config.methods):
            stop.raise_if_requested()
            method_name = config.methods[state["method_index"]]
            if method_name not in metrics["methods"]:
                restore_adapter(model, initial)
                random.seed(config.seed)
                torch.manual_seed(config.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(config.seed)
                optimizer = make_optimizer(parameters, config)
                metrics["methods"][method_name] = new_method()
                metrics["methods"][method_name]["initial_adapter_sha256"] = (
                    adapter_hash(adapter_state(model))
                )
                artifacts.emit("method_started", initial_adapter_sha256=initial_hash)
            method = metrics["methods"][method_name]
            while state["task"] < config.data.num_tasks:
                task = state["task"]
                update_position(artifacts, state, config)
                if not state["before_done"]:
                    evaluation = evaluate(
                        model,
                        tokenizer,
                        encoded,
                        dataset,
                        config,
                        device,
                        stop,
                        artifacts.emit,
                    )
                    method["boundaries"].append(
                        {
                            "task": task,
                            "phase": "before",
                            "updates": method["budgets"]["optimizer_updates"],
                            "evaluation": evaluation,
                        }
                    )
                    state["before_done"] = True
                    save_checkpoint(
                        output,
                        state,
                        model,
                        optimizer,
                        fingerprints,
                        initial_hash,
                        artifacts,
                    )
                while state["step"] < config.pilot.steps_per_task:
                    stop.raise_if_requested()
                    step = state["step"]
                    current, reference = step_examples(
                        dataset,
                        task,
                        step,
                        config.pilot.current_batch_size,
                        config.pilot.reference_batch_size,
                    )
                    current_encoded = [
                        encoded[example.example_id] for example in current
                    ]
                    reference_encoded = [
                        encoded[example.example_id] for example in reference
                    ]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    synchronize(device)
                    start = time.perf_counter()
                    current_gradient, current_loss, current_counts = training_gradient(
                        model,
                        current_encoded,
                        tokenizer,
                        parameters,
                        device,
                        task,
                        "current",
                    )
                    reference_gradient = None
                    reference_loss = None
                    reference_counts = {key: 0 for key in current_counts}
                    if reference:
                        reference_gradient, reference_loss, reference_counts = (
                            training_gradient(
                                model,
                                reference_encoded,
                                tokenizer,
                                parameters,
                                device,
                                task,
                                "reference",
                            )
                        )
                    direction, conflict = choose_gradient(
                        method_name,
                        current_gradient,
                        reference_gradient,
                        config.optimization.replay_weight,
                        config.optimization.max_grad_norm,
                    )
                    before_parameters = torch.cat(
                        [parameter.detach().reshape(-1) for parameter in parameters]
                    )
                    assign_gradient(parameters, direction)
                    optimizer.step()
                    after_parameters = torch.cat(
                        [parameter.detach().reshape(-1) for parameter in parameters]
                    )
                    actual_direction = (
                        before_parameters - after_parameters
                    ) / config.optimization.learning_rate
                    if not torch.isfinite(after_parameters).all():
                        raise FloatingPointError(
                            "[optimizer] non-finite adapter weights after SGD"
                        )
                    conflict["actual_parameter_delta_norm"] = torch.linalg.vector_norm(
                        before_parameters - after_parameters
                    ).item()
                    conflict["actual_sgd_reference_dot"] = (
                        torch.dot(
                            actual_direction.double(), reference_gradient.double()
                        ).item()
                        if reference
                        else None
                    )
                    synchronize(device)
                    seconds = time.perf_counter() - start
                    state["step"] += 1
                    state["optimizer_updates"] += 1
                    budgets = method["budgets"]
                    budgets["optimizer_updates"] += 1
                    budgets["forward_backward_passes"] += 1 + bool(reference)
                    for role, counts in (
                        ("current", current_counts),
                        ("reference", reference_counts),
                    ):
                        for key, count in counts.items():
                            budgets[f"{role}_{key}"] += count
                    method["training_seconds"] += seconds
                    method["gradient_reference_steps"] += bool(reference)
                    method["conflicting_steps"] += bool(conflict["conflict"])
                    method["projected_steps"] += conflict["projected"]
                    current_ids = [example.example_id for example in current]
                    reference_ids = [example.example_id for example in reference]
                    method["schedule_sha256"] = digest(
                        [method["schedule_sha256"], current_ids, reference_ids]
                    )
                    record = {
                        "task": task,
                        "step": step + 1,
                        "update": budgets["optimizer_updates"],
                        "current_ids": current_ids,
                        "reference_ids": reference_ids,
                        "current_loss": current_loss,
                        "reference_loss": reference_loss,
                        "gradient": conflict,
                        "current_counts": current_counts,
                        "reference_counts": reference_counts,
                        "seconds": seconds,
                        "objective_supervised_tokens": current_counts[
                            "supervised_tokens"
                        ]
                        + (
                            reference_counts["supervised_tokens"]
                            if method_name == "replay"
                            else 0
                        ),
                    }
                    method["steps"].append(record)
                    update_position(artifacts, state, config)
                    artifacts.emit(
                        "step_finished",
                        **record,
                        checkpoint_generation=state["checkpoint_generation"],
                    )
                    if (
                        state["step"] % config.pilot.checkpoint_every_steps == 0
                        or state["step"] == config.pilot.steps_per_task
                        or stop.signal is not None
                    ):
                        save_checkpoint(
                            output,
                            state,
                            model,
                            optimizer,
                            fingerprints,
                            initial_hash,
                            artifacts,
                        )
                    stop.raise_if_requested()
                evaluation = evaluate(
                    model,
                    tokenizer,
                    encoded,
                    dataset,
                    config,
                    device,
                    stop,
                    artifacts.emit,
                )
                method["boundaries"].append(
                    {
                        "task": task,
                        "phase": "after",
                        "updates": method["budgets"]["optimizer_updates"],
                        "adapter_sha256": adapter_hash(adapter_state(model)),
                        "evaluation": evaluation,
                    }
                )
                method["summary"] = summarize_method(
                    method["boundaries"], config.data.num_tasks
                )
                state["task"] += 1
                state["step"] = 0
                state["before_done"] = False
                update_position(artifacts, state, config)
                save_checkpoint(
                    output,
                    state,
                    model,
                    optimizer,
                    fingerprints,
                    initial_hash,
                    artifacts,
                )
            method["status"] = "completed"
            denominator = method["gradient_reference_steps"]
            method["gradient_conflict_rate"] = (
                method["conflicting_steps"] / denominator if denominator else None
            )
            method["final_adapter_sha256"] = adapter_hash(adapter_state(model))
            model.save_pretrained(
                output / "adapters" / method_name, safe_serialization=True
            )
            method["adapter_path"] = f"adapters/{method_name}"
            metrics["budget_match"] = validate_budget_match(metrics["methods"])
            state["method_index"] += 1
            state["task"] = 0
            update_position(artifacts, state, config)
            save_checkpoint(
                output, state, model, optimizer, fingerprints, initial_hash, artifacts
            )
        metrics["status"] = "completed"
        metrics["completed_at"] = now()
        metrics["optimizer_updates"] = state["optimizer_updates"]
        metrics["measured_training_seconds"] = sum(
            method["training_seconds"] for method in metrics["methods"].values()
        )
        metrics["measured_evaluation_seconds"] = metrics["headroom"]["evaluation"][
            "seconds"
        ] + sum(
            boundary["evaluation"]["seconds"]
            for method in metrics["methods"].values()
            for boundary in method["boundaries"]
        )
        metrics["peak_gpu_memory_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        save_checkpoint(
            output, state, model, optimizer, fingerprints, initial_hash, artifacts
        )
        artifacts.emit(
            "completed",
            optimizer_updates=state["optimizer_updates"],
            budget_match=metrics["budget_match"],
        )
        artifacts.metrics(metrics)
        artifacts.progress("completed")
        return metrics
    except InterruptedRun as exc:
        if state is not None and metrics.get("headroom", {}).get("passed"):
            save_checkpoint(
                output, state, model, optimizer, fingerprints, initial_hash, artifacts
            )
        metrics["status"] = "interrupted"
        metrics["last_interruption"] = str(exc)
        artifacts.emit(
            "interrupted",
            reason=str(exc),
            resume="Same config/output resumes from checkpoint.pt; uncommitted events are retained with attempt IDs",
        )
        artifacts.metrics(metrics)
        artifacts.progress("interrupted", reason=str(exc))
        return metrics
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        artifacts.emit("failed", failure=metrics["failure"])
        artifacts.metrics(metrics)
        artifacts.progress("failed", failure=metrics["failure"])
        LOGGER.exception(
            "[continual-failed] resume requires the same fingerprinted config, code, data and model"
        )
        raise


def run_experiment(config, output, stop=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"[output] another process owns {output}") from exc
        return execute(config, output, stop or StopFlag())


def main():
    parser = argparse.ArgumentParser(
        description="Matched-budget sequential LoRA, replay and A-GEM on newly sampled rules"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [continual] %(message)s"
    )
    config = RunConfig.from_dict(json.loads(args.config.read_text()))
    stop = StopFlag()
    signal.signal(signal.SIGTERM, stop.handle)
    signal.signal(signal.SIGINT, stop.handle)
    metrics = run_experiment(config, args.output_dir, stop)
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "metrics": str(args.output_dir / "metrics.json"),
            }
        ),
        flush=True,
    )
    return {"completed": 0, "interrupted": 143, "rejected_base_headroom": 2}[
        metrics["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
