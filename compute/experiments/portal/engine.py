import gc
import json
import math
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace

import torch
from peft import LoraConfig, get_peft_model
from portallib import PortalEvaluator, collate_gold_batch
from portallib.evaluation import PortalInjector
from safetensors.torch import load_file, save_file

from inventory import tensor_digest
from runtime import (
    CapabilityBlocked,
    atomic_json,
    ensure_gpu,
    load_local_base,
    load_port,
    registry_from,
    select_port,
)
from tasks import evaluation_dataset, make_tasks


@dataclass(frozen=True)
class Pilot:
    steps_per_task: int = 32
    batch_size: int = 4
    train_examples: int = 128
    validation_examples: int = 32
    test_examples: int = 64
    eval_batch_size: int = 8
    max_prompt_tokens: int = 128
    checkpoint_every: int = 8
    latent_learning_rate: float = 0.002
    lora_learning_rate: float = 0.0002
    gradient_clip: float = 1.0
    gradient_checkpointing: bool = True
    source_native_lora: bool = True

    def __post_init__(self):
        integer_fields = (
            self.steps_per_task,
            self.batch_size,
            self.train_examples,
            self.validation_examples,
            self.test_examples,
            self.eval_batch_size,
            self.max_prompt_tokens,
            self.checkpoint_every,
        )
        if any(not isinstance(value, int) or value < 1 for value in integer_fields):
            raise ValueError("pilot_counts_must_be_positive_integers")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.latent_learning_rate,
                self.lora_learning_rate,
                self.gradient_clip,
            )
        ):
            raise ValueError(
                "pilot_learning_rates_and_gradient_clip_must_be_positive_finite"
            )


class Adaptation(AbstractContextManager):
    def __init__(self, base, portal, kind, initial_latent, seed):
        self.kind = kind
        self.original_base = base
        self.base = base
        self.portal = portal
        self.injector = None
        self.peft = None
        self.core_digest = tensor_digest(portal.core.state_dict())
        self.alignment_digest = tensor_digest(portal.alignment.state_dict())
        torch.manual_seed(seed)
        base.freeze()
        if kind == "latent":
            portal.to(device=base.device, dtype=torch.float32).requires_grad_(False)
            self.latent = torch.nn.Parameter(
                initial_latent.to(device=base.device, dtype=torch.float32).clone()
            )
            self.parameters = {"latent": self.latent}
            self.injector = PortalInjector(base.model, portal.config)
        elif kind == "native_lora":
            paths = [path for _target, path in portal.config.resolved_targets()]
            self.peft = get_peft_model(
                base.model,
                LoraConfig(
                    base_model_name_or_path=base.model_id,
                    revision=base.revision,
                    task_type="CAUSAL_LM",
                    r=portal.config.rank,
                    lora_alpha=portal.config.alpha,
                    target_modules=paths,
                    lora_dropout=0,
                    bias="none",
                ),
            )
            self.base = replace(base, model=self.peft)
            self.parameters = {
                name: param
                for name, param in self.peft.named_parameters()
                if param.requires_grad
            }
            if not self.parameters or any(
                "lora_" not in name for name in self.parameters
            ):
                raise RuntimeError("native_lora_trainable_contract_failed")
        else:
            raise ValueError(f"unknown_adaptation_kind: {kind}")

    def activation(self):
        if self.injector is None:
            return nullcontext()
        return self.injector.activate(self.portal(self.latent))

    def state(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.parameters.items()
        }

    def restore(self, state):
        if set(state) != set(self.parameters):
            raise ValueError("checkpoint_trainable_names_mismatch")
        with torch.no_grad():
            for name, value in state.items():
                target = self.parameters[name]
                if target.shape != value.shape:
                    raise ValueError(f"checkpoint_trainable_shape_mismatch: {name}")
                target.copy_(value)

    def verify_frozen(self):
        if tensor_digest(self.portal.core.state_dict()) != self.core_digest:
            raise RuntimeError("released_core_was_modified")
        if tensor_digest(self.portal.alignment.state_dict()) != self.alignment_digest:
            raise RuntimeError("released_alignment_was_modified")
        trained = {id(parameter) for parameter in self.parameters.values()}
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in self.base.model.parameters()
            if id(parameter) not in trained
        ):
            raise RuntimeError("frozen_base_parameter_received_gradient")

    def __exit__(self, *_args):
        if self.injector is not None:
            self.injector.close()
        if self.peft is not None:
            self.peft.unload()
        self.original_base.freeze()
        return False


def batch_schedule(rows, steps, batch_size, seed):
    rng = random.Random(seed)
    order = []
    while len(order) < steps * batch_size:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        order.extend(indices)
    return [
        order[start : start + batch_size]
        for start in range(0, steps * batch_size, batch_size)
    ]


def sync_device(base):
    if base.device.type == "cuda":
        torch.cuda.synchronize(base.device)


def save_checkpoint(path, adaptation, optimizer, step, stats, log, task, start_digest):
    content = {
        "config_sha256": log.digest,
        "task": task,
        "step": step,
        "stats": stats,
        "start_state_sha256": start_digest,
        "trainable_state": adaptation.state(),
        "optimizer": optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(adaptation.base.device)
        if adaptation.base.device.type == "cuda"
        else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def fit_stage(adaptation, rows, task, arm, pilot, log, seed):
    directory = log.output / arm / task
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    learning_rate = (
        pilot.latent_learning_rate
        if adaptation.kind == "latent"
        else pilot.lora_learning_rate
    )
    optimizer = torch.optim.AdamW(
        list(adaptation.parameters.values()), lr=learning_rate, weight_decay=0
    )
    stats = {
        "steps": 0,
        "examples_seen": 0,
        "answer_tokens_seen": 0,
        "input_tokens_seen": 0,
        "optimizer_seconds": 0.0,
        "trainable_parameters": sum(p.numel() for p in adaptation.parameters.values()),
        "learning_rate": learning_rate,
        "losses": [],
        "gradient_norms": [],
    }
    start_digest = tensor_digest(adaptation.state())
    step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved["config_sha256"] != log.digest
            or saved["task"] != task
            or saved["start_state_sha256"] != start_digest
        ):
            raise ValueError("checkpoint_config_task_or_start_state_mismatch")
        adaptation.restore(saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"])
        if saved["rng_cuda"] is not None:
            torch.cuda.set_rng_state(saved["rng_cuda"], adaptation.base.device)
        stats = saved["stats"]
        step = saved["step"]
        log.event("training_resumed", arm=arm, task=task, step=step)
    schedule = batch_schedule(rows, pilot.steps_per_task, pilot.batch_size, seed)
    if step > pilot.steps_per_task:
        raise ValueError("checkpoint_exceeds_requested_budget")
    adaptation.base.model.train()
    for index in range(step, pilot.steps_per_task):
        log.check_stop()
        batch = [rows[row_index] for row_index in schedule[index]]
        sync_device(adaptation.base)
        started = time.perf_counter()
        ids, mask, labels = collate_gold_batch(
            adaptation.base.tokenizer,
            batch,
            max_prompt=pilot.max_prompt_tokens,
            device=adaptation.base.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with adaptation.activation():
            loss = adaptation.base.model(
                input_ids=ids, attention_mask=mask, labels=labels, use_cache=False
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"training_nonfinite_loss: {arm} {task} step={index}"
                )
            loss.backward()
        active = [
            parameter
            for parameter in adaptation.parameters.values()
            if parameter.grad is not None
        ]
        if not active or any(
            not torch.isfinite(parameter.grad).all() for parameter in active
        ):
            raise RuntimeError(
                f"training_missing_or_nonfinite_gradient: {arm} {task} step={index}"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active, pilot.gradient_clip, error_if_nonfinite=True
        )
        if gradient_norm == 0:
            raise RuntimeError(f"training_zero_gradient: {arm} {task} step={index}")
        optimizer.step()
        if any(
            not torch.isfinite(parameter).all()
            for parameter in adaptation.parameters.values()
        ):
            raise RuntimeError(
                f"training_nonfinite_parameter: {arm} {task} step={index}"
            )
        sync_device(adaptation.base)
        stats["optimizer_seconds"] += time.perf_counter() - started
        stats["steps"] = index + 1
        stats["examples_seen"] += len(batch)
        stats["answer_tokens_seen"] += int((labels[:, 1:] != -100).sum())
        stats["input_tokens_seen"] += int(mask.sum())
        stats["losses"].append(float(loss.detach()))
        stats["gradient_norms"].append(float(gradient_norm))
        log.event(
            "optimizer_step",
            arm=arm,
            task=task,
            step=index + 1,
            loss=float(loss.detach()),
            grad_norm=float(gradient_norm),
            examples_seen=stats["examples_seen"],
        )
        if (
            (index + 1) % pilot.checkpoint_every == 0
            or index + 1 == pilot.steps_per_task
            or log.stopping
        ):
            save_checkpoint(
                checkpoint,
                adaptation,
                optimizer,
                index + 1,
                stats,
                log,
                task,
                start_digest,
            )
        log.check_stop()
    adaptation.verify_frozen()
    stats["trainable_state_sha256"] = tensor_digest(adaptation.state())
    if stats["trainable_state_sha256"] == start_digest:
        raise RuntimeError(f"training_did_not_change_parameters: {arm} {task}")
    atomic_json(directory / "training.json", stats)
    return stats


def wilson_interval(accuracy, count):
    z = 1.959963984540054
    divisor = 1 + z * z / count
    center = (accuracy + z * z / (2 * count)) / divisor
    half = (
        z
        * math.sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count))
        / divisor
    )
    return [center - half, center + half]


def evaluate(adaptation, base, data, split, pilot):
    evaluator = PortalEvaluator(
        max_prompt=pilot.max_prompt_tokens, batch_size=pilot.eval_batch_size
    )
    base.model.eval()
    with torch.no_grad(), adaptation.activation() if adaptation else nullcontext():
        result = evaluator.evaluate(base, evaluation_dataset(data, split)).to_dict()
    for value in result["tasks"].values():
        value["accuracy_wilson_95"] = wilson_interval(
            value["accuracy"], value["examples"]
        )
    return result


def measure(adaptation, base, data, pilot, log, name):
    log.check_stop()
    sync_device(base)
    started = time.perf_counter()
    result = {
        split: evaluate(adaptation, base, data, split, pilot)
        for split in ("validation", "test")
    }
    sync_device(base)
    result["seconds"] = time.perf_counter() - started
    log.metrics["measurements"][name] = result
    log.metrics["runtime"]["gpu_execution"] = base.device.type == "cuda"
    log.event(
        "evaluation_completed",
        measurement=name,
        test_accuracy={
            task: value["accuracy"] for task, value in result["test"]["tasks"].items()
        },
    )
    log.check_stop()
    return result


def run_arm(base, portal, kind, initial_latent, data, pilot, log, name, seed):
    directory = log.output / name
    directory.mkdir(parents=True, exist_ok=True)
    result = {"kind": kind, "snapshots": {}, "training": {}}
    with Adaptation(base, portal, kind, initial_latent, seed) as adaptation:
        result["trainable_parameters"] = sum(
            parameter.numel() for parameter in adaptation.parameters.values()
        )
        for stage, task in (
            ("initial", None),
            ("after_a", "acquisition_a"),
            ("after_b", "acquisition_b"),
        ):
            if task is not None:
                result["training"][task] = fit_stage(
                    adaptation,
                    data.train[task],
                    task,
                    name,
                    pilot,
                    log,
                    seed + (101 if task == "acquisition_a" else 202),
                )
            state = adaptation.state()
            save_file(state, directory / f"{stage}.safetensors")
            result["snapshots"][stage] = measure(
                adaptation, adaptation.base, data, pilot, log, f"{name}/{stage}"
            )
        adaptation.verify_frozen()
    atomic_json(directory / "result.json", result)
    return result


def learning_deltas(arm, no_adapter):
    snapshots = arm["snapshots"]
    initial = snapshots["initial"]["test"]["tasks"]
    after_a = snapshots["after_a"]["test"]["tasks"]
    after_b = snapshots["after_b"]["test"]["tasks"]
    raw = no_adapter["test"]["tasks"]
    return {
        "a_acquisition_accuracy_delta_vs_initial": after_a["acquisition_a"]["accuracy"]
        - initial["acquisition_a"]["accuracy"],
        "a_acquisition_accuracy_delta_vs_no_adapter": after_a["acquisition_a"][
            "accuracy"
        ]
        - raw["acquisition_a"]["accuracy"],
        "a_gold_nll_reduction_vs_initial": initial["acquisition_a"]["gold_nll"]
        - after_a["acquisition_a"]["gold_nll"],
        "b_acquisition_accuracy_delta": after_b["acquisition_b"]["accuracy"]
        - after_a["acquisition_b"]["accuracy"],
        "a_retention_accuracy_delta_after_b": after_b["acquisition_a"]["accuracy"]
        - after_a["acquisition_a"]["accuracy"],
        "a_retention_nll_increase_after_b": after_b["acquisition_a"]["gold_nll"]
        - after_a["acquisition_a"]["gold_nll"],
        "control_accuracy_delta_after_b_vs_no_adapter": after_b["control_copy"][
            "accuracy"
        ]
        - raw["control_copy"]["accuracy"],
        "positive_acquisition_observed": (
            after_a["acquisition_a"]["accuracy"] > initial["acquisition_a"]["accuracy"]
            and after_a["acquisition_a"]["accuracy"] > raw["acquisition_a"]["accuracy"]
        ),
        "uncertainty": "Per-cell Wilson intervals are reported. Small pilot deltas are descriptive, without significance claims.",
    }


def verify_shared_core(source, target):
    if source.config.architecture_kwargs() != target.config.architecture_kwargs():
        raise CapabilityBlocked("source_target_canonical_architecture_mismatch")
    if tensor_digest(source.core.state_dict()) != tensor_digest(
        target.core.state_dict()
    ):
        raise CapabilityBlocked(
            "source_target_canonical_core_weights_differ: copying a latent would not be valid transfer"
        )


def source_phase(row, spec, config, data, pilot, device, log):
    portal = load_port(row, config, log)
    base = load_local_base(
        row, spec["model_path"], device, log, pilot.gradient_checkpointing
    )
    initial_latent = portal.task_latents.detach().mean(dim=0).cpu()
    raw = measure(None, base, data, pilot, log, "source_no_adapter")
    latent_result = run_arm(
        base,
        portal,
        "latent",
        initial_latent,
        data,
        pilot,
        log,
        "source_latent",
        config["seed"],
    )
    result = {
        "no_adapter": raw,
        "source_latent": latent_result,
        "learning": learning_deltas(latent_result, raw),
    }
    if pilot.source_native_lora:
        result["source_native_lora"] = run_arm(
            base,
            portal,
            "native_lora",
            initial_latent,
            data,
            pilot,
            log,
            "source_native_lora",
            config["seed"],
        )
    portal.to("cpu")
    del base
    return result, portal, initial_latent


def target_phase(row, config, source_portal, initial_latent, data, pilot, device, log):
    portal = load_port(row, config, log)
    verify_shared_core(source_portal, portal)
    base = load_local_base(
        row, config["model_path"], device, log, pilot.gradient_checkpointing
    )
    raw = measure(None, base, data, pilot, log, "target_no_adapter")
    transfer_result = {
        "snapshots": {},
        "new_task_target_optimizer_steps": 0,
        "new_task_target_training_examples": 0,
    }
    with Adaptation(
        base, portal, "latent", initial_latent, config["seed"]
    ) as adaptation:
        for stage in ("initial", "after_a", "after_b"):
            state = load_file(log.output / "source_latent" / f"{stage}.safetensors")
            adaptation.restore(state)
            before = tensor_digest(adaptation.state())
            transfer_result["snapshots"][stage] = measure(
                adaptation, base, data, pilot, log, f"portable_transfer/{stage}"
            )
            if tensor_digest(adaptation.state()) != before:
                raise RuntimeError("target_transfer_modified_source_latent")
        adaptation.verify_frozen()
    target_latent = run_arm(
        base,
        portal,
        "latent",
        initial_latent,
        data,
        pilot,
        log,
        "target_native_latent",
        config["seed"],
    )
    target_lora = run_arm(
        base,
        portal,
        "native_lora",
        initial_latent,
        data,
        pilot,
        log,
        "target_native_lora",
        config["seed"],
    )
    result = {
        "no_adapter": raw,
        "portable_transfer": transfer_result,
        "native_latent": target_latent,
        "native_lora": target_lora,
        "transfer_learning": learning_deltas(transfer_result, raw),
        "native_latent_learning": learning_deltas(target_latent, raw),
        "native_lora_learning": learning_deltas(target_lora, raw),
    }
    result["transfer_vs_native"] = {
        stage: {
            task: {
                "transfer_accuracy": transfer_result["snapshots"][stage]["test"][
                    "tasks"
                ][task]["accuracy"],
                "minus_native_latent": transfer_result["snapshots"][stage]["test"][
                    "tasks"
                ][task]["accuracy"]
                - target_latent["snapshots"][stage]["test"]["tasks"][task]["accuracy"],
                "minus_native_lora": transfer_result["snapshots"][stage]["test"][
                    "tasks"
                ][task]["accuracy"]
                - target_lora["snapshots"][stage]["test"]["tasks"][task]["accuracy"],
            }
            for task in ("acquisition_a", "acquisition_b", "control_copy")
        }
        for stage in ("after_a", "after_b")
    }
    return result


def transfer(config, log):
    registry = registry_from(config)
    if not registry["shared_core_identical_across_all_releases"]:
        raise CapabilityBlocked("registry_shared_core_identity_not_verified")
    source_spec = config["source"]
    source_row = select_port(registry, source_spec["model_id"], source_spec["revision"])
    target_row = select_port(registry, config["model_id"], config["revision"])
    if source_row["key"] == target_row["key"]:
        raise ValueError(
            "cross_model_transfer_requires_distinct_source_and_target_bases"
        )
    for row in (source_row, target_row):
        blocked = [
            blocker
            for blocker in row["status"]["blockers"]
            if blocker["code"] != "gated_base"
        ]
        if blocked:
            raise CapabilityBlocked(json.dumps(blocked))
    pilot = Pilot(**config.get("pilot", {}))
    data = make_tasks(
        config["seed"],
        pilot.train_examples,
        pilot.validation_examples,
        pilot.test_examples,
    )
    atomic_json(log.output / "data.json", data.as_dict())
    log.metrics["provenance"]["data"] = data.provenance
    log.metrics["scientific_contract"] = {
        "method": "novel_latent_learning_through_released_frozen_ports",
        "new_task_name": "one shared continual latent; A then B update the same weights",
        "frozen": [
            "all base weights",
            "released canonical cores",
            "released base alignments",
        ],
        "zero_target_training_for_transfer": True,
        "direct_lora_portability_assumed": False,
        "prior_port_training": "Released alignments and core were trained on the fourteen published benchmark tasks; zero target training refers only to the two new pilot tasks.",
        "latent_initialization": "mean of the fourteen released task latents, shared across every arm",
        "fixed_training_budget": {
            "steps_per_task_per_trained_arm": pilot.steps_per_task,
            "examples_per_task_per_trained_arm": pilot.steps_per_task
            * pilot.batch_size,
        },
        "budget_matching": "identical example order, batch size and optimizer steps; native latent also matches trainable parameter count",
        "unmatched_budget_dimensions": [
            "LoRA trainable parameter count",
            "token counts across tokenizers",
            "wall time and FLOPs across model sizes",
        ],
        "test_use": "fixed predeclared initial/after-A/after-B measurements; no test-driven selection",
        "limitations": [
            "Novel task acquisition in the frozen released latent space is unproven until measured.",
            "Transferred performance is not successful new learning unless source acquisition is observed.",
            "Explicit task commands select behavior, but no training examples or hidden mapping are supplied during evaluation.",
            "Four-choice accuracy does not establish unconstrained generation proficiency.",
        ],
    }
    device = ensure_gpu(config, log)
    log.check_stop()
    source_result, source_portal, initial = source_phase(
        source_row, source_spec, config, data, pilot, device, log
    )
    log.metrics["source"] = source_result
    log.event("source_phase_completed", source_learning=source_result["learning"])
    gc.collect()
    torch.cuda.empty_cache()
    target_result = target_phase(
        target_row, config, source_portal, initial, data, pilot, device, log
    )
    log.metrics["target"] = target_result
    source_acquired = source_result["learning"]["positive_acquisition_observed"]
    transferred = target_result["transfer_learning"]["positive_acquisition_observed"]
    log.metrics["acquisition_transfer_status"] = {
        "source_acquisition_point_estimate_positive": source_acquired,
        "target_transfer_point_estimate_positive": transferred,
        "status": "candidate_positive_transfer_requires_replication"
        if source_acquired and transferred
        else "no_positive_transfer_demonstrated",
        "criterion": "Heldout A accuracy after source A training exceeds both no-adapter and initial-latent accuracy on each base; this point-estimate gate is not a significance test.",
    }
    log.metrics["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    log.metrics["claim_scope"] = (
        "Measured pilot of new shared-latent acquisition, sequential interference and transfer through released ports; not a reproduction of Ramp's benchmark claims."
    )
