from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from portallib import PortalModel
from safetensors.torch import load_file, save_file

from compare import BOUNDARIES
from data import SEQUENCE_TASKS, digest, prepare_data, read_data, write_json
from learner import (
    MAX_PARAMETER_MISMATCH,
    NUMERICAL_GATE,
    SequenceLearner,
    compare_probes,
    evaluate,
    extend_portal,
    frozen_base_tensors,
    initialization_parity,
    load_base,
    load_portal,
    make_persistent_lora,
    parameter_plan,
    probe_logits,
    save_native,
    shared_hash,
    tensor_hash,
    tensor_storage,
    transplant,
)
from metrics import SELECTION_RULE, sequence_metrics, transport_metrics

ROOT = Path(__file__).resolve().parent
COUNT_KEYS = ("examples", "supervised_tokens", "input_tokens", "padded_tokens")


def emit(output: Path, event: str, **fields: object) -> None:
    row = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    line = json.dumps(row, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    with (output / "events.jsonl").open("a") as stream:
        stream.write(line + "\n")


def validate_config(config: dict) -> None:
    expected = {
        "arm",
        "seed",
        "method",
        "source",
        "train_model",
        "transport_model",
        "models",
        "sequence",
        "training",
        "evaluation",
    }
    if set(config) != expected:
        raise ValueError(
            f"CONFIG_KEYS: expected {sorted(expected)}, got {sorted(config)}"
        )
    if config["method"] not in {"native", "lora"}:
        raise ValueError("UNKNOWN_METHOD")
    recipe = config["training"]
    if recipe["train_core"] is not (config["method"] == "native"):
        raise ValueError("TRAINABLE_CORE_METHOD_MISMATCH")
    if recipe["replay_weight"] not in (0.0, 0.25):
        raise ValueError("UNDECLARED_REPLAY_WEIGHT")
    fixed = {
        "steps": 96,
        "batch_size": 4,
        "replay_batch_size": 4,
        "replay_every": 4,
        "eval_every": 24,
        "grad_clip": 1.0,
        "capacity_relative_tolerance": MAX_PARAMETER_MISMATCH,
        "core_lr": 3e-5,
        "latent_lr": 0.002,
        "lora_lr": 0.0002,
        "max_prompt": 768,
    }
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise ValueError("UNDECLARED_SEQUENCE_TRAINING_RECIPE")
    if set(recipe) != set(fixed) | {"replay_weight", "train_core"}:
        raise ValueError("TRAINING_CONFIG_KEYS")
    if config["evaluation"] != {"batch_size": 8}:
        raise ValueError("UNDECLARED_EVALUATION_RECIPE")
    if (config["source"], config["train_model"], config["transport_model"]) != (
        "qwen4",
        "qwen8",
        "qwen17",
    ):
        raise ValueError("SEQUENCE_BACKBONE_CONTRACT_MISMATCH")
    fixture = config["sequence"]
    seeds = {"primary": (303, 30301), "reserved_confirmation": (419, 41901)}
    if seeds.get(fixture["role"]) != (fixture["fixture_seed"], config["seed"]):
        raise ValueError("SEQUENCE_FIXTURE_SEED_MISMATCH")
    if set(config["models"]) != {"qwen4", "qwen8", "qwen17"}:
        raise ValueError("MODEL_CATALOG_MISMATCH")
    for model in config["models"].values():
        if set(model) != {"base", "portal"}:
            raise ValueError("MODEL_SPEC_KEYS")
        for spec in model.values():
            if set(spec) != {"repo_id", "revision", "local_path", "cache_dir"}:
                raise ValueError("MODEL_PIN_KEYS")
            if not isinstance(spec["local_path"], str) or not spec["local_path"]:
                raise ValueError("LOCAL_MODEL_SNAPSHOT_REQUIRED")
            if len(spec["revision"]) != 40 or any(
                c not in "0123456789abcdef" for c in spec["revision"]
            ):
                raise ValueError(f"UNPINNED_MODEL: {spec['repo_id']}")


def require_gpu(output: Path, seed: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "GPU_CONTRACT: parent must expose exactly one GPU to this process"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    emit(
        output,
        "runtime",
        packages={
            name: version(name)
            for name in ("torch", "peft", "transformers", "portallib")
        },
        gpu=torch.cuda.get_device_name(0),
        hip=torch.version.hip,
        visible_gpus=torch.cuda.device_count(),
        requested_base_dtype="float32",
        trainable_dtype="float32",
        base_forward_autocast=False,
        native_core_generation_autocast=False,
    )


class Cycle:
    def __init__(self, rows: list, seed: int):
        if not rows:
            raise ValueError("EMPTY_EXPOSURE_CYCLE")
        self.rows = rows
        self.rng = random.Random(seed)
        self.pending: list = []

    def draw(self, count: int) -> list:
        result = []
        while len(result) < count:
            if not self.pending:
                self.pending = self.rows.copy()
                self.rng.shuffle(self.pending)
            result.append(self.pending.pop())
        return result


def stage_schedule(data: dict, stage: int, seed: int, recipe: dict):
    task = SEQUENCE_TASKS[stage]
    current = Cycle(data[f"{task}_train"], seed + stage * 97)
    previous = {
        name: Cycle(data[f"{name}_train"], seed + 100003 + stage * 997 + index)
        for index, name in enumerate(SEQUENCE_TASKS[:stage])
    }
    for step in range(1, recipe["steps"] + 1):
        rows = current.draw(recipe["batch_size"])
        references = []
        if previous and step % recipe["replay_every"] == 0:
            count = recipe["replay_batch_size"] // len(previous)
            if count * len(previous) != recipe["replay_batch_size"]:
                raise ValueError("UNBALANCED_REFERENCE_BATCH")
            for cycle in previous.values():
                references.extend(cycle.draw(count))
        identity = {
            "stage": stage,
            "task": task,
            "step": step,
            "current_ids": [row.id for row in rows],
            "reference_ids": [row.id for row in references],
        }
        yield rows, references, identity


def initial_portal(config: dict) -> tuple[PortalModel, dict]:
    source = load_portal(config["models"][config["source"]]["portal"])
    target = load_portal(config["models"][config["train_model"]]["portal"])
    source_hash = shared_hash(source)
    if source_hash != shared_hash(target):
        raise ValueError("PUBLISHED_SHARED_STATE_MISMATCH")
    initial = extend_portal(transplant(source, target), SEQUENCE_TASKS)
    return initial, {
        "source_shared_sha256": source_hash,
        "source_core_sha256": tensor_hash(source.core.state_dict()),
        "train_alignment_sha256": tensor_hash(target.alignment.state_dict()),
        "published_vector_sha256": tensor_hash({"published": source.task_latents}),
        "initial_vector_source": "rte",
    }


def adaptation_state(learner: SequenceLearner) -> dict:
    if learner.portal is not None:
        return learner.portal.state_dict()
    return get_peft_model_state_dict(learner.base.model, save_embedding_layers=False)


def frozen_snapshot(learner: SequenceLearner) -> dict:
    result = {"base_sha256": tensor_hash(frozen_base_tensors(learner.base.model))}
    if learner.portal is not None:
        portal = learner.portal
        result.update(
            {
                "alignment_sha256": tensor_hash(portal.alignment.state_dict()),
                "published_vectors_sha256": tensor_hash(
                    {"published": portal.task_latents[: -len(SEQUENCE_TASKS)]}
                ),
                "inactive_vectors": {
                    task: tensor_hash({"vector": vector})
                    for task, vector in learner.vectors.items()
                    if not vector.requires_grad
                },
                "inactive_vector_optimizer": learner.inactive_optimizer_hashes(),
            }
        )
    return result


def save_checkpoint(learner: SequenceLearner, directory: Path, rows: list) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    learner.sync_vectors()
    checkpoint = directory / "checkpoint"
    if learner.portal is not None:
        save_native(learner.portal, checkpoint)
    else:
        learner.base.model.save_pretrained(checkpoint, save_embedding_layers=False)
    optimizer = learner.save_optimizer(directory / "optimizer.pt")
    probes = probe_logits(learner.base, rows, learner.portal)
    save_file(probes, directory / "probe_logits.safetensors")
    write_json(directory / "probe_ids.json", [row.id for row in rows])
    state = adaptation_state(learner)
    storage = tensor_storage(state)
    storage["serialized_checkpoint_bytes"] = sum(
        path.stat().st_size for path in checkpoint.iterdir() if path.is_file()
    )
    storage["serialized_optimizer_bytes"] = (directory / "optimizer.pt").stat().st_size
    storage["active_trainable_elements"] = sum(
        value.numel() for value in learner.active.values()
    )
    if learner.portal is not None:
        portal = learner.portal
        storage["native_vector_state"] = {
            "published_vectors": len(portal.config.tasks) - len(SEQUENCE_TASKS),
            "published_elements": portal.task_latents[: -len(SEQUENCE_TASKS)].numel(),
            "retained_sequence_vectors": len(SEQUENCE_TASKS),
            "retained_sequence_elements": len(SEQUENCE_TASKS) * portal.config.d_z,
            "active_sequence_elements": portal.config.d_z,
            "frozen_sequence_elements": (len(SEQUENCE_TASKS) - 1) * portal.config.d_z,
        }
    else:
        storage["native_vector_state"] = None
    receipt = {
        "pid": os.getpid(),
        "stage": learner.stage,
        "checkpoint_sha256": tensor_hash(state),
        "probe_sha256": tensor_hash(probes),
        "optimizer": optimizer,
        "storage": storage,
        "frozen": frozen_snapshot(learner),
    }
    write_json(directory / "checkpoint_receipt.json", receipt)
    return receipt


def learner_recipe(config: dict, plan: dict) -> dict:
    return {**config["training"], "inherited_rank": plan["published_rank"]}


def evaluation_rows(data: dict, split: str, tasks: tuple = SEQUENCE_TASKS) -> list:
    return [row for task in tasks for row in data[f"{task}_{split}"]]


def eval_rows(base, portal, rows: list, config: dict) -> dict:
    return evaluate(
        base,
        rows,
        config["training"]["max_prompt"],
        config["evaluation"]["batch_size"],
        portal,
    )


def initial_paths(output: Path, method: str) -> Path:
    return output / "initial" / method


def initialize(config: dict, output: Path) -> None:
    if (output / "initial").exists():
        raise ValueError("INITIALIZATION_ARTIFACTS_ALREADY_EXIST")
    require_gpu(output, config["seed"])
    split_names = tuple(f"{task}_validation" for task in SEQUENCE_TASKS)
    data = read_data(output, split_names)
    validation = evaluation_rows(data, "validation")
    probes = [
        data[f"{task}_validation"][index] for task in SEQUENCE_TASKS for index in (0, 1)
    ]
    portal, provenance = initial_portal(config)
    plan = parameter_plan(portal)
    base = load_base(config["models"][config["train_model"]]["base"])
    portal.to(base.device)
    arithmetic = {
        "base": tensor_storage(frozen_base_tensors(base.model)),
        "native": tensor_storage(portal.state_dict()),
        "forward_autocast": False,
        "probes": {"native": {}, "lora": {}},
    }
    before_base = tensor_hash(frozen_base_tensors(base.model))
    native_validation = eval_rows(base, portal, validation, config)
    native_probes = probe_logits(
        base, probes, portal, arithmetic=arithmetic["probes"]["native"]
    )
    native = SequenceLearner(
        base, portal, SEQUENCE_TASKS, 0, learner_recipe(config, plan)
    )
    native_checkpoint = save_checkpoint(native, initial_paths(output, "native"), probes)
    native.close()
    del native
    portal.requires_grad_(False)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    peft_model, embedding = make_persistent_lora(
        base, portal, output / "initial" / "exported_rte"
    )
    adapted = replace(base, model=peft_model)
    arithmetic["lora"] = tensor_storage(
        get_peft_model_state_dict(peft_model, save_embedding_layers=False)
    )
    lora_validation = eval_rows(adapted, None, validation, config)
    lora_probes = probe_logits(
        adapted, probes, None, arithmetic=arithmetic["probes"]["lora"]
    )
    emit(output, "initialization_arithmetic", **arithmetic)
    parity = initialization_parity(
        native_validation, lora_validation, native_probes, lora_probes
    )
    write_json(
        output / "initialization_parity.json",
        {"embedding": embedding, "storage_arithmetic": arithmetic, **parity},
    )
    save_file(
        native_probes, output / "initial" / "native_validation_probes.safetensors"
    )
    save_file(lora_probes, output / "initial" / "lora_validation_probes.safetensors")
    write_json(
        output / "initial" / "validation_predictions.json",
        {"native": native_validation, "lora": lora_validation},
    )
    if not parity["passed"]:
        raise ValueError(
            "INITIALIZATION_PARITY_GATE_FAILED: inspect initialization_parity.json; do not relax thresholds"
        )
    lora = SequenceLearner(
        adapted, None, SEQUENCE_TASKS, 0, learner_recipe(config, plan)
    )
    lora_checkpoint = save_checkpoint(lora, initial_paths(output, "lora"), probes)
    lora.close()
    after_base = tensor_hash(frozen_base_tensors(adapted.model))
    if after_base != before_base:
        raise ValueError("INITIAL_EVALUATION_CHANGED_BASE")
    portal.cpu()
    del lora, adapted, peft_model, base, portal
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "pid": os.getpid(),
        "provenance": provenance,
        "parameter_match": plan,
        "parity": parity,
        "embedding": embedding,
        "storage_arithmetic": arithmetic,
        "checkpoints": {"native": native_checkpoint, "lora": lora_checkpoint},
        "heldout_reads": [],
    }
    write_json(output / "initialization.json", result)
    write_json(
        output / "initialization_receipt.json",
        {
            "parity": {"passed": parity["passed"], "thresholds": parity["thresholds"]},
            "parameter_match": plan,
            "provenance": provenance,
        },
    )
    emit(
        output,
        "initialization_qualified",
        parity_passed=True,
        relative_parameter_difference=plan["relative_difference"],
    )


def evaluate_baselines(config: dict, output: Path) -> None:
    require_gpu(output, config["seed"])
    training = json.loads((output / "training_receipt.json").read_text())
    if training["pid"] == os.getpid() or not all(training["checks"].values()):
        raise ValueError("BASELINES_REQUIRE_COMPLETED_SEPARATE_TRAINER")
    if (output / "baselines.json").exists():
        raise ValueError("BASELINE_ARTIFACTS_ALREADY_EXIST")
    initialization = json.loads((output / "initialization.json").read_text())
    data = read_data(output, tuple(f"{task}_test" for task in SEQUENCE_TASKS))
    tests = evaluation_rows(data, "test")
    base = load_base(config["models"][config["train_model"]]["base"])
    base_hash = tensor_hash(frozen_base_tensors(base.model))
    if base_hash != initialization["checkpoints"]["native"]["frozen"]["base_sha256"]:
        raise ValueError("RAW_BASELINE_IS_NOT_PRISTINE")
    raw = eval_rows(base, None, tests, config)
    native = PortalModel.from_pretrained(
        initial_paths(output, "native") / "checkpoint",
        device=str(base.device),
        dtype=torch.float32,
    ).requires_grad_(False)
    native_hash = tensor_hash(native.state_dict())
    native_vector_hash = tensor_hash({"task_vectors": native.task_latents})
    if native_hash != initialization["checkpoints"]["native"]["checkpoint_sha256"]:
        raise ValueError("NATIVE_BASELINE_INITIAL_STATE_MISMATCH")
    native_initial = eval_rows(base, native, tests, config)
    if native_hash != tensor_hash(native.state_dict()):
        raise ValueError("NATIVE_BASELINE_CHANGED_INITIAL_STATE")
    native.cpu()
    del native
    model = PeftModel.from_pretrained(
        base.model,
        initial_paths(output, "lora") / "checkpoint",
        is_trainable=False,
        autocast_adapter_dtype=True,
    )
    adapted = replace(base, model=model)
    lora_hash = tensor_hash(
        get_peft_model_state_dict(model, save_embedding_layers=False)
    )
    if lora_hash != initialization["checkpoints"]["lora"]["checkpoint_sha256"]:
        raise ValueError("LORA_BASELINE_INITIAL_STATE_MISMATCH")
    lora_initial = eval_rows(adapted, None, tests, config)
    if lora_hash != tensor_hash(
        get_peft_model_state_dict(model, save_embedding_layers=False)
    ) or base_hash != tensor_hash(frozen_base_tensors(model)):
        raise ValueError("LORA_BASELINE_CHANGED_INITIAL_STATE")
    del adapted, model, base
    gc.collect()
    torch.cuda.empty_cache()
    target = load_portal(config["models"][config["transport_model"]]["portal"])
    if shared_hash(target) != initialization["provenance"]["source_shared_sha256"]:
        raise ValueError("TARGET_PUBLISHED_SHARED_STATE_MISMATCH")
    target_initial = extend_portal(target, SEQUENCE_TASKS)
    target_initial_hash = tensor_hash(target_initial.state_dict())
    target_base = load_base(config["models"][config["transport_model"]]["base"])
    target_initial.to(target_base.device)
    target_before = tensor_hash(frozen_base_tensors(target_base.model))
    target_raw = eval_rows(target_base, None, tests, config)
    target_initialized = eval_rows(target_base, target_initial, tests, config)
    if target_before != tensor_hash(
        frozen_base_tensors(target_base.model)
    ) or target_initial_hash != tensor_hash(target_initial.state_dict()):
        raise ValueError("TARGET_BASELINE_CHANGED_INITIAL_STATE")
    result = {
        "pid": os.getpid(),
        "trainer_pid": training["pid"],
        "post_training": True,
        "baselines": {
            "train": {
                "raw": raw,
                "initial": native_initial
                if config["method"] == "native"
                else lora_initial,
            },
            "target": {"raw": target_raw, "initial": target_initialized},
        },
        "method_initializations": {"native": native_initial, "lora": lora_initial},
        "initial_checkpoint_identity": {
            "native_sha256": native_hash,
            "native_vectors_sha256": native_vector_hash,
            "lora_sha256": lora_hash,
            "raw_base_sha256": base_hash,
            "target_initial_sha256": target_initial_hash,
        },
        "target_base_sha256": target_before,
        "target_alignment_sha256": tensor_hash(target.alignment.state_dict()),
        "target_calibration_examples": 0,
    }
    write_json(output / "baselines.json", result)
    emit(
        output,
        "post_training_baselines_complete",
        initial_checkpoint_identity=result["initial_checkpoint_identity"],
    )


def restore_learner(
    config: dict, output: Path, directory: Path, stage: int
) -> SequenceLearner:
    initialization = json.loads((output / "initialization_receipt.json").read_text())
    base = load_base(config["models"][config["train_model"]]["base"])
    if config["method"] == "native":
        portal = PortalModel.from_pretrained(
            directory / "checkpoint", device=str(base.device), dtype=torch.float32
        )
    else:
        portal = None
        base = replace(
            base,
            model=PeftModel.from_pretrained(
                base.model,
                directory / "checkpoint",
                is_trainable=True,
                autocast_adapter_dtype=True,
            ),
        )
    learner = SequenceLearner(
        base,
        portal,
        SEQUENCE_TASKS,
        stage,
        learner_recipe(config, initialization["parameter_match"]),
    )
    expected = json.loads((directory / "checkpoint_receipt.json").read_text())
    if tensor_hash(adaptation_state(learner)) != expected["checkpoint_sha256"]:
        raise ValueError("CHECKPOINT_STATE_RELOAD_MISMATCH")
    learner.load_optimizer(directory / "optimizer.pt")
    if frozen_snapshot(learner) != expected["frozen"]:
        raise ValueError("CHECKPOINT_FROZEN_STATE_RELOAD_MISMATCH")
    return learner


def check_optimizer_steps(learner: SequenceLearner, updates_per_stage: int) -> bool:
    steps = learner.optimizer_receipt()["steps"]
    for name, step in steps.items():
        if name.startswith("vector."):
            index = SEQUENCE_TASKS.index(name.removeprefix("vector."))
            expected = updates_per_stage if index <= learner.stage else 0
        else:
            expected = updates_per_stage * (learner.stage + 1)
        if step != expected:
            raise ValueError(f"OPTIMIZER_CONTINUITY_FAILED: {name}: {step}/{expected}")
    return True


def aggregate_steps(rows: list[dict]) -> dict:
    result = {
        category: {key: sum(row[category][key] for row in rows) for key in COUNT_KEYS}
        for category in (
            "current_computed",
            "reference_computed",
            "reference_effective",
        )
    }
    for key in (
        "optimizer_updates",
        "current_forward_backward_groups",
        "reference_forward_backward_groups",
        "reference_backward_calls",
    ):
        result[key] = sum(row[key] for row in rows)
    result["reference_batches"] = sum(
        bool(row["reference_computed"]["examples"]) for row in rows
    )
    result["reference_batches_with_nonzero_gradient"] = sum(
        row["reference_gradient_norm"] > 0 for row in rows
    )
    result["extra_rank_gradient_seen"] = {
        name: any(
            row["extra_rank_gradient_nonzero_elements"].get(name, 0) > 0 for row in rows
        )
        for name in {
            name for row in rows for name in row["extra_rank_gradient_nonzero_elements"]
        }
    }
    result["current_gradient_seen"] = {
        name: any(row["current_gradient_seen"].get(name, False) for row in rows)
        for name in {name for row in rows for name in row["current_gradient_seen"]}
    }
    return result


def train(config: dict, output: Path) -> None:
    if (output / "stages").exists():
        raise ValueError("TRAINING_ARTIFACTS_ALREADY_EXIST")
    require_gpu(output, config["seed"])
    initialization = json.loads((output / "initialization_receipt.json").read_text())
    if not initialization["parity"]["passed"]:
        raise ValueError("TRAINING_REQUIRES_INITIALIZATION_PARITY")
    learner = restore_learner(
        config, output, initial_paths(output, config["method"]), 0
    )
    recipe = config["training"]
    all_steps = []
    schedule = []
    stages = {}
    try:
        for stage, task in enumerate(SEQUENCE_TASKS):
            if stage:
                transition = learner.advance_stage()
                emit(
                    output,
                    "stage_transition",
                    stage=stage,
                    optimizer_state_unchanged=transition["optimizer_state_unchanged"],
                )
            names = tuple(
                f"{name}_{split}"
                for name in SEQUENCE_TASKS[: stage + 1]
                for split in ("train", "validation")
            )
            data = read_data(output, names)
            directory = output / "stages" / BOUNDARIES[stage + 1]
            directory.mkdir(parents=True, exist_ok=True)
            before = frozen_snapshot(learner)
            before_state = tensor_hash(adaptation_state(learner))
            before_optimizer = learner.optimizer_receipt()
            steps = []
            for current, reference, identity in stage_schedule(
                data, stage, config["seed"], recipe
            ):
                row = {
                    **identity,
                    **learner.step(current, reference, recipe["replay_weight"]),
                }
                steps.append(row)
                schedule.append(identity)
                with (directory / "updates.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                if identity["step"] % recipe["eval_every"] == 0:
                    validation = eval_rows(
                        learner.base,
                        learner.portal,
                        evaluation_rows(
                            data, "validation", SEQUENCE_TASKS[: stage + 1]
                        ),
                        config,
                    )
                    write_json(
                        directory / f"validation_{identity['step']}.json", validation
                    )
                    emit(
                        output,
                        "validation",
                        stage=stage,
                        task=task,
                        step=identity["step"],
                        metrics=validation["metrics"],
                    )
            after = frozen_snapshot(learner)
            if before != after:
                raise ValueError(f"FROZEN_STATE_CHANGED_IN_STAGE: {task}")
            if before_state == tensor_hash(adaptation_state(learner)):
                raise ValueError(f"NO_LEARNED_STATE_CHANGE: {task}")
            check_optimizer_steps(learner, recipe["steps"])
            probes = [
                data[f"{name}_validation"][index]
                for name in SEQUENCE_TASKS[: stage + 1]
                for index in (0, 1)
            ]
            checkpoint = save_checkpoint(learner, directory, probes)
            budget = aggregate_steps(steps)
            if config["method"] == "lora" and not all(
                budget["extra_rank_gradient_seen"].values()
            ):
                raise ValueError(f"DEAD_EXTRA_LORA_RANKS: {task}")
            stage_receipt = {
                "stage": stage,
                "task": task,
                "pid": os.getpid(),
                "data_reads": names,
                "schedule_sha256": digest(
                    [row for row in schedule if row["stage"] == stage]
                ),
                "budget": budget,
                "frozen_before": before,
                "frozen_after": after,
                "optimizer_before": before_optimizer,
                "checkpoint": checkpoint,
                "checks": {
                    "frozen_state_unchanged": before == after,
                    "learned_state_changed": before_state
                    != checkpoint["checkpoint_sha256"],
                    "optimizer_continuity": True,
                    "fixed_final_step": len(steps) == recipe["steps"],
                    "all_active_gradients_seen": all(
                        budget["current_gradient_seen"].values()
                    ),
                    "reference_computation_completed": budget[
                        "reference_backward_calls"
                    ]
                    == (recipe["steps"] // recipe["replay_every"] if stage else 0)
                    and budget["reference_forward_backward_groups"]
                    == stage * (recipe["steps"] // recipe["replay_every"]),
                    "active_capacity_matched": abs(
                        checkpoint["storage"]["active_trainable_elements"]
                        - initialization["parameter_match"]["native_active_parameters"]
                    )
                    / initialization["parameter_match"]["native_active_parameters"]
                    <= MAX_PARAMETER_MISMATCH,
                    "extra_lora_rank_gradients_live": all(
                        budget["extra_rank_gradient_seen"].values()
                    ),
                },
            }
            if not all(stage_receipt["checks"].values()):
                raise ValueError(
                    f"STAGE_MECHANICAL_QUALIFICATION_FAILED: {task}/{stage_receipt['checks']}"
                )
            write_json(directory / "training_receipt.json", stage_receipt)
            stages[BOUNDARIES[stage + 1]] = stage_receipt
            all_steps.extend(steps)
            emit(
                output,
                "stage_saved",
                stage=stage,
                task=task,
                checkpoint_sha256=checkpoint["checkpoint_sha256"],
            )
    finally:
        learner.close()
    budget = aggregate_steps(all_steps)
    expected_reference = (
        recipe["replay_batch_size"] * (recipe["steps"] // recipe["replay_every"]) * 2
    )
    checks = {
        "all_stage_checks": all(all(row["checks"].values()) for row in stages.values()),
        "single_training_process": len({row["pid"] for row in stages.values()}) == 1,
        "total_updates": budget["optimizer_updates"] == recipe["steps"] * 3,
        "total_current_examples": budget["current_computed"]["examples"]
        == recipe["steps"] * recipe["batch_size"] * 3,
        "total_computed_reference_examples": budget["reference_computed"]["examples"]
        == expected_reference,
        "total_effective_reference_examples": budget["reference_effective"]["examples"]
        == (expected_reference if recipe["replay_weight"] else 0),
    }
    receipt = {
        "pid": os.getpid(),
        "method": config["method"],
        "tasks": SEQUENCE_TASKS,
        "stages": stages,
        "budget": budget,
        "computed_schedule_sha256": digest(schedule),
        "checks": checks,
        "parameter_match": initialization["parameter_match"],
        "reference_gradient_policy": "Compute unweighted reference gradients in both conditions; discard before clipping at weight 0, add weight 0.25 before clipping otherwise; exactly one Adam step.",
        "parameterization_boundary": "Matched active scalar counts and scheduled examples/tokens/updates do not assert equal FLOPs or equal functional capacity.",
    }
    write_json(output / "computed_schedule.json", schedule)
    write_json(output / "training_receipt.json", receipt)
    if not all(checks.values()):
        raise ValueError("SEQUENCE_TRAINING_QUALIFICATION_FAILED")
    emit(
        output,
        "training_complete",
        computed_schedule_sha256=receipt["computed_schedule_sha256"],
        budget=budget,
    )


def reload_probes(
    learner: SequenceLearner, output: Path, directory: Path, tasks: tuple
) -> dict:
    expected_receipt = json.loads((directory / "checkpoint_receipt.json").read_text())
    data = read_data(output, tuple(f"{task}_validation" for task in tasks))
    by_id = {row.id: row for row in evaluation_rows(data, "validation", tasks)}
    wanted = json.loads((directory / "probe_ids.json").read_text())
    if len(wanted) != len(set(wanted)):
        raise ValueError("DUPLICATE_RELOAD_PROBE_IDS")
    actual = probe_logits(learner.base, [by_id[key] for key in wanted], learner.portal)
    expected = load_file(directory / "probe_logits.safetensors")
    comparison = compare_probes(actual, expected)
    receipt = {
        "trainer_pid": expected_receipt["pid"],
        "evaluator_pid": os.getpid(),
        "fresh_process": expected_receipt["pid"] != os.getpid(),
        "checkpoint_sha256": tensor_hash(adaptation_state(learner)),
        "optimizer_sha256": learner.optimizer_receipt()["state_sha256"],
        "optimizer_exact": learner.optimizer_receipt()["state_sha256"]
        == expected_receipt["optimizer"]["state_sha256"],
        "probe_logits": comparison,
        "probe_ids": wanted,
    }
    receipt["passed"] = (
        receipt["fresh_process"]
        and receipt["optimizer_exact"]
        and comparison["close"]
        and receipt["checkpoint_sha256"] == expected_receipt["checkpoint_sha256"]
    )
    if not receipt["passed"]:
        raise ValueError(f"FRESH_PROCESS_RELOAD_FAILED: {receipt}")
    return receipt


def evaluate_checkpoint(config: dict, output: Path, stage: int) -> None:
    require_gpu(output, config["seed"])
    boundary = BOUNDARIES[stage + 1]
    directory = output / "stages" / boundary
    if (directory / "evaluation.json").exists():
        raise ValueError("EVALUATION_ARTIFACT_ALREADY_EXISTS")
    learner = restore_learner(config, output, directory, stage)
    reload = reload_probes(learner, output, directory, SEQUENCE_TASKS[: stage + 1])
    learner.close()
    data = read_data(output, tuple(f"{task}_test" for task in SEQUENCE_TASKS))
    tests = evaluation_rows(data, "test")
    before = frozen_snapshot(learner)
    state_before = tensor_hash(adaptation_state(learner))
    evaluated = eval_rows(learner.base, learner.portal, tests, config)
    if (
        frozen_snapshot(learner) != before
        or tensor_hash(adaptation_state(learner)) != state_before
    ):
        raise ValueError("EVALUATION_CHANGED_SOURCE_STATE")
    result = {
        "stage": stage,
        "trained_task": SEQUENCE_TASKS[stage],
        "seen_tasks": SEQUENCE_TASKS[: stage + 1],
        "evaluated_tasks": SEQUENCE_TASKS,
        "evaluation": evaluated,
        "reload": reload,
        "task_routing": "Known task name in both prompts; native additionally selects that task vector."
        if learner.portal is not None
        else "One persistent LoRA adapter for all task-name prompts.",
    }
    if learner.portal is not None:
        learned = learner.portal.cpu().requires_grad_(False)
        learned_shared = shared_hash(learned)
        del learner
        gc.collect()
        torch.cuda.empty_cache()
        initialization = json.loads((output / "baselines.json").read_text())
        target = load_portal(config["models"][config["transport_model"]]["portal"])
        transported = transplant(learned, target)
        if shared_hash(transported) != learned_shared:
            raise ValueError("TRANSPORT_CHANGED_SHARED_STATE")
        alignment = tensor_hash(target.alignment.state_dict())
        save_native(transported, directory / "transport_checkpoint")
        saved = PortalModel.from_pretrained(
            directory / "transport_checkpoint", dtype=torch.float32
        )
        if tensor_hash(saved.state_dict()) != tensor_hash(transported.state_dict()):
            raise ValueError("TRANSPORT_SERIALIZATION_MISMATCH")
        target_base = load_base(config["models"][config["transport_model"]]["base"])
        transported.to(target_base.device)
        target_before = tensor_hash(frozen_base_tensors(target_base.model))
        transported_before = tensor_hash(transported.state_dict())
        target_evaluation = eval_rows(target_base, transported, tests, config)
        checks = {
            "target_base_unchanged": target_before
            == tensor_hash(frozen_base_tensors(target_base.model))
            == initialization["target_base_sha256"],
            "target_alignment_original": tensor_hash(transported.alignment.state_dict())
            == alignment
            == initialization["target_alignment_sha256"],
            "shared_core_and_vectors_exact": shared_hash(transported) == learned_shared,
            "transport_state_unchanged": tensor_hash(transported.state_dict())
            == transported_before,
            "no_trainable_target_parameters": all(
                not value.requires_grad for value in transported.parameters()
            )
            and all(
                not value.requires_grad for value in target_base.model.parameters()
            ),
        }
        result["transport"] = {
            "measured": True,
            "target": config["transport_model"],
            "calibration_examples": 0,
            "target_optimizer_updates": 0,
            "evaluated_tasks": SEQUENCE_TASKS,
            "seen_tasks": SEQUENCE_TASKS[: stage + 1],
            "evaluation": target_evaluation,
            "source_shared_sha256": learned_shared,
            "target_alignment_sha256": alignment,
            "checks": checks,
        }
        if not all(checks.values()):
            raise ValueError(f"TRANSPORT_QUALIFICATION_FAILED: {checks}")
    else:
        result["transport"] = {
            "measured": False,
            "reason": "Direct 8B LoRA to 1.7B transfer is undefined for these projection shapes; source retention uses the same persistent adapter.",
        }
    write_json(directory / "evaluation.json", result)
    emit(
        output,
        "checkpoint_evaluated",
        boundary=boundary,
        reload_passed=reload["passed"],
    )


def finish(config: dict, output: Path) -> None:
    initialization = json.loads((output / "initialization.json").read_text())
    baselines = json.loads((output / "baselines.json").read_text())
    training = json.loads((output / "training_receipt.json").read_text())
    manifest = json.loads((output / "data_manifest.json").read_text())
    timeline = {
        "initial": {
            "seen_tasks": [],
            "evaluated_tasks": SEQUENCE_TASKS,
            "evaluation": baselines["baselines"]["train"]["initial"],
            "transport": {
                "evaluation": baselines["baselines"]["target"]["initial"],
                "calibration_examples": 0,
            },
        }
    }
    for boundary in BOUNDARIES[1:]:
        timeline[boundary] = json.loads(
            (output / "stages" / boundary / "evaluation.json").read_text()
        )
    evaluations = {
        "baselines": baselines["baselines"],
        "timeline": timeline,
        "initialization": initialization,
        "baseline_receipt": baselines,
    }
    write_json(output / "evaluation.json", evaluations)
    qualified = (
        initialization["parity"]["passed"]
        and baselines["post_training"]
        and baselines["trainer_pid"] == training["pid"]
        and baselines["pid"] != training["pid"]
        and all(training["checks"].values())
        and all(timeline[name]["reload"]["passed"] for name in BOUNDARIES[1:])
        and all(
            timeline[name]["reload"]["trainer_pid"] == training["pid"]
            for name in BOUNDARIES[1:]
        )
    )
    if config["method"] == "native":
        qualified = qualified and all(
            all(timeline[name]["transport"]["checks"].values())
            for name in BOUNDARIES[1:]
        )
    source = sequence_metrics(
        timeline, baselines["baselines"]["train"]["raw"], config["seed"]
    )
    transport = (
        transport_metrics(
            {
                boundary: {"evaluation": timeline[boundary]["transport"]["evaluation"]}
                for boundary in BOUNDARIES
            },
            baselines["baselines"]["target"]["raw"],
            config["seed"],
        )
        if config["method"] == "native"
        else {"measured": False, "reason": timeline["after_c"]["transport"]["reason"]}
    )
    result = {
        "arm": config["arm"],
        "method": config["method"],
        "replay_weight": config["training"]["replay_weight"],
        "qualified": qualified,
        "mechanically_qualified": qualified,
        "sequence_learning_signal": qualified and source["sequence_learning_signal"],
        "source": source,
        "transport": transport,
        "tasks": SEQUENCE_TASKS,
        "fixture": config["sequence"],
        "fixture_sha256": config["sequence"]["sha256"],
        "config_sha256": digest(config),
        "data_sha256": manifest["data_sha256"],
        "model_pins": config["models"],
        "computed_schedule_sha256": training["computed_schedule_sha256"],
        "parameter_match": training["parameter_match"],
        "budget": training["budget"],
        "storage_by_boundary": {
            name: training["stages"][name]["checkpoint"]["storage"]
            for name in BOUNDARIES[1:]
        },
        "parent_paired_review_required": True,
        "claim_boundary": "Three known tasks in one procedural family after one pinned backbone upgrade; paired fixture uncertainty is not training-seed uncertainty or general continual-learning evidence.",
        "task_routing": "Identical external task name in prompts; native additionally selects a frozen or active task vector. LoRA remains one persistent adapter for every task.",
    }
    write_json(output / "result.json", result)
    emit(
        output,
        "arm_complete",
        qualified=qualified,
        sequence_learning_signal=result["sequence_learning_signal"],
    )


def source_fingerprint() -> dict:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "data.py",
            "learner.py",
            "run.py",
            "metrics.py",
            "qualify.py",
            "compare.py",
            "protocol.json",
        )
    }


def prepare(config: dict, output: Path) -> None:
    entries = {path.name: path for path in output.iterdir()}
    bootstrap = {
        "execution.json",
        "task.json",
        "run.log",
        "packages.txt",
        "config.json",
    }
    if entries:
        if (
            not bootstrap <= set(entries)
            or set(entries) - bootstrap - {"attempts"}
            or any(
                entries[name].is_symlink() or not entries[name].is_file()
                for name in bootstrap
            )
        ):
            raise ValueError("OUTPUT_CONTAINS_RESEARCH_OR_INVALID_BOOTSTRAP")
        if "attempts" in entries:
            attempts = entries["attempts"]
            if (
                attempts.is_symlink()
                or not attempts.is_dir()
                or any(
                    path.is_symlink() or not (path.is_file() or path.is_dir())
                    for path in attempts.rglob("*")
                )
            ):
                raise ValueError("INVALID_SUPERVISOR_ATTEMPT_ARCHIVE")
        existing = json.loads(entries["config.json"].read_text())
        task = json.loads(entries["task.json"].read_text())
        execution = json.loads(entries["execution.json"].read_text())
        if existing != config or task.get("config") != config:
            raise ValueError("SUPERVISOR_BOOTSTRAP_CONFIG_MISMATCH")
        if (
            task.get("id") != output.name
            or execution.get("task_id") != task["id"]
            or execution.get("status") != "running"
            or execution.get("task") != task
        ):
            raise ValueError("SUPERVISOR_BOOTSTRAP_TASK_MISMATCH")
    else:
        write_json(output / "config.json", config)
    write_json(
        output / "protocol.json", json.loads((ROOT / "protocol.json").read_text())
    )
    write_json(
        output / "preregistration.json",
        {
            "config_sha256": digest(config),
            "source_sha256": source_fingerprint(),
            "numerical_gate": NUMERICAL_GATE,
            "selection_rule": SELECTION_RULE,
        },
    )
    prepare_data(config, output)
    emit(output, "prepared", config_sha256=digest(config))


def verify_prepared(config: dict, output: Path) -> None:
    preregistration = json.loads((output / "preregistration.json").read_text())
    if digest(config) != preregistration["config_sha256"]:
        raise ValueError("PREPARED_CONFIG_CHANGED")
    if source_fingerprint() != preregistration["source_sha256"]:
        raise ValueError("PREREGISTERED_SOURCE_CHANGED")
    if (
        NUMERICAL_GATE != preregistration["numerical_gate"]
        or SELECTION_RULE != preregistration["selection_rule"]
    ):
        raise ValueError("PREREGISTERED_GATE_CHANGED")


def child_command(output: Path, phase: str, stage: int | None = None) -> list[str]:
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(ROOT / "run.py"),
        "--config",
        str(output / "config.json"),
        "--output-dir",
        str(output),
        "--phase",
        phase,
    ]
    if stage is not None:
        command.extend(("--stage", str(stage)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "run",
            "prepare",
            "initialize",
            "train",
            "baselines",
            "evaluate",
            "finish",
        ),
        default="run",
    )
    parser.add_argument("--stage", type=int, choices=range(3))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config)
    prepared = args.phase not in {"run", "prepare"}
    try:
        if args.phase in {"run", "prepare"}:
            prepare(config, output)
            prepared = True
        else:
            verify_prepared(config, output)
        if args.phase == "run":
            for phase in ("initialize", "train", "baselines"):
                subprocess.run(child_command(output, phase), check=True)
            for stage in range(3):
                subprocess.run(child_command(output, "evaluate", stage), check=True)
            finish(config, output)
        elif args.phase == "initialize":
            initialize(config, output)
        elif args.phase == "train":
            train(config, output)
        elif args.phase == "baselines":
            evaluate_baselines(config, output)
        elif args.phase == "evaluate":
            if args.stage is None:
                raise ValueError("EVALUATION_STAGE_REQUIRED")
            evaluate_checkpoint(config, output, args.stage)
        elif args.phase == "finish":
            finish(config, output)
    except Exception as exc:
        if prepared:
            failure = output / "failure.json"
            if not failure.exists():
                write_json(
                    failure,
                    {
                        "phase": args.phase,
                        "stage": args.stage,
                        "type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            emit(output, "failed", phase=args.phase, error=str(exc))
        raise


if __name__ == "__main__":
    main()
