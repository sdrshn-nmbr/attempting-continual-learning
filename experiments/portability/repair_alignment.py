from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import torch
from portallib import PortalConfig, PortalModel
from safetensors import safe_open
from safetensors.torch import save_file

PUBLISHED_TASKS = (
    "truthfulqa",
    "rte",
    "cb",
    "copa",
    "wic",
    "wsc",
    "boolq",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "winogrande",
    "commonsense_qa",
    "sciq",
)
ARMS = ("identity", "repair", "mismatch")
CONFIG_KEYS = {
    "kind",
    "version",
    "source_initial",
    "source_learned",
    "targets",
    "train_tasks",
    "holdout_tasks",
    "mismatch_task_map",
    "arms",
    "fit",
    "gates",
    "provenance",
}
FIT_KEYS = {
    "device",
    "parameter_dtype",
    "gram_dtype",
    "cpu_threads",
    "steps",
    "tasks_per_step",
    "seed",
    "learning_rate",
    "weight_decay",
    "adam_epsilon",
    "gradient_clip",
    "normalization_epsilon",
    "schedule",
    "checkpoint_selection",
    "trainable",
}
GATE_KEYS = {
    "training_mean_final_over_initial_max",
    "heldout_mean_final_over_initial_max",
    "every_heldout_task_nonworsening",
    "correct_over_mismatch_heldout_max",
    "identity_relative_squared_error_max",
    "identity_alignment_max_abs_delta",
    "negligible_initial_error",
    "frozen_core_and_all_vectors_exact",
    "saved_reload_exact",
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_hash(tensors: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        hasher.update(name.encode())
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        hasher.update(memoryview(raw))
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def emit(event: str, **fields: object) -> None:
    print(
        json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False),
        flush=True,
    )


def require_keys(value: object, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"REPAIR_CONFIG_KEYS_MISMATCH: {name}")


def positive_number(value: object, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"REPAIR_POSITIVE_FINITE_NUMBER_REQUIRED: {name}")


def checkpoint_specs(config: dict) -> dict:
    return {
        "source_initial": config["source_initial"],
        "source_learned": config["source_learned"],
        **config["targets"],
    }


def validate_config(config: dict) -> None:
    require_keys(config, CONFIG_KEYS, "root")
    if (
        config["kind"] != "portal_alignment_geometry_repair"
        or type(config["version"]) is not int
        or config["version"] != 1
    ):
        raise ValueError("REPAIR_CONFIG_KIND_OR_VERSION_MISMATCH")
    require_keys(config["targets"], {"qwen4", "mistral7"}, "targets")
    if config["arms"] != list(ARMS):
        raise ValueError("REPAIR_ALL_THREE_ORDERED_ARMS_REQUIRED")
    for name in ("train_tasks", "holdout_tasks"):
        tasks = config[name]
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(type(task) is not str for task in tasks)
        ):
            raise ValueError(f"REPAIR_TASK_LIST_REQUIRED: {name}")
        if len(set(tasks)) != len(tasks) or not set(tasks) <= set(PUBLISHED_TASKS):
            raise ValueError(f"REPAIR_PUBLISHED_TASKS_ONLY: {name}")
    train, holdout = set(config["train_tasks"]), set(config["holdout_tasks"])
    if train & holdout or train | holdout != set(PUBLISHED_TASKS) or len(train) < 2:
        raise ValueError("REPAIR_TASK_SPLIT_MUST_PARTITION_PUBLISHED_TASKS")
    tasks = config["train_tasks"]
    if config["mismatch_task_map"] != dict(
        zip(tasks, tasks[1:] + tasks[:1], strict=True)
    ):
        raise ValueError("REPAIR_CYCLIC_TRAIN_ONLY_MISMATCH_REQUIRED")
    fit = config["fit"]
    require_keys(fit, FIT_KEYS, "fit")
    constants = {
        "device": "cpu",
        "parameter_dtype": "float32",
        "gram_dtype": "float64",
        "schedule": "balanced_shuffled_cycles",
        "checkpoint_selection": "fixed_final_step",
        "trainable": "alignment_only",
        "weight_decay": 0,
    }
    if any(fit[key] != value for key, value in constants.items()):
        raise ValueError("REPAIR_FIXED_FIT_CONTRACT_MISMATCH")
    for name in ("steps", "tasks_per_step", "cpu_threads"):
        if type(fit[name]) is not int or fit[name] < 1:
            raise ValueError(f"REPAIR_POSITIVE_INTEGER_REQUIRED: {name}")
    if fit["tasks_per_step"] > len(train) or fit["steps"] * fit["tasks_per_step"] % len(
        train
    ):
        raise ValueError("REPAIR_COMPLETE_BALANCED_TASK_CYCLES_REQUIRED")
    if type(fit["seed"]) is not int or not 0 <= fit["seed"] < 2**63:
        raise ValueError("REPAIR_INVALID_SEED")
    for name in (
        "learning_rate",
        "adam_epsilon",
        "gradient_clip",
        "normalization_epsilon",
    ):
        positive_number(fit[name], name)
    gates = config["gates"]
    require_keys(gates, GATE_KEYS, "gates")
    for name, value in gates.items():
        if name in (
            "every_heldout_task_nonworsening",
            "frozen_core_and_all_vectors_exact",
            "saved_reload_exact",
        ):
            if value is not True:
                raise ValueError(f"REPAIR_REQUIRED_GATE_DISABLED: {name}")
        else:
            positive_number(value, name)
    if not isinstance(config["provenance"], dict):
        raise TypeError("REPAIR_PROVENANCE_REQUIRED")
    for name, spec in checkpoint_specs(config).items():
        require_keys(spec, {"path", "files_sha256"}, name)
        if not isinstance(spec["path"], str) or not Path(spec["path"]).is_absolute():
            raise ValueError(f"REPAIR_ABSOLUTE_LOCAL_CHECKPOINT_REQUIRED: {name}")
        require_keys(spec["files_sha256"], {"config.json", "model.safetensors"}, name)
        for checksum in spec["files_sha256"].values():
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(char not in "0123456789abcdef" for char in checksum)
            ):
                raise ValueError(f"REPAIR_INVALID_SHA256_PIN: {name}")


def verify_checkpoint_files(spec: dict) -> dict:
    actual = {
        name: file_hash(Path(spec["path"]) / name) for name in spec["files_sha256"]
    }
    if actual != spec["files_sha256"]:
        raise ValueError(f"REPAIR_CHECKPOINT_FILE_SHA256_MISMATCH: {spec['path']}")
    return actual


@dataclass
class Checkpoint:
    config: PortalConfig
    core: dict[str, torch.Tensor]
    latents: torch.Tensor
    alignment: dict[str, torch.Tensor]
    receipt: dict


def load_checkpoint(spec: dict, *, alignment: bool) -> Checkpoint:
    files = verify_checkpoint_files(spec)
    path = Path(spec["path"])
    config = PortalConfig.from_dict(json.loads((path / "config.json").read_bytes()))
    if tuple(config.tasks[: len(PUBLISHED_TASKS)]) != PUBLISHED_TASKS:
        raise ValueError(f"REPAIR_PUBLISHED_TASK_PREFIX_MISMATCH: {path}")
    if alignment and tuple(config.tasks) != PUBLISHED_TASKS:
        raise ValueError(f"REPAIR_TARGET_MUST_CONTAIN_ONLY_PUBLISHED_TASKS: {path}")
    with safe_open(
        path / "model.safetensors", framework="pt", device="cpu"
    ) as artifact:
        metadata = artifact.metadata() or {}
        if metadata.get("format") != "portallib" or metadata.get(
            "format_version"
        ) != str(config.format_version):
            raise ValueError(f"REPAIR_CHECKPOINT_FORMAT_MISMATCH: {path}")
        latent_slice = artifact.get_slice("task_latents")
        if latent_slice.get_shape() != [len(config.tasks), config.d_z]:
            raise ValueError(f"REPAIR_LATENT_SHAPE_MISMATCH: {path}")
        latents = latent_slice[: len(PUBLISHED_TASKS)].clone()
        tensor_names = artifact.keys()
        core = {
            key.removeprefix("core."): artifact.get_tensor(key)
            for key in tensor_names
            if key.startswith("core.")
        }
        maps = {
            key.removeprefix("alignment."): artifact.get_tensor(key)
            for key in tensor_names
            if alignment and key.startswith("alignment.")
        }
    used = {f"core.{key}": value for key, value in core.items()}
    used.update({f"alignment.{key}": value for key, value in maps.items()})
    used["published_task_latents"] = latents
    for name, value in used.items():
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"REPAIR_INVALID_FLOAT32_CPU_TENSOR: {path}/{name}")
    return Checkpoint(
        replace(config, tasks=PUBLISHED_TASKS),
        core,
        latents,
        maps,
        {
            "path": str(path),
            "files_sha256": files,
            "core_sha256": tensor_hash(core),
            "published_latents_sha256": tensor_hash({"latents": latents}),
            "alignment_sha256": tensor_hash(maps) if alignment else None,
            "used_tensor_sha256": tensor_hash(used),
            "latent_rows_loaded": [0, len(PUBLISHED_TASKS)],
            "added_latent_rows_loaded": 0,
            "hash_scope": "Full sealed file bytes are pinned; tensor loading uses only core, published latent prefix, and target alignment.",
        },
    )


def make_model(target: Checkpoint, shared: Checkpoint) -> PortalModel:
    if target.config.architecture_kwargs() != shared.config.architecture_kwargs():
        raise ValueError("REPAIR_CORE_ARCHITECTURE_MISMATCH")
    model = PortalModel(target.config, shared.latents).to(
        device="cpu", dtype=torch.float32
    )
    model.core.load_state_dict(shared.core, strict=True)
    model.alignment.load_state_dict(target.alignment, strict=True)
    return model.requires_grad_(False).eval()


def product_squared_error(
    a: torch.Tensor,
    b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
) -> torch.Tensor:
    """Implicit BA distance with exactly zero value and gradient at identical factors."""
    a, b, reference_a, reference_b = (
        value.to(torch.float64) for value in (a, b, reference_a, reference_b)
    )
    left = torch.cat((b - reference_b, reference_b), dim=1)
    right = torch.cat((a, a - reference_a), dim=0)
    return ((left.T @ left) * (right @ right.T).T).sum()


def product_squared_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.to(torch.float64), b.to(torch.float64)
    return ((b.T @ b) * (a @ a.T).T).sum()


def product_loss(
    generated: dict, reference: dict, epsilon: float, scale: float
) -> torch.Tensor:
    if not generated or generated.keys() != reference.keys():
        raise ValueError("REPAIR_PROJECTION_KEYS_MISMATCH")
    losses = []
    for key, (a, b) in generated.items():
        reference_a, reference_b = reference[key]
        error = scale**2 * product_squared_error(a, b, reference_a, reference_b)
        norm = scale**2 * product_squared_norm(reference_a, reference_b)
        losses.append(error / (norm + epsilon))
    return torch.stack(losses).mean()


def frozen_hashes(model: PortalModel) -> dict:
    return {
        "core_sha256": tensor_hash(model.core.state_dict()),
        "published_latents_sha256": tensor_hash({"latents": model.task_latents}),
    }


def task_factors(model: PortalModel, task: str) -> dict:
    if task not in PUBLISHED_TASKS:
        raise ValueError(f"REPAIR_PUBLISHED_TASKS_ONLY: {task}")
    return model(model.task_latents[model.config.tasks.index(task)])


@torch.no_grad()
def evaluate(
    model: PortalModel, teacher: PortalModel, tasks: list[str], epsilon: float
) -> dict:
    metrics = {}
    scale = model.config.alpha / model.config.rank
    for task in tasks:
        generated, reference = task_factors(model, task), task_factors(teacher, task)
        modules = {}
        for key, (a, b) in generated.items():
            reference_a, reference_b = reference[key]
            squared_error = scale**2 * float(
                product_squared_error(a, b, reference_a, reference_b)
            )
            squared_norm = scale**2 * float(
                product_squared_norm(reference_a, reference_b)
            )
            normalized = squared_error / (squared_norm + epsilon)
            if not math.isfinite(normalized) or normalized < -1e-9:
                raise ValueError(
                    f"REPAIR_INVALID_PRODUCT_METRIC: {task}/{key}/{normalized}"
                )
            modules[str(key)] = {
                "squared_error": squared_error,
                "reference_squared_norm": squared_norm,
                "normalized_squared_error": normalized,
            }
        metrics[task] = {
            "mean_normalized_squared_error": sum(
                value["normalized_squared_error"] for value in modules.values()
            )
            / len(modules),
            "max_normalized_squared_error": max(
                value["normalized_squared_error"] for value in modules.values()
            ),
            "projections": modules,
            "generated_factors_sha256": tensor_hash(
                {
                    f"{key}/{factor}": value
                    for key, pair in generated.items()
                    for factor, value in zip(("A", "B"), pair, strict=True)
                }
            ),
        }
    return {
        "mean_normalized_squared_error": sum(
            value["mean_normalized_squared_error"] for value in metrics.values()
        )
        / len(metrics),
        "max_normalized_squared_error": max(
            value["max_normalized_squared_error"] for value in metrics.values()
        ),
        "tasks": metrics,
    }


def make_schedule(tasks: list[str], fit: dict) -> list[list[str]]:
    generator = random.Random(fit["seed"])
    order = []
    while len(order) < fit["steps"] * fit["tasks_per_step"]:
        cycle = tasks.copy()
        generator.shuffle(cycle)
        order.extend(cycle)
    width = fit["tasks_per_step"]
    return [
        order[start : start + width] for start in range(0, fit["steps"] * width, width)
    ]


def train_alignment(
    model: PortalModel,
    teacher: PortalModel,
    train_tasks: list[str],
    fit: dict,
    schedule: list[list[str]],
    mapping: dict,
    arm: str,
) -> list[dict]:
    model.alignment.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.alignment.parameters(),
        lr=fit["learning_rate"],
        weight_decay=fit["weight_decay"],
        eps=fit["adam_epsilon"],
        foreach=False,
    )
    with torch.no_grad():
        references = {task: task_factors(teacher, task) for task in train_tasks}
    training = []
    for step, batch in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [
                product_loss(
                    task_factors(model, task),
                    references[mapping[task]],
                    fit["normalization_epsilon"],
                    model.config.alpha / model.config.rank,
                )
                for task in batch
            ]
        ).mean()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value) or loss_value < -1e-9:
            raise ValueError(f"REPAIR_NONFINITE_OR_NEGATIVE_LOSS: {arm}/{step}")
        loss.backward()
        maximum_gradient = 0.0
        for parameter_name, parameter in model.alignment.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(
                    f"REPAIR_INVALID_ALIGNMENT_GRADIENT: {arm}/{step}/{parameter_name}"
                )
            maximum_gradient = max(maximum_gradient, float(parameter.grad.abs().max()))
        if (
            any(parameter.grad is not None for parameter in model.core.parameters())
            or model.task_latents.grad is not None
        ):
            raise ValueError(f"REPAIR_FROZEN_GRADIENT: {arm}/{step}")
        if arm == "identity" and (loss_value != 0 or maximum_gradient != 0):
            raise ValueError(f"REPAIR_IDENTITY_NOT_STATIONARY: {step}")
        norm = torch.nn.utils.clip_grad_norm_(
            model.alignment.parameters(),
            fit["gradient_clip"],
            error_if_nonfinite=True,
            foreach=False,
        )
        optimizer.step()
        row = {
            "step": step,
            "tasks": batch,
            "loss": loss_value,
            "maximum_alignment_gradient": maximum_gradient,
            "gradient_norm_before_clip": float(norm),
        }
        training.append(row)
        emit("repair_alignment_step", arm=arm, **row)
    model.requires_grad_(False)
    return training


def save_checkpoint(model: PortalModel, path: Path) -> dict:
    path.mkdir(parents=True)
    write_json(path / "config.json", model.config.to_dict())
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        path / "model.safetensors",
        metadata={
            "format": "portallib",
            "format_version": str(model.config.format_version),
        },
    )
    return {
        "path": str(path),
        "files_sha256": {
            name: file_hash(path / name)
            for name in ("config.json", "model.safetensors")
        },
    }


def run_arm(
    name: str,
    target: Checkpoint,
    shared: Checkpoint,
    source_spec: dict,
    teacher: PortalModel,
    config: dict,
    schedule: list[list[str]],
    output: Path,
) -> dict:
    model = make_model(target, shared)
    frozen_before = frozen_hashes(model)
    alignment_before = tensor_hash(model.alignment.state_dict())
    fit = config["fit"]
    initial_evaluation = {
        split: evaluate(
            model, teacher, config[f"{split}_tasks"], fit["normalization_epsilon"]
        )
        for split in ("train", "holdout")
    }
    mapping = (
        config["mismatch_task_map"]
        if name == "mismatch"
        else {task: task for task in config["train_tasks"]}
    )
    emit(
        "repair_arm_started",
        arm=name,
        steps=fit["steps"],
        target=target.config.base_model_name_or_path,
    )
    training = train_alignment(
        model, teacher, config["train_tasks"], fit, schedule, mapping, name
    )
    frozen_after = frozen_hashes(model)
    if frozen_before != frozen_after:
        raise ValueError(f"REPAIR_FROZEN_STATE_CHANGED: {name}")
    alignment_after = tensor_hash(model.alignment.state_dict())
    delta = max(
        float((value - target.alignment[key]).abs().max())
        for key, value in model.alignment.state_dict().items()
    )
    if name == "identity" and alignment_before != alignment_after:
        raise ValueError("REPAIR_IDENTITY_ALIGNMENT_CHANGED")
    evaluation = {
        split: evaluate(
            model, teacher, config[f"{split}_tasks"], fit["normalization_epsilon"]
        )
        for split in ("train", "holdout")
    }
    checkpoint = save_checkpoint(model, output / name / "checkpoint")
    restored = load_checkpoint(checkpoint, alignment=True)
    reloaded = make_model(restored, restored)
    reload_evaluation = {
        split: evaluate(
            reloaded, teacher, config[f"{split}_tasks"], fit["normalization_epsilon"]
        )
        for split in ("train", "holdout")
    }
    exact = (
        tensor_hash(model.state_dict()) == tensor_hash(reloaded.state_dict())
        and evaluation == reload_evaluation
    )
    if not exact:
        raise ValueError(f"REPAIR_CHECKPOINT_RELOAD_MISMATCH: {name}")
    return {
        "optimizer_updates": len(training),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": fit["learning_rate"],
            "weight_decay": fit["weight_decay"],
            "epsilon": fit["adam_epsilon"],
            "gradient_clip": fit["gradient_clip"],
        },
        "schedule_sha256": digest(schedule),
        "task_mapping": mapping,
        "training": training,
        "initial_evaluation": initial_evaluation,
        "evaluation": evaluation,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "alignment_before_sha256": alignment_before,
        "alignment_after_sha256": alignment_after,
        "alignment_max_abs_delta": delta,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.alignment.parameters()
        ),
        "trainable_names": [
            f"alignment.{key}" for key, _ in model.alignment.named_parameters()
        ],
        "checkpoint": checkpoint,
        "alignment_carrier": {
            "tasks": list(PUBLISHED_TASKS),
            "shared_source_checkpoint": source_spec,
            "future_graft": "This carrier contains only 14 published vectors. For a behavioral run, graft only its alignment onto the full pinned source shared core and task-vector table; added vectors were never loaded by repair.",
        },
        "reload": {
            "state_and_all_old_task_metrics_exact": exact,
            "fresh_model_from_disk": True,
            "separate_process": False,
            "optimizer_updates": 0,
        },
        "holdout_access": "Baseline and fixed-final/reload evaluation outside train_alignment; no heldout generation or selection in the trainer.",
    }


def apply_gates(arms: dict, gates: dict) -> dict:
    repair, mismatch, identity = (
        arms[name] for name in ("repair", "mismatch", "identity")
    )
    initial = {
        split: repair["initial_evaluation"][split]["mean_normalized_squared_error"]
        for split in ("train", "holdout")
    }
    final = {
        split: repair["evaluation"][split]["mean_normalized_squared_error"]
        for split in ("train", "holdout")
    }
    negligible = {
        split: value < gates["negligible_initial_error"]
        for split, value in initial.items()
    }
    checks = {
        "training_improvement": not negligible["train"]
        and final["train"]
        <= gates["training_mean_final_over_initial_max"] * initial["train"],
        "heldout_improvement": not negligible["holdout"]
        and final["holdout"]
        <= gates["heldout_mean_final_over_initial_max"] * initial["holdout"],
        "every_heldout_task_nonworsening": all(
            value["mean_normalized_squared_error"]
            <= repair["initial_evaluation"]["holdout"]["tasks"][task][
                "mean_normalized_squared_error"
            ]
            for task, value in repair["evaluation"]["holdout"]["tasks"].items()
        ),
        "correct_better_than_mismatch": final["holdout"]
        <= gates["correct_over_mismatch_heldout_max"]
        * mismatch["evaluation"]["holdout"]["mean_normalized_squared_error"],
        "identity_error": max(
            identity["evaluation"][split]["max_normalized_squared_error"]
            for split in ("train", "holdout")
        )
        <= gates["identity_relative_squared_error_max"],
        "identity_alignment_delta": identity["alignment_max_abs_delta"]
        <= gates["identity_alignment_max_abs_delta"],
        "frozen_loaded_core_and_vectors": all(
            arm["frozen_before"] == arm["frozen_after"] for arm in arms.values()
        ),
        "saved_reload_exact": all(
            arm["reload"]["state_and_all_old_task_metrics_exact"]
            for arm in arms.values()
        ),
    }
    passed = all(checks.values())
    status = "passed" if passed else "failed"
    if any(negligible.values()):
        status = "no_substantive_geometry_error"
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "thresholds": gates,
        "negligible_initial_error": negligible,
        "final_over_initial": {
            split: None if negligible[split] else final[split] / initial[split]
            for split in initial
        },
        "scope": "CPU geometry gate only; parent owns review and any later behavioral evaluation.",
    }


def run(config: dict, output: Path, *, config_file_sha256: str | None = None) -> dict:
    validate_config(config)
    output = output.resolve()
    for spec in checkpoint_specs(config).values():
        path = Path(spec["path"]).resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("REPAIR_OUTPUT_OVERLAPS_INPUT_CHECKPOINT")
    if output.exists() and any(output.iterdir()):
        raise ValueError("REPAIR_OUTPUT_DIRECTORY_NOT_EMPTY")
    fit = config["fit"]
    torch.set_num_threads(fit["cpu_threads"])
    torch.manual_seed(fit["seed"])
    torch.use_deterministic_algorithms(True)
    emit("repair_loading_checkpoints", target_base_loaded=False, device="cpu")
    initial = load_checkpoint(config["source_initial"], alignment=False)
    learned = load_checkpoint(config["source_learned"], alignment=False)
    targets = {
        name: load_checkpoint(spec, alignment=True)
        for name, spec in config["targets"].items()
    }
    if any(
        target.receipt["core_sha256"] != initial.receipt["core_sha256"]
        for target in targets.values()
    ):
        raise ValueError("REPAIR_RELEASED_SOURCE_TARGET_CORE_MISMATCH")
    if (
        len(
            {
                checkpoint.receipt["published_latents_sha256"]
                for checkpoint in (initial, learned, *targets.values())
            }
        )
        != 1
    ):
        raise ValueError("REPAIR_PUBLISHED_LATENTS_CHANGED")
    schedule = make_schedule(config["train_tasks"], fit)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "schedule.json", schedule)
    receipt = {
        "status": "running",
        "kind": config["kind"],
        "version": config["version"],
        "config": config,
        "config_sha256": digest(config),
        "config_file_sha256": config_file_sha256,
        "schedule_sha256": digest(schedule),
        "implementation_sha256": file_hash(Path(__file__)),
        "inputs": {
            "source_initial": initial.receipt,
            "source_learned": learned.receipt,
            **{name: target.receipt for name, target in targets.items()},
        },
        "runtime": {
            "device": "cpu",
            "parameter_dtype": "float32",
            "gram_dtype": "float64",
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "portallib": version("portallib"),
            "safetensors": version("safetensors"),
        },
        "objective": "Equal task means of equal projection means: ||sBA-sB0A0||_F^2 / (||sB0A0||_F^2 + normalization_epsilon), where s=alpha/rank, using implicit residual factors and FP64 Gram accumulation.",
        "controls": {
            "released_source_target_core_identical": True,
            "published_latents_identical": True,
            "all_arms_same_initial_alignment_and_schedule": True,
        },
        "target_base_loaded": False,
        "base_forward_calls": 0,
        "abc_data_reads": 0,
        "added_latent_rows_loaded": 0,
        "frozen_vector_scope": "Loaded published vectors and core are hashed before/after fitting. Unloaded added vectors remain in full source files, whose pins are reverified after all fits.",
        "claim_boundary": "Old generated-adapter product agreement only. No base behavior, new-task acquisition, or transfer was measured. No heldout-driven checkpoint selection.",
        "provenance": config["provenance"],
        "targets": {},
    }
    try:
        for name, target in targets.items():
            teacher = make_model(target, initial)
            arms = {}
            receipt["targets"][name] = {"arms": arms}
            for arm in ARMS:
                shared, spec = (
                    (initial, config["source_initial"])
                    if arm == "identity"
                    else (learned, config["source_learned"])
                )
                arms[arm] = run_arm(
                    arm, target, shared, spec, teacher, config, schedule, output / name
                )
                write_json(output / "result.json", receipt)
            receipt["targets"][name]["gates"] = apply_gates(arms, config["gates"])
            write_json(output / "result.json", receipt)
        for spec in checkpoint_specs(config).values():
            verify_checkpoint_files(spec)
        receipt["input_files_reverified_exact"] = True
        receipt["status"] = "completed"
        write_json(output / "result.json", receipt)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "failure.json", receipt)
        raise
    emit(
        "repair_completed",
        targets=list(receipt["targets"]),
        result=str(output / "result.json"),
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only alignment repair using published old-task adapter products."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.config.read_bytes()
        run(
            json.loads(raw),
            args.output_dir,
            config_file_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except Exception as error:
        print(
            f"REPAIR_FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
