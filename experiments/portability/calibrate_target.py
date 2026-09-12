from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import learner
import torch
from data import SEQUENCE_TASKS, Example, digest, write_json
from learner import (
    evaluate,
    frozen_base_tensors,
    load_base,
    save_native,
    shared_hash,
    supervised_loss,
    tensor_hash,
)
from portallib import PortalConfig, PortalModel
from portallib.evaluation import PortalInjector
from portallib.model import PortalAlignment
from safetensors.torch import load_file, save_file

TASKS = SEQUENCE_TASKS
LANE_ROOT = Path(__file__).resolve().parent
TRAINED = ("learned_calibrated", "initial_calibrated", "lora_from_zero")
FLOORS = (
    "old_alignment_learned",
    "old_alignment_initial",
    "fresh_learned",
    "fresh_initial",
)
TRAINING = {
    "examples_per_task": 16,
    "epochs": 5,
    "batch_size_per_task": 4,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "gradient_clip": 1.0,
    "ema_decay": 0.9,
    "ema_floor": 1e-6,
    "max_prompt": 768,
    "loss": "equal_task_mean_ema_normalized_gold_answer_token_nll",
    "selection": "fixed_final_epoch",
    "optimizer_updates": 20,
    "task_microbatches": 60,
}
GATES = {
    "minimum_lora_gain_over_raw": 0.05,
    "minimum_learned_gain_over_initial_calibrated": 0.05,
    "maximum_accuracy": 1.0,
    "insufficient_headroom": "inconclusive_not_failure",
    "use_for_training_or_selection": False,
}
CONTRACT = {
    "source_training_seed": 30301,
    "alignment_seed": 73303,
    "data_seed": 83303,
    "lora_seed": 93303,
    "tasks": list(TASKS),
    "training": TRAINING,
    "gates": GATES,
    "fresh_alignment": "sdk_zero_output_including_fresh_layer_embeddings",
    "architecture_choice": "published_target_config_no_search",
    "lora": "one_persistent_rank8_qv_alpha16_adapter_for_ABC",
    "evaluation": {"batch_size": 8, "selection": "all_conditions_final_only"},
}


def input_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else LANE_ROOT / path


def file_pin(path: Path) -> dict:
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": checksum}


def verify_file(path: Path, pin: dict) -> dict:
    actual = file_pin(path)
    if actual["bytes"] != pin["bytes"]:
        raise ValueError(f"CALIBRATION_FILE_SIZE_MISMATCH: {path}")
    if pin.get("sha256"):
        valid = actual["sha256"] == pin["sha256"]
    elif pin.get("git_blob_sha1"):
        hasher = hashlib.sha1(f"blob {actual['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        valid = hasher.hexdigest() == pin["git_blob_sha1"]
    else:
        raise ValueError(f"CALIBRATION_UNPINNED_FILE: {path}")
    if not valid:
        raise ValueError(f"CALIBRATION_FILE_HASH_MISMATCH: {path}")
    return {"path": str(path.resolve()), **actual}


def verify_bundle(spec: dict) -> dict:
    root = input_path(spec["path"])
    actual_names = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and p.relative_to(root).parts[:2] != (".cache", "huggingface")
        and p.relative_to(root) != Path("README.md")
    }
    if actual_names != set(spec["files"]):
        raise ValueError(f"CALIBRATION_BUNDLE_FILE_SET_MISMATCH: {root}")
    return {name: verify_file(root / name, pin) for name, pin in spec["files"].items()}


def pin_bundle(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "files": {
            str(p.relative_to(path)): file_pin(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        },
    }


def validate_config(config: dict) -> None:
    if (
        config["kind"] != "portal_target_task_calibration"
        or config["protocol"] != CONTRACT
    ):
        raise ValueError("CALIBRATION_FROZEN_PROTOCOL_CHANGED")
    if config["protocol_sha256"] != digest(config["protocol"]):
        raise ValueError("CALIBRATION_PROTOCOL_HASH_MISMATCH")
    if config["runtime"] not in (
        {"device": "cpu", "dtype": "float32", "autocast": False, "cpu_threads": 1},
        {"device": "cuda:0", "dtype": "float32", "autocast": False, "cpu_threads": 4},
    ):
        raise ValueError("CALIBRATION_RUNTIME_CONTRACT")
    if config["role"] not in {"mechanistic_exploratory", "cpu_contract_test"}:
        raise ValueError("CALIBRATION_ROLE")
    if set(config["inputs"]) != {
        "source_initial",
        "source_learned",
        "target_portal",
        "target_base",
        "fixture",
        "fresh_fixture",
    }:
        raise ValueError("CALIBRATION_INPUT_SET")
    if config["role"] == "mechanistic_exploratory":
        if (
            config["base"]["repo_id"] != "Qwen/Qwen3-4B"
            or config["expected_target_hooks"] != 72
        ):
            raise ValueError("CALIBRATION_FIXED_TARGET_CHANGED")
        if config["expected_source_hooks"] != 72:
            raise ValueError("CALIBRATION_FIXED_SOURCE_CHANGED")
    if config["data_exposure"] != {
        "training": "source_train_only",
        "old_validation_test": "previously_exposed_exploratory",
        "fresh288": "previously_exposed_exploratory_288_inputs_per_task",
        "heldout_access": "after_all_final_checkpoints_only",
    }:
        raise ValueError("CALIBRATION_DATA_EXPOSURE_CHANGED")
    selected = config["calibration_ids"]
    if set(selected) != set(TASKS) or any(
        len(v) != 16 or len(set(v)) != 16 for v in selected.values()
    ):
        raise ValueError("CALIBRATION_FIXED_EXAMPLE_IDS")


def verify_inputs(config: dict) -> dict:
    result = {}
    for name, spec in config["inputs"].items():
        result[name] = (
            verify_bundle(spec)
            if "files" in spec
            else verify_file(input_path(spec["path"]), spec)
        )
    if (
        Path(config["base"]["local_path"]).resolve()
        != input_path(config["inputs"]["target_base"]["path"]).resolve()
    ):
        raise ValueError("CALIBRATION_BASE_PATH_NOT_PINNED")
    return result


def runtime(config: dict) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(CONTRACT["alignment_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def emit(output: Path, event: str, **fields) -> None:
    row = {"event": event, "pid": os.getpid(), **fields}
    line = json.dumps(row, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    with (output / "events.jsonl").open("a") as stream:
        stream.write(line + "\n")


def calibration_rows(config: dict) -> dict[str, list[Example]]:
    splits = json.loads(input_path(config["inputs"]["fixture"]["path"]).read_text())[
        "splits"
    ]
    selected = {}
    for task in TASKS:
        pool = {row["id"]: Example.from_dict(row) for row in splits[f"{task}_train"]}
        rows = [pool[row_id] for row_id in config["calibration_ids"][task]]
        if any(row.task != task for row in rows):
            raise ValueError("CALIBRATION_TASK_MISMATCH")
        selected[task] = rows
    return selected


def balanced_schedule(rows: dict[str, list[Example]]) -> list[dict]:
    schedule = []
    for epoch in range(1, TRAINING["epochs"] + 1):
        orders = {}
        for task in TASKS:
            orders[task] = [row.id for row in rows[task]]
            random.Random(
                int(digest([CONTRACT["data_seed"], epoch, task]), 16)
            ).shuffle(orders[task])
        for start in range(
            0, TRAINING["examples_per_task"], TRAINING["batch_size_per_task"]
        ):
            schedule.append(
                {
                    "epoch": epoch,
                    "tasks": {
                        task: orders[task][
                            start : start + TRAINING["batch_size_per_task"]
                        ]
                        for task in TASKS
                    },
                }
            )
    return schedule


def load_native_bundle(spec: dict) -> PortalModel:
    verify_bundle(spec)
    root = input_path(spec["path"])
    config = PortalConfig.from_dict(json.loads((root / "config.json").read_text()))
    state = load_file(root / "model.safetensors", device="cpu")
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
        raise ValueError("CALIBRATION_SOURCE_NOT_FP32")
    with torch.random.fork_rng(devices=[]):
        model = PortalModel(config, state["task_latents"])
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False)


def make_target(
    shared: PortalModel, target: PortalModel, alignment: dict
) -> PortalModel:
    if shared.config.architecture_kwargs() != target.config.architecture_kwargs():
        raise ValueError("CALIBRATION_ARCHITECTURE_MISMATCH")
    if shared.config.tasks[: len(target.config.tasks)] != target.config.tasks:
        raise ValueError("CALIBRATION_PUBLISHED_TASK_PREFIX_MISMATCH")
    config = replace(target.config, tasks=shared.config.tasks)
    with torch.random.fork_rng(devices=[]):
        model = PortalModel(config, shared.task_latents)
    model.core.load_state_dict(shared.core.state_dict(), strict=True)
    model.alignment.load_state_dict(alignment, strict=True)
    model.requires_grad_(False)
    if shared_hash(model) != shared_hash(shared):
        raise ValueError("CALIBRATION_SHARED_STATE_CHANGED_DURING_TRANSPLANT")
    return model


def fresh_alignment(target: PortalModel) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CONTRACT["alignment_seed"])
        alignment = PortalAlignment(target.config, zero_output=True)
    return {
        name: value.detach().clone() for name, value in alignment.state_dict().items()
    }


def assert_fp32(values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        if value.is_floating_point() and (
            value.dtype != torch.float32 or not torch.isfinite(value).all()
        ):
            raise ValueError(f"CALIBRATION_NONFINITE_OR_NON_FP32: {name}")


def hook_receipt(base, config: PortalConfig, expected: int) -> list[dict]:
    rows = []
    for target, path in config.resolved_targets():
        module = base.model.get_submodule(path)
        dimensions = (module.in_features, module.out_features)
        if not isinstance(module, torch.nn.Linear) or dimensions != target.dimensions:
            raise ValueError(f"CALIBRATION_HOOK_SHAPE_MISMATCH: {path}")
        rows.append(
            {
                "path": path,
                "layer": target.layer_index,
                "module": target.module_name,
                "A": [config.rank, dimensions[0]],
                "B": [dimensions[1], config.rank],
            }
        )
    if len(rows) != expected or len({r["path"] for r in rows}) != expected:
        raise ValueError("CALIBRATION_HOOK_COUNT_MISMATCH")
    return rows


class ZeroLora(torch.nn.Module):
    def __init__(self, config: PortalConfig):
        super().__init__()
        self.config = config
        parameters = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CONTRACT["lora_seed"])
            for target in config.projection_targets:
                key = f"l{target.layer_index}_{target.module_name}"
                a = torch.empty(config.rank, target.in_features)
                torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                parameters[f"{key}_a"] = torch.nn.Parameter(a)
                parameters[f"{key}_b"] = torch.nn.Parameter(
                    torch.zeros(target.out_features, config.rank)
                )
        self.factors = torch.nn.ParameterDict(parameters)

    def forward(self, task: str) -> dict:
        if task not in TASKS:
            raise ValueError(f"CALIBRATION_LORA_UNKNOWN_TASK: {task}")
        return {
            target.key: (
                self.factors[f"l{target.layer_index}_{target.module_name}_a"],
                self.factors[f"l{target.layer_index}_{target.module_name}_b"],
            )
            for target in self.config.projection_targets
        }


def factors_for(adapter, task: str) -> dict:
    if isinstance(adapter, PortalModel):
        return adapter(adapter.task_latents[adapter.config.tasks.index(task)])
    return adapter(task)


def frozen_state(base, adapter) -> dict:
    result = {"base": tensor_hash(frozen_base_tensors(base.model))}
    if isinstance(adapter, PortalModel):
        result.update(
            core=tensor_hash(adapter.core.state_dict()),
            latents=tensor_hash({"all": adapter.task_latents}),
        )
    return result


def fit(base, adapter, rows: dict, schedule: list, output: Path | None = None) -> dict:
    base.freeze(gradient_checkpointing=False)
    base.model.eval()
    if isinstance(adapter, PortalModel):
        adapter.requires_grad_(False)
        adapter.alignment.requires_grad_(True)
        trainable = dict(adapter.alignment.named_parameters())
    else:
        trainable = dict(adapter.named_parameters())
    adapter.train()
    if isinstance(adapter, PortalModel):
        adapter.core.eval()
    assert_fp32(adapter.state_dict())
    assert_fp32(frozen_base_tensors(base.model))
    before = frozen_state(base, adapter)
    initial_trainable = tensor_hash(trainable)
    optimizer = torch.optim.AdamW(
        list(trainable.values()),
        lr=TRAINING["learning_rate"],
        betas=tuple(TRAINING["betas"]),
        eps=TRAINING["epsilon"],
        weight_decay=TRAINING["weight_decay"],
        foreach=False,
    )
    index = {row.id: row for task_rows in rows.values() for row in task_rows}
    if schedule != balanced_schedule(rows):
        raise ValueError("CALIBRATION_SCHEDULE_CHANGED")
    ema = {}
    trace = []
    nonzero = set()
    with PortalInjector(base.model, adapter.config) as injector:
        for step, entry in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            losses, tokens = {}, {}
            for task in TASKS:
                batch = [index[row_id] for row_id in entry["tasks"][task]]
                with (
                    torch.autocast(device_type=base.device.type, enabled=False),
                    injector.activate(factors_for(adapter, task)),
                ):
                    result = supervised_loss(base, batch, TRAINING["max_prompt"])
                    value = float(result.loss.detach())
                    ema[task] = (
                        TRAINING["ema_decay"] * ema.get(task, value)
                        + (1 - TRAINING["ema_decay"]) * value
                    )
                    (
                        result.loss / max(ema[task], TRAINING["ema_floor"]) / len(TASKS)
                    ).backward()
                losses[task], tokens[task] = value, result.supervised_tokens
            for name, value in trainable.items():
                if value.grad is None or not torch.isfinite(value.grad).all():
                    raise FloatingPointError(
                        f"CALIBRATION_MISSING_OR_NONFINITE_GRADIENT: {name}"
                    )
                if torch.count_nonzero(value.grad):
                    nonzero.add(name)
            for name, parameter in base.model.named_parameters():
                if parameter.grad is not None or parameter.requires_grad:
                    raise ValueError(f"CALIBRATION_BASE_RECEIVED_GRADIENT: {name}")
            if isinstance(adapter, PortalModel):
                for name, parameter in adapter.named_parameters():
                    if not name.startswith("alignment.") and (
                        parameter.grad is not None or parameter.requires_grad
                    ):
                        raise ValueError(
                            f"CALIBRATION_FROZEN_RECEIVED_GRADIENT: {name}"
                        )
            norm = torch.nn.utils.clip_grad_norm_(
                list(trainable.values()),
                TRAINING["gradient_clip"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            record = {
                "step": step,
                "epoch": entry["epoch"],
                "losses": losses,
                "answer_tokens": tokens,
                "gradient_norm": float(norm),
                "ids": entry["tasks"],
            }
            trace.append(record)
            if output is not None:
                emit(output, "calibration_update", **record)
    assert_fp32(adapter.state_dict())
    after = frozen_state(base, adapter)
    if before != after:
        raise ValueError("CALIBRATION_FROZEN_FULL_TENSOR_STATE_CHANGED")
    if set(trainable) != nonzero or tensor_hash(trainable) == initial_trainable:
        raise ValueError(
            f"CALIBRATION_INACTIVE_TRAINABLE_PARAMETERS: {sorted(set(trainable) - nonzero)}"
        )
    return {
        "optimizer_updates": len(trace),
        "task_microbatches": len(trace) * len(TASKS),
        "microbatches_per_task": dict(Counter(t for row in trace for t in row["ids"])),
        "example_exposures_per_task": {
            task: sum(len(row["ids"][task]) for row in trace) for task in TASKS
        },
        "schedule_sha256": digest(schedule),
        "trainable_names": sorted(trainable),
        "trainable_elements": sum(v.numel() for v in trainable.values()),
        "initial_trainable_sha256": initial_trainable,
        "final_trainable_sha256": tensor_hash(trainable),
        "all_trainables_received_finite_nonzero_gradient": True,
        "frozen_before": before,
        "frozen_after": after,
        "trace": trace,
    }


def save_adapter(adapter, path: Path) -> dict:
    if isinstance(adapter, PortalModel):
        save_native(adapter, path)
        kind = "portal"
    else:
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "config.json", adapter.config.to_dict())
        save_file(
            {
                name: v.detach().cpu().contiguous()
                for name, v in adapter.state_dict().items()
            },
            path / "model.safetensors",
        )
        kind = "lora"
    return {
        "kind": kind,
        "artifact": pin_bundle(path),
        "tensor_sha256": tensor_hash(adapter.state_dict()),
        "shared_sha256": shared_hash(adapter) if kind == "portal" else None,
    }


def restore_adapter(receipt: dict, device: str):
    spec = receipt["artifact"]
    verify_bundle(spec)
    if receipt["kind"] == "portal":
        adapter = load_native_bundle(spec)
    else:
        root = input_path(spec["path"])
        adapter = ZeroLora(
            PortalConfig.from_dict(json.loads((root / "config.json").read_text()))
        )
        adapter.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    if tensor_hash(adapter.state_dict()) != receipt["tensor_sha256"]:
        raise ValueError("CALIBRATION_RELOAD_TENSOR_MISMATCH")
    if (
        isinstance(adapter, PortalModel)
        and shared_hash(adapter) != receipt["shared_sha256"]
    ):
        raise ValueError("CALIBRATION_RELOAD_SHARED_MISMATCH")
    return adapter.to(device).requires_grad_(False).eval()


def final_rows(config: dict) -> dict:
    value = json.loads(input_path(config["inputs"]["fixture"]["path"]).read_text())
    fresh = json.loads(
        input_path(config["inputs"]["fresh_fixture"]["path"]).read_text()
    )
    panels = {
        "old_validation_test": [
            Example.from_dict(row)
            for task in TASKS
            for split in ("validation", "test")
            for row in value["splits"][f"{task}_{split}"]
        ],
        "fresh288": [Example.from_dict(row) for row in fresh["rows"]],
    }
    training_groups = {
        row["group"] for task in TASKS for row in value["splits"][f"{task}_train"]
    }
    previous = set(training_groups)
    for panel, rows in panels.items():
        if (
            not rows
            or {r.task for r in rows} != set(TASKS)
            or len({r.id for r in rows}) != len(rows)
        ):
            raise ValueError(f"CALIBRATION_INVALID_EVALUATION_ROWS: {panel}")
        expected = 96 if panel == "old_validation_test" else 288
        if config["role"] == "mechanistic_exploratory" and Counter(
            r.task for r in rows
        ) != dict.fromkeys(TASKS, expected):
            raise ValueError(f"CALIBRATION_EVALUATION_ROW_COUNT: {panel}")
        groups = {r.group for r in rows}
        if groups & previous:
            raise ValueError(f"CALIBRATION_TRAIN_EVALUATION_INPUT_OVERLAP: {panel}")
        previous |= groups
    return panels


def evaluate_adapter(base, adapter, rows: list) -> dict:
    if adapter is None or isinstance(adapter, PortalModel):
        return evaluate(
            base,
            rows,
            TRAINING["max_prompt"],
            CONTRACT["evaluation"]["batch_size"],
            adapter,
        )
    with (
        torch.inference_mode(),
        PortalInjector(base.model, adapter.config) as injector,
        injector.activate(adapter(TASKS[0])),
    ):
        return evaluate(
            base, rows, TRAINING["max_prompt"], CONTRACT["evaluation"]["batch_size"]
        )


def headroom_result(raw: float, initial: float, learned: float, lora: float) -> dict:
    if any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in (raw, initial, learned, lora)
    ):
        raise ValueError("CALIBRATION_INVALID_ACCURACY")
    denominator = lora - raw
    feasible = (
        initial + GATES["minimum_learned_gain_over_initial_calibrated"] <= 1 + 1e-12
    )
    enough = denominator >= GATES["minimum_lora_gain_over_raw"] - 1e-12
    return {
        "learned_minus_initial_calibrated": learned - initial,
        "learned_minus_lora_from_zero": learned - lora,
        "lora_minus_raw": denominator,
        "source_benefit_gate_feasible": feasible,
        "lora_headroom_sufficient": enough,
        "source_benefit_gate": (
            learned - initial
            >= GATES["minimum_learned_gain_over_initial_calibrated"] - 1e-12
        )
        if feasible and enough
        else None,
        "recovered_lift": (learned - raw) / denominator if enough else None,
        "status": "descriptive_gate_evaluable"
        if feasible and enough
        else "inconclusive_insufficient_headroom",
    }


def reload_evaluate(config: dict, output: Path, parent_pid: int) -> dict:
    if os.getpid() == parent_pid:
        raise ValueError("CALIBRATION_NEW_PID_REQUIRED")
    runtime(config)
    receipt_path = output / "training_receipt.json"
    verify_file(
        receipt_path, json.loads((output / "training_receipt.pin.json").read_text())
    )
    receipt = json.loads(receipt_path.read_text())
    if receipt["pid"] != parent_pid or receipt["config_sha256"] != digest(config):
        raise ValueError("CALIBRATION_PARENT_RECEIPT_MISMATCH")
    if verify_inputs(config) != receipt["input_manifest"]:
        raise ValueError("CALIBRATION_INPUTS_CHANGED_SINCE_TRAINING")
    panels = final_rows(config)
    base = load_base(config["base"], device=config["runtime"]["device"])
    before = tensor_hash(frozen_base_tensors(base.model))
    if before != receipt["base_sha256"]:
        raise ValueError("CALIBRATION_RELOADED_BASE_FULL_TENSOR_MISMATCH")
    results, reload_checks = {}, {}
    probes = [row for task in TASKS for row in calibration_rows(config)[task][:1]]
    for condition in ("raw", *FLOORS, *TRAINED):
        saved = receipt["checkpoints"].get(condition)
        adapter = restore_adapter(saved, config["runtime"]["device"]) if saved else None
        if adapter is not None:
            hook_receipt(base, adapter.config, config["expected_target_hooks"])
            if (
                evaluate_adapter(base, adapter, probes)
                != receipt["reload_probes"][condition]
            ):
                raise ValueError(f"CALIBRATION_RELOAD_PROBE_MISMATCH: {condition}")
        results[condition] = {
            panel: evaluate_adapter(base, adapter, rows)
            for panel, rows in panels.items()
        }
        if tensor_hash(frozen_base_tensors(base.model)) != before:
            raise ValueError(f"CALIBRATION_EVALUATION_BASE_CHANGED: {condition}")
        if (
            adapter is not None
            and tensor_hash(adapter.state_dict()) != saved["tensor_sha256"]
        ):
            raise ValueError(f"CALIBRATION_EVALUATION_ADAPTER_CHANGED: {condition}")
        reload_checks[condition] = {
            "full_base_tensors_exact": True,
            "saved_adapter_tensors_exact": True,
            "calibration_probe_predictions_and_scores_exact": adapter is not None,
        }
        emit(output, "final_evaluation", condition=condition)
        del adapter
    comparisons = {}
    for panel in panels:
        comparisons[panel] = {}
        for task in TASKS:
            metrics = {
                arm: result[panel]["metrics"][task] for arm, result in results.items()
            }
            comparison = headroom_result(
                *(
                    metrics[arm]["accuracy"]
                    for arm in (
                        "raw",
                        "initial_calibrated",
                        "learned_calibrated",
                        "lora_from_zero",
                    )
                )
            )
            comparison["initial_minus_learned_gold_nll"] = (
                metrics["initial_calibrated"]["gold_nll"]
                - metrics["learned_calibrated"]["gold_nll"]
            )
            comparisons[panel][task] = comparison
    result = {
        "status": "completed",
        "pid": os.getpid(),
        "training_pid": parent_pid,
        "new_pid_reload": True,
        "results": results,
        "comparisons": comparisons,
        "reload_checks": reload_checks,
        "claim_boundary": config["claim_boundary"],
        "data_exposure": config["data_exposure"],
        "config_sha256": digest(config),
    }
    write_json(output / "result.json", result)
    return result


def run_experiment(config: dict, output: Path, *, prepare_only: bool = False) -> dict:
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "packages.txt", "execution.json", "task.json", "run.log"}
    if any(
        path.is_symlink()
        or not (
            (path.name in allowed and path.is_file())
            or (path.name == "attempts" and path.is_dir())
        )
        for path in output.iterdir()
    ):
        raise FileExistsError(f"CALIBRATION_OUTPUT_ALREADY_USED: {output}")
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("CALIBRATION_DISPATCH_CONFIG_MISMATCH")
    runtime(config)
    if not config_path.exists():
        write_json(config_path, config)
    manifest = verify_inputs(config)
    write_json(output / "input_manifest.json", manifest)
    emit(
        output,
        "all_input_bytes_verified_before_model_load",
        config_sha256=digest(config),
    )
    selected = calibration_rows(config)
    schedule = balanced_schedule(selected)
    write_json(
        output / "calibration_rows.json",
        {task: [asdict(row) for row in rows] for task, rows in selected.items()},
    )
    write_json(output / "schedule.json", schedule)
    if prepare_only:
        return {"status": "prepared_not_trained", "input_manifest": manifest}
    initial = load_native_bundle(config["inputs"]["source_initial"])
    learned = load_native_bundle(config["inputs"]["source_learned"])
    target = load_native_bundle(config["inputs"]["target_portal"])
    if initial.config != learned.config or initial.config.tasks[-3:] != TASKS:
        raise ValueError("CALIBRATION_SOURCE_PAIR_MISMATCH")
    for name, model in (("initial", initial), ("learned", learned)):
        if (
            tensor_hash(model.state_dict())
            != config["source_provenance"][f"{name}_tensor_sha256"]
        ):
            raise ValueError(f"CALIBRATION_ORIGINAL_SOURCE_CHECKPOINT_MISMATCH: {name}")
    if any(
        len(model.config.projection_targets) != config["expected_source_hooks"]
        for model in (initial, learned)
    ):
        raise ValueError("CALIBRATION_SOURCE_HOOK_COUNT_MISMATCH")
    if learned.config.rank != 8 or target.config.rank != 8 or target.config.alpha != 16:
        raise ValueError("CALIBRATION_RANK8_ALPHA16_REQUIRED")
    if shared_hash(initial) == shared_hash(learned):
        raise ValueError("CALIBRATION_SOURCE_KNOWLEDGE_CONTROL_IDENTICAL")
    if tensor_hash(initial.alignment.state_dict()) != tensor_hash(
        learned.alignment.state_dict()
    ):
        raise ValueError("CALIBRATION_SOURCE_ALIGNMENT_WAS_NOT_FROZEN")
    if not torch.equal(initial.task_latents[:-3], learned.task_latents[:-3]):
        raise ValueError("CALIBRATION_SOURCE_PUBLISHED_VECTORS_CHANGED")
    rte = initial.task_latents[initial.config.tasks.index("rte")]
    if not torch.equal(initial.task_latents[-3:], rte.expand(3, -1)):
        raise ValueError("CALIBRATION_INITIAL_ABC_VECTORS_NOT_ORIGINAL_RTE")
    fresh = fresh_alignment(target)
    save_file(fresh, output / "fresh_alignment.safetensors")
    base = load_base(config["base"], device=config["runtime"]["device"])
    assert_fp32(frozen_base_tensors(base.model))
    hooks = hook_receipt(base, target.config, config["expected_target_hooks"])
    if hooks != config["target_hooks"]:
        raise ValueError("CALIBRATION_DECLARED_TARGET_HOOKS_MISMATCH")
    receipt = {
        "pid": os.getpid(),
        "config_sha256": digest(config),
        "input_manifest": manifest,
        "base_sha256": tensor_hash(frozen_base_tensors(base.model)),
        "hooks": hooks,
        "fresh_alignment_sha256": tensor_hash(fresh),
        "fresh_alignment_file": file_pin(output / "fresh_alignment.safetensors"),
        "source_shared_sha256": {
            "initial": shared_hash(initial),
            "learned": shared_hash(learned),
        },
        "source_hooks": [
            {
                "path": path,
                "A": [initial.config.rank, spec.in_features],
                "B": [spec.out_features, initial.config.rank],
            }
            for spec, path in initial.config.resolved_targets()
        ],
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "portallib", "safetensors")
        },
        "implementation": {
            str(Path(p).resolve()): file_pin(Path(p))
            for p in (
                __file__,
                learner.__file__,
                inspect.getfile(PortalAlignment),
                inspect.getfile(PortalModel),
                inspect.getfile(PortalInjector),
            )
        },
        "training": {},
        "checkpoints": {},
        "reload_probes": {},
    }
    probes = [row for task in TASKS for row in selected[task][:1]]
    for condition in (*FLOORS, *TRAINED):
        if condition == "lora_from_zero":
            adapter = ZeroLora(target.config)
        else:
            shared = initial if "initial" in condition else learned
            alignment = (
                target.alignment.state_dict()
                if condition.startswith("old_alignment")
                else fresh
            )
            adapter = make_target(shared, target, alignment)
            if (
                not condition.startswith("old_alignment")
                and tensor_hash(adapter.alignment.state_dict())
                != receipt["fresh_alignment_sha256"]
            ):
                raise ValueError("CALIBRATION_PAIRED_INITIALIZATION_MISMATCH")
            if condition.startswith("fresh"):
                with torch.no_grad():
                    for task in TASKS:
                        if any(
                            torch.count_nonzero(b)
                            for _, b in factors_for(adapter, task).values()
                        ):
                            raise ValueError(
                                "CALIBRATION_FRESH_ALIGNMENT_NOT_ZERO_DELTA"
                            )
        adapter = adapter.to(config["runtime"]["device"])
        if condition in TRAINED:
            arm_output = output / condition
            arm_output.mkdir()
            receipt["training"][condition] = fit(
                base, adapter, selected, schedule, arm_output
            )
        receipt["checkpoints"][condition] = save_adapter(
            adapter, output / "checkpoints" / condition
        )
        receipt["reload_probes"][condition] = evaluate_adapter(base, adapter, probes)
        emit(output, "checkpoint_saved", condition=condition)
        del adapter
    if (
        receipt["training"]["learned_calibrated"]["initial_trainable_sha256"]
        != receipt["training"]["initial_calibrated"]["initial_trainable_sha256"]
    ):
        raise ValueError("CALIBRATION_ARMS_DID_NOT_START_IDENTICALLY")
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("CALIBRATION_BASE_CHANGED_ACROSS_ARMS")
    write_json(output / "training_receipt.json", receipt)
    write_json(
        output / "training_receipt.pin.json", file_pin(output / "training_receipt.json")
    )
    del base, initial, learned, target
    gc.collect()
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(Path(__file__).resolve()),
        "--config",
        str(output / "config.json"),
        "--output-dir",
        str(output),
        "--reload-evaluate",
        str(os.getpid()),
    ]
    subprocess.run(
        command,
        check=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    return json.loads((output / "result.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reload-evaluate", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    try:
        if args.reload_evaluate is not None:
            reload_evaluate(config, args.output_dir.resolve(), args.reload_evaluate)
        else:
            run_experiment(
                config, args.output_dir.resolve(), prepare_only=args.prepare_only
            )
    except Exception as exc:
        if args.output_dir.is_dir():
            emit(
                args.output_dir,
                "calibration_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
