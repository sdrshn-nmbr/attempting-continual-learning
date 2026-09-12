from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch
from portallib import PortalBase, PortalModel
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

from compare import BOUNDARIES, load_run, prediction_index, same_examples
from data import SEQUENCE_TASKS, digest, sequence_splits, write_json
from learner import (
    NUMERICAL_GATE,
    evaluate,
    fp32_forward,
    frozen_base_tensors,
    load_portal,
    save_native,
    shared_hash,
    tensor_hash,
    tensor_storage,
    transplant,
)
from metrics import SELECTION_RULE, paired_group_interval, sequence_metrics, task_rows
from run import emit, require_gpu

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("raw", "initial", "native_no_replay", "native_replay")
TARGETS = ("qwen4", "mistral7", "gemma3", "gemma4")
CONFIG_DIGESTS = {
    ("qwen4", 303): "97ec30db3bbc1cb28c4f4672dc91b1aad6e9ae4f127959c9f69e94e02e6babc6",
    (
        "mistral7",
        303,
    ): "78f0850951d0fa15694326e5e75298a182b118ccd015b954b02a325d18f13bd1",
    ("gemma3", 303): "bf165b3bf4340ca3b025d57edb3f11f557f69a96ef1db9586e2c0c2a103a83a8",
    ("gemma4", 303): "e972d933fb1445c8498ff7294a8b8e4f93a370de984e6f39a065e75ceff068af",
    ("qwen4", 419): "468af3b91840e6161b99fe4f6903662dc0ccd1d0ab95fd7afbc272465c0ac645",
    (
        "mistral7",
        419,
    ): "29663812f33b1f60908002adef474a3559cd4afdcb20e9b5fc3a508eb7cdb67c",
    ("gemma3", 419): "ab83ebaef1f58d29c94e12accf10844667a4ddab39e022b002c368bf1f825855",
    ("gemma4", 419): "860bd17fffbd0b878daca2ad5694bfee9a90cbdf19700f3b6f888c0117be62b8",
}
PRIMARY_BUNDLE = "370dfd20b44b1fa2f66a8b0a6b67ca15217b239cff4c112ad38a98077b53533c"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixed_config(path: Path) -> dict:
    config = read_json(path)
    target = config.get("target", {}).get("name")
    fixture = config.get("fixture", {}).get("fixture_seed")
    if digest(config) != CONFIG_DIGESTS.get((target, fixture)):
        raise ValueError("UNDECLARED_TARGET_EVALUATION_CONFIG")
    return config


def verify_primary_files(config: dict) -> dict:
    expected = config["primary_source_files"]
    if len(expected) != 20 or config["primary_source_bundle_sha256"] != PRIMARY_BUNDLE:
        raise ValueError("PRIMARY_SOURCE_MANIFEST_MISMATCH")
    combined = hashlib.sha256()
    for name in sorted(expected, key=Path):
        checksum = expected[name]
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"PRIMARY_SOURCE_CHANGED: {name}")
        combined.update(name.encode() + b"\0" + data)
    if combined.hexdigest() != PRIMARY_BUNDLE:
        raise ValueError("PRIMARY_SOURCE_BUNDLE_MISMATCH")
    return expected


def validate_bootstrap(config: dict, output: Path) -> bool:
    entries = {path.name: path for path in output.iterdir()}
    required = {"task.json", "execution.json", "config.json", "packages.txt", "run.log"}
    if not entries:
        return False
    if (
        not required <= entries.keys()
        or entries.keys() - required - {"attempts"}
        or any(
            entries[name].is_symlink() or not entries[name].is_file()
            for name in required
        )
    ):
        raise ValueError("TARGET_OUTPUT_CONTAINS_RESEARCH_OR_INVALID_BOOTSTRAP")
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
    task, execution = (
        read_json(entries[name]) for name in ("task.json", "execution.json")
    )
    if read_json(entries["config.json"]) != config or task.get("config") != config:
        raise ValueError("SUPERVISOR_BOOTSTRAP_CONFIG_MISMATCH")
    if (
        task.get("id") != output.name
        or execution.get("task_id") != task["id"]
        or execution.get("status") != "running"
        or execution.get("task") != task
    ):
        raise ValueError("SUPERVISOR_BOOTSTRAP_TASK_MISMATCH")
    return True


def source_readiness(config: dict) -> dict[str, Path]:
    directories = {}
    for condition, spec in config["source_runs"].items():
        directory = Path(config["source_runs_root"]) / spec["id"]
        execution = read_json(directory / "execution.json")
        if execution.get("status") != "completed" or execution.get("exit_code") != 0:
            raise ValueError(f"SOURCE_RUN_NOT_COMPLETED: {spec['id']}")
        for name in ("result.json", "training_receipt.json", "evaluation.json"):
            if not (directory / name).is_file():
                raise ValueError(
                    f"SOURCE_REQUIRED_RECEIPT_MISSING: {spec['id']}/{name}"
                )
        result = read_json(directory / "result.json")
        if (
            result.get("qualified") is not True
            or result.get("mechanically_qualified") is not True
        ):
            raise ValueError(f"SOURCE_NOT_MECHANICALLY_QUALIFIED: {spec['id']}")
        source = result.get("source", {})
        if (
            source.get("selection_rule") != SELECTION_RULE
            or set(source.get("tasks", {})) != set(SEQUENCE_TASKS)
            or any(
                type(source["tasks"][task].get("acquired")) is not bool
                for task in SEQUENCE_TASKS
            )
        ):
            raise ValueError(f"SOURCE_ACQUISITION_UNAVAILABLE: {spec['id']}")
        task = read_json(directory / "task.json")
        primary_config = read_json(ROOT / spec["config_path"])
        if (
            directory.name != spec["id"]
            or task.get("id") != spec["id"]
            or execution.get("task_id") != spec["id"]
            or execution.get("task") != task
            or task.get("source_sha256") != PRIMARY_BUNDLE
            or execution.get("runtime_sha256") != config["runtime_sha256"]
            or task.get("config") != primary_config
            or read_json(directory / "config.json") != primary_config
        ):
            raise ValueError(f"SOURCE_TASK_CONFIG_OR_RUNTIME_MISMATCH: {spec['id']}")
        if (
            file_sha256(directory / "training_receipt.json")
            != spec["training_receipt_sha256"]
        ):
            raise ValueError(f"SOURCE_TRAINING_RECEIPT_CHANGED: {spec['id']}")
        directories[condition] = directory
    if tuple(directories) != CONDITIONS[2:]:
        raise ValueError("EXACT_TWO_NATIVE_SOURCE_ARMS_REQUIRED")
    return directories


def fixture_test_rows(config: dict) -> list:
    splits, _ = sequence_splits(config["fixture"])
    rows = [row for task in SEQUENCE_TASKS for row in splits[f"{task}_test"]]
    if len(rows) != 192 or Counter(row.task for row in rows) != dict.fromkeys(
        SEQUENCE_TASKS, 64
    ):
        raise ValueError("EXACT_192_TEST_ROWS_REQUIRED")
    return rows


def row_identity(rows: list) -> dict:
    return {
        row.id: {
            "task": row.task,
            "group": row.group,
            "gold": row.gold_idx,
            "prompt_sha256": digest(row.prompt),
        }
        for row in rows
    }


def frozen_fp32(module: torch.nn.Module, label: str) -> dict:
    tensors = dict(module.named_parameters()) | {
        f"buffer:{name}": value for name, value in module.named_buffers()
    }
    for name, value in tensors.items():
        if value.requires_grad or value.grad is not None:
            raise ValueError(f"TARGET_STATE_NOT_FROZEN: {label}/{name}")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"TARGET_STATE_NOT_FP32: {label}/{name}/{value.dtype}")
    return tensor_storage(tensors)


def portal_identity(portal: PortalModel) -> dict:
    return {
        "checkpoint_sha256": tensor_hash(portal.state_dict()),
        "config_sha256": digest(portal.config.to_dict()),
        "shared_sha256": shared_hash(portal),
        "core_sha256": tensor_hash(portal.core.state_dict()),
        "task_table_sha256": tensor_hash({"task_latents": portal.task_latents}),
        "published_vectors_sha256": tensor_hash(
            {"published": portal.task_latents[:14]}
        ),
        "sequence_vectors_sha256": {
            task: tensor_hash(
                {"vector": portal.task_latents[portal.config.tasks.index(task)]}
            )
            for task in SEQUENCE_TASKS
            if task in portal.config.tasks
        },
        "alignment_sha256": tensor_hash(portal.alignment.state_dict()),
        "tasks": list(portal.config.tasks),
        "architecture": {
            **portal.config.architecture_kwargs(),
            "modules": list(portal.config.modules),
        },
    }


def checkpoint_files(directory: Path) -> dict:
    return {
        name: file_sha256(directory / name)
        for name in ("config.json", "model.safetensors")
    }


def load_source_checkpoint(
    directory: Path, expected: str, primary: dict, config: dict
) -> PortalModel:
    portal = (
        PortalModel.from_pretrained(
            directory, local_files_only=True, device="cpu", dtype=torch.float32
        )
        .requires_grad_(False)
        .eval()
    )
    frozen_fp32(portal, str(directory))
    portal.validate_base_model(
        primary["models"]["qwen8"]["base"]["repo_id"],
        primary["models"]["qwen8"]["base"]["revision"],
    )
    if tuple(portal.config.tasks) != (*config["published_tasks"], *SEQUENCE_TASKS):
        raise ValueError("SOURCE_TASK_TABLE_MISMATCH")
    if tensor_hash(portal.state_dict()) != expected:
        raise ValueError(f"SOURCE_CHECKPOINT_HASH_MISMATCH: {directory}")
    return portal


def validated_sources(config: dict) -> tuple[dict, dict[str, PortalModel], list]:
    directories = source_readiness(config)
    verify_primary_files(config)
    rows = fixture_test_rows(config)
    expected_rows = row_identity(rows)
    receipts, states = {}, {}
    for condition, directory in directories.items():
        record = load_run(directory)
        primary = record["config"]
        preregistration = record["preregistration"]
        if (
            primary["sequence"] != config["fixture"]
            or preregistration["selection_rule"] != SELECTION_RULE
            or preregistration["numerical_gate"] != NUMERICAL_GATE
            or preregistration["source_sha256"]
            != {
                name: config["primary_source_files"][name]
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
        ):
            raise ValueError("SOURCE_PREREGISTRATION_MISMATCH")
        evaluation = record["evaluation"]
        for boundary in BOUNDARIES:
            same_examples(
                expected_rows,
                prediction_index(
                    evaluation["timeline"][boundary]["evaluation"]["predictions"]
                ),
            )
        acquisition = sequence_metrics(
            evaluation["timeline"],
            evaluation["baselines"]["train"]["raw"],
            primary["seed"],
        )
        if acquisition != record["result"]["source"]:
            raise ValueError("SOURCE_ACQUISITION_RECEIPT_MISMATCH")
        training = record["training_receipt"]
        if (
            training["budget"]["optimizer_updates"] != 288
            or not training["checks"]
            or not all(training["checks"].values())
        ):
            raise ValueError("SOURCE_TRAINING_INCOMPLETE")
        for boundary in BOUNDARIES[1:]:
            reloaded = evaluation["timeline"][boundary]["reload"]
            if (
                reloaded["passed"] is not True
                or reloaded["fresh_process"] is not True
                or reloaded["trainer_pid"] != training["pid"]
                or reloaded["evaluator_pid"] == training["pid"]
            ):
                raise ValueError(f"SOURCE_EVALUATOR_RECEIPT_MISMATCH: {boundary}")
        initial_directory = directory / "initial/native/checkpoint"
        final_directory = directory / "stages/after_c/checkpoint"
        initial_receipt = read_json(
            initial_directory.parent / "checkpoint_receipt.json"
        )
        final_receipt = read_json(final_directory.parent / "checkpoint_receipt.json")
        expected_initial = config["initial_checkpoint_sha256"]
        expected_final = config["source_runs"][condition]["final_checkpoint_sha256"]
        if (
            initial_receipt["checkpoint_sha256"] != expected_initial
            or evaluation["initialization"]["checkpoints"]["native"][
                "checkpoint_sha256"
            ]
            != expected_initial
            or final_receipt["checkpoint_sha256"] != expected_final
            or training["stages"]["after_c"]["checkpoint"]["checkpoint_sha256"]
            != expected_final
            or evaluation["timeline"]["after_c"]["reload"]["checkpoint_sha256"]
            != expected_final
            or evaluation["timeline"]["after_c"]["reload"]["passed"] is not True
        ):
            raise ValueError("SOURCE_CHECKPOINT_RECEIPT_MISMATCH")
        initial = load_source_checkpoint(
            initial_directory, expected_initial, primary, config
        )
        final = load_source_checkpoint(final_directory, expected_final, primary, config)
        initial_identity, final_identity = (
            portal_identity(initial),
            portal_identity(final),
        )
        if (
            initial_identity["published_vectors_sha256"]
            != final_identity["published_vectors_sha256"]
            or initial_identity["alignment_sha256"]
            != final_identity["alignment_sha256"]
            or initial_receipt["frozen"]["base_sha256"]
            != final_receipt["frozen"]["base_sha256"]
            or initial_identity["checkpoint_sha256"]
            == final_identity["checkpoint_sha256"]
        ):
            raise ValueError("SOURCE_FROZEN_OR_LEARNED_STATE_MISMATCH")
        rte = initial.task_latents[initial.config.tasks.index("rte")]
        if not torch.equal(initial.task_latents[-3:], rte.unsqueeze(0).repeat(3, 1)):
            raise ValueError("SOURCE_INITIAL_VECTORS_NOT_UNTOUCHED_RTE")
        if (
            "initial" in states
            and portal_identity(states["initial"]) != initial_identity
        ):
            raise ValueError("NATIVE_SOURCE_INITIAL_CHECKPOINTS_DIFFER")
        states.setdefault("initial", initial)
        states[condition] = final
        files = {
            str(directory / name): value
            for name, value in record["provenance"]["artifact_sha256"].items()
        }
        for checkpoint in (initial_directory, final_directory):
            files.update(
                {
                    str(checkpoint / name): value
                    for name, value in checkpoint_files(checkpoint).items()
                }
            )
            files[str(checkpoint.parent / "checkpoint_receipt.json")] = file_sha256(
                checkpoint.parent / "checkpoint_receipt.json"
            )
        receipts[condition] = {
            "task_id": config["source_runs"][condition]["id"],
            "source_bundle_sha256": PRIMARY_BUNDLE,
            "runtime_sha256": record["execution"]["runtime_sha256"],
            "initial": initial_identity,
            "final": final_identity,
            "acquisition": acquisition,
            "files": files,
        }
    return receipts, states, rows


def verify_snapshot(spec: dict) -> dict:
    root = Path(spec["local_path"])
    if root.name != spec["revision"]:
        raise ValueError("TARGET_SNAPSHOT_REVISION_MISMATCH")
    checked = {}
    for expected in spec["files"]:
        path = root / expected["path"]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"TARGET_SNAPSHOT_FILE_MISMATCH: {path}")
        checksum = file_sha256(path)
        if expected["sha256"] is not None:
            valid = checksum == expected["sha256"]
        else:
            blob = path.read_bytes()
            valid = (
                hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
                == expected["git_blob_sha1"]
            )
        if not valid:
            raise ValueError(f"TARGET_SNAPSHOT_HASH_MISMATCH: {path}")
        checked[expected["path"]] = checksum
    return checked


def load_target_base(target: dict, device: str) -> tuple[PortalBase, dict]:
    spec = target["base"]
    files = verify_snapshot(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["local_path"], local_files_only=True, trust_remote_code=False
    )
    if target["pad_token"]:
        tokenizer.pad_token = target["pad_token"]
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("TARGET_TOKENIZER_HAS_NO_PADDING_TOKEN")
    if target["loader"] == "causal_lm":
        model_class = AutoModelForCausalLM
    elif target["loader"] == "multimodal_lm":
        model_class = AutoModelForMultimodalLM
    else:
        raise ValueError("UNDECLARED_TARGET_MODEL_LOADER")
    model = model_class.from_pretrained(
        spec["local_path"],
        dtype=torch.float32,
        device_map={"": device},
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    base = PortalBase(
        model_id=spec["repo_id"],
        revision=spec["revision"],
        model=model,
        tokenizer=tokenizer,
        layer_path=target["layer_path"],
        allow_heterogeneous_targets=target["heterogeneous"],
    )
    base.freeze(gradient_checkpointing=False)
    model.eval()
    storage = frozen_fp32(model, target["name"])
    for projection in target["projection_targets"]:
        path = f"{target['layer_path']}.{projection['layer_index']}.{projection['module_path']}"
        module = model.get_submodule(path)
        if not isinstance(module, torch.nn.Linear) or (
            module.in_features,
            module.out_features,
        ) != (projection["in_features"], projection["out_features"]):
            raise ValueError(f"TARGET_PROJECTION_TOPOLOGY_MISMATCH: {path}")
    return base, {
        "files": files,
        "storage": storage,
        "loader": model_class.__name__,
        "layer_path": base.layer_path,
        "heterogeneous": base.allow_heterogeneous_targets,
    }


def prepare_target(config: dict, output: Path) -> dict:
    bootstrap = validate_bootstrap(config, output)
    sources, states, rows = validated_sources(config)
    target = config["target"]
    target_files = verify_snapshot(target["portal"])
    original = load_portal(target["portal"]).eval()
    frozen_fp32(original, "original_target_portal")
    original.validate_base_model(target["base"]["repo_id"], target["base"]["revision"])
    if (
        list(original.config.tasks) != config["published_tasks"]
        or original.config.layer_path != target["layer_path"]
        or list(original.config.to_dict()["projection_targets"])
        != target["projection_targets"]
        or not torch.equal(original.task_latents, states["initial"].task_latents[:14])
    ):
        raise ValueError("ORIGINAL_TARGET_PORTAL_CONTRACT_MISMATCH")
    original_identity = portal_identity(original)
    if not bootstrap:
        write_json(output / "config.json", config)
    checkpoints = {}
    for condition in CONDITIONS[1:]:
        source = states[condition]
        adapted = transplant(source, original).requires_grad_(False).eval()
        frozen_fp32(adapted, condition)
        identity = portal_identity(adapted)
        source_identity = portal_identity(source)
        if (
            any(
                identity[key] != source_identity[key]
                for key in (
                    "shared_sha256",
                    "core_sha256",
                    "task_table_sha256",
                    "published_vectors_sha256",
                    "sequence_vectors_sha256",
                    "tasks",
                    "architecture",
                )
            )
            or identity["alignment_sha256"] != original_identity["alignment_sha256"]
        ):
            raise ValueError(f"TARGET_TRANSPLANT_IDENTITY_MISMATCH: {condition}")
        directory = output / "checkpoints" / condition
        save_native(adapted, directory)
        checkpoints[condition] = {
            "identity": identity,
            "files": checkpoint_files(directory),
        }
    prepared = {
        "parent_pid": os.getpid(),
        "config_sha256": digest(config),
        "primary_source_files": config["primary_source_files"],
        "sources": sources,
        "target_original": original_identity,
        "target_portal_files": target_files,
        "checkpoints": checkpoints,
        "fixture_sha256": config["fixture"]["sha256"],
        "test_identity_sha256": digest(row_identity(rows)),
        "test_rows": len(rows),
        "target_optimizer_updates": 0,
        "calibration_examples": 0,
    }
    write_json(output / "prepared.json", prepared)
    emit(
        output,
        "target_checkpoints_prepared",
        target=target["name"],
        checkpoints=3,
        parent_pid=os.getpid(),
    )
    return prepared


def verify_source_files(prepared: dict) -> None:
    for source in prepared["sources"].values():
        for name, expected in source["files"].items():
            if file_sha256(Path(name)) != expected:
                raise ValueError(
                    f"SOURCE_ARTIFACT_CHANGED_DURING_TARGET_EVALUATION: {name}"
                )


def interpretation(panels: dict, sources: dict, seed: int, fixture_seed: int) -> dict:
    result = {}
    for condition in CONDITIONS[2:]:
        source = sources[condition]["acquisition"]
        tasks = {}
        for task in SEQUENCE_TASKS:
            current = panels[condition]["metrics"][task]["accuracy"]
            raw = panels["raw"]["metrics"][task]["accuracy"]
            initial = panels["initial"]["metrics"][task]["accuracy"]
            acquired = source["tasks"][task]["acquired"]
            retained = (
                acquired
                and source["tasks"][task]["maximum_forgetting"]
                <= SELECTION_RULE["maximum_task_forgetting_at_any_later_boundary"]
                + 1e-12
            )
            gain = current - max(raw, initial)
            tasks[task] = {
                "source_acquired": acquired,
                "source_acquired_and_retained": retained,
                "source_acquisition": source["tasks"][task],
                "gain_over_raw": current - raw,
                "gain_over_initial": current - initial,
                "gain_over_both_floors": gain,
                "clears_target_gain_threshold": gain
                >= SELECTION_RULE["minimum_target_gain"] - 1e-12,
                "acquired_source_target_gain": acquired
                and gain >= SELECTION_RULE["minimum_target_gain"] - 1e-12,
                "retained_source_target_gain": retained
                and gain >= SELECTION_RULE["minimum_target_gain"] - 1e-12,
                "paired_rows": 64,
                "gain_intervals": {
                    floor: paired_group_interval(
                        task_rows(panels[condition], task),
                        task_rows(panels[floor], task),
                        seed,
                    )
                    for floor in ("raw", "initial")
                },
            }
        result[condition] = {
            "tasks": tasks,
            "source_sequence_learning_signal": source["sequence_learning_signal"],
            "all_tasks_acquired_source_target_gain": all(
                row["acquired_source_target_gain"] for row in tasks.values()
            ),
            "all_tasks_retained_source_target_gain": all(
                row["retained_source_target_gain"] for row in tasks.values()
            ),
            "paired_rows": 192,
        }
    return {
        "arms": result,
        "replay_minus_no_replay": paired_group_interval(
            panels["native_replay"]["predictions"],
            panels["native_no_replay"]["predictions"],
            seed,
        ),
        "selected_arm": None,
        "claim_boundary": f"Descriptive paired results on fixture{fixture_seed}. Acquired-source target gain requires that task's own source acquisition and gain over both target floors. Retained-source target gain additionally requires source retention. No target or arm is selected.",
    }


def evaluate_prepared(
    config: dict, output: Path, prepared_sha256: str, device: str = "cuda:0"
) -> dict:
    prepared_path = output / "prepared.json"
    if file_sha256(prepared_path) != prepared_sha256:
        raise ValueError("PREPARED_TARGET_RECEIPT_CHANGED")
    prepared = read_json(prepared_path)
    if prepared["parent_pid"] == os.getpid() or prepared["config_sha256"] != digest(
        config
    ):
        raise ValueError("FRESH_TARGET_EVALUATOR_PROCESS_REQUIRED")
    verify_primary_files(config)
    verify_source_files(prepared)
    rows = fixture_test_rows(config)
    if (
        digest(row_identity(rows)) != prepared["test_identity_sha256"]
        or len(rows) != prepared["test_rows"]
    ):
        raise ValueError("PREPARED_TARGET_FIXTURE_MISMATCH")
    if device != "cpu":
        require_gpu(output, config["seed"])
    base, loaded = load_target_base(config["target"], device)
    original = load_portal(config["target"]["portal"]).eval()
    if (
        verify_snapshot(config["target"]["portal"]) != prepared["target_portal_files"]
        or portal_identity(original) != prepared["target_original"]
    ):
        raise ValueError("ORIGINAL_TARGET_PORTAL_CHANGED")
    before_base = tensor_hash(frozen_base_tensors(base.model))
    expected_rows = row_identity(rows)
    panels, checks, arithmetic = {}, {}, {}
    for condition in CONDITIONS:
        portal = None
        identity = None
        if condition != "raw":
            directory = output / "checkpoints" / condition
            expected = prepared["checkpoints"][condition]
            if checkpoint_files(directory) != expected["files"]:
                raise ValueError(f"TRANSPORTED_CHECKPOINT_FILES_CHANGED: {condition}")
            portal = (
                PortalModel.from_pretrained(
                    directory, local_files_only=True, device=device, dtype=torch.float32
                )
                .requires_grad_(False)
                .eval()
            )
            identity = portal_identity(portal)
            if (
                identity != expected["identity"]
                or identity["alignment_sha256"]
                != prepared["target_original"]["alignment_sha256"]
            ):
                raise ValueError(f"TRANSPORTED_CHECKPOINT_RELOAD_MISMATCH: {condition}")
            frozen_fp32(portal, condition)
        observations = []

        def inspect_logits(
            module, inputs, result, *, condition=condition, observations=observations
        ):
            observed = {
                "logits_dtype": str(result.logits.dtype),
                "autocast_enabled": torch.is_autocast_enabled(base.device.type),
                "grad_enabled": torch.is_grad_enabled(),
            }
            if observed != {
                "logits_dtype": "torch.float32",
                "autocast_enabled": False,
                "grad_enabled": False,
            }:
                raise ValueError(
                    f"TARGET_FORWARD_ARITHMETIC_MISMATCH: {condition}/{observed}"
                )
            observations.append(observed)

        handle = base.model.register_forward_hook(inspect_logits)
        try:
            with torch.inference_mode(), fp32_forward(base):
                panel = evaluate(
                    base,
                    rows,
                    config["evaluation"]["max_prompt"],
                    config["evaluation"]["batch_size"],
                    portal,
                )
        finally:
            handle.remove()
        same_examples(expected_rows, prediction_index(panel["predictions"]))
        after_base = tensor_hash(frozen_base_tensors(base.model))
        frozen_fp32(base.model, "base_after_evaluation")
        if before_base != after_base or (
            portal is not None and portal_identity(portal) != identity
        ):
            raise ValueError(f"TARGET_EVALUATION_CHANGED_STATE: {condition}")
        if not observations:
            raise ValueError("TARGET_EVALUATOR_DID_NOT_RUN")
        panels[condition] = panel
        arithmetic[condition] = {"forward_calls": len(observations), **observations[0]}
        checks[condition] = {
            "base_before_sha256": before_base,
            "base_after_sha256": after_base,
            "checkpoint": identity,
            "frozen": True,
            "reload_from_disk": condition != "raw",
        }
        del portal
    verify_source_files(prepared)
    result = {
        "kind": f"sequence{config['fixture']['fixture_seed']}_single_target_evaluation",
        "target": config["target"]["name"],
        "config_sha256": digest(config),
        "fixture_seed": config["fixture"]["fixture_seed"],
        "fixture_sha256": config["fixture"]["sha256"],
        "prepared_sha256": prepared_sha256,
        "parent_pid": prepared["parent_pid"],
        "evaluator_pid": os.getpid(),
        "fresh_process": True,
        "device": str(base.device),
        "mechanically_qualified": True,
        "conditions": list(CONDITIONS),
        "panel_count": 4,
        "prediction_count": 768,
        "paired_rows": 192,
        "task_paired_rows": dict.fromkeys(SEQUENCE_TASKS, 64),
        "target_optimizer_updates": 0,
        "calibration_examples": 0,
        "target_loader": loaded,
        "panels": panels,
        "checks": checks,
        "arithmetic": arithmetic,
        "source_identity": {
            condition: {
                key: source[key]
                for key in (
                    "task_id",
                    "source_bundle_sha256",
                    "runtime_sha256",
                    "initial",
                    "final",
                )
            }
            for condition, source in prepared["sources"].items()
        },
        "interpretation": interpretation(
            panels,
            prepared["sources"],
            config["seed"],
            config["fixture"]["fixture_seed"],
        ),
    }
    write_json(output / "result.json", result)
    emit(
        output,
        "target_evaluation_complete",
        target=result["target"],
        panels=4,
        predictions=768,
        target_optimizer_updates=0,
    )
    return result


def aggregate_results(reports: list[dict]) -> dict:
    indexed = {report["target"]: report for report in reports}
    if len(reports) != 4 or set(indexed) != set(TARGETS):
        raise ValueError("EXACT_FOUR_TARGET_RESULTS_REQUIRED")
    reference = reports[0]
    examples = prediction_index(reference["panels"]["raw"]["predictions"])
    for report in reports:
        if (
            report["mechanically_qualified"] is not True
            or report["config_sha256"]
            != CONFIG_DIGESTS.get((report["target"], report["fixture_seed"]))
            or report["fixture_seed"] != reference["fixture_seed"]
            or report["kind"]
            != f"sequence{report['fixture_seed']}_single_target_evaluation"
            or report["panel_count"] != 4
            or report["prediction_count"] != 768
            or report["fresh_process"] is not True
            or report["parent_pid"] == report["evaluator_pid"]
            or report["conditions"] != list(CONDITIONS)
            or set(report["panels"]) != set(CONDITIONS)
            or report["fixture_sha256"] != reference["fixture_sha256"]
            or report["source_identity"] != reference["source_identity"]
            or report["target_optimizer_updates"] != 0
            or report["calibration_examples"] != 0
        ):
            raise ValueError("TARGET_RESULT_MATRIX_MISMATCH")
        for panel in report["panels"].values():
            same_examples(examples, prediction_index(panel["predictions"]))
    if (
        len(examples) != 192
        or sum(
            len(panel["predictions"])
            for report in reports
            for panel in report["panels"].values()
        )
        != 3072
    ):
        raise ValueError("TARGET_RESULT_PREDICTION_COUNT_MISMATCH")
    return {
        "targets": indexed,
        "fixture_seed": reference["fixture_seed"],
        "fixture_sha256": reference["fixture_sha256"],
        "panel_count": 16,
        "prediction_count": 3072,
        "paired_rows": 192,
        "selected_target": None,
        "interpretation": "All four preregistered targets and conditions are reported; no best-target or best-arm selection.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "evaluate"), default="run")
    parser.add_argument("--prepared-sha256")
    args = parser.parse_args()
    config = fixed_config(args.config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "evaluate":
            if args.prepared_sha256 is None:
                raise ValueError("PARENT_PREPARED_RECEIPT_HASH_REQUIRED")
            evaluate_prepared(config, output, args.prepared_sha256)
            return
        prepare_target(config, output)
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
            "--phase",
            "evaluate",
            "--prepared-sha256",
            file_sha256(output / "prepared.json"),
        ]
        with (output / "evaluation.log").open("w") as stream:
            subprocess.run(
                command,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                },
            )
        result = read_json(output / "result.json")
        if (
            result["parent_pid"] != os.getpid()
            or not result["fresh_process"]
            or result["panel_count"] != 4
            or result["prediction_count"] != 768
        ):
            raise ValueError("TARGET_CHILD_COMPLETION_RECEIPT_MISMATCH")
        emit(
            output,
            "target_job_complete",
            target=result["target"],
            panels=4,
            predictions=768,
        )
    except Exception as exc:
        failure = output / "failure.json"
        if not failure.exists():
            write_json(
                failure,
                {
                    "phase": args.phase,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        emit(output, "target_evaluation_failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
