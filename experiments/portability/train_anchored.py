from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from portallib import PortalModel
from safetensors.torch import load_file

import run as original
from anchoring import AnchoredSequenceLearner, anchor_schedule
from data import SEQUENCE_TASKS, digest, prepare_data, read_data, write_json
from learner import (
    SequenceLearner,
    evaluate,
    frozen_base_tensors,
    load_base,
    parameter_plan,
    tensor_hash,
)
from repair_alignment import PUBLISHED_TASKS, file_hash, verify_checkpoint_files

ROOT = Path(__file__).resolve().parent
CONFIG_KEYS = {
    "kind",
    "version",
    "arm",
    "anchor_weight",
    "source_config",
    "source_initial",
    "targets",
    "anchors",
    "gates",
    "target_confirmation",
    "frozen_dependencies_sha256",
    "claim_boundary",
}
TRAIN_TASKS = [
    "rte",
    "cb",
    "copa",
    "wic",
    "wsc",
    "boolq",
    "arc_easy",
    "hellaswag",
    "openbookqa",
    "commonsense_qa",
]
HOLDOUT_TASKS = ["truthfulqa", "arc_challenge", "winogrande", "sciq"]
ANCHOR_PROTOCOL = {
    "surfaces": ["qwen8", "qwen4", "mistral7"],
    "train_tasks": TRAIN_TASKS,
    "holdout_tasks": HOLDOUT_TASKS,
    "seed": 90302,
    "parameter_dtype": "float32",
    "gram_dtype": "float64",
    "normalization_epsilon": 1e-12,
    "published_reference": "initial_checkpoint",
    "prior_reference": "own_acquisition_boundary",
    "pool_weighting": "equal_available_pools",
    "surface_weighting": "equal",
    "projection_weighting": "equal",
    "published_schedule": "shuffled_cycles_one_per_step",
    "prior_schedule": "round_robin_one_per_step",
    "trainable": "shared_core_and_current_vector",
    "alignments": "frozen_all_three",
    "teacher_factor_generation": "same_device_fp32_before_first_use_no_llm_forward",
}
SOURCE_GATES = {
    "own_train_accuracy_min": 0.9,
    "own_validation_accuracy_min": 0.9,
    "own_validation_gain_min": 0.5,
    "max_later_validation_forgetting": 0.05,
    "failure_action": "skip_target_evaluation_after_completing_fixed_budget",
    "checkpoint_selection": "fixed_final_per_stage",
    "updates_each_arm": 288,
    "reloaded_all_boundaries": True,
}
CONFIRMATION = {
    "path": "data/sequence303_unused_inputs.json",
    "sha256": "4f298979c5f6e76ed4a1ca8d07d1880a20499bd5911d8991591ece3934e8ce2e",
    "targets": ["qwen4", "mistral7"],
    "rows_each_task": 288,
    "scope": "all_previously_unused_303_input_triples",
    "source_trainer_access": "file_hash_only_never_deserialize",
    "new_mapping_replication": False,
    "new_training_seed_replication": False,
    "owner": "parent_behavioral_evaluation",
}


def validate_config(config: dict) -> None:
    if (
        set(config) != CONFIG_KEYS
        or config["kind"] != "portal_function_anchoring_sequence"
        or type(config["version"]) is not int
        or config["version"] != 1
    ):
        raise ValueError("ANCHOR_CONFIG_SCHEMA_MISMATCH")
    weight = config["anchor_weight"]
    if (
        type(weight) not in (int, float)
        or not math.isfinite(weight)
        or weight not in (0, 1)
        or config["arm"] != f"native_replay_anchor{int(weight)}"
    ):
        raise ValueError("ANCHOR_FIXED_MATCHED_PAIR_REQUIRED")
    if (
        config["anchors"] != ANCHOR_PROTOCOL
        or config["gates"] != SOURCE_GATES
        or config["target_confirmation"] != CONFIRMATION
    ):
        raise ValueError("ANCHOR_FROZEN_PROTOCOL_CHANGED")
    if set(config["targets"]) != {"qwen4", "mistral7"}:
        raise ValueError("ANCHOR_BOTH_TARGETS_REQUIRED")
    for spec in (config["source_initial"], *config["targets"].values()):
        if (
            set(spec) != {"path", "files_sha256"}
            or not Path(spec["path"]).is_absolute()
            or set(spec["files_sha256"]) != {"config.json", "model.safetensors"}
        ):
            raise ValueError("ANCHOR_PINNED_CHECKPOINT_REQUIRED")
    source = config["source_config"]
    if source != {
        "path": "configs/sequence_native_replay.json",
        "sha256": "d99d2a9f941fdc67b63f6309d531721fafda16aa361825c5f8f76684fb6afbbc",
    }:
        raise ValueError("ANCHOR_ORIGINAL_SOURCE_CONFIG_REQUIRED")
    dependencies = config["frozen_dependencies_sha256"]
    if not isinstance(dependencies, dict) or len(dependencies) != 32:
        raise ValueError("ANCHOR_FROZEN_DEPENDENCY_CLOSURE_REQUIRED")
    for name, expected in dependencies.items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("ANCHOR_INVALID_DEPENDENCY_PIN")
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"ANCHOR_FROZEN_DEPENDENCY_CHANGED: {name}")
    if file_hash(ROOT / CONFIRMATION["path"]) != CONFIRMATION["sha256"]:
        raise ValueError("ANCHOR_CONFIRMATION_FILE_CHANGED")


def read_source_config(config: dict) -> dict:
    path = ROOT / config["source_config"]["path"]
    if file_hash(path) != config["source_config"]["sha256"]:
        raise ValueError("ANCHOR_SOURCE_CONFIG_CHANGED")
    source = json.loads(path.read_text())
    original.validate_config(source)
    if (
        source["method"] != "native"
        or source["training"]["replay_weight"] != 0.25
        or source["seed"] != 30301
    ):
        raise ValueError("ANCHOR_NATIVE_REPLAY_303_REQUIRED")
    return source


def fingerprint(config: dict) -> dict:
    names = set(config["frozen_dependencies_sha256"]) | {
        "anchoring.py",
        "train_anchored.py",
    }
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def prepare(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
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
            raise ValueError("ANCHOR_OUTPUT_CONTAINS_RESEARCH_OR_INVALID_BOOTSTRAP")
        if "attempts" in entries and (
            entries["attempts"].is_symlink() or not entries["attempts"].is_dir()
        ):
            raise ValueError("ANCHOR_INVALID_ATTEMPT_ARCHIVE")
        task = json.loads(entries["task.json"].read_text())
        execution = json.loads(entries["execution.json"].read_text())
        if (
            json.loads(entries["config.json"].read_text()) != config
            or task.get("config") != config
        ):
            raise ValueError("ANCHOR_BOOTSTRAP_CONFIG_MISMATCH")
        if (
            task.get("id") != output.name
            or execution.get("task_id") != task["id"]
            or execution.get("status") != "running"
            or execution.get("task") != task
        ):
            raise ValueError("ANCHOR_BOOTSTRAP_TASK_MISMATCH")
    else:
        write_json(output / "config.json", config)
    for spec in (config["source_initial"], *config["targets"].values()):
        verify_checkpoint_files(spec)
    source = read_source_config(config)
    prepare_data(source, output)
    anchor_rows = anchor_schedule(
        config["anchors"]["train_tasks"],
        source["training"]["steps"],
        config["anchors"]["seed"],
    )
    write_json(output / "anchor_schedule.json", anchor_rows)
    write_json(
        output / "preregistration.json",
        {
            "config_sha256": digest(config),
            "source_sha256": fingerprint(config),
            "anchor_schedule_sha256": digest(anchor_rows),
            "source_updates_each_arm": 288,
            "target_confirmation": config["target_confirmation"],
            "selection": "Fixed step96 for all3stages. Acquisition/retention never changes the288-update training budget.",
            "claim_boundary": config["claim_boundary"],
        },
    )
    original.emit(
        output,
        "anchored_prepared",
        config_sha256=digest(config),
        anchor_weight=config["anchor_weight"],
    )


def verify_prepared(config: dict, output: Path) -> None:
    receipt = json.loads((output / "preregistration.json").read_text())
    if receipt["config_sha256"] != digest(config) or receipt[
        "source_sha256"
    ] != fingerprint(config):
        raise ValueError("ANCHOR_PREREGISTERED_CODE_OR_CONFIG_CHANGED")
    if (
        digest(json.loads((output / "anchor_schedule.json").read_text()))
        != receipt["anchor_schedule_sha256"]
    ):
        raise ValueError("ANCHOR_PREREGISTERED_SCHEDULE_CHANGED")
    for spec in (config["source_initial"], *config["targets"].values()):
        verify_checkpoint_files(spec)


def load_native(spec: dict, device: torch.device) -> PortalModel:
    verify_checkpoint_files(spec)
    return PortalModel.from_pretrained(
        spec["path"], local_files_only=True, device=str(device), dtype=torch.float32
    ).requires_grad_(False)


def checkpoint_spec(directory: Path) -> dict:
    path = directory / "checkpoint"
    return {
        "path": str(path),
        "files_sha256": {
            name: file_hash(path / name)
            for name in ("config.json", "model.safetensors")
        },
    }


def vector_snapshot(learner: SequenceLearner) -> dict:
    return {
        "source_alignment": tensor_hash(learner.portal.alignment.state_dict()),
        "published_vectors": tensor_hash(
            {"vectors": learner.portal.task_latents[: len(PUBLISHED_TASKS)]}
        ),
        "inactive_vectors": {
            task: tensor_hash({"vector": vector})
            for task, vector in learner.vectors.items()
            if not vector.requires_grad
        },
        "inactive_optimizer": learner.inactive_optimizer_hashes(),
    }


def train(config: dict, output: Path) -> None:
    if (output / "stages").exists():
        raise ValueError("ANCHOR_TRAINING_ALREADY_EXISTS")
    source = read_source_config(config)
    original.require_gpu(output, source["seed"])
    base = load_base(source["models"]["qwen8"]["base"])
    portal = load_native(config["source_initial"], base.device)
    devices = [base.device.index or 0] if base.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        targets = {
            name: load_native(spec, base.device)
            for name, spec in config["targets"].items()
        }
    plan = parameter_plan(portal)
    recipe = original.learner_recipe(source, plan)
    learner = AnchoredSequenceLearner(
        base,
        portal,
        SEQUENCE_TASKS,
        0,
        recipe,
        targets=targets,
        protocol=config["anchors"],
        anchor_weight=config["anchor_weight"],
    )
    anchors = learner.anchors
    anchor_rows = json.loads((output / "anchor_schedule.json").read_text())
    base_before = tensor_hash(frozen_base_tensors(base.model))
    write_json(
        output / "initial_state.json",
        {
            "pid": os.getpid(),
            "checkpoint": config["source_initial"],
            "state_sha256": tensor_hash(portal.state_dict()),
            "optimizer": learner.optimizer_receipt(),
            "parameter_plan": plan,
            "base_sha256": base_before,
            "frozen_anchors": anchors.verify_frozen(),
        },
    )
    reference_files = {
        task: anchors.save_reference(task, output / "anchors")
        for task in PUBLISHED_TASKS
    }
    write_json(output / "anchor_references.json", reference_files)
    count = [0]

    def record_base_forward(*_args) -> None:
        count[0] += 1

    handle = base.model.register_forward_pre_hook(record_base_forward)
    steps, schedule, stages = [], [], {}
    try:
        for stage, task in enumerate(SEQUENCE_TASKS):
            if stage:
                learner.advance_stage()
            names = tuple(f"{name}_train" for name in SEQUENCE_TASKS[: stage + 1])
            data = read_data(output, names)
            directory = output / "stages" / original.BOUNDARIES[stage + 1]
            directory.mkdir(parents=True)
            before = vector_snapshot(learner)
            reference_before = anchors.verify_frozen()
            stage_steps = []
            for current, reference, identity in original.stage_schedule(
                data, stage, source["seed"], recipe
            ):
                anchor_row = anchor_rows[stage * recipe["steps"] + identity["step"] - 1]
                if (anchor_row["stage"], anchor_row["step"]) != (
                    stage,
                    identity["step"],
                ):
                    raise ValueError("ANCHOR_CURRENT_AND_FUNCTION_SCHEDULE_MISMATCH")
                learner.select_anchor(anchor_row)
                forwards_before = count[0]
                native_before = anchors.forward_calls["student"]
                row = {
                    **identity,
                    **learner.step(current, reference, recipe["replay_weight"]),
                }
                row["actual_llm_forward_calls"] = count[0] - forwards_before
                row["actual_anchor_native_forward_calls"] = (
                    anchors.forward_calls["student"] - native_before
                )
                if row["actual_llm_forward_calls"] != 1 + len(
                    {item.task for item in reference}
                ):
                    raise ValueError("ANCHOR_UNEXPECTED_LLM_FORWARD_COUNT")
                if row["actual_anchor_native_forward_calls"] != (
                    3 if stage == 0 else 6
                ):
                    raise ValueError("ANCHOR_UNEXPECTED_NATIVE_FORWARD_COUNT")
                stage_steps.append(row)
                steps.append(row)
                schedule.append(identity)
                with (directory / "updates.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                if identity["step"] % 8 == 0:
                    original.emit(
                        output,
                        "anchored_update",
                        stage=stage,
                        step=identity["step"],
                        weight=config["anchor_weight"],
                        supervised_loss=row["anchor"]["supervised_current_loss"],
                        anchor_loss=row["anchor"]["loss"],
                    )
            if (
                vector_snapshot(learner) != before
                or anchors.verify_frozen() != reference_before
            ):
                raise ValueError("ANCHOR_FROZEN_STATE_CHANGED_DURING_STAGE")
            original.check_optimizer_steps(learner, recipe["steps"])
            validation = read_data(
                output,
                tuple(f"{name}_validation" for name in SEQUENCE_TASKS[: stage + 1]),
            )
            probes = [
                validation[f"{name}_validation"][index]
                for name in SEQUENCE_TASKS[: stage + 1]
                for index in (0, 1)
            ]
            saved = original.save_checkpoint(learner, directory, probes)
            geometry = anchors.diagnostics()
            write_json(directory / "geometry.json", geometry)
            stage_result = {
                "stage": stage,
                "task": task,
                "pid": os.getpid(),
                "checkpoint": checkpoint_spec(directory),
                "checkpoint_receipt": saved,
                "budget": original.aggregate_steps(stage_steps),
                "actual_llm_training_forward_calls": sum(
                    row["actual_llm_forward_calls"] for row in stage_steps
                ),
                "actual_anchor_native_forward_calls": sum(
                    row["actual_anchor_native_forward_calls"] for row in stage_steps
                ),
                "frozen_before": before,
                "frozen_after": vector_snapshot(learner),
                "reference_hashes_before": reference_before,
                "reference_hashes_after": anchors.verify_frozen(),
                "selection": "fixed_step96",
                "source_gate_used_for_training": False,
            }
            if stage < 2:
                anchors.capture_acquired(task, learner.vectors[task], stage)
                reference_files[task] = anchors.save_reference(task, output / "anchors")
                write_json(output / "anchor_references.json", reference_files)
            write_json(directory / "training_receipt.json", stage_result)
            stages[original.BOUNDARIES[stage + 1]] = stage_result
            original.emit(
                output,
                "anchored_stage_saved",
                stage=stage,
                checkpoint_sha256=saved["checkpoint_sha256"],
                optimizer_updates=len(stage_steps),
            )
        base_after = tensor_hash(frozen_base_tensors(base.model))
        if base_after != base_before:
            raise ValueError("ANCHOR_FROZEN_BASE_CHANGED")
        anchors.verify_frozen()
        budget = original.aggregate_steps(steps)
        if (
            budget["optimizer_updates"] != 288
            or budget["current_computed"]["examples"] != 1152
            or budget["reference_effective"]["examples"] != 192
        ):
            raise ValueError("ANCHOR_FIXED_SEQUENCE_BUDGET_MISMATCH")
        receipt = {
            "status": "completed",
            "pid": os.getpid(),
            "arm": config["arm"],
            "anchor_weight": config["anchor_weight"],
            "stages": stages,
            "budget": budget,
            "computed_schedule_sha256": digest(schedule),
            "anchor_schedule_sha256": digest(anchor_rows),
            "reference_files": reference_files,
            "base_sha256_before": base_before,
            "base_sha256_after": base_after,
            "actual_llm_training_forward_calls": sum(
                row["actual_llm_forward_calls"] for row in steps
            ),
            "actual_anchor_native_forward_calls": sum(
                row["actual_anchor_native_forward_calls"] for row in steps
            ),
            "native_generator_calls": anchors.forward_calls,
            "total_llm_forward_calls_in_training_process": count[0],
            "target_llm_forward_calls": 0,
            "target_abc_data_reads": 0,
            "fixed_budget_completed_before_any_source_gate": True,
        }
        write_json(output / "computed_schedule.json", schedule)
        write_json(output / "training_receipt.json", receipt)
    finally:
        handle.remove()
        learner.close()


def source_gates(timeline: dict, gates: dict) -> dict:
    if set(timeline) != set(original.BOUNDARIES):
        raise ValueError("ANCHOR_ALL_SOURCE_BOUNDARIES_REQUIRED")
    tasks = {}
    for index, task in enumerate(SEQUENCE_TASKS):
        own_boundary = original.BOUNDARIES[index + 1]
        train_accuracy = timeline[own_boundary]["train"]["metrics"][task]["accuracy"]
        accuracy = timeline[own_boundary]["validation"]["metrics"][task]["accuracy"]
        initial = timeline["initial"]["validation"]["metrics"][task]["accuracy"]
        later = {
            boundary: {
                "accuracy": timeline[boundary]["validation"]["metrics"][task][
                    "accuracy"
                ],
                "forgetting_from_acquisition": max(
                    0.0,
                    accuracy
                    - timeline[boundary]["validation"]["metrics"][task]["accuracy"],
                ),
            }
            for boundary in original.BOUNDARIES[index + 2 :]
        }
        observed = [
            train_accuracy,
            accuracy,
            initial,
            *(row["accuracy"] for row in later.values()),
        ]
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in observed
        ):
            raise ValueError("ANCHOR_INVALID_SOURCE_ACCURACY")
        checks = {
            "own_train": train_accuracy >= gates["own_train_accuracy_min"],
            "own_validation": accuracy >= gates["own_validation_accuracy_min"],
            "own_validation_gain": accuracy - initial
            >= gates["own_validation_gain_min"],
            "all_later_boundaries_retained": all(
                row["forgetting_from_acquisition"]
                <= gates["max_later_validation_forgetting"]
                for row in later.values()
            ),
        }
        tasks[task] = {
            "own_boundary": own_boundary,
            "own_train_accuracy": train_accuracy,
            "own_validation_accuracy": accuracy,
            "initial_validation_accuracy": initial,
            "validation_gain": accuracy - initial,
            "later": later,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(row["passed"] for row in tasks.values())
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "tasks": tasks,
        "thresholds": gates,
        "target_evaluation_eligible": passed,
        "training_budget_action": "all288updates_already_completed",
        "forgetting_reference": "own_acquisition_boundary_at_every_later_saved_boundary",
        "claim_boundary": "Original four-choice source accuracy. This does not establish free-generation competence or target transfer.",
    }


def verify_reference_files(training: dict) -> None:
    for task, reference in training["reference_files"].items():
        path = Path(reference["path"])
        if (
            file_hash(path) != reference["file_sha256"]
            or tensor_hash(load_file(path)) != reference["tensor_sha256"]
        ):
            raise ValueError(f"ANCHOR_SAVED_REFERENCE_CHANGED: {task}")


def evaluate_sources(config: dict, output: Path) -> None:
    source = read_source_config(config)
    original.require_gpu(output, source["seed"])
    training = json.loads((output / "training_receipt.json").read_text())
    if (
        training["status"] != "completed"
        or training["pid"] == os.getpid()
        or training["budget"]["optimizer_updates"] != 288
    ):
        raise ValueError("ANCHOR_EVALUATION_REQUIRES_COMPLETED_FRESH_PROCESS_TRAINING")
    verify_reference_files(training)
    for stage in training["stages"].values():
        verify_checkpoint_files(stage["checkpoint"])
    if (output / "result.json").exists() or (
        output / "source_evaluation.json"
    ).exists():
        raise ValueError("ANCHOR_SOURCE_EVALUATION_ALREADY_EXISTS")
    base = load_base(source["models"]["qwen8"]["base"])
    base_before = tensor_hash(frozen_base_tensors(base.model))
    if base_before != training["base_sha256_after"]:
        raise ValueError("ANCHOR_EVALUATION_BASE_CHANGED")
    data = read_data(
        output,
        tuple(
            f"{task}_{split}"
            for task in SEQUENCE_TASKS
            for split in ("train", "validation")
        ),
    )
    initial = json.loads((output / "initial_state.json").read_text())
    timeline = {}
    for index, boundary in enumerate(original.BOUNDARIES):
        spec = (
            config["source_initial"]
            if index == 0
            else training["stages"][boundary]["checkpoint"]
        )
        portal = load_native(spec, base.device)
        tasks = SEQUENCE_TASKS if index == 0 else SEQUENCE_TASKS[:index]
        if index == 0:
            if tensor_hash(portal.state_dict()) != initial["state_sha256"]:
                raise ValueError("ANCHOR_INITIAL_STATE_RELOAD_CHANGED")
            reload_receipt = {
                "passed": True,
                "fresh_process": True,
                "trainer_pid": training["pid"],
                "evaluator_pid": os.getpid(),
                "state_exact": True,
                "optimizer_updates": 0,
            }
        else:
            directory = output / "stages" / boundary
            learner = SequenceLearner(
                base,
                portal,
                SEQUENCE_TASKS,
                index - 1,
                original.learner_recipe(source, initial["parameter_plan"]),
            )
            try:
                learner.load_optimizer(directory / "optimizer.pt")
                expected = training["stages"][boundary]["checkpoint_receipt"]
                if (
                    tensor_hash(portal.state_dict()) != expected["checkpoint_sha256"]
                    or original.frozen_snapshot(learner) != expected["frozen"]
                ):
                    raise ValueError(
                        "ANCHOR_SAVED_STATE_OR_FROZEN_STATE_RELOAD_CHANGED"
                    )
                reload_receipt = original.reload_probes(
                    learner, output, directory, tasks
                )
                reload_receipt["optimizer_updates"] = 0
            finally:
                learner.close()
            portal.requires_grad_(False)
        measured = {
            split: evaluate(
                base,
                original.evaluation_rows(data, split, tasks),
                source["training"]["max_prompt"],
                source["evaluation"]["batch_size"],
                portal,
            )
            for split in ("train", "validation")
        }
        measured["reload"] = reload_receipt
        measured["checkpoint"] = spec
        timeline[boundary] = measured
        original.emit(
            output,
            "anchored_source_evaluated",
            boundary=boundary,
            validation={
                task: values["accuracy"]
                for task, values in measured["validation"]["metrics"].items()
            },
        )
    verify_reference_files(training)
    base_after = tensor_hash(frozen_base_tensors(base.model))
    if base_after != base_before:
        raise ValueError("ANCHOR_EVALUATION_MUTATED_BASE")
    for spec in (config["source_initial"], *config["targets"].values()):
        verify_checkpoint_files(spec)
    gates = source_gates(timeline, config["gates"])
    write_json(
        output / "source_evaluation.json",
        {
            "pid": os.getpid(),
            "training_pid": training["pid"],
            "timeline": timeline,
            "gates": gates,
        },
    )
    write_json(
        output / "result.json",
        {
            "status": "completed",
            "kind": config["kind"],
            "arm": config["arm"],
            "anchor_weight": config["anchor_weight"],
            "mechanically_qualified": True,
            "source_gates": gates,
            "target_evaluation_eligible": gates["passed"],
            "source_evaluation_pid": os.getpid(),
            "training_pid": training["pid"],
            "separate_process_reload": True,
            "budget": training["budget"],
            "actual_llm_training_forward_calls": training[
                "actual_llm_training_forward_calls"
            ],
            "actual_anchor_native_forward_calls": training[
                "actual_anchor_native_forward_calls"
            ],
            "computed_schedule_sha256": training["computed_schedule_sha256"],
            "anchor_schedule_sha256": training["anchor_schedule_sha256"],
            "config_sha256": digest(config),
            "source_sha256": fingerprint(config),
            "target_confirmation": config["target_confirmation"],
            "target_llm_forward_calls": 0,
            "target_abc_data_reads": 0,
            "evaluation_optimizer_updates": 0,
            "parent_matched_pair_review_required": True,
            "claim_boundary": config["claim_boundary"],
        },
    )


def child_command(output: Path, phase: str) -> list[str]:
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(ROOT / "train_anchored.py"),
        "--config",
        str(output / "config.json"),
        "--output-dir",
        str(output),
        "--phase",
        phase,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed matched native sequence learning with generated-adapter function anchors."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("run", "prepare", "train", "evaluate"), default="run"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    validate_config(config)
    prepared = False
    try:
        if args.phase in ("run", "prepare"):
            prepare(config, output)
        else:
            verify_prepared(config, output)
        prepared = True
        if args.phase == "run":
            for phase in ("train", "evaluate"):
                subprocess.run(child_command(output, phase), check=True)
            verify_prepared(config, output)
        elif args.phase == "train":
            train(config, output)
        elif args.phase == "evaluate":
            evaluate_sources(config, output)
    except Exception as error:
        if prepared:
            write_json(
                output / "failure.json",
                {
                    "phase": args.phase,
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            original.emit(output, "anchored_failed", phase=args.phase, error=str(error))
        raise


if __name__ == "__main__":
    main()
