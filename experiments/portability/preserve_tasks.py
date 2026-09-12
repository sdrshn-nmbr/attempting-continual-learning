import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from portallib.evaluation import PortalInjector
from safetensors.torch import save_file

import follow_through as follow
from calibrate_target import assert_fp32, emit, file_pin, save_adapter, verify_file
from data import SEQUENCE_TASKS, digest, write_json
from learner import frozen_base_tensors, load_base, supervised_loss, tensor_hash
from repair_alignment import product_loss, product_squared_norm

OLD_TASKS = ("boolq", "commonsense_qa", "hellaswag", "winogrande")
PRESERVATION = {
    "method": "native_adapter_retention_regularization",
    "tasks": list(OLD_TASKS),
    "tasks_per_update": 4,
    "matrix_target": "effective_scaled_B_at_A",
    "normalization": "mean_over_tasks_and_projections_of_squared_error_over_reference_squared_norm_plus_epsilon",
    "epsilon": 1e-12,
    "gram_dtype": "float64",
    "reference": "source_initial_original_frozen_task_vectors",
    "reference_capture": "once_before_updates_in_the_current_arithmetic_environment",
    "weights": [0.0, 1.0],
    "behavioral_retention_max_drop_per_task": 0.0625,
    "behavioral_metrics_never_select_training_checkpoints": True,
    "old_behavioral_examples_for_training": 0,
}


def validate(config):
    follow.validate(config)
    protocol = config["protocol"]
    if protocol.get("preservation") != PRESERVATION or protocol["mode"] != "examples":
        raise ValueError("PRESERVATION_SEALED_OBJECTIVE_CHANGED")
    arms = protocol["arms"]
    if [arm["name"] for arm in arms] != ["control", "preserved"]:
        raise ValueError("PRESERVATION_PAIRED_ARMS_REQUIRED")
    if [arm["preservation_weight"] for arm in arms] != [0.0, 1.0]:
        raise ValueError("PRESERVATION_SEALED_WEIGHTS_CHANGED")
    control, protected = [
        {
            key: value
            for key, value in arm.items()
            if key not in {"name", "preservation_weight"}
        }
        for arm in arms
    ]
    if control != protected or (
        control["source"] != "source_initial"
        or control["train"] != "heads"
        or control["routing"] != "task_latents"
        or not control["train_latents"]
        or control["initialization"] != "released"
    ):
        raise ValueError("PRESERVATION_TRAINABILITY_OR_MATCHING_CHANGED")
    for name, pin in config["code_dependencies"].items():
        verify_file(Path(__file__).parent / name, pin)


class MatrixReferences:
    def __init__(self, source, expected_hooks):
        source.requires_grad_(False).eval()
        self.source_sha256 = tensor_hash(source.state_dict())
        self.scale = source.config.scaling
        self.factors = {}
        self.vectors = {}
        self.energies = {}
        with torch.no_grad():
            for task in OLD_TASKS:
                index = source.config.tasks.index(task)
                self.vectors[task] = source.task_latents[index].detach().clone()
                self.factors[task] = {
                    key: tuple(value.detach().clone() for value in pair)
                    for key, pair in source.generate(task).items()
                }
                if len(self.factors[task]) != expected_hooks:
                    raise ValueError(f"PRESERVATION_REFERENCE_HOOK_COUNT: {task}")
                energy = {
                    key: float(product_squared_norm(a, b) * self.scale**2)
                    for key, (a, b) in self.factors[task].items()
                }
                if any(
                    not math.isfinite(value) or value <= PRESERVATION["epsilon"]
                    for value in energy.values()
                ):
                    raise ValueError(
                        f"PRESERVATION_DEGENERATE_REFERENCE_MATRIX: {task}"
                    )
                self.energies[task] = energy
                if float(self.loss(source, task)) != 0:
                    raise ValueError(f"PRESERVATION_INITIAL_IDENTITY_FAILED: {task}")
        self.sha256 = tensor_hash(self.tensors())

    def tensors(self):
        values = {f"{task}.vector": vector for task, vector in self.vectors.items()}
        values.update(
            {
                f"{task}.{layer}.{module}.{factor}": value
                for task, projections in self.factors.items()
                for (layer, module), pair in projections.items()
                for factor, value in zip(("A", "B"), pair, strict=True)
            }
        )
        return values

    def verify(self):
        if any(
            value.requires_grad or value.grad is not None
            for value in self.tensors().values()
        ):
            raise ValueError("PRESERVATION_REFERENCE_RECEIVED_GRADIENT")
        if tensor_hash(self.tensors()) != self.sha256:
            raise ValueError("PRESERVATION_REFERENCE_TENSORS_CHANGED")

    def loss(self, model, task):
        if model.config.scaling != self.scale:
            raise ValueError("PRESERVATION_ADAPTER_SCALE_CHANGED")
        value = product_loss(
            model(self.vectors[task]),
            self.factors[task],
            PRESERVATION["epsilon"],
            self.scale,
        )
        if not torch.isfinite(value) or float(value.detach()) < -1e-10:
            raise FloatingPointError(
                f"PRESERVATION_NONFINITE_OR_NEGATIVE_PRODUCT_LOSS: {task}"
            )
        return value

    def diagnostics(self, model):
        with torch.no_grad():
            values = {task: float(self.loss(model, task)) for task in OLD_TASKS}
        return {
            "per_task_mean_normalized_squared_error": values,
            "mean": sum(values.values()) / len(values),
        }

    def save(self, output):
        self.verify()
        path = output / "preservation_reference.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in self.tensors().items()
            },
            path,
            metadata={
                "kind": PRESERVATION["method"],
                "source_tensor_sha256": self.source_sha256,
            },
        )
        result = {
            "source_tensor_sha256": self.source_sha256,
            "reference_tensor_sha256": self.sha256,
            "file": file_pin(path),
            "tasks": {
                task: {
                    "matrices": len(self.factors[task]),
                    "reference_energy_min": min(self.energies[task].values()),
                    "reference_energy_max": max(self.energies[task].values()),
                    "reference_factor_elements": sum(
                        v.numel() for pair in self.factors[task].values() for v in pair
                    ),
                    "vector_sha256": tensor_hash({"vector": self.vectors[task]}),
                }
                for task in OLD_TASKS
            },
            "behavioral_examples_used": 0,
        }
        write_json(output / "preservation_reference.json", result)
        return result


def apply_preservation(adapter, references, weight, counters):
    values = {}
    for task in OLD_TASKS:
        with torch.set_grad_enabled(weight != 0):
            counters["forward_calls"] += 1
            loss = references.loss(adapter, task)
            if weight:
                counters["backward_calls"] += 1
                (weight * loss / len(OLD_TASKS)).backward()
        values[task] = float(loss.detach())
    return {
        "weight": weight,
        "tasks": list(OLD_TASKS),
        "per_task_mean_normalized_squared_error": values,
        "unweighted_mean": sum(values.values()) / len(values),
        "weighted_loss": weight * sum(values.values()) / len(values),
        "backward_calls": len(values) if weight else 0,
    }


def fit_arm(base, adapter, arm, config, rows, schedule, output, references):
    protocol = config["protocol"]
    named, groups, vectors = follow.parameter_groups(adapter, arm)
    assert_fp32(adapter.state_dict())
    assert_fp32(named)
    references.verify()
    before_frozen = follow.frozen_adapter(adapter)
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
    )
    index = {row.id: row for values in rows.values() for row in values}
    train_probe = [r for task in SEQUENCE_TASKS for r in rows[task]]
    validation = follow.validation_rows(config)
    initial_validation = follow.measure(base, adapter, validation, arm, config)
    checkpoints, trace, ema, active = {}, [], {}, set()
    preservation_calls = {"forward_calls": 0, "backward_calls": 0}

    def checkpoint(step):
        follow.sync_vectors(adapter, vectors)
        assert_fp32(adapter.state_dict())
        references.verify()
        if follow.frozen_adapter(adapter) != before_frozen:
            raise ValueError("PRESERVATION_FROZEN_COMPONENT_CHANGED")
        location = output / "checkpoints" / f"step-{step:04d}"
        saved = save_adapter(adapter, location)
        optimizer_path = location.parent / f"optimizer-{step:04d}.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        trained = follow.measure(base, adapter, train_probe, arm, config)
        validated = follow.measure(base, adapter, validation, arm, config)
        record = {
            "step": step,
            "saved": saved,
            "optimizer": file_pin(optimizer_path),
            "train": trained,
            "validation": validated,
            "scale": follow.scale_diagnostics(adapter, arm),
            "qualification": follow.qualification(
                trained, validated, initial_validation, protocol
            ),
            "frozen_adapter_sha256": before_frozen,
            "preservation": references.diagnostics(adapter),
        }
        checkpoints[str(step)] = record
        write_json(output / f"checkpoint-{step:04d}.json", record)
        emit(
            output,
            "acquisition_checkpoint",
            step=step,
            qualification=record["qualification"],
            preservation=record["preservation"],
        )

    checkpoint(0)
    status, updates = "completed_budget", 0
    exposures = dict.fromkeys(SEQUENCE_TASKS, 0)
    with PortalInjector(base.model, adapter.config) as injector:
        for entry in schedule:
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            try:
                for task in SEQUENCE_TASKS:
                    batch = [index[key] for key in entry["ids"][task]]
                    exposures[task] += len(batch)
                    with (
                        torch.autocast(device_type=base.device.type, enabled=False),
                        injector.activate(follow.factors(adapter, task, arm, vectors)),
                    ):
                        loss = supervised_loss(base, batch, protocol["max_prompt"])
                        value = float(loss.loss.detach())
                        ema[task] = 0.9 * ema.get(task, value) + 0.1 * value
                        (
                            loss.loss / max(ema[task], 1e-6) / len(SEQUENCE_TASKS)
                        ).backward()
                    losses[task] = {
                        "nll": value,
                        "tokens": loss.supervised_tokens,
                        "ema": ema[task],
                    }
                example_gradients = follow.gradient_diagnostics(named)
                preservation = apply_preservation(
                    adapter, references, arm["preservation_weight"], preservation_calls
                )
                gradients = follow.gradient_diagnostics(named)
                active.update(
                    name for name, group in gradients.items() if group["norm"] > 0
                )
                if (
                    entry["step"] >= protocol["gradient_gate_step"]
                    and set(gradients) - active
                ):
                    status = "failed_gradient_eligibility"
                    emit(
                        output,
                        status,
                        step=entry["step"],
                        inactive=sorted(set(gradients) - active),
                    )
                    break
                if not any(group["norm"] > 0 for group in gradients.values()):
                    status = "failed_zero_gradient"
                    emit(output, status, step=entry["step"])
                    break
                norm = math.sqrt(
                    sum(group["norm"] ** 2 for group in gradients.values())
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("PRESERVATION_NONFINITE_GLOBAL_GRADIENT")
                coefficient = min(1.0, protocol["gradient_clip"] / (norm + 1e-6))
                for parameter in named.values():
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient)
                optimizer.step()
                updates += 1
                for name, parameter in named.items():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"PRESERVATION_NONFINITE_UPDATED_PARAMETER: {name}"
                        )
            except FloatingPointError as exc:
                status = "failed_nonfinite_optimization"
                emit(output, status, step=entry["step"], error=str(exc))
                break
            for name, parameter in base.model.named_parameters():
                if parameter.grad is not None or parameter.requires_grad:
                    raise ValueError(f"PRESERVATION_BASE_RECEIVED_GRADIENT: {name}")
            record = {
                **entry,
                "losses": losses,
                "gradients": gradients,
                "example_gradient_norm": math.sqrt(
                    sum(group["norm"] ** 2 for group in example_gradients.values())
                ),
                "gradient_norm": norm,
                "preservation": preservation,
            }
            trace.append(record)
            emit(output, "training_update", **record)
            if entry["step"] in protocol["checkpoints"]:
                checkpoint(entry["step"])
    references.verify()
    if follow.frozen_adapter(adapter) != before_frozen:
        raise ValueError("PRESERVATION_FROZEN_COMPONENT_CHANGED_AT_END")
    result = {
        "status": status,
        "optimizer_updates": updates,
        "finite_completed_updates": len(trace),
        "example_exposures_per_task": exposures,
        "trainable_parameters": sum(p.numel() for p in named.values()),
        "trainable_names": sorted(named),
        "resident_parameters": sum(p.numel() for p in adapter.parameters()),
        "frozen_adapter_sha256": before_frozen,
        "reference_tensor_sha256": references.sha256,
        "preservation_weight": arm["preservation_weight"],
        "preservation_training_forward_calls": preservation_calls["forward_calls"],
        "preservation_training_backward_calls": preservation_calls["backward_calls"],
        "preservation_training_matrix_comparisons": preservation_calls["forward_calls"]
        * config["expected_hooks"],
        "preservation_checkpoint_forward_calls": len(checkpoints) * len(OLD_TASKS),
        "old_behavioral_training_examples": 0,
        "checkpoints": checkpoints,
        "trace": trace,
        "schedule_sha256": digest(schedule),
    }
    write_json(output / "training.json", result)
    return result


def compare_final(config, output, result):
    training = json.loads((output / "training_receipt.json").read_text())
    verify_file(
        output / "preservation_reference.safetensors",
        training["matrix_reference"]["file"],
    )
    last = str(config["protocol"]["checkpoints"][-1])
    comparisons = {}
    for name in ("control", "preserved"):
        arm = result["arms"][name]
        checkpoint = training["arms"][name]["checkpoints"].get(last)
        observed_steps = sorted(training["arms"][name]["checkpoints"], key=int)
        if arm["retention_steps"] != {
            "before": int(observed_steps[0]),
            "after": int(observed_steps[-1]),
        }:
            raise ValueError("PRESERVATION_RETENTION_ENDPOINT_MISMATCH")
        retained = arm["retention"]
        comparisons[name] = {
            "completed_planned_budget": checkpoint is not None
            and training["arms"][name]["status"] == "completed_budget",
            "abc_acquisition_qualified": checkpoint["qualification"][
                "all_tasks_qualified"
            ]
            if checkpoint
            else None,
            "initial_ability_qualified": {
                task: retained[task]["initial_ability_qualified"] for task in OLD_TASKS
            },
            "retention_within_two_rows_per_task": all(
                retained[task]["change"]
                >= -PRESERVATION["behavioral_retention_max_drop_per_task"]
                for task in OLD_TASKS
            ),
            "final_mean_old_task_accuracy": sum(
                retained[task]["after"] for task in OLD_TASKS
            )
            / len(OLD_TASKS),
            "final_matrix_error": checkpoint["preservation"] if checkpoint else None,
        }
    comparisons["protected_minus_control_old_task_accuracy"] = (
        comparisons["preserved"]["final_mean_old_task_accuracy"]
        - comparisons["control"]["final_mean_old_task_accuracy"]
    )
    comparisons["joint_acquisition_and_retention_qualified"] = (
        all(
            comparisons[name]["completed_planned_budget"]
            and comparisons[name]["abc_acquisition_qualified"]
            for name in ("control", "preserved")
        )
        and all(comparisons["preserved"]["initial_ability_qualified"].values())
        and comparisons["preserved"]["retention_within_two_rows_per_task"]
    )
    result["native_adapter_retention_regularization"] = comparisons
    write_json(output / "result.json", result)
    return result


def run(config, output, prepare_only=False):
    validate(config)
    manifest, rows, schedule = follow.prepare(config, output)
    follow.setup_runtime(config)
    source = follow.load_source(config, "source_initial")
    if prepare_only:
        reference = MatrixReferences(source, config["expected_hooks"])
        saved = reference.save(output)
        emit(output, "matrix_preservation_preflight_passed", reference=saved)
        return {"status": "prepared_not_trained", "reference": saved}
    base = load_base(config["base"], device=config["runtime"]["device"])
    base.model.eval()
    reference = MatrixReferences(source.to(base.device), config["expected_hooks"])
    saved_reference = reference.save(output)
    del source
    base_sha = tensor_hash(frozen_base_tensors(base.model))
    receipt = {
        "pid": os.getpid(),
        "config_sha256": digest(config),
        "input_manifest": manifest,
        "base_sha256": base_sha,
        "matrix_reference": saved_reference,
        "arms": {},
        "entrypoint": file_pin(Path(__file__)),
        "code_dependencies": config["code_dependencies"],
    }
    for arm in config["protocol"]["arms"]:
        destination = output / arm["name"]
        destination.mkdir()
        adapter = follow.build_adapter(config, arm).to(base.device)
        if (
            adapter.config.base_model_name_or_path != base.model_id
            or adapter.config.base_model_revision != base.revision
        ):
            raise ValueError("PRESERVATION_ADAPTER_BASE_IDENTITY_MISMATCH")
        follow.hook_receipt(base, adapter.config, config["expected_hooks"])
        initial = reference.diagnostics(adapter)
        if any(
            value != 0
            for value in initial["per_task_mean_normalized_squared_error"].values()
        ):
            raise ValueError("PRESERVATION_ARM_NOT_INITIALIZED_FROM_REFERENCE")
        receipt["arms"][arm["name"]] = fit_arm(
            base, adapter, arm, config, rows, schedule, destination, reference
        )
        if tensor_hash(frozen_base_tensors(base.model)) != base_sha:
            raise ValueError("PRESERVATION_BASE_TENSORS_CHANGED")
        write_json(output / "partial_training_receipt.json", receipt)
        del adapter
        gc.collect()
    if (
        receipt["arms"]["control"]["checkpoints"]["0"]["saved"]["tensor_sha256"]
        != receipt["arms"]["preserved"]["checkpoints"]["0"]["saved"]["tensor_sha256"]
    ):
        raise ValueError("PRESERVATION_ARMS_STARTED_DIFFERENTLY")
    write_json(output / "training_receipt.json", receipt)
    write_json(
        output / "training_receipt.pin.json", file_pin(output / "training_receipt.json")
    )
    del base, reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run(
        [
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
        ],
        check=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    return json.loads((output / "result.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reload-evaluate", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    try:
        if args.reload_evaluate is not None:
            result = follow.evaluate_saved(
                config, args.output_dir.resolve(), args.reload_evaluate
            )
            compare_final(config, args.output_dir.resolve(), result)
        else:
            run(config, args.output_dir.resolve(), args.prepare_only)
    except Exception as exc:
        if args.output_dir.is_dir():
            emit(
                args.output_dir,
                "preservation_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
