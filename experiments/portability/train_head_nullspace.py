import argparse
import copy
import gc
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from portallib import PortalModel
from portallib.evaluation import PortalInjector
from safetensors.torch import save_file

import follow_through as follow
from calibrate_target import assert_fp32, emit, file_pin, verify_file
from data import SEQUENCE_TASKS, digest, write_json
from head_nullspace import (
    CONTRACT,
    HeadSpacePortal,
    conservation,
    feature_space,
    restore_adapter,
    save_adapter,
    trainable_plan,
)
from learner import frozen_base_tensors, load_base, supervised_loss, tensor_hash
from preserve_tasks import OLD_TASKS, MatrixReferences


def validate(config):
    follow.validate(config)
    protocol = config["protocol"]
    if protocol["mode"] != "examples" or protocol["head_constraint"] != CONTRACT:
        raise ValueError("HEAD_NULLSPACE_PROTOCOL_CHANGED")
    if [arm["name"] for arm in protocol["arms"]] != [
        "head_only_control",
        "head_nullspace",
    ]:
        raise ValueError("HEAD_NULLSPACE_MATCHED_ARMS_REQUIRED")
    arms = [
        {key: value for key, value in arm.items() if key != "name"}
        for arm in protocol["arms"]
    ]
    if arms[0] != arms[1] or any(
        arm["routing"] != "task_latents"
        or arm["train"] != "heads"
        or not arm["train_latents"]
        or arm["source"] != "source_initial"
        or arm["initialization"] != "released"
        for arm in arms
    ):
        raise ValueError("HEAD_NULLSPACE_ARM_TRAINABILITY_CHANGED")
    for name, pin in config["code_dependencies"].items():
        verify_file(Path(__file__).parent / name, pin)
    for name, pin in config["sdk_files"].items():
        verify_file(Path(inspect.getfile(PortalModel)).parent / name, pin)


def frozen_signature(adapter):
    return digest(
        {
            "frozen_parameters": follow.frozen_adapter(adapter),
            "basis": tensor_hash({"basis": adapter.head_basis}),
        }
    )


def fit_arm(base, adapter, arm, config, rows, schedule, output, reference):
    protocol = config["protocol"]
    named, groups, vectors = trainable_plan(adapter, arm)
    initial_frozen = frozen_signature(adapter)
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
    )
    index = {row.id: row for values in rows.values() for row in values}
    train_probe = [row for values in rows.values() for row in values]
    validation = follow.validation_rows(config)
    initial_validation = follow.measure(base, adapter, validation, arm, config)
    checkpoints, trace, ema, active = {}, [], {}, set()

    def checkpoint(step):
        follow.sync_vectors(adapter, vectors)
        assert_fp32(
            {
                name: value
                for name, value in adapter.state_dict().items()
                if name != "head_basis"
            }
        )
        if frozen_signature(adapter) != initial_frozen:
            raise ValueError("HEAD_NULLSPACE_FROZEN_PARAMETERS_OR_BASIS_CHANGED")
        audit = conservation(adapter, reference)
        location = output / "checkpoints" / f"step-{step:04d}"
        saved = save_adapter(adapter, location)
        optimizer_path = location.parent / f"optimizer-{step:04d}.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        train_result = follow.measure(base, adapter, train_probe, arm, config)
        val_result = follow.measure(base, adapter, validation, arm, config)
        record = {
            "step": step,
            "saved": saved,
            "optimizer": file_pin(optimizer_path),
            "train": train_result,
            "validation": val_result,
            "qualification": follow.qualification(
                train_result, val_result, initial_validation, protocol
            ),
            "conservation": audit,
            "frozen_signature": initial_frozen,
        }
        checkpoints[str(step)] = record
        write_json(output / f"checkpoint-{step:04d}.json", record)
        emit(
            output,
            "head_acquisition_checkpoint",
            step=step,
            qualification=record["qualification"],
            conservation=audit,
        )
        return audit["passed"]

    if not checkpoint(0):
        raise ValueError("HEAD_NULLSPACE_INITIAL_SOURCE_FACTORS_DIFFER")
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
                        if not math.isfinite(value):
                            raise FloatingPointError(
                                f"HEAD_NULLSPACE_NONFINITE_LOSS: {task}"
                            )
                        ema[task] = 0.9 * ema.get(task, value) + 0.1 * value
                        (
                            loss.loss / max(ema[task], 1e-6) / len(SEQUENCE_TASKS)
                        ).backward()
                    losses[task] = {
                        "nll": value,
                        "tokens": loss.supervised_tokens,
                        "ema": ema[task],
                    }
                gradients = follow.gradient_diagnostics(named)
                active.update(
                    name for name, value in gradients.items() if value["norm"] > 0
                )
                if not any(value["norm"] > 0 for value in gradients.values()):
                    status = "failed_zero_gradient"
                    emit(output, status, step=entry["step"])
                    break
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
                norm = math.sqrt(
                    sum(value["norm"] ** 2 for value in gradients.values())
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("HEAD_NULLSPACE_NONFINITE_GLOBAL_GRADIENT")
                coefficient = min(1.0, protocol["gradient_clip"] / (norm + 1e-6))
                for parameter in named.values():
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient)
                optimizer.step()
                updates += 1
                for name, parameter in named.items():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"HEAD_NULLSPACE_NONFINITE_UPDATED_PARAMETER: {name}"
                        )
            except FloatingPointError as exc:
                status = "failed_nonfinite_optimization"
                emit(output, status, step=entry["step"], error=str(exc))
                break
            if any(
                parameter.requires_grad or parameter.grad is not None
                for parameter in base.model.parameters()
            ):
                raise ValueError("HEAD_NULLSPACE_BASE_RECEIVED_GRADIENT")
            record = {
                **entry,
                "losses": losses,
                "gradients": gradients,
                "gradient_norm": norm,
            }
            trace.append(record)
            emit(output, "head_training_update", **record)
            if entry["step"] in protocol["checkpoints"]:
                conserved = checkpoint(entry["step"])
                if arm["name"] == "head_nullspace" and not conserved:
                    status = "failed_conservation"
                    emit(output, status, step=entry["step"])
                    break
    if frozen_signature(adapter) != initial_frozen:
        raise ValueError("HEAD_NULLSPACE_FROZEN_PARAMETERS_CHANGED_AT_END")
    reference.verify()
    result = {
        "status": status,
        "optimizer_updates": updates,
        "finite_completed_updates": len(trace),
        "example_exposures_per_task": exposures,
        "trainable_parameters": sum(value.numel() for value in named.values()),
        "resident_parameters": sum(value.numel() for value in adapter.parameters()),
        "additional_training_vector_elements": sum(
            value.numel() for value in vectors.parameters()
        ),
        "basis_bytes": adapter.head_basis.numel() * adapter.head_basis.element_size(),
        "active_gradient_families": sorted(active),
        "frozen_signature": initial_frozen,
        "checkpoints": checkpoints,
        "trace": trace,
        "schedule_sha256": digest(schedule),
    }
    write_json(output / "training.json", result)
    return result


def control_qualified(training, config):
    last = str(config["protocol"]["checkpoints"][-1])
    return (
        training["status"] == "completed_budget"
        and last in training["checkpoints"]
        and training["checkpoints"][last]["qualification"]["all_tasks_qualified"]
    )


def evaluate_saved(config, output, training_pid):
    if training_pid == os.getpid():
        raise ValueError("HEAD_NULLSPACE_NEW_EVALUATION_PID_REQUIRED")
    verify_file(
        output / "training_receipt.json",
        json.loads((output / "training_receipt.pin.json").read_text()),
    )
    receipt = json.loads((output / "training_receipt.json").read_text())
    if (
        receipt["pid"] != training_pid
        or receipt["config_sha256"] != digest(config)
        or follow.verify_inputs(config) != receipt["input_manifest"]
    ):
        raise ValueError("HEAD_NULLSPACE_EVALUATION_RECEIPT_MISMATCH")
    follow.setup_runtime(config)
    base = load_base(config["base"], device=config["runtime"]["device"])
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("HEAD_NULLSPACE_RELOADED_BASE_CHANGED")
    source = follow.load_source(config, "source_initial").to(base.device)
    reference = MatrixReferences(source, config["expected_hooks"])
    if reference.sha256 != receipt["reference_tensor_sha256"]:
        raise ValueError("HEAD_NULLSPACE_RELOADED_REFERENCE_CHANGED")
    retained, panels = follow.retention_rows(config), follow.final_panels(config)
    raw = {
        name: follow.measure(base, None, rows, None, config)
        for name, rows in {**panels, "released_tasks": retained}.items()
    }
    results = {}
    for arm in config["protocol"]["arms"]:
        training = receipt["arms"][arm["name"]]
        if training["status"].startswith("skipped_"):
            results[arm["name"]] = training
            continue
        steps = sorted(training["checkpoints"], key=int)
        selected = list(dict.fromkeys((steps[0], steps[-1])))
        measured = {}
        for step in selected:
            checkpoint = training["checkpoints"][step]
            adapter = restore_adapter(checkpoint["saved"], base.device)
            if (
                follow.measure(
                    base, adapter, follow.validation_rows(config), arm, config
                )
                != checkpoint["validation"]
            ):
                raise ValueError(
                    f"HEAD_NULLSPACE_RELOAD_PREDICTIONS_DIFFER: {arm['name']}/{step}"
                )
            audit = conservation(adapter, reference)
            if audit != checkpoint["conservation"]:
                raise ValueError("HEAD_NULLSPACE_RELOAD_CONSERVATION_DIFFERS")
            metrics = {
                "released_tasks": follow.measure(base, adapter, retained, arm, config),
                "conservation": audit,
            }
            if step == steps[-1]:
                metrics.update(
                    {
                        name: follow.measure(base, adapter, rows, arm, config)
                        for name, rows in panels.items()
                    }
                )
            if (
                tensor_hash(adapter.state_dict())
                != checkpoint["saved"]["tensor_sha256"]
            ):
                raise ValueError("HEAD_NULLSPACE_EVALUATION_MUTATED_ADAPTER")
            measured[step] = metrics
            emit(
                output,
                "head_final_checkpoint_evaluated",
                arm=arm["name"],
                step=int(step),
            )
            del adapter
        before, after = (
            measured[step]["released_tasks"] for step in (steps[0], steps[-1])
        )
        initial_choices = {
            row["id"]: row["prediction"] for row in before["predictions"]
        }
        agreement = all(
            initial_choices[row["id"]] == row["prediction"]
            for row in after["predictions"]
        )
        results[arm["name"]] = {
            "status": training["status"],
            "checkpoints": measured,
            "retention_steps": {"before": int(steps[0]), "after": int(steps[-1])},
            "retention": follow.retention_comparison(
                before, after, raw["released_tasks"]
            ),
            "old_choice_predictions_all_preserved": agreement,
            "final_acquisition_qualified": control_qualified(training, config),
            "reload_predictions_exact": True,
        }
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("HEAD_NULLSPACE_EVALUATION_BASE_CHANGED")
    reference.verify()
    protected = results["head_nullspace"]
    last = str(config["protocol"]["checkpoints"][-1])
    joint = (
        protected["status"] == "completed_budget"
        and results["head_only_control"]["final_acquisition_qualified"]
        and protected["final_acquisition_qualified"]
        and protected["checkpoints"][last]["conservation"]["passed"]
        and protected["old_choice_predictions_all_preserved"]
        and all(
            protected["retention"][task]["initial_ability_qualified"]
            for task in OLD_TASKS
        )
    )
    result = {
        "status": "completed",
        "training_pid": training_pid,
        "evaluation_pid": os.getpid(),
        "new_pid_reload": True,
        "config_sha256": digest(config),
        "raw": raw,
        "arms": results,
        "joint_acquisition_conservation_retention_qualified": joint,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output / "result.json", result)
    return result


def run(config, output, prepare_only=False):
    validate(config)
    manifest, rows, schedule = follow.prepare(config, output)
    follow.setup_runtime(config)
    source = follow.load_source(config, "source_initial")
    if not prepare_only:
        source = source.to(config["runtime"]["device"])
    spaces, proof = feature_space(source)
    write_json(output / "feature_space.json", proof)
    save_file(spaces, output / "feature_space.safetensors")
    emit(output, "head_feature_space_qualified", proof=proof)
    if prepare_only:
        return {"status": "prepared_not_trained", "proof": proof}
    base = load_base(config["base"], device=config["runtime"]["device"])
    if (
        source.config.base_model_name_or_path != base.model_id
        or source.config.base_model_revision != base.revision
    ):
        raise ValueError("HEAD_NULLSPACE_ADAPTER_BASE_IDENTITY_MISMATCH")
    follow.hook_receipt(base, source.config, config["expected_hooks"])
    base_sha = tensor_hash(frozen_base_tensors(base.model))
    reference = MatrixReferences(source, config["expected_hooks"])
    receipt = {
        "pid": os.getpid(),
        "config_sha256": digest(config),
        "input_manifest": manifest,
        "base_sha256": base_sha,
        "reference_tensor_sha256": reference.sha256,
        "feature_space": {
            "file": file_pin(output / "feature_space.safetensors"),
            "proof": proof,
        },
        "entrypoint": file_pin(Path(__file__)),
        "code_dependencies": config["code_dependencies"],
        "arms": {},
    }
    for arm in config["protocol"]["arms"]:
        if arm["name"] == "head_nullspace" and not control_qualified(
            receipt["arms"]["head_only_control"], config
        ):
            receipt["arms"][arm["name"]] = {
                "status": "skipped_control_acquisition_unqualified",
                "optimizer_updates": 0,
                "decision": "The frozen-alignment architecture has not qualified from examples. Do not interpret this as failure of nullspace preservation; return to architecture or acquisition design in a separately sealed experiment.",
            }
            emit(output, "head_constraint_prerequisite_failed", arm=arm["name"])
            continue
        destination = output / arm["name"]
        destination.mkdir()
        basis = spaces[
            "null_basis" if arm["name"] == "head_nullspace" else "full_basis"
        ].to(base.device)
        adapter = HeadSpacePortal(copy.deepcopy(source), basis)
        receipt["arms"][arm["name"]] = fit_arm(
            base, adapter, arm, config, rows, schedule, destination, reference
        )
        if tensor_hash(frozen_base_tensors(base.model)) != base_sha:
            raise ValueError("HEAD_NULLSPACE_TRAINING_BASE_CHANGED")
        write_json(output / "partial_training_receipt.json", receipt)
        del adapter
        gc.collect()
    write_json(output / "training_receipt.json", receipt)
    write_json(
        output / "training_receipt.pin.json", file_pin(output / "training_receipt.json")
    )
    del base, source, reference
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
            evaluate_saved(config, args.output_dir.resolve(), args.reload_evaluate)
        else:
            run(config, args.output_dir.resolve(), args.prepare_only)
    except Exception as exc:
        if args.output_dir.is_dir():
            emit(
                args.output_dir,
                "head_nullspace_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
