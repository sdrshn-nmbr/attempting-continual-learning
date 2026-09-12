import argparse
import gc
import inspect
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from calibrate_target import (
    ZeroLora,
    assert_fp32,
    emit,
    file_pin,
    fresh_alignment,
    hook_receipt,
    input_path,
    load_native_bundle,
    make_target,
    restore_adapter,
    save_adapter,
    verify_bundle,
    verify_file,
)
from data import SEQUENCE_TASKS, Example, digest, write_json
from learner import (
    evaluate,
    frozen_base_tensors,
    load_base,
    supervised_loss,
    tensor_hash,
)
from portallib import PortalModel
from portallib.evaluation import PortalInjector

TASKS = SEQUENCE_TASKS
MODES = {"examples", "calibration", "retention"}
ROOT = Path(__file__).resolve().parent


def validate(config):
    protocol = config["protocol"]
    if config["kind"] != "portal_follow_through" or protocol["mode"] not in MODES:
        raise ValueError("FOLLOW_THROUGH_UNKNOWN_EXPERIMENT")
    if digest(protocol) != config["protocol_sha256"]:
        raise ValueError("FOLLOW_THROUGH_PROTOCOL_HASH_MISMATCH")
    if protocol["scoring"] != "character_normalized_four_choice_gold_answer_nll":
        raise ValueError("FOLLOW_THROUGH_SCORING_CHANGED")
    if protocol["mode"] == "examples" and "source_constructed" in config["inputs"]:
        raise ValueError("EXAMPLES_ONLY_MUST_NOT_ACCESS_SUCCESSFUL_WEIGHTS")
    if protocol["mode"] in {"calibration", "retention"} and not config["inputs"].get(
        "source_constructed", {}
    ).get("tensor_sha256"):
        raise ValueError("CONSTRUCTED_GENERATOR_TENSOR_PIN_REQUIRED")
    checkpoints = protocol["checkpoints"]
    if (
        not checkpoints
        or checkpoints[0] != 0
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError("FOLLOW_THROUGH_CHECKPOINT_ORDER")
    if protocol["mode"] != "retention" and (
        len(checkpoints) < 3 or checkpoints[-1] <= protocol["gradient_gate_step"]
    ):
        raise ValueError("FOLLOW_THROUGH_INSUFFICIENT_ACQUISITION_BUDGET")
    arms = protocol["arms"]
    if len({arm["name"] for arm in arms}) != len(arms):
        raise ValueError("FOLLOW_THROUGH_DUPLICATE_ARM")
    for arm in arms:
        if arm["routing"] not in {"fixed_rte", "task_latents", "persistent"}:
            raise ValueError("FOLLOW_THROUGH_UNKNOWN_ROUTING")
        if arm["train"] not in {"none", "alignment", "heads", "full", "lora"}:
            raise ValueError("FOLLOW_THROUGH_UNKNOWN_PARAMETER_SCOPE")
        if arm["train_latents"] and (
            protocol["mode"] != "examples" or arm["routing"] != "task_latents"
        ):
            raise ValueError("FOLLOW_THROUGH_INVALID_LATENT_TRAINING")
        if protocol["mode"] == "calibration" and arm["train"] not in {
            "alignment",
            "lora",
        }:
            raise ValueError("TRANSFER_CORE_AND_LATENTS_MUST_BE_FROZEN")
    if config["runtime"]["dtype"] != "float32" or config["runtime"]["autocast"]:
        raise ValueError("FOLLOW_THROUGH_FP32_REQUIRED")


def verify_inputs(config):
    return {
        name: verify_bundle(spec)
        if "files" in spec
        else verify_file(input_path(spec["path"]), spec)
        for name, spec in config["inputs"].items()
    }


def load_source(config, name):
    spec = config["inputs"][name]
    source = load_native_bundle(spec)
    if tensor_hash(source.state_dict()) != spec["tensor_sha256"]:
        raise ValueError(f"FOLLOW_THROUGH_WRONG_SOURCE_TENSORS: {name}")
    return source


def read_json_input(config, name):
    return json.loads(input_path(config["inputs"][name]["path"]).read_text())


def training_rows(config):
    fixture = read_json_input(config, "fixture")
    rows = {}
    for task in TASKS:
        index = {
            r["id"]: Example.from_dict(r) for r in fixture["splits"][f"{task}_train"]
        }
        ids = config["protocol"]["training_ids"][task]
        if len(ids) != len(set(ids)) or not set(ids) <= index.keys():
            raise ValueError("FOLLOW_THROUGH_TRAIN_ROWS_MUST_BE_DISTINCT_TRAIN_IDS")
        rows[task] = [index[key] for key in ids]
    counts = {len(items) for items in rows.values()}
    batch = config["protocol"]["batch_per_task"]
    if len(counts) != 1 or min(counts) < batch or min(counts) % batch:
        raise ValueError("FOLLOW_THROUGH_UNBALANCED_TRAINING_ROWS")
    return rows


def schedule_for(config, rows):
    protocol = config["protocol"]
    batch = protocol["batch_per_task"]
    per_epoch = len(rows[TASKS[0]]) // batch
    result = []
    for step in range(protocol["checkpoints"][-1]):
        epoch, offset = divmod(step, per_epoch)
        selected = {}
        for task in TASKS:
            ids = [row.id for row in rows[task]]
            random.Random(
                int(digest([protocol["data_seed"], epoch, task]), 16)
            ).shuffle(ids)
            selected[task] = ids[offset * batch : (offset + 1) * batch]
        result.append({"step": step + 1, "epoch": epoch + 1, "ids": selected})
    return result


def validation_rows(config):
    fixture = read_json_input(config, "fixture")
    return [
        Example.from_dict(row)
        for task in TASKS
        for row in fixture["splits"][f"{task}_validation"]
    ]


def retention_rows(config):
    saved = read_json_input(config, "retention_probes")
    if digest(saved["indices"]) != config["protocol"]["retention_indices_sha256"]:
        raise ValueError("FOLLOW_THROUGH_RELEASED_PROBES_CHANGED")
    rows = []
    for original, identity in zip(saved["validation"], saved["indices"], strict=True):
        if original["task"] != identity["task"]:
            raise ValueError("FOLLOW_THROUGH_RETENTION_TASK_MISMATCH")
        key = identity["prompt_sha256"]
        rows.append(
            Example(
                f"released-{key}",
                original["task"],
                original["prompt"],
                tuple(original["choices"]),
                original["gold_idx"],
                key,
            )
        )
    return rows


def final_panels(config):
    fixture = read_json_input(config, "fixture")
    fresh = read_json_input(config, "length_holdout")
    panels = {
        "old_validation_test": [
            Example.from_dict(r)
            for task in TASKS
            for split in ("validation", "test")
            for r in fixture["splits"][f"{task}_{split}"]
        ],
        "old_exhaustive_triples": [
            Example.from_dict(r) for r in read_json_input(config, "old_holdout")["rows"]
        ],
        "sealed_length4": [Example.from_dict(r) for r in fresh["rows"]],
    }
    training_groups = {
        r["group"] for task in TASKS for r in fixture["splits"][f"{task}_train"]
    }
    previous_groups = set(training_groups)
    for name, rows in panels.items():
        groups = {r.group for r in rows}
        if (
            not rows
            or groups & previous_groups
            or len({r.id for r in rows}) != len(rows)
        ):
            raise ValueError(f"FOLLOW_THROUGH_EVALUATION_INPUT_OVERLAP: {name}")
        previous_groups.update(groups)
        for row in rows:
            inputs = list(map(int, row.group.split()))
            gold = " " + " ".join(
                str(fixture["provenance"]["rules"][row.task][v]) for v in inputs
            )
            if row.choices[row.gold_idx] != gold or len(row.choices) != 4:
                raise ValueError(f"FOLLOW_THROUGH_WRONG_ORACLE: {row.id}")
            if name == "sealed_length4" and len(inputs) != 4:
                raise ValueError("FOLLOW_THROUGH_LENGTH_HOLDOUT_NOT_FRESH")
    return panels


def setup_runtime(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(config["protocol"]["model_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_adapter(config, arm):
    source = load_source(config, arm["source"])
    if config["protocol"]["mode"] == "calibration":
        target = load_source(config, "target_portal")
        source = make_target(source, target, fresh_alignment(target))
    if arm["initialization"] == "zero":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["protocol"]["model_seed"])
            source = PortalModel(source.config, source.task_latents)
    elif arm["initialization"] != "released":
        raise ValueError("FOLLOW_THROUGH_UNKNOWN_INITIALIZATION")
    if arm["train"] == "lora":
        lora = ZeroLora(source.config)
        if (
            arm["initialization"] == "released"
            and config["protocol"]["mode"] == "examples"
        ):
            with torch.no_grad():
                generated = source.generate("rte")
                for target in source.config.projection_targets:
                    a, b = generated[target.key]
                    prefix = f"l{target.layer_index}_{target.module_name}"
                    lora.factors[f"{prefix}_a"].copy_(a)
                    lora.factors[f"{prefix}_b"].copy_(b)
        return lora
    return source.requires_grad_(False)


def factors(adapter, task, arm, vectors=None):
    if not isinstance(adapter, PortalModel):
        return adapter(TASKS[0])
    route = "rte" if task in TASKS and arm["routing"] == "fixed_rte" else task
    z = (
        vectors[route]
        if vectors is not None and route in vectors
        else adapter.task_latents[adapter.config.tasks.index(route)]
    )
    return adapter(z)


def measure(base, adapter, rows, arm, config):
    if adapter is None:
        return evaluate(
            base,
            rows,
            config["protocol"]["max_prompt"],
            config["protocol"]["eval_batch"],
        )
    results = {"metrics": {}, "predictions": []}
    with torch.inference_mode(), PortalInjector(base.model, adapter.config) as injector:
        for task in sorted({row.task for row in rows}):
            selected = [row for row in rows if row.task == task]
            with injector.activate(factors(adapter, task, arm)):
                result = evaluate(
                    base,
                    selected,
                    config["protocol"]["max_prompt"],
                    config["protocol"]["eval_batch"],
                )
            results["metrics"].update(result["metrics"])
            results["predictions"].extend(result["predictions"])
    return results


def parameter_groups(adapter, arm):
    adapter.requires_grad_(False)
    vectors = torch.nn.ParameterDict()
    if arm["train"] == "lora":
        adapter.requires_grad_(True)
    else:
        for name, value in adapter.named_parameters():
            train = arm["train"] in {"alignment", "heads", "full"} and name.startswith(
                "alignment."
            )
            if arm["train"] == "heads":
                train |= name.startswith(("core.A.", "core.B."))
                if arm["routing"] == "fixed_rte":
                    train &= not name.startswith("alignment.layer_embeddings.")
            if arm["train"] == "full":
                train |= name.startswith("core.")
            value.requires_grad_(train)
        if arm["train_latents"]:
            vectors.update(
                {
                    task: torch.nn.Parameter(
                        adapter.task_latents[adapter.config.tasks.index(task)]
                        .detach()
                        .clone()
                    )
                    for task in TASKS
                }
            )
    named = {name: p for name, p in adapter.named_parameters() if p.requires_grad}
    groups = [{"params": list(named.values()), "lr": arm["learning_rate"]}]
    if vectors:
        groups.append(
            {"params": list(vectors.parameters()), "lr": arm["latent_learning_rate"]}
        )
        named.update({f"latent.{task}": p for task, p in vectors.items()})
    return named, groups, vectors


def sync_vectors(adapter, vectors):
    with torch.no_grad():
        for task, vector in vectors.items():
            adapter.task_latents[adapter.config.tasks.index(task)].copy_(vector)


def frozen_adapter(adapter):
    frozen = {
        name: p
        for name, p in adapter.named_parameters()
        if not p.requires_grad and name != "task_latents"
    }
    if isinstance(adapter, PortalModel):
        indices = [
            i for i, task in enumerate(adapter.config.tasks) if task not in TASKS
        ]
        frozen["published_task_latents"] = adapter.task_latents[indices]
    return tensor_hash(frozen)


def parameter_family(name):
    if name.startswith("alignment."):
        return ".".join(name.split(".")[:2])
    if name.startswith("core."):
        return (
            "core_heads" if name.startswith(("core.A.", "core.B.")) else "core_hidden"
        )
    if name.startswith("factors."):
        return "lora"
    return "latents"


def gradient_diagnostics(named):
    groups = defaultdict(
        lambda: {
            "squared_norm": 0.0,
            "parameters": 0,
            "nonzero_parameters": 0,
            "missing": [],
        }
    )
    for name, p in named.items():
        group = groups[parameter_family(name)]
        group["parameters"] += p.numel()
        if p.grad is None:
            group["missing"].append(name)
            continue
        if not torch.isfinite(p.grad).all():
            raise FloatingPointError(f"FOLLOW_THROUGH_NONFINITE_GRADIENT: {name}")
        squared = float(p.grad.double().square().sum())
        group["squared_norm"] += squared
        group["nonzero_parameters"] += int(squared > 0)
    for group in groups.values():
        group["norm"] = math.sqrt(group.pop("squared_norm"))
    return dict(groups)


def scale_diagnostics(adapter, arm):
    records = {}
    with torch.no_grad():
        for task in TASKS:
            rows = []
            for key, (a, b) in factors(adapter, task, arm).items():
                a64, b64 = a.double(), b.double()
                square = float(((b64.T @ b64) * (a64 @ a64.T).T).sum())
                rows.append(
                    {
                        "layer": key[0],
                        "module": key[1],
                        "a_rms": float(a64.square().mean().sqrt()),
                        "b_rms": float(b64.square().mean().sqrt()),
                        "delta_frobenius": adapter.config.scaling
                        * math.sqrt(max(0.0, square)),
                    }
                )
            records[task] = rows
    return records


def accuracy(result):
    return sum(row["correct"] for row in result["predictions"]) / len(
        result["predictions"]
    )


def qualification(train_result, val_result, initial_val, protocol):
    train = {t: train_result["metrics"][t]["accuracy"] for t in TASKS}
    validation = {t: val_result["metrics"][t]["accuracy"] for t in TASKS}
    learned = {
        t: train[t] >= protocol["acquisition_train_floor"]
        and validation[t] >= protocol["acquisition_validation_floor"]
        for t in TASKS
    }
    return {
        "train": train,
        "validation": validation,
        "gain_from_start": accuracy(val_result) - accuracy(initial_val),
        "qualified_tasks": learned,
        "all_tasks_qualified": all(learned.values()),
        "thresholds_are_absolute_and_feasible": True,
    }


def fit_arm(base, adapter, arm, config, rows, schedule, output):
    protocol = config["protocol"]
    named, groups, vectors = parameter_groups(adapter, arm)
    assert_fp32(adapter.state_dict())
    assert_fp32(named)
    initial_frozen = frozen_adapter(adapter)
    latents_before = (
        tensor_hash({"all": adapter.task_latents})
        if isinstance(adapter, PortalModel)
        else None
    )
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
    )
    index = {row.id: row for task_rows in rows.values() for row in task_rows}
    train_probe = [r for task in TASKS for r in rows[task]]
    validation = validation_rows(config)
    checkpoints, traces, ema, active_families = {}, [], {}, set()
    initial_validation = measure(base, adapter, validation, arm, config)

    def checkpoint(step):
        sync_vectors(adapter, vectors)
        assert_fp32(adapter.state_dict())
        if frozen_adapter(adapter) != initial_frozen:
            raise ValueError("FOLLOW_THROUGH_FROZEN_ADAPTER_STATE_CHANGED")
        if (
            isinstance(adapter, PortalModel)
            and not arm["train_latents"]
            and tensor_hash({"all": adapter.task_latents}) != latents_before
        ):
            raise ValueError("FOLLOW_THROUGH_FROZEN_LATENTS_CHANGED")
        location = output / "checkpoints" / f"step-{step:04d}"
        saved = save_adapter(adapter, location)
        torch.save(optimizer.state_dict(), location.parent / f"optimizer-{step:04d}.pt")
        train_result = measure(base, adapter, train_probe, arm, config)
        val_result = measure(base, adapter, validation, arm, config)
        record = {
            "step": step,
            "saved": saved,
            "optimizer": file_pin(location.parent / f"optimizer-{step:04d}.pt"),
            "train": train_result,
            "validation": val_result,
            "scale": scale_diagnostics(adapter, arm),
            "qualification": qualification(
                train_result, val_result, initial_validation, protocol
            ),
            "frozen_adapter_sha256": initial_frozen,
        }
        write_json(output / f"checkpoint-{step:04d}.json", record)
        checkpoints[str(step)] = record
        emit(
            output,
            "acquisition_checkpoint",
            step=step,
            qualification=record["qualification"],
        )

    checkpoint(0)
    status = "completed_budget"
    updates = 0
    exposures = dict.fromkeys(TASKS, 0)
    with PortalInjector(base.model, adapter.config) as injector:
        for entry in schedule:
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            try:
                for task in TASKS:
                    batch = [index[key] for key in entry["ids"][task]]
                    exposures[task] += len(batch)
                    with (
                        torch.autocast(device_type=base.device.type, enabled=False),
                        injector.activate(factors(adapter, task, arm, vectors)),
                    ):
                        loss = supervised_loss(base, batch, protocol["max_prompt"])
                        value = float(loss.loss.detach())
                        ema[task] = 0.9 * ema.get(task, value) + 0.1 * value
                        (loss.loss / max(ema[task], 1e-6) / len(TASKS)).backward()
                    losses[task] = {
                        "nll": value,
                        "tokens": loss.supervised_tokens,
                        "ema": ema[task],
                    }
                gradients = gradient_diagnostics(named)
                active_families.update(k for k, v in gradients.items() if v["norm"] > 0)
                if (
                    entry["step"] >= protocol["gradient_gate_step"]
                    and set(gradients) - active_families
                ):
                    status = "failed_gradient_eligibility"
                    emit(
                        output,
                        status,
                        step=entry["step"],
                        inactive=sorted(set(gradients) - active_families),
                        gradients=gradients,
                    )
                    break
                if not any(v["norm"] > 0 for v in gradients.values()):
                    status = "failed_zero_gradient"
                    emit(output, status, step=entry["step"], gradients=gradients)
                    break
                norm = math.sqrt(
                    sum(group["norm"] ** 2 for group in gradients.values())
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("FOLLOW_THROUGH_NONFINITE_GLOBAL_GRADIENT")
                coefficient = min(1.0, protocol["gradient_clip"] / (norm + 1e-6))
                for parameter in named.values():
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient)
                optimizer.step()
                updates += 1
                for name, parameter in named.items():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"FOLLOW_THROUGH_NONFINITE_UPDATED_PARAMETER: {name}"
                        )
            except FloatingPointError as exc:
                status = "failed_nonfinite_optimization"
                emit(output, status, step=entry["step"], error=str(exc))
                break
            for name, p in base.model.named_parameters():
                if p.grad is not None or p.requires_grad:
                    raise ValueError(f"FOLLOW_THROUGH_BASE_RECEIVED_GRADIENT: {name}")
            record = {
                **entry,
                "losses": losses,
                "gradients": gradients,
                "gradient_norm": float(norm),
            }
            traces.append(record)
            emit(output, "training_update", **record)
            if entry["step"] in protocol["checkpoints"]:
                checkpoint(entry["step"])
    if frozen_adapter(adapter) != initial_frozen:
        raise ValueError("FOLLOW_THROUGH_FROZEN_ADAPTER_CHANGED_AT_END")
    families = defaultdict(int)
    for name, p in named.items():
        families[parameter_family(name)] += p.numel()
    result = {
        "status": status,
        "optimizer_updates": updates,
        "finite_completed_updates": len(traces),
        "example_exposures_per_task": exposures,
        "trainable_parameters": sum(p.numel() for p in named.values()),
        "trainable_groups": dict(families),
        "resident_parameters": sum(p.numel() for p in adapter.parameters()),
        "active_gradient_families": sorted(active_families),
        "frozen_adapter_sha256": initial_frozen,
        "checkpoints": checkpoints,
        "trace": traces,
        "schedule_sha256": digest(schedule),
    }
    write_json(output / "training.json", result)
    return result


def retention_comparison(before, after, raw):
    return {
        task: {
            "before": old["accuracy"],
            "after": after["metrics"][task]["accuracy"],
            "change": after["metrics"][task]["accuracy"] - old["accuracy"],
            "raw": raw["metrics"][task]["accuracy"],
            "initial_ability_qualified": old["accuracy"] >= 0.6,
            "initial_adapter_gain_qualified": old["accuracy"]
            - raw["metrics"][task]["accuracy"]
            >= 0.05,
            "examples": old["examples"],
        }
        for task, old in before["metrics"].items()
    }


def insertion_comparison(results, raw):
    untouched = results["untouched"]["checkpoints"]["0"]["released_tasks"]
    constructed = results["constructed"]["checkpoints"]["0"]["released_tasks"]
    return retention_comparison(untouched, constructed, raw["released_tasks"])


def evaluate_saved(config, output, training_pid):
    if training_pid == os.getpid():
        raise ValueError("FOLLOW_THROUGH_NEW_EVALUATION_PID_REQUIRED")
    receipt = json.loads((output / "training_receipt.json").read_text())
    verify_file(
        output / "training_receipt.json",
        json.loads((output / "training_receipt.pin.json").read_text()),
    )
    if (
        receipt["pid"] != training_pid
        or receipt["config_sha256"] != digest(config)
        or verify_inputs(config) != receipt["input_manifest"]
    ):
        raise ValueError("FOLLOW_THROUGH_EVALUATION_RECEIPT_MISMATCH")
    setup_runtime(config)
    base = load_base(config["base"], device=config["runtime"]["device"])
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("FOLLOW_THROUGH_RELOADED_BASE_CHANGED")
    panels, retained = final_panels(config), retention_rows(config)
    raw = {
        name: measure(base, None, rows, None, config)
        for name, rows in {**panels, "released_tasks": retained}.items()
    }
    results = {}
    for arm in config["protocol"]["arms"]:
        training = receipt["arms"][arm["name"]]
        arm_results = {}
        steps = sorted(training["checkpoints"], key=int)
        for step in steps:
            checkpoint = training["checkpoints"][step]
            adapter = restore_adapter(checkpoint["saved"], config["runtime"]["device"])
            if (
                measure(base, adapter, validation_rows(config), arm, config)
                != checkpoint["validation"]
            ):
                raise ValueError(
                    f"FOLLOW_THROUGH_RELOAD_PREDICTIONS_DIFFER: {arm['name']}/{step}"
                )
            measured = {
                name: measure(base, adapter, rows, arm, config)
                for name, rows in panels.items()
            }
            if step in (steps[0], steps[-1]):
                measured["released_tasks"] = measure(
                    base, adapter, retained, arm, config
                )
            if (
                tensor_hash(adapter.state_dict())
                != checkpoint["saved"]["tensor_sha256"]
            ):
                raise ValueError("FOLLOW_THROUGH_EVALUATION_MUTATED_ADAPTER")
            arm_results[step] = measured
            emit(output, "heldout_checkpoint_evaluated", arm=arm["name"], step=step)
            del adapter
        before = arm_results[steps[0]]["released_tasks"]
        after = arm_results[steps[-1]]["released_tasks"]
        results[arm["name"]] = {
            "status": training["status"],
            "checkpoints": arm_results,
            "reload_predictions_exact": True,
        }
        if config["protocol"]["mode"] != "retention":
            results[arm["name"]]["retention_steps"] = {
                "before": int(steps[0]),
                "after": int(steps[-1]),
            }
            results[arm["name"]]["retention"] = retention_comparison(
                before, after, raw["released_tasks"]
            )
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("FOLLOW_THROUGH_EVALUATION_BASE_CHANGED")
    result = {
        "status": "completed",
        "training_pid": training_pid,
        "evaluation_pid": os.getpid(),
        "new_pid_reload": True,
        "config_sha256": digest(config),
        "raw": raw,
        "arms": results,
        "claim_boundary": config["claim_boundary"],
    }
    if config["protocol"]["mode"] == "retention":
        result["insertion_comparison"] = insertion_comparison(results, raw)
    write_json(output / "result.json", result)
    return result


def prepare(config, output):
    validate(config)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {
        "config.json",
        "packages.txt",
        "execution.json",
        "task.json",
        "run.log",
        "attempts",
    }
    if any(p.is_symlink() or p.name not in allowed for p in output.iterdir()):
        raise FileExistsError(f"FOLLOW_THROUGH_OUTPUT_ALREADY_USED: {output}")
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("FOLLOW_THROUGH_DISPATCH_CONFIG_MISMATCH")
    if (
        input_path(config["inputs"]["base"]["path"]).resolve()
        != Path(config["base"]["local_path"]).resolve()
    ):
        raise ValueError("FOLLOW_THROUGH_BASE_PATH_NOT_PINNED")
    manifest = verify_inputs(config)
    rows = training_rows(config)
    schedule = schedule_for(config, rows)
    write_json(config_path, config)
    write_json(output / "input_manifest.json", manifest)
    write_json(
        output / "training_rows.json",
        {t: [asdict(r) for r in items] for t, items in rows.items()},
    )
    write_json(output / "schedule.json", schedule)
    emit(output, "all_inputs_verified_before_model_load", config_sha256=digest(config))
    return manifest, rows, schedule


def run(config, output, prepare_only=False):
    manifest, rows, schedule = prepare(config, output)
    if prepare_only:
        return {"status": "prepared_not_trained", "inputs": manifest}
    setup_runtime(config)
    base = load_base(config["base"], device=config["runtime"]["device"])
    base.model.eval()
    base_sha = tensor_hash(frozen_base_tensors(base.model))
    receipt = {
        "pid": os.getpid(),
        "config_sha256": digest(config),
        "input_manifest": manifest,
        "base_sha256": base_sha,
        "arms": {},
        "implementation": {
            str(Path(p).resolve()): file_pin(Path(p))
            for p in (
                __file__,
                inspect.getfile(PortalModel),
                inspect.getfile(PortalInjector),
            )
        },
    }
    for arm in config["protocol"]["arms"]:
        arm_output = output / arm["name"]
        arm_output.mkdir()
        adapter = build_adapter(config, arm).to(config["runtime"]["device"])
        if (
            adapter.config.base_model_name_or_path != base.model_id
            or adapter.config.base_model_revision != base.revision
        ):
            raise ValueError("FOLLOW_THROUGH_ADAPTER_BASE_IDENTITY_MISMATCH")
        hooks = hook_receipt(base, adapter.config, config["expected_hooks"])
        write_json(arm_output / "hooks.json", hooks)
        if arm["train"] == "none":
            saved = save_adapter(adapter, arm_output / "checkpoints/step-0000")
            training = {
                "status": "evaluation_only",
                "optimizer_updates": 0,
                "checkpoints": {
                    "0": {
                        "saved": saved,
                        "validation": measure(
                            base, adapter, validation_rows(config), arm, config
                        ),
                    }
                },
            }
        else:
            training = fit_arm(base, adapter, arm, config, rows, schedule, arm_output)
        receipt["arms"][arm["name"]] = training
        if tensor_hash(frozen_base_tensors(base.model)) != base_sha:
            raise ValueError("FOLLOW_THROUGH_TRAINING_BASE_CHANGED")
        write_json(output / "partial_training_receipt.json", receipt)
        del adapter
        gc.collect()
    write_json(output / "training_receipt.json", receipt)
    write_json(
        output / "training_receipt.pin.json", file_pin(output / "training_receipt.json")
    )
    del base
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
                "follow_through_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
