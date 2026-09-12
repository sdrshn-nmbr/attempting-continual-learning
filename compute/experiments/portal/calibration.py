import json
import math
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
from portallib import PortalEvaluator
from safetensors.torch import load_file, save_file

from engine import Adaptation, Pilot, batch_schedule, fit_stage
from inventory import sha256_file, tensor_digest
from runtime import (
    atomic_json,
    config_digest,
    ensure_gpu,
    load_local_base,
    load_port,
    registry_from,
    select_port,
)
from tasks import make_calibration_tasks


@dataclass(frozen=True)
class CalibrationPlan:
    max_steps: int = 256
    batch_size: int = 4
    checkpoint_steps: tuple[int, ...] = (32, 128, 256)
    latent_learning_rates: tuple[float, ...] = (0.002, 0.02)
    lora_learning_rate: float = 0.0002
    train_examples: int = 128
    validation_examples: int = 128
    validation_seed: int = 2718
    prior_evaluation_examples: int = 96
    eval_batch_size: int = 8
    max_prompt_tokens: int = 128
    gradient_clip: float = 1.0
    gradient_checkpointing: bool = True
    min_train_accuracy: float = 0.9
    min_validation_accuracy: float = 0.8
    min_validation_accuracy_gain: float = 0.2
    min_validation_nll_drop: float = 0.1
    reload_atol: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "checkpoint_steps", tuple(self.checkpoint_steps))
        object.__setattr__(
            self, "latent_learning_rates", tuple(self.latent_learning_rates)
        )
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 256:
            raise ValueError("calibration_max_steps_must_be_between_1_and_256")
        if self.batch_size != 4:
            raise ValueError("calibration_batch_size_must_be_four")
        if (
            not self.checkpoint_steps
            or any(type(step) is not int or step < 1 for step in self.checkpoint_steps)
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
            or self.checkpoint_steps[-1] != self.max_steps
        ):
            raise ValueError("calibration_checkpoints_must_increase_to_max_steps")
        if (
            self.latent_learning_rates != (0.002, 0.02)
            or self.lora_learning_rate != 0.0002
        ):
            raise ValueError(
                "calibration_learning_rates_differ_from_preregistered_arms"
            )
        for count in (self.train_examples, self.validation_examples):
            if type(count) is not int or not 4 <= count <= 128 or count % 4:
                raise ValueError(
                    "calibration_split_sizes_must_be_multiples_of_four_up_to_128"
                )
        for count in (self.eval_batch_size, self.max_prompt_tokens):
            if type(count) is not int or count < 1:
                raise ValueError("calibration_inference_counts_must_be_positive")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("calibration_gradient_clip_must_be_positive_finite")
        minimums = (
            (self.min_train_accuracy, 0.9),
            (self.min_validation_accuracy, 0.8),
            (self.min_validation_accuracy_gain, 0.2),
            (self.min_validation_nll_drop, 0.1),
        )
        if any(
            not math.isfinite(value) or value < minimum for value, minimum in minimums
        ):
            raise ValueError("calibration_qualification_threshold_must_not_be_weakened")
        if not math.isfinite(self.reload_atol) or not 0 <= self.reload_atol <= 1e-6:
            raise ValueError("calibration_reload_tolerance_must_not_exceed_1e_6")

    def arms(self):
        return [
            {"name": f"latent_lr_{rate:g}", "kind": "latent", "learning_rate": rate}
            for rate in self.latent_learning_rates
        ] + [
            {
                "name": "native_lora",
                "kind": "native_lora",
                "learning_rate": self.lora_learning_rate,
            }
        ]

    def pilot(self, steps, learning_rate):
        return Pilot(
            steps_per_task=steps,
            batch_size=self.batch_size,
            train_examples=self.train_examples,
            validation_examples=self.validation_examples,
            eval_batch_size=self.eval_batch_size,
            max_prompt_tokens=self.max_prompt_tokens,
            checkpoint_every=self.max_steps,
            latent_learning_rate=learning_rate,
            lora_learning_rate=self.lora_learning_rate,
            gradient_clip=self.gradient_clip,
            gradient_checkpointing=self.gradient_checkpointing,
        )


def score_split(adaptation, base, data, split, plan, log):
    examples = data.rows(split)
    evaluator = PortalEvaluator(
        max_prompt=plan.max_prompt_tokens, batch_size=plan.eval_batch_size
    )
    was_training = base.model.training
    base.model.eval()
    predictions = []
    try:
        with torch.no_grad(), adaptation.activation() if adaptation else nullcontext():
            for index, example in enumerate(examples):
                log.check_stop()
                scores, nll_sum, gold_tokens = evaluator._score_rows(base, [example])
                choice_scores = scores[0]
                if gold_tokens < 1 or not all(
                    math.isfinite(value) for value in (*choice_scores, nll_sum)
                ):
                    raise RuntimeError(
                        f"calibration_invalid_scores: {split} row={index}"
                    )
                predicted = max(
                    range(len(choice_scores)), key=choice_scores.__getitem__
                )
                predictions.append(
                    {
                        "row_index": index,
                        "example_sha256": config_digest(example.to_dict()),
                        **example.to_dict(),
                        "predicted_index": predicted,
                        "predicted_answer": example.choices[predicted],
                        "correct": predicted == example.gold_idx,
                        "choice_logprob_per_character": choice_scores,
                        "gold_nll_sum": nll_sum,
                        "gold_tokens": gold_tokens,
                        "gold_nll_per_token": nll_sum / gold_tokens,
                    }
                )
    finally:
        base.model.train(was_training)
    tokens = sum(row["gold_tokens"] for row in predictions)
    return {
        "split": split,
        "examples": len(predictions),
        "accuracy": sum(row["correct"] for row in predictions) / len(predictions),
        "gold_nll": sum(row["gold_nll_sum"] for row in predictions) / tokens,
        "gold_tokens": tokens,
        "prediction_sha256": config_digest(predictions),
        "rows": predictions,
    }


def summary(measurement):
    return {key: value for key, value in measurement.items() if key != "rows"}


def read_bound(path, fingerprint, log):
    expected_hash = log.metrics.get("calibration_artifacts_sha256", {}).get(
        str(path.relative_to(log.output))
    )
    if expected_hash is None:
        return None
    if sha256_file(path) != expected_hash:
        raise ValueError(f"calibration_receipt_hash_mismatch: {path}")
    result = json.loads(path.read_text())
    if result["fingerprint"] != fingerprint:
        raise ValueError(f"calibration_artifact_identity_mismatch: {path}")
    for name, expected in result.get("files_sha256", {}).items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(
                f"calibration_artifact_hash_mismatch: {path.parent / name}"
            )
    return result


def save_bound(path, result, log):
    atomic_json(path, result)
    log.metrics.setdefault("calibration_artifacts_sha256", {})[
        str(path.relative_to(log.output))
    ] = sha256_file(path)


def frozen_digests(base, portal):
    return {
        "base": tensor_digest(base.model.state_dict()),
        "released_port": tensor_digest(portal.state_dict()),
    }


def anchors(base, portal, initial_latent, data, plan, log, fingerprint, seed):
    path = log.output / "anchors.json"
    cached = read_bound(path, fingerprint, log)
    if cached is not None:
        return cached
    result = {"fingerprint": fingerprint}
    result["no_adapter"] = {
        split: score_split(None, base, data, split, plan, log)
        for split in ("train", "validation")
    }
    with Adaptation(base, portal, "latent", initial_latent, seed) as adaptation:
        result["initial_latent"] = {
            split: score_split(adaptation, base, data, split, plan, log)
            for split in ("train", "validation")
        }
        adaptation.verify_frozen()
    save_bound(path, result, log)
    log.event("calibration_anchors_measured", fingerprint=fingerprint)
    return result


def verify_reload(expected, actual, tolerance):
    if len(expected["rows"]) != len(actual["rows"]):
        raise RuntimeError("calibration_reload_row_count_changed")
    max_error = 0.0
    for left, right in zip(expected["rows"], actual["rows"], strict=True):
        if (
            left["example_sha256"] != right["example_sha256"]
            or left["predicted_index"] != right["predicted_index"]
        ):
            raise RuntimeError("calibration_reload_predictions_changed")
        differences = [abs(left["gold_nll_per_token"] - right["gold_nll_per_token"])]
        differences.extend(
            abs(a - b)
            for a, b in zip(
                left["choice_logprob_per_character"],
                right["choice_logprob_per_character"],
                strict=True,
            )
        )
        max_error = max(max_error, *differences)
    if max_error > tolerance:
        raise RuntimeError(
            f"calibration_reload_scores_changed: max_error={max_error} tolerance={tolerance}"
        )
    return {
        "passed": True,
        "predictions_identical": True,
        "maximum_nll_or_choice_score_error": max_error,
        "atol": tolerance,
        "reloaded_prediction_sha256": actual["prediction_sha256"],
        "scope": "Saved trainable state loaded into a fresh adaptation on the same frozen source base and released port; no optimizer step during reload verification.",
    }


def fit_arm(base, portal, initial_latent, data, plan, log, arm, fingerprint, seed):
    milestones = {}
    for steps in plan.checkpoint_steps:
        log.check_stop()
        directory = log.output / arm["name"] / f"step-{steps}"
        result_path = directory / "result.json"
        cached = read_bound(result_path, fingerprint, log)
        if cached is not None:
            milestones[str(steps)] = cached
            continue
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "trainable.safetensors"
        with Adaptation(base, portal, arm["kind"], initial_latent, seed) as adaptation:
            initial_state = adaptation.state()
            training = fit_stage(
                adaptation,
                data.rows("train"),
                "acquisition_a",
                arm["name"],
                plan.pilot(steps, arm["learning_rate"]),
                log,
                seed + 101,
            )
            state = adaptation.state()
            state_sha = tensor_digest(state)
            change = math.sqrt(
                sum(
                    float((state[name] - initial_state[name]).double().square().sum())
                    for name in state
                )
            )
            if not math.isfinite(change) or change <= 0:
                raise RuntimeError(
                    f"calibration_trainable_state_unchanged_or_nonfinite: {arm['name']}"
                )
            temporary = state_path.with_suffix(".tmp")
            save_file(state, temporary)
            temporary.replace(state_path)
            measured = {
                split: score_split(adaptation, adaptation.base, data, split, plan, log)
                for split in ("train", "validation")
            }
            adaptation.verify_frozen()
        with Adaptation(base, portal, arm["kind"], initial_latent, seed) as reloaded:
            reloaded.restore(load_file(state_path))
            if tensor_digest(reloaded.state()) != state_sha:
                raise RuntimeError("calibration_reload_trainable_state_changed")
            validation = score_split(
                reloaded, reloaded.base, data, "validation", plan, log
            )
            reload_result = verify_reload(
                measured["validation"], validation, plan.reload_atol
            )
            reloaded.verify_frozen()
        latest_checkpoint = log.output / arm["name"] / "acquisition_a" / "checkpoint.pt"
        temporary = directory / "checkpoint.pt.tmp"
        shutil.copyfile(latest_checkpoint, temporary)
        temporary.replace(directory / "checkpoint.pt")
        for split, measurement in measured.items():
            atomic_json(directory / f"{split}.json", measurement)
        result = {
            "fingerprint": fingerprint,
            "arm": arm,
            "steps": steps,
            "train": summary(measured["train"]),
            "validation": summary(measured["validation"]),
            "training": training,
            "initial_state_sha256": tensor_digest(initial_state),
            "trainable_state_sha256": state_sha,
            "parameter_change_l2": change,
            "latent_change_l2": change if arm["kind"] == "latent" else None,
            "all_gradients_finite_and_nonzero": len(training["gradient_norms"]) == steps
            and all(
                math.isfinite(norm) and norm > 0 for norm in training["gradient_norms"]
            ),
            "reload": reload_result,
            "files_sha256": {
                name: sha256_file(directory / name)
                for name in (
                    "trainable.safetensors",
                    "checkpoint.pt",
                    "train.json",
                    "validation.json",
                )
            },
        }
        save_bound(result_path, result, log)
        milestones[str(steps)] = result
        log.metrics["measurements"][f"calibration/{arm['name']}/{steps}"] = result
        log.event(
            "calibration_checkpoint_measured",
            arm=arm["name"],
            steps=steps,
            train=result["train"],
            validation=result["validation"],
            reload=reload_result,
        )
    return {**arm, "milestones": milestones}


def qualify(result, references, plan):
    final = result["milestones"][str(plan.max_steps)]
    validation = final["validation"]
    initial_name = "initial_latent" if result["kind"] == "latent" else "no_adapter"
    initial = references[initial_name]["validation"]
    raw = references["no_adapter"]["validation"]
    checks = {
        "train_accuracy": final["train"]["accuracy"] >= plan.min_train_accuracy,
        "validation_accuracy": validation["accuracy"] >= plan.min_validation_accuracy,
        "validation_gain_vs_initial": validation["accuracy"] - initial["accuracy"]
        >= plan.min_validation_accuracy_gain,
        "validation_gain_vs_no_adapter": validation["accuracy"] - raw["accuracy"]
        >= plan.min_validation_accuracy_gain,
        "validation_nll_drop_vs_initial": initial["gold_nll"] - validation["gold_nll"]
        >= plan.min_validation_nll_drop,
        "validation_nll_drop_vs_no_adapter": raw["gold_nll"] - validation["gold_nll"]
        >= plan.min_validation_nll_drop,
        "finite_nonzero_gradients": final["all_gradients_finite_and_nonzero"],
        "nonzero_parameter_change": final["parameter_change_l2"] > 0,
        "reload_equivalent": final["reload"]["passed"],
        "full_fixed_budget": final["training"]["examples_seen"]
        == plan.max_steps * plan.batch_size
        and final["training"]["steps"] == plan.max_steps,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "qualification_step": plan.max_steps,
    }


def source_calibration(config, log):
    allowed = {
        "mode",
        "model_id",
        "revision",
        "model_path",
        "seed",
        "device",
        "registry_path",
        "artifact_cache_dir",
        "source_training_rows_sha256",
        "calibration",
    }
    if config.keys() - allowed or config["mode"] != "source_calibration":
        raise ValueError(
            "source_calibration_config_must_not_include_target_or_other_modes"
        )
    plan = CalibrationPlan(**config["calibration"])
    data = make_calibration_tasks(
        config["seed"],
        plan.validation_seed,
        plan.train_examples,
        plan.validation_examples,
        plan.prior_evaluation_examples,
    )
    if data.provenance["train_sha256"] != config["source_training_rows_sha256"]:
        raise ValueError("source_calibration_training_rows_changed")
    registry = registry_from(config)
    row = select_port(registry, config["model_id"], config["revision"])
    device = ensure_gpu(config, log)
    portal = load_port(row, config, log)
    base = load_local_base(
        row, config["model_path"], device, log, plan.gradient_checkpointing
    )
    initial_latent = portal.task_latents.detach().mean(dim=0).cpu()
    provenance = log.metrics["provenance"]
    contract = {
        "config_sha256": log.digest,
        "source_sha256": config_digest(provenance["implementation"]),
        "model_id": base.model_id,
        "revision": base.revision,
        "base_provenance_sha256": config_digest(provenance[base.model_id]),
        "released_port_provenance_sha256": config_digest(
            provenance[row["artifact"]["id"]]
        ),
        "frozen_state_sha256": frozen_digests(base, portal),
        "runtime": {
            name: log.metrics["runtime"][name]
            for name in (
                "torch",
                "hip",
                "transformers",
                "portallib",
                "peft",
                "safetensors",
            )
        },
        "initial_latent_sha256": tensor_digest({"latent": initial_latent}),
        "data": data.provenance,
        "plan": asdict(plan),
        "schedule_sha256": config_digest(
            batch_schedule(
                data.train, plan.max_steps, plan.batch_size, config["seed"] + 101
            )
        ),
        "scoring": "Pinned portallib 0.2.1 per-example choice scoring; character-normalized candidate logprobs and token-normalized gold NLL.",
    }
    contract = json.loads(json.dumps(contract))
    fingerprint = config_digest(contract)
    previous = log.metrics.get("calibration_contract")
    if previous is not None and previous != contract:
        raise ValueError("source_calibration_resume_provenance_changed")
    log.metrics["calibration_contract"] = contract
    log.metrics["scientific_contract"] = {
        "scope": "Source-only train/dev latent capacity qualification; frozen released core/alignment and frozen base.",
        "arms": plan.arms(),
        "selection": "All arms run the full fixed budget. Qualification uses the final checkpoint only; intermediate checkpoints are diagnostics.",
        "validation": f"Fresh validation seed {plan.validation_seed}; validation values never enter optimizer updates or the minibatch schedule.",
        "test_access": False,
        "target_model_loaded": False,
        "unmatched_budget": "Native LoRA has more trainable parameters than the latent; identical examples and steps do not imply equal capacity or FLOPs.",
    }
    atomic_json(log.output / "data.json", data.as_dict())
    log.event(
        "source_calibration_preregistered", fingerprint=fingerprint, contract=contract
    )
    references = anchors(
        base, portal, initial_latent, data, plan, log, fingerprint, config["seed"]
    )
    results = {}
    for arm in plan.arms():
        result = fit_arm(
            base,
            portal,
            initial_latent,
            data,
            plan,
            log,
            arm,
            fingerprint,
            config["seed"],
        )
        result["qualification"] = qualify(result, references, plan)
        results[arm["name"]] = result
        log.metrics["source_calibration"] = results
        log.event(
            "source_calibration_arm_complete",
            arm=arm["name"],
            qualification=result["qualification"],
        )
    if frozen_digests(base, portal) != contract["frozen_state_sha256"]:
        raise RuntimeError("source_calibration_frozen_weights_changed")
    control_passed = results["native_lora"]["qualification"]["passed"]
    latent_passed = [
        name
        for name, result in results.items()
        if result["kind"] == "latent" and result["qualification"]["passed"]
    ]
    gpu = base.device.type == "cuda"
    log.metrics["runtime"]["gpu_execution"] = gpu
    log.metrics["qualification"] = {
        "status": "source_fit_qualified_for_further_study"
        if gpu and control_passed and latent_passed
        else "source_fit_not_qualified",
        "native_lora_control_passed": control_passed,
        "latent_arms_passing_final_dev_criteria": latent_passed,
        "gpu_execution_observed": gpu,
        "full_frozen_weight_hashes_unchanged": True,
        "fingerprint": fingerprint,
        "automatic_transfer_expansion": False,
    }
    log.metrics["claim_scope"] = (
        "Preregistered source-only train/dev calibration. Passing is an engineering qualification, not a research conclusion about heldout generalization, continual retention, or cross-model transfer."
    )
    log.event("source_calibration_complete", qualification=log.metrics["qualification"])
