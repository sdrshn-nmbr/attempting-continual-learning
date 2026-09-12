import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from consolidation import observed_rows, training_plan
from environment import (
    ARMS,
    binary_entropy,
    collect_device,
    digest,
    evaluate_rows,
    make_rules,
    manifest,
    partition,
    prompt,
    stage_devices,
)
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn import functional
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

LOGGER = logging.getLogger("active-evidence")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def parameter_hash(model, trainable_only=False):
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        hasher.update(name.encode())
        data = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hasher.update(data)
    return hasher.hexdigest()


def frozen_parameter_probe(model):
    samples = []
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        samples.append(
            [
                name,
                tuple(parameter.shape),
                flat[:2].float().cpu().tolist(),
                flat[-2:].float().cpu().tolist(),
            ]
        )
    return digest(samples)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.settings = config["training"]
        self.path = config["model"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.answer_ids = [
            self.tokenizer.encode(str(label), add_special_tokens=False)
            for label in range(2)
        ]
        if any(len(ids) != 1 for ids in self.answer_ids):
            raise RuntimeError(
                f"Binary labels are not single tokens: {self.answer_ids}"
            )
        self.answer_ids = [ids[0] for ids in self.answer_ids]
        self.model = self.load_base()
        self.base_probe = frozen_parameter_probe(self.model)
        self.inference_tokens = 0

    def load_base(self):
        LOGGER.info("MODEL_LOAD path=%s", self.path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
        model.config.use_cache = False
        model.requires_grad_(False)
        if any(parameter.device.type != "cuda" for parameter in model.parameters()):
            raise RuntimeError("The model is not fully resident on the assigned GPU")
        return model

    def encode(self, prompts):
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for value in prompts
        ]
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] > self.settings["max_input_tokens"]:
            raise RuntimeError(
                f"Input would exceed the declared token limit: {encoded['input_ids'].shape[1]}"
            )
        return encoded.to("cuda:0")

    def forward(self, inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("NONFINITE_LOGITS")
        return logits

    def predictions(self, prompts):
        self.model.eval()
        results = []
        batch_size = self.settings["eval_batch_size"]
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                inputs = self.encode(prompts[start : start + batch_size])
                logits = self.forward(inputs)
                log_probs = logits.log_softmax(-1)
                selected = logits[:, self.answer_ids].softmax(-1)
                unrestricted = logits.argmax(-1)
                mass = log_probs[:, self.answer_ids].exp().sum(-1)
                self.inference_tokens += int(inputs["attention_mask"].sum())
                for probability, token_id, answer_mass in zip(
                    selected[:, 1].tolist(),
                    unrestricted.tolist(),
                    mass.tolist(),
                    strict=True,
                ):
                    results.append(
                        {
                            "prediction": int(probability > 0.5),
                            "probability_one": probability,
                            "answer_probability_mass": answer_mass,
                            "unrestricted_prediction": self.answer_ids.index(token_id)
                            if token_id in self.answer_ids
                            else None,
                            "unrestricted_token_id": token_id,
                        }
                    )
        return results

    def uncertainty(self, device):
        def score(candidates, evidence):
            prompts = [
                prompt({"device": device, "input": value}, evidence=evidence)
                for value in candidates
            ]
            return [
                binary_entropy(row["probability_one"])
                for row in self.predictions(prompts)
            ]

        return score

    def new_adapter(self, seed):
        if isinstance(self.model, PeftModel):
            raise RuntimeError("Previous adapter was not removed")
        seed_everything(seed)
        targets = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and ".language_model." in name
            and name.rsplit(".", 1)[-1] in self.settings["target_modules"]
        ]
        found_suffixes = {name.rsplit(".", 1)[-1] for name in targets}
        if not {"q_proj", "v_proj"}.issubset(found_suffixes):
            raise RuntimeError(
                f"Expected language attention targets missing: {found_suffixes}"
            )
        adapter = LoraConfig(
            r=self.settings["rank"],
            lora_alpha=self.settings["alpha"],
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
        )
        self.model = get_peft_model(self.model, adapter, autocast_adapter_dtype=True)
        trainable = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise RuntimeError("Unexpected trainable parameters")
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return {
            "target_modules": targets,
            "trainable_parameters": sum(
                parameter.numel() for _, parameter in trainable
            ),
            "trainable_dtypes": sorted(
                {str(parameter.dtype) for _, parameter in trainable}
            ),
            "base_dtype": "torch.bfloat16",
            "compute_dtype": "torch.bfloat16",
            "initial_adapter_sha256": parameter_hash(self.model, trainable_only=True),
        }

    def optimizer(self):
        return torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self.settings["learning_rate"],
            weight_decay=0.0,
        )

    def save_reload(self, destination, optimizer, probe_rows):
        self.model.eval()
        prompts = [prompt(row) for row in probe_rows]
        with torch.inference_mode():
            before = self.forward(self.encode(prompts)).cpu()
        original_hash = parameter_hash(self.model, trainable_only=True)
        self.model.save_pretrained(destination, safe_serialization=True)
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")
        del optimizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        base = self.load_base()
        if frozen_parameter_probe(base) != self.base_probe:
            raise RuntimeError("Original base model changed across loads")
        self.model = PeftModel.from_pretrained(
            base, destination, is_trainable=True, autocast_adapter_dtype=True
        ).eval()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        reloaded_hash = parameter_hash(self.model, trainable_only=True)
        if reloaded_hash != original_hash:
            raise RuntimeError("Checkpoint reload changed adapter tensors")
        with torch.inference_mode():
            after = self.forward(self.encode(prompts)).cpu()
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        optimizer = self.optimizer()
        optimizer.load_state_dict(
            torch.load(
                destination / "optimizer.pt", map_location="cuda:0", weights_only=True
            )
        )
        proof = {
            "adapter_sha256": original_hash,
            "adapter_file_sha256": file_hash(destination / "adapter_model.safetensors"),
            "reload_tensor_equality": True,
            "reload_logit_equality": True,
            "reload_max_logit_difference": float((after - before).abs().max()),
            "fresh_base_model_object": True,
            "probe_count": len(prompts),
            "evidence_in_probe_prompt": False,
            "use_cache": False,
        }
        write_json(destination / "reload-proof.json", proof)
        LOGGER.info("CHECKPOINT_RELOAD_PASSED %s", destination)
        return optimizer, proof

    def remove_adapter(self):
        self.model.gradient_checkpointing_disable()
        self.model = self.model.unload().eval()
        self.model.requires_grad_(False)
        if frozen_parameter_probe(self.model) != self.base_probe:
            raise RuntimeError("FROZEN_BASE_PROBE_CHANGED")
        gc.collect()
        torch.cuda.empty_cache()


def aggregate(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups["all"].append(row)
        groups[row["kind"]].append(row)
        if row["profile"] != row["kind"]:
            groups[row["profile"]].append(row)
        if row["kind"] == "single":
            groups[f"{row['profile']}/context{row['context']}"].append(row)
    metrics = {}
    for name, rows in groups.items():
        metrics[name] = {
            "n": len(rows),
            "accuracy": sum(row["prediction"] == row["label"] for row in rows)
            / len(rows),
            "unrestricted_accuracy": sum(
                row["unrestricted_prediction"] == row["label"] for row in rows
            )
            / len(rows),
            "valid_output_rate": sum(
                row["unrestricted_prediction"] is not None for row in rows
            )
            / len(rows),
            "answer_probability_mass": sum(
                row["answer_probability_mass"] for row in rows
            )
            / len(rows),
            "binary_log_loss": -sum(
                math.log(
                    max(
                        1e-9,
                        row["probability_one"]
                        if row["label"]
                        else 1 - row["probability_one"],
                    )
                )
                for row in rows
            )
            / len(rows),
        }
    return metrics


def evaluate(
    runner, rows, destination, evidence=(), specification=None, reloaded=False
):
    prompts = [
        prompt(row, evidence=evidence, specification=specification) for row in rows
    ]
    predictions = [
        {**row, **prediction}
        for row, prediction in zip(rows, runner.predictions(prompts), strict=True)
    ]
    metrics = aggregate(predictions)
    write_json(
        destination,
        {
            "metrics": metrics,
            "predictions": predictions,
            "prompt_sha256": digest(prompts),
            "evidence_in_prompt": bool(evidence),
            "true_specification_in_prompt": specification is not None,
            "checkpoint_reloaded": reloaded,
            "use_cache": False,
            "scoring": "Primary: forced choice over single tokens 0/1. Unrestricted next-token accuracy and answer mass are also reported.",
        },
    )
    LOGGER.info(
        "EVALUATION %s single=%s composition=%s",
        destination.name,
        metrics.get("single", {}).get("accuracy"),
        metrics.get("composition", {}).get("accuracy"),
    )
    return metrics


def collect_arm(runner, config, arm, output):
    seed, settings = config["seed"], config["environment"]
    rules = make_rules(seed)
    datasets = []
    for stage in range(2):
        dataset = []
        for device in stage_devices(stage):
            flip_steps = (
                settings["noise_flip_steps"] if stage == 1 and device == 3 else ()
            )

            def observe(
                value, query_index, rule=rules[stage][device], flips=flip_steps
            ):
                return rule(value) ^ int(query_index in flips)

            evidence, events = collect_device(
                seed,
                stage,
                device,
                arm,
                settings,
                observe,
                runner.uncertainty(device) if arm == "model_uncertainty" else None,
            )
            dataset.extend(evidence)
            for event in events:
                event["evaluator_only_clean_label"] = rules[stage][device](
                    event["input"]
                )
                event["evaluator_only_was_noise"] = (
                    event["observed"] != event["evaluator_only_clean_label"]
                )
                append_jsonl(output / "queries.jsonl", event)
            LOGGER.info(
                "ACQUISITION arm=%s stage=%d device=%d queries=%d candidates=%.2f",
                arm,
                stage,
                device,
                len(evidence),
                events[-1]["posterior_after"]["effective_candidates"],
            )
        datasets.append(dataset)
        write_json(output / f"observations-stage{stage}.json", dataset)
    _, heldout = partition(seed)
    if any(row["input"] in set(heldout) for dataset in datasets for row in dataset):
        raise RuntimeError("ACQUISITION_HELDOUT_LEAKAGE")
    if any(row["device"] == 0 for row in datasets[1]):
        raise RuntimeError("Stable retained device was exposed during the second stage")
    return datasets


def train_stage(runner, rows, plan, optimizer, config, stage, output, eval_rows):
    settings = config["training"]
    prompts = [prompt(row) for row in rows]
    encoded = runner.encode(prompts)
    lengths = encoded["attention_mask"].sum(-1).tolist()
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"Matched training-token budget violated by variable prompt lengths: {sorted(set(lengths))}"
        )
    token_length = lengths[0]
    batch_size = settings["batch_size"]
    trainable = [
        parameter for parameter in runner.model.parameters() if parameter.requires_grad
    ]
    initial_hash = parameter_hash(runner.model, trainable_only=True)
    losses, midpoint = [], None
    for update in range(settings["updates_per_stage"]):
        runner.model.train()
        indices = list(range(update * batch_size, (update + 1) * batch_size))
        inputs = {name: values[indices] for name, values in encoded.items()}
        labels = torch.tensor(
            [runner.answer_ids[rows[index]["label"]] for index in indices],
            device="cuda:0",
        )
        optimizer.zero_grad(set_to_none=True)
        logits = runner.forward(inputs)
        loss = functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NONFINITE_LOSS stage={stage} update={update}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, settings["max_grad_norm"], error_if_nonfinite=True
        )
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise RuntimeError(
                f"INVALID_GRADIENT stage={stage} update={update} norm={norm}"
            )
        if any(
            parameter.grad is not None
            for parameter in runner.model.parameters()
            if not parameter.requires_grad
        ):
            raise RuntimeError("FROZEN_PARAMETER_RECEIVED_GRADIENT")
        optimizer.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        append_jsonl(
            output / "training.jsonl",
            {
                "stage": stage,
                "update": update + 1,
                "loss": loss_value,
                "gradient_norm": float(norm),
                "examples": batch_size,
                "input_tokens": int(token_length * batch_size),
                "supervised_tokens": batch_size,
                "old_replay_examples": sum(
                    rows[index]["old_replay"] for index in indices
                ),
                "finite_gradient": True,
            },
        )
        if update == 0 or (update + 1) % 10 == 0:
            LOGGER.info(
                "TRAIN stage=%d update=%d loss=%.5f grad_norm=%.5f",
                stage,
                update + 1,
                loss_value,
                norm,
            )
        if (
            stage == 1
            and settings["midpoint_eval"]
            and update + 1 == settings["updates_per_stage"] // 2
        ):
            midpoint = evaluate(runner, eval_rows, output / "eval-stage1-midpoint.json")
    final_hash = parameter_hash(runner.model, trainable_only=True)
    if final_hash == initial_hash:
        raise RuntimeError("ADAPTER_PARAMETERS_DID_NOT_CHANGE")
    return {
        "updates": len(losses),
        "input_tokens": int(token_length * batch_size * len(losses)),
        "padded_input_tokens": int(
            encoded["input_ids"].shape[1] * batch_size * len(losses)
        ),
        "supervised_tokens": batch_size * len(losses),
        "examples": batch_size * len(losses),
        "old_replay_examples": plan["old_examples"],
        "current_examples": plan["current_examples"],
        "consolidation": plan,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "parameters_changed": True,
        "initial_adapter_sha256": initial_hash,
        "final_adapter_sha256": final_hash,
        "midpoint_metrics": midpoint,
    }


def run_arm(runner, config, arm, destination, evaluations):
    destination.mkdir(parents=True, exist_ok=False)
    inference_before = runner.inference_tokens
    datasets = collect_arm(runner, config, arm, destination)
    acquisition_tokens = runner.inference_tokens - inference_before
    context_metrics = []
    for stage in range(2):
        evidence = datasets[stage] + (
            [row for row in datasets[0] if row["device"] == 0] if stage else []
        )
        context_metrics.append(
            evaluate(
                runner,
                evaluations[stage],
                destination / f"context-only-stage{stage}.json",
                evidence=evidence,
            )
        )
    metadata = runner.new_adapter(config["seed"])
    optimizer = runner.optimizer()
    stages = []
    for stage in range(2):
        before = evaluate(
            runner,
            evaluations[stage],
            destination / f"eval-before-stage{stage}.json",
            reloaded=stage > 0,
        )
        rows, plan = training_plan(
            config,
            stage,
            datasets[stage],
            [row for row in datasets[0] if row["device"] == 0] if stage else [],
        )
        write_json(destination / f"training-rows-stage{stage}.json", rows)
        write_json(destination / f"consolidation-stage{stage}.json", plan)
        training = train_stage(
            runner,
            rows,
            plan,
            optimizer,
            config,
            stage,
            destination,
            evaluations[stage],
        )
        optimizer, reload_proof = runner.save_reload(
            destination / f"adapter-stage{stage}", optimizer, evaluations[stage][:4]
        )
        metrics = evaluate(
            runner,
            evaluations[stage],
            destination / f"eval-stage{stage}.json",
            reloaded=True,
        )
        fit = evaluate(
            runner,
            observed_rows(datasets[stage]),
            destination / f"training-fit-stage{stage}.json",
            reloaded=True,
        )
        stages.append(
            {
                "stage": stage,
                "before_metrics": before,
                "metrics": metrics,
                "training_fit": fit,
                "training": training,
                "reload": reload_proof,
            }
        )
        write_json(destination / "partial.json", {"arm": arm, "stages": stages})
    del optimizer
    runner.remove_adapter()
    result = {
        "arm": arm,
        "consolidation": config["consolidation"],
        "adapter": metadata,
        "stages": stages,
        "context_only": context_metrics,
        "environment_queries": sum(len(dataset) for dataset in datasets),
        "acquisition_model_input_tokens": acquisition_tokens,
        "inference_input_tokens": runner.inference_tokens - inference_before,
        "stable_retention_change": stages[1]["metrics"]["stable"]["accuracy"]
        - stages[0]["metrics"]["stable"]["accuracy"],
        "new_skill_learning_gain": stages[1]["metrics"]["new_skill"]["accuracy"]
        - stages[1]["before_metrics"]["new_skill"]["accuracy"],
        "boundary": {
            "student_evidence_prompt": False,
            "selector_frozen_base": True,
            "selector_accesses_heldout": False,
            "oracle_exact_family": arm == "hypothesis_elimination_oracle",
            "stable_device_replayed_in_stage1": config["consolidation"][
                "old_examples_per_batch"
            ]
            > 0,
            "posterior_training_uses_only_queried_ledger": True,
            "cache_enabled": False,
        },
    }
    write_json(destination / "result.json", result)
    return result


def summarize(results, config):
    signatures = [
        {
            "queries": result["environment_queries"],
            "stages": [
                {
                    key: stage["training"][key]
                    for key in (
                        "updates",
                        "input_tokens",
                        "padded_input_tokens",
                        "supervised_tokens",
                        "examples",
                        "old_replay_examples",
                        "current_examples",
                    )
                }
                for stage in result["stages"]
            ],
        }
        for result in results
    ]
    if len({digest(signature) for signature in signatures}) != 1:
        raise RuntimeError(f"CROSS_ARM_BUDGET_MISMATCH: {signatures}")
    if len({result["adapter"]["initial_adapter_sha256"] for result in results}) != 1:
        raise RuntimeError("CROSS_ARM_INITIALIZATION_MISMATCH")
    reference = next(result for result in results if result["arm"] == "random")
    rows = []
    for result in results:
        gain = (
            result["stages"][1]["metrics"]["single"]["accuracy"]
            - reference["stages"][1]["metrics"]["single"]["accuracy"]
        )
        retention_difference = (
            result["stable_retention_change"] - reference["stable_retention_change"]
        )
        durable_gain = (
            result["stages"][0]["metrics"]["single"]["accuracy"]
            - result["stages"][0]["before_metrics"]["single"]["accuracy"]
        )
        next_skill_difference = (
            result["new_skill_learning_gain"] - reference["new_skill_learning_gain"]
        )
        learnable = (
            min(
                result["stages"][0]["training_fit"]["single"]["accuracy"],
                reference["stages"][0]["training_fit"]["single"]["accuracy"],
            )
            >= config["screen"]["minimum_train_accuracy"]
        )
        if config["screen"]["qualification_only"]:
            decision = "qualification_only"
        elif result["arm"] == "random":
            decision = "baseline"
        elif not learnable:
            decision = "inconclusive_learner_not_qualified"
        elif (
            durable_gain < config["screen"]["minimum_durable_gain"]
            or result["new_skill_learning_gain"]
            < config["screen"]["minimum_new_skill_gain"]
        ):
            decision = "inconclusive_durable_learning_not_qualified"
        elif (
            gain >= config["screen"]["minimum_effect"]
            and retention_difference
            >= -config["screen"]["maximum_retention_regression"]
            and result["stable_retention_change"]
            >= -config["screen"]["maximum_stable_forgetting"]
            and next_skill_difference
            >= -config["screen"]["maximum_next_skill_regression"]
        ):
            decision = "keep_for_independent_replication"
        else:
            decision = "discard_screen_hypothesis"
        rows.append(
            {
                "arm": result["arm"],
                "decision": decision,
                "stage0_accuracy": result["stages"][0]["metrics"]["single"]["accuracy"],
                "stage1_accuracy": result["stages"][1]["metrics"]["single"]["accuracy"],
                "stage1_delta_vs_random": gain,
                "stage0_gain_vs_unadapted_base": durable_gain,
                "retention_difference_vs_random": retention_difference,
                "stable_retention_change": result["stable_retention_change"],
                "maximum_stable_forgetting": config["screen"][
                    "maximum_stable_forgetting"
                ],
                "next_skill_gain_difference_vs_random": next_skill_difference,
                "stage1_composition_accuracy": result["stages"][1]["metrics"][
                    "composition"
                ]["accuracy"],
                "new_skill_learning_gain": result["new_skill_learning_gain"],
                "learner_fit_qualified": learnable,
            }
        )
    return {
        "decisions": rows,
        "matched_budgets": signatures[0],
        "budgets_equal": True,
        "initialization_equal": True,
        "claim_limit": config["screen"]["claim"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [active-evidence] %(message)s"
    )
    config = json.loads(args.config.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in (
            "runtime.json",
            "data-manifest.json",
            "result.json",
            "failure.json",
        )
    ):
        raise RuntimeError(
            f"Output already contains lane execution artifacts: {output}"
        )
    saved_config = output / "config.json"
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise RuntimeError("Dispatcher and runner configurations disagree")
    if not saved_config.exists():
        write_json(saved_config, config)
    start = time.time()
    try:
        if config["screen"]["iterations"] != 1 or config["environment"]["bits"] != 10:
            raise ValueError(
                "This bounded screen implements one iteration with ten-bit environments"
            )
        if (
            len(config["arms"]) not in (2, 3)
            or "random" not in config["arms"]
            or any(arm not in ARMS for arm in config["arms"])
        ):
            raise ValueError(
                "Screen requires random plus one or two declared acquisition arms"
            )
        if (
            not torch.cuda.is_available()
            or not torch.version.hip
            or torch.cuda.device_count() != 1
        ):
            raise RuntimeError(
                f"Exactly one visible ROCm GPU is required; found {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        seed_everything(config["seed"])
        write_json(
            output / "runtime.json",
            {
                "hostname": platform.node(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "peft", "transformers", "accelerate")
                },
                "rocm": torch.version.hip,
                "gpu": str(torch.cuda.get_device_properties(0)),
                "visible_devices": {
                    name: os.environ.get(name)
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                    )
                },
                "source_sha256": {
                    path.name: file_hash(path)
                    for path in Path(__file__).parent.glob("*.py")
                },
                "config_sha256": file_hash(args.config),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_json(
            output / "data-manifest.json",
            manifest(config["seed"], config["environment"]),
        )
        evaluations = [
            evaluate_rows(config["seed"], stage, config["environment"])
            for stage in range(2)
        ]
        runner = ModelRunner(config)
        anchors = []
        for stage in range(2):
            anchors.append(
                {
                    "base": evaluate(
                        runner, evaluations[stage], output / f"base-stage{stage}.json"
                    ),
                    "specification_ceiling": evaluate(
                        runner,
                        evaluations[stage],
                        output / f"specification-ceiling-stage{stage}.json",
                        specification=make_rules(config["seed"])[stage],
                    ),
                }
            )
        results = []
        for arm in config["arms"]:
            results.append(run_arm(runner, config, arm, output / arm, evaluations))
        result = summarize(results, config)
        result.update(
            {
                "status": "complete",
                "arms": results,
                "anchors": anchors,
                "seconds": time.time() - start,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(output / "result.json", result)
        ledger = (
            output
            / "autoresearch"
            / ("screen-" + datetime.now(timezone.utc).strftime("%y%m%d-%H%M"))
        )
        ledger.mkdir(parents=True, exist_ok=True)
        columns = list(result["decisions"][0])
        lines = ["\t".join(columns)] + [
            "\t".join(str(row[column]) for column in columns)
            for row in result["decisions"]
        ]
        (ledger / "screen-results.tsv").write_text("\n".join(lines) + "\n")
        write_json(
            ledger / "handoff.json",
            {
                "status": "complete",
                "result": str(output / "result.json"),
                "decisions": result["decisions"],
            },
        )
        LOGGER.info("SCREEN_COMPLETE %s", json.dumps(result["decisions"]))
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "seconds": time.time() - start,
            },
        )
        LOGGER.exception("SCREEN_FAILED")
        raise


if __name__ == "__main__":
    main()
