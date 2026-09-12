import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tasks import (
    COHORTS,
    LABELS,
    STAGES,
    audit_corpus,
    build_corpus,
    digest,
    render,
    stage_batches,
    write_corpus,
)
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

LOGGER = logging.getLogger("plasticity")
ARMS = {"continue", "optimizer_reset", "capacity_refresh", "replay"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def event(output, kind, **values):
    record = {"event": kind, "time": datetime.now(timezone.utc).isoformat(), **values}
    with (Path(output) / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    LOGGER.info("%s %s", kind, json.dumps(values, sort_keys=True, allow_nan=False))


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(
        value.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def tree_hash(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_hash(value),
        }
    if isinstance(value, dict):
        return {str(key): tree_hash(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree_hash(item) for item in value]
    return value


def adapter_hash(model):
    return digest(
        {
            name: tree_hash(parameter)
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        }
    )


def frozen_adapter_hash(model):
    return digest(
        {
            name: tree_hash(parameter)
            for name, parameter in model.named_parameters()
            if ".lora_" in name and not parameter.requires_grad
        }
    )


def frozen_probe_hash(model):
    probes = {}
    for name, parameter in model.named_parameters():
        if ".lora_" not in name and parameter.numel():
            flat = parameter.detach().view(-1)
            probes[name.replace(".base_layer.", ".")] = tree_hash(
                flat[
                    torch.tensor(
                        [0, flat.numel() // 2, flat.numel() - 1], device=flat.device
                    )
                ]
            )
    return digest(probes)


def activate(model, adapters, trainable):
    model.base_model.set_adapter(adapters)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".lora_" in name and f".{trainable}." in name)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(".lora_" not in name for name in trainable_names):
        raise RuntimeError(
            "TRAINABLE_BOUNDARY: optimizer must own only the current LoRA adapter"
        )


def lora_config(base, config):
    targets = [
        name
        for name, module in base.named_modules()
        if name.startswith("model.language_model.")
        and name.rsplit(".", 1)[-1] in config["target_suffixes"]
        and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise RuntimeError("ADAPTER_TARGETS: no Qwen3.5 text projection targets found")
    return LoraConfig(
        r=config["rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )


def new_optimizer(model, config):
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config["weight_decay"],
        foreach=False,
    )


def load_base(config):
    LOGGER.info("MODEL_LOAD path=%s", config["model_path"])
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_path"],
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=config["device"],
        attn_implementation=config["attention"],
    )
    base.config.use_cache = False
    if config["gradient_checkpointing"]:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if any(
        parameter.device != torch.device(config["device"])
        for parameter in base.parameters()
    ):
        raise RuntimeError(
            "MODEL_PLACEMENT: all parameters must be on the assigned GPU"
        )
    return base


def new_model(config):
    torch.manual_seed(config["seed"])
    base = load_base(config)
    model = get_peft_model(
        base,
        lora_config(base, config),
        adapter_name="acquired",
        autocast_adapter_dtype=True,
    )
    activate(model, ["acquired"], "acquired")
    return model, new_optimizer(model, config)


class EncodedCorpus:
    def __init__(self, tokenizer, corpus, config):
        self.tokenizer = tokenizer
        self.device = config["device"]
        self.max_length = config["max_length"]
        self.label_ids = [
            tokenizer.encode(label, add_special_tokens=False) for label in LABELS
        ]
        if any(len(ids) != 1 for ids in self.label_ids):
            raise RuntimeError(
                f"LABEL_TOKENIZATION: each answer must have one token: {self.label_ids}"
            )
        self.label_ids = [ids[0] for ids in self.label_ids]
        self.cache = {}
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise RuntimeError("TOKENIZER_PADDING: no pad or EOS token")
        for splits in corpus.values():
            for rows in splits.values():
                for row in rows:
                    ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": render(row)}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_dict=False,
                    )
                    if len(ids) > self.max_length:
                        raise RuntimeError(
                            f"CONTEXT_TRUNCATION: {row['id']} requires {len(ids)} tokens"
                        )
                    padding = self.max_length - len(ids)
                    self.cache[row["id"]] = (
                        torch.tensor([pad] * padding + ids),
                        torch.tensor([0] * padding + [1] * len(ids)),
                    )

    def batch(self, rows, observed=False):
        values = [self.cache[row["id"]] for row in rows]
        inputs = {
            "input_ids": torch.stack([item[0] for item in values]).to(self.device),
            "attention_mask": torch.stack([item[1] for item in values]).to(self.device),
        }
        key = "observed_label" if observed else "label"
        labels = torch.tensor(
            [self.label_ids[LABELS.index(row[key])] for row in rows], device=self.device
        )
        return inputs, labels


def next_logits(model, inputs):
    device_type = next(model.parameters()).device.type
    precision = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else contextlib.nullcontext()
    )
    with precision:
        logits = (
            model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError("NONFINITE_LOGITS: forward pass returned NaN or infinity")
    return logits


def wilson(correct, total):
    if total == 0:
        return None
    p = correct / total
    z = 1.959963984540054
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate(model, encoded, rows, batch_size, predictions_path=None):
    model.eval()
    records = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            inputs, labels = encoded.batch(chunk)
            logits = next_logits(model, inputs)
            predictions = logits.argmax(-1).cpu().tolist()
            candidates = logits[:, encoded.label_ids].argmax(-1).cpu().tolist()
            losses = F.cross_entropy(logits, labels, reduction="none").cpu().tolist()
            for row, token, candidate, loss in zip(
                chunk, predictions, candidates, losses, strict=True
            ):
                predicted = (
                    LABELS[encoded.label_ids.index(token)]
                    if token in encoded.label_ids
                    else "OTHER"
                )
                records.append(
                    {
                        "id": row["id"],
                        "cohort": row["cohort"],
                        "split": row["split"],
                        "scope": row["scope"],
                        "label": row["label"],
                        "predicted": predicted,
                        "token_id": token,
                        "decoded_token": encoded.tokenizer.decode([token]),
                        "candidate_prediction": LABELS[candidate],
                        "correct": predicted == row["label"],
                        "candidate_correct": LABELS[candidate] == row["label"],
                        "nll": loss,
                        "old_label": row["old_label"],
                        "global_base_label": row["global_base_label"],
                    }
                )
    if predictions_path:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
    correct = sum(record["correct"] for record in records)
    changed = [record for record in records if record["old_label"] != record["label"]]
    scopes = {}
    for scope in ("ordinary", "exception"):
        subset = [record for record in records if record["scope"] == scope]
        scopes[scope] = {
            "n": len(subset),
            "accuracy": sum(record["correct"] for record in subset) / len(subset)
            if subset
            else None,
        }
    return {
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "accuracy_wilson95": wilson(correct, len(records)),
        "candidate_accuracy": sum(record["candidate_correct"] for record in records)
        / len(records),
        "format_compliance": sum(record["predicted"] != "OTHER" for record in records)
        / len(records),
        "mean_nll": sum(record["nll"] for record in records) / len(records),
        "scopes": scopes,
        "revised_cases": len(changed),
        "stale_rule_rate": sum(
            record["predicted"] == record["old_label"] for record in changed
        )
        / len(changed)
        if changed
        else None,
    }


def evaluate_cohorts(
    model, encoded, corpus, cohorts, split_prefix, config, output=None
):
    metrics = {}
    for cohort in cohorts:
        for kind in ("iid", "composition"):
            split = f"{split_prefix}_{kind}"
            path = Path(output) / f"{cohort}.{split}.jsonl" if output else None
            metrics[f"{cohort}/{kind}"] = evaluate(
                model, encoded, corpus[cohort][split], config["eval_batch_size"], path
            )
    return metrics


def checkpoint_load(path, config, load_optimizer=True):
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if (
        metadata["model_id"] != config["model_id"]
        or metadata["model_revision"] != config["model_revision"]
    ):
        raise RuntimeError(
            "CHECKPOINT_BASE_IDENTITY: checkpoint references a different model revision"
        )
    for relative, expected in metadata["files"].items():
        if hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"CHECKPOINT_CORRUPTION: {relative}")
    base = load_base(config)
    adapters = metadata["active_adapters"]
    model = PeftModel.from_pretrained(
        base,
        path / "adapters" / adapters[0],
        adapter_name=adapters[0],
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    for adapter in adapters[1:]:
        model.load_adapter(
            path / "adapters" / adapter,
            adapter_name=adapter,
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
    activate(model, adapters, metadata["trainable_adapter"])
    optimizer = new_optimizer(model, config)
    if load_optimizer:
        optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    if adapter_hash(model) != metadata["adapter_tensor_sha256"]:
        raise RuntimeError(
            "CHECKPOINT_TENSOR_IDENTITY: adapter tensors changed after reload"
        )
    if (
        load_optimizer
        and digest(tree_hash(optimizer.state_dict())) != metadata["optimizer_sha256"]
    ):
        raise RuntimeError(
            "CHECKPOINT_OPTIMIZER_IDENTITY: optimizer state changed after reload"
        )
    return model, optimizer, metadata


def save_and_verify(
    model, optimizer, adapters, trainable, path, config, encoded, sentinels
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.eval()
    inputs, _ = encoded.batch(sentinels)
    with torch.inference_mode():
        before = next_logits(model, inputs).cpu()
    model.save_pretrained(
        path / "adapters",
        selected_adapters=adapters,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    metadata = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "active_adapters": adapters,
        "trainable_adapter": trainable,
        "adapter_tensor_sha256": adapter_hash(model),
        "optimizer_sha256": digest(tree_hash(optimizer.state_dict())),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_adapter_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        ),
        "trainable_rank": config["rank"],
        "cumulative_rank_upper_bound": len(adapters) * config["rank"],
        "files": {
            str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(path.rglob("*"))
            if file.is_file()
        },
    }
    write_json(path / "checkpoint.json", metadata)
    reloaded, reloaded_optimizer, _ = checkpoint_load(path, config)
    reloaded.eval()
    with torch.inference_mode():
        after = next_logits(reloaded, inputs).cpu()
    max_difference = float((after - before).abs().max())
    identity = {
        "adapter_tensors_exact": True,
        "optimizer_state_exact": True,
        "logits_exact": torch.equal(before, after),
        "logits_max_abs_difference": max_difference,
        "argmax_equal": bool(torch.equal(before.argmax(-1), after.argmax(-1))),
        "sentinel_ids": [row["id"] for row in sentinels],
    }
    write_json(path / "reload.json", identity)
    if not identity["logits_exact"]:
        raise RuntimeError(
            f"CHECKPOINT_LOGIT_IDENTITY: reloaded logits differ by {max_difference}"
        )
    return reloaded, reloaded_optimizer, identity


def train_stage(
    model, optimizer, adapters, trainable, arm, stage, config, corpus, encoded, output
):
    before_hash = adapter_hash(model)
    frozen_before = frozen_probe_hash(model)
    frozen_adapters_before = frozen_adapter_hash(model)
    curves = []
    budget = Counter()
    gradient_norms = []
    losses = []
    new_cohort = STAGES[stage]["new"]
    stage_output = Path(output) / "metrics" / arm / stage
    stage_output.mkdir(parents=True, exist_ok=True)
    optimizer_steps = [
        float(state["step"]) for state in optimizer.state.values() if "step" in state
    ]
    event(
        output,
        "STAGE_START",
        arm=arm,
        stage=stage,
        active_adapters=adapters,
        trainable_adapter=trainable,
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        adapter_tensor_sha256=before_hash,
        optimizer_state_sha256=digest(tree_hash(optimizer.state_dict())),
        optimizer_step_min=min(optimizer_steps) if optimizer_steps else 0,
        optimizer_step_max=max(optimizer_steps) if optimizer_steps else 0,
    )

    def probe(step):
        metrics = evaluate_cohorts(
            model, encoded, corpus, [new_cohort], "validation", config
        )
        point = {"step": step, "new_examples": budget["new"], "metrics": metrics}
        curves.append(point)
        write_json(stage_output / "curve.json", curves)
        event(
            output,
            "LEARNING_PROBE",
            arm=arm,
            stage=stage,
            step=step,
            iid=metrics[f"{new_cohort}/iid"]["accuracy"],
            composition=metrics[f"{new_cohort}/composition"]["accuracy"],
        )

    probe(0)
    began = time.perf_counter()
    for step, (rows, roles) in enumerate(
        stage_batches(corpus, stage, arm, config), start=1
    ):
        model.train()
        inputs, labels = encoded.batch(rows, observed=True)
        optimizer.zero_grad(set_to_none=True)
        logits = next_logits(model, inputs)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NONFINITE_LOSS: {arm}/{stage}/{step}")
        loss.backward()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, config["grad_clip"], error_if_nonfinite=True
            )
        )
        if grad_norm <= 0:
            raise RuntimeError(f"ZERO_GRADIENT: {arm}/{stage}/{step}")
        optimizer.step()
        gradient_norms.append(grad_norm)
        losses.append(float(loss.detach()))
        budget.update(roles)
        budget["updates"] += 1
        budget["examples"] += len(rows)
        budget["padded_input_tokens"] += int(inputs["input_ids"].numel())
        budget["input_tokens"] += int(inputs["attention_mask"].sum())
        budget["label_tokens"] += len(rows)
        budget["corrupted_labels"] += sum(row["label_corrupted"] for row in rows)
        with (stage_output / "updates.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "gradient_norm": grad_norm,
                        "ids": [row["id"] for row in rows],
                        "roles": roles,
                    }
                )
                + "\n"
            )
        if step in config["eval_steps"]:
            probe(step)
        elif step % 8 == 0:
            event(
                output,
                "TRAIN_PROGRESS",
                arm=arm,
                stage=stage,
                step=step,
                loss=losses[-1],
                gradient_norm=grad_norm,
            )
    after_hash = adapter_hash(model)
    if before_hash == after_hash:
        raise RuntimeError(f"NO_PARAMETER_UPDATE: {arm}/{stage}")
    if frozen_probe_hash(model) != frozen_before:
        raise RuntimeError(f"FROZEN_PARAMETER_CHANGE: {arm}/{stage}")
    if frozen_adapter_hash(model) != frozen_adapters_before:
        raise RuntimeError(f"FROZEN_ADAPTER_CHANGE: {arm}/{stage}")
    if (
        budget["updates"] != config["updates_per_stage"]
        or budget["examples"] != config["updates_per_stage"] * config["batch_size"]
    ):
        raise RuntimeError(f"UPDATE_BUDGET: {arm}/{stage}")
    validation = evaluate_cohorts(
        model,
        encoded,
        corpus,
        list(COHORTS),
        "validation",
        config,
        stage_output / "validation",
    )
    checkpoint = Path(output) / "checkpoints" / arm / stage
    sentinels = corpus[new_cohort]["validation_iid"][:4]
    reloaded, reloaded_optimizer, identity = save_and_verify(
        model, optimizer, adapters, trainable, checkpoint, config, encoded, sentinels
    )
    result = {
        "arm": arm,
        "stage": stage,
        "curve": curves,
        "validation": validation,
        "budget": dict(budget),
        "checkpoint": str(checkpoint),
        "reload": identity,
        "adapter_before_sha256": before_hash,
        "adapter_after_sha256": after_hash,
        "real_parameter_update": True,
        "frozen_parameter_probes_unchanged": True,
        "frozen_historical_adapters_exact": True,
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "seconds": time.perf_counter() - began,
    }
    write_json(stage_output / "stage.json", result)
    event(
        output,
        "STAGE_COMPLETE",
        arm=arm,
        stage=stage,
        checkpoint=str(checkpoint),
        budget=dict(budget),
        reload_exact=True,
    )
    return reloaded, reloaded_optimizer, result


def score(metrics, cohort):
    return (
        sum(metrics[f"{cohort}/{kind}"]["accuracy"] for kind in ("iid", "composition"))
        / 2
    )


def curve_auc(stage_result):
    cohort = STAGES[stage_result["stage"]]["new"]
    points = stage_result["curve"]
    area = sum(
        (right["step"] - left["step"])
        * (score(left["metrics"], cohort) + score(right["metrics"], cohort))
        / 2
        for left, right in pairwise(points)
    )
    absolute = area / points[-1]["step"]
    return {
        "accuracy_auc": absolute,
        "gain_auc": absolute - score(points[0]["metrics"], cohort),
        "initial_accuracy": score(points[0]["metrics"], cohort),
        "final_accuracy": score(points[-1]["metrics"], cohort),
    }


def analyze(results, config, output):
    metrics = {}
    for arm, stages in results["arms"].items():
        metrics[arm] = {
            stage: curve_auc(value)
            for stage, value in stages.items()
            if stage in STAGES
        }
    references_learnable = all(
        metrics[f"fresh_{stage}"][stage]["final_accuracy"]
        >= config["min_fresh_reference_accuracy"]
        for stage in config["fresh_reference_stages"]
    )
    reference = results["arms"]["continue"]
    baseline_auc = (
        sum(metrics["continue"][stage]["accuracy_auc"] for stage in ("B", "C")) / 2
    )
    baseline_retention = (
        sum(score(reference["C"]["validation"], cohort) for cohort in ("A0", "B1")) / 2
    )
    baseline_test = (
        sum(score(reference["test"], cohort) for cohort in ("A0", "A2", "B1", "C2")) / 4
    )
    ledger = []
    for arm in config["arms"]:
        auc = sum(metrics[arm][stage]["accuracy_auc"] for stage in ("B", "C")) / 2
        retention = (
            sum(
                score(results["arms"][arm]["C"]["validation"], cohort)
                for cohort in ("A0", "B1")
            )
            / 2
        )
        test_score = (
            sum(
                score(results["arms"][arm]["test"], cohort)
                for cohort in ("A0", "A2", "B1", "C2")
            )
            / 4
        )
        gain = auc - baseline_auc
        retained = retention - baseline_retention
        candidate = (
            arm != "continue"
            and references_learnable
            and gain >= config["candidate_auc_gain"]
            and retained >= -config["max_retention_regression"]
        )
        test_consistent = (
            test_score - baseline_test >= -config["max_retention_regression"]
        )
        ledger.append(
            {
                "arm": arm,
                "new_task_auc": auc,
                "auc_gain": gain,
                "retention": retention,
                "retention_delta": retained,
                "locked_test_score": test_score,
                "locked_test_delta": test_score - baseline_test,
                "fresh_references_learnable": references_learnable,
                "decision": "baseline"
                if arm == "continue"
                else "keep_for_replication"
                if candidate and test_consistent
                else "discard_at_this_budget",
            }
        )
    directory = Path(output) / "autoresearch" / "plasticity-screen"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "screen-results.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(ledger)
    candidates = [
        row["arm"] for row in ledger if row["decision"] == "keep_for_replication"
    ]
    write_json(
        directory / "handoff.json",
        {
            "status": "SCREEN_COMPLETE",
            "candidates": candidates,
            "next_action": "replicate baseline and selected candidates on training seed 29"
            if candidates
            else "report no candidate at this budget",
            "selection_used": "validation AUC and retention, with locked test used once for consistency",
            "replication_required": True,
        },
    )
    return {
        "curves": metrics,
        "ledger": ledger,
        "candidates": candidates,
        "fresh_references_learnable": references_learnable,
        "verdict": "candidate_requires_replication"
        if candidates
        else "no_candidate_at_this_budget",
    }


def validate_config(config):
    revision = config["model_revision"]
    snapshot = Path(config["model_path"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "MODEL_REVISION: expected an immutable 40-character hexadecimal revision"
        )
    if not snapshot.is_absolute() or snapshot.name != revision:
        raise ValueError(
            "MODEL_SNAPSHOT_PATH: expected an absolute snapshot path ending in the pinned revision"
        )
    if set(config["arms"]) != ARMS:
        raise ValueError(
            "SCREEN_ARMS: the initial screen requires all four predeclared arms"
        )
    if (
        config["eval_steps"][0] != 0
        or config["eval_steps"][-1] != config["updates_per_stage"]
    ):
        raise ValueError(
            "CURVE_ENDPOINTS: evaluation steps must include zero and final update"
        )
    if sorted(set(config["eval_steps"])) != config["eval_steps"]:
        raise ValueError("CURVE_ORDER: evaluation steps must be strictly increasing")
    if config["device"] != "cuda:0":
        raise ValueError("GPU_SLOT: use exactly one visible GPU, addressed as cuda:0")


def runtime_receipt(config):
    snapshot = Path(config["model_path"])
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise RuntimeError(f"MODEL_SNAPSHOT_MISSING: {snapshot}")
    index = snapshot / "model.safetensors.index.json"
    weights = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.is_file()
        else {"model.safetensors"}
    )
    missing = [name for name in weights if not (snapshot / name).is_file()]
    if missing or not (snapshot / "tokenizer.json").is_file():
        raise RuntimeError(
            f"MODEL_SNAPSHOT_INCOMPLETE: weights={missing}, tokenizer_present={(snapshot / 'tokenizer.json').is_file()}"
        )
    if (
        not torch.cuda.is_available()
        or not torch.version.hip
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            f"GPU_SLOT: expected exactly one ROCm GPU; count={torch.cuda.device_count()}, hip={torch.version.hip}"
        )
    torch.cuda.set_device(0)
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "rocm": torch.version.hip,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "peft", "safetensors")
        },
        "gpu": str(torch.cuda.get_device_properties(0)),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "dtype": "bfloat16 backbone and autocast; float32 adapter parameters and optimizer state",
        "sdk": {
            "peft_from_pretrained": str(inspect.signature(PeftModel.from_pretrained)),
            "qwen_forward": str(
                inspect.signature(Qwen3_5ForConditionalGeneration.forward)
            ),
        },
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "snapshot_path": str(snapshot),
        "snapshot_weight_files": sorted(weights),
        "snapshot_files_exist": True,
        "teacher": None,
        "retrieval": False,
        "context_messages_per_example": 1,
        "base_weights_trainable": False,
    }


def run(config, output, corpus):
    write_json(Path(output) / "runtime.json", runtime_receipt(config))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True
    )
    encoded = EncodedCorpus(tokenizer, corpus, config)
    write_json(
        Path(output) / "tokenization.json",
        {
            "label_token_ids": dict(zip(LABELS, encoded.label_ids, strict=True)),
            "max_length": encoded.max_length,
            "examples": len(encoded.cache),
            "no_truncation": True,
        },
    )
    model, optimizer = new_model(config)
    results = {"status": "running", "seed": config["seed"], "arms": {}}
    model, optimizer, acquisition = train_stage(
        model,
        optimizer,
        ["acquired"],
        "acquired",
        "shared",
        "A",
        config,
        corpus,
        encoded,
        output,
    )
    acquired = acquisition["validation"]
    gain = score(acquired, "A0") - score(acquisition["curve"][0]["metrics"], "A0")
    qualified = (
        acquired["A0/iid"]["accuracy"] >= config["min_acquisition_iid"]
        and acquired["A0/composition"]["accuracy"]
        >= config["min_acquisition_composition"]
        and gain >= config["min_acquisition_gain"]
    )
    results["acquisition"] = acquisition
    results["acquisition_qualified"] = qualified
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if not qualified:
        results["status"] = "unqualified_acquisition"
        results["conclusion"] = (
            "No plasticity claim: task A did not meet the predeclared held-out acquisition gate."
        )
        directory = Path(output) / "autoresearch" / "plasticity-screen"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "screen-results.tsv").write_text(
            "arm\tdecision\tiid\tcomposition\tgain\nshared_A\tdiscard_unqualified_acquisition\t"
            + str(acquired["A0/iid"]["accuracy"])
            + "\t"
            + str(acquired["A0/composition"]["accuracy"])
            + "\t"
            + str(gain)
            + "\n"
        )
        write_json(
            directory / "handoff.json",
            {
                "status": "UNQUALIFIED",
                "next_action": "inspect acquisition curve before running continual-learning comparisons",
            },
        )
        return results
    for arm in config["arms"]:
        model, optimizer, _ = checkpoint_load(acquisition["checkpoint"], config)
        adapters = ["acquired"]
        trainable = "acquired"
        stages = {}
        for stage in ("B", "C"):
            if arm == "optimizer_reset":
                optimizer = new_optimizer(model, config)
            elif arm == "capacity_refresh":
                torch.manual_seed(config["seed"] + 2003 * (ord(stage) - ord("A")))
                trainable = f"stage_{stage}"
                base_config = model.peft_config["acquired"]
                model.add_adapter(
                    trainable,
                    LoraConfig(
                        r=config["rank"],
                        lora_alpha=config["lora_alpha"],
                        target_modules=list(base_config.target_modules),
                        lora_dropout=0.0,
                        bias="none",
                        init_lora_weights=True,
                    ),
                )
                adapters.append(trainable)
                activate(model, adapters, trainable)
                optimizer = new_optimizer(model, config)
            model, optimizer, stage_result = train_stage(
                model,
                optimizer,
                adapters,
                trainable,
                arm,
                stage,
                config,
                corpus,
                encoded,
                output,
            )
            stages[stage] = stage_result
        stages["test"] = evaluate_cohorts(
            model,
            encoded,
            corpus,
            list(COHORTS),
            "test",
            config,
            Path(output) / "metrics" / arm / "locked_test",
        )
        results["arms"][arm] = stages
        write_json(Path(output) / "results.partial.json", results)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    for stage in config["fresh_reference_stages"]:
        arm = f"fresh_{stage}"
        model, optimizer = new_model(config)
        model, optimizer, stage_result = train_stage(
            model,
            optimizer,
            ["acquired"],
            "acquired",
            arm,
            stage,
            config,
            corpus,
            encoded,
            output,
        )
        test = evaluate_cohorts(
            model,
            encoded,
            corpus,
            [STAGES[stage]["new"]],
            "test",
            config,
            Path(output) / "metrics" / arm / "locked_test",
        )
        results["arms"][arm] = {stage: stage_result, "test": test}
        write_json(Path(output) / "results.partial.json", results)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    results["analysis"] = analyze(results, config, output)
    results["status"] = "complete"
    return results


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [plasticity] %(message)s"
    )
    config = json.loads(args.config.read_text())
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    owned_artifacts = {
        "status.json",
        "events.jsonl",
        "source.json",
        "code",
        "data",
        "data-audit.json",
        "checkpoints",
        "metrics",
        "results.json",
        "results.partial.json",
        "autoresearch",
        "runtime.json",
        "tokenization.json",
    }
    if any(path.name in owned_artifacts for path in args.output_dir.iterdir()):
        raise RuntimeError(
            "OUTPUT_ALREADY_USED: lane artifacts already exist in this output directory"
        )
    if args.config.resolve() != (args.output_dir / "config.json").resolve():
        write_json(args.output_dir / "config.json", config)
    source = Path(__file__).resolve().parent
    (args.output_dir / "code").mkdir()
    for name in ("run.py", "tasks.py", "requirements.txt"):
        shutil.copy2(source / name, args.output_dir / "code" / name)
    write_json(
        args.output_dir / "source.json",
        {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted((args.output_dir / "code").iterdir())
        },
    )
    write_json(args.output_dir / "status.json", {"status": "validating"})
    try:
        corpus = build_corpus(config)
        audit = audit_corpus(corpus)
        write_corpus(corpus, args.output_dir / "data")
        write_json(args.output_dir / "data-audit.json", audit)
        event(
            args.output_dir,
            "DATA_AUDIT_PASS",
            rows=audit["rows"],
            sha256=audit["sha256"],
        )
        if args.validate_only:
            write_json(
                args.output_dir / "status.json",
                {"status": "data_validated", "gpu_training_run": False},
            )
            return
        write_json(args.output_dir / "status.json", {"status": "running"})
        results = run(config, args.output_dir, corpus)
        write_json(args.output_dir / "results.json", results)
        write_json(
            args.output_dir / "status.json",
            {
                "status": results["status"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        event(
            args.output_dir,
            "RUN_FINISHED",
            status=results["status"],
            output=str(args.output_dir),
        )
    except Exception as error:
        write_json(
            args.output_dir / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        LOGGER.exception("LANE_FAILURE output=%s", args.output_dir)
        raise


if __name__ == "__main__":
    main()
