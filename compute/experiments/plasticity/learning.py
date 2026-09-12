import math
import re
from itertools import pairwise

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3_5ForConditionalGeneration,
)

from config import derived_seed
from data import collate


def attach_adapter(base, cfg):
    targets = [
        name
        for name, module in base.named_modules()
        if isinstance(module, torch.nn.Linear)
        and re.search(r"\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$", name)
    ]
    if not targets:
        raise RuntimeError("LORA_TARGETS_MISSING: expected Qwen text MLP linear layers")
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=True,
        ),
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if ".lora_A." not in name and ".lora_B." not in name:
                raise RuntimeError(f"UNEXPECTED_TRAINABLE {name}")
            parameter.data = parameter.data.float()
    return model


def load_model(cfg):
    config = AutoConfig.from_pretrained(
        cfg.model_path, local_files_only=True, trust_remote_code=False
    )
    model_class = (
        Qwen3_5ForConditionalGeneration
        if config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    if config.model_type not in ("qwen3_5", "qwen3", "qwen2"):
        raise ValueError(
            f"Unsupported model_type={config.model_type}; expected Qwen3.5, Qwen3, or Qwen2"
        )
    base = model_class.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, cfg.dtype),
        device_map=cfg.device,
        attn_implementation="sdpa",
    )
    return attach_adapter(base, cfg)


def trainables(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def snapshot(model, cpu=True):
    return {
        name: p.detach().to("cpu" if cpu else p.device, copy=True)
        for name, p in trainables(model).items()
    }


@torch.no_grad()
def restore(model, values):
    parameters = trainables(model)
    if parameters.keys() != values.keys():
        raise RuntimeError("CHECKPOINT_TRAINABLE_MISMATCH")
    for name, parameter in parameters.items():
        parameter.copy_(values[name])


def optimizer_for(model, cfg):
    return torch.optim.AdamW(
        list(trainables(model).values()),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
        foreach=False,
        fused=False,
    )


def adapter_layers(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer) and "default" in module.lora_A
    ]


@torch.no_grad()
def selective_reset(model, cfg, task_index):
    changes = []
    for name, module in adapter_layers(model):
        a = module.lora_A["default"].weight
        b = module.lora_B["default"].weight
        scale = module.scaling["default"]
        contribution = a.float().norm(dim=1) * b.float().norm(dim=0) * abs(scale)
        count = int(a.shape[0] * cfg.reset_fraction)
        chosen = torch.argsort(contribution, stable=True)[:count]
        generator = torch.Generator(device="cpu").manual_seed(
            derived_seed(cfg.seed, "renewal", task_index, name)
        )
        replacement = torch.empty((count, a.shape[1]), dtype=torch.float32)
        replacement.uniform_(
            -1 / math.sqrt(a.shape[1]), 1 / math.sqrt(a.shape[1]), generator=generator
        )
        a[chosen] = replacement.to(a.device, a.dtype)
        b[:, chosen] = 0
        changes.append(
            {
                "module": name,
                "components": chosen.cpu().tolist(),
                "removed_component_frobenius_norms": contribution[chosen]
                .cpu()
                .tolist(),
                "rule": "lowest ||B[:,j]||2 * ||A[j,:]||2 * scaling; A fresh Kaiming-uniform, B zero",
            }
        )
    if not changes:
        raise RuntimeError("NO_ADAPTER_COMPONENTS_TO_RENEW")
    return changes


@torch.no_grad()
def apply_boundary(model, optimizer, initial, method, task_index, cfg):
    changes = []
    if task_index == 0 or method == "fresh_base":
        restore(model, initial)
        optimizer = optimizer_for(model, cfg)
    elif method == "optimizer_reset":
        optimizer = optimizer_for(model, cfg)
    elif method == "selective_reset":
        changes = selective_reset(model, cfg, task_index)
        optimizer = optimizer_for(model, cfg)
    elif method not in ("persistent", "balanced_replay"):
        raise ValueError(f"Unknown method {method}")
    return optimizer, changes


def class_logits(model, inputs, code_token_ids):
    outputs = model(**inputs, logits_to_keep=1, use_cache=False, return_dict=True)
    selected = outputs.logits[:, -1, code_token_ids].float()
    if not torch.isfinite(selected).all():
        raise FloatingPointError("NONFINITE_CLASS_LOGITS")
    return selected


def score_logits(logits, labels, task_codes):
    local_codes = torch.tensor(task_codes, dtype=torch.long, device=logits.device)
    matches = labels[:, None] == local_codes[None, :]
    if not matches.any(dim=1).all():
        raise ValueError("Evaluation labels are outside the supplied task")
    local_labels = matches.long().argmax(dim=1)
    local_logits = logits[:, local_codes]
    return {
        "global_correct": int((logits.argmax(-1) == labels).sum()),
        "global_loss_sum": float(F.cross_entropy(logits, labels, reduction="sum")),
        "task_local_correct": int((local_logits.argmax(-1) == local_labels).sum()),
        "task_local_loss_sum": float(
            F.cross_entropy(local_logits, local_labels, reduction="sum")
        ),
        "global_confidence_sum": float(logits.softmax(-1).amax(-1).sum()),
    }


@torch.no_grad()
def evaluate(model, task, split, prepared, cfg):
    model.eval()
    totals = {
        "global_correct": 0,
        "global_loss_sum": 0.0,
        "task_local_correct": 0,
        "task_local_loss_sum": 0.0,
        "global_confidence_sum": 0.0,
    }
    predictions = []
    rows = task.rows[split]
    for start in range(0, len(rows), cfg.eval_batch_size):
        batch = rows[start : start + cfg.eval_batch_size]
        inputs, labels = collate(batch, prepared.pad_token_id, cfg.device)
        logits = class_logits(model, inputs, prepared.code_token_ids)
        scored = score_logits(logits, labels, task.codes)
        for key, value in scored.items():
            totals[key] += value
        predictions.extend(logits.argmax(-1).cpu().tolist())
    n = len(rows)
    return {
        "task": task.index,
        "split": split,
        "examples": n,
        "accuracy": totals["global_correct"] / n,
        "loss": totals["global_loss_sum"] / n,
        "correct": totals["global_correct"],
        "task_local_accuracy": totals["task_local_correct"] / n,
        "task_local_loss": totals["task_local_loss_sum"] / n,
        "mean_confidence": totals["global_confidence_sum"] / n,
        "predictions": predictions,
        "input_tokens": sum(len(row["input_ids"]) for row in rows),
    }


@torch.no_grad()
def factor_norm_squared(a, b):
    return float(((b.T @ b) * (a @ a.T)).sum().clamp_min(0))


@torch.no_grad()
def update_diagnostics(model, before):
    squared = torch.zeros(
        (), device=next(model.parameters()).device, dtype=torch.float32
    )
    parameter_squared = torch.zeros_like(squared)
    for name, p in trainables(model).items():
        squared += (p - before[name].to(p.device)).float().square().sum()
        parameter_squared += p.float().square().sum()
    effective_squared = 0.0
    for name, layer in adapter_layers(model):
        a = layer.lora_A["default"].weight.float()
        b = layer.lora_B["default"].weight.float()
        old_a = before[f"{name}.lora_A.default.weight"].to(a.device)
        old_b = before[f"{name}.lora_B.default.weight"].to(b.device)
        joined_a = torch.cat((old_a, a - old_a), dim=0)
        joined_b = torch.cat((b - old_b, b), dim=1)
        effective_squared += (
            factor_norm_squared(joined_a, joined_b) * layer.scaling["default"] ** 2
        )
    return {
        "adapter_parameter_update_l2": float(squared.sqrt()),
        "adapter_parameter_l2": float(parameter_squared.sqrt()),
        "effective_adapter_update_frobenius": math.sqrt(effective_squared),
    }


def activation_statistics(features, gates):
    features = features.double()
    centered = features - features.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-30)
    nonzero = probabilities[probabilities > 0]
    effective_rank = (
        float((-nonzero * nonzero.log()).sum().exp()) if len(nonzero) else 0.0
    )
    return {
        "sample_count": len(features),
        "sampled_features": features.shape[1],
        "centered_effective_rank": effective_rank,
        "centered_stable_rank": float(
            singular.square().sum() / singular[0].square().clamp_min(1e-30)
        ),
        "maximum_centered_rank": min(len(features) - 1, features.shape[1]),
        "near_zero_activation_fraction_abs_lt_1e_3": float(
            (features.abs() < 1e-3).double().mean()
        ),
        "low_variance_feature_fraction_var_lt_1e_6": float(
            (features.var(dim=0, unbiased=False) < 1e-6).double().mean()
        ),
        "silu_negative_saturation_proxy_gate_lt_minus_5": float(
            (gates < -5).double().mean()
        ),
        "gate_abs_gt_10_fraction": float((gates.abs() > 10).double().mean()),
        "activation_rms": float(features.square().mean().sqrt()),
    }


@torch.no_grad()
def activation_probe(model, task, prepared, cfg):
    modules = dict(model.named_modules())
    downs = [name for name, _ in adapter_layers(model) if name.endswith(".down_proj")]
    if not downs:
        raise RuntimeError("DIAGNOSTIC_TEXT_MLP_NOT_FOUND")
    down_name = downs[-1]
    gate_name = down_name.removesuffix("down_proj") + "gate_proj"
    feature_values, gate_values = [], []

    def sample(tensor):
        values = tensor[:, -1, :].float()
        indices = torch.linspace(
            0,
            values.shape[-1] - 1,
            min(cfg.diagnostic_features, values.shape[-1]),
            device=values.device,
        ).long()
        return values[:, indices].cpu()

    def down_hook(module, args):
        feature_values.append(sample(args[0]))

    def gate_hook(module, args, output):
        gate_values.append(sample(output))

    hooks = [
        modules[down_name].register_forward_pre_hook(down_hook),
        modules[gate_name].register_forward_hook(gate_hook),
    ]
    model.eval()
    rows = sorted(
        task.rows["train"],
        key=lambda row: derived_seed(cfg.seed, "diagnostic", row["source_index"]),
    )[: cfg.diagnostic_examples]
    try:
        for start in range(0, len(rows), cfg.eval_batch_size):
            inputs, _ = collate(
                rows[start : start + cfg.eval_batch_size],
                prepared.pad_token_id,
                cfg.device,
            )
            class_logits(model, inputs, prepared.code_token_ids)
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "split": "train",
        "source_indices": [row["source_index"] for row in rows],
        "module": down_name,
        "gate_module": gate_name,
        "input_tokens": sum(len(row["input_ids"]) for row in rows),
        "interpretation": "Last-token, feature-subsampled SwiGLU diagnostics; saturation proxies are not proof of plasticity loss",
        **activation_statistics(torch.cat(feature_values), torch.cat(gate_values)),
    }


def acquisition_summary(curve, threshold):
    last_step = curve[-1]["updates"]
    summary = {}
    for metric in ("accuracy", "task_local_accuracy", "loss", "task_local_loss"):
        area = sum(
            (right["updates"] - left["updates"]) * (left[metric] + right[metric]) / 2
            for left, right in pairwise(curve)
        )
        summary[f"{metric}_auc_per_update"] = area / last_step if last_step else None
        summary[f"{metric}_change"] = curve[-1][metric] - curve[0][metric]
    for metric in ("accuracy", "task_local_accuracy"):
        attained = next((point for point in curve if point[metric] >= threshold), None)
        summary[f"{metric}_threshold"] = {
            "threshold": threshold,
            "first_observed_update": attained["updates"] if attained else None,
            "first_observed_training_input_tokens": attained["training_input_tokens"]
            if attained
            else None,
            "right_censored": attained is None,
        }
    return summary
