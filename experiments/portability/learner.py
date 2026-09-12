from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from portallib import PortalBase, PortalEvaluator, PortalModel, collate_gold_batch
from portallib.evaluation import PortalInjector
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import Example, digest, write_json

INITIAL_TASK = "rte"
MAX_PARAMETER_MISMATCH = 0.005
NUMERICAL_GATE = {
    "minimum_choice_agreement": 31 / 32,
    "maximum_accuracy_gap": 1 / 32,
    "maximum_choice_score_gap": 0.02,
    "maximum_probe_abs_gap": 0.125,
    "maximum_probe_relative_l2": 0.01,
}


def tensor_hash(tensors: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        hasher.update(name.encode())
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        hasher.update(memoryview(raw))
    return hasher.hexdigest()


def state_hash(value: object) -> str:
    hasher = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            hasher.update(b"tensor:")
            hasher.update(tensor_hash({"value": item}).encode())
        elif isinstance(item, dict):
            hasher.update(b"dict:")
            for key in sorted(
                item, key=lambda value: (type(value).__name__, str(value))
            ):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            hasher.update(type(item).__name__.encode() + b":")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            hasher.update(type(item).__name__.encode() + b":")
            hasher.update(json.dumps(item, allow_nan=False).encode())
        else:
            raise TypeError(f"UNSUPPORTED_STATE_VALUE: {type(item).__name__}")
        hasher.update(b"\0")

    visit(value)
    return hasher.hexdigest()


def shared_hash(portal: PortalModel) -> str:
    return tensor_hash(
        {
            key: value
            for key, value in portal.state_dict().items()
            if not key.startswith("alignment.")
        }
    )


def model_source(spec: dict) -> str:
    local = spec["local_path"]
    if not isinstance(local, str) or not local:
        raise ValueError("LOCAL_MODEL_SNAPSHOT_REQUIRED")
    path = Path(local)
    if path.name != spec["revision"] or not (path / "config.json").is_file():
        raise ValueError(f"INVALID_PINNED_SNAPSHOT: {path}")
    index = path / "model.safetensors.index.json"
    if not index.is_file() and not (path / "model.safetensors").is_file():
        raise ValueError(f"MISSING_SNAPSHOT_WEIGHTS: {path}")
    return local


def load_portal(spec: dict) -> PortalModel:
    portal = PortalModel.from_pretrained(
        model_source(spec),
        revision=spec["revision"],
        cache_dir=spec["cache_dir"],
        local_files_only=True,
        dtype=torch.float32,
    )
    return portal.requires_grad_(False)


def load_base(spec: dict, device: str = "cuda:0") -> PortalBase:
    path = model_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        revision=spec["revision"],
        cache_dir=spec["cache_dir"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        revision=spec["revision"],
        cache_dir=spec["cache_dir"],
        dtype=torch.float32,
        device_map={"": device},
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,
    )
    base = PortalBase(spec["repo_id"], model, tokenizer, revision=spec["revision"])
    base.freeze(gradient_checkpointing=False)
    for name, value in frozen_base_tensors(model).items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"BASE_STORAGE_NOT_FP32: {name}/{value.dtype}")
    model.eval()
    return base


def extend_portal(portal: PortalModel, tasks: tuple[str, ...]) -> PortalModel:
    if len(tasks) != 3 or len(set(tasks)) != 3:
        raise ValueError("THREE_DISTINCT_SEQUENCE_TASKS_REQUIRED")
    if set(tasks) & set(portal.config.tasks):
        raise ValueError("SEQUENCE_TASK_ALREADY_EXISTS")
    z = portal.task_latents[portal.config.tasks.index(INITIAL_TASK)].detach()
    latents = torch.cat(
        [portal.task_latents.detach(), z.unsqueeze(0).repeat(len(tasks), 1)], dim=0
    )
    config = replace(portal.config, tasks=portal.config.tasks + tasks)
    extended = PortalModel(config, latents)
    extended.core.load_state_dict(portal.core.state_dict(), strict=True)
    extended.alignment.load_state_dict(portal.alignment.state_dict(), strict=True)
    return extended.requires_grad_(False)


def transplant(shared: PortalModel, target: PortalModel) -> PortalModel:
    if shared.config.architecture_kwargs() != target.config.architecture_kwargs():
        raise ValueError("INCOMPATIBLE_CANONICAL_ARCHITECTURE")
    if shared.config.tasks[: len(target.config.tasks)] != target.config.tasks:
        raise ValueError("TASK_TABLE_PREFIX_MISMATCH")
    config = replace(target.config, tasks=shared.config.tasks)
    result = PortalModel(config, shared.task_latents.detach())
    result.core.load_state_dict(shared.core.state_dict(), strict=True)
    result.alignment.load_state_dict(target.alignment.state_dict(), strict=True)
    return result.requires_grad_(False)


def save_native(portal: PortalModel, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", portal.config.to_dict())
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in portal.state_dict().items()
        },
        output / "model.safetensors",
        metadata={"format": "portallib", "format_version": "1"},
    )


def parameter_plan(portal: PortalModel) -> dict:
    groups = defaultdict(list)
    for target, path in portal.config.resolved_targets():
        groups[target.module_name].append(
            (path, target.in_features + target.out_features)
        )
    core_count = sum(parameter.numel() for parameter in portal.core.parameters())
    native_count = core_count + portal.config.d_z
    rank_unit = sum(size for rows in groups.values() for _, size in rows)
    lower_rank = native_count // rank_unit
    if lower_rank < portal.config.rank:
        raise ValueError("INHERITED_RANK_EXCEEDS_MATCHED_BUDGET")
    names = sorted(groups)
    choices = []
    for ranks in itertools.product((lower_rank, lower_rank + 1), repeat=len(names)):
        count = sum(
            rank * sum(size for _, size in groups[name])
            for name, rank in zip(names, ranks, strict=True)
        )
        choices.append((abs(count - native_count), ranks, count))
    error, ranks, lora_count = min(choices)
    relative = error / native_count
    if relative > MAX_PARAMETER_MISMATCH:
        raise ValueError(
            f"PARAMETER_MATCH_FAILED: native={native_count}, lora={lora_count}, "
            f"relative_error={relative:.8f}, limit={MAX_PARAMETER_MISMATCH}"
        )
    scale = portal.config.alpha / portal.config.rank
    rank_pattern = {}
    alpha_pattern = {}
    projection_ranks = dict(zip(names, ranks, strict=True))
    for name in names:
        rank = projection_ranks[name]
        alpha = rank * scale
        if not alpha.is_integer():
            raise ValueError("NONINTEGER_MATCHED_LORA_ALPHA")
        for path, _ in groups[name]:
            rank_pattern[path] = rank
            alpha_pattern[path] = int(alpha)
    return {
        "native_core_parameters": core_count,
        "native_current_vector_parameters": portal.config.d_z,
        "native_active_parameters": native_count,
        "lora_parameters": lora_count,
        "absolute_difference": error,
        "relative_difference": relative,
        "maximum_relative_difference": MAX_PARAMETER_MISMATCH,
        "published_rank": portal.config.rank,
        "published_alpha": portal.config.alpha,
        "scaling": scale,
        "projection_ranks": projection_ranks,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "allocation": "Uniform rank within each projection type; closest adjacent-rank allocation.",
    }


def make_persistent_lora(
    base: PortalBase, portal: PortalModel, export_directory: Path
) -> tuple[PeftModel, dict]:
    plan = parameter_plan(portal)
    with fp32_forward(base):
        portal.export_peft(INITIAL_TASK, export_directory)
    inherited = load_file(export_directory / "adapter_model.safetensors")
    rank = min(plan["rank_pattern"].values())
    model = get_peft_model(
        base.model,
        LoraConfig(
            r=rank,
            lora_alpha=int(rank * plan["scaling"]),
            rank_pattern=plan["rank_pattern"],
            alpha_pattern=plan["alpha_pattern"],
            target_modules=list(plan["rank_pattern"]),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        ),
        autocast_adapter_dtype=True,
    )
    template = get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(template) != set(inherited):
        raise ValueError("INHERITED_LORA_TARGET_MISMATCH")
    expanded = {}
    source_rank = portal.config.rank
    for key, value in template.items():
        if value.dtype != torch.float32:
            raise ValueError(f"LORA_TRAINABLE_NOT_FP32: {key}/{value.dtype}")
        widened = value.detach().clone()
        original = inherited[key].to(device=value.device, dtype=value.dtype)
        if key.endswith(".lora_A.weight"):
            widened[:source_rank].copy_(original)
        elif key.endswith(".lora_B.weight"):
            widened.zero_()
            widened[:, :source_rank].copy_(original)
        else:
            raise ValueError(f"UNEXPECTED_LORA_PARAMETER: {key}")
        expanded[key] = widened
    incompatible = set_peft_model_state_dict(model, expanded)
    missing_adapters = [key for key in incompatible.missing_keys if "lora_" in key]
    if missing_adapters or incompatible.unexpected_keys:
        raise ValueError(
            f"LORA_INITIALIZATION_LOAD_FAILED: {missing_adapters}, "
            f"{incompatible.unexpected_keys}"
        )
    actual = get_peft_model_state_dict(model, save_embedding_layers=False)
    checks = {}
    projection_probe_errors = {}
    rng = torch.Generator(device="cpu").manual_seed(9181)
    for target, path in portal.config.resolved_targets():
        prefix = f"base_model.model.{path}"
        a = actual[f"{prefix}.lora_A.weight"]
        b = actual[f"{prefix}.lora_B.weight"]
        old_a = inherited[f"{prefix}.lora_A.weight"].to(a)
        old_b = inherited[f"{prefix}.lora_B.weight"].to(b)
        module = model.get_base_model().get_submodule(path)
        checks[path] = {
            "inherited_a_bitwise_equal": torch.equal(a[:source_rank], old_a),
            "inherited_b_bitwise_equal": torch.equal(b[:, :source_rank], old_b),
            "extra_b_exact_zero": not bool(torch.count_nonzero(b[:, source_rank:])),
            "extra_a_rows_nonzero": bool(
                torch.all(torch.count_nonzero(a[source_rank:], dim=1) > 0)
            ),
            "scaling_equal": module.scaling["default"] == plan["scaling"],
            "rank_matches_plan": a.shape[0] == plan["rank_pattern"][path],
        }
        x = torch.randn(2, target.in_features, generator=rng).to(a)
        with torch.no_grad(), fp32_forward(base):
            left = (
                torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
                * plan["scaling"]
            )
            right = (
                torch.nn.functional.linear(torch.nn.functional.linear(x, old_a), old_b)
                * plan["scaling"]
            )
        projection_probe_errors[path] = float((left - right).abs().max())
    if not all(value for row in checks.values() for value in row.values()):
        raise ValueError(f"INITIAL_ADAPTER_EMBEDDING_FAILED: {checks}")
    trainable = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    count = sum(value.numel() for value in trainable.values())
    if count != plan["lora_parameters"] or any(
        "lora_" not in name for name in trainable
    ):
        raise ValueError(
            f"LORA_PARAMETER_COUNT_MISMATCH: {count}/{plan['lora_parameters']}"
        )
    if any(value.dtype != torch.float32 for value in trainable.values()):
        raise ValueError("LORA_TRAINABLE_DTYPE_CHANGED")
    receipt = {
        "parameter_match": plan,
        "factor_checks": checks,
        "algebraic_effective_delta_equal": True,
        "zero_initial_effect_of_extra_ranks": True,
        "projection_probe_max_abs": projection_probe_errors,
        "projection_probe_overall_max_abs": max(projection_probe_errors.values()),
        "inherited_adapter_sha256": tensor_hash(inherited),
        "initial_lora_sha256": tensor_hash(actual),
        "trainable_parameters": count,
        "trainable_dtype": "float32",
        "numerical_model_behavior": "Measured separately with actual model logits; factor embedding does not imply bitwise model equivalence.",
    }
    return model, receipt


def frozen_base_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    underlying = model.get_base_model() if isinstance(model, PeftModel) else model
    values = {}
    for kind, iterator in (
        ("parameter", underlying.named_parameters()),
        ("buffer", underlying.named_buffers()),
    ):
        for name, value in iterator:
            if "lora_" in name:
                continue
            canonical = f"{kind}:{name.replace('.base_layer.', '.')}"
            if canonical in values:
                raise ValueError(f"FROZEN_STATE_NAME_COLLISION: {canonical}")
            if value.requires_grad:
                raise ValueError(f"BASE_PARAMETER_UNFROZEN: {canonical}")
            values[canonical] = value
    return values


def tensor_storage(tensors: dict[str, torch.Tensor]) -> dict:
    return {
        "tensors": len(tensors),
        "elements": sum(value.numel() for value in tensors.values()),
        "tensor_bytes": sum(
            value.numel() * value.element_size() for value in tensors.values()
        ),
        "floating_dtypes": sorted(
            {
                str(value.dtype)
                for value in tensors.values()
                if value.is_floating_point()
            }
        ),
    }


def fp32_forward(base: PortalBase):
    return torch.autocast(device_type=base.device.type, enabled=False)


@dataclass
class LossBatch:
    loss: torch.Tensor
    examples: int
    supervised_tokens: int
    input_tokens: int
    padded_tokens: int

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.examples,
            "supervised_tokens": self.supervised_tokens,
            "input_tokens": self.input_tokens,
            "padded_tokens": self.padded_tokens,
        }


def supervised_loss(
    base: PortalBase, rows: list[Example], max_prompt: int
) -> LossBatch:
    input_ids, attention_mask, labels = collate_gold_batch(
        base.tokenizer,
        [row.choice() for row in rows],
        max_prompt=max_prompt,
        device=base.device,
    )
    with fp32_forward(base):
        loss = base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
    if not torch.isfinite(loss):
        raise FloatingPointError("NONFINITE_TRAINING_LOSS")
    return LossBatch(
        loss,
        len(rows),
        int((labels != -100).sum()),
        int(attention_mask.sum()),
        input_ids.numel(),
    )


class SequenceLearner:
    def __init__(
        self,
        base: PortalBase,
        portal: PortalModel | None,
        tasks: tuple[str, ...],
        stage: int,
        recipe: dict,
    ):
        if stage not in range(len(tasks)):
            raise ValueError(f"INVALID_SEQUENCE_STAGE: {stage}")
        self.base = base
        self.portal = portal
        self.tasks = tasks
        self.stage = stage
        self.recipe = recipe
        self.vectors = {}
        self.named_parameters = {}
        self.injector = None
        if portal is not None:
            if tuple(portal.config.tasks[-len(tasks) :]) != tasks:
                raise ValueError("SEQUENCE_VECTOR_ORDER_MISMATCH")
            portal.requires_grad_(False)
            portal.core.requires_grad_(True)
            self.named_parameters.update(
                (f"core.{name}", value)
                for name, value in portal.core.named_parameters()
            )
            groups = [
                {"params": list(portal.core.parameters()), "lr": recipe["core_lr"]}
            ]
            for index, task in enumerate(tasks):
                vector = torch.nn.Parameter(
                    portal.task_latents[portal.config.tasks.index(task)]
                    .detach()
                    .clone(),
                    requires_grad=index == stage,
                )
                self.vectors[task] = vector
                self.named_parameters[f"vector.{task}"] = vector
                groups.append({"params": [vector], "lr": recipe["latent_lr"]})
            self.injector = PortalInjector(base.model, portal.config)
        else:
            if not isinstance(base.model, PeftModel):
                raise ValueError("PERSISTENT_PEFT_MODEL_REQUIRED")
            self.named_parameters = {
                name: value
                for name, value in base.model.named_parameters()
                if value.requires_grad
            }
            if not self.named_parameters or any(
                "lora_" not in name for name in self.named_parameters
            ):
                raise ValueError("LORA_ONLY_TRAINABLE_STATE_REQUIRED")
            groups = [
                {
                    "params": list(self.named_parameters.values()),
                    "lr": recipe["lora_lr"],
                }
            ]
        if any(
            value.dtype != torch.float32 for value in self.named_parameters.values()
        ):
            raise ValueError("TRAINABLE_STATE_MUST_BE_FP32")
        self.optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        frozen_base_tensors(base.model)

    @property
    def active(self) -> dict[str, torch.nn.Parameter]:
        return {
            name: value
            for name, value in self.named_parameters.items()
            if value.requires_grad
        }

    def inactive_optimizer_hashes(self) -> dict[str, str]:
        return {
            name: state_hash(self.optimizer.state.get(value, {}))
            for name, value in self.named_parameters.items()
            if not value.requires_grad
        }

    def sync_vectors(self) -> None:
        if self.portal is not None:
            with torch.no_grad():
                for task, vector in self.vectors.items():
                    self.portal.task_latents[
                        self.portal.config.tasks.index(task)
                    ].copy_(vector)

    def close(self) -> None:
        if self.injector is not None:
            self.injector.close()

    def advance_stage(self) -> dict:
        if self.stage + 1 >= len(self.tasks):
            raise ValueError("SEQUENCE_ALREADY_FINISHED")
        before = self.optimizer_receipt()
        self.optimizer.zero_grad(set_to_none=True)
        self.sync_vectors()
        self.stage += 1
        for index, task in enumerate(self.tasks):
            if self.portal is not None:
                vector = self.vectors[task]
                vector.requires_grad_(index == self.stage)
                if index == self.stage and self.optimizer.state.get(vector):
                    raise ValueError("FUTURE_VECTOR_HAS_OPTIMIZER_HISTORY")
        after = self.optimizer_receipt()
        if before["state_sha256"] != after["state_sha256"]:
            raise ValueError("STAGE_SWITCH_CHANGED_OPTIMIZER_STATE")
        return {"before": before, "after": after, "optimizer_state_unchanged": True}

    def loss(self, rows: list[Example]) -> LossBatch:
        if not rows:
            raise ValueError("EMPTY_LEARNING_BATCH")
        groups = defaultdict(list)
        for row in rows:
            if row.task not in self.tasks[: self.stage + 1]:
                raise ValueError(f"FUTURE_TASK_TRAINING_ACCESS: {row.task}")
            groups[row.task].append(row)
        batches = []
        for task, subset in groups.items():
            with fp32_forward(self.base):
                activation = (
                    self.injector.activate(self.portal(self.vectors[task]))
                    if self.portal is not None
                    else nullcontext()
                )
            with activation:
                batches.append(
                    supervised_loss(self.base, subset, self.recipe["max_prompt"])
                )
        tokens = sum(batch.supervised_tokens for batch in batches)
        combined = (
            sum(
                (batch.loss * batch.supervised_tokens for batch in batches),
                start=torch.zeros((), device=self.base.device),
            )
            / tokens
        )
        return LossBatch(
            combined,
            sum(batch.examples for batch in batches),
            tokens,
            sum(batch.input_tokens for batch in batches),
            sum(batch.padded_tokens for batch in batches),
        )

    def step(
        self, current: list[Example], reference: list[Example], replay_weight: float
    ) -> dict:
        if any(row.task != self.tasks[self.stage] for row in current):
            raise ValueError("CURRENT_BATCH_TASK_MISMATCH")
        if any(row.task not in self.tasks[: self.stage] for row in reference):
            raise ValueError("REFERENCE_MUST_CONTAIN_ONLY_PREVIOUS_TASKS")
        if replay_weight not in (0.0, 0.25):
            raise ValueError("UNDECLARED_REPLAY_WEIGHT")
        self.optimizer.zero_grad(set_to_none=True)
        self.base.model.train()
        named = self.active
        parameters = tuple(named.values())
        current_batch = self.loss(current)
        current_gradients = torch.autograd.grad(
            current_batch.loss, parameters, allow_unused=True
        )
        reference_batch = None
        reference_gradients = (None,) * len(parameters)
        if reference:
            devices = (
                [self.base.device.index or 0] if self.base.device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=devices):
                reference_batch = self.loss(reference)
                reference_gradients = torch.autograd.grad(
                    reference_batch.loss, parameters, allow_unused=True
                )
        gradient_seen = {}
        extra_rank_gradients = {}
        reference_norm_squared = 0.0
        for (name, parameter), current_gradient, reference_gradient in zip(
            named.items(), current_gradients, reference_gradients, strict=True
        ):
            for source, gradient in (
                ("current", current_gradient),
                ("reference", reference_gradient),
            ):
                if gradient is not None and not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        f"NONFINITE_{source.upper()}_GRADIENT: {name}"
                    )
            gradient_seen[name] = current_gradient is not None and bool(
                torch.count_nonzero(current_gradient)
            )
            if reference_gradient is not None:
                reference_norm_squared += float(
                    reference_gradient.float().square().sum()
                )
            gradient = current_gradient
            if replay_weight and reference_gradient is not None:
                contribution = reference_gradient * replay_weight
                gradient = contribution if gradient is None else gradient + contribution
            parameter.grad = gradient
            if self.portal is None and gradient is not None:
                rank = self.recipe["inherited_rank"]
                extra = gradient[rank:] if ".lora_A." in name else gradient[:, rank:]
                extra_rank_gradients[name] = int(torch.count_nonzero(extra))
        inactive = {
            name: parameter
            for name, parameter in self.named_parameters.items()
            if not parameter.requires_grad
        }
        if any(parameter.grad is not None for parameter in inactive.values()):
            raise ValueError("FROZEN_VECTOR_RECEIVED_GRADIENT")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, self.recipe["grad_clip"], error_if_nonfinite=True
            )
        )
        self.optimizer.step()
        self.sync_vectors()
        empty = {
            "examples": 0,
            "supervised_tokens": 0,
            "input_tokens": 0,
            "padded_tokens": 0,
        }
        reference_counts = (
            reference_batch.counts() if reference_batch is not None else empty
        )
        return {
            "current_loss": float(current_batch.loss.detach()),
            "reference_loss": float(reference_batch.loss.detach())
            if reference_batch is not None
            else None,
            "gradient_norm": gradient_norm,
            "current_gradient_seen": gradient_seen,
            "extra_rank_gradient_nonzero_elements": extra_rank_gradients,
            "reference_gradient_norm": reference_norm_squared**0.5,
            "current_computed": current_batch.counts(),
            "reference_computed": reference_counts,
            "reference_effective": reference_counts if replay_weight else empty,
            "current_forward_backward_groups": len({row.task for row in current}),
            "reference_forward_backward_groups": len({row.task for row in reference}),
            "reference_backward_calls": int(reference_batch is not None),
            "optimizer_updates": 1,
        }

    def optimizer_receipt(self) -> dict:
        tensors = {
            f"{name}.{key}": state
            for name, parameter in self.named_parameters.items()
            for key, state in self.optimizer.state.get(parameter, {}).items()
            if isinstance(state, torch.Tensor)
        }
        return {
            "parameter_names": list(self.named_parameters),
            "state_sha256": state_hash(self.optimizer.state_dict()),
            "steps": {
                name: float(self.optimizer.state.get(value, {}).get("step", 0))
                for name, value in self.named_parameters.items()
            },
            "active_parameters": sum(value.numel() for value in self.active.values()),
            "inactive_vector_optimizer_sha256": self.inactive_optimizer_hashes(),
            "storage": tensor_storage(tensors),
            "parameter_groups": [
                {
                    "names": [
                        name
                        for name, value in self.named_parameters.items()
                        if any(value is member for member in group["params"])
                    ],
                    "lr": group["lr"],
                    "active_elements": sum(
                        value.numel()
                        for value in group["params"]
                        if value.requires_grad
                    ),
                    "retained_elements": sum(
                        value.numel() for value in group["params"]
                    ),
                }
                for group in self.optimizer.param_groups
            ],
        }

    def save_optimizer(self, path: Path) -> dict:
        receipt = self.optimizer_receipt()
        torch.save(
            {
                "parameter_names": list(self.named_parameters),
                "optimizer": self.optimizer.state_dict(),
                "stage": self.stage,
                "receipt": receipt,
            },
            path,
        )
        return receipt

    def load_optimizer(self, path: Path) -> dict:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["parameter_names"] != list(self.named_parameters):
            raise ValueError("OPTIMIZER_PARAMETER_ORDER_MISMATCH")
        if saved["stage"] not in (self.stage, self.stage - 1):
            raise ValueError("OPTIMIZER_STAGE_MISMATCH")
        self.optimizer.load_state_dict(saved["optimizer"])
        if state_hash(self.optimizer.state_dict()) != saved["receipt"]["state_sha256"]:
            raise ValueError("OPTIMIZER_ROUNDTRIP_MISMATCH")
        if self.portal is not None and saved["stage"] < self.stage:
            fresh = self.vectors[self.tasks[self.stage]]
            if self.optimizer.state.get(fresh):
                raise ValueError("FUTURE_VECTOR_HAS_OPTIMIZER_HISTORY")
        return saved["receipt"]


def score_rows(
    base: PortalBase, rows: list[Example], max_prompt: int, batch_size: int
) -> dict:
    base.model.eval()
    with torch.inference_mode(), fp32_forward(base):
        scores, gold_nll, gold_tokens = PortalEvaluator(
            max_prompt=max_prompt, batch_size=batch_size
        )._score_rows(base, [row.choice() for row in rows])
    results = []
    for row, values in zip(rows, scores, strict=True):
        if not torch.isfinite(torch.tensor(values)).all():
            raise FloatingPointError(f"NONFINITE_EVALUATION: {row.id}")
        prediction = max(range(len(values)), key=values.__getitem__)
        results.append(
            {
                "id": row.id,
                "task": row.task,
                "group": row.group,
                "gold": row.gold_idx,
                "prediction": prediction,
                "correct": int(prediction == row.gold_idx),
                "scores": values,
                "prompt_sha256": digest(row.prompt),
            }
        )
    return {"predictions": results, "gold_nll": gold_nll / gold_tokens}


def evaluate(
    base: PortalBase,
    rows: list[Example],
    max_prompt: int,
    batch_size: int,
    portal: PortalModel | None = None,
) -> dict:
    if portal is not None:
        portal.validate_base_model(base.model_id, base.revision)
    predictions = []
    metrics = {}
    injector = PortalInjector(base.model, portal.config) if portal is not None else None
    try:
        for task in sorted({row.task for row in rows}):
            subset = [row for row in rows if row.task == task]
            with fp32_forward(base):
                activation = (
                    injector.activate(portal.generate(task))
                    if injector is not None
                    else nullcontext()
                )
            with activation:
                scored = score_rows(base, subset, max_prompt, batch_size)
                results = scored["predictions"]
            predictions.extend(results)
            metrics[task] = {
                "accuracy": sum(row["correct"] for row in results) / len(results),
                "gold_nll": scored["gold_nll"],
                "examples": len(results),
                "truncated_prompts": sum(
                    len(
                        base.tokenizer(
                            row.prompt.rstrip(), add_special_tokens=True
                        ).input_ids
                    )
                    > max_prompt
                    for row in subset
                ),
                "gold_char_mean_logp": sum(
                    result["scores"][result["gold"]] for result in results
                )
                / len(results),
            }
    finally:
        if injector is not None:
            injector.close()
    return {"metrics": metrics, "predictions": predictions}


def probe_logits(
    base: PortalBase,
    rows: list[Example],
    portal: PortalModel | None,
    *,
    arithmetic: dict | None = None,
) -> dict[str, torch.Tensor]:
    base.model.eval()
    probes = {}
    injector = PortalInjector(base.model, portal.config) if portal is not None else None
    try:
        for row in rows:
            if row.id in probes:
                raise ValueError(f"DUPLICATE_PROBE_ID: {row.id}")
            inputs = base.tokenizer(row.prompt, return_tensors="pt").to(base.device)
            with fp32_forward(base):
                activation = (
                    injector.activate(portal.generate(row.task))
                    if injector is not None
                    else nullcontext()
                )
            with torch.inference_mode(), activation, fp32_forward(base):
                logits = base.model(**inputs, use_cache=False).logits[0, -1]
                if arithmetic is not None:
                    arithmetic[row.id] = {
                        "logits_dtype": str(logits.dtype),
                        "autocast_enabled": torch.is_autocast_enabled(base.device.type),
                    }
                probes[row.id] = logits.float().cpu().contiguous()
    finally:
        if injector is not None:
            injector.close()
    return probes


def compare_probes(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict:
    if actual.keys() != expected.keys():
        raise ValueError("PROBE_KEYS_MISMATCH")
    errors = {key: float((actual[key] - expected[key]).abs().max()) for key in actual}
    return {
        "bitwise_equal": all(torch.equal(actual[key], expected[key]) for key in actual),
        "close": all(
            torch.allclose(actual[key], expected[key], rtol=1e-5, atol=1e-4)
            for key in actual
        ),
        "max_abs": errors,
        "overall_max_abs": max(errors.values()),
    }


def initialization_parity(
    native: dict,
    lora: dict,
    native_probes: dict[str, torch.Tensor],
    lora_probes: dict[str, torch.Tensor],
) -> dict:
    left = {row["id"]: row for row in native["predictions"]}
    right = {row["id"]: row for row in lora["predictions"]}
    if not left or left.keys() != right.keys():
        raise ValueError("INITIALIZATION_PANEL_MISMATCH")
    tasks = {}
    for task in sorted({row["task"] for row in left.values()}):
        rows = [row for row in left.values() if row["task"] == task]
        gaps = []
        margins = []
        disagreements = []
        for row in rows:
            other = right[row["id"]]
            if any(
                row[key] != other[key]
                for key in ("task", "gold", "group", "prompt_sha256")
            ):
                raise ValueError("INITIALIZATION_IDENTITY_MISMATCH")
            gaps.extend(
                abs(a - b) for a, b in zip(row["scores"], other["scores"], strict=True)
            )
            ranked = sorted(row["scores"], reverse=True)
            margins.append(ranked[0] - ranked[1])
            if row["prediction"] != other["prediction"]:
                disagreements.append(row["id"])
        agreement = 1 - len(disagreements) / len(rows)
        accuracy_gap = abs(
            sum(row["correct"] - right[row["id"]]["correct"] for row in rows)
            / len(rows)
        )
        tasks[task] = {
            "examples": len(rows),
            "choice_agreement": agreement,
            "absolute_accuracy_gap": accuracy_gap,
            "max_choice_score_gap": max(gaps),
            "minimum_native_choice_margin": min(margins),
            "disagreement_ids": disagreements,
            "passed": agreement >= NUMERICAL_GATE["minimum_choice_agreement"]
            and accuracy_gap <= NUMERICAL_GATE["maximum_accuracy_gap"]
            and max(gaps) <= NUMERICAL_GATE["maximum_choice_score_gap"],
        }
    probe = compare_probes(lora_probes, native_probes)
    relative = {
        key: float(
            torch.linalg.vector_norm(lora_probes[key] - value)
            / torch.linalg.vector_norm(value).clamp_min(1e-12)
        )
        for key, value in native_probes.items()
    }
    probe["relative_l2"] = relative
    probe["overall_relative_l2"] = max(relative.values())
    passed = (
        all(row["passed"] for row in tasks.values())
        and probe["overall_max_abs"] <= NUMERICAL_GATE["maximum_probe_abs_gap"]
        and probe["overall_relative_l2"] <= NUMERICAL_GATE["maximum_probe_relative_l2"]
    )
    return {
        "passed": passed,
        "thresholds": NUMERICAL_GATE,
        "selection_data": "validation only",
        "tasks": tasks,
        "probe": probe,
        "arithmetic": "FP32 base storage, trainable state, native generation, and model forwards; autocast disabled for both methods.",
    }
