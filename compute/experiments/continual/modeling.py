import hashlib
import importlib.metadata
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from config import RunConfig
from data import Example, require_gradient_examples
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)


@dataclass(frozen=True)
class EncodedExample:
    example: Example
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inspect_model_provenance(config: RunConfig):
    path = Path(config.model_path).resolve(strict=True)
    weights = sorted(path.glob("*.safetensors"))
    if not weights or not (path / "config.json").is_file():
        raise ValueError(f"[model-provenance] incomplete local model at {path}")
    files = sorted(
        {
            *weights,
            *path.glob("*.json"),
            *path.glob("*.jinja"),
            *path.glob("*.txt"),
            *path.glob("*.model"),
        }
    )
    records = [
        {"name": file.name, "bytes": file.stat().st_size, "sha256": file_sha256(file)}
        for file in files
        if file.is_file()
    ]
    metadata_revisions = {}
    metadata_root = path / ".cache/huggingface/download"
    for file in files:
        metadata = metadata_root / f"{file.name}.metadata"
        if metadata.is_file():
            metadata_revisions[file.name] = metadata.read_text().splitlines()[0]
    if config.model_source == "huggingface":
        mismatches = {
            name: revision
            for name, revision in metadata_revisions.items()
            if revision != config.revision
        }
        if mismatches:
            raise ValueError(
                f"[model-provenance] cached revisions differ from requested commit: {mismatches}"
            )
        if not metadata_revisions and path.name != config.revision:
            raise ValueError(
                "[model-provenance] require a pinned snapshot path or hf download commit metadata"
            )
    return {
        "model_id": config.model_id,
        "requested_revision": config.revision,
        "source": config.model_source,
        "local_path": str(path),
        "hf_source": f"https://huggingface.co/{config.model_id}/tree/{config.revision}"
        if config.model_source == "huggingface"
        else None,
        "revision_evidence": "hf_download_metadata"
        if metadata_revisions
        else "snapshot_directory_name",
        "hf_download_revisions": metadata_revisions,
        "files": records,
        "weight_bytes": sum(file.stat().st_size for file in weights),
        "content_verification": "full local SHA-256; upstream association from cache metadata/path",
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(config: RunConfig):
    start = time.perf_counter()
    device = torch.device(config.runtime.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "[model-load] expected one GPU exposed as cuda:0 by ROCR_VISIBLE_DEVICES"
            )
        torch.cuda.set_device(device)
        if config.runtime.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("[model-load] bf16 is unavailable on the isolated GPU")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.runtime.dtype]
    hf_config = AutoConfig.from_pretrained(
        config.model_path, local_files_only=True, trust_remote_code=False
    )
    if hf_config.model_type == "qwen3_5":
        factory = Qwen3_5ForConditionalGeneration
        text_config = hf_config.text_config
        layer_pattern = (
            r"model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)"
        )
    elif hf_config.model_type in {"qwen3", "qwen2"}:
        factory = AutoModelForCausalLM
        text_config = hf_config
        layer_pattern = r"model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)"
    else:
        raise ValueError(
            f"[model-load] unsupported explicit model_type={hf_config.model_type}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("[tokenizer] require an EOS token and a model chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = factory.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": str(device)},
        attn_implementation=config.runtime.attention,
    )
    model.config.use_cache = False
    if hf_config.model_type == "qwen3_5":
        model.config.text_config.use_cache = False
    if config.lora.last_n_layers > text_config.num_hidden_layers:
        raise ValueError("[lora] last_n_layers exceeds the number of text layers")
    first_layer = text_config.num_hidden_layers - config.lora.last_n_layers
    targets = []
    for name, module in model.named_modules():
        match = re.fullmatch(layer_pattern, name)
        if match and int(match[1]) >= first_layer and match[3] in config.lora.modules:
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(
                    f"[lora] expected Linear at {name}, got {type(module).__name__}"
                )
            targets.append(name)
    if not targets:
        raise ValueError("[lora] no text attention/MLP targets matched")
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=0.0,
            target_modules=targets,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    trainable = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if ".lora_" not in name or any(
                part in name for part in ("visual", "embed_tokens", "lm_head")
            ):
                raise RuntimeError(f"[lora] unexpected trainable parameter {name}")
            parameter.data = parameter.data.float()
            trainable.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                    "dtype": str(parameter.dtype),
                }
            )
        if parameter.device != device:
            raise RuntimeError(
                f"[model-load] parameter is off device: {name} on {parameter.device}"
            )
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    stochastic = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.p
    ]
    if stochastic or getattr(text_config, "attention_dropout", 0.0):
        raise ValueError(
            f"[model-load] matched deterministic gradient diagnostics require zero dropout: {stochastic}"
        )
    synchronize(device)
    info = {
        "model_class": type(model.get_base_model()).__name__,
        "model_type": hf_config.model_type,
        "text_config": text_config.to_dict(),
        "device": str(device),
        "base_dtype": str(dtype),
        "lora_target_modules": targets,
        "trainable_parameters": trainable,
        "trainable_numel": sum(entry["numel"] for entry in trainable),
        "total_numel": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": time.perf_counter() - start,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
        "python": platform.python_version(),
        "rocm": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "gradient_checkpointing": config.runtime.gradient_checkpointing,
        "deterministic_algorithms": config.runtime.deterministic_algorithms,
    }
    return model, tokenizer, info


def encode_examples(examples, tokenizer, max_length, max_new_tokens):
    encoded = {}
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        target = tokenizer.encode(example.answer, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if not prompt or not target or len(prompt) + len(target) > max_length:
            raise ValueError(
                f"[tokenizer] refused truncation of {example.example_id}: prompt={len(prompt)}, target={len(target)}"
            )
        if len(target) > max_new_tokens:
            raise ValueError(
                f"[tokenizer] generation budget cannot emit gold answer and EOS: {example.example_id}"
            )
        encoded[example.example_id] = EncodedExample(
            example, tuple(prompt), tuple(target)
        )
    return encoded


def collate(encoded, pad_id, device, generation=False):
    lengths = [
        len(item.prompt_ids) + (0 if generation else len(item.target_ids))
        for item in encoded
    ]
    width = max(lengths)
    input_ids = torch.full(
        (len(encoded), width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (item, length) in enumerate(zip(encoded, lengths)):
        offset = width - length if generation else 0
        ids = item.prompt_ids if generation else item.prompt_ids + item.target_ids
        input_ids[row, offset : offset + length] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        attention_mask[row, offset : offset + length] = 1
        if not generation:
            labels[row, len(item.prompt_ids) : length] = torch.tensor(
                item.target_ids, dtype=torch.long, device=device
            )
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    if not generation:
        batch["labels"] = labels
    return batch


def batch_accounting(encoded, width):
    return {
        "examples": len(encoded),
        "input_tokens": sum(
            len(item.prompt_ids) + len(item.target_ids) for item in encoded
        ),
        "supervised_tokens": sum(len(item.target_ids) for item in encoded),
        "padded_tokens": len(encoded) * width,
    }


def training_gradient(
    model, encoded, tokenizer, parameters, device, current_task, role
):
    require_gradient_examples([item.example for item in encoded], current_task, role)
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError(
            "[gradient] training gradient requires train mode and enabled autograd"
        )
    batch = collate(encoded, tokenizer.pad_token_id, device)
    loss = model(**batch, use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError(
            f"[loss] non-finite {role} loss for task {current_task}"
        )
    gradient = torch.autograd.grad(loss, parameters, allow_unused=False)
    flat = torch.cat([part.detach().float().reshape(-1) for part in gradient])
    return flat, loss.item(), batch_accounting(encoded, batch["input_ids"].shape[1])
