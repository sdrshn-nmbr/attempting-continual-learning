import copy
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration

from data import digest
from sandbox import grade_text, render

LOGGER = logging.getLogger("skill_transfer")
ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def event(path, kind, **values):
    record = {"event": kind, **values}
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    LOGGER.info("%s %s", kind, json.dumps(values, sort_keys=True, allow_nan=False))


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tree_record(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
    if isinstance(value, dict):
        return {str(key): tree_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree_record(item) for item in value]
    return value


def frozen_hash(model):
    return digest({name: tensor_hash(value) for name, value in model.named_parameters() if not value.requires_grad})


def file_identity(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def verify_snapshot(config, tokenizer_only=False):
    manifest_path = ROOT / config["snapshot_manifest"]
    payload = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(payload).hexdigest()
    if manifest_hash != config["snapshot_manifest_sha256"]:
        raise RuntimeError("SNAPSHOT_MANIFEST_CHANGED")
    manifest = json.loads(payload)
    if manifest["model_id"] != config["model_id"] or manifest["revision"] != config["model_revision"]:
        raise RuntimeError("SNAPSHOT_REVISION_MISMATCH")
    root = Path(config["model_path"])
    expected = {item["path"]: item for item in manifest["files"]}
    permitted = set(expected) | set(manifest["ignored_nonloading_documents"])
    for path in root.iterdir():
        if path.name == manifest["optional_metadata_directory"] and path.is_dir() and not path.is_symlink():
            continue
        if path.name not in permitted or not path.is_file():
            raise RuntimeError(f"SNAPSHOT_UNPINNED_ENTRY: {path.name}")
    checked = {}
    for name, item in expected.items():
        if tokenizer_only and name not in manifest["tokenizer_files"] and item["required"]:
            continue
        path = root / name
        if not path.exists() and not item["required"]:
            continue
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"SNAPSHOT_SIZE: {name}")
        identity = file_identity(path)
        LOGGER.info("SNAPSHOT_HASH path=%s bytes=%d", path, item["bytes"])
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != item["sha256"]:
            raise RuntimeError(f"SNAPSHOT_HASH: {name}")
        if file_identity(path) != identity:
            raise RuntimeError(f"SNAPSHOT_CHANGED_DURING_HASH: {name}")
        checked[name] = {"sha256": checksum.hexdigest(), "bytes": item["bytes"], "file_identity": identity}
    return {
        "manifest_sha256": manifest_hash,
        "revision": manifest["revision"],
        "model_path": str(root),
        "tokenizer_only": tokenizer_only,
        "files": checked,
    }


def check_snapshot_unchanged(proof):
    for name, record in proof["files"].items():
        if file_identity(Path(proof["model_path"]) / name) != record["file_identity"]:
            raise RuntimeError(f"SNAPSHOT_CHANGED_DURING_LOAD: {name}")


def require_rocm_runtime(config):
    if config["device"] != "cuda:0":
        raise RuntimeError("RUNTIME_DEVICE: use cuda:0 inside one isolated ROCm GPU")
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("REQUIRE_ONE_VISIBLE_ROCM_GPU")
    if torch.is_autocast_enabled():
        raise RuntimeError("AUTOCAST_FORBIDDEN: adapters and dense offsets use explicit FP32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "device": config["device"],
        "gpu_count": torch.cuda.device_count(),
        "rocm": torch.version.hip,
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "base_load_dtype": "bfloat16",
        "adapter_dtype": "float32",
        "autocast": False,
        "tf32": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def finite_tensor(value):
    return all(bool(torch.isfinite(chunk).all()) for chunk in value.detach().reshape(-1).split(4 * 1024 * 1024))


def check_model_precision(model, device):
    allowed_placements = {str(device), device}
    if device.type == "cuda":
        allowed_placements.add(device.index)
    placement = getattr(model, "hf_device_map", None)
    tensors = [
        (kind, name, value)
        for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
        for name, value in values
    ]
    hooks = {
        name: f"{type(hook).__module__}.{type(hook).__qualname__}: {hook!r}"
        for name, module in model.named_modules()
        if (hook := getattr(module, "_hf_hook", None)) is not None
    }
    placement_record = {
        "expected_device": str(device),
        "tensor_devices": dict(Counter(f"{kind}:{value.device}" for kind, _, value in tensors)),
        "hf_device_map_present": hasattr(model, "hf_device_map"),
        "hf_device_map_type": f"{type(placement).__module__}.{type(placement).__qualname__}",
        "hf_device_map_repr": repr(placement),
        "accelerate_hooks": hooks,
    }
    diagnostics = json.dumps(placement_record, sort_keys=True)
    LOGGER.info("MODEL_PLACEMENT_AUDIT %s", diagnostics)
    for kind, name, value in tensors:
        if value.device != device:
            raise RuntimeError(f"MODEL_PLACEMENT: {kind}:{name} is on {value.device}; {diagnostics}")
    if placement is not None:
        if not isinstance(placement, Mapping):
            raise RuntimeError(f"MODEL_DEVICE_MAP_TYPE: expected optional mapping; {diagnostics}")
        if any(
            not isinstance(value, (str, int, torch.device))
            or isinstance(value, bool)
            or value not in allowed_placements
            for value in placement.values()
        ):
            raise RuntimeError(f"MODEL_OFFLOAD: device-map entry differs from assigned device; {diagnostics}")
    if hooks:
        raise RuntimeError(f"MODEL_DISPATCH_HOOK: uniform placement requires no Accelerate hooks; {diagnostics}")
    counts = Counter()
    base_bf16 = 0
    adapter_parameters = {id(value) for layer in layers(model).values() for value in (layer.A, layer.B)}
    for kind, name, value in tensors:
        if value.is_floating_point() or value.is_complex():
            expected = (
                {torch.float32}
                if id(value) in adapter_parameters or name.endswith(".offset")
                else {torch.float32, torch.bfloat16}
            )
            if value.dtype not in expected:
                raise RuntimeError(f"MODEL_PRECISION: {kind}:{name} is {value.dtype}; {diagnostics}")
            if not finite_tensor(value):
                raise RuntimeError(f"MODEL_NONFINITE: {kind}:{name}; {diagnostics}")
            if kind == "parameter" and id(value) not in adapter_parameters and value.dtype == torch.bfloat16:
                base_bf16 += value.numel()
        counts[f"{kind}:{value.dtype}"] += value.numel()
    if not base_bf16:
        raise RuntimeError(f"MODEL_BASE_PRECISION: BF16 base weights are required; {diagnostics}")
    return {
        **placement_record,
        "all_tensors_on_assigned_device": True,
        "no_accelerate_hooks": True,
        "all_floating_tensors_finite": True,
        "base_load_dtype": "bfloat16",
        "native_fp32_components_allowed": True,
        "adapter_and_offset_dtype": "float32",
        "tensor_elements": dict(counts),
        "offload": False,
    }


class LowRankLinear(torch.nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.A = torch.nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = torch.nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device)
        )
        torch.nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.register_buffer("offset", None)

    def forward(self, value):
        result = self.base(value)
        delta = self.scaling * F.linear(F.linear(value.float(), self.A), self.B)
        if self.offset is not None:
            delta = delta + F.linear(value.float(), self.offset)
        return result + delta.to(result.dtype)


def layers(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, LowRankLinear)}


def install_adapters(model, config):
    model.requires_grad_(False)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and name.rsplit(".", 1)[-1] in config["target_suffixes"]
        and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise RuntimeError("ADAPTER_TARGETS: no Qwen3.5 text projection targets found")
    for name, module in targets:
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, LowRankLinear(module, config["rank"], config["lora_alpha"]))
    owned = {id(value) for layer in layers(model).values() for value in (layer.A, layer.B)}
    trainable = {id(value) for value in model.parameters() if value.requires_grad}
    if owned != trainable:
        raise RuntimeError("TRAINABLE_BOUNDARY: only rank-eight factors may be trainable")


def capture(model):
    return {
        name: {
            "A": layer.A.detach().cpu().clone(),
            "B": layer.B.detach().cpu().clone(),
            "offset": None if layer.offset is None else layer.offset.detach().cpu().clone(),
        }
        for name, layer in layers(model).items()
    }


def restore(model, state):
    if set(state) != set(layers(model)):
        raise RuntimeError("ADAPTER_KEYS")
    for name, values in state.items():
        for key, value in values.items():
            if value is not None and (value.dtype != torch.float32 or not finite_tensor(value)):
                raise RuntimeError(f"ADAPTER_NOT_FINITE_FP32: {name}:{key}")
    with torch.no_grad():
        for name, layer in layers(model).items():
            layer.A.copy_(state[name]["A"])
            layer.B.copy_(state[name]["B"])
            offset = state[name]["offset"]
            layer.offset = None if offset is None else offset.to(device=layer.A.device, dtype=torch.float32).clone()
    for parameter in model.parameters():
        parameter.grad = None
    if digest(tree_record(capture(model))) != digest(tree_record(state)):
        raise RuntimeError("ADAPTER_RESTORE_CHANGED")


def new_optimizer(model, config):
    return torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=config["learning_rate"],
        weight_decay=0.0,
        foreach=False,
    )


def load_model(config):
    runtime = require_rocm_runtime(config)
    snapshot = verify_snapshot(config)
    torch.manual_seed(config["optimization_seed"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_path"],
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": config["device"]},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    install_adapters(model, config)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_guard = check_model_precision(model, torch.device(config["device"]))
    check_snapshot_unchanged(snapshot)
    model.skill_transfer_load_proof = {"runtime": runtime, "model_guard": model_guard, "snapshot": snapshot}
    return model


class Encoded:
    def __init__(self, tokenizer, conventions, config):
        self.tokenizer = tokenizer
        self.conventions = conventions
        self.config = config
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos
        if self.eos is None or self.eos not in tokenizer.all_special_ids:
            raise ValueError("NATIVE_EOS_REQUIRED")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Protocol check."}, {"role": "assistant", "content": "[]"}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if not text.rstrip().endswith(tokenizer.eos_token):
            raise ValueError("NATIVE_EOS_MISMATCH: tokenizer EOS must close the assistant turn")
        self.cache = {}

    def tokens(self, row):
        if row.id not in self.cache:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": render(row, self.conventions)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            text = json.dumps(row.calls, separators=(",", ":"))
            answer = self.tokenizer.encode(text, add_special_tokens=False)
            if self.tokenizer.decode(answer, clean_up_tokenization_spaces=False) != text or self.eos in answer:
                raise ValueError("ANSWER_ROUNDTRIP")
            answer = answer + [self.eos]
            if len(answer) > self.config["max_new_tokens"] or len(prompt) + len(answer) > self.config["max_length"]:
                raise ValueError(f"CONTEXT_TRUNCATION: {row.id}, prompt={len(prompt)}, answer={len(answer)}")
            self.cache[row.id] = (list(prompt), answer)
        return self.cache[row.id]

    def batch(self, rows, training):
        encoded = [self.tokens(row) for row in rows]
        sequences = [prompt + answer if training else prompt for prompt, answer in encoded]
        width = max(map(len, sequences))
        inputs = {
            "input_ids": torch.tensor(
                [[self.pad] * (width - len(ids)) + ids for ids in sequences], device=self.config["device"]
            ),
            "attention_mask": torch.tensor(
                [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences], device=self.config["device"]
            ),
        }
        labels = None
        if training:
            labels = torch.tensor(
                [[-100] * (width - len(answer)) + answer for _, answer in encoded], device=self.config["device"]
            )
        return inputs, labels, max(len(answer) for _, answer in encoded)

    def generation_config(self, use_cache=None):
        return GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.config["max_new_tokens"],
            eos_token_id=self.eos,
            pad_token_id=self.pad,
            bos_token_id=self.tokenizer.bos_token_id,
            use_cache=self.config["generation_use_cache"] if use_cache is None else use_cache,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
        )


def load_tokenizer(config, conventions):
    snapshot = verify_snapshot(config, tokenizer_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, trust_remote_code=False)
    check_snapshot_unchanged(snapshot)
    return Encoded(tokenizer, conventions, config)


def train_update(model, optimizer, encoded, rows):
    if any(row.split != "train" for row in rows):
        raise ValueError("TRAIN_SPLIT_REQUIRED")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, labels, keep = encoded.batch(rows, training=True)
    logits = model(**inputs, use_cache=False, logits_to_keep=keep + 1).logits[:, :-1].float()
    labels = labels[:, -keep:]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view_as(labels)
    mask = labels != -100
    row_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    loss = row_losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError("NONFINITE_LOSS")
    loss.backward()
    parameters = [value for value in model.parameters() if value.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, encoded.config["grad_clip"], error_if_nonfinite=True))
    if norm == 0:
        raise RuntimeError("ZERO_GRADIENT")
    if any(value.grad is not None for value in model.parameters() if not value.requires_grad):
        raise RuntimeError("FROZEN_GRADIENT")
    optimizer.step()
    if any(not finite_tensor(value) for value in parameters):
        raise RuntimeError("NONFINITE_ADAPTER_AFTER_UPDATE")
    return {
        "loss": float(loss.detach()),
        "eos_loss": float(losses[:, -1].mean().detach()),
        "gradient_norm": norm,
        "input_tokens": int(inputs["attention_mask"].sum()),
        "target_tokens": int(mask.sum()),
        "example_ids": [row.id for row in rows],
    }


def generate(model, encoded, rows, use_cache=None):
    model.eval()
    records = []
    with torch.inference_mode():
        for offset in range(0, len(rows), encoded.config["eval_batch_size"]):
            chunk = rows[offset : offset + encoded.config["eval_batch_size"]]
            inputs, _, _ = encoded.batch(chunk, training=False)
            model.generation_config = encoded.generation_config(use_cache)
            tokens = model.generate(**inputs, generation_config=model.generation_config)
            suffixes = tokens[:, inputs["input_ids"].shape[1] :].cpu().tolist()
            for row, suffix in zip(chunk, suffixes, strict=True):
                terminated = encoded.eos in suffix
                end = suffix.index(encoded.eos) if terminated else len(suffix)
                padding_only = all(token == encoded.pad for token in suffix[end + 1 :])
                answer = encoded.tokenizer.decode(
                    suffix[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                records.append(
                    {
                        **grade_text(answer, row, encoded.conventions, terminated, padding_only),
                        "generated_ids": suffix[: end + 1],
                        "raw_generation_ids": suffix,
                    }
                )
    return records


def score(records):
    if not records:
        raise ValueError("EMPTY_EVALUATION")
    metrics = {
        key: sum(record[key] for record in records) / len(records)
        for key in ("correct", "format_valid", "executable", "native_eos")
    }
    patterns = sorted({record["pattern"] for record in records})
    return {
        "n": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": metrics.pop("correct"),
        **metrics,
        "errors": dict(
            Counter(record["error"] or ("WRONG_STATE" if not record["correct"] else "CORRECT") for record in records)
        ),
        "per_pattern": {
            pattern: sum(record["correct"] for record in records if record["pattern"] == pattern)
            / sum(record["pattern"] == pattern for record in records)
            for pattern in patterns
        },
    }


def evaluate(model, encoded, rows, path):
    records = generate(model, encoded, rows)
    write_json(path, {"metrics": score(records), "records": records})
    return score(records)


def save_checkpoint(model, optimizer, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    state = capture(model)
    optimizer_state = None if optimizer is None else copy.deepcopy(optimizer.state_dict())
    torch.save({"adapter": state, "optimizer": optimizer_state}, path / "state.pt")
    record = {
        **metadata,
        "tensor_manifest": tree_record(state),
        "optimizer_sha256": digest(tree_record(optimizer_state)),
    }
    write_json(path / "metadata.json", record)
    loaded = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if (
        tree_record(loaded["adapter"]) != tree_record(state)
        or digest(tree_record(loaded["optimizer"])) != record["optimizer_sha256"]
    ):
        raise RuntimeError("SERIALIZATION_CHANGED")
    return state


def load_checkpoint(model, path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    saved = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if tree_record(saved["adapter"]) != metadata["tensor_manifest"]:
        raise RuntimeError("CHECKPOINT_ADAPTER_MANIFEST_MISMATCH")
    if digest(tree_record(saved["optimizer"])) != metadata["optimizer_sha256"]:
        raise RuntimeError("CHECKPOINT_OPTIMIZER_MANIFEST_MISMATCH")
    restore(model, saved["adapter"])
    return saved["optimizer"]


def verify_checkpoint(model, encoded, checkpoint, rows, path):
    before = generate(model, encoded, rows)
    loaded = torch.load(Path(checkpoint) / "state.pt", map_location="cpu", weights_only=True)
    for layer in layers(model).values():
        with torch.no_grad():
            layer.B.add_(1)
    restore(model, loaded["adapter"])
    after = generate(model, encoded, rows)
    if before != after:
        raise RuntimeError("RELOAD_GENERATION_CHANGED")
    write_json(
        path, {"same_process_reload": True, "all_generated_records_exact": True, "ids": [row.id for row in rows]}
    )
