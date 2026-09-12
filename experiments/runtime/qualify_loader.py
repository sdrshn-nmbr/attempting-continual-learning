import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from loader_guard import Channel
from transformers import AutoModelForCausalLM

LOG = logging.getLogger("loader-qualification-child")
MAX_HOST_CHUNK_BYTES = 64 * 1024**2
FILE_CHUNK_BYTES = 8 * 1024**2
ENVIRONMENT_KEYS = (
    "HF_DEACTIVATE_ASYNC_LOAD",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "POD_UID",
    "LOADER_GUARD_CHANNEL_FD",
)
START_ENVIRONMENT = {name: os.environ.get(name) for name in ENVIRONMENT_KEYS}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def file_digests(path, git_blob=False):
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1() if git_blob else None
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        if sha1 is not None:
            sha1.update(f"blob {before.st_size}\0".encode())
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            sha256.update(chunk)
            if sha1 is not None:
                sha1.update(chunk)
        after = os.fstat(handle.fileno())
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in identity):
        raise RuntimeError(f"FILE_CHANGED_DURING_HASH: {path}")
    return {
        "sha256": sha256.hexdigest(),
        "git_blob_sha1": sha1.hexdigest() if sha1 is not None else None,
        "bytes": before.st_size,
        "identity": {key: getattr(before, key) for key in identity},
    }


def write_exclusive(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            written = handle.write(data)
            if written != len(data):
                raise OSError(f"SHORT_RECEIPT_WRITE: {path}")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_stage(output, event, payload):
    record = {
        "schema_version": 1,
        "stage": event,
        "at": utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        **payload,
    }
    return write_exclusive(output / f"{event}.json", canonical_json(record) + b"\n")


def read_config(path):
    data = Path(path).read_bytes()
    config = json.loads(data)
    if config.get("schema_version") != 1:
        raise ValueError("CONFIG_INVALID: schema_version must be 1")
    if (
        config.get("model_id") != "Qwen/Qwen3-8B"
        or config.get("revision") != "b968826d9c46dd6066d109eabc6255188de91218"
    ):
        raise ValueError("CONFIG_INVALID: qualification is pinned to Qwen3-8B")
    for key in ("model_path", "manifest_path"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"CONFIG_INVALID: {key} must be absolute")
    chunk_bytes = config["hash_chunk_bytes"]
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_HOST_CHUNK_BYTES:
        raise ValueError("CONFIG_INVALID: hash_chunk_bytes must be at most 64MiB")
    for key, maximum in (("cpu_threads", 8), ("cpu_interop_threads", 4)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError("CONFIG_INVALID: CPU thread count out of bounds")
    timeout = config["channel_timeout_seconds"]
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 1 <= timeout <= 7200
    ):
        raise ValueError("CONFIG_INVALID: channel timeout out of bounds")
    if config["expected_vocab_size"] != 151936 or type(config["seed"]) is not int:
        raise ValueError("CONFIG_INVALID: vocabulary or seed")
    probes = config["probe_token_ids"]
    if len(probes) != 3 or len({tuple(tokens) for tokens in probes}) != 3:
        raise ValueError("CONFIG_INVALID: require three distinct fixed token sequences")
    for tokens in probes:
        if not 1 <= len(tokens) <= 32 or any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or not 0 <= token < config["expected_vocab_size"]
            for token in tokens
        ):
            raise ValueError("CONFIG_INVALID: invalid fixed probe tokens")
    return config, hashlib.sha256(data).hexdigest()


def verify_environment(config, hold_after_probe):
    current = {name: os.environ.get(name) for name in ENVIRONMENT_KEYS}
    if current != START_ENVIRONMENT:
        raise RuntimeError("ENVIRONMENT_CHANGED_AFTER_IMPORT")
    mode = current["HF_DEACTIVATE_ASYNC_LOAD"]
    if mode not in {"0", "1"} or hold_after_probe != (mode == "1"):
        raise RuntimeError(
            "LOAD_MODE_INVALID: async=0 exits; sync=1 requires --hold-after-probe"
        )
    if any(current[key] != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("OFFLINE_ENVIRONMENT_REQUIRED")
    if sys.platform != "linux" or sys.byteorder != "little":
        raise RuntimeError("RUNTIME_INVALID: require little-endian Linux ROCm worker")
    if not current["POD_UID"]:
        raise RuntimeError("POD_UID_REQUIRED: guard must supply verified pod identity")
    uuid.UUID(current["POD_UID"])
    expected = config["expected_runtime"]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", *expected["packages"])
    }
    for name, version in expected["packages"].items():
        if versions[name] != version:
            raise RuntimeError(
                f"PACKAGE_VERSION_MISMATCH: {name}={versions[name]}, expected {version}"
            )
    if (
        torch.__version__ != expected["torch_version"]
        or torch.version.hip != expected["hip_version"]
    ):
        raise RuntimeError(
            f"VENDOR_RUNTIME_MISMATCH: torch={torch.__version__}, hip={torch.version.hip}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "GPU_VISIBILITY_MISMATCH: require exactly one visible ROCm GPU"
        )
    properties = torch.cuda.get_device_properties(0)
    if properties.gcnArchName.split(":", 1)[0] != expected["gpu_arch"]:
        raise RuntimeError(f"GPU_ARCH_MISMATCH: {properties.gcnArchName}")
    return {
        "environment": current,
        "packages": versions,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu": {
            "name": properties.name,
            "arch": properties.gcnArchName,
            "total_memory": properties.total_memory,
        },
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "pod_uid": current["POD_UID"],
        "mode": "sync" if mode == "1" else "async",
    }


def fingerprint_sources(config):
    package = Path(transformers.__file__).parent
    records = {}
    for name, expected in config["transformers_source_sha256"].items():
        path = package / name
        record = file_digests(path)
        if record["sha256"] != expected:
            raise RuntimeError(f"PINNED_TRANSFORMERS_SOURCE_MISMATCH: {path}")
        records[f"transformers/{name}"] = {"path": str(path), **record}
    torch_root = Path(torch.__file__).parent
    paths = {
        "child": Path(__file__),
        "qwen3_implementation": package / "models/qwen3/modeling_qwen3.py",
        "torch_python": Path(torch.__file__),
        "torch_version": Path(torch.version.__file__),
        "torch_extension": Path(torch._C.__file__),
        **{
            name: torch_root / "lib" / name
            for name in (
                "libtorch_hip.so",
                "libtorch_cpu.so",
                "libc10.so",
                "libc10_hip.so",
            )
        },
    }
    for name, path in paths.items():
        records[name] = {"path": str(path), **file_digests(path)}
    return records


def verify_snapshot(config):
    manifest_path = Path(config["manifest_path"])
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != config["manifest_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_MISMATCH")
    manifest = json.loads(manifest_bytes)
    asset = manifest["assets"][config["manifest_asset_key"]]
    if asset["repo"] != config["model_id"] or asset["revision"] != config["revision"]:
        raise RuntimeError("FROZEN_ASSET_IDENTITY_MISMATCH")
    root = Path(config["model_path"])
    files = []
    for expected in sorted(asset["files"], key=lambda value: value["path"]):
        relative = Path(expected["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"SNAPSHOT_PATH_INVALID: {relative}")
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"SNAPSHOT_CONTENT_MISMATCH: {path}")
        record = file_digests(path, git_blob=expected["git_blob_sha1"] is not None)
        for key in ("sha256", "git_blob_sha1"):
            if expected[key] is not None and record[key] != expected[key]:
                raise RuntimeError(f"SNAPSHOT_CONTENT_MISMATCH: {path}: {key}")
        if expected["sha256"] is None and expected["git_blob_sha1"] is None:
            raise RuntimeError(f"SNAPSHOT_DIGEST_MISSING: {path}")
        files.append({"path": str(path), "relative_path": str(relative), **record})
    return {
        "repo": asset["repo"],
        "revision": asset["revision"],
        "path": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "files": files,
        "verified_files": len(files),
        "verified_bytes": sum(file["bytes"] for file in files),
        "content_sha256": hashlib.sha256(
            canonical_json(
                [
                    {key: file[key] for key in ("relative_path", "sha256", "bytes")}
                    for file in files
                ]
            )
        ).hexdigest(),
    }


def check_snapshot_identity(snapshot):
    for record in snapshot["files"]:
        stat = Path(record["path"]).stat()
        if any(
            getattr(stat, key) != value for key, value in record["identity"].items()
        ):
            raise RuntimeError(f"SNAPSHOT_CHANGED_AFTER_BASELINE: {record['path']}")


def tensor_pieces(tensor, max_elements):
    if tensor.numel() == 0:
        return
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), max_elements):
            yield flat.narrow(0, start, min(max_elements, flat.numel() - start))
    elif tensor.numel() <= max_elements:
        yield tensor
    else:
        row_elements = math.prod(tensor.shape[1:])
        if row_elements > max_elements:
            for index in range(tensor.shape[0]):
                yield from tensor_pieces(tensor.select(0, index), max_elements)
        else:
            rows = max(1, max_elements // row_elements)
            for start in range(0, tensor.shape[0], rows):
                yield tensor.narrow(0, start, min(rows, tensor.shape[0] - start))


def hash_loaded_state(model, chunk_bytes):
    if not 1 <= chunk_bytes <= MAX_HOST_CHUNK_BYTES:
        raise ValueError("STATE_HASH_CHUNK_INVALID")
    tensors = [
        (name, "parameter", tensor)
        for name, tensor in model.named_parameters(remove_duplicate=False)
    ] + [
        (name, "buffer", tensor)
        for name, tensor in model.named_buffers(remove_duplicate=False)
    ]
    records = []
    for name, kind, tensor in sorted(tensors, key=lambda value: value[0]):
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise RuntimeError(f"STATE_TENSOR_UNSUPPORTED: {name}")
        max_elements = chunk_bytes // tensor.element_size()
        if max_elements < 1:
            raise ValueError(f"STATE_HASH_CHUNK_TOO_SMALL: {name}")
        digest = hashlib.sha256()
        seen = 0
        for piece in tensor_pieces(tensor.detach(), max_elements):
            host = piece.to(
                device="cpu", copy=True, memory_format=torch.contiguous_format
            )
            data = memoryview(host.reshape(-1).view(torch.uint8).numpy()).cast("B")
            if len(data) > chunk_bytes:
                raise RuntimeError(f"STATE_HASH_CHUNK_EXCEEDED: {name}")
            digest.update(data)
            seen += len(data)
            data.release()
            del data, host
        if seen != tensor.numel() * tensor.element_size():
            raise RuntimeError(f"STATE_HASH_LENGTH_MISMATCH: {name}")
        records.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": seen,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "algorithm": "sha256 of sorted canonical tensor records including raw-byte SHA256",
        "state_sha256": hashlib.sha256(canonical_json(records)).hexdigest(),
        "tensors": records,
        "tensor_count": len(records),
        "tensor_bytes": sum(record["bytes"] for record in records),
        "host_chunk_bytes": chunk_bytes,
        "includes_duplicate_names_and_nonpersistent_buffers": True,
    }


def load_model(config):
    return AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        disable_mmap=False,
    )


def validate_loaded_model(model, config):
    if (
        type(model).__name__ != "Qwen3ForCausalLM"
        or model.config.vocab_size != config["expected_vocab_size"]
    ):
        raise RuntimeError("LOADED_MODEL_IDENTITY_MISMATCH")
    placement = {}
    for kind, tensors in (
        ("parameters", model.named_parameters()),
        ("buffers", model.named_buffers()),
    ):
        counts = {
            "tensor_count": 0,
            "element_count": 0,
            "device_counts": {},
            "dtype_counts": {},
        }
        for name, tensor in tensors:
            device, dtype = tensor.device, tensor.dtype
            if device != torch.device("cuda:0"):
                raise RuntimeError(f"LOADED_TENSOR_NOT_ON_CUDA0: {name}: {device}")
            if kind == "parameters" and dtype != torch.bfloat16:
                raise RuntimeError(f"LOADED_PARAMETER_DTYPE_MISMATCH: {name}: {dtype}")
            counts["tensor_count"] += 1
            counts["element_count"] += tensor.numel()
            for key, value in (
                ("device_counts", str(device)),
                ("dtype_counts", str(dtype)),
            ):
                counts[key][value] = counts[key].get(value, 0) + 1
        placement[kind] = counts
    if model.config._attn_implementation != "sdpa":
        raise RuntimeError("LOADED_ATTENTION_BACKEND_MISMATCH")
    return placement


def run_probes(model, config, output):
    records = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, tokens in enumerate(config["probe_token_ids"]):
            inputs = torch.tensor([tokens], dtype=torch.long, device=device)
            result = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=False,
                logits_to_keep=0,
            )
            logits = result.logits.to(device="cpu", dtype=torch.float32).contiguous()
            expected_shape = (1, len(tokens), config["expected_vocab_size"])
            if tuple(logits.shape) != expected_shape or not bool(
                torch.isfinite(logits).all()
            ):
                raise RuntimeError(
                    f"PROBE_LOGITS_INVALID: index={index}, shape={tuple(logits.shape)}"
                )
            data = memoryview(logits.numpy()).cast("B")
            payload = write_exclusive(output / f"probe-{index:03d}.f32", data)
            data.release()
            record = {
                "index": index,
                "input_ids": tokens,
                "shape": list(logits.shape),
                "source_dtype": str(result.logits.dtype),
                "saved_dtype": "float32",
                "byteorder": "little",
                "format": "raw IEEE754 float32, C order",
                "predicted_token_ids": logits.argmax(dim=-1).tolist(),
                "payload": payload,
            }
            metadata = write_exclusive(
                output / f"probe-{index:03d}.json", canonical_json(record) + b"\n"
            )
            records.append({**record, "metadata": metadata})
            del data, logits, result, inputs
    return records


def report_failure(channel, output, stage, error):
    fields = {
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "reason": str(error)[:4096],
    }
    receipt = None
    if output is not None:
        try:
            receipt = write_stage(
                output, "failure", {**fields, "research_result": False}
            )
        except Exception as receipt_error:
            fields["receipt_error"] = (
                f"{type(receipt_error).__name__}: {receipt_error}"[:4096]
            )
            LOG.exception("failure receipt unavailable stage=%s", stage)
    try:
        channel.send("failed", receipt, **fields)
    except Exception:
        LOG.exception("failure notification unavailable stage=%s", stage)


def qualify(config_path, output_dir, hold_after_probe=False):
    config, config_sha256 = read_config(config_path)
    channel = Channel(config["channel_timeout_seconds"])
    output = None
    stage = "preflight"
    try:
        candidate = Path(output_dir).absolute()
        candidate.mkdir(mode=0o700, parents=True, exist_ok=False)
        output = candidate
        runtime = verify_environment(config, hold_after_probe)
        torch.set_num_threads(config["cpu_threads"])
        torch.set_num_interop_threads(config["cpu_interop_threads"])
        sources = fingerprint_sources(config)
        provenance = {
            "config": config,
            "config_sha256": config_sha256,
            "runtime": runtime,
            "sources": sources,
            "training_steps": 0,
            "research_result": False,
        }
        request = write_stage(output, "request", provenance)
        stage = "cache_verified"
        snapshot = verify_snapshot(config)
        cache_receipt = write_stage(
            output, stage, {"request": request, "snapshot": snapshot}
        )
        channel.send(stage, cache_receipt, snapshot_sha256=snapshot["content_sha256"])
        channel.wait_for_continue()
        check_snapshot_identity(snapshot)
        if {
            name: os.environ.get(name) for name in ENVIRONMENT_KEYS
        } != START_ENVIRONMENT:
            raise RuntimeError("ENVIRONMENT_CHANGED_BEFORE_LOAD")
        stage = "load_started"
        started = write_stage(
            output,
            stage,
            {"request": request, "snapshot": cache_receipt, "mode": runtime["mode"]},
        )
        channel.send(stage, started)
        torch.manual_seed(config["seed"])
        model = load_model(config)
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        torch.cuda.synchronize(0)
        placement = validate_loaded_model(model, config)
        stage = "load_complete"
        loaded = write_stage(
            output,
            stage,
            {
                "model_class": type(model).__name__,
                "requested_device_map": {"": 0},
                "actual_placement": placement,
                "hf_device_map": getattr(model, "hf_device_map", None),
                "parameter_count": placement["parameters"]["element_count"],
                "gpu_allocated_bytes": torch.cuda.memory_allocated(0),
                "gpu_reserved_bytes": torch.cuda.memory_reserved(0),
            },
        )
        channel.send(stage, loaded)
        channel.wait_for_continue()
        stage = "state_hash_started"
        hash_started = write_stage(
            output,
            stage,
            {"load_complete": loaded, "host_chunk_bytes": config["hash_chunk_bytes"]},
        )
        channel.send(stage, hash_started)
        state = hash_loaded_state(model, config["hash_chunk_bytes"])
        stage = "state_hash_complete"
        hashed = write_stage(output, stage, state)
        channel.send(stage, hashed, state_sha256=state["state_sha256"])
        stage = "probe_started"
        probing = write_stage(
            output,
            stage,
            {
                "state_hash_complete": hashed,
                "fixed_input_sha256": hashlib.sha256(
                    canonical_json(config["probe_token_ids"])
                ).hexdigest(),
            },
        )
        channel.send(stage, probing)
        probes = run_probes(model, config, output)
        torch.cuda.synchronize(0)
        stage = "probe_complete"
        receipt = write_stage(
            output,
            stage,
            {
                **provenance,
                "snapshot": cache_receipt,
                "snapshot_content_sha256": snapshot["content_sha256"],
                "loaded_state": hashed,
                "state_sha256": state["state_sha256"],
                "probes": probes,
                "hold_after_probe": hold_after_probe,
                "receipt_semantics": "load/state/probe stage complete; not a process-completion or guard-kill receipt",
            },
        )
        channel.send(
            stage,
            receipt,
            state_sha256=state["state_sha256"],
            hold_after_probe=hold_after_probe,
        )
        if hold_after_probe:
            stage = "holding_for_guard_kill"
            channel.hold_for_guard_kill()
        return 0
    except Exception as exc:
        LOG.exception("qualification failed stage=%s", stage)
        report_failure(channel, output, stage, exc)
        raise
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hold-after-probe", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [loader-child] %(message)s"
    )
    return qualify(args.config, args.output_dir, args.hold_after_probe)


if __name__ == "__main__":
    raise SystemExit(main())
