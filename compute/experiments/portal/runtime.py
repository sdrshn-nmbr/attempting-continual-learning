import fcntl
import hashlib
import json
import os
import platform
import signal
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from inventory import sha256_file, tensor_digest
from portallib import PortalBase, PortalModel
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


class CapabilityBlocked(RuntimeError):
    pass


class InterruptedRun(RuntimeError):
    pass


class OutputDirectoryLocked(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config_digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RunLog:
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.output / ".portal-run.lock").open("a+")
        self._lock_acquired = False
        try:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OutputDirectoryLocked(
                    f"output_directory_locked: another worker owns {self.output.resolve()}"
                ) from exc
            self._lock_acquired = True
            self._initialize(config)
        except BaseException:
            self.close()
            raise

    def _initialize(self, config):
        self.config = config
        self.digest = config_digest(config)
        self.started = time.perf_counter()
        self.stopping = False
        self.metrics = {
            "status": "running",
            "config": config,
            "config_sha256": self.digest,
            "measurements": {},
            "provenance": {},
            "timing_seconds": {},
            "invocations": [],
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "transformers": version("transformers"),
                "accelerate": version("accelerate"),
                "portallib": version("portallib"),
                "peft": version("peft"),
                "huggingface_hub": version("huggingface-hub"),
                "safetensors": version("safetensors"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "cuda_available": torch.cuda.is_available(),
                "gpu_execution": False,
            },
        }
        current_runtime = self.metrics["runtime"]
        old = self.output / "metrics.json"
        if old.exists():
            previous = json.loads(old.read_text())
            if previous["config_sha256"] != self.digest:
                raise ValueError(
                    "resume_config_mismatch: use an output directory for exactly this config"
                )
            self.metrics = previous
            self.metrics["status"] = "running"
        code_hashes = {
            file.name: sha256_file(file) for file in sorted(ROOT.glob("*.py"))
        }
        previous_code = self.metrics["provenance"].get("implementation")
        if previous_code and previous_code != code_hashes:
            raise ValueError(
                "resume_implementation_mismatch: code changed; start a new output directory"
            )
        self.metrics["provenance"]["implementation"] = code_hashes
        self.metrics["runtime"] = current_runtime
        self.metrics["invocations"].append(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "runtime": current_runtime,
            }
        )
        for key in ("error", "error_type", "traceback", "blockers"):
            self.metrics.pop(key, None)
        atomic_json(self.output / "config.json", config)
        self.event("run_started")

    def close(self):
        handle = self._lock_handle
        if handle is None:
            return
        self._lock_handle = None
        try:
            if self._lock_acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._lock_acquired = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False

    def request_stop(self, signum, _frame):
        self.stopping = True

    def check_stop(self):
        if self.stopping:
            raise InterruptedRun(
                "SIGTERM/SIGINT requested; resume from the last atomic checkpoint"
            )

    def event(self, event, **values):
        if self._lock_handle is None:
            raise RuntimeError("run_log_closed: output lock is no longer held")
        row = {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "elapsed_seconds": time.perf_counter() - self.started,
            **values,
        }
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
        atomic_json(
            self.output / "progress.json", {"status": self.metrics["status"], **row}
        )
        atomic_json(self.output / "metrics.json", self.metrics)
        print(json.dumps(row, allow_nan=False), flush=True)

    def finish(self, status, **values):
        self.metrics["status"] = status
        self.metrics["timing_seconds"]["last_invocation"] = (
            time.perf_counter() - self.started
        )
        self.metrics.update(values)
        self.event("run_" + status)

    def install_signal_handlers(self):
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)


def registry_from(config):
    path = Path(config.get("registry_path", "registry.json"))
    if not path.is_absolute():
        path = ROOT / path
    registry = json.loads(path.read_text())
    if not registry["all_official_ports_accounted_for"]:
        raise CapabilityBlocked("incomplete_official_port_inventory")
    return registry


def select_port(registry, model_id, revision):
    for row in registry["ports"]:
        if row["base"]["id"] == model_id:
            if row["base"]["revision"] != revision:
                raise CapabilityBlocked(
                    f"base_revision_mismatch: {model_id} requires {row['base']['revision']}"
                )
            return row
    raise CapabilityBlocked(
        f"no_official_port: {model_id}; direct LoRA reuse is unsupported"
    )


def load_port(row, config, log):
    artifact = row["artifact"]
    artifact_root = config.get("artifact_cache_dir")
    shipped = ROOT / ".artifacts" / row["key"]
    if artifact_root:
        local = Path(artifact_root) / artifact["id"] / artifact["revision"]
    elif shipped.exists():
        local = shipped
    else:
        local = log.output / "artifacts" / row["key"]
    log.event(
        "artifact_load_started",
        artifact_id=artifact["id"],
        revision=artifact["revision"],
    )
    if (
        not (local / "model.safetensors").exists()
        or not (local / "config.json").exists()
    ):
        if artifact_root:
            raise CapabilityBlocked(f"parent_cached_artifact_missing: {local}")
        for name in ("config.json", "model.safetensors"):
            hf_hub_download(
                artifact["id"], name, revision=artifact["revision"], local_dir=local
            )
    expected = next(
        file["sha256"]
        for file in artifact["files"]
        if file["name"] == "model.safetensors"
    )
    if sha256_file(local / "model.safetensors") != expected:
        raise CapabilityBlocked(f"artifact_weights_digest_mismatch: {local}")
    if sha256_file(local / "config.json") != artifact["config_sha256"]:
        raise CapabilityBlocked(f"artifact_config_digest_mismatch: {local}")
    portal = PortalModel.from_pretrained(str(local), local_files_only=True)
    portal.validate_base_model(row["base"]["id"], row["base"]["revision"])
    log.metrics["provenance"][artifact["id"]] = {
        "revision": artifact["revision"],
        "weights_sha256": expected,
        "path": str(local),
        "core_sha256": tensor_digest(portal.core.state_dict()),
    }
    return portal


def ensure_gpu(config, log):
    if config.get("device", "cuda:0") != "cuda:0":
        raise CapabilityBlocked(
            "production_lane_requires_cuda_0: CPU testing uses the test suite"
        )
    if not torch.cuda.is_available():
        raise CapabilityBlocked(
            "gpu_unavailable: native ROCm forward/backward has not run"
        )
    if torch.cuda.device_count() != 1:
        raise CapabilityBlocked(
            "gpu_isolation_required: expect exactly one visible GPU via ROCR_VISIBLE_DEVICES"
        )
    if not torch.cuda.is_bf16_supported():
        raise CapabilityBlocked("bf16_unavailable")
    props = torch.cuda.get_device_properties(0)
    log.metrics["runtime"].update(
        {"gpu": props.name, "device_count": 1, "total_memory_bytes": props.total_memory}
    )
    return torch.device("cuda:0")


def load_local_base(row, model_path, device, log, gradient_checkpointing=False):
    blocked = [b for b in row["status"]["blockers"] if b["code"] != "gated_base"]
    if blocked:
        raise CapabilityBlocked(json.dumps(blocked))
    path = Path(model_path)
    if not path.is_dir():
        raise CapabilityBlocked(f"base_snapshot_missing: {path}")
    base_info = row["base"]
    if path.name != base_info["revision"]:
        raise CapabilityBlocked(
            f"snapshot_revision_path_mismatch: {path}; expected exact revision directory"
        )
    weight_paths = sorted(path.glob("*.safetensors"))
    if not weight_paths:
        raise CapabilityBlocked(f"base_weights_missing: {path}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        expected = set(json.loads(index.read_text())["weight_map"].values())
        if expected - {weight.name for weight in weight_paths}:
            raise CapabilityBlocked(f"base_weight_shards_missing: {path}")
    snapshot = {
        "id": base_info["id"],
        "revision": base_info["revision"],
        "path": str(path),
        "config_sha256": sha256_file(path / "config.json"),
        "weight_files": [
            {"name": weight.name, "bytes": weight.stat().st_size}
            for weight in weight_paths
        ],
        "base_weight_hash_verification": "not performed; exact revision supplied by parent model cache",
        "tokenizer_files": {
            file.name: sha256_file(file)
            for file in path.iterdir()
            if file.is_file()
            and (
                file.name.startswith("tokenizer")
                or file.name == "special_tokens_map.json"
            )
        },
    }
    log.metrics["provenance"][base_info["id"]] = snapshot
    log.event(
        "base_load_started", model_id=base_info["id"], revision=base_info["revision"]
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise CapabilityBlocked("tokenizer_has_no_pad_or_eos_token")
    loader = (
        AutoModelForMultimodalLM
        if base_info["loader"] == "multimodal_lm"
        else AutoModelForCausalLM
    )
    model = loader.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="sdpa",
    )
    base = PortalBase(
        model_id=base_info["id"],
        revision=base_info["revision"],
        model=model,
        tokenizer=tokenizer,
        layer_path=base_info["layer_path"],
        allow_heterogeneous_targets=base_info["allow_heterogeneous_targets"],
    )
    base.freeze(gradient_checkpointing=gradient_checkpointing)
    base.model.eval()
    log.metrics["timing_seconds"]["load_" + row["key"]] = time.perf_counter() - started
    log.event(
        "base_loaded",
        model_id=base_info["id"],
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
    return base
