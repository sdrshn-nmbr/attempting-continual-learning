import argparse
import fcntl
import importlib.metadata
import json
import logging
import platform
import random
import re
import signal
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import Config, digest
from data import load_clinc
from engine import Experiment, Reporter, StopRequest, atomic_json, file_sha256, utc_now
from learning import load_model, trainables

logger = logging.getLogger(__name__)
LANE_ROOT = Path(__file__).resolve().parent
SUPERVISOR_FILES = {
    "execution.json",
    "output.log",
    "run.log",
    "packages.txt",
    "config.json",
    "task.json",
    "attempts",
}


def model_provenance(cfg, reporter=None):
    path = Path(cfg.model_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"LOCAL_MODEL_MISSING {path}")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"LOCAL_MODEL_WEIGHTS_MISSING {path}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        expected = set(json.loads(index.read_text())["weight_map"].values())
        missing = expected - {p.name for p in weights}
        if missing:
            raise FileNotFoundError(f"LOCAL_MODEL_SHARDS_MISSING {sorted(missing)}")
    paths = sorted(
        {
            p
            for pattern in ("*.json", "*.safetensors", "*.model", "*.jinja")
            for p in path.glob(pattern)
        }
    )
    files = []
    for item in paths:
        if reporter is not None:
            reporter.preparing("hashing_local_model", file=item.name)
        files.append(
            {
                "path": item.name,
                "bytes": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    return {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "local_path": str(path),
        "identity_verification": "All listed local file contents SHA256-hashed. Hub model_id/revision supplied by parent; no remote weight download or independent Hub identity check.",
        "files": files,
        "content_manifest_sha256": digest(files),
        "hf_config": json.loads((path / "config.json").read_text()),
    }


def runtime_provenance(cfg):
    result = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "peft",
                "datasets",
                "huggingface-hub",
            )
        },
        "rocm": torch.version.hip,
        "device": cfg.device,
        "base_dtype": cfg.dtype,
        "adapter_dtype": "float32",
        "attention": "sdpa",
        "use_cache": False,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "determinism": "Seeded RNG, fixed minibatches, dropout zero. GPU kernels are not guaranteed bitwise deterministic.",
    }
    if cfg.device == "cuda:0":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "GPU_ISOLATION_REQUIRED: expected exactly one visible GPU, cuda:0"
            )
        if not torch.version.hip:
            raise RuntimeError(
                "ROCM_RUNTIME_REQUIRED: this GPU lane is qualified for the supplied ROCm image"
            )
        torch.cuda.set_device(0)
        if cfg.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16_UNAVAILABLE")
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "architecture": properties.gcnArchName,
        }
    return result


def source_provenance():
    paths = sorted(LANE_ROOT.glob("*.py"))
    paths = [path for path in paths if not path.name.startswith("test_")]
    paths.extend([LANE_ROOT / "requirements.txt", LANE_ROOT / "protocol.json"])
    return {path.name: file_sha256(path) for path in paths}


def fail(reporter, error):
    details = {"type": type(error).__name__, "message": str(error), "at": utc_now()}
    reporter.event("failed", error=details)
    for name in ("metrics.json", "progress.json"):
        path = reporter.output_dir / name
        previous = json.loads(path.read_text()) if path.exists() else {}
        atomic_json(
            path,
            {
                **previous,
                "status": "failed",
                "error": details,
                "resume_checkpoint_exists": (
                    reporter.output_dir / "checkpoint.pt"
                ).exists(),
                "results_are_from_last_committed_checkpoint": True,
            },
        )


def validate_supervisor_bootstrap(cfg, output_dir):
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RuntimeError(
            f"RUN_ID_ALREADY_EXISTS: not a supervisor output directory: {output_dir}"
        )
    entries = {path.name: path for path in output_dir.iterdir()}
    required = {"execution.json", "packages.txt", "config.json"}
    if (
        not required <= entries.keys()
        or not {"output.log", "run.log"} & entries.keys()
        or entries.keys() - SUPERVISOR_FILES
    ):
        raise RuntimeError(
            f"RUN_ID_ALREADY_EXISTS: output contains research data or is not a supervisor bootstrap: {output_dir}"
        )
    for name, path in entries.items():
        valid_type = path.is_dir() if name == "attempts" else path.is_file()
        if path.is_symlink() or not valid_type:
            raise RuntimeError(f"SUPERVISOR_BOOTSTRAP_INVALID_FILE {path}")
    if Config(**json.loads(entries["config.json"].read_text())) != cfg:
        raise RuntimeError("SUPERVISOR_CONFIG_MISMATCH: refusing existing output")
    execution = json.loads(entries["execution.json"].read_text())
    if (
        execution.get("status") != "running"
        or execution.get("task_id") != output_dir.name
    ):
        raise RuntimeError(
            "SUPERVISOR_EXECUTION_MISMATCH: bootstrap is terminal or belongs to another run"
        )
    if "task.json" in entries:
        task = json.loads(entries["task.json"].read_text())
        if task["id"] != output_dir.name or Config(**task["config"]) != cfg:
            raise RuntimeError("SUPERVISOR_TASK_MISMATCH: refusing existing output")
    if "attempts" in entries:
        for attempt in entries["attempts"].iterdir():
            if (
                attempt.is_symlink()
                or not attempt.is_dir()
                or not re.fullmatch(
                    r"(?:[0-9a-f]{32}|prior-[0-9a-f]{64})", attempt.name
                )
            ):
                raise RuntimeError(f"SUPERVISOR_HISTORY_INVALID {attempt}")
            for receipt in attempt.iterdir():
                if (
                    receipt.is_symlink()
                    or not receipt.is_file()
                    or not re.fullmatch(r"[0-9a-f]{64}\.json", receipt.name)
                    or file_sha256(receipt) != receipt.stem
                ):
                    raise RuntimeError(f"SUPERVISOR_HISTORY_INVALID {receipt}")
                if json.loads(receipt.read_text())["task_id"] != output_dir.name:
                    raise RuntimeError(f"SUPERVISOR_HISTORY_RUN_MISMATCH {receipt}")


def prepare_output(cfg, output_dir, resume):
    if not resume:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            validate_supervisor_bootstrap(cfg, output_dir)
        return None
    checkpoint = output_dir / "checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"RESUME_CHECKPOINT_MISSING {output_dir}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["state"]["completed"]:
        raise RuntimeError(f"COMPLETED_RUN_IMMUTABLE {output_dir}")
    provenance = json.loads((output_dir / "provenance.json").read_text())
    if (
        payload["signature"]
        != digest({"config": cfg.as_dict(), "provenance": provenance})
        or provenance["source_sha256"] != source_provenance()
    ):
        raise RuntimeError(
            "RESUME_SIGNATURE_MISMATCH: config or source changed; choose a unique run ID"
        )
    return provenance


def execute(cfg, output_dir, resume=False):
    if not resume:
        prepare_output(cfg, output_dir, resume=False)
    with (output_dir / ".run.lock").open("r" if resume else "x") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"OUTPUT_DIRECTORY_IN_USE {output_dir}") from error
        previous_provenance = (
            prepare_output(cfg, output_dir, resume=True) if resume else None
        )
        reporter = Reporter(output_dir, cfg)
        write_failure = not resume
        stop = StopRequest()
        previous_handlers = {
            s: signal.signal(s, stop.handle) for s in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            started = time.perf_counter()
            if not resume:
                reporter.preparing("runtime")
            runtime = runtime_provenance(cfg)
            model_info = model_provenance(cfg, None if resume else reporter)
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if not resume:
                reporter.preparing("tokenizer_and_data")
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_path, local_files_only=True, trust_remote_code=False
            )
            prepared = load_clinc(cfg, tokenizer)
            if not resume:
                reporter.preparing("loading_model")
            model = load_model(cfg)
            expected_device = torch.device(cfg.device)
            if any(p.device != expected_device for p in model.parameters()):
                raise RuntimeError("MODEL_NOT_FULLY_RESIDENT_ON_REQUESTED_DEVICE")
            provenance = {
                "model": model_info,
                "data": prepared.provenance,
                "runtime": runtime,
                "source_sha256": source_provenance(),
                "protocol": json.loads((LANE_ROOT / "protocol.json").read_text()),
                "trainable_parameters": {
                    name: list(p.shape) for name, p in trainables(model).items()
                },
                "trainable_parameter_count": sum(
                    p.numel() for p in trainables(model).values()
                ),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
            provenance_path = output_dir / "provenance.json"
            if resume and digest(previous_provenance) != digest(provenance):
                raise RuntimeError(
                    "PROVENANCE_CHANGED: refusing to overwrite the prior run; use a separate output directory"
                )
            write_failure = True
            if not resume:
                atomic_json(provenance_path, provenance)
            reporter.event(
                "prepared",
                setup_seconds=time.perf_counter() - started,
                hostname=platform.node(),
                config=cfg.as_dict(),
                provenance_sha256=digest(provenance),
            )
            experiment = Experiment(model, prepared, cfg, reporter, provenance, stop)
            return experiment.run()
        except Exception as error:
            logger.exception("PLASTICITY_FAILED")
            if write_failure:
                fail(reporter, error)
            return 1
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(
        description="Sequential CLINC LoRA screen; new runs require a unique output directory"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an unfinished run with identical config, source, model and data; completed runs are immutable",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [plasticity] %(message)s"
    )
    cfg = Config(**json.loads(args.config.read_text()))
    return execute(cfg, args.output_dir.resolve(), resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
