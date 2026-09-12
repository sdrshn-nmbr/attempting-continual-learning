import fcntl
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from protocol import canonical_config, digest

LOGGER = logging.getLogger("nla")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class StopRequested(Exception):
    pass


class Store:
    def __init__(self, output_dir, config, code_hash):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("NLA_OUTPUT_LOCKED: another process owns this output directory") from None
        self.config_hash = digest(canonical_config(config))
        self.code_hash = code_hash
        self.started = time.monotonic()
        self.previous_elapsed = 0.0
        self.stop_signal = None
        self.previous_handlers = {}
        path = self.root / "metrics.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if not config.get("resume", False):
                self.lock.close()
                raise ValueError("NLA_OUTPUT_EXISTS: use a new output directory or set resume=true")
            if previous["config_sha256"] != self.config_hash or previous["code_sha256"] != code_hash:
                self.lock.close()
                raise ValueError("NLA_RESUME_IDENTITY: code/config changed; use a new output directory")
            self.metrics = previous
            self.previous_elapsed = previous.get("elapsed_seconds", 0.0)
        else:
            self.metrics = {
                "status": "initializing",
                "started_at": now(),
                "config": canonical_config(config),
                "config_sha256": self.config_hash,
                "code_sha256": code_hash,
                "measurements": {},
            }
            atomic_json(path, self.metrics)
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self.request_stop)
        self.event("session_started", resume=config.get("resume", False))

    def request_stop(self, signum, frame):
        self.stop_signal = signum

    def check_stop(self):
        if self.stop_signal is not None:
            raise StopRequested(f"signal {self.stop_signal}")

    def elapsed(self):
        return self.previous_elapsed + time.monotonic() - self.started

    def save(self, status=None):
        if status is not None:
            self.metrics["status"] = status
        self.metrics["updated_at"] = now()
        self.metrics["elapsed_seconds"] = self.elapsed()
        atomic_json(self.root / "metrics.json", self.metrics)

    def event(self, event, **details):
        entry = {"at": now(), "elapsed_seconds": self.elapsed(), "event": event, **details}
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        LOGGER.info("%s %s", event, json.dumps(details, ensure_ascii=False, allow_nan=False))

    def progress(self, phase, **details):
        entry = {
            "at": now(),
            "status": self.metrics["status"],
            "phase": phase,
            "elapsed_seconds": self.elapsed(),
            **details,
        }
        atomic_json(self.root / "progress.json", entry)

    def finish(self, status, **details):
        self.metrics.update(details)
        self.save(status)
        self.event("session_finished", status=status)
        self.progress("finished", result_status=status)

    def close(self):
        for sig, handler in self.previous_handlers.items():
            signal.signal(sig, handler)
        self.lock.close()


def gpu_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
