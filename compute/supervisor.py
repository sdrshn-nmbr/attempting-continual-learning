import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def receipt_failure(path, reason):
    logger.error("RECEIPT_INTEGRITY: %s: %s", path, reason)
    raise RuntimeError(f"RECEIPT_INTEGRITY: {path}: {reason}")


def atomic_bytes(path, content, *, immutable=False, staging=None):
    with tempfile.NamedTemporaryFile(dir=staging or path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.flush()
            os.fchmod(file.fileno(), 0o444 if immutable else 0o644)
            os.fsync(file.fileno())
            if immutable:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or path.read_bytes() != content
                    ):
                        receipt_failure(
                            path, "immutable history collision or corruption"
                        )
            else:
                temporary.replace(path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def decode_receipt(path, content):
    try:
        receipt = json.loads(content)
    except (ValueError, UnicodeError) as error:
        receipt_failure(path, f"invalid receipt JSON: {error}")
    if (
        not isinstance(receipt, dict)
        or receipt.get("task_id") != path.parent.name
        or not isinstance(receipt.get("status"), str)
        or receipt.get("status")
        not in {"running", "interrupted", "completed", "failed", "blocked"}
    ):
        receipt_failure(path, "invalid task identity or execution status")
    if "attempt_id" in receipt and (
        not isinstance(receipt["attempt_id"], str)
        or not re.fullmatch(
            r"(?:[0-9a-f]{32}|prior-[0-9a-f]{64})", receipt["attempt_id"]
        )
    ):
        receipt_failure(path, "invalid attempt identity")
    return receipt


def archive_path(path, content):
    receipt = decode_receipt(path, content)
    digest = hashlib.sha256(content).hexdigest()
    attempt_id = receipt.get("attempt_id", f"prior-{digest}")
    return path.parent / "attempts" / attempt_id / f"{digest}.json"


def read_receipt(path):
    if path.is_symlink():
        receipt_failure(path, "execution receipt is a symlink")
    content = path.read_bytes() if path.exists() else None
    history = path.parent / "attempts"
    if history.is_symlink():
        receipt_failure(history, "history is a symlink")
    snapshots = {}
    if history.exists():
        if not history.is_dir():
            receipt_failure(history, "history is not a directory")
        for attempt in history.iterdir():
            if (
                attempt.is_symlink()
                or not attempt.is_dir()
                or not re.fullmatch(
                    r"(?:[0-9a-f]{32}|prior-[0-9a-f]{64})", attempt.name
                )
            ):
                receipt_failure(attempt, "invalid history directory")
            for archived in attempt.iterdir():
                if archived.is_symlink() or not archived.is_file():
                    receipt_failure(archived, "invalid history file")
                snapshot = archived.read_bytes()
                if archive_path(path, snapshot) != archived:
                    receipt_failure(
                        archived, "history identity or content hash mismatch"
                    )
                snapshots[str(archived.relative_to(path.parent))] = snapshot
                if content is None:
                    receipt_failure(
                        path, "current receipt missing while history exists"
                    )
    receipt = decode_receipt(path, content) if content is not None else None
    if receipt is not None and "previous_receipt" in receipt:
        current_archive = str(archive_path(path, content).relative_to(path.parent))
        if snapshots.get(current_archive) != content:
            receipt_failure(path, "current receipt is missing from immutable history")
    for snapshot in snapshots.values():
        previous = decode_receipt(path, snapshot).get("previous_receipt")
        if previous is not None and (
            not isinstance(previous, str) or previous not in snapshots
        ):
            receipt_failure(path, "history predecessor is missing or invalid")
    return receipt


def publish_receipt(path, receipt):
    read_receipt(path)
    contents = [path.read_bytes()] if path.exists() else []
    receipt["previous_receipt"] = (
        str(archive_path(path, contents[0]).relative_to(path.parent))
        if contents
        else None
    )
    contents.append(json_bytes(receipt))
    for content in contents:
        archived = archive_path(path, content)
        for directory in (archived.parent.parent, archived.parent):
            if directory.is_symlink():
                receipt_failure(directory, "history directory is a symlink")
            directory.mkdir(exist_ok=True)
            sync_directory(directory.parent)
        atomic_bytes(archived, content, immutable=True, staging=path.parent)
    atomic_bytes(path, contents[-1])


def check_recovery(path, supervisor):
    prior = read_receipt(path)
    if prior is not None and prior["status"] == "running":
        prior_bytes = path.read_bytes()
        prior = {
            **prior,
            "attempt_id": archive_path(path, prior_bytes).parent.name,
            "status": "blocked",
            "blocked_reason": "unclean_supervisor_termination",
            "recovery": {
                "observed_at": now(),
                "supervisor": supervisor,
                "prior_receipt_sha256": hashlib.sha256(prior_bytes).hexdigest(),
                "required_action": "Orchestrator review is required before restarting this queue; use a new task ID for another attempt.",
            },
        }
        publish_receipt(path, prior)
    if (
        prior is not None
        and prior.get("blocked_reason") == "unclean_supervisor_termination"
    ):
        logger.error(
            "UNCLEAN_EXECUTION_BLOCKED: %s from pod=%s; entire queue held for orchestrator review; recorded PIDs were not signalled",
            path.parent.name,
            prior.get("pod"),
        )
        raise RuntimeError(
            "UNCLEAN_EXECUTION_BLOCKED: queue requires orchestrator review"
        )
    return prior


def signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--idle-timeout", type=int, default=900)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [portfolio] %(message)s"
    )
    root = args.root
    queue = root / "queue"
    runs = root / "runs"
    queue.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    runtime_gates = json.loads((root / "runtime-gates.json").read_text())
    ownership = (root / ".supervisor.lock").open("a")
    try:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "Another supervisor already owns this experiment queue"
        ) from error
    active = {}
    seen = set()
    stopping = []
    idle_since = time.monotonic()
    supervisor = {
        "id": uuid4().hex,
        "pid": os.getpid(),
        "started_at": now(),
        "pod": os.environ.get("POD_NAME"),
        "pod_uid": os.environ.get("POD_UID"),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "python_version": sys.version,
    }
    for output in sorted(runs.iterdir()):
        if output.is_dir():
            check_recovery(output / "execution.json", supervisor)
    runtime_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("run_task.py").read_bytes()
        + Path(__file__).with_name("runtime-requirements.txt").read_bytes()
    ).hexdigest()

    def stop(signum, _frame):
        if not stopping:
            stopping.extend([time.monotonic(), signum])
            logger.warning(
                "Termination %s: checkpointing %s workers", signum, len(active)
            )
            for item in active.values():
                signal_group(item["process"], signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            for task_id, item in list(active.items()):
                process = item["process"]
                code = process.poll()
                if code is None and time.monotonic() > item["deadline"]:
                    if not item.get("timed_out"):
                        item["timed_out"] = True
                        item["deadline"] = time.monotonic() + 45
                        logger.error(
                            "%s exceeded its execution limit: requesting checkpoint",
                            task_id,
                        )
                        signal_group(process, signal.SIGTERM)
                    else:
                        signal_group(process, signal.SIGKILL)
                if code is None:
                    continue
                item["log"].close()
                receipt = {
                    **item["receipt"],
                    "finished_at": now(),
                    "exit_code": code,
                    "timed_out": item.get("timed_out", False),
                    "status": "interrupted"
                    if stopping
                    else "failed"
                    if item.get("timed_out") or code != 0
                    else "completed",
                }
                publish_receipt(runs / task_id / "execution.json", receipt)
                logger.info("%s %s (exit=%s)", task_id, receipt["status"], code)
                del active[task_id]
                idle_since = time.monotonic()

            if not stopping:
                used = {gpu for item in active.values() for gpu in item["gpus"]}
                free = [gpu for gpu in range(args.gpus) if gpu not in used]
                tasks = [json.loads(path.read_text()) for path in queue.glob("*.json")]
                for task in sorted(
                    tasks, key=lambda item: (item.get("priority", 100), item["id"])
                ):
                    if stopping:
                        break
                    task_id = task["id"]
                    if task_id in seen:
                        continue
                    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", task_id):
                        raise ValueError(f"Invalid task id: {task_id}")
                    output = runs / task_id
                    existing = output / "execution.json"
                    prior = check_recovery(existing, supervisor)
                    if prior is not None and prior["status"] in {
                        "completed",
                        "failed",
                        "blocked",
                    }:
                        seen.add(task_id)
                        continue
                    dependency_states = {}
                    prerequisites = (
                        []
                        if task.get("lane") in {"runtime", "qualification"}
                        else runtime_gates
                    )
                    for dependency in dict.fromkeys(
                        [*prerequisites, *task.get("depends_on", [])]
                    ):
                        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", dependency):
                            raise ValueError(f"Invalid dependency id: {dependency}")
                        receipt = runs / dependency / "execution.json"
                        dependency_states[dependency] = (
                            json.loads(receipt.read_text()).get("status")
                            if receipt.exists()
                            else None
                        )
                    if any(
                        state in {"failed", "blocked"}
                        for state in dependency_states.values()
                    ):
                        output.mkdir(parents=True, exist_ok=True)
                        publish_receipt(
                            existing,
                            {
                                "task_id": task_id,
                                "status": "blocked",
                                "dependencies": dependency_states,
                                "observed_at": now(),
                            },
                        )
                        seen.add(task_id)
                        continue
                    if any(
                        state != "completed" for state in dependency_states.values()
                    ):
                        continue
                    needed = task.get("gpus", 1)
                    if needed < 1 or needed > args.gpus:
                        raise ValueError(f"Invalid GPU count for {task_id}: {needed}")
                    if needed > len(free):
                        break
                    gpus, free = free[:needed], free[needed:]
                    output.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    for variable in (
                        "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "GPU_DEVICE_ORDINAL",
                    ):
                        env.pop(variable, None)
                    env["ROCR_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                    env["PYTHONUNBUFFERED"] = "1"
                    if stopping:
                        break
                    command = [
                        "uv",
                        "run",
                        "--no-project",
                        "--python",
                        sys.executable,
                        "python",
                        str(Path(__file__).with_name("run_task.py")),
                        "--task",
                        str(output / "task.json"),
                        "--output-dir",
                        str(output),
                    ]
                    receipt = {
                        "task_id": task_id,
                        "attempt_id": uuid4().hex,
                        "task": task,
                        "task_sha256": hashlib.sha256(json_bytes(task)).hexdigest(),
                        "config_sha256": hashlib.sha256(
                            json_bytes(task.get("config"))
                        ).hexdigest(),
                        "source_sha256": task.get("source_sha256"),
                        "started_at": now(),
                        "status": "running",
                        "gpus": gpus,
                        "pid": None,
                        "pod": os.environ.get("POD_NAME"),
                        "supervisor": supervisor,
                        "runtime_sha256": runtime_hash,
                    }
                    if prior is not None:
                        receipt["resumed_from"] = str(
                            archive_path(existing, existing.read_bytes()).relative_to(
                                output
                            )
                        )
                    publish_receipt(existing, receipt)
                    atomic_json(output / "task.json", task)
                    deadline = time.monotonic() + task.get("max_seconds", 7200)
                    log = (output / "run.log").open("a", buffering=1)
                    try:
                        process = subprocess.Popen(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except Exception:
                        log.close()
                        logger.exception(
                            "WORKER_LAUNCH_FAILED: %s; retaining unfinished receipt",
                            task_id,
                        )
                        raise
                    active[task_id] = {
                        "process": process,
                        "gpus": gpus,
                        "log": log,
                        "receipt": receipt,
                        "deadline": deadline,
                    }
                    if stopping:
                        signal_group(process, signal.SIGTERM)
                    receipt["pid"] = process.pid
                    publish_receipt(existing, receipt)
                    seen.add(task_id)
                    logger.info("Started %s on GPU(s) %s", task_id, gpus)
                    idle_since = time.monotonic()

            atomic_json(
                root / "supervisor.json",
                {
                    "observed_at": now(),
                    "pod": os.environ.get("POD_NAME"),
                    "active": {key: value["receipt"] for key, value in active.items()},
                    "seen": sorted(seen),
                    "stopping": bool(stopping),
                },
            )
            if stopping:
                if not active:
                    raise SystemExit(128 + stopping[1])
                if time.monotonic() - stopping[0] > 50:
                    for item in active.values():
                        signal_group(item["process"], signal.SIGKILL)
            elif not active and time.monotonic() - idle_since > args.idle_timeout:
                logger.info("Queue idle: releasing node")
                return
            time.sleep(2)
    finally:
        if active:
            logger.error(
                "SUPERVISOR_ABORT: stopping owned workers; retaining unfinished receipts"
            )
            for item in active.values():
                if item["process"].poll() is None:
                    signal_group(item["process"], signal.SIGKILL)
            for item in active.values():
                item["process"].wait(timeout=5)
                item["log"].close()
        ownership.close()


if __name__ == "__main__":
    main()
