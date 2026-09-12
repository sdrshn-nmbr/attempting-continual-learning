import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("campaign")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--idle-seconds", type=int, default=1800)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [campaign] %(message)s")
    queue = args.root / "queue"
    receipts = args.root / "receipts"
    logs = args.root / "logs"
    for directory in (queue, receipts, logs):
        directory.mkdir(parents=True, exist_ok=True)
    active = {}
    stopping = False
    last_work = time.monotonic()

    def stop(signum, frame):
        nonlocal stopping
        logger.warning(
            "Signal %s: forwarding termination to running experiments", signum
        )
        stopping = True
        for task in active.values():
            os.killpg(task["process"].pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for path in receipts.glob("*.json"):
        previous = json.loads(path.read_text())
        if previous["status"] == "running":
            previous.update(status="interrupted", finished_at=now())
            write_json(path, previous)
    while True:
        for task_id, task in list(active.items()):
            process = task["process"]
            if process.poll() is None and time.monotonic() > task["deadline"]:
                logger.error("Experiment %s exceeded its deadline", task_id)
                os.killpg(process.pid, signal.SIGTERM)
                task["timed_out"] = True
                task["deadline"] = float("inf")
                task["kill_at"] = time.monotonic() + 45
            if process.poll() is None and time.monotonic() > task.get(
                "kill_at", float("inf")
            ):
                os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.poll()
            if exit_code is None:
                continue
            result = task["receipt"]
            result.update(
                status="completed" if exit_code == 0 else "failed",
                exit_code=exit_code,
                timed_out=task.get("timed_out", False),
                finished_at=now(),
            )
            write_json(receipts / f"{task_id}.json", result)
            with (args.root / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            task["log"].close()
            del active[task_id]
            last_work = time.monotonic()
            logger.info("Experiment %s finished: %s", task_id, result["status"])
        occupied = {task["receipt"]["gpu"] for task in active.values()}
        pending = 0
        for path in sorted(queue.glob("*.json")):
            task_id = path.stem
            if (receipts / f"{task_id}.json").exists():
                continue
            pending += 1
            if stopping or (args.root / "STOP").exists():
                continue
            free = sorted(set(range(args.gpus)) - occupied)
            if not free:
                break
            raw = path.read_bytes()
            specification = json.loads(raw)
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", task_id):
                raise ValueError(f"Invalid experiment identifier: {task_id}")
            command = specification["command"]
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(x, str) for x in command)
            ):
                raise ValueError(f"Invalid command for experiment {task_id}")
            gpu = specification.get("gpu", free[0])
            if gpu not in free:
                continue
            environment = dict(os.environ)
            environment.pop("HIP_VISIBLE_DEVICES", None)
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment.update(specification.get("env", {}))
            environment["ROCR_VISIBLE_DEVICES"] = str(gpu)
            log = (logs / f"{task_id}.log").open("ab")
            process = subprocess.Popen(
                command,
                cwd=specification["cwd"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt = {
                "id": task_id,
                "lane": specification["lane"],
                "status": "running",
                "gpu": gpu,
                "pid": process.pid,
                "started_at": now(),
                "spec_sha256": hashlib.sha256(raw).hexdigest(),
                "specification": specification,
                "log": str(log.name),
            }
            write_json(receipts / f"{task_id}.json", receipt)
            active[task_id] = {
                "process": process,
                "receipt": receipt,
                "log": log,
                "deadline": time.monotonic()
                + specification.get("timeout_seconds", 3600),
            }
            occupied.add(gpu)
            last_work = time.monotonic()
            logger.info("Started %s on GPU %s, pid %s", task_id, gpu, process.pid)
        write_json(
            args.root / "heartbeat.json",
            {
                "observed_at": now(),
                "active": list(active),
                "pending": pending,
                "gpus": args.gpus,
                "stopping": stopping,
            },
        )
        if not active and (
            stopping or (args.root / "STOP").exists() or (args.root / "SEALED").exists()
        ):
            logger.info("Campaign queue drained; stopping worker")
            return
        if not active and time.monotonic() - last_work > args.idle_seconds:
            logger.warning("Campaign idle limit reached; stopping worker")
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
