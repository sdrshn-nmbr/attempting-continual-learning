import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


class SupervisorIntegrationTests(unittest.TestCase):
    def prepare_probe(self, root):
        (root / "runtime-gates.json").write_text("[]")
        (root / "queue").mkdir()
        code = root / "code"
        code.mkdir()
        (code / "runtime-requirements.txt").write_text("")
        shutil.copyfile(
            Path(__file__).with_name("supervisor.py"), code / "supervisor.py"
        )
        (code / "run_task.py").write_text(
            textwrap.dedent("""
            import argparse
            import hashlib
            import json
            import os
            import time
            from pathlib import Path
            parser = argparse.ArgumentParser()
            parser.add_argument("--task", type=Path)
            parser.add_argument("--output-dir", type=Path)
            args = parser.parse_args()
            task = json.loads(args.task.read_text())
            raw = (args.output_dir / "execution.json").read_bytes()
            receipt = json.loads(raw)
            archive = args.output_dir / "attempts" / receipt["attempt_id"] / (hashlib.sha256(raw).hexdigest() + ".json")
            assert archive.read_bytes() == raw
            assert receipt["status"] == "running"
            assert receipt["task"] == task
            with (args.output_dir.parent.parent / "worker-launches").open("a") as launches:
                launches.write(task["id"] + "\\n")
            (args.output_dir / "ready").write_text(str(os.getpid()))
            while task.get("hold") and not (args.output_dir / "release").exists():
                (args.output_dir / "heartbeat").write_text(str(time.monotonic()))
                time.sleep(0.05)
            (args.output_dir / "work.json").write_text(json.dumps({
                "sum_squares": sum(value * value for value in range(10000)),
                "gpus": os.environ["ROCR_VISIBLE_DEVICES"],
                "pid": os.getpid(),
            }))
        """)
        )
        return [
            sys.executable,
            str(code / "supervisor.py"),
            "--root",
            str(root),
            "--gpus",
            "1",
            "--idle-timeout",
            "0",
        ]

    def history(self, output):
        return {
            path.relative_to(output): path.read_bytes()
            for path in (output / "attempts").glob("*/*.json")
        }

    def test_invalid_supervisor_capacity_creates_no_queue_state(self):
        for count in (0, 9):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "unused-root"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("supervisor.py")),
                        "--root",
                        str(root),
                        "--gpus",
                        str(count),
                    ],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr.decode())
                self.assertIn(b"INVALID_GPU_COUNT", result.stderr)
                self.assertFalse(root.exists())

    def test_invalid_gpu_count_does_not_bypass_receipt_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.prepare_probe(root)
            for task in ({"id": "healthy"}, {"id": "invalid", "gpus": 0}):
                (root / "queue" / f"{task['id']}.json").write_text(json.dumps(task))
            output = root / "runs/invalid"
            output.mkdir(parents=True)
            receipt = output / "execution.json"
            receipt.write_bytes(b"{")
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"RECEIPT_INTEGRITY", result.stderr)
            self.assertEqual(receipt.read_bytes(), b"{")
            self.assertFalse((root / "worker-launches").exists())
            self.assertFalse((root / "runs/healthy").exists())
            self.assertEqual(self.history(output), {})

    def test_invalid_gpu_tasks_leave_active_worker_and_healthy_queue_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.prepare_probe(root)
            (root / "queue/active.json").write_text(
                json.dumps({"id": "active", "gpus": 1, "hold": True})
            )
            owner = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            active = root / "runs/active"
            invalid = {}
            try:
                deadline = time.monotonic() + 20
                while not (active / "ready").exists():
                    if owner.poll() is not None or time.monotonic() > deadline:
                        self.fail("Resource isolation probe did not start its worker")
                    time.sleep(0.05)
                running = (active / "execution.json").read_bytes()
                for index, count in enumerate((0, -1, 9, 2, True, 1.0, "1", None)):
                    task_id = f"invalid-{index}"
                    task = {
                        "id": task_id,
                        "gpus": count,
                        "priority": 0,
                        "depends_on": [] if index == 0 else ["missing-prerequisite"],
                        "config": {"probe": index},
                    }
                    invalid[task_id] = task
                    staged = root / "queue" / f"{task_id}.staged"
                    staged.write_text(json.dumps(task))
                    staged.rename(staged.with_suffix(".json"))
                for task in (
                    {"id": "dependent", "depends_on": ["invalid-0"]},
                    {"id": "healthy", "gpus": 1},
                ):
                    staged = root / "queue" / f"{task['id']}.staged"
                    staged.write_text(json.dumps(task))
                    staged.rename(staged.with_suffix(".json"))
                deadline = time.monotonic() + 10
                while not all(
                    (root / "runs" / name / "execution.json").exists()
                    for name in invalid
                ):
                    if owner.poll() is not None or time.monotonic() > deadline:
                        self.fail("Invalid resources aborted or stalled the supervisor")
                    time.sleep(0.05)
                self.assertIsNone(owner.poll())
                self.assertEqual((active / "execution.json").read_bytes(), running)
                for task_id, task in invalid.items():
                    output = root / "runs" / task_id
                    receipt = (output / "execution.json").read_bytes()
                    record = json.loads(receipt)
                    self.assertEqual(record["status"], "blocked")
                    self.assertEqual(record["blocked_reason"], "INVALID_GPU_COUNT")
                    self.assertEqual(record["task"], task)
                    self.assertEqual(record["requested_gpus"], task["gpus"])
                    self.assertEqual(record["available_gpus"], 1)
                    self.assertEqual(record["supervisor"]["pid"], owner.pid)
                    self.assertEqual(
                        record["task_sha256"],
                        hashlib.sha256(
                            (json.dumps(task, indent=2) + "\n").encode()
                        ).hexdigest(),
                    )
                    self.assertNotIn("started_at", record)
                    self.assertNotIn("exit_code", record)
                    self.assertNotIn("pid", record)
                    self.assertFalse((output / "ready").exists())
                    self.assertIn(receipt, self.history(output).values())
                    for path, content in self.history(output).items():
                        self.assertEqual(path.stem, hashlib.sha256(content).hexdigest())
                        self.assertEqual((output / path).stat().st_mode & 0o222, 0)
                (active / "release").write_text("finish the computation")
                _, stderr = owner.communicate(timeout=15)
                self.assertEqual(owner.returncode, 0, stderr.decode())
                self.assertIn(b"INVALID_GPU_COUNT", stderr)
                self.assertNotIn(b"SUPERVISOR_ABORT", stderr)
                for task_id in ("active", "healthy"):
                    output = root / "runs" / task_id
                    record = json.loads((output / "execution.json").read_bytes())
                    observed = json.loads((output / "work.json").read_bytes())
                    self.assertEqual(record["status"], "completed")
                    self.assertEqual(record["exit_code"], 0)
                    self.assertEqual(observed["sum_squares"], 333283335000)
                    self.assertEqual(observed["gpus"], "0")
                    self.assertEqual(
                        observed["pid"], int((output / "ready").read_text())
                    )
                    if task_id == "active":
                        self.assertEqual(
                            record["attempt_id"], json.loads(running)["attempt_id"]
                        )
                self.assertIn(running, self.history(active).values())
                dependent = json.loads(
                    (root / "runs/dependent/execution.json").read_bytes()
                )
                self.assertEqual(dependent["status"], "blocked")
                self.assertEqual(dependent["dependencies"], {"invalid-0": "blocked"})
                self.assertEqual(
                    (root / "worker-launches").read_text(), "active\nhealthy\n"
                )
                before = {
                    path.relative_to(root): path.read_bytes()
                    for path in (root / "runs").rglob("*.json")
                }
                (root / "queue/invalid-0.json").write_text(
                    json.dumps({**invalid["invalid-0"], "gpus": 1, "depends_on": []})
                )
                subprocess.run(command, capture_output=True, check=True, timeout=10)
                self.assertEqual(
                    before,
                    {
                        path.relative_to(root): path.read_bytes()
                        for path in (root / "runs").rglob("*.json")
                    },
                )
                self.assertEqual(
                    (root / "worker-launches").read_text(), "active\nhealthy\n"
                )
            finally:
                if owner.poll() is None:
                    owner.send_signal(signal.SIGTERM)
                owner.communicate(timeout=10)

    def test_sigterm_during_process_launch_stops_new_child_and_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime-gates.json").write_text("[]")
            code = root / "code"
            code.mkdir()
            (code / "runtime-requirements.txt").write_text("")
            shutil.copyfile(
                Path(__file__).with_name("supervisor.py"), code / "supervisor.py"
            )
            (code / "run_task.py").write_text("import time\ntime.sleep(60)\n")
            (code / "launch_probe.py").write_text(
                textwrap.dedent("""
                import hashlib
                import json
                import os
                import signal
                import subprocess
                from pathlib import Path
                from unittest.mock import patch
                import supervisor
                launch = subprocess.Popen
                def interrupted_launch(*args, **kwargs):
                    output = Path(args[0][-1])
                    raw = (output / "execution.json").read_bytes()
                    receipt = json.loads(raw)
                    assert receipt["status"] == "running"
                    assert receipt["pid"] is None
                    archive = output / "attempts" / receipt["attempt_id"] / (hashlib.sha256(raw).hexdigest() + ".json")
                    assert archive.read_bytes() == raw
                    process = launch(*args, **kwargs)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return process
                with patch.object(supervisor.subprocess, "Popen", interrupted_launch):
                    supervisor.main()
            """)
            )
            queue = root / "queue"
            queue.mkdir()
            for task_id in ("a", "b"):
                (queue / f"{task_id}.json").write_text(json.dumps({"id": task_id}))
            command = [
                sys.executable,
                str(code / "launch_probe.py"),
                "--root",
                str(root),
                "--gpus",
                "2",
                "--idle-timeout",
                "0",
            ]
            result = subprocess.run(
                command, capture_output=True, timeout=10, check=False
            )
            self.assertEqual(result.returncode, 143, result.stderr.decode())
            receipt = json.loads((root / "runs/a/execution.json").read_text())
            self.assertEqual(receipt["status"], "interrupted")
            self.assertEqual(receipt["exit_code"], -signal.SIGTERM)
            self.assertFalse((root / "runs/b/execution.json").exists())

    def test_single_owner_and_sigterm_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime-gates.json").write_text("[]")
            code = root / "code"
            code.mkdir()
            (code / "runtime-requirements.txt").write_text("")
            shutil.copyfile(
                Path(__file__).with_name("supervisor.py"), code / "supervisor.py"
            )
            (code / "run_task.py").write_text(
                textwrap.dedent("""
                import argparse
                import signal
                import time
                from pathlib import Path
                parser = argparse.ArgumentParser()
                parser.add_argument("--task", type=Path)
                parser.add_argument("--output-dir", type=Path)
                args = parser.parse_args()
                checkpoint = args.output_dir / "checkpoint"
                def stop(signum, frame):
                    checkpoint.write_text("committed")
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, stop)
                if checkpoint.exists():
                    (args.output_dir / "resumed").write_text(checkpoint.read_text())
                else:
                    (args.output_dir / "ready").touch()
                    while True:
                        time.sleep(0.1)
            """)
            )
            queue = root / "queue"
            queue.mkdir()
            task = {
                "id": "a",
                "gpus": 1,
                "source_sha256": "b" * 64,
                "code_dir": str(code),
                "config": {"seed": 19},
            }
            (queue / "a.json").write_text(json.dumps(task))
            command = [
                sys.executable,
                str(code / "supervisor.py"),
                "--root",
                str(root),
                "--gpus",
                "1",
                "--idle-timeout",
                "0",
            ]
            owner = subprocess.Popen(
                command,
                env=dict(os.environ, POD_NAME="pod-a"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                deadline = time.monotonic() + 20
                while not (root / "runs/a/ready").exists():
                    if owner.poll() is not None or time.monotonic() > deadline:
                        self.fail("Supervisor worker did not become ready")
                    time.sleep(0.05)
                duplicate = subprocess.run(
                    command, capture_output=True, timeout=10, check=False
                )
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertIn(b"Another supervisor already owns", duplicate.stderr)
                output = root / "runs/a"
                running = (output / "execution.json").read_bytes()
                owner.send_signal(signal.SIGTERM)
                _, stderr = owner.communicate(timeout=15)
                self.assertEqual(owner.returncode, 143, stderr.decode())
                receipt = root / "runs/a/execution.json"
                interrupted = receipt.read_bytes()
                interrupted_record = json.loads(interrupted)
                self.assertEqual(
                    json.loads(receipt.read_text())["status"], "interrupted"
                )
                self.assertEqual((root / "runs/a/checkpoint").read_text(), "committed")
                history = self.history(output)
                self.assertIn(running, history.values())
                self.assertIn(interrupted, history.values())
                subprocess.run(
                    command,
                    env=dict(os.environ, POD_NAME="pod-b"),
                    capture_output=True,
                    check=True,
                    timeout=15,
                )
                self.assertEqual(json.loads(receipt.read_text())["status"], "completed")
                self.assertEqual((root / "runs/a/resumed").read_text(), "committed")
                completed = json.loads(receipt.read_bytes())
                self.assertNotEqual(
                    completed["attempt_id"], interrupted_record["attempt_id"]
                )
                self.assertNotEqual(
                    completed["supervisor"]["id"],
                    interrupted_record["supervisor"]["id"],
                )
                self.assertEqual(interrupted_record["pod"], "pod-a")
                self.assertEqual(completed["pod"], "pod-b")
                self.assertEqual(completed["task"], task)
                self.assertEqual(completed["source_sha256"], task["source_sha256"])
                self.assertEqual(
                    completed["task_sha256"],
                    hashlib.sha256((output / "task.json").read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    completed["config_sha256"],
                    hashlib.sha256(b'{\n  "seed": 19\n}\n').hexdigest(),
                )
                self.assertEqual(
                    completed["runtime_sha256"],
                    hashlib.sha256(
                        (code / "supervisor.py").read_bytes()
                        + (code / "run_task.py").read_bytes()
                        + (code / "runtime-requirements.txt").read_bytes()
                    ).hexdigest(),
                )
                self.assertEqual(
                    (output / completed["resumed_from"]).read_bytes(), interrupted
                )
                after = self.history(output)
                self.assertIn(receipt.read_bytes(), after.values())
                for path, content in history.items():
                    self.assertEqual(after[path], content)
                for path, content in after.items():
                    self.assertEqual(path.stem, hashlib.sha256(content).hexdigest())
                    self.assertEqual((output / path).stat().st_mode & 0o222, 0)
            finally:
                if owner.poll() is None:
                    owner.kill()
                owner.communicate(timeout=5)

    def test_gpu_exclusion_dependency_failure_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime-gates.json").write_text('["a"]')
            code = root / "code"
            code.mkdir()
            (code / "runtime-requirements.txt").write_text("")
            shutil.copyfile(
                Path(__file__).with_name("supervisor.py"), code / "supervisor.py"
            )
            (code / "run_task.py").write_text(
                textwrap.dedent("""
                import argparse
                import json
                import os
                import time
                from pathlib import Path
                parser = argparse.ArgumentParser()
                parser.add_argument("--task", type=Path)
                parser.add_argument("--output-dir", type=Path)
                args = parser.parse_args()
                task = json.loads(args.task.read_text())
                start = time.time()
                time.sleep(0.15)
                receipt = {"start": start, "end": time.time(), "gpus": os.environ["ROCR_VISIBLE_DEVICES"], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "hip": os.environ.get("HIP_VISIBLE_DEVICES")}
                (args.output_dir / "observed.json").write_text(json.dumps(receipt))
                raise SystemExit(task.get("exit_code", 0))
            """)
            )
            queue = root / "queue"
            queue.mkdir()
            tasks = [
                {"id": "a", "gpus": 1, "lane": "runtime"},
                {"id": "b", "gpus": 1, "exit_code": 3, "lane": "qualification"},
                {"id": "c", "gpus": 2, "depends_on": ["a"]},
                {"id": "d", "gpus": 1, "depends_on": ["b"]},
            ]
            for task in tasks:
                (queue / (task["id"] + ".json")).write_text(json.dumps(task))
            environment = dict(
                os.environ, CUDA_VISIBLE_DEVICES="7", HIP_VISIBLE_DEVICES="7"
            )
            command = [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(code / "supervisor.py"),
                "--root",
                str(root),
                "--gpus",
                "2",
                "--idle-timeout",
                "0",
            ]
            subprocess.run(
                command, env=environment, check=True, capture_output=True, timeout=30
            )
            receipts = {
                name: json.loads((root / "runs" / name / "execution.json").read_text())
                for name in ("a", "b", "c", "d")
            }
            self.assertEqual(
                [receipts[name]["status"] for name in ("a", "b", "c", "d")],
                ["completed", "failed", "completed", "blocked"],
            )
            observed = {
                name: json.loads((root / "runs" / name / "observed.json").read_text())
                for name in ("a", "b", "c")
            }
            self.assertNotEqual(observed["a"]["gpus"], observed["b"]["gpus"])
            self.assertEqual(observed["c"]["gpus"], "0,1")
            self.assertGreaterEqual(
                observed["c"]["start"], max(observed["a"]["end"], observed["b"]["end"])
            )
            self.assertTrue(
                all(
                    item["cuda"] is None and item["hip"] is None
                    for item in observed.values()
                )
            )
            before = {
                name: (root / "runs" / name / "execution.json").read_bytes()
                for name in receipts
            }
            subprocess.run(
                command, env=environment, check=True, capture_output=True, timeout=15
            )
            for name, content in before.items():
                self.assertEqual(
                    content, (root / "runs" / name / "execution.json").read_bytes()
                )

    def test_stale_running_on_another_pod_holds_entire_queue_across_restarts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.prepare_probe(root)
            for task_id in ("a-fresh", "z-stale"):
                (root / "queue" / f"{task_id}.json").write_text(
                    json.dumps({"id": task_id})
                )
            output = root / "runs/z-stale"
            output.mkdir(parents=True)
            receipt = output / "execution.json"
            prior = (
                json.dumps(
                    {
                        "task_id": "z-stale",
                        "status": "running",
                        "started_at": "2026-09-07T01:00:00+00:00",
                        "pod": "pod-a",
                        "pid": 8675309,
                        "runtime_sha256": "a" * 64,
                        "scientific_failure": {
                            "reason": "model-cache-missing",
                            "checkpoint": "unverified",
                        },
                    },
                    indent=3,
                )
                + "\r\n"
            ).encode()
            receipt.write_bytes(prior)
            evidence = {
                output / "task.json": b'{"id":"z-stale","config":{"seed":23}}\n',
                output / "run.log": b"Original scientific failure evidence\n",
                output / "failure.json": b'{"failure":"must remain intact"}\n',
            }
            for path, content in evidence.items():
                path.write_bytes(content)
            for terminal in ("completed", "failed", "blocked"):
                path = root / "runs" / terminal / "execution.json"
                path.parent.mkdir()
                evidence[path] = json.dumps(
                    {"task_id": terminal, "status": terminal}
                ).encode()
                path.write_bytes(evidence[path])
            result = subprocess.run(
                command,
                env=dict(os.environ, POD_NAME="pod-b"),
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"UNCLEAN_EXECUTION_BLOCKED", result.stderr)
            blocked = receipt.read_bytes()
            record = json.loads(blocked)
            self.assertEqual(record["status"], "blocked")
            self.assertEqual(record["blocked_reason"], "unclean_supervisor_termination")
            self.assertEqual(record["pod"], "pod-a")
            self.assertEqual(record["recovery"]["supervisor"]["pod"], "pod-b")
            self.assertEqual(
                record["scientific_failure"], json.loads(prior)["scientific_failure"]
            )
            history = self.history(output)
            self.assertIn(prior, history.values())
            self.assertIn(blocked, history.values())
            self.assertEqual(len({path.parent for path in history}), 1)
            self.assertFalse((root / "worker-launches").exists())
            self.assertFalse((root / "runs/a-fresh").exists())
            (root / "queue/z-stale.json").unlink()
            result = subprocess.run(
                command, capture_output=True, timeout=10, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"UNCLEAN_EXECUTION_BLOCKED", result.stderr)
            self.assertEqual(receipt.read_bytes(), blocked)
            self.assertEqual(self.history(output), history)
            self.assertFalse((root / "worker-launches").exists())
            for path, content in evidence.items():
                self.assertEqual(path.read_bytes(), content)

    def test_sigkill_same_pod_with_live_orphan_holds_all_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.prepare_probe(root)
            (root / "queue/z-orphan.json").write_text(
                json.dumps({"id": "z-orphan", "hold": True})
            )
            environment = dict(os.environ, POD_NAME="same-pod")
            owner = subprocess.Popen(
                command,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            group = None
            output = root / "runs/z-orphan"
            receipt = output / "execution.json"
            try:
                deadline = time.monotonic() + 20
                while True:
                    if receipt.exists():
                        group = json.loads(receipt.read_bytes())["pid"]
                    if (output / "ready").exists() and group is not None:
                        break
                    if owner.poll() is not None or time.monotonic() > deadline:
                        self.fail("Abrupt-death probe worker did not become ready")
                    time.sleep(0.05)
                prior = receipt.read_bytes()
                self.assertEqual(json.loads(prior)["supervisor"]["pid"], owner.pid)
                owner.kill()
                owner.communicate(timeout=5)
                self.assertEqual(owner.returncode, -signal.SIGKILL)
                (root / "queue/a-fresh.json").write_text(
                    json.dumps({"id": "a-fresh", "priority": 0})
                )
                result = subprocess.run(
                    command,
                    env=environment,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"UNCLEAN_EXECUTION_BLOCKED", result.stderr)
                self.assertIn(prior, self.history(output).values())
                record = json.loads(receipt.read_bytes())
                self.assertEqual(record["status"], "blocked")
                self.assertEqual(record["attempt_id"], json.loads(prior)["attempt_id"])
                self.assertEqual(record["pod"], "same-pod")
                self.assertEqual(record["recovery"]["supervisor"]["pod"], "same-pod")
                self.assertNotEqual(
                    record["supervisor"]["id"], record["recovery"]["supervisor"]["id"]
                )
                self.assertEqual((root / "worker-launches").read_text(), "z-orphan\n")
                self.assertFalse((root / "runs/a-fresh").exists())
                heartbeat = (output / "heartbeat").read_bytes()
                deadline = time.monotonic() + 5
                while (output / "heartbeat").read_bytes() == heartbeat:
                    if time.monotonic() > deadline:
                        self.fail("Recovery signalled or stopped the orphan worker")
                    time.sleep(0.05)
            finally:
                if owner.poll() is None:
                    owner.send_signal(signal.SIGTERM)
                    owner.communicate(timeout=10)
                if group is not None:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                owner.communicate(timeout=5)

    def test_receipt_corruption_or_history_collision_refuses_all_dispatch(self):
        for problem in (
            "current",
            "collision",
            "older-history",
            "missing-current",
            "missing-predecessor",
            "missing-current-snapshot",
        ):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                command = self.prepare_probe(root)
                for task_id in ("a-fresh", "z-resume"):
                    (root / "queue" / f"{task_id}.json").write_text(
                        json.dumps({"id": task_id})
                    )
                output = root / "runs/z-resume"
                output.mkdir(parents=True)
                receipt = output / "execution.json"
                record = {
                    "task_id": "z-resume",
                    "status": "interrupted",
                    "attempt_id": "a" * 32,
                }
                if problem == "missing-predecessor":
                    record["previous_receipt"] = (
                        "attempts/" + "a" * 32 + "/" + "f" * 64 + ".json"
                    )
                elif problem == "missing-current-snapshot":
                    record["previous_receipt"] = None
                prior = json.dumps(record).encode()
                receipt.write_bytes(b"{" if problem == "current" else prior)
                if problem != "current":
                    history = output / "attempts" / ("a" * 32)
                    history.mkdir(parents=True)
                    archived = history / (hashlib.sha256(prior).hexdigest() + ".json")
                    archived.write_bytes(
                        b"corrupt" if problem == "collision" else prior
                    )
                    if problem == "older-history":
                        archived.rename(history / ("b" * 64 + ".json"))
                    if problem == "missing-current":
                        receipt.unlink()
                    if problem == "missing-current-snapshot":
                        archived.unlink()
                before = receipt.read_bytes() if receipt.exists() else None
                history_before = self.history(output)
                result = subprocess.run(
                    command, capture_output=True, timeout=10, check=False
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"RECEIPT_INTEGRITY", result.stderr)
                self.assertEqual(
                    receipt.read_bytes() if receipt.exists() else None, before
                )
                self.assertEqual(self.history(output), history_before)
                self.assertFalse((root / "worker-launches").exists())
                self.assertFalse((root / "runs/a-fresh").exists())

    def test_publication_failure_after_popen_reaps_owned_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.prepare_probe(root)
            (root / "queue/a.json").write_text(json.dumps({"id": "a", "hold": True}))
            probe = root / "code/launch_probe.py"
            probe.write_text(
                textwrap.dedent("""
                import json
                from unittest.mock import patch
                import supervisor
                publish = supervisor.atomic_bytes
                def failed_pid_publication(path, content, **kwargs):
                    if path.name == "execution.json" and json.loads(content).get("pid"):
                        (path.parent / "launched-pid").write_text(str(json.loads(content)["pid"]))
                        raise OSError("Injected receipt publication failure")
                    publish(path, content, **kwargs)
                with patch.object(supervisor, "atomic_bytes", failed_pid_publication):
                    supervisor.main()
            """)
            )
            command[1] = str(probe)
            result = subprocess.run(
                command, capture_output=True, timeout=10, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Injected receipt publication failure", result.stderr)
            self.assertIn(b"SUPERVISOR_ABORT", result.stderr)
            output = root / "runs/a"
            pid = int((output / "launched-pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            prior = (output / "execution.json").read_bytes()
            self.assertEqual(json.loads(prior)["status"], "running")
            self.assertIn(prior, self.history(output).values())
            command[1] = str(root / "code/supervisor.py")
            result = subprocess.run(
                command, capture_output=True, timeout=10, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"UNCLEAN_EXECUTION_BLOCKED", result.stderr)
            self.assertIn(prior, self.history(output).values())


if __name__ == "__main__":
    unittest.main()
