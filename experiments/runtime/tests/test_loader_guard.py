"""CPU proof: real Linux process teardown with fake cgroup files and file I/O faults.

The suite does not simulate kernel D-state, an OOM killer, ROCm allocations, or
permanent loss of every artifact destination. It never creates memory pressure.
"""

import ctypes
import dataclasses
import hashlib
import json
import os
import random
import select
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import loader_guard as guard

PROOF_ROOT = Path(os.environ.get("LOADER_PROOF_ROOT", "."))
LIMIT = 1024 * 1024
TRIP = LIMIT // 2
ORIGINAL_SAMPLE_WORKER = guard._sample_worker


def wait_for(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("CPU_PROOF_DEADLINE")


def put(path, value):
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(str(value))
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text())


def injected_sample_worker(channel, root, pid, policy):
    original_send = guard._send
    scenario = read_json(root.parent / "spec.json")["scenario"]
    injected = False

    def send(destination, message):
        nonlocal injected
        if (
            scenario in {"premature_hash", "catalog_postload_bypass"}
            and message.get("kind") == "smaps"
            and message.get("stage") == "load_complete"
        ):
            return
        if (
            scenario == "catalog_preload_bypass"
            and message.get("kind") == "smaps"
            and message.get("stage") == "cache_verified"
        ):
            return
        if (
            scenario == "real_threshold_after_probe"
            and message.get("kind") == "sample"
            and not injected
            and (root.parent / "trip-after-probe").exists()
        ):
            events = root.parent / "attempt/events.jsonl"
            wait_for(
                lambda: any(
                    record.get("event") == "child_stage"
                    and record["stage"].get("event") == "probe_complete"
                    for record in (
                        json.loads(line) for line in events.read_text().splitlines()
                    )
                ),
                seconds=0.3,
            )
            injected = True
            unsafe = {
                **message,
                "sample": {
                    **message["sample"],
                    "current": TRIP,
                    "started": time.monotonic(),
                },
            }
            unsafe["sample"]["finished"] = time.monotonic()
            original_send(destination, unsafe)
            return
        if (
            message.get("kind") == "sample"
            and not injected
            and (root.parent / "inject-batch").exists()
        ):
            injected = True
            unsafe = {**message, "sample": dict(message["sample"])}
            changes = {
                "queued_current": {"current": TRIP},
                "queued_max": {"max": LIMIT + 1},
                "queued_stale": {
                    "started": message["sample"]["started"] - policy.stale_seconds - 1
                },
            }
            unsafe["sample"].update(changes[scenario])
            controller = os.getppid()
            os.kill(controller, signal.SIGSTOP)
            try:
                original_send(destination, unsafe)
                original_send(destination, message)
            finally:
                os.kill(controller, signal.SIGCONT)
            guard.write_once(
                root.parent / "injected-batch.json",
                {
                    "both_datagrams_sent_before_controller_resumed": True,
                    "samples": [unsafe["sample"], message["sample"]],
                    "real_memory_pressure_created": False,
                },
            )
            return
        original_send(destination, message)

    guard._send = send
    ORIGINAL_SAMPLE_WORKER(channel, root, pid, policy)


def catalog_fixture(spec):
    channel = guard.Channel(15)
    root = Path(spec["root"])
    scenario = spec["scenario"]
    guard.write_once(
        root / "child-ready.json",
        {"loader": guard.process_identity(os.getpid()), "children": []},
    )
    try:
        channel.send("cache_verified", fixture=True)
        if scenario != "catalog_preload_bypass":
            channel.wait_for_continue()
        channel.send("load_started")
        (root / "loaded").write_text(str(os.getpid()))
        channel.send("load_complete")
        if scenario not in {"catalog_preload_bypass", "catalog_postload_bypass"}:
            channel.wait_for_continue()
        if scenario == "catalog_wrong_profile_stage":
            channel.send("state_hash_started")
        elif scenario == "catalog_terminal_out_of_order":
            channel.send("evaluation_complete")
        else:
            channel.send("evaluation_started")
            if scenario == "catalog_missing_terminal":
                return
            if scenario == "catalog_reported_failure":
                path = root / "catalog-result.json"
                digest = guard.write_once(path, {"status": "failed"})
                channel.send(
                    "failed",
                    {"path": str(path), "sha256": digest},
                    failed_stage="evaluation_started",
                    reason="CPU_FIXTURE_PEFT_FAILURE",
                )
            elif scenario in {"catalog_success", "catalog_error_after_terminal"}:
                path = root / "catalog-result.json"
                digest = guard.write_once(path, {"status": "qualified"})
                channel.send(
                    "evaluation_complete", {"path": str(path), "sha256": digest}
                )
                if scenario == "catalog_success":
                    return
                events = root / "attempt/events.jsonl"

                def terminal_logged():
                    complete_lines = events.read_text().rsplit("\n", 1)[0]
                    return any(
                        record.get("event") == "child_stage"
                        and record["stage"].get("event") == "evaluation_complete"
                        for record in (
                            json.loads(line) for line in complete_lines.splitlines()
                        )
                    )

                wait_for(terminal_logged)
                raise RuntimeError("CPU_FIXTURE_FAILURE_AFTER_TERMINAL")
        while True:
            time.sleep(0.1)
    finally:
        channel.close()


def fixture(spec):
    if spec["stage_profile"] == "catalog":
        return catalog_fixture(spec)
    channel = socket.socket(fileno=int(os.environ["LOADER_GUARD_CHANNEL_FD"]))
    channel.settimeout(15)
    root = Path(spec["root"])
    if spec["scenario"] == "warm_baseline":
        put(root / "cgroup/memory.current", TRIP)
    channel.send(json.dumps({"event": "cache_verified", "fixture": True}).encode())
    assert json.loads(channel.recv(65536)) == {"command": "continue"}
    (root / "loaded").write_text(str(os.getpid()))
    children = []
    if spec["scenario"] in {
        "threshold",
        "controller_death",
        "controller_stall",
        "outer_term",
        "outer_kill",
        "natural_orphan",
        "intentional_trip",
    }:
        children.append(
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        )
        if spec["scenario"] in {
            "threshold",
            "controller_death",
            "outer_term",
            "outer_kill",
            "natural_orphan",
        }:
            children.append(
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    start_new_session=True,
                )
            )
    child_records = [guard.process_identity(child.pid) for child in children]
    (root / "child-ready.json").write_text(
        json.dumps(
            {"loader": guard.process_identity(os.getpid()), "children": child_records}
        )
    )
    channel.send(json.dumps({"event": "load_started"}).encode())
    scenario = spec["scenario"]
    if scenario == "threshold":
        put(root / "cgroup/memory.current", TRIP)
    elif scenario == "missing_read":
        (root / "cgroup/memory.current").unlink()
    elif scenario == "malformed_read":
        put(root / "cgroup/memory.current", "not-a-number")
    elif scenario == "changed_limit":
        put(root / "cgroup/memory.max", LIMIT + 1)
    elif scenario == "changed_cgroup_identity":
        original = root / "cgroup"
        saved = root / "prior-cgroup"
        original.rename(saved)
        original.mkdir()
        for path in saved.iterdir():
            (original / path.name).write_bytes(path.read_bytes())
    elif scenario == "reported_failure":
        channel.send(
            json.dumps(
                {
                    "event": "failed",
                    "failed_stage": "load_started",
                    "reason": "CPU_FIXTURE_FAILURE",
                }
            ).encode()
        )
    elif scenario == "bad_stage_order":
        channel.send(json.dumps({"event": "state_hash_complete"}).encode())
    elif scenario in {"queued_current", "queued_max", "queued_stale"}:
        (root / "inject-batch").touch()
    elif scenario == "blocked_read":
        path = root / "cgroup/memory.current"
        path.unlink()
        os.mkfifo(path)
    elif scenario == "blocked_diagnostics":
        path = root / "cgroup/memory.stat"
        path.unlink()
        os.mkfifo(path)
    elif scenario in {
        "success",
        "natural_orphan",
        "intentional_trip",
        "premature_hash",
        "real_threshold_after_probe",
    }:
        guard.write_once(
            root / "probe.json",
            {"status": "load_probe_complete", "process_complete": False},
        )
        channel.send(json.dumps({"event": "load_complete"}).encode())
        if scenario != "premature_hash":
            assert json.loads(channel.recv(65536)) == {"command": "continue"}
        for stage in ("state_hash_started", "state_hash_complete", "probe_started"):
            channel.send(json.dumps({"event": stage}).encode())
        if scenario == "real_threshold_after_probe":
            (root / "trip-after-probe").touch()
        channel.send(
            json.dumps(
                {"event": "probe_complete", "receipt": str(root / "probe.json")}
            ).encode()
        )
        if scenario not in {
            "intentional_trip",
            "premature_hash",
            "real_threshold_after_probe",
        }:
            return
    elif scenario == "child_error":
        raise RuntimeError("INJECTED_CHILD_ERROR")
    while True:
        time.sleep(0.1)


def run_case(spec):
    root = Path(spec["root"])
    if spec["scenario"] in {
        "queued_current",
        "queued_max",
        "queued_stale",
        "premature_hash",
        "real_threshold_after_probe",
        "catalog_preload_bypass",
        "catalog_postload_bypass",
    }:
        guard._sample_worker = injected_sample_worker
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(45)"], start_new_session=True
    )
    os.kill(foreign.pid, signal.SIGSTOP)
    foreign_before = guard.process_identity(foreign.pid)
    policy = guard.Policy(
        tripwire_bytes=TRIP,
        expected_max_bytes=LIMIT,
        stage_profile=spec["stage_profile"],
        poll_seconds=0.05,
        stale_seconds=0.45,
        logger_seconds=0.45,
        control_seconds=1.2,
        max_seconds=6,
        cleanup_seconds=2,
        detailed_seconds=0.1,
        trip_after_probe=spec["scenario"]
        in {"intentional_trip", "real_threshold_after_probe"},
        pod_uid="cpu-fixture-only",
    )

    def publish_launch(path, value):
        if spec["scenario"] == "launch_failure":
            raise OSError("INJECTED_LAUNCH_PUBLICATION_FAILURE")
        if spec["scenario"] == "launch_stall":
            (root / "publication-started").touch()
            with (root / "launch-fifo").open("wb", buffering=0) as sink:
                sink.write(b"ready")
        return guard.write_once(path, value)

    try:
        result = guard.run_guard(
            [sys.executable, __file__, "fixture", str(root / "spec.json")],
            root / "attempt",
            policy,
            cgroup_dir=root / "cgroup",
            event_sink=Path(spec["event_sink"]) if spec.get("event_sink") else None,
            launch_writer=publish_launch,
        )
        foreign_after = guard.process_identity(foreign.pid)
        guard.write_once(
            root / "outer-evidence.json",
            {
                "result": result,
                "foreign_before": foreign_before,
                "foreign_after": foreign_after,
                "foreign_alive": foreign.poll() is None,
            },
        )
    finally:
        foreign.kill()
        foreign.wait(timeout=3)


class PolicyTests(unittest.TestCase):
    def test_profile_is_explicit_and_probe_trip_cannot_target_catalog(self):
        with self.assertRaises(TypeError):
            guard.Policy(TRIP, LIMIT)
        with self.assertRaisesRegex(ValueError, "GUARD_INVALID_STAGE_PROFILE"):
            guard.Policy(TRIP, LIMIT, "arbitrary_protocol").validate()
        with self.assertRaisesRegex(
            ValueError, "GUARD_INTENTIONAL_PROBE_TRIP_REQUIRES_QUALIFICATION"
        ):
            guard.Policy(TRIP, LIMIT, "catalog", trip_after_probe=True).validate()

    def test_threshold_equality_and_total_memory_are_conservative(self):
        policy = guard.Policy(TRIP, LIMIT, "qualification")
        self.assertIsNone(
            guard.sample_failure({"current": TRIP - 1, "max": LIMIT}, policy)
        )
        self.assertEqual(
            guard.sample_failure({"current": TRIP, "max": LIMIT}, policy),
            "memory_tripwire",
        )
        self.assertEqual(
            guard.sample_failure({"current": TRIP + 1, "max": LIMIT}, policy),
            "memory_tripwire",
        )
        self.assertEqual(
            guard.sample_failure({"current": 0, "max": LIMIT + 1}, policy),
            "cgroup_limit_changed",
        )
        for seed in range(300):
            rng = random.Random(seed)
            current = rng.randrange(LIMIT)
            sample = {"current": current, "max": LIMIT, "stat": {"file": current}}
            self.assertEqual(
                guard.sample_failure(sample, policy),
                "memory_tripwire" if current >= TRIP else None,
            )

    def test_invalid_policy_refuses(self):
        for changes in (
            {"tripwire_bytes": LIMIT},
            {"poll_seconds": 0},
            {"stale_seconds": 0.01},
            {"max_seconds": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(
                    guard.Policy(TRIP, LIMIT, "qualification"), **changes
                ).validate()


class ChannelTests(unittest.TestCase):
    def test_connected_datagram_envelope_and_continue(self):
        peer, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        descriptor = child.detach()
        with patch.dict(os.environ, {"LOADER_GUARD_CHANNEL_FD": str(descriptor)}):
            channel = guard.Channel(1)
        try:
            channel.send(
                "evaluation_started",
                {"path": "/fixture/receipt.json", "sha256": "a" * 64},
            )
            message = json.loads(peer.recv(guard.MAX_MESSAGE_BYTES))
            self.assertEqual(message["event"], "evaluation_started")
            self.assertEqual(message["sequence"], 1)
            self.assertEqual(message["pid"], os.getpid())
            self.assertEqual(message["receipt_sha256"], "a" * 64)
            peer.send(b'{"command":"continue"}')
            channel.wait_for_continue()
            peer.send(b'{"command":"invalid"}')
            with self.assertRaisesRegex(RuntimeError, "GUARD_COMMAND_INVALID"):
                channel.wait_for_continue()
        finally:
            channel.close()
            peer.close()

    def test_channel_rejects_unguarded_or_stream_execution(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GUARD_CHANNEL_REQUIRED"):
                guard.Channel(1)
        peer, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        descriptor = child.detach()
        try:
            with patch.dict(os.environ, {"LOADER_GUARD_CHANNEL_FD": str(descriptor)}):
                with self.assertRaisesRegex(RuntimeError, "GUARD_CHANNEL_INVALID"):
                    guard.Channel(1)
        finally:
            peer.close()


@unittest.skipUnless(
    sys.platform == "linux", "requires real Linux pidfds, prctl and procfs"
)
class LinuxGuardTests(unittest.TestCase):
    def prepare(self, scenario, stage_profile="qualification"):
        root = PROOF_ROOT / self.id().split(".")[-1]
        root.mkdir(parents=True, exist_ok=False)
        cgroup = root / "cgroup"
        cgroup.mkdir()
        values = {
            "memory.current": 100,
            "memory.max": LIMIT,
            "memory.peak": 100,
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
            "memory.stat": "anon 20\nfile 80\nkernel 0\nfile_mapped 0\nshmem 0\nunevictable 0\nfile_dirty 0\nfile_writeback 0\n",
        }
        for name, value in values.items():
            (cgroup / name).write_text(str(value))
        spec = {"root": str(root), "scenario": scenario, "stage_profile": stage_profile}
        (root / "spec.json").write_text(json.dumps(spec))
        return root, spec

    def launch(self, root, spec):
        (root / "spec.json").write_text(json.dumps(spec))
        output = (root / "runner.log").open("xb")
        process = subprocess.Popen(
            [sys.executable, __file__, "case", str(root / "spec.json")],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(output.close)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        return process

    def finish(self, root, process, reasons):
        self.assertEqual(process.wait(timeout=12), 0, (root / "runner.log").read_text())
        result = read_json(root / "attempt/result.json")
        self.assertIn(result["reason"], reasons, result)
        self.assertTrue(result["cleanup"]["all_reaped"], result)
        evidence = read_json(root / "outer-evidence.json")
        self.assertTrue(evidence["foreign_alive"])
        before, after = evidence["foreign_before"], evidence["foreign_after"]
        self.assertEqual(
            (before["pid"], before["start_ticks"]), (after["pid"], after["start_ticks"])
        )
        self.assertEqual(after["state"], "T")
        self.assertNotIn(before["pid"], [r["pid"] for r in result["cleanup"]["reaped"]])
        frozen = (root / "attempt/result.json").read_bytes()
        with self.assertRaises(FileExistsError):
            guard.write_once(root / "attempt/result.json", {"corrupt": True})
        self.assertEqual((root / "attempt/result.json").read_bytes(), frozen)
        self.assertEqual(
            result["request_sha256"],
            hashlib.sha256((root / "attempt/request.json").read_bytes()).hexdigest(),
        )
        ready = root / "child-ready.json"
        if ready.exists():
            for record in [read_json(ready)["loader"], *read_json(ready)["children"]]:
                self.assertFalse(guard.identity_exists(record), record)
                self.assertFalse((Path("/proc") / str(record["pid"])).exists(), record)
                with self.assertRaises(ProcessLookupError):
                    os.kill(record["pid"], 0)
        return result

    def check_case(self, scenario, reasons, stage_profile="qualification"):
        root, spec = self.prepare(scenario, stage_profile)
        process = self.launch(root, spec)
        return root, self.finish(root, process, reasons)

    def test_success_and_no_output_overwrite(self):
        root, result = self.check_case("success", {"completed"})
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["baseline_after_hash"]["allowed"])
        self.assertGreaterEqual(
            result["baseline_after_hash"]["sample_started"],
            result["baseline_after_hash"]["verified_received"],
        )
        self.assertTrue(result["telemetry_log_complete"])
        self.assertEqual(
            set(result["milestone_smaps"]), {"cache_verified", "load_complete"}
        )
        for record in result["milestone_smaps"].values():
            self.assertIn("Rss:", record["raw"])
        events = [
            json.loads(line)
            for line in (root / "attempt/events.jsonl").read_text().splitlines()
        ]
        samples = [event for event in events if event["event"] == "memory_sample"]
        self.assertGreaterEqual(len(samples), 2)
        self.assertTrue(
            all(
                "current" in sample and "sample_started" in sample for sample in samples
            )
        )
        self.assertEqual(
            result["stage_sequence"],
            list(
                guard.LOAD_STAGES
                + guard.STAGE_PROFILES["qualification"]["post_load_stages"]
            ),
        )
        with self.assertRaises(FileExistsError):
            guard.run_guard(
                ["never-executed"],
                root / "attempt",
                guard.Policy(TRIP, LIMIT, "qualification"),
            )

    def test_catalog_completes_only_its_declared_protocol(self):
        root, result = self.check_case("catalog_success", {"completed"}, "catalog")
        self.assertEqual(result["stage_profile"], "catalog")
        self.assertEqual(result["terminal_stage"], "evaluation_complete")
        self.assertTrue(result["terminal_complete"])
        self.assertNotIn("probe_complete", result)
        self.assertTrue(result["load_continue_granted"])
        self.assertEqual(
            set(result["milestone_smaps"]), {"cache_verified", "load_complete"}
        )
        self.assertEqual(
            result["stage_sequence"],
            [
                "cache_verified",
                "load_started",
                "load_complete",
                "evaluation_started",
                "evaluation_complete",
            ],
        )
        self.assertTrue(result["telemetry_log_complete"])
        self.assertEqual(result["unacknowledged_log_sequences"], [])
        self.assertEqual(read_json(root / "catalog-result.json")["status"], "qualified")

    def test_catalog_cannot_bypass_before_load_barrier(self):
        _, result = self.check_case(
            "catalog_preload_bypass", {"controller_error"}, "catalog"
        )
        self.assertIn("GUARD_LOAD_BEFORE_BASELINE", result["detail"])
        self.assertFalse(result["terminal_complete"])

    def test_catalog_cannot_bypass_after_load_barrier(self):
        _, result = self.check_case(
            "catalog_postload_bypass", {"controller_error"}, "catalog"
        )
        self.assertIn("GUARD_POST_LOAD_BEFORE_OBSERVATION", result["detail"])
        self.assertFalse(result["load_continue_granted"])
        self.assertFalse(result["terminal_complete"])

    def test_catalog_clean_exit_without_terminal_fails(self):
        _, result = self.check_case(
            "catalog_missing_terminal", {"child_failed"}, "catalog"
        )
        self.assertFalse(result["terminal_complete"])

    def test_catalog_terminal_cannot_replace_evaluation_started(self):
        _, result = self.check_case(
            "catalog_terminal_out_of_order", {"controller_error"}, "catalog"
        )
        self.assertIn("GUARD_CHILD_STAGE_ORDER", result["detail"])
        self.assertFalse(result["terminal_complete"])

    def test_catalog_rejects_qualification_stage(self):
        _, result = self.check_case(
            "catalog_wrong_profile_stage", {"controller_error"}, "catalog"
        )
        self.assertIn("GUARD_CHILD_STAGE_ORDER", result["detail"])
        self.assertFalse(result["terminal_complete"])

    def test_catalog_evaluation_failure_cannot_be_success(self):
        root, result = self.check_case(
            "catalog_reported_failure", {"child_reported_failure"}, "catalog"
        )
        self.assertFalse(result["terminal_complete"])
        self.assertEqual(result["detail"]["reason"], "CPU_FIXTURE_PEFT_FAILURE")
        self.assertEqual(read_json(root / "catalog-result.json")["status"], "failed")

    def test_catalog_terminal_cannot_hide_failed_process_exit(self):
        _, result = self.check_case(
            "catalog_error_after_terminal", {"child_failed"}, "catalog"
        )
        self.assertTrue(result["terminal_complete"])
        self.assertNotEqual(result["child_exit_code"], 0)

    def test_after_hash_file_cache_at_limit_refuses_loading(self):
        root, result = self.check_case("warm_baseline", {"memory_tripwire"})
        self.assertFalse((root / "loaded").exists())
        self.assertFalse(result["baseline_after_hash"]["allowed"])
        self.assertIn(
            result["baseline_after_hash"]["phase"],
            {"after_hash", "before_hash_complete"},
        )

    def test_live_threshold_cleans_owned_and_escaped_descendants(self):
        _, result = self.check_case("threshold", {"memory_tripwire"})
        self.assertGreaterEqual(len(result["cleanup"]["reaped"]), 3)

    def test_required_read_failure(self):
        self.check_case("missing_read", {"telemetry_error"})

    def test_malformed_required_read(self):
        self.check_case("malformed_read", {"telemetry_error"})

    def test_changed_cgroup_limit_refuses(self):
        self.check_case("changed_limit", {"cgroup_limit_changed"})

    def test_queued_safe_current_cannot_erase_observed_trip(self):
        _, result = self.check_case("queued_current", {"memory_tripwire"})
        self.assertEqual(result["last_sample"]["current"], TRIP)

    def test_queued_safe_max_cannot_erase_observed_limit_change(self):
        _, result = self.check_case("queued_max", {"cgroup_limit_changed"})
        self.assertEqual(result["last_sample"]["max"], LIMIT + 1)

    def test_queued_fresh_sample_cannot_erase_observed_staleness(self):
        _, result = self.check_case("queued_stale", {"telemetry_stale"})
        self.assertEqual(result["first_violating_sample"], result["last_sample"])

    def test_real_threshold_after_probe_is_never_tagged_intentional(self):
        _, result = self.check_case("real_threshold_after_probe", {"memory_tripwire"})
        self.assertTrue(result["terminal_complete"])
        self.assertEqual(result["effective_tripwire_bytes"], TRIP)
        self.assertEqual(result["first_violating_sample"]["current"], TRIP)

    def test_ordered_hash_cannot_bypass_load_observation_gate(self):
        _, result = self.check_case("premature_hash", {"controller_error"})
        self.assertIn("GUARD_POST_LOAD_BEFORE_OBSERVATION", result["detail"])

    def test_changed_cgroup_identity_refuses_even_with_same_memory_values(self):
        self.check_case(
            "changed_cgroup_identity", {"cgroup_identity_changed", "telemetry_error"}
        )

    def test_out_of_order_child_stage_aborts(self):
        _, result = self.check_case("bad_stage_order", {"controller_error"})
        self.assertIn("GUARD_CHILD_STAGE_ORDER", result["detail"])

    def test_child_reported_failure_is_supported_and_terminal(self):
        _, result = self.check_case("reported_failure", {"child_reported_failure"})
        self.assertEqual(result["detail"]["reason"], "CPU_FIXTURE_FAILURE")

    def test_blocked_required_read_cannot_block_kill(self):
        self.check_case("blocked_read", {"telemetry_stale"})

    def test_blocked_diagnostics_cannot_block_kill(self):
        self.check_case("blocked_diagnostics", {"telemetry_stale"})

    def test_failed_logger_preserves_final_receipt(self):
        root, spec = self.prepare("hold")
        sink = root / "bad-sink"
        sink.mkdir()
        spec["event_sink"] = str(sink)
        self.finish(root, self.launch(root, spec), {"logger_error"})

    def test_blocked_logger_cannot_block_kill_or_final_receipt(self):
        root, spec = self.prepare("hold")
        sink = root / "blocked-sink"
        os.mkfifo(sink)
        spec["event_sink"] = str(sink)
        self.finish(root, self.launch(root, spec), {"logger_stale"})

    def test_actual_guard_parent_death_cleans_all_owned_processes(self):
        root, spec = self.prepare("controller_death")
        process = self.launch(root, spec)
        wait_for(lambda: (root / "child-ready.json").exists())
        launch = wait_for(
            lambda: (
                read_json(root / "attempt/launch.json")
                if (root / "attempt/launch.json").exists()
                else None
            )
        )
        child = read_json(root / "child-ready.json")["loader"]
        self.assertEqual(child["ppid"], launch["controller"]["pid"])
        os.kill(launch["controller"]["pid"], signal.SIGKILL)
        result = self.finish(root, process, {"controller_died"})
        self.assertTrue(result["parent_death_protection_armed"])
        self.assertTrue(
            any(
                r["pid"] == child["pid"] and r["signal"] == signal.SIGKILL
                for r in result["cleanup"]["reaped"]
            )
        )

    def test_stopped_controller_does_not_leave_loader_running(self):
        root, spec = self.prepare("controller_stall")
        process = self.launch(root, spec)
        wait_for(lambda: (root / "child-ready.json").exists())
        controller = read_json(root / "attempt/launch.json")["controller"]
        os.kill(controller["pid"], signal.SIGSTOP)
        self.finish(root, process, {"controller_stale"})

    def test_successful_child_cannot_leave_orphans(self):
        _, result = self.check_case("natural_orphan", {"unexpected_descendants"})
        self.assertGreaterEqual(len(result["unexpected_descendants"]), 2)

    def test_outer_sigterm_cleans_tree_and_writes_final_receipt(self):
        root, spec = self.prepare("outer_term")
        process = self.launch(root, spec)
        wait_for(lambda: (root / "child-ready.json").exists())
        os.kill(process.pid, signal.SIGTERM)
        self.finish(root, process, {"interrupted"})

    def test_outer_sigkill_boundary_is_recorded_by_surviving_test_parent(self):
        libc = ctypes.CDLL(None, use_errno=True)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        root, spec = self.prepare("outer_kill")
        process = self.launch(root, spec)
        wait_for(lambda: (root / "child-ready.json").exists())
        ready = read_json(root / "child-ready.json")
        request = read_json(root / "attempt/request.json")
        launch = read_json(root / "attempt/launch.json")
        controller_pid = launch["controller"]["pid"]
        foreign_pid = request["protected_children"][0]["pid"]
        targets = [ready["loader"], *ready["children"]]
        pinned = {record["pid"]: os.pidfd_open(record["pid"]) for record in targets}
        original = {
            name: (root / "attempt" / name).read_bytes()
            for name in ("request.json", "launch.json")
        }
        try:
            os.kill(process.pid, signal.SIGKILL)
            self.assertEqual(process.wait(timeout=3), -signal.SIGKILL)
            loader_fd = pinned[ready["loader"]["pid"]]
            wait_for(lambda: bool(select.select([loader_fd], [], [], 0)[0]))
            alive_descendants = [
                record["pid"]
                for record in ready["children"]
                if (Path("/proc") / str(record["pid"])).exists()
            ]
            self.assertEqual(
                set(alive_descendants), {record["pid"] for record in ready["children"]}
            )
            self.assertFalse((root / "attempt/result.json").exists())
            self.assertEqual(
                (Path("/proc") / str(foreign_pid) / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[0],
                "T",
            )
            guard.write_once(
                root / "outer-loss-observation.json",
                {
                    "author": "surviving_cpu_test_parent_not_the_killed_guard",
                    "killed_process": "top_level_outer_launcher",
                    "outer_pid": process.pid,
                    "controller_pid": controller_pid,
                    "loader_parent_death_observed": True,
                    "surviving_descendant_pids_before_test_cleanup": alive_descendants,
                    "whole_tree_cleanup_guaranteed": False,
                    "guard_final_receipt_exists": False,
                    "foreign_supervisor_still_stopped": True,
                    "boundary": "Outer SIGKILL cascades parent-death signals to the inner controller and loader; forked descendants do not inherit that signal setting. The independent test parent performs the remaining cleanup.",
                },
            )
            for name, raw in original.items():
                self.assertEqual((root / "attempt" / name).read_bytes(), raw)
        finally:
            for descriptor in pinned.values():
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.close(descriptor)
            for pid in [controller_pid, foreign_pid]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    child, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not child:
                    time.sleep(0.02)
            for record in targets:
                self.assertFalse((Path("/proc") / str(record["pid"])).exists())

    def test_intentional_trip_is_separate_from_probe_completion(self):
        root, result = self.check_case("intentional_trip", {"intentional_trip"})
        self.assertEqual(result["status"], "infrastructure_abort_guard")
        self.assertEqual(
            read_json(root / "probe.json")["status"], "load_probe_complete"
        )
        self.assertFalse(read_json(root / "probe.json")["process_complete"])
        self.assertEqual(result["detail"]["threshold_decision"], "memory_tripwire")
        self.assertGreaterEqual(
            result["last_sample"]["current"], result["effective_tripwire_bytes"]
        )
        self.assertEqual((root / "cgroup/memory.current").read_text(), "100")

    def test_child_failure_has_immutable_final_receipt(self):
        self.check_case("child_error", {"child_failed"})

    def test_identity_mismatch_cannot_signal_foreign_process(self):
        foreign = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            start_new_session=True,
        )
        try:
            identity = guard.process_identity(foreign.pid)
            invalid = dict(identity, start_ticks=identity["start_ticks"] + 1)
            with self.assertRaises(guard.IdentityError):
                guard.open_owned_pid(invalid, expected_parent=os.getpid())
            self.assertIsNone(foreign.poll())
        finally:
            foreign.kill()
            foreign.wait(timeout=3)

    def test_launch_publication_failure_never_executes_loader(self):
        root, _ = self.check_case("launch_failure", {"outer_error"})
        self.assertFalse((root / "loaded").exists())
        self.assertFalse((root / "attempt/launch.json").exists())

    def test_blocked_launch_publication_cannot_grant_exec_permission(self):
        root, spec = self.prepare("launch_stall")
        fifo = root / "launch-fifo"
        os.mkfifo(fifo)
        process = self.launch(root, spec)
        wait_for(lambda: (root / "publication-started").exists())
        children = (
            (Path("/proc") / str(process.pid) / "task" / str(process.pid) / "children")
            .read_text()
            .split()
        )
        self.assertEqual(len(children), 2)
        for child in children:
            self.assertEqual(
                (Path("/proc") / child / "task" / child / "children")
                .read_text()
                .strip(),
                "",
            )
        self.assertFalse((root / "loaded").exists())
        os.kill(process.pid, signal.SIGTERM)
        reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        try:
            self.finish(root, process, {"interrupted"})
        finally:
            os.close(reader)
        self.assertFalse((root / "loaded").exists())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "fixture":
        fixture(read_json(Path(sys.argv[2])))
    elif len(sys.argv) > 2 and sys.argv[1] == "case":
        run_case(read_json(Path(sys.argv[2])))
    else:
        unittest.main()
