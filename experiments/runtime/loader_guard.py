"""Linux loader guard with isolated I/O and an outer orphan reaper.

Polling is an operational tripwire, not an OOM-prevention guarantee. Kernel
D-state, scheduler starvation, loss of both controllers, and permanent loss of
the artifact filesystem are outside its termination/persistence guarantees.
"""

import argparse
import ctypes
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import select
import signal
import socket
import sys
import time
import uuid
from pathlib import Path

LIBC = ctypes.CDLL(None, use_errno=True)
PACKET_BYTES = 65536
PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16384
CHANNEL_LOG = logging.getLogger("loader-guard-child")
REQUIRED_STAT = {
    "anon",
    "file",
    "kernel",
    "file_mapped",
    "shmem",
    "unevictable",
    "file_dirty",
    "file_writeback",
}
REQUIRED_EVENTS = {"low", "high", "max", "oom", "oom_kill"}
LOAD_STAGES = (
    "cache_verified",
    "load_started",
    "load_complete",
)
STAGE_PROFILES = {
    "qualification": {
        "post_load_stages": (
            "state_hash_started",
            "state_hash_complete",
            "probe_started",
            "probe_complete",
        ),
        "terminal_stage": "probe_complete",
    },
    "catalog": {
        "post_load_stages": ("evaluation_started", "evaluation_complete"),
        "terminal_stage": "evaluation_complete",
    },
}


class Channel:
    def __init__(self, timeout_seconds):
        value = os.environ.get("LOADER_GUARD_CHANNEL_FD")
        if value is None or not value.isdecimal():
            raise RuntimeError(
                "GUARD_CHANNEL_REQUIRED: refuse standalone qualification"
            )
        self.socket = socket.socket(fileno=int(value))
        try:
            if (
                self.socket.family != socket.AF_UNIX
                or self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
                != socket.SOCK_DGRAM
            ):
                raise RuntimeError(
                    "GUARD_CHANNEL_INVALID: require connected AF_UNIX SOCK_DGRAM"
                )
            self.socket.getpeername()
            self.socket.settimeout(timeout_seconds)
        except Exception:
            self.socket.close()
            raise
        self.sequence = 0

    def close(self):
        self.socket.close()

    def send(self, event, receipt=None, **fields):
        self.sequence += 1
        message = {
            "protocol_version": PROTOCOL_VERSION,
            "event": event,
            "sequence": self.sequence,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "pgid": os.getpgrp(),
            "pod_uid": os.environ.get("POD_UID"),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        if receipt is not None:
            message["receipt"] = receipt["path"]
            message["receipt_sha256"] = receipt["sha256"]
        data = json.dumps(
            message, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if len(data) > MAX_MESSAGE_BYTES or self.socket.send(data) != len(data):
            raise RuntimeError("GUARD_MESSAGE_NOT_DELIVERED")
        CHANNEL_LOG.info(
            "event=%s sequence=%d receipt=%s",
            event,
            self.sequence,
            message.get("receipt"),
        )

    def receive(self):
        data, _, flags, _ = self.socket.recvmsg(MAX_MESSAGE_BYTES)
        if not data or flags & socket.MSG_TRUNC:
            raise RuntimeError("GUARD_COMMAND_INVALID: empty or truncated datagram")
        command = json.loads(data)
        if not isinstance(command, dict):
            raise TypeError("GUARD_COMMAND_INVALID: expected JSON object")
        return command

    def wait_for_continue(self):
        if self.receive().get("command") != "continue":
            raise RuntimeError(
                "GUARD_COMMAND_INVALID: expected continue at guarded milestone"
            )

    def hold_for_guard_kill(self):
        command = self.receive()
        raise RuntimeError(f"GUARD_HOLD_INTERRUPTED: unexpected command {command!r}")


class IdentityError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Policy:
    tripwire_bytes: int
    expected_max_bytes: int
    stage_profile: str
    poll_seconds: float = 0.25
    stale_seconds: float = 2.0
    logger_seconds: float = 2.0
    control_seconds: float = 4.0
    max_seconds: float = 7200.0
    cleanup_seconds: float = 5.0
    detailed_seconds: float = 1.0
    trip_after_probe: bool = False
    pod_uid: str = ""

    def validate(self):
        if self.stage_profile not in STAGE_PROFILES:
            raise ValueError("GUARD_INVALID_STAGE_PROFILE")
        if self.trip_after_probe and self.stage_profile != "qualification":
            raise ValueError("GUARD_INTENTIONAL_PROBE_TRIP_REQUIRES_QUALIFICATION")
        if not 0 < self.tripwire_bytes < self.expected_max_bytes:
            raise ValueError("GUARD_INVALID_MEMORY_BOUNDS")
        durations = (
            self.poll_seconds,
            self.stale_seconds,
            self.logger_seconds,
            self.control_seconds,
            self.max_seconds,
            self.cleanup_seconds,
            self.detailed_seconds,
        )
        if (
            any(not math.isfinite(value) or value <= 0 for value in durations)
            or self.stale_seconds <= self.poll_seconds
        ):
            raise ValueError("GUARD_INVALID_DEADLINES")


def write_once(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _prctl(option, value):
    function = LIBC.prctl
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    function.restype = ctypes.c_int
    if function(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "GUARD_PRCTL_FAILED")


def arm_parent_death(expected_parent):
    _prctl(1, signal.SIGKILL)
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGKILL)


def process_identity(pid):
    root = Path("/proc") / str(pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
        "state": fields[0],
        "uid": root.stat().st_uid,
    }


def identity_exists(identity):
    try:
        current = process_identity(identity["pid"])
        return current["start_ticks"] == identity["start_ticks"]
    except FileNotFoundError:
        return False


def open_owned_pid(identity, expected_parent):
    pid = identity["pid"]
    if pid <= 1 or identity["ppid"] != expected_parent:
        raise IdentityError("GUARD_FOREIGN_PARENT")
    descriptor = os.pidfd_open(pid)
    try:
        current = process_identity(pid)
        if any(
            current[key] != identity[key]
            for key in ("pid", "ppid", "start_ticks", "pgid", "session", "uid")
        ):
            raise IdentityError("GUARD_PROCESS_IDENTITY_CHANGED")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _kill_pidfd(descriptor):
    try:
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _send(channel, message):
    raw = json.dumps(message, separators=(",", ":")).encode()
    if len(raw) >= PACKET_BYTES:
        raise ValueError("GUARD_PACKET_TOO_LARGE")
    channel.send(raw)


def _messages(channel):
    messages = []
    for _ in range(64):
        try:
            raw = channel.recv(PACKET_BYTES)
        except BlockingIOError:
            break
        if not raw or len(raw) == PACKET_BYTES:
            raise ValueError("GUARD_INVALID_PACKET")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise TypeError("GUARD_INVALID_MESSAGE")
        messages.append(message)
    return messages


def _socket_pair():
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    first.setblocking(False)
    second.setblocking(False)
    return first, second


def _number(path):
    with path.open() as source:
        value = source.read(128).strip()
    if not value.isdigit():
        raise ValueError(f"GUARD_INVALID_COUNTER:{path.name}")
    return int(value)


def _keyed(path, required):
    with path.open() as source:
        raw = source.read(PACKET_BYTES)
    if len(raw) == PACKET_BYTES:
        raise ValueError(f"GUARD_COUNTER_FILE_TOO_LARGE:{path.name}")
    result = {}
    for line in raw.splitlines():
        key, value = line.split()
        if key in result or not value.isdigit():
            raise ValueError(f"GUARD_MALFORMED_COUNTER:{path.name}")
        result[key] = int(value)
    if not required <= result.keys():
        raise ValueError(f"GUARD_MISSING_COUNTER:{path.name}")
    return result


def _sample_worker(channel, root, pid, policy):
    detail_at = 0.0
    details = {}
    while True:
        started = time.monotonic()
        try:
            for message in _messages(channel):
                if message.get("command") != "smaps":
                    raise ValueError("GUARD_UNKNOWN_DIAGNOSTIC_REQUEST")
                with (Path("/proc") / str(pid) / "smaps_rollup").open() as source:
                    smaps = source.read(PACKET_BYTES)
                if len(smaps) == PACKET_BYTES:
                    raise ValueError("GUARD_SMAPS_TOO_LARGE")
                _send(
                    channel,
                    {
                        "kind": "smaps",
                        "stage": message["stage"],
                        "started": started,
                        "finished": time.monotonic(),
                        "raw": smaps,
                    },
                )
            identity = root.stat()
            sample = {
                "started": started,
                "current": _number(root / "memory.current"),
                "max": _number(root / "memory.max"),
                "cgroup_inode": identity.st_ino,
                "cgroup_device": identity.st_dev,
            }
            if started - detail_at >= policy.detailed_seconds:
                details = {
                    "cgroup_lifetime_peak": _number(root / "memory.peak"),
                    "events": _keyed(root / "memory.events", REQUIRED_EVENTS),
                    "stat": _keyed(root / "memory.stat", REQUIRED_STAT),
                }
                try:
                    status = (Path("/proc") / str(pid) / "status").read_text()
                    details["process_status"] = {
                        key: value.strip()
                        for key, value in (
                            line.split(":", 1) for line in status.splitlines()
                        )
                        if key
                        in {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "State"}
                    }
                except FileNotFoundError:
                    details["process_status"] = {"state": "exited"}
                detail_at = started
            sample.update(details)
            sample["details_started"] = detail_at
            sample["finished"] = time.monotonic()
            _send(channel, {"kind": "sample", "sample": sample})
        except (OSError, ValueError, TypeError, KeyError) as error:
            _send(
                channel, {"kind": "error", "error": f"{type(error).__name__}:{error}"}
            )
            return
        time.sleep(policy.poll_seconds)


def _log_worker(channel, path):
    channel.setblocking(True)
    try:
        with path.open("ab", buffering=0) as output:
            while True:
                message = json.loads(channel.recv(PACKET_BYTES))
                raw = (json.dumps(message, separators=(",", ":")) + "\n").encode()
                view = memoryview(raw)
                while view:
                    written = output.write(view)
                    if not written:
                        raise OSError("GUARD_LOG_SHORT_WRITE")
                    view = view[written:]
                os.fsync(output.fileno())
                _send(channel, {"kind": "ack", "sequence": message["sequence"]})
    except (OSError, ValueError, TypeError, KeyError) as error:
        _send(channel, {"kind": "error", "error": f"{type(error).__name__}:{error}"})


def _fork_worker(function, *arguments):
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        arm_parent_death(parent)
        try:
            function(*arguments)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            os.write(
                2, f"GUARD_WORKER_FAILURE:{type(error).__name__}:{error}\n".encode()
            )
            os._exit(70)
        finally:
            os._exit(0)
    return pid


def sample_failure(sample, policy):
    if sample["max"] != policy.expected_max_bytes:
        return "cgroup_limit_changed"
    if sample["current"] >= policy.tripwire_bytes:
        return "memory_tripwire"
    return None


def _exec_child(channel, command, environment):
    os.setsid()
    channel.setblocking(True)
    channel.settimeout(30)
    _send(
        channel,
        {
            "event": "bootstrap_ready",
            "identity": process_identity(os.getpid()),
            "parent_death_armed": True,
        },
    )
    if json.loads(channel.recv(PACKET_BYTES)) != {"command": "exec"}:
        raise ValueError("GUARD_EXEC_HANDSHAKE_FAILED")
    channel.set_inheritable(True)
    child_environment = dict(
        environment,
        LOADER_GUARD_CHANNEL_FD=str(channel.fileno()),
        LOADER_GUARD_PARENT_PID=str(os.getppid()),
    )
    os.execvpe(command[0], command, child_environment)


def _exit_code(pid):
    event = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if event is None:
        return None
    return event.si_status if event.si_code == os.CLD_EXITED else -event.si_status


def _controller(channel, command, environment, root, sink, policy):
    profile = STAGE_PROFILES[policy.stage_profile]
    stages = LOAD_STAGES + profile["post_load_stages"]
    terminal_stage = profile["terminal_stage"]
    channel.settimeout(policy.control_seconds)
    if json.loads(channel.recv(PACKET_BYTES)) != {"command": "start_controller"}:
        raise ValueError("GUARD_CONTROLLER_START_NOT_AUTHORIZED")
    channel.setblocking(False)
    sampler, sampler_child = _socket_pair()
    writer, writer_child = _socket_pair()
    loader, loader_child = _socket_pair()
    loader_pid = _fork_worker(_exec_child, loader_child, command, environment)
    helper_pids = [
        _fork_worker(_sample_worker, sampler_child, root, loader_pid, policy),
        _fork_worker(_log_worker, writer_child, sink),
    ]
    sampler_child.close()
    writer_child.close()
    loader_child.close()
    helper_fds = [os.pidfd_open(pid) for pid in helper_pids]
    _send(
        channel,
        {
            "kind": "helper_identities",
            "identities": [process_identity(pid) for pid in helper_pids],
        },
    )
    loader_fd = None
    identity = None
    started = time.monotonic()
    last_sample = started
    heartbeat_at = 0.0
    logged_details = -1.0
    stage_at_receipt = "bootstrap"
    sequence = 0
    pending_logs = {}
    sample = None
    baseline = None
    verified_at = None
    terminal_at = None
    executed = False
    terminal_complete = False
    reason = None
    detail = None
    observed_peak = 0
    child_exit = None
    logger_ready = False
    cgroup_identity = None
    smaps = {}
    stage_index = 0
    load_continue_granted = False
    first_violating_sample = None
    effective_policy = policy
    intentional_trip = None

    def log(event):
        nonlocal sequence
        if len(pending_logs) >= 16:
            raise RuntimeError("GUARD_LOG_QUEUE_FULL")
        sequence += 1
        _send(writer, {"sequence": sequence, "monotonic": time.monotonic(), **event})
        pending_logs[sequence] = time.monotonic()

    try:
        log(
            {
                "event": "controller_started",
                "pid": os.getpid(),
                "loader_pid": loader_pid,
            }
        )
        while reason is None:
            now = time.monotonic()
            if now - heartbeat_at >= min(0.25, policy.poll_seconds):
                _send(channel, {"kind": "heartbeat", "at": now})
                heartbeat_at = now
            readable, _, _ = select.select(
                [sampler, writer, loader], [], [], policy.poll_seconds
            )
            for connection in readable:
                for message in _messages(connection):
                    if connection is sampler:
                        if message["kind"] == "error":
                            reason, detail = "telemetry_error", message["error"]
                            break
                        if message["kind"] == "smaps":
                            stage = message["stage"]
                            smaps[stage] = message
                            log({"event": "smaps_rollup", **message})
                            if stage == "load_complete":
                                _send(loader, {"command": "continue"})
                                load_continue_granted = True
                            continue
                        sample = message["sample"]
                        observed_identity = (
                            sample["cgroup_device"],
                            sample["cgroup_inode"],
                        )
                        if cgroup_identity is None:
                            cgroup_identity = observed_identity
                        elif cgroup_identity != observed_identity:
                            reason = "cgroup_identity_changed"
                            break
                        last_sample = sample["started"]
                        observed_peak = max(observed_peak, sample["current"])
                        log(
                            {
                                "event": "memory_sample",
                                "sample_started": sample["started"],
                                "sample_finished": sample["finished"],
                                "current": sample["current"],
                                "max": sample["max"],
                                "stage_at_receipt": stage_at_receipt,
                            }
                        )
                        if sample["details_started"] != logged_details:
                            log({"event": "memory_details", **sample})
                            logged_details = sample["details_started"]
                        reason = sample_failure(sample, policy)
                        if (
                            reason is None
                            and time.monotonic() - sample["started"]
                            > policy.stale_seconds
                        ):
                            reason = "telemetry_stale"
                        if reason is not None:
                            first_violating_sample = sample
                            if reason == "memory_tripwire" and baseline is None:
                                baseline = {
                                    "allowed": False,
                                    "sample_started": sample["started"],
                                    "verified_received": verified_at,
                                    "current": sample["current"],
                                    "file": sample.get("stat", {}).get("file"),
                                    "phase": "after_hash"
                                    if verified_at is not None
                                    else "before_hash_complete",
                                }
                            break
                    elif connection is writer:
                        if message["kind"] == "error":
                            reason, detail = "logger_error", message["error"]
                            break
                        acknowledged = message["sequence"]
                        if acknowledged not in pending_logs:
                            raise ValueError("GUARD_UNKNOWN_LOG_ACK")
                        pending_logs.pop(acknowledged)
                        logger_ready = True
                    else:
                        event = message.get("event")
                        stage_at_receipt = event
                        log({"event": "child_stage", "stage": message})
                        if event == "bootstrap_ready":
                            identity = message["identity"]
                            if (
                                identity["pid"] != loader_pid
                                or identity["pgid"] != loader_pid
                                or identity["session"] != loader_pid
                            ):
                                raise IdentityError("GUARD_LOADER_GROUP_MISMATCH")
                            loader_fd = open_owned_pid(identity, os.getpid())
                            _send(
                                channel,
                                {
                                    "kind": "loader_identity",
                                    "identity": identity,
                                    "parent_death_armed": message["parent_death_armed"],
                                },
                            )
                        elif event == "failed":
                            reason, detail = "child_reported_failure", message
                            break
                        else:
                            if (
                                not executed
                                or stage_index >= len(stages)
                                or event != stages[stage_index]
                            ):
                                raise ValueError(f"GUARD_CHILD_STAGE_ORDER:{event}")
                            if (
                                stage_index >= len(LOAD_STAGES)
                                and not load_continue_granted
                            ):
                                raise ValueError("GUARD_POST_LOAD_BEFORE_OBSERVATION")
                            stage_index += 1
                            if event == "cache_verified":
                                verified_at = time.monotonic()
                                _send(sampler, {"command": "smaps", "stage": event})
                            elif event == "load_started":
                                if baseline is None or not baseline["allowed"]:
                                    raise ValueError("GUARD_LOAD_BEFORE_BASELINE")
                            elif event == "load_complete":
                                _send(sampler, {"command": "smaps", "stage": event})
                            elif event == terminal_stage:
                                terminal_complete = True
                                terminal_at = time.monotonic()
                if reason is not None:
                    break
            now = time.monotonic()
            if reason is not None:
                break
            if now - last_sample > policy.stale_seconds:
                reason = "telemetry_stale"
            elif (
                pending_logs
                and now - min(pending_logs.values()) > policy.logger_seconds
            ):
                reason = "logger_stale"
            elif now - started > policy.max_seconds:
                reason = "deadline"
            elif sample is not None:
                if (
                    policy.trip_after_probe
                    and terminal_at is not None
                    and sample["started"] >= terminal_at
                ):
                    effective_policy = dataclasses.replace(
                        policy, tripwire_bytes=max(0, sample["current"] - 1)
                    )
                    intentional_trip = {
                        "test_tripwire_bytes": effective_policy.tripwire_bytes,
                        "current": sample["current"],
                        "memory_pressure_created": False,
                    }
                    decision = sample_failure(sample, effective_policy)
                    if decision != "memory_tripwire":
                        raise RuntimeError("GUARD_INTENTIONAL_TRIP_DECISION_INVALID")
                    reason, detail = (
                        "intentional_trip",
                        {**intentional_trip, "threshold_decision": decision},
                    )
                if (
                    reason is None
                    and identity is not None
                    and logger_ready
                    and not executed
                ):
                    _send(loader, {"command": "exec"})
                    executed = True
                if (
                    reason is None
                    and verified_at is not None
                    and baseline is None
                    and sample["started"] >= verified_at
                    and "cache_verified" in smaps
                ):
                    baseline = {
                        "allowed": True,
                        "sample_started": sample["started"],
                        "verified_received": verified_at,
                        "current": sample["current"],
                        "file": sample.get("stat", {}).get("file"),
                        "phase": "after_hash",
                    }
                    log({"event": "baseline_after_hash", **baseline})
                    _send(loader, {"command": "continue"})
            if (
                reason is None
                and loader_fd is not None
                and select.select([loader_fd], [], [], 0)[0]
            ):
                child_exit = _exit_code(loader_pid)
                reason = (
                    "completed"
                    if child_exit == 0
                    and terminal_complete
                    and stage_index == len(stages)
                    and load_continue_granted
                    and {"cache_verified", "load_complete"} <= smaps.keys()
                    else "child_failed"
                )
    except Exception as error:  # noqa: BLE001 - Any controller fault must stop the loader.
        reason, detail = "controller_error", f"{type(error).__name__}:{error}"
    finally:
        killed_at = time.monotonic()
        if identity is not None:
            try:
                os.killpg(loader_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            temporary_fd = os.pidfd_open(loader_pid)
            _kill_pidfd(temporary_fd)
            os.close(temporary_fd)
        _kill_pidfd(helper_fds[0])
        final_log_error = None
        if reason not in {"logger_error", "logger_stale"}:
            log_deadline = time.monotonic() + policy.logger_seconds
            while pending_logs and time.monotonic() < log_deadline:
                _send(channel, {"kind": "heartbeat", "at": time.monotonic()})
                if select.select([writer], [], [], policy.poll_seconds)[0]:
                    for message in _messages(writer):
                        if message["kind"] == "error":
                            final_log_error = message["error"]
                            break
                        pending_logs.pop(message["sequence"], None)
                if final_log_error is not None:
                    break
            if pending_logs and reason == "completed":
                reason = "logger_error" if final_log_error else "logger_stale"
        for descriptor in helper_fds:
            _kill_pidfd(descriptor)
            os.close(descriptor)
        if loader_fd is not None:
            os.close(loader_fd)
        _send(
            channel,
            {
                "kind": "result",
                "result": {
                    "reason": reason,
                    "detail": detail,
                    "baseline_after_hash": baseline,
                    "last_sample": sample,
                    "first_violating_sample": first_violating_sample,
                    "attempt_max_observed_current_including_hashes": observed_peak,
                    "stage_profile": policy.stage_profile,
                    "terminal_stage": terminal_stage,
                    "terminal_complete": terminal_complete,
                    "child_exit_code": child_exit,
                    "kill_requested_at": killed_at,
                    "unacknowledged_log_sequences": sorted(pending_logs),
                    "telemetry_log_complete": not pending_logs,
                    "log_finalization_error": final_log_error,
                    "milestone_smaps": smaps,
                    "effective_tripwire_bytes": effective_policy.tripwire_bytes,
                    "stage_sequence": list(stages[:stage_index]),
                    "load_continue_granted": load_continue_granted,
                },
            },
        )


def _children():
    path = Path("/proc") / str(os.getpid()) / "task" / str(os.getpid()) / "children"
    return [int(value) for value in path.read_text().split()]


def _cleanup(protected, seconds):
    deadline = time.monotonic() + seconds
    reaped = []
    remaining = []
    while time.monotonic() < deadline:
        remaining = []
        for pid in _children()[:256]:
            try:
                identity = process_identity(pid)
            except FileNotFoundError:
                continue
            if (pid, identity["start_ticks"]) in protected:
                continue
            descriptor = open_owned_pid(identity, os.getpid())
            try:
                _kill_pidfd(descriptor)
            finally:
                os.close(descriptor)
            child, wait_status = os.waitpid(pid, os.WNOHANG)
            if child:
                exit_code = os.waitstatus_to_exitcode(wait_status)
                reaped.append(
                    {
                        **identity,
                        "exit_code": exit_code,
                        "signal": -exit_code if exit_code < 0 else None,
                    }
                )
            else:
                remaining.append(identity)
        if not remaining:
            current = []
            for pid in _children():
                try:
                    identity = process_identity(pid)
                except FileNotFoundError:
                    continue
                if (pid, identity["start_ticks"]) not in protected:
                    current.append(identity)
            if not current:
                return {
                    "all_reaped": True,
                    "reaped": reaped,
                    "remaining": [],
                    "finished_at": time.monotonic(),
                }
            remaining = current
        time.sleep(0.02)
    return {
        "all_reaped": False,
        "reaped": reaped,
        "remaining": remaining,
        "finished_at": time.monotonic(),
    }


def run_guard(
    command,
    output_dir,
    policy,
    *,
    cgroup_dir=None,
    event_sink=None,
    environment=None,
    launch_writer=write_once,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    policy.validate()
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        raise RuntimeError("GUARD_REQUIRES_LINUX_PIDFD")
    _prctl(36, 1)
    root = Path(cgroup_dir) if cgroup_dir is not None else Path("/sys/fs/cgroup")
    protected = {(pid, process_identity(pid)["start_ticks"]) for pid in _children()}
    environment = dict(os.environ if environment is None else environment)
    if cgroup_dir is None and (
        not policy.pod_uid or environment.get("POD_UID") != policy.pod_uid
    ):
        raise IdentityError("GUARD_POD_UID_MISMATCH")
    request = {
        "kind": "loader_guard_attempt",
        "attempt_id": uuid.uuid4().hex,
        "command": command,
        "policy": dataclasses.asdict(policy),
        "stage_contract": {
            "stages": LOAD_STAGES
            + STAGE_PROFILES[policy.stage_profile]["post_load_stages"],
            "terminal_stage": STAGE_PROFILES[policy.stage_profile]["terminal_stage"],
            "before_load_continue": "cache_verified",
            "after_load_continue": "load_complete",
        },
        "cgroup_dir": str(root),
        "telemetry_boundary": "live_cgroup"
        if cgroup_dir is None
        else "explicit_cpu_fixture",
        "cgroup_membership": Path("/proc/self/cgroup").read_text(),
        "outer_identity": process_identity(os.getpid()),
        "protected_children": [
            {"pid": pid, "start_ticks": start} for pid, start in sorted(protected)
        ],
        "platform": platform.platform(),
        "python": sys.version,
        "guard_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            name: environment.get(name)
            for name in (
                "HF_DEACTIVATE_ASYNC_LOAD",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
            )
        },
    }
    request_sha = write_once(output / "request.json", request)
    (output / "events.jsonl").touch(exist_ok=False)
    sink = Path(event_sink) if event_sink is not None else output / "events.jsonl"
    outer, inner = _socket_pair()
    controller_pid = None
    controller_fd = None
    result = None
    loader_identity = None
    armed = False
    helpers = []
    interrupted = []
    old_handlers = {
        number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)
    }
    for number in old_handlers:
        signal.signal(number, lambda signum, frame: interrupted.append(signum))
    try:
        controller_pid = _fork_worker(
            _controller, inner, command, environment, root, sink, policy
        )
        inner.close()
        controller_identity = process_identity(controller_pid)
        controller_fd = open_owned_pid(controller_identity, os.getpid())
        launch_writer(
            output / "launch.json",
            {
                "controller": controller_identity,
                "request_sha256": request_sha,
                "controller_requires_start_permission": True,
            },
        )
        if not interrupted:
            _send(outer, {"command": "start_controller"})
        last_contact = time.monotonic()
        while True:
            readable, _, _ = select.select(
                [outer, controller_fd], [], [], min(policy.poll_seconds, 0.25)
            )
            if outer in readable:
                for message in _messages(outer):
                    last_contact = time.monotonic()
                    if message["kind"] == "loader_identity":
                        loader_identity = message["identity"]
                        armed = message["parent_death_armed"]
                    elif message["kind"] == "result":
                        result = message["result"]
                    elif message["kind"] == "helper_identities":
                        helpers = message["identities"]
                    elif message["kind"] != "heartbeat":
                        raise ValueError("GUARD_UNKNOWN_CONTROLLER_MESSAGE")
            if interrupted:
                result = {"reason": "interrupted", "signal": interrupted[0]}
                break
            if controller_fd in readable:
                if result is None:
                    result = {
                        "reason": "controller_died",
                        "controller_exit_code": _exit_code(controller_pid),
                    }
                break
            if time.monotonic() - last_contact > policy.control_seconds:
                result = {"reason": "controller_stale"}
                break
    except Exception as error:  # noqa: BLE001 - Preserve a terminal receipt after any launch/control failure.
        result = {"reason": "outer_error", "detail": f"{type(error).__name__}:{error}"}
    finally:
        if controller_fd is not None:
            _kill_pidfd(controller_fd)
        cleanup = _cleanup(protected, policy.cleanup_seconds)
        if controller_fd is not None:
            os.close(controller_fd)
        inner.close()
        outer.close()
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    result = dict(result or {"reason": "controller_died"})
    expected_pids = {controller_pid, *[item["pid"] for item in helpers]}
    if loader_identity is not None:
        expected_pids.add(loader_identity["pid"])
    unexpected = [
        record for record in cleanup["reaped"] if record["pid"] not in expected_pids
    ]
    if result["reason"] == "completed" and unexpected:
        result["reason"] = "unexpected_descendants"
    result.update(
        {
            "status": "completed"
            if result["reason"] == "completed" and cleanup["all_reaped"]
            else "infrastructure_abort_guard",
            "request_sha256": request_sha,
            "loader_identity": loader_identity,
            "parent_death_protection_armed": armed,
            "cleanup": cleanup,
            "unexpected_descendants": unexpected,
            "limits": [
                "Polling and reserve do not guarantee OOM prevention.",
                "Uninterruptible kernel D-state may prevent timely termination/reaping.",
                "Top-level outer SIGKILL also kills the inner controller and loader through parent-death signaling, but does not guarantee cleanup of their forked descendants; no surviving guard remains to write a final receipt. The immutable request/launch remain.",
                "Permanent loss of the output filesystem can prevent a final receipt.",
                "Cgroup stat categories overlap and file cache is never subtracted from the tripwire.",
            ],
        }
    )
    result["events_sha256"] = hashlib.sha256(
        (output / "events.jsonl").read_bytes()
    ).hexdigest()
    os.chmod(output / "events.jsonl", 0o444)
    write_once(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tripwire-bytes", type=int, required=True)
    parser.add_argument("--expected-max-bytes", type=int, required=True)
    parser.add_argument("--stage-profile", choices=tuple(STAGE_PROFILES), required=True)
    parser.add_argument("--pod-uid", required=True)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--trip-after-probe", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required")
    policy = Policy(
        args.tripwire_bytes,
        args.expected_max_bytes,
        args.stage_profile,
        max_seconds=args.max_seconds,
        trip_after_probe=args.trip_after_probe,
        pod_uid=args.pod_uid,
    )
    result = run_guard(command, args.output_dir, policy)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
