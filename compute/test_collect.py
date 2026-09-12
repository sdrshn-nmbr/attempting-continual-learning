import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Lock

import collect
import pytest


class LocalTransport:
    def __init__(self, root, run):
        self.root = root
        self.run = run
        self.stream_codes = []
        self.fail = None
        self.mutex = Lock()

    def call(self, command, *args, **kwargs):
        if command[: len(collect.KUBE)] != collect.KUBE:
            return self.run(command, *args, **kwargs)
        code = command[-1]
        if "stdout" in kwargs:
            with self.mutex:
                self.stream_codes.append(code)
            if self.fail and self.fail(code):
                kwargs["stdout"].flush()
                os.write(kwargs["stdout"].fileno(), b"partial")
                return subprocess.CompletedProcess(command, 1, stderr=b"injected")
        return self.run(
            ["uv", "run", "--no-project", "--python", sys.executable, "python", "-c", code],
            *args,
            **kwargs,
        )

    def offsets(self, name):
        return [
            int(match.group(1))
            for code in self.stream_codes
            if repr(name) in code
            for match in [re.search(r"f.seek\((\d+)\)", code)]
            if match
        ]


@pytest.fixture
def remote(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    root.mkdir()
    transport = LocalTransport(root, subprocess.run)
    monkeypatch.setattr(collect, "REMOTE", str(root))
    monkeypatch.setattr(collect, "CHUNK", 32)
    monkeypatch.setattr(collect.subprocess, "run", transport.call)
    return transport


def write_files(root, files):
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)


def expected_manifest(files):
    return {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in files.items()
    }


def assert_published(destination, identity, files):
    target = destination / "runs" / identity
    assert not (destination / ".collecting" / identity).exists()
    for name, value in files.items():
        assert (target / name).read_bytes() == value
    receipt = json.loads((target / "collection.json").read_text())
    assert receipt["execution_status"] == "completed"
    assert receipt["files"] == expected_manifest(files)


def test_resume_retains_valid_bytes_and_redownloads_partial_after_three_retries(
    remote, tmp_path
):
    identity = "run-resume"
    files = {"a.bin": b"A" * 80, "b.bin": b"B" * 80, "nested/note": b"small"}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    remote.fail = lambda code: "'b.bin'" in code and "f.seek(32)" in code
    with pytest.raises(RuntimeError, match="COLLECTION_STREAM_FAILED"):
        collect.collect_run(identity, "completed", destination, "cpu-only")
    staging = destination / ".collecting" / identity
    assert not (destination / "runs" / identity).exists()
    assert (staging / "a.bin").read_bytes() == files["a.bin"]
    assert (staging / "b.bin").read_bytes() == b"B" * 32 + b"partial"
    assert remote.offsets("b.bin") == [0, 32, 32, 32]
    previous = (staging / "a.bin").stat()
    remote.fail = None
    remote.stream_codes.clear()
    result = collect.collect_run(identity, "completed", destination, "cpu-only")
    assert result["resumed_files"] == 1
    assert remote.offsets("a.bin") == []
    assert remote.offsets("b.bin") == [0, 32, 64]
    current = (destination / "runs" / identity / "a.bin").stat()
    assert (current.st_ino, current.st_mtime_ns) == (
        previous.st_ino, previous.st_mtime_ns
    )
    assert_published(destination, identity, files)


def test_corrupt_same_size_and_partial_files_ignore_old_receipt(remote, tmp_path):
    identity = "run-corrupt"
    files = {"corrupt.bin": b"C" * 80, "partial.bin": b"P" * 80, "note": b"good"}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    staging = destination / ".collecting" / identity
    write_files(staging, {
        "corrupt.bin": b"X" * 80,
        "partial.bin": b"P" * 17,
        "note": b"bad!",
        "collection.json": json.dumps({"files": expected_manifest(files)}).encode(),
    })
    result = collect.collect_run(identity, "completed", destination, "cpu-only")
    assert result["resumed_files"] == 0
    assert result["downloaded_files"] == 3
    assert_published(destination, identity, files)


def test_remote_manifest_drift_retains_staging_without_promotion(
    remote, tmp_path, monkeypatch
):
    identity = "run-drift"
    original = {"state.bin": b"A" * 80}
    changed = {"state.bin": b"B" * 80}
    write_files(remote.root / identity, original)
    destination = tmp_path / "destination"
    manifest = collect.manifest
    calls = 0

    def drifting(name, control):
        nonlocal calls
        calls += 1
        if calls == 2:
            write_files(remote.root / identity, changed)
        return manifest(name, control)

    monkeypatch.setattr(collect, "manifest", drifting)
    with pytest.raises(ValueError, match="COLLECTION_HASH_MISMATCH"):
        collect.collect_run(identity, "completed", destination, "cpu-only")
    staging = destination / ".collecting" / identity
    assert (staging / "state.bin").read_bytes() == original["state.bin"]
    assert not (staging / "collection.json").exists()
    assert not (destination / "runs" / identity).exists()
    collect.collect_run(identity, "completed", destination, "cpu-only")
    assert_published(destination, identity, changed)


@pytest.mark.parametrize("already_staged", [False, True])
def test_large_duplicate_payload_downloaded_once_and_copied_independently(
    remote, tmp_path, already_staged
):
    identity = "run-duplicates"
    payload = b"same" * 20
    files = {f"checkpoint{i}/weights.bin": payload for i in range(3)}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    if already_staged:
        write_files(destination / ".collecting" / identity, {
            "checkpoint2/weights.bin": payload,
        })
    result = collect.collect_run(identity, "completed", destination, "cpu-only")
    assert result["downloaded_files"] == (0 if already_staged else 1)
    assert result["downloaded_payload_bytes"] == (0 if already_staged else len(payload))
    assert result["local_duplicate_copies"] == 2
    assert len(remote.stream_codes) == (0 if already_staged else 3)
    assert_published(destination, identity, files)
    paths = [destination / "runs" / identity / name for name in files]
    assert len({path.stat().st_ino for path in paths}) == 3
    assert all(path.stat().st_nlink == 1 for path in paths)
    with paths[0].open("r+b") as handle:
        handle.write(b"changed")
    assert all(path.read_bytes() == payload for path in paths[1:])


def test_corrupt_download_is_not_reused_for_duplicate(remote, tmp_path, monkeypatch):
    identity = "run-bad-download"
    files = {"first.bin": b"F" * 80, "second.bin": b"F" * 80}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    stream = collect.stream

    def corrupt(code, handle, control):
        offset = handle.tell()
        stream(code, handle, control)
        if "'first.bin'" in code and offset == 0:
            end = handle.tell()
            handle.seek(0)
            handle.write(b"wrong")
            handle.seek(end)

    monkeypatch.setattr(collect, "stream", corrupt)
    with pytest.raises(ValueError, match="COLLECTION_FILE_HASH_MISMATCH"):
        collect.collect_run(identity, "completed", destination, "cpu-only")
    staging = destination / ".collecting" / identity
    assert (staging / "first.bin").read_bytes().startswith(b"wrong")
    assert not (staging / "second.bin").exists()
    assert not (destination / "runs" / identity).exists()


def test_small_remote_hardlinks_are_collected_as_independent_files(remote, tmp_path):
    identity = "run-remote-links"
    write_files(remote.root / identity, {"one": b"same"})
    os.link(remote.root / identity / "one", remote.root / identity / "two")
    files = {"one": b"same", "two": b"same"}
    destination = tmp_path / "destination"
    collect.collect_run(identity, "completed", destination, "cpu-only")
    assert_published(destination, identity, files)
    paths = [destination / "runs" / identity / name for name in files]
    assert paths[0].stat().st_ino != paths[1].stat().st_ino
    assert all(path.stat().st_nlink == 1 for path in paths)


def test_failed_atomic_promotion_resumes_without_redownloading(
    remote, tmp_path, monkeypatch
):
    identity = "run-promotion"
    files = {"state.bin": b"S" * 80, "note": b"N"}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    rename = Path.rename

    def interrupted(path, target):
        assert not target.exists()
        assert json.loads((path / "collection.json").read_text())["files"] == expected_manifest(files)
        raise OSError("injected before atomic promotion")

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError, match="before atomic promotion"):
        collect.collect_run(identity, "completed", destination, "cpu-only")
    assert not (destination / "runs" / identity).exists()
    assert (destination / ".collecting" / identity / "collection.json").exists()
    remote.stream_codes.clear()
    monkeypatch.setattr(Path, "rename", rename)
    result = collect.collect_run(identity, "completed", destination, "cpu-only")
    assert result["resumed_files"] == 2
    assert not remote.stream_codes
    assert_published(destination, identity, files)


def test_concurrent_claim_excludes_second_thread_and_process(
    remote, tmp_path, monkeypatch
):
    identity = "run-exclusive"
    files = {"state.bin": b"S" * 80}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    entered, release = Event(), Event()
    manifest = collect.manifest

    def paused(name, control):
        entered.set()
        assert release.wait(10)
        return manifest(name, control)

    monkeypatch.setattr(collect, "manifest", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(collect.collect_run, identity, "completed", destination, "cpu-only")
        try:
            assert entered.wait(10)
            with pytest.raises(RuntimeError, match="COLLECTION_ALREADY_CLAIMED"):
                collect.collect_run(identity, "completed", destination, "cpu-only")
            lock = destination / ".collecting/.locks/runs" / identity
            result = subprocess.run(
                [
                    "uv", "run", "--no-project", "--python", sys.executable, "python", "-c",
                    (
                        "import sys\nfrom pathlib import Path\nfrom collect import exclusive_lock\n"
                        "try:\n    with exclusive_lock(Path(sys.argv[1])):\n        pass\n"
                        "except RuntimeError as error:\n    print(error)\n    sys.exit(23)\n"
                    ),
                    str(lock),
                ],
                cwd=Path(collect.__file__).parent,
                check=False, capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 23, result.stderr
            assert "COLLECTION_ALREADY_CLAIMED" in result.stdout
        finally:
            release.set()
        first.result(timeout=10)
    assert_published(destination, identity, files)
    before = list(remote.stream_codes)
    assert collect.collect_run(identity, "completed", destination, "cpu-only") is None
    assert remote.stream_codes == before


@pytest.mark.parametrize("workers", [None, 2, 4])
def test_worker_bound_and_single_source_collection_after_all_runs(
    remote, tmp_path, monkeypatch, workers
):
    limit = workers or 1
    identities = [f"run-parallel-{i}" for i in range(limit + 1)]
    files = {"execution.json": b'{"status":"completed"}', "state.bin": b"S" * 80}
    for identity in identities:
        write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    args = ["collect.py", "--output-dir", str(destination), "--control-pod", "cpu-only", "--prefix", "run-parallel-"]
    if workers is not None:
        args.extend(["--workers", str(workers)])
    monkeypatch.setattr(sys, "argv", args)
    barrier, mutex = Barrier(limit), Lock()
    active, peak, started = 0, 0, 0
    copy_files = collect.copy_files
    sources = []

    def copying(*args):
        nonlocal active, peak, started
        with mutex:
            active += 1
            started += 1
            first_batch = started <= limit
            peak = max(peak, active)
        try:
            if first_batch:
                barrier.wait(timeout=10)
            return copy_files(*args)
        finally:
            with mutex:
                active -= 1

    def collect_sources(states, output, control):
        assert active == 0
        assert set(states) == set(identities)
        for identity in identities:
            assert_published(output, identity, files)
        sources.append((states, output, control))

    monkeypatch.setattr(collect, "copy_files", copying)
    monkeypatch.setattr(collect, "collect_sources", collect_sources)
    collect.main()
    assert peak == limit
    assert len(sources) == 1


def test_failed_run_keeps_staging_and_does_not_cancel_other_runs(
    remote, tmp_path, monkeypatch
):
    files = {"execution.json": b'{"status":"completed"}', "state.bin": b"S" * 80}
    for identity in ("run-good", "run-failed"):
        write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    monkeypatch.setattr(sys, "argv", [
        "collect.py", "--output-dir", str(destination), "--control-pod", "cpu-only",
        "--prefix", "run-", "--workers", "2",
    ])
    copy_files = collect.copy_files
    calls = []

    def copying(identity, *args):
        if identity == "run-failed":
            raise OSError("injected copy failure")
        return copy_files(identity, *args)

    def collect_sources(*args):
        assert_published(destination, "run-good", files)
        assert (destination / ".collecting/run-failed").exists()
        assert not (destination / "runs/run-failed").exists()
        calls.append(args)

    monkeypatch.setattr(collect, "copy_files", copying)
    monkeypatch.setattr(collect, "collect_sources", collect_sources)
    with pytest.raises(RuntimeError, match="COLLECTION_RUNS_FAILED.*run-failed"):
        collect.main()
    assert len(calls) == 1


def test_separate_invocations_serialize_source_archive_writers(
    remote, tmp_path, monkeypatch
):
    destination = tmp_path / "destination"
    monkeypatch.setattr(sys, "argv", [
        "collect.py", "--output-dir", str(destination), "--control-pod", "cpu-only",
        "--prefix", "no-runs-",
    ])
    ready = Barrier(3)
    entered, release = Event(), Event()
    mutex = Lock()
    calls = []
    lock = collect.exclusive_lock

    @contextmanager
    def observed_lock(path, blocking=False):
        assert path.name == "sources" and blocking
        ready.wait(timeout=10)
        with lock(path, blocking=blocking):
            yield

    def collect_sources(*args):
        with mutex:
            calls.append(args)
            first = len(calls) == 1
        if first:
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr(collect, "exclusive_lock", observed_lock)
    monkeypatch.setattr(collect, "collect_sources", collect_sources)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with lock(destination / ".collecting/.locks/sources"):
            futures = [pool.submit(collect.main) for _ in range(2)]
            ready.wait(timeout=10)
            assert not entered.is_set()
        try:
            assert entered.wait(10)
            assert len(calls) == 1
        finally:
            release.set()
        for future in futures:
            future.result(timeout=10)
    assert len(calls) == 2


@pytest.mark.parametrize("workers", ["0", "5", "-1", "many"])
def test_invalid_workers_rejected_before_transport_or_staging(
    tmp_path, monkeypatch, workers
):
    destination = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", [
        "collect.py", "--output-dir", str(destination), "--control-pod", "must-not-contact",
        "--prefix", "run-", "--workers", workers,
    ])

    def unexpected(*args):
        pytest.fail("transport must not run")

    monkeypatch.setattr(collect, "execute", unexpected)
    with pytest.raises(SystemExit) as error:
        collect.main()
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_repair_never_mutates_linked_external_bytes(remote, tmp_path, link):
    identity = "run-links"
    files = {"state.bin": b"S" * 80}
    write_files(remote.root / identity, files)
    destination = tmp_path / "destination"
    staging = destination / ".collecting" / identity
    staging.mkdir(parents=True)
    outside = tmp_path / "independent"
    outside.write_bytes(b"X" * 80)
    path = staging / "state.bin"
    if link == "symlink":
        path.symlink_to(outside)
    else:
        os.link(outside, path)
    collect.collect_run(identity, "completed", destination, "cpu-only")
    assert outside.read_bytes() == b"X" * 80
    assert_published(destination, identity, files)
    assert (destination / "runs" / identity / "state.bin").stat().st_nlink == 1
