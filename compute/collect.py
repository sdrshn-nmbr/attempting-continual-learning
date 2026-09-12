import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

KUBE = ["kubectl", "--context", "us-mi355x-nambiar-k8s"]
REMOTE = "/mnt/shared/cl-portfolio/runs"
CHUNK = 16 * 1024 * 1024


def execute(code, control):
    result = subprocess.run(
        KUBE
        + ["exec", control, "--", "uv", "run", "--no-project", "python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def manifest(identity, control):
    return execute(
        "import hashlib,json; from pathlib import Path; "
        f"p=Path({REMOTE!r})/{identity!r}; "
        "files={str(x.relative_to(p)):{'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} "
        "for x in sorted(p.rglob('*')) if x.is_file()}; "
        "print(json.dumps(files))",
        control,
    )


def stream(code, handle, control):
    offset = handle.tell()
    for attempt in range(3):
        handle.seek(offset)
        handle.truncate()
        result = subprocess.run(
            KUBE
            + [
                "exec",
                control,
                "--",
                "uv",
                "run",
                "--no-project",
                "python",
                "-c",
                code,
            ],
            check=False,
            stdout=handle,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return
        if attempt == 2:
            raise RuntimeError(f"COLLECTION_STREAM_FAILED: {result.stderr.decode()}")


def file_info(path):
    with path.open("rb") as handle:
        return {
            "bytes": os.fstat(handle.fileno()).st_size,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }


def directory(path):
    if path.is_symlink():
        raise ValueError(f"COLLECTION_DIRECTORY_SYMLINK: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def exclusive_lock(path, blocking=False):
    directory(path.parent)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        try:
            fcntl.flock(
                handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            )
        except BlockingIOError as error:
            raise RuntimeError(f"COLLECTION_ALREADY_CLAIMED: {path.name}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_manifest(files):
    for name, info in files.items():
        path = Path(name)
        if (
            not name
            or path.is_absolute()
            or str(path) != name
            or ".." in path.parts
            or name in {".", "collection.json"}
            or not isinstance(info["bytes"], int)
            or info["bytes"] < 0
            or len(info["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in info["sha256"])
        ):
            raise ValueError(f"COLLECTION_MANIFEST_INVALID: {name}")


def reusable(path, expected):
    if not path.exists() or path.is_symlink():
        return False
    metadata = path.stat()
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and metadata.st_size == expected["bytes"]
        and file_info(path) == expected
    )


def prepare_file(temporary, name):
    parent = temporary
    for part in Path(name).parts[:-1]:
        parent = directory(parent / part)
    destination = temporary / name
    destination.unlink(missing_ok=True)
    return destination


def verify_file(path, expected):
    if not reusable(path, expected):
        raise ValueError(f"COLLECTION_FILE_HASH_MISMATCH: {path}")


def copy_files(identity, temporary, files, control):
    directory(temporary)
    validate_manifest(files)
    verified, payloads = set(), {}
    for name, info in files.items():
        parent = temporary
        for part in Path(name).parts[:-1]:
            parent = directory(parent / part)
        path = temporary / name
        if reusable(path, info):
            verified.add(name)
            if info["bytes"] >= CHUNK:
                payloads[(info["sha256"], info["bytes"])] = path
    counts = {
        "resumed_files": len(verified),
        "downloaded_files": 0,
        "downloaded_payload_bytes": 0,
        "local_duplicate_copies": 0,
        "local_duplicate_bytes": 0,
    }
    groups, group, total = [], [], 0
    for name, info in files.items():
        if name in verified:
            continue
        if info["bytes"] >= CHUNK:
            destination = prepare_file(temporary, name)
            key = (info["sha256"], info["bytes"])
            if key in payloads:
                shutil.copyfile(payloads[key], destination)
                verify_file(destination, info)
                counts["local_duplicate_copies"] += 1
                counts["local_duplicate_bytes"] += info["bytes"]
                continue
            with destination.open("wb") as handle:
                for offset in range(0, info["bytes"], CHUNK):
                    size = min(CHUNK, info["bytes"] - offset)
                    code = (
                        "import sys; from pathlib import Path; "
                        f"p=Path({REMOTE!r})/{identity!r}/{name!r}; f=p.open('rb'); "
                        f"f.seek({offset}); sys.stdout.buffer.write(f.read({size}))"
                    )
                    stream(code, handle, control)
                    if handle.tell() != offset + size:
                        raise ValueError(f"COLLECTION_SHORT_SEGMENT: {identity}/{name}")
            verify_file(destination, info)
            payloads[key] = destination
            counts["downloaded_files"] += 1
            counts["downloaded_payload_bytes"] += info["bytes"]
            continue
        if group and total + info["bytes"] > CHUNK:
            groups.append(group)
            group, total = [], 0
        group.append(name)
        total += info["bytes"]
    if group:
        groups.append(group)
    for group in groups:
        code = (
            "import sys,tarfile; from pathlib import Path; "
            f"p=Path({REMOTE!r})/{identity!r}; "
            "t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|',dereference=True); "
            f"[t.add(p/name,arcname=name,recursive=False) for name in {group!r}]; t.close()"
        )
        with tempfile.TemporaryFile() as handle:
            stream(code, handle, control)
            handle.seek(0)
            with tarfile.open(fileobj=handle) as tar:
                members = tar.getmembers()
                if (
                    len(members) != len(group)
                    or {member.name for member in members} != set(group)
                    or any(
                        not member.isfile()
                        or member.size != files[member.name]["bytes"]
                        for member in members
                    )
                ):
                    raise ValueError(f"COLLECTION_TAR_MEMBERS_INVALID: {identity}")
                for name in group:
                    prepare_file(temporary, name)
                tar.extractall(temporary, filter="data")
        for name in group:
            verify_file(temporary / name, files[name])
            counts["downloaded_files"] += 1
            counts["downloaded_payload_bytes"] += files[name]["bytes"]
    return counts


def local_manifest(temporary):
    files = {}
    for path in sorted(temporary.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"COLLECTION_LOCAL_FILE_INVALID: {path}")
        files[str(path.relative_to(temporary))] = file_info(path)
    return files


def collect_run(identity, state, destination, control):
    if (
        not identity
        or not identity[0].isalnum()
        or any(not (c.isalnum() or c in "_.-") for c in identity)
    ):
        raise ValueError(f"COLLECTION_ID_INVALID: {identity}")
    staging = directory(destination / ".collecting")
    locks = directory(staging / ".locks")
    with exclusive_lock(locks / "runs" / identity):
        target = directory(destination / "runs") / identity
        if target.exists():
            return
        before = manifest(identity, control)
        validate_manifest(before)
        temporary = directory(staging / identity)
        (temporary / "collection.json").unlink(missing_ok=True)
        counts = copy_files(identity, temporary, before, control)
        local = local_manifest(temporary)
        after = manifest(identity, control)
        if local != before or before != after:
            raise ValueError(f"COLLECTION_HASH_MISMATCH: {identity}")
        receipt = {
            "remote": f"{REMOTE}/{identity}",
            "execution_status": state,
            "files": local,
        }
        (temporary / "collection.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        if target.exists():
            raise FileExistsError(f"COLLECTION_TARGET_APPEARED: {identity}")
        temporary.rename(target)
        result = {
            "collected": identity,
            "status": state,
            "files": len(before),
            "bytes": sum(x["bytes"] for x in before.values()),
            **counts,
        }
        print(json.dumps(result), flush=True)
        return result


def collect_sources(states, destination, control):
    directory = destination / "code"
    directory.mkdir(exist_ok=True)
    references = execute(
        "import json; from pathlib import Path; "
        f"root=Path({REMOTE!r}); "
        f"print(json.dumps({{name:json.loads((root/name/'execution.json').read_text())['source_sha256'] for name in {list(states)!r}}}))",
        control,
    )
    for code_hash in sorted(set(references.values())):
        archive = directory / f"{code_hash}.tar"
        receipt = directory / f"{code_hash}.json"
        if archive.exists() and receipt.exists():
            continue
        remote = f"/mnt/shared/cl-portfolio/code/{code_hash}.tar"
        info = execute(
            "import hashlib,json; from pathlib import Path; "
            f"p=Path({remote!r}); "
            "print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))",
            control,
        )
        with archive.open("wb") as handle:
            for offset in range(0, info["bytes"], CHUNK):
                stream(
                    "import sys; from pathlib import Path; "
                    f"f=Path({remote!r}).open('rb'); f.seek({offset}); "
                    f"sys.stdout.buffer.write(f.read({min(CHUNK, info['bytes'] - offset)}))",
                    handle,
                    control,
                )
        if hashlib.sha256(archive.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"SOURCE_ARCHIVE_HASH_MISMATCH: {code_hash}")
        digest = hashlib.sha256()
        with tarfile.open(archive) as tar:
            members = sorted(tar.getmembers(), key=lambda member: Path(member.name))
            for member in members:
                if (
                    not member.isfile()
                    or member.name.startswith("/")
                    or ".." in Path(member.name).parts
                ):
                    raise ValueError(f"SOURCE_ARCHIVE_MEMBER_INVALID: {member.name}")
                digest.update(
                    member.name.encode() + b"\0" + tar.extractfile(member).read()
                )
        if digest.hexdigest() != code_hash:
            raise ValueError(f"SOURCE_CLOSURE_HASH_MISMATCH: {code_hash}")
        receipt.write_text(
            json.dumps(
                {
                    "remote": remote,
                    "source_sha256": code_hash,
                    "archive": info,
                    "files": len(members),
                    "tasks": sorted(k for k, v in references.items() if v == code_hash),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "collected_source": code_hash,
                    "files": len(members),
                    "bytes": info["bytes"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-pod", required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    states = execute(
        "import json; from pathlib import Path; "
        f"root=Path({REMOTE!r}); "
        "print(json.dumps({p.name:json.loads((p/'execution.json').read_text())['status'] "
        f"for p in sorted(root.iterdir()) if p.name.startswith({tuple(args.prefix)!r}) "
        "and (p/'execution.json').is_file()}))",
        args.control_pod,
    )
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                collect_run, identity, state, destination, args.control_pod
            ): identity
            for identity, state in states.items()
            if state in {"failed", "completed", "blocked", "interrupted"}
        }
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                identity = futures[future]
                failures[identity] = f"{type(error).__name__}: {error}"
                print(
                    json.dumps({"collection_failed": identity, "error": str(error)}),
                    flush=True,
                )
    staging = directory(destination / ".collecting")
    with exclusive_lock(directory(staging / ".locks") / "sources", blocking=True):
        collect_sources(
            {
                name: state
                for name, state in states.items()
                if state in {"completed", "failed", "interrupted"}
            },
            destination,
            args.control_pod,
        )
    if failures:
        raise RuntimeError(f"COLLECTION_RUNS_FAILED: {json.dumps(failures, sort_keys=True)}")


if __name__ == "__main__":
    main()
