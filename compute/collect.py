import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
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


def copy_files(identity, temporary, files, control):
    temporary.mkdir()
    groups, group, total = [], [], 0
    for name, info in files.items():
        if info["bytes"] >= CHUNK:
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
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
            continue
        if group and total + info["bytes"] > CHUNK:
            groups.append(group)
            group, total = [], 0
        group.append(name)
        total += info["bytes"]
    if group:
        groups.append(group)
    for group in groups:
        archive = temporary.parent / "transfer.tar"
        code = (
            "import sys,tarfile; from pathlib import Path; "
            f"p=Path({REMOTE!r})/{identity!r}; "
            "t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|'); "
            f"[t.add(p/name,arcname=name,recursive=False) for name in {group!r}]; t.close()"
        )
        with archive.open("wb") as handle:
            stream(code, handle, control)
        with tarfile.open(archive) as tar:
            tar.extractall(temporary, filter="data")
        archive.unlink()


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
    (destination / "runs").mkdir(exist_ok=True)
    for identity, state in states.items():
        target = destination / "runs" / identity
        if state not in {"failed", "completed", "blocked", "interrupted"} or target.exists():
            continue
        before = manifest(identity, args.control_pod)
        with tempfile.TemporaryDirectory(
            prefix="cl-collect-", dir=destination
        ) as directory:
            temporary = Path(directory) / identity
            copy_files(identity, temporary, before, args.control_pod)
            local = {
                str(path.relative_to(temporary)): {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            if local != before or before != manifest(identity, args.control_pod):
                raise ValueError(f"COLLECTION_HASH_MISMATCH: {identity}")
            receipt = {
                "remote": f"{REMOTE}/{identity}",
                "execution_status": state,
                "files": local,
            }
            (temporary / "collection.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            shutil.move(temporary, target)
        print(
            json.dumps(
                {
                    "collected": identity,
                    "status": state,
                    "files": len(before),
                    "bytes": sum(x["bytes"] for x in before.values()),
                }
            ),
            flush=True,
        )
    collect_sources(
        {
            name: state
            for name, state in states.items()
            if state in {"completed", "failed"}
        },
        destination,
        args.control_pod,
    )


if __name__ == "__main__":
    main()
