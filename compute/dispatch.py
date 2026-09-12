import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

CONTEXT = "us-mi355x-nambiar-k8s"
CONTROL = "cl-portfolio-control"
REMOTE = "/mnt/shared/cl-portfolio"


def kubectl(*args):
    subprocess.run(
        ["kubectl", "--context", CONTEXT, "-n", "default", *map(str, args)],
        check=True,
    )


def sources(directory):
    excluded = {"__pycache__", "tests", "outputs", "runs"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") or part in excluded
            for part in path.relative_to(directory).parts
        )
        and (
            path.suffix in {".py", ".json", ".txt", ".yaml", ".yml"}
            or path.name.startswith(("LICENSE", "NOTICE"))
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--entrypoint", default="run.py")
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", args.id):
        raise ValueError("Task ID must be one path component")
    config = json.loads(args.config.read_text())
    digest = hashlib.sha256()
    files = sources(args.lane_dir)
    captured = [(path.relative_to(args.lane_dir), path.read_bytes()) for path in files]
    for name, content in captured:
        digest.update(str(name).encode() + b"\0" + content)
    code_hash = digest.hexdigest()
    remote_code = f"{REMOTE}/code/{code_hash}"
    task = {
        "id": args.id,
        "lane": args.lane_dir.name,
        "code_dir": remote_code,
        "source_sha256": code_hash,
        "config": config,
        "gpus": args.gpus,
        "entrypoint": args.entrypoint,
        "depends_on": args.depends_on,
        "priority": args.priority,
        "max_seconds": args.max_seconds,
    }
    output = Path("outputs/portfolio/manifests")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{args.id}.json"
    if manifest.exists() and json.loads(manifest.read_text()) != task:
        raise ValueError(f"Task ID already records a different experiment: {args.id}")
    manifest.write_text(json.dumps(task, indent=2) + "\n")
    if not args.submit:
        print(
            json.dumps(
                {"manifest": str(manifest.resolve()), "source_files": len(files)}
            )
        )
        return
    with tempfile.TemporaryDirectory(prefix="cl-stage-") as temporary:
        archive = Path(temporary) / "code.tar"
        with tarfile.open(archive, "w") as tar:
            for name, content in captured:
                entry = tarfile.TarInfo(str(name))
                entry.size = len(content)
                entry.mode = 0o644
                tar.addfile(entry, io.BytesIO(content))
        kubectl("exec", CONTROL, "--", "mkdir", "-p", remote_code)
        remote_archive = f"{REMOTE}/code/{code_hash}.tar"
        kubectl("cp", archive, f"{CONTROL}:{remote_archive}")
        kubectl("exec", CONTROL, "--", "tar", "-xf", remote_archive, "-C", remote_code)
    staged = f"{REMOTE}/queue/{args.id}.staged"
    kubectl("cp", manifest, f"{CONTROL}:{staged}")
    kubectl("exec", CONTROL, "--", "mv", staged, f"{REMOTE}/queue/{args.id}.json")
    print(
        json.dumps(
            {
                "submitted": args.id,
                "source_sha256": code_hash,
                "source_files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
