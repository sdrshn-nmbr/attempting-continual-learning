import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
import shutil
import site
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [task-runtime] %(message)s"
    )
    task = json.loads(args.task.read_text())
    code = Path(task["code_dir"])
    requirements = code / task.get("requirements", "requirements.txt")
    baseline = Path(__file__).with_name("runtime-requirements.txt")
    dependency_text = baseline.read_text() + requirements.read_text()
    torch_requirement = "torch==" + version("torch") + "\n"
    vendor_sites = [
        str(Path(path).resolve())
        for path in site.getsitepackages()
        if Path(path).is_dir()
    ]
    digest = hashlib.sha256(
        (dependency_text + torch_requirement + json.dumps(vendor_sites)).encode()
    ).hexdigest()[:16]
    environment = Path("/tmp/cl-environments") / digest
    environment.parent.mkdir(exist_ok=True)
    python = environment / "bin/python"
    constraint = environment.parent / f"{digest}-torch.txt"
    constraint.write_text(torch_requirement)
    with (environment.parent / f"{digest}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (environment / ".ready").exists():
            subprocess.run(
                [
                    "uv",
                    "venv",
                    "--system-site-packages",
                    "--python",
                    sys.executable,
                    str(environment),
                ],
                check=True,
            )
            local_site = (
                environment
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            local_site.mkdir(parents=True, exist_ok=True)
            (local_site / "vendor-runtime.pth").write_text(
                "\n".join(vendor_sites) + "\n"
            )
            for vendor in vendor_sites:
                for metadata in Path(vendor).glob("*.dist-info"):
                    destination = local_site / metadata.name
                    if destination.exists():
                        continue
                    shutil.copytree(metadata, destination)
                    files = sorted(
                        path for path in destination.rglob("*") if path.is_file()
                    )
                    with (destination / "RECORD").open("w", newline="") as record:
                        writer = csv.writer(record)
                        writer.writerows(
                            (str(path.relative_to(local_site)), "", "")
                            for path in files
                        )
            subprocess.run(
                ["uv", "pip", "show", "--python", str(python), "torch"], check=True
            )
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--constraint",
                    str(constraint),
                    "-r",
                    str(baseline),
                    "-r",
                    str(requirements),
                ],
                check=True,
            )
            verify = (
                "import json, torch; from importlib.metadata import version; assert torch.version.hip, 'Vendor ROCm Torch was replaced'; assert version('torch') == "
                + repr(version("torch"))
                + "; print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'gpus':torch.cuda.device_count()}))"
            )
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--python",
                    str(python),
                    "python",
                    "-c",
                    verify,
                ],
                check=True,
            )
            (environment / ".ready").touch()
    with (args.output_dir / "packages.txt").open("w") as output:
        subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python)], stdout=output, check=True
        )
    config = args.output_dir / "config.json"
    config.write_text(json.dumps(task["config"], indent=2) + "\n")
    os.environ["VIRTUAL_ENV"] = str(environment)
    os.environ["PATH"] = str(environment / "bin") + ":" + os.environ["PATH"]
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        str(python),
        "python",
        str(code / task.get("entrypoint", "run.py")),
        "--config",
        str(config),
        "--output-dir",
        str(args.output_dir),
    ]
    logger.info(
        "Launching %s with environment %s and visible GPUs %s",
        task["id"],
        digest,
        os.environ.get("ROCR_VISIBLE_DEVICES"),
    )
    os.chdir(code)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
