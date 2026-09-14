import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(client, bucket, descriptor, destination):
    client.download_file(bucket, descriptor["key"], str(destination))
    if (
        destination.stat().st_size != descriptor["bytes"]
        or digest(destination) != descriptor["sha256"]
    ):
        raise RuntimeError(f"ARCHIVE_CHECKSUM_MISMATCH: {descriptor['key']}")


def extract_archive(archive, destination, names, original_root):
    with subprocess.Popen(
        ["zstd", "-q", "-d", "-c", str(archive)], stdout=subprocess.PIPE
    ) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as contents:
                for member in contents:
                    if member.name not in names:
                        continue
                    if member.islnk() and member.linkname not in names:
                        raise RuntimeError(
                            f"HARDLINK_TARGET_NOT_SELECTED: restore a broader prefix including {member.linkname}"
                        )
                    if member.issym() and Path(member.linkname).is_absolute():
                        target = Path(member.linkname).relative_to(original_root)
                        member = member.replace(
                            linkname=os.path.relpath(
                                destination / target,
                                (destination / member.name).parent,
                            )
                        )
                    contents.extract(member, destination, filter="data")
        except BaseException:
            process.terminate()
            raise
        if process.wait() != 0:
            raise RuntimeError(f"DECOMPRESSION_FAILED: {archive}")


def main():
    parser = argparse.ArgumentParser(
        description="Restore verified portfolio paths from the private R2 archive."
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Original path prefix inside outputs/portfolio; empty selects everything.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="New, nonexistent directory receiving the selected portfolio paths.",
    )
    parser.add_argument(
        "--credentials", type=Path, default=Path.home() / ".config/axport/r2.env"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching file paths without restoring.",
    )
    args = parser.parse_args()
    reference = json.loads(
        (Path(__file__).resolve().parents[1] / "research/r2-portfolio.json").read_text()
    )
    if reference["status"] != "verified":
        raise RuntimeError("ARCHIVE_NOT_VERIFIED: upload is not complete")
    credentials = dict(
        line.split("=", 1)
        for line in args.credentials.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    client = boto3.client(
        "s3",
        endpoint_url=reference["endpoint"],
        region_name="auto",
        aws_access_key_id=credentials["R2_ACCESS_KEY_ID"].strip("\"'"),
        aws_secret_access_key=credentials["R2_SECRET_ACCESS_KEY"].strip("\"'"),
        config=Config(
            retries={"mode": "standard", "max_attempts": 8},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    if not shutil.which("zstd"):
        raise RuntimeError("ZSTD_REQUIRED: install the zstd command before restoring")
    with tempfile.TemporaryDirectory(prefix="portfolio-restore-") as scratch:
        scratch = Path(scratch)
        compressed_index = scratch / "files.jsonl.zst"
        download(client, reference["bucket"], reference["file_index"], compressed_index)
        raw_index = subprocess.check_output(
            ["zstd", "-q", "-d", "-c", str(compressed_index)], text=True
        )
        selected = [
            entry
            for line in raw_index.splitlines()
            if (entry := json.loads(line))["path"].startswith(args.prefix)
        ]
        if not selected:
            raise RuntimeError(f"NO_MATCHING_PATHS: {args.prefix}")
        if args.list:
            for entry in selected:
                print(entry["path"])
            return
        if args.destination is None or args.destination.exists():
            parser.error("--destination must name a new, nonexistent directory")
        names = {entry["path"] for entry in selected}
        parts = {entry["archive"] for entry in selected}
        required = sum(entry["size"] for entry in selected) + 3 * 1024**3
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.destination.parent).free < required:
            raise RuntimeError(
                f"INSUFFICIENT_SPACE: allow {required} bytes for restored files and a temporary archive"
            )
        args.destination.mkdir()
        args.destination = args.destination.resolve()
        for descriptor in reference["archives"]:
            relative_key = descriptor["key"].removeprefix(reference["prefix"] + "/")
            if relative_key not in parts:
                continue
            archive = scratch / "part.tar.zst"
            download(client, reference["bucket"], descriptor, archive)
            extract_archive(
                archive,
                args.destination,
                names,
                Path(reference["source_absolute_path"]),
            )
            archive.unlink()
            print(f"Restored verified archive: {relative_key}", flush=True)
        for entry in selected:
            path = args.destination / entry["path"]
            if not path.exists() and not path.is_symlink():
                raise RuntimeError(f"RESTORE_PATH_MISSING: {entry['path']}")
        print(f"Restored {len(selected)} paths into {args.destination}")


if __name__ == "__main__":
    main()
