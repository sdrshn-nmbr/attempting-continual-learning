import argparse
import shutil
from pathlib import Path

from common import (
    asset_path,
    emit,
    read_json,
    sha256_file,
    utc_now,
    verify_file,
    write_json,
)
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", default=["artifact", "dataset", "base"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--max-asset-gib", type=float, default=240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    args.root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = {manifest["dataset"]}
    for entry in manifest["models"]:
        if args.models is None or entry["name"] in args.models:
            selected.update([entry["artifact"], entry["base"]])
    assets = [
        asset
        for key, asset in manifest["assets"].items()
        if key in selected and asset["kind"] in args.kinds
    ]
    assets.sort(
        key=lambda asset: (
            {"artifact": 0, "dataset": 1, "base": 2}[asset["kind"]],
            asset["bytes"],
        )
    )
    receipt = {
        "at": utc_now(),
        "manifest_sha256": sha256_file(args.manifest),
        "root": str(args.root),
        "assets": [],
    }
    for asset in assets:
        target = asset_path(args.root, asset)
        record = {
            "repo": asset["repo"],
            "revision": asset["revision"],
            "kind": asset["kind"],
            "path": str(target),
        }
        missing_bytes = sum(
            file["bytes"]
            for file in asset["files"]
            if not (target / file["path"]).is_file()
        )
        free = shutil.disk_usage(args.root).free
        if asset["bytes"] > args.max_asset_gib * 1024**3:
            record.update(status="blocked_asset_ceiling", required_bytes=asset["bytes"])
        elif missing_bytes + 10 * 1024**3 > free:
            record.update(
                status="blocked_storage_capacity",
                required_bytes=missing_bytes,
                free_bytes=free,
            )
        elif args.dry_run:
            record.update(
                status="downloadable", bytes=asset["bytes"], missing_bytes=missing_bytes
            )
        else:
            try:
                emit("download_start", **record, bytes=asset["bytes"])
                snapshot_download(
                    asset["repo"],
                    revision=asset["revision"],
                    repo_type=asset["repo_type"],
                    local_dir=target,
                    allow_patterns=[file["path"] for file in asset["files"]],
                    max_workers=args.workers,
                )
                failures = [
                    file["path"]
                    for file in asset["files"]
                    if not verify_file(target / file["path"], file)
                ]
                if failures:
                    raise RuntimeError(
                        f"checksum verification failed for {asset['repo']}: {failures}"
                    )
                record.update(
                    status="verified", files=len(asset["files"]), bytes=asset["bytes"]
                )
            except HfHubHTTPError as exc:
                record.update(
                    status="blocked_hub_access",
                    error_type=type(exc).__name__,
                    http_status=exc.response.status_code,
                )
        receipt["assets"].append(record)
        write_json(args.output / "download.json", receipt)
        emit("download_result", **record)
    receipt["complete"] = all(
        item["status"] == "verified" for item in receipt["assets"]
    )
    write_json(args.output / "download.json", receipt)


if __name__ == "__main__":
    main()
