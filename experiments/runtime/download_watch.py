import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger("download-watch")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[0]
    except FileNotFoundError:
        return False
    return state not in {"Z", "X"}


def verify_receipt(receipt, config):
    if receipt.get("manifest_sha256") != config["manifest_sha256"]:
        raise ValueError("DOWNLOAD_MANIFEST_MISMATCH")
    if receipt.get("root") != config["root"]:
        raise ValueError("DOWNLOAD_ROOT_MISMATCH")
    assets = receipt.get("assets", [])
    if len(assets) != 1:
        raise ValueError("DOWNLOAD_ASSET_COUNT_MISMATCH")
    asset = assets[0]
    for key, value in config["asset"].items():
        if asset.get(key) != value:
            raise ValueError(f"DOWNLOAD_ASSET_MISMATCH: {key}")
    if receipt.get("complete") is not True or asset.get("status") != "verified":
        raise ValueError(f"DOWNLOAD_NOT_VERIFIED: {asset.get('status')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [download-watch] %(message)s"
    )
    config = json.loads(args.config.read_text())
    if os.environ.get("POD_NAME") != config["pod"]:
        raise RuntimeError("NODE_LOCAL_DOWNLOAD_POD_CHANGED")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launch = json.loads(Path(config["launch_path"]).read_text())
    if launch["pid"] != config["pid"] or launch["pod"] != config["pod"]:
        raise RuntimeError("DOWNLOAD_PROCESS_IDENTITY_CHANGED")
    receipt_path = Path(config["receipt_path"])
    previous_count = -1
    started = time.monotonic()
    LOGGER.info(
        "Watching existing download pid=%s pod=%s; no downloads are initiated",
        config["pid"],
        config["pod"],
    )
    while True:
        if receipt_path.exists():
            payload = receipt_path.read_bytes()
            receipt = json.loads(payload)
            if "complete" not in receipt:
                if not process_alive(config["pid"]):
                    raise RuntimeError("DOWNLOAD_EXITED_WITH_PARTIAL_RECEIPT")
                time.sleep(config["poll_seconds"])
                continue
            verify_receipt(receipt, config)
            result = {
                "kind": "download_verification_only",
                "qualified": True,
                "scientific_result": None,
                "pod": config["pod"],
                "asset": config["asset"],
                "download_receipt": str(receipt_path),
                "receipt_sha256": hashlib.sha256(payload).hexdigest(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(args.output_dir / "result.json", result)
            LOGGER.info("DOWNLOAD_VERIFIED %s", json.dumps(result))
            return
        if not process_alive(config["pid"]):
            raise RuntimeError(
                f"DOWNLOAD_PROCESS_EXITED_WITHOUT_VERIFIED_RECEIPT: {config['pid']}"
            )
        files = list(Path(config["asset"]["path"]).glob("*.safetensors"))
        progress = {
            "pod": config["pod"],
            "pid": config["pid"],
            "completed_weight_files_unverified": len(files),
            "completed_weight_file_bytes_unverified": sum(
                path.stat().st_size for path in files
            ),
            "elapsed_seconds": time.monotonic() - started,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(args.output_dir / "progress.json", progress)
        if len(files) != previous_count:
            LOGGER.info("DOWNLOAD_PROGRESS %s", json.dumps(progress))
            previous_count = len(files)
        time.sleep(config["poll_seconds"])


if __name__ == "__main__":
    main()
