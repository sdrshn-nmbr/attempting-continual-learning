import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

ROOT = Path("/mnt/shared/cl-smoke")
MANIFEST = Path("/code/models.json")
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [model-cache] %(message)s"
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    marker = ROOT / "storage-check.txt"
    marker.write_text("National Compute shared storage is writable.\n")
    logger.info("Shared storage write/read passed: %s", marker.read_text().strip())
    os.environ["HF_HOME"] = str(ROOT / "huggingface")
    receipts = []
    for model in json.loads(MANIFEST.read_text()):
        destination = ROOT / "models" / model["id"] / model["revision"]
        logger.info("Downloading %s at %s with hf CLI", model["id"], model["revision"])
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "hf",
                "download",
                model["id"],
                "--revision",
                model["revision"],
                "--local-dir",
                str(destination),
                "--max-workers",
                "8",
            ],
            check=True,
        )
        weights = sorted(destination.glob("*.safetensors"))
        if not weights:
            raise RuntimeError(f"No model weights downloaded for {model['id']}")
        index_path = destination / "model.safetensors.index.json"
        if index_path.exists():
            expected = set(json.loads(index_path.read_text())["weight_map"].values())
            missing = expected - {path.name for path in weights}
            if missing:
                raise RuntimeError(
                    f"Missing weight shards for {model['id']}: {sorted(missing)}"
                )
        receipts.append(
            {
                **model,
                "path": str(destination),
                "weight_bytes": sum(path.stat().st_size for path in weights),
                "weight_files": [path.name for path in weights],
                "config_sha256": hashlib.sha256(
                    (destination / "config.json").read_bytes()
                ).hexdigest(),
            }
        )
        logger.info(
            "Download complete: %s (%s weight files)", model["id"], len(weights)
        )
    result = {
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "models": receipts,
    }
    temporary = ROOT / "downloads.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(ROOT / "downloads.json")
    print(json.dumps({"event": "models_cached", **result}), flush=True)


if __name__ == "__main__":
    main()
