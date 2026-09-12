import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def materialize(entry, root):
    model_id, revision = entry["id"], entry["revision"]
    if len(revision) != 40:
        raise ValueError(f"[model-factory] Unpinned model: {model_id}")
    target = Path(entry.get("path", root / "models" / model_id / revision))
    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"[model-factory] Revision mismatch: {model_id}")
    logging.info("[model-factory] Caching %s@%s", model_id, revision)
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=target,
        allow_patterns=entry.get(
            "allow_patterns", ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.yaml", "*.yml", "LICENSE*", "NOTICE*"]
        ),
        max_workers=4,
    )
    weights = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
    if not weights:
        raise ValueError(f"[model-factory] No safe tensor weights: {model_id}")
    for file in weights:
        local = target / file.rfilename
        if not local.is_file() or local.stat().st_size != file.size:
            raise ValueError(f"[model-factory] Incomplete weight file: {local}")
    return {
        **entry,
        "path": str(target),
        "weight_bytes": sum(f.size for f in weights),
        "license": info.card_data.get("license") if info.card_data else None,
        "config_sha256": hashlib.sha256(
            (target / "config.json").read_bytes()
        ).hexdigest(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    entries = json.loads(args.manifest.read_text())
    receipts = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(materialize, entry, args.root): entry for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                receipts[entry["id"]] = future.result()
            except Exception as error:
                logging.exception("[model-factory] Failed: %s", entry["id"])
                failures[entry["id"]] = str(error)
            atomic_json(
                args.root / "downloads.json", {"models": receipts, "failures": failures}
            )
    if failures:
        raise RuntimeError(f"[model-factory] Failed models: {list(failures)}")


if __name__ == "__main__":
    main()
