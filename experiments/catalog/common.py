import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    if expected["sha256"]:
        return sha256_file(path) == expected["sha256"]
    data = path.read_bytes()
    return (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == expected["git_blob_sha1"]
    )


def verify_asset(path, asset):
    for expected in asset["files"]:
        if not verify_file(path / expected["path"], expected):
            raise ValueError(
                f"SNAPSHOT_CONTENT_MISMATCH: {asset['repo']}@{asset['revision']}: {path / expected['path']}"
            )
    emit(
        "snapshot_content_verified",
        path=str(path),
        repo=asset["repo"],
        revision=asset["revision"],
        files=len(asset["files"]),
    )


def emit(event, **fields):
    print(
        json.dumps({"at": utc_now(), "event": event, **fields}, allow_nan=False),
        flush=True,
    )


def asset_path(root, asset):
    for relative in [asset["relative_path"], *asset.get("reuse_paths", [])]:
        candidate = Path(root) / relative
        if all(
            (candidate / file["path"]).is_file()
            and (candidate / file["path"]).stat().st_size == file["bytes"]
            for file in asset["files"]
        ):
            return candidate
    return Path(root) / asset["relative_path"]


def entry_for(manifest, name):
    return next(entry for entry in manifest["models"] if entry["name"] == name)
