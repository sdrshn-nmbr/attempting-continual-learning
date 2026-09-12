import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import emit, read_json, sha256_file, utc_now, write_json
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError

SOURCE_SHA = "1a7b8c5b0200c301060bfbceca12653f86d2fa31"
DATASET_SHA = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
EXPECTED_PORTS = {
    "portal-qwen3-1.7b",
    "portal-qwen3-4b",
    "portal-qwen3-8b",
    "portal-gemma-3-4b",
    "portal-gemma-4-e2b",
    "portal-mistral-7b",
    "portal-inkling",
}


def metadata_access(repo, revision, filename, token):
    try:
        get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision), token=token)
        return {"status": "readable"}
    except HfHubHTTPError as exc:
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "http_status": exc.response.status_code,
        }


def asset_info(api, repo, revision, kind):
    info = (api.dataset_info if kind == "dataset" else api.model_info)(
        repo,
        revision=revision,
        files_metadata=True,
    )
    files = {file.rfilename: file for file in info.siblings}
    repo_type = "dataset" if kind == "dataset" else "model"
    if kind == "base":
        if "model.safetensors.index.json" in files:
            index = read_json(
                hf_hub_download(repo, "model.safetensors.index.json", revision=info.sha)
            )
            weights = set(index["weight_map"].values())
        elif "model.safetensors" in files:
            weights = {"model.safetensors"}
        else:
            raise ValueError(f"no standard safetensors checkpoint: {repo}@{info.sha}")
        metadata = {
            name
            for name in files
            if Path(name).suffix in {".json", ".model", ".tiktoken", ".jinja", ".txt"}
            and not name.startswith(("original/", "onnx/"))
        }
        chosen = weights | metadata | ({"README.md"} & files.keys())
    else:
        chosen = set(files)
        weights = {name for name in chosen if name.endswith(".safetensors")}
    selected = []
    for name in sorted(chosen):
        file = files[name]
        selected.append(
            {
                "path": name,
                "bytes": file.size,
                "sha256": file.lfs.sha256 if file.lfs else None,
                "git_blob_sha1": file.blob_id if not file.lfs else None,
            }
        )
    relative = (
        f"cl-portfolio/portal-cache/models/{repo}/{info.sha}"
        if kind != "dataset"
        else f"cl-campaign/datasets/{repo}/{info.sha}"
    )
    result = {
        "key": f"{repo_type}:{repo}@{info.sha}",
        "repo": repo,
        "revision": info.sha,
        "repo_type": repo_type,
        "kind": kind,
        "relative_path": relative,
        "files": selected,
        "bytes": sum(file["bytes"] for file in selected),
        "weight_bytes": sum(files[name].size for name in weights),
        "gated": info.gated,
        "private": info.private,
    }
    if kind != "dataset":
        result["reuse_paths"] = [
            f"cl-smoke/models/{repo}/{info.sha}",
            f"cl-portfolio/models/{repo}/{info.sha}",
        ]
    if kind == "base":
        weight = min(weights)
        result["access"] = {
            "anonymous": metadata_access(repo, info.sha, weight, False),
            "existing_login": metadata_access(repo, info.sha, weight, None),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    listing = list(api.list_models(author="RampPublic", full=True))
    ports = [item for item in listing if item.id.split("/")[-1].startswith("portal-")]
    names = {item.id.split("/")[-1] for item in ports}
    if not EXPECTED_PORTS.issubset(names):
        raise RuntimeError(
            f"expected public ports disappeared: {EXPECTED_PORTS - names}"
        )

    def inspect_port(item):
        artifact = asset_info(api, item.id, item.sha, "artifact")
        config = read_json(hf_hub_download(item.id, "config.json", revision=item.sha))
        base = asset_info(
            api,
            config["base_model_name_or_path"],
            config["base_model_revision"],
            "base",
        )
        base_config = read_json(
            hf_hub_download(base["repo"], "config.json", revision=base["revision"])
        )
        name = item.id.split("/")[-1]
        multimodal = name in {
            "portal-gemma-3-4b",
            "portal-gemma-4-e2b",
            "portal-inkling",
        }
        entry = {
            "name": name,
            "artifact": artifact["key"],
            "base": base["key"],
            "tasks": config["tasks"],
            "layer_path": config["layer_path"],
            "projection_targets": config["projection_targets"],
            "rank": config["rank"],
            "alpha": config["alpha"],
            "loader": "multimodal_lm" if multimodal else "causal_lm",
            "architectures": base_config["architectures"],
            "model_type": base_config["model_type"],
            "trust_remote_code": False,
            "pad_token": "<|endoftext|>" if name == "portal-inkling" else None,
            "heterogeneous": name in {"portal-gemma-4-e2b", "portal-inkling"},
            "relationship": "joint_source"
            if name in {"portal-qwen3-1.7b", "portal-qwen3-4b"}
            else "alignment_refit",
            "single_gpu_status": "blocked_weight_capacity"
            if base["weight_bytes"] > 240 * 1024**3
            else "pending_execution",
            "raw_config_sha256": hashlib.sha256(
                Path(
                    hf_hub_download(item.id, "config.json", revision=item.sha)
                ).read_bytes()
            ).hexdigest(),
        }
        emit(
            "inventory_port",
            name=name,
            base=base["repo"],
            revision=base["revision"],
            bytes=base["bytes"],
        )
        return entry, artifact, base

    with ThreadPoolExecutor(max_workers=7) as pool:
        inspected = list(pool.map(inspect_port, ports))
    dataset = asset_info(api, "RampPublic/portallib-tasks", DATASET_SHA, "dataset")
    assets = {
        asset["key"]: asset
        for _, artifact, base in inspected
        for asset in (artifact, base)
    }
    assets[dataset["key"]] = dataset
    source_dir = Path(__file__).parent / "upstream/src/portallib"
    code_hashes = {
        str(path.relative_to(source_dir)): sha256_file(path)
        for path in sorted(source_dir.rglob("*.py"))
    }
    if not code_hashes:
        raise FileNotFoundError(
            "pinned upstream source checkout required to record library code hashes"
        )
    manifest = {
        "schema_version": 1,
        "observed_at": utc_now(),
        "inventory": {
            "author": "RampPublic",
            "all_public_repos": [item.id for item in listing],
            "port_count": len(ports),
        },
        "source": {
            "repository": "https://github.com/ramp-public/portallib",
            "revision": SOURCE_SHA,
            "package": "portallib",
            "version": "0.2.1",
            "file_sha256": code_hashes,
        },
        "dataset": dataset["key"],
        "assets": assets,
        "models": sorted(
            [item[0] for item in inspected], key=lambda entry: entry["name"]
        ),
        "reusable_caches": [
            {
                "repo": "Qwen/Qwen3.5-4B",
                "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            },
            {
                "repo": "Qwen/Qwen3.5-9B",
                "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            },
        ],
        "proof_boundary": "Published artifact evaluation and serialization only; no training or continual learning claim.",
    }
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
