import argparse
import hashlib
import inspect
import json
import logging
import platform
import re
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from portallib import PortalConfig, PortalModel
from portallib.evaluation import PortalInjector
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
GITHUB = "https://api.github.com/repos/ramp-public/portallib"
logger = logging.getLogger(__name__)


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def get_text(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inventory_port(release):
    repo, tag = release
    info = get_json(
        f"https://huggingface.co/api/models/{repo}/revision/{tag}?blobs=true"
    )
    revision = info["sha"]
    config_text = get_text(
        f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
    )
    config = json.loads(config_text)
    base_id = config["base_model_name_or_path"]
    base_revision = config["base_model_revision"]
    base = get_json(
        f"https://huggingface.co/api/models/{base_id}/revision/{base_revision}?blobs=true"
    )
    files = [
        {
            "name": row["rfilename"],
            "bytes": row.get("size"),
            "sha256": row.get("lfs", {}).get("sha256"),
        }
        for row in info["siblings"]
    ]
    parameters = base.get("safetensors", {}).get("total")
    weight_bytes = sum(
        row.get("size", 0)
        for row in base["siblings"]
        if row["rfilename"].endswith(".safetensors")
    )
    if not parameters:
        parameters = (
            sum(base.get("safetensors", {}).get("parameters", {}).values()) or None
        )
    base_config = None
    access_error = None
    try:
        base_config = get_json(
            f"https://huggingface.co/{base_id}/resolve/{base_revision}/config.json"
        )
    except urllib.error.HTTPError as exc:
        access_error = {"status": exc.code, "reason": str(exc.reason)}
    blockers = []
    if base.get("gated"):
        blockers.append(
            {
                "code": "gated_base",
                "detail": "Requires existing authorized HF access or a verified local snapshot; credentials were not changed.",
            }
        )
    if (parameters and parameters * 2 > 250 * 1024**3) or weight_bytes > 250 * 1024**3:
        blockers.append(
            {
                "code": "single_gpu_capacity",
                "detail": "Full base weights exceed the conservative 250 GiB weight allowance on one 288 GiB GPU. No quantization, sharding, or offload substitution is implemented.",
            }
        )
    if base_id == "thinkingmachines/Inkling":
        blockers.append(
            {
                "code": "official_eight_gpu_recipe",
                "detail": "Official Inkling recipe uses eight GPUs. This lane is restricted to one GPU; capability inspection only until separately qualified.",
            }
        )
    loader = "multimodal_lm" if "gemma-4" in repo or "inkling" in repo else "causal_lm"
    return {
        "key": repo.split("/", 1)[1].removeprefix("portal-"),
        "artifact": {
            "id": repo,
            "tag": tag,
            "revision": revision,
            "license": info.get("cardData", {}).get("license"),
            "gated": info.get("gated"),
            "files": files,
            "total_bytes": sum(row["bytes"] or 0 for row in files),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "config": config,
        },
        "base": {
            "id": base_id,
            "revision": base_revision,
            "resolved_revision": base["sha"],
            "license": base.get("cardData", {}).get("license"),
            "gated": base.get("gated"),
            "parameters": parameters,
            "safetensors_file_bytes": weight_bytes,
            "bf16_weight_bytes_estimate": parameters * 2 if parameters else None,
            "architectures": (base_config or {}).get("architectures"),
            "model_type": (base_config or {}).get("model_type"),
            "anonymous_config_access_error": access_error,
            "loader": loader,
            "layer_path": config["layer_path"],
            "allow_heterogeneous_targets": "gemma-4" in repo or "inkling" in repo,
        },
        "status": {
            "artifact_available": True,
            "gpu_tested": False,
            "transfer_tested": False,
            "priority": 0 if base_id.startswith("Qwen/") else 1,
            "run_eligibility": "blocked"
            if any(b["code"] != "gated_base" for b in blockers)
            else "requires_base_and_runtime_smoke",
            "blockers": blockers,
            "ready_config": f"compute/experiments/portal/configs/{'smoke' if 'inkling' in repo else 'pilot'}-{repo.split('/')[-1].removeprefix('portal-')}.json",
        },
        "sources": [
            f"https://huggingface.co/{repo}/tree/{revision}",
            f"https://huggingface.co/{base_id}/tree/{base_revision}",
        ],
    }


def inspect_weights(row, artifact_dir):
    artifact = row["artifact"]
    local_dir = artifact_dir / row["key"]
    paths = {
        name: hf_hub_download(
            artifact["id"], name, revision=artifact["revision"], local_dir=local_dir
        )
        for name in ("config.json", "model.safetensors")
    }
    weights = next(
        item for item in artifact["files"] if item["name"] == "model.safetensors"
    )
    actual_sha = sha256_file(paths["model.safetensors"])
    if actual_sha != weights["sha256"]:
        raise ValueError(f"artifact_digest_mismatch: {artifact['id']}")
    tensors = {}
    with safe_open(paths["model.safetensors"], framework="pt") as handle:
        metadata = handle.metadata()
        keys = list(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
        tensors = {key: handle.get_tensor(key) for key in keys}
    portal = PortalModel.from_pretrained(str(local_dir), local_files_only=True)
    portal.validate_base_model(row["base"]["id"], row["base"]["revision"])
    generated = portal.generate("rte")
    for target, _path in portal.config.resolved_targets():
        a, b = generated[target.key]
        if a.shape != (portal.config.rank, target.in_features) or b.shape != (
            target.out_features,
            portal.config.rank,
        ):
            raise ValueError(f"generated_shape_mismatch: {artifact['id']} {target.key}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(
                f"nonfinite_generated_factors: {artifact['id']} {target.key}"
            )
    row["inspection"] = {
        "status": "passed_cpu_artifact_load_and_factor_generation",
        "sha256": actual_sha,
        "metadata": metadata,
        "tensor_shapes": shapes,
        "generated_projections": len(generated),
        "core_sha256": tensor_digest(
            {k: v for k, v in tensors.items() if k.startswith("core.")}
        ),
        "task_latents_sha256": tensor_digest({"task_latents": tensors["task_latents"]}),
        "weight_download_and_hash_verified": True,
        "base_model_loaded": False,
        "gpu_execution": False,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=Path, default=ROOT.parent.parent / "portal-models.json"
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / ".artifacts")
    args = parser.parse_args()
    torch.set_num_threads(2)
    commit = get_json(f"{GITHUB}/commits/main")["sha"]
    readme_url = (
        f"https://raw.githubusercontent.com/ramp-public/portallib/{commit}/README.md"
    )
    readme = get_text(readme_url)
    pypi = get_json("https://pypi.org/pypi/portallib/0.2.1/json")
    releases = sorted(
        set(
            re.findall(
                r"https://huggingface.co/(RampPublic/portal-[^/)]+)/tree/(v[0-9.]+)",
                readme,
            )
        )
    )
    observed = sorted(
        item.id
        for item in HfApi(token=False).list_models(author="RampPublic")
        if item.id.startswith("RampPublic/portal-")
    )
    if sorted(repo for repo, _tag in releases) != observed:
        raise ValueError(
            f"release_inventory_mismatch: README={releases}, HF={observed}"
        )
    with ThreadPoolExecutor(max_workers=7) as pool:
        rows = list(pool.map(inventory_port, releases))
    registry = {
        "schema_version": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "https://labs.ramp.com/research/",
            "https://labs.ramp.com/research/portal-portable-task-adaptation/",
            readme_url,
        ],
        "github_commit": commit,
        "all_official_ports_accounted_for": True,
        "hf_author_artifact_ids": observed,
        "release_count": len(rows),
        "sdk": {
            "name": "portallib",
            "version": version("portallib"),
            "license": "Apache-2.0",
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "minimal_additional_requirement": "portallib==0.2.1",
            "package_files": [
                {
                    key: file[key]
                    for key in ("filename", "size", "digests", "upload_time_iso_8601")
                }
                for file in pypi["urls"]
            ],
            "requires_dist": pypi["info"]["requires_dist"],
            "datasets_extra_required": False,
            "config_fields": [field.name for field in fields(PortalConfig)],
            "signatures": {
                name: str(inspect.signature(obj))
                for name, obj in {
                    "PortalModel": PortalModel,
                    "PortalModel.forward": PortalModel.forward,
                    "PortalModel.from_pretrained": PortalModel.from_pretrained,
                    "PortalModel.validate_base_model": PortalModel.validate_base_model,
                    "PortalInjector": PortalInjector,
                    "PortalInjector.activate": PortalInjector.activate,
                    "hf_hub_download": hf_hub_download,
                }.items()
            },
            "pinned_submodule_api": "portallib.evaluation.PortalInjector (used by official trainer; not root-exported)",
            "rocm_status": "No CUDA-only extension found in inspected SDK; live ROCm forward/backward remains required.",
        },
        "unsupported_cached_bases": [
            {"id": "Qwen/Qwen3.5-4B", "status": "no_official_released_port"},
            {"id": "Qwen/Qwen3.5-9B", "status": "no_official_released_port"},
        ],
        "ports": rows,
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        try:
            inspect_weights(row, args.artifact_dir)
        except Exception as exc:
            logger.exception("portal_artifact_inspection_failed: %s", row["key"])
            row["inspection"] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            row["status"]["run_eligibility"] = "blocked"
            row["status"]["blockers"].append(
                {"code": "artifact_inspection_failed", "detail": str(exc)}
            )
        args.registry.write_text(json.dumps(registry, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "port": row["key"],
                    "inspection": row["inspection"]["status"],
                    "artifact_bytes": row["artifact"]["total_bytes"],
                    "base_bytes": row["base"]["safetensors_file_bytes"],
                    "blockers": row["status"]["blockers"],
                }
            ),
            flush=True,
        )
    core_hashes = {row["inspection"].get("core_sha256") for row in rows}
    latent_hashes = {row["inspection"].get("task_latents_sha256") for row in rows}
    registry["shared_core_identical_across_all_releases"] = (
        len(core_hashes) == 1 and None not in core_hashes
    )
    registry["released_task_latents_identical_across_all_releases"] = (
        len(latent_hashes) == 1 and None not in latent_hashes
    )
    registry["transfer_contract"] = {
        "method": "Optimize a new latent through the frozen released source core and alignment; copy only that latent into target's identical frozen core and released target alignment.",
        "direct_lora_copy": False,
        "zero_target_training_for_transfer": True,
        "frozen_components": [
            "base weights",
            "released canonical core",
            "released source and target alignments",
        ],
        "limitation": "Novel task learning in a fixed released 256-dimensional latent space is an experimental extension; released benchmark transfer claims do not establish its effectiveness.",
    }
    args.registry.write_text(json.dumps(registry, indent=2) + "\n")
    (ROOT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")


if __name__ == "__main__":
    main()
