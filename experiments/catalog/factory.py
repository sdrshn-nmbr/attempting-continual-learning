from pathlib import Path

import portallib
import torch
from common import asset_path, sha256_file, verify_asset
from portallib import PortalBase, PortalModel
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer


def verify_library(manifest):
    package = Path(portallib.__file__).parent
    failures = [
        name
        for name, digest in manifest["source"]["file_sha256"].items()
        if not (package / name).is_file() or sha256_file(package / name) != digest
    ]
    if failures:
        raise ValueError(f"installed portallib differs from pinned source: {failures}")


def local_asset(manifest, root, key):
    asset = manifest["assets"][key]
    path = asset_path(root, asset)
    verify_asset(path, asset)
    return path


def load_portal(manifest, entry, root):
    path = local_asset(manifest, root, entry["artifact"])
    portal = PortalModel.from_pretrained(
        path, local_files_only=True, device="cpu", dtype=torch.float32
    )
    portal.requires_grad_(False)
    portal.eval()
    return portal


def load_base(manifest, entry, root, expected_gpus=1, base_path=None, *, before_load):
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(
            f"expected {expected_gpus} visible GPU(s); observed {torch.cuda.device_count()}"
        )
    asset = manifest["assets"][entry["base"]]
    path = (
        Path(base_path)
        if base_path is not None
        else local_asset(manifest, root, entry["base"])
    )
    if base_path is not None:
        verify_asset(path, asset)
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    if entry["pad_token"]:
        tokenizer.pad_token = entry["pad_token"]
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(f"tokenizer has no usable padding token: {asset['repo']}")
    model_class = (
        AutoModelForMultimodalLM
        if entry["loader"] == "multimodal_lm"
        else AutoModelForCausalLM
    )
    placement = {"": 0} if expected_gpus == 1 else "balanced"
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": placement,
        "local_files_only": True,
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
    }
    if expected_gpus > 1:
        kwargs["max_memory"] = {index: "270GiB" for index in range(expected_gpus)}
    before_load(
        {
            "asset_key": entry["base"],
            "repo": asset["repo"],
            "revision": asset["revision"],
            "path": str(path.resolve()),
            "verified_files": len(asset["files"]),
            "verified_bytes": sum(file["bytes"] for file in asset["files"]),
            "files": asset["files"],
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "requested_device_map": placement,
            "expected_visible_gpus": expected_gpus,
        }
    )
    model = model_class.from_pretrained(path, **kwargs)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    base = PortalBase(
        model_id=asset["repo"],
        revision=asset["revision"],
        model=model,
        tokenizer=tokenizer,
        layer_path=entry["layer_path"],
        allow_heterogeneous_targets=entry["heterogeneous"],
    )
    for target in entry["projection_targets"]:
        exact = f"{entry['layer_path']}.{target['layer_index']}.{target['module_path']}"
        module = model.get_submodule(exact)
        if (module.in_features, module.out_features) != (
            target["in_features"],
            target["out_features"],
        ):
            raise ValueError(f"base projection dimensions changed: {exact}")
    return base
