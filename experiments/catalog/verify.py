import argparse
import gc
import hashlib
from pathlib import Path

import torch
from common import emit, read_json, sha256_file, write_json
from data import prepare_dataset
from factory import load_portal, verify_library
from run import roundtrip_portal
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(
            name.encode()
            + str(tuple(tensor.shape)).encode()
            + str(tensor.dtype).encode()
        )
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def check_topology(manifest, entry):
    asset = manifest["assets"][entry["base"]]
    config = AutoConfig.from_pretrained(
        asset["repo"], revision=asset["revision"], trust_remote_code=False
    )
    loader = (
        AutoModelForMultimodalLM
        if entry["loader"] == "multimodal_lm"
        else AutoModelForCausalLM
    )
    with torch.device("meta"):
        model = loader.from_config(
            config, attn_implementation="eager", dtype=torch.bfloat16
        )
    for target in entry["projection_targets"]:
        path = f"{entry['layer_path']}.{target['layer_index']}.{target['module_path']}"
        module = model.get_submodule(path)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"PorTAL target is not Linear: {path}")
        if (module.in_features, module.out_features) != (
            target["in_features"],
            target["out_features"],
        ):
            raise ValueError(f"PorTAL target dimensions mismatch: {path}")
    result = {
        "architecture": type(model).__name__,
        "targets": len(entry["projection_targets"]),
        "meta_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "actual_weights_loaded": False,
    }
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-topology", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest, config = read_json(args.manifest), read_json(args.config)
    verify_library(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, selection = prepare_dataset(manifest, args.cache_root, config, args.output_dir)
    report = {
        "manifest_sha256": sha256_file(args.manifest),
        "selection_sha256": selection["selection_sha256"],
        "selection_counts": selection["counts"],
        "models": [],
        "scope": "CPU artifact and data qualification",
    }
    for entry in sorted(
        manifest["models"],
        key=lambda model: (not model["name"].startswith("portal-qwen"), model["name"]),
    ):
        portal = load_portal(manifest, entry, args.cache_root)
        output = args.output_dir / entry["name"]
        output.mkdir(parents=True, exist_ok=True)
        record = {
            "name": entry["name"],
            "roundtrip": roundtrip_portal(portal, output, entry["tasks"]),
            "shared_state_sha256": state_digest(
                {
                    name: value
                    for name, value in portal.state_dict().items()
                    if name == "task_latents" or name.startswith("core.")
                }
            ),
        }
        exports = []
        for task in entry["tasks"]:
            target = portal.export_peft(task, output / "peft" / task)
            weights = load_file(target / "adapter_model.safetensors")
            if len(weights) != 2 * len(entry["projection_targets"]) or not all(
                torch.isfinite(value).all() for value in weights.values()
            ):
                raise ValueError(
                    f"invalid generated PEFT factors: {entry['name']}/{task}"
                )
            exports.append(
                {
                    "task": task,
                    "factor_tensors": len(weights),
                    "sha256": sha256_file(target / "adapter_model.safetensors"),
                }
            )
        record["exports"] = exports
        if args.inspect_topology:
            record["topology"] = check_topology(manifest, entry)
        record["status"] = "qualified_cpu"
        report["models"].append(record)
        write_json(args.output_dir / "verification.json", report)
        emit(
            "artifact_qualified",
            name=entry["name"],
            exported_tasks=len(exports),
            shared_state_sha256=record["shared_state_sha256"],
        )
        del portal
        gc.collect()
    report["identical_shared_state_across_ports"] = (
        len({record["shared_state_sha256"] for record in report["models"]}) == 1
    )
    report["complete"] = len(report["models"]) == manifest["inventory"]["port_count"]
    if not report["identical_shared_state_across_ports"]:
        raise ValueError("published ports do not share exact task-latent/core state")
    write_json(args.output_dir / "verification.json", report)


if __name__ == "__main__":
    main()
