import argparse
import json
from pathlib import Path

import torch
from data import digest, write_json
from evaluate_rank import evaluate_adapter_panels
from evaluate_repair import measure_panel
from evaluate_targets import file_sha256, validate_bootstrap
from fresh_inputs import load_fresh_rows
from learner import frozen_base_tensors, load_base, tensor_hash
from metrics import paired_group_interval
from portallib import PortalModel
from portallib.evaluation import PortalInjector
from run import require_gpu

STATIC = ("rank8_final", "constructed_export")


def verify(config):
    if (
        config["kind"] != "constructed_native_generator_behavior"
        or set(config["adapters"]) != set(STATIC)
        or config["fixed_task"] != "rte"
        or config["evaluation"]
        != {
            "batch_size": 8,
            "max_prompt": 768,
            "dtype": "float32",
            "autocast": False,
        }
    ):
        raise ValueError("CONSTRUCTED_BEHAVIOR_PROTOCOL_CHANGED")
    pins = {}
    for spec in [*config["adapters"].values(), config["native"]]:
        for filename, expected in spec["files"].items():
            path = Path(spec["path"]) / filename
            if file_sha256(path) != expected:
                raise ValueError(f"CONSTRUCTED_INPUT_CHANGED: {path}")
            pins[str(path)] = expected
    proof = config["construction_result"]
    if file_sha256(Path(proof["path"])) != proof["sha256"]:
        raise ValueError("CONSTRUCTED_CPU_RECEIPT_CHANGED")
    pins[proof["path"]] = proof["sha256"]
    rows, fixture = load_fresh_rows(config["fresh_inputs"])
    return pins, rows, fixture


def paired_identity(left, right):
    reference = {row["id"]: row for row in right["predictions"]}
    candidate = {row["id"]: row for row in left["predictions"]}
    if reference.keys() != candidate.keys():
        raise ValueError("CONSTRUCTED_ROW_IDENTITY_CHANGED")
    return {
        "choice_agreement": sum(
            row["prediction"] == reference[key]["prediction"]
            for key, row in candidate.items()
        )
        / len(reference),
        "maximum_choice_score_gap": max(
            abs(a - b)
            for key, row in candidate.items()
            for a, b in zip(row["scores"], reference[key]["scores"], strict=True)
        ),
    }


def run(config, output, device="cuda:0"):
    pins, rows, fixture = verify(config)
    if (output / "behavior-seal.json").exists():
        raise FileExistsError("CONSTRUCTED_BEHAVIOR_ALREADY_STARTED")
    write_json(
        output / "behavior-seal.json",
        {
            "config_sha256": digest(config),
            "input_pins": pins,
            "rows": len(rows),
            "groups": len(fixture["unique_inputs"]),
            "behavior_observed": False,
            "optimizer_steps": 0,
        },
    )
    if device != "cpu":
        require_gpu(output, config["seed"])
    base_files, adapters, panels, checks = evaluate_adapter_panels(
        config, rows, output, STATIC, device
    )
    base = load_base(config["source_base"], device)
    if (
        tensor_hash(frozen_base_tensors(base.model))
        != config["source_base_tensor_sha256"]
    ):
        raise ValueError("CONSTRUCTED_BASE_IDENTITY_CHANGED")
    native = (
        PortalModel.from_pretrained(
            config["native"]["path"],
            local_files_only=True,
            dtype=torch.float32,
            device=device,
        )
        .requires_grad_(False)
        .eval()
    )
    native.validate_base_model(base.model_id, base.revision)
    before = tensor_hash(native.state_dict())
    if before != config["native"]["tensor_sha256"]:
        raise ValueError("CONSTRUCTED_NATIVE_RELOAD_CHANGED")
    with (
        torch.inference_mode(),
        torch.autocast(device_type=base.device.type, enabled=False),
        PortalInjector(base.model, native.config) as injector,
        injector.activate(native.generate(config["fixed_task"])),
    ):
        panels["native_fixed_rte"], checks["native_fixed_rte"] = measure_panel(
            base, None, rows, config, output, "native_fixed_rte"
        )
    if before != tensor_hash(native.state_dict()) or verify(config)[0] != pins:
        raise ValueError("CONSTRUCTED_NATIVE_OR_INPUT_MUTATED")
    export = paired_identity(panels["native_fixed_rte"], panels["constructed_export"])
    result = {
        "status": "completed",
        "config_sha256": digest(config),
        "base_files": base_files,
        "adapters": adapters,
        "native_tensor_sha256": before,
        "native_task_for_all_ABC": config["fixed_task"],
        "optimizer_steps": 0,
        "panels": panels,
        "checks": checks,
        "correct": {
            name: sum(row["correct"] for row in panel["predictions"])
            for name, panel in panels.items()
        },
        "rows": len(rows),
        "native_export_parity": export,
        "native_export_parity_passed": export["choice_agreement"] == 1
        and export["maximum_choice_score_gap"] <= 1e-3,
        "native_source_parity": paired_identity(
            panels["native_fixed_rte"], panels["rank8_final"]
        ),
        "native_minus_source": paired_group_interval(
            panels["native_fixed_rte"]["predictions"],
            panels["rank8_final"]["predictions"],
            config["seed"],
        ),
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_bootstrap(config, args.output_dir)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
