import argparse
import json
import os
from pathlib import Path

from data import SEQUENCE_TASKS, digest, write_json
from evaluate_rank import evaluate_adapter_panels
from evaluate_targets import file_sha256, validate_bootstrap
from fresh_inputs import load_fresh_rows
from metrics import paired_group_interval, task_rows
from run import require_gpu

CONDITIONS = ("rank8_initial", "initial_exact", "rank8_final", "projection")
EVALUATION = {
    "batch_size": 8,
    "max_prompt": 768,
    "dtype": "float32",
    "autocast": False,
}


def verify(config):
    if (
        config["kind"] != "fixed_alignment_projection_behavior"
        or tuple(config["conditions"]) != CONDITIONS
        or set(config["adapters"]) != set(CONDITIONS)
        or config["evaluation"] != EVALUATION
    ):
        raise ValueError("PROJECTION_BEHAVIOR_CONTRACT_CHANGED")
    checked = {}
    for name, spec in config["adapters"].items():
        if set(spec["files"]) != {"adapter_config.json", "adapter_model.safetensors"}:
            raise ValueError("PROJECTION_ADAPTER_FILE_SET")
        for filename, expected in spec["files"].items():
            path = Path(spec["path"]) / filename
            if file_sha256(path) != expected:
                raise ValueError(f"PROJECTION_INPUT_CHANGED: {path}")
            checked[str(path)] = expected
        adapter = json.loads((Path(spec["path"]) / "adapter_config.json").read_text())
        if {adapter["r"], *adapter.get("rank_pattern", {}).values()} != {8}:
            raise ValueError(f"PROJECTION_RANK_CHANGED: {name}")
    for spec in config["geometry"].values():
        if file_sha256(Path(spec["path"])) != spec["sha256"]:
            raise ValueError("PROJECTION_GEOMETRY_CHANGED")
        checked[spec["path"]] = spec["sha256"]
    rows, fixture = load_fresh_rows(config["fresh_inputs"])
    return checked, rows, fixture


def compare(panels, seed):
    counts = {
        name: sum(row["correct"] for row in panel["predictions"])
        for name, panel in panels.items()
    }
    original = {row["id"]: row for row in panels["rank8_initial"]["predictions"]}
    exact = {row["id"]: row for row in panels["initial_exact"]["predictions"]}
    if original.keys() != exact.keys():
        raise ValueError("PROJECTION_INITIAL_ROW_MISMATCH")
    agreement = sum(
        row["prediction"] == original[key]["prediction"] for key, row in exact.items()
    ) / len(original)
    gap = max(
        abs(a - b)
        for key, row in exact.items()
        for a, b in zip(row["scores"], original[key]["scores"], strict=True)
    )
    return {
        "correct": counts,
        "rows_per_condition": len(original),
        "historical_rank8_850_reproduced": counts["rank8_final"] == 850,
        "initial_exact_control": {
            "choice_agreement": agreement,
            "maximum_choice_score_gap": gap,
            "passed": agreement == 1 and gap <= 1e-4,
        },
        "projection_minus_rank8": {
            task: paired_group_interval(
                task_rows(panels["projection"], task),
                task_rows(panels["rank8_final"], task),
                seed,
            )
            for task in SEQUENCE_TASKS
        },
    }


def run(config, output, device="cuda:0"):
    checked, rows, fixture = verify(config)
    seal_path = output / "behavior-seal.json"
    if seal_path.exists():
        raise FileExistsError("PROJECTION_BEHAVIOR_ALREADY_STARTED")
    write_json(
        seal_path,
        {
            "config_sha256": digest(config),
            "inputs": checked,
            "rows": len(rows),
            "groups": len(fixture["unique_inputs"]),
            "predictions_observed": False,
            "geometry_known_before_behavior": True,
            "optimizer_updates": 0,
        },
    )
    if device != "cpu":
        require_gpu(output, config["seed"])
    base_files, adapters, panels, checks = evaluate_adapter_panels(
        config, rows, output, CONDITIONS, device
    )
    if verify(config)[0] != checked:
        raise ValueError("PROJECTION_INPUTS_CHANGED_DURING_EVALUATION")
    result = {
        "status": "completed",
        "pid": os.getpid(),
        "config_sha256": digest(config),
        "behavior_seal_sha256": file_sha256(seal_path),
        "base_files": base_files,
        "adapter_tensor_hashes": adapters,
        "optimizer_updates": 0,
        "panels": panels,
        "checks": checks,
        "comparisons": compare(panels, config["seed"]),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_bootstrap(config, args.output_dir)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
