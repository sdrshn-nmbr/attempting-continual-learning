import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from peft import PeftModel, get_peft_model_state_dict

from data import SEQUENCE_TASKS, digest, write_json
from evaluate_repair import measure_panel
from evaluate_targets import file_sha256, validate_bootstrap, verify_snapshot
from fresh_inputs import load_fresh_rows
from learner import frozen_base_tensors, load_base, tensor_hash
from metrics import paired_group_interval, task_rows
from run import emit, require_gpu

CONDITIONS = ("full_initial", "full_final", "rank8_initial", "rank8_final")
CRITERIA = {
    "minimum_final_accuracy": 0.90,
    "minimum_fraction_of_full_final_accuracy": 0.95,
    "minimum_gain_over_both_initializations": 0.50,
    "initial_choice_agreement": 1.0,
    "maximum_initial_choice_score_gap": 1e-4,
}


def validate(config):
    if tuple(config["conditions"]) != CONDITIONS or set(config["adapters"]) != set(
        CONDITIONS
    ):
        raise ValueError("RANK_EVAL_FOUR_FIXED_CONDITIONS_REQUIRED")
    if config["criteria"] != CRITERIA or config["evaluation"] != {
        "batch_size": 8,
        "max_prompt": 768,
        "dtype": "float32",
        "autocast": False,
    }:
        raise ValueError("RANK_EVAL_FROZEN_PROTOCOL_CHANGED")
    if not config["source_base"].get("files"):
        raise ValueError("RANK_EVAL_PINNED_BASE_FILES_REQUIRED")


def verify_inputs(config):
    checked = {}
    for name, spec in config["adapters"].items():
        if set(spec["files"]) != {"adapter_config.json", "adapter_model.safetensors"}:
            raise ValueError(f"RANK_EVAL_ADAPTER_FILE_SET: {name}")
        for filename, checksum in spec["files"].items():
            path = Path(spec["path"]) / filename
            if file_sha256(path) != checksum:
                raise ValueError(f"RANK_EVAL_ADAPTER_FILE_CHANGED: {path}")
            checked[str(path)] = checksum
        adapter = json.loads((Path(spec["path"]) / "adapter_config.json").read_text())
        ranks = {adapter["r"], *adapter.get("rank_pattern", {}).values()}
        if name.startswith("rank8") and ranks != {8}:
            raise ValueError(f"RANK_EVAL_PROJECTED_RANK_MISMATCH: {name}")
    proof = config["projection_proof"]
    if file_sha256(Path(proof["path"])) != proof["sha256"]:
        raise ValueError("RANK_EVAL_PROJECTION_PROOF_CHANGED")
    checked[proof["path"]] = proof["sha256"]
    rows, fixture = load_fresh_rows(config["fresh_inputs"])
    return checked, rows, fixture


def prepare(config, output):
    validate(config)
    if (output / "prepared.json").exists():
        raise ValueError("RANK_EVAL_ALREADY_PREPARED")
    checked, rows, fixture = verify_inputs(config)
    receipt = {
        "parent_pid": os.getpid(),
        "config_sha256": digest(config),
        "input_files": checked,
        "row_count": len(rows),
        "groups": len(fixture["unique_inputs"]),
        "criteria": CRITERIA,
    }
    write_json(output / "prepared.json", receipt)
    emit(output, "rank_evaluation_prepared", rows=len(rows), optimizer_updates=0)


def compare(panels, seed):
    results = {}
    for task in SEQUENCE_TASKS:
        scores = {
            name: panel["metrics"][task]["accuracy"] for name, panel in panels.items()
        }
        fraction = (
            scores["rank8_final"] / scores["full_final"] if scores["full_final"] else 0
        )
        gain = scores["rank8_final"] - max(
            scores["full_initial"], scores["rank8_initial"]
        )
        maximum_gain = 1 - max(scores["full_initial"], scores["rank8_initial"])
        results[task] = {
            "accuracies": scores,
            "fraction_of_full_final_accuracy": fraction,
            "gain_over_both_initializations": gain,
            "maximum_possible_gain": maximum_gain,
            "gain_criterion_feasible": maximum_gain >= 0.50,
            "retained_skill_flag": scores["rank8_final"] >= 0.90
            and fraction >= 0.95
            and gain >= 0.50,
            "rank8_final_minus_full_final": paired_group_interval(
                task_rows(panels["rank8_final"], task),
                task_rows(panels["full_final"], task),
                seed,
            ),
        }
    full = {row["id"]: row for row in panels["full_initial"]["predictions"]}
    compressed = panels["rank8_initial"]["predictions"]
    agreement = sum(
        row["prediction"] == full[row["id"]]["prediction"] for row in compressed
    ) / len(full)
    gap = max(
        abs(left - right)
        for row in compressed
        for left, right in zip(row["scores"], full[row["id"]]["scores"], strict=True)
    )
    return {
        "tasks": results,
        "all_tasks_retain_skill": all(
            row["retained_skill_flag"] for row in results.values()
        ),
        "initial_projection_control": {
            "choice_agreement": agreement,
            "maximum_choice_score_gap": gap,
            "passed": agreement == 1.0 and gap <= 1e-4,
        },
        "qualified_rank_control": agreement == 1.0
        and gap <= 1e-4
        and all(row["retained_skill_flag"] for row in results.values()),
    }


def evaluate_prepared(config, output, prepared_sha256, device="cuda:0"):
    validate(config)
    path = output / "prepared.json"
    if file_sha256(path) != prepared_sha256:
        raise ValueError("RANK_EVAL_PREPARED_FILE_CHANGED")
    prepared = json.loads(path.read_text())
    if prepared["parent_pid"] == os.getpid() or prepared["config_sha256"] != digest(
        config
    ):
        raise ValueError("RANK_EVAL_FRESH_PROCESS_REQUIRED")
    checked, rows, fixture = verify_inputs(config)
    if checked != prepared["input_files"]:
        raise ValueError("RANK_EVAL_INPUT_IDENTITY_CHANGED")
    if device != "cpu":
        require_gpu(output, config["seed"])
    base_files = verify_snapshot(config["source_base"])
    base = load_base(config["source_base"], device)
    if (
        tensor_hash(frozen_base_tensors(base.model))
        != config["source_base_tensor_sha256"]
    ):
        raise ValueError("RANK_EVAL_BASE_TENSOR_IDENTITY_MISMATCH")
    model = PeftModel.from_pretrained(
        base.model,
        config["adapters"][CONDITIONS[0]]["path"],
        adapter_name=CONDITIONS[0],
        is_trainable=False,
        local_files_only=True,
    )
    for name in CONDITIONS[1:]:
        model.load_adapter(
            config["adapters"][name]["path"],
            name,
            is_trainable=False,
            local_files_only=True,
        )
    model.requires_grad_(False).eval()
    base = replace(base, model=model)

    def adapter_hashes():
        return {
            name: tensor_hash(
                get_peft_model_state_dict(
                    model, adapter_name=name, save_embedding_layers=False
                )
            )
            for name in CONDITIONS
        }

    before = adapter_hashes()
    if before != {
        name: config["adapters"][name]["tensor_sha256"] for name in CONDITIONS
    }:
        raise ValueError("RANK_EVAL_RELOADED_ADAPTER_IDENTITY_MISMATCH")
    panels, checks = {}, {}
    for name in CONDITIONS:
        model.set_adapter(name, inference_mode=True)
        model.requires_grad_(False).eval()
        if model.active_adapters != [name]:
            raise ValueError(f"RANK_EVAL_ADAPTER_SELECTION_MISMATCH: {name}")
        panels[name], checks[name] = measure_panel(
            base, None, rows, config, output, name
        )
        if adapter_hashes() != before:
            raise ValueError(f"RANK_EVAL_ADAPTER_MUTATION: {name}")
    verify_inputs(config)
    result = {
        "status": "completed",
        "config_sha256": digest(config),
        "prepared_sha256": prepared_sha256,
        "parent_pid": prepared["parent_pid"],
        "evaluator_pid": os.getpid(),
        "base_files": base_files,
        "adapter_tensor_hashes": before,
        "fresh_inputs_sha256": config["fresh_inputs"]["sha256"],
        "rows": len(rows),
        "groups": len(fixture["unique_inputs"]),
        "prediction_count": sum(len(panel["predictions"]) for panel in panels.values()),
        "optimizer_updates": 0,
        "panels": panels,
        "checks": checks,
        "comparisons": compare(panels, config["seed"]),
        "criteria": CRITERIA,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output / "result.json", result)
    emit(
        output,
        "rank_evaluation_completed",
        predictions=result["prediction_count"],
        optimizer_updates=0,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "evaluate"), default="run")
    parser.add_argument("--prepared-sha256")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "run":
        validate_bootstrap(config, args.output_dir)
        prepare(config, args.output_dir)
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(Path(__file__).resolve()),
                "--config",
                str(args.config.resolve()),
                "--output-dir",
                str(args.output_dir.resolve()),
                "--phase",
                "evaluate",
                "--prepared-sha256",
                file_sha256(args.output_dir / "prepared.json"),
            ],
            check=True,
        )
    else:
        if not args.prepared_sha256:
            raise ValueError("RANK_EVAL_PREPARED_HASH_REQUIRED")
        evaluate_prepared(config, args.output_dir, args.prepared_sha256)


if __name__ == "__main__":
    main()
