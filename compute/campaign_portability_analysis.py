import argparse
import hashlib
import json
import random
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_contrast(after, before, seed):
    previous = {row["id"]: row for row in before}
    if len(previous) != len(before) or len(after) != len(before):
        raise ValueError("UNPAIRED_EXAMPLES")
    if len({row["group"] for row in after}) != len(after):
        raise ValueError("EXPECTED_ONE_EXAMPLE_PER_INPUT_TRIPLE")
    differences = []
    for row in sorted(after, key=lambda row: row["group"]):
        reference = previous[row["id"]]
        if (row["group"], row["gold"]) != (reference["group"], reference["gold"]):
            raise ValueError("LABEL_OR_GROUP_MISMATCH")
        differences.append(int(row["correct"]) - int(reference["correct"]))
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(2000)
    )
    return {
        "effect": sum(differences) / len(differences),
        "interval_95": [draws[49], draws[1949]],
        "input_triples": len(differences),
        "improved": differences.count(1),
        "regressed": differences.count(-1),
        "scope": "Paired input-triple bootstrap, 2000 replicates; not training-seed or fresh-fixture uncertainty.",
    }


def analyze(roots):
    records, evaluations, data_hashes, source_hashes, budgets = {}, {}, [], [], []
    seeds = []
    for arm, root in roots.items():
        execution = read(root / "execution.json")
        result = read(root / "result.json")
        training = read(root / "training_receipt.json")
        evaluation = read(root / "evaluation.json")
        config = read(root / "config.json")
        data = read(root / "data_manifest.json")
        if execution["status"] != "completed" or not result["qualified"]:
            raise ValueError(f"UNQUALIFIED_RUN: {root}")
        if not all(training["checks"].values()) or not all(
            evaluation["reload"][name]
            for name in (
                "tensor_hash_equal",
                "probe_logits_exact",
                "probe_logits_close",
            )
        ):
            raise ValueError(f"INCOMPLETE_TRAINING_RELOAD_PROOF: {root}")
        if training["training_pid"] == evaluation["evaluation_pid"]:
            raise ValueError("EVALUATION_NOT_IN_FRESH_PROCESS")
        data_hashes.append(result["data_sha256"])
        source_hashes.append(read(root / "task.json")["source_sha256"])
        budgets.append(
            {key: training[key] for key in ("steps", "b_examples", "b_tokens")}
        )
        seeds.append(config["seed"])
        for stage in ("raw", "initial", "learned"):
            rows = evaluation[stage]["b_test"]["predictions"]
            for row in rows:
                predicted = max(
                    range(len(row["scores"])), key=row["scores"].__getitem__
                )
                if row["prediction"] != predicted or row["correct"] != int(
                    predicted == row["gold"]
                ):
                    raise ValueError("RAW_SCORES_AND_RECORDED_PREDICTION_DISAGREE")
        records[arm] = {
            "run_id": root.name,
            "raw_b_accuracy": result["b_raw_accuracy"],
            "initial_b_accuracy": result["b_initial_accuracy"],
            "learned_b_accuracy": result["b_accuracy"],
            "b_gain": result["b_gain"],
            "b_gain_intervals": result["b_gain_intervals"],
            "a_macro_forgetting": result["a_macro_forgetting"],
            "transport_b_accuracy": result["transport_b_accuracy"],
            "transport_b_gain": result["new_b_transport_gain"],
            "transport_gain_intervals": result["transport_gain_intervals"],
            "transport_a_macro_forgetting": result["transport_a_macro_forgetting"],
            "post_upgrade_learning_signal": result["after_upgrade_signal"],
            "new_skill_transport_signal": result["new_b_transport_signal"],
            "trainable_parameters": training["trainable_parameters"],
            "extra_a_examples": training["a_examples"],
            "extra_a_tokens": training["a_tokens"],
            "source_sha256": source_hashes[-1],
            "config_file_sha256": file_hash(root / "config.json"),
            "config_semantic_sha256": result["config_sha256"],
            "data_sha256": result["data_sha256"],
            "fixture_sha256": result["fixture_sha256"],
            "fixture_seed": data["fixture"]["fixture_seed"],
            "checkpoint_tensor_sha256": training["checkpoint_sha256"],
            "full_base_tensor_sha256": training["base_sha256"],
            "reload_probe_max_error": max(evaluation["reload"]["max_abs"].values()),
            "result_path": str((root / "result.json").resolve()),
            "result_sha256": file_hash(root / "result.json"),
        }
        evaluations[arm] = evaluation
    if (
        len(set(data_hashes)) != 1
        or len(set(source_hashes)) != 1
        or len(set(seeds)) != 1
    ):
        raise ValueError("CROSS_ARM_SOURCE_DATA_OR_SEED_MISMATCH")
    if any(budget != budgets[0] for budget in budgets):
        raise ValueError("UNMATCHED_NEW_SKILL_EXPOSURE")
    for arm in ("latent", "lora"):
        for stage in ("raw", "initial"):
            if (
                evaluations[arm][stage]["b_test"]
                != evaluations["core"][stage]["b_test"]
            ):
                raise ValueError("BASELINE_EVALUATIONS_DIFFER_ACROSS_ARMS")
    contrasts = {
        f"core_minus_{arm}": paired_contrast(
            evaluations["core"]["learned"]["b_test"]["predictions"],
            evaluations[arm]["learned"]["b_test"]["predictions"],
            seeds[0],
        )
        for arm in ("latent", "lora")
    }
    return {
        "status": "reviewed_three_arm_gpu_results",
        "training_seed": seeds[0],
        "source_sha256": source_hashes[0],
        "arms": records,
        "matched_new_skill_budget": budgets[0],
        "raw_and_initial_baselines_identical": True,
        "full_frozen_base_hash_checks_passed": True,
        "fresh_process_checkpoint_and_probe_equality_passed": True,
        "comparisons": contrasts,
        "limits": [
            "The core arm adds old-task replay; only new-skill exposure is matched.",
            "Trainable capacities differ. This is a feasibility comparison, not an efficiency or superiority claim.",
            "LoRA and latent-only old-task retention are protected by external task routing.",
            "Published old-task validation measures retention on that sample; prior release selection exposure is possible.",
            "Learning after the upgrade and transferring newly learned skill through a frozen alignment are separate outcomes.",
            "Tensor equality is based on recorded full-hash and reload assertions; this analysis does not load checkpoint tensors.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    for arm in ("core", "latent", "lora"):
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze({arm: getattr(args, arm) for arm in ("core", "latent", "lora")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arms": {
                    name: {
                        key: value
                        for key, value in arm.items()
                        if key
                        in (
                            "learned_b_accuracy",
                            "b_gain",
                            "a_macro_forgetting",
                            "transport_b_accuracy",
                            "transport_b_gain",
                        )
                    }
                    for name, arm in result["arms"].items()
                },
                "comparisons": result["comparisons"],
            }
        )
    )


if __name__ == "__main__":
    main()
