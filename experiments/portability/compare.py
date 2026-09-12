from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from data import SEQUENCE_TASKS, digest, read_data, write_json

ARMS = ("native_no_replay", "native_replay", "lora_no_replay", "lora_replay")
BOUNDARIES = ("initial", "after_a", "after_b", "after_c")


def prediction_index(rows: list[dict]) -> dict[str, dict]:
    if not rows:
        raise ValueError("EMPTY_COMPARISON_PREDICTIONS")
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("DUPLICATE_COMPARISON_ID")
    if any(row["correct"] not in (0, 1) for row in rows):
        raise ValueError("INVALID_COMPARISON_CORRECTNESS")
    return indexed


def same_examples(reference: dict[str, dict], rows: dict[str, dict]) -> None:
    if rows.keys() != reference.keys():
        raise ValueError("UNPAIRED_COMPARISON_IDS")
    for key, previous in reference.items():
        if any(
            rows[key][field] != previous[field]
            for field in ("task", "group", "gold", "prompt_sha256")
        ):
            raise ValueError(f"COMPARISON_IDENTITY_MISMATCH: {key}")


def factorial_configs(configs: list[dict]) -> tuple[dict[str, dict], dict]:
    arms = {}
    normalized = []
    for config in configs:
        method = config["method"]
        weight = config["training"]["replay_weight"]
        if method not in {"native", "lora"} or weight not in {0.0, 0.25}:
            raise ValueError("UNKNOWN_FACTORIAL_ARM")
        if config["training"]["train_core"] is not (method == "native"):
            raise ValueError("FACTORIAL_TRAINABLE_CORE_MISMATCH")
        key = f"{method}_{'replay' if weight else 'no_replay'}"
        if key in arms:
            raise ValueError(f"DUPLICATE_FACTORIAL_ARM: {key}")
        arms[key] = config
        common = deepcopy(config)
        del common["arm"], common["method"]
        del common["training"]["replay_weight"], common["training"]["train_core"]
        normalized.append(common)
    if set(arms) != set(ARMS):
        raise ValueError("INCOMPLETE_FOUR_ARM_COMPARISON")
    if len({config["arm"] for config in configs}) != 4:
        raise ValueError("DUPLICATE_FACTORIAL_ARM_NAME")
    if any(config != normalized[0] for config in normalized[1:]):
        raise ValueError("UNMATCHED_FACTORIAL_CONFIGURATIONS")
    return arms, normalized[0]


def contrast_interval(
    terms: list[tuple[float, list[dict]]], seed: int, replicates: int = 2000
) -> dict:
    if not terms or replicates < 100:
        raise ValueError("INVALID_COMPARISON_BUDGET")
    indexed = []
    for coefficient, rows in terms:
        if not math.isfinite(coefficient) or not rows:
            raise ValueError("INVALID_COMPARISON_TERM")
        indexed.append((coefficient, prediction_index(rows)))
    reference = indexed[0][1]
    for _, rows in indexed[1:]:
        same_examples(reference, rows)
    grouped = defaultdict(list)
    coverage = defaultdict(list)
    for key, row in sorted(reference.items()):
        coverage[row["group"]].append(row["task"])
        grouped[row["group"]].append(
            sum(coefficient * rows[key]["correct"] for coefficient, rows in indexed)
        )
    panels = {tuple(sorted(tasks)) for tasks in coverage.values()}
    if len(panels) != 1 or any(
        len(tasks) != len(set(tasks)) for tasks in coverage.values()
    ):
        raise ValueError("UNEQUAL_COMPARISON_GROUP_COVERAGE")
    means = [sum(values) / len(values) for _, values in sorted(grouped.items())]
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choices(means, k=len(means))) / len(means) for _ in range(replicates)
    )
    tail = int(replicates * 0.025)
    return {
        "estimate": sum(means) / len(means),
        "interval_95": [draws[max(0, tail - 1)], draws[replicates - tail - 1]],
        "groups": len(means),
        "examples": len(reference),
        "group_unit": "input_digit_triple_shared_across_tasks",
        "replicates": replicates,
        "bootstrap_seed": seed,
        "scope": "Descriptive paired resampling within one fixture. It does not measure training-seed uncertainty, confirm superiority, or correct for multiple comparisons.",
    }


def timeline_endpoints(timeline: dict) -> dict[str, list[tuple[float, list[dict]]]]:
    if set(timeline) != set(BOUNDARIES):
        raise ValueError("COMPARISON_BOUNDARY_MATRIX_INCOMPLETE")
    panels = {
        boundary: record["evaluation"]["predictions"]
        for boundary, record in timeline.items()
    }
    initial = prediction_index(panels["initial"])
    if {row["task"] for row in initial.values()} != set(SEQUENCE_TASKS):
        raise ValueError("COMPARISON_TASK_MATRIX_INCOMPLETE")
    for rows in panels.values():
        same_examples(initial, prediction_index(rows))
    acquired = []
    before = []
    for index, task in enumerate(SEQUENCE_TASKS):
        acquired.extend(
            row for row in panels[BOUNDARIES[index + 1]] if row["task"] == task
        )
        before.extend(row for row in panels[BOUNDARIES[index]] if row["task"] == task)
    previous_tasks = set(SEQUENCE_TASKS[:-1])
    later_tasks = set(SEQUENCE_TASKS[1:])
    return {
        "initial_accuracy": [(1, panels["initial"])],
        "final_accuracy": [(1, panels["after_c"])],
        "acquisition_gain_over_own_initial": [(1, acquired), (-1, panels["initial"])],
        "own_stage_increment": [(1, acquired), (-1, before)],
        "retention_change_after_c": [
            (1, [row for row in panels["after_c"] if row["task"] in previous_tasks]),
            (-1, [row for row in acquired if row["task"] in previous_tasks]),
        ],
        "forward_transfer_before_arrival": [
            (1, [row for row in before if row["task"] in later_tasks]),
            (-1, [row for row in panels["initial"] if row["task"] in later_tasks]),
        ],
    }


def compare_timelines(timelines: dict[str, dict], seed: int) -> dict:
    if set(timelines) != set(ARMS):
        raise ValueError("INCOMPLETE_FOUR_ARM_TIMELINES")
    endpoints = {arm: timeline_endpoints(timelines[arm]) for arm in ARMS}
    arm_values = {
        arm: {name: contrast_interval(terms, seed) for name, terms in endpoint.items()}
        for arm, endpoint in endpoints.items()
    }
    contrasts = {
        "native_minus_lora_without_replay": {
            "native_no_replay": 1,
            "lora_no_replay": -1,
        },
        "native_minus_lora_with_replay": {"native_replay": 1, "lora_replay": -1},
        "replay_effect_native": {"native_replay": 1, "native_no_replay": -1},
        "replay_effect_lora": {"lora_replay": 1, "lora_no_replay": -1},
        "method_replay_interaction": {
            "native_replay": 1,
            "native_no_replay": -1,
            "lora_replay": -1,
            "lora_no_replay": 1,
        },
    }
    estimates = {}
    for name, weights in contrasts.items():
        estimates[name] = {
            endpoint: contrast_interval(
                [
                    (weight * coefficient, rows)
                    for arm, weight in weights.items()
                    for coefficient, rows in endpoints[arm][endpoint]
                ],
                seed,
            )
            for endpoint in endpoints[ARMS[0]]
        }
    return {
        "arm_endpoints": arm_values,
        "paired_contrasts": estimates,
        "directions": {
            "native_minus_lora": "Positive favors the tested native recipe at the specified replay setting.",
            "replay_effect": "Positive means replay improves the endpoint for the specified method.",
            "method_replay_interaction": "Native replay effect minus LoRA replay effect.",
            "retention_change_after_c": "Signed change for A/B from their own acquisition checkpoints to after C; positive means improvement. C is excluded because it has no later stage.",
            "forward_transfer_before_arrival": "B/C accuracy immediately before their own stage minus their own initialized floor. A is excluded because it has no previous stage.",
            "own_stage_increment": "Descriptive increment during a task's own stage. Positive forward transfer may make this small without invalidating acquisition.",
        },
        "selection_boundary": "Cross-arm estimates do not override per-run acquisition, every-boundary retention, or mechanical qualification gates.",
    }


def load_run(directory: Path) -> dict:
    names = (
        "config",
        "result",
        "training_receipt",
        "evaluation",
        "data_manifest",
        "computed_schedule",
        "preregistration",
        "task",
        "execution",
    )
    record = {
        name: json.loads((directory / f"{name}.json").read_text()) for name in names
    }
    config, result, training = (
        record[name] for name in ("config", "result", "training_receipt")
    )
    if (
        record["execution"]["status"] != "completed"
        or record["execution"]["exit_code"] != 0
    ):
        raise ValueError(f"RUN_NOT_COMPLETED: {directory}")
    if result["qualified"] is not True or result["mechanically_qualified"] is not True:
        raise ValueError(f"RUN_NOT_MECHANICALLY_QUALIFIED: {directory}")
    if not all(training["checks"].values()):
        raise ValueError(f"TRAINING_CHECK_FAILED: {directory}")
    if (
        record["task"]["config"] != config
        or record["execution"]["task_id"] != record["task"]["id"]
    ):
        raise ValueError(f"TASK_IDENTITY_MISMATCH: {directory}")
    if result["config_sha256"] != digest(config) or record["preregistration"][
        "config_sha256"
    ] != digest(config):
        raise ValueError(f"RESULT_CONFIG_MISMATCH: {directory}")
    if (
        result["arm"] != config["arm"]
        or result["method"] != config["method"]
        or result["replay_weight"] != config["training"]["replay_weight"]
        or result["fixture_sha256"] != config["sequence"]["sha256"]
        or result["data_sha256"] != record["data_manifest"]["data_sha256"]
        or result["data_sha256"] != config["sequence"]["generator_sha256"]
    ):
        raise ValueError(f"RESULT_FIXTURE_OR_ARM_MISMATCH: {directory}")
    schedule_hash = digest(record["computed_schedule"])
    if (
        schedule_hash != training["computed_schedule_sha256"]
        or schedule_hash != result["computed_schedule_sha256"]
    ):
        raise ValueError(f"REALIZED_SCHEDULE_HASH_MISMATCH: {directory}")
    if (
        not isinstance(record["computed_schedule"], list)
        or len(record["computed_schedule"]) != 3 * config["training"]["steps"]
    ):
        raise ValueError(f"REALIZED_SCHEDULE_LENGTH_MISMATCH: {directory}")
    if (
        result["budget"] != training["budget"]
        or result["parameter_match"] != training["parameter_match"]
    ):
        raise ValueError(f"RESULT_TRAINING_RECEIPT_MISMATCH: {directory}")
    test_data = read_data(directory, tuple(f"{task}_test" for task in SEQUENCE_TASKS))
    expected = {
        row.id: {
            "task": row.task,
            "group": row.group,
            "gold": row.gold_idx,
            "prompt_sha256": digest(row.prompt),
        }
        for rows in test_data.values()
        for row in rows
    }
    evaluation = record["evaluation"]
    panels = [
        evaluation["baselines"]["train"]["raw"],
        evaluation["baselines"]["train"]["initial"],
        *(evaluation["timeline"][boundary]["evaluation"] for boundary in BOUNDARIES),
    ]
    if set(result["source"]["tasks"]) != set(SEQUENCE_TASKS):
        raise ValueError(f"RESULT_SOURCE_TASKS_INCOMPLETE: {directory}")
    if config["method"] == "native":
        panels.extend(evaluation["baselines"]["target"].values())
        for boundary in BOUNDARIES[1:]:
            transport = evaluation["timeline"][boundary]["transport"]
            if (
                transport["measured"] is not True
                or transport["calibration_examples"] != 0
                or transport["target_optimizer_updates"] != 0
                or not all(transport["checks"].values())
            ):
                raise ValueError(f"TRANSPORT_CONTRACT_MISMATCH: {directory}/{boundary}")
            panels.append(transport["evaluation"])
    for panel in panels:
        same_examples(expected, prediction_index(panel["predictions"]))
        for row in panel["predictions"]:
            scores = row["scores"]
            if len(scores) != 4 or not all(math.isfinite(value) for value in scores):
                raise ValueError(f"INVALID_COMPARISON_SCORES: {directory}/{row['id']}")
            prediction = max(range(4), key=scores.__getitem__)
            if row["prediction"] != prediction or row["correct"] != int(
                prediction == row["gold"]
            ):
                raise ValueError(
                    f"PREDICTION_RECEIPT_MISMATCH: {directory}/{row['id']}"
                )
    record["provenance"] = {
        "directory": str(directory.resolve()),
        "task_id": record["task"]["id"],
        "source_sha256": record["task"]["source_sha256"],
        "runtime_sha256": record["execution"]["runtime_sha256"],
        "artifact_sha256": {
            f"{name}.json": hashlib.sha256(
                (directory / f"{name}.json").read_bytes()
            ).hexdigest()
            for name in names
        },
    }
    return record


def validate_comparability(records: list[dict]) -> tuple[dict[str, dict], dict]:
    configs, common = factorial_configs([record["config"] for record in records])
    by_name = {record["config"]["arm"]: record for record in records}
    arms = {key: by_name[config["arm"]] for key, config in configs.items()}
    matched = []
    recipe = common["training"]
    for arm, record in arms.items():
        training, result = record["training_receipt"], record["result"]
        budget = training["budget"]
        reference_examples = (
            2
            * (recipe["steps"] // recipe["replay_every"])
            * recipe["replay_batch_size"]
        )
        effective = (
            reference_examples if record["config"]["training"]["replay_weight"] else 0
        )
        if (
            budget["optimizer_updates"] != 3 * recipe["steps"]
            or budget["current_computed"]["examples"]
            != 3 * recipe["steps"] * recipe["batch_size"]
            or budget["reference_computed"]["examples"] != reference_examples
            or budget["reference_effective"]["examples"] != effective
        ):
            raise ValueError(f"REALIZED_SEQUENCE_BUDGET_MISMATCH: {arm}")
        expected_effective = (
            budget["reference_computed"]
            if effective
            else dict.fromkeys(budget["reference_computed"], 0)
        )
        if budget["reference_effective"] != expected_effective:
            raise ValueError(f"EFFECTIVE_REFERENCE_TOKEN_MISMATCH: {arm}")
        plan = result["parameter_match"]
        target = plan["native_active_parameters"]
        expected_active = (
            target
            if record["config"]["method"] == "native"
            else plan["lora_parameters"]
        )
        if (
            abs(expected_active - target) / target
            > recipe["capacity_relative_tolerance"]
        ):
            raise ValueError(f"UNMATCHED_ACTIVE_CAPACITY: {arm}")
        if set(result["storage_by_boundary"]) != set(BOUNDARIES[1:]):
            raise ValueError(f"MISSING_STAGE_STORAGE: {arm}")
        if any(
            storage["active_trainable_elements"] != expected_active
            for storage in result["storage_by_boundary"].values()
        ):
            raise ValueError(f"ACTUAL_ACTIVE_CAPACITY_MISMATCH: {arm}")
        matched.append(
            {
                "source": record["task"]["source_sha256"],
                "runtime": record["execution"]["runtime_sha256"],
                "preregistered_source": record["preregistration"]["source_sha256"],
                "selection_rule": record["preregistration"]["selection_rule"],
                "numerical_gate": record["preregistration"]["numerical_gate"],
                "parameter_match": plan,
                "computed_schedule_sha256": training["computed_schedule_sha256"],
                "current_computed": budget["current_computed"],
                "reference_computed": budget["reference_computed"],
                "optimizer_updates": budget["optimizer_updates"],
                "current_forward_backward_groups": budget[
                    "current_forward_backward_groups"
                ],
                "reference_forward_backward_groups": budget[
                    "reference_forward_backward_groups"
                ],
            }
        )
    if any(value != matched[0] for value in matched[1:]):
        raise ValueError("UNMATCHED_REALIZED_COMPARISON_CONTRACT")
    return arms, common


def compare_runs(directories: list[Path]) -> dict:
    arms, common = validate_comparability([load_run(path) for path in directories])
    comparison = compare_timelines(
        {arm: record["evaluation"]["timeline"] for arm, record in arms.items()},
        common["seed"],
    )
    return {
        "kind": "paired_four_arm_sequence_comparison",
        "qualified_comparison": True,
        "fixture": common["sequence"],
        "training_seed": common["seed"],
        "provenance": {arm: record["provenance"] for arm, record in arms.items()},
        "sequence_learning_signals": {
            arm: record["result"]["sequence_learning_signal"]
            for arm, record in arms.items()
        },
        "capacity_exposure": {
            arm: {
                "budget": record["result"]["budget"],
                "parameter_match": record["result"]["parameter_match"],
                "storage_by_boundary": record["result"]["storage_by_boundary"],
            }
            for arm, record in arms.items()
        },
        "source_task_metrics": {
            arm: record["result"]["source"]["tasks"] for arm, record in arms.items()
        },
        "native_transport": {
            arm: record["result"]["transport"]
            for arm, record in arms.items()
            if record["config"]["method"] == "native"
        },
        **comparison,
        "interpretation": "The complete factorial is paired on one fixture and qualified receipts. Acquisition and every-boundary retention remain per-task conditions. Transfer is separate; no confirmed superiority or automatic confirmation launch follows from these contrasts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True, dest="runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.runs)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "qualified_comparison": True,
                "sequence_learning_signals": result["sequence_learning_signals"],
            }
        )
    )


if __name__ == "__main__":
    main()
