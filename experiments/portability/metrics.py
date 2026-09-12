from __future__ import annotations

from compare import BOUNDARIES, contrast_interval
from data import SEQUENCE_TASKS

SELECTION_RULE = {
    "minimum_task_acquisition_gain": 0.10,
    "maximum_task_forgetting_at_any_later_boundary": 0.05,
    "acquisition_floor": "max(raw, method initialization)",
    "pre_stage_score": "Diagnostic increment and acquisition timing only; positive forward transfer can establish acquisition before arrival.",
    "minimum_target_gain": 0.05,
    "required_acquired_tasks": list(SEQUENCE_TASKS),
    "retention_boundaries": {
        "sequence_a": ["after_b", "after_c"],
        "sequence_b": ["after_c"],
    },
    "absolute_accuracy_floor": None,
    "scope": "Source-backbone learning only; target portability is reported separately.",
}


def task_rows(evaluation: dict, task: str) -> list[dict]:
    rows = [row for row in evaluation["predictions"] if row["task"] == task]
    if not rows:
        raise ValueError(f"MISSING_TASK_METRIC_ROWS: {task}")
    return rows


def paired_group_interval(after: list[dict], before: list[dict], seed: int) -> dict:
    return contrast_interval([(1.0, after), (-1.0, before)], seed)


def sequence_metrics(timeline: dict, raw: dict, seed: int, select: bool = True) -> dict:
    if set(timeline) != set(BOUNDARIES):
        raise ValueError("SEQUENCE_BOUNDARY_MATRIX_INCOMPLETE")
    evaluations = {name: timeline[name]["evaluation"] for name in BOUNDARIES}
    panels = {"raw": raw, **evaluations}
    for panel in panels.values():
        if set(panel["metrics"]) != set(SEQUENCE_TASKS):
            raise ValueError("SEQUENCE_TASK_MATRIX_INCOMPLETE")
        contrast_interval(
            [(1.0, panel["predictions"]), (-1.0, raw["predictions"])], seed
        )
        for task in SEQUENCE_TASKS:
            rows = task_rows(panel, task)
            observed = sum(row["correct"] for row in rows) / len(rows)
            if abs(observed - panel["metrics"][task]["accuracy"]) > 1e-12:
                raise ValueError("METRIC_ACCURACY_RECEIPT_MISMATCH")
    result = {}
    for index, task in enumerate(SEQUENCE_TASKS):
        acquisition_boundary = BOUNDARIES[index + 1]
        pre_boundary = BOUNDARIES[index]
        accuracies = {
            name: panel["metrics"][task]["accuracy"] for name, panel in panels.items()
        }
        acquired_accuracy = accuracies[acquisition_boundary]
        floor = max(accuracies["raw"], accuracies["initial"])
        acquired = (
            acquired_accuracy - floor
            >= SELECTION_RULE["minimum_task_acquisition_gain"] - 1e-12
        )
        reached_before = (
            accuracies[pre_boundary] - floor
            >= SELECTION_RULE["minimum_task_acquisition_gain"] - 1e-12
        )
        later = {}
        for boundary in BOUNDARIES[index + 2 :]:
            backward = accuracies[boundary] - acquired_accuracy
            later[boundary] = {
                "accuracy": accuracies[boundary],
                "backward_transfer": backward,
                "forgetting": max(0.0, -backward),
                "interval": paired_group_interval(
                    task_rows(evaluations[boundary], task),
                    task_rows(evaluations[acquisition_boundary], task),
                    seed,
                ),
            }
        maximum_forgetting = max(
            (row["forgetting"] for row in later.values()), default=0.0
        )
        timing = None
        retained = None
        if select:
            if not acquired:
                timing = "not_acquired"
            elif reached_before:
                timing = "before_arrival_via_forward_transfer"
            else:
                timing = "during_own_stage"
            if later:
                retained = (
                    acquired
                    and maximum_forgetting
                    <= SELECTION_RULE["maximum_task_forgetting_at_any_later_boundary"]
                    + 1e-12
                )
        result[task] = {
            "accuracy": accuracies,
            "acquisition_boundary": acquisition_boundary,
            "pre_stage_boundary": pre_boundary,
            "acquisition_floor": floor,
            "stage_acquisition_gain": acquired_accuracy - accuracies[pre_boundary],
            "gain_over_strongest_floor": acquired_accuracy - floor,
            "gain_over_raw": acquired_accuracy - accuracies["raw"],
            "gain_over_initial": acquired_accuracy - accuracies["initial"],
            "pre_stage_forward_transfer": accuracies[pre_boundary]
            - accuracies["initial"],
            "threshold_reached_before_arrival": reached_before if select else None,
            "acquisition_timing": timing,
            "acquisition_intervals": {
                name: paired_group_interval(
                    task_rows(evaluations[acquisition_boundary], task),
                    task_rows(panels[name], task),
                    seed,
                )
                for name in dict.fromkeys(("raw", "initial", pre_boundary))
            },
            "later_boundaries": later,
            "maximum_forgetting": maximum_forgetting,
            "acquired": acquired if select else None,
            "acquired_and_retained": retained,
        }
    signal = (
        all(row["acquired"] for row in result.values())
        and all(
            row["maximum_forgetting"]
            <= SELECTION_RULE["maximum_task_forgetting_at_any_later_boundary"] + 1e-12
            for row in result.values()
        )
        if select
        else None
    )
    return {
        "tasks": result,
        "selection_rule": SELECTION_RULE if select else None,
        "sequence_learning_signal": signal,
        "post_c_macro_accuracy": sum(
            evaluations["after_c"]["metrics"][task]["accuracy"]
            for task in SEQUENCE_TASKS
        )
        / len(SEQUENCE_TASKS),
        "after_c_gain_interval": paired_group_interval(
            evaluations["after_c"]["predictions"],
            evaluations["initial"]["predictions"],
            seed,
        ),
    }


def transport_metrics(timeline: dict, raw: dict, seed: int) -> dict:
    result = sequence_metrics(timeline, raw, seed, select=False)
    threshold = SELECTION_RULE["minimum_target_gain"]
    boundaries = {}
    for boundary in BOUNDARIES[1:]:
        tasks = {}
        for task, row in result["tasks"].items():
            scores = row["accuracy"]
            gain = scores[boundary] - max(scores["raw"], scores["initial"])
            tasks[task] = {
                "gain_over_both_target_floors": gain,
                "clears_target_gain_threshold": gain >= threshold - 1e-12,
            }
        boundaries[boundary] = {
            "tasks": tasks,
            "all_tasks_clear_target_gain": all(
                row["clears_target_gain_threshold"] for row in tasks.values()
            ),
        }
    return {
        **result,
        "measured": True,
        "minimum_gain_over_both_target_floors": threshold,
        "target_gain_by_boundary": boundaries,
        "after_c_target_gain_signal": boundaries["after_c"][
            "all_tasks_clear_target_gain"
        ],
        "claim_boundary": "Target gain is evaluated separately from source acquisition and retention; this flag alone does not establish a retained transported skill.",
    }
