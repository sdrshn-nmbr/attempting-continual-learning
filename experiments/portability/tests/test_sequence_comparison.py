from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compare import (
    ARMS,
    BOUNDARIES,
    compare_runs,
    compare_timelines,
    contrast_interval,
    factorial_configs,
    load_run,
    timeline_endpoints,
    validate_comparability,
)
from data import SEQUENCE_TASKS, digest, prepare_data, read_data, write_json
from run import stage_schedule

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-comparison")
    outputs = []
    for config_path in sorted((ROOT / "configs").glob("sequence_*.json")):
        config = json.loads(config_path.read_text())
        directory = root / config["arm"]
        directory.mkdir()
        manifest = prepare_data(config, directory)
        data = read_data(
            directory,
            tuple(
                f"{task}_{split}"
                for task in SEQUENCE_TASKS
                for split in ("train", "test")
            ),
        )
        predictions = []
        for task in SEQUENCE_TASKS:
            for index, row in enumerate(data[f"{task}_test"]):
                correct = int(index % 4 == 0)
                chosen = row.gold_idx if correct else (row.gold_idx + 1) % 4
                scores = [-10.0] * 4
                scores[chosen] = 0.0
                predictions.append(
                    {
                        "id": row.id,
                        "task": row.task,
                        "group": row.group,
                        "gold": row.gold_idx,
                        "prediction": chosen,
                        "correct": correct,
                        "scores": scores,
                        "prompt_sha256": digest(row.prompt),
                    }
                )
        panel = {"predictions": predictions}
        timeline = {boundary: {"evaluation": panel} for boundary in BOUNDARIES}
        native = config["method"] == "native"
        if native:
            for boundary in BOUNDARIES[1:]:
                timeline[boundary]["transport"] = {
                    "measured": True,
                    "calibration_examples": 0,
                    "target_optimizer_updates": 0,
                    "checks": {"synthetic_fixture": True},
                    "evaluation": panel,
                }
        schedule = [
            identity
            for stage in range(3)
            for _, _, identity in stage_schedule(
                data, stage, config["seed"], config["training"]
            )
        ]
        reference = {
            "examples": 192,
            "supervised_tokens": 1152,
            "input_tokens": 4608,
            "padded_tokens": 4608,
        }
        budget = {
            "optimizer_updates": 288,
            "current_computed": {
                "examples": 1152,
                "supervised_tokens": 6912,
                "input_tokens": 27648,
                "padded_tokens": 27648,
            },
            "reference_computed": reference,
            "reference_effective": reference
            if config["training"]["replay_weight"]
            else dict.fromkeys(reference, 0),
            "current_forward_backward_groups": 288,
            "reference_forward_backward_groups": 72,
        }
        plan = {"native_active_parameters": 17484032, "lora_parameters": 17436672}
        training = {
            "checks": {"synthetic_fixture": True},
            "budget": budget,
            "parameter_match": plan,
            "computed_schedule_sha256": digest(schedule),
        }
        result = {
            "arm": config["arm"],
            "method": config["method"],
            "replay_weight": config["training"]["replay_weight"],
            "qualified": True,
            "mechanically_qualified": True,
            "sequence_learning_signal": False,
            "config_sha256": digest(config),
            "fixture_sha256": config["sequence"]["sha256"],
            "data_sha256": manifest["data_sha256"],
            "computed_schedule_sha256": digest(schedule),
            "budget": budget,
            "parameter_match": plan,
            "storage_by_boundary": {
                boundary: {
                    "active_trainable_elements": 17484032 if native else 17436672
                }
                for boundary in BOUNDARIES[1:]
            },
            "source": {"tasks": {task: {"acquired": False} for task in SEQUENCE_TASKS}},
            "transport": {"synthetic_fixture": True},
        }
        objects = {
            "config": config,
            "result": result,
            "training_receipt": training,
            "computed_schedule": schedule,
            "evaluation": {
                "timeline": timeline,
                "baselines": {
                    "train": {"raw": panel, "initial": panel},
                    "target": {"raw": panel, "initial": panel},
                },
            },
            "preregistration": {
                "config_sha256": digest(config),
                "source_sha256": {"synthetic_fixture": "3" * 64},
                "selection_rule": {"synthetic_fixture": True},
                "numerical_gate": {"synthetic_fixture": True},
            },
            "task": {
                "id": f"synthetic-{config['arm']}",
                "config": config,
                "source_sha256": "1" * 64,
                "synthetic_fixture": True,
            },
            "execution": {
                "status": "completed",
                "exit_code": 0,
                "task_id": f"synthetic-{config['arm']}",
                "runtime_sha256": "2" * 64,
                "synthetic_fixture": True,
            },
        }
        for name, value in objects.items():
            write_json(directory / f"{name}.json", value)
        outputs.append(directory)
    return outputs


def test_complete_artifact_comparison_keeps_no_learning_distinct_from_qualification(
    synthetic_runs,
):
    result = compare_runs(synthetic_runs)
    assert result["qualified_comparison"] is True
    assert not any(result["sequence_learning_signals"].values())
    for arm in result["arm_endpoints"].values():
        assert arm["final_accuracy"]["estimate"] == 0.25
    for contrast in result["paired_contrasts"].values():
        assert all(endpoint["estimate"] == 0.0 for endpoint in contrast.values())


@pytest.mark.parametrize(
    "drift", ["tokens", "schedule", "source", "active_capacity", "replay_exposure"]
)
def test_actual_receipt_drift_blocks_the_four_arm_comparison(synthetic_runs, drift):
    records = [load_run(path) for path in synthetic_runs]
    record = records[-1]
    if drift == "tokens":
        record["training_receipt"]["budget"]["current_computed"][
            "supervised_tokens"
        ] += 1
    elif drift == "schedule":
        record["training_receipt"]["computed_schedule_sha256"] = "different-schedule"
    elif drift == "source":
        record["task"]["source_sha256"] = "different-source"
    elif drift == "active_capacity":
        record["result"]["storage_by_boundary"]["after_b"][
            "active_trainable_elements"
        ] = 1
    else:
        record["training_receipt"]["budget"]["reference_effective"]["examples"] = 0
    with pytest.raises(
        ValueError,
        match="UNMATCHED_REALIZED|ACTUAL_ACTIVE_CAPACITY|REALIZED_SEQUENCE_BUDGET",
    ):
        validate_comparability(records)


def configs() -> list[dict]:
    return [
        {
            "arm": f"sequence_{method}_{weight}",
            "method": method,
            "seed": 30301,
            "sequence": {"fixture_seed": 303, "sha256": "same-fixture"},
            "training": {
                "steps": 96,
                "batch_size": 4,
                "replay_weight": weight,
                "train_core": method == "native",
            },
        }
        for method in ("native", "lora")
        for weight in (0.0, 0.25)
    ]


def test_factorial_pairing_requires_exactly_the_four_planned_conditions():
    arms, common = factorial_configs(configs())
    assert set(arms) == {
        "native_no_replay",
        "native_replay",
        "lora_no_replay",
        "lora_replay",
    }
    assert common["training"] == {"steps": 96, "batch_size": 4}
    with pytest.raises(ValueError, match="INCOMPLETE_FOUR_ARM_COMPARISON"):
        factorial_configs(configs()[:3])
    repeated = configs()
    repeated[3] = copy.deepcopy(repeated[2])
    with pytest.raises(ValueError, match="DUPLICATE_FACTORIAL_ARM"):
        factorial_configs(repeated)


@pytest.mark.parametrize("drift", ["fixture", "seed", "budget"])
def test_pairing_rejects_hidden_drift_between_arms(drift):
    values = configs()
    if drift == "fixture":
        values[3]["sequence"]["sha256"] = "different-fixture"
    elif drift == "seed":
        values[3]["seed"] = 41901
    else:
        values[3]["training"]["steps"] = 192
    with pytest.raises(ValueError, match="UNMATCHED_FACTORIAL_CONFIGURATIONS"):
        factorial_configs(values)


def predictions(values: list[int], tasks: tuple[str, ...] = ("a",)) -> list[dict]:
    return [
        {
            "id": f"{task}:{group}",
            "task": task,
            "group": str(group),
            "gold": 0,
            "correct": value,
            "prompt_sha256": f"prompt-{task}-{group}",
        }
        for task in tasks
        for group, value in enumerate(values)
    ]


def timeline() -> dict:
    values = {
        "initial": ([0, 0], [0, 0], [0, 0]),
        "after_a": ([1, 1], [1, 0], [0, 0]),
        "after_b": ([1, 0], [1, 1], [1, 0]),
        "after_c": ([1, 0], [0, 1], [1, 1]),
    }
    return {
        boundary: {
            "evaluation": {
                "predictions": [
                    row
                    for task, correct in zip(
                        ("sequence_a", "sequence_b", "sequence_c"), tasks, strict=True
                    )
                    for row in predictions(correct, (task,))
                ]
            }
        }
        for boundary, tasks in values.items()
    }


def test_timeline_separates_acquisition_retention_and_forward_transfer():
    endpoints = timeline_endpoints(timeline())
    observed = {
        name: contrast_interval(terms, seed=91)["estimate"]
        for name, terms in endpoints.items()
    }
    assert observed["acquisition_gain_over_own_initial"] == 1.0
    assert observed["own_stage_increment"] == pytest.approx(2 / 3)
    assert observed["retention_change_after_c"] == -0.5
    assert observed["forward_transfer_before_arrival"] == 0.5


def test_identical_sequences_have_zero_all_factorial_contrasts():
    result = compare_timelines({arm: timeline() for arm in ARMS}, seed=91)
    for contrast in result["paired_contrasts"].values():
        for endpoint in contrast.values():
            assert endpoint["estimate"] == 0.0
            assert endpoint["interval_95"] == [0.0, 0.0]


def test_changed_future_task_panel_is_rejected_even_if_unused_in_endpoint():
    values = timeline()
    future_row = next(
        row
        for row in values["after_a"]["evaluation"]["predictions"]
        if row["task"] == "sequence_c"
    )
    future_row["gold"] = 1
    with pytest.raises(ValueError, match="COMPARISON_IDENTITY_MISMATCH"):
        timeline_endpoints(values)


def test_method_replay_interaction_uses_all_four_paired_arms():
    native_replay = predictions([1, 1, 1, 1])
    native_none = predictions([1, 0, 0, 0])
    lora_replay = predictions([1, 1, 0, 0])
    lora_none = predictions([1, 0, 0, 0])
    result = contrast_interval(
        [
            (1, native_replay),
            (-1, native_none),
            (-1, lora_replay),
            (1, lora_none),
        ],
        seed=30301,
    )
    assert result["estimate"] == 0.5
    assert result["groups"] == 4
    assert result["interval_95"][0] <= 0.5 <= result["interval_95"][1]


def test_repeated_tasks_do_not_become_independent_bootstrap_samples():
    after = predictions([1, 0], ("a", "b", "c"))
    before = predictions([0, 1], ("a", "b", "c"))
    result = contrast_interval([(1, after), (-1, before)], seed=91)
    assert result["estimate"] == 0.0
    assert result["groups"] == 2
    assert result["examples"] == 6
    assert result["interval_95"] == [-1.0, 1.0]


def test_within_arm_acquisition_cancels_a_different_starting_floor():
    native_after = predictions([1, 1, 1, 1])
    native_before = predictions([1, 1, 0, 0])
    lora_after = predictions([1, 1, 0, 0])
    lora_before = predictions([0, 0, 0, 0])
    result = contrast_interval(
        [(1, native_after), (-1, native_before), (-1, lora_after), (1, lora_before)],
        seed=91,
    )
    assert result["estimate"] == 0.0


@pytest.mark.parametrize("field", ["task", "group", "gold", "prompt_sha256"])
def test_same_ids_with_changed_examples_are_rejected(field):
    before = predictions([0, 1])
    after = copy.deepcopy(before)
    after[0][field] = "changed"
    with pytest.raises(ValueError, match="COMPARISON_IDENTITY_MISMATCH"):
        contrast_interval([(1, after), (-1, before)], seed=91)


def test_missing_or_duplicate_rows_are_rejected():
    rows = predictions([0, 1])
    with pytest.raises(ValueError, match="UNPAIRED_COMPARISON_IDS"):
        contrast_interval([(1, rows), (-1, rows[:1])], seed=91)
    with pytest.raises(ValueError, match="DUPLICATE_COMPARISON_ID"):
        contrast_interval([(1, rows + rows[:1])], seed=91)


def test_identical_predictions_have_zero_contrast_and_zero_interval():
    rows = predictions([0, 1, 1, 0])
    result = contrast_interval([(1, rows), (-1, rows)], seed=91)
    assert result["estimate"] == 0.0
    assert result["interval_95"] == [0.0, 0.0]


def test_same_size_groups_with_different_task_panels_are_rejected():
    rows = predictions([0, 1], ("a", "b", "c"))
    rows = [row for row in rows if row["id"] not in {"b:0", "c:1"}]
    with pytest.raises(ValueError, match="UNEQUAL_COMPARISON_GROUP_COVERAGE"):
        contrast_interval([(1, rows)], seed=91)
