import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from config import Config, digest


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def correct_count(score, rows, task_index, classes):
    predictions = score["predictions"]
    correct = sum(
        prediction == row["code"]
        for prediction, row in zip(predictions, rows, strict=True)
    )
    require(
        score["task"] == task_index
        and score["split"] == "test"
        and score["examples"] == len(predictions) == len(rows)
        and all(
            type(prediction) is int and 0 <= prediction < classes
            for prediction in predictions
        )
        and correct == score["correct"],
        f"SUPPORT_RAW_SCORE_MISMATCH task={task_index}",
    )
    return {"correct": correct, "examples": len(rows)}


def support_gates(full, bounded):
    pairs = list(zip(full["acquisition"], bounded["acquisition"], strict=True))
    pairs.append((full["final_old"], bounded["final_old"]))
    require(
        bool(full["acquisition"])
        and all(left["examples"] == right["examples"] > 0 for left, right in pairs),
        "SUPPORT_GATE_DENOMINATOR_MISMATCH",
    )
    require(
        all(
            type(score["correct"]) is int and 0 <= score["correct"] <= score["examples"]
            for pair in pairs
            for score in pair
        ),
        "SUPPORT_GATE_COUNT_INVALID",
    )
    return {
        "both_acquisition_at_least_90_percent_every_task": all(
            10 * score["correct"] >= 9 * score["examples"]
            for pair in pairs[:-1]
            for score in pair
        ),
        "bounded_acquisition_at_most_5_points_below_full": all(
            20 * (left["correct"] - right["correct"]) <= left["examples"]
            for left, right in pairs[:-1]
        ),
        "bounded_final_old_at_least_90_percent": 10 * bounded["final_old"]["correct"]
        >= 9 * bounded["final_old"]["examples"],
        "bounded_final_old_at_most_5_points_below_full": 20
        * (full["final_old"]["correct"] - bounded["final_old"]["correct"])
        <= full["final_old"]["examples"],
    }


def support_result(metrics):
    cfg = Config(**metrics["config"])
    data = metrics["provenance"]["data"]
    require(
        metrics["status"] == "completed"
        and metrics["completed"]
        and metrics["total_updates"] == cfg.tasks * cfg.updates_per_task,
        "SUPPORT_RUN_INCOMPLETE",
    )
    require(
        cfg.methods == ["balanced_replay"]
        and set(metrics["methods"]) == {"balanced_replay"},
        "SUPPORT_METHOD_MISMATCH",
    )
    require(
        metrics["signature"]
        == digest({"config": cfg.as_dict(), "provenance": metrics["provenance"]}),
        "SUPPORT_SIGNATURE_MISMATCH",
    )
    stages = metrics["methods"]["balanced_replay"]["tasks"]
    require(
        len(stages) == len(data["selected_rows"]) == cfg.tasks,
        "SUPPORT_STAGE_COUNT_MISMATCH",
    )
    retained = []
    acquisition = []
    final_scores = []
    slots = []
    for task_index, stage in enumerate(stages):
        require(
            stage["completed"] and stage["updates"] == cfg.updates_per_task,
            "SUPPORT_STAGE_INCOMPLETE",
        )
        before = stage["replay_buffer_before"]
        require(
            before["members"] == retained
            and before["members_sha256"] == digest(retained)
            and before["examples"] == len(retained),
            "SUPPORT_BUFFER_BEFORE_MISMATCH",
        )
        eligible = {tuple(ref) for ref in retained}
        task_slots = []
        require(
            len(stage["training_curve"]) == cfg.updates_per_task,
            "SUPPORT_TRAINING_CURVE_INCOMPLETE",
        )
        current_count = replay_count = 0
        for update in stage["training_curve"]:
            refs = update["training_refs"]
            require(
                len(refs) == cfg.effective_batch_size
                and update["batch_sha256"] == digest(refs),
                "SUPPORT_BATCH_MISMATCH",
            )
            batch_slots = []
            batch_current = batch_replay = 0
            for source, index in refs:
                require(
                    0 <= source <= task_index and index >= 0, "SUPPORT_FUTURE_REFERENCE"
                )
                row = data["selected_rows"][source]["train"][index]
                require(row["source_split"] == "train", "SUPPORT_NONTRAIN_REFERENCE")
                if source == task_index:
                    batch_current += 1
                    batch_slots.append(["current", source, index])
                else:
                    require(
                        (source, index) in eligible, "SUPPORT_REFERENCE_OUTSIDE_BUFFER"
                    )
                    batch_replay += 1
                    batch_slots.append(["replay", row["code"]])
            expected_replay = (
                int(cfg.effective_batch_size * cfg.replay_fraction) if task_index else 0
            )
            require(
                batch_replay == expected_replay
                and batch_current == cfg.effective_batch_size - expected_replay,
                "SUPPORT_EXPOSURES_MISMATCH",
            )
            require(
                update["current_examples"] == batch_current
                and update["replay_examples"] == batch_replay,
                "SUPPORT_UPDATE_RECEIPT_MISMATCH",
            )
            current_count += batch_current
            replay_count += batch_replay
            task_slots.append(batch_slots)
        require(
            stage["current_examples"] == current_count
            and stage["replay_examples"] == replay_count
            and stage["training_examples"] == current_count + replay_count,
            "SUPPORT_STAGE_EXPOSURES_MISMATCH",
        )
        slots.append(task_slots)
        after = stage["replay_buffer_after"]
        allowed = eligible | {
            (task_index, i)
            for i in range(len(data["selected_rows"][task_index]["train"]))
        }
        members = {tuple(ref) for ref in after["members"]}
        require(
            members <= allowed and len(members) == len(after["members"]),
            "SUPPORT_EVICTED_ROW_REACQUIRED",
        )
        limit = (
            len(allowed)
            if cfg.replay_capacity is None
            else min(cfg.replay_capacity, len(allowed))
        )
        counts = Counter(
            str(data["selected_rows"][t]["train"][i]["code"]) for t, i in members
        )
        require(
            len(members) == limit
            and len(counts) == (task_index + 1) * cfg.classes_per_task
            and max(counts.values()) - min(counts.values()) <= 1,
            "SUPPORT_RETAINED_QUOTA_MISMATCH",
        )
        require(
            after["examples"] == len(members)
            and after["examples_by_code"] == dict(counts)
            and after["members_sha256"] == digest(after["members"])
            and after == data["replay_buffers_after_task"][task_index],
            "SUPPORT_BUFFER_AFTER_MISMATCH",
        )
        retained = after["members"]
        require(
            len(stage["test_after_task"]) == task_index + 1,
            "SUPPORT_ENDPOINT_COUNT_MISMATCH",
        )
        final_scores = [
            correct_count(
                score,
                data["selected_rows"][index]["test"],
                index,
                cfg.tasks * cfg.classes_per_task,
            )
            for index, score in enumerate(stage["test_after_task"])
        ]
        acquisition.append(final_scores[-1])
    return {
        "acquisition": acquisition,
        "final_old": {
            key: sum(score[key] for score in final_scores[:-1])
            for key in ("correct", "examples")
        },
        "shared_slots": slots,
    }


def compare_support(full, bounded):
    full_cfg, bounded_cfg = Config(**full["config"]), Config(**bounded["config"])
    require(
        full_cfg.replay_capacity is None and bounded_cfg.replay_capacity == 32,
        "SUPPORT_CAPACITIES_MUST_BE_FULL_AND_32",
    )
    require(
        {**full_cfg.as_dict(), "replay_capacity": 32} == bounded_cfg.as_dict(),
        "SUPPORT_CONFIGS_DIFFER_BEYOND_CAPACITY",
    )
    require(
        full_cfg.acquisition_floor == 0.9 and full_cfg.max_acquisition_drop == 0.05,
        "SUPPORT_PREDECLARED_THRESHOLD_MISMATCH",
    )
    require(
        {key: value for key, value in full["provenance"].items() if key != "data"}
        == {
            key: value for key, value in bounded["provenance"].items() if key != "data"
        },
        "SUPPORT_MODEL_RUNTIME_SOURCE_MISMATCH",
    )
    for key in ("selected_rows", "code_token_ids", "evaluation_sha256"):
        require(
            full["provenance"]["data"][key] == bounded["provenance"]["data"][key],
            f"SUPPORT_DATA_MISMATCH {key}",
        )
    results = {"full": support_result(full), "bounded": support_result(bounded)}
    require(
        results["full"]["shared_slots"] == results["bounded"]["shared_slots"],
        "SUPPORT_CURRENT_OR_CLASS_SCHEDULE_MISMATCH",
    )
    schedule_sha256 = digest(results["full"].pop("shared_slots"))
    results["bounded"].pop("shared_slots")
    gates = support_gates(results["full"], results["bounded"])
    return {
        "seed": full_cfg.seed,
        "capacity_is_only_config_difference": True,
        "current_refs_replay_class_slots_and_exposures_match": True,
        "shared_slots_sha256": schedule_sha256,
        "raw_counts": results,
        "gates": gates,
        "primary_gate_passed": all(gates.values()),
        "scope": "Bounded replay candidate support; the loader retains historical rows, so no bounded host-RAM claim.",
    }


def read_metrics(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-metrics", type=Path, required=True)
    parser.add_argument("--bounded-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full, full_hash = read_metrics(args.full_metrics)
    bounded, bounded_hash = read_metrics(args.bounded_metrics)
    result = compare_support(full, bounded)
    result["metrics_sha256"] = {"full": full_hash, "bounded": bounded_hash}
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    pin = protocol["descriptive_no_replay"][str(result["seed"])]
    reference, reference_hash = read_metrics(Path(pin["metrics_path"]))
    require(
        reference_hash == pin["metrics_sha256"],
        "SUPPORT_DESCRIPTIVE_REFERENCE_HASH_MISMATCH",
    )
    require(
        reference["config"]["seed"] == result["seed"],
        "SUPPORT_DESCRIPTIVE_REFERENCE_SEED_MISMATCH",
    )
    scores = reference["methods"]["persistent"]["tasks"][-1]["test_after_task"][:-1]
    result["descriptive_no_replay"] = {
        "metrics_sha256": reference_hash,
        "final_old": {
            key: sum(score[key] for score in scores) for key in ("correct", "examples")
        },
        "enters_primary_gate": False,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "seed": result["seed"],
                "primary_gate_passed": result["primary_gate_passed"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
