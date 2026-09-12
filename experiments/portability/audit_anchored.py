import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

TASKS = ("sequence_a", "sequence_b", "sequence_c")
BOUNDARIES = ("initial", "after_a", "after_b", "after_c")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(directory, name):
    return json.loads((directory / name).read_text())


def audit_arm(directory):
    result = read(directory, "result.json")
    evaluation = read(directory, "source_evaluation.json")
    training = read(directory, "training_receipt.json")
    execution = read(directory, "execution.json")
    assert execution["exit_code"] == 0 and result["status"] == "completed"
    for name, digest in read(directory, "copy-receipt.json")[
        "all_remote_hashes_exact"
    ].items():
        assert checksum(directory / name) == digest, name
    assert training["pid"] == result["training_pid"] != evaluation["pid"]
    assert training["base_sha256_before"] == training["base_sha256_after"]
    assert training["budget"]["optimizer_updates"] == 288
    assert training["budget"]["current_computed"]["examples"] == 1152
    assert training["budget"]["reference_effective"]["examples"] == 192
    assert training["actual_llm_training_forward_calls"] == 360
    assert training["actual_anchor_native_forward_calls"] == 1440
    assert (
        training["target_abc_data_reads"] == training["target_llm_forward_calls"] == 0
    )
    assert result["evaluation_optimizer_updates"] == 0
    assert len(training["reference_files"]) == 16
    steps = []
    for stage, boundary in enumerate(BOUNDARIES[1:]):
        rows = [
            json.loads(line)
            for line in (directory / f"stages/{boundary}/updates.jsonl")
            .read_text()
            .splitlines()
        ]
        receipt = training["stages"][boundary]
        assert len(rows) == 96 and [row["step"] for row in rows] == list(range(1, 97))
        assert receipt["frozen_before"] == receipt["frozen_after"]
        assert receipt["reference_hashes_before"] == receipt["reference_hashes_after"]
        assert not receipt["source_gate_used_for_training"]
        for row in rows:
            assert row["stage"] == stage and row["task"] == TASKS[stage]
            assert row["optimizer_updates"] == 1
            assert row["anchor"]["weight"] == result["anchor_weight"]
            assert row["actual_anchor_native_forward_calls"] == (3 if stage == 0 else 6)
            assert all(
                math.isfinite(row[key])
                for key in ("current_loss", "gradient_norm")
            )
            if row["reference_ids"]:
                assert math.isfinite(row["reference_loss"])
            else:
                assert row["reference_loss"] is None
                assert row["reference_computed"]["examples"] == 0
            assert math.isfinite(row["anchor"]["loss"])
        steps.extend(rows)
    counts, predictions, regraded = {}, {}, 0
    for index, boundary in enumerate(BOUNDARIES):
        panel = evaluation["timeline"][boundary]
        assert panel["reload"]["passed"] and panel["reload"]["fresh_process"]
        assert panel["reload"]["optimizer_updates"] == 0
        if index:
            assert panel["reload"]["optimizer_exact"]
            assert panel["reload"]["probe_logits"]["bitwise_equal"]
        counts[boundary], predictions[boundary] = {}, {}
        for split, per_task in (("train", 128), ("validation", 32)):
            scored = panel[split]["predictions"]
            required = TASKS if index == 0 else TASKS[:index]
            assert len(scored) == per_task * len(required)
            assert len({row["id"] for row in scored}) == len(scored)
            observed, correct = Counter(), Counter()
            for row in scored:
                assert row["task"] in required
                assert len(row["scores"]) == 4 and all(
                    math.isfinite(x) for x in row["scores"]
                )
                answer = max(range(4), key=row["scores"].__getitem__)
                assert row["prediction"] == answer
                assert row["correct"] == int(answer == row["gold"])
                observed[row["task"]] += 1
                correct[row["task"]] += row["correct"]
            for task in required:
                assert observed[task] == per_task
                assert (
                    panel[split]["metrics"][task]["accuracy"]
                    == correct[task] / per_task
                )
            counts[boundary][split] = dict(correct)
            predictions[boundary][split] = {row["id"]: row for row in scored}
            regraded += len(scored)
    infeasible = {}
    for task, gate in result["source_gates"]["tasks"].items():
        initial = counts["initial"]["validation"][task] / 32
        threshold = result["source_gates"]["thresholds"]["own_validation_gain_min"]
        if initial + threshold > 1:
            infeasible[task] = {
                "initial_validation_accuracy": initial,
                "maximum_possible_gain": 1 - initial,
                "required_gain": threshold,
                "interpretation": "Preregistered criterion unattainable because of baseline headroom. This gate is not evidence against the learning algorithm.",
            }
        own = counts[gate["own_boundary"]]["validation"][task] / 32
        assert gate["validation_gain"] == own - initial
        for boundary, later in gate["later"].items():
            accuracy = counts[boundary]["validation"][task] / 32
            assert later["accuracy"] == accuracy
            assert later["forgetting_from_acquisition"] == max(0, own - accuracy)
    return (
        {
            "result_sha256": checksum(directory / "result.json"),
            "prediction_count_regraded": regraded,
            "counts": counts,
            "infeasible_preregistered_criteria": infeasible,
            "source_gates_as_recorded": result["source_gates"],
            "final_geometry": read(directory, "stages/after_c/geometry.json"),
        },
        predictions,
        steps,
    )


def main():
    root = Path(__file__).resolve().parent / "outputs/anchored-gpu"
    artifacts = read(root, "live-artifact-audit.json")
    assert artifacts["all_saved_references_and_checkpoints_unchanged"]
    arms, predictions, updates = {}, {}, {}
    for name in ("anchor0", "anchor1"):
        arms[name], predictions[name], updates[name] = audit_arm(root / name)
    for filename in (
        "computed_schedule.json",
        "anchor_schedule.json",
        "data_manifest.json",
    ):
        assert read(root / "anchor0", filename) == read(root / "anchor1", filename)
    for left, right in zip(updates["anchor0"], updates["anchor1"], strict=True):
        for key in (
            "current_ids",
            "reference_ids",
            "current_computed",
            "reference_computed",
            "reference_effective",
        ):
            assert left[key] == right[key]
        for key in ("published", "previous"):
            assert left["anchor"][key] == right["anchor"][key]
    before = predictions["anchor0"]["after_c"]["validation"]
    after = predictions["anchor1"]["after_c"]["validation"]
    assert set(before) == set(after)
    for key in before:
        assert all(
            before[key][field] == after[key][field]
            for field in ("task", "group", "gold", "prompt_sha256")
        )
    contrasts = {}
    for selected in (*TASKS, "macro"):
        grouped = {}
        for key, row in sorted(before.items()):
            if selected == "macro" or row["task"] == selected:
                grouped.setdefault(row["group"], []).append(
                    after[key]["correct"] - row["correct"]
                )
        values = [sum(grouped[key]) / len(grouped[key]) for key in sorted(grouped)]
        rng = random.Random(90311)
        draws = sorted(
            sum(rng.choices(values, k=len(values))) / len(values) for _ in range(20000)
        )
        contrasts[selected] = {
            "anchor1_minus_anchor0": sum(values) / len(values),
            "interval_95": [draws[499], draws[19499]],
            "groups": len(values),
        }
    proof = {
        "mechanical_audit_passed": True,
        "audit_source_sha256": checksum(Path(__file__)),
        "matched_288_update_data_and_anchor_schedules": True,
        "saved_references_and_checkpoint_files_verified": artifacts["files"],
        "total_predictions_regraded": sum(
            arm["prediction_count_regraded"] for arm in arms.values()
        ),
        "arms": arms,
        "final_validation_paired_contrasts": contrasts,
        "statistical_scope": "Descriptive paired bootstrap over the32validation input groups,20000draws,seed90311. One mapping and training seed; no independent replication or multiplicity correction.",
        "gate_limitation": "TaskC's50point gain requirement is impossible from the59.375percent initial validation baseline. The overall pass flag cannot establish algorithm success or failure. Raw gates are preserved; feasible retention components remain interpretable. No target evaluation was run.",
    }
    (root / "independent-audit.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in proof.items() if key != "arms"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
