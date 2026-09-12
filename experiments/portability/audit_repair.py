import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_target(directory, expected, fixture_hash):
    result = json.loads((directory / "result.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    copied = json.loads((directory / "copy-receipt.json").read_text())
    assert execution["exit_code"] == 0 and result["status"] == "completed"
    assert result["parent_pid"] != result["evaluator_pid"]
    assert result["fresh_inputs_sha256"] == fixture_hash
    assert result["optimizer_updates"] == 0
    assert result["prepared_sha256"] == file_hash(directory / "prepared.json")
    for name, checksum in copied["all_remote_hashes_exact"].items():
        assert file_hash(directory / name) == checksum, name
    counts, indexed = {}, {}
    for condition, panel in result["panels"].items():
        saved = json.loads((directory / f"panel_{condition}.json").read_text())
        checks = result["checks"][condition]
        assert panel == saved["evaluation"], condition
        extra = {"lora_tensor_sha256"} if condition == "lora_replay" else set()
        assert set(checks) == set(saved["checks"]) | extra, condition
        assert all(checks[key] == value for key, value in saved["checks"].items())
        predictions = panel["predictions"]
        assert len(predictions) == 864
        assert {row["id"] for row in predictions} == set(expected)
        correct = Counter()
        indexed[condition] = {}
        for prediction in predictions:
            row = expected[prediction["id"]]
            assert prediction["task"] == row["task"]
            assert prediction["group"] == row["group"]
            assert prediction["gold"] == row["gold_idx"]
            assert prediction["prompt_sha256"] == canonical_hash(row["prompt"])
            scores = prediction["scores"]
            assert len(scores) == 4 and all(math.isfinite(score) for score in scores)
            answer = max(range(4), key=scores.__getitem__)
            observed = int(answer == row["gold_idx"])
            assert answer == prediction["prediction"]
            assert prediction["correct"] == observed
            correct[row["task"]] += observed
            indexed[condition][(row["task"], row["group"])] = observed
        for task, metrics in panel["metrics"].items():
            assert metrics["accuracy"] == correct[task] / 288
            assert metrics["examples"] == 288 and metrics["truncated_prompts"] == 0
        assert checks["base_and_portal_unchanged"]
        assert not checks["autocast"] and not checks["grad_enabled"]
        assert checks["forward_dtype"] == "torch.float32"
        counts[condition] = dict(correct)
    assert len({value["base_sha256"] for value in result["checks"].values()}) == 1
    if result["role"] == "target":
        identities = [
            result["checks"][condition]["portal_identity"]
            for condition in ("unchanged", "repair", "mismatch")
        ]
        assert len({identity["shared_sha256"] for identity in identities}) == 1
        assert len({identity["alignment_sha256"] for identity in identities}) == 3
        for task, comparison in result["comparisons"].items():
            groups = sorted(group for name, group in indexed["repair"] if name == task)
            for condition, interval in comparison["paired_intervals"].items():
                differences = [
                    indexed["repair"][(task, group)] - indexed[condition][(task, group)]
                    for group in groups
                ]
                assert len(differences) == 288
                assert sum(differences) / 288 == interval["estimate"]
                generator = random.Random(interval["bootstrap_seed"])
                draws = sorted(
                    sum(generator.choices(differences, k=288)) / 288
                    for _ in range(interval["replicates"])
                )
                tail = int(len(draws) * 0.025)
                assert interval["interval_95"] == [draws[tail - 1], draws[-tail - 1]]
            scores = {name: values[task] / 288 for name, values in counts.items()}
            assert comparison["repair_specific_gain_flag"] == (
                scores["repair"] - max(scores["unchanged"], scores["mismatch"])
                >= 0.05 - 1e-12
            )
            assert comparison["target_gain_flag"] == (
                comparison["source_acquired_and_retained_on_original_test"]
                and scores["repair"] - max(scores["raw"], scores["initial"])
                >= 0.05 - 1e-12
            )
    return {
        "task": execution["task_id"],
        "source_sha256": execution["source_sha256"],
        "result_sha256": file_hash(directory / "result.json"),
        "all_prediction_scores_regraded": True,
        "all_reported_paired_intervals_recomputed": result["role"] == "target",
        "prediction_count": result["prediction_count"],
        "fresh_process": True,
        "fp32_no_grad_and_no_updates": True,
        "counts": counts,
    }


def main():
    root = Path(__file__).resolve().parent
    fixture = root / "data/sequence303_unused_inputs.json"
    expected = {row["id"]: row for row in json.loads(fixture.read_text())["rows"]}
    assert len(expected) == 864
    output = root / "outputs/repair-eval-gpu"
    results = {
        name: audit_target(output / name, expected, file_hash(fixture))
        for name in ("source", "qwen4", "mistral7")
    }
    proof = {
        "status": "passed",
        "audit_source_sha256": file_hash(Path(__file__)),
        "fixture_sha256": file_hash(fixture),
        "row_count": 864,
        "groups": 288,
        "total_predictions_regraded": sum(
            value["prediction_count"] for value in results.values()
        ),
        "results": results,
        "claim_boundary": "All remaining inputs of fixture303; no independent mapping or training replicate. Independent audit uses saved choice scores, not another model inference.",
    }
    (output / "independent-audit.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
