import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/portfolio/resume-20260911/runs/resume-20260911-lens-recoverability-sealed"


def accuracy(records, rows):
    if not rows or len(records) != len(rows):
        raise ValueError("HISTORICAL_CODE_CARDINALITY")
    for record, row in zip(records, rows, strict=True):
        if record["text_sha256"] != row["text_sha256"]:
            raise ValueError("HISTORICAL_CODE_ROW_BINDING")
    return sum(record["output"]["code_prediction"] == row["target"] for record, row in zip(records, rows, strict=True)) / len(rows)


def main():
    dataset = ROOT / "experiments/recoverability/inputs/cohort.json"
    data = json.loads(dataset.read_text())
    observations = json.loads((RUN / "study/observations.json").read_text())
    outcomes = json.loads((RUN / "study/outcomes.json").read_text())
    collection = json.loads((RUN / "collection.json").read_text())
    for name in ("study/observations.json", "study/outcomes.json", "study/seal.json"):
        if hashlib.sha256((RUN / name).read_bytes()).hexdigest() != collection["files"][name]["sha256"]:
            raise ValueError("HISTORICAL_CODE_ARTIFACT_HASH")
    seal = json.loads((RUN / "study/seal.json").read_text())["payload"]
    units = []
    for unit in data["units"]:
        observed = observations["units"][unit["id"]]
        before, after = (accuracy(observed[phase]["label"], unit["label"]) for phase in ("before", "after"))
        eligible = before >= 0.75 and after <= 0.5 and before - after >= 0.25
        guard_rows = data["guards"][unit["run"]]
        guard_late = accuracy(observations["guards"][unit["run"]]["after"], guard_rows)
        actions = {}
        for action, result in outcomes[unit["id"]].items():
            score = accuracy(result["label"], unit["label"])
            guard = accuracy(result["guard"], guard_rows)
            actions[action] = {
                "accuracy": score, "guard_accuracy": guard,
                "recovered": eligible and score >= 0.75 and score - after >= 0.25 and guard_late - guard <= 0.1 + 1e-12,
                "utility": score - after - max(0, guard_late - guard),
            }
        units.append({"id": unit["id"], "skill": unit["skill"], "split": unit["split"], "before": before, "after": after, "eligible": eligible, "guard_after": guard_late, "actions": actions})
    result = {
        "kind": "post_hoc_original_code_score_feasibility",
        "boundary": "Post-hoc rescoring of previously executed observations using their original 16-code classifier. This does not revise the failed sealed primary experiment or qualify a recovery predictor. No new GPU operations or repair updates.",
        "inputs": {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "seal_payload_keys": list(seal), "collection_sha256": hashlib.sha256((RUN / "collection.json").read_bytes()).hexdigest()},
        "counts": {split: {"previously_learned": sum(row["before"] >= 0.75 for row in units if row["split"] == split), "forgotten": sum(row["eligible"] for row in units if row["split"] == split), "recovered": dict(Counter(action for row in units if row["split"] == split for action, value in row["actions"].items() if value["recovered"]))} for split in ("train", "test")},
        "units": units,
    }
    target = Path(__file__).with_name("historical-code-recovery-audit.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(target), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
