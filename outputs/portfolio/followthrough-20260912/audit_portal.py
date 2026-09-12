import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_panels(repo, config):
    data = {}
    for key in ("fixture", "old_holdout", "length_holdout", "retention_probes"):
        spec = config["inputs"][key]
        path = repo / "experiments/portability" / spec["path"]
        if checksum(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError(f"AUDIT_INPUT_CHANGED: {key}")
        data[key] = read(path)
    fixture = data["fixture"]
    tasks = ("sequence_a", "sequence_b", "sequence_c")
    panels = {
        "old_validation_test": [
            row for task in tasks for split in ("validation", "test")
            for row in fixture["splits"][f"{task}_{split}"]
        ],
        "old_exhaustive_triples": data["old_holdout"]["rows"],
        "sealed_length4": data["length_holdout"]["rows"],
    }
    seen = {r["group"] for task in tasks for r in fixture["splits"][f"{task}_train"]}
    for panel, rows in panels.items():
        groups = {r["group"] for r in rows}
        if seen & groups:
            raise ValueError(f"AUDIT_INPUT_OVERLAP: {panel}")
        seen.update(groups)
        for row in rows:
            gold = " " + " ".join(str(fixture["provenance"]["rules"][row["task"]][int(v)]) for v in row["group"].split())
            if gold != row["choices"][row["gold_idx"]]:
                raise ValueError("AUDIT_ORACLE")
    panels["validation"] = [row for task in tasks for row in fixture["splits"][f"{task}_validation"]]
    released = data["retention_probes"]
    panels["released_tasks"] = []
    for row, identity in zip(released["validation"], released["indices"], strict=True):
        key = hashlib.sha256(" ".join(row["prompt"].split()).casefold().encode()).hexdigest()
        if key != identity["prompt_sha256"] or row["task"] != identity["task"]:
            raise ValueError("AUDIT_RELEASED_IDENTITY")
        panels["released_tasks"].append({**row, "id": f"released-{key}", "group": key})
    return panels


def panel_score(panel, rows):
    expected = {row["id"]: row for row in rows}
    records = panel["predictions"]
    if len(records) != len(expected) or {r["id"] for r in records} != set(expected):
        raise ValueError("AUDIT_INCOMPLETE_PREDICTIONS")
    counts = defaultdict(lambda: {"correct": 0, "n": 0})
    for record in records:
        row = expected[record["id"]]
        scores = record["scores"]
        if len(scores) != len(row["choices"]) or not all(math.isfinite(s) for s in scores):
            raise ValueError("AUDIT_CHOICE_SUPPORT")
        predicted = max(range(len(scores)), key=scores.__getitem__)
        valid = {
            "prediction": predicted, "gold": row["gold_idx"],
            "correct": int(predicted == row["gold_idx"]), "task": row["task"],
            "prompt_sha256": semantic_hash(row["prompt"]),
            "group": row["group"],
        }
        if any(record[k] != v for k, v in valid.items()):
            raise ValueError(f"AUDIT_ROW_BINDING_OR_SCORE: {row['id']}")
        counts[row["task"]]["correct"] += valid["correct"]
        counts[row["task"]]["n"] += 1
    for task, value in counts.items():
        value["accuracy"] = value["correct"] / value["n"]
        saved = panel["metrics"][task]
        if saved["examples"] != value["n"] or saved["accuracy"] != value["accuracy"]:
            raise ValueError("AUDIT_METRIC_RECOMPUTATION")
    return {
        "per_task": dict(counts), "correct": sum(x["correct"] for x in counts.values()),
        "n": len(records),
    }


def audit(repo, run):
    result = read(run / "result.json")
    execution = read(run / "execution.json")
    if execution["status"] != "completed" or execution["exit_code"] != 0:
        raise ValueError("AUDIT_EXECUTION_NOT_COMPLETE")
    correction = result.get("kind") == "portal_final_retention"
    training_pid = result["source_training_pid"] if correction else result["training_pid"]
    if not result["new_pid_reload"] or result["evaluation_pid"] == training_pid:
        raise ValueError("AUDIT_FRESH_PROCESS_REQUIRED")
    config = read(run / "config.json")
    if correction:
        source = run.parent / Path(config["source_run"]["path"]).name
        original_config = read(source / "config.json")
        if semantic_hash(original_config) != config["source_run"]["config_sha256"]:
            raise ValueError("AUDIT_CORRECTED_SOURCE_CONFIG")
        if result["evaluation_pid"] == result["source_evaluation_pid"]:
            raise ValueError("AUDIT_CORRECTION_FRESH_PROCESS_REQUIRED")
        panels = expected_panels(repo, original_config)
    else:
        panels = expected_panels(repo, config)
    raw = {name: panel_score(value, panels[name]) for name, value in result["raw"].items()}
    arms, retention_coverage = {}, {}
    for arm, value in result["arms"].items():
        if not correction and not value["reload_predictions_exact"]:
            raise ValueError("AUDIT_RELOAD_MISMATCH")
        if correction and not all(item["reload_predictions_exact"] for item in value["checkpoints"].values()):
            raise ValueError("AUDIT_CORRECTED_RELOAD_MISMATCH")
        arms[arm] = {
            step: {name: panel_score(panel, panels[name]) for name, panel in measured.items() if isinstance(panel, dict) and "predictions" in panel}
            for step, measured in value["checkpoints"].items()
        }
        if "retention" in value:
            steps = sorted(arms[arm], key=int)
            measured_steps = [step for step in steps if "released_tasks" in arms[arm][step]]
            retention_coverage[arm] = {
                "measured_steps": measured_steps, "intended_final_step": steps[-1],
                "final_measured": steps[-1] in measured_steps,
            }
            if correction and (int(steps[-1]) != config["expected_final_step"] or measured_steps != ["0", steps[-1]]):
                raise ValueError("AUDIT_CORRECTED_FINAL_STEP")
            for task, comparison in value["retention"].items():
                before = arms[arm][measured_steps[0]]["released_tasks"]["per_task"][task]["accuracy"]
                after = arms[arm][measured_steps[-1]]["released_tasks"]["per_task"][task]["accuracy"]
                if comparison["before"] != before or comparison["after"] != after or comparison["change"] != after - before:
                    raise ValueError("AUDIT_RETENTION_CHANGE")
    if "insertion_comparison" in result:
        for task, comparison in result["insertion_comparison"].items():
            before = arms["untouched"]["0"]["released_tasks"]["per_task"][task]["accuracy"]
            after = arms["constructed"]["0"]["released_tasks"]["per_task"][task]["accuracy"]
            if comparison["before"] != before or comparison["after"] != after or comparison["change"] != after - before:
                raise ValueError("AUDIT_INSERTION_CHANGE")
    return {
        "result_sha256": checksum(run / "result.json"),
        "source_sha256": execution["source_sha256"], "all_predictions_recomputed": True,
        "input_and_prompt_binding": True, "fresh_pid_reload": True,
        "raw": raw, "arms": arms, "retention_coverage": retention_coverage,
        "insertion_comparison": result.get("insertion_comparison"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--wave", type=Path, required=True)
    args = parser.parse_args()
    output = {}
    for run in sorted((args.wave / "runs").glob("followthrough-20260912-portal-*")):
        if (run / "result.json").exists():
            output[run.name] = audit(args.repo, run)
    report = args.wave / "portal-audit.json"
    report.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audited_runs": list(output), "report": str(report.resolve())}))


if __name__ == "__main__":
    main()
