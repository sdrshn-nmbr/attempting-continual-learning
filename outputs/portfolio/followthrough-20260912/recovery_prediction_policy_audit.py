import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(left, right):
    if not math.isclose(left, right, abs_tol=1e-11, rel_tol=1e-11):
        raise ValueError(f"POLICY_AUDIT_NUMERICAL_DISAGREEMENT {left} != {right}")


def grouped(values, streams):
    sources = {
        source: mean(v for v, s in zip(values, streams, strict=True) if s == source)
        for source in sorted(set(streams))
    }
    return {
        "pooled": mean(values),
        "by_source": sources,
        "equal_source_mean": mean(sources.values()),
    }


def check_grouped(actual, expected):
    close(actual["pooled"], expected["pooled"])
    close(actual["equal_source_mean"], expected["equal_source_mean"])
    for source in actual["by_source"]:
        close(actual["by_source"][source], expected["by_source"][source])


class Inputs:
    def __init__(self, root):
        self.root = root
        self.hashes = {}
        self.executions = {}

    def load(self, suffix, filename):
        run = self.root / ("followthrough-20260912-recovery-predict-" + suffix)
        if not (run / "collection.json").is_file():
            raise ValueError(f"POLICY_AUDIT_WAIT_FOR_COMPLETE_COLLECTION {suffix}")
        execution = read(run / "execution.json")
        if (
            execution["status"] != "completed"
            or execution["exit_code"] != 0
            or execution["source_sha256"]
            != "99ac3030bb56b44579b07e0f672ce098b0831f9c871b350d087201ad546ba8d1"
        ):
            raise ValueError(f"POLICY_AUDIT_EXECUTION {suffix}")
        self.executions[suffix] = execution
        receipt = read(run / "study/receipt.json")
        if digest(receipt["payload"]) != receipt["sha256"]:
            raise ValueError("POLICY_AUDIT_RECEIPT_HASH")
        path = run / "study" / filename
        actual = sha(path)
        if receipt["payload"]["files"][filename] != actual:
            raise ValueError(f"POLICY_AUDIT_INPUT_HASH {path}")
        self.hashes[str(path)] = actual
        return read(path)


def score(records, rows, codes):
    if len(records) != len(rows) or not records:
        raise ValueError("POLICY_AUDIT_ROWS")
    correct = 0
    for record, row in zip(records, rows, strict=True):
        if any(record[key] != row[key] for key in ("intent", "target", "text_sha256")):
            raise ValueError("POLICY_AUDIT_ROW_BINDING")
        logits = record["output"]["code_logits"]
        if len(logits) != 16 or not all(math.isfinite(x) for x in logits):
            raise ValueError("POLICY_AUDIT_SIXTEEN_CODES")
        prediction = codes[max(range(16), key=logits.__getitem__)]
        if prediction != record["output"]["prediction"]:
            raise ValueError("POLICY_AUDIT_NATIVE_ARGMAX")
        correct += prediction == row["target"]
    return correct / len(records)


def outcome(unit, data):
    stream = data["streams"][unit["stream"]]
    target = stream["units"][unit["intent"]]["test"]
    guards = [
        row for intent in stream["new"] for row in stream["units"][intent]["test"]
    ]
    after = score(unit["baseline"]["after"]["test"], target, data["codes"])
    guard_before = score(unit["baseline"]["after"]["guard"], guards, data["codes"])
    actions = {}
    for budget, panel in unit["actions"].items():
        accuracy = score(panel["test"], target, data["codes"])
        guard = score(panel["guard"], guards, data["codes"])
        gain, guard_loss = accuracy - after, max(0, guard_before - guard)
        actions[budget] = {
            "accuracy": accuracy,
            "gain": gain,
            "guard_accuracy": guard,
            "utility": gain - guard_loss,
            "recovered": accuracy >= 0.8
            and gain >= 0.3 - 1e-12
            and guard_loss <= 0.1 + 1e-12,
        }
    return {
        "id": unit["id"],
        "actions": actions,
        "after": after,
        "guard_before": guard_before,
    }


def audit(root, dataset_path):
    inputs = Inputs(root)
    analysis = inputs.load("analyze", "analysis.json")
    forecast = inputs.load("forecast", "forecasts.json")
    fitted = inputs.load("fit", "predictors.json")
    if (
        sha(dataset_path)
        != "14781a4cd46816767903e43a0a1cd74b5d7d80b096ad7c0eeb92b69fd29ebc93"
    ):
        raise ValueError("POLICY_AUDIT_DATASET_HASH")
    data = read(dataset_path)
    raw = [
        unit
        for source in ("test-a", "test-b")
        for unit in inputs.load("repair-" + source, "units.json")
    ]
    if [u["id"] for u in raw] != forecast["unit_ids"] or [u["id"] for u in raw] != [
        u["id"] for u in analysis["units"]
    ]:
        raise ValueError("POLICY_AUDIT_UNIT_ORDER")
    streams = [u["stream"] for u in raw]
    for source in ("test-a", "test-b"):
        if datetime.fromisoformat(
            inputs.executions["forecast"]["finished_at"]
        ) > datetime.fromisoformat(inputs.executions["repair-" + source]["started_at"]):
            raise ValueError("POLICY_AUDIT_FORECAST_AFTER_TEST_REPAIR_START")
    outcomes = [outcome(unit, data) for unit in raw]
    for observed, expected in zip(outcomes, analysis["units"], strict=True):
        close(observed["after"], expected["after"])
        if observed["actions"] != expected["actions"]:
            raise ValueError("POLICY_AUDIT_RAW_SCORING_DISAGREEMENT")
    labels = fitted["permutations"]["labels"]
    activity, inert = {}, []
    for draw in fitted["permutations"]["draws"]:
        indices = draw["indices"]
        if sorted(indices) != list(range(len(fitted["training_units"]))):
            raise ValueError("POLICY_AUDIT_INVALID_LABEL_PERMUTATION")
        if draw["scheme"] == "within_source":
            original_streams = [
                key.rsplit("/", 1)[0] for key in fitted["training_units"]
            ]
            if any(
                original_streams[i] != original_streams[j]
                for i, j in enumerate(indices)
            ):
                raise ValueError("POLICY_AUDIT_PERMUTATION_CROSSED_SOURCE")
        changes = {
            key: sum(values[i] != values[j] for i, j in enumerate(indices))
            for key, values in labels.items()
        }
        if changes != draw["changed_labels"]:
            raise ValueError("POLICY_AUDIT_INERT_LABEL_COUNT")
        activity.setdefault(draw["scheme"], {})[str(draw["seed"])] = changes
        inert.extend(
            {"scheme": draw["scheme"], "seed": draw["seed"], "target": key}
            for key, count in changes.items()
            if count == 0
        )
    utility_metrics, max_coefficient_error = {}, 0
    for name, model in fitted["models"].items():
        utility_metrics[name] = {}
        for budget in ("2", "4", "8", "16"):
            key = "utility/" + budget
            values = []
            for unit in raw:
                if "constants" in model:
                    value = model["constants"][key]
                else:
                    parameters = model["targets"][key]
                    vector = unit["features"][model["feature"]]
                    value = parameters["intercept"] + sum(
                        (x - center) / scale * weight
                        for x, center, scale, weight in zip(
                            vector,
                            parameters["center"],
                            parameters["scale"],
                            parameters["weights"],
                            strict=True,
                        )
                    )
                values.append(value)
            for value, saved in zip(
                values, forecast["models"][name]["utility"][budget], strict=True
            ):
                close(value, saved)
                max_coefficient_error = max(max_coefficient_error, abs(value - saved))
            errors = [
                value - unit["actions"][budget]["utility"]
                for value, unit in zip(values, outcomes, strict=True)
            ]
            utility_metrics[name][budget] = {
                "utility_mse": grouped([e * e for e in errors], streams),
                "utility_mae": grouped([abs(e) for e in errors], streams),
            }
            for metric, value in utility_metrics[name][budget].items():
                check_grouped(
                    value, analysis["metrics"][name]["by_budget"][budget][metric]
                )
    policies = {}
    for name, saved in analysis["policies"].items():
        choices = saved["choices"]
        if (
            name in forecast["models"]
            and choices != forecast["models"][name]["choices"]
        ):
            raise ValueError("POLICY_AUDIT_FORECAST_CHOICE_CHANGED")
        actions = [
            u["actions"][choice]
            if choice != "never"
            else {"recovered": False, "utility": 0}
            for u, choice in zip(outcomes, choices, strict=True)
        ]
        steps = [int(choice) if choice != "never" else 0 for choice in choices]
        recomputed = {
            "qualified_fraction": grouped(
                [float(a["recovered"]) for a in actions], streams
            ),
            "mean_updates": grouped(steps, streams),
            "utility": grouped([a["utility"] for a in actions], streams),
        }
        for key, value in recomputed.items():
            check_grouped(value, saved[key])
        policies[name] = recomputed | {
            "qualified": sum(a["recovered"] for a in actions),
            "total_updates": sum(steps),
        }
    changed = []
    for unit, choice in zip(
        outcomes, forecast["models"]["tuned"]["choices"], strict=True
    ):
        if choice == "2":
            continue
        alternative = (
            unit["actions"][choice]
            if choice != "never"
            else {"recovered": False, "utility": 0}
        )
        changed.append(
            {
                "id": unit["id"],
                "chosen": choice,
                "always2": unit["actions"]["2"],
                "tuned": alternative,
                "utility_delta": alternative["utility"]
                - unit["actions"]["2"]["utility"],
            }
        )
    flat_metrics = {
        name: {
            metric: mean(
                values[budget][metric]["equal_source_mean"] for budget in values
            )
            for metric in ("utility_mse", "utility_mae")
        }
        for name, values in utility_metrics.items()
    }
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "consumed_file_sha256": inputs.hashes | {str(dataset_path): sha(dataset_path)},
        "units": len(raw),
        "source_clusters": len(set(streams)),
        "policies": policies,
        "tuned_changed_choices": changed,
        "all_old_accuracy_and_gain_pass_at_two": all(
            u["actions"]["2"]["accuracy"] >= 0.8 and u["actions"]["2"]["gain"] >= 0.3
            for u in outcomes
        ),
        "always2_guard_failure_ids": [
            u["id"] for u in outcomes if not u["actions"]["2"]["recovered"]
        ],
        "utility_forecast_metrics": utility_metrics,
        "posthoc_equal_budget_equal_source_utility_errors": flat_metrics,
        "maximum_utility_coefficient_forecast_error": max_coefficient_error,
        "label_changes_per_draw_per_target": activity,
        "inert_targets": inert,
        "label_redraws": fitted["permutations"]["redraws"],
        "scope": "Independent native16 scoring, guard-aware policy arithmetic, utility forecasts from saved coefficients, and permutation activity. Parent runs full existing provenance/prefix/predictor-fit auditor separately.",
        "boundary": "Two source clusters; budgets and permutation draws are correlated diagnostics, not independent models. Equal-budget combined errors are post-hoc descriptive summaries. Always8 success does not prospectively select a new default; no frozen protocol changed.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    repo = directory.parents[2]
    result = audit(
        directory / "runs",
        repo / "experiments/recoverability/inputs/budget-prediction-cohort.json",
    )
    result["audit_script_sha256"] = sha(Path(__file__).resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "units": result["units"],
                "source_clusters": result["source_clusters"],
                "tuned": result["policies"]["tuned"],
                "old_accuracy_and_gain_pass_at_two": result[
                    "all_old_accuracy_and_gain_pass_at_two"
                ],
                "coefficient_forecast_error": result[
                    "maximum_utility_coefficient_forecast_error"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
