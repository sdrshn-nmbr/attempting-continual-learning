import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

SOURCE = "99ac3030bb56b44579b07e0f672ce098b0831f9c871b350d087201ad546ba8d1"
PREFIX = "followthrough-20260912-recovery-predict-"
FEATURES = ("confidence", "output", "frozen", "tuned")
METRICS = ("top_probability", "entropy", "top_margin", "target_logp", "target_margin")
BUDGETS = (2, 4, 8, 16)
CLASSES = ("2", "4", "8", "16", "never")


def require(condition, detail):
    if not condition:
        raise ValueError("PREDICTION_INDEPENDENT_AUDIT: " + detail)


def doc(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(actual, expected, name="value", tolerance=1e-8):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{name}: dictionary required")
        for key, value in expected.items():
            require(key in actual, f"{name}/{key}: missing")
            close(actual[key], value, f"{name}/{key}", tolerance)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f"{name}: list length")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close(left, right, f"{name}/{index}", tolerance)
    elif isinstance(expected, (float, np.floating)):
        require(
            math.isfinite(actual)
            and math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance),
            f"{name}: {actual} != {expected}",
        )
    else:
        require(actual == expected, f"{name}: {actual} != {expected}")


def read_run(path, task, identity):
    execution_path = path / "execution.json"
    if not execution_path.exists():
        return {"state": "awaiting_collection", "path": path}
    execution = doc(execution_path)
    require(
        execution["task_id"] == task["id"] and execution["source_sha256"] == SOURCE,
        "execution source/task",
    )
    require(
        execution["task"]["config"] == task["config"],
        "execution config differs from staged dispatch",
    )
    result = {"path": path, "execution": execution, "state": execution["status"]}
    if execution["status"] != "completed":
        if (path / "study/failure.json").exists():
            result["failure"] = doc(path / "study/failure.json")
        return result
    require(
        execution["exit_code"] == 0 and execution.get("finished_at"),
        "completed execution must have successful exit and finish time",
    )
    receipt_path, seal_path = path / "study/receipt.json", path / "study/seal.json"
    if not receipt_path.exists() or not seal_path.exists():
        return result | {"state": "collection_incomplete"}
    receipt, seal = doc(receipt_path), doc(seal_path)
    for value in (receipt, seal):
        require(value["sha256"] == digest(value["payload"]), "receipt/seal digest")
    body, sealed = receipt["payload"], seal["payload"]
    missing = [name for name in body["files"] if not (path / "study" / name).is_file()]
    if missing:
        return result | {"state": "collection_incomplete", "missing": missing}
    actual = {
        str(item.relative_to(path / "study"))
        for item in (path / "study").rglob("*")
        if item.is_file() and item.name != "receipt.json"
    }
    require(actual == set(body["files"]), "receipt file set")
    for name, expected in body["files"].items():
        require(
            file_hash(path / "study" / name) == expected,
            f"artifact hash {task['id']}/{name}",
        )
    require(
        sealed["prediction_identity"] == identity
        and sealed["dispatch_manifest"] == task["config"],
        "sealed prediction identity/config",
    )
    require(body["pipeline_stage"] == task["config"]["stage"], "receipt pipeline stage")
    require(
        doc(execution_path) == execution and doc(receipt_path) == receipt,
        "source changed during snapshot",
    )
    return result | {
        "state": "verified",
        "receipt": body,
        "seal": sealed,
        "receipt_sha256": file_hash(receipt_path),
        "execution_sha256": file_hash(execution_path),
    }


def time_order(before, after, label):
    end = datetime.fromisoformat(before["finished_at"])
    start = datetime.fromisoformat(after["started_at"])
    require(
        end.tzinfo is not None and start.tzinfo is not None,
        label + ": timezone required",
    )
    require(
        end <= start, label + ": dependent run started before prerequisite completed"
    )
    return {
        "before_finished_at": before["finished_at"],
        "after_started_at": after["started_at"],
        "separation_seconds": (start - end).total_seconds(),
    }


def binding(row):
    return row["text_sha256"], row["intent"], row["target"]


def code_metrics(panel, target, codes):
    logits = panel["code_logits"]
    require(
        len(codes) == len(set(codes)) == len(logits) == 16
        and all(math.isfinite(value) for value in logits),
        "finite native 16-code logits",
    )
    predicted = codes[max(range(16), key=logits.__getitem__)]
    require(
        panel["prediction"] == predicted, "saved prediction is not native code argmax"
    )
    maximum = max(logits)
    logsum = math.log(sum(math.exp(value - maximum) for value in logits))
    logp = [value - maximum - logsum for value in logits]
    probability = [math.exp(value) for value in logp]
    ordered = sorted(probability, reverse=True)
    target_index = codes.index(target)
    values = {
        "top_probability": ordered[0],
        "entropy": -sum(p * lp for p, lp in zip(probability, logp, strict=True)),
        "top_margin": ordered[0] - ordered[1],
        "target_logp": logp[target_index],
        "target_margin": logits[target_index]
        - max(value for index, value in enumerate(logits) if index != target_index),
    }
    close(panel, values, "native features", tolerance=1e-5)
    return predicted, values


def score(records, codes, expected=None, counter=None):
    require(bool(records), "no empty accuracy denominators")
    if expected is not None:
        require(
            Counter(map(binding, records)) == Counter(map(binding, expected)),
            "observation row/code binding",
        )
    correct, total = Counter(), Counter()
    for row in records:
        prediction, _ = code_metrics(row["output"], row["target"], codes)
        correct[row["intent"]] += prediction == row["target"]
        total[row["intent"]] += 1
    if counter is not None:
        counter["native_records_checked"] += len(records)
    return {
        "accuracy": sum(correct.values()) / sum(total.values()),
        "by_intent": {name: correct[name] / count for name, count in total.items()},
    }


def rows_for(stream, kind, role):
    return [row for intent in stream[kind] for row in stream["units"][intent][role]]


def learning(run, stream, spec, codes, counter):
    observations = doc(run["path"] / "study/observations.json")
    phases = {}
    for phase, panels in observations.items():
        phases[phase] = {
            kind: score(
                panels[kind + "_gate"], codes, rows_for(stream, kind, "gate"), counter
            )["by_intent"]
            for kind in ("old", "new")
        }
    rule = spec["gates"]
    acquired = sorted(
        intent
        for intent in stream["old"]
        if phases["acquired"]["old"][intent] >= rule["acquired_min"]
        and phases["acquired"]["old"][intent] - phases["initial"]["old"][intent]
        >= rule["acquisition_gain_min"] - 1e-12
    )
    eligible, new = [], []
    acquisition_passed = len(acquired) >= rule["minimum_acquired_per_stream"]
    if "forgotten" in phases:
        require(acquisition_passed, "forgetting ran after failed acquisition")
        eligible = sorted(
            intent
            for intent in acquired
            if phases["forgotten"]["old"][intent] <= rule["after_max"]
            and phases["acquired"]["old"][intent] - phases["forgotten"]["old"][intent]
            >= rule["drop_min"] - 1e-12
        )
        new = sorted(
            intent
            for intent in stream["new"]
            if phases["forgotten"]["new"][intent] >= rule["acquired_min"]
            and phases["forgotten"]["new"][intent] - phases["acquired"]["new"][intent]
            >= rule["acquisition_gain_min"] - 1e-12
        )
    else:
        require(
            not acquisition_passed,
            "qualified acquisition did not produce forgetting observations",
        )
    qualified = (
        acquisition_passed
        and len(eligible) >= rule["minimum_forgotten_per_stream"]
        and len(new) >= rule["minimum_new_acquired_per_stream"]
    )
    receipt = run["receipt"]
    close(
        receipt,
        {
            "acquired": acquired,
            "eligible": eligible,
            "new_acquired": new,
            "qualified": qualified,
            "repair_updates": 0,
            "test_rows_evaluated": 0,
            "lens_updates": 0,
        },
    )
    acquired_updates = len(doc(run["path"] / "study/acquisition-updates.json"))
    forgotten_updates = (
        len(doc(run["path"] / "study/forgetting-updates.json"))
        if acquisition_passed
        else 0
    )
    require(
        acquired_updates == run["seal"]["config"]["acquisition_updates"]
        and forgotten_updates
        == (run["seal"]["config"]["forgetting_updates"] if acquisition_passed else 0),
        "predeclared learning budgets",
    )
    close(
        receipt,
        {
            "acquisition_updates": acquired_updates,
            "forgetting_updates": forgotten_updates,
            "optimizer_steps": {
                "acquired": [acquired_updates],
                "final": [acquired_updates + forgotten_updates],
            },
        },
    )
    return {
        "qualified": qualified,
        "acquired": acquired,
        "new_acquired": new,
        "eligible": eligible,
        "phases": phases,
        "stream": stream["id"],
        "split": stream["split"],
    }


def lens(run, data, spec):
    training = doc(run["path"] / "study/lens-training.json")
    late = doc(run["path"] / "study/lens-late-check.json")
    require(
        training["model_parameters_unchanged"] and training["reload_exact"],
        "lens changed source model or failed reload",
    )
    require(
        training["lens_sha256"]
        == file_hash(run["path"] / "study/tuned-lens.safetensors"),
        "lens file hash",
    )
    for part in ("fit", "check"):
        expected = {row["text_sha256"]: row for row in data["lens_" + part]}
        positions = training[part]["positions"]
        require(
            set(expected) == {row["text_sha256"] for row in positions},
            "lens background identities",
        )
        require(
            all(row["source_split"] == "train" for row in expected.values()),
            "lens used non-TRAIN background",
        )
        require(
            all(
                0 <= row["position"] < len(expected[row["text_sha256"]]["input_ids"])
                for row in positions
            ),
            "lens positions outside prompt",
        )
    require(len(training["history"]) == spec["lens"]["updates"], "lens update budget")
    require(
        all(
            len(step["positions"]) == spec["lens"]["token_batch_size"]
            and all(
                0 <= index < training["fit"]["tokens"] for index in step["positions"]
            )
            for step in training["history"]
        ),
        "lens optimization used non-fit activations",
    )
    close(
        late["metadata"],
        {key: training["check"][key] for key in ("positions", "prompts", "tokens")},
        "late lens validation rows",
    )
    improvements = {}
    for phase, values in (
        ("acquired", training["final_heldout_kl"]),
        ("forgotten", late["heldout_kl"]),
    ):
        require(set(values) == {str(layer) for layer in spec["layers"]}, "lens layers")
        require(
            all(
                math.isfinite(value)
                for row in values.values()
                for value in row.values()
            ),
            "nonfinite lens KL",
        )
        improvements[phase] = 1 - sum(row["tuned"] for row in values.values()) / max(
            sum(row["frozen"] for row in values.values()), 1e-12
        )
    qualified = all(
        value >= spec["lens"]["min_relative_kl_improvement"]
        for value in improvements.values()
    )
    require(qualified == run["receipt"]["qualified"], "lens qualification result")
    require(run["receipt"]["repair_updates"] == 0, "readout stage performed repair")
    if qualified:
        close(
            run["receipt"],
            {phase + "_kl_improvement": value for phase, value in improvements.items()},
        )
    else:
        require(
            not (run["path"] / "study/readouts.json").exists(),
            "failed lens emitted qualified readouts",
        )
    return {
        "qualified": qualified,
        "relative_kl_improvement": improvements,
        "updates": len(training["history"]),
    }


def probe_features(unit, stream, spec, codes, counter):
    result = {feature: [] for feature in FEATURES}
    for phase in ("before", "after"):
        records = unit["probes"][phase]
        score(records, codes, stream["units"][unit["intent"]]["probe"], counter)
        base = [
            mean(
                code_metrics(row["output"], row["target"], codes)[1][metric]
                for row in records
            )
            for metric in METRICS
        ]
        result["confidence"].extend(base[:3])
        result["output"].extend(base)
        for kind in ("frozen", "tuned"):
            result[kind].extend(base)
            for layer in spec["layers"]:
                values = [
                    code_metrics(row[kind][str(layer)], row["target"], codes)[1]
                    for row in records
                ]
                result[kind].extend(
                    mean(row[metric] for row in values) for metric in METRICS[-2:]
                )
    close(unit["features"], result, "probe feature vector", tolerance=1e-5)
    return result


def outcomes(unit, codes, rule, stream=None, counter=None):
    expected_old = stream["units"][unit["intent"]]["test"] if stream else None
    expected_guard = rows_for(stream, "new", "test") if stream else None
    before = score(unit["baseline"]["before"]["test"], codes, expected_old, counter)[
        "accuracy"
    ]
    after = score(unit["baseline"]["after"]["test"], codes, expected_old, counter)[
        "accuracy"
    ]
    guard_before = score(
        unit["baseline"]["after"]["guard"], codes, expected_guard, counter
    )["accuracy"]
    require(
        set(unit["actions"]) == {str(budget) for budget in BUDGETS},
        "complete 2/4/8/16 action grid required",
    )
    actions = {}
    for budget in BUDGETS:
        panel = unit["actions"][str(budget)]
        accuracy = score(panel["test"], codes, expected_old, counter)["accuracy"]
        guard = score(panel["guard"], codes, expected_guard, counter)["accuracy"]
        gain, guard_drop = accuracy - after, max(0.0, guard_before - guard)
        actions[str(budget)] = {
            "accuracy": accuracy,
            "guard_accuracy": guard,
            "gain": gain,
            "guard_drop": guard_drop,
            "utility": gain - guard_drop,
            "recovered": accuracy >= rule["accuracy_min"]
            and gain >= rule["gain_min"] - 1e-12
            and guard_drop <= rule["guard_drop_max"] + 1e-12,
        }
    passed = [budget for budget in BUDGETS if actions[str(budget)]["recovered"]]
    return {
        "id": unit["id"],
        "stream": unit["stream"],
        "split": unit["split"],
        "intent": unit["intent"],
        "before": before,
        "after": after,
        "guard_after": guard_before,
        "actions": actions,
        "minimum_budget": str(min(passed)) if passed else "never",
        "qualified_budgets": passed,
        "later_qualification_loss": bool(passed)
        and any(
            not actions[str(budget)]["recovered"]
            for budget in BUDGETS
            if budget > min(passed)
        ),
    }


def action_prefixes(run, unit, stream, learning_run):
    root = run["path"] / "study/actions" / unit["intent"]
    expected_rows = stream["units"][unit["intent"]]["repair"] + rows_for(
        stream, "new", "repair"
    )
    target_count = len(stream["units"][unit["intent"]]["repair"])
    full = doc(root / "16-updates.json")["updates"]
    require(len(full) == 16, "full repair budget")
    for budget in BUDGETS:
        action = unit["actions"][str(budget)]
        history = doc(root / f"{budget}-updates.json")
        require(
            history["rows"] == expected_rows and history["updates"] == full[:budget],
            "actual loss/gradient/data update prefix",
        )
        batches = [step["rows"] for step in history["updates"]]
        require(
            [step["step"] for step in history["updates"]] == list(range(1, budget + 1)),
            "repair step counts",
        )
        require(
            all(
                len(batch) == 8
                and sum(index < target_count for index in batch) == 4
                and all(0 <= index < len(expected_rows) for index in batch)
                for batch in batches
            ),
            "balanced actual old/new exposures",
        )
        for stop in (value for value in BUDGETS if value <= budget):
            require(
                file_hash(
                    root / str(budget) / f"prefix-{stop}/adapter_model.safetensors"
                )
                == file_hash(root / "16" / f"prefix-{stop}/adapter_model.safetensors"),
                "actual saved-weight prefix",
            )
        require(
            action["adapter_sha256"]
            == file_hash(
                root / str(budget) / f"prefix-{budget}/adapter_model.safetensors"
            ),
            "final adapter identity",
        )
        require(
            action["start_adapter_sha256"]
            == file_hash(
                learning_run["path"] / "study/forgotten/adapter_model.safetensors"
            ),
            "same forgotten start checkpoint",
        )
        require(
            action["trace_sha256"] == digest(history["updates"])
            and action["schedule_sha256"]
            == digest({"rows": expected_rows, "batches": batches}),
            "repair trace/schedule digest",
        )
        close(
            action,
            {
                "updates": budget,
                "old_exposures": 4 * budget,
                "new_exposures": 4 * budget,
                "fresh_optimizer": True,
                "reload_exact": True,
            },
        )
        require(
            action == doc(root / f"{budget}-outcome.json"),
            "action publication differs from unit",
        )


def class_gate(units, protocol):
    require(
        bool(units) and all(unit["split"] == "train" for unit in units),
        "TRAIN class gate scope",
    )
    counts = {
        label: sum(unit["minimum_budget"] == label for unit in units)
        for label in CLASSES
    }
    sources = sorted({unit["stream"] for unit in units})
    by_source = {
        source: {
            label: sum(
                unit["stream"] == source and unit["minimum_budget"] == label
                for unit in units
            )
            for label in CLASSES
        }
        for source in sources
    }
    supported = [
        label
        for label in CLASSES
        if counts[label] >= protocol["train_gate"]["minimum_skills_per_class"]
    ]
    qualified = (
        len(supported) >= protocol["train_gate"]["minimum_budget_classes"]
        and len(units) >= protocol["train_gate"]["minimum_train_skills"]
        and sources == protocol["train_streams"]
    )
    return {
        "qualified": qualified,
        "counts": counts,
        "counts_by_source": by_source,
        "supported_classes": supported,
        "unobserved_classes": [label for label in CLASSES if not counts[label]],
        "train_skills": len(units),
        "source_clusters": len(sources),
        "test_readout_permitted": qualified,
    }


def check_permutations(plan, units, protocol, reported=None):
    require(
        plan["unit_ids"] == [unit["id"] for unit in units]
        and plan["source_streams"] == [unit["stream"] for unit in units],
        "permutation source order",
    )
    require(plan["redraws"] == 0, "permutation redraw")
    declared = [
        (scheme, seed)
        for scheme in protocol["label_permutations"]["schemes"]
        for seed in protocol["label_permutations"]["seeds"]
    ]
    require(
        [(draw["scheme"], draw["seed"]) for draw in plan["draws"]] == declared,
        "predeclared permutations",
    )
    labels = {"minimum_budget": [unit["minimum_budget"] for unit in units]}
    labels.update(
        {
            f"{budget}/{target}": [
                unit["actions"][str(budget)][target] for unit in units
            ]
            for budget in BUDGETS
            for target in ("recovered", "utility")
        }
    )
    labels.update(
        {
            "class/" + label: [float(unit["minimum_budget"] == label) for unit in units]
            for label in CLASSES
        }
    )
    results = []
    for draw in plan["draws"]:
        indices = list(range(len(units)))
        groups = (
            [indices]
            if draw["scheme"] == "pooled_train"
            else [
                [index for index, unit in enumerate(units) if unit["stream"] == source]
                for source in sorted({unit["stream"] for unit in units})
            ]
        )
        rng = np.random.default_rng(draw["seed"])
        for group in groups:
            for index, replacement in zip(
                group, rng.permutation(group).tolist(), strict=True
            ):
                indices[index] = replacement
        require(
            indices == draw["indices"], "permutation indices were changed or redrawn"
        )
        require(
            draw["fixed_indices"]
            == [
                index
                for index, replacement in enumerate(indices)
                if index == replacement
            ],
            "fixed permutation indices",
        )
        changed = {
            name: sum(
                value != values[indices[index]] for index, value in enumerate(values)
            )
            for name, values in labels.items()
        }
        results.append(
            draw
            | {
                "changed_labels": changed,
                "inert_targets": [name for name, count in changed.items() if not count],
            }
        )
    activity = {
        scheme: {
            name: {
                "active_draws": sum(
                    draw["changed_labels"][name] > 0
                    for draw in results
                    if draw["scheme"] == scheme
                ),
                "inert_draws": sum(
                    draw["changed_labels"][name] == 0
                    for draw in results
                    if draw["scheme"] == scheme
                ),
            }
            for name in labels
        }
        for scheme in protocol["label_permutations"]["schemes"]
    }
    if reported is not None:
        require(reported["redraws"] == 0, "reported permutation redraw")
        close(reported["labels"], labels, "permutation labels")
        close(reported["activity_by_scheme"], activity, "permutation activity")
        require(len(reported["draws"]) == len(results), "permutation diagnostic count")
        for actual, expected in zip(reported["draws"], results, strict=True):
            close(
                actual,
                {
                    key: value
                    for key, value in expected.items()
                    if key != "inert_targets"
                },
                "permutation diagnostic",
            )
            require(
                set(actual["inert_targets"]) == set(expected["inert_targets"]),
                "inert scalar labels",
            )
    return {"draws": results, "activity_by_scheme": activity, "redraws": 0}


def check_predictors(fitted, units, observed, plan, protocol):
    require(
        fitted["training_units"] == [unit["id"] for unit in units]
        and fitted["training_intents"] == [unit["intent"] for unit in units],
        "fitted TRAIN unit/intent identities",
    )
    require(
        fitted["training_source_streams"] == protocol["train_streams"]
        and fitted["training_data_sha256"] == digest(units),
        "fitted TRAIN data/source binding",
    )
    gate = class_gate(observed, protocol)
    require(gate["qualified"], "predictors fit without informative TRAIN classes")
    close(fitted["gate"], gate, "fitted TRAIN gate")
    targets = {
        "class/" + label: [float(unit["minimum_budget"] == label) for unit in observed]
        for label in CLASSES
    }
    targets.update(
        {
            f"{target}/{budget}": [
                float(unit["actions"][str(budget)][target]) for unit in observed
            ]
            for budget in BUDGETS
            for target in ("recovered", "utility")
        }
    )
    required = {
        "constant_train",
        *FEATURES,
        *(f"permuted/{draw['scheme']}/{draw['seed']}" for draw in plan["draws"]),
    }
    require(
        set(fitted["models"]) == required and fitted["classes"] == list(CLASSES),
        "all predictors/fixed classes must remain",
    )
    close(
        fitted["models"]["constant_train"]["constants"],
        {name: mean(values) for name, values in targets.items()},
        "TRAIN constants",
    )
    close(
        fitted["label_variation"],
        {name: len(set(values)) > 1 for name, values in targets.items()},
        "scalar TRAIN variation",
    )
    nulls = {
        f"permuted/{draw['scheme']}/{draw['seed']}": draw["indices"]
        for draw in plan["draws"]
    }
    residual_max = 0.0
    for name, model in fitted["models"].items():
        if name == "constant_train":
            continue
        require(
            model["feature"] == ("tuned" if name in nulls else name),
            "feature source of predictor",
        )
        x = np.asarray(
            [unit["features"][model["feature"]] for unit in units], dtype=float
        )
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-12] = 1
        z = (x - center) / scale
        require(set(model["targets"]) == set(targets), "ridge target set")
        for target, saved in model["targets"].items():
            y = np.asarray(targets[target], dtype=float)[
                nulls.get(name, list(range(len(units))))
            ]
            close(
                saved,
                {
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                    "intercept": float(y.mean()),
                },
                "TRAIN-only ridge scaler/intercept",
            )
            weight = np.asarray(saved["weights"])
            residual = (
                z.T @ (z @ weight - (y - y.mean())) + protocol["ridge_alpha"] * weight
            )
            residual_max = max(residual_max, float(abs(residual).max()))
            require(
                np.allclose(residual, 0, atol=1e-7, rtol=0),
                "saved ridge does not solve TRAIN-label objective",
            )
    return {
        "models": len(required),
        "scalar_ridge_solutions_checked": 13 * (len(required) - 1),
        "max_ridge_stationarity_error": residual_max,
        "permutations": check_permutations(
            plan, observed, protocol, fitted["permutations"]
        ),
    }


def predict(fitted, units):
    require(
        bool(units) and all(unit["split"] == "test" for unit in units),
        "forecast heldout-only scope",
    )
    for field, saved in (
        ("id", "training_units"),
        ("intent", "training_intents"),
        ("stream", "training_source_streams"),
    ):
        require(
            not set(fitted[saved]).intersection(unit[field] for unit in units),
            "forecast overlaps TRAIN " + field,
        )
    results = {}
    for name, model in fitted["models"].items():
        if name == "constant_train":
            values = {
                target: np.full(len(units), value)
                for target, value in model["constants"].items()
            }
        else:
            x = np.asarray([unit["features"][model["feature"]] for unit in units])
            values = {
                target: (x - np.asarray(saved["center"]))
                / np.asarray(saved["scale"])
                @ np.asarray(saved["weights"])
                + saved["intercept"]
                for target, saved in model["targets"].items()
            }
        require(
            all(np.isfinite(value).all() for value in values.values()),
            "finite forecasts",
        )
        raw = np.array([np.clip(values["class/" + label], 0, 1) for label in CLASSES]).T
        require(np.all(raw.sum(axis=1) > 0), "nonempty class score")
        probabilities = raw / raw.sum(axis=1)[:, None]
        results[name] = {
            "class_probabilities": probabilities.tolist(),
            "choices": [CLASSES[index] for index in probabilities.argmax(axis=1)],
            "recovery": {
                str(budget): np.clip(values[f"recovered/{budget}"], 0, 1).tolist()
                for budget in BUDGETS
            },
            "utility": {
                str(budget): values[f"utility/{budget}"].tolist() for budget in BUDGETS
            },
        }
    return {
        "unit_ids": [unit["id"] for unit in units],
        "source_streams": [unit["stream"] for unit in units],
        "classes": list(CLASSES),
        "models": results,
        "fitted_predictors_sha256": digest(fitted),
        "readouts_sha256": digest(
            [
                {
                    key: unit[key]
                    for key in ("id", "stream", "intent", "split", "features")
                }
                for unit in units
            ]
        ),
    }


def aggregate(values, streams):
    require(
        len(values) == len(streams) and bool(streams), "nonempty metric denominator"
    )
    by_source = {
        source: mean(
            float(value)
            for value, group in zip(values, streams, strict=True)
            if source == group
        )
        for source in sorted(set(streams))
    }
    return {
        "pooled": mean(map(float, values)),
        "by_source": by_source,
        "equal_source_mean": mean(by_source.values()),
    }


def policy(choices, observed):
    streams = [unit["stream"] for unit in observed]
    actions = [
        unit["actions"][choice]
        if choice != "never"
        else {"recovered": False, "utility": 0.0}
        for unit, choice in zip(observed, choices, strict=True)
    ]
    return {
        "choices": choices,
        "qualified_fraction": aggregate(
            [action["recovered"] for action in actions], streams
        ),
        "mean_updates": aggregate(
            [int(choice) if choice != "never" else 0 for choice in choices], streams
        ),
        "utility": aggregate([action["utility"] for action in actions], streams),
        "qualified_by_intent": {
            unit["id"]: action["recovered"]
            for unit, action in zip(observed, actions, strict=True)
        },
    }


def evaluate(predicted, observed, fitted):
    streams = [unit["stream"] for unit in observed]
    require(
        [unit["id"] for unit in observed] == predicted["unit_ids"],
        "outcome/forecast order",
    )
    actual_class = [CLASSES.index(unit["minimum_budget"]) for unit in observed]
    metrics, policies = {}, {}
    for name, prediction in predicted["models"].items():
        probabilities = np.asarray(prediction["class_probabilities"])
        metrics[name] = {
            "minimum_class_accuracy": aggregate(
                probabilities.argmax(axis=1) == actual_class, streams
            ),
            "minimum_class_brier": aggregate(
                ((probabilities - np.eye(5)[actual_class]) ** 2).sum(axis=1), streams
            ),
            "by_budget": {},
        }
        for budget in BUDGETS:
            key = str(budget)
            recovered = [unit["actions"][key]["recovered"] for unit in observed]
            error = np.asarray(prediction["utility"][key]) - [
                unit["actions"][key]["utility"] for unit in observed
            ]
            metrics[name]["by_budget"][key] = {
                "recovery_brier": aggregate(
                    (np.asarray(prediction["recovery"][key]) - recovered) ** 2, streams
                ),
                "utility_mse": aggregate(error**2, streams),
                "utility_mae": aggregate(abs(error), streams),
                "train_binary_variation": fitted["label_variation"][
                    f"recovered/{budget}"
                ],
                "test_binary_variation": len(set(recovered)) > 1,
                "test_recovered": sum(recovered),
                "test_total": len(recovered),
            }
        policies[name] = policy(prediction["choices"], observed)
    for choice in CLASSES:
        policies["always/" + choice] = policy([choice] * len(observed), observed)
    for value in policies.values():
        comparisons = {}
        for baseline in ("always/2", "always/16", "constant_train"):
            reference = policies[baseline]
            loss = [
                reference["qualified_by_intent"][unit["id"]]
                and not value["qualified_by_intent"][unit["id"]]
                for unit in observed
            ]
            gain = [
                value["qualified_by_intent"][unit["id"]]
                and not reference["qualified_by_intent"][unit["id"]]
                for unit in observed
            ]
            comparisons[baseline] = {
                "lost_qualification_fraction": aggregate(loss, streams),
                "newly_qualified_fraction": aggregate(gain, streams),
                "lost_intents": [
                    unit["id"]
                    for unit, lost in zip(observed, loss, strict=True)
                    if lost
                ],
                "newly_qualified_intents": [
                    unit["id"]
                    for unit, gained in zip(observed, gain, strict=True)
                    if gained
                ],
                "paired_differences": {
                    metric: aggregate(
                        [
                            value[metric]["by_source"][stream]
                            - reference[metric]["by_source"][stream]
                            for stream in streams
                        ],
                        streams,
                    )
                    for metric in ("qualified_fraction", "mean_updates", "utility")
                },
            }
        value["comparisons"] = comparisons
        value["descriptive_savings_without_qualification_loss_on_each_source"] = all(
            policies["always/16"]["qualified_fraction"]["by_source"][stream] > 0
            and value["qualified_fraction"]["by_source"][stream]
            >= policies["always/16"]["qualified_fraction"]["by_source"][stream]
            and value["mean_updates"]["by_source"][stream] < 16
            for stream in set(streams)
        )
        value["savings_without_any_paired_intent_loss"] = (
            value["descriptive_savings_without_qualification_loss_on_each_source"]
            and not value["comparisons"]["always/16"]["lost_intents"]
        )
    return {
        "metrics": metrics,
        "policies": policies,
        "source_counts": dict(Counter(streams)),
        "independent_test_source_clusters": len(set(streams)),
        "train_class_counts": fitted["gate"]["counts"],
        "test_class_counts": {
            label: sum(unit["minimum_budget"] == label for unit in observed)
            for label in CLASSES
        },
    }


def audit(runs_root, source_root, dispatch_path):
    interface = doc(source_root / "budget-prediction-interface.json")
    require(
        file_hash(source_root / "budget-prediction-interface.json")
        == (source_root / "budget-prediction-interface.sha256").read_text().strip(),
        "interface seal",
    )
    for name, expected in interface["required_file_sha256"].items():
        require(
            file_hash(source_root / name) == expected, "frozen source closure: " + name
        )
    dispatch = doc(dispatch_path)
    require(
        dispatch["source_sha256"] == SOURCE and len(dispatch["tasks"]) == 19,
        "staged source/task graph",
    )
    tasks = {task["id"]: task for task in dispatch["tasks"]}
    require(set(tasks) == set(interface["dispatch"]), "staged task identities")
    for name, task in tasks.items():
        require(
            task["config"] == doc(source_root / interface["dispatch"][name]["config"]),
            "staged manifest differs from sealed source",
        )
        require(
            task["depends_on"] == interface["dispatch"][name]["depends_on"],
            "staged dependencies differ from sealed graph",
        )
    protocol = doc(source_root / "budget-prediction-protocol.json")
    spec = doc(source_root / "budget-prediction-learning.json")
    data = doc(source_root / "inputs/budget-prediction-cohort.json")
    excluded = set(data["source"]["excluded_prior_intents"])
    cohorts = [
        set(stream["old"] + stream["new"]) for stream in data["streams"].values()
    ]
    require(
        len(excluded) == 80
        and len(set.union(*cohorts)) == sum(map(len, cohorts)) == 64
        and not set.union(*cohorts).intersection(excluded),
        "four fresh disjoint cohorts exclude all 80 prior intents",
    )
    runs = {
        name: read_run(
            runs_root / name, task, interface["source_boundary"]["prediction_identity"]
        )
        for name, task in tasks.items()
    }
    for run in runs.values():
        if run["state"] == "verified":
            require(
                run["seal"]["prediction_implementation"]
                == interface["source_boundary"]["relevant_implementation_sha256"],
                "measured implementation differs from frozen closure",
            )
            require(
                all(
                    run["seal"]["implementation"][name] == expected
                    for name, expected in interface["source_boundary"][
                        "relevant_implementation_sha256"
                    ].items()
                ),
                "measured Python source hashes",
            )
            require(
                run["seal"]["prediction_protocol"] == protocol
                and run["seal"]["protocol"] == spec
                and run["seal"]["dataset_sha256"] == digest(data),
                "measured protocol and fresh dataset identity",
            )
    report = {
        "source_sha256": SOURCE,
        "frozen_runtime_files_verified": len(interface["required_file_sha256"]),
        "staged_tasks": len(tasks),
        "fresh_intents": 64,
        "source_cluster_scope": "Two fresh TRAIN and two fresh TEST trained source streams. Exploratory inference across two TEST clusters; intents, layers and permutation draws are not independent models.",
        "stages": {},
        "ordering": {},
        "native_records_checked": 0,
        "learning": {},
        "readout": {},
        "repairs": {},
        "pending_scientific_checks": [],
        "scientific_conclusion": None,
    }
    for name, run in runs.items():
        report["stages"][name] = {
            key: run[key]
            for key in (
                "state",
                "receipt_sha256",
                "execution_sha256",
                "missing",
                "failure",
            )
            if key in run
        }
        if run["state"] == "verified":
            report["stages"][name].update(
                qualified=run["receipt"]["qualified"],
                scientific_status=run["receipt"]["status"],
            )
        if "execution" not in run:
            continue
        for dependency in tasks[name]["depends_on"]:
            before = runs.get(dependency)
            external = runs_root / dependency / "execution.json"
            execution = (
                before.get("execution")
                if before
                else doc(external)
                if external.exists()
                else None
            )
            if execution and execution["status"] == "completed":
                report["ordering"][dependency + " -> " + name] = time_order(
                    execution, run["execution"], dependency + " -> " + name
                )
            else:
                report["pending_scientific_checks"].append(
                    "dependency completion: " + dependency + " -> " + name
                )
    ready = {
        name.removeprefix(PREFIX): run
        for name, run in runs.items()
        if run["state"] == "verified"
    }
    collected_count = len(ready)
    while True:
        waiting = [
            suffix
            for suffix in ready
            if any(
                dependency in tasks and dependency.removeprefix(PREFIX) not in ready
                for dependency in tasks[PREFIX + suffix]["depends_on"]
            )
        ]
        if not waiting:
            break
        for suffix in waiting:
            del ready[suffix]
            report["pending_scientific_checks"].append(
                "awaiting complete upstream collections: " + PREFIX + suffix
            )

    def get(suffix):
        return ready.get(suffix)

    def run_for(stream, stage):
        return get(stage + "-" + stream.removeprefix("fresh-"))

    authorization = get("authorize")
    if authorization:
        gate_path = (
            runs_root / protocol["minimum_budget_gate"]["run_id"] / "study/receipt.json"
        )
        if gate_path.exists():
            require(
                authorization["receipt"]["minimum_gate_receipt_sha256"]
                == file_hash(gate_path),
                "official minimum gate receipt binding",
            )
            require(
                authorization["receipt"]["qualified"]
                == doc(gate_path)["payload"]["qualified"],
                "authorization scientific gate",
            )
    for suffix, run in ready.items():
        if suffix != "authorize":
            require(
                authorization is not None,
                "completed downstream missing collected authorization",
            )
            require(
                run["receipt"]["authorization_sha256"]
                == authorization["receipt_sha256"]
                and authorization["receipt"]["qualified"],
                "downstream work after failed/mismatched authorization",
            )
    for stream, definition in data["streams"].items():
        run = run_for(stream, "learn")
        if run:
            report["learning"][stream] = learning(
                run, definition, spec, data["codes"], report
            )
    cohort = get("cohort")
    if cohort and len(report["learning"]) == 4:
        qualified = all(
            value["qualified"] for value in report["learning"].values()
        ) and all(
            sum(
                len(value["eligible"])
                for value in report["learning"].values()
                if value["split"] == split
            )
            >= spec["gates"]["minimum_forgotten"][split]
            for split in ("train", "test")
        )
        require(cohort["receipt"]["qualified"] == qualified, "global cohort gate")
        for name, value in report["learning"].items():
            close(
                cohort["receipt"]["streams"][name],
                {
                    "passed": value["qualified"],
                    "eligible": value["eligible"],
                    "split": value["split"],
                    "receipt_sha256": run_for(name, "learn")["receipt_sha256"],
                },
                "cohort source",
            )
        report["cohort_qualified"] = qualified
    readouts, repaired, measured_outcomes = {}, {}, {}
    for stream, definition in data["streams"].items():
        run = run_for(stream, "readout")
        if not run:
            continue
        require(
            report.get("cohort_qualified"), "readout without valid collected cohort"
        )
        require(
            run["receipt"]["cohort_sha256"] == cohort["receipt_sha256"],
            "readout cohort digest",
        )
        report["readout"][stream] = lens(run, data, spec)
        if not run["receipt"]["qualified"]:
            continue
        units = doc(run["path"] / "study/readouts.json")
        require(
            [unit["intent"] for unit in units]
            == report["learning"][stream]["eligible"],
            "readout eligible intent set",
        )
        require(
            all(
                unit["stream"] == stream
                and unit["split"] == definition["split"]
                and unit["id"] == stream + "/" + unit["intent"]
                for unit in units
            ),
            "readout source/intent identity",
        )
        for unit in units:
            require(not unit["actions"], "repair outcomes leaked into frozen readout")
            probe_features(unit, definition, spec, data["codes"], report)
            for phase in ("before", "after"):
                score(
                    unit["baseline"][phase]["test"],
                    data["codes"],
                    definition["units"][unit["intent"]]["test"],
                    report,
                )
                score(
                    unit["baseline"][phase]["guard"],
                    data["codes"],
                    rows_for(definition, "new", "test"),
                    report,
                )
        require(
            run["receipt"]["readouts_sha256"]
            == file_hash(run["path"] / "study/readouts.json"),
            "readout file digest",
        )
        readouts[stream] = units
    readout_gates = {}
    for split in ("train", "test"):
        gate = get("readout-gate-" + split)
        if gate and all(
            name in report["readout"] for name in protocol[split + "_streams"]
        ):
            passed = all(
                report["readout"][name]["qualified"]
                for name in protocol[split + "_streams"]
            )
            require(
                gate["receipt"]["qualified"] == passed,
                "readout global gate cannot vacuously pass",
            )
            require(
                gate["receipt"]["unit_ids"]
                == [
                    unit["id"]
                    for name in protocol[split + "_streams"]
                    for unit in readouts.get(name, [])
                ],
                "readout gate eligible units",
            )
            readout_gates[split] = passed
    for stream, definition in data["streams"].items():
        run = run_for(stream, "repair")
        if not run:
            continue
        split = definition["split"]
        require(
            readout_gates.get(split) and stream in readouts,
            "repair without qualified collected readouts",
        )
        require(
            run["receipt"]["readout_gate_sha256"]
            == get("readout-gate-" + split)["receipt_sha256"],
            "repair readout gate digest",
        )
        if split == "test":
            require(
                get("forecast") is not None
                and get("fit") is not None
                and get("fit")["receipt"]["qualified"],
                "TEST repairs without forecasts or TRAIN qualification",
            )
            require(
                run["receipt"]["forecasts_sha256"] == get("forecast")["receipt_sha256"],
                "TEST repair forecast digest",
            )
            time_order(
                get("forecast")["execution"],
                run["execution"],
                "FORECAST BEFORE TEST REPAIR",
            )
        units = doc(run["path"] / "study/units.json")
        require(
            doc(run["path"] / "study/features-frozen.json") == readouts[stream],
            "pre-repair frozen features differ from readouts",
        )
        require(
            [{**unit, "actions": {}} for unit in units] == readouts[stream],
            "post-repair feature or cohort mutation",
        )
        require(
            run["receipt"]["repair_updates"] == 30 * len(units),
            "complete matched repair budget",
        )
        measured_outcomes[stream] = []
        published = doc(run["path"] / "study/outcomes.json")
        require(len(published) == len(units), "outcome publication count")
        for unit, saved in zip(units, published, strict=True):
            action_prefixes(run, unit, definition, run_for(stream, "learn"))
            result = outcomes(
                unit, data["codes"], protocol["recovered"], definition, report
            )
            close(result, saved, "published guard-aware outcome")
            measured_outcomes[stream].append(result)
        repaired[stream] = units
        run_level = {
            "before_accuracy": mean(
                unit["before"] for unit in measured_outcomes[stream]
            ),
            "after_accuracy": mean(unit["after"] for unit in measured_outcomes[stream]),
            "actions": {
                str(budget): {
                    metric: mean(
                        float(unit["actions"][str(budget)][metric])
                        for unit in measured_outcomes[stream]
                    )
                    for metric in (
                        "accuracy",
                        "guard_accuracy",
                        "gain",
                        "utility",
                        "recovered",
                    )
                }
                for budget in BUDGETS
            },
            "mean_readouts": {
                name: np.mean(
                    [unit["features"][name] for unit in units], axis=0
                ).tolist()
                for name in FEATURES
            },
        }
        close(
            run["receipt"]["run_level"],
            run_level,
            "aggregated source readouts/recovery",
        )
        report["repairs"][stream] = {
            "run_level": run_level,
            "units": len(units),
            "updates": 30 * len(units),
            "minimum_class_counts": dict(
                Counter(unit["minimum_budget"] for unit in measured_outcomes[stream])
            ),
            "nonmonotonic_intents": [
                unit["id"]
                for unit in measured_outcomes[stream]
                if unit["later_qualification_loss"]
            ],
        }
    fitted = None
    if get("fit") and all(name in repaired for name in protocol["train_streams"]):
        train_units = [
            unit for name in protocol["train_streams"] for unit in repaired[name]
        ]
        train_outcomes = [
            unit
            for name in protocol["train_streams"]
            for unit in measured_outcomes[name]
        ]
        gate = class_gate(train_outcomes, protocol)
        close(get("fit")["receipt"], gate, "fresh TRAIN minimum-class gate")
        report["fresh_train_gate"] = gate
        if not gate["qualified"]:
            require(
                get("fit")["receipt"]["predictor_fits"] == 0
                and not (get("fit")["path"] / "study/predictors.json").exists(),
                "failed TRAIN gate still fit predictors",
            )
            require(
                not any(
                    run_for(name, stage)
                    for name in protocol["test_streams"]
                    for stage in ("readout", "repair")
                ),
                "TEST work after failed TRAIN class gate",
            )
        else:
            fitted = doc(get("fit")["path"] / "study/predictors.json")
            require(
                get("fit")["receipt"]["predictors_sha256"]
                == file_hash(get("fit")["path"] / "study/predictors.json"),
                "predictor file digest",
            )
            plan = doc(cohort["path"] / "study/permutation-plan.json")
            report["predictors"] = check_predictors(
                fitted, train_units, train_outcomes, plan, protocol
            )
            require(
                get("fit")["receipt"]["predictor_fits"]
                == report["predictors"]["scalar_ridge_solutions_checked"],
                "predictor fit count",
            )
    predictions = None
    if (
        get("forecast")
        and fitted
        and all(name in readouts for name in protocol["test_streams"])
    ):
        test_units = [
            unit for name in protocol["test_streams"] for unit in readouts[name]
        ]
        predictions = predict(fitted, test_units)
        close(
            doc(get("forecast")["path"] / "study/forecasts.json"),
            predictions,
            "saved forecasts from fixed TRAIN coefficients",
        )
        require(
            get("forecast")["receipt"]["forecasts_file_sha256"]
            == file_hash(get("forecast")["path"] / "study/forecasts.json"),
            "forecast file digest",
        )
        require(
            get("forecast")["receipt"]["fitted_sha256"] == get("fit")["receipt_sha256"],
            "forecast fitted-source digest",
        )
        report["forecasts"] = {
            "units": len(test_units),
            "models": len(predictions["models"]),
            "recomputed_without_test_outcomes": True,
        }
    if predictions and all(
        name in measured_outcomes for name in protocol["test_streams"]
    ):
        test_outcomes = [
            unit
            for name in protocol["test_streams"]
            for unit in measured_outcomes[name]
        ]
        result = evaluate(predictions, test_outcomes, fitted)
        report["independent_analysis"] = result
        if get("analyze"):
            published = doc(get("analyze")["path"] / "study/analysis.json")
            close(
                published,
                {
                    key: result[key]
                    for key in (
                        "metrics",
                        "source_counts",
                        "independent_test_source_clusters",
                        "train_class_counts",
                        "test_class_counts",
                    )
                },
                "analysis metrics",
            )
            for name, policy_result in result["policies"].items():
                close(
                    published["policies"][name],
                    {
                        key: policy_result[key]
                        for key in (
                            "choices",
                            "qualified_fraction",
                            "mean_updates",
                            "utility",
                            "descriptive_savings_without_qualification_loss_on_each_source",
                        )
                    },
                    "published policy",
                )
                close(
                    published["policies"][name]["paired_differences"],
                    {
                        baseline: value["paired_differences"]
                        for baseline, value in policy_result["comparisons"].items()
                    },
                    "published paired policy differences",
                )
                for source, joint in published["policy_report"][name][
                    "by_source"
                ].items():
                    close(
                        joint,
                        {
                            metric: policy_result[metric]["by_source"][source]
                            for metric in (
                                "qualified_fraction",
                                "mean_updates",
                                "utility",
                            )
                        },
                        "joint source policy report",
                    )
                for partition in ("pooled", "equal_source_mean"):
                    close(
                        published["policy_report"][name][partition],
                        {
                            metric: policy_result[metric][partition]
                            for metric in (
                                "qualified_fraction",
                                "mean_updates",
                                "utility",
                            )
                        },
                        "joint aggregate policy report",
                    )
            report["published_analysis_verified"] = True
        report["scientific_conclusion"] = {
            "policy_and_forecast_error_are_distinct": True,
            "main_policy_baselines": ["always/2", "always/16", "constant_train"],
            "inspect_joint_qualification_updates_and_utility": True,
            "paired_loss_boundary": "Equal source qualification rates can conceal lost and newly qualified intents; those pairs are separately reported.",
            "history_boundary": "No matched small-budget retain-only reference in this study. Response speed does not by itself establish stored-binding reuse.",
            "exploratory_test_source_clusters": result[
                "independent_test_source_clusters"
            ],
        }
    report["completed_verified_stages"] = collected_count
    report["stages_with_complete_upstream_collections"] = len(ready)
    report["scientific_blocks"] = [
        {"task_id": PREFIX + suffix, "status": run["receipt"]["status"]}
        for suffix, run in ready.items()
        if not run["receipt"]["qualified"]
    ]
    report["status"] = (
        "complete_verified"
        if len(ready) == 19 and report.get("published_analysis_verified")
        else "prerequisite_failure_verified"
        if report["scientific_blocks"]
        else "partial_no_final_scientific_conclusion"
        if report["scientific_conclusion"] is None
        else "outcomes_verified_analysis_pending"
    )
    report["missing_stages"] = [
        name for name, run in runs.items() if run["state"] != "verified"
    ]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path, default=Path(__file__).parent / "runs"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).parents[3] / "experiments/recoverability",
    )
    parser.add_argument(
        "--dispatch",
        type=Path,
        default=Path(__file__).parent / "recovery-prediction-dispatch.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "immutable audit output already exists")
    result = audit(args.runs_root, args.source_root, args.dispatch)
    result.update(
        audited_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=file_hash(Path(__file__)),
        auditor_imports_experiment_modules=False,
        writes_to_experiment_source=False,
        input_runs_root=str(args.runs_root),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = (
        args.output.parent
        / "recovery-prediction-audit.sources"
        / (result["auditor_sha256"] + ".py")
    )
    snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():
        with snapshot.open("xb") as handle:
            handle.write(Path(__file__).read_bytes())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "completed_verified_stages": result["completed_verified_stages"],
                "native_records_checked": result["native_records_checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
