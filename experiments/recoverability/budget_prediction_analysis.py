import numpy as np

from minimum_budget import distribution_gate, minimum_outcome, permutation_diagnostics, permutation_plan
from prospective_analysis import FEATURES, ridge_apply, ridge_fit
from protocol import digest


def budget_rule(protocol):
    return {"budgets": protocol["budgets"], "train_streams": protocol["train_streams"], "gate": protocol["train_gate"]}


def prediction_gate(outcomes, protocol):
    result = distribution_gate(outcomes, budget_rule(protocol))
    classes = [str(budget) for budget in protocol["budgets"]] + ["never"]
    result["counts"] = {label: result["counts"].get(label, 0) for label in classes}
    result["counts_by_source"] = {
        source: {label: counts.get(label, 0) for label in classes}
        for source, counts in result["counts_by_source"].items()
    }
    result["unobserved_classes"] = [label for label, count in result["counts"].items() if not count]
    result.pop("test_authorized")
    result["test_readout_permitted"] = result["qualified"]
    result["decision"] = (
        "fit_and_freeze_predictors_then_TEST_readouts"
        if result["qualified"]
        else "stop_before_predictors_and_TEST; new_TRAIN_protocol_required"
    )
    return result


def fit_models(units, spec, protocol, plan):
    if any(unit["split"] != "train" for unit in units):
        raise ValueError("BUDGET_PREDICTION_FIT_TRAIN_ONLY")
    outcomes = [minimum_outcome(unit, spec, protocol["budgets"]) for unit in units]
    gate = prediction_gate(outcomes, protocol)
    if not gate["qualified"]:
        raise ValueError("BUDGET_PREDICTION_NO_TRAIN_LABEL_VARIATION")
    expected = permutation_plan(
        [unit["id"] for unit in units],
        [unit["stream"] for unit in units],
        protocol["label_permutations"]["seeds"],
        protocol["label_permutations"]["schemes"],
    )
    if expected != plan:
        raise ValueError("BUDGET_PREDICTION_PLANNED_LABEL_ORDER")
    classes = [str(value) for value in protocol["budgets"]] + ["never"]
    targets = {f"class/{label}": [float(unit["minimum_budget"] == label) for unit in outcomes] for label in classes}
    targets.update(
        {
            f"{label}/{budget}": [float(unit["actions"][str(budget)][label]) for unit in outcomes]
            for budget in protocol["budgets"]
            for label in ("recovered", "utility")
        }
    )
    models = {"constant_train": {"constants": {key: float(np.mean(value)) for key, value in targets.items()}}}

    def fitted(feature, indices):
        x = [unit["features"][feature] for unit in units]
        if not np.isfinite(x).all():
            raise ValueError("BUDGET_PREDICTION_NONFINITE_FIT_FEATURES")
        return {
            "feature": feature,
            "targets": {
                name: ridge_fit(x, np.asarray(y)[indices], protocol["ridge_alpha"]) for name, y in targets.items()
            },
        }

    for feature in FEATURES:
        models[feature] = fitted(feature, list(range(len(units))))
    for draw in plan["draws"]:
        name = f"permuted/{draw['scheme']}/{draw['seed']}"
        models[name] = fitted("tuned", draw["indices"])
    permutations = permutation_diagnostics(plan, outcomes, protocol["budgets"])
    permutations.pop("predictor_fits")
    permutations["labels"].update({key: value for key, value in targets.items() if key.startswith("class/")})
    for draw in permutations["draws"]:
        for target, values in targets.items():
            if target.startswith("class/"):
                changed = sum(value != values[draw["indices"][index]] for index, value in enumerate(values))
                draw["changed_labels"][target] = changed
                if not changed:
                    draw["inert_targets"].append(target)
    permutations["activity_by_scheme"] = {
        scheme: {
            target: {
                "active_draws": sum(
                    draw["changed_labels"][target] > 0 for draw in permutations["draws"] if draw["scheme"] == scheme
                ),
                "inert_draws": sum(
                    draw["changed_labels"][target] == 0 for draw in permutations["draws"] if draw["scheme"] == scheme
                ),
            }
            for target in permutations["labels"]
        }
        for scheme in protocol["label_permutations"]["schemes"]
    }
    return {
        "models": models,
        "classes": classes,
        "training_units": [unit["id"] for unit in units],
        "training_intents": [unit["intent"] for unit in units],
        "training_source_streams": sorted({unit["stream"] for unit in units}),
        "training_data_sha256": digest(units),
        "ridge_alpha": protocol["ridge_alpha"],
        "gate": gate,
        "permutations": permutations,
        "label_variation": {name: len(set(values)) > 1 for name, values in targets.items()},
        "training_unit_weight": "Equal per intent; all evaluation is also aggregated by independent source stream.",
    }


def forecasts(fitted, units):
    if (
        not units
        or any(unit["split"] != "test" for unit in units)
        or set(fitted["training_units"]).intersection(unit["id"] for unit in units)
        or set(fitted["training_intents"]).intersection(unit["intent"] for unit in units)
        or set(fitted["training_source_streams"]).intersection(unit["stream"] for unit in units)
    ):
        raise ValueError("BUDGET_PREDICTION_HELDOUT_IDENTITIES")
    result = {}
    for name, model in fitted["models"].items():
        if "constants" in model:
            values = {target: np.full(len(units), value) for target, value in model["constants"].items()}
        else:
            x = [unit["features"][model["feature"]] for unit in units]
            values = {target: ridge_apply(parameters, x) for target, parameters in model["targets"].items()}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("BUDGET_PREDICTION_NONFINITE_FORECAST")
        scores = np.stack([np.clip(values[f"class/{label}"], 0, 1) for label in fitted["classes"]], axis=1)
        if np.any(scores.sum(axis=1) <= 0):
            raise ValueError("BUDGET_PREDICTION_EMPTY_CLASS_SCORES")
        probabilities = scores / scores.sum(axis=1, keepdims=True)
        choices = [fitted["classes"][index] for index in probabilities.argmax(axis=1)]
        result[name] = {
            "class_probabilities": probabilities.tolist(),
            "choices": choices,
            "recovery": {
                key.removeprefix("recovered/"): np.clip(value, 0, 1).tolist()
                for key, value in values.items()
                if key.startswith("recovered/")
            },
            "utility": {
                key.removeprefix("utility/"): value.tolist()
                for key, value in values.items()
                if key.startswith("utility/")
            },
        }
    return {
        "unit_ids": [unit["id"] for unit in units],
        "source_streams": [unit["stream"] for unit in units],
        "classes": fitted["classes"],
        "models": result,
        "fitted_predictors_sha256": digest(fitted),
        "readouts_sha256": digest(
            [{key: unit[key] for key in ("id", "stream", "intent", "split", "features")} for unit in units]
        ),
    }


def grouped(values, streams):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(streams) or not len(values) or not np.isfinite(values).all():
        raise ValueError("BUDGET_PREDICTION_METRIC_SHAPE")
    by_source = {
        name: float(np.mean([value for value, stream in zip(values, streams, strict=True) if stream == name]))
        for name in sorted(set(streams))
    }
    return {
        "pooled": float(values.mean()),
        "by_source": by_source,
        "equal_source_mean": float(np.mean(list(by_source.values()))),
    }


def policy_metrics(choices, outcomes, streams, budgets):
    actions = [
        unit["actions"][choice] if choice != "never" else {"recovered": False, "utility": 0.0}
        for unit, choice in zip(outcomes, choices, strict=True)
    ]
    return {
        "choices": choices,
        "qualified_fraction": grouped([float(action["recovered"]) for action in actions], streams),
        "mean_updates": grouped([0 if value == "never" else int(value) for value in choices], streams),
        "utility": grouped([action["utility"] for action in actions], streams),
        "underbudget_failure_fraction": grouped(
            [
                float(
                    unit["minimum_budget"] != "never"
                    and not action["recovered"]
                    and (choice == "never" or int(choice) < int(unit["minimum_budget"]))
                )
                for unit, choice, action in zip(outcomes, choices, actions, strict=True)
            ],
            streams,
        ),
        "never_qualified_attempt_fraction": grouped(
            [
                float(unit["minimum_budget"] == "never" and choice != "never")
                for unit, choice in zip(outcomes, choices, strict=True)
            ],
            streams,
        ),
        "excess_steps_among_qualified": [
            int(choice) - int(unit["minimum_budget"])
            for unit, choice, action in zip(outcomes, choices, actions, strict=True)
            if action["recovered"]
        ],
        "reference_budget": max(budgets),
    }


def evaluate(fitted, predicted, units, spec, protocol):
    if forecasts(fitted, units) != predicted:
        raise ValueError("BUDGET_PREDICTION_SEALED_FORECAST_RECOMPUTATION")
    outcomes = [minimum_outcome(unit, spec, protocol["budgets"]) for unit in units]
    streams = predicted["source_streams"]
    labels = np.array([predicted["classes"].index(unit["minimum_budget"]) for unit in outcomes])
    one_hot = np.eye(len(predicted["classes"]))[labels]
    metrics, policies = {}, {}
    for name, values in predicted["models"].items():
        probabilities = np.asarray(values["class_probabilities"])
        metrics[name] = {
            "minimum_class_accuracy": grouped((probabilities.argmax(axis=1) == labels).astype(float), streams),
            "minimum_class_brier": grouped(((probabilities - one_hot) ** 2).sum(axis=1), streams),
            "by_budget": {},
        }
        for budget in protocol["budgets"]:
            key = str(budget)
            y = np.array([float(unit["actions"][key]["recovered"]) for unit in outcomes])
            utility = np.array([unit["actions"][key]["utility"] for unit in outcomes])
            error = np.asarray(values["utility"][key]) - utility
            metrics[name]["by_budget"][key] = {
                "recovery_brier": grouped((np.asarray(values["recovery"][key]) - y) ** 2, streams),
                "utility_mse": grouped(error**2, streams),
                "utility_mae": grouped(abs(error), streams),
                "train_binary_variation": fitted["label_variation"][f"recovered/{budget}"],
                "test_binary_variation": len(set(y)) > 1,
                "test_recovered": int(y.sum()),
                "test_total": len(y),
            }
        policies[name] = policy_metrics(values["choices"], outcomes, streams, protocol["budgets"])
    for budget in [*protocol["budgets"], "never"]:
        policies[f"always/{budget}"] = policy_metrics(
            [str(budget)] * len(units), outcomes, streams, protocol["budgets"]
        )
    reference_names = ("always/2", "always/16", "constant_train")
    for policy in policies.values():
        policy["paired_differences"] = {
            reference: {
                metric: {
                    "pooled": policy[metric]["pooled"] - policies[reference][metric]["pooled"],
                    "equal_source_mean": policy[metric]["equal_source_mean"]
                    - policies[reference][metric]["equal_source_mean"],
                    "by_source": {
                        name: value - policies[reference][metric]["by_source"][name]
                        for name, value in policy[metric]["by_source"].items()
                    },
                }
                for metric in ("qualified_fraction", "mean_updates", "utility")
            }
            for reference in reference_names
        }
        delta = policy["paired_differences"]["always/16"]
        policy["descriptive_savings_without_qualification_loss_on_each_source"] = all(
            policies["always/16"]["qualified_fraction"]["by_source"][name] > 0
            and delta["qualified_fraction"]["by_source"][name] >= 0
            and delta["mean_updates"]["by_source"][name] < 0
            for name in set(streams)
        )
    return {
        "policy_report": {
            name: {
                "pooled": {
                    metric: policy[metric]["pooled"] for metric in ("qualified_fraction", "mean_updates", "utility")
                },
                "equal_source_mean": {
                    metric: policy[metric]["equal_source_mean"]
                    for metric in ("qualified_fraction", "mean_updates", "utility")
                },
                "by_source": {
                    source: {
                        metric: policy[metric]["by_source"][source]
                        for metric in ("qualified_fraction", "mean_updates", "utility")
                    }
                    for source in sorted(set(streams))
                },
            }
            for name, policy in policies.items()
        },
        "train_class_counts": fitted["gate"]["counts"],
        "test_class_counts": {
            label: sum(unit["minimum_budget"] == label for unit in outcomes) for label in fitted["classes"]
        },
        "train_majority_rule": "constant_train chooses the largest TRAIN class frequency, with the same predeclared smaller-budget tie break.",
        "metrics": metrics,
        "policies": policies,
        "units": outcomes,
        "source_counts": {name: streams.count(name) for name in sorted(set(streams))},
        "independent_test_source_clusters": len(set(streams)),
        "inference": "Exploratory paired source-stream results. Intents, layers, prefixes and permutation draws are not independent trained models. No p-values or intent-level bootstrap claims. Normalized ridge scores are evaluated, not assumed calibrated. Budget savings, qualification and continuous utility are distinct outcomes.",
    }
