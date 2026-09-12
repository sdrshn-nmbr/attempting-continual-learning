from collections import Counter

import numpy as np

from protocol import digest

DEPLOYABLE = ("none", "replay_target", "replay_balanced")
TRAINABLE = DEPLOYABLE[1:]
FEATURES = ("confidence", "output", "frozen", "tuned")
METRICS = ("top_probability", "entropy", "top_margin", "target_logp", "target_margin")


def accuracy(records):
    if not records:
        raise ValueError("PROSPECTIVE_EMPTY_ACCURACY")
    return sum(row["output"]["prediction"] == row["target"] for row in records) / len(records)


def per_intent(records):
    return {
        intent: accuracy([row for row in records if row["intent"] == intent])
        for intent in sorted({row["intent"] for row in records})
    }


def learned(initial, scores, spec):
    if set(initial) != set(scores):
        raise ValueError("PROSPECTIVE_ACQUISITION_BASELINE_BINDING")
    return sorted(
        intent
        for intent, score in scores.items()
        if score >= spec["gates"]["acquired_min"]
        and score - initial[intent] >= spec["gates"]["acquisition_gain_min"] - 1e-12
    )


def eligible(initial, before, after, spec):
    return sorted(
        intent
        for intent in learned(initial, before, spec)
        if after[intent] <= spec["gates"]["after_max"]
        and before[intent] - after[intent] >= spec["gates"]["drop_min"] - 1e-12
    )


def feature_vector(before, after, layers):
    result = {key: [] for key in FEATURES}
    for rows in (before, after):
        base = [float(np.mean([row["output"][metric] for row in rows])) for metric in METRICS]
        result["confidence"].extend(base[:3])
        result["output"].extend(base)
        for lens in ("frozen", "tuned"):
            result[lens].extend(base)
            for layer in layers:
                result[lens].extend(
                    float(np.mean([row[lens][str(layer)][metric] for row in rows])) for metric in METRICS[-2:]
                )
    return result


def unit_outcomes(unit, spec):
    before = accuracy(unit["baseline"]["before"]["test"])
    after = accuracy(unit["baseline"]["after"]["test"])
    guard_after = accuracy(unit["baseline"]["after"]["guard"])
    actions = {}
    for name, action in unit["actions"].items():
        score, guard = accuracy(action["test"]), accuracy(action["guard"])
        gain, guard_loss = score - after, max(0.0, guard_after - guard)
        actions[name] = {
            "accuracy": score,
            "guard_accuracy": guard,
            "gain": gain,
            "utility": gain - guard_loss,
            "recovered": bool(
                score >= spec["recovered"]["accuracy_min"]
                and gain >= spec["recovered"]["gain_min"] - 1e-12
                and guard_loss <= spec["recovered"]["guard_drop_max"] + 1e-12
            ),
        }
    return {
        "id": unit["id"],
        "stream": unit["stream"],
        "intent": unit["intent"],
        "split": unit["split"],
        "before": before,
        "after": after,
        "actions": actions,
    }


def stream_summary(units, spec):
    outcomes = [unit_outcomes(unit, spec) for unit in units]
    actions = sorted(set.intersection(*(set(row["actions"]) for row in outcomes)))
    summary = {
        "stream": units[0]["stream"],
        "split": units[0]["split"],
        "intents": len(units),
        "before_accuracy": float(np.mean([row["before"] for row in outcomes])),
        "after_accuracy": float(np.mean([row["after"] for row in outcomes])),
        "actions": {
            action: {
                key: float(np.mean([row["actions"][action][key] for row in outcomes]))
                for key in ("accuracy", "guard_accuracy", "gain", "utility", "recovered")
            }
            for action in actions
        },
        "dependency_unit": "One independently trained source stream, not its intents/tokens/layers.",
    }
    if all("features" in unit for unit in units):
        summary["mean_readouts"] = {
            key: np.mean([unit["features"][key] for unit in units], axis=0).tolist() for key in FEATURES
        }
    return summary


def repair_feasibility(units, spec):
    if not units or any(unit["split"] != "train" for unit in units):
        raise ValueError("PROSPECTIVE_REPAIR_GATE_TRAIN_ONLY")
    outcomes = [unit_outcomes(unit, spec) for unit in units]
    rule = spec["repair_feasibility"]
    responders = [
        row["id"]
        for row in outcomes
        if max(row["actions"][action]["gain"] for action in TRAINABLE) >= rule["response_gain_min"] - 1e-12
    ]
    utilities = np.array([[row["actions"][action]["utility"] for action in TRAINABLE] for row in outcomes])
    paired_range = float(np.ptp(utilities[:, 0] - utilities[:, 1]))
    utility_range = float(np.ptp(utilities.max(axis=1)))
    signal = (
        utility_range >= rule["utility_range_min"] - 1e-12 or paired_range >= rule["paired_action_range_min"] - 1e-12
    )
    qualified = len(responders) >= rule["minimum_responsive_train_skills"] and signal
    return {
        "qualified": bool(qualified),
        "responsive_train_units": responders,
        "utility_range": utility_range,
        "paired_action_range": paired_range,
        "units": outcomes,
        "single_class_binary_actions": [
            action for action in TRAINABLE if len({row["actions"][action]["recovered"] for row in outcomes}) < 2
        ],
        "decision": "heldout_actions_allowed" if qualified else "stop_before_heldout_actions",
        "next_config_if_failed": "configs/prospective-repair64.json",
        "scope": "Signal qualification uses TRAIN intents only. Constant binary labels do not invalidate continuous utility.",
    }


def ridge_fit(x, y, alpha):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - center) / scale
    intercept = float(y.mean())
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return {"center": center.tolist(), "scale": scale.tolist(), "intercept": intercept, "weights": weights.tolist()}


def ridge_apply(model, x):
    z = (np.asarray(x, dtype=np.float64) - np.array(model["center"])) / np.array(model["scale"])
    return z @ np.array(model["weights"]) + model["intercept"]


def fit_predictors(units, spec):
    if any(unit["split"] != "train" for unit in units):
        raise ValueError("PROSPECTIVE_PREDICTOR_TRAIN_ONLY")
    outcomes = [unit_outcomes(unit, spec) for unit in units]
    models = {}
    permutation = np.random.default_rng(spec["data_seed"]).permutation(len(units)).tolist()
    for action in TRAINABLE:
        models[action] = {}
        for label in ("recovered", "utility"):
            y = np.array([row["actions"][action][label] for row in outcomes], dtype=np.float64)
            models[action][label] = {"prevalence": float(y.mean()), "estimable_binary": len(set(y)) > 1, "models": {}}
            for feature in (*FEATURES, "permuted"):
                key = "tuned" if feature == "permuted" else feature
                x = [unit["features"][key] for unit in units]
                target = y[permutation] if feature == "permuted" else y
                models[action][label]["models"][feature] = ridge_fit(x, target, spec["predictor"]["alpha"])
    return {
        "models": models,
        "train_units": [unit["id"] for unit in units],
        "permutation": permutation,
        "training_data_sha256": digest(units),
        "alpha": spec["predictor"]["alpha"],
    }


def evaluate_predictors(fitted, units, spec):
    if not units or any(unit["split"] != "test" for unit in units):
        raise ValueError("PROSPECTIVE_PREDICTOR_TEST_ONLY")
    if set(fitted["train_units"]).intersection(unit["id"] for unit in units):
        raise ValueError("PROSPECTIVE_PREDICTOR_ID_OVERLAP")
    rows = [unit_outcomes(unit, spec) for unit in units]
    prediction, utilities = {}, {key: {"none": np.zeros(len(rows))} for key in (*FEATURES, "permuted")}
    for action in TRAINABLE:
        y = np.array([row["actions"][action]["recovered"] for row in rows], dtype=float)
        trained = fitted["models"][action]
        probabilities = {"prevalence": np.full(len(rows), trained["recovered"]["prevalence"])}
        for feature in utilities:
            key = "tuned" if feature == "permuted" else feature
            x = [unit["features"][key] for unit in units]
            probabilities[feature] = np.clip(ridge_apply(trained["recovered"]["models"][feature], x), 0, 1)
            utilities[feature][action] = ridge_apply(trained["utility"]["models"][feature], x)
        errors = {key: (value - y) ** 2 for key, value in probabilities.items()}
        prediction[action] = {
            "labels": y.tolist(),
            "predictions": {key: value.tolist() for key, value in probabilities.items()},
            "brier": {key: float(value.mean()) for key, value in errors.items()},
            "brier_per_skill": {key: value.tolist() for key, value in errors.items()},
            "train_binary_variation": trained["recovered"]["estimable_binary"],
            "test_binary_variation": len(set(y)) > 1,
            "test_units": [row["id"] for row in rows],
        }

    def policy(choices):
        actual = [row["actions"][action]["utility"] for row, action in zip(rows, choices, strict=True)]
        oracle = [max(row["actions"][action]["utility"] for action in DEPLOYABLE) for row in rows]
        return {
            "choices": choices,
            "utility_per_skill": actual,
            "mean_utility": float(np.mean(actual)),
            "mean_oracle_regret": float(np.mean(np.array(oracle) - actual)),
            "by_stream": {
                stream: float(
                    np.mean([value for row, value in zip(rows, actual, strict=True) if row["stream"] == stream])
                )
                for stream in sorted({row["stream"] for row in rows})
            },
        }

    policies = {
        key: policy([max(DEPLOYABLE, key=lambda action, i=i: values[action][i]) for i in range(len(rows))])
        for key, values in utilities.items()
    }
    policies.update({f"always_{action}": policy([action] * len(rows)) for action in DEPLOYABLE})
    return {
        "status": "exploratory_heldout_intent_prediction",
        "units": rows,
        "recovery_prediction": prediction,
        "policies": policies,
        "counts": dict(Counter(row["stream"] for row in rows)),
        "run_level": {
            stream: {
                "intents": sum(row["stream"] == stream for row in rows),
                "mean_readouts": {
                    key: np.mean([unit["features"][key] for unit in units if unit["stream"] == stream], axis=0).tolist()
                    for key in FEATURES
                },
                "mean_gain": {
                    action: float(np.mean([row["actions"][action]["gain"] for row in rows if row["stream"] == stream]))
                    for action in TRAINABLE
                },
            }
            for stream in sorted({row["stream"] for row in rows})
        },
        "scope": "Internal scores are diagnostics only, never learner training targets. Two test runs support exploratory intent-level comparisons, not a model-level Jacobian or J-Access reproduction.",
    }
