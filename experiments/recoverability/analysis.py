from collections import Counter

import numpy as np

from protocol import ACTIONS, DEPLOYABLE

OUTPUT_FEATURES = ("top_probability", "entropy", "top_margin", "target_logp", "target_code_margin")


def features(probes):
    confidence, output, lens = [], [], []
    for phase in ("before", "after"):
        rows = probes[phase]
        confidence.extend(float(np.mean([r["output"][key] for r in rows])) for key in OUTPUT_FEATURES[:3])
        output.extend(float(np.mean([r["output"][key] for r in rows])) for key in OUTPUT_FEATURES)
        for key in ("target_logp", "target_code_margin"):
            layer_values = np.array([[v[key] for v in r["lens"].values()] for r in rows])
            lens.extend([float(layer_values.mean()), float(layer_values.max(axis=1).mean())])
    return {"confidence": confidence, "output": output, "lens": output + lens}


def accuracy(records, rows, key="prediction"):
    if not records or len(records) != len(rows):
        raise ValueError("RECOVERY_EVALUATION_CARDINALITY")
    if any(record["text_sha256"] != row["text_sha256"] for record, row in zip(records, rows, strict=True)):
        raise ValueError("RECOVERY_EVALUATION_ROW_BINDING")
    return sum(record["output"][key] == row["target"] for record, row in zip(records, rows, strict=True)) / len(rows)


def forgotten(before, after, spec):
    gate = spec["forgotten"]
    return before >= gate["before_min"] and after <= gate["after_max"] and before - after >= gate["drop_min"]


def recovered(before, after, repaired, guard_before, guard_after, spec):
    gate = spec["recovered"]
    return bool(
        forgotten(before, after, spec)
        and repaired >= gate["accuracy_min"]
        and repaired - after >= gate["gain_min"]
        and guard_before - guard_after <= gate["guard_drop_max"] + 1e-12
    )


def ridge_predict(train_x, train_y, test_x, alpha):
    train_x, test_x, train_y = (np.asarray(x, dtype=np.float64) for x in (train_x, test_x, train_y))
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-12] = 1
    train_z, test_z = (train_x - center) / scale, (test_x - center) / scale
    intercept = train_y.mean(axis=0)
    weights = np.linalg.solve(train_z.T @ train_z + alpha * np.eye(train_z.shape[1]), train_z.T @ (train_y - intercept))
    return test_z @ weights + intercept


def labels(data, observations, outcomes, spec):
    result = []
    for unit in data["units"]:
        raw = observations["units"][unit["id"]]
        before, after = (accuracy(raw[phase]["label"], unit["label"]) for phase in ("before", "after"))
        guard_rows = data["guards"][unit["run"]]
        guard_late = accuracy(observations["guards"][unit["run"]]["after"], guard_rows)
        row = {
            "id": unit["id"],
            "skill": unit["skill"],
            "run": unit["run"],
            "split": unit["split"],
            "before": before,
            "after": after,
            "forgotten": forgotten(before, after, spec),
            "actions": {},
        }
        for action in ACTIONS:
            observed = outcomes[unit["id"]][action]
            score, guard = accuracy(observed["label"], unit["label"]), accuracy(observed["guard"], guard_rows)
            row["actions"][action] = {
                "accuracy": score,
                "guard_accuracy": guard,
                "code_accuracy": accuracy(observed["label"], unit["label"], "code_prediction"),
                "recovered": recovered(before, after, score, guard_late, guard, spec),
                "utility": score - after - max(0, guard_late - guard),
            }
        result.append(row)
    return result


def analyze(data, observations, outcomes, feature_table, spec):
    measured = labels(data, observations, outcomes, spec)
    train = [row for row in measured if row["split"] == "train" and row["forgotten"]]
    test = [row for row in measured if row["split"] == "test" and row["forgotten"]]
    eligibility = "estimable"
    if min(len(train), len(test)) < spec["minimum_forgotten_per_split"]:
        eligibility = "insufficient_eligible_skills"
    test_clusters = len({r["run"] for r in test})
    result = {
        "status": "descriptive_pilot",
        "feasibility": "insufficient_independent_runs",
        "predictor_feasibility": eligibility,
        "single_class_actions": [],
        "claim": "Pilot estimates only. No general prediction or causal claim about internal readouts.",
        "units": measured,
        "counts": {
            "all_skills": len(measured),
            "forgotten_train": len(train),
            "forgotten_test": len(test),
            "independent_test_clusters": test_clusters,
            "raw_skills_by_split": dict(Counter(row["split"] for row in measured)),
            "raw_clusters_by_split": {
                split: len({row["run"] for row in measured if row["split"] == split}) for split in ("train", "test")
            },
            "source_clusters": dict(Counter(row["run"] for row in measured)),
        },
    }
    result["control_observations"] = {
        split: {
            action: float(np.mean([row["actions"][action]["accuracy"] for row in measured if row["split"] == split]))
            for action in ("none", "restore", "sham")
        }
        for split in ("train", "test")
    }
    if not train or not test:
        result["predictor_estimate"] = None
        return result
    action_results, utilities = {}, {key: {} for key in ("confidence", "output", "lens", "permuted")}
    alpha = spec["predictor"]["alpha"]
    for action in ("replay_short", "replay_long"):
        train_y = np.array([row["actions"][action]["recovered"] for row in train], dtype=float)
        test_y = np.array([row["actions"][action]["recovered"] for row in test], dtype=float)
        train_u = np.array([row["actions"][action]["utility"] for row in train])
        if len(set(train_y)) < 2 or len(set(test_y)) < 2:
            result["single_class_actions"].append(action)
        rng = np.random.default_rng(spec["runtime"]["seed"])
        permutation = rng.permutation(len(train))
        predictions = {"prevalence": np.full(len(test), train_y.mean())}
        for key in utilities:
            feature_key = "lens" if key == "permuted" else key
            train_x = [feature_table[row["id"]][feature_key] for row in train]
            test_x = [feature_table[row["id"]][feature_key] for row in test]
            target, utility = (train_y[permutation], train_u[permutation]) if key == "permuted" else (train_y, train_u)
            predictions[key] = np.clip(ridge_predict(train_x, target, test_x, alpha), 0, 1)
            utilities[key][action] = ridge_predict(train_x, utility, test_x, alpha)
        errors = {key: (value - test_y) ** 2 for key, value in predictions.items()}
        action_results[action] = {
            "brier": {key: float(value.mean()) for key, value in errors.items()},
            "output_minus_lens_per_skill": (errors["output"] - errors["lens"]).tolist(),
            "predictions": {key: value.tolist() for key, value in predictions.items()},
            "test_units": [row["id"] for row in test],
            "labels": test_y.tolist(),
        }
    policies = {}
    for key, values in utilities.items():
        values["none"] = np.zeros(len(test))
        choices = [max(DEPLOYABLE, key=lambda action, index=i: values[action][index]) for i in range(len(test))]
        policies[key] = policy_metrics(test, choices)
    for action in DEPLOYABLE:
        policies[f"always_{action}"] = policy_metrics(test, [action] * len(test))
    result.update(
        {
            "recovery_prediction": action_results,
            "policies": policies,
            "controls": {
                action: sum(row["actions"][action]["recovered"] for row in test) / len(test)
                for action in ("restore", "sham")
            },
        }
    )
    result["lens_minus_output_policy_utility"] = policies["lens"]["mean_utility"] - policies["output"]["mean_utility"]
    return result


def policy_metrics(rows, choices):
    utility = [row["actions"][action]["utility"] for row, action in zip(rows, choices, strict=True)]
    oracle = [max(row["actions"][action]["utility"] for action in DEPLOYABLE) for row in rows]
    return {
        "choices": choices,
        "mean_utility": float(np.mean(utility)),
        "mean_oracle_regret": float(np.mean(np.array(oracle) - np.array(utility))),
        "utility_per_skill": utility,
    }
