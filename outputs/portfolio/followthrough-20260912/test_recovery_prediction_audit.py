import json
import math
from pathlib import Path

import numpy as np
import pytest

import recovery_prediction_audit as audit

CODES = list(range(32, 48))
RULE = {"accuracy_min": 0.8, "gain_min": 0.3, "guard_drop_max": 0.1}


def records(intent, code, correct, total=20):
    result = []
    denominator = math.exp(4) + 15
    top, rest = math.exp(4) / denominator, 1 / denominator
    for index in range(total):
        prediction = code if index < correct else (code + 1) % 16
        logits = [2.0 if label == prediction else -2.0 for label in range(16)]
        result.append(
            {
                "intent": intent,
                "target": CODES[code],
                "text_sha256": f"{intent}/{index}",
                "output": {
                    "code_logits": logits,
                    "prediction": CODES[prediction],
                    "top_probability": top,
                    "entropy": -top * math.log(top) - 15 * rest * math.log(rest),
                    "top_margin": top - rest,
                    "target_logp": math.log(top if code == prediction else rest),
                    "target_margin": 4.0 if code == prediction else -4.0,
                },
            }
        )
    return result


def units(split, labels):
    result = []
    for index, label in enumerate(labels):
        stream = f"fresh-{split}-{'a' if index % 2 == 0 else 'b'}"
        intent = f"{split}-{index}"
        result.append(
            {
                "id": f"{stream}/{intent}",
                "stream": stream,
                "intent": intent,
                "split": split,
                "baseline": {
                    "before": {
                        "test": records(intent, 0, 20),
                        "guard": records("guard", 1, 0),
                    },
                    "after": {
                        "test": records(intent, 0, 0),
                        "guard": records("guard", 1, 20),
                    },
                },
                "features": {
                    name: [-1.0 if label == "2" else 1.0] for name in audit.FEATURES
                },
                "actions": {
                    str(budget): {
                        "test": records(
                            intent,
                            0,
                            20 if label != "never" and budget >= int(label) else 10,
                        ),
                        "guard": records("guard", 1, 20),
                    }
                    for budget in audit.BUDGETS
                },
            }
        )
    return result


@pytest.fixture
def protocol():
    return {
        "train_streams": ["fresh-train-a", "fresh-train-b"],
        "ridge_alpha": 10.0,
        "train_gate": {
            "minimum_skills_per_class": 2,
            "minimum_budget_classes": 2,
            "minimum_train_skills": 12,
        },
        "label_permutations": {
            "schemes": ["within_source", "pooled_train"],
            "seeds": list(range(2026091301, 2026091321)),
        },
    }


def plan_for(train, protocol):
    draws = []
    for scheme in protocol["label_permutations"]["schemes"]:
        for seed in protocol["label_permutations"]["seeds"]:
            indices = list(range(len(train)))
            rng = np.random.default_rng(seed)
            groups = (
                [indices.copy()]
                if scheme == "pooled_train"
                else [
                    [
                        index
                        for index, unit in enumerate(train)
                        if unit["stream"] == source
                    ]
                    for source in protocol["train_streams"]
                ]
            )
            for group in groups:
                indices_for_group = rng.permutation(group)
                for destination, source in zip(group, indices_for_group, strict=True):
                    indices[destination] = int(source)
            draws.append(
                {
                    "scheme": scheme,
                    "seed": seed,
                    "indices": indices,
                    "fixed_indices": [
                        index for index, source in enumerate(indices) if source == index
                    ],
                }
            )
    return {
        "unit_ids": [unit["id"] for unit in train],
        "source_streams": [unit["stream"] for unit in train],
        "draws": draws,
        "redraws": 0,
    }


@pytest.fixture
def fitted_case(protocol):
    train = units("train", ["2"] * 6 + ["16"] * 6)
    observed = [audit.outcomes(unit, CODES, RULE) for unit in train]
    plan = plan_for(train, protocol)
    targets = {
        "class/" + label: [float(unit["minimum_budget"] == label) for unit in observed]
        for label in audit.CLASSES
    }
    targets.update(
        {
            f"{target}/{budget}": [
                float(unit["actions"][str(budget)][target]) for unit in observed
            ]
            for budget in audit.BUDGETS
            for target in ("recovered", "utility")
        }
    )
    models = {
        "constant_train": {
            "constants": {key: float(np.mean(value)) for key, value in targets.items()}
        }
    }
    for name, feature, indices in [
        *((name, name, list(range(12))) for name in audit.FEATURES),
        *(
            (f"permuted/{draw['scheme']}/{draw['seed']}", "tuned", draw["indices"])
            for draw in plan["draws"]
        ),
    ]:
        x = np.asarray([unit["features"][feature] for unit in train])
        center, scale = x.mean(axis=0), x.std(axis=0)
        z = (x - center) / scale
        design = np.vstack((z, math.sqrt(protocol["ridge_alpha"]) * np.eye(z.shape[1])))
        saved = {}
        for target, values in targets.items():
            y = np.asarray(values)[indices]
            weights = np.linalg.lstsq(
                design, np.concatenate((y - y.mean(), np.zeros(z.shape[1]))), rcond=None
            )[0]
            saved[target] = {
                "center": center.tolist(),
                "scale": scale.tolist(),
                "intercept": float(y.mean()),
                "weights": weights.tolist(),
            }
        models[name] = {"feature": feature, "targets": saved}
    diagnostics = audit.check_permutations(plan, observed, protocol)
    diagnostics["labels"] = {
        "minimum_budget": [unit["minimum_budget"] for unit in observed]
    }
    diagnostics["labels"].update(
        {
            f"{budget}/{target}": [
                unit["actions"][str(budget)][target] for unit in observed
            ]
            for budget in audit.BUDGETS
            for target in ("recovered", "utility")
        }
    )
    diagnostics["labels"].update(
        {key: values for key, values in targets.items() if key.startswith("class/")}
    )
    fitted = {
        "classes": list(audit.CLASSES),
        "training_units": [unit["id"] for unit in train],
        "training_intents": [unit["intent"] for unit in train],
        "training_source_streams": protocol["train_streams"],
        "training_data_sha256": audit.digest(train),
        "models": models,
        "gate": audit.class_gate(observed, protocol),
        "label_variation": {
            key: len(set(values)) > 1 for key, values in targets.items()
        },
        "permutations": diagnostics,
    }
    return fitted, train, observed, plan


def test_native_contract_and_raw_feature_recalculation():
    source = records("old", 0, 16)
    assert audit.score(source, CODES)["accuracy"] == 0.8
    source[0]["output"]["prediction"] = CODES[-1]
    with pytest.raises(ValueError, match="argmax"):
        audit.score(source, CODES)


def test_guard_failure_and_nonmonotonic_recovery_are_preserved():
    unit = units("test", ["2"])[0]
    unit["actions"]["2"]["guard"] = records("guard", 1, 17)
    unit["actions"]["8"]["guard"] = records("guard", 1, 16)
    observed = audit.outcomes(unit, CODES, RULE)
    assert observed["minimum_budget"] == "4"
    assert observed["qualified_budgets"] == [4, 16]
    assert observed["later_qualification_loss"]
    assert observed["actions"]["2"]["accuracy"] == 1.0
    assert observed["actions"]["2"]["guard_drop"] == pytest.approx(0.15)
    assert observed["actions"]["2"]["utility"] == pytest.approx(0.85)
    assert not observed["actions"]["2"]["recovered"]


def test_gain_gate_and_boundary_thresholds():
    unit = units("test", ["2"])[0]
    unit["baseline"]["after"]["test"] = records(unit["intent"], 0, 10)
    for action in unit["actions"].values():
        action["test"] = records(unit["intent"], 0, 16)
        action["guard"] = records("guard", 1, 18)
    assert audit.outcomes(unit, CODES, RULE)["minimum_budget"] == "2"
    unit["baseline"]["after"]["test"] = records(unit["intent"], 0, 11)
    assert audit.outcomes(unit, CODES, RULE)["minimum_budget"] == "never"


def test_forecast_must_finish_before_repair_process_start():
    before = {"finished_at": "2026-09-12T06:45:41+00:00"}
    after = {"started_at": "2026-09-12T06:45:42+00:00"}
    assert audit.time_order(before, after, "forecast/repair")["separation_seconds"] == 1
    after["started_at"] = "2026-09-12T06:45:40+00:00"
    with pytest.raises(ValueError, match="started before"):
        audit.time_order(before, after, "forecast/repair")


def test_constant_minimum_classes_are_not_an_informative_task(protocol):
    observed = [
        audit.outcomes(unit, CODES, RULE) for unit in units("train", ["2"] * 12)
    ]
    result = audit.class_gate(observed, protocol)
    assert not result["qualified"] and not result["test_readout_permitted"]
    assert result["counts"] == {"2": 12, "4": 0, "8": 0, "16": 0, "never": 0}


def test_inert_binary_and_one_hot_nulls_are_counted_without_redrawing(
    fitted_case, protocol
):
    fitted, train, observed, plan = fitted_case
    result = audit.check_predictors(fitted, train, observed, plan, protocol)
    assert result["models"] == 45 and result["scalar_ridge_solutions_checked"] == 572
    assert result["max_ridge_stationarity_error"] < 1e-7
    for scheme, values in result["permutations"]["activity_by_scheme"].items():
        assert values["16/recovered"] == {"active_draws": 0, "inert_draws": 20}
        assert values["class/4"] == {"active_draws": 0, "inert_draws": 20}
        assert values["minimum_budget"]["active_draws"] > 0
        assert scheme in protocol["label_permutations"]["schemes"]


@pytest.mark.parametrize("mutation", ["weight", "center", "label", "indices"])
def test_saved_ridge_or_label_shuffle_tampering_is_rejected(
    fitted_case, protocol, mutation
):
    fitted, train, observed, plan = fitted_case
    if mutation == "weight":
        fitted["models"]["tuned"]["targets"]["class/2"]["weights"][0] += 0.1
    elif mutation == "center":
        fitted["models"]["tuned"]["targets"]["class/2"]["center"][0] += 0.1
    elif mutation == "label":
        fitted["permutations"]["labels"]["16/recovered"][0] = False
    else:
        plan["draws"][0]["indices"] = list(range(12))
    with pytest.raises(ValueError, match="PREDICTION_INDEPENDENT_AUDIT"):
        audit.check_predictors(fitted, train, observed, plan, protocol)


def test_forecast_recomputation_does_not_consume_test_outcomes(fitted_case):
    fitted = fitted_case[0]
    test = units("test", ["2", "16"])
    expected = audit.predict(fitted, test)
    test[0]["actions"] = {}
    test[1]["actions"] = {"fabricated": "not read"}
    assert audit.predict(fitted, test) == expected


def test_equal_source_metric_is_not_an_intent_pooled_metric():
    result = audit.aggregate([0, 1, 1], ["a", "b", "b"])
    assert result["pooled"] == 2 / 3
    assert result["equal_source_mean"] == 0.5


def test_good_class_accuracy_cannot_hide_under_repair(fitted_case):
    fitted = fitted_case[0]
    test = units("test", ["2"] * 12 + ["16"] * 2)
    result = audit.evaluate(
        audit.predict(fitted, test),
        [audit.outcomes(unit, CODES, RULE) for unit in test],
        fitted,
    )
    majority = result["policies"]["constant_train"]
    assert (
        result["metrics"]["constant_train"]["minimum_class_accuracy"]["pooled"]
        == 12 / 14
    )
    assert majority["qualified_fraction"]["equal_source_mean"] == 6 / 7
    assert majority["mean_updates"]["equal_source_mean"] == 2
    assert not majority["descriptive_savings_without_qualification_loss_on_each_source"]
    assert len(majority["comparisons"]["always/16"]["lost_intents"]) == 2
    assert (
        result["policies"]["always/16"]["qualified_fraction"]["equal_source_mean"] == 1
    )


def test_equal_qualification_rates_can_hide_different_lost_intents(fitted_case):
    fitted = fitted_case[0]
    test = units("test", ["2", "2", "16", "16"])
    for unit in test[:2]:
        unit["actions"]["16"]["guard"] = records("guard", 1, 0)
    predictions = audit.predict(fitted, test)
    observed = [audit.outcomes(unit, CODES, RULE) for unit in test]
    result = audit.evaluate(predictions, observed, fitted)
    majority = result["policies"]["constant_train"]
    assert majority["qualified_fraction"]["equal_source_mean"] == 0.5
    assert (
        result["policies"]["always/16"]["qualified_fraction"]["equal_source_mean"]
        == 0.5
    )
    assert majority["descriptive_savings_without_qualification_loss_on_each_source"]
    assert not majority["savings_without_any_paired_intent_loss"]
    assert len(majority["comparisons"]["always/16"]["lost_intents"]) == 2
    assert len(majority["comparisons"]["always/16"]["newly_qualified_intents"]) == 2


def test_continuous_utility_error_is_separate_from_policy_error(fitted_case):
    fitted = fitted_case[0]
    test = units("test", ["2", "2", "16", "16"])
    result = audit.evaluate(
        audit.predict(fitted, test),
        [audit.outcomes(unit, CODES, RULE) for unit in test],
        fitted,
    )
    scalar = result["metrics"]["constant_train"]["by_budget"]["2"]
    assert scalar["utility_mse"]["pooled"] == 0.0625
    assert scalar["utility_mae"]["pooled"] == 0.25
    assert scalar["recovery_brier"]["pooled"] == 0.25


def publish(tmp_path):
    root = tmp_path / "run"
    study = root / "study"
    study.mkdir(parents=True)
    config = {"stage": "authorize"}
    seal = {"prediction_identity": "identity", "dispatch_manifest": config}
    (study / "seal.json").write_text(
        json.dumps({"payload": seal, "sha256": audit.digest(seal)})
    )
    (study / "result.json").write_text("{}")
    body = {
        "pipeline_stage": "authorize",
        "qualified": True,
        "files": {
            name: audit.file_hash(study / name) for name in ("seal.json", "result.json")
        },
    }
    (study / "receipt.json").write_text(
        json.dumps({"payload": body, "sha256": audit.digest(body)})
    )
    task = {"id": "run", "config": config}
    execution = {
        "task_id": "run",
        "task": task,
        "source_sha256": audit.SOURCE,
        "status": "completed",
        "exit_code": 0,
        "finished_at": "2026-09-12T06:46:00+00:00",
    }
    (root / "execution.json").write_text(json.dumps(execution))
    return root, task


def test_partial_collection_is_pending_not_success(tmp_path):
    root, task = publish(tmp_path)
    (root / "study/result.json").unlink()
    result = audit.read_run(root, task, "identity")
    assert result["state"] == "collection_incomplete"
    assert result["missing"] == ["result.json"]


def test_sealed_file_and_implementation_mutation_fail(tmp_path):
    root, task = publish(tmp_path)
    assert audit.read_run(root, task, "identity")["state"] == "verified"
    with pytest.raises(ValueError, match="prediction identity"):
        audit.read_run(root, task, "different")
    (root / "study/result.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="artifact hash"):
        audit.read_run(root, task, "identity")


def test_missing_collection_produces_no_scientific_result(tmp_path):
    wave = Path(__file__).parent
    root = wave.parents[2] / "experiments/recoverability"
    result = audit.audit(tmp_path, root, wave / "recovery-prediction-dispatch.json")
    assert result["completed_verified_stages"] == 0
    assert result["scientific_conclusion"] is None
    assert result["status"] == "partial_no_final_scientific_conclusion"
    assert len(result["missing_stages"]) == 19
