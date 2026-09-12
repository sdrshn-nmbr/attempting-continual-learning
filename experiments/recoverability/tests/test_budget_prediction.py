import copy
import json
from pathlib import Path

import numpy as np
import pytest
import test_minimum_budget as pilot_tests
import test_prospective as source_tests
from test_prospective import records, rows

import budget_prediction as controller
import budget_prediction_analysis as analysis
import budget_prediction_model as measurement
from budget_prediction_analysis import evaluate, fit_models, forecasts, grouped, prediction_gate
from minimum_budget import minimum_outcome, permutation_plan
from prospective import finish, verify_result
from prospective_analysis import FEATURES
from prospective_model import parameter_hashes
from protocol import ROOT, digest, file_hash, read_json, write_json

design = source_tests.design
tiny_model = source_tests.tiny_model
small_stream = pilot_tests.small_stream


@pytest.fixture(scope="module")
def context():
    return controller.load_study(read_json(ROOT / "configs/budget-prediction-authorize.json"))


def sealed_result(directory, context, payload, manifest=None, files=None, execution=None):
    directory.mkdir(parents=True)
    seal = {"prediction_identity": context["identity"], "dispatch_manifest": manifest or {}}
    write_json(directory / "seal.json", {"payload": seal, "sha256": digest(seal)})
    for name, content in (files or {}).items():
        write_json(directory / name, content)
    finish(directory, payload)
    if execution is not None:
        write_json(directory.parent / "execution.json", execution)
    return directory


def execution(context):
    rule = context["protocol"]["minimum_budget_gate"]
    return {
        "status": "completed",
        "exit_code": 0,
        "finished_at": "2026-09-12T00:00:00+00:00",
        "task_id": rule["run_id"],
        "source_sha256": rule["source_sha256"],
    }


def forbidden(*args, **kwargs):
    pytest.fail("Dependent model or prediction work occurred after a failed prerequisite")


def test_prepared_cohorts_are_new_and_contract_is_unchanged(context):
    data, protocol = context["data"], context["protocol"]
    excluded = set(read_json(ROOT / "minimum-budget-protocol.json")["future_holdout"]["excluded_intents"])
    groups = [set(stream["old"] + stream["new"]) for stream in data["streams"].values()]
    assert len(excluded) == 80
    assert len(set.union(*groups)) == sum(map(len, groups)) == 64
    assert not set.union(*groups).intersection(excluded)
    assert len(data["codes"]) == 16
    assert len(data["lens_fit"]) == 512 and len(data["lens_check"]) == 128
    old = read_json(ROOT / "prospective-protocol.json")
    assert context["spec"]["gates"] == old["gates"] == protocol["learning"]["gates"]
    assert context["spec"]["recovered"] == old["recovered"] == protocol["recovered"]
    assert len(protocol["train_streams"]) == len(protocol["test_streams"]) == 2
    assert file_hash(ROOT / context["config"]["dataset"]) == protocol["dataset_sha256"]


def test_all_dispatch_manifests_use_only_front_door_and_replacement_gate(context):
    paths = sorted((ROOT / "configs").glob("budget-prediction-*.json"))
    assert len(paths) == 19
    for path in paths:
        manifest = read_json(path)
        controller.validate_manifest(manifest, context)
        if manifest["stage"] == "authorize":
            assert Path(manifest["budget_gate"]).parent.name.endswith("-gate-slot")
        else:
            assert Path(manifest["authorization"]).parent.name.endswith("-predict-authorize")
        with pytest.raises(ValueError, match="MANIFEST_FIELDS"):
            controller.validate_manifest(manifest | {"old_test_reuse": True}, context)
    manifest = read_json(ROOT / "configs/budget-prediction-readout-test-a.json")
    del manifest["fitted"]
    with pytest.raises(ValueError, match="MANIFEST_FIELDS"):
        controller.validate_manifest(manifest, context)


@pytest.mark.parametrize("state", [None, "running", "failed", "completed_without_timestamp"])
def test_authorization_requires_completed_execution_not_raw_counts(tmp_path, context, state):
    done = execution(context)
    if state == "completed_without_timestamp":
        done["finished_at"] = None
    elif state is not None:
        done["status"] = state
    source = sealed_result(
        tmp_path / "gate/study",
        context,
        {"stage": "minimum-budget-qualify", "qualified": False},
        execution=done if state else None,
    )
    with pytest.raises(ValueError, match="COMPLETED_EXECUTION_REQUIRED|SOURCE_NOT_COMPLETED"):
        controller.authorize(source, context)


def test_failed_official_gate_seals_zero_work_from_existing_output_directory(tmp_path, context, monkeypatch):
    gate = sealed_result(
        tmp_path / "gate/study",
        context,
        {"stage": "minimum-budget-qualify", "qualified": False},
        execution=execution(context),
    )
    manifest = read_json(ROOT / "configs/budget-prediction-authorize.json") | {"budget_gate": str(gate)}
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    run = tmp_path / "existing-run"
    run.mkdir()
    monkeypatch.setattr(controller, "verify_model", forbidden)
    monkeypatch.setattr(controller.ProspectiveModel, "fresh", forbidden)
    monkeypatch.setattr(controller, "fit_models", forbidden)
    controller.main(["--config", str(path), "--output-dir", str(run)])
    receipt, _ = verify_result(run / "study")
    assert receipt["status"] == "blocked_minimum_budget_gate"
    assert not receipt["qualified"]
    assert (
        receipt["model_loads"] == receipt["repair_updates"] == receipt["lens_updates"] == receipt["predictor_fits"] == 0
    )
    with pytest.raises(FileExistsError):
        controller.main(["--config", str(path), "--output-dir", str(run)])


@pytest.mark.parametrize(
    "suffix",
    ["learn-train-a", "learn-test-a", "cohort", "readout-train-a", "repair-train-a", "fit", "forecast", "analyze"],
)
def test_no_downstream_stage_runs_under_failed_authorization(tmp_path, context, monkeypatch, suffix):
    auth = sealed_result(
        tmp_path / "auth/study",
        context,
        {"pipeline_stage": "authorize", "qualified": False},
        execution=execution(context),
    )
    manifest = read_json(ROOT / f"configs/budget-prediction-{suffix}.json") | {"authorization": str(auth)}
    for name in ("verify_model", "fit_models", "checked_cohort", "cohort_sources", "readout", "repair"):
        monkeypatch.setattr(controller, name, forbidden)
    monkeypatch.setattr(controller.ProspectiveModel, "fresh", forbidden)
    with pytest.raises(ValueError, match="QUALIFIED_SOURCE_REQUIRED"):
        controller.run_stage(manifest, context, tmp_path)


def test_authorization_rejects_original_gate_id_and_changed_scientific_source(tmp_path, context):
    for key, value in (
        ("task_id", "followthrough-20260912-recovery-budget-pilot-gate"),
        ("source_sha256", "different"),
    ):
        source = sealed_result(
            tmp_path / key / "study",
            context,
            {"stage": "minimum-budget-qualify", "qualified": False},
            execution=execution(context) | {key: value},
        )
        with pytest.raises(ValueError, match="MINIMUM_GATE_SOURCE_BINDING"):
            controller.authorize(source, context)


def test_positive_authorization_requires_exact_recomputed_gate(tmp_path, context, monkeypatch):
    manifest, config, spec, data, pilot = controller.load_pilot_manifest(ROOT / "configs/minimum-budget-gate.json")
    payload = {"stage": "minimum-budget-qualify", "qualified": True, "counts": {"2": 9, "16": 3, "8": 1, "never": 1}}
    gate = sealed_result(tmp_path / "gate/study", context, payload, manifest=manifest, execution=execution(context))
    actual = controller.completed

    def complete(path):
        if Path(path) == gate:
            return actual(path)
        assert path in manifest["sources"].values()
        return {}, {}, execution(context)

    monkeypatch.setattr(controller, "completed", complete)
    monkeypatch.setattr(controller, "checked_learning_sources", lambda *args: {"qualified": True})
    calls = []

    def recomputed(sources, prerequisites, given_config, given_spec, given_data, given_pilot):
        assert sources == manifest["sources"] and prerequisites["qualified"]
        assert (given_config, given_spec, given_data, given_pilot) == (config, spec, data, pilot)
        calls.append(True)
        return payload

    monkeypatch.setattr(controller, "recompute_pilot_gate", recomputed)
    result = controller.authorize(gate, context)
    assert result["qualified"] and calls == [True]
    assert result["model_loads"] == result["repair_updates"] == 0
    monkeypatch.setattr(controller, "recompute_pilot_gate", lambda *args: payload | {"counts": {"2": 14}})
    with pytest.raises(ValueError, match="MINIMUM_GATE_RECOMPUTATION"):
        controller.authorize(gate, context)


def synthetic_units(split, labels):
    result = []
    for index, label in enumerate(labels):
        stream = f"fresh-{split}-{'a' if index % 2 == 0 else 'b'}"
        intent = f"{split}-intent-{index}"
        old, new = rows(intent, 0, "test", 20), rows(f"{stream}/new", 1, "test", 20)
        value = -1.0 if label == "2" else 1.0
        result.append(
            {
                "id": f"{stream}/{intent}",
                "intent": intent,
                "stream": stream,
                "split": split,
                "baseline": {
                    "before": {"test": records(old, 20), "guard": records(new, 0)},
                    "after": {"test": records(old, 0), "guard": records(new, 20)},
                },
                "features": {name: [value] for name in FEATURES},
                "actions": {
                    str(budget): {
                        "test": records(old, 20 if label != "never" and budget >= int(label) else 10),
                        "guard": records(new, 20),
                    }
                    for budget in (2, 4, 8, 16)
                },
            }
        )
    return result


def planned(units, protocol):
    return permutation_plan(
        [unit["id"] for unit in units],
        [unit["stream"] for unit in units],
        protocol["label_permutations"]["seeds"],
        protocol["label_permutations"]["schemes"],
    )


@pytest.fixture
def fitted(context):
    units = synthetic_units("train", ["2"] * 6 + ["16"] * 6)
    return fit_models(units, context["spec"], context["protocol"], planned(units, context["protocol"]))


@pytest.mark.parametrize("labels", [["2"] * 12, ["2"] * 11 + ["16"]])
def test_uninformative_fresh_train_distribution_stops_before_any_fit(context, monkeypatch, labels):
    units = synthetic_units("train", labels)
    outcomes = [minimum_outcome(unit, context["spec"], [2, 4, 8, 16]) for unit in units]
    gate = prediction_gate(outcomes, context["protocol"])
    assert not gate["qualified"] and not gate["test_readout_permitted"]
    monkeypatch.setattr(analysis, "ridge_fit", forbidden)
    with pytest.raises(ValueError, match="NO_TRAIN_LABEL_VARIATION"):
        fit_models(units, context["spec"], context["protocol"], planned(units, context["protocol"]))


def test_fit_stage_preserves_failed_train_receipt_and_does_not_fit(tmp_path, context, monkeypatch):
    manifest = read_json(ROOT / "configs/budget-prediction-fit.json")
    auth, cohort, readout = [tmp_path / name for name in ("auth", "cohort", "readout")]
    for path in (auth, cohort, readout):
        path.mkdir()
        write_json(path / "receipt.json", {"fixture": path.name})
    manifest.update(authorization=str(auth), cohort=str(cohort), readout_gate=str(readout))
    monkeypatch.setattr(controller, "source", lambda *args: ({"qualified": True}, {}, {}))
    monkeypatch.setattr(controller, "checked_cohort", lambda *args: {"qualified": True})
    monkeypatch.setattr(controller, "checked_readout_gate", lambda *args: ({"qualified": True}, []))
    monkeypatch.setattr(controller, "repaired_sources", lambda *args: synthetic_units("train", ["2"] * 14))
    monkeypatch.setattr(controller, "fit_models", forbidden)
    result = controller.run_stage(manifest, context, tmp_path)
    assert result["status"] == "no_identifiable_prediction_task"
    assert not result["qualified"] and not result["test_readout_permitted"]
    assert result["predictor_fits"] == 0
    assert not (tmp_path / "predictors.json").exists()


def test_predictors_keep_zero_support_classes_and_all_fixed_permutations(fitted):
    assert len(fitted["models"]) == 45
    assert fitted["gate"]["counts"] == {"2": 6, "4": 0, "8": 0, "16": 6, "never": 0}
    assert fitted["gate"]["unobserved_classes"] == ["4", "8", "never"]
    assert fitted["classes"] == ["2", "4", "8", "16", "never"]
    assert len(fitted["permutations"]["draws"]) == 40
    assert fitted["permutations"]["redraws"] == 0
    for scheme, targets in fitted["permutations"]["activity_by_scheme"].items():
        assert targets["16/recovered"] == {"active_draws": 0, "inert_draws": 20}
        assert targets["minimum_budget"]["active_draws"] > 0
        assert targets["class/4"] == {"active_draws": 0, "inert_draws": 20}
        assert sum(targets["minimum_budget"].values()) == 20
        for draw in fitted["permutations"]["draws"]:
            if draw["scheme"] == scheme:
                assert draw["changed_labels"]["16/recovered"] == 0
                assert "16/recovered" in draw["inert_targets"]


def test_permutation_mutation_and_test_labels_cannot_enter_fit(context, monkeypatch):
    train = synthetic_units("train", ["2"] * 6 + ["16"] * 6)
    plan = planned(train, context["protocol"])
    plan["draws"].pop()
    monkeypatch.setattr(analysis, "ridge_fit", forbidden)
    with pytest.raises(ValueError, match="PLANNED_LABEL_ORDER"):
        fit_models(train, context["spec"], context["protocol"], plan)
    train[0]["split"] = "test"
    with pytest.raises(ValueError, match="FIT_TRAIN_ONLY"):
        fit_models(train, context["spec"], context["protocol"], plan)


def test_forecasts_ignore_test_actions_and_are_saved_coefficients_only(fitted, context, monkeypatch):
    test = synthetic_units("test", ["2", "2", "16", "16"])
    initial_fit = digest(fitted)
    prediction = forecasts(fitted, test)
    monkeypatch.setattr(analysis, "ridge_fit", forbidden)
    original = evaluate(fitted, prediction, test, context["spec"], context["protocol"])
    test[0]["actions"]["2"]["test"] = records(rows(test[0]["intent"], 0, "test", 20), 0)
    assert forecasts(fitted, test) == prediction
    changed = evaluate(fitted, prediction, test, context["spec"], context["protocol"])
    assert (
        changed["policies"]["tuned"]["qualified_fraction"]["pooled"]
        < original["policies"]["tuned"]["qualified_fraction"]["pooled"]
    )
    assert digest(fitted) == initial_fit


@pytest.mark.parametrize("field", ["id", "intent", "stream", "split"])
def test_heldout_prediction_rejects_training_identity_overlap(fitted, field):
    test = synthetic_units("test", ["2"])
    test[0][field] = {
        "id": fitted["training_units"][0],
        "intent": fitted["training_intents"][0],
        "stream": fitted["training_source_streams"][0],
        "split": "train",
    }[field]
    with pytest.raises(ValueError, match="HELDOUT_IDENTITIES"):
        forecasts(fitted, test)


def test_policy_report_pairs_updates_with_qualification_and_majority_baseline(fitted, context):
    test = synthetic_units("test", ["2", "2", "16", "16"])
    result = evaluate(fitted, forecasts(fitted, test), test, context["spec"], context["protocol"])
    report = result["policy_report"]
    assert report["tuned"]["pooled"] == {"qualified_fraction": 1.0, "mean_updates": 9.0, "utility": 1.0}
    assert report["always/16"]["pooled"]["qualified_fraction"] == 1.0
    assert report["always/16"]["pooled"]["mean_updates"] == 16
    for name in ("constant_train", "always/2", "always/4", "always/8"):
        assert report[name]["pooled"]["qualified_fraction"] == 0.5
        assert not result["policies"][name]["descriptive_savings_without_qualification_loss_on_each_source"]
    assert result["policies"]["tuned"]["descriptive_savings_without_qualification_loss_on_each_source"]
    assert set(result["policies"]["tuned"]["paired_differences"]) == {"always/2", "always/16", "constant_train"}
    assert result["independent_test_source_clusters"] == 2
    assert result["source_counts"] == {"fresh-test-a": 2, "fresh-test-b": 2}
    for source in report["tuned"]["by_source"].values():
        assert source["qualified_fraction"] == 1.0 and source["mean_updates"] == 9.0


def test_constant_utility_binary_metrics_and_unequal_source_aggregation(fitted, context):
    test = synthetic_units("test", ["2", "2", "16", "16"])
    result = evaluate(fitted, forecasts(fitted, test), test, context["spec"], context["protocol"])
    metrics = result["metrics"]["constant_train"]["by_budget"]
    assert metrics["2"]["recovery_brier"]["pooled"] == 0.25
    assert metrics["2"]["utility_mse"]["pooled"] == 0.0625
    assert metrics["2"]["utility_mae"]["pooled"] == 0.25
    assert not metrics["16"]["train_binary_variation"] and not metrics["16"]["test_binary_variation"]
    assert grouped([0, 1, 1], ["a", "b", "b"]) == {
        "pooled": 2 / 3,
        "by_source": {"a": 0.0, "b": 1.0},
        "equal_source_mean": 0.5,
    }


def test_failed_fitted_gate_or_wrong_cohort_cannot_enable_test(tmp_path, context, fitted):
    write_json(tmp_path / "predictors.json", fitted)
    predictor_hash = file_hash(tmp_path / "predictors.json")
    for name, qualified, cohort in (("failed", False, "correct"), ("different", True, "wrong")):
        path = sealed_result(
            tmp_path / name / "study",
            context,
            {
                "pipeline_stage": "fit",
                "qualified": qualified,
                "authorization_sha256": "auth",
                "predictors_sha256": predictor_hash,
                "cohort_sha256": cohort,
            },
            files={"predictors.json": fitted},
            execution=execution(context),
        )
        with pytest.raises(ValueError, match="QUALIFIED_SOURCE_REQUIRED|FITTED_TRAIN_GATE"):
            controller.checked_fitted(path, context, "auth", "correct")


def test_actual_repair_freezes_readouts_before_updates_and_checks_all_prefixes(
    tiny_model, small_stream, context, tmp_path, monkeypatch
):
    observer, spec = tiny_model, tiny_model.spec
    learning, output = tmp_path / "learning", tmp_path / "study"
    output.mkdir()
    observer.checkpoint(learning / "acquired")
    observer.checkpoint(learning / "forgotten")
    frozen = parameter_hashes(observer.model, frozen_only=True)
    panel = {
        "test": observer.observe(small_stream["units"]["old"]["test"]),
        "guard": observer.observe(small_stream["units"]["new"]["test"]),
    }
    unit = {
        "id": "train-a/old",
        "stream": "train-a",
        "intent": "old",
        "split": "train",
        "baseline": {"before": panel, "after": panel},
        "features": {name: [0.0] for name in FEATURES},
        "probes": {},
        "actions": {},
    }
    original = observer.update
    calls = []

    def witness(rows, schedule, optimizer):
        assert read_json(output / "features-frozen.json") == [unit]
        calls.append(len(schedule))
        return original(rows, schedule, optimizer)

    monkeypatch.setattr(observer, "update", witness)
    receipt = measurement.repair(observer, spec, context["protocol"], small_stream, learning, [unit], output)
    measured = read_json(output / "units.json")[0]
    measurement.verify_grid(output, measured, small_stream, spec, context["protocol"])
    assert sum(calls) == receipt["repair_updates"] == 30
    assert set(measured["actions"]) == {"16", "2", "4", "8"}
    assert frozen == parameter_hashes(observer.model, frozen_only=True)
    assert observer.observe(small_stream["units"]["old"]["test"]) == panel["test"]
    altered = read_json(output / "actions/old/2-updates.json")
    altered["updates"][0]["loss"] += 1
    (output / "actions/old/2-updates.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="GRID_RECOMPUTATION"):
        measurement.verify_grid(output, measured, small_stream, spec, context["protocol"])


def test_baseline_mismatch_stops_before_any_repair(tiny_model, small_stream, context, tmp_path, monkeypatch):
    tiny_model.checkpoint(tmp_path / "acquired")
    tiny_model.checkpoint(tmp_path / "forgotten")
    unit = synthetic_units("train", ["2"])[0]
    unit.update(intent="old", actions={})
    monkeypatch.setattr(measurement, "grid", forbidden)
    with pytest.raises(ValueError, match="PREUPDATE_BASELINE_PARITY"):
        measurement.repair(tiny_model, tiny_model.spec, context["protocol"], small_stream, tmp_path, [unit], tmp_path)


def lens_fixture(tmp_path, spec):
    spec = copy.deepcopy(spec)
    spec["lens"].update(updates=2, token_batch_size=2)
    data = {key: rows(key, 0, "train", 2) for key in ("lens_fit", "lens_check")}
    metadata = {
        key: {
            "positions": [{"text_sha256": row["text_sha256"], "position": 1} for row in data["lens_" + key]],
            "prompts": 2,
            "tokens": 2,
            "terminal_max_error": 0.0,
        }
        for key in ("fit", "check")
    }
    values = {str(layer): {"frozen": 1.0, "tuned": 0.8} for layer in spec["layers"]}
    improvement = 1 - sum(value["tuned"] for value in values.values()) / len(values)
    (tmp_path / "tuned-lens.safetensors").write_bytes(b"lens-proof-fixture")
    lens_hash = file_hash(tmp_path / "tuned-lens.safetensors")
    training = metadata | {
        "final_heldout_kl": values,
        "model_parameters_unchanged": True,
        "lens_sha256": lens_hash,
        "reload_exact": True,
        "history": [{"step": i + 1, "positions": [0, 1]} for i in range(2)],
    }
    late = {"metadata": metadata["check"] | {"terminal_max_error": 1e-8}, "heldout_kl": copy.deepcopy(values)}
    write_json(tmp_path / "lens-training.json", training)
    write_json(tmp_path / "lens-late-check.json", late)
    receipt = {
        "acquired_kl_improvement": improvement,
        "forgotten_kl_improvement": improvement,
        "lens_sha256": lens_hash,
    }
    return data, spec, receipt, training, late


def test_lens_requires_calibration_before_and_after_forgetting_on_same_train_positions(tmp_path, context):
    data, spec, receipt, _, late = lens_fixture(tmp_path, context["spec"])
    controller.lens_valid(tmp_path, receipt, data, spec)
    for value in late["heldout_kl"].values():
        value["tuned"] = value["frozen"]
    receipt["forgotten_kl_improvement"] = 0.0
    (tmp_path / "lens-late-check.json").write_text(json.dumps(late))
    with pytest.raises(ValueError, match="LENS_KL_QUALIFICATION"):
        controller.lens_valid(tmp_path, receipt, data, spec)


@pytest.mark.parametrize("failure", ["test_row", "nonfinite", "fit_budget", "model_mutation", "new_positions"])
def test_lens_gate_rejects_invalid_calibration_receipts(tmp_path, context, failure):
    data, spec, receipt, training, late = lens_fixture(tmp_path, context["spec"])
    if failure == "test_row":
        data["lens_fit"][0]["source_split"] = "test"
    elif failure == "nonfinite":
        training["final_heldout_kl"][str(spec["layers"][0])]["tuned"] = float("nan")
    elif failure == "fit_budget":
        training["history"][0]["positions"] = [2, 3]
    elif failure == "model_mutation":
        training["model_parameters_unchanged"] = False
    else:
        late["metadata"]["positions"] = late["metadata"]["positions"][::-1]
    (tmp_path / "lens-training.json").write_text(json.dumps(training))
    (tmp_path / "lens-late-check.json").write_text(json.dumps(late))
    with pytest.raises(ValueError, match="BUDGET_PREDICTION_LENS"):
        controller.lens_valid(tmp_path, receipt, data, spec)


def test_linear_model_scalers_remain_train_only(fitted):
    before = copy.deepcopy(fitted)
    test = synthetic_units("test", ["2"])
    for feature in test[0]["features"]:
        test[0]["features"][feature] = [100.0]
    forecasts(fitted, test)
    assert before == fitted
    assert np.isfinite(fitted["models"]["tuned"]["targets"]["class/2"]["weights"]).all()


def test_high_minimum_class_accuracy_does_not_hide_under_repair(fitted, context):
    test = synthetic_units("test", ["2"] * 12 + ["16"] * 2)
    prediction = forecasts(fitted, test)
    result = evaluate(fitted, prediction, test, context["spec"], context["protocol"])
    majority = result["policies"]["constant_train"]
    assert result["metrics"]["constant_train"]["minimum_class_accuracy"]["pooled"] == 12 / 14
    assert majority["qualified_fraction"]["pooled"] == 12 / 14
    assert majority["mean_updates"]["pooled"] == 2
    assert majority["underbudget_failure_fraction"]["pooled"] == 2 / 14
    assert not majority["descriptive_savings_without_qualification_loss_on_each_source"]
