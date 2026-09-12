from copy import deepcopy

import numpy as np
import pytest
import torch

from analysis import calibration_diagnostics, fidelity_gate, forgetting_prediction, ridge_predict
from native import generation_diagnostics, reconstruction_metrics
from protocol import Record, make_calibration, matched_donors


@pytest.fixture
def diagnostic_case(diagnostic_config, native_tokenizers):
    calibration = make_calibration(diagnostic_config)
    donors = matched_donors(calibration, diagnostic_config["seed"] + 211)
    vectors = torch.eye(len(calibration))
    states = {row["id"]: {"activation": vectors[i]} for i, row in enumerate(calibration)}
    tokenizer = native_tokenizers["av"]
    readouts = {}
    for i, row in enumerate(calibration):
        result = {"reconstructions": {}}
        for condition in ("true", "empty", "shuffled", "random"):
            explanation = calibration[donors[i]]["aliases"][0] if condition == "shuffled" else row["aliases"][0]
            text = f"<explanation>{explanation}" + ("" if condition == "empty" else "</explanation>")
            ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            measured = generation_diagnostics(tokenizer, ids, 192)
            measured.update(
                metrics=None,
                injected_raw_norm=0.0 if condition == "empty" else 1.0,
                injection_slot_norm=0.0 if condition == "empty" else 150.0,
            )
            if measured["format_valid"]:
                vector = vectors[donors[i]] if condition == "shuffled" else vectors[i]
                result["reconstructions"][condition] = vector
                measured["metrics"] = reconstruction_metrics(vector, vectors[i], len(calibration) ** 0.5)
            result[condition] = measured
        readouts[row["id"]] = result
    return calibration, states, readouts


def test_independent_diagnostics_report_every_control_when_zero_is_invalid(diagnostic_case, diagnostic_config):
    calibration, states, readouts = diagnostic_case
    before = fidelity_gate(calibration, states, readouts, diagnostic_config)
    diagnostics = calibration_diagnostics(calibration, states, readouts)
    assert set(diagnostics["conditions"]) == {"true", "empty", "shuffled", "random"}
    assert diagnostics["random_norm_matching"]["n_total"] == 16
    assert diagnostics["random_norm_matching"]["max_raw_relative_norm_error"] == 0
    assert diagnostics["random_norm_matching"]["max_injection_slot_norm_difference_from_true"] == 0
    assert not before["passed"] and not before["checks"]["all_controls_scorable"]
    assert before == fidelity_gate(calibration, states, readouts, diagnostic_config)
    empty = diagnostics["conditions"]["empty"]
    assert empty["output"]["n_total"] == empty["output"]["eos_before_closing_tag_count"] == 16
    assert empty["output"]["n_scorable"] == empty["output"]["n_exact_format_valid"] == 0
    assert empty["topic_information"]["mean_recipient_cosine"] is None
    assert empty["fact_binding"]["n_pairs_scorable"] == 0
    assert len(empty["fact_binding"]["pairs"]) == 4
    assert empty["fact_binding"]["both_members_correct_rate_among_scorable_pairs"] is None
    for condition in ("true", "shuffled", "random"):
        assert diagnostics["conditions"][condition]["output"]["n_scorable"] == 16
        assert diagnostics["conditions"][condition]["fact_binding"]["n_pairs_scorable"] == 4
    true = diagnostics["conditions"]["true"]
    assert true["topic_information"]["mean_recipient_cosine"] == 1
    assert true["topic_information"]["retrieval_accuracy_among_scorable"] == 1
    assert true["fact_binding"]["both_members_correct_count"] == 4
    assert diagnostics["conditions"]["shuffled"]["fact_binding"]["both_members_correct_count"] == 0
    paired = diagnostics["paired_condition_comparisons"]
    assert paired["empty"]["topic"]["n_paired_scorable"] == 0
    assert paired["empty"]["topic"]["mean_true_minus_control_cosine"] is None
    assert paired["shuffled"]["topic"]["n_paired_scorable"] == 8
    assert paired["shuffled"]["topic"]["mean_true_minus_control_cosine"] == 1
    assert diagnostics["conditions"]["random"]["topic_information"]["mean_recipient_cosine"] == 1
    assert not fidelity_gate(calibration, states, readouts, diagnostic_config)["passed"]


@pytest.mark.parametrize("collapse_to_tie", [False, True])
def test_topic_success_does_not_hide_failed_counterfactual_binding_pairs(diagnostic_case, collapse_to_tie):
    calibration, states, readouts = diagnostic_case
    bindings = [row for row in calibration if row["group"] != "topic"]
    for first, second in zip(bindings[::2], bindings[1::2], strict=True):
        first_vector = states[first["id"]]["activation"]
        second_vector = states[second["id"]]["activation"]
        vector = first_vector + second_vector if collapse_to_tie else first_vector
        for member in (first, second):
            readouts[member["id"]]["reconstructions"]["true"] = vector
            readouts[member["id"]]["true"]["metrics"] = reconstruction_metrics(
                vector, states[member["id"]]["activation"], len(calibration) ** 0.5
            )
            readouts[member["id"]]["true"]["explanation"] = f"{member['aliases'][0]}, not {member['other_answer']}"
    result = calibration_diagnostics(calibration, states, readouts)["conditions"]["true"]
    assert result["topic_information"]["keyword_hit_rate_among_valid"] == 1
    assert result["topic_information"]["retrieval_accuracy_among_scorable"] == 1
    binding = result["fact_binding"]
    assert binding["n_states_scorable"] == 8 and binding["n_pairs_scorable"] == 4
    assert binding["direction_correct_count"] == (0 if collapse_to_tie else 4)
    assert binding["direction_tie_count"] == (8 if collapse_to_tie else 0)
    assert binding["both_members_correct_count"] == 0
    assert binding["color_mention_counts"]["both"] == 8


def test_native_gate_cannot_pass_unscorable_or_prior_only_controls(config):
    calibration = make_calibration(config)
    states, readouts = {}, {}
    for i, row in enumerate(calibration):
        vector = torch.eye(len(calibration))[i]
        states[row["id"]] = {"activation": vector}
        conditions = {
            name: {"metrics": {"cosine": value}, "explanation": row["aliases"][0] if name == "true" else "empty"}
            for name, value in (("true", 1.0), ("shuffled", 0.0), ("empty", 0.0))
        }
        readouts[row["id"]] = {**conditions, "reconstructions": {"true": vector}}
    assert fidelity_gate(calibration, states, readouts, config)["passed"]
    bad = deepcopy(readouts)
    for row in bad.values():
        row["shuffled"]["metrics"]["cosine"] = 1.0
    assert not fidelity_gate(calibration, states, bad, config)["passed"]
    bad = deepcopy(readouts)
    bad[calibration[0]["id"]]["empty"]["metrics"] = None
    result = fidelity_gate(calibration, states, bad, config)
    assert not result["passed"]
    assert not result["measurements"]["all_controls_scorable"]


def prediction_fixture(config):
    config["prediction"].update(min_development=4, min_heldout=4, projection_dim=2, bootstrap_samples=100)
    rng = np.random.default_rng(18)
    records, before, after, readouts = [], {}, {}, {}
    for i in range(12):
        record = Record(
            f"r{i}", f"entity{i}", "native_fact", "development" if i < 6 else "heldout", "query", ("red", "blue"), "red"
        )
        records.append(record)
        hidden = torch.tensor(rng.normal(size=8), dtype=torch.float32)
        behavior = {
            "gold_logprob_mean": -0.1 - 0.01 * i,
            "gold_margin": 2.0 + 0.3 * i,
            "candidate_entropy": 0.3,
            "gold_probability_within_candidates": 0.9,
            "greedy_exact": 1,
            "candidate_correct": 1,
        }
        before[record.id] = {"behavior": behavior, "activation": hidden}
        after[record.id] = {"behavior": {**behavior, "gold_logprob_mean": behavior["gold_logprob_mean"] - 0.03 * i}}
        readouts[record.id] = {
            name: {"metrics": {"cosine": value}, "generated_tokens": 30, "explanation": "red"}
            for name, value in (("true", 0.6 + 0.01 * i), ("empty", 0.2), ("shuffled", 0.1))
        }
    return records, before, after, readouts


def test_forecast_fits_only_development_labels(config):
    records, before, after, readouts = prediction_fixture(config)
    first = forgetting_prediction(records, before, after, readouts, "a", config)
    assert first["status"] == "measured_exploratory"
    changed = deepcopy(after)
    for record in records:
        if record.split == "heldout":
            changed[record.id]["behavior"]["gold_logprob_mean"] -= 100
    second = forgetting_prediction(records, before, changed, readouts, "a", config)
    for a, b in zip(first["heldout_predictions"], second["heldout_predictions"], strict=True):
        for key in a.keys() - {"target_logprob_drop"}:
            assert a[key] == b[key]
    assert first["scores"]["logprobs"]["alpha"] == second["scores"]["logprobs"]["alpha"]


def test_missing_sample_headroom_emits_no_forecast_metric(config):
    records, before, after, readouts = prediction_fixture(config)
    config["prediction"]["min_development"] = 20
    result = forgetting_prediction(records, before, after, readouts, "a", config)
    assert result["status"] == "insufficient_previously_correct_scorable_entities"
    assert result["scores"] is None


def test_small_probe_scaling_uses_only_training_samples():
    x = np.array([[0.0, 2], [1, 2], [2, 2], [3, 2]])
    y = np.array([0.0, 1, 2, 3])
    single = ridge_predict(x, y, np.array([[1.5, 2]]), 1.0)
    with_extreme_heldout = ridge_predict(x, y, np.array([[1.5, 2], [1e10, -1e10]]), 1.0)
    assert single[0] == pytest.approx(with_extreme_heldout[0])
