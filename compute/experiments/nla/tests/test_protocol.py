from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from protocol import (
    candidate_metrics,
    digest,
    ensure_paired_states,
    load_config,
    make_calibration,
    make_records,
    matched_donors,
    training_query,
    validate_config,
)


def test_data_is_seeded_balanced_and_entity_disjoint(config):
    records = make_records(config)
    assert records == make_records(config)
    different = deepcopy(config)
    different["seed"] += 1
    assert records != make_records(different)
    assert len({r.entity for r in records}) == len(records)
    assert {r.entity for r in records if r.split == "development"}.isdisjoint(
        {r.entity for r in records if r.split == "heldout"}
    )
    for stage in ("a", "b"):
        counts = Counter(r.answer for r in records if r.learning_stage == stage)
        assert max(counts.values()) - min(counts.values()) <= 1
    for record in records:
        assert record.answer.lower() not in record.query.lower()
        if record.learning_stage:
            assert all(training_query(record, i) != record.query for i in range(2))


def test_control_donors_never_cross_entity_split_or_task(config):
    records = make_records(config)
    donors = matched_donors(records, 311)
    assert donors == matched_donors(records, 311)
    assert sorted(donors.values()) == list(range(len(records)))
    for i, j in donors.items():
        assert i != j
        a, b = records[i], records[j]
        assert (a.cohort, a.split, a.learning_stage) == (b.cohort, b.split, b.learning_stage)


def test_binding_controls_swap_both_owners_and_preserve_query(config):
    calibration = make_calibration(config)
    donors = matched_donors(calibration, 4)
    for i, record in enumerate(calibration):
        if record["group"].startswith("binding_pair_"):
            donor = calibration[donors[i]]
            assert record["group"] == donor["group"]
            assert record["aliases"][0] == donor["other_answer"]
            assert (
                record["content"].split(". The registered color")[1]
                == donor["content"].split(". The registered color")[1]
            )


def test_native_diagnostic_config_keeps_original_thresholds_and_larger_fixed_corpus(
    diagnostic_config, native_tokenizers
):
    original = load_config(Path(__file__).parents[1] / "smoke.json")
    config = validate_config(diagnostic_config)
    assert config["mode"] == "calibrate" and config["updates"] == [] and config["seed"] == 2718
    assert config["gate"] == original["gate"]
    assert config["max_new_tokens"] == original["max_new_tokens"] == 192
    records = make_calibration(config)
    assert len(records) == 16
    assert Counter(row["group"] for row in records) == {"topic": 8, **{f"binding_pair_{i}": 2 for i in range(4)}}
    assert records == make_calibration(config)
    donors = matched_donors(records, config["seed"] + 211)
    assert all(i != j and records[i]["group"] == records[j]["group"] for i, j in donors.items())
    tokenizer = native_tokenizers["base"]
    assert all(len(tokenizer(row["content"])["input_ids"]) < config["max_sequence_length"] - 16 for row in records)


def test_native_calibration_rejects_any_training_updates(config):
    config["mode"] = "calibrate"
    with pytest.raises(ValueError, match="CALIBRATION_NO_TRAINING"):
        validate_config(config)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"model_id": "Qwen/Qwen3.5-4B"}, "UNSUPPORTED_MODEL"),
        ({"revision": "main"}, "CONFIG_REVISION"),
        ({"device": "cuda:1"}, "RUNTIME_CONTRACT"),
        ({"source_token_policy": "last_gold_token"}, "CONFIG_UNKNOWN"),
        ({"lora": {"r": 8, "lora_alpha": 16, "target_modules": ["lm_head"]}}, "CONFIG_LORA"),
    ],
)
def test_unsupported_contracts_fail_early(config, mutation, reason):
    config.update(mutation)
    with pytest.raises(ValueError, match=reason):
        validate_config(config)


def test_sequential_budgets_must_match(config):
    config["updates"].append({**config["updates"][0], "name": "b", "steps": 19})
    with pytest.raises(ValueError, match="UNMATCHED_BUDGET"):
        validate_config(config)


def test_singleton_shuffle_control_is_rejected(config):
    with pytest.raises(ValueError, match="CONTROL_DONOR"):
        matched_donors(make_records(config)[:1], 4)


def test_pair_alignment_rejects_output_gold_and_position_drift(config):
    record = make_records(config)[0]
    state = {
        "id": record.id,
        "source_layer": 20,
        "activation": torch.ones(4),
        "source_has_output_or_gold_tokens": False,
        "query": {"input_ids": [1, 2], "source_token_index": 1, "source_token_id": 2, "prompt_sha256": "a"},
    }
    before = {record.id: state}
    after = deepcopy(before)
    ensure_paired_states([record], before, after)
    after[record.id]["query"]["source_token_index"] = 0
    with pytest.raises(ValueError, match="PAIRED_QUERY"):
        ensure_paired_states([record], before, after)
    after = deepcopy(before)
    after[record.id]["source_has_output_or_gold_tokens"] = True
    with pytest.raises(ValueError, match="PAIRED_GOLD_LEAK"):
        ensure_paired_states([record], before, after)


def test_candidate_metrics_keep_length_normalization_separate():
    result = candidate_metrics([-2, -3], [1, 3], ["short", "long answer"], "long answer")
    assert result["gold_logprob_sum"] == -3
    assert result["gold_logprob_mean"] == -1
    assert result["candidate_prediction"] == "short"
    assert result["candidate_correct"] == 0
    assert result["gold_margin"] == -1
    assert digest(result) == digest(deepcopy(result))
