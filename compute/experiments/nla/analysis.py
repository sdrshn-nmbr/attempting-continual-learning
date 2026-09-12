import re

import numpy as np

from native import cosine
from protocol import mean_or_none, normalized_text


def alias_hit(text, aliases):
    lowered = (text or "").lower()
    return int(any(alias.lower() in lowered for alias in aliases))


def answer_hit(text, answer):
    return int(
        re.search(r"(?<!\w)" + re.escape(normalized_text(answer)) + r"(?!\w)", normalized_text(text or "")) is not None
    )


def fidelity_gate(calibration, states, readouts, config):
    topic_indices = [i for i, row in enumerate(calibration) if row["group"] == "topic"]
    true_cos, shuffled_cos, empty_cos = [], [], []
    true_hits, shuffled_hits, empty_hits, retrieval = [], [], [], []
    binding_wins = []
    for i, row in enumerate(calibration):
        result = readouts[row["id"]]
        if any(result[k]["metrics"] is None for k in ("true", "empty", "shuffled")):
            continue
        if row["group"] == "topic":
            true_cos.append(result["true"]["metrics"]["cosine"])
            shuffled_cos.append(result["shuffled"]["metrics"]["cosine"])
            empty_cos.append(result["empty"]["metrics"]["cosine"])
            true_hits.append(alias_hit(result["true"]["explanation"], row["aliases"]))
            shuffled_hits.append(alias_hit(result["shuffled"]["explanation"], row["aliases"]))
            empty_hits.append(alias_hit(result["empty"]["explanation"], row["aliases"]))
            similarities = [
                cosine(result["reconstructions"]["true"], states[calibration[j]["id"]]["activation"])
                for j in topic_indices
            ]
            retrieval.append(int(topic_indices[int(np.argmax(similarities))] == i))
        else:
            binding_wins.append(int(result["true"]["metrics"]["cosine"] > result["shuffled"]["metrics"]["cosine"]))
    complete = len(true_cos) == len(topic_indices) and len(binding_wins) == len(calibration) - len(topic_indices)
    gate = config["gate"]
    measurements = {
        "topic_count": len(topic_indices),
        "scorable_topic_count": len(true_cos),
        "all_controls_scorable": complete,
        "mean_true_cosine": mean_or_none(true_cos),
        "mean_true_minus_shuffled_cosine": mean_or_none(np.asarray(true_cos) - np.asarray(shuffled_cos)),
        "mean_true_minus_empty_cosine": mean_or_none(np.asarray(true_cos) - np.asarray(empty_cos)),
        "true_topic_hit_rate": mean_or_none(true_hits),
        "shuffled_topic_hit_rate": mean_or_none(shuffled_hits),
        "empty_topic_hit_rate": mean_or_none(empty_hits),
        "topic_retrieval_accuracy": mean_or_none(retrieval),
        "paired_binding_direction_win_rate": mean_or_none(binding_wins),
    }
    checks = {"all_controls_scorable": complete}
    for measurement, threshold in (
        ("mean_true_cosine", "min_cosine"),
        ("mean_true_minus_shuffled_cosine", "min_shuffled_gap"),
        ("mean_true_minus_empty_cosine", "min_empty_gap"),
        ("true_topic_hit_rate", "min_topic_hit_rate"),
        ("topic_retrieval_accuracy", "min_retrieval_accuracy"),
        ("paired_binding_direction_win_rate", "min_binding_win_rate"),
    ):
        value = measurements[measurement]
        checks[measurement] = value is not None and value >= gate[threshold]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": measurements,
        "thresholds": gate,
        "interpretation": "Predeclared small native calibration gate; passing does not establish readout truth or portability after weight changes.",
    }


def topic_diagnostics(topics, states, readouts, condition):
    rows = []
    for topic in topics:
        measured = readouts[topic["id"]][condition]
        row = {
            "id": topic["id"],
            "keyword_hit": alias_hit(measured["explanation"], topic["aliases"]) if measured["format_valid"] else None,
            "recipient_cosine": measured["metrics"]["cosine"] if measured["metrics"] is not None else None,
            "retrieval_correct": None,
            "retrieval_tie": None,
        }
        if measured["metrics"] is not None:
            reconstructed = readouts[topic["id"]]["reconstructions"][condition]
            similarities = {other["id"]: cosine(reconstructed, states[other["id"]]["activation"]) for other in topics}
            margin = similarities[topic["id"]] - max(value for key, value in similarities.items() if key != topic["id"])
            row.update(
                retrieval_correct=int(margin > 1e-6),
                retrieval_tie=int(abs(margin) <= 1e-6),
                retrieval_margin=margin,
            )
        rows.append(row)
    valid = [row for row in rows if row["keyword_hit"] is not None]
    scorable = [row for row in rows if row["recipient_cosine"] is not None]
    return {
        "n_total": len(rows),
        "n_format_valid": len(valid),
        "n_scorable": len(scorable),
        "mean_recipient_cosine": mean_or_none([row["recipient_cosine"] for row in scorable]),
        "keyword_hit_count": sum(row["keyword_hit"] for row in valid),
        "keyword_hit_rate_among_valid": mean_or_none([row["keyword_hit"] for row in valid]),
        "retrieval_correct_count": sum(row["retrieval_correct"] for row in scorable),
        "retrieval_accuracy_among_scorable": mean_or_none([row["retrieval_correct"] for row in scorable]),
        "retrieval_tie_count": sum(row["retrieval_tie"] for row in scorable),
        "rows": rows,
    }


def binding_diagnostics(bindings, states, readouts, condition):
    groups = {}
    for item in bindings:
        groups.setdefault(item["group"], []).append(item)
    pairs = []
    for group, members in groups.items():
        if len(members) != 2:
            raise ValueError(f"NLA_BINDING_PAIR: {group} must have exactly two counterfactual members")
        rows = []
        for member, other in (members, members[::-1]):
            measured = readouts[member["id"]][condition]
            target = states[member["id"]]["activation"]
            counterfactual = states[other["id"]]["activation"]
            mentions = "invalid_format"
            if measured["format_valid"]:
                target_hit = answer_hit(measured["explanation"], member["aliases"][0])
                other_hit = answer_hit(measured["explanation"], member["other_answer"])
                mentions = {(0, 0): "neither", (1, 0): "target_only", (0, 1): "other_only", (1, 1): "both"}[
                    target_hit, other_hit
                ]
            gap = None
            if measured["metrics"] is not None:
                reconstructed = readouts[member["id"]]["reconstructions"][condition]
                gap = cosine(reconstructed, target) - cosine(reconstructed, counterfactual)
            rows.append(
                {
                    "id": member["id"],
                    "counterfactual_id": other["id"],
                    "target_entity": member["target_entity"],
                    "target_color": member["aliases"][0],
                    "counterfactual_color": member["other_answer"],
                    "source_counterfactual_cosine": cosine(target, counterfactual),
                    "target_minus_counterfactual_cosine": gap,
                    "direction_correct": int(gap > 1e-6) if gap is not None else None,
                    "direction_tie": int(abs(gap) <= 1e-6) if gap is not None else None,
                    "color_mentions": mentions,
                }
            )
        scorable = all(row["direction_correct"] is not None for row in rows)
        pairs.append(
            {
                "pair": group,
                "both_members_scorable": scorable,
                "both_members_correct": int(all(row["direction_correct"] for row in rows)) if scorable else None,
                "rows": rows,
            }
        )
    rows = [row for pair in pairs for row in pair["rows"]]
    scorable_rows = [row for row in rows if row["direction_correct"] is not None]
    scorable_pairs = [pair for pair in pairs if pair["both_members_scorable"]]
    return {
        "n_states_total": len(rows),
        "n_states_scorable": len(scorable_rows),
        "mean_target_minus_counterfactual_cosine": mean_or_none(
            [row["target_minus_counterfactual_cosine"] for row in scorable_rows]
        ),
        "direction_correct_count": sum(row["direction_correct"] for row in scorable_rows),
        "direction_accuracy_among_scorable": mean_or_none([row["direction_correct"] for row in scorable_rows]),
        "direction_tie_count": sum(row["direction_tie"] for row in scorable_rows),
        "n_pairs_total": len(pairs),
        "n_pairs_scorable": len(scorable_pairs),
        "both_members_correct_count": sum(pair["both_members_correct"] for pair in scorable_pairs),
        "both_members_correct_rate_among_scorable_pairs": mean_or_none(
            [pair["both_members_correct"] for pair in scorable_pairs]
        ),
        "color_mention_counts": {
            label: sum(row["color_mentions"] == label for row in rows)
            for label in ("target_only", "other_only", "both", "neither", "invalid_format")
        },
        "pairs": pairs,
    }


def calibration_diagnostics(calibration, states, readouts):
    groups = {
        "topic": [row for row in calibration if row["group"] == "topic"],
        "binding": [row for row in calibration if row["group"] != "topic"],
    }
    conditions = {}
    for condition in ("true", "empty", "shuffled", "random"):
        measured = [readouts[row["id"]][condition] for row in calibration]
        conditions[condition] = {
            "output": {
                "n_total": len(measured),
                "n_format_valid": sum(row["format_valid"] for row in measured),
                "n_exact_format_valid": sum(row["exact_format_valid"] for row in measured),
                "n_scorable": sum(row["metrics"] is not None for row in measured),
                "stop_reason_counts": {
                    reason: sum(row["stop_reason"] == reason for row in measured)
                    for reason in ("eos", "length", "other")
                },
                "eos_before_closing_tag_count": sum(row["eos_before_closing_tag"] for row in measured),
                "hit_token_cap_count": sum(row["hit_token_cap"] for row in measured),
                "mean_generated_tokens": mean_or_none([row["generated_tokens"] for row in measured]),
                "mean_injected_raw_norm": mean_or_none([row["injected_raw_norm"] for row in measured]),
                "mean_injection_slot_norm": mean_or_none([row["injection_slot_norm"] for row in measured]),
            },
            "topic_information": topic_diagnostics(groups["topic"], states, readouts, condition),
            "fact_binding": binding_diagnostics(groups["binding"], states, readouts, condition),
        }
    comparisons = {}
    for control in ("empty", "shuffled", "random"):
        comparisons[control] = {}
        for group, items in groups.items():
            paired = [
                readouts[row["id"]]
                for row in items
                if all(readouts[row["id"]][name]["metrics"] is not None for name in ("true", control))
            ]
            gaps = [row["true"]["metrics"]["cosine"] - row[control]["metrics"]["cosine"] for row in paired]
            comparisons[control][group] = {
                "n_total": len(items),
                "n_paired_scorable": len(paired),
                "mean_true_minus_control_cosine": mean_or_none(gaps),
                "positive_gap_count": sum(gap > 1e-6 for gap in gaps),
                "tie_count": sum(abs(gap) <= 1e-6 for gap in gaps),
            }
    return {
        "conditions": conditions,
        "paired_condition_comparisons": comparisons,
        "random_norm_matching": {
            "n_total": len(calibration),
            "max_raw_relative_norm_error": max(
                abs(
                    readouts[row["id"]]["random"]["injected_raw_norm"]
                    / readouts[row["id"]]["true"]["injected_raw_norm"]
                    - 1
                )
                for row in calibration
            ),
            "max_injection_slot_norm_difference_from_true": max(
                abs(
                    readouts[row["id"]]["random"]["injection_slot_norm"]
                    - readouts[row["id"]]["true"]["injection_slot_norm"]
                )
                for row in calibration
            ),
            "interpretation": "Raw L2 is matched in float32. Every nonzero control uses the released injection scale; measured slot norms include BF16 rounding.",
        },
        "original_gate_conditions": ["true", "empty", "shuffled"],
        "random_role": "Diagnostic-only deterministic Gaussian direction, recipient raw L2 matched, then released AV normalization; never replaces empty in the gate.",
        "empty_role": "The original empty condition is a zero vector at the injection slot, not an absent prompt.",
        "format_contract": "format_valid retains the original one-nonempty-regex-match parser. exact_format_valid additionally requires exactly one tag pair and no outer text except whitespace and terminal EOS. Token IDs and stop details are saved for every output.",
        "scoring_contract": "Each condition uses its own valid/scorable denominator. Reconstruction scores always reference the recipient source. Diagnostic comparisons treat absolute margins <=1e-6 as ties; retrieval counts ties as incorrect. Original gate logic is unchanged.",
        "interpretation": "Topic keywords and topic retrieval measure broad topic information. Binding tests counterfactual direction discrimination with both members required per pair. Pairs are the units of analysis in this small fixed corpus; color mentions are lexical evidence only and may include negation. These diagnostics do not establish semantic binding truth, durable learning, or rescue a failed native gate.",
    }


def behavior_change(records, before, after):
    rows = []
    for record in records:
        a, b = before[record.id]["behavior"], after[record.id]["behavior"]
        rows.append(
            {
                "id": record.id,
                "split": record.split,
                "gold_logprob_mean_before": a["gold_logprob_mean"],
                "gold_logprob_mean_after": b["gold_logprob_mean"],
                "forgetting_logprob_drop": a["gold_logprob_mean"] - b["gold_logprob_mean"],
                "candidate_correct_before": a["candidate_correct"],
                "candidate_correct_after": b["candidate_correct"],
                "greedy_exact_before": a["greedy_exact"],
                "greedy_exact_after": b["greedy_exact"],
            }
        )
    return {
        "n": len(rows),
        "mean_gold_logprob_gain": mean_or_none([-r["forgetting_logprob_drop"] for r in rows]),
        "candidate_accuracy_before": mean_or_none([r["candidate_correct_before"] for r in rows]),
        "candidate_accuracy_after": mean_or_none([r["candidate_correct_after"] for r in rows]),
        "greedy_exact_before": mean_or_none([r["greedy_exact_before"] for r in rows]),
        "greedy_exact_after": mean_or_none([r["greedy_exact_after"] for r in rows]),
        "lost_exact_count": sum(r["greedy_exact_before"] and not r["greedy_exact_after"] for r in rows),
        "gained_exact_count": sum(not r["greedy_exact_before"] and r["greedy_exact_after"] for r in rows),
        "rows": rows,
    }


def eligible_retention(record, previous, stage):
    old_task = record.learning_stage is not None and record.learning_stage < stage
    behavior = previous[record.id]["behavior"]
    return (record.cohort == "native_fact" or old_task) and bool(
        behavior["greedy_exact"] and behavior["candidate_correct"]
    )


def stage_behavior(records, base, before, after, frozen, stage):
    groups = {
        "acquisition_heldout_paraphrases": [r for r in records if r.learning_stage == stage],
        "retention_all_native_facts": [r for r in records if r.cohort == "native_fact"],
        "retention_previously_correct": [r for r in records if eligible_retention(r, before, stage)],
        "retention_previous_task_all": [
            r for r in records if r.learning_stage is not None and r.learning_stage < stage
        ],
        "unexposed_random_binding_control": [
            r for r in records if r.learning_stage is not None and r.learning_stage > stage
        ],
    }
    frozen_error = max(
        abs(base[r.id]["behavior"]["gold_logprob_mean"] - frozen[r.id]["behavior"]["gold_logprob_mean"])
        for r in records
    )
    frozen_vector_error = max(
        float((base[r.id]["activation"] - frozen[r.id]["activation"]).abs().max()) for r in records
    )
    return {
        "groups": {name: behavior_change(group, before, after) for name, group in groups.items()},
        "frozen_repeat": {
            "n": len(records),
            "max_gold_logprob_error": frozen_error,
            "max_activation_absolute_error": frozen_vector_error,
        },
        "comparison": "Sequential LoRA versus measured adapter-disabled frozen source. Frozen control has zero optimizer steps; inference/data budgets match.",
    }


def readout_drift(records, before, after, readouts_before, readouts_after):
    rows = []
    for record in records:
        left, right = readouts_before[record.id], readouts_after[record.id]
        row = {
            "id": record.id,
            "source_direction_cosine": cosine(before[record.id]["activation"], after[record.id]["activation"]),
        }
        for name, state, readout in (("before", before, left), ("after", after, right)):
            row[name] = {
                "true_cosine": readout["true"]["metrics"]["cosine"] if readout["true"]["metrics"] else None,
                "source_norm": float(state[record.id]["activation"].norm()),
                "answer_mentioned": answer_hit(readout["true"]["explanation"], record.answer),
                "format_valid": readout["true"]["format_valid"],
            }
        if "true" in left["reconstructions"] and "true" in right["reconstructions"]:
            row["stale_before_readout_cosine_on_after_state"] = cosine(
                left["reconstructions"]["true"], after[record.id]["activation"]
            )
            row["after_readout_cosine_on_before_state"] = cosine(
                right["reconstructions"]["true"], before[record.id]["activation"]
            )
            row["reconstructed_direction_cosine"] = cosine(
                left["reconstructions"]["true"], right["reconstructions"]["true"]
            )
        rows.append(row)
    return {
        "rows": rows,
        "interpretation": "Source direction drift, current-state reconstruction fidelity, and stale-readout mismatch are separate diagnostics. AR agreement is not causal evidence of semantic truth.",
    }


def ridge_predict(x_train, y_train, x_test, alpha):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    target_mean = y_train.mean()
    gram = train @ train.T
    weights = np.linalg.solve(gram + alpha * np.eye(len(train)), y_train - target_mean)
    return target_mean + test @ train.T @ weights


def select_alpha(x, y, alphas):
    errors = []
    for alpha in alphas:
        predictions = np.empty_like(y)
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            predictions[i] = ridge_predict(x[keep], y[keep], x[i : i + 1], alpha)[0]
        errors.append(float(np.square(predictions - y).mean()))
    best = int(np.argmin(errors))
    return alphas[best], errors[best]


def forgetting_prediction(records, before, after, prior_readouts, stage, config):
    prediction = config["prediction"]
    selected = [
        r
        for r in records
        if eligible_retention(r, before, stage)
        and all(prior_readouts[r.id][k]["metrics"] is not None for k in ("true", "empty", "shuffled"))
    ]
    train = np.array([r.split == "development" for r in selected], dtype=bool)
    test = ~train
    counts = {"development": int(train.sum()), "heldout": int(test.sum())}
    if counts["development"] < prediction["min_development"] or counts["heldout"] < prediction["min_heldout"]:
        return {"status": "insufficient_previously_correct_scorable_entities", "counts": counts, "scores": None}
    if set(r.entity for r in selected if r.split == "development") & set(
        r.entity for r in selected if r.split == "heldout"
    ):
        raise ValueError("NLA_PREDICTOR_LEAK: entity appears on both sides of the split")
    behavior, semantic, vectors, targets = [], [], [], []
    for record in selected:
        prior = before[record.id]["behavior"]
        readout = prior_readouts[record.id]
        true = readout["true"]["metrics"]["cosine"]
        behavior.append(
            [
                prior["gold_logprob_mean"],
                prior["gold_margin"],
                prior["candidate_entropy"],
                prior["gold_probability_within_candidates"],
            ]
        )
        semantic.append(
            [
                true,
                true - readout["empty"]["metrics"]["cosine"],
                true - readout["shuffled"]["metrics"]["cosine"],
                readout["true"]["generated_tokens"],
                answer_hit(readout["true"]["explanation"], record.answer),
            ]
        )
        vector = before[record.id]["activation"].numpy().astype(np.float64)
        vectors.append(vector / np.linalg.norm(vector))
        targets.append(prior["gold_logprob_mean"] - after[record.id]["behavior"]["gold_logprob_mean"])
    behavior, semantic, vectors, targets = map(np.asarray, (behavior, semantic, vectors, targets))
    if any(not np.isfinite(value).all() for value in (behavior, semantic, vectors, targets)):
        raise ValueError("NLA_PREDICTOR_NONFINITE: invalid feature/target")
    if targets[train].std() < prediction["min_target_std"]:
        return {"status": "insufficient_development_forgetting_variation", "counts": counts, "scores": None}
    rng = np.random.default_rng(config["seed"] + 917)
    projection = rng.normal(size=(vectors.shape[1], prediction["projection_dim"])) / np.sqrt(
        prediction["projection_dim"]
    )
    probe = vectors @ projection
    features = {
        "logprobs": behavior,
        "logprobs_plus_nla": np.concatenate((behavior, semantic), axis=1),
        "logprobs_plus_small_probe": np.concatenate((behavior, probe), axis=1),
        "logprobs_plus_small_probe_plus_nla": np.concatenate((behavior, probe, semantic), axis=1),
    }
    predictions = {"development_mean": np.full(int(test.sum()), targets[train].mean())}
    scores = {}
    for name, values in features.items():
        alpha, cv_error = select_alpha(values[train], targets[train], prediction["ridge_alphas"])
        predictions[name] = ridge_predict(values[train], targets[train], values[test], alpha)
        scores[name] = {"alpha": alpha, "development_loo_mse": cv_error, "feature_count": values.shape[1]}
    permutation = np.random.default_rng(config["seed"] + 919).permutation(targets[train])
    values = features["logprobs_plus_small_probe_plus_nla"]
    alpha, _ = select_alpha(values[train], permutation, prediction["ridge_alphas"])
    predictions["shuffled_development_target_control"] = ridge_predict(values[train], permutation, values[test], alpha)
    for name, values in predictions.items():
        scores.setdefault(name, {})["heldout_mse"] = float(np.square(values - targets[test]).mean())
        scores[name]["heldout_mae"] = float(np.abs(values - targets[test]).mean())
    paired_gains = {}
    rng = np.random.default_rng(config["seed"] + 921)
    for baseline, augmented in (
        ("logprobs", "logprobs_plus_nla"),
        ("logprobs_plus_small_probe", "logprobs_plus_small_probe_plus_nla"),
    ):
        gains = np.square(predictions[baseline] - targets[test]) - np.square(predictions[augmented] - targets[test])
        samples = rng.choice(gains, size=(prediction["bootstrap_samples"], len(gains)), replace=True).mean(axis=1)
        paired_gains[augmented] = {
            "heldout_mse_improvement": float(gains.mean()),
            "entity_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return {
        "status": "measured_exploratory",
        "counts": counts,
        "scores": scores,
        "nla_increment": paired_gains,
        "heldout_predictions": [
            {
                "id": r.id,
                "target_logprob_drop": float(targets[i]),
                **{name: float(predictions[name][j]) for name in predictions},
            }
            for j, (i, r) in enumerate((i, r) for i, r in enumerate(selected) if test[i])
        ],
        "protocol": "All features precede the update. Targets are before-minus-after mean answer logprob. Entity split is fixed before inference. Standardization and ridge alpha selection use development entities only. Every arm uses identical eligible entities and target budgets. The small probe is a fixed random projection of native hidden states followed by ridge regression.",
        "limitation": "Single-seed, single-update forecast calibration is exploratory; it does not establish general predictive value across update distributions.",
    }
