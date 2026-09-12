import math
from collections import Counter
from statistics import mean

from minimum_budget import checked_records
from prospective_analysis import accuracy, learned, per_intent
from prospective_data import balanced_batches, rows_for, seed_for
from protocol import digest, read_json


def alternate_mapping(stream):
    old = stream["old"]
    original = {intent: stream["units"][intent]["learn"][0]["code"] for intent in old}
    alternate = {intent: original[old[(index + 1) % len(old)]] for index, intent in enumerate(old)}
    if len(set(original.values())) != len(old) or any(original[key] == alternate[key] for key in old):
        raise ValueError("HISTORY_CONTROL_ALTERNATE_FIXED_POINT_OR_DUPLICATE")
    return original, alternate


def relabel(rows, mapping, codes):
    return [row | {"code": mapping[row["intent"]], "target": codes[mapping[row["intent"]]]} for row in rows]


def input_schedule(rows, batches):
    return [
        [
            {key: rows[index][key] for key in ("intent", "text_sha256", "source_split", "source_index", "input_ids")}
            for index in batch
        ]
        for batch in batches
    ]


def learning_schedules(source, config, spec, stream):
    schedules = {}
    for kind, filename, budget, seed_name in (
        ("old", "acquisition-updates.json", config["acquisition_updates"], "acquisition"),
        ("new", "forgetting-updates.json", config["forgetting_updates"], "forgetting"),
    ):
        rows = rows_for(stream, stream[kind], "learn")
        trace = read_json(source / filename)
        batches = balanced_batches(rows, budget, spec["optimizer"]["batch_size"], seed_for(stream["seed"], seed_name))
        if (
            [entry["step"] for entry in trace] != list(range(1, budget + 1))
            or [entry["rows"] for entry in trace] != batches
            or any(not math.isfinite(entry[key]) for entry in trace for key in ("loss", "gradient_norm"))
        ):
            raise ValueError(f"HISTORY_CONTROL_HISTORICAL_TRAINING_SCHEDULE kind={kind}")
        schedules[kind] = {
            "rows": rows,
            "batches": batches,
            "input_schedule_sha256": digest(input_schedule(rows, batches)),
            "historical_trace_sha256": digest(trace),
        }
    return schedules


def learning_gate(observations, condition, stream, spec, protocol, codes):
    if condition not in protocol["conditions"] or "initial" not in observations:
        raise ValueError("HISTORY_CONTROL_LEARNING_CONDITION_OR_INITIAL")
    _, alternate = alternate_mapping(stream)
    old_rows = rows_for(stream, stream["old"], "gate")
    own_rows = relabel(old_rows, alternate, codes) if condition == "alternate_binding" else old_rows
    new_rows = rows_for(stream, stream["new"], "gate")
    for phase in observations.values():
        for key, rows in (("old_own", own_rows), ("old_original", old_rows), ("new", new_rows)):
            checked_records(phase[key], rows, codes)
        if any(
            left["output"]["code_logits"] != right["output"]["code_logits"]
            for left, right in zip(phase["old_own"], phase["old_original"], strict=True)
        ):
            raise ValueError("HISTORY_CONTROL_MAPPING_CHANGED_MODEL_PREDICTIONS")
    acquired = []
    old_passed = condition == "new_only"
    before_new = "initial"
    if condition == "alternate_binding":
        if "acquired" not in observations:
            raise ValueError("HISTORY_CONTROL_ALTERNATE_ACQUISITION_MISSING")
        acquired = learned(
            per_intent(observations["initial"]["old_own"]), per_intent(observations["acquired"]["old_own"]), spec
        )
        old_passed = len(acquired) >= protocol["source_qualification"]["minimum_alternate_old_acquired"]
        before_new = "acquired"
    if not old_passed and ("post_stream" in observations or "new_midpoint" in observations):
        raise ValueError("HISTORY_CONTROL_NEW_UPDATES_AFTER_FAILED_OLD_GATE")
    new_acquired = []
    if "post_stream" in observations:
        if "new_midpoint" not in observations:
            raise ValueError("HISTORY_CONTROL_NEW_MIDPOINT_MISSING")
        new_acquired = learned(
            per_intent(observations[before_new]["new"]), per_intent(observations["post_stream"]["new"]), spec
        )
    new_passed = len(new_acquired) >= protocol["source_qualification"]["minimum_new_acquired"]
    return {
        "qualified": old_passed and new_passed,
        "status": "qualified"
        if old_passed and new_passed
        else ("alternate_acquisition_failed" if not old_passed else "new_acquisition_failed"),
        "alternate_acquired": acquired,
        "old_acquisition_required": condition == "alternate_binding",
        "old_source_passed": old_passed,
        "new_acquired": new_acquired,
        "new_source_passed": new_passed,
        "new_baseline_phase": before_new,
        "scores": {
            phase: {key: per_intent(value) for key, value in panel.items()} for phase, panel in observations.items()
        },
        "repair_updates": 0,
        "test_rows_evaluated": 0,
    }


def common_gate(historical, controls, streams, protocol):
    if set(historical) != set(streams) or set(controls) != set(streams):
        raise ValueError("HISTORY_CONTROL_ALL_SOURCES_REQUIRED")
    groups, failures = {}, []
    for name, stream in streams.items():
        if set(controls[name]) != set(protocol["conditions"]):
            raise ValueError("HISTORY_CONTROL_BOTH_CONDITIONS_REQUIRED")
        old = historical[name]["eligible"]
        alternate = set(controls[name]["alternate_binding"]["alternate_acquired"])
        common = [intent for intent in old if intent in alternate]
        failed = [condition for condition, check in controls[name].items() if not check["qualified"]]
        if failed:
            failures.append({"stream": name, "failed_conditions": failed})
        groups[name] = {
            "split": stream["split"],
            "historical_eligible": old,
            "common": common,
            "exclusions": [
                {"intent": intent, "reason": "alternate_own_binding_acquisition_failed"}
                for intent in old
                if intent not in alternate
            ],
            "source_gates_passed": not failed,
            "enough_common": len(common) >= protocol["common_gate"]["minimum_per_stream"],
        }
    counts = {
        split: sum(len(value["common"]) for value in groups.values() if value["split"] == split)
        for split in ("train", "test")
    }
    clusters = {
        split: sum(
            value["source_gates_passed"] and value["enough_common"]
            for value in groups.values()
            if value["split"] == split
        )
        for split in counts
    }
    passed = (
        not failures
        and all(value["enough_common"] for value in groups.values())
        and min(counts.values()) >= protocol["common_gate"]["minimum_per_split"]
        and min(clusters.values()) >= protocol["common_gate"]["source_clusters_per_split"]
    )
    return {
        "qualified": passed,
        "status": "qualified" if passed else "common_eligibility_failed",
        "streams": groups,
        "counts": counts,
        "source_clusters": clusters,
        "source_failures": failures,
        "repair_updates": 0,
        "predictor_fits": 0,
        "decision": "matched_control_repairs_allowed"
        if passed
        else "stop_all_control_repairs; new_TRAIN_protocol_required",
    }


def curve(unit, qualification, budgets):
    baseline = unit["baseline"]
    base_accuracy, base_guard = accuracy(baseline["test"]), accuracy(baseline["guard"])
    panels = {"0": baseline} | unit["actions"]
    if set(panels) != {"0", *(str(budget) for budget in budgets)}:
        raise ValueError("HISTORY_CONTROL_INCOMPLETE_BUDGET_CURVE")
    values = {}
    for budget, panel in panels.items():
        score, guard = accuracy(panel["test"]), accuracy(panel["guard"])
        gain, guard_change = score - base_accuracy, guard - base_guard
        loss = max(0.0, -guard_change)
        qualified = (
            score >= qualification["accuracy_min"]
            and gain >= qualification["gain_min"] - 1e-12
            and loss <= qualification["guard_drop_max"] + 1e-12
        )
        values[budget] = {
            "accuracy": score,
            "gain": gain,
            "guard": guard,
            "guard_change": guard_change,
            "guard_loss": loss,
            "utility": gain - loss,
            "qualified": qualified,
            "accuracy_passed": score >= qualification["accuracy_min"],
            "guard_passed": loss <= qualification["guard_drop_max"] + 1e-12,
            "updates": int(budget),
        }
    qualified_budgets = [budget for budget in budgets if values[str(budget)]["qualified"]]
    return {
        "id": unit["id"],
        "intent": unit["intent"],
        "stream": unit["stream"],
        "split": unit["split"],
        "curve": values,
        "minimum_budget": str(min(qualified_budgets)) if qualified_budgets else "never",
        "qualified_budgets": qualified_budgets,
        "later_qualification_loss": any(
            not values[str(budget)]["qualified"]
            for budget in budgets
            if qualified_budgets and budget > min(qualified_budgets)
        ),
    }


def paired_analysis(all_units, streams, common, protocol):
    conditions = ["original_history", *protocol["conditions"]]
    if set(all_units) != set(streams):
        raise ValueError("HISTORY_CONTROL_ANALYSIS_SOURCE_SET")
    per_source = {}
    metrics = (
        "accuracy",
        "gain",
        "guard",
        "guard_change",
        "guard_loss",
        "utility",
        "qualified",
        "accuracy_passed",
        "guard_passed",
        "updates",
    )
    for name, stream in streams.items():
        if set(all_units[name]) != set(conditions):
            raise ValueError("HISTORY_CONTROL_ANALYSIS_CONDITION_SET")
        expected = [f"{name}/{intent}" for intent in common["streams"][name]["common"]]
        curves = {}
        for condition in conditions:
            units = all_units[name][condition]
            if (
                not expected
                or [unit["id"] for unit in units] != expected
                or any(unit["stream"] != name or unit["split"] != stream["split"] for unit in units)
            ):
                raise ValueError("HISTORY_CONTROL_ANALYSIS_COMMON_PAIRED_SET")
            curves[condition] = [
                curve(unit, protocol["repair"]["qualification"], protocol["repair"]["budgets"]) for unit in units
            ]
        means, pairs = {}, {}
        for condition, values in curves.items():
            means[condition] = {
                budget: {key: mean(unit["curve"][budget][key] for unit in values) for key in metrics}
                for budget in values[0]["curve"]
            }
        for condition in protocol["conditions"]:
            pairs[condition] = {}
            for budget in means[condition]:
                rows = []
                for historical, control in zip(curves["original_history"], curves[condition], strict=True):
                    left, right = historical["curve"][budget], control["curve"][budget]
                    rows.append(
                        {
                            "id": historical["id"],
                            "excess": {key: float(left[key]) - float(right[key]) for key in metrics},
                            "only_historical_qualified": left["qualified"] and not right["qualified"],
                            "only_control_qualified": right["qualified"] and not left["qualified"],
                        }
                    )
                pairs[condition][budget] = {
                    "pairs": rows,
                    "mean_excess": {key: mean(row["excess"][key] for row in rows) for key in metrics},
                    "only_historical_qualified": sum(row["only_historical_qualified"] for row in rows),
                    "only_control_qualified": sum(row["only_control_qualified"] for row in rows),
                }
        per_source[name] = {
            "split": stream["split"],
            "common_intents": len(expected),
            "curves": curves,
            "means": means,
            "paired_historical_excess": pairs,
            "minimum_budget_counts": {
                condition: {
                    label: Counter(unit["minimum_budget"] for unit in values)[label]
                    for label in [*(str(budget) for budget in protocol["repair"]["budgets"]), "never"]
                }
                for condition, values in curves.items()
            },
        }
    aggregate = {}
    for split in ("train", "test"):
        groups = [value for value in per_source.values() if value["split"] == split]
        if len(groups) < protocol["common_gate"]["source_clusters_per_split"]:
            raise ValueError("HISTORY_CONTROL_ANALYSIS_TOO_FEW_CLUSTERS")
        aggregate[split] = {"source_clusters": len(groups), "intents": sum(g["common_intents"] for g in groups)}
        for aggregation in ("equal_source", "pooled_intents_descriptive"):
            weights = [1 if aggregation == "equal_source" else g["common_intents"] for g in groups]
            condition_means = {
                condition: {
                    budget: {
                        key: sum(
                            g["means"][condition][budget][key] * weight
                            for g, weight in zip(groups, weights, strict=True)
                        )
                        / sum(weights)
                        for key in metrics
                    }
                    for budget in groups[0]["means"][condition]
                }
                for condition in conditions
            }
            aggregate[split][aggregation] = {
                "means": condition_means,
                "paired_historical_excess": {
                    condition: {
                        budget: {key: condition_means["original_history"][budget][key] - values[key] for key in metrics}
                        for budget, values in condition_means[condition].items()
                    }
                    for condition in protocol["conditions"]
                },
            }
    return {
        "per_source": per_source,
        "aggregate": aggregate,
        "common_eligibility": common,
        "scope": protocol["scope"],
        "interpretation_limit": protocol["analysis"]["boundaries"],
        "control_qualification_meaning": protocol["repair"]["terminology"],
        "independence": "Two source clusters per split, paired histories within source; intents are not independent models.",
        "predictor_fits": 0,
    }
