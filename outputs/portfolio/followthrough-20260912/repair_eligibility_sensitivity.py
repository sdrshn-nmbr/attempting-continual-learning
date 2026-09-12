import argparse
import copy
import hashlib
import json
import logging
import math
import sys
import unittest
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from budget_prediction_analysis import evaluate, fit_models, forecasts, prediction_gate
from minimum_budget import minimum_outcome, permutation_diagnostics, permutation_plan
from prospective_analysis import FEATURES, feature_vector, ridge_fit

WAVE = Path(__file__).resolve().parent
REPO = WAVE.parents[2]
SOURCE = REPO / "experiments/recoverability"
RUNS = WAVE / "runs"
PREFIX = "followthrough-20260912-recovery-predict-"
ORIGINAL_SOURCE = "99ac3030bb56b44579b07e0f672ce098b0831f9c871b350d087201ad546ba8d1"
DECISION = WAVE / "repair_eligibility_sensitivity.decision.json"
FITS = WAVE / "repair_eligibility_sensitivity.fit.json"
OUTPUT = WAVE / "repair_eligibility_sensitivity.json"
STREAMS = ("train-a", "train-b", "test-a", "test-b")
LOG = logging.getLogger("repair-eligibility-sensitivity")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, detail):
    if not condition:
        raise ValueError(f"REPAIR_ELIGIBILITY_SENSITIVITY {detail}")


def write_sealed(path, payload):
    envelope = {"payload": payload, "sha256": digest(payload)}
    with path.open("x") as handle:
        json.dump(envelope, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return envelope


def read_sealed(path):
    value = read(path)
    require(digest(value["payload"]) == value["sha256"], f"seal hash {path}")
    return value


def compare_tree(left, right, path="root"):
    if isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), f"keys {path}")
        for key in left:
            compare_tree(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), f"length {path}")
        for index, (x, y) in enumerate(zip(left, right, strict=True)):
            compare_tree(x, y, f"{path}/{index}")
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        require(
            math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10),
            f"numeric disagreement {path}: {left} != {right}",
        )
    else:
        require(left == right, f"value disagreement {path}")


class Evidence:
    def __init__(self):
        self.hashes = {}
        self.native_rows = 0
        self.native_panels = 0
        self.executions = {}

    def pin(self, path, expected=None):
        value = file_hash(path)
        if expected is not None:
            require(value == expected, f"file hash {path}")
        self.hashes[str(path.relative_to(REPO))] = value
        return value

    def source(self, filename):
        path = SOURCE / filename
        self.pin(path)
        return read(path)

    def artifact(self, suffix, filename, load=True):
        run = RUNS / (PREFIX + suffix)
        require((run / "collection.json").is_file(), f"collection incomplete {suffix}")
        self.pin(run / "collection.json")
        self.pin(run / "execution.json")
        execution = read(run / "execution.json")
        require(
            execution["status"] == "completed"
            and execution["exit_code"] == 0
            and execution["source_sha256"] == ORIGINAL_SOURCE,
            f"original terminal execution {suffix}",
        )
        self.executions[suffix] = execution
        self.pin(run / "study/receipt.json")
        receipt = read_sealed(run / "study/receipt.json")["payload"]
        path = run / "study" / filename
        self.pin(path, receipt["files"][filename])
        return read(path) if load else None

    def score(self, records, rows, codes):
        require(len(records) == len(rows) and bool(rows), "scoring denominator")
        require(len(codes) == len(set(codes)) == 16, "16-code contract")
        correct = 0
        for record, row in zip(records, rows, strict=True):
            require(
                all(
                    record[key] == row[key]
                    for key in ("intent", "target", "text_sha256")
                ),
                "native scoring row binding",
            )
            logits = record["output"]["code_logits"]
            require(
                len(logits) == 16 and all(math.isfinite(v) for v in logits),
                "native logits",
            )
            predicted = codes[max(range(16), key=logits.__getitem__)]
            require(predicted == record["output"]["prediction"], "native argmax")
            correct += predicted == row["target"]
        self.native_panels += 1
        self.native_rows += len(rows)
        return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def maximum_gain_impossible(correct, total, gain):
    require(0 <= correct <= total and total > 0, "invalid score fraction")
    return Fraction(total - correct, total) < Fraction(str(gain))


def panel_eligibility(unit, scored, initial_matches, spec):
    before, after = scored["before"]["test"], scored["after"]["test"]
    gates = spec["gates"]
    impossible = maximum_gain_impossible(
        after["correct"], after["total"], spec["recovered"]["gain_min"]
    )
    failures = []
    checks = {
        "acquired_floor": before["accuracy"] >= gates["acquired_min"],
        "forgotten_after_max": after["accuracy"] <= gates["after_max"],
        "forgetting_drop": before["accuracy"] - after["accuracy"]
        >= gates["drop_min"] - 1e-12,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    require(
        not initial_matches,
        "unexpected initial actual-panel evidence; decision requires review",
    )
    return {
        "id": unit["id"],
        "intent": unit["intent"],
        "stream": unit["stream"],
        "split": unit["split"],
        "actual_repair_panel": {"acquired": before, "pre_repair": after},
        "maximum_possible_gain": {
            "numerator": after["total"] - after["correct"],
            "denominator": after["total"],
            "value": 1 - after["accuracy"],
        },
        "primary_excluded_impossible_gain": impossible,
        "primary_retained": not impossible,
        "diagnostic_checks": checks,
        "diagnostic_known_failures": failures,
        "actual_initial_panel": {
            "measured_rows": 0,
            "required_rows": after["total"],
            "acquisition_gain_passed": None,
            "status": "not_recorded; validation rows cannot supply actual-panel initial scores",
        },
        "fully_eligible_actual_panel": False if failures else None,
        "diagnostic_filter_used_for_refit": False,
    }


def load_source_contract(evidence):
    interface = evidence.source("budget-prediction-interface.json")
    for name, expected in interface["required_file_sha256"].items():
        evidence.pin(SOURCE / name, expected)
    protocol = evidence.source("budget-prediction-protocol.json")
    spec = evidence.source("budget-prediction-learning.json")
    data = evidence.source("inputs/budget-prediction-cohort.json")
    require(
        file_hash(SOURCE / "inputs/budget-prediction-cohort.json")
        == protocol["dataset_sha256"],
        "dataset",
    )
    require(
        file_hash(SOURCE / "budget-prediction-learning.json")
        == protocol["learning"]["spec_sha256"],
        "spec",
    )
    require(protocol["recovered"] == spec["recovered"], "unchanged recovery thresholds")
    require(
        protocol["ridge_alpha"] == 10 and protocol["budgets"] == [2, 4, 8, 16],
        "fixed recipe",
    )
    require(protocol["features"] == list(FEATURES), "fixed feature models")
    return protocol, spec, data


def load_units(evidence, suffix, spec, data):
    units = evidence.artifact("repair-" + suffix, "units.json")
    readouts = evidence.artifact("readout-" + suffix, "readouts.json")
    require(len(units) == len(readouts), "readout/repair count")
    stream = data["streams"]["fresh-" + suffix]
    require(
        [u["id"] for u in units] == [u["id"] for u in readouts],
        "readout/repair ordering",
    )
    guard_rows = [
        row for name in stream["new"] for row in stream["units"][name]["test"]
    ]
    scores = {}
    for unit, readout in zip(units, readouts, strict=True):
        require(
            unit["stream"] == stream["id"] and unit["split"] == stream["split"],
            "unit source",
        )
        for key in (
            "id",
            "intent",
            "stream",
            "split",
            "baseline",
            "probes",
            "features",
        ):
            require(
                unit[key] == readout[key],
                f"frozen pre-repair readout {unit['id']}/{key}",
            )
        require(not readout["actions"], "readouts precede actions")
        compare_tree(
            feature_vector(
                unit["probes"]["before"], unit["probes"]["after"], spec["layers"]
            ),
            unit["features"],
        )
        target = stream["units"][unit["intent"]]
        scores[unit["id"]] = {}
        for phase, panel in unit["baseline"].items():
            scores[unit["id"]][phase] = {
                "test": evidence.score(panel["test"], target["test"], data["codes"]),
                "guard": evidence.score(panel["guard"], guard_rows, data["codes"]),
            }
            evidence.score(unit["probes"][phase], target["probe"], data["codes"])
        for budget, panel in unit["actions"].items():
            scores[unit["id"]][budget] = {
                "test": evidence.score(panel["test"], target["test"], data["codes"]),
                "guard": evidence.score(panel["guard"], guard_rows, data["codes"]),
            }
    return units, scores


def training_inputs(evidence, spec, data):
    units, scores, eligibility = [], {}, []
    for suffix in STREAMS[:2]:
        rows, scored = load_units(evidence, suffix, spec, data)
        observations = evidence.artifact("learn-" + suffix, "observations.json")
        stream = data["streams"]["fresh-" + suffix]
        for unit in rows:
            actual_keys = {
                r["text_sha256"] for r in stream["units"][unit["intent"]]["test"]
            }
            initial_matches = [
                r
                for r in observations["initial"]["old_gate"]
                if r["text_sha256"] in actual_keys
            ]
            result = panel_eligibility(unit, scored[unit["id"]], initial_matches, spec)
            result["validation_gate_scores"] = {
                phase: evidence.score(
                    [
                        r
                        for r in observations[phase]["old_gate"]
                        if r["intent"] == unit["intent"]
                    ],
                    stream["units"][unit["intent"]]["gate"],
                    data["codes"],
                )
                for phase in ("initial", "acquired", "forgotten")
            }
            eligibility.append(result)
        units.extend(rows)
        scores.update(scored)
    require(
        len(units) == 13 and len({u["id"] for u in units}) == 13,
        "original TRAIN denominator",
    )
    return units, scores, eligibility


def retained_units(units, eligibility):
    require(
        [u["id"] for u in units] == [u["id"] for u in eligibility], "eligibility order"
    )
    return [
        unit
        for unit, decision in zip(units, eligibility, strict=True)
        if decision["primary_retained"]
    ]


def check_plan(plan, units, protocol):
    require(
        plan["unit_ids"] == [u["id"] for u in units], "permutation retained identities"
    )
    require(
        plan["source_streams"] == [u["stream"] for u in units], "permutation sources"
    )
    expected = [
        (scheme, seed)
        for scheme in protocol["label_permutations"]["schemes"]
        for seed in protocol["label_permutations"]["seeds"]
    ]
    require(
        [(d["scheme"], d["seed"]) for d in plan["draws"]] == expected,
        "fixed forty draws",
    )
    for draw in plan["draws"]:
        require(
            sorted(draw["indices"]) == list(range(len(units))), "permutation bijection"
        )
        if draw["scheme"] == "within_source":
            require(
                all(
                    units[i]["stream"] == units[j]["stream"]
                    for i, j in enumerate(draw["indices"])
                ),
                "permutation crossed source",
            )
    require(plan["redraws"] == 0, "no permutation redraws")


def seal_decision():
    require(
        not DECISION.exists() and not FITS.exists() and not OUTPUT.exists(),
        "analysis already initialized",
    )
    evidence = Evidence()
    protocol, spec, data = load_source_contract(evidence)
    units, _, eligibility = training_inputs(evidence, spec, data)
    retained = retained_units(units, eligibility)
    plan = permutation_plan(
        [u["id"] for u in retained],
        [u["stream"] for u in retained],
        protocol["label_permutations"]["seeds"],
        protocol["label_permutations"]["schemes"],
    )
    check_plan(plan, retained, protocol)
    for suffix in STREAMS[2:]:
        evidence.artifact("repair-" + suffix, "units.json", load=False)
        evidence.artifact("readout-" + suffix, "readouts.json", load=False)
    for suffix, filename in (
        ("fit", "predictors.json"),
        ("forecast", "forecasts.json"),
        ("analyze", "analysis.json"),
    ):
        evidence.artifact(suffix, filename, load=False)
    gate = prediction_gate(
        [minimum_outcome(u, spec, protocol["budgets"]) for u in retained], protocol
    )
    payload = {
        "name": "one-posthoc-TRAIN-impossible-gain-sensitivity",
        "sealed_at": now(),
        "scope": "Posthoc sensitivity after original TEST outcomes were observed. Not prospective, independently held-out replication, or a revised original study.",
        "authorization": "User explicitly requests one TRAIN-only impossible-gain exclusion and fixed-recipe posthoc refit against unchanged 13 TEST units.",
        "primary_rule": {
            "exclude": "1 - actual pre-repair repair-evaluation-panel accuracy < required gain",
            "split": "train",
            "required_gain": spec["recovered"]["gain_min"],
            "arithmetic": "Exact integer correct/total fractions; strict inequality. Equality remains eligible.",
            "unit_name_used_in_selection": False,
        },
        "secondary_eligibility": "Report actual acquired-panel floor, actual after maximum, actual before-after drop. Initial actual-panel scores are absent, so actual acquisition gain is unknown. These diagnostics add no exclusions.",
        "training_eligibility": eligibility,
        "retained_training_units": [u["id"] for u in retained],
        "excluded_training_units": [
            r["id"] for r in eligibility if not r["primary_retained"]
        ],
        "original_training_units_sha256": digest(units),
        "retained_training_units_sha256": digest(retained),
        "fixed_recipe": {
            key: protocol[key]
            for key in (
                "budgets",
                "features",
                "ridge_alpha",
                "class_rule",
                "policy_rule",
                "predictor",
                "recovered",
                "label_permutations",
                "train_gate",
            )
        },
        "prospective_class_gate_after_exclusion": gate,
        "posthoc_gate_boundary": "Report the unchanged prospective class-support gate even when failed. This user-authorized diagnostic uses its frozen ridge, targets, scalers and class rule without the prospective dispatch gate. No new TEST training or prospective authorization follows.",
        "permutation_plan": plan,
        "permutation_plan_sha256": digest(plan),
        "permutation_generation_count": 1,
        "fits_so_far": 0,
        "refit_variants_authorized": 1,
        "hyperparameter_searches_authorized": 0,
        "baselines": ["always/2", "always/8", "always/16", "constant_train"],
        "evaluation": "All fixed models and all 40 nulls. Joint qualification, updates, old accuracy, new guard, utility, class Brier and accuracy, per-budget recovery Brier and utility MSE/MAE. Per-source, pooled and equal-source; no policy selection, significance tests, intent bootstrap, or permutation redraws.",
        "interpretation": "A sensitivity benefit would require new prospective data before a prediction benefit claim. Thirteen TEST intents comprise two source clusters; those sources and outcomes were already observed.",
        "original_source_sha256": ORIGINAL_SOURCE,
        "input_hashes": evidence.hashes,
        "numpy_at_seal": np.__version__,
    }
    result = write_sealed(DECISION, payload)
    LOG.info(
        "DECISION_SEALED sha=%s retained=%d excluded=%s gate=%s fits=0",
        result["sha256"],
        len(retained),
        payload["excluded_training_units"],
        gate["qualified"],
    )


def refit_recipe(units, spec, protocol, plan):
    require(units and all(u["split"] == "train" for u in units), "refit TRAIN only")
    check_plan(plan, units, protocol)
    outcomes = [minimum_outcome(u, spec, protocol["budgets"]) for u in units]
    gate = prediction_gate(outcomes, protocol)
    classes = [str(b) for b in protocol["budgets"]] + ["never"]
    targets = {
        f"class/{label}": [float(u["minimum_budget"] == label) for u in outcomes]
        for label in classes
    }
    targets.update(
        {
            f"{label}/{budget}": [
                float(u["actions"][str(budget)][label]) for u in outcomes
            ]
            for budget in protocol["budgets"]
            for label in ("recovered", "utility")
        }
    )
    models = {
        "constant_train": {
            "constants": {key: float(np.mean(value)) for key, value in targets.items()}
        }
    }
    for name, feature, indices in [
        (feature, feature, list(range(len(units)))) for feature in FEATURES
    ] + [
        (f"permuted/{d['scheme']}/{d['seed']}", "tuned", d["indices"])
        for d in plan["draws"]
    ]:
        x = [u["features"][feature] for u in units]
        require(np.isfinite(x).all(), "finite fit features")
        models[name] = {
            "feature": feature,
            "targets": {
                key: ridge_fit(x, np.asarray(y)[indices], protocol["ridge_alpha"])
                for key, y in targets.items()
            },
        }
    permutations = permutation_diagnostics(plan, outcomes, protocol["budgets"])
    permutations.pop("predictor_fits")
    permutations["labels"].update(
        {key: value for key, value in targets.items() if key.startswith("class/")}
    )
    for draw in permutations["draws"]:
        for target, values in targets.items():
            if target.startswith("class/"):
                changed = sum(
                    v != values[draw["indices"][i]] for i, v in enumerate(values)
                )
                draw["changed_labels"][target] = changed
                if not changed:
                    draw["inert_targets"].append(target)
    permutations["activity_by_scheme"] = {
        scheme: {
            target: {
                "active_draws": sum(
                    d["changed_labels"][target] > 0
                    for d in permutations["draws"]
                    if d["scheme"] == scheme
                ),
                "inert_draws": sum(
                    d["changed_labels"][target] == 0
                    for d in permutations["draws"]
                    if d["scheme"] == scheme
                ),
            }
            for target in permutations["labels"]
        }
        for scheme in protocol["label_permutations"]["schemes"]
    }
    return {
        "models": models,
        "classes": classes,
        "training_units": [u["id"] for u in units],
        "training_intents": [u["intent"] for u in units],
        "training_source_streams": sorted({u["stream"] for u in units}),
        "training_data_sha256": digest(units),
        "ridge_alpha": protocol["ridge_alpha"],
        "gate": gate,
        "permutations": permutations,
        "label_variation": {
            key: len(set(values)) > 1 for key, values in targets.items()
        },
        "training_unit_weight": "Equal per intent; all evaluation is also aggregated by independent source stream.",
    }


def policy_counts(result, raw_units):
    groups = {"pooled": raw_units} | {
        name: [u for u in raw_units if u["stream"] == name]
        for name in sorted({u["stream"] for u in raw_units})
    }
    outcomes = {u["id"]: u for u in result["units"]}
    report = {}
    for name, policy in result["policies"].items():
        choices = dict(
            zip([u["id"] for u in raw_units], policy["choices"], strict=True)
        )
        report[name] = {}
        for group, units in groups.items():
            rows = []
            for unit in units:
                choice = choices[unit["id"]]
                panel = (
                    unit["baseline"]["after"]
                    if choice == "never"
                    else unit["actions"][choice]
                )
                old = sum(
                    r["output"]["prediction"] == r["target"] for r in panel["test"]
                )
                guard = sum(
                    r["output"]["prediction"] == r["target"] for r in panel["guard"]
                )
                rows.append(
                    {
                        "id": unit["id"],
                        "choice": choice,
                        "qualified": choice != "never"
                        and outcomes[unit["id"]]["actions"][choice]["recovered"],
                        "updates": 0 if choice == "never" else int(choice),
                        "old_correct": old,
                        "old_total": len(panel["test"]),
                        "guard_correct": guard,
                        "guard_total": len(panel["guard"]),
                    }
                )
            report[name][group] = {
                "units": len(rows),
                "qualified": sum(r["qualified"] for r in rows),
                "total_updates": sum(r["updates"] for r in rows),
                "old_correct": sum(r["old_correct"] for r in rows),
                "old_total": sum(r["old_total"] for r in rows),
                "guard_correct": sum(r["guard_correct"] for r in rows),
                "guard_total": sum(r["guard_total"] for r in rows),
                "paired_cases": rows,
            }
    return report


def verify_unchanged(hashes):
    for relative, expected in hashes.items():
        require(
            file_hash(REPO / relative) == expected,
            f"immutable input changed {relative}",
        )


def run_sensitivity():
    require(
        not FITS.exists() and not OUTPUT.exists(),
        "one fit only; existing fit/result preserved",
    )
    sealed = read_sealed(DECISION)
    decision = sealed["payload"]
    verify_unchanged(decision["input_hashes"])
    evidence = Evidence()
    protocol, spec, data = load_source_contract(evidence)
    training, training_scores, eligibility = training_inputs(evidence, spec, data)
    require(
        eligibility == decision["training_eligibility"],
        "sealed eligibility recomputation",
    )
    require(
        digest(training) == decision["original_training_units_sha256"],
        "original TRAIN unchanged",
    )
    retained = retained_units(training, eligibility)
    require(
        digest(retained) == decision["retained_training_units_sha256"],
        "retained TRAIN unchanged",
    )
    plan = decision["permutation_plan"]
    require(
        digest(plan) == decision["permutation_plan_sha256"], "sealed permutation plan"
    )
    original_fitted = evidence.artifact("fit", "predictors.json")
    require(
        original_fitted["training_units"] == [u["id"] for u in training],
        "original TRAIN identity",
    )
    require(
        original_fitted["training_data_sha256"] == digest(training),
        "original TRAIN content",
    )
    fit_started = now()
    require(
        datetime.fromisoformat(decision["sealed_at"])
        < datetime.fromisoformat(fit_started),
        "decision before fit",
    )
    LOG.info(
        "POSTHOC_REFIT_START units=%d variants=1 alpha=%s nulls=40",
        len(retained),
        protocol["ridge_alpha"],
    )
    fitted = refit_recipe(retained, spec, protocol, plan)
    fit_artifact = write_sealed(
        FITS,
        {
            "decision_sha256": sealed["sha256"],
            "script_sha256": file_hash(Path(__file__)),
            "started_at": fit_started,
            "finished_at": now(),
            "refit_variants": 1,
            "scalar_ridge_fits": sum(
                len(m.get("targets", {})) for m in fitted["models"].values()
            ),
            "numpy": np.__version__,
            "fitted": fitted,
        },
    )
    test_units, test_scores = [], {}
    for suffix in STREAMS[2:]:
        units, scores = load_units(evidence, suffix, spec, data)
        test_units.extend(units)
        test_scores.update(scores)
    require(len(test_units) == 13, "unchanged 13 TEST units")
    original_forecast = evidence.artifact("forecast", "forecasts.json")
    original_analysis = evidence.artifact("analyze", "analysis.json")
    require(
        [u["id"] for u in test_units] == original_forecast["unit_ids"],
        "unchanged TEST identity/order",
    )
    compare_tree(forecasts(original_fitted, test_units), original_forecast)
    compare_tree(
        [minimum_outcome(u, spec, protocol["budgets"]) for u in test_units],
        original_analysis["units"],
    )
    predicted = forecasts(fitted, test_units)
    result = evaluate(fitted, predicted, test_units, spec, protocol)
    comparisons = {}
    for name, policy in result["policies"].items():
        references = {
            reference: result["policies"][reference]
            for reference in decision["baselines"]
        }
        references["original_prospective_same_policy"] = original_analysis["policies"][
            name
        ]
        comparisons[name] = {
            reference: {
                metric: {
                    "pooled": policy[metric]["pooled"] - baseline[metric]["pooled"],
                    "equal_source_mean": policy[metric]["equal_source_mean"]
                    - baseline[metric]["equal_source_mean"],
                    "by_source": {
                        source: policy[metric]["by_source"][source]
                        - baseline[metric]["by_source"][source]
                        for source in policy[metric]["by_source"]
                    },
                }
                for metric in ("qualified_fraction", "mean_updates", "utility")
            }
            for reference, baseline in references.items()
        }
    changed_choices = {
        name: [
            {"id": unit["id"], "original": old, "posthoc": new}
            for unit, old, new in zip(
                test_units,
                original_analysis["policies"][name]["choices"],
                policy["choices"],
                strict=True,
            )
            if old != new
        ]
        for name, policy in result["policies"].items()
    }
    verify_unchanged(decision["input_hashes"])
    verify_unchanged(evidence.hashes)
    payload = {
        "status": "completed_posthoc_sensitivity",
        "finished_at": now(),
        "decision": {
            "path": str(DECISION.relative_to(REPO)),
            "file_sha256": file_hash(DECISION),
            "payload_sha256": sealed["sha256"],
            "sealed_at": decision["sealed_at"],
        },
        "fit_artifact": {
            "path": str(FITS.relative_to(REPO)),
            "file_sha256": file_hash(FITS),
            "payload_sha256": fit_artifact["sha256"],
        },
        "implementation": {
            "script": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": file_hash(Path(__file__)),
            "numpy": np.__version__,
            "python": sys.version,
        },
        "original_results_unchanged": True,
        "original_source_sha256": ORIGINAL_SOURCE,
        "refit_variants": 1,
        "hyperparameter_searches": 0,
        "permutation_redraws": 0,
        "new_gpu_updates": 0,
        "training_eligibility": eligibility,
        "excluded_training_units": decision["excluded_training_units"],
        "retained_training_units": decision["retained_training_units"],
        "prospective_gate_after_exclusion": fitted["gate"],
        "posthoc_gate_boundary": decision["posthoc_gate_boundary"],
        "scoring_denominators": {
            "train_original": len(training),
            "train_retained": len(retained),
            "test_unchanged": len(test_units),
            "train_sources": dict(Counter(u["stream"] for u in retained)),
            "test_sources": dict(Counter(u["stream"] for u in test_units)),
            "training_panels": training_scores,
            "test_panels": test_scores,
        },
        "input_hashes": evidence.hashes,
        "original_prospective": {
            "analysis_sha256": file_hash(
                RUNS / (PREFIX + "analyze") / "study/analysis.json"
            ),
            "forecast_sha256": file_hash(
                RUNS / (PREFIX + "forecast") / "study/forecasts.json"
            ),
            "policy_report": original_analysis["policy_report"],
            "metrics": original_analysis["metrics"],
            "train_class_counts": original_analysis["train_class_counts"],
            "permutation_activity": original_fitted["permutations"][
                "activity_by_scheme"
            ],
        },
        "posthoc_forecasts": predicted,
        "posthoc_forecasts_sha256": digest(predicted),
        "posthoc_evaluation": result,
        "posthoc_evaluation_sha256": digest(result),
        "posthoc_policy_counts": policy_counts(result, test_units),
        "paired_policy_differences": comparisons,
        "changed_policy_choices": changed_choices,
        "permutations": fitted["permutations"],
        "verification": {
            "decision_before_fit": True,
            "original_forecasts_recomputed_from_original_saved_coefficients": True,
            "original_test_outcomes_recomputed": True,
            "all_consumed_sources_and_original_results_unchanged": True,
            "raw_native_panels": evidence.native_panels,
            "raw_native_rows": evidence.native_rows,
            "scoring": "Argmax across the same 16 token codes; row intent/target/text hashes bound to frozen dataset. Each score retains integer numerator and denominator.",
        },
        "interpretation": decision["scope"] + " " + decision["interpretation"],
    }
    output = write_sealed(OUTPUT, payload)
    LOG.info("POSTHOC_COMPLETE sha=%s output=%s", output["sha256"], OUTPUT)
    for name in ("constant_train", *FEATURES, "always/2", "always/8", "always/16"):
        counts = payload["posthoc_policy_counts"][name]["pooled"]
        LOG.info(
            "POLICY %s qualified=%d/%d updates=%d equal_source=%s",
            name,
            counts["qualified"],
            counts["units"],
            counts["total_updates"],
            result["policy_report"][name]["equal_source_mean"],
        )


class Controls(unittest.TestCase):
    def test_strict_fraction_boundary(self):
        self.assertFalse(maximum_gain_impossible(14, 20, 0.3))
        self.assertTrue(maximum_gain_impossible(15, 20, 0.3))
        self.assertTrue(maximum_gain_impossible(18, 20, 0.3))
        self.assertFalse(maximum_gain_impossible(0, 20, 0.3))

    def test_score_rejects_wrong_binding(self):
        evidence = Evidence()
        record = {
            "intent": "a",
            "target": 0,
            "text_sha256": "x",
            "output": {"code_logits": [1] + [0] * 15, "prediction": 0},
        }
        with self.assertRaisesRegex(ValueError, "row binding"):
            evidence.score(
                [record],
                [{"intent": "b", "target": 0, "text_sha256": "x"}],
                list(range(16)),
            )

    def test_score_recomputes_argmax_and_denominator(self):
        evidence = Evidence()
        record = {
            "intent": "a",
            "target": 1,
            "text_sha256": "x",
            "output": {"code_logits": [1] + [0] * 15, "prediction": 0},
        }
        self.assertEqual(
            evidence.score([record], [record], list(range(16))),
            {"correct": 0, "total": 1, "accuracy": 0},
        )
        record["output"]["prediction"] = 1
        with self.assertRaisesRegex(ValueError, "argmax"):
            evidence.score([record], [record], list(range(16)))

    def test_secondary_failure_does_not_change_primary_filter(self):
        spec = {
            "gates": {"acquired_min": 0.8, "after_max": 0.5, "drop_min": 0.3},
            "recovered": {"gain_min": 0.3},
        }
        unit = {
            "id": "train/arbitrary",
            "intent": "arbitrary",
            "stream": "train",
            "split": "train",
        }
        scores = {
            "before": {"test": {"correct": 10, "total": 20, "accuracy": 0.5}},
            "after": {"test": {"correct": 0, "total": 20, "accuracy": 0}},
        }
        value = panel_eligibility(unit, scores, [], spec)
        self.assertTrue(value["primary_retained"])
        self.assertEqual(value["diagnostic_known_failures"], ["acquired_floor"])
        self.assertIsNone(value["actual_initial_panel"]["acquisition_gain_passed"])

    def test_refit_exact_frozen_recipe_on_synthetic_data(self):
        protocol = read(SOURCE / "budget-prediction-protocol.json")
        spec = read(SOURCE / "budget-prediction-learning.json")
        units = []
        for index in range(12):
            source = protocol["train_streams"][index // 6]
            good = {"target": 1, "output": {"prediction": 1}}
            bad = {"target": 1, "output": {"prediction": 0}}
            panel = {"test": [bad], "guard": [good]}
            actions = {
                str(b): {
                    "test": [good if index % 2 == 0 or b >= 8 else bad],
                    "guard": [good],
                }
                for b in protocol["budgets"]
            }
            units.append(
                {
                    "id": f"{source}/synthetic-{index}",
                    "intent": f"synthetic-{index}",
                    "stream": source,
                    "split": "train",
                    "baseline": {
                        "before": {"test": [good], "guard": [bad]},
                        "after": panel,
                    },
                    "actions": actions,
                    "features": {
                        feature: [float(index), float(index % 3), 1.0]
                        for feature in FEATURES
                    },
                }
            )
        plan = permutation_plan(
            [u["id"] for u in units],
            [u["stream"] for u in units],
            protocol["label_permutations"]["seeds"],
            protocol["label_permutations"]["schemes"],
        )
        self.assertEqual(
            refit_recipe(units, spec, protocol, plan),
            fit_models(units, spec, protocol, plan),
        )
        polluted = copy.deepcopy(units)
        polluted[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            refit_recipe(polluted, spec, protocol, plan)
        broken = copy.deepcopy(plan)
        broken["draws"][0]["indices"][0] = 11
        with self.assertRaisesRegex(ValueError, "bijection|crossed source"):
            check_plan(broken, units, protocol)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "self-test", "run"))
    args = parser.parse_args()
    if args.mode == "seal":
        seal_decision()
    elif args.mode == "self-test":
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
        )
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        run_sensitivity()


if __name__ == "__main__":
    main()
