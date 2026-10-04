"""Step 4: three sanity checks on the τ-banking token scores before reading them as results. The pass criteria are
fixed here, before looking at the numbers. Confidence intervals resample whole tasks (BOOTSTRAP draws, fixed seed).
  1 documents_only values: full beats none. Pass if the 95% intervals of the mean per-value log-probability gain and
    of the greedy exact-match gain (full minus none) both lie above zero.
  2 copy values: every condition stays close to full. Pass if, for each condition, the 95% interval of its greedy
    exact-match difference from full lies inside +-COPY_TOLERANCE points.
  3 agreement with real runs: across tasks, the full-documents greedy exact-match rate rises with Qwen3-4B's action
    recall in the τ-Knowledge pilot's gold-documents runs. Pass if the Spearman correlation's 95% interval lies above
    zero. (Per-task pass rates are too rare at Qwen3-4B's 5.4% to carry this comparison.)

  uv run --no-project --with numpy --with scipy python sanity.py --scores results/scores.jsonl \
      --pilot-gold <pilot>/qwen3-4b-instruct-2507-golden_retrieval/results.json \
      --pilot-none <pilot>/qwen3-4b-instruct-2507-no_knowledge/results.json --out results/sanity.json
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BOOTSTRAP = 4000
SEED = 0
COPY_TOLERANCE = 5.0


def log(message):
    print(f"[sanity {time.strftime('%H:%M:%S')}] {message}", flush=True)


def values_by_condition(path):
    """{condition: {(decision_id, index): (task_id, label, mean_logprob, greedy_exact)}}"""
    table = defaultdict(dict)
    with path.open() as handle:
        for record in map(json.loads, handle):
            for index, value in enumerate(record["values"]):
                table[record["condition"]][(record["decision_id"], index)] = (
                    record["task_id"], value["label"], value["mean_logprob"], float(value["greedy_exact"]))
    return table


def interval(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def clustered_mean(rows, rng):
    """Mean of rows[:, 1] with a 95% interval from resampling the tasks in rows[:, 0]."""
    tasks = sorted({task for task, _ in rows})
    by_task = {task: np.array([x for t, x in rows if t == task]) for task in tasks}
    sums = np.array([by_task[t].sum() for t in tasks])
    counts = np.array([len(by_task[t]) for t in tasks])
    draws = rng.integers(0, len(tasks), size=(BOOTSTRAP, len(tasks)))
    return float(sums.sum() / counts.sum()), interval(sums[draws].sum(1) / counts[draws].sum(1))


def paired(table, condition, reference, label, field, rng, scale=1.0):
    keys = [k for k, v in table[reference].items() if v[1] == label]
    rows = [(table[reference][k][0], scale * (table[condition][k][field] - table[reference][k][field])) for k in keys]
    mean, ci = clustered_mean(rows, rng)
    return {"mean": round(mean, 4), "ci95": ci, "values": len(rows), "tasks": len({t for t, _ in rows})}


def rate(table, condition, label, field, scale=1.0):
    rows = [v[field] for v in table[condition].values() if v[1] == label]
    return round(scale * float(np.mean(rows)), 4)


def pilot_tasks(path):
    data = json.loads(path.read_text())
    recall, passed = defaultdict(list), defaultdict(list)
    for simulation in data["simulations"]:
        info = simulation.get("reward_info") or {}
        checks = info.get("action_checks") or []
        if checks:
            recall[simulation["task_id"]].append(sum(c["action_match"] for c in checks) / len(checks))
        reward = info.get("reward")
        passed[simulation["task_id"]].append(reward is not None and abs(reward - 1.0) <= 1e-6)
    return ({t: float(np.mean(v)) for t, v in recall.items()}, {t: float(np.mean(v)) for t, v in passed.items()},
            data["info"]["agent_info"]["llm"], data["info"].get("git_commit"))


def task_scores(table, condition, field):
    rows = defaultdict(list)
    for task, _, logprob, exact in table[condition].values():
        rows[task].append((logprob, exact)[field - 2])
    return {task: float(np.mean(v)) for task, v in rows.items()}


def correlation(x_by_task, y_by_task, rng):
    tasks = sorted(set(x_by_task) & set(y_by_task))
    x, y = np.array([x_by_task[t] for t in tasks]), np.array([y_by_task[t] for t in tasks])
    rho = spearmanr(x, y).statistic
    draws = rng.integers(0, len(tasks), size=(BOOTSTRAP, len(tasks)))
    boot = np.array([spearmanr(x[d], y[d]).statistic for d in draws])
    return {"spearman": round(float(rho), 4), "ci95": interval(boot[np.isfinite(boot)]), "tasks": len(tasks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--pilot-gold", type=Path, required=True)
    parser.add_argument("--pilot-none", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    table = values_by_condition(args.scores)
    others = [c for c in table if c != "full"]
    exact, logprob = 3, 2

    gain_logprob = paired(table, "full", "none", "documents_only", logprob, rng)
    gain_exact = paired(table, "full", "none", "documents_only", exact, rng, scale=100)
    check1 = {"criterion": "95% intervals of full-minus-none log-probability and exact-match gains above zero",
              "exact_rate_points": {c: rate(table, c, "documents_only", exact, 100) for c in table},
              "mean_logprob": {c: rate(table, c, "documents_only", logprob) for c in table},
              "gain_logprob": gain_logprob, "gain_exact_points": gain_exact,
              "passed": gain_logprob["ci95"][0] > 0 and gain_exact["ci95"][0] > 0}

    differences = {c: paired(table, c, "full", "copy", exact, rng, scale=100) for c in others}
    check2 = {"criterion": f"each condition's exact-match difference from full inside +-{COPY_TOLERANCE} points (95%)",
              "exact_rate_points": {c: rate(table, c, "copy", exact, 100) for c in table},
              "mean_logprob": {c: rate(table, c, "copy", logprob) for c in table},
              "difference_from_full_points": differences,
              "difference_from_full_logprob": {c: paired(table, c, "full", "copy", logprob, rng) for c in others},
              "within_tolerance": {c: -COPY_TOLERANCE <= d["ci95"][0] and d["ci95"][1] <= COPY_TOLERANCE
                                   for c, d in differences.items()}}
    check2["passed"] = all(check2["within_tolerance"].values())

    gold_recall, gold_pass, agent, commit = pilot_tasks(args.pilot_gold)
    none_recall, none_pass, _, _ = pilot_tasks(args.pilot_none)
    full_exact, none_exact = task_scores(table, "full", exact), task_scores(table, "none", exact)
    primary = correlation(full_exact, gold_recall, rng)
    gains = {t: full_exact[t] - none_exact[t] for t in full_exact}
    pilot_gains = {t: gold_recall[t] - none_recall[t] for t in gold_recall if t in none_recall}
    check3 = {"criterion": "Spearman(full exact-match rate, pilot gold action recall) 95% interval above zero",
              "pilot_agent": agent, "pilot_tau2_commit": commit,
              "full_exact_vs_gold_recall": primary,
              "full_logprob_vs_gold_recall": correlation(task_scores(table, "full", logprob), gold_recall, rng),
              "full_exact_vs_gold_pass_rate": correlation(full_exact, gold_pass, rng),
              "token_gain_vs_pilot_recall_gain": correlation(gains, pilot_gains, rng),
              "pilot_mean_recall": {"gold": round(float(np.mean(list(gold_recall.values()))), 4),
                                    "none": round(float(np.mean(list(none_recall.values()))), 4)},
              "pilot_tasks_passed_at_least_once": {"gold": sum(v > 0 for v in gold_pass.values()),
                                                   "none": sum(v > 0 for v in none_pass.values())},
              "passed": primary["ci95"][0] > 0}

    report = {"bootstrap": BOOTSTRAP, "seed": SEED,
              "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
              "pilot_gold_sha256": hashlib.sha256(args.pilot_gold.read_bytes()).hexdigest(),
              "pilot_none_sha256": hashlib.sha256(args.pilot_none.read_bytes()).hexdigest(),
              "check1_documents_only_full_beats_none": check1, "check2_copy_values_unchanged": check2,
              "check3_agreement_with_pilot": check3}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"check 1 {'PASS' if check1['passed'] else 'FAIL'}, check 2 {'PASS' if check2['passed'] else 'FAIL'}, "
        f"check 3 {'PASS' if check3['passed'] else 'FAIL'} -> {args.out}")


if __name__ == "__main__":
    main()
