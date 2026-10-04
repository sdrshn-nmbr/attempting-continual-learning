"""Step 5: the τ-banking token-level results on Qwen3-4B-Instruct-2507. For each value label and condition: greedy
exact-match rate and mean token log-probability. For documents_only values, the share of the gap between no documents
and full documents that each condition closes, in exact match and in TRACE's log-ratio form (d = EPSILON - mean token
log-probability; share = sum log(d_none / d) / sum log(d_none / d_full)). STILL appears per seed and as "still", the
per-value average of the three seeds. Intervals resample whole tasks. Also: documents_only exact match by compression
(prefix tokens / 164 slots).

  uv run --no-project --with numpy --with scipy python report.py --scores results/scores.jsonl --out results/report.json
"""
import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from sanity import BOOTSTRAP, SEED, clustered_mean, interval

EPSILON = 0.01
SEEDS = ("still-17", "still-23", "still-29")
CONDITIONS = ("none", "full", "streaming", "am", "am-budgets", "still", *SEEDS)
LABELS = ("documents_only", "copy", "derived")
SLOTS = 164
COMPRESSION_BINS = ((0, 5), (5, 15), (15, 30), (30, math.inf))


def log(message):
    print(f"[report {time.strftime('%H:%M:%S')}] {message}", flush=True)


def load(path):
    """{(decision_id, index): {"task", "label", "prefix_tokens", condition: (mean_logprob, greedy_exact)}}"""
    values = defaultdict(dict)
    with path.open() as handle:
        for record in map(json.loads, handle):
            for index, value in enumerate(record["values"]):
                row = values[(record["decision_id"], index)]
                row.update(task=record["task_id"], label=value["label"], prefix_tokens=record["prefix_tokens"])
                row[record["condition"]] = (value["mean_logprob"], float(value["greedy_exact"]))
    for row in values.values():
        row["still"] = tuple(float(np.mean([row[s][i] for s in SEEDS])) for i in range(2))
    return list(values.values())


def summary(rows, condition, field, rng, scale=1.0):
    mean, ci = clustered_mean([(r["task"], scale * r[condition][field]) for r in rows], rng)
    return {"mean": round(mean, 4), "ci95": [round(x, 4) for x in ci]}


def clustered_share(rows, numerator, denominator, rng):
    """sum(numerator) / sum(denominator) over values, with a 95% interval from resampling tasks."""
    tasks = sorted({r["task"] for r in rows})
    top = np.array([sum(numerator(r) for r in rows if r["task"] == t) for t in tasks])
    bottom = np.array([sum(denominator(r) for r in rows if r["task"] == t) for t in tasks])
    draws = rng.integers(0, len(tasks), size=(BOOTSTRAP, len(tasks)))
    return {"share": round(float(top.sum() / bottom.sum()), 4),
            "ci95": [round(x, 4) for x in interval(top[draws].sum(1) / bottom[draws].sum(1))]}


def distance(row, condition):
    return EPSILON - row[condition][0]


def gap_shares(rows, condition, rng):
    exact = clustered_share(rows, lambda r: r[condition][1] - r["none"][1], lambda r: r["full"][1] - r["none"][1], rng)
    trace = clustered_share(rows, lambda r: math.log(distance(r, "none") / distance(r, condition)),
                            lambda r: math.log(distance(r, "none") / distance(r, "full")), rng)
    return {"exact_match": exact, "trace_log_ratio": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows = load(args.scores)
    order = [c for c in CONDITIONS if c in rows[0]]
    documents = [r for r in rows if r["label"] == "documents_only"]

    by_label = {}
    for label in LABELS:
        subset = [r for r in rows if r["label"] == label]
        by_label[label] = {"values": len(subset), "tasks": len({r["task"] for r in subset}),
                           "exact_match_points": {c: summary(subset, c, 1, rng, 100) for c in order},
                           "mean_logprob": {c: summary(subset, c, 0, rng) for c in order}}
    shares = {c: gap_shares(documents, c, rng) for c in order if c not in ("none", "full")}
    seed_spread = {kind: {"mean": round(float(np.mean([shares[s][kind]["share"] for s in SEEDS])), 4),
                          "sd": round(float(np.std([shares[s][kind]["share"] for s in SEEDS], ddof=1)), 4)}
                   for kind in ("exact_match", "trace_log_ratio")}

    compression = []
    for low, high in COMPRESSION_BINS:
        subset = [r for r in documents if low <= r["prefix_tokens"] / SLOTS < high]
        if subset:
            compression.append({"compression": f"{low}-{high}x", "tasks": len({r["task"] for r in subset}),
                                "values": len(subset),
                                "exact_match_points": {c: round(100 * float(np.mean([r[c][1] for r in subset])), 2)
                                                       for c in order}})

    report = {"model": "Qwen/Qwen3-4B-Instruct-2507", "slots": SLOTS, "epsilon": EPSILON, "bootstrap": BOOTSTRAP,
              "seed": SEED, "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
              "values_by_label": by_label, "documents_only_gap_closed": shares,
              "documents_only_gap_closed_still_seed_spread": seed_spread,
              "documents_only_exact_match_by_compression": compression}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
