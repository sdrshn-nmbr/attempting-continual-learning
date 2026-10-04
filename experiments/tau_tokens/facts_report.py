"""Results of the short-answer τ-banking eval (facts.py) on Qwen3-4B-Instruct-2507. Pass criteria for the sanity
checks are fixed here, before looking at the scores. Intervals resample whole tasks (sanity.BOOTSTRAP, fixed seed).
  1 original answers: full beats none. Pass if the intervals of the exact-match gain and the mean log-probability gain
    (full minus none) both lie above zero.
  2 edited answers: the same check.
  3 full documents are read, not recalled: with the edited documents, full's exact match on the edited value is at
    least its exact match on the original documents minus READ_TOLERANCE points, and full gives the edited value a
    higher log-probability than the stale (original) value on at least PREFER_MIN of questions.
  4 pipeline: "none" sees identical text for an original question and its stale twin, so their log-probabilities
    agree within IDENTITY_TOLERANCE for every question.
For each version and condition: exact match, mean log-probability, and the share of the none-to-full gap closed, in
exact match and in TRACE's log-ratio form (report.gap_shares). With edited documents, per condition: how often the
greedy answer is the edited value, how often it is the stale value, and how often the edited value is more likely.

  uv run --no-project --with numpy --with scipy python facts_report.py --scores <scores.jsonl> ... --edits <edits.jsonl> \
      --out results/facts/report.json
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from report import COMPRESSION_BINS, SEEDS, SLOTS, gap_shares
from sanity import BOOTSTRAP, SEED, clustered_mean

CONDITIONS = ("none", "full", "streaming", "am", "am-budgets", "still", *SEEDS)
READ_TOLERANCE = 10.0
PREFER_MIN = 0.9
IDENTITY_TOLERANCE = 0.05


def log(message):
    print(f"[facts_report {time.strftime('%H:%M:%S')}] {message}", flush=True)


def load(paths):
    """{decision_id: {"task", "label", "question", "prefix_tokens", condition: (mean_logprob, greedy_exact)}}"""
    rows = defaultdict(dict)
    for path in paths:
        with path.open() as handle:
            for record in map(json.loads, handle):
                (value,) = record["values"]
                question = record["decision_id"] if value["label"] == "original" \
                    else record["decision_id"].rsplit(":", 1)[0]
                row = rows[record["decision_id"]]
                row.update(task=record["task_id"], label=value["label"], question=question,
                           prefix_tokens=record["prefix_tokens"])
                if record["condition"] in row:
                    raise SystemExit(f"DUPLICATE_RECORD {record['decision_id']} {record['condition']}")
                row[record["condition"]] = (value["mean_logprob"], float(value["greedy_exact"]))
    for decision_id, row in rows.items():
        missing = [c for c in CONDITIONS if c != "still" and c not in row]
        if missing:
            raise SystemExit(f"MISSING_CONDITIONS {decision_id} {missing}")
        row["still"] = tuple(float(np.mean([row[s][i] for s in SEEDS])) for i in range(2))
    return rows


def summary(pairs, rng):
    mean, ci = clustered_mean(pairs, rng)
    return {"mean": round(mean, 4), "ci95": [round(x, 4) for x in ci]}


def gain(rows, field, rng, scale=1.0):
    result = summary([(r["task"], scale * (r["full"][field] - r["none"][field])) for r in rows], rng)
    return {**result, "pass": result["ci95"][0] > 0}


def version(rows, rng):
    return {"questions": len(rows), "tasks": len({r["task"] for r in rows}),
            "exact_match_points": {c: summary([(r["task"], 100 * r[c][1]) for r in rows], rng) for c in CONDITIONS},
            "mean_logprob": {c: summary([(r["task"], r[c][0]) for r in rows], rng) for c in CONDITIONS},
            "gap_closed": {c: gap_shares(rows, c, rng) for c in CONDITIONS if c not in ("none", "full")}}


def exact_points(rows, condition):
    return round(100 * float(np.mean([r[condition][1] for r in rows])), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows = load(args.scores)
    with args.edits.open() as handle:
        kinds = {e["id"]: e["kind"] for e in map(json.loads, handle)}
    original = [r for r in rows.values() if r["label"] == "original"]
    edited = [r for r in rows.values() if r["label"] == "edited"]
    stale = {r["question"]: r for r in rows.values() if r["label"] == "stale"}
    by_question = {r["question"]: r for r in original}
    if set(by_question) != set(kinds) or {r["question"] for r in edited} != set(kinds) or set(stale) != set(kinds):
        raise SystemExit("QUESTION_SETS_DIFFER")

    choice = {c: {"greedy_is_edited_points": summary([(r["task"], 100 * r[c][1]) for r in edited], rng),
                  "greedy_is_stale_points": summary([(r["task"], 100 * stale[r["question"]][c][1]) for r in edited],
                                                    rng),
                  "edited_more_likely_points": summary([(r["task"], 100 * float(r[c][0] > stale[r["question"]][c][0]))
                                                        for r in edited], rng)}
              for c in CONDITIONS}

    full_original, full_edited = exact_points(original, "full"), exact_points(edited, "full")
    full_prefers = choice["full"]["edited_more_likely_points"]["mean"] / 100
    identity = max(abs(by_question[q]["none"][0] - stale[q]["none"][0]) for q in kinds)
    checks = {
        "1_original_full_beats_none": {"exact_match_points": gain(original, 1, rng, 100),
                                       "mean_logprob": gain(original, 0, rng)},
        "2_edited_full_beats_none": {"exact_match_points": gain(edited, 1, rng, 100),
                                     "mean_logprob": gain(edited, 0, rng)},
        "3_full_reads_the_edit": {"full_exact_original_points": full_original, "full_exact_edited_points": full_edited,
                                  "full_edited_more_likely": round(full_prefers, 4),
                                  "pass": full_edited >= full_original - READ_TOLERANCE and full_prefers >= PREFER_MIN},
        "4_none_identical_for_original_and_stale": {"max_abs_logprob_difference": round(identity, 6),
                                                    "pass": identity <= IDENTITY_TOLERANCE},
    }
    for name in ("1_original_full_beats_none", "2_edited_full_beats_none"):
        checks[name]["pass"] = all(v["pass"] for v in checks[name].values())

    by_kind = {}
    for kind in sorted(set(kinds.values())):
        subset = [r for r in original if kinds[r["question"]] == kind]
        subset_edited = [r for r in edited if kinds[r["question"]] == kind]
        by_kind[kind] = {"questions": len(subset),
                         "original_exact_match_points": {c: exact_points(subset, c) for c in CONDITIONS},
                         "edited_exact_match_points": {c: exact_points(subset_edited, c) for c in CONDITIONS}}
    compression = []
    for low, high in COMPRESSION_BINS:
        subset = [r for r in original if low <= r["prefix_tokens"] / SLOTS < high]
        subset_edited = [r for r in edited if low <= r["prefix_tokens"] / SLOTS < high]
        if subset:
            compression.append({"compression": f"{low}-{high}x", "tasks": len({r["task"] for r in subset}),
                                "questions": len(subset),
                                "original_exact_match_points": {c: exact_points(subset, c) for c in CONDITIONS},
                                "edited_exact_match_points": {c: exact_points(subset_edited, c) for c in CONDITIONS}})
    seed_sd = {name: {kind: round(float(np.std([gap_shares(rs, s, rng)[kind]["share"] for s in SEEDS], ddof=1)), 4)
                      for kind in ("exact_match", "trace_log_ratio")}
               for name, rs in (("original", original), ("edited", edited))}
    report = {"model": "Qwen/Qwen3-4B-Instruct-2507", "slots": SLOTS, "bootstrap": BOOTSTRAP, "seed": SEED,
              "scores_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.scores},
              "sanity": checks, "original": version(original, rng), "edited": version(edited, rng),
              "edited_choice": choice, "still_seed_sd_of_gap_closed": seed_sd, "by_kind": by_kind,
              "by_compression": compression}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {args.out}; sanity {[(k, v['pass']) for k, v in checks.items()]}")


if __name__ == "__main__":
    main()
