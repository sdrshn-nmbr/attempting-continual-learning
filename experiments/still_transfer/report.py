"""Summarize transfer evaluations: accuracy with bootstrap CI, unparsed answers, share of the receiver's
full-document gain retained ((mode - none) / (full - none)), and accuracy by question type."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def bootstrap(correct, seed=0, samples=4000):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(correct) for _ in correct) / len(correct) for _ in range(samples))
    return round(100 * means[int(0.025 * samples)], 1), round(100 * means[int(0.975 * samples) - 1], 1)


def load(directory):
    records = []
    for path in sorted(directory.glob("predictions-rank*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return records


def summarize(records):
    by_mode = defaultdict(list)
    for record in records:
        by_mode[record["mode"]].append(record)
    accuracy = {m: sum(r["prediction"] == r["gold"] for r in rows) / len(rows) for m, rows in by_mode.items()}
    floor, ceiling = accuracy.get("none"), accuracy.get("full")
    report = {}
    for mode, rows in sorted(by_mode.items()):
        correct = [int(r["prediction"] == r["gold"]) for r in rows]
        types = defaultdict(list)
        for r in rows:
            types[r["type"]].append(int(r["prediction"] == r["gold"]))
        retained = None
        if floor is not None and ceiling is not None and ceiling > floor:
            retained = round(100 * (accuracy[mode] - floor) / (ceiling - floor), 1)
        report[mode] = {"n": len(rows), "accuracy": round(100 * accuracy[mode], 1), "ci95": bootstrap(correct),
                        "retained_gain": retained, "unparsed": sum(r["prediction"] is None for r in rows),
                        "by_type": {t: round(100 * sum(v) / len(v), 1) for t, v in sorted(types.items())}}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+", help="evaluation directories with predictions-rank*.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combined = {}
    for directory in args.runs:
        combined[directory.name] = summarize(load(directory))
        for mode, row in combined[directory.name].items():
            print(f"{directory.name:>18} {mode:<13} n={row['n']:<4} acc={row['accuracy']:<5} ci={row['ci95']} "
                  f"retained={row['retained_gain']} unparsed={row['unparsed']} by_type={row['by_type']}")
    args.out.write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
