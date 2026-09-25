import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def predictions(directory):
    rows = []
    for path in sorted(Path(directory).glob("predictions-rank*.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def accuracy(rows, key=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key] if key else "all"].append(row["prediction"] == row["gold"])
    return {group: round(sum(values) / len(values), 4) for group, values in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.items.open() as handle:
        kinds = {item["id"]: item["type"] for item in map(json.loads, handle)}
    report = {"modes": {}, "seeds": {}}
    for row_set in [predictions(args.baselines)]:
        for mode in sorted({r["mode"] for r in row_set}):
            rows = [{**r, "type": kinds[r["id"]]} for r in row_set if r["mode"] == mode]
            report["modes"][mode] = {"n": len(rows), **accuracy(rows), "by_type": accuracy(rows, "type"),
                                     "by_domain": accuracy(rows, "domain")}
    scores = []
    for directory in args.seeds:
        rows = [{**r, "type": kinds[r["id"]]} for r in predictions(directory) if r["mode"] == "still"]
        entry = {"n": len(rows), **accuracy(rows), "by_type": accuracy(rows, "type"),
                 "by_domain": accuracy(rows, "domain")}
        report["seeds"][directory.name] = entry
        scores.append(entry["all"])
    report["still"] = {"mean": round(statistics.mean(scores), 4),
                       "sd": round(statistics.stdev(scores), 4) if len(scores) > 1 else None}
    full, none = report["modes"]["full"]["all"], report["modes"]["none"]["all"]
    report["still"]["utilization"] = round((report["still"]["mean"] - none) / (full - none), 4)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
