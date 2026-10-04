"""Per-head STILL slot counts from an official Attention Matching per-head budget file, allocated the way the official
code allocates them (load_budgets_from_json, apply_max_ratio_cap, then int(proportion * target * heads)). STILL
compresses the whole prefix, so the target is the full slot budget and the article is the whole prefix. Run from the
official checkout in its pinned environment:
  cd <checkout> && PYTHONPATH=. python <this dir>/budget_counts.py \
      --budgets head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json --out <counts.json>
"""
import argparse
import hashlib
import json
from pathlib import Path

# The official package only imports cleanly with evaluation before compaction (evaluation -> compaction -> evaluation).
from evaluation.configs.utils import load_algorithm_config  # noqa: F401
from compaction.compaction_methods.base import apply_max_ratio_cap, load_budgets_from_json

LAYERS = 36
HEADS = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--max-ratio-per-head", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    total = LAYERS * HEADS
    proportions = load_budgets_from_json(str(args.budgets), LAYERS, HEADS)
    missing = [key for key, value in proportions.items() if value is None]
    if missing:
        raise SystemExit(f"BUDGET_MISSING_HEADS {missing[:5]}")
    capped = apply_max_ratio_cap(proportions, args.max_ratio_per_head, args.slots / args.prefix_tokens, total)
    counts = [[int(capped[(layer, head)] * args.slots * total) for head in range(HEADS)] for layer in range(LAYERS)]
    flat = [c for row in counts for c in row]
    if min(flat) < 1:
        raise SystemExit(f"BUDGET_EMPTY_HEAD min={min(flat)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": str(args.budgets), "source_sha256": hashlib.sha256(args.budgets.read_bytes()).hexdigest(),
        "slots": args.slots, "prefix_tokens": args.prefix_tokens, "max_ratio_per_head": args.max_ratio_per_head,
        "mean": sum(flat) / total, "min": min(flat), "max": max(flat), "total": sum(flat), "counts": counts}, indent=1) + "\n")
    print(f"[budget_counts] mean={sum(flat) / total:.2f} min={min(flat)} max={max(flat)} total={sum(flat)} -> {args.out}")


if __name__ == "__main__":
    main()
