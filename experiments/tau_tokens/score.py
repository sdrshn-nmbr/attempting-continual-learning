"""Step 3 of the τ-banking token-level eval: score every decision under every documents condition. For each value
the gold call specifies, record the log-probability of each of its tokens given everything before it, their mean
(TRACE's average gold log-probability), and whether greedy decoding reproduces every token. STILL runs once per
compactor seed; each checkpoint must match the hash recorded with the three-seed reproduction.

Run on the GPU worker, one shard of tasks per rank:
  torchrun --standalone --nproc-per-node 8 score.py --inputs <conditions.py out> --model <Qwen3-4B-Instruct-2507> \
      --am-dir <compact_am.py out> --checkpoint 17=<path> --checkpoint 23=<path> --checkpoint 29=<path> --out <dir>
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "still_repro"))
import attention_matching
import conditions
from still import StillCompactor, build_cache

CHUNK = 2048
SEED_HASHES = HERE.parent / "still_repro" / "results" / "luna-3seed"


def log(message):
    print(f"[score {time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_compactor(model, seed, path, device):
    expected = (SEED_HASHES / f"compactor-seed{seed}.sha256").read_text().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"CHECKPOINT_HASH_MISMATCH seed={seed} path={path} expected={expected} actual={actual}")
    compactor = StillCompactor(model.config, slots=conditions.SLOTS).to(device)
    compactor.load_state_dict(torch.load(path, map_location=device))
    return compactor.eval()


@torch.no_grad()
def value_scores(model, pairs, logical, rest_ids, values, chunk=CHUNK):
    """Teacher-force rest_ids after the prefix cache, chunk by chunk, up to the last value token. Token t is
    predicted from the hidden state at t - 1; logits are computed only at those positions."""
    device = pairs[0][0].device
    wanted = sorted({t - 1 for v in values for t in range(v["token_start"], v["token_end"])})
    if wanted[0] < 0:
        raise SystemExit("VALUE_AT_FIRST_REST_TOKEN")
    ids = torch.tensor([rest_ids], device=device)
    cache, hidden = build_cache(pairs), {}
    for start in range(0, wanted[-1] + 1, chunk):
        end = min(start + chunk, wanted[-1] + 1)
        positions = torch.arange(logical + start, logical + end, device=device)[None]
        output = model.model(input_ids=ids[:, start:end], past_key_values=cache, position_ids=positions,
                             use_cache=True)
        cache = output.past_key_values
        hidden.update({t: output.last_hidden_state[0, t - start] for t in wanted if start <= t < end})
    logprobs = torch.log_softmax(model.lm_head(torch.stack([hidden[t] for t in wanted])).float(), dim=-1)
    row = {t: i for i, t in enumerate(wanted)}
    scores = []
    for value in values:
        rows = [row[t - 1] for t in range(value["token_start"], value["token_end"])]
        gold = ids[0, value["token_start"]:value["token_end"]]
        token = logprobs[rows].gather(1, gold[:, None])[:, 0]
        scores.append({"path": value["path"], "label": value["label"], "token_logprobs": token.tolist(),
                       "mean_logprob": token.mean().item(),
                       "greedy_exact": bool((logprobs[rows].argmax(-1) == gold).all())})
    return scores


def runs(names, compactors):
    for name in names:
        if name == "still":
            yield from ((f"still-{seed}", "still", compactor) for seed, compactor in sorted(compactors.items()))
        else:
            yield name, name, None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--am-dir", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], help="seed=path")
    parser.add_argument("--conditions", nargs="+", default=list(conditions.CONDITIONS))
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    unknown = set(args.conditions) - set(conditions.CONDITIONS)
    if unknown:
        raise SystemExit(f"UNKNOWN_CONDITIONS {sorted(unknown)}")
    if "am" in args.conditions and args.am_dir is None:
        raise SystemExit("AM_REQUIRES_AM_DIR")
    if "still" in args.conditions and not args.checkpoint:
        raise SystemExit("STILL_REQUIRES_CHECKPOINTS")

    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation=attention_matching.NAME).to(device).eval()
    compactors = {}
    for entry in args.checkpoint if "still" in args.conditions else []:
        seed, path = entry.split("=", 1)
        compactors[int(seed)] = load_compactor(model, int(seed), Path(path), device)
    with (args.inputs / "prefixes.jsonl").open() as handle:
        prefixes = {row["task_id"]: row for row in map(json.loads, handle)}
    decisions = defaultdict(list)
    with (args.inputs / "inputs.jsonl").open() as handle:
        for row in map(json.loads, handle):
            decisions[row["task_id"]].append(row)
    tasks = sorted(prefixes)[:args.limit_tasks][rank::world]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"scores-rank{rank}.jsonl"
    done = set()
    if path.exists():
        with path.open() as handle:
            done = {(r["decision_id"], r["condition"]) for r in map(json.loads, handle)}
    log(f"rank {rank}: {len(tasks)} tasks, conditions {[name for name, _, _ in runs(args.conditions, compactors)]}, "
        f"{len(done)} records already written")
    with path.open("a") as out:
        for number, task_id in enumerate(tasks, 1):
            prefix, started = prefixes[task_id], time.time()
            for name, kind, compactor in runs(args.conditions, compactors):
                pending = [d for d in decisions[task_id] if (d["decision_id"], name) not in done]
                if not pending:
                    continue
                am = torch.load(args.am_dir / f"{task_id}.pt", map_location=device) if kind == "am" else None
                pairs, logical, bias = conditions.prefix_state(model, kind, prefix, compactor=compactor, am=am)
                attention_matching.STATE.bias = bias
                try:
                    for decision in pending:
                        scores = value_scores(model, pairs, logical, decision["rest_ids"], decision["values"])
                        out.write(json.dumps({"decision_id": decision["decision_id"], "task_id": task_id,
                                              "condition": name, "prefix_tokens": len(prefix["prefix_ids"]),
                                              "prefix_physical": pairs[0][0].shape[-2], "logical_start": logical,
                                              "values": scores}) + "\n")
                finally:
                    attention_matching.STATE.bias = None
                out.flush()
            log(f"rank {rank}: {number}/{len(tasks)} {task_id} ({len(prefix['prefix_ids'])} prefix tokens, "
                f"{len(decisions[task_id])} decisions) in {time.time() - started:.1f}s")
    log(f"rank {rank}: done")


if __name__ == "__main__":
    main()
