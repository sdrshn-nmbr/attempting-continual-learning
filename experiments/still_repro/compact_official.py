"""Compact evaluation documents with the official Attention Matching code (github.com/adamzweiger/compaction).
Run from the official checkout in its pinned environment (transformers==4.57.1), for example:
    cd <checkout> && PYTHONPATH=. torchrun --standalone --nproc-per-node 8 <this dir>/compact_official.py ...
The chat header stays uncompacted, as in the official evaluator, and header plus compacted document fill --slots
positions per head on average, the budget STILL gets. Each document is saved as {domain}-{row}.pt in the packed
format of cache_format.py for evaluate.py --modes am."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import cache_format
import layout
# The official package only imports cleanly with evaluation before compaction (evaluation -> compaction -> evaluation).
from evaluation.configs.utils import load_algorithm_config, load_query_config
from compaction.compaction_methods.registry import get_compaction_method
from models.qwen3 import Qwen3ForCausalLM

DOMAINS = ["financial", "gutenberg", "legal", "code"]


def pack(compacted):
    return cache_format.pack([(c1[0].cpu(), beta[0].cpu(), c2[0].cpu()) for c1, beta, c2 in compacted])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--method", default="AM-HighestAttnKeys")
    parser.add_argument("--algorithm-config", default="default")
    parser.add_argument("--query-config", default="repeat")
    parser.add_argument("--budget-path")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(device).eval()
    method_kwargs = dict(load_algorithm_config(args.algorithm_config)[args.method])
    if args.budget_path:
        method_kwargs["precomputed_budget_path"] = args.budget_path
    method = get_compaction_method(args.method, method_kwargs=method_kwargs)
    query_config = load_query_config(args.query_config)
    windows = {d: np.load(args.corpus / f"eval-{d}.npy", mmap_mode="r") for d in DOMAINS}
    with args.items.open() as handle:
        items = [json.loads(line) for line in handle]
    items = items[:args.limit] if args.limit else items
    documents = sorted({(i["domain"], i["row"]) for i in items})[rank::world]
    header = len(layout.header_ids(tokenizer))
    print(f"[am {time.strftime('%H:%M:%S')}] rank {rank} documents {len(documents)} header {header} "
          f"method {args.method} {method_kwargs}", flush=True)
    with (args.out / f"log-rank{rank}.jsonl").open("a") as log:
        for number, (domain, row) in enumerate(documents, 1):
            path = args.out / f"{domain}-{row}.pt"
            if path.exists():
                continue
            started = time.time()
            prefix = layout.prefix_ids(tokenizer, windows[domain][row].tolist())
            formatted = tokenizer.decode(prefix)
            if tokenizer(formatted, add_special_tokens=False).input_ids != prefix:
                raise ValueError(f"AM_ROUNDTRIP_MISMATCH {domain}-{row}")
            with torch.no_grad():
                cache = model(input_ids=torch.tensor([prefix], device=device), use_cache=True).past_key_values
                compacted, _ = method.compact_kv_cache(past_key_values=cache, target_size=args.slots,
                                                       indices=range(header, len(prefix)), query_config=query_config,
                                                       model=model, tokenizer=tokenizer, formatted_context=formatted)
            packed = pack(compacted)
            lengths = packed["lengths"]
            if args.budget_path is None and not (lengths == args.slots).all():
                raise ValueError(f"AM_LENGTH_MISMATCH {domain}-{row} lengths={lengths.unique().tolist()} "
                                 f"slots={args.slots}")
            torch.save({**packed, "method": args.method, "kwargs": method_kwargs, "slots": args.slots,
                        "header": header}, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            record = {"domain": domain, "row": row, "seconds": round(time.time() - started, 1),
                      "head_min": int(lengths.min()), "head_max": int(lengths.max()),
                      "effective": lengths.float().mean().item(), "beta_min": packed["beta"].min().item(),
                      "beta_max": packed["beta"].max().item()}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[am {time.strftime('%H:%M:%S')}] rank {rank} {number}/{len(documents)} {json.dumps(record)}",
                  flush=True)
    print(f"[am {time.strftime('%H:%M:%S')}] rank {rank} done", flush=True)


if __name__ == "__main__":
    main()
