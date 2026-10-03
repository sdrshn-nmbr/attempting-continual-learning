"""Compress each τ-banking documents prefix (conditions.py prefixes.jsonl) with the official Attention Matching code,
as compact_official.py does for the STILL evaluation documents: the system header stays uncompressed and header plus
compressed documents fill exactly --slots positions. Run from the official checkout in its pinned environment:
    cd <checkout> && PYTHONPATH=. torchrun --standalone --nproc-per-node 8 <this dir>/compact_am.py ...
Each task is saved as {task_id}.pt with keys/values [L,H,t,D], beta [L,H,t] and the header length, which
conditions.prefix_state(..., "am") checks."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "still_repro"))
from compact_official import Qwen3ForCausalLM, get_compaction_method, load_algorithm_config, load_query_config, stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--method", default="AM-HighestAttnKeys")
    parser.add_argument("--algorithm-config", default="default")
    parser.add_argument("--query-config", default="repeat")
    args = parser.parse_args()

    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(device).eval()
    method_kwargs = dict(load_algorithm_config(args.algorithm_config)[args.method])
    method = get_compaction_method(args.method, method_kwargs=method_kwargs)
    query_config = load_query_config(args.query_config)
    with args.prefixes.open() as handle:
        prefixes = [json.loads(line) for line in handle][rank::world]
    print(f"[am-tau {time.strftime('%H:%M:%S')}] rank {rank} tasks {len(prefixes)} method {args.method} "
          f"{method_kwargs}", flush=True)
    with (args.out / f"log-rank{rank}.jsonl").open("a") as log:
        for number, prefix in enumerate(prefixes, 1):
            path = args.out / f"{prefix['task_id']}.pt"
            if path.exists():
                continue
            started = time.time()
            ids, header = prefix["prefix_ids"], len(prefix["header_ids"])
            formatted = tokenizer.decode(ids)
            if tokenizer(formatted, add_special_tokens=False).input_ids != ids:
                raise ValueError(f"AM_ROUNDTRIP_MISMATCH {prefix['task_id']}")
            with torch.no_grad():
                cache = model(input_ids=torch.tensor([ids], device=device), use_cache=True).past_key_values
                compacted, _ = method.compact_kv_cache(past_key_values=cache, target_size=args.slots,
                                                       indices=range(header, len(ids)), query_config=query_config,
                                                       model=model, tokenizer=tokenizer, formatted_context=formatted)
            keys, beta, values = stack(compacted)
            if keys.shape[2] != args.slots:
                raise ValueError(f"AM_LENGTH_MISMATCH {prefix['task_id']} physical={keys.shape[2]} slots={args.slots}")
            finite = beta[torch.isfinite(beta)]
            torch.save({"keys": keys.cpu(), "beta": beta.float().cpu(), "values": values.cpu(), "method": args.method,
                        "kwargs": method_kwargs, "slots": args.slots, "header": header}, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            record = {"task_id": prefix["task_id"], "prefix_tokens": len(ids), "seconds": round(time.time() - started, 1),
                      "beta_min": finite.min().item(), "beta_max": finite.max().item()}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[am-tau {time.strftime('%H:%M:%S')}] rank {rank} {number}/{len(prefixes)} {json.dumps(record)}",
                  flush=True)
    print(f"[am-tau {time.strftime('%H:%M:%S')}] rank {rank} done", flush=True)


if __name__ == "__main__":
    main()
