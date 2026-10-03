import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_matching
import layout
from still import StillCompactor, build_cache, prefill, streaming_pairs

DOMAINS = ["financial", "gutenberg", "legal", "code"]
AM_CACHE = {}


def am_fit(model, tokenizer, windows, item, slots):
    key = (item["domain"], item["row"])
    if key not in AM_CACHE:
        AM_CACHE.clear()
        AM_CACHE[key] = attention_matching.compact(model, tokenizer, windows[key[0]][key[1]].tolist(), slots)
    return AM_CACHE[key]


def prefix_state(model, compactor, tokenizer, mode, windows, items, device, slots):
    if mode == "none":
        header = torch.tensor([layout.header_ids(tokenizer)], device=device)
        pairs = [(k.expand(len(items), -1, -1, -1), v.expand(len(items), -1, -1, -1)) for k, v in prefill(model, header)]
        return pairs, header.shape[1], None
    if mode == "am":
        fitted = [am_fit(model, tokenizer, windows, i, slots) for i in items]
        pairs = [(torch.cat([f[0][layer][0] for f in fitted]), torch.cat([f[0][layer][1] for f in fitted]))
                 for layer in range(len(fitted[0][0]))]
        betas = [torch.cat([f[1][layer] for f in fitted]) for layer in range(len(fitted[0][1]))]
        return pairs, layout.PREFIX_TOKENS, betas
    prefixes = torch.tensor([layout.prefix_ids(tokenizer, windows[i["domain"]][i["row"]].tolist()) for i in items],
                            device=device)
    pairs = prefill(model, prefixes)
    if mode == "streaming":
        pairs = streaming_pairs(pairs, slots)
    elif mode in ("still", "untrained"):
        with torch.no_grad():
            pairs = compactor(model, pairs)
    elif mode != "full":
        raise ValueError(f"EVAL_MODE {mode}")
    return pairs, layout.PREFIX_TOKENS, None


@torch.no_grad()
def generate(model, tokenizer, pairs, logical, items, device, max_new, suffix=""):
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    questions = [layout.encode(tokenizer, layout.question_text(i) + suffix) for i in items]
    width = max(len(q) for q in questions)
    ids = torch.full((len(items), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    real = torch.zeros((len(items), width), dtype=torch.long, device=device)
    for row, question in enumerate(questions):
        ids[row, width - len(question):] = torch.tensor(question, device=device)
        real[row, width - len(question):] = 1
    physical = pairs[0][0].shape[-2]
    mask = torch.cat((torch.ones((len(items), physical), dtype=torch.long, device=device), real), dim=1)
    positions = (logical + real.cumsum(1) - 1).clamp(min=logical)
    output = model(input_ids=ids, past_key_values=build_cache(pairs), attention_mask=mask, position_ids=positions,
                   use_cache=True)
    last = positions[:, -1]
    generated = [[] for _ in items]
    done = torch.zeros(len(items), dtype=torch.bool, device=device)
    for _ in range(max_new):
        token = output.logits[:, -1].argmax(-1)
        for row in range(len(items)):
            if not done[row]:
                generated[row].append(token[row].item())
        done |= token == end_id
        if done.all():
            break
        mask = torch.cat((mask, torch.ones((len(items), 1), dtype=torch.long, device=device)), dim=1)
        last = last + 1
        output = model(input_ids=token[:, None], past_key_values=output.past_key_values, attention_mask=mask,
                       position_ids=last[:, None], use_cache=True)
    return [tokenizer.decode([t for t in g if t != end_id]) for g in generated]


def bootstrap(correct, seed=0, samples=2000):
    rng = random.Random(seed)
    values = sorted(sum(rng.choice(correct) for _ in correct) / len(correct) for _ in range(samples))
    return values[int(0.025 * samples)], values[int(0.975 * samples) - 1]


def summarize(records):
    summary = {}
    for mode in sorted({r["mode"] for r in records}):
        rows = [r for r in records if r["mode"] == mode]
        correct = [int(r["prediction"] == r["gold"]) for r in rows]
        by_domain = {}
        for domain in DOMAINS:
            subset = [int(r["prediction"] == r["gold"]) for r in rows if r["domain"] == domain]
            if subset:
                by_domain[domain] = {"n": len(subset), "accuracy": sum(subset) / len(subset)}
        summary[mode] = {"n": len(rows), "accuracy": sum(correct) / len(rows), "ci95": bootstrap(correct),
                         "by_domain": by_domain, "unparsed": sum(r["prediction"] is None for r in rows),
                         "predicted_letters": dict(Counter(r["prediction"] for r in rows))}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["full", "none", "streaming", "untrained"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=320)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation=attention_matching.NAME).to(device).eval()
    untrained = StillCompactor(model.config, args.slots).to(device).eval()
    trained = StillCompactor(model.config, args.slots).to(device).eval()
    if "still" in args.modes:
        if args.checkpoint is None:
            raise ValueError("EVAL_STILL_REQUIRES_CHECKPOINT")
        trained.load_state_dict(torch.load(args.checkpoint, map_location=device))
    windows = {d: np.load(args.corpus / f"eval-{d}.npy", mmap_mode="r") for d in DOMAINS}
    with args.items.open() as handle:
        items = [json.loads(line) for line in handle]
    items = items[:args.limit] if args.limit else items
    documents = sorted({(i["domain"], i["row"]) for i in items})[rank::world]
    mine = sorted((i for i in items if (i["domain"], i["row"]) in set(documents)),
                  key=lambda i: (i["domain"], i["row"], i["id"]))
    shard = args.out / f"predictions-rank{rank}.jsonl"
    started = time.time()
    with shard.open("w") as handle:
        for mode in args.modes:
            for start in range(0, len(mine), args.batch):
                batch = mine[start:start + args.batch]
                compactor = trained if mode == "still" else untrained
                pairs, logical, betas = prefix_state(model, compactor, tokenizer, mode, windows, batch, device,
                                                     args.slots)
                attention_matching.STATE.bias = betas
                try:
                    texts = generate(model, tokenizer, pairs, logical, batch, device, args.max_new)
                finally:
                    attention_matching.STATE.bias = None
                for item, text in zip(batch, texts, strict=True):
                    handle.write(json.dumps({"id": item["id"], "domain": item["domain"], "mode": mode,
                                             "gold": item["gold"], "prediction": layout.parse_letter(text),
                                             "physical_prefix": pairs[0][0].shape[-2], "text": text}) + "\n")
                handle.flush()
            print(f"[eval {time.strftime('%H:%M:%S')}] rank {rank} mode {mode} done "
                  f"({time.time() - started:.0f}s)", flush=True)
    dist.barrier()
    if rank == 0:
        records = []
        for other in range(world):
            with (args.out / f"predictions-rank{other}.jsonl").open() as handle:
                records.extend(json.loads(line) for line in handle)
        summary = summarize(records)
        summary["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({m: {"accuracy": s["accuracy"], "ci95": s["ci95"]} for m, s in summary.items()
                          if isinstance(s, dict)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
