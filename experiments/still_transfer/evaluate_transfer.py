"""Evaluate Qwen3-4B-Instruct-2507 caches moved into a receiver model with the fitted KV mapper, alongside the
receiver's own no-document, full-document and same-size truncated baselines, on the STILL question set."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "still_repro"))
import attention_matching
import kvmap
import layout
from evaluate import DOMAINS, generate, prefix_state, summarize
from still import StillCompactor, prefill

NATIVE = ("none", "full", "streaming")
MAPPED = ("mapped_full", "mapped_still")
THINK_OFF = "<think>\n\n</think>\n\n"


def mapped_state(source, receiver, compactor, mapper, tokenizer, mode, windows, items, device, slots):
    prefixes = torch.tensor([layout.prefix_ids(tokenizer, windows[i["domain"]][i["row"]].tolist()) for i in items],
                            device=device)
    with torch.no_grad():
        pairs = prefill(source, prefixes)
        if mode == "mapped_full":
            positions = torch.arange(layout.PREFIX_TOKENS, device=device).unsqueeze(0)
        else:
            pairs = compactor(source, pairs)
            positions = torch.linspace(0.0, float(layout.PREFIX_TOKENS - 1), slots, device=device).unsqueeze(0)
        content = kvmap.to_content(source, pairs, positions)
        del pairs
        mapped = kvmap.apply(mapper, content)
        del content
        return kvmap.from_content(receiver, mapped, positions, torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=list(NATIVE + MAPPED))
    parser.add_argument("--think-off", action="store_true")
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=320)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    load = lambda name: AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation=attention_matching.NAME).to(device).eval()
    receiver = load(args.receiver)
    source = compactor = mapper = None
    if any(m in MAPPED for m in args.modes):
        source = load(args.source)
        compactor = StillCompactor(source.config, args.slots).to(device).eval()
        compactor.load_state_dict(torch.load(args.checkpoint, map_location=device))
        saved = torch.load(args.mapper, map_location=device)
        mapper = {"keys": saved["keys"], "values": saved["values"]}
        if len(mapper["keys"]["selections"]) != receiver.config.num_hidden_layers:
            raise SystemExit(f"MAPPER_RECEIVER_MISMATCH layers={len(mapper['keys']['selections'])}")
    suffix = THINK_OFF if args.think_off else ""
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
                if mode in NATIVE:
                    pairs, logical, _ = prefix_state(receiver, None, tokenizer, mode, windows, batch, device,
                                                     args.slots)
                elif mode in MAPPED:
                    pairs = mapped_state(source, receiver, compactor, mapper, tokenizer, mode, windows, batch,
                                         device, args.slots)
                    logical = layout.PREFIX_TOKENS
                else:
                    raise SystemExit(f"TRANSFER_MODE {mode}")
                texts = generate(receiver, tokenizer, pairs, logical, batch, device, args.max_new, suffix)
                for item, text in zip(batch, texts, strict=True):
                    handle.write(json.dumps({"id": item["id"], "domain": item["domain"], "type": item["type"],
                                             "mode": mode, "gold": item["gold"],
                                             "prediction": layout.parse_letter(text),
                                             "physical_prefix": pairs[0][0].shape[-2], "text": text}) + "\n")
                handle.flush()
            print(f"[transfer {time.strftime('%H:%M:%S')}] rank {rank} mode {mode} done "
                  f"({time.time() - started:.0f}s)", flush=True)
    dist.barrier()
    if rank == 0:
        records = []
        for other in range(world):
            with (args.out / f"predictions-rank{other}.jsonl").open() as handle:
                records.extend(json.loads(line) for line in handle)
        summary = summarize(records)
        summary["receiver"], summary["mapper"] = args.receiver, str(args.mapper)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({m: {"accuracy": s["accuracy"], "ci95": s["ci95"]} for m, s in summary.items()
                          if isinstance(s, dict)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
