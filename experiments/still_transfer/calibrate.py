"""Fit the cross-model KV mapper on FineWeb-Edu text and pick the number of source layers k by the
receiver's held-out next-token KL when it reads a mapped cache instead of its own."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "still_repro"))
import kvmap
from still import continue_from, prefill


def log(rank, message):
    print(f"[calibrate {time.strftime('%H:%M:%S')}] rank {rank} {message}", flush=True)


def sequences(path, tokenizer, count, length):
    kept = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=256):
        for text in batch.column("text").to_pylist():
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) >= length:
                kept.append(ids[:length])
            if len(kept) == count:
                return kept
    raise SystemExit(f"FINEWEB_TOO_FEW_SEQUENCES have={len(kept)} need={count}")


def chunks(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def token_kl(reference, other):
    p = F.log_softmax(reference.float(), dim=-1)
    q = F.log_softmax(other.float(), dim=-1)
    return (p.exp() * (p - q)).sum(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--fineweb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=500)
    parser.add_argument("--heldout", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--prefix", type=int, default=768)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 36])
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    load = lambda name: AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                             attn_implementation="sdpa").to(device).eval()
    source, receiver = load(args.source), load(args.receiver)
    if source.config.head_dim != receiver.config.head_dim or \
            source.config.num_key_value_heads != receiver.config.num_key_value_heads:
        raise SystemExit("KV_SHAPE_MISMATCH: mapper needs equal KV heads and head_dim")
    dim, heads = source.config.head_dim, source.config.num_key_value_heads
    data = sequences(args.fineweb, tokenizer, args.calibration + args.heldout, args.length)
    calibration, heldout = data[:args.calibration], data[args.calibration:]

    width = lambda model: model.config.num_hidden_layers * dim
    keys = kvmap.Moments(heads, width(source), width(receiver), device)
    values = kvmap.Moments(heads, width(source), width(receiver), device)
    positions = torch.arange(args.length, device=device).unsqueeze(0)
    started = time.time()
    for batch in chunks(calibration[rank::world], args.batch):
        ids = torch.tensor(batch, device=device)
        with torch.no_grad():
            src = kvmap.to_content(source, prefill(source, ids), positions)
            tgt = kvmap.to_content(receiver, prefill(receiver, ids), positions)
        keys.add(kvmap.stack_heads(src, 0), kvmap.stack_heads(tgt, 0))
        values.add(kvmap.stack_heads(src, 1), kvmap.stack_heads(tgt, 1))
    for tensor in keys.tensors() + values.tensors():
        dist.all_reduce(tensor)
    log(rank, f"moments from {int(keys.count.item())} tokens in {time.time() - started:.0f}s")

    key_stats, value_stats = kvmap.statistics(keys, dim, args.lam), kvmap.statistics(values, dim, args.lam)
    prefix_positions = torch.arange(args.prefix, device=device).unsqueeze(0)
    mine = heldout[rank::world]
    results, mappers = {}, {}
    for k in [None] + args.ks:
        mapper = None if k is None else {"keys": kvmap.fit(key_stats, k), "values": kvmap.fit(value_stats, k)}
        total = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in chunks(mine, args.batch):
            ids = torch.tensor(batch, device=device)
            prefix, continuation = ids[:, :args.prefix], ids[:, args.prefix:]
            with torch.no_grad():
                own = continue_from(receiver, prefill(receiver, prefix), continuation, args.prefix).logits
                if mapper is None:
                    later = torch.arange(args.prefix, args.length, device=device).unsqueeze(0)
                    other = receiver(input_ids=continuation, position_ids=later.expand(len(batch), -1)).logits
                else:
                    content = kvmap.to_content(source, prefill(source, prefix), prefix_positions)
                    mapped = kvmap.from_content(receiver, kvmap.apply(mapper, content), prefix_positions,
                                                torch.bfloat16)
                    other = continue_from(receiver, mapped, continuation, args.prefix).logits
            values_kl = token_kl(own, other)
            total += torch.stack((values_kl.sum().double(), torch.tensor(values_kl.numel(), device=device,
                                                                         dtype=torch.float64)))
        dist.all_reduce(total)
        name = "no_prefix" if k is None else f"k{k}"
        results[name] = (total[0] / total[1]).item()
        if mapper is not None:
            mappers[k] = mapper
        log(rank, f"{name} heldout KL {results[name]:.4f}")

    best = min(args.ks, key=lambda k: results[f"k{k}"])
    if rank == 0:
        torch.save({"k": best, "lam": args.lam, "source": args.source, "receiver": args.receiver,
                    "keys": mappers[best]["keys"], "values": mappers[best]["values"]}, args.out / "mapper.pt")
        report = {"source": args.source, "receiver": args.receiver, "tokens": int(keys.count.item()),
                  "lam": args.lam, "heldout_kl": results, "chosen_k": best,
                  "key_scores": key_stats["scores"].tolist(), "value_scores": value_stats["scores"].tolist(),
                  "selections": {"keys": mappers[best]["keys"]["selections"],
                                 "values": mappers[best]["values"]["selections"]}}
        (args.out / "calibration.json").write_text(json.dumps(report, indent=2))
        log(rank, f"chose k={best}; wrote {args.out}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
