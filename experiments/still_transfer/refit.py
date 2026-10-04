"""Refit the KV mapper on STILL outputs by behavior: starting from the ridge solution, train only the mapper so
the receiver reading mapped STILL slots matches the receiver reading the full document (support KL on answer
tokens). The source model, the STILL compactor and the receiver stay frozen."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "still_repro"))
import kvmap
import layout
from still import StillCompactor, continue_from, prefill, support_kl
from train import collate

DOMAINS = ["financial", "gutenberg", "legal", "code"]
THINK_OFF = "<think>\n\n</think>\n\n"


class Mapper(nn.Module):
    """Trainable KV mapper. With rank > 0 the saved weights stay frozen and only a low-rank correction A @ B per
    layer and head trains (B starts at zero, so training starts from the saved map)."""

    def __init__(self, saved, rank=0):
        super().__init__()
        self.selections = {part: saved[part]["selections"] for part in ("keys", "values")}
        self.rank = rank
        trainable = rank == 0
        wrap = lambda tensors: nn.ParameterList([nn.Parameter(t.float().clone(), requires_grad=trainable)
                                                 for t in tensors])
        self.key_weights, self.key_biases = wrap(saved["keys"]["weights"]), wrap(saved["keys"]["biases"])
        self.value_weights, self.value_biases = wrap(saved["values"]["weights"]), wrap(saved["values"]["biases"])
        if rank > 0:
            def factors(weights):
                down = nn.ParameterList([nn.Parameter(torch.randn(w.shape[0], w.shape[1], rank) / w.shape[1] ** 0.5)
                                         for w in weights])
                up = nn.ParameterList([nn.Parameter(torch.zeros(w.shape[0], rank, w.shape[2])) for w in weights])
                return down, up
            self.key_down, self.key_up = factors(saved["keys"]["weights"])
            self.value_down, self.value_up = factors(saved["values"]["weights"])

    def tensors(self, part):
        weights, biases = (self.key_weights, self.key_biases) if part == "keys" else \
            (self.value_weights, self.value_biases)
        if self.rank == 0:
            return list(weights), list(biases)
        down, up = (self.key_down, self.key_up) if part == "keys" else (self.value_down, self.value_up)
        return [w + d @ u for w, d, u in zip(weights, down, up, strict=True)], list(biases)

    def parts(self):
        return {part: {"selections": self.selections[part], "weights": self.tensors(part)[0],
                       "biases": self.tensors(part)[1]} for part in ("keys", "values")}

    def trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, content):
        return kvmap.apply(self.parts(), content)

    def export(self):
        with torch.no_grad():
            return {part: {"selections": self.selections[part],
                           "weights": [w.detach().cpu() for w in self.tensors(part)[0]],
                           "biases": [b.detach().cpu() for b in self.tensors(part)[1]]} for part in ("keys", "values")}


def identity_map(layers, heads, dim):
    part = lambda: {"selections": [[layer] for layer in range(layers)],
                    "weights": [torch.eye(dim).expand(heads, dim, dim).clone() for _ in range(layers)],
                    "biases": [torch.zeros(heads, dim) for _ in range(layers)]}
    return {"k": 1, "lam": None, "keys": part(), "values": part()}


def step_loss(source, compactor, receiver, mapper, batch, logical, slots, top_k=200):
    prefixes, ids, (rows, columns), gold = batch
    with torch.no_grad():
        teacher = continue_from(receiver, prefill(receiver, prefixes), ids, logical).logits[rows, columns]
        compact = compactor(source, prefill(source, prefixes))
        positions = torch.linspace(0.0, float(logical - 1), slots, device=prefixes.device).unsqueeze(0)
        content = kvmap.to_content(source, compact, positions)
    mapped = kvmap.from_content(receiver, mapper(content), positions, next(receiver.parameters()).dtype)
    student = continue_from(receiver, mapped, ids, logical).logits[rows, columns]
    return support_kl(teacher, student, gold, top_k).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, help="saved ridge map to start from; omit with --init identity")
    parser.add_argument("--init", choices=["saved", "identity"], default="saved")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--think-off", action="store_true")
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
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
    for model in (source, receiver):
        model.requires_grad_(False)
    compactor = StillCompactor(source.config, args.slots).to(device).eval().requires_grad_(False)
    compactor.load_state_dict(torch.load(args.checkpoint, map_location=device))
    if args.init == "identity":
        if source.config.num_hidden_layers != receiver.config.num_hidden_layers:
            raise SystemExit("IDENTITY_INIT_NEEDS_EQUAL_DEPTH")
        saved = identity_map(receiver.config.num_hidden_layers, receiver.config.num_key_value_heads,
                             receiver.config.head_dim)
    else:
        if args.mapper is None:
            raise SystemExit("REFIT_SAVED_INIT_NEEDS_MAPPER")
        saved = torch.load(args.mapper, map_location="cpu")
    module = Mapper(saved, args.rank).to(device)
    if rank == 0:
        print(f"[refit] init={args.init} rank={args.rank} k={saved['k']} trainable_parameters={module.trainable()}",
              flush=True)
    mapper = DistributedDataParallel(module, device_ids=[device.index])
    optimizer = torch.optim.AdamW([p for p in mapper.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    windows = {d: np.load(args.corpus / f"train-{d}.npy", mmap_mode="r") for d in DOMAINS}
    with args.items.open() as handle:
        items = [json.loads(line) for line in handle]
    suffix = layout.encode(tokenizer, THINK_OFF) if args.think_off else []
    order = list(range(len(items)))
    random.Random(args.seed).shuffle(order)
    needed = args.steps * args.batch * world
    if needed > len(order):
        raise SystemExit(f"REFIT_NOT_ENOUGH_ITEMS need={needed} have={len(order)}")
    log_path = args.out / "metrics.jsonl"
    started = time.time()
    for step in range(args.steps):
        start = (step * world + rank) * args.batch
        chosen = [items[i] for i in order[start:start + args.batch]]
        examples = [(layout.prefix_ids(tokenizer, windows[i["domain"]][i["row"]].tolist()),
                     layout.encode(tokenizer, layout.question_text(i)) + suffix, i["answer_ids"]) for i in chosen]
        batch = collate(examples, tokenizer.pad_token_id, device)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        loss = step_loss(source, compactor, receiver, mapper, batch, layout.PREFIX_TOKENS, args.slots)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = loss.detach().clone()
        dist.all_reduce(value)
        value = value.item() / world
        if rank == 0:
            with log_path.open("a") as handle:
                handle.write(json.dumps({"step": step + 1, "loss": value, "seconds": time.time() - started}) + "\n")
            if (step + 1) % 10 == 0 or step == 0:
                print(f"[refit {time.strftime('%H:%M:%S')}] step {step + 1}/{args.steps} loss {value:.4f}", flush=True)
    if rank == 0:
        exported = mapper.module.export()
        torch.save({"k": saved["k"], "lam": saved["lam"], "source": args.source, "receiver": args.receiver,
                    "init": args.init, "rank": args.rank, "trainable_parameters": module.trainable(),
                    "refit_steps": args.steps, "keys": exported["keys"], "values": exported["values"]},
                   args.out / "mapper.pt")
        print(f"[refit] wrote {args.out / 'mapper.pt'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
