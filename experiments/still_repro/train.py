import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_matching
import layout
from still import BudgetedStillCompactor, StillCompactor, budget_counts, continue_from, prefill, support_kl

DOMAINS = ["financial", "gutenberg", "legal", "code"]


class Examples:
    def __init__(self, corpus, items_path, tokenizer, split):
        with open(items_path) as handle:
            self.items = [json.loads(line) for line in handle]
        self.windows = {d: np.load(corpus / f"{split}-{d}.npy", mmap_mode="r") for d in DOMAINS}
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def get(self, index):
        item = self.items[index]
        prefix = layout.prefix_ids(self.tokenizer, self.windows[item["domain"]][item["row"]].tolist())
        return prefix, layout.encode(self.tokenizer, layout.question_text(item)), item["answer_ids"]


def collate(examples, pad_id, device):
    prefixes = torch.tensor([prefix for prefix, _, _ in examples], device=device)
    sequences = [question + answer for _, question, answer in examples]
    width = max(len(s) for s in sequences)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long, device=device)
    rows, columns, gold = [], [], []
    for row, ((_, question, answer), sequence) in enumerate(zip(examples, sequences, strict=True)):
        ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
        for offset in range(len(question), len(sequence)):
            rows.append(row)
            columns.append(offset - 1)
            gold.append(sequence[offset])
    return prefixes, ids, (torch.tensor(rows, device=device), torch.tensor(columns, device=device)), \
        torch.tensor(gold, device=device)


def step_loss(model, compactor, batch, top_k):
    prefixes, ids, (rows, columns), gold = batch
    pairs = prefill(model, prefixes)
    with torch.no_grad():
        teacher = continue_from(model, pairs, ids, layout.PREFIX_TOKENS).logits[rows, columns]
    compact = compactor(model, pairs)
    compact, betas = compact if isinstance(compact, tuple) else (compact, None)
    attention_matching.STATE.bias = betas
    try:
        student = continue_from(model, compact, ids, layout.PREFIX_TOKENS).logits[rows, columns]
    finally:
        attention_matching.STATE.bias = None
    return support_kl(teacher, student, gold, top_k).mean()


def learning_rate(step, peak, warmup, start=1e-6):
    return start + (peak - start) * min(1.0, step / warmup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=164)
    parser.add_argument("--budgets", type=Path, help="per-head slot counts from budget_counts.py; replaces --slots")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=17)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    def log(message):
        if rank == 0:
            print(f"[train {time.strftime('%H:%M:%S')}] {message}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    attention = attention_matching.NAME if args.budgets else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation=attention).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = BudgetedStillCompactor(model.config, budget_counts(args.budgets)) if args.budgets \
        else StillCompactor(model.config, args.slots)
    compactor = DistributedDataParallel(module.to(device), device_ids=[device.index])
    decay = [p for p in compactor.parameters() if p.ndim >= 2]
    other = [p for p in compactor.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95))
    data = Examples(args.corpus, args.items, tokenizer, "train")
    order = list(range(len(data)))
    random.Random(args.data_seed).shuffle(order)
    validation, training = order[:args.validation], order[args.validation:]
    random.Random(args.seed).shuffle(training)
    global_batch = args.micro * world
    slots = f"budgets={args.budgets} mean_slots={module.mean_slots():.2f} width={module.width}" if args.budgets \
        else f"slots={args.slots}"
    log(f"items={len(data)} train={len(training)} validation={len(validation)} world={world} "
        f"global_batch={global_batch} {slots} attention={attention} "
        f"parameters={sum(p.numel() for p in compactor.parameters())}")
    if args.steps * global_batch > len(training):
        log(f"WARNING epochs={args.steps * global_batch / len(training):.2f} (training items reused)")
    metrics = (args.out / "metrics.jsonl").open("a") if rank == 0 else None

    def validate(step):
        compactor.eval()
        total = torch.zeros(2, device=device)
        with torch.no_grad():
            for start in range(rank * args.micro, len(validation), world * args.micro):
                batch = collate([data.get(i) for i in validation[start:start + args.micro]], tokenizer.pad_token_id, device)
                total += torch.stack([step_loss(model, compactor, batch, args.top_k), torch.ones((), device=device)])
        dist.all_reduce(total)
        compactor.train()
        value = (total[0] / total[1]).item()
        log(f"step {step} validation_kl={value:.4f}")
        if metrics:
            metrics.write(json.dumps({"step": step, "validation_kl": value}) + "\n")
            metrics.flush()

    validate(0)
    started = time.time()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.lr, args.warmup)
        window = [training[(i + (step - 1) * global_batch) % len(training)] for i in range(global_batch)]
        mine = window[rank * args.micro:(rank + 1) * args.micro]
        loss = step_loss(model, compactor, collate([data.get(i) for i in mine], tokenizer.pad_token_id, device), args.top_k)
        if not torch.isfinite(loss):
            raise RuntimeError(f"TRAIN_NONFINITE_LOSS step={step} rank={rank} value={loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(compactor.parameters(), 1.0)
        optimizer.step()
        reduced = loss.detach().clone()
        dist.all_reduce(reduced)
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - started
            record = {"step": step, "loss": reduced.item() / world, "grad_norm": norm.item(),
                      "lr": optimizer.param_groups[0]["lr"], "seconds_per_step": elapsed / step,
                      "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30}
            log(json.dumps(record))
            if metrics:
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
        if step % args.checkpoint_every == 0 or step == args.steps:
            if rank == 0:
                path = args.out / f"compactor-{step}.pt"
                torch.save(compactor.module.state_dict(), path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (args.out / f"compactor-{step}.sha256").write_text(digest + "\n")
                log(f"checkpoint {path} sha256={digest}")
            validate(step)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
