"""Turn written candidate questions into training and evaluation items on the GPU node.

Drops questions the base model answers correctly without the document, records the base model's
full-document rationale (training keeps only correct, finished rationales), and balances evaluation
answer letters within each domain."""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout

DOMAINS = ["financial", "gutenberg", "legal", "code"]


def log(message):
    print(f"[questions {time.strftime('%H:%M:%S')}] {message}", flush=True)


def candidate_items(path, split, seed):
    items = []
    with path.open() as handle:
        documents = sorted(map(json.loads, handle), key=lambda d: (d["domain"], d["row"]))
    for document in documents:
        for number, question in enumerate(document["questions"]):
            index = len(items)
            options, gold = layout.arrange_options(question["correct"], question["distractors"], index, seed)
            items.append({"id": f"{split}-{document['domain']}-{document['row']}-{number}",
                          "domain": document["domain"], "row": document["row"], "type": question["type"],
                          "sections": question["sections"], "writer": document["writer"],
                          "question": question["question"], "options": options, "gold": gold})
    return items


def worker(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    header = layout.header_ids(tokenizer)
    items = candidate_items(args.candidates, args.split, args.seed)[args.rank::args.world]
    stats = Counter(candidates=len(items))
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=layout.PREFIX_TOKENS + 1024,
              enable_prefix_caching=True, gpu_memory_utilization=0.85, seed=args.seed)

    blind = [TokensPrompt(prompt_token_ids=header + layout.encode(tokenizer, layout.question_text(i, letter_only=True)))
             for i in items]
    answers = llm.generate(blind, SamplingParams(temperature=0.0, max_tokens=3))
    kept = []
    for item, answer in zip(items, answers, strict=True):
        text = answer.outputs[0].text.strip()
        item["nocontext_letter"] = text[:1] if text[:1] in layout.LETTERS else None
        if item["nocontext_letter"] == item["gold"]:
            stats["answerable_without_context"] += 1
        else:
            kept.append(item)

    windows = {d: np.load(args.corpus / f"{args.split}-{d}.npy", mmap_mode="r") for d in DOMAINS}
    full = [TokensPrompt(prompt_token_ids=layout.prefix_ids(tokenizer, windows[i["domain"]][i["row"]].tolist())
                         + layout.encode(tokenizer, layout.question_text(i))) for i in kept]
    rationales = llm.generate(full, SamplingParams(temperature=0.0, max_tokens=320))
    result = []
    for item, rationale in zip(kept, rationales, strict=True):
        completion = rationale.outputs[0]
        item["teacher_text"] = completion.text
        item["teacher_letter"] = layout.parse_letter(completion.text)
        item["teacher_finished"] = completion.finish_reason == "stop"
        if args.split == "train":
            if item["teacher_letter"] != item["gold"] or not item["teacher_finished"]:
                stats["teacher_wrong_or_unfinished"] += 1
                continue
            item["answer_ids"] = [t for t in completion.token_ids if t != end_id] + [end_id]
        result.append(item)
    stats["kept"] = len(result)
    with (args.out / f"{args.split}-rank{args.rank}.jsonl").open("w") as handle:
        for item in result:
            handle.write(json.dumps(item) + "\n")
    (args.out / f"{args.split}-rank{args.rank}.stats.json").write_text(json.dumps(stats, indent=2))
    log(f"rank {args.rank}: {dict(stats)}")


def balance(items, per_domain):
    chosen = []
    for domain in DOMAINS:
        pools = [[i for i in items if i["domain"] == domain and i["gold"] == letter] for letter in layout.LETTERS]
        per_letter = min(per_domain // 4, *(len(pool) for pool in pools))
        for pool in pools:
            chosen.extend(pool[:per_letter])
    return chosen


def merge(args):
    items, stats = [], Counter()
    for rank in range(args.world):
        with (args.out / f"{args.split}-rank{rank}.jsonl").open() as handle:
            items.extend(json.loads(line) for line in handle)
        stats.update(json.loads((args.out / f"{args.split}-rank{rank}.stats.json").read_text()))
    items.sort(key=lambda i: i["id"])
    if args.split == "eval":
        items = balance(items, args.eval_per_domain)
    with (args.out / f"items-{args.split}.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    summary = {"stats": dict(stats), "items": len(items),
               "by_domain": dict(Counter(i["domain"] for i in items)),
               "by_type": dict(Counter(i["type"] for i in items)),
               "gold_letters": dict(Counter(i["gold"] for i in items)),
               "teacher_accuracy": sum(i["teacher_letter"] == i["gold"] for i in items) / max(len(items), 1)}
    (args.out / f"items-{args.split}.summary.json").write_text(json.dumps(summary, indent=2))
    log(f"merged {args.split}: {json.dumps(summary)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--eval-per-domain", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rank is not None:
        worker(args)
        return
    procs = []
    for rank in range(args.world):
        env = {**os.environ, "HIP_VISIBLE_DEVICES": str(rank), "CUDA_VISIBLE_DEVICES": str(rank)}
        command = [sys.executable, __file__, *sys.argv[1:], "--rank", str(rank)]
        log_path = args.out / f"{args.split}-rank{rank}.log"
        procs.append((rank, subprocess.Popen(command, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT)))
    failed = [rank for rank, proc in procs if proc.wait() != 0]
    if failed:
        raise RuntimeError(f"QUESTIONS_WORKER_FAILED split={args.split} ranks={failed}")
    merge(args)


if __name__ == "__main__":
    main()
