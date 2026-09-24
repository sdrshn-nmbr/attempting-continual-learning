import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout

DOMAINS = ["financial", "gutenberg", "legal", "code"]
DESCRIPTIONS = {"financial": "a company's SEC annual filing", "gutenberg": "a book from Project Gutenberg",
                "legal": "a U.S. court opinion", "code": "a software repository's source files"}
CHUNK_TOKENS = 1024
GENERATE = """Below is an excerpt from {description}.

<excerpt>
{chunk}
</excerpt>

Write one multiple-choice question that tests a specific fact, detail, number, name, or relationship stated in this excerpt. Requirements:
- The answer must be stated in, or directly supported by, the excerpt.
- It must not be answerable from general knowledge alone.
- Do not refer to "the excerpt" or "the passage"; name the entities involved so the question is clear when asked about the whole document.
- Give the correct answer and exactly three distractors that are plausible, similar in length and style, and clearly wrong according to the excerpt.

Return only a JSON object: {{"question": "...", "correct": "...", "distractors": ["...", "...", "..."]}}"""
JUDGE = """<excerpt>
{chunk}
</excerpt>

Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}

Proposed answer: {letter}

According to the excerpt, is the proposed answer correct and are all other options incorrect? Reply with exactly YES or NO."""


def log(message):
    print(f"[questions {time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_mcq(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    question, correct, distractors = data.get("question"), data.get("correct"), data.get("distractors")
    if not isinstance(question, str) or not isinstance(correct, str) or not isinstance(distractors, list):
        return None
    options = [correct.strip(), *[str(d).strip() for d in distractors]]
    if len(options) != 4 or len({o.lower() for o in options}) != 4 or not all(0 < len(o) <= 300 for o in options):
        return None
    if not 10 <= len(question.strip()) <= 500:
        return None
    return question.strip(), options[0], options[1:]


def chat(tokenizer, content):
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def worker(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    header = layout.header_ids(tokenizer)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=layout.PREFIX_TOKENS + 1024,
              enable_prefix_caching=True, gpu_memory_utilization=0.85, seed=args.seed)
    rng = random.Random(f"{args.seed}:{args.split}:{args.rank}")
    candidates = []
    for domain in DOMAINS:
        windows = np.load(args.corpus / f"{args.split}-{domain}.npy", mmap_mode="r")
        for row in range(args.rank, windows.shape[0], args.world):
            document = windows[row]
            span = len(document) // CHUNK_TOKENS
            for slot in sorted(rng.sample(range(span), min(args.questions_per_window, span))):
                candidates.append({"domain": domain, "row": row, "chunk_start": slot * CHUNK_TOKENS,
                                   "chunk": tokenizer.decode(document[slot * CHUNK_TOKENS:(slot + 1) * CHUNK_TOKENS])})
    log(f"rank {args.rank}: {len(candidates)} candidate chunks")
    stats = Counter(candidates=len(candidates))

    prompts = [chat(tokenizer, GENERATE.format(description=DESCRIPTIONS[c["domain"]], chunk=c["chunk"]))
               for c in candidates]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, seed=args.seed))
    items = []
    for candidate, output in zip(candidates, outputs, strict=True):
        parsed = parse_mcq(output.outputs[0].text)
        if parsed is None:
            stats["malformed"] += 1
            continue
        question, correct, distractors = parsed
        index = len(items)
        options, gold = layout.arrange_options(correct, distractors, index, f"{args.seed}:{args.rank}")
        items.append({**candidate, "id": f"{args.split}-{args.rank}-{index}", "question": question,
                      "options": options, "gold": gold})
    log(f"rank {args.rank}: {len(items)} parsed questions")

    judge_prompts = [chat(tokenizer, JUDGE.format(chunk=i["chunk"], question=i["question"], a=i["options"][0],
                                                  b=i["options"][1], c=i["options"][2], d=i["options"][3],
                                                  letter=i["gold"])) for i in items]
    verdicts = llm.generate(judge_prompts, SamplingParams(temperature=0.0, max_tokens=3))
    kept = []
    for item, verdict in zip(items, verdicts, strict=True):
        item["judge"] = verdict.outputs[0].text.strip()
        if item["judge"].upper().startswith("YES"):
            kept.append(item)
        else:
            stats["judge_rejected"] += 1
    items = kept

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
    items = kept

    windows = {d: np.load(args.corpus / f"{args.split}-{d}.npy", mmap_mode="r") for d in DOMAINS}
    items.sort(key=lambda i: (i["domain"], i["row"]))
    full = [TokensPrompt(prompt_token_ids=layout.prefix_ids(tokenizer, windows[i["domain"]][i["row"]].tolist())
                         + layout.encode(tokenizer, layout.question_text(i))) for i in items]
    rationales = llm.generate(full, SamplingParams(temperature=0.0, max_tokens=320))
    result = []
    for item, rationale in zip(items, rationales, strict=True):
        completion = rationale.outputs[0]
        ids = [t for t in completion.token_ids if t != end_id]
        item["teacher_text"] = completion.text
        item["teacher_letter"] = layout.parse_letter(completion.text)
        item["teacher_finished"] = completion.finish_reason == "stop"
        if args.split == "train":
            if item["teacher_letter"] != item["gold"] or not item["teacher_finished"]:
                stats["teacher_wrong_or_unfinished"] += 1
                continue
            item["answer_ids"] = ids + [end_id]
        item.pop("chunk")
        result.append(item)
    stats["kept"] = len(result)
    shard = args.out / f"{args.split}-rank{args.rank}.jsonl"
    with shard.open("w") as handle:
        for item in result:
            handle.write(json.dumps(item) + "\n")
    (args.out / f"{args.split}-rank{args.rank}.stats.json").write_text(json.dumps(stats, indent=2))
    log(f"rank {args.rank}: kept {len(result)} {dict(stats)}")


def merge(args):
    items = []
    stats = Counter()
    for rank in range(args.world):
        with (args.out / f"{args.split}-rank{rank}.jsonl").open() as handle:
            items.extend(json.loads(line) for line in handle)
        stats.update(json.loads((args.out / f"{args.split}-rank{rank}.stats.json").read_text()))
    items.sort(key=lambda i: i["id"])
    if args.split == "eval":
        capped = []
        for domain in DOMAINS:
            pools = [[i for i in items if i["domain"] == domain and i["gold"] == letter] for letter in layout.LETTERS]
            per_letter = min(args.eval_per_domain // 4, *(len(pool) for pool in pools))
            for pool in pools:
                capped.extend(pool[:per_letter])
        items = capped
    with (args.out / f"items-{args.split}.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    summary = {"stats": dict(stats), "items": len(items),
               "by_domain": dict(Counter(i["domain"] for i in items)),
               "gold_letters": dict(Counter(i["gold"] for i in items)),
               "teacher_accuracy": sum(i.get("teacher_letter") == i["gold"] for i in items) / max(len(items), 1)}
    (args.out / f"items-{args.split}.summary.json").write_text(json.dumps(summary, indent=2))
    log(f"merged {args.split}: {json.dumps(summary)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--questions-per-window", type=int, default=4)
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
