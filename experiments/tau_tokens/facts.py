"""Short-answer τ-banking eval in TRACE's form. After the documents, ask for one exact value that only the documents
decide (a fee, limit, rate, age or time window) and score the gold answer's tokens under each documents condition.
Two versions share the questions:
  original  the task's required documents as published
  edited    every questioned value replaced by a different value of the same form, so the gold answer contradicts
            anything the model knew beforehand; each question is scored against the edited value ("edited") and
            against the original value ("stale")
Same 62 tasks, documents block, slots and conditions as conditions.py. The prompt is the system turn (documents block,
then tau2's no-knowledge instructions and policy), one user question, and the assistant's answer.

Run inside a tau2 v1.0.1 checkout:
  cd <tau2-bench> && uv run --with torch --with transformers --with httpx python <this> write \
      --scores <tau_tokens results/scores.jsonl> --out <tau_tokens results/facts/questions.jsonl>
  cd <tau2-bench> && uv run --with torch --with transformers --with httpx python <this> build \
      --questions <questions.jsonl> --scores <tau_tokens results/scores.jsonl> --out <dir>
"""
import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conditions
from build import RETRIEVAL, TAU2_COMMIT

VALUE = re.compile(r"(?<![\w$.,/:-])(\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)(?![\w%/:]|[.,]\d)")
KINDS = ("fee", "limit", "rate", "eligibility", "time", "other")
PER_TASK = 8
EDIT_SEED = 17
ASK = "Reply with only the value, written exactly as it appears in the bank's documents."
WRITER_INSTRUCTIONS = """You write short-answer questions for a memory test. A small model reads a bank's policy documents once and must answer from them. Write the way one engineer asks another: plain words, short sentences."""
WRITER_PROMPT = """Below are the policy documents a Rho-Bank customer service agent needs for one customer's case.

{documents}

Each of these values appears exactly once in the documents:
{candidates}

Write up to {count} questions. Each question's answer is exactly one value from the list above.
- Pick values that decide what an agent does or tells a customer: fees, limits, rates, eligibility ages or thresholds, time windows and deadlines. Skip values that are only examples, phone numbers, identifiers or counts of items in a list.
- Use a different value for each question, and spread the questions across the documents when you can.
- Name the product, account and situation precisely, using the documents' own names, so that exactly one value answers the question.
- Never put the answer in the question. Ask for the value as stated; never ask for a calculation.

Return only JSON in this form:
{{"questions": [{{"answer": "<value copied exactly from the list>", "kind": "fee" | "limit" | "rate" | "eligibility" | "time" | "other", "question": "..."}}]}}"""


def log(message):
    print(f"[facts {time.strftime('%H:%M:%S')}] {message}", flush=True)


def matches(text):
    return [(m.start(1), m.end(1), m.group(1)) for m in VALUE.finditer(text)]


def parse(value):
    prefix = "$" if value.startswith("$") else ""
    suffix = "%" if value.endswith("%") else ""
    digits = value[len(prefix):len(value) - len(suffix)]
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    return prefix, Decimal(digits.replace(",", "")), decimals, "," in digits, suffix


def render(prefix, amount, decimals, comma, suffix):
    return prefix + (f"{amount:,.{decimals}f}" if comma else f"{amount:.{decimals}f}") + suffix


def candidates(block):
    """Values that occur exactly once in the block and round-trip through parse/render, without zeros, years and
    bare 0/1."""
    counts = Counter(value for _, _, value in matches(block))
    kept = []
    for value, count in counts.items():
        prefix, amount, decimals, comma, suffix = parse(value)
        bare = not prefix and not suffix and decimals == 0
        if count != 1 or amount == 0 or (bare and (amount < 2 or 1900 <= amount <= 2100)):
            continue
        if render(prefix, amount, decimals, comma, suffix) == value:
            kept.append(value)
    return kept


def rounded(amount, decimals):
    if decimals == 0:
        step = Decimal(10) ** max(0, len(str(int(amount))) - 2)
        return (amount / step).quantize(Decimal(1), ROUND_HALF_UP) * step
    if decimals == 2:
        quarter = (amount * 4).quantize(Decimal(1), ROUND_HALF_UP) / 4
        return (amount.quantize(Decimal(1), ROUND_HALF_UP) if amount >= 10 else quarter).quantize(Decimal("0.01"))
    return amount.quantize(Decimal(1).scaleb(-decimals), ROUND_HALF_UP)


def edited_value(value, taken, avoid, key):
    """A different positive value of the same form (symbol, decimals, digit grouping), 0.5-0.8x or 1.25-1.9x the
    original and rounded the way policy values are written, that is not already a value in the block and does not
    occur in the text after it (avoid). None when no such value exists."""
    prefix, amount, decimals, comma, suffix = parse(value)
    rng = random.Random(f"{EDIT_SEED}:{key}")
    for _ in range(200):
        factor = Decimal(str(round(rng.uniform(0.5, 0.8) if rng.random() < 0.5 else rng.uniform(1.25, 1.9), 3)))
        new = rounded(amount * factor, decimals)
        text = render(prefix, new, decimals, comma, suffix)
        if new > 0 and new != amount and text not in taken and text not in avoid:
            return text
    return None


def edit_block(block, answers, key, avoid=""):
    """Replace each answer's single occurrence with an edited value; every other character stays the same. Answers
    without a possible edited value are left unedited and absent from the returned edits."""
    counts = Counter(value for _, _, value in matches(block))
    for answer in answers:
        if counts[answer] != 1:
            raise SystemExit(f"NOT_A_SINGLE_OCCURRENCE task={key} answer={answer} count={counts[answer]}")
    spans = {value: (start, end) for start, end, value in matches(block)}
    taken, edits = {value for _, _, value in matches(block)}, {}
    for answer in answers:
        new = edited_value(answer, taken, avoid, f"{key}:{answer}")
        if new is not None:
            edits[answer] = new
            taken.add(new)
    edited = block
    for answer in sorted(edits, key=lambda a: spans[a][0], reverse=True):
        start, end = spans[answer]
        edited = edited[:start] + edits[answer] + edited[end:]
    found = Counter(value for _, _, value in matches(edited))
    for answer, new in edits.items():
        if found[answer] or found[new] != 1:
            raise SystemExit(f"EDIT_NOT_CLEAN task={key} answer={answer} edited={new} "
                             f"left={found[answer]} new_count={found[new]}")
    return edited, edits


def task_blocks(tau2, task_ids):
    domain = tau2 / conditions.DOMAIN
    intro, tags = conditions.block_template((domain / "prompts" / "required_docs.md").read_text(),
                                            (domain / "prompts" / "no_knowledge.md").read_text())
    tasks = {task.id: task for task in get_tasks()}
    blocks = {}
    for task_id in task_ids:
        documents = [json.loads((KNOWLEDGE_DOCUMENTS_DIR / f"{d}.json").read_text())
                     for d in tasks[task_id].required_documents]
        blocks[task_id] = conditions.documents_block(intro, tags, documents)
    return tasks, blocks


def step2_lengths(scores):
    with scores.open() as handle:
        return {r["task_id"]: r["prefix_tokens"] for r in map(json.loads, handle)}


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return None


def validate(entry, allowed):
    if not isinstance(entry, dict):
        return None
    answer, kind, question = entry.get("answer"), entry.get("kind"), entry.get("question")
    if not isinstance(answer, str) or not isinstance(question, str) or answer not in allowed or kind not in KINDS:
        return None
    bare = parse(answer)[1]
    if answer in question or f"{bare:f}" in question.replace(",", "") or not 10 <= len(question.strip()) <= 300:
        return None
    return {"answer": answer, "kind": kind, "question": question.strip()}


async def call(client, args, prompt):
    body = {"model": args.model, "instructions": WRITER_INSTRUCTIONS, "store": False, "stream": True,
            "reasoning": {"effort": args.effort},
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    errors = []
    for attempt in range(4):
        text = []
        try:
            async with client.stream("POST", args.endpoint, json=body, timeout=600) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code} {(await response.aread())[:300]!r}")
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and '"response.output_text.delta"' in line:
                        text.append(json.loads(line[6:]).get("delta", ""))
            return "".join(text)
        except (httpx.HTTPError, RuntimeError) as error:
            errors.append(str(error)[:300])
            await asyncio.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"WRITER_CALL_FAILED {errors}")


async def write(args):
    lengths = step2_lengths(args.scores)
    _, blocks = task_blocks(args.tau2, sorted(lengths))
    done = set()
    if args.out.exists():
        with args.out.open() as handle:
            done = {r["task_id"] for r in map(json.loads, handle)}
    semaphore, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    async def one(client, task_id):
        allowed = candidates(blocks[task_id])
        async with semaphore:
            prompt = WRITER_PROMPT.format(documents=blocks[task_id], count=PER_TASK,
                                          candidates="\n".join(f"- {v}" for v in allowed))
            text = await call(client, args, prompt)
        raw = (extract_json(text) or {}).get("questions", [])
        kept, seen = [], set()
        for question in (validate(entry, allowed) for entry in raw if isinstance(raw, list)):
            if question and question["answer"] not in seen:
                seen.add(question["answer"])
                kept.append(question)
        async with lock:
            with args.out.open("a") as handle:
                handle.write(json.dumps({"task_id": task_id, "writer": args.model, "candidates": len(allowed),
                                         "written": len(raw) if isinstance(raw, list) else 0,
                                         "questions": kept[:PER_TASK]}) + "\n")
        log(f"{task_id}: {len(allowed)} candidates, {len(kept)} questions kept")

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, t) for t in sorted(lengths) if t not in done))


def system_prompt(tasks, task_ids):
    prompts = {SYSTEM_PROMPT.format(domain_policy=get_environment(retrieval_variant=RETRIEVAL, task=tasks[t]).get_policy(),
                                    agent_instruction=AGENT_INSTRUCTION) for t in task_ids}
    if len(prompts) != 1:
        raise SystemExit(f"SYSTEM_PROMPT_VARIES_BY_TASK count={len(prompts)}")
    return prompts.pop()


def record(tokenizer, system, task_id, decision_id, question, answer, label, kind):
    chat = [{"role": "system", "content": system}, {"role": "user", "content": f"{question}\n{ASK}"}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    if answer in text:
        raise SystemExit(f"ANSWER_IN_PROMPT {decision_id} {answer}")
    start = len(text)
    return {"decision_id": decision_id, "task_id": task_id, "action_name": kind, "text": text + answer,
            "values": [{"path": "answer", "value": answer, "label": label, "nesting": 0, "start": start,
                        "end": start + len(answer)}]}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=KNOWLEDGE_DOCUMENTS_DIR, capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != TAU2_COMMIT:
        raise SystemExit(f"TAU2_COMMIT_MISMATCH expected={TAU2_COMMIT} actual={commit}")
    lengths = step2_lengths(args.scores)
    with args.questions.open() as handle:
        rows = sorted(map(json.loads, handle), key=lambda r: r["task_id"])
    tasks, blocks = task_blocks(args.tau2, [r["task_id"] for r in rows])
    system = system_prompt(tasks, [r["task_id"] for r in rows])
    tokenizer = AutoTokenizer.from_pretrained(conditions.TOKENIZER)
    versions = {"original": ([], []), "edited": ([], [])}
    dropped, edits_log = Counter(), []
    for row in rows:
        task_id, block = row["task_id"], blocks[row["task_id"]]
        candidates_kept = []
        for number, q in enumerate(row["questions"]):
            if q["answer"] in system:
                dropped["answer_in_system_prompt"] += 1
                continue
            candidates_kept.append({**q, "id": f"{task_id}:q{number}"})
        avoid = "\n".join([system, *(q["question"] for q in candidates_kept)])
        edited_block, edits = edit_block(block, [q["answer"] for q in candidates_kept], task_id, avoid=avoid)
        dropped["no_edited_value"] += len(candidates_kept) - len(edits)
        questions = [q for q in candidates_kept if q["answer"] in edits]
        if not questions:
            dropped["task_without_questions"] += 1
            continue
        for name, text in (("original", block), ("edited", edited_block)):
            header_ids, prefix_ids = conditions.task_prefix(tokenizer, text)
            if name == "original" and len(prefix_ids) != lengths[task_id]:
                raise SystemExit(f"BLOCK_DIFFERS_FROM_STEP2 task={task_id} tokens={len(prefix_ids)} "
                                 f"step2={lengths[task_id]}")
            prefixes, inputs = versions[name]
            prefixes.append({"task_id": task_id, "block": text, "header_ids": header_ids, "prefix_ids": prefix_ids})
            for q in questions:
                if edits[q["answer"]] in q["question"]:
                    raise SystemExit(f"EDITED_VALUE_IN_QUESTION {q['id']}")
                golds = [("original", q["answer"])] if name == "original" else \
                    [("edited", edits[q["answer"]]), ("stale", q["answer"])]
                for label, answer in golds:
                    decision_id = q["id"] if name == "original" else f"{q['id']}:{label}"
                    inputs.append(conditions.decision_inputs(
                        tokenizer, record(tokenizer, system, task_id, decision_id, q["question"], answer, label,
                                          q["kind"]), header_ids, prefix_ids, text))
        edits_log += [{"id": q["id"], "kind": q["kind"], "question": q["question"], "original": q["answer"],
                       "edited": edits[q["answer"]]} for q in questions]

    manifest = {"tau2_commit": TAU2_COMMIT, "tokenizer": conditions.TOKENIZER, "slots": conditions.SLOTS,
                "edit_seed": EDIT_SEED, "ask": ASK, "questions_sha256": sha256(args.questions),
                "tasks": len(versions["original"][0]), "questions": len(edits_log), "dropped": dict(dropped),
                "questions_by_kind": dict(Counter(e["kind"] for e in edits_log))}
    for name, (prefixes, inputs) in versions.items():
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, items in (("prefixes.jsonl", prefixes), ("inputs.jsonl", inputs)):
            with (directory / filename).open("w") as handle:
                for item in items:
                    handle.write(json.dumps(item) + "\n")
        tokens = sorted(len(p["prefix_ids"]) for p in prefixes)
        manifest[name] = {"decisions": len(inputs), "prefix_tokens_median": tokens[len(tokens) // 2],
                          "values_with_exact_token_boundaries": sum(v["exact"] for i in inputs for v in i["values"]),
                          "prefixes_sha256": sha256(directory / "prefixes.jsonl"),
                          "inputs_sha256": sha256(directory / "inputs.jsonl")}
    with (args.out / "edits.jsonl").open("w") as handle:
        for entry in edits_log:
            handle.write(json.dumps(entry) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"{manifest['questions']} questions over {manifest['tasks']} tasks; dropped {dict(dropped)}; "
        f"original {manifest['original']['decisions']} decisions, edited {manifest['edited']['decisions']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    writer = commands.add_parser("write")
    writer.add_argument("--scores", type=Path, required=True)
    writer.add_argument("--out", type=Path, required=True)
    writer.add_argument("--model", default="gpt-6-luna")
    writer.add_argument("--effort", default="medium")
    writer.add_argument("--endpoint", default="http://127.0.0.1:10100/v1/responses")
    writer.add_argument("--concurrency", type=int, default=16)
    builder = commands.add_parser("build")
    builder.add_argument("--questions", type=Path, required=True)
    builder.add_argument("--scores", type=Path, required=True)
    builder.add_argument("--out", type=Path, required=True)
    for sub in (writer, builder):
        sub.add_argument("--tau2", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "write":
        asyncio.run(write(args))
    else:
        build(args)


if __name__ == "__main__":
    main()
