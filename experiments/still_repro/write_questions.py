import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout

DOMAINS = ["financial", "gutenberg", "legal", "code"]
DESCRIPTIONS = {"financial": "a company's SEC annual filing", "gutenberg": "a book from Project Gutenberg",
                "legal": "a U.S. court opinion", "code": "a software repository's source files"}
SECTIONS = 8
PER_DOCUMENT = 5
SECTION_MENTION = re.compile(r"\bS\d\b|\bsections?\b|\bthe (document|passage|text|excerpt)\b", re.IGNORECASE)
INSTRUCTIONS = """You write reading-comprehension questions for a memory test. A small model will read the whole document, and must answer from memory of it.

Write every question the way one engineer would ask another: plain words, short sentences, no jargon beyond the names the document itself uses. Keep answer options short (a few words, a number, or a name), and make all four options similar in length and style."""
PROMPT = """Below is {description}, split into {sections} numbered sections.

{document}

Write exactly {count} multiple-choice questions about this document:
- 2 "detail" questions. Each asks about one specific fact (a number, name, date, condition, or what someone did) stated in a single section. The three wrong options must also look like real details from this document, such as a nearby number or another name that appears in it, so that guessing or skimming does not work.
- 3 "connect" questions. Each needs facts from two different sections that are at least two sections apart (for example S2 and S6). A reader who remembers only one of the two sections should not be able to answer.

Rules for every question:
- The correct answer must be stated in, or directly follow from, the document.
- It must not be answerable from general knowledge.
- Name the people, companies, functions, or parties involved. Never say "the document", "the passage", or "section".
- Exactly one option is correct.

Return only JSON in this form:
{{"questions": [{{"type": "detail" or "connect", "sections": [section numbers used], "question": "...", "correct": "...", "distractors": ["...", "...", "..."]}}]}}"""


def log(message):
    print(f"[write {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sectioned(tokenizer, ids):
    size = -(-len(ids) // SECTIONS)
    parts = [tokenizer.decode(ids[i * size:(i + 1) * size]) for i in range(SECTIONS)]
    return "\n\n".join(f"[S{i + 1}]\n{part}" for i, part in enumerate(parts))


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def validate(entry):
    if not isinstance(entry, dict):
        return None
    kind, sections = entry.get("type"), entry.get("sections")
    question, correct, distractors = entry.get("question"), entry.get("correct"), entry.get("distractors")
    if kind not in ("detail", "connect") or not isinstance(sections, list) or not isinstance(distractors, list):
        return None
    if not isinstance(question, str) or not isinstance(correct, str) or len(distractors) != 3:
        return None
    try:
        sections = sorted({int(s) for s in sections})
    except (TypeError, ValueError):
        return None
    if not sections or not all(1 <= s <= SECTIONS for s in sections):
        return None
    if kind == "connect" and (len(sections) < 2 or sections[-1] - sections[0] < 2):
        return None
    options = [correct.strip(), *[str(d).strip() for d in distractors]]
    if len({o.lower() for o in options}) != 4 or not all(0 < len(o) <= 200 for o in options):
        return None
    if not 10 <= len(question.strip()) <= 400:
        return None
    if any(SECTION_MENTION.search(text) for text in (question, *options)):
        return None
    return {"type": kind, "sections": sections, "question": question.strip(), "correct": options[0],
            "distractors": options[1:]}


async def call(client, args, prompt, attempt_log):
    body = {"model": args.model, "instructions": INSTRUCTIONS, "store": False, "stream": True,
            "reasoning": {"effort": args.effort},
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    for attempt in range(4):
        text = []
        try:
            async with client.stream("POST", args.endpoint, json=body, timeout=600) as response:
                if response.status_code != 200:
                    detail = (await response.aread())[:300]
                    raise RuntimeError(f"HTTP {response.status_code} {detail!r}")
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and '"response.output_text.delta"' in line:
                        text.append(json.loads(line[6:]).get("delta", ""))
            return "".join(text)
        except (httpx.HTTPError, RuntimeError) as error:
            attempt_log.append(str(error)[:300])
            await asyncio.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"WRITE_CALL_FAILED attempts={attempt_log[-4:]}")


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    done = set()
    if args.out.exists():
        with args.out.open() as handle:
            done = {(r["domain"], r["row"]) for r in map(json.loads, handle)}
    jobs = []
    for domain in DOMAINS:
        windows = np.load(args.corpus / f"{args.split}-{domain}.npy", mmap_mode="r")
        rows = list(range(min(args.per_domain, windows.shape[0])))
        jobs.extend((domain, row, windows) for row in rows if (domain, row) not in done)
    random.Random(args.seed).shuffle(jobs)
    log(f"{len(jobs)} documents to write ({len(done)} already done) with {args.model} effort={args.effort}")
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counts = {"documents": 0, "questions": 0, "rejected": 0, "failed": 0}

    async def one(client, domain, row, windows):
        async with semaphore:
            prompt = PROMPT.format(description=DESCRIPTIONS[domain], sections=SECTIONS, count=PER_DOCUMENT,
                                   document=sectioned(tokenizer, windows[row].tolist()))
            attempts = []
            try:
                text = await call(client, args, prompt, attempts)
            except RuntimeError as error:
                counts["failed"] += 1
                log(f"FAILED {domain}/{row}: {error}")
                return
            parsed = extract_json(text) or {}
            raw = parsed.get("questions", []) if isinstance(parsed, dict) else []
            kept = [q for q in map(validate, raw) if q]
            async with lock:
                counts["documents"] += 1
                counts["questions"] += len(kept)
                counts["rejected"] += len(raw) - len(kept) + max(0, PER_DOCUMENT - len(raw))
                with args.out.open("a") as handle:
                    handle.write(json.dumps({"domain": domain, "row": row, "writer": args.model,
                                             "questions": kept, "retries": len(attempts)}) + "\n")
                if counts["documents"] % 50 == 0:
                    log(json.dumps(counts))

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        await asyncio.gather(*(one(client, *job) for job in jobs))
    log(f"finished {json.dumps(counts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--per-domain", type=int, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--endpoint", default="http://127.0.0.1:10100/v1/responses")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
