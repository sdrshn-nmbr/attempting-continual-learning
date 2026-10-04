"""Step 2 of the τ-banking token-level eval: the model inputs for each documents condition. Every condition shares
one layout. The system turn opens with the documents block, then tau2's instructions, no-knowledge policy and tools,
then the conversation up to and including the gold call. Only the documents block differs:
  none       no block
  full       the task's required documents as text, in tau2's gold-documents wording and format
  still      header plus block compressed by the trained STILL compactor into SLOTS positions
  am         header kept, block compressed by the official Attention Matching code; SLOTS positions per head, exactly
             with equal budgets or on average with per-head budgets
  streaming  the first SINKS and last SLOTS - SINKS positions of header plus block
The prefix (header plus block) and the rest are tokenized separately, so every condition sees identical token ids
after the prefix; compressed conditions continue at the uncompressed prefix length.

Build the inputs from step 1's decisions:
  uv run --no-project --with torch --with transformers python conditions.py --decisions <decisions.jsonl> \
      --tau2 <tau2-bench v1.0.1 checkout> --out <dir>
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "still_repro"))
import cache_format
from still import prefill, streaming_pairs

TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
HEADER = "<|im_start|>system\n"
SLOTS = 164
SINKS = 4
MAX_PREFIX = 8192
CONDITIONS = ("none", "full", "still", "am", "streaming")
DOMAIN = Path("data/tau2/domains/banking_knowledge")
POLICY_HEADER = "{{component:policy_header}}\n\n"
INSTRUCTIONS = "{{component:additional_instructions}}"
DOCUMENTS = "{{required_documents}}"


def log(message):
    print(f"[tau_tokens {time.strftime('%H:%M:%S')}] {message}", flush=True)


def block_template(gold, plain):
    """Split tau2's gold-documents template into the intro and the documents tags, checking that removing both
    leaves exactly the no-knowledge template."""
    if not gold.startswith(POLICY_HEADER) or INSTRUCTIONS not in gold or DOCUMENTS not in gold:
        raise SystemExit("GOLD_TEMPLATE_SHAPE")
    intro, tail = gold[len(POLICY_HEADER):].split(INSTRUCTIONS, 1)
    if not tail.startswith("\n\n") or POLICY_HEADER + INSTRUCTIONS + "\n" != plain:
        raise SystemExit("GOLD_TEMPLATE_IS_NOT_PLAIN_PLUS_BLOCK")
    return intro, tail[2:]


def documents_block(intro, tags, documents):
    """The block opening the system turn: tau2's intro, then its tags around the documents formatted as tau2's
    golden_prompt formats them, then a blank line before the instructions."""
    body = "\n\n---\n\n".join(f"## {d['title']}\n\n{d['content']}" for d in documents)
    return intro + tags.replace(DOCUMENTS, body) + "\n"


def token_span(offsets, start, end):
    covered = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not covered:
        raise SystemExit(f"VALUE_WITHOUT_TOKENS start={start} end={end}")
    first, last = covered[0], covered[-1] + 1
    return first, last, offsets[first][0] == start and offsets[last - 1][1] == end


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids


def task_prefix(tokenizer, block):
    header_ids = encode(tokenizer, HEADER)
    prefix_ids = encode(tokenizer, HEADER + block)
    if prefix_ids[:len(header_ids)] != header_ids:
        raise SystemExit("HEADER_NOT_A_TOKEN_PREFIX")
    return header_ids, prefix_ids


def decision_inputs(tokenizer, record, header_ids, prefix_ids, block):
    text = record["text"]
    if not text.startswith(HEADER):
        raise SystemExit(f"DECISION_WITHOUT_SYSTEM_HEADER {record['decision_id']}")
    rest = text[len(HEADER):]
    encoded = tokenizer(rest, add_special_tokens=False, return_offsets_mapping=True)
    rest_ids, offsets = encoded.input_ids, encoded.offset_mapping
    if header_ids + rest_ids != encode(tokenizer, text):
        raise SystemExit(f"NONE_TOKENIZATION_SPLITS_DIFFERENTLY {record['decision_id']}")
    if prefix_ids + rest_ids != encode(tokenizer, HEADER + block + rest):
        raise SystemExit(f"FULL_TOKENIZATION_SPLITS_DIFFERENTLY {record['decision_id']}")
    values = []
    for value in record["values"]:
        first, last, exact = token_span(offsets, value["start"] - len(HEADER), value["end"] - len(HEADER))
        values.append({"path": value["path"], "value": value["value"], "label": value["label"],
                       "nesting": value["nesting"], "token_start": first, "token_end": last, "exact": exact})
    return {"decision_id": record["decision_id"], "task_id": record["task_id"], "action_name": record["action_name"],
            "rest_ids": rest_ids, "values": values}


def prefix_state(model, condition, prefix, compactor=None, am=None):
    """(pairs, logical_start, bias) for one prefix record. Pairs are per-layer (keys, values) [1, H, t, D]; bias is
    the official Attention Matching beta per layer [1, H, t] or None. am is a saved compact_am.py cache; heads shorter
    than the longest are padded with beta=-inf."""
    device = next(model.parameters()).device
    if condition == "none":
        return prefill(model, torch.tensor([prefix["header_ids"]], device=device)), len(prefix["header_ids"]), None
    logical = len(prefix["prefix_ids"])
    if condition == "am":
        average = am["lengths"].float().mean().item()
        if am["header"] != len(prefix["header_ids"]) or average > SLOTS:
            raise SystemExit(f"AM_CACHE_SHAPE task={prefix['task_id']} header={am['header']} "
                             f"average_per_head={average:.1f} slots={SLOTS}")
        keys, beta, values = cache_format.unpack(am)
        pairs = [(keys[layer][None].to(device), values[layer][None].to(device)) for layer in range(keys.shape[0])]
        return pairs, logical, [beta[layer][None].to(device) for layer in range(beta.shape[0])]
    pairs = prefill(model, torch.tensor([prefix["prefix_ids"]], device=device))
    if condition == "full":
        return pairs, logical, None
    if condition == "streaming":
        return streaming_pairs(pairs, SLOTS, sinks=SINKS), logical, None
    if condition == "still":
        if compactor is None or compactor.slots != SLOTS:
            raise SystemExit(f"STILL_COMPACTOR_SLOTS expected={SLOTS}")
        with torch.no_grad():
            return compactor(model, pairs), logical, None
    raise SystemExit(f"UNKNOWN_CONDITION {condition}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--tau2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    domain = args.tau2 / DOMAIN
    intro, tags = block_template((domain / "prompts" / "required_docs.md").read_text(),
                                 (domain / "prompts" / "no_knowledge.md").read_text())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    with args.decisions.open() as handle:
        records = [json.loads(line) for line in handle]

    prefixes, excluded = {}, {}
    for task_id in dict.fromkeys(r["task_id"] for r in records):
        ids = next(r["required_documents"] for r in records if r["task_id"] == task_id)
        documents = []
        for document_id in ids:
            path = domain / "documents" / f"{document_id}.json"
            if not path.exists():
                raise SystemExit(f"REQUIRED_DOCUMENT_MISSING task={task_id} document={document_id}")
            documents.append(json.loads(path.read_text()))
        block = documents_block(intro, tags, documents)
        header_ids, prefix_ids = task_prefix(tokenizer, block)
        if len(prefix_ids) > MAX_PREFIX:
            excluded[task_id] = len(prefix_ids)
            continue
        if len(prefix_ids) <= SLOTS:
            raise SystemExit(f"PREFIX_NOT_LONGER_THAN_SLOTS task={task_id} length={len(prefix_ids)}")
        prefixes[task_id] = {"task_id": task_id, "block": block, "header_ids": header_ids, "prefix_ids": prefix_ids}

    inputs = [decision_inputs(tokenizer, r, prefixes[r["task_id"]]["header_ids"], prefixes[r["task_id"]]["prefix_ids"],
                              prefixes[r["task_id"]]["block"]) for r in records if r["task_id"] in prefixes]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("prefixes.jsonl", prefixes.values()), ("inputs.jsonl", inputs)):
        with (args.out / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    lengths = sorted(len(p["prefix_ids"]) for p in prefixes.values())
    values = [v for row in inputs for v in row["values"]]
    manifest = {"tokenizer": TOKENIZER, "slots": SLOTS, "sinks": SINKS, "max_prefix": MAX_PREFIX,
                "conditions": list(CONDITIONS), "decisions_source_sha256": sha256(args.decisions),
                "tasks": len(prefixes), "tasks_excluded_prefix_tokens": excluded, "decisions": len(inputs),
                "decisions_excluded": len(records) - len(inputs),
                "prefix_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
                "compression_at_median": round(lengths[len(lengths) // 2] / SLOTS, 1),
                "values_by_label": dict(Counter(v["label"] for v in values)),
                "values_with_exact_token_boundaries": sum(v["exact"] for v in values), "values": len(values),
                "prefixes_sha256": sha256(args.out / "prefixes.jsonl"), "inputs_sha256": sha256(args.out / "inputs.jsonl")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"{len(inputs)} decisions over {len(prefixes)} tasks; excluded {excluded}; prefix tokens {manifest['prefix_tokens']}")


if __name__ == "__main__":
    main()
