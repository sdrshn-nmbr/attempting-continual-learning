import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout

SERVER = "https://datasets-server.huggingface.co/parquet?dataset="
CODE_LANGUAGES = ["Python-all", "Java-all", "JavaScript-all", "C++-all", "Go-all", "TypeScript-all"]
DOMAINS = {
    "financial": {"dataset": "PleIAs/SEC", "configs": ["default"], "per_source": 3, "skip": 0},
    "gutenberg": {"dataset": "emozilla/pg19", "configs": ["default"], "per_source": 4, "skip": 2048},
    "legal": {"dataset": "harvard-lil/cold-cases", "configs": ["default"], "per_source": 2, "skip": 0},
    "code": {"dataset": "codeparrot/github-code-clean", "configs": CODE_LANGUAGES, "per_source": 2, "skip": 0},
}


def log(message):
    print(f"[corpus {time.strftime('%H:%M:%S')}] {message}", flush=True)


def shard_urls(dataset, configs):
    listing = requests.get(SERVER + quote(dataset, safe=""), timeout=60)
    listing.raise_for_status()
    files = [f for f in listing.json()["parquet_files"] if f["split"] == "train" and f["config"] in configs]
    by_config = {config: [f for f in files if f["config"] == config] for config in configs}
    ordered = []
    for index in range(max(len(v) for v in by_config.values())):
        ordered.extend(v[index] for v in by_config.values() if index < len(v))
    if not ordered:
        raise RuntimeError(f"CORPUS_NO_SHARDS dataset={dataset} configs={configs}")
    return ordered


def download(entry, raw):
    target = raw / hashlib.sha256(entry["url"].encode()).hexdigest()[:16] / entry["filename"]
    if target.exists() and target.stat().st_size == entry["size"]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    with requests.get(entry["url"], stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1 << 22):
                handle.write(chunk)
    if partial.stat().st_size != entry["size"]:
        raise RuntimeError(f"CORPUS_DOWNLOAD_SIZE {entry['url']} got={partial.stat().st_size} want={entry['size']}")
    partial.rename(target)
    log(f"downloaded {entry['url']} ({entry['size'] / 1e6:.0f} MB)")
    return target


def financial_sources(path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "text"], batch_size=256):
        for source, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist()):
            if text and len(text) > 40000:
                yield f"sec:{source}", text


def gutenberg_sources(path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["url", "text"], batch_size=64):
        for source, text in zip(batch.column("url").to_pylist(), batch.column("text").to_pylist()):
            if text and len(text) > 60000:
                yield f"pg19:{source}", text


def legal_sources(path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "case_name", "opinions"], batch_size=512):
        rows = zip(batch.column("id").to_pylist(), batch.column("case_name").to_pylist(),
                   batch.column("opinions").to_pylist())
        for source, name, opinions in rows:
            body = "\n\n".join(o["opinion_text"] for o in opinions or [] if o.get("opinion_text"))
            if len(body) > 40000:
                yield f"case:{source}", f"{name}\n\n{body}"


def code_sources(path):
    repos = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=["repo_name", "path", "code"], batch_size=2048):
        for repo, file_path, code in zip(batch.column("repo_name").to_pylist(), batch.column("path").to_pylist(),
                                         batch.column("code").to_pylist()):
            if code and len(code) < 100000:
                repos.setdefault(repo, []).append((file_path, code))
    for repo in sorted(repos):
        files = sorted(repos[repo])
        text = "".join(f"=== File: {file_path} ===\n{code}\n\n" for file_path, code in files)
        if len(text) > 40000:
            yield f"repo:{repo}", text


READERS = {"financial": financial_sources, "gutenberg": gutenberg_sources, "legal": legal_sources,
           "code": code_sources}


def windows(tokenizer, texts, budget, per_source, skip):
    limit = (skip + per_source * budget) * 8
    encoded = tokenizer([text[:limit] for text in texts], add_special_tokens=False).input_ids
    result = []
    for ids in encoded:
        count = min(per_source, max(0, (len(ids) - skip) // budget))
        result.append([(skip + i * budget, ids[skip + i * budget: skip + (i + 1) * budget]) for i in range(count)])
    return result


def build_domain(name, spec, tokenizer, budget, raw, want_train, want_eval, seed):
    rng = random.Random(f"{seed}:{name}")
    train, evaluation = [], []
    for entry in shard_urls(spec["dataset"], spec["configs"]):
        path = download(entry, raw)
        sources = list(READERS[name](path))
        rng.shuffle(sources)
        for start in range(0, len(sources), 64):
            if len(train) >= want_train and len(evaluation) >= want_eval:
                break
            batch = sources[start:start + 64]
            target = evaluation if len(evaluation) < want_eval else train
            per_source = 1 if target is evaluation else spec["per_source"]
            taken = windows(tokenizer, [text for _, text in batch], budget, per_source, spec["skip"])
            for (source, _), source_windows in zip(batch, taken, strict=True):
                target.extend((source, offset, ids) for offset, ids in source_windows)
        log(f"{name}: shard {entry['filename']} -> train={len(train)} eval={len(evaluation)}")
        if len(train) >= want_train and len(evaluation) >= want_eval:
            return train[:want_train], evaluation[:want_eval]
    raise RuntimeError(f"CORPUS_EXHAUSTED domain={name} train={len(train)} eval={len(evaluation)}")


def save(out, split, name, rows):
    array = np.asarray([ids for _, _, ids in rows], dtype=np.uint32)
    np.save(out / f"{split}-{name}.npy", array)
    with (out / f"{split}-{name}.jsonl").open("w") as handle:
        for index, (source, offset, _) in enumerate(rows):
            handle.write(json.dumps({"row": index, "domain": name, "source": source, "offset": offset}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-windows", type=int, default=7500)
    parser.add_argument("--eval-windows", type=int, default=120)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    budget = layout.document_budget(tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.domains:
        train, evaluation = build_domain(name, DOMAINS[name], tokenizer, budget, args.raw,
                                         args.train_windows, args.eval_windows, args.seed)
        overlap = {s for s, _, _ in train} & {s for s, _, _ in evaluation}
        if overlap:
            raise RuntimeError(f"CORPUS_SPLIT_OVERLAP domain={name} sources={sorted(overlap)[:3]}")
        save(args.out, "train", name, train)
        save(args.out, "eval", name, evaluation)
        log(f"{name}: saved train={len(train)} eval={len(evaluation)} budget={budget}")
    manifest = {"tokenizer": args.tokenizer, "document_budget": budget, "prefix_tokens": layout.PREFIX_TOKENS,
                "header": layout.HEADER, "seed": args.seed, "domains": {n: DOMAINS[n] for n in args.domains}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
