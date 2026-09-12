import argparse
import random
import string
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from prospective_data import PROMPT, ROLES, load_design, seed_for, validate_data
from protocol import ROOT, file_hash, normalized_hash, read_json, write_json


def prepare(cache, tokenizer_path, config_path):
    config, spec = load_design(config_path)
    splits = {
        name: Dataset.from_file(str(cache / f"clinc_oos-{name}.arrow")) for name in ("train", "validation", "test")
    }
    if {key: len(value) for key, value in splits.items()} != {"train": 7600, "validation": 3100, "test": 5500}:
        raise ValueError("PROSPECTIVE_DATASET_COUNTS")
    names = splits["train"].features["intent"].names
    if len(names) != 151 or names.count("oos") != 1:
        raise ValueError("PROSPECTIVE_INTENT_SCHEMA")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    code_ids = [tokenizer.encode(code, add_special_tokens=False) for code in string.ascii_uppercase[:16]]
    if any(len(ids) != 1 for ids in code_ids) or len(set(ids[0] for ids in code_ids)) != 16:
        raise ValueError("PROSPECTIVE_CODE_TOKENIZATION")
    codes = [ids[0] for ids in code_ids]
    grouped = {split: defaultdict(list) for split in splits}
    reserved = set()
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            text_hash = normalized_hash(row["text"])
            if text_hash in reserved:
                continue
            reserved.add(text_hash)
            grouped[split][names[row["intent"]]].append((index, row["text"]))
    candidates = sorted(name for name in names if name != "oos")
    excluded = {
        name: {split: len(grouped[split][name]) for split in splits}
        for name in candidates
        if any(len(grouped[split][name]) < count for split, count in (("train", 40), ("validation", 20), ("test", 20)))
    }
    intents = [name for name in candidates if name not in excluded]
    random.Random(spec["data_seed"]).shuffle(intents)
    seen_text, seen_tokens = set(), set()

    def select(intent, source_split, count, salt, code=None):
        candidates = grouped[source_split][intent].copy()
        random.Random(seed_for(spec["data_seed"], intent, source_split, salt)).shuffle(candidates)
        selected = []
        for index, text in candidates:
            text_hash = normalized_hash(text)
            ids = tokenizer.encode(PROMPT.format(text=text), add_special_tokens=True)
            if text_hash in seen_text or tuple(ids) in seen_tokens or len(ids) > spec["runtime"]["max_length"]:
                continue
            selected.append(
                {
                    "intent": intent,
                    "text": text,
                    "text_sha256": text_hash,
                    "input_ids": ids,
                    "source_split": source_split,
                    "source_index": index,
                    "code": code,
                    "target": None if code is None else codes[code],
                }
            )
            seen_text.add(text_hash)
            seen_tokens.add(tuple(ids))
            if len(selected) == count:
                return selected
        raise ValueError(f"PROSPECTIVE_INSUFFICIENT_ROWS {intent=} {source_split=} {salt=} {len(selected)=}")

    streams = {}
    for index, stream in enumerate(spec["streams"]):
        chosen = intents[16 * index : 16 * (index + 1)]
        mapping = list(range(16))
        random.Random(seed_for(stream["seed"], "codes")).shuffle(mapping)
        old, new = chosen[:8], chosen[8:]
        random.Random(seed_for(stream["seed"], "old_order")).shuffle(old)
        random.Random(seed_for(stream["seed"], "new_order")).shuffle(new)
        units = {}
        for intent, code in zip(chosen, mapping, strict=True):
            units[intent] = {role: select(intent, split, count, role, code) for role, (split, count) in ROLES.items()}
        streams[stream["id"]] = {**stream, "old": old, "new": new, "units": units}
    background = intents[16 * len(streams) : 16 * len(streams) + spec["lens"]["background_intents"]]
    data = {
        "codes": codes,
        "pad_token_id": tokenizer.pad_token_id,
        "prompt": PROMPT,
        "streams": streams,
        "lens_fit": [row for intent in background for row in select(intent, "train", 32, "lens_fit")],
        "lens_check": [row for intent in background for row in select(intent, "train", 8, "lens_check")],
        "source": {
            "dataset": "clinc/clinc_oos",
            "subset": "small",
            "paper": "https://aclanthology.org/D19-1131/",
            "scope": "In-scope intent classification only; cached Arrow files are content-pinned below.",
            "arrow_sha256": {
                f"clinc_oos-{split}.arrow": file_hash(cache / f"clinc_oos-{split}.arrow") for split in splits
            },
            "tokenizer_sha256": {
                name: file_hash(tokenizer_path / name) for name in ("tokenizer.json", "tokenizer_config.json")
            },
            "selection_seed": spec["data_seed"],
        },
        "base_manifest": read_json(ROOT / "inputs/cohort.json")["base_manifest"],
    }
    data["source"]["excluded_before_selection_for_insufficient_unique_rows"] = excluded
    validate_data(data, spec)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.cache, args.tokenizer_path, args.config)
    write_json(args.output, data)
    print(
        {
            "dataset_sha256": file_hash(args.output),
            "streams": list(data["streams"]),
            "candidate_skills": sum(len(stream["old"]) for stream in data["streams"].values()),
        }
    )


if __name__ == "__main__":
    main()
