import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from prospective_data import PROMPT, ROLES, load_design, seed_for, validate_data
from protocol import ROOT, digest, file_hash, normalized_hash, read_json, write_json


def prepare(cache, tokenizer_path, config_path):
    config, spec = load_design(config_path)
    prior = read_json(ROOT / "inputs/prospective-cohort.json")
    exclusions = set(read_json(ROOT / "minimum-budget-protocol.json")["future_holdout"]["excluded_intents"])
    for name, expected in prior["source"]["arrow_sha256"].items():
        if file_hash(cache / name) != expected:
            raise ValueError("BUDGET_PREDICTION_ARROW_IDENTITY")
    for name, expected in prior["source"]["tokenizer_sha256"].items():
        if file_hash(tokenizer_path / name) != expected:
            raise ValueError("BUDGET_PREDICTION_TOKENIZER_IDENTITY")
    splits = {
        name: Dataset.from_file(str(cache / f"clinc_oos-{name}.arrow")) for name in ("train", "validation", "test")
    }
    names = splits["train"].features["intent"].names
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    if [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(16)] != [[x] for x in prior["codes"]]:
        raise ValueError("BUDGET_PREDICTION_CODE_TOKENIZATION")
    grouped, reserved = {split: defaultdict(list) for split in splits}, set()
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            text_hash = normalized_hash(row["text"])
            if text_hash not in reserved:
                grouped[split][names[row["intent"]]].append((index, row["text"]))
                reserved.add(text_hash)
    candidates = sorted(set(names) - exclusions - {"oos"})
    insufficient = {
        name: {split: len(grouped[split][name]) for split in splits}
        for name in candidates
        if any(len(grouped[split][name]) < count for split, count in (("train", 40), ("validation", 20), ("test", 20)))
    }
    candidates = [name for name in candidates if name not in insufficient]
    random.Random(spec["data_seed"]).shuffle(candidates)
    if len(candidates) < 16 * len(spec["streams"]):
        raise ValueError("BUDGET_PREDICTION_INSUFFICIENT_FRESH_INTENTS")
    prior_rows = (
        prior["lens_fit"]
        + prior["lens_check"]
        + [
            row
            for stream in prior["streams"].values()
            for unit in stream["units"].values()
            for rows in unit.values()
            for row in rows
        ]
    )
    seen_text = {row["text_sha256"] for row in prior_rows}
    seen_tokens = {tuple(row["input_ids"]) for row in prior_rows}

    def select(intent, split, count, role, code):
        rows = grouped[split][intent].copy()
        random.Random(seed_for(spec["data_seed"], intent, split, role)).shuffle(rows)
        selected = []
        for index, text in rows:
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
                    "source_split": split,
                    "source_index": index,
                    "code": code,
                    "target": prior["codes"][code],
                }
            )
            seen_text.add(text_hash)
            seen_tokens.add(tuple(ids))
            if len(selected) == count:
                return selected
        raise ValueError(f"BUDGET_PREDICTION_INSUFFICIENT_UNIQUE_ROWS intent={intent} split={split} role={role}")

    streams = {}
    for index, stream in enumerate(spec["streams"]):
        chosen = candidates[16 * index : 16 * (index + 1)]
        mapping = list(range(16))
        random.Random(seed_for(stream["seed"], "codes")).shuffle(mapping)
        old, new = chosen[:8], chosen[8:]
        random.Random(seed_for(stream["seed"], "old_order")).shuffle(old)
        random.Random(seed_for(stream["seed"], "new_order")).shuffle(new)
        units = {
            intent: {role: select(intent, split, count, role, code) for role, (split, count) in ROLES.items()}
            for intent, code in zip(chosen, mapping, strict=True)
        }
        streams[stream["id"]] = stream | {"old": old, "new": new, "units": units}
    data = {
        key: copy.deepcopy(prior[key])
        for key in ("codes", "pad_token_id", "prompt", "base_manifest", "lens_fit", "lens_check")
    }
    data["streams"] = streams
    data["source"] = {
        "prior_dataset_sha256": digest(prior),
        "arrow_sha256": prior["source"]["arrow_sha256"],
        "tokenizer_sha256": prior["source"]["tokenizer_sha256"],
        "selection_seed": spec["data_seed"],
        "excluded_prior_intents": sorted(exclusions),
        "insufficient_unique_rows": insufficient,
        "fresh_candidate_count": len(candidates),
        "unused_candidates": candidates[16 * len(streams) :],
        "background": "Reuse only the original disjoint TRAIN background fit/check rows; fit new translators per fresh model.",
        "preparation_implementation_sha256": file_hash(Path(__file__)),
        "dataset": "clinc/clinc_oos",
        "subset": "small",
    }
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
            "fresh_candidate_count": data["source"]["fresh_candidate_count"],
        }
    )


if __name__ == "__main__":
    main()
