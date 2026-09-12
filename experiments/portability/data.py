from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from portallib import ChoiceExample

SEQUENCE_TASKS = ("sequence_a", "sequence_b", "sequence_c")
SEQUENCE_PROMPT = re.compile(
    r"Apply the (sequence_[abc]) code\.\nInput: ([0-7]) ([0-7]) ([0-7])\nOutput:"
)
OLD_AMBER_MAPPINGS = {
    "17": [3, 1, 0, 4, 5, 2, 7, 6],
    "29": [3, 7, 0, 6, 5, 4, 2, 1],
}
SPLIT_SIZES = {"train": 128, "validation": 32, "test": 64}
ROOT = Path(__file__).resolve().parent


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Example:
    id: str
    task: str
    prompt: str
    choices: tuple[str, ...]
    gold_idx: int
    group: str

    def choice(self) -> ChoiceExample:
        return ChoiceExample(self.task, self.prompt, self.choices, self.gold_idx)

    @classmethod
    def from_dict(cls, value: dict) -> Example:
        return cls(**{**value, "choices": tuple(value["choices"])})


def sequence_splits(spec: dict) -> tuple[dict[str, list[Example]], dict]:
    path = ROOT / spec["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"FROZEN_FIXTURE_HASH_MISMATCH: {path}")
    value = json.loads(raw)
    provenance = value["provenance"]
    if (
        provenance["generator"] != "seeded_three_stage_hidden_digit_permutations"
        or provenance["tasks"] != list(SEQUENCE_TASKS)
        or provenance["split_sizes_per_task"] != SPLIT_SIZES
        or provenance["fixture_seed"] != spec["fixture_seed"]
        or provenance["chance_accuracy"] != 0.25
        or provenance["excluded_old_amber_mappings"] != OLD_AMBER_MAPPINGS
    ):
        raise ValueError("SEQUENCE_FIXTURE_CONTRACT_MISMATCH")
    if (
        digest(value["splits"]) != spec["generator_sha256"]
        or provenance["generator_sha256"] != spec["generator_sha256"]
        or provenance["source_generator_sha256"] != spec["source_generator_sha256"]
        or digest(value["input_splits"]) != provenance["input_sha256"]
    ):
        raise ValueError("SEQUENCE_GENERATOR_PROVENANCE_MISMATCH")
    mappings = provenance["rules"]
    if set(mappings) != set(SEQUENCE_TASKS) or any(
        len(mapping) != 8
        or any(type(digit) is not int for digit in mapping)
        or set(mapping) != set(range(8))
        for mapping in mappings.values()
    ):
        raise ValueError("INVALID_SEQUENCE_PERMUTATION")
    map_keys = {tuple(mapping) for mapping in mappings.values()}
    if len(map_keys) != 3 or map_keys & {
        tuple(mapping) for mapping in OLD_AMBER_MAPPINGS.values()
    }:
        raise ValueError("SEQUENCE_MAPPING_FRESHNESS_FAILURE")
    if set(value["input_splits"]) != set(SPLIT_SIZES):
        raise ValueError("SEQUENCE_INPUT_SPLIT_NAMES_MISMATCH")
    inputs = {}
    for split, size in SPLIT_SIZES.items():
        triples = value["input_splits"][split]
        if len(triples) != size or any(
            len(triple) != 3
            or any(type(digit) is not int or digit not in range(8) for digit in triple)
            for triple in triples
        ):
            raise ValueError(f"INVALID_SEQUENCE_INPUTS: {split}")
        inputs[split] = {tuple(triple) for triple in triples}
        if len(inputs[split]) != size:
            raise ValueError(f"DUPLICATE_SEQUENCE_INPUT: {split}")
    if len(set.union(*inputs.values())) != sum(SPLIT_SIZES.values()):
        raise ValueError("GLOBAL_SEQUENCE_INPUT_SPLIT_OVERLAP")
    expected_names = {
        f"{task}_{split}" for task in SEQUENCE_TASKS for split in SPLIT_SIZES
    }
    if set(value["splits"]) != expected_names:
        raise ValueError("SEQUENCE_SPLIT_NAMES_MISMATCH")
    splits = {}
    audit_keys = {
        kind: {split: set() for split in SPLIT_SIZES}
        for kind in ("id", "normalized_prompt", "task_input_pair")
    }
    for task in SEQUENCE_TASKS:
        for split, size in SPLIT_SIZES.items():
            name = f"{task}_{split}"
            values = value["splits"][name]
            if len(values) != size:
                raise ValueError(f"SEQUENCE_SPLIT_SIZE_MISMATCH: {name}")
            rows, observed_inputs = [], []
            for item in values:
                if set(item) != {
                    "id",
                    "task",
                    "prompt",
                    "choices",
                    "gold_idx",
                    "group",
                }:
                    raise ValueError(f"SEQUENCE_ROW_SCHEMA_MISMATCH: {name}")
                row = Example.from_dict(item)
                matched = SEQUENCE_PROMPT.fullmatch(row.prompt)
                if row.task != task or matched is None or matched.group(1) != task:
                    raise ValueError(f"INVALID_SEQUENCE_PROMPT: {name}/{row.id}")
                digits = tuple(map(int, matched.groups()[1:]))
                if row.group != " ".join(map(str, digits)) or row.id != digest(
                    {"task": task, "input": digits}
                ):
                    raise ValueError(f"SEQUENCE_ID_INPUT_MISMATCH: {name}/{row.id}")
                if (
                    type(row.gold_idx) is not int
                    or row.gold_idx not in range(4)
                    or len(row.choices) != 4
                    or len(set(row.choices)) != 4
                    or any(
                        re.fullmatch(r" [0-7] [0-7] [0-7]", choice) is None
                        for choice in row.choices
                    )
                ):
                    raise ValueError(f"INVALID_SEQUENCE_CHOICES: {name}/{row.id}")
                gold = " " + " ".join(str(mappings[task][digit]) for digit in digits)
                if row.choices[row.gold_idx] != gold:
                    raise ValueError(f"SEQUENCE_ORACLE_MISMATCH: {name}/{row.id}")
                keys = {
                    "id": row.id,
                    "normalized_prompt": " ".join(row.prompt.split()).casefold(),
                    "task_input_pair": (task, digits),
                }
                for kind, key in keys.items():
                    if key in audit_keys[kind][split]:
                        raise ValueError(f"DUPLICATE_SEQUENCE_{kind.upper()}: {name}")
                    audit_keys[kind][split].add(key)
                rows.append(row)
                observed_inputs.append(list(digits))
            if observed_inputs != value["input_splits"][split]:
                raise ValueError(f"GLOBAL_SEQUENCE_INPUT_ALIGNMENT_MISMATCH: {name}")
            if Counter(row.gold_idx for row in rows) != {
                index: size // 4 for index in range(4)
            }:
                raise ValueError(f"UNBALANCED_SEQUENCE_GOLD_POSITIONS: {name}")
            splits[name] = rows
        task_hash = digest(
            {split: value["splits"][f"{task}_{split}"] for split in SPLIT_SIZES}
        )
        if task_hash != provenance["task_sha256"][task]:
            raise ValueError(f"SEQUENCE_TASK_HASH_MISMATCH: {task}")
    overlap = {}
    for kind, by_split in {**audit_keys, "input_triple": inputs}.items():
        overlap[kind] = {}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            count = len(by_split[left] & by_split[right])
            overlap[kind][f"{left}/{right}"] = count
            if count:
                raise ValueError(f"SEQUENCE_CROSS_SPLIT_OVERLAP: {kind}/{left}/{right}")
    return splits, {
        "file": spec["path"],
        "file_sha256": spec["sha256"],
        "fixture_seed": provenance["fixture_seed"],
        "training_seed": provenance["training_seed"],
        "role": spec["role"],
        "tasks": list(SEQUENCE_TASKS),
        "generator_sha256": provenance["generator_sha256"],
        "source_generator_sha256": provenance["source_generator_sha256"],
        "input_sha256": provenance["input_sha256"],
        "task_sha256": provenance["task_sha256"],
        "mapping_sha256": {task: digest(mapping) for task, mapping in mappings.items()},
        "group_unit": "input_digit_triple",
        "shared_unique_inputs": sum(SPLIT_SIZES.values()),
        "test_groups_per_task": SPLIT_SIZES["test"],
        "chance_accuracy": 0.25,
        "overlap_audit": overlap,
        "heldout_contract": provenance["heldout_contract"],
        "evidence_boundary": "Three sequential procedural mapping skills on one fixed fixture; neither open-ended agent capability nor general backbone portability follows from this test alone.",
    }


def prepare_data(config: dict, output: Path) -> dict:
    all_splits, fixture = sequence_splits(config["sequence"])
    if fixture["training_seed"] != config["seed"]:
        raise ValueError("SEQUENCE_TRAINING_SEED_MISMATCH")
    serialized = {
        name: [asdict(row) for row in rows] for name, rows in all_splits.items()
    }
    manifest = {
        "training_seed": config["seed"],
        "sequence_tasks": list(SEQUENCE_TASKS),
        "fixture": fixture,
        "data_sha256": digest(serialized),
        "counts": {name: len(rows) for name, rows in all_splits.items()},
        "split_sha256": {name: digest(values) for name, values in serialized.items()},
        "ordered_example_ids_sha256": {
            name: digest([row.id for row in rows]) for name, rows in all_splits.items()
        },
        "ordered_inputs_sha256": {
            name: digest([row.group for row in rows])
            for name, rows in all_splits.items()
        },
        "labels": {
            name: {
                str(label): count
                for label, count in Counter(row.gold_idx for row in rows).items()
            }
            for name, rows in all_splits.items()
        },
        "student_context": "One prompt only, no hidden digit mapping, demonstration, previous episode, or KV state.",
        "primary_retention": "Learned sequence A/B/C only. Published task performance is a separate optional diagnostic.",
        "split_access": "Preparation validates the complete frozen fixture. Training readers must explicitly request only train/validation files; held-out test files are read by evaluation.",
    }
    for name, values in serialized.items():
        write_json(output / "data" / f"{name}.json", values)
    write_json(output / "data_manifest.json", manifest)
    return manifest


def read_data(output: Path, splits: tuple[str, ...]) -> dict[str, list[Example]]:
    if not splits or isinstance(splits, str):
        raise ValueError("EXPLICIT_SEQUENCE_SPLITS_REQUIRED")
    if len(set(splits)) != len(splits):
        raise ValueError("DUPLICATE_SEQUENCE_SPLIT_REQUEST")
    allowed = {f"{task}_{split}" for task in SEQUENCE_TASKS for split in SPLIT_SIZES}
    if set(splits) - allowed:
        raise ValueError(f"UNKNOWN_SEQUENCE_SPLIT_REQUEST: {set(splits) - allowed}")
    manifest = json.loads((output / "data_manifest.json").read_text())
    result = {}
    for split in splits:
        rows = json.loads((output / "data" / f"{split}.json").read_text())
        if digest(rows) != manifest["split_sha256"][split]:
            raise ValueError(f"DATA_HASH_MISMATCH: {split}")
        if len(rows) != manifest["counts"][split]:
            raise ValueError(f"DATA_COUNT_MISMATCH: {split}")
        result[split] = [Example.from_dict(row) for row in rows]
    return result
