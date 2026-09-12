import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from data import SEQUENCE_TASKS, Example, digest, sequence_splits


def rng_for(seed, *keys):
    return random.Random(int(digest([seed, *keys]), 16))


def build_fresh_fixture(spec):
    if spec["fixture_seed"] != 303:
        raise ValueError("FRESH_INPUTS_FIXTURE303_ONLY")
    splits, validated = sequence_splits(spec)
    used = {row.group for rows in splits.values() for row in rows}
    original = json.loads((Path(__file__).parent / spec["path"]).read_text())
    triples = list(itertools.product(range(8), repeat=3))
    rng_for(303, "global-input-partition").shuffle(triples)
    if {" ".join(map(str, row)) for row in triples[:224]} != used:
        raise ValueError("FRESH_INPUTS_ORIGINAL_PARTITION_MISMATCH")
    inputs = triples[224:]
    if len(inputs) != 288 or len(set(inputs)) != 288:
        raise ValueError("FRESH_INPUTS_ALL_REMAINING_TRIPLES_REQUIRED")
    rows = []
    for task in SEQUENCE_TASKS:
        mapping = original["provenance"]["rules"][task]
        rng = rng_for(303, "choices", task, "fresh288")
        positions = [index % 4 for index in range(len(inputs))]
        rng.shuffle(positions)
        for digits, gold_index in zip(inputs, positions, strict=True):
            group = " ".join(map(str, digits))
            gold = " " + " ".join(str(mapping[digit]) for digit in digits)
            distractors = set()
            while len(distractors) < 3:
                candidate = " " + " ".join(str(rng.randrange(8)) for _ in digits)
                if candidate != gold:
                    distractors.add(candidate)
            choices = sorted(distractors)
            rng.shuffle(choices)
            choices.insert(gold_index, gold)
            rows.append(
                asdict(
                    Example(
                        digest({"task": task, "input": digits}),
                        task,
                        f"Apply the {task} code.\nInput: {group}\nOutput:",
                        tuple(choices),
                        gold_index,
                        group,
                    )
                )
            )
    if any(row["group"] in used for row in rows):
        raise ValueError("FRESH_INPUTS_PRIOR_SPLIT_OVERLAP")
    return {
        "kind": "sequence303_all_unused_input_confirmation",
        "original_fixture": spec,
        "original_fixture_sha256": validated["file_sha256"],
        "rows": rows,
        "unique_inputs": inputs,
        "rows_sha256": digest(rows),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": {
            "selected_before_new_behavioral_outcomes": True,
            "new_mapping_or_training_replication": False,
            "all_remaining_domain_inputs": True,
            "old_input_count": 224,
            "new_input_count": 288,
            "rows_per_task": dict.fromkeys(SEQUENCE_TASKS, 288),
            "choice_counts": {
                task: dict(
                    Counter(row["gold_idx"] for row in rows if row["task"] == task)
                )
                for task in SEQUENCE_TASKS
            },
            "choice_policy": "Same original family: three random distinct non-gold three-digit distractors, independently balanced gold positions. No answer choices enter model prompt.",
            "source_access": "Evaluation only. Existing source checkpoints and prospective anchored trainers must not use these inputs or choices for optimization, stopping or hyperparameter selection.",
            "claim_boundary": "Input confirmation on existing fixture303 mappings, with all previous train, validation and test triples excluded. This is not a new mapping or training-seed replication.",
        },
    }


def load_fresh_rows(spec):
    path = Path(__file__).parent / spec["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("FRESH_INPUTS_FILE_HASH_MISMATCH")
    value = json.loads(raw)
    expected = build_fresh_fixture(value["original_fixture"])
    if digest(value) != digest(expected):
        raise ValueError("FRESH_INPUTS_GENERATOR_ROUNDTRIP_MISMATCH")
    return [Example.from_dict(row) for row in value["rows"]], value
