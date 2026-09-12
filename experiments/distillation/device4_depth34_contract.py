import itertools
import json
import random
from collections import Counter
from pathlib import Path

from device4_curriculum import verify_records
from generated_contract import digest, file_digest, make_corpus

from tasks import FAMILIES, OPS, audit_dataset, make_task, paired_gate, to_records

ROOT = Path(__file__).resolve().parent
CONTRACT = "device4_unseen_composition_context_distillation_20260912"
DEEP_SPLITS = ("train", "validation", "test")
DATA_SPEC = {
    "input_seed": 91234001, "program_seed": 91234003, "assignment_seed": 91234005,
    "rows_per_family_depth": {"train": 64, "validation": 24, "test": 48},
    "program_counts": {"3": {"train": 18, "validation": 4, "test": 5}, "4": {"train": 54, "validation": 13, "test": 14}},
    "exclude_all_historical_inputs": True, "programs_disjoint_across_new_splits": True,
}


def build_data(original, spec):
    historical = make_corpus(original)
    used = {task.initial for tasks in historical.values() for task in tasks}
    values = [value for value in range(10000) if tuple(map(int, f"{value:04d}")) not in used]
    random.Random(spec["input_seed"]).shuffle(values)
    pools, cursor = {}, 0
    for split in DEEP_SPLITS:
        count = spec["rows_per_family_depth"][split]
        size = len(FAMILIES) * 2 * count
        pools[split] = values[cursor:cursor + size]
        cursor += size
    if cursor > len(values):
        raise ValueError("DEPTH34_INSUFFICIENT_UNUSED_INPUTS")
    programs = {split: {} for split in pools}
    for depth in (3, 4):
        values = list(itertools.product(OPS, repeat=depth))
        random.Random(spec["program_seed"] + depth).shuffle(values)
        cursor = 0
        for split in DEEP_SPLITS:
            count = spec["program_counts"][str(depth)][split]
            programs[split][str(depth)] = values[cursor:cursor + count]
            cursor += count
        if cursor != len(values):
            raise ValueError("DEPTH34_PROGRAM_PARTITION_INCOMPLETE")
    corpus = {"demonstration": historical["demonstration"]}
    for split, inputs in pools.items():
        corpus[split] = []
        cursor = 0
        for family in FAMILIES:
            for depth in (3, 4):
                count = spec["rows_per_family_depth"][split]
                assigned = programs[split][str(depth)]
                rng = random.Random(f"{spec['assignment_seed']}:{split}:{family}:{depth}")
                assigned = [assigned[i % len(assigned)] for i in range(count)]
                rng.shuffle(assigned)
                for value, program in zip(inputs[cursor:cursor + count], assigned, strict=True):
                    corpus[split].append(make_task(family, split, tuple(map(int, f"{value:04d}")), program))
                cursor += count
    audit_dataset(corpus)
    split_inputs = {split: {task.initial for task in rows} for split, rows in corpus.items()}
    if any(split_inputs[left] & split_inputs[right] for left in corpus for right in corpus if left != right):
        raise ValueError("DEPTH34_INPUT_OVERLAP")
    if any(task.initial in used for split in pools for task in corpus[split]):
        raise ValueError("DEPTH34_HISTORICAL_INPUT_REUSED")
    split_programs = {split: {task.program for task in corpus[split]} for split in pools}
    if any(split_programs[left] & split_programs[right] for left in pools for right in pools if left != right):
        raise ValueError("DEPTH34_PROGRAM_LEAK")
    return {
        "contract": CONTRACT, "spec": spec, "tasks": to_records(corpus),
        "programs": {split: {depth: [list(program) for program in values] for depth, values in panel.items()} for split, panel in programs.items()},
        "audit": {
            "dataset_sha256": digest(to_records(corpus)), "rows_per_split": {split: len(rows) for split, rows in corpus.items()},
            "family_depth_counts": {split: dict(Counter(f"{task.family}/depth{len(task.program)}" for task in rows)) for split, rows in corpus.items()},
            "historical_unique_inputs_excluded": len(used), "historical_input_overlap": 0,
            "cross_split_input_overlap": 0, "cross_split_program_overlap": 0,
            "historical_task_manifest_sha256": digest(to_records(historical)),
        },
    }, corpus


def load_data(design, original):
    spec = design["data"]
    path = ROOT / spec["path"]
    if file_digest(path) != spec["sha256"]:
        raise ValueError("DEPTH34_SEALED_DATA_FILE_CHANGED")
    generated, corpus = build_data(original, design["data_spec"])
    if digest(generated) != digest(json.loads(path.read_text())):
        raise ValueError("DEPTH34_DATA_GENERATOR_OR_ORACLE_CHANGED")
    return corpus, generated["audit"]


def train_schedule(corpus, design):
    sequence = []
    recipe = design["training"]
    for family in FAMILIES:
        for depth in (3, 4):
            rows = [task for task in corpus["train"] if task.family == family and len(task.program) == depth]
            if not rows:
                raise ValueError("DEPTH34_MISSING_TRAIN_CELL")
            ids = [task.uid for task in rows]
            random.Random(f"{recipe['order_seed']}:{family}:{depth}").shuffle(ids)
            sequence.extend(ids[i % len(ids)] for i in range(recipe["exposures_per_family_depth"]))
    random.Random(recipe["order_seed"]).shuffle(sequence)
    batch = recipe["examples_per_update"]
    if len(sequence) != recipe["updates_per_method"] * batch:
        raise ValueError("DEPTH34_TRAIN_BUDGET_MISMATCH")
    return [sequence[i:i + batch] for i in range(0, len(sequence), batch)]


def train_probes(corpus, design):
    return [
        task for family in FAMILIES for depth in (3, 4)
        for task in [row for row in corpus["train"] if row.family == family and len(row.program) == depth][:design["training"]["probe_rows_per_family_depth"]]
    ]


def deep_gate(corpus, original, raw, teacher, eos, specials, gate):
    panels = {}
    for split in ("train", "validation"):
        verify_records(corpus[split], raw[split], corpus, original, eos, specials, privileged=False)
        verify_records(corpus[split], teacher[split], corpus, original, eos, specials)
        for family in FAMILIES:
            for depth in (3, 4):
                indices = [i for i, task in enumerate(corpus[split]) if task.family == family and len(task.program) == depth]
                if not indices:
                    raise ValueError("DEPTH34_EMPTY_QUALIFICATION_CELL")
                panels[f"{family}/depth{depth}/{split}"] = paired_gate(
                    [raw[split][i]["generation"]["correct"] for i in indices],
                    [teacher[split][i]["generation"]["correct"] for i in indices],
                    gate["minimum_accuracy"], gate["minimum_gain"], gate["maximum_p"],
                )
    return {"passed": all(value["passed"] for value in panels.values()), "panels": panels}


def feasible_gate(corpus, original, raw, eos, specials, gate):
    panels = {}
    for split in ("train", "validation"):
        verify_records(corpus[split], raw[split], corpus, original, eos, specials, privileged=False)
        for family in FAMILIES:
            for depth in (3, 4):
                indices = [i for i, task in enumerate(corpus[split]) if task.family == family and len(task.program) == depth]
                values = [raw[split][i]["generation"]["correct"] for i in indices]
                panels[f"{family}/depth{depth}/{split}"] = paired_gate(values, [True] * len(values), gate["minimum_accuracy"], gate["minimum_gain"], gate["maximum_p"])
    return {"passed": all(value["passed"] for value in panels.values()), "panels": panels}
