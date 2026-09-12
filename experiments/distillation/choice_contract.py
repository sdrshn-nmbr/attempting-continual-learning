import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ("sequence_a", "sequence_b", "sequence_c")
METHODS = ("teacher_choice_kl", "oracle_sft")
CONTRACT = "sequence303_qualified_choice_consolidation_20260912"
SCORING = {
    "prompt": "raw_original_no_chat_template_no_choices",
    "continuation": "original_boundary_whitespace_separate_tokenization_no_eos",
    "score": "sum_full_vocabulary_token_log_probability_divided_by_characters",
    "decision": "argmax_of_four_scores_first_index_breaks_ties",
    "dtype": "float32",
    "autocast": False,
    "max_prompt": 768,
    "candidate_batch_size": 8,
}
SOURCE_FILES = ("choice_contract.py", "choice_consolidation.py", "requirements.txt")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_json(spec):
    path = ROOT / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError(f"CONSOLIDATE_INPUT_CHANGED: {path}")
    return json.loads(path.read_text())


def validate_protocol(protocol):
    if protocol["contract"] != CONTRACT or protocol["scoring"] != SCORING:
        raise ValueError("CONSOLIDATE_SCORING_CONTRACT_CHANGED")
    if protocol["qualification"] != {
        "splits": ["train", "validation"],
        "minimum_accuracy_per_task_and_split": 0.9,
        "require_all_tasks": True,
    }:
        raise ValueError("CONSOLIDATE_QUALIFICATION_CONTRACT_CHANGED")
    if protocol["learner"] != {
        "rank": 8,
        "alpha": 16,
        "targets": ["q_proj", "v_proj"],
        "dropout": 0.0,
        "initialization": "independent_random_A_zero_B",
        "expected_trainable_parameters": 3833856,
    }:
        raise ValueError("CONSOLIDATE_LEARNER_CAPACITY_CHANGED")
    recipe = protocol["training"]
    if (
        recipe["methods"] != list(METHODS)
        or recipe["epochs"] != 4
        or recipe["examples_per_update"] != 4
        or recipe["updates_per_arm"] != 384
        or recipe["example_exposures_per_arm"] != 1536
        or recipe["learning_rate"] != 0.0003
        or recipe["temperature"] != 1.0
        or recipe["weight_decay"] != 0.0
        or recipe["max_grad_norm"] != 1.0
    ):
        raise ValueError("CONSOLIDATE_TRAINING_BUDGET_CHANGED")
    base = protocol["source_base"]
    if (
        re.fullmatch(r"[0-9a-f]{40}", base["revision"]) is None
        or Path(base["local_path"]).name != base["revision"]
        or not base["files"]
    ):
        raise ValueError("CONSOLIDATE_PINNED_BASE_REQUIRED")
    if protocol["evaluation_splits"] != ["test", "unused288"]:
        raise ValueError("CONSOLIDATE_HELDOUT_CONTRACT_CHANGED")


def load_corpus(protocol):
    original = checked_json(protocol["fixtures"]["original"])
    unused = checked_json(protocol["fixtures"]["unused"])
    if original["provenance"]["fixture_seed"] != 303:
        raise ValueError("CONSOLIDATE_WRONG_MAPPING")
    corpus = {
        split: [row for task in TASKS for row in original["splits"][f"{task}_{split}"]]
        for split in ("train", "validation", "test")
    }
    corpus["unused288"] = unused["rows"]
    sizes = {"train": 128, "validation": 32, "test": 64, "unused288": 288}
    groups, ids = {}, set()
    for split, rows in corpus.items():
        if Counter(row["task"] for row in rows) != dict.fromkeys(TASKS, sizes[split]):
            raise ValueError(f"CONSOLIDATE_SPLIT_COUNTS: {split}")
        groups[split] = {row["group"] for row in rows}
        if len(groups[split]) != sizes[split]:
            raise ValueError(f"CONSOLIDATE_GLOBAL_INPUT_COUNTS: {split}")
        for row in rows:
            task = row["task"]
            inputs = tuple(map(int, row["group"].split()))
            expected_prompt = f"Apply the {task} code.\nInput: {row['group']}\nOutput:"
            if (
                len(inputs) != 3
                or any(item not in range(8) for item in inputs)
                or row["prompt"] != expected_prompt
                or row["id"] != digest({"task": task, "input": inputs})
                or row["id"] in ids
                or len(row["choices"]) != 4
                or len(set(row["choices"])) != 4
                or type(row["gold_idx"]) is not int
                or row["gold_idx"] not in range(4)
                or any(re.fullmatch(r" [0-7] [0-7] [0-7]", c) is None for c in row["choices"])
            ):
                raise ValueError(f"CONSOLIDATE_ROW_CONTRACT: {row['id']}")
            gold = " " + " ".join(str(original["provenance"]["rules"][task][i]) for i in inputs)
            if row["choices"][row["gold_idx"]] != gold:
                raise ValueError(f"CONSOLIDATE_ORACLE_MISMATCH: {row['id']}")
            ids.add(row["id"])
    if len(set.union(*groups.values())) != sum(sizes.values()):
        raise ValueError("CONSOLIDATE_CROSS_SPLIT_INPUT_LEAK")
    return corpus, {
        "rows_sha256": {split: digest(rows) for split, rows in corpus.items()},
        "unique_inputs_per_split": sizes,
        "cross_split_input_overlap": 0,
        "prior_campaign_observed_holdout": True,
    }


def prediction(row, scores):
    if len(scores) != 4 or any(not math.isfinite(score) for score in scores):
        raise ValueError("CONSOLIDATE_INVALID_CHOICE_SCORES")
    chosen = max(range(4), key=scores.__getitem__)
    return {
        "id": row["id"], "task": row["task"], "group": row["group"],
        "row_sha256": digest(row), "gold": row["gold_idx"],
        "prediction": chosen, "correct": chosen == row["gold_idx"], "scores": scores,
    }


def verify_predictions(rows, records):
    if len(rows) != len(records):
        raise ValueError("CONSOLIDATE_INCOMPLETE_PREDICTIONS")
    for row, record in zip(rows, records, strict=True):
        if record != prediction(row, record["scores"]):
            raise ValueError("CONSOLIDATE_PREDICTION_BINDING_MISMATCH")


def metrics(records):
    return {
        task: {
            "correct": sum(row["correct"] for row in records if row["task"] == task),
            "count": sum(row["task"] == task for row in records),
            "accuracy": sum(row["correct"] for row in records if row["task"] == task)
            / sum(row["task"] == task for row in records),
        }
        for task in TASKS
    }


def qualification_gate(protocol, corpus, panels):
    expected = set(protocol["qualification"]["splits"])
    if set(panels) != expected:
        raise ValueError("CONSOLIDATE_QUALIFICATION_SPLITS")
    result = {}
    for split in protocol["qualification"]["splits"]:
        if set(panels[split]) != {"teacher", "untouched"}:
            raise ValueError("CONSOLIDATE_QUALIFICATION_CONDITIONS")
        for records in panels[split].values():
            verify_predictions(corpus[split], records)
        teacher, untouched = [metrics(panels[split][c]) for c in ("teacher", "untouched")]
        for task in TASKS:
            result[f"{split}/{task}"] = {
                "teacher": teacher[task], "untouched": untouched[task],
                "gain": teacher[task]["accuracy"] - untouched[task]["accuracy"],
                "passed": teacher[task]["accuracy"]
                >= protocol["qualification"]["minimum_accuracy_per_task_and_split"],
            }
    return {"passed": all(item["passed"] for item in result.values()), "panels": result}


def training_schedule(protocol, rows):
    recipe = protocol["training"]
    schedule = []
    for epoch in range(recipe["epochs"]):
        indices = list(range(len(rows)))
        random.Random(recipe["order_seed"] + epoch).shuffle(indices)
        for offset in range(0, len(indices), recipe["examples_per_update"]):
            schedule.append(indices[offset:offset + recipe["examples_per_update"]])
    if (
        len(schedule) != recipe["updates_per_arm"]
        or sum(map(len, schedule)) != recipe["example_exposures_per_arm"]
        or any(len(batch) != recipe["examples_per_update"] for batch in schedule)
    ):
        raise ValueError("CONSOLIDATE_UNEQUAL_EXPOSURE_BUDGET")
    return schedule


def verify_qualification(directory, protocol, corpus, audit):
    directory = Path(directory)
    receipt = json.loads((directory / "qualification.json").read_text())
    if (
        receipt["protocol_sha256"] != digest(protocol)
        or receipt["source_sha256"] != source_hashes()
        or receipt["dataset"] != audit
        or receipt["optimizer_updates"] != 0
        or receipt["teacher_tensor_sha256"] != protocol["teacher"]["tensor_sha256"]
        or receipt["base_tensor_sha256"] != protocol["source_base_tensor_sha256"]
    ):
        raise ValueError("CONSOLIDATE_QUALIFICATION_IDENTITY_MISMATCH")
    panels = checked_json({
        "path": str(directory / "qualification_panels.json"),
        "sha256": receipt["panels_sha256"],
    })
    gate = qualification_gate(protocol, corpus, panels)
    if gate != receipt["gate"] or not gate["passed"] or receipt["status"] != "qualified":
        raise ValueError("CONSOLIDATE_TEACHER_NOT_QUALIFIED_NO_LEARNER_UPDATES")
    return receipt, panels["train"]["teacher"]
