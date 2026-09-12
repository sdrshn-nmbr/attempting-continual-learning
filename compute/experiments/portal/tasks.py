import hashlib
import itertools
import json
import random
from dataclasses import dataclass

from portallib import ChoiceDataset, ChoiceExample


@dataclass(frozen=True)
class TaskData:
    train: dict
    validation: dict
    test: dict
    provenance: dict

    def as_dict(self):
        return {
            "splits": {
                split: {
                    task: [row.to_dict() for row in rows]
                    for task, rows in getattr(self, split).items()
                }
                for split in ("train", "validation", "test")
            },
            "provenance": self.provenance,
        }


def oracle(task, digits, mapping_a, mapping_b):
    if task == "acquisition_a":
        result = [mapping_a[digit] for digit in digits]
    elif task == "acquisition_b":
        result = [mapping_b[digit] for digit in reversed(digits)]
    elif task == "control_copy":
        result = digits
    else:
        raise ValueError(f"unknown_task: {task}")
    return " " + " ".join(str(digit) for digit in result)


def seeded_problem(seed):
    rng = random.Random(seed)
    mapping_a = list(range(8))
    mapping_b = list(range(8))
    rng.shuffle(mapping_a)
    rng.shuffle(mapping_b)
    if mapping_a == mapping_b:
        raise ValueError("mapping_seed_collision")
    triples = list(itertools.product(range(8), repeat=3))
    rng.shuffle(triples)
    return rng, mapping_a, mapping_b, triples


def task_rows(task, command, pool, mapping_a, mapping_b, rng):
    rows = []
    positions = [index % 4 for index in range(len(pool))]
    rng.shuffle(positions)
    for digits, gold_index in zip(pool, positions, strict=True):
        gold = oracle(task, digits, mapping_a, mapping_b)
        distractors = set()
        while len(distractors) < 3:
            candidate = " " + " ".join(str(rng.randrange(8)) for _ in digits)
            if candidate != gold:
                distractors.add(candidate)
        choices = sorted(distractors)
        rng.shuffle(choices)
        choices.insert(gold_index, gold)
        prompt = f"{command}.\nInput: {' '.join(map(str, digits))}\nOutput:"
        rows.append(ChoiceExample(task, prompt, tuple(choices), gold_index))
    return rows


def make_tasks(seed=17, train_examples=128, validation_examples=32, test_examples=64):
    sizes = (train_examples, validation_examples, test_examples)
    if any(not isinstance(size, int) or size < 4 or size % 4 for size in sizes):
        raise ValueError("split_sizes_must_be_positive_multiples_of_four")
    if sum(sizes) > 8**3:
        raise ValueError("requested_split_sizes_exceed_unique_digit_triples")
    rng, mapping_a, mapping_b, triples = seeded_problem(seed)
    splits = {}
    cursor = 0
    for split, size in zip(("train", "validation", "test"), sizes, strict=True):
        pool = triples[cursor : cursor + size]
        cursor += size
        rows_by_task = {}
        for task, command in (
            ("acquisition_a", "Apply the amber code"),
            ("acquisition_b", "Apply the violet code"),
            ("control_copy", "Copy the input digits unchanged"),
        ):
            rows_by_task[task] = task_rows(
                task, command, pool, mapping_a, mapping_b, rng
            )
        splits[split] = rows_by_task
    serialized = {
        split: {task: [row.to_dict() for row in rows] for task, rows in tasks.items()}
        for split, tasks in splits.items()
    }
    provenance = {
        "generator": "seeded_hidden_digit_relabeling_and_reverse",
        "version": 1,
        "seed": seed,
        "rules": {"amber": mapping_a, "violet_reverse_then_map": mapping_b},
        "split_sizes_per_task": dict(zip(splits, sizes, strict=True)),
        "sha256": hashlib.sha256(
            json.dumps(serialized, sort_keys=True).encode()
        ).hexdigest(),
        "heldout_contract": "Input digit triples are disjoint across train, validation, and test for all tasks. Mappings are never included in model prompts.",
        "chance_accuracy": 0.25,
        "choice_lengths_equal": True,
        "validation_use": "fixed-budget diagnostic only; no test-driven checkpoint or hyperparameter selection",
        "primary_metric": "four-choice continuation accuracy; this is not free-generation accuracy",
    }
    return TaskData(**splits, provenance=provenance)


def evaluation_dataset(data, split):
    rows = [row for task_rows in getattr(data, split).values() for row in task_rows]
    train = [row for task_rows in data.train.values() for row in task_rows]
    return ChoiceDataset(train, rows)


def rows_digest(rows):
    return hashlib.sha256(
        json.dumps([row.to_dict() for row in rows], sort_keys=True).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationData:
    train: tuple
    validation: tuple
    provenance: dict

    def rows(self, split):
        if split not in ("train", "validation"):
            raise ValueError(f"calibration_split_forbidden: {split}")
        return getattr(self, split)

    def as_dict(self):
        return {
            "splits": {
                split: [row.to_dict() for row in self.rows(split)]
                for split in ("train", "validation")
            },
            "provenance": self.provenance,
        }


def make_calibration_tasks(
    seed=17,
    validation_seed=2718,
    train_examples=128,
    validation_examples=128,
    prior_evaluation_examples=96,
):
    if any(
        type(size) is not int or size < 4 or size % 4
        for size in (train_examples, validation_examples)
    ):
        raise ValueError("calibration_split_sizes_must_be_multiples_of_four")
    if type(prior_evaluation_examples) is not int or prior_evaluation_examples < 0:
        raise ValueError("calibration_prior_evaluation_count_invalid")
    if train_examples + prior_evaluation_examples + validation_examples > 8**3:
        raise ValueError("calibration_has_insufficient_fresh_input_triples")
    rng, mapping_a, mapping_b, triples = seeded_problem(seed)
    task, command = "acquisition_a", "Apply the amber code"
    train = task_rows(
        task, command, triples[:train_examples], mapping_a, mapping_b, rng
    )
    validation_rng = random.Random(validation_seed)
    fresh = triples[train_examples + prior_evaluation_examples :]
    validation_rng.shuffle(fresh)
    validation = task_rows(
        task, command, fresh[:validation_examples], mapping_a, mapping_b, validation_rng
    )
    provenance = {
        "generator": "source_only_amber_train_dev_calibration",
        "training_seed": seed,
        "validation_seed": validation_seed,
        "task": task,
        "rules": {"amber": mapping_a},
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_sha256": rows_digest(train),
        "validation_sha256": rows_digest(validation),
        "excluded_prior_evaluation_input_count": prior_evaluation_examples,
        "split_contract": "Original training rows are unchanged. Fresh validation inputs exclude training and the previous pilot's reserved evaluation inputs. No test examples are constructed or exposed.",
        "metric": "four-choice continuation accuracy and token-normalized gold NLL",
    }
    return CalibrationData(tuple(train), tuple(validation), provenance)
