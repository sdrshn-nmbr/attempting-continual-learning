import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass

from tasks import LABELS, digest


@dataclass(frozen=True)
class RoutingTask:
    identity: str
    axes: tuple[int, int]
    mapping: tuple[str, str, str, str]

    def answer(self, values):
        first, second = self.axes
        return self.mapping[2 * int(values[first] >= 8) + int(values[second] >= 8)]


TASKS = (
    RoutingTask("VELA", (0, 1), ("C", "A", "D", "B")),
    RoutingTask("SORI", (1, 2), ("B", "D", "A", "C")),
    RoutingTask("KEMU", (0, 2), ("D", "B", "C", "A")),
    RoutingTask("PAVI", (1, 0), ("A", "C", "B", "D")),
    RoutingTask("RULO", (2, 1), ("C", "D", "B", "A")),
    RoutingTask("ZEDI", (2, 0), ("B", "A", "D", "C")),
    RoutingTask("FENO", (0, 1), ("D", "C", "A", "B")),
    RoutingTask("HUKA", (1, 2), ("A", "B", "D", "C")),
)
TASK_IDS = tuple(task.identity for task in TASKS)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Example:
    id: str
    task_id: str
    group_id: str
    values: tuple[int, int, int]
    split: str
    label: str


def render(row):
    x, y, z = row.values
    return (
        "Apply the learned routing policy for this station. "
        "Answer with exactly one letter: A, B, C, or D, then end your response.\n"
        f"Station: {row.task_id}\nSignal x: {x}\nSignal y: {y}\nSignal z: {z}"
    )


def seeded_rng(seed, purpose, identity=""):
    return random.Random(int(digest([seed, purpose, identity]), 16))


def split_groups(config):
    counts = {split: config[f"{split}_examples"] for split in SPLITS}
    if any(
        type(count) is not int or count <= 0 or count % 8 for count in counts.values()
    ):
        raise ValueError(
            "SPLIT_BALANCE: positive split sizes must be multiples of eight"
        )
    if sum(counts.values()) > 4096:
        raise ValueError("GROUP_CAPACITY: shared input universe has 4096 groups")
    result = {split: [] for split in SPLITS}
    for high in itertools.product(range(2), repeat=3):
        values = list(itertools.product(*(range(bit * 8, bit * 8 + 8) for bit in high)))
        seeded_rng(config["data_seed"], "group-split", high).shuffle(values)
        offset = 0
        for split, count in counts.items():
            result[split].extend(values[offset : offset + count // 8])
            offset += count // 8
    return {split: tuple(sorted(values)) for split, values in result.items()}


def build_corpus(config, tasks=TASKS):
    groups = split_groups(config)
    registry = {task.identity: task for task in tasks}
    return {
        identity: {
            split: tuple(
                Example(
                    id=digest(["long-stream", identity, values]),
                    task_id=identity,
                    group_id=digest(["routing-input", values]),
                    values=values,
                    split=split,
                    label=registry[identity].answer(values),
                )
                for values in group_values
            )
            for split, group_values in groups.items()
        }
        for identity in config["task_order"]
    }


def audit_corpus(corpus, config, tasks=TASKS):
    registry = {task.identity: task for task in tasks}
    expected_groups = split_groups(config)
    global_splits = {}
    ids = set()
    counts = {}
    for identity, splits in corpus.items():
        if set(splits) != set(SPLITS):
            raise ValueError("CORPUS_SPLITS: train, validation and test are required")
        for split, rows in splits.items():
            observed = set()
            for row in rows:
                if row.split != split or row.task_id != identity:
                    raise ValueError("CORPUS_MEMBERSHIP: incorrect split or task")
                if row.id in ids:
                    raise ValueError("CORPUS_DUPLICATE: repeated example ID")
                ids.add(row.id)
                if row.group_id != digest(["routing-input", row.values]):
                    raise ValueError("GROUP_IDENTITY: input group has changed")
                if row.id != digest(["long-stream", identity, row.values]):
                    raise ValueError("EXAMPLE_IDENTITY: example ID has changed")
                prior = global_splits.setdefault(row.group_id, split)
                if prior != split:
                    raise ValueError(
                        "HELDOUT_LEAK: input group crosses splits across tasks"
                    )
                if row.label != registry[identity].answer(row.values):
                    raise ValueError(
                        "TASK_MAPPING: historical answer differs from fixed policy"
                    )
                observed.add(row.values)
            if observed != set(expected_groups[split]) or len(rows) != len(observed):
                raise ValueError(
                    "SHARED_GROUPS: tasks must share exactly the same split groups"
                )
            labels = Counter(row.label for row in rows)
            if labels != Counter({label: len(rows) // 4 for label in LABELS}):
                raise ValueError("LABEL_BALANCE: every task and split must be balanced")
            counts[f"{identity}/{split}"] = {
                "examples": len(rows),
                "labels": dict(labels),
            }
    if tuple(corpus) != tuple(config["task_order"]):
        raise ValueError("TASK_ORDER: corpus must follow the fixed task order")
    return {
        "sha256": corpus_digest(corpus),
        "counts": counts,
        "shared_input_groups": {
            split: len(rows) for split, rows in expected_groups.items()
        },
        "cross_task_group_disjointness": True,
        "balanced": True,
        "task_mappings": [asdict(registry[identity]) for identity in corpus],
    }


def corpus_digest(corpus):
    return digest(
        {
            identity: {
                split: [asdict(row) for row in rows] for split, rows in splits.items()
            }
            for identity, splits in corpus.items()
        }
    )


class Reservoir:
    def __init__(self, capacity, seed):
        if capacity != 64:
            raise ValueError(
                "BUFFER_CAPACITY: this protocol requires exactly 64 total examples"
            )
        self.capacity = capacity
        self.seen = 0
        self.rows = []
        self.rng = seeded_rng(seed, "reservoir-admission")

    def offer(self, rows):
        for row in rows:
            if row.split != "train":
                raise ValueError(
                    "BUFFER_HELDOUT: only training examples may enter replay"
                )
            self.seen += 1
            if len(self.rows) < self.capacity:
                self.rows.append(row)
            else:
                index = self.rng.randrange(self.seen)
                if index < self.capacity:
                    self.rows[index] = row

    def payload(self):
        return {
            "capacity": self.capacity,
            "seen_unique_training_examples": self.seen,
            "rng_state": self.rng.getstate(),
            "rows": [asdict(row) for row in self.rows],
        }

    def storage(self, encoded, old_tasks):
        payload = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        tokens = sum(len(encoded.prompt_ids(row)) + 2 for row in self.rows)
        return {
            "capacity_examples": self.capacity,
            "examples": len(self.rows),
            "serialized_bytes_including_rng": len(payload),
            "prompt_label_eos_tokens": tokens,
            "token_int64_equivalent_bytes": tokens * 8,
            "stored_representation": "immutable examples plus RNG state; no cached token tensors",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "seen_unique_training_examples": self.seen,
            "ids": [row.id for row in self.rows],
            "old_ids": [row.id for row in self.rows if row.task_id in old_tasks],
            "task_counts": dict(Counter(row.task_id for row in self.rows)),
        }


def stage_batches(current_rows, reservoir_rows, arm, config, task_id):
    if arm not in ("continue", "replay"):
        raise ValueError("STREAM_ARM: expected continue or replay")
    if not current_rows or any(
        row.split != "train" or row.task_id != task_id for row in current_rows
    ):
        raise ValueError(
            "CURRENT_SOURCE: only the current task training split may be sampled"
        )
    if any(row.split != "train" or row.task_id == task_id for row in reservoir_rows):
        raise ValueError("REPLAY_SOURCE: only old training rows may be replayed")
    if len(reservoir_rows) > config["buffer_capacity"]:
        raise ValueError(
            "REPLAY_CAPACITY: historical training source exceeds total capacity"
        )
    replay_count = (
        config["replay_per_batch"] if arm == "replay" and reservoir_rows else 0
    )
    current_count = config["batch_size"] - replay_count
    rng = seeded_rng(config["seed"], "current-samples", task_id)
    replay_rng = seeded_rng(config["seed"], "replay-samples", task_id)
    cycle = []
    offset = 0
    for _ in range(config["updates_per_stage"]):
        batch = []
        for _ in range(current_count):
            if offset == len(cycle):
                cycle = list(current_rows)
                rng.shuffle(cycle)
                offset = 0
            batch.append(cycle[offset])
            offset += 1
        batch.extend(replay_rng.choice(reservoir_rows) for _ in range(replay_count))
        yield tuple(batch), ("new",) * current_count + ("replay",) * replay_count
