import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

LABELS = ("A", "B", "C", "D")
COHORTS = {
    "A0": ("MERA", 0),
    "A1": ("MERA", 1),
    "A2": ("MERA", 2),
    "B1": ("NOVA", 1),
    "C2": ("TAVI", 2),
}
STAGES = {
    "A": {"new": "A0", "revision": None, "old": []},
    "B": {"new": "B1", "revision": "A1", "old": ["A0"]},
    "C": {"new": "C2", "revision": "A2", "old": ["A0", "A1", "B1"]},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def oracle(station, edition, x, y, z, coastal, sealed):
    high = (x >= 8, y >= 8, z >= 8)
    selectors = {"MERA": (0, 1), "NOVA": (1, 2), "TAVI": (0, 2)}
    first, second = selectors[station]
    route = 2 * int(high[first]) + int(high[second])
    if station == "MERA":
        route += int(edition >= 1 and coastal and sealed)
        route += 2 * int(edition >= 2)
    return LABELS[route % 4]


def render(row):
    return (
        "Apply the learned routing policy for this station and edition. "
        "Answer with exactly one letter: A, B, C, or D.\n"
        f"Station: {row['station']}\nEdition: {row['edition']}\n"
        f"Signal x: {row['x']}\nSignal y: {row['y']}\nSignal z: {row['z']}\n"
        f"Region: {'coastal' if row['coastal'] else 'inland'}\n"
        f"Seal: {'sealed' if row['sealed'] else 'open'}"
    )


def features(row):
    return tuple(row[key] for key in ("x", "y", "z", "coastal", "sealed"))


def structure(row):
    return (row["x"] >= 8, row["y"] >= 8, row["z"] >= 8, row["coastal"], row["sealed"])


def split_for(values, seed):
    x, y, z, coastal, sealed = values
    composition = (int(x >= 8) + int(y >= 8) + int(z >= 8) + coastal + sealed) % 2 == 0
    bucket = int(digest([seed, values])[:8], 16) % 100
    if composition:
        return "validation_composition" if bucket < 50 else "test_composition"
    if bucket < 12:
        return "validation_iid"
    if bucket < 32:
        return "test_iid"
    return "train"


def build_corpus(config):
    seed = config["data_seed"]
    counts = {"train": config["train_per_cohort"]}
    counts.update(
        {
            f"validation_{kind}": config["validation_per_split"]
            for kind in ("iid", "composition")
        }
    )
    counts.update(
        {f"test_{kind}": config["test_per_split"] for kind in ("iid", "composition")}
    )
    if any(count % 4 for count in counts.values()):
        raise ValueError("DATA_BALANCE: all split counts must be divisible by four")
    corpus = {}
    for cohort, (station, edition) in COHORTS.items():
        pools = {(split, label): [] for split in counts for label in LABELS}
        for values in itertools.product(
            range(16), range(16), range(16), range(2), range(2)
        ):
            x, y, z, coastal, sealed = values
            split = split_for(values, seed)
            label = oracle(station, edition, *values)
            row = {
                "id": digest([cohort, values]),
                "cohort": cohort,
                "split": split,
                "station": station,
                "edition": edition,
                "x": x,
                "y": y,
                "z": z,
                "coastal": bool(coastal),
                "sealed": bool(sealed),
                "label": label,
                "observed_label": label,
                "label_corrupted": False,
                "scope": "exception" if coastal and sealed else "ordinary",
                "old_label": oracle(station, max(0, edition - 1), *values),
                "global_base_label": oracle(station, 0, *values),
            }
            if (
                split == "train"
                and int(digest([seed, "noise", row["id"]])[:8], 16) / 2**32
                < config["noise_fraction"]
            ):
                row["observed_label"] = LABELS[(LABELS.index(label) + 1) % 4]
                row["label_corrupted"] = True
            pools[(split, label)].append(row)
        corpus[cohort] = {}
        for split, count in counts.items():
            rows = []
            for label in LABELS:
                pool = sorted(
                    pools[(split, label)],
                    key=lambda row: digest([seed, "selection", row["id"]]),
                )
                if len(pool) < count // 4:
                    raise ValueError(
                        f"DATA_CAPACITY: {cohort}/{split}/{label} has only {len(pool)} cases"
                    )
                rows.extend(pool[: count // 4])
            corpus[cohort][split] = sorted(rows, key=lambda row: row["id"])
    return corpus


def audit_corpus(corpus):
    train_features = set()
    heldout_features = set()
    train_structures = set()
    composition_structures = set()
    prompts = {}
    counts = {}
    for cohort, splits in corpus.items():
        for split, rows in splits.items():
            counts[f"{cohort}/{split}"] = {
                "n": len(rows),
                "labels": dict(Counter(row["label"] for row in rows)),
                "corrupted": sum(row["label_corrupted"] for row in rows),
                "exception": sum(row["scope"] == "exception" for row in rows),
            }
            for row in rows:
                prompt = render(row)
                if prompt in prompts:
                    raise AssertionError(f"DATA_OVERLAP: duplicate input {row['id']}")
                prompts[prompt] = row["observed_label"]
                if row["label"] != oracle(
                    row["station"], row["edition"], *features(row)
                ):
                    raise AssertionError(f"ORACLE_MISMATCH: {row['id']}")
                if split == "train":
                    train_features.add(features(row))
                    train_structures.add(structure(row))
                else:
                    heldout_features.add(features(row))
                    if row["label_corrupted"]:
                        raise AssertionError(
                            "HELDOUT_NOISE: evaluation labels must be clean"
                        )
                    if split.endswith("composition"):
                        composition_structures.add(structure(row))
    if train_features & heldout_features:
        raise AssertionError(
            "DATA_LEAKAGE: raw input tuple occurs in train and held-out sets"
        )
    if train_structures & composition_structures:
        raise AssertionError(
            "COMPOSITION_LEAKAGE: held-out structural combination occurs in training"
        )
    all_rows = [
        row for splits in corpus.values() for rows in splits.values() for row in rows
    ]
    return {
        "passed": True,
        "counts": counts,
        "rows": len(all_rows),
        "sha256": digest(all_rows),
        "train_heldout_raw_tuple_overlap": 0,
        "train_composition_structure_overlap": 0,
        "train_structures": len(train_structures),
        "heldout_structures": len(composition_structures),
        "contradictory_identical_prompts": 0,
        "context_boundary": "one fresh user message; no demonstrations, retrieval, teacher, or oracle specification",
        "revision_boundary": "edition is an explicit input; older editions retain their historical labels",
    }


def write_corpus(corpus, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for cohort, splits in corpus.items():
        for split, rows in splits.items():
            path = output / f"{cohort}.{split}.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )


class Stream:
    def __init__(self, rows, seed):
        self.rows = rows
        self.rng = random.Random(seed)
        self.indices = []

    def take(self, count):
        result = []
        while len(result) < count:
            if not self.indices:
                self.indices = list(range(len(self.rows)))
                self.rng.shuffle(self.indices)
            result.append(self.rows[self.indices.pop()])
        return result


def stage_batches(corpus, stage, arm, config):
    info = STAGES[stage]
    seed = config["seed"] + 1009 * (ord(stage) - ord("A"))
    current = Stream(corpus[info["new"]]["train"], seed)
    revision = (
        Stream(corpus[info["revision"]]["train"], seed + 11)
        if info["revision"]
        else None
    )
    past_rows = [row for cohort in info["old"] for row in corpus[cohort]["train"]]
    past = Stream(past_rows, seed + 29) if past_rows else None
    revision_count = config["revision_per_batch"] if revision else 0
    replay_count = config["replay_per_batch"] if arm == "replay" and past else 0
    new_count = config["batch_size"] - revision_count
    if replay_count >= new_count:
        raise ValueError(
            "BATCH_BUDGET: replay must leave at least one new-task example"
        )
    for _ in range(config["updates_per_stage"]):
        new_rows = current.take(new_count)
        rows = new_rows[: new_count - replay_count]
        roles = ["new"] * len(rows)
        if revision:
            rows.extend(revision.take(revision_count))
            roles.extend(["revision"] * revision_count)
        if replay_count:
            rows.extend(past.take(replay_count))
            roles.extend(["replay"] * replay_count)
        yield rows, roles
