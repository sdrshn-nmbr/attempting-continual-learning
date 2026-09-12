import hashlib
import json
import random
import string
from collections import Counter

from sandbox import FAMILIES, SCHEMAS, Example, execute, grade_text, render


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rng_for(*parts):
    return random.Random(int(digest(parts), 16))


def token(rng, length=6):
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def stream_spec(seed):
    rng = rng_for(seed, "conventions")
    aliases = [f"{a}{b}" for a in ("ve", "ro", "mi", "ka") for b in ("lu", "sa", "no", "ti")]
    rng.shuffle(aliases)
    conventions = {}
    for family in FAMILIES:
        conventions[family] = {operation: aliases.pop() for operation in SCHEMAS[family]}
    order = list(FAMILIES)
    rng_for(seed, "order").shuffle(order)
    return {"data_seed": seed, "conventions": conventions, "order": order}


def specifications(family, rng):
    ids = [token(rng) for _ in range(5)]
    if len(set(ids)) != len(ids):
        raise RuntimeError("ENTITY_COLLISION")
    if family == "records":
        scores = rng.sample(range(10, 99), 4)
        if scores == sorted(scores, reverse=True):
            scores[0], scores[1] = scores[1], scores[0]
        teams = ["amber", "blue", "amber", "blue"]
        rng.shuffle(teams)
        state = [{"id": ids[i], "team": teams[i], "score": scores[i]} for i in range(4)]
        team = rng.choice(teams)
        primitive = [
            ("filter", {"field": "team", "value": team}, f"Keep only records whose team equals {team}."),
            ("sort", {"field": "score", "descending": True}, "Sort records by score, largest first."),
            ("take", {"count": 2}, "Keep only the first two records, in their current order."),
            ("project", {"fields": ["id", "score"]}, "Keep only the id and score fields of each record."),
        ]
        workflows = [(0, 1, 3), (1, 2, 3), (0, 2, 3), (0, 1, 2)]
    elif family == "text":
        state = [f"ALERT {ids[0]}", f"Note {ids[1]}", f"ALERT {ids[2]}"]
        primitive = [
            ("lower", {}, "Convert every message to lowercase."),
            ("keep", {"contains": "ALERT"}, "Keep only messages containing the exact text ALERT."),
            ("replace", {"old": "ALERT", "new": "TASK"}, "Replace every occurrence of ALERT with TASK."),
            ("join", {"separator": " | "}, "Join the messages into one string separated by ' | '."),
        ]
        workflows = [(1, 0, 3), (2, 0, 3), (1, 2, 3), (0, 1, 2)]
        if rng.randrange(2):
            primitive[3] = ("join", {"separator": "; "}, "Join the messages into one string separated by '; '.")
    elif family == "files":
        source, other, temp, dest, content = ids
        state = {source: f"draft-{content}", other: f"note-{content}"}
        primitive = [
            ("copy", {"source": source, "destination": temp}, f"Copy {source} to the new path {temp}."),
            ("move", {"source": other, "destination": dest}, f"Move {other} to the new path {dest}."),
            ("write", {"path": source, "text": content}, f"Set the content of {source} to exactly {content}."),
            ("delete", {"path": source}, f"Delete the path {source}."),
        ]
        workflows = [(0, 1, 2), (0, 1, 3), (2, 0, 3), (1, 2, 3)]
    else:
        first, second, new, title, alternate = ids
        start = rng.randrange(12, 48) * 15
        state = {first: {"start": start, "title": title}, second: {"start": start + 180, "title": alternate}}
        primitive = [
            (
                "add",
                {"id": new, "start": start + 90, "title": title},
                f"Add event {new} at minute {start + 90} titled {title}.",
            ),
            ("shift", {"id": first, "minutes": 30}, f"Shift event {first} later by 30 minutes."),
            ("rename", {"id": first, "title": alternate}, f"Rename event {first} to {alternate}."),
            ("cancel", {"id": second}, f"Cancel event {second}."),
        ]
        workflows = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]
    return state, primitive, workflows


def make_example(spec, family, kind, split, index):
    rng = rng_for(spec["data_seed"], family, kind, split, index)
    state, primitive, workflows = specifications(family, rng)
    if kind == "primitive":
        indices = (index % 4,)
        pattern = primitive[indices[0]][0]
    else:
        pattern_index = index % 2 + (2 if split == "novel_test" else 0)
        indices = workflows[pattern_index]
        pattern = str(pattern_index)
    selected = [primitive[i] for i in indices]
    if family == "text" and kind == "workflow" and indices == (0, 1, 2):
        selected[1] = ("keep", {"contains": "alert"}, "Keep only messages containing the exact text alert.")
        selected[2] = ("replace", {"old": "alert", "new": "task"}, "Replace every occurrence of alert with task.")
    if family == "files" and kind == "workflow" and indices == (0, 1, 2):
        temp = primitive[0][1]["destination"]
        dest = primitive[1][1]["destination"]
        content = primitive[2][1]["text"]
        selected = [
            primitive[0],
            ("write", {"path": temp, "text": content}, f"Set the content of {temp} to exactly {content}."),
            ("move", {"source": temp, "destination": dest}, f"Move {temp} to the new path {dest}."),
        ]
    calls = [{"tool": spec["conventions"][family][op], "args": args} for op, args, _ in selected]
    expected, _ = execute(calls, state, family, spec["conventions"])
    if expected == state:
        raise RuntimeError("NOOP_ORACLE: every example must require a state change")
    group_id = digest({"family": family, "state": state})
    return Example(
        id=digest([spec["data_seed"], family, kind, split, index]),
        group_id=group_id,
        family=family,
        split=split,
        kind=kind,
        pattern=pattern,
        state=state,
        request=" Then ".join(item[2] for item in selected),
        calls=calls,
        expected=expected,
    )


def build_corpus(config):
    spec = stream_spec(config["data_seed"])
    corpus = {"primitive": {}, "workflow": {}}
    for family in FAMILIES:
        for kind in corpus:
            splits = {split: config[f"{kind}_{split}_examples"] for split in ("train", "validation", "test")}
            if kind == "workflow":
                splits["novel_test"] = config["workflow_test_examples"]
            corpus[kind][family] = {
                split: [make_example(spec, family, kind, split, i) for i in range(count)]
                for split, count in splits.items()
            }
    return spec, corpus


def audit_corpus(spec, corpus):
    ids, groups, prompts = set(), set(), set()
    counts = Counter()
    for kind, families in corpus.items():
        for family, splits in families.items():
            for split, rows in splits.items():
                for row in rows:
                    if row.id in ids or row.group_id in groups:
                        raise RuntimeError("SPLIT_LEAKAGE: repeated identity or initial scene")
                    prompt = render(row, spec["conventions"])
                    if prompt in prompts:
                        raise RuntimeError("PROMPT_LEAKAGE")
                    grade = grade_text(json.dumps(row.calls), row, spec["conventions"])
                    if not grade["correct"]:
                        raise RuntimeError(f"ORACLE_EXECUTION: {row.id}")
                    if kind == "primitive" and len(row.calls) != 1:
                        raise RuntimeError("HISTORY_COMPOSITION_LEAKAGE")
                    if kind == "workflow" and split == "novel_test" and row.pattern in {"0", "1"}:
                        raise RuntimeError("WORKFLOW_PATTERN_LEAKAGE")
                    ids.add(row.id)
                    groups.add(row.group_id)
                    prompts.add(prompt)
                    counts[f"{kind}/{family}/{split}"] += 1
    return {
        "rows": len(ids),
        "unique_scenes": len(groups),
        "unique_prompts": len(prompts),
        "counts": dict(counts),
        "all_oracles_execute": True,
        "primitive_history_contains_no_composition": True,
        "novel_test_patterns_disjoint_from_workflow_training": True,
        "corpus_sha256": digest(
            {
                kind: {
                    family: {split: [r.record() for r in rows] for split, rows in splits.items()}
                    for family, splits in families.items()
                }
                for kind, families in corpus.items()
            }
        ),
    }


def scheduled_batch(rows, step, batch_size, seed):
    rng = rng_for(seed, "batch", step)
    if any(row.split != "train" for row in rows):
        raise ValueError("TRAIN_SPLIT_REQUIRED")
    return rng.sample(rows, batch_size)


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = rng_for(seed, "reservoir")
        self.rows = []
        self.seen_count = 0
        self.completed_families = set()

    def observe(self, rows):
        if any(row.split != "train" for row in rows):
            raise ValueError("REPLAY_LEAKAGE")
        families = {row.family for row in rows}
        if len(families) != 1 or families & self.completed_families:
            raise ValueError("REPLAY_STAGE: admit each completed family exactly once")
        self.completed_families.update(families)
        for row in {row.id: row for row in rows}.values():
            self.seen_count += 1
            if len(self.rows) < self.capacity:
                self.rows.append(row)
            else:
                slot = self.rng.randrange(self.seen_count)
                if slot < self.capacity:
                    self.rows[slot] = row

    def sample(self, count, seed, step, current_family):
        if any(row.family == current_family for row in self.rows):
            raise ValueError("REPLAY_CURRENT_TASK: update memory only at stage boundaries")
        if not 0 <= count <= len(self.rows):
            raise ValueError("REPLAY_CAPACITY")
        return rng_for(seed, "replay", step).sample(self.rows, count)

    def record(self):
        return {
            "capacity": self.capacity,
            "seen_unique": self.seen_count,
            "completed_families": sorted(self.completed_families),
            "retained_seen_id_index": False,
            "ids": [row.id for row in self.rows],
        }
