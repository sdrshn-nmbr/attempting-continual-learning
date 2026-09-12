import random
from collections import Counter, defaultdict

from protocol import ROOT, digest, file_hash, normalized_hash, read_json

PROMPT = "Classify the intent of this utterance using its learned letter code.\nUtterance: {text}\nIntent code:"
ROLES = {
    "learn": ("train", 32),
    "repair": ("train", 8),
    "gate": ("validation", 10),
    "probe": ("validation", 10),
    "test": ("test", 20),
}


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:15], 16)


def load_design(config_path):
    config = read_json(config_path)
    spec_path = ROOT / config["protocol"]
    if file_hash(spec_path) != spec_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("PROSPECTIVE_PROTOCOL_HASH")
    spec = read_json(spec_path)
    if any(
        type(config[key]) is not int or config[key] <= 0
        for key in ("acquisition_updates", "forgetting_updates", "repair_updates")
    ):
        raise ValueError("PROSPECTIVE_UPDATE_BUDGET")
    if (config["model_id"], config["revision"]) != ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"):
        raise ValueError("PROSPECTIVE_MODEL_SCOPE")
    return config, spec


def validate_data(data, spec):
    codes = data["codes"]
    if len(codes) != 16 or len(set(codes)) != 16 or any(type(code) is not int for code in codes):
        raise ValueError("PROSPECTIVE_SIXTEEN_CODES")
    expected = {item["id"]: item for item in spec["streams"]}
    if set(data["streams"]) != set(expected) or data["prompt"] != PROMPT:
        raise ValueError("PROSPECTIVE_STREAM_BINDING")
    seen_intents, seen_text, seen_tokens, seen_sources = set(), set(), set(), set()

    def check_row(row, split, intent, code):
        ids = row["input_ids"]
        source = (row["source_split"], row["source_index"])
        if (
            row["intent"] != intent
            or row["source_split"] != split
            or row["code"] != code
            or row["target"] != (None if code is None else codes[code])
            or row["text_sha256"] != normalized_hash(row["text"])
        ):
            raise ValueError("PROSPECTIVE_ROW_BINDING")
        if not ids or len(ids) > spec["runtime"]["max_length"] or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("PROSPECTIVE_INPUT_IDS")
        if row["text_sha256"] in seen_text or tuple(ids) in seen_tokens or source in seen_sources:
            raise ValueError("PROSPECTIVE_ROW_LEAKAGE")
        seen_text.add(row["text_sha256"])
        seen_tokens.add(tuple(ids))
        seen_sources.add(source)

    for name, stream in data["streams"].items():
        if any(stream[key] != expected[name][key] for key in ("id", "split", "seed")):
            raise ValueError("PROSPECTIVE_STREAM_SEED_OR_SPLIT")
        intents = stream["old"] + stream["new"]
        if (
            len(stream["old"]) != 8
            or len(stream["new"]) != 8
            or len(set(intents)) != 16
            or set(intents) != set(stream["units"])
            or seen_intents.intersection(intents)
        ):
            raise ValueError("PROSPECTIVE_INTENT_LEAKAGE")
        seen_intents.update(intents)
        mapping = []
        for intent in intents:
            unit = stream["units"][intent]
            if set(unit) != set(ROLES):
                raise ValueError("PROSPECTIVE_ROLES")
            code = unit["learn"][0]["code"]
            mapping.append(code)
            for role, (split, count) in ROLES.items():
                if len(unit[role]) != count:
                    raise ValueError("PROSPECTIVE_ROLE_COUNT")
                for row in unit[role]:
                    check_row(row, split, intent, code)
        if sorted(mapping) != list(range(16)):
            raise ValueError("PROSPECTIVE_CODE_MAPPING")
    background = set()
    for role, count in (("lens_fit", 32), ("lens_check", 8)):
        counts = Counter(row["intent"] for row in data[role])
        if len(counts) != spec["lens"]["background_intents"] or any(n != count for n in counts.values()):
            raise ValueError("PROSPECTIVE_LENS_ROW_COUNT")
        if seen_intents.intersection(counts) or (background and background != set(counts)):
            raise ValueError("PROSPECTIVE_LENS_INTENT_LEAKAGE")
        background = set(counts)
        for row in data[role]:
            check_row(row, "train", row["intent"], None)


def rows_for(stream, intents, role):
    return [row for intent in intents for row in stream["units"][intent][role]]


def balanced_batches(rows, steps, batch_size, seed):
    by_code = defaultdict(list)
    for index, row in enumerate(rows):
        by_code[row["code"]].append(index)
    if not rows or steps < 1 or batch_size % len(by_code):
        raise ValueError("PROSPECTIVE_BALANCED_BATCH_SIZE")
    rng = random.Random(seed)
    queues = {code: [] for code in by_code}
    batches = []
    for _ in range(steps):
        batch = []
        for code, indices in sorted(by_code.items()):
            for _ in range(batch_size // len(by_code)):
                if not queues[code]:
                    queues[code] = rng.sample(indices, len(indices))
                batch.append(queues[code].pop())
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def repair_batches(target_rows, guard_rows, spec, seed):
    batch_size, steps = spec["repair"]["batch_size"], spec["repair"]["updates"]
    rng = random.Random(seed)
    rows = target_rows + guard_rows
    batches = []
    target_queue, guard_queue = [], []
    for _ in range(steps):
        batch = []
        for _ in range(batch_size // 2):
            if not target_queue:
                target_queue = rng.sample(range(len(target_rows)), len(target_rows))
            if not guard_queue:
                guard_queue = rng.sample(range(len(target_rows), len(rows)), len(guard_rows))
            batch.extend((target_queue.pop(), guard_queue.pop()))
        rng.shuffle(batch)
        batches.append(batch)
    return rows, batches
