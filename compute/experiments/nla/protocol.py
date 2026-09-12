import hashlib
import json
import math
import random
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
AV_ID = "kitft/nla-qwen2.5-7b-L20-av"
AR_ID = "kitft/nla-qwen2.5-7b-L20-ar"
COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "white", "black")
FACTS = (
    ("France", "What is the capital of France?", "Paris", "Rome", "Berlin", "Madrid"),
    ("Japan", "What is the capital of Japan?", "Tokyo", "Seoul", "Beijing", "Bangkok"),
    ("Italy", "What is the capital of Italy?", "Rome", "Paris", "Athens", "Lisbon"),
    ("Germany", "What is the capital of Germany?", "Berlin", "Vienna", "Prague", "Warsaw"),
    ("Spain", "What is the capital of Spain?", "Madrid", "Lisbon", "Paris", "Rome"),
    ("Portugal", "What is the capital of Portugal?", "Lisbon", "Madrid", "Dublin", "Oslo"),
    ("Egypt", "What is the capital of Egypt?", "Cairo", "Rabat", "Tunis", "Algiers"),
    ("Canada", "What is the capital of Canada?", "Ottawa", "Toronto", "Montreal", "Vancouver"),
    ("Australia", "What is the capital of Australia?", "Canberra", "Sydney", "Melbourne", "Perth"),
    ("Brazil", "What is the capital of Brazil?", "Brasilia", "Lima", "Bogota", "Santiago"),
    ("Greece", "What is the capital of Greece?", "Athens", "Rome", "Sofia", "Belgrade"),
    ("Norway", "What is the capital of Norway?", "Oslo", "Stockholm", "Helsinki", "Copenhagen"),
    ("Sweden", "What is the capital of Sweden?", "Stockholm", "Oslo", "Helsinki", "Copenhagen"),
    ("Finland", "What is the capital of Finland?", "Helsinki", "Oslo", "Stockholm", "Tallinn"),
    ("Denmark", "What is the capital of Denmark?", "Copenhagen", "Oslo", "Stockholm", "Helsinki"),
    ("Poland", "What is the capital of Poland?", "Warsaw", "Prague", "Berlin", "Vienna"),
    ("Austria", "What is the capital of Austria?", "Vienna", "Berlin", "Prague", "Bern"),
    ("Ireland", "What is the capital of Ireland?", "Dublin", "London", "Cardiff", "Belfast"),
    ("Iceland", "What is the capital of Iceland?", "Reykjavik", "Oslo", "Dublin", "Helsinki"),
    ("Peru", "What is the capital of Peru?", "Lima", "Quito", "Bogota", "Santiago"),
    ("Chile", "What is the capital of Chile?", "Santiago", "Lima", "Quito", "Bogota"),
    ("Colombia", "What is the capital of Colombia?", "Bogota", "Lima", "Quito", "Caracas"),
    ("Ecuador", "What is the capital of Ecuador?", "Quito", "Lima", "Bogota", "Caracas"),
    ("Thailand", "What is the capital of Thailand?", "Bangkok", "Hanoi", "Manila", "Jakarta"),
    ("Vietnam", "What is the capital of Vietnam?", "Hanoi", "Bangkok", "Manila", "Jakarta"),
    ("Kenya", "What is the capital of Kenya?", "Nairobi", "Kampala", "Kigali", "Lusaka"),
    ("Morocco", "What is the capital of Morocco?", "Rabat", "Cairo", "Tunis", "Algiers"),
    ("Senegal", "What is the capital of Senegal?", "Dakar", "Accra", "Lagos", "Bamako"),
    ("Nepal", "What is the capital of Nepal?", "Kathmandu", "Dhaka", "Thimphu", "Colombo"),
    ("Bangladesh", "What is the capital of Bangladesh?", "Dhaka", "Kathmandu", "Thimphu", "Kabul"),
    ("Mongolia", "What is the capital of Mongolia?", "Ulaanbaatar", "Astana", "Bishkek", "Tashkent"),
    ("New Zealand", "What is the capital of New Zealand?", "Wellington", "Auckland", "Hamilton", "Dunedin"),
)
TOPICS = (
    (
        "baking",
        "In the bakery, the cook mixes flour with water, kneads the dough, and puts the bread into the oven",
        ("bread", "baking", "bakery", "dough", "oven"),
    ),
    (
        "astronomy",
        "Through a telescope at night, the astronomer studies distant stars, galaxies, and planets in outer space",
        ("astronom", "telescope", "space", "galax", "star"),
    ),
    (
        "music",
        "The musician sits at the piano, reads the notes on the score, and plays a gentle piece of music",
        ("music", "piano", "melody", "musician"),
    ),
    (
        "ocean",
        "Beneath the ocean waves, a diver watches colorful fish swim around a coral reef under the water",
        ("ocean", "coral", "reef", "fish", "diver", "underwater"),
    ),
    (
        "railway",
        "At the railway station, the passengers wait on the platform before boarding a train for their journey",
        ("train", "rail", "passenger", "station"),
    ),
    (
        "gardening",
        "In the garden, a gardener plants seeds in the soil and waters the flowers to help the plants grow",
        ("garden", "plant", "flower", "soil"),
    ),
    (
        "snow",
        "High in the cold mountains, the skiers slide down a steep slope covered in fresh white snow",
        ("ski", "snow", "winter", "mountain"),
    ),
    (
        "library",
        "At the quiet library, a reader chooses a novel from the shelves and sits down to read a book",
        ("book", "librar", "read", "novel"),
    ),
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Record:
    id: str
    entity: str
    cohort: str
    split: str
    query: str
    candidates: tuple[str, ...]
    answer: str
    learning_stage: str | None = None

    def payload(self):
        return asdict(self)


def validate_config(config):
    allowed = {
        "model_id",
        "revision",
        "model_path",
        "av_model_id",
        "av_revision",
        "av_model_path",
        "ar_model_id",
        "ar_revision",
        "ar_model_path",
        "seed",
        "mode",
        "pilot",
        "updates",
        "lora",
        "verify_weight_hashes",
        "resume",
        "max_sequence_length",
        "max_new_tokens",
        "gate",
        "prediction",
        "device",
        "dtype",
        "checkpoint_every_steps",
    }
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"NLA_CONFIG_UNKNOWN: {sorted(unknown)}")
    for key, expected in (("model_id", BASE_ID), ("av_model_id", AV_ID), ("ar_model_id", AR_ID)):
        if config[key] != expected:
            raise ValueError(f"NLA_UNSUPPORTED_MODEL: {key} must be {expected}; no cross-model decoder support")
    if config.get("device", "cuda:0") != "cuda:0" or config.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("NLA_RUNTIME_CONTRACT: GPU-isolated cuda:0 and bfloat16 required")
    if config["mode"] not in {"calibrate", "continual"}:
        raise ValueError("NLA_CONFIG_MODE: use calibrate or continual")
    for key in ("revision", "av_revision", "ar_revision"):
        if len(config[key]) != 40 or any(c not in "0123456789abcdef" for c in config[key]):
            raise ValueError(f"NLA_CONFIG_REVISION: {key} must be a full immutable commit")
    if not isinstance(config["seed"], int):
        raise ValueError("NLA_CONFIG_SEED: integer required")
    pilot = config["pilot"]
    if pilot.keys() - {"entities_per_stage", "retention_facts", "calibration_topics", "binding_pairs"}:
        raise ValueError("NLA_CONFIG_PILOT: unknown pilot field")
    n = pilot["entities_per_stage"]
    if n < 4 or n % 2:
        raise ValueError("NLA_CONFIG_ENTITIES: even entities_per_stage >= 4 required for grouped controls")
    if not 4 <= pilot["retention_facts"] <= len(FACTS) or pilot["retention_facts"] % 2:
        raise ValueError("NLA_CONFIG_RETENTION: even retention_facts from 4 through 32 required")
    if not 4 <= pilot["calibration_topics"] <= len(TOPICS):
        raise ValueError("NLA_CONFIG_CALIBRATION: calibration_topics must be 4 through 8")
    if not 1 <= pilot["binding_pairs"] <= 8:
        raise ValueError("NLA_CONFIG_CALIBRATION: binding_pairs must be 1 through 8")
    stages = [u["name"] for u in config["updates"]]
    if stages not in ([], ["a"], ["a", "b"]):
        raise ValueError("NLA_CONFIG_STAGES: supported sequences are [], [a], [a,b]")
    if config["mode"] == "continual" and not stages:
        raise ValueError("NLA_CONFIG_STAGES: continual mode needs at least update a")
    if config["mode"] == "calibrate" and stages:
        raise ValueError("NLA_CALIBRATION_NO_TRAINING: calibrate mode requires updates=[]")
    if config["max_sequence_length"] < 64 or not 16 <= config["max_new_tokens"] <= 512:
        raise ValueError("NLA_CONFIG_LENGTH: sequence >=64 and generation from 16 through 512 required")
    if config["checkpoint_every_steps"] < 1:
        raise ValueError("NLA_CONFIG_CHECKPOINT: positive checkpoint interval required")
    if set(config["lora"]) != {"r", "lora_alpha", "target_modules"}:
        raise ValueError("NLA_CONFIG_LORA: expected r, lora_alpha, target_modules")
    if config["lora"]["r"] < 1 or config["lora"]["lora_alpha"] < 1:
        raise ValueError("NLA_CONFIG_LORA: positive rank and alpha required")
    modules = config["lora"]["target_modules"]
    if not modules or len(set(modules)) != len(modules) or set(modules) - {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("NLA_CONFIG_LORA: only distinct native attention projections are supported")
    gate_keys = {
        "min_cosine",
        "min_shuffled_gap",
        "min_empty_gap",
        "min_topic_hit_rate",
        "min_retrieval_accuracy",
        "min_binding_win_rate",
    }
    if set(config["gate"]) != gate_keys or any(not 0 <= float(v) <= 1 for v in config["gate"].values()):
        raise ValueError("NLA_CONFIG_GATE: specify every native gate threshold in [0, 1]")
    prediction = config["prediction"]
    if set(prediction) != {
        "min_development",
        "min_heldout",
        "projection_dim",
        "ridge_alphas",
        "bootstrap_samples",
        "min_target_std",
    }:
        raise ValueError("NLA_CONFIG_PREDICTION: unknown or missing predictor field")
    if (
        prediction["min_development"] < 4
        or prediction["min_heldout"] < 4
        or prediction["projection_dim"] < 1
        or prediction["bootstrap_samples"] < 100
    ):
        raise ValueError("NLA_CONFIG_PREDICTION: inadequate sample, projection, or bootstrap setting")
    if (
        not prediction["ridge_alphas"]
        or any(not math.isfinite(a) or a <= 0 for a in prediction["ridge_alphas"])
        or prediction["min_target_std"] <= 0
    ):
        raise ValueError("NLA_CONFIG_PREDICTION: finite positive ridge penalties and target floor required")
    for update in config["updates"]:
        if set(update) != {"name", "steps", "batch_size", "learning_rate"}:
            raise ValueError("NLA_CONFIG_UPDATE: expected name, steps, batch_size, learning_rate")
        if update["steps"] < 1 or update["batch_size"] < 1 or not 0 < update["learning_rate"] <= 0.01:
            raise ValueError("NLA_CONFIG_UPDATE: invalid update budget")
        if n % update["batch_size"] or update["steps"] * update["batch_size"] < n:
            raise ValueError("NLA_CONFIG_UPDATE: whole batches and at least one exposure per training entity required")
    if len(config["updates"]) == 2:
        budgets = [{k: v for k, v in u.items() if k != "name"} for u in config["updates"]]
        if budgets[0] != budgets[1]:
            raise ValueError("NLA_UNMATCHED_BUDGET: a/b steps, batches and learning rate must match")
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def make_records(config):
    rng = random.Random(config["seed"])
    records = []
    for stage in ("a", "b"):
        labels = [COLORS[i % len(COLORS)] for i in range(config["pilot"]["entities_per_stage"])]
        rng.shuffle(labels)
        for i in range(config["pilot"]["entities_per_stage"]):
            entity = f"{stage.upper()}{rng.randrange(100000, 999999)}-{i:03d}"
            answer = labels[i]
            records.append(
                Record(
                    f"binding_{stage}_{i:03d}",
                    entity,
                    "random_binding",
                    "development" if i % 2 == 0 else "heldout",
                    f"In the invented Neral registry, which color is assigned to entity {entity}? Respond with the color name only.",
                    COLORS,
                    answer,
                    stage,
                )
            )
    selected = list(FACTS)
    rng.shuffle(selected)
    for i, (entity, query, answer, *distractors) in enumerate(selected[: config["pilot"]["retention_facts"]]):
        candidates = [answer, *distractors]
        rng.shuffle(candidates)
        records.append(
            Record(
                f"retention_{entity.lower().replace(' ', '_')}",
                entity,
                "native_fact",
                "development" if i % 2 == 0 else "heldout",
                query + " Respond with the city name only.",
                tuple(candidates),
                answer,
            )
        )
    if len({r.entity for r in records}) != len(records):
        raise ValueError("NLA_DATA_COLLISION: duplicate entity")
    return records


def training_query(record, variant):
    templates = (
        "Neral registry lookup. Entity {entity} has which registered color? Answer with just the color.",
        "State the assigned color of {entity} in the fictional Neral registry. Give only a color name.",
    )
    return templates[variant % len(templates)].format(entity=record.entity)


def make_calibration(config):
    records = []
    for topic, content, aliases in TOPICS[: config["pilot"]["calibration_topics"]]:
        records.append({"id": f"topic_{topic}", "group": "topic", "content": content, "aliases": aliases})
    for i in range(config["pilot"]["binding_pairs"]):
        first, second = COLORS[(2 * i) % len(COLORS)], COLORS[(2 * i + 1) % len(COLORS)]
        for color, other in ((first, second), (second, first)):
            records.append(
                {
                    "id": f"calibration_binding_{i}_{color}",
                    "group": f"binding_pair_{i}",
                    "content": f"The fictional registry assigns entity Talin{i} to {color} and entity Varek{i} to {other}. The registered color of Talin{i} is",
                    "aliases": (color,),
                    "other_answer": other,
                    "target_entity": f"Talin{i}",
                    "other_entity": f"Varek{i}",
                }
            )
    return records


def matched_donors(records, seed):
    groups = {}
    for i, record in enumerate(records):
        if isinstance(record, Record):
            key = (record.cohort, record.split, record.learning_stage)
        else:
            key = record["group"]
        groups.setdefault(key, []).append(i)
    rng = random.Random(seed)
    mapping = {}
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("NLA_CONTROL_DONOR: each control group needs at least two distinct entities")
        rng.shuffle(indices)
        for i, donor in zip(indices, indices[1:] + indices[:1], strict=True):
            mapping[i] = donor
    return mapping


def candidate_metrics(logprobs, token_counts, candidates, answer):
    values = np.asarray(logprobs, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) != len(candidates) or min(token_counts) < 1:
        raise ValueError("NLA_BEHAVIOR_INVALID: non-finite scores or candidate alignment failure")
    gold = candidates.index(answer)
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    other = np.delete(values, gold)
    return {
        "candidate_logprobs": values.tolist(),
        "candidate_token_counts": list(token_counts),
        "gold_logprob_sum": float(values[gold]),
        "gold_logprob_mean": float(values[gold] / token_counts[gold]),
        "gold_probability_within_candidates": float(probabilities[gold]),
        "gold_margin": float(values[gold] - other.max()),
        "candidate_entropy": float(-(probabilities * np.log(probabilities.clip(1e-300))).sum()),
        "candidate_correct": int(values.argmax() == gold),
        "candidate_prediction": candidates[int(values.argmax())],
    }


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def normalized_text(text):
    decomposed = unicodedata.normalize("NFKD", text)
    letters = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(letters.strip().lower().rstrip(".").split())


def canonical_config(config):
    return {k: v for k, v in config.items() if k != "resume"}


def ensure_paired_states(records, before, after):
    for record in records:
        a, b = before[record.id], after[record.id]
        if (
            a["id"] != b["id"]
            or a["source_layer"] != b["source_layer"]
            or a["activation"].shape != b["activation"].shape
        ):
            raise ValueError("NLA_PAIRED_STATE: identity, layer, or shape mismatch")
        for key in ("input_ids", "source_token_index", "source_token_id", "prompt_sha256"):
            if a["query"][key] != b["query"][key]:
                raise ValueError(f"NLA_PAIRED_QUERY: {key} changed across weight states")
        if a["source_has_output_or_gold_tokens"] or b["source_has_output_or_gold_tokens"]:
            raise ValueError("NLA_PAIRED_GOLD_LEAK: output or gold tokens in source input")
