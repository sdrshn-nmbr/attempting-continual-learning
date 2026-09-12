import hashlib
import random
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
from datasets import load_dataset
from huggingface_hub import HfApi

from config import DATASET_REVISION, derived_seed, digest

DATASET_ID = "clinc/clinc_oos"
SPLIT_COUNTS = {"train": 7600, "validation": 3100, "test": 5500}
PROMPT = "Classify the intent of this utterance using its learned letter code.\nUtterance: {text}\nIntent code:"


@dataclass
class Task:
    index: int
    intents: list[int]
    codes: list[int]
    rows: dict[str, list[dict]]
    schedule: list[list[int]]


@dataclass
class PreparedData:
    tasks: list[Task]
    code_token_ids: list[int]
    pad_token_id: int
    provenance: dict
    schedules: dict
    replay_buffers: list[list[tuple[int, int]]]


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def select_rows(splits, names, cfg):
    intents = [i for i, name in enumerate(names) if name != "oos"]
    random.Random(derived_seed(cfg.seed, "task_order")).shuffle(intents)
    intents = intents[: cfg.tasks * cfg.classes_per_task]
    codes = list(range(len(intents)))
    random.Random(derived_seed(cfg.seed, "code_mapping")).shuffle(codes)
    code_for_intent = dict(zip(intents, codes, strict=True))
    selected = {split: defaultdict(list) for split in SPLIT_COUNTS}
    excluded = Counter()
    seen = set()
    for split in SPLIT_COUNTS:
        grouped = defaultdict(list)
        for index, row in enumerate(splits[split]):
            if row["intent"] in code_for_intent:
                grouped[row["intent"]].append((index, row))
        for intent in intents:
            candidates = grouped[intent]
            random.Random(derived_seed(cfg.seed, split, intent)).shuffle(candidates)
            limit = getattr(cfg, f"{split}_per_class")
            for index, row in candidates:
                text_hash = hashlib.sha256(
                    normalized_text(row["text"]).encode()
                ).hexdigest()
                if text_hash in seen:
                    excluded[split] += 1
                    continue
                if len(selected[split][intent]) < limit:
                    selected[split][intent].append(
                        {
                            "source_split": split,
                            "source_index": index,
                            "text": row["text"],
                            "text_sha256": text_hash,
                            "intent": intent,
                            "code": code_for_intent[intent],
                        }
                    )
                seen.add(text_hash)
            if len(selected[split][intent]) != limit:
                raise ValueError(
                    f"CLINC_SPLIT_TOO_SMALL {split=} {intent=} requested={limit}"
                )
    groups = [
        intents[i : i + cfg.classes_per_task]
        for i in range(0, len(intents), cfg.classes_per_task)
    ]
    tasks = [
        Task(
            index=i,
            intents=group,
            codes=[code_for_intent[label] for label in group],
            rows={
                split: [r for label in group for r in selected[split][label]]
                for split in SPLIT_COUNTS
            },
            schedule=[],
        )
        for i, group in enumerate(groups)
    ]
    return tasks, {
        "intent_to_code": {
            names[label]: string.ascii_uppercase[code_for_intent[label]]
            for label in intents
        },
        "task_intent_ids": groups,
        "task_intent_names": [[names[label] for label in group] for group in groups],
        "duplicate_policy": "NFKC/casefold/whitespace dedup; train before validation before test; all candidate rows reserved",
        "excluded_duplicates": dict(excluded),
    }


def balanced_schedule(rows, cfg, task_index, batch_examples=None):
    batch_examples = (
        cfg.effective_batch_size if batch_examples is None else batch_examples
    )
    by_code = defaultdict(list)
    for i, row in enumerate(rows):
        by_code[row["code"]].append(i)
    rng = random.Random(derived_seed(cfg.seed, "minibatches", task_index))
    queues = {code: [] for code in by_code}
    batches = []
    for _ in range(cfg.updates_per_task):
        batch = []
        for code, indices in sorted(by_code.items()):
            for _ in range(batch_examples // len(by_code)):
                if not queues[code]:
                    queues[code] = rng.sample(indices, len(indices))
                batch.append(queues[code].pop())
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def retain_replay(previous, completed_task, tasks, cfg):
    incoming = [
        (completed_task.index, i) for i in range(len(completed_task.rows["train"]))
    ]
    candidates = [tuple(ref) for ref in previous] + incoming
    if len(candidates) != len(set(candidates)) or any(
        source < 0
        or source >= completed_task.index
        or not 0 <= index < len(tasks[source].rows["train"])
        for source, index in previous
    ):
        raise RuntimeError("REPLAY_RETENTION_INVALID_PREVIOUS_MEMBERS")
    by_code = defaultdict(list)
    for source, index in candidates:
        row = tasks[source].rows["train"][index]
        if row["source_split"] != "train":
            raise RuntimeError("REPLAY_RETENTION_NONTRAIN_ROW")
        by_code[row["code"]].append((source, index))
    capacity = len(candidates) if cfg.replay_capacity is None else cfg.replay_capacity
    per_class, extra = divmod(min(capacity, len(candidates)), len(by_code))
    retained = []
    for position, code in enumerate(sorted(by_code)):
        quota = per_class + (position < extra)
        if quota < 1 or len(by_code[code]) < quota:
            raise RuntimeError(f"REPLAY_QUOTA_UNAVAILABLE code={code} quota={quota}")
        ranked = sorted(
            by_code[code],
            key=lambda ref: derived_seed(cfg.seed, "replay_retention", *ref),
        )
        retained.extend(ranked[:quota])
    return sorted(retained)


def replay_buffers(tasks, cfg):
    history = []
    retained = []
    for task in tasks:
        retained = retain_replay(retained, task, tasks, cfg)
        history.append(retained)
    return history


def replay_buffer_record(tasks, members):
    return {
        "members": [list(ref) for ref in members],
        "examples": len(members),
        "examples_by_code": dict(
            sorted(
                Counter(
                    str(tasks[t].rows["train"][i]["code"]) for t, i in members
                ).items()
            )
        ),
        "members_sha256": digest(members),
    }


def validate_replay_refs(refs, members, task_index):
    eligible = {tuple(ref) for ref in members}
    if any(
        source < 0
        or source > task_index
        or index < 0
        or (source < task_index and (source, index) not in eligible)
        for source, index in refs
    ):
        raise RuntimeError(f"REPLAY_REFERENCE_OUTSIDE_CURRENT_BUFFER task={task_index}")


def training_schedules(tasks, cfg, buffers):
    schedules = {}
    for method in cfg.methods:
        schedules[method] = []
        for task in tasks:
            if method != "balanced_replay" or task.index == 0:
                schedule = [[(task.index, i) for i in batch] for batch in task.schedule]
            else:
                replay_size = int(cfg.effective_batch_size * cfg.replay_fraction)
                current = balanced_schedule(
                    task.rows["train"],
                    cfg,
                    task.index,
                    cfg.effective_batch_size - replay_size,
                )
                by_code = defaultdict(list)
                for old in tasks[: task.index]:
                    for i, row in enumerate(old.rows["train"]):
                        by_code[row["code"]].append((old.index, i))
                rng = random.Random(derived_seed(cfg.seed, "replay", task.index))
                queues = {code: [] for code in by_code}
                eligible = set(buffers[task.index - 1])
                candidate_cycles = {}
                candidate_queues = {code: [] for code in by_code}
                code_queue = []
                schedule = []
                for batch in current:
                    mixed = [(task.index, i) for i in batch]
                    for _ in range(replay_size):
                        if not code_queue:
                            code_queue = rng.sample(sorted(by_code), len(by_code))
                        code = code_queue.pop()
                        if not queues[code]:
                            queues[code] = rng.sample(by_code[code], len(by_code[code]))
                            candidate_cycles[code] = [
                                ref for ref in queues[code] if ref in eligible
                            ]
                            candidate_queues[code] = candidate_cycles[code].copy()
                        # The full-class draw clock fixes RNG consumption and mixing across capacities.
                        queues[code].pop()
                        if not candidate_queues[code]:
                            candidate_queues[code] = candidate_cycles[code].copy()
                        if not candidate_queues[code]:
                            raise RuntimeError(
                                f"REPLAY_CLASS_HAS_NO_CANDIDATES code={code}"
                            )
                        mixed.append(candidate_queues[code].pop())
                    rng.shuffle(mixed)
                    schedule.append(mixed)
            schedules[method].append(schedule)
    return schedules


def schedule_budget(tasks, schedule, task_index, cfg):
    refs = [ref for batch in schedule for ref in batch]
    rows = [tasks[t].rows["train"][i] for t, i in refs]
    current = [(t, i) for t, i in refs if t == task_index]
    replay = [(t, i) for t, i in refs if t < task_index]
    if len(current) + len(replay) != len(refs) or any(
        row["source_split"] != "train" for row in rows
    ):
        raise RuntimeError("TRAINING_PLAN_LEAKAGE: future or heldout examples")
    pool_examples = sum(len(old.rows["train"]) for old in tasks[:task_index])
    if cfg.replay_capacity is not None:
        pool_examples = min(pool_examples, cfg.replay_capacity)
    padded_tokens = 0
    for batch in schedule:
        for start in range(0, len(batch), cfg.batch_size):
            microbatch = batch[start : start + cfg.batch_size]
            padded_tokens += len(microbatch) * max(
                len(tasks[t].rows["train"][i]["input_ids"]) for t, i in microbatch
            )
    return {
        "updates": len(schedule),
        "training_examples": len(refs),
        "training_input_tokens": sum(len(row["input_ids"]) for row in rows),
        "training_padded_tokens": padded_tokens,
        "current_examples": len(current),
        "replay_examples": len(replay),
        "current_input_tokens": sum(
            len(tasks[t].rows["train"][i]["input_ids"]) for t, i in current
        ),
        "replay_input_tokens": sum(
            len(tasks[t].rows["train"][i]["input_ids"]) for t, i in replay
        ),
        "examples_by_source_task": [
            sum(t == source for t, _ in refs) for source in range(len(tasks))
        ],
        "examples_by_code": {
            str(code): count
            for code, count in sorted(Counter(row["code"] for row in rows).items())
        },
        "unique_current_examples": len(set(current)),
        "unique_replay_examples": len(set(replay)),
        "replay_pool_examples": pool_examples if replay else 0,
        "schedule_sha256": digest(schedule),
    }


def tokenize_tasks(tasks, tokenizer, cfg, provenance):
    code_ids = []
    for code in string.ascii_uppercase[: cfg.tasks * cfg.classes_per_task]:
        token_ids = tokenizer.encode(code, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != code:
            raise ValueError(f"CODE_NOT_SINGLE_TOKEN {code=} {token_ids=}")
        code_ids.append(token_ids[0])
    if len(set(code_ids)) != len(code_ids):
        raise ValueError("CODE_TOKEN_COLLISION")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide a pad_token_id")
    max_observed = 0
    for task in tasks:
        for rows in task.rows.values():
            for row in rows:
                token_ids = tokenizer.encode(
                    PROMPT.format(text=row["text"]), add_special_tokens=False
                )
                if not token_ids or len(token_ids) > cfg.max_length:
                    raise ValueError(
                        f"PROMPT_LENGTH_INVALID source={row['source_split']}:{row['source_index']} length={len(token_ids)}; increase max_length"
                    )
                row["input_ids"] = token_ids
                max_observed = max(max_observed, len(token_ids))
        task.schedule = balanced_schedule(task.rows["train"], cfg, task.index)
    buffers = replay_buffers(tasks, cfg)
    schedules = training_schedules(tasks, cfg, buffers)
    provenance.update(
        {
            "prompt_template": PROMPT,
            "prompt_uses_chat_template": False,
            "code_token_ids": code_ids,
            "padding": "left",
            "truncation": False,
            "max_observed_input_tokens": max_observed,
            "selected_rows": [
                {split: rows for split, rows in task.rows.items()} for task in tasks
            ],
            "training_reference_format": "[source_task_index, selected_train_row_index]",
            "training_schedules": schedules,
            "replay_support_capacity": cfg.replay_capacity,
            "replay_buffers_after_task": [
                replay_buffer_record(tasks, members) for members in buffers
            ],
            "replay_support_scope": "Replay candidate support only. The loader retains historical rows; this is not a bounded host-RAM claim.",
            "training_budget_per_method": {
                method: [
                    schedule_budget(tasks, schedule, i, cfg)
                    for i, schedule in enumerate(plans)
                ]
                for method, plans in schedules.items()
            },
            "evaluation_sha256": digest(
                [
                    {split: task.rows[split] for split in ("validation", "test")}
                    for task in tasks
                ]
            ),
        }
    )
    provenance["selection_sha256"] = digest(provenance)
    return PreparedData(
        tasks, code_ids, tokenizer.pad_token_id, provenance, schedules, buffers
    )


def load_clinc(cfg, tokenizer):
    info = HfApi().dataset_info(
        DATASET_ID, revision=DATASET_REVISION, files_metadata=True, timeout=60
    )
    if info.sha != DATASET_REVISION:
        raise RuntimeError("DATASET_REVISION_MISMATCH")
    licenses = info.card_data.to_dict().get("license", [])
    if "cc-by-3.0" not in licenses:
        raise RuntimeError(f"DATASET_LICENSE_MISMATCH {licenses=}")
    splits = load_dataset(
        DATASET_ID, "small", revision=DATASET_REVISION, cache_dir=cfg.dataset_cache_dir
    )
    counts = {key: len(value) for key, value in splits.items()}
    if counts != SPLIT_COUNTS:
        raise RuntimeError(f"DATASET_SPLIT_MISMATCH {counts=}")
    names = splits["train"].features["intent"].names
    if len(names) != 151 or names.count("oos") != 1:
        raise RuntimeError("DATASET_LABEL_SCHEMA_MISMATCH")
    tasks, selection = select_rows(splits, names, cfg)
    provenance = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "subset": "small",
        "license": licenses,
        "official_split_counts": counts,
        "excluded_intent": "oos",
        "source": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}",
        "paper": "https://aclanthology.org/D19-1131/",
        "authors": "Larson et al. (2019)",
        "files": [
            {
                "path": f.rfilename,
                "bytes": f.size,
                "git_blob": f.blob_id,
                "lfs_sha256": f.lfs.sha256 if f.lfs else None,
            }
            for f in info.siblings
            if f.rfilename.startswith("small/") or f.rfilename == "README.md"
        ],
        "fingerprints": {
            split: dataset._fingerprint for split, dataset in splits.items()
        },
        **selection,
    }
    return tokenize_tasks(tasks, tokenizer, cfg, provenance)


def collate(rows, pad_token_id, device):
    length = max(len(row["input_ids"]) for row in rows)
    inputs = torch.full(
        (len(rows), length), pad_token_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(inputs)
    for i, row in enumerate(rows):
        size = len(row["input_ids"])
        inputs[i, -size:] = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        mask[i, -size:] = 1
    labels = torch.tensor(
        [row["code"] for row in rows], dtype=torch.long, device=device
    )
    return {"input_ids": inputs, "attention_mask": mask}, labels
