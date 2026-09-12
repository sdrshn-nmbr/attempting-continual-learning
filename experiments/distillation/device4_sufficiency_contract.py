import random
from collections import Counter

from generated_contract import digest

CONTRACT = "device4_fixed_budget_train_sufficiency_20260912"
RECIPE = {
    "start_update": 128, "final_update": 1024, "checkpoints": [128, 256, 512, 1024],
    "examples_per_update": 4, "train_rows": 128, "new_updates": 896,
    "new_example_exposures": 3584, "order_seed": 91235001,
    "learning_rate": 0.0002, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "expected_trainable_parameters": 8519680,
}
ARMS = {
    "A_oracle": ("A", "oracle_sft"),
    "A_chain": ("A", "chain_output_flat_mixed"),
    "B_oracle": ("B", "oracle_sft"),
    "C_oracle": ("C", "oracle_sft"),
}


def schedule_for(tasks, family, recipe):
    ids = [task.uid for task in tasks if task.family == family]
    count = recipe["final_update"] * recipe["examples_per_update"]
    if len(ids) != recipe["train_rows"] or count % len(ids):
        raise ValueError("SUFFICIENCY_EXACT_EPOCHS_REQUIRED")
    rng, stream = random.Random(f"{recipe['order_seed']}:{family}"), []
    for _ in range(count // len(ids)):
        epoch = ids.copy()
        rng.shuffle(epoch)
        stream.extend(epoch)
    if Counter(stream) != {uid: count // len(ids) for uid in ids}:
        raise ValueError("SUFFICIENCY_EXPOSURE_MULTISET_CHANGED")
    batch = recipe["examples_per_update"]
    return [stream[i:i + batch] for i in range(0, len(stream), batch)]


def verify_prefix(schedule, old_schedule, ledger, start):
    if (
        len(old_schedule) != start or schedule[:start] != old_schedule
        or [r["ids"] for r in ledger] != old_schedule
        or [r["step"] for r in ledger] != list(range(1, start + 1))
    ):
        raise ValueError("SUFFICIENCY_ORIGINAL_128_PREFIX_NOT_EXACT")


def acquisition(records, overall=0.95, each_depth=0.9):
    groups = {"all": records, **{f"depth{d}": [r for r in records if r["depth"] == d] for d in (3, 4)}}
    if not records or any(not rows for rows in groups.values()):
        raise ValueError("SUFFICIENCY_MISSING_TRAIN_DEPTH")
    counts = {}
    for name, rows in groups.items():
        counts[name] = {
            "n": len(rows), "gold_correct": sum(r["generation"]["correct"] for r in rows),
            "supplied_target_match": sum(r["supplied_target_match"] for r in rows),
            "format_valid": sum(r["generation"]["format_valid"] for r in rows),
            "native_eos": sum(r["generation"]["terminated"] for r in rows),
            "mean_gold_teacher_forced_token_loss": sum(r["gold_teacher_forced_token_loss"] for r in rows) / len(rows),
            "mean_supplied_teacher_forced_token_loss": sum(r["supplied_teacher_forced_token_loss"] for r in rows) / len(rows),
        }
    passed = all(row["gold_correct"] / row["n"] >= (overall if name == "all" else each_depth) for name, row in counts.items())
    return {"passed": passed, "panels": counts, "criterion": "gold_whole_answer_native_eos", "overall_threshold": overall, "depth_threshold": each_depth}


def verify_cached_targets(tasks, cache, expected_sha):
    if digest(cache) != expected_sha or [r["uid"] for r in cache] != [t.uid for t in tasks]:
        raise ValueError("SUFFICIENCY_CACHED_TEACHER_TARGETS_CHANGED")
    return {r["uid"]: r for r in cache}
