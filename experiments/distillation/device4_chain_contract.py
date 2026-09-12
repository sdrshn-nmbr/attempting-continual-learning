import random
import re
from collections import Counter
from dataclasses import asdict, replace

from device4_curriculum import ARMS, verify_records
from generated_contract import digest, grade_generation, teacher_prompt

from tasks import FAMILIES, paired_gate

CONTRACT = "device4_external_chain_to_persistent_family_learner_20260912"
CANDIDATES = ("raw_privileged", *ARMS)
GATE = {"minimum_accuracy": 0.875, "minimum_gain": 0.25, "maximum_p": 0.05}
TRAINING = {
    "updates_per_method": 128, "examples_per_update": 4, "example_exposures_per_method": 512,
    "epochs": 4, "checkpoint_updates": [32, 96, 128], "selection": "fixed_final128",
    "order_seed": 91235001, "probe_rows_per_depth": 4, "learning_rate": 0.0002,
    "weight_decay": 0.0, "max_grad_norm": 1.0, "rank": 16, "alpha": 32,
    "expected_trainable_parameters": 8519680, "shared_oracle_per_family": True,
}


def family_rows(corpus, split, family):
    return [task for task in corpus[split] if task.family == family]


def primitive_rows(corpus):
    return {
        split: [task for task in corpus[split] if len(task.program) == 1]
        for split in ("train", "fallback_validation")
    }


def gates_by_family(tasks, raw, candidates, original, corpus, eos, specials, gate):
    results = {candidate: {} for candidate in CANDIDATES}
    for split, rows in tasks.items():
        verify_records(rows, raw[split], corpus, original, eos, specials, privileged=False)
        for candidate in CANDIDATES:
            records = candidates[candidate][split]
            verify_records(rows, records, corpus, original, eos, specials)
            for family in FAMILIES:
                indices = [i for i, task in enumerate(rows) if task.family == family]
                if not indices:
                    raise ValueError("CHAIN_MISSING_PRIMITIVE_FAMILY_SPLIT")
                result = paired_gate(
                    [raw[split][i]["generation"]["correct"] for i in indices],
                    [records[i]["generation"]["correct"] for i in indices], **gate,
                )
                comparison = paired_gate(
                    [candidates["raw_privileged"][split][i]["generation"]["correct"] for i in indices],
                    [records[i]["generation"]["correct"] for i in indices], **gate,
                )
                results[candidate].setdefault(family, {"panels": {}, "versus_raw_privileged_diagnostic": {}})
                results[candidate][family]["panels"][split] = result
                results[candidate][family]["versus_raw_privileged_diagnostic"][split] = comparison
    for panels in results.values():
        for result in panels.values():
            result["passed"] = all(panel["passed"] for panel in result["panels"].values())
    return results


def step_prompt(task, state, index, demonstrations):
    if any(row.family == task.family and row.initial == tuple(state) for row in demonstrations):
        return None
    step = replace(task, initial=tuple(state), program=(task.program[index],), answer=())
    return teacher_prompt(step, demonstrations)


def state_from_generation(generation, eos, specials, cap):
    tokens, body = generation["token_ids"], generation["body_text"]
    if not tokens or len(tokens) > cap or tokens[-1] != eos:
        return None
    if any(token in specials for token in tokens[:-1]) or re.fullmatch(r"[0-9](?: [0-9]){3}", body) is None:
        return None
    return tuple(map(int, body.split(" ")))


def chain_record(task, steps, demonstrations, original, eos, specials):
    state, aborted = task.initial, None
    if not steps or len(steps) > len(task.program):
        raise ValueError("CHAIN_TRACE_STEP_COUNT")
    calls = prompt_tokens = response_tokens = 0
    for index, step in enumerate(steps):
        prompt = step_prompt(task, state, index, demonstrations)
        if step["index"] != index or step["input_state"] != list(state) or step["command"] != task.program[index]:
            raise ValueError("CHAIN_STATE_MUST_EQUAL_PREVIOUS_EMISSION")
        if prompt is None:
            if step["generation"] is not None:
                raise ValueError("CHAIN_DEMONSTRATION_COLLISION_MUST_ABORT")
            aborted = "demonstration_input_collision"
        else:
            generated = step["generation"]
            if generated is None or generated["prompt"] != prompt:
                raise ValueError("CHAIN_SINGLE_COMMAND_PROMPT_CHANGED")
            calls += 1
            prompt_tokens += len(generated["prompt_token_ids"])
            response_tokens += len(generated["token_ids"])
            state = state_from_generation(generated, eos, specials, original["max_new_tokens"])
            if state is None:
                aborted = "malformed_or_missing_native_eos"
        if aborted is not None and index != len(steps) - 1:
            raise ValueError("CHAIN_CONTINUED_AFTER_INVALID_STEP")
    if aborted is None and len(steps) != len(task.program):
        raise ValueError("CHAIN_INCOMPLETE_WITHOUT_ABORT")
    final = steps[-1]["generation"] if aborted is None else None
    correct = final is not None and grade_generation(
        task, final["token_ids"], final["body_text"], eos, specials, original["max_new_tokens"],
    )["correct"]
    return {
        "uid": task.uid, "family": task.family, "split": task.split,
        "depth": len(task.program), "program": list(task.program), "task_sha256": digest(asdict(task)),
        "steps": steps, "aborted": aborted, "completed": final is not None, "correct": correct,
        "final_generation": final, "model_calls": calls,
        "prompt_tokens": prompt_tokens, "response_tokens": response_tokens,
    }


def verify_chains(tasks, records, corpus, original, eos, specials):
    if [r["uid"] for r in records] != [t.uid for t in tasks]:
        raise ValueError("CHAIN_ROWS_MISSING_REORDERED_OR_SUBSTITUTED")
    for task, record in zip(tasks, records, strict=True):
        if record != chain_record(task, record["steps"], corpus["demonstration"], original, eos, specials):
            raise ValueError("CHAIN_TRACE_RECOMPUTATION_FAILED")


def family_gate(corpus, family, raw, chains, original, eos, specials, gate):
    panels, train_complete = {}, True
    for split in ("train", "validation"):
        rows = family_rows(corpus, split, family)
        verify_records(rows, raw[split], corpus, original, eos, specials, privileged=False)
        verify_chains(rows, chains[split], corpus, original, eos, specials)
        for depth in (3, 4):
            indices = [i for i, task in enumerate(rows) if len(task.program) == depth]
            if not indices:
                raise ValueError("CHAIN_MISSING_DEEP_FAMILY_CELL")
            panels[f"{split}/depth{depth}"] = paired_gate(
                [raw[split][i]["generation"]["correct"] for i in indices],
                [chains[split][i]["correct"] for i in indices], **gate,
            )
        if split == "train":
            train_complete = all(row["completed"] for row in chains[split])
    accuracy_passed = all(panel["passed"] for panel in panels.values())
    return {
        "panels": panels, "whole_chain_gate_passed": accuracy_passed,
        "complete_train_cache": train_complete, "passed": accuracy_passed and train_complete,
    }


def family_feasibility(corpus, family, raw, original, eos, specials, gate):
    panels = {}
    for split in ("train", "validation"):
        rows = family_rows(corpus, split, family)
        verify_records(rows, raw[split], corpus, original, eos, specials, privileged=False)
        for depth in (3, 4):
            scores = [record["generation"]["correct"] for task, record in zip(rows, raw[split], strict=True) if len(task.program) == depth]
            panels[f"{split}/depth{depth}"] = paired_gate(scores, [True] * len(scores), **gate)
    return {"passed": all(v["passed"] for v in panels.values()), "panels": panels}


def cache_outputs(tasks, records):
    if len(tasks) != len(records) or any(not r["completed"] for r in records):
        raise ValueError("CHAIN_COMPLETE_TRAIN_OUTPUTS_REQUIRED_NO_FILTERING")
    cache = []
    for task, record in zip(tasks, records, strict=True):
        if task.uid != record["uid"] or task.split != "train":
            raise ValueError("CHAIN_CACHE_ONLY_EXACT_TRAIN_ROWS")
        generated = record["final_generation"]
        cache.append({
            "uid": task.uid, "task_sha256": digest(asdict(task)), "chain_sha256": digest(record),
            "response_token_ids": generated["token_ids"], "body_text": generated["body_text"],
            "raw_text": generated["raw_text"], "teacher_correct": record["correct"],
        })
    return cache


def train_schedule(corpus, family, recipe):
    rows = family_rows(corpus, "train", family)
    rng, ids = random.Random(f"{recipe['order_seed']}:{family}"), []
    for _ in range(recipe["epochs"]):
        epoch = [row.uid for row in rows]
        rng.shuffle(epoch)
        ids.extend(epoch)
    batch = recipe["examples_per_update"]
    schedule = [ids[i:i + batch] for i in range(0, len(ids), batch)]
    if (
        len(ids) != recipe["example_exposures_per_method"] or len(schedule) != recipe["updates_per_method"]
        or any(len(part) != batch for part in schedule)
        or Counter(ids) != {row.uid: recipe["epochs"] for row in rows}
    ):
        raise ValueError("CHAIN_FAMILY_EXPOSURE_BUDGET_CHANGED")
    return schedule


def train_probes(corpus, family, recipe):
    return [
        task for depth in (3, 4)
        for task in [t for t in family_rows(corpus, "train", family) if len(t.program) == depth][:recipe["probe_rows_per_depth"]]
    ]


def methods_for(qualified, family):
    candidates = qualified["eligible"][family]
    return {"oracle_sft": None, **{f"chain_output_{name}": name for name in candidates}}
