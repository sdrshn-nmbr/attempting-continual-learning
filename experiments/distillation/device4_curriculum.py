import argparse
import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn import functional

from generated_contract import (
    digest,
    file_digest,
    grade_generation,
    learner_prompt,
    make_corpus,
    qualification_gate,
    qualification_rows,
    teacher_prompt,
    validate_protocol,
    verify_qualification,
)
from generated_contract import source_hashes as legacy_source_hashes
from qualified import QualifiedRunner
from run import adapter_parameters, tensor_digest
from tasks import FAMILIES, audit_dataset, digits, execute, paired_gate, to_records

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("distillation.device4")
CONTRACT = "device4_teacher_budget_and_curriculum_20260912"
ARMS = ("flat_mixed", "primitive_first", "legacy96_budget")
SPLITS = ("demonstration", "train", "fallback_validation")
SOURCE_FILES = (
    "device4_curriculum.py", "generated_contract.py", "qualified.py", "run.py",
    "tasks.py", "objectives.py", "requirements.txt",
)


def source_hashes():
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_design(design):
    if design["contract"] != CONTRACT:
        raise ValueError("DEVICE4_UNKNOWN_CONTRACT")
    path = ROOT / design["original_protocol"]["path"]
    if file_digest(path) != design["original_protocol"]["sha256"]:
        raise ValueError("DEVICE4_ORIGINAL_PROTOCOL_CHANGED")
    original = json.loads(path.read_text())
    validate_protocol(original)
    if legacy_source_hashes() != design["original_source_sha256"]:
        raise ValueError("DEVICE4_ORIGINAL_SOURCE_CHANGED")
    if design["training"] != {
        "arms": list(ARMS), "updates_per_arm": 384, "examples_per_update": 4,
        "example_exposures_per_arm": 1536, "checkpoints": [24, 96, 384],
        "learning_rate": 0.0002, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "order_seed": 9111704, "remainder_order_seed": 9111705, "warmup_primitive_exposures_per_family": 64,
        "selection": "fixed_final384", "expected_trainable_parameters": 8519680,
    }:
        raise ValueError("DEVICE4_FIXED_TRAINING_CONTRACT_CHANGED")
    if design["evaluation"] != {
        "validation_split": "fallback_validation", "new_process": True,
        "original_family_gates": True, "minimum_depth_accuracy_for_consolidation": 0.875,
        "test_predictions": False, "composition_predictions": False,
    }:
        raise ValueError("DEVICE4_FIXED_EVALUATION_CONTRACT_CHANGED")
    return original


def legacy_pool(corpus):
    return [task for family in FAMILIES for task in [t for t in corpus["train"] if t.family == family][:32]]


def make_schedules(corpus, design):
    recipe = design["training"]
    old = legacy_pool(corpus)
    flat = [task.uid for task in corpus["train"]] * 5 + [task.uid for task in old]
    random.Random(recipe["order_seed"]).shuffle(flat)
    lookup = {task.uid: task for task in corpus["train"]}
    warm, remaining, counts = [], [], Counter()
    for uid in flat:
        task = lookup[uid]
        if len(task.program) == 1 and counts[task.family] < recipe["warmup_primitive_exposures_per_family"]:
            warm.append(uid)
            counts[task.family] += 1
        else:
            remaining.append(uid)
    if counts != Counter({family: recipe["warmup_primitive_exposures_per_family"] for family in FAMILIES}):
        raise ValueError("DEVICE4_INSUFFICIENT_PRIMITIVE_EXPOSURES")
    random.Random(recipe["remainder_order_seed"]).shuffle(remaining)
    old_schedule = []
    rng = random.Random(recipe["order_seed"])
    for _ in range(16):
        epoch = [task.uid for task in old]
        rng.shuffle(epoch)
        old_schedule.extend(epoch)
    sequences = {"flat_mixed": flat, "primitive_first": warm + remaining, "legacy96_budget": old_schedule}
    if Counter(flat) != Counter(sequences["primitive_first"]):
        raise ValueError("DEVICE4_UNMATCHED_MAIN_EXPOSURE_MULTISET")
    batch = recipe["examples_per_update"]
    schedules = {arm: [ids[i:i + batch] for i in range(0, len(ids), batch)] for arm, ids in sequences.items()}
    if any(len(ids) != recipe["example_exposures_per_arm"] for ids in sequences.values()) or any(
        len(batches) != recipe["updates_per_arm"] or any(len(row) != batch for row in batches)
        for batches in schedules.values()
    ):
        raise ValueError("DEVICE4_EXPOSURE_BUDGET_MISMATCH")
    return schedules


def schedule_audit(corpus, schedules, design):
    lookup = {task.uid: task for task in corpus["train"]}
    result = {}
    for arm, batches in schedules.items():
        ids = [uid for batch in batches for uid in batch]
        panels = Counter(f"{lookup[uid].family}/depth{len(lookup[uid].program)}" for uid in ids)
        result[arm] = {
            "updates": len(batches), "example_exposures": len(ids), "unique_training_rows": len(set(ids)),
            "per_uid_exposures": dict(sorted(Counter(ids).items())), "family_depth_exposures": dict(sorted(panels.items())),
            "checkpoint_exposures": {
                str(step): dict(sorted(Counter(f"{lookup[uid].family}/depth{len(lookup[uid].program)}" for uid in ids[:step * design['training']['examples_per_update']]).items()))
                for step in design["training"]["checkpoints"]
            },
        }
    return result


def verify_history(design, original, corpus, schedules):
    spec = design["history"]
    receipts = {}
    for name in ("initial", "fallback24"):
        folder = Path(spec[name]["path"])
        for filename, expected in spec[name]["files_sha256"].items():
            if file_digest(folder / filename) != expected:
                raise ValueError(f"DEVICE4_HISTORICAL_EVIDENCE_CHANGED: {name}/{filename}")
        receipt = json.loads((folder / "qualification.json").read_text())
        if (
            receipt["status"] != "rejected" or receipt["protocol_sha256"] != digest(original)
            or receipt["source_sha256"] != legacy_source_hashes()
            or receipt["learner_optimizer_updates"] != 0 or receipt["qualification_optimizer_updates"] != 0
        ):
            raise ValueError("DEVICE4_WRONG_HISTORICAL_FAILURE")
        receipts[name] = {**receipt, "pairs": json.loads((folder / "pairs.json").read_text())}
    initial = verify_qualification(Path(spec["initial"]["path"]), original, required_status="rejected")
    fallback = receipts["fallback24"]
    gate = qualification_gate(fallback["pairs"], corpus, original, fallback["eos_token_id"], fallback["special_token_ids"], "fallback_validation")
    historical_training = json.loads((Path(spec["fallback24"]["path"]) / "teacher_training.json").read_text())
    if (
        fallback["candidate"] != "trained_teacher" or fallback["optimizer_updates"] != 24
        or gate != fallback["gate"] or gate["passed"]
        or fallback["adapter_before_sha256"] != initial["adapter_before_sha256"]
        or fallback["adapter_after_sha256"] != initial["adapter_before_sha256"]
        or fallback["snapshot_sha256"] != initial["snapshot_sha256"]
        or historical_training["updates"] != 24 or historical_training["learner_updates"] != 0
        or historical_training["validation_used_in_training"]
        or historical_training["initial_failure_sha256"] != initial["receipt_sha256"]
        or historical_training["training_ids"] != [uid for batch in schedules["legacy96_budget"][:24] for uid in batch]
    ):
        raise ValueError("DEVICE4_HISTORICAL_BUDGET_OR_IDENTITY_MISMATCH")
    feasibility = gate_feasibility(fallback["pairs"], corpus, original, fallback["eos_token_id"], fallback["special_token_ids"])
    return {"initial": initial, "fallback24": fallback, "feasibility": feasibility, "historical_teacher24_sha256": historical_training["teacher_sha256"]}


def gate_feasibility(pairs, corpus, original, eos, specials):
    panels = {}
    tasks = qualification_rows(corpus, original, "fallback_validation")
    if [row["uid"] for row in pairs] != [task.uid for task in tasks]:
        raise ValueError("DEVICE4_RAW_FEASIBILITY_ROWS")
    for task, record in zip(tasks, pairs, strict=True):
        raw = record["learner"]
        if raw["prompt"] != learner_prompt(task):
            raise ValueError("DEVICE4_RAW_PROMPT_CHANGED")
        correct = grade_generation(task, raw["token_ids"], raw["body_text"], eos, specials, original["max_new_tokens"])["correct"]
        panels.setdefault(f"{task.family}/{task.split}", []).append(correct)
    gate = original["qualification"]
    feasible = {key: paired_gate(values, [True] * len(values), gate["minimum_accuracy"], gate["minimum_gain"], gate["maximum_p"]) for key, values in panels.items()}
    if not all(item["passed"] for item in feasible.values()):
        raise ValueError("DEVICE4_ORIGINAL_GAIN_GATE_INFEASIBLE_NO_UPDATES")
    return feasible


def load_runner(original, design, corpus, history, output):
    runner = QualifiedRunner(original, output)
    runner.load()
    runner.corpus = corpus
    runner.check_initialization(history["initial"])
    runner.check_initialization(history["fallback24"])
    if sum(p.numel() for p in runner.student_parameters) != design["training"]["expected_trainable_parameters"]:
        raise ValueError("DEVICE4_ADAPTER_CAPACITY_CHANGED")
    return runner


def forced_diagnostics(runner, task, prompt):
    response = runner.target_tokens(task)
    encoded = runner.tokenizer(digits(task.answer), add_special_tokens=False, return_offsets_mapping=True)
    if encoded.input_ids != response[:-1].tolist():
        raise ValueError("DEVICE4_GOLD_TOKENIZATION_BOUNDARY")
    with torch.no_grad():
        logits = runner.logits(runner.prompt_ids(prompt), response).float()
        logps = functional.log_softmax(logits, dim=-1)
        selected = logps.gather(-1, response[:, None]).squeeze(-1)
        predictions = logits.argmax(dim=-1).tolist()
    positions = []
    for position in range(4):
        indices = [i for i, (start, end) in enumerate(encoded.offset_mapping) if start <= 2 * position < end]
        if not indices:
            raise ValueError("DEVICE4_DIGIT_OFFSET_MISSING")
        positions.append(all(predictions[i] == encoded.input_ids[i] for i in indices))
    logp = float(selected.sum())
    return {
        "gold_sequence_log_probability_including_eos": logp,
        "gold_sequence_probability_including_eos": math.exp(logp),
        "teacher_forced_position_correct": positions,
        "teacher_forced_eos_probability": float(selected[-1].exp()),
        "teacher_forced_eos_argmax_correct": predictions[-1] == runner.eos,
    }


def diagnostic_record(runner, task, demonstrations, privileged=True, forced=True):
    prompt = teacher_prompt(task, demonstrations) if privileged else learner_prompt(task)
    generated = runner.generated_record(task, prompt)
    gold = digits(task.answer)
    body = generated["body_text"]
    wrong = not generated["correct"]
    row = {
        "uid": task.uid, "family": task.family, "split": task.split, "depth": len(task.program),
        "task_sha256": digest(asdict(task)), "program": list(task.program), "generation": generated,
        "literal_position_correct": [len(body) > 2 * i and body[2 * i] == gold[2 * i] for i in range(4)],
        "wrong_state_with_valid_format_and_eos": wrong and generated["format_valid"] and generated["terminated"],
        "returned_input_wrong_state": wrong and body == digits(task.initial),
        "stopped_after_first_command_wrong_state": wrong and len(task.program) == 2 and body == digits(execute(task.family, task.initial, task.program[:1])),
    }
    if forced:
        row["forced"] = forced_diagnostics(runner, task, prompt)
    return row


def measure(runner, tasks, output, label, privileged=True, forced=True):
    records = []
    for task in tasks:
        row = diagnostic_record(runner, task, runner.corpus["demonstration"], privileged, forced)
        records.append(row)
        runner.event("device4_measure", label=label, uid=task.uid, depth=len(task.program), correct=row["generation"]["correct"])
    runner.assert_frozen()
    return records


def verify_records(tasks, records, corpus, original, eos, specials, privileged=True):
    if [row["uid"] for row in records] != [task.uid for task in tasks]:
        raise ValueError("DEVICE4_INCOMPLETE_OR_REORDERED_RECORDS")
    for task, row in zip(tasks, records, strict=True):
        generated = row["generation"]
        prompt = teacher_prompt(task, corpus["demonstration"]) if privileged else learner_prompt(task)
        grade = grade_generation(task, generated["token_ids"], generated["body_text"], eos, specials, original["max_new_tokens"])
        if (
            row["task_sha256"] != digest(asdict(task)) or row["family"] != task.family
            or row["split"] != task.split or row["depth"] != len(task.program)
            or row["program"] != list(task.program) or generated["prompt"] != prompt
            or len(generated["token_ids"]) > original["max_new_tokens"]
            or any(generated[key] != value for key, value in grade.items())
        ):
            raise ValueError("DEVICE4_GENERATION_RECORD_CHANGED")


def summarize(records):
    groups = {}
    for row in records:
        for key in ("all", row["family"], f"depth{row['depth']}", f"{row['family']}/depth{row['depth']}", f"{row['family']}/program/{','.join(row['program'])}"):
            groups.setdefault(key, []).append(row)
    result = {}
    for key, rows in groups.items():
        n = len(rows)
        panel = {
            "n": n, "correct": sum(row["generation"]["correct"] for row in rows),
            "accuracy": sum(row["generation"]["correct"] for row in rows) / n,
            "format_valid": sum(row["generation"]["format_valid"] for row in rows),
            "terminated": sum(row["generation"]["terminated"] for row in rows),
            "literal_position_accuracy": [sum(row["literal_position_correct"][i] for row in rows) / n for i in range(4)],
        }
        for field in ("wrong_state_with_valid_format_and_eos", "returned_input_wrong_state", "stopped_after_first_command_wrong_state"):
            panel[field] = sum(row[field] for row in rows)
        if all("forced" in row for row in rows):
            for field in ("gold_sequence_log_probability_including_eos", "gold_sequence_probability_including_eos", "teacher_forced_eos_probability"):
                panel[f"mean_{field}"] = sum(row["forced"][field] for row in rows) / n
            panel["teacher_forced_position_accuracy"] = [sum(row["forced"]["teacher_forced_position_correct"][i] for row in rows) / n for i in range(4)]
        result[key] = panel
    return result


def frozen_digest(runner):
    return tensor_digest({name: value for name, value in runner.model.named_parameters() if ".lora_" not in name})


def reset_teacher(runner):
    runner.model.set_adapter("teacher")
    runner.model.eval()
    with torch.no_grad():
        for name, parameter in adapter_parameters(runner.model, "teacher").items():
            parameter.copy_(runner.initial_adapter[name])
    if tensor_digest(adapter_parameters(runner.model, "teacher")) != tensor_digest(runner.initial_adapter):
        raise ValueError("DEVICE4_MATCHED_INITIALIZATION_FAILED")
    return list(adapter_parameters(runner.model, "teacher").values())


def save_checkpoint(runner, folder):
    runner.model.save_pretrained(folder, selected_adapters=["teacher"], save_embedding_layers=False, safe_serialization=True)
    path = folder / "teacher"
    return {
        "path": str(path.resolve()), "tensor_sha256": tensor_digest(adapter_parameters(runner.model, "teacher")),
        "files_sha256": {name: file_digest(path / name) for name in ("adapter_model.safetensors", "adapter_config.json")},
    }


def train_arm(runner, design, arm, schedule, output):
    recipe = design["training"]
    torch.manual_seed(recipe["order_seed"])
    parameters = reset_teacher(runner)
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    tasks = {task.uid: task for task in runner.corpus["train"]}
    before_base = frozen_digest(runner)
    ledger, checkpoints = [], {}
    for step, ids in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        losses, responses = [], []
        for uid in ids:
            task = tasks[uid]
            prompt = teacher_prompt(task, runner.corpus["demonstration"])
            response = runner.target_tokens(task)
            logits = runner.logits(runner.prompt_ids(prompt), response)
            loss = functional.cross_entropy(logits.float(), response)
            if not torch.isfinite(loss):
                raise ValueError(f"DEVICE4_NONFINITE_LOSS: {arm}/{step}")
            (loss / len(ids)).backward()
            losses.append(float(loss.detach()))
            responses.append(response.tolist())
            del loss, logits
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, recipe["max_grad_norm"]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"DEVICE4_INVALID_GRADIENT: {arm}/{step}")
        optimizer.step()
        runner.assert_frozen()
        if any(parameter.grad is not None for parameter in runner.student_parameters):
            raise ValueError("DEVICE4_RAW_REFERENCE_RECEIVED_GRADIENT")
        record = {"step": step, "ids": ids, "response_sha256": [digest(row) for row in responses], "loss_tokens": sum(len(row) for row in responses), "loss": sum(losses) / len(losses), "gradient_norm": norm}
        ledger.append(record)
        runner.event("device4_teacher_update", arm=arm, **record)
        if step in recipe["checkpoints"]:
            optimizer.zero_grad(set_to_none=True)
            folder = output / arm / f"checkpoint{step}"
            checkpoint = save_checkpoint(runner, folder)
            torch.save(optimizer.state_dict(), folder / "optimizer.pt")
            records = measure(runner, runner.corpus["train"], output, f"{arm}/train/{step}")
            write_new(folder / "train_diagnostics.json", records)
            checkpoints[str(step)] = {
                "adapter": checkpoint, "optimizer_sha256": file_digest(folder / "optimizer.pt"),
                "train_diagnostics_sha256": file_digest(folder / "train_diagnostics.json"), "train_summary": summarize(records),
            }
    runner.assert_frozen()
    if frozen_digest(runner) != before_base or tensor_digest(adapter_parameters(runner.model, "student")) != tensor_digest(runner.initial_adapter):
        raise ValueError("DEVICE4_FROZEN_BASE_OR_RAW_REFERENCE_MUTATED")
    if checkpoints[str(recipe["updates_per_arm"])]["adapter"]["tensor_sha256"] == tensor_digest(runner.initial_adapter):
        raise ValueError("DEVICE4_TEACHER_UNCHANGED_AFTER_UPDATES")
    write_new(output / arm / "ledger.json", ledger)
    return {
        "updates": len(ledger), "example_exposures": sum(len(row["ids"]) for row in ledger),
        "loss_token_exposures": sum(row["loss_tokens"] for row in ledger), "ledger_sha256": file_digest(output / arm / "ledger.json"),
        "checkpoints": checkpoints, "selected_checkpoint": recipe["updates_per_arm"], "learner_updates": 0,
        "base_before_sha256": before_base, "base_after_sha256": frozen_digest(runner),
    }


def train(design, original, corpus, schedules, history, output):
    runner = load_runner(original, design, corpus, history, output)
    reset_teacher(runner)
    baseline = measure(runner, corpus["train"], output, "raw_instruction_demo/train")
    write_new(output / "raw_instruction_demo_train.json", baseline)
    initial = save_checkpoint(runner, output / "initial")
    arms = {arm: train_arm(runner, design, arm, schedules[arm], output) for arm in ARMS}
    if arms["flat_mixed"]["loss_token_exposures"] != arms["primitive_first"]["loss_token_exposures"]:
        raise ValueError("DEVICE4_MATCHED_TOKEN_BUDGET_FAILED")
    result = {
        "status": "trained_evaluation_pending", "pid": os.getpid(), "source_sha256": source_hashes(), "design_sha256": digest(design),
        "original_protocol_sha256": digest(original), "dataset_sha256": digest(to_records(corpus)),
        "snapshot_sha256": runner.snapshot, "eos_token_id": runner.eos, "special_token_ids": runner.special_ids,
        "initial": initial, "arms": arms, "learner_updates": 0, "new_validation_predictions_observed_during_training": False,
        "historical_validation_receipts_read_before_updates": True,
        "test_or_composition_predictions_observed": False, "trainable_parameters": sum(p.numel() for p in runner.student_parameters),
        "historical24_tensor_parity": arms["legacy96_budget"]["checkpoints"]["24"]["adapter"]["tensor_sha256"] == history["historical_teacher24_sha256"],
        "historical24_parity_boundary": "Exact row order, initialization, model, prompt, loss and budget are matched. Tensor equality is measured; a different runtime or numerical path can prevent bitwise replication.",
        "files_sha256": {name: file_digest(output / name) for name in ("schedule.json", "schedule_audit.json", "raw_instruction_demo_train.json", "seal.json")},
    }
    write_new(output / "training.json", result)
    return result


def verify_training(folder, design, original, corpus, schedules):
    training = json.loads((folder / "training.json").read_text())
    if (
        training["status"] != "trained_evaluation_pending" or training["pid"] == os.getpid()
        or training["source_sha256"] != source_hashes() or training["design_sha256"] != digest(design)
        or training["original_protocol_sha256"] != digest(original) or training["dataset_sha256"] != digest(to_records(corpus))
        or training["new_validation_predictions_observed_during_training"] or training["test_or_composition_predictions_observed"]
        or training["learner_updates"] != 0 or set(training["arms"]) != set(ARMS)
        or training["trainable_parameters"] != design["training"]["expected_trainable_parameters"]
    ):
        raise ValueError("DEVICE4_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
    for name, expected in training["files_sha256"].items():
        if file_digest(folder / name) != expected:
            raise ValueError(f"DEVICE4_TRAINING_ARTIFACT_CHANGED: {name}")
    if json.loads((folder / "schedule.json").read_text()) != schedules:
        raise ValueError("DEVICE4_SCHEDULE_CHANGED")
    for arm, receipt in training["arms"].items():
        recipe = design["training"]
        if (
            receipt["updates"] != recipe["updates_per_arm"] or receipt["example_exposures"] != recipe["example_exposures_per_arm"]
            or receipt["selected_checkpoint"] != recipe["updates_per_arm"] or receipt["learner_updates"] != 0
            or receipt["base_before_sha256"] != receipt["base_after_sha256"]
            or set(receipt["checkpoints"]) != {str(step) for step in recipe["checkpoints"]}
            or file_digest(folder / arm / "ledger.json") != receipt["ledger_sha256"]
        ):
            raise ValueError("DEVICE4_ARM_BUDGET_OR_CHECKPOINT_CHANGED")
        ledger = json.loads((folder / arm / "ledger.json").read_text())
        if [row["ids"] for row in ledger] != schedules[arm] or [row["step"] for row in ledger] != list(range(1, recipe["updates_per_arm"] + 1)):
            raise ValueError("DEVICE4_EXPOSURE_LEDGER_CHANGED")
        if sum(row["loss_tokens"] for row in ledger) != receipt["loss_token_exposures"]:
            raise ValueError("DEVICE4_TOKEN_EXPOSURE_CHANGED")
        for step, checkpoint in receipt["checkpoints"].items():
            directory = folder / arm / f"checkpoint{step}"
            if file_digest(directory / "optimizer.pt") != checkpoint["optimizer_sha256"] or file_digest(directory / "train_diagnostics.json") != checkpoint["train_diagnostics_sha256"]:
                raise ValueError("DEVICE4_CHECKPOINT_DIAGNOSTICS_CHANGED")
            records = json.loads((directory / "train_diagnostics.json").read_text())
            verify_records(corpus["train"], records, corpus, original, training["eos_token_id"], training["special_token_ids"])
            if summarize(records) != checkpoint["train_summary"]:
                raise ValueError("DEVICE4_TRAIN_DIAGNOSTIC_SUMMARY_CHANGED")
    return training


def restore(runner, checkpoint):
    path = Path(checkpoint["path"])
    if set(checkpoint["files_sha256"]) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("DEVICE4_CHECKPOINT_FILE_SET")
    for name, expected in checkpoint["files_sha256"].items():
        if file_digest(path / name) != expected:
            raise ValueError(f"DEVICE4_CHECKPOINT_CHANGED: {name}")
    state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    set_peft_model_state_dict(runner.model, state, adapter_name="teacher")
    runner.model.set_adapter("teacher", inference_mode=True)
    runner.model.eval()
    if tensor_digest(adapter_parameters(runner.model, "teacher")) != checkpoint["tensor_sha256"]:
        raise ValueError("DEVICE4_SAVED_TEACHER_IDENTITY_MISMATCH")


def evaluate(design, original, corpus, schedules, history, folder, output):
    training = verify_training(folder, design, original, corpus, schedules)
    runner = load_runner(original, design, corpus, history, output)
    if runner.snapshot != training["snapshot_sha256"] or runner.eos != training["eos_token_id"] or runner.special_ids != training["special_token_ids"]:
        raise ValueError("DEVICE4_RELOAD_RUNTIME_IDENTITY_CHANGED")
    tasks = qualification_rows(corpus, original, "fallback_validation")
    runner.model.set_adapter("student", inference_mode=True)
    raw = measure(runner, tasks, output, "raw_learner/qualification", privileged=False, forced=False)
    raw_pairs = [{"uid": row["uid"], "learner": row["generation"]} for row in raw]
    gate_feasibility(raw_pairs, corpus, original, runner.eos, runner.special_ids)
    write_new(output / "raw_learner.json", raw)
    results, panels = {}, {}
    lookup = {task.uid: task for task in corpus["train"]}
    for arm in ARMS:
        receipt = training["arms"][arm]
        ledger = json.loads((folder / arm / "ledger.json").read_text())
        for batch, row in zip(schedules[arm], ledger, strict=True):
            targets = [runner.target_tokens(lookup[uid]).tolist() for uid in batch]
            if row["response_sha256"] != [digest(target) for target in targets] or row["loss_tokens"] != sum(len(target) for target in targets):
                raise ValueError("DEVICE4_GOLD_TARGET_LEDGER_CHANGED")
        checkpoint = receipt["checkpoints"][str(design["training"]["updates_per_arm"])]["adapter"]
        restore(runner, checkpoint)
        measured = measure(runner, tasks, output, f"{arm}/qualification")
        saved = json.loads((folder / arm / f"checkpoint{design['training']['updates_per_arm']}" / "train_diagnostics.json").read_text())
        saved_train = {row["uid"]: row for row in saved}
        if any(row["generation"] != saved_train[row["uid"]]["generation"] for row in measured if row["split"] == "train"):
            raise ValueError(f"DEVICE4_RELOAD_GENERATION_PARITY: {arm}")
        if frozen_digest(runner) != receipt["base_before_sha256"]:
            raise ValueError("DEVICE4_RELOADED_BASE_CHANGED")
        pairs = [
            {"uid": task.uid, "family": task.family, "split": task.split, "learner": base["generation"], "teacher": trained["generation"]}
            for task, base, trained in zip(tasks, raw, measured, strict=True)
        ]
        gate = qualification_gate(pairs, corpus, original, runner.eos, runner.special_ids, "fallback_validation")
        summaries = {split: summarize([row for row in measured if row["split"] == split]) for split in ("train", "fallback_validation")}
        depth_passed = all(summaries[split][f"{family}/depth{depth}"]["accuracy"] >= design["evaluation"]["minimum_depth_accuracy_for_consolidation"] for split in summaries for family in FAMILIES for depth in (1, 2))
        panels[arm] = {"pairs": pairs, "diagnostics": measured}
        results[arm] = {
            "status": "qualified" if gate["passed"] else "rejected", "gate": gate, "summary": summaries,
            "depth_accuracy_check_passed": depth_passed, "eligible_for_depth3_4_prerequisite": gate["passed"] and depth_passed,
            "checkpoint": checkpoint, "checkpoint_update": design["training"]["updates_per_arm"],
            "reload_generation_parity": True, "learner_updates": 0,
        }
    write_new(output / "panels.json", panels)
    comparisons = {}
    for split in ("train", "fallback_validation"):
        comparisons[split] = {
            group: {
                "primitive_first_minus_flat_mixed": results["primitive_first"]["summary"][split][group]["accuracy"] - results["flat_mixed"]["summary"][split][group]["accuracy"],
                "flat_mixed_minus_legacy96_budget": results["flat_mixed"]["summary"][split][group]["accuracy"] - results["legacy96_budget"]["summary"][split][group]["accuracy"],
            }
            for group in ("all", "depth1", "depth2", *FAMILIES)
        }
    result = {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"],
        "source_sha256": source_hashes(), "design_sha256": digest(design), "training_sha256": file_digest(folder / "training.json"),
        "panels_sha256": file_digest(output / "panels.json"), "arms": results, "comparisons": comparisons,
        "learner_updates": 0, "test_predictions": 0, "composition_predictions": 0,
        "claim_boundary": design["claim_boundary"],
        "next_experiment": design["conditional_followthrough"],
    }
    write_new(output / "qualification.json", result)
    return result


def evaluate_child(output, dispatch):
    child = {"stage": "evaluate", "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "training_dir": str(output.resolve())}
    write_new(output / "evaluation_config.json", child)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run([
        "uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "device4_curriculum.py"),
        "--config", str((output / "evaluation_config.json").resolve()), "--output-dir", str((output / "evaluation").resolve()),
    ], check=True, cwd=ROOT)
    path = output / "evaluation/qualification.json"
    result = json.loads(path.read_text())
    if result["status"] != "completed" or result["pid"] == os.getpid() or result["training_pid"] != os.getpid():
        raise ValueError("DEVICE4_CHILD_EVALUATION_PID_MISMATCH")
    write_new(output / "result.json", {
        "status": "completed", "training_pid": os.getpid(), "evaluation_pid": result["pid"],
        "qualification_path": str(path), "qualification_sha256": file_digest(path),
        "arms": result["arms"], "comparisons": result["comparisons"], "learner_updates": 0,
        "claim_boundary": result["claim_boundary"],
    })


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [device4] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"run", "train", "evaluate"}:
        raise ValueError("DEVICE4_DISPATCH_PROTOCOL_MISMATCH")
    original = validate_design(design)
    corpus = make_corpus(original, SPLITS)
    schedules = make_schedules(corpus, design)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("DEVICE4_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("DEVICE4_OUTPUT_DISPATCH_MISMATCH")
    seal = {"design": design, "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset_sha256": digest(to_records(corpus)), "dataset_audit": audit_dataset(corpus), "pid": os.getpid(), "dispatch": dispatch}
    write_new(output / "seal.json", seal)
    write_new(output / "schedule.json", schedules)
    write_new(output / "schedule_audit.json", schedule_audit(corpus, schedules, design))
    if args.validate_only:
        write_new(output / "validation.json", {"status": "cpu_contract_validated", "gpu_qualified": False})
        return
    try:
        history = verify_history(design, original, corpus, schedules)
        if dispatch["stage"] == "evaluate":
            evaluate(design, original, corpus, schedules, history, Path(dispatch["training_dir"]), output)
        else:
            train(design, original, corpus, schedules, history, output)
            if source_hashes() != seal["source_sha256"]:
                raise ValueError("DEVICE4_SOURCE_CHANGED_DURING_TRAINING")
            if dispatch["stage"] == "run":
                evaluate_child(output, dispatch)
        if source_hashes() != seal["source_sha256"]:
            raise ValueError("DEVICE4_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_new(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("DEVICE4_FAILED")
        raise


if __name__ == "__main__":
    main()
