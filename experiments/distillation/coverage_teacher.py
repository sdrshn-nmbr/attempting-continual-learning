import argparse
import gc
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file

from choice_consolidation import adapter_state, check_base, load_base, tensor_hash
from choice_consolidation import evaluate as evaluate_choices
from choice_contract import ROOT, TASKS, digest, file_hash, load_corpus, write_json
from generation_consolidation import source_hashes as previous_source_hashes
from generation_consolidation import validate_design as validate_previous_design
from generation_consolidation import verify_checkpoint_files, verify_teacher_receipt
from generation_teacher import generation_gate, measure, verify_generated_rows

CONTRACT = "sequence303_complete_train_binding_coverage_20260912"
SOURCE_FILES = (
    "coverage_teacher.py", "generation_consolidation.py", "generation_teacher.py",
    "choice_consolidation.py", "choice_contract.py", "requirements.txt",
)
LOGGER = logging.getLogger("distillation.coverage_teacher")


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    if design["contract"] != CONTRACT or design["coverage"] != {
        "candidates": [32, 128, 384], "split": "train", "required_cells": 72,
        "minimum_position_accuracy": 1.0, "require_native_eos_and_exact_format": True,
        "selection": "earliest_all_nonempty_cells_perfect_else_reject",
        "teacher_optimizer_updates": 0,
    }:
        raise ValueError("COVERAGE_FIXED_TRAIN_RULE_CHANGED")
    previous_path = ROOT / design["previous_design"]["path"]
    if file_hash(previous_path) != design["previous_design"]["sha256"] or previous_source_hashes() != design["previous_source_sha256"]:
        raise ValueError("COVERAGE_PREVIOUS_EXPERIMENT_CHANGED")
    previous = json.loads(previous_path.read_text())
    teacher_design, choice = validate_previous_design(previous)
    return previous, teacher_design, choice


def coverage_gate(rows, records, teacher_design, eos, specials):
    verify_generated_rows(rows, records, teacher_design, eos, specials)
    cells = {
        f"{task}/position{position + 1}/input{digit}": {"n": 0, "correct": 0, "format_and_eos_valid": 0, "expected_digits": set(), "wrong_ids": []}
        for task in TASKS for position in range(3) for digit in range(8)
    }
    for row, record in zip(rows, records, strict=True):
        inputs = [int(value) for value in row["group"].split()]
        gold = [int(value) for value in row["choices"][row["gold_idx"]].split()]
        if len(inputs) != 3 or len(gold) != 3 or any(not 0 <= value <= 7 for value in inputs + gold):
            raise ValueError("COVERAGE_SEQUENCE_BINDING_CONTRACT")
        generated = record["generation"]
        valid = generated["format_valid"] and generated["terminated"]
        for position, digit in enumerate(inputs):
            cell = cells[f"{row['task']}/position{position + 1}/input{digit}"]
            correct = valid and generated["digit_position_correct"][position]
            cell["n"] += 1
            cell["correct"] += int(correct)
            cell["format_and_eos_valid"] += int(valid)
            cell["expected_digits"].add(gold[position])
            if not correct:
                cell["wrong_ids"].append(row["id"])
    for cell in cells.values():
        if len(cell["expected_digits"]) > 1:
            raise ValueError("COVERAGE_FIXTURE_IS_NOT_POSITIONWISE_BINDING")
        cell["expected_digits"] = sorted(cell["expected_digits"])
        cell["accuracy"] = cell["correct"] / cell["n"] if cell["n"] else None
        cell["passed"] = cell["n"] > 0 and cell["correct"] == cell["n"]
    return {
        "passed": all(cell["passed"] for cell in cells.values()), "cells": cells,
        "required_cells": 72, "present_cells": sum(cell["n"] > 0 for cell in cells.values()),
        "passed_cells": sum(cell["passed"] for cell in cells.values()),
        "whole_answers_correct": sum(row["generation"]["correct"] for row in records), "rows": len(rows),
    }


def select_checkpoint(candidates):
    return next((step for step in (32, 128, 384) if candidates[str(step)]["coverage"]["passed"]), None)


def verify_archive(design, previous, teacher_design, choice, corpus, audit):
    proof, records = verify_teacher_receipt(previous, "whole_answer_trained", teacher_design, choice, corpus, audit)
    root = Path(previous["teacher_sources"]["whole_answer_trained"]["training_dir"])
    if file_hash(root / "training.json") != design["teacher_training_sha256"] or proof["qualification_sha256"] != design["original32_qualification_sha256"]:
        raise ValueError("COVERAGE_WRONG_ARCHIVED_TEACHER_TRAINING")
    training = json.loads((root / "training.json").read_text())
    for step in design["coverage"]["candidates"]:
        verify_checkpoint_files(training["checkpoints"][str(step)])
    if training["selected_checkpoint"] != 32 or proof["history"]["selected_checkpoint"] != 32:
        raise ValueError("COVERAGE_ORIGINAL32_SELECTION_CHANGED")
    return {"training": training, "original32_proof": proof, "original32_train_records": records}


def load_teacher(choice, checkpoint, output):
    verify_checkpoint_files(checkpoint)
    base, tokenizer = load_base(choice, output)
    model = PeftModel.from_pretrained(base, checkpoint["path"], adapter_name="teacher", is_trainable=False, local_files_only=True).requires_grad_(False).eval()
    if tensor_hash(adapter_state(model, "teacher")) != checkpoint["tensor_sha256"]:
        raise ValueError("COVERAGE_TEACHER_RELOAD_IDENTITY")
    return model, tokenizer


def screen(design, teacher_design, choice, corpus, audit, archive, output):
    proof = archive["original32_proof"]
    candidates = {}
    records = archive["original32_train_records"]
    gate = coverage_gate(corpus["train"], records, teacher_design, proof["eos_token_id"], proof["special_token_ids"])
    write_json(output / "train32.json", records)
    candidates["32"] = {
        "coverage": gate, "records_sha256": file_hash(output / "train32.json"),
        "checkpoint": archive["training"]["checkpoints"]["32"], "records_origin": "unchanged_original32_qualification_train_panel",
    }
    model, tokenizer = load_teacher(choice, archive["training"]["checkpoints"]["128"], output)
    if tokenizer.eos_token_id != proof["eos_token_id"] or tokenizer.all_special_ids != proof["special_token_ids"]:
        raise ValueError("COVERAGE_TOKENIZER_CHANGED")
    for step in (128, 384):
        checkpoint = archive["training"]["checkpoints"][str(step)]
        if step != 128:
            state = load_file(str(Path(checkpoint["path"]) / "adapter_model.safetensors"), device="cpu")
            set_peft_model_state_dict(model, state, adapter_name="teacher")
            model.set_adapter("teacher", inference_mode=True)
            model.requires_grad_(False).eval()
        if tensor_hash(adapter_state(model, "teacher")) != checkpoint["tensor_sha256"] or any(p.requires_grad for p in model.parameters()):
            raise ValueError("COVERAGE_FROZEN_CANDIDATE_IDENTITY")
        records = measure(model, tokenizer, corpus["train"], teacher_design, output, f"coverage/train{step}", detailed=False)
        gate = coverage_gate(corpus["train"], records, teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
        write_json(output / f"train{step}.json", records)
        candidates[str(step)] = {
            "coverage": gate, "records_sha256": file_hash(output / f"train{step}.json"),
            "checkpoint": checkpoint, "records_origin": "full_train_greedy_evaluation_of_existing_checkpoint",
        }
        if tensor_hash(adapter_state(model, "teacher")) != checkpoint["tensor_sha256"]:
            raise ValueError("COVERAGE_EVALUATION_MUTATED_TEACHER")
    check_base(model, choice["source_base_tensor_sha256"])
    selected = select_checkpoint(candidates)
    receipt = {
        "status": "selected" if selected is not None else "rejected", "pid": os.getpid(),
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "teacher_training_sha256": design["teacher_training_sha256"], "original32_proof": proof,
        "candidates": candidates, "selected_checkpoint": selected,
        "eos_token_id": tokenizer.eos_token_id, "special_token_ids": tokenizer.all_special_ids,
        "teacher_optimizer_updates": 0, "learner_updates": 0, "new_validation_predictions": 0, "test_predictions": 0,
    }
    write_json(output / "coverage.json", receipt)
    return receipt


def verify_screen(folder, design, teacher_design, corpus, audit, archive):
    folder = Path(folder)
    receipt = json.loads((folder / "coverage.json").read_text())
    if (
        receipt["design_sha256"] != digest(design) or receipt["source_sha256"] != source_hashes()
        or receipt["dataset"] != audit or receipt["teacher_training_sha256"] != design["teacher_training_sha256"]
        or receipt["original32_proof"] != archive["original32_proof"]
        or receipt["eos_token_id"] != archive["original32_proof"]["eos_token_id"]
        or receipt["special_token_ids"] != archive["original32_proof"]["special_token_ids"]
        or receipt["teacher_optimizer_updates"] != 0 or receipt["learner_updates"] != 0
        or receipt["new_validation_predictions"] != 0 or receipt["test_predictions"] != 0
        or set(receipt["candidates"]) != {"32", "128", "384"}
    ):
        raise ValueError("COVERAGE_SCREEN_RECEIPT_CHANGED")
    records_by_step = {}
    for step, candidate in receipt["candidates"].items():
        path = folder / f"train{step}.json"
        if file_hash(path) != candidate["records_sha256"] or candidate["checkpoint"] != archive["training"]["checkpoints"][step]:
            raise ValueError("COVERAGE_SCREEN_CANDIDATE_CHANGED")
        records = json.loads(path.read_text())
        gate = coverage_gate(corpus["train"], records, teacher_design, receipt["eos_token_id"], receipt["special_token_ids"])
        if gate != candidate["coverage"] or (step == "32" and records != archive["original32_train_records"]):
            raise ValueError("COVERAGE_RECOMPUTED_TRAIN_GATE_CHANGED")
        records_by_step[step] = records
    selected = select_checkpoint(receipt["candidates"])
    if selected != receipt["selected_checkpoint"] or receipt["status"] != ("selected" if selected is not None else "rejected"):
        raise ValueError("COVERAGE_TRAIN_ONLY_SELECTION_CHANGED")
    if selected is None:
        raise ValueError("COVERAGE_NO_FULLY_COVERED_TEACHER_NO_FURTHER_WORK")
    return receipt, records_by_step[str(selected)]


def qualify(design, teacher_design, choice, corpus, audit, archive, screen_dir, output):
    selected, train_records = verify_screen(screen_dir, design, teacher_design, corpus, audit, archive)
    if selected["pid"] == os.getpid():
        raise ValueError("COVERAGE_QUALIFICATION_REQUIRES_FRESH_PROCESS")
    checkpoint = selected["candidates"][str(selected["selected_checkpoint"])]["checkpoint"]
    model, tokenizer = load_teacher(choice, checkpoint, output)
    if tokenizer.eos_token_id != selected["eos_token_id"] or tokenizer.all_special_ids != selected["special_token_ids"]:
        raise ValueError("COVERAGE_QUALIFICATION_TOKENIZER_CHANGED")
    panels = {"train": {"generated": train_records, "choice": evaluate_choices(model, tokenizer, corpus["train"])}}
    panels["validation"] = {
        "generated": measure(model, tokenizer, corpus["validation"], teacher_design, output, "coverage_selected/validation", detailed=False),
        "choice": evaluate_choices(model, tokenizer, corpus["validation"]),
    }
    gate = generation_gate(teacher_design, corpus, panels, tokenizer.eos_token_id, tokenizer.all_special_ids)
    validation_coverage = coverage_gate(corpus["validation"], panels["validation"]["generated"], teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
    check_base(model, choice["source_base_tensor_sha256"])
    if tensor_hash(adapter_state(model, "teacher")) != checkpoint["tensor_sha256"]:
        raise ValueError("COVERAGE_QUALIFICATION_CHANGED_TEACHER")
    write_json(output / "qualification_panels.json", panels)
    receipt = {
        "status": "qualified" if gate["passed"] else "rejected", "pid": os.getpid(), "screen_pid": selected["pid"],
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "screen_dir": str(Path(screen_dir).resolve()), "screen_sha256": file_hash(Path(screen_dir) / "coverage.json"),
        "panels_sha256": file_hash(output / "qualification_panels.json"), "selected_checkpoint": selected["selected_checkpoint"],
        "checkpoint": checkpoint, "teacher_tensor_sha256": checkpoint["tensor_sha256"],
        "eos_token_id": tokenizer.eos_token_id, "special_token_ids": tokenizer.all_special_ids,
        "gate": gate, "train_coverage": selected["candidates"][str(selected["selected_checkpoint"])]["coverage"],
        "validation_coverage_diagnostic_only": validation_coverage,
        "teacher_optimizer_updates": 0, "learner_updates": 0, "test_predictions": 0,
        "claim_boundary": design["claim_boundary"],
    }
    write_json(output / "qualification.json", receipt)
    return receipt


def verify_qualified(folder, design, previous, teacher_design, choice, corpus, audit):
    folder = Path(folder)
    archive = verify_archive(design, previous, teacher_design, choice, corpus, audit)
    receipt = json.loads((folder / "qualification.json").read_text())
    if (
        receipt["status"] != "qualified" or receipt["design_sha256"] != digest(design)
        or receipt["source_sha256"] != source_hashes() or receipt["dataset"] != audit
        or receipt["teacher_optimizer_updates"] != 0 or receipt["learner_updates"] != 0 or receipt["test_predictions"] != 0
        or receipt["pid"] == receipt["screen_pid"]
        or file_hash(Path(receipt["screen_dir"]) / "coverage.json") != receipt["screen_sha256"]
        or file_hash(folder / "qualification_panels.json") != receipt["panels_sha256"]
    ):
        raise ValueError("COVERAGE_QUALIFIED_TEACHER_REQUIRED_BEFORE_LEARNER")
    selection, train_records = verify_screen(receipt["screen_dir"], design, teacher_design, corpus, audit, archive)
    candidate = selection["candidates"][str(selection["selected_checkpoint"])]
    if (
        receipt["selected_checkpoint"] != selection["selected_checkpoint"] or receipt["checkpoint"] != candidate["checkpoint"]
        or receipt["train_coverage"] != candidate["coverage"] or receipt["screen_pid"] != selection["pid"]
        or receipt["eos_token_id"] != selection["eos_token_id"] or receipt["special_token_ids"] != selection["special_token_ids"]
        or receipt["teacher_tensor_sha256"] != candidate["checkpoint"]["tensor_sha256"]
    ):
        raise ValueError("COVERAGE_QUALIFIED_SELECTION_CHANGED")
    panels = json.loads((folder / "qualification_panels.json").read_text())
    gate = generation_gate(teacher_design, corpus, panels, receipt["eos_token_id"], receipt["special_token_ids"])
    if gate != receipt["gate"] or not gate["passed"] or panels["train"]["generated"] != train_records:
        raise ValueError("COVERAGE_QUALIFIED_GATE_OR_TRAIN_TARGETS_CHANGED")
    validation_coverage = coverage_gate(corpus["validation"], panels["validation"]["generated"], teacher_design, receipt["eos_token_id"], receipt["special_token_ids"])
    if validation_coverage != receipt["validation_coverage_diagnostic_only"]:
        raise ValueError("COVERAGE_VALIDATION_DIAGNOSTIC_CHANGED")
    verify_checkpoint_files(receipt["checkpoint"])
    return receipt, train_records


def qualify_child(output, dispatch):
    config = {"stage": "qualify", "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "screen_dir": str(output.resolve())}
    write_json(output / "qualification_config.json", config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run([
        "uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "coverage_teacher.py"),
        "--config", str((output / "qualification_config.json").resolve()), "--output-dir", str((output / "qualification").resolve()),
    ], cwd=ROOT, check=True)
    receipt = json.loads((output / "qualification/qualification.json").read_text())
    if receipt["pid"] == os.getpid() or receipt["screen_pid"] != os.getpid():
        raise ValueError("COVERAGE_CHILD_PID_IDENTITY")
    write_json(output / "result.json", {"status": receipt["status"], "selected_checkpoint": receipt["selected_checkpoint"], "qualification_sha256": file_hash(output / "qualification/qualification.json"), "teacher_optimizer_updates": 0, "learner_updates": 0})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [coverage-teacher] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"run", "screen", "qualify"}:
        raise ValueError("COVERAGE_DISPATCH_DESIGN_CHANGED")
    previous, teacher_design, choice = validate_design(design)
    corpus, audit = load_corpus(choice)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("COVERAGE_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("COVERAGE_OUTPUT_DISPATCH_CHANGED")
    sources = source_hashes()
    write_json(output / "seal.json", {"design": design, "source_sha256": sources, "dataset": audit, "dispatch": dispatch, "pid": os.getpid()})
    if args.validate_only:
        write_json(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        archive = verify_archive(design, previous, teacher_design, choice, corpus, audit)
        if dispatch["stage"] == "qualify":
            qualify(design, teacher_design, choice, corpus, audit, archive, dispatch["screen_dir"], output)
        else:
            selected = screen(design, teacher_design, choice, corpus, audit, archive, output)
            if source_hashes() != sources:
                raise ValueError("COVERAGE_SOURCE_CHANGED_DURING_SCREEN")
            if dispatch["stage"] == "run" and selected["status"] == "selected":
                qualify_child(output, dispatch)
            elif selected["status"] == "rejected":
                write_json(output / "result.json", {"status": "rejected_no_complete_train_coverage", "teacher_optimizer_updates": 0, "learner_updates": 0})
        if source_hashes() != sources:
            raise ValueError("COVERAGE_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("COVERAGE_TEACHER_FAILED")
        raise


if __name__ == "__main__":
    main()
