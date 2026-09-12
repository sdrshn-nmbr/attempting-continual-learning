import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from choice_consolidation import (
    adapter_state,
    base_tensors,
    check_base,
    event,
    load_base,
    make_learner,
    save_adapter,
    tensor_hash,
)
from choice_contract import ROOT, digest, file_hash, load_corpus, write_json
from generation_consolidation import (
    learner_protocol,
    target_sequences,
    training_probes,
    validate_cache_tokens,
    verify_checkpoint_files,
)
from generation_teacher import measure, summarize, verify_generated_rows
from onpolicy303 import (
    check_frozen,
    frozen_versions,
    generated_diagnostic,
    load_saved_learner,
    model_guard,
    optimizer_clocks,
    qualify_actual_teacher,
    row_loss,
    verify_training,
)
from onpolicy303_budget_contract import (
    ADDITIONAL,
    CONTRACT,
    FINAL,
    METHODS,
    SOURCE_ARCHIVE,
    START,
    EvaluationReadBan,
    continuation_schedule,
    model_mapping,
    require,
    require_teacher_absent,
    restore_optimizer,
    runtime_versions,
    schedule_receipt,
    source_hashes,
    token_accounting,
    validate_design,
    validate_optimizer,
    verify_child_identity,
    verify_reload,
    verify_runtime_versions,
    verify_source_files,
)
from onpolicy303_contract import (
    execution_identity,
    retention_evaluate,
    retention_rows,
    retention_summary,
)


def update(student, expert, tokenizer, rows, prompts, targets, indices, method,
           original, optimizer, parameters, step):
    require(method in METHODS and len(indices) == 4, "fixed cached four-row update")
    clocks = optimizer_clocks(optimizer, parameters)
    require(clocks["minimum"] == clocks["maximum"] == step - 1, "continuous AdamW clock before update")
    optimizer.zero_grad(set_to_none=True)
    records = []
    for microstep, index in enumerate(indices):
        response = list(targets[index])
        loss, diagnostics = row_loss(student, expert, prompts[index], response,
                                     targets[index], tokenizer, method,
                                     original["sampling"]["max_new_tokens"])
        (loss / 4).backward()
        records.append({
            "step": step, "microstep": microstep, "row_index": index,
            "id": rows[index]["id"], "task": rows[index]["task"],
            "row_sha256": digest(rows[index]), "method": method,
            "generation": generated_diagnostic(tokenizer, rows[index], prompts[index], response,
                                                 original["sampling"]["max_new_tokens"]),
            "diagnostics": diagnostics, "sampling": None,
            "canonical_teacher_response_sha256": digest(targets[index]),
            "student_optimizer_clock_before_update": step - 1,
        })
        del loss
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, original["training"]["max_grad_norm"]))
    require(math.isfinite(norm), f"nonfinite gradient at update{step}")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in parameters), f"nonfinite learner at update{step}")
    clocks = optimizer_clocks(optimizer, parameters)
    require(clocks["minimum"] == clocks["maximum"] == step, "continuous AdamW clock after update")
    entry = {"step": step, "indices": indices, "ids": [rows[i]["id"] for i in indices],
             "trajectory_sha256": [digest(record) for record in records],
             "loss_tokens": sum(record["diagnostics"]["loss_tokens"] for record in records),
             "loss": sum(record["diagnostics"]["row_loss"] for record in records) / 4,
             "gradient_norm": norm, "optimizer_clocks": clocks}
    return entry, records


def continue_updates(student, expert, tokenizer, corpus, prompts, targets, schedule,
                     method, original, optimizer, parameters, output):
    require(len(schedule) == FINAL, "fixed405 schedule")
    student.set_adapter("learner")
    student.eval()
    base_snapshot = frozen_versions(base_tensors(student))
    expert_snapshot = [] if expert is None else frozen_versions({**dict(expert.named_parameters()), **dict(expert.named_buffers())})
    ledger = []
    with (output / "trajectories.jsonl").open("x") as stream:
        for step, indices in enumerate(schedule[START:], START + 1):
            entry, records = update(student, expert, tokenizer, corpus["train"], prompts, targets,
                                    indices, method, original, optimizer, parameters, step)
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            check_frozen(base_snapshot, "student_base")
            check_frozen(expert_snapshot, "expert")
            ledger.append(entry)
            event(output, "onpolicy303_budget_update", method=method, **entry)
    require(len(ledger) == ADDITIONAL and sum(e["loss_tokens"] for e in ledger) == 588,
            "exact21 updates and588 additional loss tokens")
    optimizer.zero_grad(set_to_none=True)
    write_json(output / "ledger.json", ledger)
    return ledger


def restored_student(protocol, descriptor, output):
    base, tokenizer = load_base(protocol, output)
    model_guard(base, "budget_student_base")
    student = make_learner(base, protocol)
    spec = descriptor["checkpoint"]["adapter"]
    verify_checkpoint_files(spec)
    saved = load_file(str(Path(spec["path"]) / "adapter_model.safetensors"), device="cpu")
    set_peft_model_state_dict(student, saved, adapter_name="learner")
    require(tensor_hash(adapter_state(student)) == spec["tensor_sha256"], "checkpoint384 exact adapter restore")
    require(model_mapping(student) == descriptor["parameter_mapping"], "checkpoint384 parameter order")
    return student, tokenizer


def training(design, original, context, corpus, audit, dispatch, execution, output):
    method = dispatch["method"]
    descriptor = design["sources"][method]
    source = Path(descriptor["run_dir"])
    old_receipt = verify_source_files(source, descriptor)
    verified, cache = verify_training(source, original, context, corpus, audit, method)
    require(verified == old_receipt, "original training verification")
    source_execution = execution_identity(source, descriptor["task_id"], True)
    require(source_execution["source"]["source_sha256"] == SOURCE_ARCHIVE, "original source archive")
    for key in ("task_id", "attempt_id", "task_sha256", "config_sha256", "source", "code_dir", "entrypoint", "config"):
        require(source_execution[key] == old_receipt["execution"][key], f"source execution {key}")
    _, _, _, teacher_design, choice = context
    protocol = learner_protocol(choice, original)
    ids = json.loads((source / "schedule.json").read_text())
    schedule = continuation_schedule(protocol, corpus["train"], ids)
    schedule_proof = schedule_receipt(protocol, corpus["train"], ids)
    require(schedule_proof == design["schedule"], "sealed continuation schedule")
    accounting = token_accounting(cache, schedule, old_receipt["arm"]["loss_token_exposures"])
    write_json(output / "schedule.json", schedule_proof)
    write_json(output / "teacher_train_cache.json", cache)
    require(file_hash(output / "teacher_train_cache.json") == descriptor["files_sha256"]["teacher_train_cache.json"], "unchanged teacher cache")
    write_json(output / "parameter_mapping.json", descriptor["parameter_mapping"])
    expert, teacher_tokenizer = None, None
    proof = old_receipt["teacher_provenance"]
    if method == "cached_teacher_kl":
        expert, teacher_tokenizer, fresh_proof, fresh_cache = qualify_actual_teacher(original, context, corpus, audit, output)
        require(fresh_cache == cache and fresh_proof == proof, "fresh384TRAIN96VAL teacher/cache identity")
        event(output, "budget_teacher_qualified_before_student_load", learner_loaded=False, teacher_updates=0)
    student, tokenizer = restored_student(protocol, descriptor, output)
    if teacher_tokenizer is not None:
        require(tokenizer.get_vocab() == teacher_tokenizer.get_vocab(), "teacher/student vocabulary")
    prompts = validate_cache_tokens(tokenizer, corpus["train"], cache, student.config.vocab_size)
    targets = target_sequences("teacher_output_sft", tokenizer, corpus["train"], cache)
    require(targets == target_sequences("oracle_sft", tokenizer, corpus["train"], cache), "all384 cached native outputs remain correct")
    optimizer_path = source / method / "checkpoint384/optimizer.pt"
    require(file_hash(optimizer_path) == descriptor["checkpoint"]["optimizer_sha256"], "saved AdamW384 file")
    saved_optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    optimizer, parameters, restored = restore_optimizer(student, original["training"], saved_optimizer,
                                                         descriptor["parameter_mapping"], descriptor["optimizer_options"],
                                                         descriptor["optimizer_tensor_sha256"])
    del saved_optimizer
    write_json(output / "optimizer_restore.json", restored)
    probes = training_probes(corpus, original["training"]["probe_rows_per_task"])
    retention = retention_rows(original, corpus)
    native = measure(student, tokenizer, probes, teacher_design, output, "restored384/train_only", detailed=False)
    generic = retention_evaluate(student, tokenizer, retention)
    write_json(output / "restored384_train_probes.json", native)
    write_json(output / "restored384_retention.json", generic)
    parity = verify_reload(native, json.loads((source / method / "checkpoint384/train_probes.json").read_text()),
                           generic, json.loads((source / "final_retention.json").read_text()), retention)
    write_json(output / "prerequisites.json", {"passed": True, "updates_this_run": 0,
               "optimizer_clock": START, "teacher_fresh_before_student_load": expert is not None,
               "source_execution": source_execution, "schedule": schedule_proof, **parity})
    event(output, "budget_prerequisites_passed", updates_this_run=0, optimizer_clock=START)
    ledger = continue_updates(student, expert, tokenizer, corpus, prompts, targets, schedule,
                               method, original, optimizer, parameters, output)
    destination = output / "checkpoint405"
    final = save_adapter(student, destination)
    require(final["tensor_sha256"] != descriptor["checkpoint"]["adapter"]["tensor_sha256"], "unchanged405 adapter")
    optimizer_proof = validate_optimizer(optimizer.state_dict(), descriptor["parameter_mapping"], FINAL, descriptor["optimizer_options"])
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    persisted = torch.load(destination / "optimizer.pt", map_location="cpu", weights_only=True)
    require(validate_optimizer(persisted, descriptor["parameter_mapping"], FINAL, descriptor["optimizer_options"]) == optimizer_proof,
            "persisted AdamW405 moments differ")
    del persisted
    native405 = measure(student, tokenizer, probes, teacher_design, output, "final405/train_only", detailed=False)
    generic405 = retention_evaluate(student, tokenizer, retention)
    write_json(destination / "train_probes.json", native405)
    write_json(destination / "retention.json", generic405)
    check_base(student, choice["source_base_tensor_sha256"])
    if expert is not None:
        check_base(expert, choice["source_base_tensor_sha256"])
        require(tensor_hash(adapter_state(expert, "teacher")) == proof["teacher_tensor_sha256"], "frozen FP32 expert unchanged")
        verify_checkpoint_files(design["teacher_checkpoint"])
    verify_source_files(source, descriptor)
    files = ["runtime_versions.json", "schedule.json", "teacher_train_cache.json", "parameter_mapping.json", "optimizer_restore.json",
             "restored384_train_probes.json", "restored384_retention.json", "prerequisites.json",
             "ledger.json", "trajectories.jsonl", "checkpoint405/optimizer.pt",
             "checkpoint405/train_probes.json", "checkpoint405/retention.json"]
    if expert is not None:
        files += ["teacher_fresh_qualification.json", "teacher_fresh_train.json", "teacher_fresh_validation.json",
                  "teacher_qualification.json", "teacher_qualification_panels.json"]
    receipt = {
        "contract": CONTRACT, "status": "trained_evaluation_pending", "method": method,
        "pid": os.getpid(), "parent_pid": os.getppid(), "execution": execution,
        "design_sha256": digest(design), "source_sha256": source_hashes(design), "dataset": audit,
        "runtime_versions": runtime_versions(),
        "original_source": descriptor, "original_execution": source_execution,
        "tokens": accounting, "updates_this_run": len(ledger), "selected_checkpoint": FINAL,
        "optimizer_restored": restored, "optimizer_final": optimizer_proof,
        "checkpoint405": final, "base_tensor_sha256": choice["source_base_tensor_sha256"],
        "teacher_tensor_sha256": proof["teacher_tensor_sha256"], "teacher_optimizer_updates": 0,
        "teacher_loaded_in_training": expert is not None, "teacher_fresh_qualified_before_student_load": expert is not None,
        "native384_reload_parity": parity, "native405_train_summary": summarize(native405),
        "generic405_summary": retention_summary(generic405),
        "trainable_parameters": sum(p.numel() for p in parameters), "resident_learner_rank": 8,
        "model_guard": model_guard(student, "budget_final405"),
        "heldout_predictions_observed_during_this_training": False,
        "post_observation_control": True, "heldouts_previously_observed": True,
        "native_export": design["native_export"], "claim_boundary": design["claim_boundary"],
        "files_sha256": {name: file_hash(output / name) for name in files},
    }
    write_json(output / "training.json", receipt)
    return receipt


def verify_continuation(output, receipt, design, audit, evaluation=False):
    require(receipt["contract"] == CONTRACT and receipt["design_sha256"] == digest(design), "continuation receipt design")
    require(receipt["source_sha256"] == source_hashes(design) and receipt["dataset"] == audit, "continuation source/dataset")
    require(receipt["runtime_versions"] == design["runtime_versions"], "continuation exact runtime fields")
    require(receipt["method"] in METHODS and receipt["status"] == "trained_evaluation_pending", "continuation method/status")
    require(receipt["updates_this_run"] == ADDITIONAL and receipt["selected_checkpoint"] == FINAL,
            "fixed final405 receipt")
    require(receipt["tokens"]["total_loss_tokens"] == 11340 and receipt["tokens"]["additional_loss_tokens"] == 588, "final loss tokens")
    require(receipt["optimizer_restored"]["clock"] == START and receipt["optimizer_final"]["clock"] == FINAL, "restored/final Adam clocks")
    require(receipt["original_source"] == design["sources"][receipt["method"]], "original source binding")
    require(receipt["teacher_optimizer_updates"] == 0 and receipt["teacher_tensor_sha256"] == design["teacher"]["tensor_sha256"], "fixed teacher receipt")
    require(receipt["teacher_fresh_qualified_before_student_load"] == (receipt["method"] == "cached_teacher_kl"), "teacher prerequisite receipt")
    require(receipt["native384_reload_parity"] == {"train24_exact_native_parity": True, "generic128_exact_score_parity": True}, "restored384 prerequisite receipt")
    evaluation_files = {"checkpoint405/train_probes.json", "checkpoint405/retention.json", "parameter_mapping.json"}
    for name, checksum in receipt["files_sha256"].items():
        if evaluation and name not in evaluation_files:
            continue
        require(file_hash(output / name) == checksum, f"continuation artifact {name}")
    require(Path(receipt["checkpoint405"]["path"]).resolve() == output / "checkpoint405/learner", "own405 path")
    verify_checkpoint_files(receipt["checkpoint405"])


def evaluate_child(design, original, context, corpus, audit, dispatch, output):
    versions = verify_runtime_versions(design["runtime_versions"], runtime_versions())
    require(not torch.cuda.is_initialized(), "fresh evaluator has no inherited CUDA model")
    training_path = output / "training.json"
    receipt = json.loads(training_path.read_text())
    launch = json.loads((output / "evaluation_launch.json").read_text())
    require(launch["training_sha256"] == file_hash(training_path) and launch["training_pid"] == receipt["pid"], "child training binding")
    require(os.getpid() != receipt["pid"] and launch["method"] == dispatch["method"] == receipt["method"], "child process/method")
    child = output / "evaluation"
    child.mkdir()
    _, _, _, teacher_design, choice = context
    base_path = Path(choice["source_base"]["local_path"])
    checkpoint = receipt["checkpoint405"]
    forbidden = design["evaluation_forbidden_paths"] + [str(output / name) for name in
                    ("teacher_train_cache.json", "checkpoint405/optimizer.pt", "trajectories.jsonl", "ledger.json")]
    forbidden += [str(output / name) for name in receipt["files_sha256"] if name.startswith("teacher_")]
    pinned_base_paths = [base_path / spec["path"] for spec in choice["source_base"]["files"]]
    ban = EvaluationReadBan(forbidden, [base_path, checkpoint["path"], *pinned_base_paths]).install()
    verify_continuation(output, receipt, design, audit, evaluation=True)
    for spec in choice["source_base"]["files"]:
        ban.check(base_path / spec["path"])
    for name in checkpoint["files"]:
        ban.check(Path(checkpoint["path"]) / name)
    base, tokenizer = load_base(learner_protocol(choice, original), child)
    guard = model_guard(base, "budget_teacher_absent_evaluator")
    model = load_saved_learner(base, checkpoint)
    absent = require_teacher_absent(model)
    probes = training_probes(corpus, original["training"]["probe_rows_per_task"])
    retention = retention_rows(original, corpus)
    native = measure(model, tokenizer, probes, teacher_design, child, "reload405/train_only", detailed=False)
    generic = retention_evaluate(model, tokenizer, retention)
    verify_reload(native, json.loads((output / "checkpoint405/train_probes.json").read_text()),
                  generic, json.loads((output / "checkpoint405/retention.json").read_text()), retention)
    write_json(child / "train_probes.json", native)
    write_json(child / "retention.json", generic)
    panels = {}
    for split in original["evaluation"]["splits"]:
        panels[split] = measure(model, tokenizer, corpus[split], teacher_design, child, f"final405/{split}", detailed=False)
        verify_generated_rows(corpus[split], panels[split], teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
    require({key: len(value) for key, value in panels.items()} == {"test": 192, "unused288": 864}, "unchanged heldout denominators")
    check_base(model, choice["source_base_tensor_sha256"])
    require(tensor_hash(adapter_state(model)) == checkpoint["tensor_sha256"], "evaluation405 adapter unchanged")
    require(not ban.denied, "unexpected attempted forbidden evaluator read")
    write_json(child / "panels.json", panels)
    result = {
        "status": "completed", "method": receipt["method"], "pid": os.getpid(), "parent_pid": os.getppid(),
        "training_pid": receipt["pid"], "training_sha256": file_hash(training_path), "child_token": launch["child_token"],
        "design_sha256": digest(design), "source_sha256": source_hashes(design), "dataset": audit,
        "runtime_versions": versions,
        "checkpoint405": checkpoint, "model_guard": guard, **absent,
        "train24_reload_parity": True, "generic128_reload_parity": True,
        "native": {split: summarize(records) for split, records in panels.items()},
        "generic_retention": retention_summary(generic), "read_ban": ban.receipt(),
        "files_sha256": {name: file_hash(child / name) for name in ("train_probes.json", "retention.json", "panels.json", "runtime.json")},
        "post_observation_control": True, "heldouts_previously_observed": True,
        "native_export": design["native_export"], "claim_boundary": design["claim_boundary"],
    }
    write_json(child / "result.json", result)
    return result


def launch_evaluation(config, output, receipt):
    training_hash = file_hash(output / "training.json")
    token = uuid.uuid4().hex
    command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-B",
               str(ROOT / "onpolicy303_budget.py"), "--config", str(config.resolve()),
               "--output-dir", str(output), "--evaluate-child"]
    write_json(output / "evaluation_launch.json", {"training_pid": os.getpid(), "training_sha256": training_hash,
               "child_token": token, "method": receipt["method"], "command": command,
               "started_unix_ns": time.time_ns(), "process_chain": "training Python -> uv launcher -> fresh evaluation Python"})
    with (output / "evaluation.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        code = process.wait()
    write_json(output / "evaluation_process.json", {"uv_launcher_pid": process.pid, "returncode": code,
               "training_pid": os.getpid(), "completed_unix_ns": time.time_ns()})
    require(code == 0, f"fresh evaluator exit{code}; see evaluation.log")
    result = json.loads((output / "evaluation/result.json").read_text())
    verify_child_identity(result, os.getpid(), process.pid, training_hash, token)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-child", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [onpolicy303_budget] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    require(digest(design) == dispatch["protocol_sha256"] and dispatch["method"] in METHODS
            and dispatch["stage"] == "continue_and_evaluate", "dispatch contract")
    original, context = validate_design(design)
    corpus, audit = load_corpus(context[-1])
    retention_rows(original, corpus)
    root = args.output_dir.resolve()
    if args.evaluate_child:
        require(not args.validate_only, "child cannot validate only")
        evaluate_child(design, original, context, corpus, audit, dispatch, root)
        return
    root.mkdir(parents=True, exist_ok=True)
    output = root / "study"
    output.mkdir()
    write_json(output / "seal.json", {"design": design, "dispatch": dispatch, "dataset": audit,
               "source_sha256": source_hashes(design), "pid": os.getpid(), "sealed_unix_ns": time.time_ns()})
    if args.validate_only:
        write_json(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False,
                   "updates": 0, "tokens": design["tokens"]})
        return
    try:
        versions = verify_runtime_versions(design["runtime_versions"], runtime_versions())
        write_json(output / "runtime_versions.json", versions)
        execution = execution_identity(root, dispatch["task_id"], False, dispatch)
        require(Path(execution["code_dir"]).resolve() == ROOT and execution["entrypoint"] == "onpolicy303_budget.py", "executed source directory")
        dependency = original["runtime_dependency"]
        gate = execution_identity(dependency["directory"], dependency["task_id"], True)
        write_json(output / "runtime_dependency.json", {"execution": gate, "numerical_boundary": dependency["numerical_boundary"]})
        receipt = training(design, original, context, corpus, audit, dispatch, execution, output)
        verify_continuation(output, receipt, design, audit)
        gc.collect()
        torch.cuda.empty_cache()
        require(torch.cuda.memory_allocated() == 0, "release all training models before fresh evaluator")
        result = launch_evaluation(args.config, output, receipt)
        require(source_hashes(design) == design["scientific_files_sha256"], "source changed during execution")
        write_json(output / "result.json", {"status": "completed", "method": dispatch["method"],
                   "tokens": receipt["tokens"], "training_sha256": file_hash(output / "training.json"),
                   "evaluation_sha256": file_hash(output / "evaluation/result.json"),
                   "native": result["native"], "generic_retention": result["generic_retention"],
                   "teacher_absent_fresh_process_completed": True, "native_export": design["native_export"],
                   "claim_boundary": design["claim_boundary"]})
        event(output, "onpolicy303_budget_completed", method=dispatch["method"], native=result["native"], generic=result["generic_retention"])
    except Exception as error:
        event(output, "onpolicy303_budget_failed", exception=type(error).__name__, detail=str(error))
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
