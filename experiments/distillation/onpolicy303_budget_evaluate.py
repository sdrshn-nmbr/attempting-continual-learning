import argparse
import copy
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from choice_consolidation import event, tensor_hash
from choice_contract import (
    ROOT,
    digest,
    file_hash,
    load_corpus,
    training_schedule,
    write_json,
)
from generation_consolidation import learner_protocol
from onpolicy303_budget import evaluate_child, verify_continuation
from onpolicy303_budget_contract import (
    METHODS,
    EvaluationReadBan,
    continuation_schedule,
    mapping_from_adapter,
    require,
    runtime_versions,
    schedule_receipt,
    token_accounting,
    validate_optimizer,
    verify_child_identity,
    verify_runtime_versions,
)
from onpolicy303_budget_contract import (
    validate_design as validate_budget_design,
)
from onpolicy303_contract import archive_identity, execution_identity, supervisor_digest

CONTRACT = "onpolicy303_budget405_zero_update_evaluation_recovery"
FAILED_SOURCE = "d11de8b7ba587cb7c99315232785ac2a20f8353130ab2db15bba3fa0e1e1d9f4"
CLEANUP_FAILURE = "ONPOLICY303_BUDGET release all training models before fresh evaluator"
EVALUATION_FILES = (
    "parameter_mapping.json",
    "checkpoint405/train_probes.json",
    "checkpoint405/retention.json",
    "checkpoint405/learner/adapter_config.json",
    "checkpoint405/learner/adapter_model.safetensors",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_design(design):
    require(design["contract"] == CONTRACT, "EVAL_ONLY contract")
    for name, checksum in design["files_sha256"].items():
        require(file_hash(ROOT / name) == checksum, f"EVAL_ONLY source changed {name}")
    old = design["budget_protocol"]
    require(file_hash(ROOT / old["path"]) == old["file_sha256"], "EVAL_ONLY frozen budget protocol")
    budget = read_json(ROOT / old["path"])
    require(digest(budget) == old["payload_sha256"], "EVAL_ONLY budget payload")
    require(design["new_updates"] == 0 and design["native_export"] == budget["native_export"], "EVAL_ONLY zero updates and fixed export")
    require(set(design["sources"]) == set(METHODS), "EVAL_ONLY source arms")
    original, context = validate_budget_design(budget)
    return budget, original, context


def verify_pins(root, spec):
    for name, checksum in spec["files_sha256"].items():
        require(file_hash(root / name) == checksum, f"EVAL_ONLY original file changed {name}")


def terminal_identity(execution, task, config, receipt, failure, spec):
    require(execution["status"] == "failed" and execution["exit_code"] == 1 and execution["timed_out"] is False,
            "EVAL_ONLY original process must have terminated with cleanup failure")
    require(execution["task_id"] == task["id"] == spec["task_id"] and execution["task"] == task,
            "EVAL_ONLY original execution/task identity")
    require(task["config"] == config and execution["config_sha256"] == supervisor_digest(config)
            and execution["task_sha256"] == supervisor_digest(task), "EVAL_ONLY original config digest")
    require(task["source_sha256"] == execution["source_sha256"] == FAILED_SOURCE
            and task["entrypoint"] == "onpolicy303_budget.py", "EVAL_ONLY original frozen archive")
    require(config["method"] == spec["method"] == receipt["method"] and config["stage"] == "continue_and_evaluate",
            "EVAL_ONLY original method")
    require(failure == {"exception": "ValueError", "detail": CLEANUP_FAILURE, "pid": receipt["pid"]},
            "EVAL_ONLY exact cleanup failure; cause is unknown residual GPU allocations")
    started = datetime.fromisoformat(execution["started_at"])
    ended = datetime.fromisoformat(execution["finished_at"])
    require(started < ended <= datetime.now(timezone.utc), "EVAL_ONLY terminal timestamp")
    previous = receipt["execution"]
    require(previous["task_id"] == task["id"] and previous["attempt_id"] == execution["attempt_id"]
            and previous["config"] == config and previous["config_sha256"] == execution["config_sha256"]
            and previous["task_sha256"] == execution["task_sha256"]
            and previous["source"]["source_sha256"] == FAILED_SOURCE
            and previous["code_dir"] == task["code_dir"], "EVAL_ONLY trained receipt belongs to ended attempt")
    require(receipt["pid"] != os.getpid(), "EVAL_ONLY independent process from original training")
    return {"task_id": task["id"], "attempt_id": execution["attempt_id"], "training_pid": receipt["pid"],
            "supervisor_child_pid": execution["pid"], "finished_at": execution["finished_at"],
            "status": "failed_after_complete_training", "cleanup_failure": failure["detail"],
            "cleanup_cause": "unknown residual GPU allocations; no claim about teacher residency",
            "source_sha256": FAILED_SOURCE}


def verify_materials(root, spec, budget, original, context, corpus, audit):
    root = Path(root)
    verify_pins(root, spec)
    study = root / "study"
    receipt = read_json(study / "training.json")
    terminal = terminal_identity(read_json(root / "execution.json"), read_json(root / "task.json"),
                                 read_json(root / "config.json"), receipt, read_json(study / "failure.json"), spec)
    require(not (study / "evaluation_launch.json").exists() and not (study / "evaluation/result.json").exists()
            and not (study / "result.json").exists(), "EVAL_ONLY never repeat an existing final evaluation")
    require(receipt["design_sha256"] == digest(budget) and receipt["source_sha256"] == budget["scientific_files_sha256"]
            and receipt["dataset"] == audit, "EVAL_ONLY trained design/source/data")
    require(receipt["status"] == "trained_evaluation_pending" and receipt["updates_this_run"] == 21
            and receipt["selected_checkpoint"] == 405 and receipt["teacher_optimizer_updates"] == 0,
            "EVAL_ONLY exactly complete21 training updates")
    require(receipt["original_source"] == budget["sources"][spec["method"]]
            and receipt["checkpoint405"] == spec["checkpoint405"], "EVAL_ONLY checkpoint/original384 identity")
    canonical = Path(spec["run_dir"]) / "study/checkpoint405/learner"
    require(Path(receipt["checkpoint405"]["path"]) == canonical, "EVAL_ONLY original405 canonical path")
    require(receipt["runtime_versions"] == budget["runtime_versions"]
            and receipt["base_tensor_sha256"] == context[-1]["source_base_tensor_sha256"]
            and receipt["teacher_tensor_sha256"] == budget["teacher"]["tensor_sha256"], "EVAL_ONLY frozen runtime/base/teacher")
    prerequisite = read_json(study / "prerequisites.json")
    require(prerequisite["passed"] and prerequisite["updates_this_run"] == 0 and prerequisite["optimizer_clock"] == 384
            and prerequisite["train24_exact_native_parity"] and prerequisite["generic128_exact_score_parity"],
            "EVAL_ONLY original384 pre-update parity gate")
    if spec["method"] == "cached_teacher_kl":
        qualification = read_json(study / "teacher_fresh_qualification.json")
        require(qualification["learner_loaded"] is False and qualification["learner_updates"] == 0
                and qualification["teacher_optimizer_updates"] == 0, "EVAL_ONLY teacher qualification before student")
        for split, count in (("train", 384), ("validation", 96)):
            require(qualification["gates"][split] == {"correct": count, "count": count, "coverage_passed": True,
                    "qualified_native_tokens_equal": True}, "EVAL_ONLY exact fresh teacher qualification")
    protocol = learner_protocol(context[-1], original)
    rows = corpus["train"]
    original_ids = [[rows[i]["id"] for i in batch] for batch in training_schedule(protocol, rows)]
    schedule = continuation_schedule(protocol, rows, original_ids)
    require(schedule_receipt(protocol, rows, original_ids) == read_json(study / "schedule.json") == budget["schedule"],
            "EVAL_ONLY sealed405 schedule and384 prefix")
    cache = read_json(study / "teacher_train_cache.json")
    require(file_hash(study / "teacher_train_cache.json") == budget["sources"][spec["method"]]["files_sha256"]["teacher_train_cache.json"],
            "EVAL_ONLY original correct TRAIN cache")
    accounting = token_accounting(cache, schedule, 10752)
    require(receipt["tokens"] == accounting, "EVAL_ONLY exact11340 total tokens")
    ledger = read_json(study / "ledger.json")
    records = [json.loads(line) for line in (study / "trajectories.jsonl").read_text().splitlines()]
    require(len(ledger) == 21 and len(records) == 84, "EVAL_ONLY 21ledger/84row denominator")
    for offset, (entry, indices) in enumerate(zip(ledger, schedule[384:], strict=True)):
        step = 385 + offset
        block = records[offset * 4:(offset + 1) * 4]
        require(entry["step"] == step and entry["indices"] == indices and entry["ids"] == [rows[i]["id"] for i in indices]
                and entry["trajectory_sha256"] == [digest(record) for record in block]
                and entry["loss_tokens"] == 28 and math.isfinite(entry["loss"]) and math.isfinite(entry["gradient_norm"])
                and entry["optimizer_clocks"] == {"minimum": step, "maximum": step, "parameters_with_state": 144},
                f"EVAL_ONLY actual ledger/clock at{step}")
        for microstep, (record, index) in enumerate(zip(block, indices, strict=True)):
            prompt, response = cache["rows"][index]["prompt_token_ids"], cache["rows"][index]["response_token_ids"]
            diagnostics = record["diagnostics"]
            require(record["step"] == step and record["microstep"] == microstep and record["row_index"] == index
                    and record["id"] == rows[index]["id"] and record["row_sha256"] == digest(rows[index])
                    and record["method"] == spec["method"] and record["sampling"] is None
                    and record["student_optimizer_clock_before_update"] == step - 1
                    and record["canonical_teacher_response_sha256"] == digest(response)
                    and record["generation"]["prompt_token_ids"] == prompt and record["generation"]["token_ids"] == response
                    and diagnostics["loss_token_ids"] == response and diagnostics["loss_tokens"] == 7
                    and diagnostics["prediction_mask"] == [False] * (len(prompt) - 1) + [True] * 7,
                    f"EVAL_ONLY actual TRAIN row/mask at{step}/{microstep}")
    require(sum(entry["loss_tokens"] for entry in ledger) == 588, "EVAL_ONLY actual588 continuation tokens")
    mapping = read_json(study / "parameter_mapping.json")
    original_source = budget["sources"][spec["method"]]
    require(mapping == original_source["parameter_mapping"], "EVAL_ONLY exact original optimizer parameter order")
    restored = read_json(study / "optimizer_restore.json")
    require(restored == receipt["optimizer_restored"] and restored["clock"] == 384 and restored["parameter_states"] == 144
            and restored["tensor_sha256"] == original_source["optimizer_tensor_sha256"]
            and restored["parameter_order_sha256"] == digest(mapping), "EVAL_ONLY original Adam384 restored")
    final = load_file(str(study / "checkpoint405/learner/adapter_model.safetensors"), device="cpu")
    require(tensor_hash(final) == spec["checkpoint405"]["tensor_sha256"] and mapping_from_adapter(final) == mapping,
            "EVAL_ONLY actual405 tensor identity before model load")
    optimizer = torch.load(study / "checkpoint405/optimizer.pt", map_location="cpu", weights_only=True)
    proof = validate_optimizer(optimizer, mapping, 405, original_source["optimizer_options"])
    require(proof == receipt["optimizer_final"], "EVAL_ONLY saved Adam405 complete moments")
    require(not torch.cuda.is_initialized(), "EVAL_ONLY preflight must not initialize CUDA")
    return receipt, {"terminal": terminal, "ledger_updates": 21, "loss_tokens": 588, "total_tokens": 11340,
                     "optimizer_initial_clock": 384, "optimizer_final": proof, "checkpoint405": spec["checkpoint405"],
                     "cuda_initialized": False, "new_optimizer_updates": 0}


def copy_bound(source, destination, checksum):
    require(not source.is_symlink(), "EVAL_ONLY input symlink")
    require(file_hash(source) == checksum, f"EVAL_ONLY input changed before copy {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    require(file_hash(destination) == checksum, f"EVAL_ONLY corrupt copy {destination.name}")


def project_receipt(original, output, origin_hash):
    projected = copy.deepcopy(original)
    require("evaluation_only_origin_sha256" not in projected, "EVAL_ONLY source must be original receipt")
    projected["checkpoint405"]["path"] = str(output / "checkpoint405/learner")
    projected["evaluation_only_origin_sha256"] = origin_hash
    return projected


def prepare_copy(source, output, spec, receipt, gate):
    origin_hash = spec["files_sha256"]["study/training.json"]
    paths = {"original_training.json": "study/training.json", **{name: "study/" + name for name in EVALUATION_FILES}}
    for destination, original_name in paths.items():
        copy_bound(source / original_name, output / destination, spec["files_sha256"][original_name])
    projected = project_receipt(receipt, output, origin_hash)
    write_json(output / "training.json", projected)
    copy_receipt = {"kind": "zero_update_evaluator_input_copy_not_a_new_training_receipt", "source_run": spec["run_dir"],
                    "origin_training_sha256": origin_hash, "projected_training_sha256": file_hash(output / "training.json"),
                    "exact_allowed_json_changes": ["/checkpoint405/path", "/evaluation_only_origin_sha256"],
                    "copies_sha256": {destination: spec["files_sha256"][original_name] for destination, original_name in paths.items()},
                    "original_optimizer_copied": False, "source_gate": gate, "new_optimizer_updates": 0}
    write_json(output / "copy.json", copy_receipt)
    return projected, copy_receipt


def verify_copy(output, spec, budget, audit):
    copied = read_json(output / "copy.json")
    require(copied["source_run"] == spec["run_dir"] and copied["new_optimizer_updates"] == 0, "EVAL_ONLY copy source")
    origin_hash = spec["files_sha256"]["study/training.json"]
    require(copied["origin_training_sha256"] == origin_hash, "EVAL_ONLY copied origin identity")
    expected = {"original_training.json": origin_hash, **{name: spec["files_sha256"]["study/" + name] for name in EVALUATION_FILES}}
    require(copied["copies_sha256"] == expected, "EVAL_ONLY complete exact copy set")
    for name, checksum in expected.items():
        require(file_hash(output / name) == checksum, f"EVAL_ONLY copied bytes changed {name}")
    origin, projected = read_json(output / "original_training.json"), read_json(output / "training.json")
    require(projected == project_receipt(origin, output, origin_hash), "EVAL_ONLY derived receipt exact path-plus-origin diff")
    require(file_hash(output / "training.json") == copied["projected_training_sha256"], "EVAL_ONLY derived receipt hash")
    verify_continuation(output, projected, budget, audit, evaluation=True)
    return projected, copied


def run_child(design, budget, original, context, corpus, audit, dispatch, output):
    source_paths = [spec["run_dir"] for spec in design["sources"].values()]
    base = Path(context[-1]["source_base"]["local_path"])
    pinned = [base / spec["path"] for spec in context[-1]["source_base"]["files"]]
    ban = EvaluationReadBan(source_paths, [base, *pinned, output / "checkpoint405/learner"]).install()
    spec = design["sources"][dispatch["method"]]
    projected, copied = verify_copy(output, spec, budget, audit)
    launch = read_json(output / "evaluation_launch.json")
    require(launch["copy_sha256"] == file_hash(output / "copy.json") and launch["origin_training_sha256"] == copied["origin_training_sha256"],
            "EVAL_ONLY child copied-origin binding")
    require(os.getpid() not in {projected["pid"], launch["evaluation_only_parent_pid"]}, "EVAL_ONLY independent fresh child")
    result = evaluate_child(budget, original, context, corpus, audit, dispatch, output)
    require(not ban.denied, "EVAL_ONLY child attempted original source read")
    write_json(output / "evaluation/origin_read_ban.json", ban.receipt())
    return result


def invoke_fresh(command, log_path):
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        code = process.wait()
    require(code == 0, f"EVAL_ONLY fresh child exit{code}; see {log_path.name}")
    return process.pid


def launch_child(config, output, projected):
    token = uuid.uuid4().hex
    training_hash = file_hash(output / "training.json")
    write_json(output / "evaluation_launch.json", {"training_pid": projected["pid"], "training_sha256": training_hash,
               "child_token": token, "method": projected["method"], "evaluation_only_parent_pid": os.getpid(),
               "origin_training_sha256": projected["evaluation_only_origin_sha256"],
               "copy_sha256": file_hash(output / "copy.json"), "new_optimizer_updates": 0})
    command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(ROOT / "onpolicy303_budget_evaluate.py"),
               "--config", str(config.resolve()), "--output-dir", str(output), "--evaluate-child"]
    launcher = invoke_fresh(command, output / "evaluation.log")
    result = read_json(output / "evaluation/result.json")
    verify_child_identity(result, projected["pid"], launcher, training_hash, token)
    require(result["pid"] != os.getpid() and launcher not in {os.getpid(), projected["pid"]}, "EVAL_ONLY launcher/new parent independent")
    write_json(output / "evaluation_process.json", {"evaluation_only_parent_pid": os.getpid(), "uv_launcher_pid": launcher,
               "evaluation_pid": result["pid"], "original_training_pid": projected["pid"], "exit_code": 0,
               "fresh_process": True, "new_optimizer_updates": 0})
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-child", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [onpolicy303_budget_evaluate] %(message)s")
    dispatch = read_json(args.config)
    design = read_json(ROOT / dispatch["protocol"])
    require(dispatch["protocol_sha256"] == digest(design) and dispatch["stage"] == "evaluate_only"
            and dispatch["method"] in METHODS, "EVAL_ONLY dispatch")
    budget, original, context = validate_design(design)
    corpus, audit = load_corpus(context[-1])
    root = args.output_dir.resolve()
    if args.evaluate_child:
        require(not args.validate_only, "EVAL_ONLY child cannot validate only")
        run_child(design, budget, original, context, corpus, audit, dispatch, root)
        return
    root.mkdir(parents=True, exist_ok=True)
    output = root / "study"
    output.mkdir()
    write_json(output / "seal.json", {"design": design, "dispatch": dispatch, "pid": os.getpid(),
               "sealed_unix_ns": time.time_ns(), "new_optimizer_updates": 0})
    if args.validate_only:
        write_json(output / "validation.json", {"status": "cpu_contract_only", "new_optimizer_updates": 0})
        return
    try:
        verify_runtime_versions(budget["runtime_versions"], runtime_versions())
        execution = execution_identity(root, dispatch["task_id"], False, dispatch)
        require(execution["entrypoint"] == "onpolicy303_budget_evaluate.py" and Path(execution["code_dir"]).resolve() == ROOT,
                "EVAL_ONLY new supervisor source")
        spec = design["sources"][dispatch["method"]]
        source = Path(spec["run_dir"])
        receipt, gate = verify_materials(source, spec, budget, original, context, corpus, audit)
        require(execution["task_id"] != spec["task_id"] and execution["attempt_id"] != gate["terminal"]["attempt_id"],
                "EVAL_ONLY distinct supervisor attempt")
        require(datetime.fromisoformat(gate["terminal"]["finished_at"]) < datetime.fromisoformat(read_json(root / "execution.json")["started_at"]),
                "EVAL_ONLY original process ended before new task began")
        previous = read_json(source / "task.json")
        gate["archive"] = archive_identity(previous["code_dir"], FAILED_SOURCE)
        require(gate["archive"] == receipt["execution"]["source"], "EVAL_ONLY original archive byte closure")
        write_json(output / "source_gate.json", gate)
        projected, copied = prepare_copy(source, output, spec, receipt, gate)
        verify_copy(output, spec, budget, audit)
        require(not torch.cuda.is_initialized(), "EVAL_ONLY wrapper never loads a GPU model")
        result = launch_child(args.config, output, projected)
        verify_copy(output, spec, budget, audit)
        verify_pins(source, spec)
        write_json(output / "result.json", {"status": "completed", "method": dispatch["method"], "execution": execution,
                   "source_execution": gate["terminal"], "origin_training_sha256": copied["origin_training_sha256"],
                   "copy_sha256": file_hash(output / "copy.json"), "evaluation_sha256": file_hash(output / "evaluation/result.json"),
                   "new_optimizer_updates": 0, "original_files_unchanged": True,
                   "native": result["native"], "generic_retention": result["generic_retention"],
                   "native_export": budget["native_export"], "claim_boundary": budget["claim_boundary"]})
        event(output, "onpolicy303_budget_evaluation_recovered", native=result["native"], new_optimizer_updates=0)
    except Exception as error:
        event(output, "onpolicy303_budget_evaluation_failed", exception=type(error).__name__, detail=str(error))
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid(), "new_optimizer_updates": 0})
        raise


if __name__ == "__main__":
    main()
