import argparse
import gc
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import device4_depth34 as native
import device4_recovered_source as recovered
import torch
from device4_depth34_contract import (
    deep_gate,
    feasible_gate,
    load_data,
    train_probes,
    train_schedule,
)
from generated_contract import digest, file_digest
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from run import adapter_parameters, tensor_digest

ROOT = Path(__file__).resolve().parent
CONTRACT = "device4_native_composition_from_recovered_teachers_20260912"
SOURCE_FILES = tuple(dict.fromkeys(("device4_native_composition.py", *native.SOURCE_FILES, *recovered.SOURCE_FILES)))
LOGGER = logging.getLogger("distillation.native_composition")
write_new = native.write_new
measure = native.measure
summarize = native.summarize
verify_records = native.verify_records
frozen_digest = native.frozen_digest
load_runner = native.load_runner
check_runtime = native.check_runtime
methods_for = native.methods_for
save_student = native.save_student
train_method = native.train_method
verify_student_files = native.verify_student_files
verify_training_tokens = native.verify_training_tokens
load_evaluator = native.load_evaluator
upstream_lane = native.upstream_lane


def source_hashes():
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    spec = design["native_design"]
    if design["contract"] != CONTRACT or file_digest(ROOT / spec["path"]) != spec["sha256"]:
        raise ValueError("NATIVE34_ORIGINAL_PROTOCOL_CHANGED")
    sealed = json.loads((ROOT / spec["path"]).read_text())
    upstream, original = native.validate_design(sealed)
    _, recovered_upstream, recovered_original, _ = recovered.validate_source(design["teacher_source"])
    if original != recovered_original or upstream != recovered_upstream or design["native_source_sha256"] != native.source_hashes():
        raise ValueError("NATIVE34_RECOVERED_RECIPE_OR_FROZEN_SOURCE_CHANGED")
    if any(design[field] != sealed[field] for field in ("data", "data_spec", "training", "qualification", "evaluation", "context_objective", "oracle_control", "learner")):
        raise ValueError("NATIVE34_SCIENTIFIC_RECIPE_MUST_REMAIN_UNCHANGED")
    return upstream, original


def qualify(design, upstream, original, corpus, audit, output):
    proof, old_corpus, history = recovered.verify_source(design["teacher_source"])
    common = {
        "pid": os.getpid(), "design_sha256": digest(design), "source_sha256": source_hashes(),
        "dataset": audit, "upstream": proof, "teacher_optimizer_updates": 0, "learner_updates": 0, "test_predictions": 0,
    }
    if not proof["eligible"]:
        receipt = {**common, "status": "rejected_no_eligible_depth12_teacher", "eligible": []}
        write_new(output / "qualification.json", receipt)
        return receipt
    runner = load_runner(original, upstream, old_corpus, history, output)
    check_runtime(runner, proof)
    runner.corpus = corpus
    runner.model.set_adapter("student", inference_mode=True)
    raw = {split: measure(runner, corpus[split], output, f"raw/{split}", privileged=False, forced=False) for split in ("train", "validation")}
    feasibility = feasible_gate(corpus, original, raw, runner.eos, runner.special_ids, design["qualification"])
    panels, results = {"raw": raw, "teachers": {}}, {}
    if feasibility["passed"]:
        for arm in proof["eligible"]:
            checkpoint = proof["checkpoints"][arm]
            upstream_lane.restore(runner, checkpoint)
            measured = {split: measure(runner, corpus[split], output, f"{arm}/{split}", privileged=True, forced=False) for split in ("train", "validation")}
            gate = deep_gate(corpus, original, raw, measured, runner.eos, runner.special_ids, design["qualification"])
            panels["teachers"][arm] = measured
            results[arm] = {"gate": gate, "status": "qualified" if gate["passed"] else "rejected", "checkpoint": checkpoint, "summary": {split: summarize(records) for split, records in measured.items()}}
            if tensor_digest(adapter_parameters(runner.model, "teacher")) != checkpoint["tensor_sha256"]:
                raise ValueError("NATIVE34_QUALIFICATION_MUTATED_TEACHER")
    runner.assert_frozen()
    if frozen_digest(runner) != proof["base_tensor_sha256"] or tensor_digest(adapter_parameters(runner.model, "student")) != proof["initial_tensor_sha256"]:
        raise ValueError("NATIVE34_QUALIFICATION_MUTATED_RAW_REFERENCE")
    eligible = [arm for arm in proof["eligible"] if arm in results and results[arm]["gate"]["passed"]]
    write_new(output / "qualification_panels.json", panels)
    receipt = {
        **common, "status": "qualified" if eligible else "rejected", "eligible": eligible,
        "arms": results, "feasibility": feasibility, "panels_sha256": file_digest(output / "qualification_panels.json"),
    }
    write_new(output / "qualification.json", receipt)
    return receipt


def read_certificate(root, design, original, corpus, audit):
    root = Path(root)
    receipt = json.loads((root / "qualification.json").read_text())
    if (
        receipt["status"] != "qualified" or not receipt["eligible"] or receipt["design_sha256"] != digest(design)
        or receipt["source_sha256"] != source_hashes() or receipt["dataset"] != audit
        or receipt["teacher_optimizer_updates"] != 0 or receipt["learner_updates"] != 0 or receipt["test_predictions"] != 0
        or file_digest(root / "qualification_panels.json") != receipt["panels_sha256"]
        or set(receipt["arms"]) != set(receipt["upstream"]["eligible"])
    ):
        raise ValueError("NATIVE34_QUALIFIED_TEACHER_REQUIRED_BEFORE_LEARNER")
    panels = json.loads((root / "qualification_panels.json").read_text())
    proof = receipt["upstream"]
    feasibility = feasible_gate(corpus, original, panels["raw"], proof["eos_token_id"], proof["special_token_ids"], design["qualification"])
    if feasibility != receipt["feasibility"] or not feasibility["passed"] or set(panels["teachers"]) != set(receipt["arms"]):
        raise ValueError("NATIVE34_QUALIFICATION_FEASIBILITY_OR_PANELS_CHANGED")
    eligible = []
    for arm in proof["eligible"]:
        measured = panels["teachers"][arm]
        gate = deep_gate(corpus, original, panels["raw"], measured, proof["eos_token_id"], proof["special_token_ids"], design["qualification"])
        expected = {"gate": gate, "status": "qualified" if gate["passed"] else "rejected", "checkpoint": proof["checkpoints"][arm], "summary": {split: summarize(records) for split, records in measured.items()}}
        if receipt["arms"][arm] != expected:
            raise ValueError("NATIVE34_DEEP_GATE_RECOMPUTATION_CHANGED")
        if gate["passed"]:
            eligible.append(arm)
    if eligible != receipt["eligible"]:
        raise ValueError("NATIVE34_ALL_PASSING_TEACHERS_MUST_BE_CARRIED")
    return receipt


def train(design, dispatch, upstream, original, corpus, audit, output):
    proof, old_corpus, history = recovered.verify_source(design["teacher_source"])
    qualified = read_certificate(dispatch["qualification_dir"], design, original, corpus, audit)
    if qualified["upstream"] != proof or qualified["pid"] == os.getpid():
        raise ValueError("NATIVE34_LEARNER_REQUIRES_EXACT_QUALIFIED_SOURCE_IN_NEW_PROCESS")
    destination = output / "qualification"
    destination.mkdir()
    for name in ("qualification.json", "qualification_panels.json"):
        with (destination / name).open("xb") as stream:
            stream.write((Path(dispatch["qualification_dir"]) / name).read_bytes())
    if read_certificate(destination, design, original, corpus, audit) != qualified:
        raise ValueError("NATIVE34_QUALIFICATION_SNAPSHOT_CHANGED")
    runner = load_runner(original, upstream, old_corpus, history, output)
    check_runtime(runner, proof)
    runner.corpus = corpus
    schedule = train_schedule(corpus, design)
    write_new(output / "schedule.json", schedule)
    initial = save_student(runner, output / "initial")
    runner.model.set_adapter("student", inference_mode=True)
    initial_probes = measure(runner, train_probes(corpus, design), output, "initial/train_only", privileged=False, forced=False)
    write_new(output / "initial_train_probes.json", initial_probes)
    methods = {}
    for method, arm in methods_for(qualified).items():
        checkpoint = proof["checkpoints"][arm] if arm is not None else None
        methods[method] = train_method(runner, design, method, checkpoint, schedule, output)
        if frozen_digest(runner) != proof["base_tensor_sha256"]:
            raise ValueError("NATIVE34_TRAINING_CHANGED_BASE")
    receipt = {
        "status": "trained_evaluation_pending", "pid": os.getpid(), "qualification_pid": qualified["pid"],
        "source_sha256": source_hashes(), "design_sha256": digest(design), "dataset": audit,
        "qualification_sha256": file_digest(destination / "qualification.json"),
        "qualification_panels_sha256": file_digest(destination / "qualification_panels.json"),
        "schedule_sha256": file_digest(output / "schedule.json"), "initial": initial,
        "initial_probes_sha256": file_digest(output / "initial_train_probes.json"), "methods": methods,
        "total_learner_updates": sum(value["updates"] for value in methods.values()), "teacher_optimizer_updates": 0,
        "new_validation_predictions_during_learning": 0, "test_predictions_during_learning": 0,
        "base_tensor_sha256": proof["base_tensor_sha256"], "snapshot_sha256": runner.snapshot,
        "eos_token_id": runner.eos, "special_token_ids": runner.special_ids,
        "trainable_parameters": sum(p.numel() for p in runner.student_parameters), "rank": original["lora_rank"],
        "raw_initialization_only": True, "shared_oracle_control": True,
    }
    write_new(output / "training.json", receipt)
    return receipt


def verify_training(root, design, original, corpus, audit):
    root = Path(root)
    training = json.loads((root / "training.json").read_text())
    if (
        training["status"] != "trained_evaluation_pending" or training["pid"] == os.getpid()
        or training["source_sha256"] != source_hashes() or training["design_sha256"] != digest(design) or training["dataset"] != audit
        or training["teacher_optimizer_updates"] != 0 or training["new_validation_predictions_during_learning"] != 0
        or training["test_predictions_during_learning"] != 0 or not training["raw_initialization_only"] or not training["shared_oracle_control"]
        or training["trainable_parameters"] != design["training"]["expected_trainable_parameters"] or training["rank"] != design["training"]["rank"]
        or file_digest(root / "qualification/qualification.json") != training["qualification_sha256"]
        or file_digest(root / "qualification/qualification_panels.json") != training["qualification_panels_sha256"]
        or file_digest(root / "schedule.json") != training["schedule_sha256"]
        or file_digest(root / "initial_train_probes.json") != training["initial_probes_sha256"]
    ):
        raise ValueError("NATIVE34_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
    qualified = read_certificate(root / "qualification", design, original, corpus, audit)
    expected_methods = methods_for(qualified)
    proof = qualified["upstream"]
    if (
        set(training["methods"]) != set(expected_methods) or training["qualification_pid"] != qualified["pid"]
        or training["qualification_pid"] == training["pid"]
        or training["initial"]["tensor_sha256"] != proof["initial_tensor_sha256"]
        or training["base_tensor_sha256"] != proof["base_tensor_sha256"] or training["snapshot_sha256"] != proof["snapshot_sha256"]
        or training["eos_token_id"] != proof["eos_token_id"] or training["special_token_ids"] != proof["special_token_ids"]
        or training["total_learner_updates"] != len(expected_methods) * design["training"]["updates_per_method"]
    ):
        raise ValueError("NATIVE34_MATCHED_METHODS_OR_RUNTIME_CHANGED")
    verify_student_files(training["initial"], root / "initial")
    schedule = train_schedule(corpus, design)
    if json.loads((root / "schedule.json").read_text()) != schedule:
        raise ValueError("NATIVE34_MATCHED_TRAIN_SCHEDULE_CHANGED")
    probes = train_probes(corpus, design)
    verify_records(probes, json.loads((root / "initial_train_probes.json").read_text()), corpus, original, training["eos_token_id"], training["special_token_ids"], privileged=False)
    for method, receipt in training["methods"].items():
        folder = root / method
        recipe = design["training"]
        arm = expected_methods[method]
        teacher_hash = proof["checkpoints"][arm]["tensor_sha256"] if arm is not None else None
        if (
            receipt["updates"] != recipe["updates_per_method"] or receipt["example_exposures"] != recipe["example_exposures_per_method"]
            or receipt["initial_tensor_sha256"] != training["initial"]["tensor_sha256"] or receipt["teacher_tensor_sha256"] != teacher_hash
            or not receipt["teacher_frozen"] or receipt["selected_checkpoint"] != recipe["updates_per_method"]
            or set(receipt["checkpoints"]) != {str(step) for step in recipe["checkpoint_updates"]}
            or file_digest(folder / "ledger.json") != receipt["ledger_sha256"] or file_digest(folder / "rows.json") != receipt["rows_sha256"]
        ):
            raise ValueError("NATIVE34_METHOD_BUDGET_TEACHER_OR_FIXED_FINAL_CHANGED")
        ledger = json.loads((folder / "ledger.json").read_text())
        rows = json.loads((folder / "rows.json").read_text())
        if [row["ids"] for row in ledger] != schedule or [row["step"] for row in ledger] != list(range(1, recipe["updates_per_method"] + 1)) or [row["uid"] for row in rows] != [uid for batch in schedule for uid in batch]:
            raise ValueError("NATIVE34_ACTUAL_EXPOSURE_LEDGER_CHANGED")
        if any(sum(row[name] for row in ledger) != receipt[name] for name in ("loss_tokens", "rollout_tokens", "teacher_tokens")):
            raise ValueError("NATIVE34_TOKEN_EXPOSURE_TOTAL_CHANGED")
        for step, checkpoint in receipt["checkpoints"].items():
            destination = folder / f"checkpoint{step}"
            verify_student_files(checkpoint["adapter"], destination)
            if file_digest(destination / "optimizer.pt") != checkpoint["optimizer_sha256"] or file_digest(destination / "train_probes.json") != checkpoint["train_probes_sha256"]:
                raise ValueError("NATIVE34_SAVED_OPTIMIZER_OR_PROBES_CHANGED")
            records = json.loads((destination / "train_probes.json").read_text())
            verify_records(probes, records, corpus, original, training["eos_token_id"], training["special_token_ids"], privileged=False)
            if summarize(records) != checkpoint["train_summary"]:
                raise ValueError("NATIVE34_PROBE_SUMMARY_CHANGED")
        if receipt["checkpoints"][str(recipe["updates_per_method"])]["adapter"]["tensor_sha256"] == training["initial"]["tensor_sha256"]:
            raise ValueError("NATIVE34_SAVED_LEARNER_UNCHANGED")
    return training, qualified


def evaluate(design, dispatch, original, corpus, audit, output):
    root = Path(dispatch["training_dir"])
    training, qualified = verify_training(root, design, original, corpus, audit)
    runner = load_evaluator(original, output)
    runner.corpus = corpus
    check_runtime(runner, qualified["upstream"])
    if set(runner.model.peft_config) != {"student"}:
        raise ValueError("NATIVE34_EVALUATOR_HAS_TEACHER_ADAPTER")
    verify_training_tokens(runner, root, training, design, corpus)
    conditions = {"untouched": training["initial"], **{method: value["checkpoints"][str(value["selected_checkpoint"])]["adapter"] for method, value in training["methods"].items()}}
    panels = {}
    for condition, spec in conditions.items():
        state = load_file(str(Path(spec["path"]) / "adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(runner.model, state, adapter_name="student")
        runner.model.set_adapter("student", inference_mode=True)
        if tensor_digest(adapter_parameters(runner.model, "student")) != spec["tensor_sha256"]:
            raise ValueError("NATIVE34_LEARNER_RELOAD_IDENTITY_CHANGED")
        actual = measure(runner, train_probes(corpus, design), output, f"{condition}/reload_probe", privileged=False, forced=False)
        saved = root / "initial_train_probes.json" if condition == "untouched" else root / condition / f"checkpoint{design['training']['updates_per_method']}" / "train_probes.json"
        if actual != json.loads(saved.read_text()):
            raise ValueError(f"NATIVE34_NEW_PROCESS_RELOAD_PARITY_FAILED: {condition}")
        panels[condition] = {split: measure(runner, corpus[split], output, f"{condition}/{split}", privileged=False, forced=False) for split in design["evaluation"]["splits"]}
        if tensor_digest(adapter_parameters(runner.model, "student")) != spec["tensor_sha256"] or frozen_digest(runner) != training["base_tensor_sha256"]:
            raise ValueError("NATIVE34_EVALUATION_MUTATED_WEIGHTS")
    comparisons = {}
    for split in design["evaluation"]["splits"]:
        for values in panels.values():
            verify_records(corpus[split], values[split], corpus, original, runner.eos, runner.special_ids, privileged=False)
        comparisons[split] = {method: summarize(values[split]) for method, values in panels.items()}
        comparisons[split]["gains_over_untouched"] = {method: comparisons[split][method]["all"]["accuracy"] - comparisons[split]["untouched"]["all"]["accuracy"] for method in training["methods"]}
        comparisons[split]["context_minus_shared_oracle"] = {method: comparisons[split][method]["all"]["accuracy"] - comparisons[split]["oracle_sft"]["all"]["accuracy"] for method in training["methods"] if method != "oracle_sft"}
    write_new(output / "panels.json", panels)
    result = {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"],
        "source_sha256": source_hashes(), "design_sha256": digest(design), "dataset": audit,
        "training_sha256": file_digest(root / "training.json"), "panels_sha256": file_digest(output / "panels.json"),
        "comparisons": comparisons, "qualified_teachers": qualified["eligible"], "learner_updates": training["total_learner_updates"],
        "evaluation_optimizer_updates": 0, "teacher_weights_loaded": False, "external_teacher_artifacts_read": False,
        "teacher_adapter_removed_before_predictions": True, "resident_adapters": ["student"],
        "fresh_process_reload_generation_parity": True, "claim_boundary": design["scope"],
    }
    write_new(output / "result.json", result)
    return result


def child_stage(stage, output, dispatch):
    child_dir = output / ("learning" if stage == "learn" else "evaluation")
    config = {"stage": stage, "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "qualification_dir" if stage == "learn" else "training_dir": str(output.resolve())}
    config_path = output / f"{stage}_config.json"
    write_new(config_path, config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run([
        "uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "device4_native_composition.py"),
        "--config", str(config_path.resolve()), "--output-dir", str(child_dir.resolve()),
    ], cwd=ROOT, check=True)
    result = json.loads((child_dir / "result.json").read_text())
    child_pid = result["pid"] if stage == "evaluate" else result["training_pid"]
    if result["status"] != "completed" or child_pid == os.getpid():
        raise ValueError("NATIVE34_CHILD_PROCESS_NOT_COMPLETED")
    if stage == "evaluate" and result["training_pid"] != os.getpid():
        raise ValueError("NATIVE34_CHILD_EVALUATION_TRAINING_PID_CHANGED")
    write_new(output / "result.json", {
        "status": "completed", "training_pid": result["training_pid"],
        "evaluation_pid": result["pid"] if stage == "evaluate" else result["evaluation_pid"],
        "child_result": str((child_dir / "result.json").resolve()), "child_result_sha256": file_digest(child_dir / "result.json"),
        "comparisons": result["comparisons"], "qualified_teachers": result["qualified_teachers"],
        "learner_updates": result["learner_updates"], "claim_boundary": result["claim_boundary"],
    })


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [native34] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"run", "qualify", "learn", "evaluate"}:
        raise ValueError("NATIVE34_DISPATCH_SEAL_CHANGED")
    upstream, original = validate_design(design)
    corpus, audit = load_data(design, original)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("NATIVE34_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("NATIVE34_OUTPUT_DISPATCH_CHANGED")
    sources = source_hashes()
    write_new(output / "seal.json", {"design": design, "source_sha256": sources, "dataset": audit, "dispatch": dispatch, "pid": os.getpid()})
    if args.validate_only:
        write_new(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        if dispatch["stage"] == "evaluate":
            evaluate(design, dispatch, original, corpus, audit, output)
        elif dispatch["stage"] == "learn":
            train(design, dispatch, upstream, original, corpus, audit, output)
            if source_hashes() != sources:
                raise ValueError("NATIVE34_SOURCE_CHANGED_DURING_LEARNING")
            child_stage("evaluate", output, dispatch)
        else:
            result = qualify(design, upstream, original, corpus, audit, output)
            if source_hashes() != sources:
                raise ValueError("NATIVE34_SOURCE_CHANGED_DURING_QUALIFICATION")
            if result["status"] == "qualified" and dispatch["stage"] == "run":
                child_stage("learn", output, dispatch)
            else:
                write_new(output / "result.json", {"status": result["status"], "learner_updates": 0, "teacher_optimizer_updates": 0, "qualified_teachers": result["eligible"], "qualification_sha256": file_digest(output / "qualification.json")})
        if source_hashes() != sources:
            raise ValueError("NATIVE34_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_new(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("NATIVE34_FAILED")
        raise


if __name__ == "__main__":
    main()
