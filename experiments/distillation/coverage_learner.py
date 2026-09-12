import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from choice_consolidation import (
    adapter_state,
    check_base,
    event,
    load_base,
    make_learner,
    save_adapter,
    tensor_hash,
)
from choice_contract import (
    ROOT,
    digest,
    file_hash,
    load_corpus,
    training_schedule,
    write_json,
)
from coverage_teacher import SOURCE_FILES as TEACHER_SOURCE_FILES
from coverage_teacher import coverage_gate, verify_qualified
from coverage_teacher import source_hashes as teacher_source_hashes
from coverage_teacher import validate_design as validate_coverage_design
from generation_consolidation import (
    METHODS,
    cache_teacher_responses,
    learner_protocol,
    target_sequences,
    train_arm,
    training_probes,
    validate_cache_tokens,
    verify_checkpoint_files,
    verify_target_ledgers,
)
from generation_teacher import (
    generation_gate,
    measure,
    summarize,
    verify_generated_rows,
)
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file

CONTRACT = "sequence303_coverage_selected_persistent_consolidation_20260912"
SOURCE_FILES = ("coverage_learner.py", *TEACHER_SOURCE_FILES)
SNAPSHOTS = (
    "teacher_qualification.json", "teacher_qualification_panels.json",
    "reference_training.json", "reference_schedule.json",
    "schedule.json", "initial_train_probes.json", "untouched_train_probes.json",
)


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    spec = design["coverage_design"]
    if design["contract"] != CONTRACT or file_hash(ROOT / spec["path"]) != spec["sha256"]:
        raise ValueError("COVERAGE_LEARNER_SEALED_TEACHER_DESIGN_CHANGED")
    if teacher_source_hashes() != design["coverage_source_sha256"]:
        raise ValueError("COVERAGE_LEARNER_QUALIFICATION_SOURCE_CHANGED")
    coverage = json.loads((ROOT / spec["path"]).read_text())
    previous, teacher, choice = validate_coverage_design(coverage)
    if design["training"] != previous["training"] or design["evaluation"] != previous["evaluation"]:
        raise ValueError("COVERAGE_LEARNER_MATCHED_RECIPE_CHANGED")
    if design["fixed_budget"] != {
        "learner_updates_per_arm": 384, "total_learner_updates": 768, "new_teacher_updates": 0,
        "trainable_parameters": 3833856, "resident_learner_rank": 8, "native_base_updated": False,
    }:
        raise ValueError("COVERAGE_LEARNER_CAPACITY_OR_BUDGET_CHANGED")
    return coverage, previous, teacher, choice


def copy_bound_file(source, destination, checksum):
    with Path(destination).open("xb") as handle:
        handle.write(Path(source).read_bytes())
    if file_hash(destination) != checksum:
        raise ValueError(f"COVERAGE_LEARNER_SNAPSHOT_CHANGED: {destination}")


def reference_training(design, coverage, previous, choice, corpus, audit, local_root=None):
    spec = design["reference_experiment"]
    root = Path(spec["training_dir"]) if local_root is None else Path(local_root)
    training_path = root / ("training.json" if local_root is None else "reference_training.json")
    schedule_path = root / ("schedule.json" if local_root is None else "reference_schedule.json")
    if file_hash(training_path) != spec["training_sha256"] or file_hash(schedule_path) != spec["schedule_sha256"]:
        raise ValueError("COVERAGE_LEARNER_ORIGINAL_EXPERIMENT_CHANGED")
    training = json.loads(training_path.read_text())
    expected = [[corpus["train"][i]["id"] for i in indices] for indices in training_schedule(learner_protocol(choice, design), corpus["train"])]
    if (
        training["status"] != "trained_evaluation_pending" or training["dataset"] != audit
        or training["design_sha256"] != digest(previous) or training["source_sha256"] != coverage["previous_source_sha256"]
        or training["initial"]["tensor_sha256"] != spec["initial_tensor_sha256"]
        or training["files_sha256"]["schedule.json"] != spec["schedule_sha256"]
        or training["base_tensor_sha256"] != choice["source_base_tensor_sha256"]
        or training["trainable_parameters"] != choice["learner"]["expected_trainable_parameters"]
        or training["resident_learner_rank"] != 8 or training["teacher_model_loaded"]
        or training["heldout_predictions_observed"] or set(training["arms"]) != set(METHODS)
        or json.loads(schedule_path.read_text()) != expected
        or any(arm["updates"] != design["training"]["updates_per_arm"] or arm["example_exposures"] != design["training"]["example_exposures_per_arm"] for arm in training["arms"].values())
    ):
        raise ValueError("COVERAGE_LEARNER_ORIGINAL_INITIALIZATION_OR_SCHEDULE_MISMATCH")
    return training


def teacher_proof(folder, qualified):
    return {
        "teacher_role": "coverage_selected", "qualification_dir": str(Path(folder).resolve()),
        "qualification_sha256": file_hash(Path(folder) / "qualification.json"),
        "qualification_panels_sha256": qualified["panels_sha256"],
        "teacher_tensor_sha256": qualified["teacher_tensor_sha256"],
        "source_sha256": qualified["source_sha256"], "selected_checkpoint": qualified["selected_checkpoint"],
        "screen_sha256": qualified["screen_sha256"], "eos_token_id": qualified["eos_token_id"],
        "special_token_ids": qualified["special_token_ids"],
    }


def verify_local_teacher(root, proof, design, coverage, teacher, corpus, audit):
    root = Path(root)
    qualified = json.loads((root / "teacher_qualification.json").read_text())
    if (
        proof["teacher_role"] != "coverage_selected" or proof["qualification_dir"] != str(Path(design["qualified_teacher_dir"]).resolve())
        or file_hash(root / "teacher_qualification.json") != proof["qualification_sha256"]
        or file_hash(root / "teacher_qualification_panels.json") != proof["qualification_panels_sha256"]
        or proof["qualification_panels_sha256"] != qualified["panels_sha256"]
        or qualified["status"] != "qualified" or qualified["design_sha256"] != digest(coverage)
        or qualified["source_sha256"] != design["coverage_source_sha256"] or qualified["dataset"] != audit
        or qualified["pid"] == qualified["screen_pid"] or qualified["learner_updates"] != 0
        or qualified["teacher_optimizer_updates"] != 0 or qualified["test_predictions"] != 0
        or qualified["selected_checkpoint"] not in coverage["coverage"]["candidates"]
        or qualified["checkpoint"]["tensor_sha256"] != qualified["teacher_tensor_sha256"]
        or any(proof[key] != qualified[key] for key in ("source_sha256", "teacher_tensor_sha256", "selected_checkpoint", "screen_sha256", "eos_token_id", "special_token_ids"))
    ):
        raise ValueError("COVERAGE_LEARNER_LOCAL_TEACHER_CERTIFICATE_CHANGED")
    panels = json.loads((root / "teacher_qualification_panels.json").read_text())
    gate = generation_gate(teacher, corpus, panels, proof["eos_token_id"], proof["special_token_ids"])
    train_gate = coverage_gate(corpus["train"], panels["train"]["generated"], teacher, proof["eos_token_id"], proof["special_token_ids"])
    validation = coverage_gate(corpus["validation"], panels["validation"]["generated"], teacher, proof["eos_token_id"], proof["special_token_ids"])
    if (
        not gate["passed"] or gate != qualified["gate"] or not train_gate["passed"]
        or train_gate != qualified["train_coverage"] or validation != qualified["validation_coverage_diagnostic_only"]
    ):
        raise ValueError("COVERAGE_LEARNER_LOCAL_TEACHER_GATE_CHANGED")
    return panels["train"]["generated"]


def train(design, coverage, previous, teacher, choice, corpus, audit, output):
    qualified, records = verify_qualified(design["qualified_teacher_dir"], coverage, previous, teacher, choice, corpus, audit)
    reference = reference_training(design, coverage, previous, choice, corpus, audit)
    proof = teacher_proof(design["qualified_teacher_dir"], qualified)
    for source, destination, checksum in (
        (Path(design["qualified_teacher_dir"]) / "qualification.json", "teacher_qualification.json", proof["qualification_sha256"]),
        (Path(design["qualified_teacher_dir"]) / "qualification_panels.json", "teacher_qualification_panels.json", proof["qualification_panels_sha256"]),
        (Path(design["reference_experiment"]["training_dir"]) / "training.json", "reference_training.json", design["reference_experiment"]["training_sha256"]),
        (Path(design["reference_experiment"]["training_dir"]) / "schedule.json", "reference_schedule.json", design["reference_experiment"]["schedule_sha256"]),
    ):
        copy_bound_file(source, output / destination, checksum)
    if verify_local_teacher(output, proof, design, coverage, teacher, corpus, audit) != records:
        raise ValueError("COVERAGE_LEARNER_QUALIFIED_TARGET_SNAPSHOT_CHANGED")
    cache = cache_teacher_responses(corpus["train"], records, proof)
    write_json(output / "teacher_train_cache.json", cache)
    protocol = learner_protocol(choice, design)
    base, tokenizer = load_base(protocol, output)
    prompts = validate_cache_tokens(tokenizer, corpus["train"], cache, base.get_input_embeddings().num_embeddings)
    targets = {method: target_sequences(method, tokenizer, corpus["train"], cache) for method in METHODS}
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    baseline = measure(base, tokenizer, probes, teacher, output, "untouched/train_only", detailed=False)
    write_json(output / "untouched_train_probes.json", baseline)
    model = make_learner(base, protocol)
    initial = {name: value.detach().clone() for name, value in adapter_state(model).items()}
    initial_hash = tensor_hash(initial)
    if (
        set(model.peft_config) != {"learner"} or initial_hash == proof["teacher_tensor_sha256"]
        or initial_hash != reference["initial"]["tensor_sha256"]
        or initial_hash != design["reference_experiment"]["initial_tensor_sha256"]
    ):
        raise ValueError("COVERAGE_LEARNER_FRESH_MATCHED_INITIALIZATION_FAILED")
    initial_checkpoint = save_adapter(model, output / "initial")
    initial_records = measure(model, tokenizer, probes, teacher, output, "initial/train_only", detailed=False)
    if initial_records != baseline:
        raise ValueError("COVERAGE_LEARNER_INITIAL_ZERO_OUTPUT_PARITY")
    write_json(output / "initial_train_probes.json", initial_records)
    schedule = training_schedule(protocol, corpus["train"])
    write_json(output / "schedule.json", [[corpus["train"][i]["id"] for i in batch] for batch in schedule])
    if file_hash(output / "schedule.json") != design["reference_experiment"]["schedule_sha256"]:
        raise ValueError("COVERAGE_LEARNER_ORIGINAL_ROW_ORDER_CHANGED")
    arms = {}
    for method in METHODS:
        set_peft_model_state_dict(model, initial, adapter_name="learner")
        if tensor_hash(adapter_state(model)) != initial_hash:
            raise ValueError("COVERAGE_LEARNER_ARM_INITIALIZATION_CHANGED")
        arms[method] = train_arm(model, tokenizer, corpus["train"], prompts, targets[method], schedule, method, design, teacher, output)
        check_base(model, choice["source_base_tensor_sha256"])
        if arms[method]["checkpoints"][str(design["training"]["updates_per_arm"])]["adapter"]["tensor_sha256"] == initial_hash:
            raise ValueError(f"COVERAGE_LEARNER_NO_PERSISTENT_UPDATE: {method}")
    receipt = {
        "status": "trained_evaluation_pending", "pid": os.getpid(), "design_sha256": digest(design),
        "source_sha256": source_hashes(), "dataset": audit, "teacher_provenance": proof,
        "teacher_cache_sha256": file_hash(output / "teacher_train_cache.json"),
        "teacher_targets_equal_oracle_count": sum(a == b for a, b in zip(targets[METHODS[0]], targets[METHODS[1]], strict=True)),
        "teacher_targets_all_equal_oracle": targets[METHODS[0]] == targets[METHODS[1]],
        "teacher_verified_before_learner_load": True, "initial_matches_original_experiment": True,
        "initial": initial_checkpoint, "arms": arms, "reference_experiment": design["reference_experiment"],
        "base_tensor_sha256": choice["source_base_tensor_sha256"],
        "trainable_parameters": choice["learner"]["expected_trainable_parameters"],
        "resident_learner_rank": 8, "teacher_model_loaded": False, "teacher_optimizer_updates": 0,
        "heldout_predictions_observed": False, "files_sha256": {name: file_hash(output / name) for name in SNAPSHOTS},
    }
    write_json(output / "training.json", receipt)
    event(output, "coverage_learner_training_finished", updates_per_arm=design["training"]["updates_per_arm"], teacher_targets_all_equal_oracle=receipt["teacher_targets_all_equal_oracle"])
    return receipt


def verify_learner_files(spec, folder):
    if Path(spec["path"]).resolve() != (Path(folder) / "learner").resolve():
        raise ValueError("COVERAGE_LEARNER_CHECKPOINT_OUTSIDE_OWN_RUN")
    verify_checkpoint_files(spec)


def verify_training(root, design, coverage, previous, teacher, choice, corpus, audit):
    root = Path(root)
    training = json.loads((root / "training.json").read_text())
    if (
        training["status"] != "trained_evaluation_pending" or training["pid"] == os.getpid()
        or training["design_sha256"] != digest(design) or training["source_sha256"] != source_hashes()
        or training["dataset"] != audit or training["teacher_model_loaded"] or training["teacher_optimizer_updates"] != 0
        or training["heldout_predictions_observed"] or not training["teacher_verified_before_learner_load"]
        or not training["initial_matches_original_experiment"] or training["reference_experiment"] != design["reference_experiment"]
        or set(training["arms"]) != set(METHODS) or set(training["files_sha256"]) != set(SNAPSHOTS)
        or training["base_tensor_sha256"] != choice["source_base_tensor_sha256"]
        or training["trainable_parameters"] != choice["learner"]["expected_trainable_parameters"]
        or training["resident_learner_rank"] != 8
        or file_hash(root / "teacher_train_cache.json") != training["teacher_cache_sha256"]
    ):
        raise ValueError("COVERAGE_LEARNER_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
    for name, checksum in training["files_sha256"].items():
        if file_hash(root / name) != checksum:
            raise ValueError(f"COVERAGE_LEARNER_TRAINING_FILE_CHANGED: {name}")
    reference = reference_training(design, coverage, previous, choice, corpus, audit, local_root=root)
    if training["initial"]["tensor_sha256"] != reference["initial"]["tensor_sha256"]:
        raise ValueError("COVERAGE_LEARNER_RECORDED_INITIALIZATION_CHANGED")
    verify_learner_files(training["initial"], root / "initial")
    schedule = training_schedule(learner_protocol(choice, design), corpus["train"])
    expected = [[corpus["train"][i]["id"] for i in batch] for batch in schedule]
    if json.loads((root / "schedule.json").read_text()) != expected or file_hash(root / "schedule.json") != design["reference_experiment"]["schedule_sha256"]:
        raise ValueError("COVERAGE_LEARNER_EXPOSURE_SCHEDULE_CHANGED")
    records = verify_local_teacher(root, training["teacher_provenance"], design, coverage, teacher, corpus, audit)
    cache = json.loads((root / "teacher_train_cache.json").read_text())
    if cache != cache_teacher_responses(corpus["train"], records, training["teacher_provenance"]):
        raise ValueError("COVERAGE_LEARNER_CACHED_TEACHER_OUTPUTS_CHANGED")
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    for name in ("initial_train_probes.json", "untouched_train_probes.json"):
        verify_generated_rows(probes, json.loads((root / name).read_text()), teacher, cache["source"]["eos_token_id"], cache["source"]["special_token_ids"])
    if json.loads((root / "initial_train_probes.json").read_text()) != json.loads((root / "untouched_train_probes.json").read_text()):
        raise ValueError("COVERAGE_LEARNER_RECORDED_INITIAL_PARITY_CHANGED")
    recipe = design["training"]
    for method, arm in training["arms"].items():
        folder = root / method
        if (
            arm["updates"] != recipe["updates_per_arm"] or arm["example_exposures"] != recipe["example_exposures_per_arm"]
            or arm["selected_checkpoint"] != recipe["updates_per_arm"] or arm["teacher_model_loaded"]
            or set(arm["checkpoints"]) != {str(step) for step in recipe["checkpoint_updates"]}
            or file_hash(folder / "ledger.json") != arm["ledger_sha256"]
        ):
            raise ValueError("COVERAGE_LEARNER_ARM_BUDGET_OR_FINAL_SELECTION_CHANGED")
        ledger = json.loads((folder / "ledger.json").read_text())
        if (
            [row["ids"] for row in ledger] != expected or [row["step"] for row in ledger] != list(range(1, recipe["updates_per_arm"] + 1))
            or sum(row["loss_tokens"] for row in ledger) != arm["loss_token_exposures"]
            or any(not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm"]) for row in ledger)
        ):
            raise ValueError("COVERAGE_LEARNER_ACTUAL_EXPOSURE_LEDGER_CHANGED")
        for step, checkpoint in arm["checkpoints"].items():
            destination = folder / f"checkpoint{step}"
            verify_learner_files(checkpoint["adapter"], destination)
            if file_hash(destination / "optimizer.pt") != checkpoint["optimizer_sha256"] or file_hash(destination / "train_probes.json") != checkpoint["train_probes_sha256"]:
                raise ValueError("COVERAGE_LEARNER_CHECKPOINT_PROBES_CHANGED")
            records = json.loads((destination / "train_probes.json").read_text())
            verify_generated_rows(probes, records, teacher, cache["source"]["eos_token_id"], cache["source"]["special_token_ids"])
            if summarize(records) != checkpoint["train_probe_summary"]:
                raise ValueError("COVERAGE_LEARNER_TRAIN_PROBE_SUMMARY_CHANGED")
        if arm["checkpoints"][str(recipe["updates_per_arm"])]["adapter"]["tensor_sha256"] == training["initial"]["tensor_sha256"]:
            raise ValueError("COVERAGE_LEARNER_RECORDED_LEARNER_UNCHANGED")
    return training, cache


def evaluate_learners(design, dispatch, coverage, previous, teacher, choice, corpus, audit, output):
    root = Path(dispatch["training_dir"])
    training, cache = verify_training(root, design, coverage, previous, teacher, choice, corpus, audit)
    base, tokenizer = load_base(learner_protocol(choice, design), output)
    validate_cache_tokens(tokenizer, corpus["train"], cache, base.get_input_embeddings().num_embeddings)
    verify_target_ledgers(root, training, tokenizer, corpus, cache, design, choice)
    panels = {split: {"untouched": measure(base, tokenizer, corpus[split], teacher, output, f"untouched/{split}", detailed=False)} for split in design["evaluation"]["splits"]}
    model = PeftModel.from_pretrained(base, training["initial"]["path"], adapter_name="learner", is_trainable=False, local_files_only=True).requires_grad_(False).eval()
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    conditions = {"initial": training["initial"], **{method: arm["checkpoints"][str(arm["selected_checkpoint"])]["adapter"] for method, arm in training["arms"].items()}}
    for condition, checkpoint in conditions.items():
        state = load_file(str(Path(checkpoint["path"]) / "adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(model, state, adapter_name="learner")
        model.set_adapter("learner", inference_mode=True)
        model.requires_grad_(False).eval()
        if set(model.peft_config) != {"learner"} or model.active_adapters != ["learner"] or tensor_hash(adapter_state(model)) != checkpoint["tensor_sha256"]:
            raise ValueError("COVERAGE_LEARNER_RELOAD_IDENTITY_OR_EXTRA_ADAPTER")
        actual = measure(model, tokenizer, probes, teacher, output, f"{condition}/reload_probe", detailed=False)
        saved = root / "initial_train_probes.json" if condition == "initial" else root / condition / f"checkpoint{design['training']['updates_per_arm']}" / "train_probes.json"
        if actual != json.loads(saved.read_text()):
            raise ValueError(f"COVERAGE_LEARNER_RELOAD_GENERATION_PARITY: {condition}")
        if condition != "initial":
            for split in design["evaluation"]["splits"]:
                panels[split][condition] = measure(model, tokenizer, corpus[split], teacher, output, f"{condition}/{split}", detailed=False)
        if tensor_hash(adapter_state(model)) != checkpoint["tensor_sha256"]:
            raise ValueError("COVERAGE_LEARNER_EVALUATION_MUTATED_ADAPTER")
        check_base(model, choice["source_base_tensor_sha256"])
    comparison, bindings = {}, {}
    for split, conditions in panels.items():
        comparison[split] = {name: summarize(records) for name, records in conditions.items()}
        bindings[split] = {
            name: coverage_gate(corpus[split], records, teacher, tokenizer.eos_token_id, tokenizer.all_special_ids)
            for name, records in conditions.items()
        }
        for left, right in ((METHODS[0], "untouched"), (METHODS[1], "untouched"), METHODS):
            differences = [int(a["generation"]["correct"]) - int(b["generation"]["correct"]) for a, b in zip(conditions[left], conditions[right], strict=True)]
            comparison[split][f"{left}_minus_{right}"] = sum(differences) / len(differences)
    write_json(output / "panels.json", panels)
    write_json(output / "binding_diagnostics.json", bindings)
    final = str(design["training"]["updates_per_arm"])
    result = {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"],
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "training_sha256": file_hash(root / "training.json"), "panels_sha256": file_hash(output / "panels.json"),
        "binding_diagnostics_sha256": file_hash(output / "binding_diagnostics.json"),
        "comparisons": comparison, "selected_teacher_checkpoint": cache["source"]["selected_checkpoint"],
        "teacher_targets_equal_oracle_count": training["teacher_targets_equal_oracle_count"],
        "teacher_targets_all_equal_oracle": training["teacher_targets_all_equal_oracle"],
        "final_arm_tensors_equal": training["arms"][METHODS[0]]["checkpoints"][final]["adapter"]["tensor_sha256"] == training["arms"][METHODS[1]]["checkpoints"][final]["adapter"]["tensor_sha256"],
        "initial_and_schedule_match_original": True, "teacher_model_loaded": False,
        "teacher_artifact_files_read_during_evaluation": False, "learner_reload_generation_parity": True,
        "resident_active_adapters": 1, "optimizer_updates": 0, "claim_boundary": design["claim_boundary"],
    }
    write_json(output / "result.json", result)
    return result


def evaluate_in_new_process(output, dispatch):
    config = {"stage": "evaluate", "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "training_dir": str(output.resolve())}
    write_json(output / "evaluation_config.json", config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run([
        "uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "coverage_learner.py"),
        "--config", str((output / "evaluation_config.json").resolve()), "--output-dir", str((output / "evaluation").resolve()),
    ], cwd=ROOT, check=True)
    path = output / "evaluation/result.json"
    result = json.loads(path.read_text())
    if result["status"] != "completed" or result["pid"] == os.getpid() or result["training_pid"] != os.getpid():
        raise ValueError("COVERAGE_LEARNER_CHILD_EVALUATOR_IDENTITY")
    write_json(output / "result.json", {
        "status": "completed", "training_pid": os.getpid(), "evaluation_pid": result["pid"],
        "training_sha256": file_hash(output / "training.json"), "evaluation_result": str(path),
        "evaluation_result_sha256": file_hash(path), "comparisons": result["comparisons"],
        "selected_teacher_checkpoint": result["selected_teacher_checkpoint"],
        "teacher_targets_all_equal_oracle": result["teacher_targets_all_equal_oracle"],
        "teacher_model_loaded": False, "claim_boundary": result["claim_boundary"],
    })


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [coverage-learner] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"run", "train", "evaluate"}:
        raise ValueError("COVERAGE_LEARNER_DISPATCH_DESIGN_CHANGED")
    coverage, previous, teacher, choice = validate_design(design)
    corpus, audit = load_corpus(choice)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("COVERAGE_LEARNER_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("COVERAGE_LEARNER_OUTPUT_DISPATCH_CHANGED")
    sources = source_hashes()
    write_json(output / "seal.json", {"design": design, "source_sha256": sources, "dataset": audit, "dispatch": dispatch, "pid": os.getpid()})
    if args.validate_only:
        write_json(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        if dispatch["stage"] == "evaluate":
            evaluate_learners(design, dispatch, coverage, previous, teacher, choice, corpus, audit, output)
        else:
            train(design, coverage, previous, teacher, choice, corpus, audit, output)
            if source_hashes() != sources:
                raise ValueError("COVERAGE_LEARNER_SOURCE_CHANGED_DURING_TRAINING")
            if dispatch["stage"] == "run":
                evaluate_in_new_process(output, dispatch)
        if source_hashes() != sources:
            raise ValueError("COVERAGE_LEARNER_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        event(output, "coverage_learner_failed", exception=type(error).__name__, detail=str(error))
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
