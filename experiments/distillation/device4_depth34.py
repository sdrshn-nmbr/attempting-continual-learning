import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import device4_curriculum as upstream_lane
import torch
from device4_depth34_contract import (
    CONTRACT,
    DATA_SPEC,
    ROOT,
    deep_gate,
    feasible_gate,
    load_data,
    train_probes,
    train_schedule,
)
from generated_contract import (
    digest,
    file_digest,
    learner_prompt,
    make_corpus,
    qualification_gate,
    qualification_rows,
    teacher_prompt,
)
from peft import set_peft_model_state_dict
from qualified import QualifiedRunner
from safetensors.torch import load_file
from torch.nn import functional

from objectives import distillation_loss
from run import adapter_parameters, append_json, tensor_digest
from tasks import FAMILIES

SOURCE_FILES = ("device4_depth34.py", "device4_depth34_contract.py", *upstream_lane.SOURCE_FILES)
LOGGER = logging.getLogger("distillation.depth34")
write_new = upstream_lane.write_new
measure = upstream_lane.measure
summarize = upstream_lane.summarize
verify_records = upstream_lane.verify_records
frozen_digest = upstream_lane.frozen_digest


def source_hashes():
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    spec = design["upstream_design"]
    if design["contract"] != CONTRACT or file_digest(ROOT / spec["path"]) != spec["sha256"]:
        raise ValueError("DEPTH34_UPSTREAM_PROTOCOL_CHANGED")
    upstream = json.loads((ROOT / spec["path"]).read_text())
    original = upstream_lane.validate_design(upstream)
    if upstream_lane.source_hashes() != design["upstream_source_sha256"] or design["data_spec"] != DATA_SPEC:
        raise ValueError("DEPTH34_FROZEN_SOURCE_OR_DATA_CONTRACT_CHANGED")
    if design["qualification"] != {
        "minimum_accuracy": 0.875, "minimum_gain": 0.25, "maximum_p": 0.05,
        "selection": "all_eligible_teachers_pass_every_family_depth_train_validation_cell", "teacher_optimizer_updates": 0,
    }:
        raise ValueError("DEPTH34_QUALIFICATION_GATE_CHANGED")
    if design["training"] != {
        "updates_per_method": 384, "examples_per_update": 4, "example_exposures_per_method": 1536,
        "exposures_per_family_depth": 256, "order_seed": 91234101, "sampling_seed": 91234103,
        "checkpoint_updates": [24, 96, 384], "probe_rows_per_family_depth": 4,
        "selection": "fixed_final384", "learning_rate": 0.0002, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "shared_oracle_control": True, "expected_trainable_parameters": 8519680, "rank": 16, "alpha": 32,
        "rollout_forward_logprob_max_error": 0.25,
    }:
        raise ValueError("DEPTH34_PERSISTENT_LEARNER_RECIPE_CHANGED")
    if design["evaluation"] != {
        "splits": ["test"], "teacher_weights_loaded": False, "teacher_adapter_removed_before_predictions": True,
        "require_fresh_pid": True, "strict_original_four_digits_native_eos": True,
    }:
        raise ValueError("DEPTH34_EVALUATION_CONTRACT_CHANGED")
    return upstream, original


def verify_adapter(spec):
    if set(spec["files_sha256"]) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("DEPTH34_ADAPTER_FILE_SET")
    for name, expected in spec["files_sha256"].items():
        if file_digest(Path(spec["path"]) / name) != expected:
            raise ValueError(f"DEPTH34_SAVED_ADAPTER_CHANGED: {name}")


def verify_upstream(design, upstream, original):
    corpus = make_corpus(original, upstream_lane.SPLITS)
    schedules = upstream_lane.make_schedules(corpus, upstream)
    history = upstream_lane.verify_history(upstream, original, corpus, schedules)
    folder = Path(design["upstream_training_dir"])
    training = upstream_lane.verify_training(folder, upstream, original, corpus, schedules)
    root = Path(design["upstream_qualification_dir"])
    qualified = json.loads((root / "qualification.json").read_text())
    if (
        qualified["status"] != "completed" or qualified["source_sha256"] != upstream_lane.source_hashes()
        or qualified["design_sha256"] != digest(upstream) or qualified["training_sha256"] != file_digest(folder / "training.json")
        or qualified["training_pid"] != training["pid"] or qualified["pid"] == training["pid"]
        or qualified["learner_updates"] != 0 or qualified["test_predictions"] != 0 or qualified["composition_predictions"] != 0
        or set(qualified["arms"]) != set(upstream_lane.ARMS)
        or file_digest(root / "panels.json") != qualified["panels_sha256"]
    ):
        raise ValueError("DEPTH34_UPSTREAM_QUALIFICATION_CHANGED")
    panels = json.loads((root / "panels.json").read_text())
    tasks = qualification_rows(corpus, original, "fallback_validation")
    eligible, checkpoints, raw_reference = [], {}, None
    for arm in upstream_lane.ARMS:
        result, panel = qualified["arms"][arm], panels[arm]
        verify_records(tasks, panel["diagnostics"], corpus, original, training["eos_token_id"], training["special_token_ids"])
        gate = qualification_gate(panel["pairs"], corpus, original, training["eos_token_id"], training["special_token_ids"], "fallback_validation")
        if [row["teacher"] for row in panel["pairs"]] != [row["generation"] for row in panel["diagnostics"]]:
            raise ValueError("DEPTH34_UPSTREAM_PAIR_DIAGNOSTIC_MISMATCH")
        raw = [row["learner"] for row in panel["pairs"]]
        if raw_reference is not None and raw_reference != raw:
            raise ValueError("DEPTH34_UPSTREAM_UNMATCHED_RAW_REFERENCE")
        raw_reference = raw
        summaries = {split: summarize([row for row in panel["diagnostics"] if row["split"] == split]) for split in ("train", "fallback_validation")}
        depth_ok = all(summaries[split][f"{family}/depth{depth}"]["accuracy"] >= upstream["evaluation"]["minimum_depth_accuracy_for_consolidation"] for split in summaries for family in FAMILIES for depth in (1, 2))
        checkpoint = training["arms"][arm]["checkpoints"][str(upstream["training"]["updates_per_arm"])]["adapter"]
        if (
            result["gate"] != gate or result["summary"] != summaries or result["depth_accuracy_check_passed"] != depth_ok
            or result["eligible_for_depth3_4_prerequisite"] != (gate["passed"] and depth_ok)
            or result["status"] != ("qualified" if gate["passed"] else "rejected")
            or result["checkpoint"] != checkpoint or result["checkpoint_update"] != upstream["training"]["updates_per_arm"]
            or not result["reload_generation_parity"] or result["learner_updates"] != 0
        ):
            raise ValueError("DEPTH34_UPSTREAM_ELIGIBILITY_RECOMPUTATION_FAILED")
        verify_adapter(checkpoint)
        checkpoints[arm] = checkpoint
        saved = json.loads((folder / arm / f"checkpoint{upstream['training']['updates_per_arm']}" / "train_diagnostics.json").read_text())
        lookup = {row["uid"]: row["generation"] for row in saved}
        if any(row["generation"] != lookup[row["uid"]] for row in panel["diagnostics"] if row["split"] == "train"):
            raise ValueError("DEPTH34_UPSTREAM_RELOAD_PARITY_CHANGED")
        if gate["passed"] and depth_ok:
            eligible.append(arm)
    proof = {
        "qualification_sha256": file_digest(root / "qualification.json"), "panels_sha256": qualified["panels_sha256"],
        "training_sha256": file_digest(folder / "training.json"), "eligible": eligible, "checkpoints": checkpoints,
        "initial_tensor_sha256": training["initial"]["tensor_sha256"], "snapshot_sha256": training["snapshot_sha256"],
        "eos_token_id": training["eos_token_id"], "special_token_ids": training["special_token_ids"],
        "base_tensor_sha256": training["arms"][upstream_lane.ARMS[0]]["base_before_sha256"],
    }
    return proof, corpus, history


def load_runner(original, upstream, old_corpus, history, output):
    return upstream_lane.load_runner(original, upstream, old_corpus, history, output)


def check_runtime(runner, proof):
    if runner.snapshot != proof["snapshot_sha256"] or runner.eos != proof["eos_token_id"] or runner.special_ids != proof["special_token_ids"]:
        raise ValueError("DEPTH34_PINNED_MODEL_OR_TOKENIZER_CHANGED")
    if tensor_digest(adapter_parameters(runner.model, "student")) != proof["initial_tensor_sha256"] or frozen_digest(runner) != proof["base_tensor_sha256"]:
        raise ValueError("DEPTH34_FRESH_INITIALIZATION_OR_BASE_CHANGED")


def qualify(design, upstream, original, corpus, audit, output):
    proof, old_corpus, history = verify_upstream(design, upstream, original)
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
                raise ValueError("DEPTH34_QUALIFICATION_MUTATED_TEACHER")
    runner.assert_frozen()
    if frozen_digest(runner) != proof["base_tensor_sha256"] or tensor_digest(adapter_parameters(runner.model, "student")) != proof["initial_tensor_sha256"]:
        raise ValueError("DEPTH34_QUALIFICATION_MUTATED_RAW_REFERENCE")
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
        raise ValueError("DEPTH34_QUALIFIED_TEACHER_REQUIRED_BEFORE_LEARNER")
    panels = json.loads((root / "qualification_panels.json").read_text())
    proof = receipt["upstream"]
    feasibility = feasible_gate(corpus, original, panels["raw"], proof["eos_token_id"], proof["special_token_ids"], design["qualification"])
    if feasibility != receipt["feasibility"] or not feasibility["passed"] or set(panels["teachers"]) != set(receipt["arms"]):
        raise ValueError("DEPTH34_QUALIFICATION_FEASIBILITY_OR_PANELS_CHANGED")
    eligible = []
    for arm in proof["eligible"]:
        measured = panels["teachers"][arm]
        gate = deep_gate(corpus, original, panels["raw"], measured, proof["eos_token_id"], proof["special_token_ids"], design["qualification"])
        expected = {"gate": gate, "status": "qualified" if gate["passed"] else "rejected", "checkpoint": proof["checkpoints"][arm], "summary": {split: summarize(records) for split, records in measured.items()}}
        if receipt["arms"][arm] != expected:
            raise ValueError("DEPTH34_DEEP_GATE_RECOMPUTATION_CHANGED")
        if gate["passed"]:
            eligible.append(arm)
    if eligible != receipt["eligible"]:
        raise ValueError("DEPTH34_ALL_PASSING_TEACHERS_MUST_BE_CARRIED")
    return receipt


def methods_for(qualified):
    return {"oracle_sft": None, **{f"context_forward_kl_{arm}": arm for arm in qualified["eligible"]}}


def save_student(runner, folder):
    runner.model.save_pretrained(folder, selected_adapters=["student"], save_embedding_layers=False, safe_serialization=True)
    path = folder / "student"
    return {"path": str(path.resolve()), "tensor_sha256": tensor_digest(adapter_parameters(runner.model, "student")), "files_sha256": {name: file_digest(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")}}


def train_method(runner, design, method, checkpoint, schedule, output):
    runner.reset_method()
    torch.manual_seed(design["training"]["sampling_seed"])
    if checkpoint is not None:
        upstream_lane.restore(runner, checkpoint)
    runner.model.set_adapter("student")
    teacher = adapter_parameters(runner.model, "teacher")
    teacher_hash = tensor_digest(teacher)
    teacher_versions = [(p, p._version) for p in teacher.values()]
    initial = tensor_digest(adapter_parameters(runner.model, "student"))
    folder = output / method
    folder.mkdir()
    lookup = {task.uid: task for task in runner.corpus["train"]}
    probes = train_probes(runner.corpus, design)
    ledger, rows, checkpoints = [], [], {}
    for step, batch in enumerate(schedule, 1):
        runner.optimizer.zero_grad(set_to_none=True)
        examples = []
        for uid in batch:
            task = lookup[uid]
            runner.model.set_adapter("student")
            prompt = learner_prompt(task)
            teacher_ids = None
            if checkpoint is None:
                ids, response = runner.prompt_ids(prompt), runner.target_tokens(task)
                logits = runner.logits(ids, response)
                loss = functional.cross_entropy(logits.float(), response)
                diagnostics = {"objective": "oracle_whole_answer_native_eos"}
            else:
                ids, response, _, old = runner.generate(prompt, sample=True)
                teacher_ids = runner.prompt_ids(teacher_prompt(task, runner.corpus["demonstration"]))
                runner.model.set_adapter("teacher", inference_mode=True)
                with torch.no_grad():
                    teacher_logits = runner.logits(teacher_ids, response).detach()
                runner.model.set_adapter("student")
                logits = runner.logits(ids, response)
                loss, diagnostics = distillation_loss(logits, teacher_logits, response, "sdft_forward", old)
                diagnostics["objective"] = "fixed_teacher_full_vocabulary_forward_kl_on_student_trajectory"
                if diagnostics["rollout_to_forward_logprob_max_error"] > design["training"]["rollout_forward_logprob_max_error"]:
                    raise ValueError("DEPTH34_ON_POLICY_ROLLOUT_FORWARD_MISMATCH")
                del teacher_logits
            if not torch.isfinite(loss):
                raise ValueError(f"DEPTH34_NONFINITE_LOSS: {method}/{step}")
            (loss / len(batch)).backward()
            record = {
                "method": method, "step": step, "uid": uid, "task_sha256": digest(asdict(task)),
                "prompt_token_ids": ids[0].tolist(), "response_token_ids": response.tolist(),
                "teacher_prompt_token_ids": teacher_ids[0].tolist() if teacher_ids is not None else None,
                "loss": float(loss.detach()), "diagnostics": diagnostics,
            }
            rows.append(record)
            examples.append(record)
            append_json(folder / "rows.partial.jsonl", record)
            del logits, loss
        norm = float(torch.nn.utils.clip_grad_norm_(runner.student_parameters, design["training"]["max_grad_norm"]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"DEPTH34_INVALID_GRADIENT: {method}/{step}")
        runner.optimizer.step()
        runner.step += 1
        runner.assert_frozen()
        if any(p.grad is not None or p._version != version for p, version in teacher_versions):
            raise ValueError("DEPTH34_FIXED_TEACHER_MUTATED_OR_HAS_GRADIENT")
        update = {
            "step": step, "ids": batch, "response_sha256": [digest(row["response_token_ids"]) for row in examples],
            "loss_tokens": sum(len(row["response_token_ids"]) for row in examples),
            "rollout_tokens": sum(len(row["response_token_ids"]) for row in examples) if checkpoint is not None else 0,
            "teacher_tokens": sum(len(row["teacher_prompt_token_ids"]) + len(row["response_token_ids"]) for row in examples) if checkpoint is not None else 0,
            "loss": sum(row["loss"] for row in examples) / len(examples), "gradient_norm": norm,
        }
        ledger.append(update)
        runner.event("depth34_optimizer_update", method=method, **update)
        if step in design["training"]["checkpoint_updates"]:
            runner.optimizer.zero_grad(set_to_none=True)
            destination = folder / f"checkpoint{step}"
            adapter = save_student(runner, destination)
            torch.save(runner.optimizer.state_dict(), destination / "optimizer.pt")
            runner.model.set_adapter("student", inference_mode=True)
            records = measure(runner, probes, output, f"{method}/train_only/{step}", privileged=False, forced=False)
            write_new(destination / "train_probes.json", records)
            checkpoints[str(step)] = {"adapter": adapter, "optimizer_sha256": file_digest(destination / "optimizer.pt"), "train_probes_sha256": file_digest(destination / "train_probes.json"), "train_summary": summarize(records)}
            if tensor_digest(teacher) != teacher_hash:
                raise ValueError("DEPTH34_TEACHER_HASH_CHANGED")
    if tensor_digest(adapter_parameters(runner.model, "student")) == initial or runner.step != design["training"]["updates_per_method"]:
        raise ValueError("DEPTH34_LEARNER_DID_NOT_PERSISTENTLY_UPDATE")
    write_new(folder / "rows.json", rows)
    write_new(folder / "ledger.json", ledger)
    return {
        "updates": len(ledger), "example_exposures": len(rows), "initial_tensor_sha256": initial,
        "teacher_tensor_sha256": teacher_hash if checkpoint is not None else None,
        "teacher_frozen": True, "checkpoints": checkpoints, "selected_checkpoint": design["training"]["updates_per_method"],
        "loss_tokens": sum(row["loss_tokens"] for row in ledger), "rollout_tokens": sum(row["rollout_tokens"] for row in ledger),
        "teacher_tokens": sum(row["teacher_tokens"] for row in ledger),
        "ledger_sha256": file_digest(folder / "ledger.json"), "rows_sha256": file_digest(folder / "rows.json"),
    }


def train(design, dispatch, upstream, original, corpus, audit, output):
    proof, old_corpus, history = verify_upstream(design, upstream, original)
    qualified = read_certificate(dispatch["qualification_dir"], design, original, corpus, audit)
    if qualified["upstream"] != proof or qualified["pid"] == os.getpid():
        raise ValueError("DEPTH34_LEARNER_REQUIRES_EXACT_QUALIFIED_SOURCE_IN_NEW_PROCESS")
    destination = output / "qualification"
    destination.mkdir()
    for name in ("qualification.json", "qualification_panels.json"):
        with (destination / name).open("xb") as stream:
            stream.write((Path(dispatch["qualification_dir"]) / name).read_bytes())
    if read_certificate(destination, design, original, corpus, audit) != qualified:
        raise ValueError("DEPTH34_QUALIFICATION_SNAPSHOT_CHANGED")
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
            raise ValueError("DEPTH34_TRAINING_CHANGED_BASE")
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


def verify_student_files(spec, folder):
    if Path(spec["path"]).resolve() != (Path(folder) / "student").resolve():
        raise ValueError("DEPTH34_STUDENT_FILE_OUTSIDE_OWN_RUN")
    verify_adapter(spec)


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
        raise ValueError("DEPTH34_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
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
        raise ValueError("DEPTH34_MATCHED_METHODS_OR_RUNTIME_CHANGED")
    verify_student_files(training["initial"], root / "initial")
    schedule = train_schedule(corpus, design)
    if json.loads((root / "schedule.json").read_text()) != schedule:
        raise ValueError("DEPTH34_MATCHED_TRAIN_SCHEDULE_CHANGED")
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
            raise ValueError("DEPTH34_METHOD_BUDGET_TEACHER_OR_FIXED_FINAL_CHANGED")
        ledger = json.loads((folder / "ledger.json").read_text())
        rows = json.loads((folder / "rows.json").read_text())
        if [row["ids"] for row in ledger] != schedule or [row["step"] for row in ledger] != list(range(1, recipe["updates_per_method"] + 1)) or [row["uid"] for row in rows] != [uid for batch in schedule for uid in batch]:
            raise ValueError("DEPTH34_ACTUAL_EXPOSURE_LEDGER_CHANGED")
        if any(sum(row[name] for row in ledger) != receipt[name] for name in ("loss_tokens", "rollout_tokens", "teacher_tokens")):
            raise ValueError("DEPTH34_TOKEN_EXPOSURE_TOTAL_CHANGED")
        for step, checkpoint in receipt["checkpoints"].items():
            destination = folder / f"checkpoint{step}"
            verify_student_files(checkpoint["adapter"], destination)
            if file_digest(destination / "optimizer.pt") != checkpoint["optimizer_sha256"] or file_digest(destination / "train_probes.json") != checkpoint["train_probes_sha256"]:
                raise ValueError("DEPTH34_SAVED_OPTIMIZER_OR_PROBES_CHANGED")
            records = json.loads((destination / "train_probes.json").read_text())
            verify_records(probes, records, corpus, original, training["eos_token_id"], training["special_token_ids"], privileged=False)
            if summarize(records) != checkpoint["train_summary"]:
                raise ValueError("DEPTH34_PROBE_SUMMARY_CHANGED")
        if receipt["checkpoints"][str(recipe["updates_per_method"])]["adapter"]["tensor_sha256"] == training["initial"]["tensor_sha256"]:
            raise ValueError("DEPTH34_SAVED_LEARNER_UNCHANGED")
    return training, qualified


def verify_training_tokens(runner, root, training, design, corpus):
    lookup = {task.uid: task for task in corpus["train"]}
    vocabulary = runner.model.get_input_embeddings().num_embeddings
    for method in training["methods"]:
        rows = json.loads((root / method / "rows.json").read_text())
        ledger = json.loads((root / method / "ledger.json").read_text())
        batch_size = design["training"]["examples_per_update"]
        for update, batch in zip(ledger, [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)], strict=True):
            for row in batch:
                task = lookup[row["uid"]]
                response = row["response_token_ids"]
                if (
                    row["method"] != method or row["step"] != update["step"] or row["task_sha256"] != digest(asdict(task))
                    or row["prompt_token_ids"] != runner.prompt_ids(learner_prompt(task))[0].tolist()
                    or not response or len(response) > runner.config["max_new_tokens"]
                    or any(type(token) is not int or not 0 <= token < vocabulary for token in response)
                    or not math.isfinite(row["loss"])
                ):
                    raise ValueError("DEPTH34_ACTUAL_TRAIN_TOKEN_BOUNDARY_CHANGED")
                if method == "oracle_sft":
                    if response != runner.target_tokens(task).tolist() or row["teacher_prompt_token_ids"] is not None or row["diagnostics"] != {"objective": "oracle_whole_answer_native_eos"}:
                        raise ValueError("DEPTH34_ORACLE_TARGET_CHANGED")
                elif (
                    row["teacher_prompt_token_ids"] != runner.prompt_ids(teacher_prompt(task, corpus["demonstration"]))[0].tolist()
                    or row["diagnostics"]["objective"] != "fixed_teacher_full_vocabulary_forward_kl_on_student_trajectory"
                    or row["diagnostics"]["support"] != "full_vocabulary" or row["diagnostics"]["vocabulary_size"] != vocabulary
                    or row["diagnostics"]["rollout_to_forward_logprob_max_error"] > design["training"]["rollout_forward_logprob_max_error"]
                ):
                    raise ValueError("DEPTH34_CONTEXT_DISTILLATION_TRACE_CHANGED")
            response_count = sum(len(row["response_token_ids"]) for row in batch)
            teacher_count = sum(len(row["teacher_prompt_token_ids"]) + len(row["response_token_ids"]) for row in batch) if method != "oracle_sft" else 0
            if (
                update["response_sha256"] != [digest(row["response_token_ids"]) for row in batch]
                or update["loss_tokens"] != response_count or update["teacher_tokens"] != teacher_count
                or update["rollout_tokens"] != (response_count if method != "oracle_sft" else 0)
                or not math.isfinite(update["gradient_norm"]) or update["gradient_norm"] <= 0
            ):
                raise ValueError("DEPTH34_ROW_AND_UPDATE_LEDGER_DISAGREE")


def load_evaluator(original, output):
    runner = QualifiedRunner(original, output)
    runner.load()
    runner.model.set_adapter("student", inference_mode=True)
    runner.model.delete_adapter("teacher")
    return runner


def evaluate(design, dispatch, original, corpus, audit, output):
    root = Path(dispatch["training_dir"])
    training, qualified = verify_training(root, design, original, corpus, audit)
    runner = load_evaluator(original, output)
    runner.corpus = corpus
    check_runtime(runner, qualified["upstream"])
    if set(runner.model.peft_config) != {"student"}:
        raise ValueError("DEPTH34_EVALUATOR_HAS_TEACHER_ADAPTER")
    verify_training_tokens(runner, root, training, design, corpus)
    conditions = {"untouched": training["initial"], **{method: value["checkpoints"][str(value["selected_checkpoint"])]["adapter"] for method, value in training["methods"].items()}}
    panels = {}
    for condition, spec in conditions.items():
        state = load_file(str(Path(spec["path"]) / "adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(runner.model, state, adapter_name="student")
        runner.model.set_adapter("student", inference_mode=True)
        if tensor_digest(adapter_parameters(runner.model, "student")) != spec["tensor_sha256"]:
            raise ValueError("DEPTH34_LEARNER_RELOAD_IDENTITY_CHANGED")
        actual = measure(runner, train_probes(corpus, design), output, f"{condition}/reload_probe", privileged=False, forced=False)
        saved = root / "initial_train_probes.json" if condition == "untouched" else root / condition / f"checkpoint{design['training']['updates_per_method']}" / "train_probes.json"
        if actual != json.loads(saved.read_text()):
            raise ValueError(f"DEPTH34_NEW_PROCESS_RELOAD_PARITY_FAILED: {condition}")
        panels[condition] = {split: measure(runner, corpus[split], output, f"{condition}/{split}", privileged=False, forced=False) for split in design["evaluation"]["splits"]}
        if tensor_digest(adapter_parameters(runner.model, "student")) != spec["tensor_sha256"] or frozen_digest(runner) != training["base_tensor_sha256"]:
            raise ValueError("DEPTH34_EVALUATION_MUTATED_WEIGHTS")
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
        "uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "device4_depth34.py"),
        "--config", str(config_path.resolve()), "--output-dir", str(child_dir.resolve()),
    ], cwd=ROOT, check=True)
    result = json.loads((child_dir / "result.json").read_text())
    child_pid = result["pid"] if stage == "evaluate" else result["training_pid"]
    if result["status"] != "completed" or child_pid == os.getpid():
        raise ValueError("DEPTH34_CHILD_PROCESS_NOT_COMPLETED")
    if stage == "evaluate" and result["training_pid"] != os.getpid():
        raise ValueError("DEPTH34_CHILD_EVALUATION_TRAINING_PID_CHANGED")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [depth34] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"run", "qualify", "learn", "evaluate"}:
        raise ValueError("DEPTH34_DISPATCH_SEAL_CHANGED")
    upstream, original = validate_design(design)
    corpus, audit = load_data(design, original)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("DEPTH34_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("DEPTH34_OUTPUT_DISPATCH_CHANGED")
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
                raise ValueError("DEPTH34_SOURCE_CHANGED_DURING_LEARNING")
            child_stage("evaluate", output, dispatch)
        else:
            result = qualify(design, upstream, original, corpus, audit, output)
            if source_hashes() != sources:
                raise ValueError("DEPTH34_SOURCE_CHANGED_DURING_QUALIFICATION")
            if result["status"] == "qualified" and dispatch["stage"] == "run":
                child_stage("learn", output, dispatch)
            elif result["status"] != "qualified":
                write_new(output / "result.json", {"status": result["status"], "learner_updates": 0, "teacher_optimizer_updates": 0, "qualified_teachers": []})
        if source_hashes() != sources:
            raise ValueError("DEPTH34_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_new(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("DEPTH34_FAILED")
        raise


if __name__ == "__main__":
    main()
