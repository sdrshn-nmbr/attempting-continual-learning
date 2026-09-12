import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path

import torch
from peft import PeftModel, set_peft_model_state_dict
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
from choice_contract import (
    ROOT,
    digest,
    file_hash,
    load_corpus,
    training_schedule,
    write_json,
)
from coverage_learner import (
    copy_bound_file,
    reference_training,
    teacher_proof,
    verify_local_teacher,
)
from coverage_teacher import coverage_gate, load_teacher, verify_qualified
from generation_consolidation import (
    cache_teacher_responses,
    learner_protocol,
    response_nll,
    target_sequences,
    training_probes,
    validate_cache_tokens,
    verify_checkpoint_files,
)
from generation_teacher import grade, measure, summarize, verify_generated_rows
from onpolicy303_contract import (
    METHODS,
    execution_identity,
    forward_kl,
    prefix_diagnostics,
    require_standalone,
    response_batch,
    retention_evaluate,
    retention_prediction,
    retention_rows,
    retention_summary,
    sample_response,
    source_hashes,
    trajectory_summary,
    validate_design,
)


def model_guard(model, label):
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    devices = sorted({str(tensor.device) for tensor in tensors.values()})
    mapping = getattr(model, "hf_device_map", None)
    bad_map = mapping is not None and any(str(device) not in {"0", "cuda", "cuda:0"} for device in mapping.values())
    if (
        torch.cuda.device_count() != 1 or not torch.version.hip or devices != ["cuda:0"]
        or any(hasattr(module, "_hf_hook") for module in model.modules()) or bad_map
        or any(t.is_floating_point() and (t.dtype != torch.float32 or not bool(torch.isfinite(t).all())) for t in tensors.values())
    ):
        raise ValueError(f"ONPOLICY_MODEL_PLACEMENT_PRECISION_FINITE: {label} devices={devices} map={mapping!r}")
    return {"devices": devices, "hf_device_map": mapping, "dtype": "float32", "finite": True,
            "visible_rocm_gpus": 1, "offload_hooks": False}


def frozen_versions(tensors):
    return [(name, tensor, tensor._version) for name, tensor in tensors.items()]


def check_frozen(snapshot, label):
    for name, tensor, version in snapshot:
        if tensor.requires_grad or tensor.grad is not None or tensor._version != version:
            raise ValueError(f"ONPOLICY_FIXED_WEIGHTS_CHANGED: {label}/{name}")


def new_optimizer(model, recipe):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    if optimizer.state:
        raise ValueError("ONPOLICY_NONEMPTY_INITIAL_OPTIMIZER")
    return optimizer, parameters


def optimizer_clocks(optimizer, parameters):
    clocks = [int(optimizer.state[p]["step"].item()) for p in parameters]
    return {"minimum": min(clocks), "maximum": max(clocks), "parameters_with_state": len(clocks)}


def generated_diagnostic(tokenizer, row, prompt, response, cap):
    body_ids = response[:-1] if response[-1] == tokenizer.eos_token_id else response
    body = tokenizer.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"prompt_token_ids": prompt, "token_ids": response, "body_text": body,
            "raw_text": tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            **grade(row, response, body, tokenizer.eos_token_id, tokenizer.all_special_ids, cap)}


def row_loss(student, expert, prompt, response, canonical, tokenizer, method, cap):
    device = next(student.parameters()).device
    ids, attention, mask = response_batch([prompt], [response], tokenizer.pad_token_id, tokenizer.eos_token_id, cap, device)
    if method == "cached_teacher_sft":
        loss = response_nll(student, prompt, response)
        with torch.no_grad():
            logits = student(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
        selected = logits[mask]
        diagnostics = {"student_argmax_token_ids": selected.argmax(-1).tolist(), "teacher_forward_performed": False}
    else:
        if expert is None or any(p.requires_grad for p in expert.parameters()):
            raise ValueError("ONPOLICY_FIXED_DETACHED_EXPERT_REQUIRED")
        with torch.no_grad():
            target_logits = expert(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
        logits = student(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
        loss, token_kl = forward_kl(logits, target_logits, mask)
        diagnostics = prefix_diagnostics(logits[mask], target_logits[mask], response, canonical, tokenizer.eos_token_id, token_kl[mask])
    if not bool(torch.isfinite(loss)):
        raise ValueError(f"ONPOLICY_NONFINITE_LOSS: {method}")
    diagnostics.update({"vocabulary_size": logits.shape[-1], "prediction_mask": mask[0].tolist(),
                        "loss_token_ids": response, "loss_tokens": int(mask.sum()), "row_loss": float(loss.detach())})
    return loss, diagnostics


def train_updates(student, expert, tokenizer, rows, prompts, targets, schedule, method, design, teacher_design, output):
    student.set_adapter("learner")
    student.eval()
    recipe = design["training"]
    optimizer, parameters = new_optimizer(student, recipe)
    base_snapshot = frozen_versions(base_tensors(student))
    expert_snapshot = [] if expert is None else frozen_versions({**dict(expert.named_parameters()), **dict(expert.named_buffers())})
    expert_hash = None if expert is None else tensor_hash(adapter_state(expert, "teacher"))
    generator = torch.Generator(device=next(student.parameters()).device).manual_seed(design["sampling"]["seed"])
    probes = training_probes({"train": rows}, recipe["probe_rows_per_task"])
    folder = output / method
    folder.mkdir()
    ledger, checkpoints, token_total = [], {}, 0
    with (folder / "trajectories.jsonl").open("x") as trajectories:
        for step, indices in enumerate(schedule, 1):
            if len(indices) != recipe["examples_per_update"]:
                raise ValueError("ONPOLICY_BATCH_EXPOSURE_MISMATCH")
            optimizer.zero_grad(set_to_none=True)
            losses, hashes, token_counts = [], [], []
            for microstep, index in enumerate(indices):
                sampling = None
                if method == "onpolicy_kl":
                    response, sampling = sample_response(student, prompts[index], tokenizer.eos_token_id, design["sampling"]["max_new_tokens"], generator)
                else:
                    response = list(targets[index])
                loss, diagnostics = row_loss(student, expert, prompts[index], response, targets[index], tokenizer, method, design["sampling"]["max_new_tokens"])
                (loss / len(indices)).backward()
                record = {
                    "step": step, "microstep": microstep, "row_index": index, "id": rows[index]["id"],
                    "task": rows[index]["task"], "row_sha256": digest(rows[index]), "method": method,
                    "generation": generated_diagnostic(tokenizer, rows[index], prompts[index], response, design["sampling"]["max_new_tokens"]),
                    "sampling": sampling, "diagnostics": diagnostics, "student_optimizer_clock_before_sampling": step - 1,
                    "canonical_teacher_response_sha256": digest(targets[index]),
                }
                trajectories.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                losses.append(float(loss.detach()))
                hashes.append(digest(record))
                token_counts.append(len(response))
                del loss
            trajectories.flush()
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, recipe["max_grad_norm"]))
            if not math.isfinite(norm):
                raise ValueError(f"ONPOLICY_NONFINITE_GRADIENT: {method}/{step}")
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise ValueError(f"ONPOLICY_NONFINITE_LEARNER: {method}/{step}")
            check_frozen(base_snapshot, "student_base")
            check_frozen(expert_snapshot, "expert")
            clocks = optimizer_clocks(optimizer, parameters)
            if clocks["minimum"] != step or clocks["maximum"] != step:
                raise ValueError("ONPOLICY_OPTIMIZER_CLOCK")
            entry = {"step": step, "ids": [rows[i]["id"] for i in indices], "trajectory_sha256": hashes,
                     "loss_tokens": sum(token_counts), "loss": sum(losses) / len(losses),
                     "gradient_norm": norm, "optimizer_clocks": clocks}
            ledger.append(entry)
            token_total += sum(token_counts)
            event(output, "onpolicy303_update", method=method, **entry)
            if step in recipe["checkpoint_updates"]:
                optimizer.zero_grad(set_to_none=True)
                destination = folder / f"checkpoint{step}"
                checkpoint = save_adapter(student, destination)
                torch.save(optimizer.state_dict(), destination / "optimizer.pt")
                records = measure(student, tokenizer, probes, teacher_design, output, f"{method}/train_only/{step}", detailed=False)
                write_json(destination / "train_probes.json", records)
                checkpoints[str(step)] = {"adapter": checkpoint, "optimizer_sha256": file_hash(destination / "optimizer.pt"),
                                          "train_probes_sha256": file_hash(destination / "train_probes.json"),
                                          "train_probe_summary": summarize(records)}
                if expert is not None and tensor_hash(adapter_state(expert, "teacher")) != expert_hash:
                    raise ValueError("ONPOLICY_TEACHER_ADAPTER_MUTATED")
    if len(ledger) != recipe["updates_per_arm"]:
        raise ValueError("ONPOLICY_INCOMPLETE_UPDATE_SCHEDULE")
    write_json(folder / "ledger.json", ledger)
    write_json(folder / "prefix_diagnostics.json", trajectory_summary(folder / "trajectories.jsonl", recipe["checkpoint_updates"]))
    return {"updates": len(ledger), "example_exposures": len(schedule) * recipe["examples_per_update"],
            "loss_token_exposures": token_total, "checkpoints": checkpoints,
            "ledger_sha256": file_hash(folder / "ledger.json"), "trajectories_sha256": file_hash(folder / "trajectories.jsonl"),
            "prefix_diagnostics_sha256": file_hash(folder / "prefix_diagnostics.json"),
            "optimizer_clocks": clocks, "teacher_optimizer_updates": 0,
            "teacher_tensor_sha256_before": expert_hash,
            "teacher_tensor_sha256_after": None if expert is None else tensor_hash(adapter_state(expert, "teacher")),
            "selected_checkpoint": recipe["updates_per_arm"], "cached_targets_sha256": digest(targets)}


def snapshot_control(design, output):
    control = design["coverage_control"]
    for name, checksum in control["files_sha256"].items():
        copy_bound_file(Path(control["directory"]) / name, output / ("control_" + name.replace("/", "_")), checksum)
    receipt = json.loads((output / "control_training.json").read_text())
    if not receipt["teacher_targets_all_equal_oracle"] or receipt["initial"]["tensor_sha256"] != control["initial_tensor_sha256"]:
        raise ValueError("ONPOLICY_CONTROL_TARGET_OR_INITIAL_IDENTITY")
    return receipt


def qualify_actual_teacher(design, context, corpus, audit, output):
    old, coverage, previous, teacher_design, choice = context
    folder = Path(old["qualified_teacher_dir"])
    if file_hash(folder / "qualification.json") != design["teacher"]["qualification_sha256"]:
        raise ValueError("ONPOLICY_TEACHER_QUALIFICATION_HASH")
    qualified, records = verify_qualified(folder, coverage, previous, teacher_design, choice, corpus, audit)
    if qualified["selected_checkpoint"] != 384 or qualified["teacher_tensor_sha256"] != design["teacher"]["tensor_sha256"]:
        raise ValueError("ONPOLICY_TEACHER_CHECKPOINT_IDENTITY")
    proof = teacher_proof(folder, qualified)
    for name, checksum in (("qualification.json", proof["qualification_sha256"]), ("qualification_panels.json", proof["qualification_panels_sha256"])):
        copy_bound_file(folder / name, output / ("teacher_" + name), checksum)
    verify_local_teacher(output, proof, old, coverage, teacher_design, corpus, audit)
    expert, tokenizer = load_teacher(choice, qualified["checkpoint"], output / "teacher_runtime")
    guard = model_guard(expert, "fixed_teacher")
    fresh, gates = {}, {}
    original = json.loads((output / "teacher_qualification_panels.json").read_text())
    for split in ("train", "validation"):
        fresh[split] = measure(expert, tokenizer, corpus[split], teacher_design, output, f"teacher_fresh/{split}", detailed=False)
        write_json(output / f"teacher_fresh_{split}.json", fresh[split])
        gate = coverage_gate(corpus[split], fresh[split], teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
        identical = all(a["generation"] == b["generation"] for a, b in zip(fresh[split], original[split]["generated"], strict=True))
        gates[split] = {"correct": sum(r["generation"]["correct"] for r in fresh[split]), "count": len(fresh[split]),
                        "coverage_passed": gate["passed"], "qualified_native_tokens_equal": identical}
        if not gate["passed"] or not identical or gates[split]["correct"] != len(corpus[split]):
            write_json(output / "teacher_rejected.json", {"gates": gates, "learner_updates": 0, "learner_loaded": False})
            raise ValueError(f"ONPOLICY_TEACHER_NATIVE_PREREQUISITE: {split}")
    check_base(expert, choice["source_base_tensor_sha256"])
    if tensor_hash(adapter_state(expert, "teacher")) != proof["teacher_tensor_sha256"]:
        raise ValueError("ONPOLICY_TEACHER_QUALIFICATION_MUTATED")
    native = {"gates": gates, "model_guard": guard, "learner_loaded": False, "learner_updates": 0,
              "teacher_optimizer_updates": 0, "off_answer_prefix_competence_certified": False}
    write_json(output / "teacher_fresh_qualification.json", native)
    return expert, tokenizer, proof, cache_teacher_responses(corpus["train"], records, proof)


def train(design, context, corpus, audit, dispatch, execution, output):
    old, coverage, previous, teacher_design, choice = context
    expert, teacher_tokenizer, proof, cache = qualify_actual_teacher(design, context, corpus, audit, output)
    control = snapshot_control(design, output)
    reference_training(old, coverage, previous, choice, corpus, audit)
    for name, checksum in (("training.json", old["reference_experiment"]["training_sha256"]), ("schedule.json", old["reference_experiment"]["schedule_sha256"])):
        copy_bound_file(Path(old["reference_experiment"]["training_dir"]) / name, output / ("reference_" + name), checksum)
    write_json(output / "teacher_train_cache.json", cache)
    if file_hash(output / "teacher_train_cache.json") != control["teacher_cache_sha256"]:
        raise ValueError("ONPOLICY_CONTROL_TEACHER_CACHE_CHANGED")
    method = dispatch["method"]
    if method == "cached_teacher_sft":
        del expert
        gc.collect()
        torch.cuda.empty_cache()
        expert = None
    protocol = learner_protocol(choice, design)
    base, tokenizer = load_base(protocol, output)
    student_guard = model_guard(base, "student_base")
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("ONPOLICY_TEACHER_STUDENT_VOCABULARY_MISMATCH")
    prompts = validate_cache_tokens(tokenizer, corpus["train"], cache, base.config.vocab_size)
    targets = target_sequences("teacher_output_sft", tokenizer, corpus["train"], cache)
    oracle_equal = targets == target_sequences("oracle_sft", tokenizer, corpus["train"], cache)
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    baseline = measure(base, tokenizer, probes, teacher_design, output, "untouched/train_only", detailed=False)
    write_json(output / "untouched_train_probes.json", baseline)
    student = make_learner(base, protocol)
    initial = save_adapter(student, output / "initial")
    if initial["tensor_sha256"] != control["initial"]["tensor_sha256"] or initial["tensor_sha256"] == proof["teacher_tensor_sha256"]:
        raise ValueError("ONPOLICY_FRESH_MATCHED_LEARNER_INITIALIZATION")
    initial_records = measure(student, tokenizer, probes, teacher_design, output, "initial/train_only", detailed=False)
    if initial_records != baseline:
        raise ValueError("ONPOLICY_ZERO_INITIAL_OUTPUT_PARITY")
    write_json(output / "initial_train_probes.json", initial_records)
    retention = retention_rows(design, corpus)
    write_json(output / "initial_retention.json", retention_evaluate(student, tokenizer, retention))
    schedule = training_schedule(protocol, corpus["train"])
    write_json(output / "schedule.json", [[corpus["train"][i]["id"] for i in batch] for batch in schedule])
    if file_hash(output / "schedule.json") != old["reference_experiment"]["schedule_sha256"]:
        raise ValueError("ONPOLICY_MATCHED_SCHEDULE_FAILED")
    arm = train_updates(student, expert, tokenizer, corpus["train"], prompts, targets, schedule, method, design, teacher_design, output)
    check_base(student, choice["source_base_tensor_sha256"])
    if expert is not None:
        check_base(expert, choice["source_base_tensor_sha256"])
    final = arm["checkpoints"][str(design["training"]["updates_per_arm"])]["adapter"]
    if final["tensor_sha256"] == initial["tensor_sha256"]:
        raise ValueError("ONPOLICY_NO_PERSISTENT_LEARNER_CHANGE")
    initial_tensors = load_file(str(Path(initial["path"]) / "adapter_model.safetensors"), device="cpu")
    final_tensors = adapter_state(student)
    changed = {name: int(torch.count_nonzero(value.detach().cpu() != initial_tensors[name])) for name, value in final_tensors.items()}
    write_json(output / "final_retention.json", retention_evaluate(student, tokenizer, retention))
    files = {str(p.relative_to(output)): file_hash(p) for p in output.glob("*.json") if p.name not in {"execution.json", "task.json", "config.json"}}
    receipt = {
        "status": "trained_evaluation_pending", "method": method, "pid": os.getpid(), "parent_pid": os.getppid(),
        "execution": execution, "design_sha256": digest(design), "source_sha256": source_hashes(design), "dataset": audit,
        "initial": initial, "arm": arm, "teacher_provenance": proof, "teacher_optimizer_updates": 0,
        "teacher_verified_native_before_learner_load": True, "teacher_targets_all_equal_oracle": oracle_equal,
        "student_weights_changed": True, "student_guard": student_guard,
        "changed_student_elements_by_tensor": changed,
        "base_tensor_sha256": choice["source_base_tensor_sha256"], "base_immutable": True,
        "immutable_weights": {
            "student_base_sha256_before_and_after": choice["source_base_tensor_sha256"],
            "teacher_base_sha256_before_and_after": choice["source_base_tensor_sha256"],
            "teacher_adapter_sha256_before_and_after": proof["teacher_tensor_sha256"],
            "teacher_versions_and_gradients_checked_every_kl_update": expert is not None,
            "sft_teacher_released_after_native_qualification": expert is None,
        },
        "trainable_parameters": choice["learner"]["expected_trainable_parameters"], "resident_learner_rank": 8,
        "teacher_resident_during_updates": expert is not None, "learner_initialized_from_teacher": False,
        "heldout_predictions_observed_during_training": False, "generic_retention_used_for_training_or_selection": False,
        "files_sha256": files,
        "coverage_sft_final_tensors_equal": final["tensor_sha256"] == design["coverage_control"]["final_tensor_sha256"],
        "new_process_persistence_evaluation_completed": False,
    }
    write_json(output / "training.json", receipt)
    return receipt


def verify_training(root, design, context, corpus, audit, method):
    root = Path(root)
    receipt = json.loads((root / "training.json").read_text())
    old, coverage, previous, teacher_design, choice = context
    if (
        receipt["status"] != "trained_evaluation_pending" or receipt["method"] != method
        or receipt["design_sha256"] != digest(design) or receipt["source_sha256"] != source_hashes(design)
        or receipt["dataset"] != audit or receipt["teacher_optimizer_updates"] != 0
        or not receipt["teacher_verified_native_before_learner_load"] or not receipt["student_weights_changed"]
        or receipt["learner_initialized_from_teacher"] or receipt["heldout_predictions_observed_during_training"]
        or receipt["initial"]["tensor_sha256"] != design["coverage_control"]["initial_tensor_sha256"]
        or receipt["resident_learner_rank"] != 8 or receipt["base_tensor_sha256"] != choice["source_base_tensor_sha256"]
    ):
        raise ValueError("ONPOLICY_TRAINING_RECEIPT_CHANGED")
    for name, checksum in receipt["files_sha256"].items():
        if file_hash(root / name) != checksum:
            raise ValueError(f"ONPOLICY_TRAINING_SNAPSHOT_CHANGED: {name}")
    proof = receipt["teacher_provenance"]
    if proof["qualification_sha256"] != design["teacher"]["qualification_sha256"]:
        raise ValueError("ONPOLICY_COPIED_TEACHER_IDENTITY")
    records = verify_local_teacher(root, proof, old, coverage, teacher_design, corpus, audit)
    expected_cache = cache_teacher_responses(corpus["train"], records, proof)
    if json.loads((root / "teacher_train_cache.json").read_text()) != expected_cache:
        raise ValueError("ONPOLICY_COPIED_CACHE_CHANGED")
    reference_training(old, coverage, previous, choice, corpus, audit, local_root=root)
    for split in ("train", "validation"):
        fresh = json.loads((root / f"teacher_fresh_{split}.json").read_text())
        verify_generated_rows(corpus[split], fresh, teacher_design, proof["eos_token_id"], proof["special_token_ids"])
        original = json.loads((root / "teacher_qualification_panels.json").read_text())[split]["generated"]
        if not all(r["generation"]["correct"] for r in fresh) or any(a["generation"] != b["generation"] for a, b in zip(fresh, original, strict=True)):
            raise ValueError("ONPOLICY_FRESH_TEACHER_PROOF_CHANGED")
    arm = receipt["arm"]
    recipe = design["training"]
    if arm["updates"] != recipe["updates_per_arm"] or arm["example_exposures"] != recipe["example_exposures_per_arm"] or arm["selected_checkpoint"] != recipe["updates_per_arm"]:
        raise ValueError("ONPOLICY_TRAINING_BUDGET_CHANGED")
    initial, final = receipt["initial"], arm["checkpoints"][str(recipe["updates_per_arm"])]["adapter"]
    for spec, directory in ((initial, root / "initial/learner"), (final, root / method / f"checkpoint{recipe['updates_per_arm']}" / "learner")):
        if Path(spec["path"]).resolve() != directory.resolve():
            raise ValueError("ONPOLICY_CHECKPOINT_OUTSIDE_OWN_RUN")
        verify_checkpoint_files(spec)
    if final["tensor_sha256"] == initial["tensor_sha256"]:
        raise ValueError("ONPOLICY_UNCHANGED_FINAL")
    verify_trajectory_ledger(root, receipt, design, corpus["train"], expected_cache)
    return receipt, expected_cache


def verify_trajectory_ledger(root, receipt, design, rows, cache):
    method, arm = receipt["method"], receipt["arm"]
    folder = root / method
    if file_hash(folder / "ledger.json") != arm["ledger_sha256"] or file_hash(folder / "trajectories.jsonl") != arm["trajectories_sha256"]:
        raise ValueError("ONPOLICY_TRAJECTORY_FILE_HASH")
    if (file_hash(folder / "prefix_diagnostics.json") != arm["prefix_diagnostics_sha256"]
        or json.loads((folder / "prefix_diagnostics.json").read_text()) != trajectory_summary(folder / "trajectories.jsonl", design["training"]["checkpoint_updates"])):
        raise ValueError("ONPOLICY_PREFIX_DIAGNOSTIC_BINDING")
    ledger = json.loads((folder / "ledger.json").read_text())
    schedule = json.loads((root / "schedule.json").read_text())
    trajectories = [json.loads(line) for line in (folder / "trajectories.jsonl").read_text().splitlines()]
    batch = design["training"]["examples_per_update"]
    if len(trajectories) != arm["example_exposures"] or len(ledger) != len(schedule):
        raise ValueError("ONPOLICY_TRAJECTORY_COUNTS")
    tokens = 0
    for step, (entry, ids) in enumerate(zip(ledger, schedule, strict=True), 1):
        block = trajectories[(step - 1) * batch:step * batch]
        if entry["step"] != step or entry["ids"] != ids or entry["trajectory_sha256"] != [digest(r) for r in block]:
            raise ValueError("ONPOLICY_SCHEDULE_LEDGER_BINDING")
        for microstep, (record, row_id) in enumerate(zip(block, ids, strict=True)):
            index = record["row_index"]
            row, cached = rows[index], cache["rows"][index]
            generation, diagnostics = record["generation"], record["diagnostics"]
            response, prompt = generation["token_ids"], generation["prompt_token_ids"]
            expected_mask = [False] * (len(prompt) - 1) + [True] * len(response)
            if (
                record["step"] != step or record["microstep"] != microstep or record["method"] != method
                or record["student_optimizer_clock_before_sampling"] != step - 1
                or record["id"] != row_id or row_id != row["id"] or record["row_sha256"] != digest(row)
                or prompt != cached["prompt_token_ids"] or not response or len(response) > design["sampling"]["max_new_tokens"]
                or cache["source"]["eos_token_id"] in response[:-1]
                or diagnostics["prediction_mask"] != expected_mask or diagnostics["loss_token_ids"] != response
                or diagnostics["loss_tokens"] != len(response)
                or record["canonical_teacher_response_sha256"] != digest(cached["response_token_ids"])
                or any(type(token) is not int or not 0 <= token < diagnostics["vocabulary_size"] for token in response)
            ):
                raise ValueError("ONPOLICY_TRAJECTORY_ROW_OR_MASK")
            graded = grade(row, response, generation["body_text"], cache["source"]["eos_token_id"], cache["source"]["special_token_ids"], design["sampling"]["max_new_tokens"])
            if any(generation[key] != value for key, value in graded.items()):
                raise ValueError("ONPOLICY_TRAJECTORY_SCORER_CHANGED")
            if method == "onpolicy_kl":
                sampling = record["sampling"]
                if (sampling["positive_support_count"] != diagnostics["vocabulary_size"]
                    or sampling["vocabulary_size"] != diagnostics["vocabulary_size"] or sampling["forced_eos"]
                    or sampling["temperature"] != 1 or len(sampling["minimum_vocabulary_probabilities"]) != len(response)
                    or any(not math.isfinite(p) or p <= 0 for p in sampling["minimum_vocabulary_probabilities"])):
                    raise ValueError("ONPOLICY_SAMPLING_SUPPORT_CHANGED")
            elif response != cached["response_token_ids"] or record["sampling"] is not None:
                raise ValueError("ONPOLICY_CACHED_TARGET_CHANGED")
            tokens += len(response)
        if entry["loss_tokens"] != sum(r["diagnostics"]["loss_tokens"] for r in block):
            raise ValueError("ONPOLICY_LOSS_TOKEN_ACCOUNTING")
    if tokens != arm["loss_token_exposures"]:
        raise ValueError("ONPOLICY_TOTAL_TOKEN_ACCOUNTING")


def load_saved_learner(base, spec):
    verify_checkpoint_files(spec)
    model = PeftModel.from_pretrained(base, spec["path"], adapter_name="learner", is_trainable=False, local_files_only=True)
    model.requires_grad_(False).eval()
    if set(model.peft_config) != {"learner"} or tensor_hash(adapter_state(model)) != spec["tensor_sha256"]:
        raise ValueError("ONPOLICY_TEACHER_ABSENT_RELOAD_IDENTITY")
    return model


def evaluate(design, context, corpus, audit, dispatch, execution, output):
    root = Path(dispatch["training_dir"])
    receipt, cache = verify_training(root, design, context, corpus, audit, dispatch["method"])
    source_execution = execution_identity(root, receipt["execution"]["task_id"], True)
    for key in ("task_id", "attempt_id", "task_sha256", "config_sha256", "source", "code_dir", "entrypoint", "config"):
        if source_execution[key] != receipt["execution"][key]:
            raise ValueError(f"ONPOLICY_SOURCE_COMPLETION_CHANGED: {key}")
    process = require_standalone(execution, receipt)
    _, _, _, teacher_design, choice = context
    base, tokenizer = load_base(learner_protocol(choice, design), output)
    guard = model_guard(base, "teacher_absent_evaluator")
    validate_cache_tokens(tokenizer, corpus["train"], cache, base.config.vocab_size)
    model = load_saved_learner(base, receipt["initial"])
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    retention = retention_rows(design, corpus)
    final_step = design["training"]["updates_per_arm"]
    final_info = receipt["arm"]["checkpoints"][str(final_step)]
    panels = {}
    for label, spec, native_path in (
        ("initial", receipt["initial"], root / "initial_train_probes.json"),
        ("final", final_info["adapter"], root / dispatch["method"] / f"checkpoint{final_step}/train_probes.json"),
    ):
        if label == "final":
            set_peft_model_state_dict(model, load_file(str(Path(spec["path"]) / "adapter_model.safetensors"), device="cpu"), adapter_name="learner")
            model.requires_grad_(False).eval()
        if tensor_hash(adapter_state(model)) != spec["tensor_sha256"] or set(model.peft_config) != {"learner"}:
            raise ValueError("ONPOLICY_FINAL_CHECKPOINT_LOAD")
        if label == "final" and file_hash(native_path) != final_info["train_probes_sha256"]:
            raise ValueError("ONPOLICY_FINAL_PROBES_HASH")
        native = measure(model, tokenizer, probes, teacher_design, output, f"reload/{label}/train_only", detailed=False)
        generic = retention_evaluate(model, tokenizer, retention)
        write_json(output / f"{label}_train_probes.json", native)
        write_json(output / f"{label}_retention.json", generic)
        if native != json.loads(native_path.read_text()):
            raise ValueError(f"ONPOLICY_NATIVE_RELOAD_PARITY: {label}")
        expected_generic = json.loads((root / f"{label}_retention.json").read_text())
        for row, record in zip(retention, expected_generic, strict=True):
            if record != retention_prediction(row, record["scores"]):
                raise ValueError("ONPOLICY_RETENTION_RECORD_BINDING")
        if generic != expected_generic:
            raise ValueError(f"ONPOLICY_RETENTION_RELOAD_PARITY: {label}")
        panels[label + "_retention"] = generic
    for split in design["evaluation"]["splits"]:
        panels[split] = measure(model, tokenizer, corpus[split], teacher_design, output, f"final/{split}", detailed=False)
        verify_generated_rows(corpus[split], panels[split], teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
    check_base(model, choice["source_base_tensor_sha256"])
    write_json(output / "panels.json", panels)
    control_panels = json.loads((root / "control_evaluation_panels.json").read_text())
    comparisons = {}
    for split in design["evaluation"]["splits"]:
        control_records = control_panels[split]["teacher_output_sft"]
        verify_generated_rows(corpus[split], control_records, teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
        comparisons[split] = {"final_correct": sum(r["generation"]["correct"] for r in panels[split]), "count": len(panels[split]),
                              "coverage_sft_correct": sum(r["generation"]["correct"] for r in control_records),
                              "exact_generated_records_equal_to_coverage_sft": panels[split] == control_records}
    result = {"status": "completed", "method": dispatch["method"], "pid": os.getpid(), "process_proof": process,
              "training_sha256": file_hash(root / "training.json"), "source_execution": source_execution,
              "execution": execution, "design_sha256": digest(design), "source_sha256": source_hashes(design),
              "dataset": audit, "panels_sha256": file_hash(output / "panels.json"), "model_guard": guard,
              "native": {split: summarize(panels[split]) for split in design["evaluation"]["splits"]},
              "generic_retention": {label: retention_summary(panels[label + "_retention"]) for label in ("initial", "final")},
              "coverage_control_comparison": comparisons,
              "coverage_sft_final_tensors_equal": receipt["coverage_sft_final_tensors_equal"],
              "teacher_model_loaded": False, "teacher_artifact_files_read_during_evaluation": False,
              "optimizer_updates": 0, "initial_and_final_exact_reload_parity": True,
              "new_process_persistence_evaluation_completed": True, "claim_boundary": design["claim_boundary"]}
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [onpolicy303] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"train", "evaluate"} or dispatch["method"] not in METHODS:
        raise ValueError("ONPOLICY_DISPATCH_DESIGN_CHANGED")
    context = validate_design(design)
    corpus, audit = load_corpus(context[-1])
    retention_rows(design, corpus)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(p.name not in allowed for p in output.iterdir()):
        raise FileExistsError("ONPOLICY_OUTPUT_ALREADY_USED")
    sources = source_hashes(design)
    write_json(output / "seal.json", {"design": design, "source_sha256": sources, "dataset": audit,
                                       "dispatch": dispatch, "pid": os.getpid(), "parent_pid": os.getppid()})
    if args.validate_only:
        write_json(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        execution = execution_identity(output, dispatch["task_id"], False, dispatch)
        if Path(execution["code_dir"]).resolve() != ROOT or execution["entrypoint"] != "onpolicy303.py":
            raise ValueError("ONPOLICY_EXECUTED_SOURCE_DIRECTORY")
        gate = execution_identity(design["runtime_dependency"]["directory"], design["runtime_dependency"]["task_id"], True)
        write_json(output / "runtime_dependency.json", {"execution": gate, "numerical_boundary": design["runtime_dependency"]["numerical_boundary"]})
        if dispatch["stage"] == "train":
            train(design, context, corpus, audit, dispatch, execution, output)
        else:
            evaluate(design, context, corpus, audit, dispatch, execution, output)
        if source_hashes(design) != sources:
            raise ValueError("ONPOLICY_SOURCE_CHANGED_DURING_EXECUTION")
    except Exception as error:
        event(output, "onpolicy303_failed", exception=type(error).__name__, detail=str(error))
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
