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
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn import functional

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
    TASKS,
    digest,
    file_hash,
    load_corpus,
    training_schedule,
    verify_qualification,
    write_json,
)
from generation_teacher import (
    eos_encoded,
    generation_gate,
    measure,
    select_checkpoint,
    summarize,
    verify_baseline,
    verify_generated_rows,
)
from generation_teacher import source_hashes as teacher_source_hashes
from generation_teacher import validate_design as validate_teacher_design

CONTRACT = "sequence303_generated_answer_persistent_consolidation_20260912"
METHODS = ("teacher_output_sft", "oracle_sft")
SOURCE_FILES = (
    "generation_consolidation.py",
    "generation_teacher.py",
    "choice_consolidation.py",
    "choice_contract.py",
    "requirements.txt",
)


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    if design["contract"] != CONTRACT:
        raise ValueError("GEN_CONSOLIDATE_UNKNOWN_CONTRACT")
    if design["training"] != {
        "methods": list(METHODS), "epochs": 4, "examples_per_update": 4,
        "updates_per_arm": 384, "example_exposures_per_arm": 1536,
        "learning_rate": 0.0003, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "initialization_seed": 91230417, "order_seed": 91230431,
        "checkpoint_updates": [32, 128, 384], "probe_rows_per_task": 8,
        "selection": "fixed_final384",
    }:
        raise ValueError("GEN_CONSOLIDATE_FIXED_TRAINING_CONTRACT_CHANGED")
    if design["evaluation"] != {
        "splits": ["test", "unused288"], "teacher_model_loaded": False,
        "require_new_process": True, "whole_answer_and_native_eos": True,
    }:
        raise ValueError("GEN_CONSOLIDATE_EVALUATION_CONTRACT_CHANGED")
    if set(design["teacher_sources"]) != {"diagnosed_source", "whole_answer_trained"}:
        raise ValueError("GEN_CONSOLIDATE_EXPLICIT_TEACHER_CHOICES_REQUIRED")
    if len({item["qualification_dir"] for item in design["teacher_sources"].values()}) != 2:
        raise ValueError("GEN_CONSOLIDATE_AMBIGUOUS_TEACHER_SOURCE")
    spec = design["teacher_design"]
    path = ROOT / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError("GEN_CONSOLIDATE_TEACHER_DESIGN_CHANGED")
    teacher_design = json.loads(path.read_text())
    choice_protocol = validate_teacher_design(teacher_design)
    if teacher_source_hashes() != design["qualified_teacher_source_dependencies"]:
        raise ValueError("GEN_CONSOLIDATE_ARCHIVED_QUALIFICATION_CODE_CHANGED")
    return teacher_design, choice_protocol


def learner_protocol(choice_protocol, design):
    return {
        **choice_protocol,
        "training": {
            **choice_protocol["training"],
            "initialization_seed": design["training"]["initialization_seed"],
            "order_seed": design["training"]["order_seed"],
        },
    }


def training_probes(corpus, count):
    return [row for task in TASKS for row in [r for r in corpus["train"] if r["task"] == task][:count]]


def verify_checkpoint_files(spec):
    if set(spec["files"]) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("GEN_CONSOLIDATE_CHECKPOINT_FILE_SET")
    for name, expected in spec["files"].items():
        if file_hash(Path(spec["path"]) / name) != expected:
            raise ValueError(f"GEN_CONSOLIDATE_CHECKPOINT_CHANGED: {name}")


def verify_trained_teacher_history(spec, qualified, teacher_design, choice_protocol, corpus, audit):
    root = Path(spec["training_dir"])
    training = json.loads((root / "training.json").read_text())
    recipe = teacher_design["training"]
    schedule = training_schedule(choice_protocol, corpus["train"])
    expected_ids = [[corpus["train"][i]["id"] for i in batch] for batch in schedule]
    if (
        training["status"] != "trained_qualification_pending"
        or training["design_sha256"] != digest(teacher_design)
        or training["source_sha256"] != teacher_source_hashes()
        or training["dataset"] != audit
        or training["pid"] != qualified["training_pid"]
        or training["pid"] == qualified["pid"]
        or training["source_teacher_sha256"] != choice_protocol["teacher"]["tensor_sha256"]
        or training["updates"] != recipe["updates"]
        or training["example_exposures"] != recipe["updates"] * recipe["examples_per_update"]
        or training["validation_predictions_observed_during_training"]
        or training["learner_updates"] != 0
        or training["trainable_parameters"] != choice_protocol["learner"]["expected_trainable_parameters"]
        or [item["ids"] for item in training["ledger"]] != expected_ids
    ):
        raise ValueError("GEN_CONSOLIDATE_TEACHER_TRAINING_PROVENANCE")
    baseline = Path(spec["diagnosis_dir"]) / "qualification.json"
    if file_hash(baseline) != training["diagnosis_sha256"]:
        raise ValueError("GEN_CONSOLIDATE_TEACHER_DIAGNOSIS_CHANGED")
    verify_baseline(spec["diagnosis_dir"], teacher_design, choice_protocol, corpus, audit)
    probes = training_probes(corpus, recipe["train_probes_per_task"])
    if set(training["checkpoints"]) != {str(step) for step in recipe["checkpoints"]}:
        raise ValueError("GEN_CONSOLIDATE_TEACHER_CHECKPOINTS_CHANGED")
    for checkpoint in training["checkpoints"].values():
        folder = Path(checkpoint["path"]).parent
        if (
            file_hash(folder / "train_probes.json") != checkpoint["train_probes_sha256"]
            or file_hash(folder / "optimizer.pt") != checkpoint["optimizer_sha256"]
        ):
            raise ValueError("GEN_CONSOLIDATE_TEACHER_TRAIN_PROBES_CHANGED")
        records = json.loads((folder / "train_probes.json").read_text())
        verify_generated_rows(probes, records, teacher_design, training["eos_token_id"], training["special_token_ids"])
        if summarize(records) != checkpoint["train_probe_summary"]:
            raise ValueError("GEN_CONSOLIDATE_TEACHER_PROBE_SUMMARY_CHANGED")
    selected = select_checkpoint(training["checkpoints"], teacher_design)
    checkpoint = training["checkpoints"][str(selected)]
    if (
        selected != training["selected_checkpoint"]
        or training["selection"] != recipe["selection"]
        or checkpoint != qualified["checkpoint"]
        or checkpoint["tensor_sha256"] != qualified["teacher_tensor_sha256"]
        or training["eos_token_id"] != qualified["eos_token_id"]
        or training["special_token_ids"] != qualified["special_token_ids"]
    ):
        raise ValueError("GEN_CONSOLIDATE_TEACHER_SELECTION_CHANGED")
    return {"training_sha256": file_hash(root / "training.json"), "diagnosis_sha256": file_hash(baseline), "selected_checkpoint": selected}


def verify_teacher_receipt(design, role, teacher_design, choice_protocol, corpus, audit):
    if role not in design["teacher_sources"]:
        raise ValueError("GEN_CONSOLIDATE_UNKNOWN_EXPLICIT_TEACHER")
    verify_qualification(design["choice_qualification_dir"], choice_protocol, corpus, audit)
    spec = design["teacher_sources"][role]
    root = Path(spec["qualification_dir"])
    qualified = json.loads((root / "qualification.json").read_text())
    if qualified["status"] != "qualified":
        raise ValueError("GEN_CONSOLIDATE_TEACHER_NOT_QUALIFIED_NO_LEARNER_UPDATES")
    if (
        qualified["design_sha256"] != digest(teacher_design)
        or qualified["source_sha256"] != teacher_source_hashes()
        or qualified["dataset"] != audit
        or qualified["learner_updates"] != 0
        or file_hash(root / "qualification_panels.json") != qualified["panels_sha256"]
    ):
        raise ValueError("GEN_CONSOLIDATE_TEACHER_QUALIFICATION_CHANGED")
    panels = json.loads((root / "qualification_panels.json").read_text())
    gate = generation_gate(teacher_design, corpus, panels, qualified["eos_token_id"], qualified["special_token_ids"])
    if not gate["passed"] or gate != qualified["gate"]:
        raise ValueError("GEN_CONSOLIDATE_TEACHER_GATE_FAILED_NO_LEARNER_UPDATES")
    if role == "diagnosed_source":
        if (
            qualified["training_pid"] is not None
            or qualified["teacher_tensor_sha256"] != choice_protocol["teacher"]["tensor_sha256"]
            or qualified["checkpoint"] != choice_protocol["teacher"]
        ):
            raise ValueError("GEN_CONSOLIDATE_WRONG_DIAGNOSED_SOURCE")
        provenance = {"kind": "original_successful_source_zero_eos_updates"}
    else:
        provenance = verify_trained_teacher_history(spec, qualified, teacher_design, choice_protocol, corpus, audit)
    verify_checkpoint_files(qualified["checkpoint"])
    proof = {
        "teacher_role": role, "qualification_dir": str(root),
        "qualification_sha256": file_hash(root / "qualification.json"),
        "qualification_panels_sha256": qualified["panels_sha256"],
        "teacher_tensor_sha256": qualified["teacher_tensor_sha256"],
        "source_sha256": qualified["source_sha256"],
        "eos_token_id": qualified["eos_token_id"], "special_token_ids": qualified["special_token_ids"],
        "gate": gate, "history": provenance,
    }
    return proof, panels["train"]["generated"]


def cache_teacher_responses(rows, records, proof):
    if len(rows) != len(records):
        raise ValueError("GEN_CONSOLIDATE_INCOMPLETE_TEACHER_CACHE")
    cached = []
    for row, record in zip(rows, records, strict=True):
        if record["id"] != row["id"] or record["row_sha256"] != digest(row):
            raise ValueError("GEN_CONSOLIDATE_TEACHER_CACHE_ROW_BINDING")
        generated = record["generation"]
        response = generated["token_ids"]
        if not response or len(response) > 16 or any(type(token) is not int or token < 0 for token in response):
            raise ValueError("GEN_CONSOLIDATE_EMPTY_OR_INVALID_TEACHER_RESPONSE")
        cached.append({
            "id": row["id"], "task": row["task"], "row_sha256": digest(row),
            "prompt_token_ids": list(generated["prompt_token_ids"]),
            "response_token_ids": list(response), "body_text": generated["body_text"],
            "raw_text": generated["raw_text"], "original_generation_sha256": digest(generated),
        })
    return {
        "source": proof, "split": "train", "rows_sha256": digest(rows), "rows": cached,
        "policy": "Every cached TRAIN generation copied verbatim. No filtering, gold replacement, decoding normalization, or EOS insertion.",
    }


def validate_cache_tokens(tokenizer, rows, cache, vocabulary_size):
    proof = cache["source"]
    if (
        tokenizer.eos_token_id != proof["eos_token_id"]
        or set(tokenizer.all_special_ids) != set(proof["special_token_ids"])
        or cache["rows_sha256"] != digest(rows)
        or cache["split"] != "train"
        or len(cache["rows"]) != len(rows)
    ):
        raise ValueError("GEN_CONSOLIDATE_TOKENIZER_OR_CACHE_IDENTITY")
    prompts = []
    for row, saved in zip(rows, cache["rows"], strict=True):
        prompt = tokenizer(row["prompt"], add_special_tokens=True).input_ids
        response = saved["response_token_ids"]
        if (
            saved["id"] != row["id"] or saved["row_sha256"] != digest(row)
            or not prompt or len(prompt) > 768 or prompt != saved["prompt_token_ids"]
            or not response or len(response) > 16
            or any(type(token) is not int or not 0 <= token < vocabulary_size for token in response)
        ):
            raise ValueError("GEN_CONSOLIDATE_CACHED_TOKEN_BOUNDARY")
        body_ids = response[:-1] if response[-1] == tokenizer.eos_token_id else response
        if (
            tokenizer.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != saved["body_text"]
            or tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False) != saved["raw_text"]
        ):
            raise ValueError("GEN_CONSOLIDATE_CACHED_DECODE_CHANGED")
        prompts.append(prompt)
    return prompts


def target_sequences(method, tokenizer, rows, cache):
    if method == "teacher_output_sft":
        return [list(row["response_token_ids"]) for row in cache["rows"]]
    if method == "oracle_sft":
        encoded = [eos_encoded(tokenizer, row) for row in rows]
        return [sequence[-length:] for sequence, length, _ in encoded]
    raise ValueError("GEN_CONSOLIDATE_UNKNOWN_METHOD")


def response_nll(model, prompt, response):
    if not prompt or not response:
        raise ValueError("GEN_CONSOLIDATE_EMPTY_SEQUENCE")
    device = next(model.parameters()).device
    if torch.is_autocast_enabled(device.type):
        raise ValueError("GEN_CONSOLIDATE_AUTOCAST_DISALLOWED")
    ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, len(prompt) - 1:-1]
    if logits.dtype != torch.float32 or logits.shape[0] != len(response):
        raise ValueError("GEN_CONSOLIDATE_RESPONSE_LOSS_BOUNDARY")
    return functional.cross_entropy(logits, ids[0, len(prompt):])


def train_arm(model, tokenizer, rows, prompts, targets, schedule, method, design, teacher_design, output):
    model.set_adapter("learner")
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    recipe = design["training"]
    frozen = [(p, p._version) for p in base_tensors(model).values()]
    optimizer = torch.optim.AdamW(params, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    ledger, checkpoints = [], {}
    folder = output / method
    folder.mkdir()
    probes = [row for task in TASKS for row in [r for r in rows if r["task"] == task][:recipe["probe_rows_per_task"]]]
    for step, indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in indices:
            loss = response_nll(model, prompts[index], targets[index])
            if not torch.isfinite(loss):
                raise ValueError(f"GEN_CONSOLIDATE_NONFINITE_LOSS: {method}/{step}")
            (loss / len(indices)).backward()
            losses.append(float(loss.detach()))
            del loss
        norm = float(torch.nn.utils.clip_grad_norm_(params, recipe["max_grad_norm"]))
        if not math.isfinite(norm):
            raise ValueError(f"GEN_CONSOLIDATE_NONFINITE_GRADIENT: {method}/{step}")
        optimizer.step()
        if any(p.requires_grad or p.grad is not None or p._version != version for p, version in frozen):
            raise ValueError("GEN_CONSOLIDATE_FROZEN_BASE_MUTATED_DURING_TRAINING")
        record = {
            "step": step, "ids": [rows[i]["id"] for i in indices],
            "response_sha256": [digest(targets[i]) for i in indices],
            "loss_tokens": sum(len(targets[i]) for i in indices),
            "loss": sum(losses) / len(losses), "gradient_norm": norm,
        }
        ledger.append(record)
        event(output, "generated_learner_update", method=method, **record)
        if step in recipe["checkpoint_updates"]:
            optimizer.zero_grad(set_to_none=True)
            destination = folder / f"checkpoint{step}"
            checkpoint = save_adapter(model, destination)
            torch.save(optimizer.state_dict(), destination / "optimizer.pt")
            records = measure(model, tokenizer, probes, teacher_design, output, f"{method}/train_only/{step}", detailed=False)
            write_json(destination / "train_probes.json", records)
            checkpoints[str(step)] = {
                "adapter": checkpoint, "optimizer_sha256": file_hash(destination / "optimizer.pt"),
                "train_probes_sha256": file_hash(destination / "train_probes.json"),
                "train_probe_summary": summarize(records),
            }
    write_json(folder / "ledger.json", ledger)
    return {
        "updates": len(ledger), "example_exposures": sum(len(row["ids"]) for row in ledger),
        "loss_token_exposures": sum(row["loss_tokens"] for row in ledger),
        "targets_sha256": digest(targets), "ledger_sha256": file_hash(folder / "ledger.json"),
        "checkpoints": checkpoints, "selected_checkpoint": recipe["updates_per_arm"],
        "teacher_model_loaded": False,
    }


def train(design, dispatch, teacher_design, choice_protocol, corpus, audit, output):
    proof, teacher_records = verify_teacher_receipt(design, dispatch["teacher_role"], teacher_design, choice_protocol, corpus, audit)
    cache = cache_teacher_responses(corpus["train"], teacher_records, proof)
    write_json(output / "teacher_train_cache.json", cache)
    protocol = learner_protocol(choice_protocol, design)
    base, tokenizer = load_base(protocol, output)
    prompts = validate_cache_tokens(tokenizer, corpus["train"], cache, base.get_input_embeddings().num_embeddings)
    targets = {method: target_sequences(method, tokenizer, corpus["train"], cache) for method in METHODS}
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    baseline = measure(base, tokenizer, probes, teacher_design, output, "untouched/train_only", detailed=False)
    write_json(output / "untouched_train_probes.json", baseline)
    model = make_learner(base, protocol)
    if set(model.peft_config) != {"learner"}:
        raise ValueError("GEN_CONSOLIDATE_EXTRA_ADAPTER_PRESENT")
    initial = {name: value.detach().clone() for name, value in adapter_state(model).items()}
    initial_hash = tensor_hash(initial)
    if initial_hash == proof["teacher_tensor_sha256"]:
        raise ValueError("GEN_CONSOLIDATE_TEACHER_WEIGHTS_IN_LEARNER")
    initial_checkpoint = save_adapter(model, output / "initial")
    initial_records = measure(model, tokenizer, probes, teacher_design, output, "initial/train_only", detailed=False)
    if initial_records != baseline:
        raise ValueError("GEN_CONSOLIDATE_INITIAL_ZERO_OUTPUT_PARITY")
    write_json(output / "initial_train_probes.json", initial_records)
    schedule = training_schedule(protocol, corpus["train"])
    write_json(output / "schedule.json", [[corpus["train"][i]["id"] for i in batch] for batch in schedule])
    arms = {}
    for method in METHODS:
        set_peft_model_state_dict(model, initial, adapter_name="learner")
        if tensor_hash(adapter_state(model)) != initial_hash:
            raise ValueError("GEN_CONSOLIDATE_MATCHED_LEARNER_INITIALIZATION")
        arms[method] = train_arm(model, tokenizer, corpus["train"], prompts, targets[method], schedule, method, design, teacher_design, output)
        check_base(model, choice_protocol["source_base_tensor_sha256"])
        if arms[method]["checkpoints"][str(design["training"]["updates_per_arm"])]["adapter"]["tensor_sha256"] == initial_hash:
            raise ValueError(f"GEN_CONSOLIDATE_LEARNER_UNCHANGED: {method}")
    result = {
        "status": "trained_evaluation_pending", "pid": os.getpid(),
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "teacher_role": dispatch["teacher_role"], "teacher_provenance": proof,
        "teacher_cache_sha256": file_hash(output / "teacher_train_cache.json"),
        "teacher_targets_equal_oracle_count": sum(a == b for a, b in zip(targets[METHODS[0]], targets[METHODS[1]], strict=True)),
        "teacher_targets_all_equal_oracle": targets[METHODS[0]] == targets[METHODS[1]],
        "initial": initial_checkpoint, "arms": arms,
        "base_tensor_sha256": choice_protocol["source_base_tensor_sha256"],
        "trainable_parameters": choice_protocol["learner"]["expected_trainable_parameters"],
        "resident_learner_rank": 8, "teacher_model_loaded": False,
        "heldout_predictions_observed": False,
        "files_sha256": {name: file_hash(output / name) for name in ("schedule.json", "initial_train_probes.json", "untouched_train_probes.json")},
    }
    write_json(output / "training.json", result)
    event(output, "generated_learner_training_finished", updates_per_arm=design["training"]["updates_per_arm"], teacher_model_loaded=False)
    return result


def verify_training(root, design, teacher_design, choice_protocol, corpus, audit):
    root = Path(root)
    training = json.loads((root / "training.json").read_text())
    if (
        training["status"] != "trained_evaluation_pending"
        or training["pid"] == os.getpid()
        or training["design_sha256"] != digest(design)
        or training["source_sha256"] != source_hashes()
        or training["dataset"] != audit
        or training["teacher_model_loaded"] or training["heldout_predictions_observed"]
        or set(training["arms"]) != set(METHODS)
        or training["base_tensor_sha256"] != choice_protocol["source_base_tensor_sha256"]
        or training["trainable_parameters"] != choice_protocol["learner"]["expected_trainable_parameters"]
        or training["resident_learner_rank"] != 8
        or file_hash(root / "teacher_train_cache.json") != training["teacher_cache_sha256"]
    ):
        raise ValueError("GEN_CONSOLIDATE_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
    for name, checksum in training["files_sha256"].items():
        if file_hash(root / name) != checksum:
            raise ValueError(f"GEN_CONSOLIDATE_TRAINING_FILE_CHANGED: {name}")
    protocol = learner_protocol(choice_protocol, design)
    schedule = training_schedule(protocol, corpus["train"])
    expected = [[corpus["train"][i]["id"] for i in indices] for indices in schedule]
    if json.loads((root / "schedule.json").read_text()) != expected:
        raise ValueError("GEN_CONSOLIDATE_TRAINING_EXPOSURE_ORDER_CHANGED")
    cache = json.loads((root / "teacher_train_cache.json").read_text())
    if cache["source"] != training["teacher_provenance"] or cache["source"]["teacher_role"] != training["teacher_role"]:
        raise ValueError("GEN_CONSOLIDATE_TEACHER_PROVENANCE_CHANGED")
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    for name in ("untouched_train_probes.json", "initial_train_probes.json"):
        records = json.loads((root / name).read_text())
        verify_generated_rows(probes, records, teacher_design, cache["source"]["eos_token_id"], cache["source"]["special_token_ids"])
    if json.loads((root / "untouched_train_probes.json").read_text()) != json.loads((root / "initial_train_probes.json").read_text()):
        raise ValueError("GEN_CONSOLIDATE_RECORDED_INITIAL_PARITY_CHANGED")
    for method, arm in training["arms"].items():
        recipe = design["training"]
        folder = root / method
        if (
            arm["updates"] != recipe["updates_per_arm"]
            or arm["example_exposures"] != recipe["example_exposures_per_arm"]
            or arm["selected_checkpoint"] != recipe["updates_per_arm"]
            or arm["teacher_model_loaded"]
            or set(arm["checkpoints"]) != {str(step) for step in recipe["checkpoint_updates"]}
            or file_hash(folder / "ledger.json") != arm["ledger_sha256"]
        ):
            raise ValueError("GEN_CONSOLIDATE_ARM_BUDGET_OR_CHECKPOINTS_CHANGED")
        ledger = json.loads((folder / "ledger.json").read_text())
        if (
            [item["ids"] for item in ledger] != expected
            or [item["step"] for item in ledger] != list(range(1, recipe["updates_per_arm"] + 1))
            or sum(item["loss_tokens"] for item in ledger) != arm["loss_token_exposures"]
        ):
            raise ValueError("GEN_CONSOLIDATE_ARM_EXPOSURE_LEDGER_CHANGED")
        for step, checkpoint in arm["checkpoints"].items():
            destination = folder / f"checkpoint{step}"
            if (
                file_hash(destination / "optimizer.pt") != checkpoint["optimizer_sha256"]
                or file_hash(destination / "train_probes.json") != checkpoint["train_probes_sha256"]
            ):
                raise ValueError("GEN_CONSOLIDATE_CHECKPOINT_PROBES_CHANGED")
            records = json.loads((destination / "train_probes.json").read_text())
            verify_generated_rows(probes, records, teacher_design, cache["source"]["eos_token_id"], cache["source"]["special_token_ids"])
            if summarize(records) != checkpoint["train_probe_summary"]:
                raise ValueError("GEN_CONSOLIDATE_CHECKPOINT_PROBE_SUMMARY_CHANGED")
    return training, cache


def verify_target_ledgers(root, training, tokenizer, corpus, cache, design, choice_protocol):
    rows = corpus["train"]
    schedule = training_schedule(learner_protocol(choice_protocol, design), rows)
    targets = {method: target_sequences(method, tokenizer, rows, cache) for method in METHODS}
    for method in METHODS:
        ledger = json.loads((Path(root) / method / "ledger.json").read_text())
        if digest(targets[method]) != training["arms"][method]["targets_sha256"]:
            raise ValueError("GEN_CONSOLIDATE_SUPERVISION_CHANGED")
        for indices, record in zip(schedule, ledger, strict=True):
            if (
                record["response_sha256"] != [digest(targets[method][i]) for i in indices]
                or record["loss_tokens"] != sum(len(targets[method][i]) for i in indices)
            ):
                raise ValueError("GEN_CONSOLIDATE_TARGET_EXPOSURE_CHANGED")
    same = sum(a == b for a, b in zip(targets[METHODS[0]], targets[METHODS[1]], strict=True))
    if (
        same != training["teacher_targets_equal_oracle_count"]
        or (same == len(rows)) != training["teacher_targets_all_equal_oracle"]
    ):
        raise ValueError("GEN_CONSOLIDATE_TARGET_EQUALITY_CHANGED")


def evaluate_learners(design, dispatch, teacher_design, choice_protocol, corpus, audit, output):
    root = Path(dispatch["training_dir"])
    training, cache = verify_training(root, design, teacher_design, choice_protocol, corpus, audit)
    protocol = learner_protocol(choice_protocol, design)
    base, tokenizer = load_base(protocol, output)
    validate_cache_tokens(tokenizer, corpus["train"], cache, base.get_input_embeddings().num_embeddings)
    verify_target_ledgers(root, training, tokenizer, corpus, cache, design, choice_protocol)
    panels = {split: {"untouched": measure(base, tokenizer, corpus[split], teacher_design, output, f"untouched/{split}", detailed=False)} for split in design["evaluation"]["splits"]}
    verify_checkpoint_files(training["initial"])
    model = PeftModel.from_pretrained(base, training["initial"]["path"], adapter_name="learner", is_trainable=False, local_files_only=True).requires_grad_(False).eval()
    probes = training_probes(corpus, design["training"]["probe_rows_per_task"])
    conditions = {"initial": training["initial"], **{method: arm["checkpoints"][str(arm["selected_checkpoint"])]["adapter"] for method, arm in training["arms"].items()}}
    for condition, checkpoint in conditions.items():
        verify_checkpoint_files(checkpoint)
        state = load_file(str(Path(checkpoint["path"]) / "adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(model, state, adapter_name="learner")
        model.set_adapter("learner", inference_mode=True)
        model.requires_grad_(False).eval()
        if (
            set(model.peft_config) != {"learner"} or model.active_adapters != ["learner"]
            or tensor_hash(adapter_state(model)) != checkpoint["tensor_sha256"]
        ):
            raise ValueError("GEN_CONSOLIDATE_SAVED_LEARNER_IDENTITY_OR_EXTRA_ADAPTER")
        actual = measure(model, tokenizer, probes, teacher_design, output, f"{condition}/reload_probe", detailed=False)
        saved = root / "initial_train_probes.json" if condition == "initial" else root / condition / f"checkpoint{design['training']['updates_per_arm']}" / "train_probes.json"
        if actual != json.loads(saved.read_text()):
            raise ValueError(f"GEN_CONSOLIDATE_RELOAD_GENERATION_PARITY: {condition}")
        if condition != "initial":
            for split in design["evaluation"]["splits"]:
                panels[split][condition] = measure(model, tokenizer, corpus[split], teacher_design, output, f"{condition}/{split}", detailed=False)
        check_base(model, choice_protocol["source_base_tensor_sha256"])
    comparison = {}
    for split, conditions in panels.items():
        for records in conditions.values():
            verify_generated_rows(corpus[split], records, teacher_design, tokenizer.eos_token_id, tokenizer.all_special_ids)
        comparison[split] = {name: summarize(records) for name, records in conditions.items()}
        for left, right in ((METHODS[0], "untouched"), (METHODS[1], "untouched"), METHODS):
            differences = [int(a["generation"]["correct"]) - int(b["generation"]["correct"]) for a, b in zip(conditions[left], conditions[right], strict=True)]
            comparison[split][f"{left}_minus_{right}"] = sum(differences) / len(differences)
    write_json(output / "panels.json", panels)
    result = {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"],
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "training_sha256": file_hash(root / "training.json"), "panels_sha256": file_hash(output / "panels.json"),
        "comparisons": comparison, "teacher_role": training["teacher_role"],
        "teacher_targets_all_equal_oracle": training["teacher_targets_all_equal_oracle"],
        "teacher_model_loaded": False, "teacher_artifact_files_read_during_evaluation": False,
        "learner_reload_generation_parity": True, "resident_active_adapters": 1, "optimizer_updates": 0,
        "claim_boundary": design["claim_boundary"],
    }
    write_json(output / "result.json", result)
    return result


def evaluate_in_new_process(output, dispatch):
    child_config = {
        "stage": "evaluate", "protocol": dispatch["protocol"],
        "protocol_sha256": dispatch["protocol_sha256"], "training_dir": str(output.resolve()),
    }
    write_json(output / "evaluation_config.json", child_config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run([
        "uv", "run", "--no-project", "--python", sys.executable, "python",
        str(ROOT / "generation_consolidation.py"), "--config", str((output / "evaluation_config.json").resolve()),
        "--output-dir", str((output / "evaluation").resolve()),
    ], check=True, cwd=ROOT)
    result_path = output / "evaluation/result.json"
    result = json.loads(result_path.read_text())
    if result["status"] != "completed" or result["pid"] == os.getpid() or result["training_pid"] != os.getpid():
        raise ValueError("GEN_CONSOLIDATE_CHILD_EVALUATOR_IDENTITY")
    write_json(output / "result.json", {
        "status": "completed", "training_pid": os.getpid(), "evaluation_pid": result["pid"],
        "training_sha256": file_hash(output / "training.json"),
        "evaluation_result": str(result_path), "evaluation_result_sha256": file_hash(result_path),
        "comparisons": result["comparisons"], "teacher_model_loaded": False,
        "teacher_targets_all_equal_oracle": result["teacher_targets_all_equal_oracle"],
        "claim_boundary": result["claim_boundary"],
    })


def prepare(output, dispatch, design, audit):
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("GEN_CONSOLIDATE_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("GEN_CONSOLIDATE_DISPATCH_CONFIG_MISMATCH")
    write_json(output / "seal.json", {"design": design, "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit, "dispatch": dispatch, "pid": os.getpid()})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [generated-consolidation] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"]:
        raise ValueError("GEN_CONSOLIDATE_SEALED_DESIGN_CHANGED")
    teacher_design, choice_protocol = validate_design(design)
    if dispatch["stage"] not in {"run", "train", "evaluate"}:
        raise ValueError("GEN_CONSOLIDATE_UNKNOWN_STAGE")
    if dispatch["stage"] != "evaluate" and dispatch.get("teacher_role") not in design["teacher_sources"]:
        raise ValueError("GEN_CONSOLIDATE_EXPLICIT_TEACHER_REQUIRED")
    corpus, audit = load_corpus(choice_protocol)
    prepare(args.output_dir, dispatch, design, audit)
    if args.validate_only:
        write_json(args.output_dir / "validation.json", {"status": "cpu_contract_validated_only", "gpu_qualified": False})
        return
    try:
        if dispatch["stage"] == "evaluate":
            evaluate_learners(design, dispatch, teacher_design, choice_protocol, corpus, audit, args.output_dir)
        else:
            train(design, dispatch, teacher_design, choice_protocol, corpus, audit, args.output_dir)
            if json.loads((args.output_dir / "seal.json").read_text())["source_sha256"] != source_hashes():
                raise ValueError("GEN_CONSOLIDATE_SOURCE_CHANGED_DURING_TRAINING")
            if dispatch["stage"] == "run":
                evaluate_in_new_process(args.output_dir, dispatch)
        if json.loads((args.output_dir / "seal.json").read_text())["source_sha256"] != source_hashes():
            raise ValueError("GEN_CONSOLIDATE_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        event(args.output_dir, "failed", exception=type(error).__name__, detail=str(error))
        write_json(args.output_dir / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
