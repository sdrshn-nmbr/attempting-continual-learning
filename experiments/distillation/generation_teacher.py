import argparse
import json
import logging
import math
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from torch.nn import functional
from transformers import GenerationConfig

from choice_consolidation import (
    adapter_state,
    base_tensors,
    candidate_logps,
    check_base,
    encode_choices,
    evaluate,
    event,
    load_base,
    tensor_hash,
    verify_teacher,
)
from choice_contract import (
    ROOT,
    TASKS,
    digest,
    file_hash,
    load_corpus,
    metrics,
    prediction,
    training_schedule,
    validate_protocol,
    verify_qualification,
    write_json,
)

CONTRACT = "sequence303_source_teacher_three_digits_native_eos_20260912"
SOURCE_FILES = (
    "generation_teacher.py", "choice_consolidation.py", "choice_contract.py", "requirements.txt",
)


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    if design["contract"] != CONTRACT or design["generation"] != {
        "max_new_tokens": 16,
        "body": "exactly_one_leading_space_then_three_0_to_7_digits_separated_by_single_spaces",
        "eos": "tokenizer_native_eos_token_id",
        "do_sample": False,
        "forced_eos": False,
    }:
        raise ValueError("GEN_TEACHER_OUTPUT_CONTRACT_CHANGED")
    if design["training"] != {
        "updates": 384, "examples_per_update": 4, "learning_rate": 0.0002,
        "weight_decay": 0.0, "max_grad_norm": 1.0,
        "checkpoints": [32, 128, 384], "train_probes_per_task": 32,
        "selection": "earliest_checkpoint_with_train_probe95_all_tasks_else384",
    }:
        raise ValueError("GEN_TEACHER_TRAINING_CONTRACT_CHANGED")
    if design["qualification"] != {
        "splits": ["train", "validation"],
        "minimum_exact_accuracy_per_task_and_split": 0.9,
        "minimum_choice_accuracy_per_task_and_split": 0.9,
        "require_all_tasks": True,
    }:
        raise ValueError("GEN_TEACHER_QUALIFICATION_CHANGED")
    spec = design["choice_protocol"]
    path = ROOT / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError("GEN_TEACHER_CHOICE_PROTOCOL_CHANGED")
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def grade(row, token_ids, body, eos, special_ids, cap):
    terminated = bool(token_ids) and token_ids[-1] == eos
    content = token_ids[:-1] if terminated else token_ids
    clean = not any(token in special_ids for token in content)
    valid = re.fullmatch(r" [0-7] [0-7] [0-7]", body) is not None and clean
    gold = row["choices"][row["gold_idx"]]
    return {
        "correct": len(token_ids) <= cap and terminated and valid and body == gold,
        "format_valid": valid, "terminated": terminated,
        "cap_without_eos": len(token_ids) >= cap and not terminated,
        "extra_body_characters": len(body) > 6,
        "correct_prefix_with_extra_text": body.startswith(gold) and body != gold,
        "exact_body_without_valid_stop": body == gold and not terminated,
        "digit_position_correct": [len(body) > index and body[index] == gold[index] for index in (1, 3, 5)],
    }


def generation_settings(tokenizer, design):
    if tokenizer.eos_token_id is None:
        raise ValueError("GEN_TEACHER_NATIVE_EOS_REQUIRED")
    return GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=design["generation"]["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        use_cache=True, repetition_penalty=1.0,
    )


def generate(model, tokenizer, row, design):
    device = next(model.parameters()).device
    prompt = tokenizer(row["prompt"], add_special_tokens=True).input_ids
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generation = generation_settings(tokenizer, design)
    model.generation_config = generation
    if isinstance(model, PeftModel):
        model.get_base_model().generation_config = generation
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
    tokens = result[0, len(prompt):].tolist()
    body_ids = tokens[:-1] if tokens and tokens[-1] == tokenizer.eos_token_id else tokens
    body = tokenizer.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "token_ids": tokens, "prompt_token_ids": prompt, "body_text": body,
        "raw_text": tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        **grade(row, tokens, body, tokenizer.eos_token_id, tokenizer.all_special_ids, design["generation"]["max_new_tokens"]),
    }


def eos_encoded(tokenizer, row):
    sequence, length, characters = encode_choices(tokenizer, row, [row["gold_idx"]])[0]
    if tokenizer.eos_token_id is None:
        raise ValueError("GEN_TEACHER_NATIVE_EOS_REQUIRED")
    return sequence + [tokenizer.eos_token_id], length + 1, characters


def forced_diagnostics(model, tokenizer, row):
    sequence, length, _ = eos_encoded(tokenizer, row)
    device = next(model.parameters()).device
    start = len(sequence) - length
    ids = torch.tensor([sequence], dtype=torch.long, device=device)
    targets = ids[0, start:]
    pieces = [tokenizer(str(i), add_special_tokens=False).input_ids for i in range(8)]
    space = tokenizer(" ", add_special_tokens=False).input_ids
    if length != 7 or len(space) != 1 or any(len(piece) != 1 for piece in pieces):
        raise ValueError("GEN_TEACHER_THREE_DIGIT_TOKEN_BOUNDARY_CHANGED")
    gold_digits = tuple(map(int, row["choices"][row["gold_idx"]].split()))
    expected = [token for digit in gold_digits for token in space + pieces[digit]] + [tokenizer.eos_token_id]
    if targets.tolist() != expected:
        raise ValueError("GEN_TEACHER_GOLD_TOKENS_CHANGED")
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, start - 1:-1].float()
        selected = functional.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        predictions = logits.argmax(-1)
    if not torch.isfinite(selected).all():
        raise ValueError("GEN_TEACHER_NONFINITE_GOLD_PROBABILITY")
    token_logps = selected.tolist()
    total = sum(token_logps)
    return {
        "target_token_ids": targets.tolist(), "argmax_token_ids": predictions.tolist(),
        "gold_token_log_probabilities": token_logps,
        "gold_sequence_log_probability_including_eos": total,
        "gold_sequence_probability_including_eos": math.exp(total),
        "gold_digit_body_log_probability": sum(token_logps[:-1]),
        "digit_position_correct": [bool(predictions[i] == targets[i]) for i in (1, 3, 5)],
        "eos_argmax_correct": bool(predictions[-1] == tokenizer.eos_token_id),
        "eos_probability_after_gold_prefix": math.exp(token_logps[-1]),
    }


def measure(model, tokenizer, rows, design, output, label, detailed=True):
    model.eval()
    records = []
    for row in rows:
        record = {
            "id": row["id"], "task": row["task"], "group": row["group"],
            "row_sha256": digest(row), "generation": generate(model, tokenizer, row, design),
        }
        if detailed:
            record["teacher_forced"] = forced_diagnostics(model, tokenizer, row)
        records.append(record)
        event(output, "generation_teacher_probe", condition=label, id=row["id"], correct=record["generation"]["correct"])
    return records


def summarize(records):
    summary = {}
    for task in TASKS:
        rows = [row for row in records if row["task"] == task]
        if not rows:
            raise ValueError("GEN_TEACHER_MISSING_TASK")
        generated = [row["generation"] for row in rows]
        result = {"count": len(rows)}
        for field in ("correct", "format_valid", "terminated", "cap_without_eos", "extra_body_characters", "correct_prefix_with_extra_text", "exact_body_without_valid_stop"):
            result[field] = sum(row[field] for row in generated) / len(generated)
        result["digit_position_accuracy"] = [sum(row["digit_position_correct"][i] for row in generated) / len(generated) for i in range(3)]
        if all("teacher_forced" in row for row in rows):
            forced = [row["teacher_forced"] for row in rows]
            result["teacher_forced_digit_position_accuracy"] = [sum(row["digit_position_correct"][i] for row in forced) / len(forced) for i in range(3)]
            for field in ("gold_sequence_log_probability_including_eos", "gold_sequence_probability_including_eos", "gold_digit_body_log_probability", "eos_probability_after_gold_prefix", "eos_argmax_correct"):
                result["mean_" + field] = sum(row[field] for row in forced) / len(forced)
        summary[task] = result
    return summary


def select_checkpoint(checkpoints, design):
    for step in design["training"]["checkpoints"]:
        summary = checkpoints[str(step)]["train_probe_summary"]
        if all(summary[task]["correct"] >= 0.95 for task in TASKS):
            return step
    return design["training"]["updates"]


def verify_generated_rows(rows, records, design, eos, special_ids):
    if len(records) != len(rows):
        raise ValueError("GEN_TEACHER_INCOMPLETE_GENERATIONS")
    for row, record in zip(rows, records, strict=True):
        if (
            record["id"] != row["id"] or record["task"] != row["task"]
            or record["group"] != row["group"] or record["row_sha256"] != digest(row)
        ):
            raise ValueError("GEN_TEACHER_GENERATION_ROW_BINDING")
        generated = record["generation"]
        expected = grade(row, generated["token_ids"], generated["body_text"], eos, special_ids, design["generation"]["max_new_tokens"])
        if any(generated[key] != value for key, value in expected.items()):
            raise ValueError("GEN_TEACHER_QUALIFICATION_GRADING_CHANGED")


def generation_gate(design, corpus, panels, eos, special_ids):
    if set(panels) != {"train", "validation"}:
        raise ValueError("GEN_TEACHER_QUALIFICATION_SPLITS")
    results = {}
    for split in ("train", "validation"):
        records = panels[split]["generated"]
        choices = panels[split]["choice"]
        rows = corpus[split]
        if len(records) != len(rows) or len(choices) != len(rows):
            raise ValueError("GEN_TEACHER_INCOMPLETE_QUALIFICATION")
        verify_generated_rows(rows, records, design, eos, special_ids)
        for row, choice in zip(rows, choices, strict=True):
            if choice != prediction(row, choice["scores"]):
                raise ValueError("GEN_TEACHER_QUALIFICATION_CHOICE_BINDING")
        summary = summarize(records)
        classified = metrics(choices)
        for task in TASKS:
            results[f"{split}/{task}"] = {
                "generated": summary[task], "choice": classified[task],
                "passed": summary[task]["correct"] >= design["qualification"]["minimum_exact_accuracy_per_task_and_split"]
                and classified[task]["accuracy"] >= design["qualification"]["minimum_choice_accuracy_per_task_and_split"],
            }
    return {"passed": all(value["passed"] for value in results.values()), "panels": results}


def load_teacher(protocol, output, trainable):
    verify_teacher(protocol)
    base, tokenizer = load_base(protocol, output)
    model = PeftModel.from_pretrained(base, protocol["teacher"]["path"], adapter_name="teacher", is_trainable=trainable, local_files_only=True).eval()
    if tensor_hash(adapter_state(model, "teacher")) != protocol["teacher"]["tensor_sha256"]:
        raise ValueError("GEN_TEACHER_SOURCE_TENSOR_MISMATCH")
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = protocol["learner"]["expected_trainable_parameters"] if trainable else 0
    if count != expected:
        raise ValueError("GEN_TEACHER_CAPACITY_MISMATCH")
    return model, tokenizer


def qualify_model(model, tokenizer, protocol, design, corpus, output, label):
    before = tensor_hash(adapter_state(model, "teacher"))
    panels = {}
    for split in ("train", "validation"):
        panels[split] = {
            "generated": measure(model, tokenizer, corpus[split], design, output, f"{label}/{split}"),
            "choice": evaluate(model, tokenizer, corpus[split]),
        }
    gate = generation_gate(design, corpus, panels, tokenizer.eos_token_id, tokenizer.all_special_ids)
    write_json(output / "qualification_panels.json", panels)
    check_base(model, protocol["source_base_tensor_sha256"])
    if tensor_hash(adapter_state(model, "teacher")) != before:
        raise ValueError("GEN_TEACHER_QUALIFICATION_MUTATED_ADAPTER")
    return gate


def receipt(output, design, audit, gate, model, tokenizer, checkpoint, training_pid=None):
    result = {
        "status": "qualified" if gate["passed"] else "rejected", "gate": gate,
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "panels_sha256": file_hash(output / "qualification_panels.json"),
        "teacher_tensor_sha256": tensor_hash(adapter_state(model, "teacher")), "checkpoint": checkpoint,
        "eos_token_id": tokenizer.eos_token_id, "special_token_ids": tokenizer.all_special_ids,
        "pid": os.getpid(), "training_pid": training_pid, "learner_updates": 0,
        "claim_boundary": design["claim_boundary"],
    }
    write_json(output / "qualification.json", result)
    return result


def verify_baseline(path, design, protocol, corpus, audit):
    root = Path(path)
    baseline = json.loads((root / "qualification.json").read_text())
    if (
        baseline["design_sha256"] != digest(design)
        or baseline["source_sha256"] != source_hashes()
        or baseline["dataset"] != audit
        or baseline["teacher_tensor_sha256"] != protocol["teacher"]["tensor_sha256"]
        or file_hash(root / "qualification_panels.json") != baseline["panels_sha256"]
    ):
        raise ValueError("GEN_TEACHER_BASELINE_IDENTITY_CHANGED")
    panels = json.loads((root / "qualification_panels.json").read_text())
    gate = generation_gate(design, corpus, panels, baseline["eos_token_id"], baseline["special_token_ids"])
    if gate != baseline["gate"]:
        raise ValueError("GEN_TEACHER_BASELINE_GATE_CHANGED")
    if gate["passed"] or baseline["status"] != "rejected":
        raise ValueError("GEN_TEACHER_ALREADY_QUALIFIED_NO_TRAINING_NEEDED")
    return baseline


def diagnose(protocol, design, corpus, audit, output):
    model, tokenizer = load_teacher(protocol, output, trainable=False)
    gate = qualify_model(model, tokenizer, protocol, design, corpus, output, "source0")
    receipt(output, design, audit, gate, model, tokenizer, protocol["teacher"])
    event(output, "source_generation_diagnosed", qualified=gate["passed"], optimizer_updates=0)


def train_eos(protocol, design, dispatch, corpus, audit, output):
    verify_baseline(dispatch["diagnosis_dir"], design, protocol, corpus, audit)
    model, tokenizer = load_teacher(protocol, output, trainable=True)
    model.set_adapter("teacher")
    model.eval()
    before = tensor_hash(adapter_state(model, "teacher"))
    frozen = [(p, p._version) for p in base_tensors(model).values()]
    params = [p for p in model.parameters() if p.requires_grad]
    recipe = design["training"]
    optimizer = torch.optim.AdamW(params, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    schedule = training_schedule(protocol, corpus["train"])
    if len(schedule) != recipe["updates"] or any(len(batch) != recipe["examples_per_update"] for batch in schedule):
        raise ValueError("GEN_TEACHER_BUDGET_SCHEDULE_MISMATCH")
    probes = [row for task in TASKS for row in [r for r in corpus["train"] if r["task"] == task][:recipe["train_probes_per_task"]]]
    checkpoints, ledger = {}, []
    for step, indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in indices:
            encoded = eos_encoded(tokenizer, corpus["train"][index])
            loss = -candidate_logps(model, tokenizer, [encoded])[0] / encoded[1]
            if not torch.isfinite(loss):
                raise ValueError("GEN_TEACHER_NONFINITE_TRAINING_LOSS")
            (loss / len(indices)).backward()
            losses.append(float(loss.detach()))
            del loss
        norm = float(torch.nn.utils.clip_grad_norm_(params, recipe["max_grad_norm"]))
        if not math.isfinite(norm):
            raise ValueError("GEN_TEACHER_NONFINITE_GRADIENT")
        optimizer.step()
        if any(p.requires_grad or p.grad is not None or p._version != version for p, version in frozen):
            raise ValueError("GEN_TEACHER_FROZEN_BASE_MUTATED_DURING_TRAINING")
        record = {"step": step, "ids": [corpus["train"][i]["id"] for i in indices], "loss": sum(losses) / len(losses), "gradient_norm": norm}
        ledger.append(record)
        event(output, "eos_teacher_update", **record)
        if step in recipe["checkpoints"]:
            optimizer.zero_grad(set_to_none=True)
            folder = output / f"checkpoint{step}"
            model.save_pretrained(folder, selected_adapters=["teacher"], save_embedding_layers=False, safe_serialization=True)
            torch.save(optimizer.state_dict(), folder / "optimizer.pt")
            checkpoint = folder / "teacher"
            scores = measure(model, tokenizer, probes, design, output, f"train_only/{step}", detailed=False)
            write_json(folder / "train_probes.json", scores)
            checkpoints[str(step)] = {
                "path": str(checkpoint), "tensor_sha256": tensor_hash(adapter_state(model, "teacher")),
                "files": {name: file_hash(checkpoint / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
                "optimizer_sha256": file_hash(folder / "optimizer.pt"),
                "train_probes_sha256": file_hash(folder / "train_probes.json"),
                "train_probe_summary": summarize(scores),
            }
    check_base(model, protocol["source_base_tensor_sha256"])
    if tensor_hash(adapter_state(model, "teacher")) == before:
        raise ValueError("GEN_TEACHER_TRAINING_DID_NOT_CHANGE_SOURCE")
    write_json(output / "training.json", {
        "status": "trained_qualification_pending", "pid": os.getpid(), "design_sha256": digest(design),
        "source_sha256": source_hashes(), "dataset": audit,
        "source_teacher_sha256": before, "checkpoints": checkpoints, "ledger": ledger,
        "updates": len(ledger), "example_exposures": sum(len(item["ids"]) for item in ledger),
        "trainable_parameters": sum(p.numel() for p in params), "learner_updates": 0,
        "validation_predictions_observed_during_training": False,
        "diagnosis_sha256": file_hash(Path(dispatch["diagnosis_dir"]) / "qualification.json"),
        "selection": recipe["selection"],
        "selected_checkpoint": select_checkpoint(checkpoints, design),
        "eos_token_id": tokenizer.eos_token_id,
        "special_token_ids": tokenizer.all_special_ids,
    })


def evaluate_teacher(protocol, design, dispatch, corpus, audit, output):
    path = Path(dispatch["training_dir"]) / "training.json"
    training = json.loads(path.read_text())
    if (
        training["status"] != "trained_qualification_pending"
        or training["pid"] == os.getpid()
        or training["design_sha256"] != digest(design)
        or training["source_sha256"] != source_hashes()
        or training["dataset"] != audit
        or training["updates"] != design["training"]["updates"]
        or training["example_exposures"] != design["training"]["updates"] * design["training"]["examples_per_update"]
        or training["validation_predictions_observed_during_training"]
        or training["source_teacher_sha256"] != protocol["teacher"]["tensor_sha256"]
    ):
        raise ValueError("GEN_TEACHER_TRAINING_RECEIPT_CHANGED_OR_NOT_FRESH_PROCESS")
    probes = [row for task in TASKS for row in [r for r in corpus["train"] if r["task"] == task][:design["training"]["train_probes_per_task"]]]
    for step in design["training"]["checkpoints"]:
        checkpoint = training["checkpoints"][str(step)]
        folder = Path(checkpoint["path"]).parent
        if (
            file_hash(folder / "train_probes.json") != checkpoint["train_probes_sha256"]
            or file_hash(folder / "optimizer.pt") != checkpoint["optimizer_sha256"]
        ):
            raise ValueError("GEN_TEACHER_TRAIN_PROBES_OR_OPTIMIZER_CHANGED")
        records = json.loads((folder / "train_probes.json").read_text())
        verify_generated_rows(probes, records, design, training["eos_token_id"], training["special_token_ids"])
        if summarize(records) != checkpoint["train_probe_summary"]:
            raise ValueError("GEN_TEACHER_TRAIN_PROBE_SUMMARY_CHANGED")
    selected = select_checkpoint(training["checkpoints"], design)
    if selected != training["selected_checkpoint"] or training["selection"] != design["training"]["selection"]:
        raise ValueError("GEN_TEACHER_TRAIN_ONLY_CHECKPOINT_SELECTION_CHANGED")
    spec = training["checkpoints"][str(selected)]
    for name, checksum in spec["files"].items():
        if file_hash(Path(spec["path"]) / name) != checksum:
            raise ValueError(f"GEN_TEACHER_CHECKPOINT_FILE_CHANGED: {name}")
    base, tokenizer = load_base(protocol, output)
    model = PeftModel.from_pretrained(base, spec["path"], adapter_name="teacher", is_trainable=False, local_files_only=True).requires_grad_(False).eval()
    if tensor_hash(adapter_state(model, "teacher")) != spec["tensor_sha256"]:
        raise ValueError("GEN_TEACHER_FINAL_RELOAD_MISMATCH")
    gate = qualify_model(model, tokenizer, protocol, design, corpus, output, f"trained{selected}")
    result = receipt(output, design, audit, gate, model, tokenizer, spec, training_pid=training["pid"])
    if gate["passed"]:
        heldout = measure(model, tokenizer, corpus["test"], design, output, "qualified/test", detailed=False)
        write_json(output / "holdout.json", {"records": heldout, "summary": summarize(heldout)})
        check_base(model, protocol["source_base_tensor_sha256"])
    event(output, "generation_teacher_finished", status=result["status"], learner_updates=0)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [source-generation-teacher] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"]:
        raise ValueError("GEN_TEACHER_SEALED_DESIGN_CHANGED")
    protocol = validate_design(design)
    if dispatch["stage"] not in {"diagnose", "train_eos", "evaluate"}:
        raise ValueError("GEN_TEACHER_UNKNOWN_STAGE")
    corpus, audit = load_corpus(protocol)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("GEN_TEACHER_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("GEN_TEACHER_DISPATCH_CONFIG_MISMATCH")
    write_json(output / "seal.json", {"design": design, "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit, "dispatch": dispatch})
    if args.validate_only:
        write_json(output / "validation.json", {"status": "cpu_contract_validated_only", "gpu_qualified": False})
        return
    try:
        verify_qualification(dispatch["choice_qualification_dir"], protocol, corpus, audit)
        if dispatch["stage"] == "diagnose":
            diagnose(protocol, design, corpus, audit, output)
        elif dispatch["stage"] == "train_eos":
            train_eos(protocol, design, dispatch, corpus, audit, output)
        else:
            evaluate_teacher(protocol, design, dispatch, corpus, audit, output)
        if json.loads((output / "seal.json").read_text())["source_sha256"] != source_hashes():
            raise ValueError("GEN_TEACHER_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        event(output, "failed", exception=type(error).__name__, detail=str(error))
        write_json(output / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
