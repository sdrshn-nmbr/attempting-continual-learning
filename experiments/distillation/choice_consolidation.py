import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from choice_contract import (
    METHODS,
    ROOT,
    SCORING,
    digest,
    file_hash,
    load_corpus,
    metrics,
    prediction,
    qualification_gate,
    source_hashes,
    training_schedule,
    validate_protocol,
    verify_predictions,
    verify_qualification,
    write_json,
)

LOGGER = logging.getLogger("distillation.choice")


def event(output, name, **fields):
    record = {"event": name, "unix_time": time.time(), "pid": os.getpid(), **fields}
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    LOGGER.info("%s", json.dumps(record, allow_nan=False))


def tensor_hash(tensors):
    hasher = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        hasher.update(name.encode())
        hasher.update(str((tuple(value.shape), value.dtype)).encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        hasher.update(memoryview(raw))
    return hasher.hexdigest()


def base_tensors(model):
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    tensors = {}
    for kind, values in (("parameter", base.named_parameters()), ("buffer", base.named_buffers())):
        for name, value in values:
            if "lora_" in name:
                continue
            key = f"{kind}:{name.replace('.base_layer.', '.')}"
            if key in tensors or value.requires_grad:
                raise ValueError(f"CONSOLIDATE_UNFROZEN_OR_DUPLICATE_BASE: {key}")
            tensors[key] = value
    return tensors


def adapter_state(model, name="learner"):
    return get_peft_model_state_dict(model, adapter_name=name, save_embedding_layers=False)


def check_base(model, expected):
    tensors = base_tensors(model)
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors.values()):
        raise ValueError("CONSOLIDATE_BASE_NOT_FP32")
    if any(t.grad is not None for t in tensors.values()) or tensor_hash(tensors) != expected:
        raise ValueError("CONSOLIDATE_FROZEN_BASE_MUTATED")


def verify_snapshot(spec):
    root = Path(spec["local_path"])
    checked = {}
    for item in spec["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"CONSOLIDATE_SNAPSHOT_SIZE: {path}")
        if item["sha256"]:
            checksum = file_hash(path)
            if checksum != item["sha256"]:
                raise ValueError(f"CONSOLIDATE_SNAPSHOT_HASH: {path}")
        else:
            raw = path.read_bytes()
            checksum = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if checksum != item["git_blob_sha1"]:
                raise ValueError(f"CONSOLIDATE_SNAPSHOT_GIT_BLOB: {path}")
        checked[item["path"]] = checksum
    unexpected = {
        path.name for path in root.iterdir()
        if path.is_file() and path.name not in checked and path.name not in {"README.md", ".gitattributes"}
    }
    if unexpected:
        raise ValueError(f"CONSOLIDATE_UNPINNED_MODEL_FILES: {sorted(unexpected)}")
    return checked


def verify_teacher(protocol):
    spec = protocol["teacher"]
    for filename, checksum in spec["files"].items():
        if file_hash(Path(spec["path"]) / filename) != checksum:
            raise ValueError(f"CONSOLIDATE_TEACHER_FILE_CHANGED: {filename}")
    config = json.loads((Path(spec["path"]) / "adapter_config.json").read_text())
    if (
        {config["r"], *config.get("rank_pattern", {}).values()} != {8}
        or {config["lora_alpha"], *config.get("alpha_pattern", {}).values()} != {16}
        or config["bias"] != "none"
        or config.get("modules_to_save")
    ):
        raise ValueError("CONSOLIDATE_TEACHER_CAPACITY_MISMATCH")


def load_base(protocol, output):
    if not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError("CONSOLIDATE_REQUIRE_ONE_VISIBLE_ROCM_GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(protocol["training"]["initialization_seed"])
    files = verify_snapshot(protocol["source_base"])
    path = protocol["source_base"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).requires_grad_(False).eval()
    model.config.use_cache = False
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("CONSOLIDATE_MODEL_OFFLOAD_DISALLOWED")
    check_base(model, protocol["source_base_tensor_sha256"])
    write_json(output / "runtime.json", {
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")},
        "hip": torch.version.hip,
        "gpu": str(torch.cuda.get_device_properties(0)),
        "base_files": files, "base_tensor_sha256": protocol["source_base_tensor_sha256"],
        "dtype": "float32", "autocast": False,
    })
    return model, tokenizer


def encode_choices(tokenizer, row, choice_indices=None):
    prompt = row["prompt"].rstrip()
    boundary = row["prompt"][len(prompt):]
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if len(prompt_ids) > SCORING["max_prompt"]:
        raise ValueError("CONSOLIDATE_PROMPT_TRUNCATION_DISALLOWED")
    if not prompt_ids:
        token = tokenizer.bos_token_id
        if token is None:
            token = tokenizer.eos_token_id
        if token is None:
            raise ValueError("CONSOLIDATE_EMPTY_PROMPT_WITHOUT_START_TOKEN")
        prompt_ids = [token]
    result = []
    for index in range(4) if choice_indices is None else choice_indices:
        continuation = boundary + row["choices"][index]
        answer = tokenizer(continuation, add_special_tokens=False).input_ids
        if not answer:
            raise ValueError("CONSOLIDATE_EMPTY_ANSWER")
        result.append((prompt_ids + answer, len(answer), len(continuation)))
    return result


def candidate_logps(model, tokenizer, encoded):
    device = next(model.parameters()).device
    if torch.is_autocast_enabled(device.type):
        raise ValueError("CONSOLIDATE_AUTOCAST_DISALLOWED")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("CONSOLIDATE_PAD_TOKEN_REQUIRED")
    ids = torch.full((len(encoded), max(len(row[0]) for row in encoded)), pad, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (sequence, _, _) in enumerate(encoded):
        ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, :len(sequence)] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    if logits.dtype != torch.float32:
        raise ValueError("CONSOLIDATE_FORWARD_NOT_FP32")
    logps = functional.log_softmax(logits, dim=-1)
    totals = []
    for i, (sequence, answer_length, _) in enumerate(encoded):
        start = len(sequence) - answer_length
        targets = ids[i, start:len(sequence)]
        selected = logps[i, start - 1:len(sequence) - 1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        totals.append(selected.sum())
    totals = torch.stack(totals)
    if not torch.isfinite(totals).all():
        raise ValueError("CONSOLIDATE_NONFINITE_LOGPS")
    return totals


def choice_scores(model, tokenizer, row):
    encoded = encode_choices(tokenizer, row)
    totals = candidate_logps(model, tokenizer, encoded)
    characters = torch.tensor([item[2] for item in encoded], device=totals.device)
    return totals / characters


def choice_kl(student_scores, teacher_scores, temperature):
    if student_scores.shape != teacher_scores.shape or student_scores.shape[-1] != 4 or temperature <= 0:
        raise ValueError("CONSOLIDATE_KL_SUPPORT_MISMATCH")
    target = functional.log_softmax(teacher_scores.detach().float() / temperature, dim=-1)
    prediction_logps = functional.log_softmax(student_scores.float() / temperature, dim=-1)
    return (target.exp() * (target - prediction_logps)).sum(dim=-1).mean() * temperature**2


def evaluate(model, tokenizer, rows):
    model.eval()
    encoded = [item for row in rows for item in encode_choices(tokenizer, row)]
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(encoded), SCORING["candidate_batch_size"]):
            batch = encoded[offset:offset + SCORING["candidate_batch_size"]]
            totals = candidate_logps(model, tokenizer, batch).tolist()
            scores.extend(total / item[2] for total, item in zip(totals, batch, strict=True))
    records = [prediction(row, scores[i * 4:i * 4 + 4]) for i, row in enumerate(rows)]
    verify_predictions(rows, records)
    return records


def make_learner(base, protocol):
    torch.manual_seed(protocol["training"]["initialization_seed"])
    recipe = protocol["learner"]
    config = LoraConfig(
        r=recipe["rank"], lora_alpha=recipe["alpha"], target_modules=recipe["targets"],
        lora_dropout=0.0, bias="none", init_lora_weights=True, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, config, adapter_name="learner").eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != recipe["expected_trainable_parameters"]:
        raise ValueError("CONSOLIDATE_LEARNER_PARAMETER_COUNT")
    if any(torch.count_nonzero(t) for name, t in adapter_state(model).items() if "lora_B" in name):
        raise ValueError("CONSOLIDATE_LEARNER_INITIAL_OUTPUT_NOT_ZERO")
    return model


def save_adapter(model, directory):
    before = tensor_hash(adapter_state(model))
    model.save_pretrained(directory, selected_adapters=["learner"], safe_serialization=True, save_embedding_layers=False)
    folder = directory / "learner"
    return {
        "path": str(folder), "tensor_sha256": before,
        "files": {name: file_hash(folder / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
    }


def qualify(protocol, corpus, audit, output):
    verify_teacher(protocol)
    base, tokenizer = load_base(protocol, output)
    panels = {split: {"untouched": evaluate(base, tokenizer, corpus[split])} for split in ("train", "validation")}
    model = PeftModel.from_pretrained(base, protocol["teacher"]["path"], adapter_name="teacher", is_trainable=False, local_files_only=True)
    model.requires_grad_(False).eval()
    before = tensor_hash(adapter_state(model, "teacher"))
    if before != protocol["teacher"]["tensor_sha256"]:
        raise ValueError("CONSOLIDATE_TEACHER_TENSOR_MISMATCH")
    for split in ("train", "validation"):
        panels[split]["teacher"] = evaluate(model, tokenizer, corpus[split])
        event(output, "teacher_scored", split=split, metrics=metrics(panels[split]["teacher"]))
    if tensor_hash(adapter_state(model, "teacher")) != before:
        raise ValueError("CONSOLIDATE_QUALIFICATION_MUTATED_TEACHER")
    check_base(model, protocol["source_base_tensor_sha256"])
    gate = qualification_gate(protocol, corpus, panels)
    write_json(output / "qualification_panels.json", panels)
    receipt = {
        "status": "qualified" if gate["passed"] else "rejected",
        "protocol_sha256": digest(protocol), "source_sha256": source_hashes(),
        "dataset": audit, "gate": gate, "optimizer_updates": 0,
        "teacher_tensor_sha256": before, "base_tensor_sha256": protocol["source_base_tensor_sha256"],
        "panels_sha256": file_hash(output / "qualification_panels.json"),
        "teacher_targets": "Training panel scores only; no teacher queries permitted during learner training or final evaluation.",
    }
    write_json(output / "qualification.json", receipt)
    event(output, "qualification_finished", status=receipt["status"], learner_updates=0)


def train_arm(model, tokenizer, rows, targets, schedule, method, protocol, output):
    if method not in METHODS:
        raise ValueError("CONSOLIDATE_UNKNOWN_METHOD")
    model.set_adapter("learner")
    model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    frozen = [(p, p._version) for p in base_tensors(model).values()]
    recipe = protocol["training"]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    ledger = []
    for step, batch in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in batch:
            row = rows[index]
            if method == "teacher_choice_kl":
                scores = choice_scores(model, tokenizer, row)
                teacher = torch.tensor(targets[index]["scores"], device=scores.device)
                loss = choice_kl(scores, teacher, recipe["temperature"])
            else:
                encoded = encode_choices(tokenizer, row, [row["gold_idx"]])
                loss = -candidate_logps(model, tokenizer, encoded)[0] / encoded[0][1]
            if not torch.isfinite(loss):
                raise ValueError(f"CONSOLIDATE_NONFINITE_LOSS: {method}/{step}")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
            del loss
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, recipe["max_grad_norm"]))
        if not math.isfinite(norm):
            raise ValueError(f"CONSOLIDATE_NONFINITE_GRADIENT: {method}/{step}")
        optimizer.step()
        if any(p.requires_grad or p.grad is not None or p._version != version for p, version in frozen):
            raise ValueError("CONSOLIDATE_BASE_CHANGED_DURING_TRAINING")
        row = {"step": step, "ids": [rows[i]["id"] for i in batch], "loss": sum(losses) / len(losses), "gradient_norm": norm}
        ledger.append(row)
        event(output, "learner_update", method=method, **row)
    optimizer.zero_grad(set_to_none=True)
    folder = output / method
    checkpoint = save_adapter(model, folder)
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    return {
        "checkpoint": checkpoint, "ledger": ledger, "updates": len(ledger),
        "example_exposures": sum(len(item["ids"]) for item in ledger),
        "candidate_continuations_per_example": 4 if method == "teacher_choice_kl" else 1,
        "optimizer_sha256": file_hash(folder / "optimizer.pt"),
        "teacher_model_loaded": False,
    }


def train(protocol, dispatch, corpus, audit, output):
    qualification, targets = verify_qualification(dispatch["qualification_dir"], protocol, corpus, audit)
    base, tokenizer = load_base(protocol, output)
    model = make_learner(base, protocol)
    initial = {name: value.detach().clone() for name, value in adapter_state(model).items()}
    initial_hash = tensor_hash(initial)
    if initial_hash == protocol["teacher"]["tensor_sha256"]:
        raise ValueError("CONSOLIDATE_TEACHER_WEIGHTS_COPIED_TO_LEARNER")
    initial_checkpoint = save_adapter(model, output / "initial")
    schedule = training_schedule(protocol, corpus["train"])
    write_json(output / "schedule.json", [[corpus["train"][i]["id"] for i in batch] for batch in schedule])
    arms = {}
    for method in METHODS:
        set_peft_model_state_dict(model, initial, adapter_name="learner")
        if tensor_hash(adapter_state(model)) != initial_hash:
            raise ValueError("CONSOLIDATE_MATCHED_INITIALIZATION_FAILURE")
        arms[method] = train_arm(model, tokenizer, corpus["train"], targets, schedule, method, protocol, output)
        check_base(model, protocol["source_base_tensor_sha256"])
        if arms[method]["checkpoint"]["tensor_sha256"] == initial_hash:
            raise ValueError(f"CONSOLIDATE_LEARNER_UNCHANGED: {method}")
    if any(arms[m]["example_exposures"] != protocol["training"]["example_exposures_per_arm"] for m in METHODS):
        raise ValueError("CONSOLIDATE_EXPOSURE_ACCOUNTING_MISMATCH")
    receipt = {
        "status": "trained_evaluation_pending", "pid": os.getpid(),
        "protocol_sha256": digest(protocol), "source_sha256": source_hashes(), "dataset": audit,
        "qualification_sha256": file_hash(Path(dispatch["qualification_dir"]) / "qualification.json"),
        "qualified_teacher_tensor_sha256": qualification["teacher_tensor_sha256"],
        "initial": initial_checkpoint, "arms": arms, "teacher_model_loaded": False,
        "schedule_sha256": file_hash(output / "schedule.json"),
        "learner_trainable_parameters": protocol["learner"]["expected_trainable_parameters"],
        "learner_final_resident_rank": 8, "base_tensor_sha256": protocol["source_base_tensor_sha256"],
        "heldout_predictions_observed": False,
    }
    write_json(output / "training.json", receipt)
    event(output, "learner_training_finished", updates_per_arm=384, status=receipt["status"])


def verify_training(dispatch, protocol, audit):
    directory = Path(dispatch["training_dir"])
    receipt = json.loads((directory / "training.json").read_text())
    if (
        receipt["status"] != "trained_evaluation_pending"
        or receipt["protocol_sha256"] != digest(protocol)
        or receipt["source_sha256"] != source_hashes()
        or receipt["dataset"] != audit
        or receipt["pid"] == os.getpid()
        or receipt["teacher_model_loaded"]
        or receipt["heldout_predictions_observed"]
        or set(receipt["arms"]) != set(METHODS)
    ):
        raise ValueError("CONSOLIDATE_FRESH_PROCESS_TRAINING_IDENTITY")
    if file_hash(directory / "schedule.json") != receipt["schedule_sha256"]:
        raise ValueError("CONSOLIDATE_TRAINING_SCHEDULE_CHANGED")
    for method, arm in receipt["arms"].items():
        if (
            arm["updates"] != protocol["training"]["updates_per_arm"]
            or arm["example_exposures"] != protocol["training"]["example_exposures_per_arm"]
            or file_hash(directory / method / "optimizer.pt") != arm["optimizer_sha256"]
        ):
            raise ValueError("CONSOLIDATE_TRAINING_BUDGET_OR_OPTIMIZER_CHANGED")
    return receipt


def evaluate_trained(protocol, dispatch, corpus, audit, output):
    training = verify_training(dispatch, protocol, audit)
    base, tokenizer = load_base(protocol, output)
    panels = {split: {"untouched": evaluate(base, tokenizer, corpus[split])} for split in protocol["evaluation_splits"]}
    model = None
    checkpoints = {"initial": training["initial"], **{name: arm["checkpoint"] for name, arm in training["arms"].items()}}
    for name, spec in checkpoints.items():
        for filename, checksum in spec["files"].items():
            if file_hash(Path(spec["path"]) / filename) != checksum:
                raise ValueError(f"CONSOLIDATE_LEARNER_FILE_CHANGED: {name}/{filename}")
        if model is None:
            model = PeftModel.from_pretrained(base, spec["path"], adapter_name=name, is_trainable=False, local_files_only=True)
        else:
            model.load_adapter(spec["path"], adapter_name=name, is_trainable=False, local_files_only=True)
        model.set_adapter(name, inference_mode=True)
        model.requires_grad_(False).eval()
        if tensor_hash(adapter_state(model, name)) != spec["tensor_sha256"]:
            raise ValueError(f"CONSOLIDATE_SAVED_LEARNER_RELOAD_MISMATCH: {name}")
        for split in protocol["evaluation_splits"]:
            records = evaluate(model, tokenizer, corpus[split])
            if name == "initial":
                baseline = panels[split]["untouched"]
                if records != baseline:
                    raise ValueError("CONSOLIDATE_ZERO_ADAPTER_BASELINE_PARITY_FAILED")
            else:
                panels[split][name] = records
            event(output, "learner_evaluated", condition=name, split=split, metrics=metrics(records))
        check_base(model, protocol["source_base_tensor_sha256"])
    write_json(output / "panels.json", panels)
    comparison = {}
    for split, conditions in panels.items():
        comparison[split] = {name: metrics(records) for name, records in conditions.items()}
        for method in METHODS:
            comparison[split][method + "_minus_untouched"] = sum(
                int(a["correct"]) - int(b["correct"])
                for a, b in zip(conditions[method], conditions["untouched"], strict=True)
            ) / len(conditions[method])
        comparison[split]["kl_minus_sft"] = sum(
            int(a["correct"]) - int(b["correct"])
            for a, b in zip(conditions[METHODS[0]], conditions[METHODS[1]], strict=True)
        ) / len(conditions[METHODS[0]])
    write_json(output / "result.json", {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"],
        "protocol_sha256": digest(protocol), "source_sha256": source_hashes(), "dataset": audit,
        "training_sha256": file_hash(Path(dispatch["training_dir"]) / "training.json"),
        "panels_sha256": file_hash(output / "panels.json"), "comparisons": comparison,
        "teacher_model_loaded": False, "optimizer_updates": 0,
        "saved_learner_tensor_checks_passed": True, "untouched_zero_adapter_parity": True,
        "claim_boundary": protocol["claim_boundary"],
    })


def prepare(output, dispatch, protocol, audit):
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError(f"CONSOLIDATE_OUTPUT_ALREADY_USED: {output}")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("CONSOLIDATE_DISPATCH_CONFIG_MISMATCH")
    write_json(output / "seal.json", {
        "dispatch": dispatch, "protocol": protocol, "protocol_sha256": digest(protocol),
        "source_sha256": source_hashes(), "dataset": audit, "pid": os.getpid(),
        "predictions_observed": False,
    })


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [choice-consolidation] %(message)s", stream=sys.stdout)
    dispatch = json.loads(args.config.read_text())
    protocol = json.loads((ROOT / dispatch["protocol"]).read_text())
    validate_protocol(protocol)
    if digest(protocol) != dispatch["protocol_sha256"]:
        raise ValueError("CONSOLIDATE_SEALED_PROTOCOL_CHANGED")
    if dispatch["stage"] not in {"qualify", "train", "evaluate"}:
        raise ValueError("CONSOLIDATE_UNKNOWN_STAGE")
    corpus, audit = load_corpus(protocol)
    prepare(args.output_dir, dispatch, protocol, audit)
    if args.validate_only:
        write_json(args.output_dir / "validation.json", {"status": "cpu_contract_validated_only", "gpu_qualified": False, "dataset": audit})
        return
    try:
        if dispatch["stage"] == "qualify":
            qualify(protocol, corpus, audit, args.output_dir)
        elif dispatch["stage"] == "train":
            train(protocol, dispatch, corpus, audit, args.output_dir)
        else:
            evaluate_trained(protocol, dispatch, corpus, audit, args.output_dir)
        if json.loads((args.output_dir / "seal.json").read_text())["source_sha256"] != source_hashes():
            raise ValueError("CONSOLIDATE_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        event(args.output_dir, "failed", exception=type(error).__name__, detail=str(error))
        write_json(args.output_dir / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
