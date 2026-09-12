from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from data import SEQUENCE_TASKS, Example, write_json
from learner import frozen_base_tensors, load_base, tensor_hash
from peft import PeftModel, get_peft_model_state_dict
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

CONFIG_SHA256 = "67f6924e22c0a1ccbe52eb1d985a9dd97b50ff163e606ccb11094294d57ba91a"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(output: Path, event: str, **values: object) -> None:
    row = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "event": event,
        **values,
    }
    line = json.dumps(row, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    with (output / "events.jsonl").open("a") as stream:
        stream.write(line + "\n")


def read_calibration_rows(specs: list[dict]) -> list[tuple[str, Example]]:
    expected = {
        (task, split) for task in SEQUENCE_TASKS for split in ("train", "validation")
    }
    if {(spec["task"], spec["split"]) for spec in specs} != expected or len(specs) != 6:
        raise ValueError(
            "CALIBRATION_SPLIT_CONTRACT: six train/validation files required"
        )
    result = []
    groups = defaultdict(set)
    for spec in specs:
        path = Path(spec["path"])
        if path.name != f"{spec['task']}_{spec['split']}.json":
            raise ValueError("CALIBRATION_FILE_NAME_MISMATCH")
        if file_hash(path) != spec["sha256"]:
            raise ValueError(f"CALIBRATION_DATA_HASH_MISMATCH: {path}")
        rows = json.loads(path.read_text())
        if len(rows) != spec["total_rows"] or not 0 < spec["take_first"] <= len(rows):
            raise ValueError("CALIBRATION_ROW_COUNT_MISMATCH")
        for raw in rows[: spec["take_first"]]:
            row = Example.from_dict(raw)
            if row.task != spec["task"] or not row.prompt.endswith("\nOutput:"):
                raise ValueError("CALIBRATION_ROW_IDENTITY_MISMATCH")
            groups[spec["split"]].add(row.group)
            result.append((spec["split"], row))
    if groups["train"] & groups["validation"]:
        raise ValueError("CALIBRATION_TRAIN_VALIDATION_INPUT_OVERLAP")
    return result


def checkpoint_files(config: dict) -> dict[str, str]:
    result = {}
    for spec in config["checkpoints"].values():
        for name, expected in spec["files"].items():
            path = Path(spec["path"]) / name
            actual = file_hash(path)
            if actual != expected:
                raise ValueError(f"CALIBRATION_CHECKPOINT_HASH_MISMATCH: {path}")
            result[str(path)] = actual
    return result


class StopAtNewline(StoppingCriteria):
    def __init__(self, tokenizer, prefix_length: int):
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor(
            [
                "\n"
                in self.tokenizer.decode(
                    ids[self.prefix_length :], skip_special_tokens=True
                )
                for ids in input_ids
            ],
            dtype=torch.bool,
            device=input_ids.device,
        )


def grade_line(text: str, expected: str) -> dict:
    line = text.split("\n", 1)[0].strip()
    return {"first_line": line, "correct": line == expected.strip()}


def generate_rows(base, rows: list[tuple[str, Example]], recipe: dict) -> list[dict]:
    model, tokenizer = base.model, base.tokenizer
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if not eos_ids or tokenizer.pad_token_id is None:
        raise ValueError("CALIBRATION_NATIVE_STOP_TOKENS_REQUIRED")
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=recipe["max_new_tokens"],
        eos_token_id=sorted(eos_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    records = []
    for offset in range(0, len(rows), recipe["batch_size"]):
        batch = rows[offset : offset + recipe["batch_size"]]
        prompts = [
            tokenizer(row.prompt.rstrip(), add_special_tokens=True).input_ids
            for _, row in batch
        ]
        if any(not ids or len(ids) > 768 for ids in prompts):
            raise ValueError("CALIBRATION_PROMPT_LENGTH")
        width = max(map(len, prompts))
        ids = torch.full(
            (len(batch), width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=base.device,
        )
        attention = torch.zeros_like(ids)
        for index, prompt in enumerate(prompts):
            ids[index, -len(prompt) :] = torch.tensor(prompt, device=base.device)
            attention[index, -len(prompt) :] = 1
        with (
            torch.inference_mode(),
            torch.autocast(device_type=base.device.type, enabled=False),
        ):
            generated = model.generate(
                input_ids=ids,
                attention_mask=attention,
                generation_config=generation,
                stopping_criteria=StoppingCriteriaList(
                    [StopAtNewline(tokenizer, width)]
                ),
            )[:, width:]
        for (split, row), prompt, tokens in zip(
            batch, prompts, generated.cpu().tolist(), strict=True
        ):
            kept = []
            for token in tokens:
                kept.append(token)
                text = tokenizer.decode(kept, skip_special_tokens=True)
                if token in eos_ids or "\n" in text:
                    break
            text = tokenizer.decode(kept, skip_special_tokens=True)
            expected = row.choices[row.gold_idx]
            records.append(
                {
                    "id": row.id,
                    "task": row.task,
                    "split": split,
                    "group": row.group,
                    "prompt": row.prompt,
                    "prompt_token_ids": prompt,
                    "generated_token_ids": kept,
                    "raw_text": text,
                    "expected": expected,
                    "hit_token_cap": len(kept) == recipe["max_new_tokens"]
                    and kept[-1] not in eos_ids
                    and "\n" not in text,
                    **grade_line(text, expected),
                }
            )
    return records


def summarize(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in records:
        grouped[f"{row['task']}/{row['split']}"].append(row)
    return {
        key: {
            "correct": sum(row["correct"] for row in rows),
            "total": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "generated_tokens": sum(len(row["generated_token_ids"]) for row in rows),
            "token_cap_hits": sum(row["hit_token_cap"] for row in rows),
        }
        for key, rows in grouped.items()
    }


def calibrate(config: dict, output: Path, device: str = "cuda:0") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "events.jsonl").exists():
        raise ValueError("CALIBRATION_OUTPUT_ALREADY_USED")
    torch.manual_seed(config["seed"])
    before_files = checkpoint_files(config)
    rows = read_calibration_rows(config["inputs"])
    emit(output, "inputs_verified", rows=len(rows), checkpoint_files=before_files)
    base = load_base(config["base"], device=device)
    base_hash = tensor_hash(frozen_base_tensors(base.model))
    if base_hash != config["base_tensor_sha256"]:
        raise ValueError("CALIBRATION_BASE_HASH_MISMATCH")
    model = PeftModel.from_pretrained(
        base.model,
        config["checkpoints"]["initial"]["path"],
        adapter_name="initial",
        is_trainable=False,
    )
    model.load_adapter(
        config["checkpoints"]["teacher"]["path"],
        adapter_name="teacher",
        is_trainable=False,
    )
    base = replace(base, model=model)
    model.requires_grad_(False).eval()
    panels, adapter_hashes = {}, {}
    for condition in ("initial", "teacher"):
        model.set_adapter(condition, inference_mode=True)
        model.requires_grad_(False).eval()
        actual = tensor_hash(get_peft_model_state_dict(model, adapter_name=condition))
        if actual != config["checkpoints"][condition]["tensor_sha256"]:
            raise ValueError(f"CALIBRATION_ADAPTER_TENSOR_MISMATCH: {condition}")
        adapter_hashes[condition] = actual
        emit(
            output,
            "generation_started",
            condition=condition,
            adapter_tensor_sha256=actual,
        )
        records = generate_rows(base, rows, config["generation"])
        write_json(output / f"{condition}.json", records)
        panels[condition] = summarize(records)
        emit(
            output,
            "generation_complete",
            condition=condition,
            metrics=panels[condition],
        )
    after_adapters = {
        condition: tensor_hash(get_peft_model_state_dict(model, adapter_name=condition))
        for condition in adapter_hashes
    }
    after_base = tensor_hash(frozen_base_tensors(model))
    if (
        after_adapters != adapter_hashes
        or after_base != base_hash
        or checkpoint_files(config) != before_files
    ):
        raise ValueError("CALIBRATION_MUTATED_FROZEN_WEIGHTS")
    if read_calibration_rows(config["inputs"]) != rows:
        raise ValueError("CALIBRATION_INPUTS_CHANGED")
    gains = {
        task: panels["teacher"][f"{task}/validation"]["accuracy"]
        - panels["initial"][f"{task}/validation"]["accuracy"]
        for task in SEQUENCE_TASKS
    }
    qualified = all(
        panel["accuracy"] >= config["gate"]["minimum_accuracy_each_task_and_split"]
        for panel in panels["teacher"].values()
    ) and all(
        gain >= config["gate"]["minimum_validation_gain_over_initial"]
        for gain in gains.values()
    )
    result = {
        "kind": config["kind"],
        "config": config,
        "panels": panels,
        "validation_gain": gains,
        "generation_teacher_qualified": qualified,
        "optimizer_updates": 0,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "base_before_sha256": base_hash,
        "base_after_sha256": after_base,
        "adapter_before_sha256": adapter_hashes,
        "adapter_after_sha256": after_adapters,
        "checkpoint_files": before_files,
        "pid": os.getpid(),
        "device": str(base.device),
        "torch_version": torch.__version__,
        "hip": torch.version.hip,
        "proof_boundary": config["proof_boundary"],
    }
    write_json(output / "result.json", result)
    emit(output, "calibration_complete", qualified=qualified)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if file_hash(args.config) != CONFIG_SHA256:
        raise ValueError("UNREGISTERED_TEACHER_CALIBRATION_CONFIG")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.version.hip
    ):
        raise RuntimeError("CALIBRATION_REQUIRES_ONE_ROCM_GPU")
    try:
        calibrate(json.loads(args.config.read_text()), args.output_dir)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "failure.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
