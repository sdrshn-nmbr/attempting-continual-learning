import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import platform
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from run import (
    activate,
    adapter_hash,
    checkpoint_load,
    event,
    new_model,
    tensor_hash,
    tree_hash,
    wilson,
    write_json,
)
from stream_tasks import (
    TASK_IDS,
    TASKS,
    Reservoir,
    audit_corpus,
    build_corpus,
    corpus_digest,
    render,
    stage_batches,
)
from tasks import LABELS, digest
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Qwen3_5ForConditionalGeneration,
)

LOGGER = logging.getLogger("long_stream")
SEEDS = (17, 29, 43)
ARMS = ("continue", "replay")
SCORE_SPLITS = ("validation", "test")


class EncodedStream:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.device = config["device"]
        self.max_length = config["max_length"]
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        if self.eos_id is None:
            raise ValueError("NATIVE_EOS: tokenizer must define an EOS token")
        if self.pad_id is None:
            self.pad_id = self.eos_id
        label_ids = {
            label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
        }
        if any(
            len(ids) != 1 or ids[0] in (self.eos_id, self.pad_id)
            for ids in label_ids.values()
        ):
            raise ValueError(
                f"LETTER_TOKENS: letters must have distinct single tokens: {label_ids}"
            )
        self.label_ids = {label: ids[0] for label, ids in label_ids.items()}
        if len(set(self.label_ids.values())) != 4:
            raise ValueError("LETTER_TOKENS: token identities must be distinct")
        for label, ids in label_ids.items():
            if (
                tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                != label
            ):
                raise ValueError(
                    "LETTER_ROUNDTRIP: decoded whole answer differs from letter"
                )
        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=4,
            eos_token_id=self.eos_id,
            pad_token_id=self.pad_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=False,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
        )

    def prompt_ids(self, row):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": render(row)}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if not ids or len(ids) + 4 > self.max_length:
            raise ValueError(
                f"CONTEXT_TRUNCATION: {row.id} needs {len(ids) + 4} tokens"
            )
        return list(ids)

    def batch(self, rows, training=False):
        sequences = [self.prompt_ids(row) for row in rows]
        targets = [[self.label_ids[row.label], self.eos_id] for row in rows]
        if training:
            sequences = [
                ids + target for ids, target in zip(sequences, targets, strict=True)
            ]
        width = max(map(len, sequences))
        inputs = {
            "input_ids": torch.tensor(
                [[self.pad_id] * (width - len(ids)) + ids for ids in sequences],
                device=self.device,
            ),
            "attention_mask": torch.tensor(
                [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences],
                device=self.device,
            ),
        }
        return inputs, torch.tensor(targets, device=self.device)


def strict_grade(token_ids, expected, tokenizer, eos_id, pad_id):
    ids = list(token_ids)
    terminated = eos_id in ids
    if terminated:
        stop = ids.index(eos_id)
        answer_ids = ids[:stop]
        trailing = ids[stop + 1 :]
        padding_only = all(token == pad_id for token in trailing)
        emitted = ids[: stop + 1]
    else:
        answer_ids = ids
        padding_only = True
        emitted = ids
    answer = tokenizer.decode(
        answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    compliant = terminated and padding_only and answer in LABELS
    return {
        "generated_ids": emitted,
        "raw_generation_ids": ids,
        "whole_answer": answer,
        "native_eos": terminated,
        "padding_only_after_eos": padding_only,
        "format_compliant": compliant,
        "correct": compliant and answer == expected,
    }


def generate_records(model, encoded, rows, batch_size):
    model.eval()
    records = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            inputs, _ = encoded.batch(chunk)
            model.generation_config = GenerationConfig.from_dict(
                encoded.generation_config.to_dict()
            )
            model.get_base_model().generation_config = model.generation_config
            generated = model.generate(
                **inputs, generation_config=model.generation_config
            )
            suffixes = generated[:, inputs["input_ids"].shape[1] :].cpu().tolist()
            for row, suffix in zip(chunk, suffixes, strict=True):
                records.append(
                    {
                        "id": row.id,
                        "task_id": row.task_id,
                        "group_id": row.group_id,
                        "split": row.split,
                        "label": row.label,
                        **strict_grade(
                            suffix,
                            row.label,
                            encoded.tokenizer,
                            encoded.eos_id,
                            encoded.pad_id,
                        ),
                    }
                )
    return records


def score_records(records):
    if not records:
        raise ValueError("EMPTY_EVALUATION: an accuracy requires held-out examples")
    count = len(records)
    correct = sum(record["correct"] for record in records)
    return {
        "n": count,
        "correct": correct,
        "accuracy": correct / count,
        "accuracy_wilson95": wilson(correct, count),
        "native_eos_rate": sum(record["native_eos"] for record in records) / count,
        "whole_answer_format_rate": sum(
            record["format_compliant"] for record in records
        )
        / count,
    }


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def evaluate_tasks(model, encoded, corpus, identities, config, directory):
    metrics = {}
    directory = Path(directory)
    for identity in identities:
        metrics[identity] = {}
        for split in SCORE_SPLITS:
            records = generate_records(
                model, encoded, corpus[identity][split], config["eval_batch_size"]
            )
            write_records(directory / f"{identity}.{split}.jsonl", records)
            metrics[identity][split] = score_records(records)
    write_json(directory / "metrics.json", metrics)
    return metrics


def base_hash(model):
    return digest(
        {
            name: tensor_hash(parameter)
            for name, parameter in model.named_parameters()
            if ".lora_" not in name
        }
    )


def training_invariants(model, optimizer, expected_step):
    if set(model.peft_config) != {"acquired"} or model.active_adapters != ["acquired"]:
        raise RuntimeError(
            "ADAPTER_HISTORY: exactly one persistent acquired adapter is required"
        )
    adapter = model.peft_config["acquired"]
    if adapter.r != 8 or adapter.rank_pattern:
        raise RuntimeError("ADAPTER_RANK: the entire stream must retain rank eight")
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if (
        not trainable
        or {id(parameter) for parameter in owned} != trainable
        or len(owned) != len(trainable)
    ):
        raise RuntimeError(
            "OPTIMIZER_OWNERSHIP: optimizer must own exactly the trainable adapter"
        )
    frozen = [
        parameter
        for name, parameter in model.named_parameters()
        if ".lora_" not in name
    ]
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in frozen
    ):
        raise RuntimeError("BASE_FREEZE: backbone is trainable or received gradients")
    if any(
        ".lora_" not in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise RuntimeError("TRAINABLE_BOUNDARY: non-adapter parameters are trainable")
    steps = [int(state["step"]) for state in optimizer.state.values()]
    if (
        expected_step and (len(steps) != len(owned) or set(steps) != {expected_step})
    ) or (not expected_step and steps):
        raise RuntimeError(
            f"OPTIMIZER_CONTINUITY: expected every parameter at step {expected_step}, got {set(steps)}"
        )
    return {
        "rank": adapter.r,
        "active_adapters": ["acquired"],
        "trainable_parameters": sum(parameter.numel() for parameter in owned),
        "optimizer_step": expected_step,
        "base_requires_grad": False,
        "base_has_gradients": False,
    }


def train_update(model, optimizer, encoded, rows, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = encoded.batch(rows, training=True)
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config["device"].startswith("cuda")
        else contextlib.nullcontext()
    )
    with precision:
        logits = (
            model(**inputs, use_cache=False, logits_to_keep=3).logits[:, -3:-1].float()
        )
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape(-1, 2)
        loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(
            "NONFINITE_LOSS: letter and EOS supervision returned NaN or infinity"
        )
    loss.backward()
    norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config["grad_clip"],
            error_if_nonfinite=True,
        )
    )
    if norm <= 0:
        raise RuntimeError("ZERO_GRADIENT: rank-eight adapter received no update")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "letter_loss": float(losses[:, 0].mean().detach()),
        "eos_loss": float(losses[:, 1].mean().detach()),
        "gradient_norm": norm,
        "input_tokens": int(inputs["attention_mask"].sum()),
        "padded_tokens": int(inputs["input_ids"].numel()),
        "supervised_letter_tokens": len(rows),
        "supervised_eos_tokens": len(rows),
    }


def train_stage(
    model, optimizer, encoded, current_rows, reservoir, arm, config, stage_index, output
):
    Path(output).mkdir(parents=True, exist_ok=True)
    identity = current_rows[0].task_id
    expected_step = stage_index * config["updates_per_stage"]
    before = training_invariants(model, optimizer, expected_step)
    before_adapter = adapter_hash(model)
    prior_rows = tuple(reservoir.rows) if reservoir is not None else ()
    trainable_objects = tuple(
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    )
    batches = stage_batches(current_rows, prior_rows, arm, config, identity)
    counts = Counter()
    unique_current = {}
    current_hash = hashlib.sha256()
    losses = []
    event(
        output,
        "STREAM_STAGE_START",
        arm=arm,
        stage=stage_index + 1,
        task_id=identity,
        invariants=before,
    )
    for step, (rows, roles) in enumerate(batches, 1):
        update = train_update(model, optimizer, encoded, rows, config)
        counts.update(roles)
        counts["updates"] += 1
        counts["examples"] += len(rows)
        for key in (
            "input_tokens",
            "padded_tokens",
            "supervised_letter_tokens",
            "supervised_eos_tokens",
        ):
            counts[key] += update[key]
        for row, role in zip(rows, roles, strict=True):
            if role == "new":
                unique_current.setdefault(row.id, row)
                current_hash.update(row.id.encode() + b"\n")
        losses.append(update["loss"])
        event(
            output,
            "STREAM_UPDATE",
            arm=arm,
            task_id=identity,
            stage=stage_index + 1,
            step=step,
            global_step=expected_step + step,
            sample_ids=[row.id for row in rows],
            roles=roles,
            **update,
        )
    after = training_invariants(
        model, optimizer, expected_step + config["updates_per_stage"]
    )
    if trainable_objects != tuple(
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ):
        raise RuntimeError(
            "ADAPTER_REPLACED: adapter parameter objects changed within the stream"
        )
    if adapter_hash(model) == before_adapter:
        raise RuntimeError("NO_TRAINING_CHANGE: adapter tensors did not change")
    buffer_before = (
        reservoir.storage(encoded, config["task_order"][:stage_index])
        if reservoir is not None
        else None
    )
    if reservoir is not None:
        reservoir.offer(unique_current.values())
    receipt = {
        "task_id": identity,
        "stage": stage_index + 1,
        "counts": {"new": 0, "replay": 0, **dict(counts)},
        "new_unique_examples": len(unique_current),
        "new_sequence_sha256": current_hash.hexdigest(),
        "initial_adapter_sha256": before_adapter,
        "final_adapter_sha256": adapter_hash(model),
        "mean_loss": sum(losses) / len(losses),
        "invariants_before": before,
        "invariants_after": after,
        "buffer_before": buffer_before,
        "buffer_after": reservoir.storage(encoded, config["task_order"][:stage_index])
        if reservoir is not None
        else None,
        "historical_training_source": "bounded reservoir only"
        if reservoir is not None
        else "none",
    }
    directory = Path(output) / "training" / arm / identity
    write_json(directory / "receipt.json", receipt)
    if reservoir is not None:
        write_json(directory / "buffer.json", reservoir.payload())
    event(
        output, "STREAM_STAGE_END", arm=arm, task_id=identity, counts=receipt["counts"]
    )
    return receipt


def save_checkpoint(model, optimizer, config, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        path / "adapters",
        selected_adapters=["acquired"],
        safe_serialization=True,
        save_embedding_layers=False,
    )
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    metadata = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "active_adapters": ["acquired"],
        "trainable_adapter": "acquired",
        "adapter_tensor_sha256": adapter_hash(model),
        "optimizer_sha256": digest(tree_hash(optimizer.state_dict())),
        "rank": 8,
        "config_sha256": digest(config),
        "files": {
            str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(path.rglob("*"))
            if file.is_file()
        },
    }
    write_json(path / "checkpoint.json", metadata)
    return metadata


def verify_reload(
    path,
    config,
    encoded,
    corpus,
    identities,
    preceding_directory,
    output_directory,
    expected_base,
):
    loaded, optimizer, metadata = checkpoint_load(path, config)
    if base_hash(loaded) != expected_base:
        raise RuntimeError(
            "RELOAD_BASE_CHANGED: checkpoint loading changed frozen backbone tensors"
        )
    evaluate_tasks(loaded, encoded, corpus, identities, config, output_directory)
    ids = []
    for identity in identities:
        for split in SCORE_SPLITS:
            name = f"{identity}.{split}.jsonl"
            before = [
                json.loads(line)
                for line in (Path(preceding_directory) / name).read_text().splitlines()
            ]
            after = [
                json.loads(line)
                for line in (Path(output_directory) / name).read_text().splitlines()
            ]
            if before != after:
                raise RuntimeError(
                    f"RELOAD_BEHAVIOR_CHANGED: full generated answer or EOS differs for {identity}/{split}"
                )
            ids.extend(record["id"] for record in before)
    proof = {
        "adapter_tensors_exact": True,
        "optimizer_state_exact": True,
        "base_all_tensors_exact": True,
        "whole_generation_records_exact": True,
        "example_ids": ids,
        "example_count": len(ids),
        "rank": metadata["rank"],
        "criterion": "full generated tokens, decoded answer, native EOS, and strict grade; no logit accuracy gate",
    }
    write_json(Path(path) / "behavioral-reload.json", proof)
    del loaded, optimizer
    gc.collect()
    return proof


def headroom_gate(initial, final, config):
    remaining = 1.0 - initial
    required = initial + config["gate_headroom_fraction"] * remaining
    eligible = remaining >= config["min_baseline_headroom"]
    acquired = final >= max(config["gate_min_accuracy"], required)
    return {
        "initial_accuracy": initial,
        "final_accuracy": final,
        "baseline_headroom": remaining,
        "required_accuracy": max(config["gate_min_accuracy"], required),
        "observed_gain": final - initial,
        "headroom_gain_fraction": (final - initial) / remaining if remaining else None,
        "sufficient_baseline_headroom": eligible,
        "passed": eligible and acquired,
        "reason": "insufficient_baseline_headroom"
        if not eligible
        else "acquired"
        if acquired
        else "not_acquired_at_fixed_budget",
    }


def summarize_matrix(initial, boundaries, pre_stage, task_order, split):
    per_task = {}
    for index, identity in enumerate(task_order):
        acquisition = boundaries[index][identity][split]["accuracy"]
        final = boundaries[-1][identity][split]["accuracy"]
        best = max(
            boundary[identity][split]["accuracy"] for boundary in boundaries[index:]
        )
        baseline = initial[identity][split]["accuracy"]
        previous = pre_stage[index][identity][split]["accuracy"]
        per_task[identity] = {
            "initial": baseline,
            "before_learning": previous,
            "after_learning": acquisition,
            "final": final,
            "forgetting": best - final,
            "backward_transfer": final - acquisition,
            "new_task_learning": acquisition - previous,
            "gain_over_initial": acquisition - baseline,
        }
    old = [per_task[identity] for identity in task_order[:-1]]
    return {
        "final_average_accuracy": sum(row["final"] for row in per_task.values())
        / len(per_task),
        "max_forgetting": max(row["forgetting"] for row in old),
        "mean_forgetting": sum(row["forgetting"] for row in old) / len(old),
        "backward_transfer": sum(row["backward_transfer"] for row in old) / len(old),
        "new_task_learning": sum(row["new_task_learning"] for row in per_task.values())
        / len(per_task),
        "per_task": per_task,
        "definitions": {
            "forgetting": "max accuracy from task acquisition through final boundary minus final accuracy",
            "backward_transfer": "mean final minus acquisition accuracy over tasks preceding the final task",
            "new_task_learning": "mean immediate post-stage minus immediate pre-stage accuracy",
        },
    }


def release_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_native_eos(model, encoded):
    native_eos = model.generation_config.eos_token_id
    if native_eos is None:
        native_eos = model.config.get_text_config().eos_token_id
    native_ids = native_eos if isinstance(native_eos, list) else [native_eos]
    tokenizer = encoded.tokenizer
    if encoded.eos_id not in tokenizer.all_special_ids:
        raise ValueError("NATIVE_EOS_MISMATCH: EOS is not a tokenizer special token")
    method = "model_generation_default"
    if encoded.eos_id not in native_ids:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Protocol check."},
                {"role": "assistant", "content": "A"},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if not rendered.rstrip().endswith(tokenizer.eos_token):
            raise ValueError(
                "NATIVE_EOS_MISMATCH: EOS does not close the assistant template"
            )
        method = "tokenizer_assistant_turn_terminator"
    return {
        "model_generation_default_ids": native_ids,
        "tokenizer_eos_id": encoded.eos_id,
        "tokenizer_eos_token": tokenizer.eos_token,
        "validated_by": method,
        "forced_eos": False,
    }


def run_study(config, output, corpus, encoded):
    output = Path(output)
    task_order = config["task_order"]
    first, final = task_order[0], task_order[-1]
    history_hash = corpus_digest(corpus)
    model, optimizer = new_model(config)
    activate(model, ["acquired"], "acquired")
    base_before = base_hash(model)
    write_json(output / "native-eos.json", validate_native_eos(model, encoded))
    initial = evaluate_tasks(
        model, encoded, corpus, task_order, config, output / "evaluation" / "initial"
    )
    shared_reservoir = Reservoir(config["buffer_capacity"], config["seed"])
    shared = train_stage(
        model,
        optimizer,
        encoded,
        corpus[first]["train"],
        shared_reservoir,
        "continue",
        config,
        0,
        output,
    )
    acquired_path = output / "evaluation" / "shared_A"
    acquired = evaluate_tasks(model, encoded, corpus, [first], config, acquired_path)
    shared_checkpoint = output / "checkpoints" / "shared_A"
    if base_hash(model) != base_before:
        raise RuntimeError(
            "BASE_CHANGED: first acquisition changed frozen backbone tensors"
        )
    save_checkpoint(model, optimizer, config, shared_checkpoint)
    del model, optimizer
    release_model()
    shared_proof = verify_reload(
        shared_checkpoint,
        config,
        encoded,
        corpus,
        [first],
        acquired_path,
        output / "reload" / "shared_A",
        base_before,
    )
    results = {
        "status": "running",
        "seed": config["seed"],
        "predeclared_seeds": config["predeclared_seeds"],
        "mode": config["mode"],
        "primary_score": "test whole-letter-answer plus native EOS exact accuracy",
        "config_sha256": digest(config),
        "corpus_sha256": history_hash,
        "initial": initial,
        "acquisition_gate": headroom_gate(
            initial[first]["validation"]["accuracy"],
            acquired[first]["validation"]["accuracy"],
            config,
        ),
        "shared_A_reload": shared_proof,
        "arms": {},
    }
    for arm in ARMS:
        model, optimizer, _ = checkpoint_load(shared_checkpoint, config)
        reservoir = shared_reservoir if arm == "replay" else None
        matrix = [acquired]
        pre_stage = [{first: initial[first]}]
        stages = [shared]
        adapter_objects = tuple(
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        )
        for index, identity in enumerate(task_order[1:], 1):
            before = evaluate_tasks(
                model,
                encoded,
                corpus,
                [identity],
                config,
                output / "evaluation" / arm / f"before_{index + 1:02}",
            )
            pre_stage.append(before)
            stages.append(
                train_stage(
                    model,
                    optimizer,
                    encoded,
                    corpus[identity]["train"],
                    reservoir,
                    arm,
                    config,
                    index,
                    output,
                )
            )
            matrix.append(
                evaluate_tasks(
                    model,
                    encoded,
                    corpus,
                    task_order[: index + 1],
                    config,
                    output / "evaluation" / arm / f"boundary_{index + 1:02}",
                )
            )
            if adapter_objects != tuple(
                id(parameter)
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError(
                    "ADAPTER_REPLACED: new adapter parameter objects appeared between stages"
                )
            write_json(
                output / "matrices" / f"{arm}.json",
                {
                    "initial": initial,
                    "before_learning": pre_stage,
                    "boundaries": matrix,
                },
            )
        base_after = base_hash(model)
        if base_before != base_after:
            raise RuntimeError(f"BASE_CHANGED: frozen backbone changed in {arm}")
        checkpoint = output / "checkpoints" / arm
        save_checkpoint(model, optimizer, config, checkpoint)
        del model, optimizer
        release_model()
        proof = verify_reload(
            checkpoint,
            config,
            encoded,
            corpus,
            task_order,
            output / "evaluation" / arm / f"boundary_{len(task_order):02}",
            output / "reload" / arm,
            base_before,
        )
        budget = Counter()
        for stage in stages:
            budget.update(stage["counts"])
        results["arms"][arm] = {
            "stages": stages,
            "counts": dict(budget),
            "metrics": {
                split: summarize_matrix(initial, matrix, pre_stage, task_order, split)
                for split in SCORE_SPLITS
            },
            "reload": proof,
            "base_all_tensors_unchanged": base_before == base_after,
            "base_tensor_sha256": base_after,
            "checkpoint": str(checkpoint),
            "persistent_adapter_rank": 8,
            "resident_replay_buffer": reservoir.storage(encoded, task_order[:-1])
            if reservoir is not None
            else {
                "examples": 0,
                "serialized_bytes_including_rng": 0,
                "prompt_label_eos_tokens": 0,
                "old_ids": [],
            },
        }
        write_json(output / "results.partial.json", results)
    model, optimizer = new_model(config)
    reference_initial = evaluate_tasks(
        model,
        encoded,
        corpus,
        [final],
        config,
        output / "evaluation" / "fresh_final" / "initial",
    )
    if reference_initial[final] != initial[final]:
        raise RuntimeError(
            "FRESH_BASELINE_CHANGED: fresh final-task reference differs from its task-specific baseline"
        )
    reference_training = train_stage(
        model,
        optimizer,
        encoded,
        corpus[final]["train"],
        None,
        "continue",
        config,
        0,
        output / "fresh_final",
    )
    reference_path = output / "evaluation" / "fresh_final" / "trained"
    reference = evaluate_tasks(model, encoded, corpus, [final], config, reference_path)
    if base_hash(model) != base_before:
        raise RuntimeError(
            "REFERENCE_BASE_CHANGED: fresh reference modified the backbone"
        )
    reference_checkpoint = output / "checkpoints" / "fresh_final"
    save_checkpoint(model, optimizer, config, reference_checkpoint)
    del model, optimizer
    release_model()
    reference_proof = verify_reload(
        reference_checkpoint,
        config,
        encoded,
        corpus,
        [final],
        reference_path,
        output / "reload" / "fresh_final",
        base_before,
    )
    results["fresh_final_reference"] = {
        "task_id": final,
        "initial": reference_initial,
        "trained": reference,
        "training": reference_training,
        "reload": reference_proof,
        "gate": headroom_gate(
            reference_initial[final]["validation"]["accuracy"],
            reference[final]["validation"]["accuracy"],
            config,
        ),
    }
    continue_counts = results["arms"]["continue"]["counts"]
    replay_counts = results["arms"]["replay"]["counts"]
    if (
        continue_counts["updates"] != replay_counts["updates"]
        or continue_counts["examples"] != replay_counts["examples"]
    ):
        raise RuntimeError(
            "BUDGET_MISMATCH: compared arms differ in updates or total examples"
        )
    if corpus_digest(corpus) != history_hash:
        raise RuntimeError(
            "HISTORY_MUTATION: old task examples or labels changed during the stream"
        )
    results["exposure_comparison"] = {
        "contract": config["exposure_matching"],
        "updates_per_arm": continue_counts["updates"],
        "examples_per_arm": continue_counts["examples"],
        "continue_new_examples": continue_counts["new"],
        "replay_new_examples": replay_counts["new"],
        "replay_old_examples": replay_counts["replay"],
        "new_exposure_difference_replay_minus_continue": replay_counts["new"]
        - continue_counts["new"],
        "current_sampling": "identical seeded new-example stream prefix; replay consumes a shorter prefix per stage",
        "shared_A_physical_execution_once": True,
        "fresh_reference_counts_separate": reference_training["counts"],
    }
    results["replay_minus_continue"] = {
        split: {
            metric: results["arms"]["replay"]["metrics"][split][metric]
            - results["arms"]["continue"]["metrics"][split][metric]
            for metric in (
                "final_average_accuracy",
                "max_forgetting",
                "backward_transfer",
                "new_task_learning",
            )
        }
        for split in SCORE_SPLITS
    }
    qualified = (
        results["acquisition_gate"]["passed"]
        and results["fresh_final_reference"]["gate"]["passed"]
    )
    results["behavioral_gates_passed"] = qualified
    results["claim_eligible"] = qualified and config["mode"] == "study"
    results["status"] = "completed"
    results["interpretation"] = (
        "CPU schedule qualification only; random tiny model is not a capability result"
        if config["mode"] == "cpu_qualification"
        else "Bounded per-seed generated-answer continual-learning comparison; combine all predeclared seeds"
        if qualified
        else "Descriptive stream measurements only: acquisition or fresh final-task learning lacked the required task-specific headroom gain"
    )
    results["history_immutable"] = True
    return results


def validate_config(config):
    if config["mode"] not in ("study", "cpu_qualification"):
        raise ValueError("STUDY_MODE: expected study or cpu_qualification")
    if config["seed"] not in SEEDS or tuple(config["predeclared_seeds"]) != SEEDS:
        raise ValueError("FIXED_SEEDS: seeds must be predeclared as 17, 29, 43")
    if (
        tuple(config["arms"]) != ARMS
        or config["rank"] != 8
        or config["buffer_capacity"] != 64
    ):
        raise ValueError(
            "STUDY_CONTRACT: requires continue, replay, rank eight and a total buffer of 64"
        )
    if tuple(config["task_order"]) != TASK_IDS[: len(config["task_order"])]:
        raise ValueError(
            "FIXED_TASKS: task identities and order must follow the frozen registry"
        )
    if config["mode"] == "study" and len(config["task_order"]) != 8:
        raise ValueError("LONG_HISTORY: production study requires all eight stages")
    if config["mode"] == "cpu_qualification" and (
        config["device"] != "cpu" or len(config["task_order"]) not in (2, 8)
    ):
        raise ValueError(
            "CPU_QUALIFICATION: only local two-stage or eight-stage controls are permitted"
        )
    if config["device"] not in ("cpu", "cuda:0"):
        raise ValueError("DEVICE: expected cpu or the parent-assigned cuda:0")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("MODEL_REVISION: use a pinned forty-character revision")
    snapshot = Path(config["model_path"])
    if not snapshot.is_absolute() or snapshot.name != config["model_revision"]:
        raise ValueError(
            "MODEL_SNAPSHOT: use an absolute local snapshot ending in the pinned revision"
        )
    for key in ("batch_size", "eval_batch_size", "updates_per_stage", "max_length"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"POSITIVE_INTEGER: {key}")
    if (
        type(config["replay_per_batch"]) is not int
        or not 0 < config["replay_per_batch"] < config["batch_size"]
    ):
        raise ValueError(
            "REPLAY_BATCH: replay must replace a proper subset of the total batch"
        )
    if config["exposure_matching"] != "match_total_and_log_new_exposure_difference":
        raise ValueError(
            "EXPOSURE_CONTRACT: fixed total examples and explicit new exposure difference required"
        )
    if (
        config["max_new_tokens"] != 4
        or config["gate_metric"] != "whole_answer_native_eos_accuracy"
    ):
        raise ValueError(
            "GENERATION_CONTRACT: maximum four unforced tokens and whole-answer native-EOS grading required"
        )
    for key in ("gate_min_accuracy", "gate_headroom_fraction", "min_baseline_headroom"):
        if not math.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"HEADROOM_GATE: invalid {key}")
    for key in ("learning_rate", "grad_clip", "lora_alpha"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"TRAINING_VALUE: invalid {key}")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("TRAINING_VALUE: invalid weight_decay")


def prepare_output(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "packages.txt", "execution.json", "task.json", "run.log"}
    entries = list(output.iterdir())
    if any(
        path.is_symlink()
        or not (
            (path.name in allowed and path.is_file())
            or (path.name == "attempts" and path.is_dir())
        )
        for path in entries
    ):
        raise RuntimeError(
            "OUTPUT_ALREADY_USED: only dispatcher config/packages/execution/task/log files may preexist"
        )
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(
                "DISPATCH_CONFIG_MISMATCH: output config differs from requested protocol"
            )
    else:
        with config_path.open("x") as handle:
            handle.write(
                json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
    dispatcher_files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in entries
        if path.is_file()
    }
    with (output / "protocol.json").open("x") as handle:
        json.dump(
            {
                "config": config,
                "config_sha256": digest(config),
                "tasks": [
                    asdict(task)
                    for task in TASKS
                    if task.identity in config["task_order"]
                ],
                "selection": "fixed protocol before all outcomes; no adaptive schedule, seed, rank or hyperparameter search",
                "gates": "validation whole-answer plus native EOS; report complete stream regardless of gate outcome",
                "replay": "Algorithm R over each actually exposed current training example once per task, admitted only after its stage",
                "scope": "synthetic routing with shared task structure and hidden station-specific mappings",
                "dispatcher_files_sha256": dispatcher_files,
            },
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    source = Path(__file__).resolve().parent
    code = output / "code"
    code.mkdir()
    for name in (
        "long_stream.py",
        "stream_tasks.py",
        "run.py",
        "tasks.py",
        "requirements.txt",
    ):
        shutil.copy2(source / name, code / name)
    write_json(
        output / "source.json",
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(code.iterdir())
        },
    )
    return dispatcher_files


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [long-stream] %(message)s"
    )
    config = json.loads(args.config.read_text())
    validate_config(config)
    dispatcher = prepare_output(config, args.output_dir)
    try:
        torch.use_deterministic_algorithms(True)
        if config["device"] == "cpu":
            torch.set_num_threads(1)
        corpus = build_corpus(config)
        audit = audit_corpus(corpus, config)
        write_json(args.output_dir / "data-audit.json", audit)
        for identity, splits in corpus.items():
            for split, rows in splits.items():
                write_records(
                    args.output_dir / "data" / f"{identity}.{split}.jsonl",
                    [asdict(row) for row in rows],
                )
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True
        )
        encoded = EncodedStream(tokenizer, config)
        lengths = [
            len(encoded.prompt_ids(row))
            for splits in corpus.values()
            for rows in splits.values()
            for row in rows
        ]
        write_json(
            args.output_dir / "runtime.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "device": config["device"],
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "local_files_only": True,
                "deterministic_algorithms": True,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "peft",
                        "accelerate",
                        "safetensors",
                    )
                },
                "forward_signature": str(
                    inspect.signature(Qwen3_5ForConditionalGeneration.forward)
                ),
                "generation_signature": str(
                    inspect.signature(Qwen3_5ForConditionalGeneration.generate)
                ),
                "generation": encoded.generation_config.to_diff_dict(),
                "label_ids": encoded.label_ids,
                "native_eos_id": encoded.eos_id,
                "prompt_length_max": max(lengths),
                "no_truncation": True,
                "full_vocabulary_training_and_generation": True,
                "supervision": "one letter token and one native EOS token; equal loss weights",
            },
        )
        write_json(args.output_dir / "status.json", {"status": "running"})
        results = run_study(config, args.output_dir, corpus, encoded)
        if any(
            hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
            != expected
            for name, expected in dispatcher.items()
            if name not in ("execution.json", "run.log")
        ):
            raise RuntimeError(
                "DISPATCHER_FILE_CHANGED: immutable dispatcher input was modified"
            )
        results["dispatcher_inputs_preserved"] = True
        write_json(args.output_dir / "results.json", results)
        write_json(
            args.output_dir / "status.json",
            {
                "status": "completed",
                "claim_eligible": results["claim_eligible"],
                "mode": config["mode"],
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "results": str(args.output_dir / "results.json"),
                    "claim_eligible": results["claim_eligible"],
                }
            )
        )
    except Exception as error:
        write_json(
            args.output_dir / "status.json",
            {
                "status": "failed",
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        LOGGER.exception("LONG_STREAM_FAILURE output=%s", args.output_dir)
        raise


if __name__ == "__main__":
    main()
