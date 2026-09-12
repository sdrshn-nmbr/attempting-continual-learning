import argparse
import hashlib
import json
import os
import signal
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from config import Config
from model import ModelIO, StopFlag, load_model
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from tasks import Example, execute, grade, input_pool, stable_seed

CONFIG_SHA256 = "cf172f4ef9ef30296c22d0436dc6fcc6b010e3c7c088fb9b8ca0310fef8d34dc"
PROGRAMS = (("dax", "wug"), ("wug", "dax"), ("fep", "kiv"), ("kiv", "fep"))
CONDITIONS = ("initial", "after_symbol_map")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def emit(output, event, **values):
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


def build_examples():
    return [
        Example(
            "permutation" if program[0] in ("dax", "wug") else "symbol_map",
            "chain_diagnostic",
            inputs,
            program,
        )
        for program in PROGRAMS
        for inputs in input_pool(37)[160:192]
    ]


def input_manifest(examples):
    pool = input_pool(37)
    roots = {x.inputs for x in examples}
    if roots & set(pool[:160]) or roots & set(pool[192:224]):
        raise ValueError("CHAIN_ROOT_PARTITION_OVERLAP")
    return {
        "seed": 37,
        "root_input_pool_range": [160, 192],
        "reserved_future_range": [192, 224],
        "examples": [asdict(x) for x in examples],
        "order_sensitive_examples": sum(
            execute(x.inputs, x.program) != execute(x.inputs, x.program[::-1])
            for x in examples
        ),
        "boundary": "Roots are unused by earlier train/dev/evaluation. Intermediate primitive inputs can overlap earlier inputs and are logged; this is a zero-update diagnostic.",
    }


def parse_output(text):
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(x) is not int or not 0 <= x <= 3 for x in value)
    ):
        return None
    return tuple(value)


def measure_example(example, query):
    one_shot = query(example)
    model_steps, oracle_steps = [], []
    current = example.inputs
    for operation in example.program:
        primitive = Example(example.task, "chain_primitive", current, (operation,))
        response = query(primitive)
        model_steps.append(response)
        current = parse_output(response["text"])
        if current is None:
            break
    for index, operation in enumerate(example.program):
        gold_input = tuple(execute(example.inputs, example.program[:index]))
        oracle_steps.append(
            query(Example(example.task, "chain_primitive", gold_input, (operation,)))
        )
    expected = tuple(execute(example.inputs, example.program))
    return {
        "key": example.key,
        "task": example.task,
        "inputs": list(example.inputs),
        "program": list(example.program),
        "expected": list(expected),
        "order_sensitive": execute(example.inputs, example.program)
        != execute(example.inputs, example.program[::-1]),
        "one_shot": {"correct": one_shot["correct"], "call": one_shot["key"]},
        "model_chain": {
            "correct": current == expected and len(model_steps) == len(example.program),
            "all_steps_correct": len(model_steps) == len(example.program)
            and all(x["correct"] for x in model_steps),
            "invalid": current is None,
            "calls": [x["key"] for x in model_steps],
        },
        "oracle_intermediate": {
            "correct": oracle_steps[-1]["correct"],
            "all_steps_correct": all(x["correct"] for x in oracle_steps),
            "calls": [x["key"] for x in oracle_steps],
        },
    }


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups["all"].append(row)
        groups[",".join(row["program"])].append(row)
        if row["order_sensitive"]:
            groups["order_sensitive"].append(row)
    result = {}
    for name, rows in groups.items():
        panel = {"total": len(rows)}
        for method in ("one_shot", "model_chain", "oracle_intermediate"):
            correct = sum(row[method]["correct"] for row in rows)
            panel[method] = {"correct": correct, "accuracy": correct / len(rows)}
            if method != "one_shot":
                panel[method]["all_steps_correct"] = sum(
                    row[method]["all_steps_correct"] for row in rows
                )
        result[name] = panel
    return result


def evaluate_condition(io, examples, output, condition):
    cache, records = {}, []
    train_inputs = set(input_pool(37)[4:20])
    previous_inputs = set(input_pool(37)[:160])
    budget = {
        "logical_calls": 0,
        "unique_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "primitive_train_input_overlap": 0,
        "primitive_previous_input_overlap": 0,
    }

    def query(example):
        budget["logical_calls"] += 1
        key = json.dumps([example.program, example.inputs], separators=(",", ":"))
        if key in cache:
            return cache[key]
        prefix = io.prefix(example, context="cue_only", privileged=False)
        tokens, generated_count = io.generate(
            prefix,
            privileged=False,
            sample=False,
            seed=stable_seed(37, "chain_control", key),
            max_tokens=io.config.max_new_tokens,
        )
        text = io.decode(tokens)
        row = {
            "key": key,
            "condition": condition,
            "program": list(example.program),
            "inputs": list(example.inputs),
            "prompt_token_ids": prefix.cpu().tolist(),
            "generated_token_ids": tokens.cpu().tolist(),
            "generated_tokens": generated_count,
            "text": text,
            "hit_token_cap": generated_count == io.config.max_new_tokens
            and int(tokens[-1]) not in io.eos_ids,
            **grade(text, example),
        }
        cache[key] = row
        budget["unique_calls"] += 1
        budget["prompt_tokens"] += len(prefix)
        budget["generated_tokens"] += generated_count
        budget["primitive_train_input_overlap"] += int(
            len(example.program) == 1 and example.inputs in train_inputs
        )
        budget["primitive_previous_input_overlap"] += int(
            len(example.program) == 1 and example.inputs in previous_inputs
        )
        with (output / f"{condition}_calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        return row

    for example in examples:
        records.append(measure_example(example, query))
        if len(records) % 16 == 0:
            emit(
                output,
                "chain_progress",
                condition=condition,
                completed=len(records),
                **budget,
            )
    if (
        budget["logical_calls"] > 5 * len(examples)
        or budget["unique_calls"] > budget["logical_calls"]
    ):
        raise RuntimeError("CHAIN_CALL_BUDGET_VIOLATION")
    result = {"records": records, "measurements": aggregate(records), "budget": budget}
    write_json(output / f"{condition}.json", result)
    return result


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(
            name.encode()
            + b"\0"
            + str(value.dtype).encode()
            + b"\0"
            + str(tuple(value.shape)).encode()
            + b"\0"
        )
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def base_hash(model):
    return tensor_hash(
        {
            name: value
            for name, value in model.state_dict().items()
            if "lora_" not in name
        }
    )


def verify_files(recipe):
    files = {spec["path"]: spec["sha256"] for spec in recipe["checkpoints"].values()}
    files[str(Path(recipe["model_config"]["model_path"]) / "config.json")] = recipe[
        "base_config_sha256"
    ]
    for path, expected in files.items():
        if file_hash(path) != expected:
            raise ValueError(f"CHAIN_INPUT_HASH_MISMATCH: {path}")
    return files


def evaluate_loaded(recipe, config, model, tokenizer, runtime, output, examples):
    original_files = verify_files(recipe)
    original_base = base_hash(model)
    stop = StopFlag()
    signal.signal(signal.SIGTERM, stop.request)
    signal.signal(signal.SIGINT, stop.request)
    io = ModelIO(model, tokenizer, config, {}, stop)
    panels = {}
    for condition in CONDITIONS:
        spec = recipe["checkpoints"][condition]
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError(f"CHAIN_ADAPTER_HASH_MISMATCH: {condition}")
        state = load_file(spec["path"], device="cpu")
        set_peft_model_state_dict(model, state)
        actual = get_peft_model_state_dict(model, save_embedding_layers=False)
        if set(actual) != set(state) or any(
            not torch.equal(actual[key].detach().cpu(), state[key]) for key in state
        ):
            raise RuntimeError(f"CHAIN_ADAPTER_RELOAD_MISMATCH: {condition}")
        model.requires_grad_(False).eval()
        before_adapter = tensor_hash(actual)
        emit(
            output,
            "chain_adapter_loaded",
            condition=condition,
            file_sha256=spec["sha256"],
            tensor_sha256=before_adapter,
        )
        panels[condition] = evaluate_condition(io, examples, output, condition)
        after_adapter = tensor_hash(
            get_peft_model_state_dict(model, save_embedding_layers=False)
        )
        if before_adapter != after_adapter or original_base != base_hash(model):
            raise RuntimeError("CHAIN_MODEL_MUTATED_DURING_EVALUATION")
        panels[condition]["identity"] = {
            "file_sha256": spec["sha256"],
            "tensor_sha256": before_adapter,
            "base_sha256": original_base,
            "base_and_adapter_unchanged": True,
        }
        write_json(output / f"{condition}.json", panels[condition])
    if verify_files(recipe) != original_files:
        raise RuntimeError("CHAIN_DISK_INPUTS_MUTATED")
    initial = panels["initial"]["measurements"]["all"]
    final = panels["after_symbol_map"]["measurements"]["all"]
    gate = {
        "final_chain_accuracy_at_least_090": final["model_chain"]["accuracy"] >= 0.90,
        "chain_gain_over_one_shot_at_least_025": final["model_chain"]["accuracy"]
        - final["one_shot"]["accuracy"]
        >= 0.25,
        "chain_gain_over_initial_at_least_025": final["model_chain"]["accuracy"]
        - initial["model_chain"]["accuracy"]
        >= 0.25,
    }
    result = {
        "status": "completed",
        "runtime": runtime,
        "protocol": recipe["protocol"],
        "panels": panels,
        "optimizer_updates": 0,
        "next_step_gate": {
            "checks": gate,
            "passed": all(gate.values()),
            "training_started": False,
        },
        "interpretation": "Model-only chaining uses extra calls and a fixed external controller. Oracle intermediates are privileged diagnostic inputs. Neither is learned one-shot composition or additional weight learning.",
    }
    write_json(output / "result.json", result)
    emit(
        output,
        "chain_completed",
        examples_per_condition=len(examples),
        measurements={name: panel["measurements"] for name, panel in panels.items()},
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", dest="output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "events.jsonl").exists():
        raise ValueError("CHAIN_OUTPUT_ALREADY_USED")
    try:
        if file_hash(args.config) != CONFIG_SHA256:
            raise ValueError("CHAIN_FROZEN_CONFIG_MISMATCH")
        recipe = json.loads(args.config.read_text())
        files = verify_files(recipe)
        payload = dict(recipe["model_config"])
        payload["arms"] = tuple(payload["arms"])
        config = Config(**payload).validate()
        examples = build_examples()
        write_json(args.output / "input_manifest.json", input_manifest(examples))
        emit(
            args.output,
            "chain_inputs_verified",
            files=files,
            examples=len(examples),
            optimizer_updates=0,
        )
        model, tokenizer, runtime = load_model(config)
        evaluate_loaded(
            recipe, config, model, tokenizer, runtime, args.output, examples
        )
    except Exception as error:
        emit(
            args.output,
            "chain_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
