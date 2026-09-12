import math
import time

import torch
from data import EVAL_SPLITS
from modeling import collate, synchronize


def score_output(output, answer):
    prediction = output.strip().split()
    target = answer.split()
    denominator = max(len(prediction), len(target))
    return {
        "exact": prediction == target,
        "symbol_accuracy": sum(left == right for left, right in zip(prediction, target))
        / denominator,
        "valid_format": bool(prediction)
        and all(
            len(value) == 1 and value.isascii() and value.isdigit()
            for value in prediction
        ),
    }


def wilson_interval(successes, count):
    z = 1.96
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


@torch.inference_mode()
def evaluate(
    model, tokenizer, encoded, dataset, config, device, stop, emit, splits=EVAL_SPLITS
):
    if any(split not in {*EVAL_SPLITS, "calibration"} for split in splits):
        raise ValueError("[evaluation] only heldout/calibration splits are allowed")
    was_training = model.training
    model.eval()
    result = {}
    start = time.perf_counter()
    try:
        for task in range(config.data.num_tasks):
            by_split = {}
            for split in splits:
                examples = dataset.select(task, split)
                predictions = []
                nll_sum = 0.0
                target_tokens = 0
                generated_tokens = 0
                for offset in range(0, len(examples), config.pilot.eval_batch_size):
                    stop.raise_if_requested()
                    chunk = [
                        encoded[example.example_id]
                        for example in examples[
                            offset : offset + config.pilot.eval_batch_size
                        ]
                    ]
                    supervised = collate(chunk, tokenizer.pad_token_id, device)
                    count = sum(len(item.target_ids) for item in chunk)
                    loss = model(**supervised, use_cache=False).loss
                    if loss is None or not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"[evaluation] non-finite heldout NLL: task={task}, split={split}"
                        )
                    nll_sum += loss.item() * count
                    target_tokens += count
                    prompts = collate(
                        chunk, tokenizer.pad_token_id, device, generation=True
                    )
                    generated = model.generate(
                        **prompts,
                        do_sample=False,
                        max_new_tokens=config.pilot.max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )[:, prompts["input_ids"].shape[1] :]
                    for item, tokens in zip(chunk, generated):
                        token_ids = tokens.tolist()
                        if tokenizer.eos_token_id in token_ids:
                            token_ids = token_ids[
                                : token_ids.index(tokenizer.eos_token_id)
                            ]
                        generated_tokens += len(token_ids)
                        output = tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        predictions.append(
                            {
                                "example_id": item.example.example_id,
                                "output": output,
                                "answer": item.example.answer,
                                **score_output(output, item.example.answer),
                            }
                        )
                    emit(
                        "eval_batch",
                        eval_task=task,
                        eval_split=split,
                        completed=min(offset + len(chunk), len(examples)),
                        total=len(examples),
                    )
                exact = sum(prediction["exact"] for prediction in predictions)
                by_split[split] = {
                    "n": len(predictions),
                    "exact_correct": exact,
                    "exact_match": exact / len(predictions),
                    "exact_match_wilson95": wilson_interval(exact, len(predictions)),
                    "symbol_accuracy": sum(
                        prediction["symbol_accuracy"] for prediction in predictions
                    )
                    / len(predictions),
                    "valid_format_rate": sum(
                        prediction["valid_format"] for prediction in predictions
                    )
                    / len(predictions),
                    "answer_nll": nll_sum / target_tokens,
                    "supervised_tokens": target_tokens,
                    "generated_tokens": generated_tokens,
                    "predictions": predictions,
                }
            result[str(task)] = by_split
    finally:
        model.train(was_training)
    synchronize(device)
    return {
        "tasks": result,
        "seconds": time.perf_counter() - start,
        "gradients_enabled": False,
    }


def summarize_method(boundaries, num_tasks):
    result = {
        "per_task": [],
        "final": {},
        "interpretation": "Acquisition, retention and transfer are reported separately; no gain is assumed.",
    }
    complete = {
        entry["task"]: entry for entry in boundaries if entry["phase"] == "after"
    }
    before = {
        entry["task"]: entry for entry in boundaries if entry["phase"] == "before"
    }
    for task in sorted(complete):
        if task not in before:
            raise ValueError(
                f"[metrics] missing before-task evaluation for task {task}"
            )
        by_split = {}
        for split in EVAL_SPLITS:
            prior = before[task]["evaluation"]["tasks"][str(task)][split]
            acquired = complete[task]["evaluation"]["tasks"][str(task)][split]
            old = []
            for previous in range(task):
                now = complete[task]["evaluation"]["tasks"][str(previous)][split]
                own = complete[previous]["evaluation"]["tasks"][str(previous)][split]
                old.append(
                    {
                        "task": previous,
                        "accuracy": now["exact_match"],
                        "backward_transfer": now["exact_match"] - own["exact_match"],
                        "nll_change_since_acquisition": now["answer_nll"]
                        - own["answer_nll"],
                    }
                )
            by_split[split] = {
                "new_task_gain": acquired["exact_match"] - prior["exact_match"],
                "new_task_nll_reduction": prior["answer_nll"] - acquired["answer_nll"],
                "before_accuracy": prior["exact_match"],
                "after_accuracy": acquired["exact_match"],
                "retention": old,
                "mean_backward_transfer": sum(item["backward_transfer"] for item in old)
                / len(old)
                if old
                else None,
            }
        result["per_task"].append({"task": task, "splits": by_split})
    if len(complete) == num_tasks:
        final_eval = complete[num_tasks - 1]["evaluation"]["tasks"]
        for split in EVAL_SPLITS:
            deltas = [
                final_eval[str(task)][split]["exact_match"]
                - complete[task]["evaluation"]["tasks"][str(task)][split]["exact_match"]
                for task in range(num_tasks - 1)
            ]
            forgetting = []
            for task in range(num_tasks - 1):
                best = max(
                    complete[stage]["evaluation"]["tasks"][str(task)][split][
                        "exact_match"
                    ]
                    for stage in range(task, num_tasks)
                )
                forgetting.append(best - final_eval[str(task)][split]["exact_match"])
            result["final"][split] = {
                "mean_accuracy": sum(
                    final_eval[str(task)][split]["exact_match"]
                    for task in range(num_tasks)
                )
                / num_tasks,
                "mean_backward_transfer": sum(deltas) / len(deltas),
                "mean_forgetting_from_best": sum(forgetting) / len(forgetting),
                "mean_new_task_gain": sum(
                    entry["splits"][split]["new_task_gain"]
                    for entry in result["per_task"]
                )
                / num_tasks,
            }
    return result
