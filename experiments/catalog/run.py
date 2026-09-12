import argparse
import csv
import importlib.metadata
import json
import math
import random
import traceback
from dataclasses import replace
from pathlib import Path

import torch
from common import emit, entry_for, read_json, sha256_file, utc_now, write_json
from data import prepare_dataset, prompt_hash
from factory import load_base, load_portal, verify_library
from loader_guard import Channel, write_once
from peft import PeftModel
from portallib import PortalEvaluator, PortalModel


class EvidenceEvaluator(PortalEvaluator):
    def __init__(self, *, output, condition, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.condition = condition
        self.records = []

    def _score_rows(self, base, rows):
        scores, gold_nll, gold_tokens = super()._score_rows(base, rows)
        if not math.isfinite(gold_nll) or gold_tokens <= 0:
            raise ValueError(
                f"nonfinite or empty likelihood evidence: {self.condition}"
            )
        with self.output.open("a") as handle:
            for row, choices in zip(rows, scores, strict=True):
                if not all(math.isfinite(score) for score in choices):
                    raise ValueError(
                        f"invalid candidate score: {self.condition}/{row.task}"
                    )
                prediction = max(range(len(choices)), key=choices.__getitem__)
                prompt_tokens = len(
                    base.tokenizer(
                        row.prompt.rstrip(), add_special_tokens=True
                    ).input_ids
                )
                record = {
                    "condition": self.condition,
                    "task": row.task,
                    "prompt_sha256": prompt_hash(row),
                    "scores": choices,
                    "prediction": prediction,
                    "gold_idx": row.gold_idx,
                    "correct": prediction == row.gold_idx,
                    "prompt_tokens": prompt_tokens,
                    "truncated_prompt_tokens": max(0, prompt_tokens - self.max_prompt),
                }
                self.records.append(record)
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        emit(
            "task_scored",
            condition=self.condition,
            task=rows[0].task,
            examples=len(rows),
            gold_nll=gold_nll / gold_tokens,
        )
        return scores, gold_nll, gold_tokens


def paired_statistics(base, adapted, seed):
    left = {(row["task"], row["prompt_sha256"]): row for row in base}
    right = {(row["task"], row["prompt_sha256"]): row for row in adapted}
    if set(left) != set(right) or len(left) != len(base) or len(right) != len(adapted):
        raise ValueError("paired evaluation rows differ or contain duplicates")
    by_task = {}
    improved = regressed = 0
    for key in left:
        difference = int(right[key]["correct"]) - int(left[key]["correct"])
        by_task.setdefault(key[0], []).append(difference)
        improved += difference == 1
        regressed += difference == -1
    rng = random.Random(seed)
    samples = sorted(
        sum(
            sum(rng.choices(values, k=len(values))) / len(values)
            for values in by_task.values()
        )
        / len(by_task)
        for _ in range(4000)
    )
    discordant = improved + regressed
    p_value = min(
        1.0,
        2
        * sum(
            math.comb(discordant, index)
            for index in range(min(improved, regressed) + 1)
        )
        / 2**discordant,
    )
    return {
        "paired_examples": len(left),
        "improved": improved,
        "regressed": regressed,
        "macro_accuracy_delta": sum(
            sum(values) / len(values) for values in by_task.values()
        )
        / len(by_task),
        "paired_stratified_bootstrap_95ci": [samples[100], samples[3899]],
        "exact_mcnemar_two_sided_p": p_value,
        "interpretation": "exploratory screen; interval is within-model and unadjusted for seven model comparisons",
    }


def roundtrip_portal(portal, output, tasks):
    destination = output / "native_checkpoint"
    portal.save_pretrained(destination)
    restored = PortalModel.from_pretrained(
        destination, local_files_only=True, device="cpu", dtype=torch.float32
    )
    restored.requires_grad_(False)
    state_equal = all(
        torch.equal(tensor, restored.state_dict()[name])
        for name, tensor in portal.state_dict().items()
    )
    factors_equal = True
    for task in tasks:
        original, reloaded = portal.generate(task), restored.generate(task)
        factors_equal &= set(original) == set(reloaded) and all(
            torch.equal(original[key][0], reloaded[key][0])
            and torch.equal(original[key][1], reloaded[key][1])
            for key in original
        )
    if not state_equal or not factors_equal:
        raise ValueError("native artifact roundtrip changed state or generated factors")
    return {
        "state_exact_equal": state_equal,
        "generated_factors_exact_equal": bool(factors_equal),
        "tasks": list(tasks),
        "checkpoint_sha256": sha256_file(destination / "model.safetensors"),
    }


def peft_reload_check(base, portal, dataset, task, output, config):
    evaluator = EvidenceEvaluator(
        output=output / "reload_predictions.jsonl",
        condition="native",
        max_prompt=config["max_prompt"],
        batch_size=config["batch_size"],
    )
    evaluator.evaluate(base, dataset, tasks=(task,), portal=portal, max_examples=1)
    native_scores = evaluator.records[-1]["scores"]
    exported = output / "exported_peft"
    portal.export_peft(task, exported)
    wrapped = PeftModel.from_pretrained(
        base.model, exported, is_trainable=False, autocast_adapter_dtype=False
    )
    wrapped.eval()
    wrapped.requires_grad_(False)
    evaluator.condition = "peft_export"
    evaluator.evaluate(
        replace(base, model=wrapped), dataset, tasks=(task,), max_examples=1
    )
    exported_scores = evaluator.records[-1]["scores"]
    saved = output / "saved_peft"
    wrapped.save_pretrained(saved)
    unwrapped = wrapped.unload()
    reloaded = PeftModel.from_pretrained(
        unwrapped, saved, is_trainable=False, autocast_adapter_dtype=False
    )
    reloaded.eval()
    reloaded.requires_grad_(False)
    evaluator.condition = "peft_reload"
    evaluator.evaluate(
        replace(base, model=reloaded), dataset, tasks=(task,), max_examples=1
    )
    reloaded_scores = evaluator.records[-1]["scores"]
    export_error = max(
        abs(a - b) for a, b in zip(native_scores, exported_scores, strict=True)
    )
    reload_error = max(
        abs(a - b) for a, b in zip(exported_scores, reloaded_scores, strict=True)
    )
    if export_error > config["peft_parity_atol"] or reload_error != 0:
        raise ValueError(
            f"adapter inference parity failed: native/PEFT={export_error}, reload={reload_error}"
        )
    return {
        "task": task,
        "examples": 1,
        "native_peft_max_score_error": export_error,
        "reload_max_score_error": reload_error,
        "reload_scores_exact_equal": reload_error == 0,
        "native_peft_atol": config["peft_parity_atol"],
    }


def publish_stage(channel, output, event, payload):
    path = output / f"{event}.json"
    digest = write_once(
        path, {"schema_version": 1, "stage": event, "at": utc_now(), **payload}
    )
    receipt = {"path": str(path.resolve()), "sha256": digest}
    channel.send(event, receipt)
    return receipt


def qualify(config, output, channel, base_path=None):
    manifest_path = Path(__file__).parent / config["manifest"]
    manifest = read_json(manifest_path)
    entry = entry_for(manifest, config["model"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "predictions.jsonl").exists():
        raise FileExistsError(
            f"output already has evidence; choose a fresh run directory: {output}"
        )
    started = utc_now()
    result = {
        "schema_version": 1,
        "model": config["model"],
        "started_at": started,
        "manifest_sha256": sha256_file(manifest_path),
        "config": config,
        "source_sha256": {
            file.name: sha256_file(file)
            for file in sorted(Path(__file__).parent.glob("*.py"))
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "portallib",
                "peft",
                "transformers",
                "torch",
                "huggingface-hub",
                "datasets",
                "pyarrow",
            )
        },
        "claim": "evaluation of published pretrained task adapters",
        "optimizer_steps": 0,
        "durable_learning_demonstrated": False,
    }
    write_json(output / "request.json", result)
    ledger = output / "autoresearch" / "qualification"
    ledger.mkdir(parents=True, exist_ok=True)
    emit("qualification_start", model=config["model"], output=str(output))
    stage = "preflight"

    def before_load(snapshot):
        nonlocal stage
        stage = "cache_verified"
        receipt = publish_stage(
            channel,
            output,
            stage,
            {"snapshot": snapshot, "manifest_sha256": result["manifest_sha256"]},
        )
        result["cache_verified"] = receipt
        channel.wait_for_continue()
        stage = "load_started"
        publish_stage(channel, output, stage, {"cache_verified": receipt})

    try:
        verify_library(manifest)
        torch.manual_seed(config["seed"])
        dataset, selection = prepare_dataset(
            manifest, config["cache_root"], config, output
        )
        if selection["selection_sha256"] != config["selection_sha256"]:
            raise ValueError("heldout selection changed from the pinned screen")
        result["selection_sha256"] = selection["selection_sha256"]
        portal = load_portal(manifest, entry, config["cache_root"])
        base = load_base(
            manifest,
            entry,
            config["cache_root"],
            config["visible_gpus"],
            base_path or config.get("base_path"),
            before_load=before_load,
        )
        visible_indices = list(range(torch.cuda.device_count()))
        for index in visible_indices:
            torch.cuda.synchronize(index)
        result["gpu"] = {
            "visible_count": len(visible_indices),
            "hip_version": torch.version.hip,
            "synchronized_device_indices": visible_indices,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in visible_indices
            ],
        }
        result["base"] = {
            "repo": base.model_id,
            "revision": base.revision,
            "model_class": type(base.model).__name__,
            "parameters": sum(p.numel() for p in base.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in base.model.parameters() if p.requires_grad
            ),
        }
        stage = "load_complete"
        loaded = publish_stage(
            channel, output, stage, {"base": result["base"], "gpu": result["gpu"]}
        )
        channel.wait_for_continue()
        stage = "evaluation_started"
        publish_stage(
            channel,
            output,
            stage,
            {"load_complete": loaded, "selection_sha256": result["selection_sha256"]},
        )
        result["native_roundtrip"] = roundtrip_portal(portal, output, config["tasks"])
        evaluator = EvidenceEvaluator(
            output=output / "predictions.jsonl",
            condition="base",
            max_prompt=config["max_prompt"],
            batch_size=config["batch_size"],
        )
        original = evaluator.evaluate(base, dataset, tasks=tuple(config["tasks"]))
        base_records = evaluator.records.copy()
        evaluator.records.clear()
        evaluator.condition = "portal"
        adapted = evaluator.evaluate(
            base, dataset, tasks=tuple(config["tasks"]), portal=portal
        )
        statistics = paired_statistics(base_records, evaluator.records, config["seed"])
        result.update(
            base_metrics=original.to_dict(),
            adapted_metrics=adapted.to_dict(),
            statistics=statistics,
        )
        write_json(output / "evaluation.json", result)
        result["peft_roundtrip"] = peft_reload_check(
            base, portal, dataset, config["tasks"][0], output, config
        )
        result["status"] = "qualified"
        result["decision"] = (
            "keep_for_followup"
            if statistics["paired_stratified_bootstrap_95ci"][0] > 0
            else "discard_positive_claim"
        )
        result["limitations"] = [
            "No training, retention, or next-skill learning was measured.",
            "Four tasks and one deterministic sample; no general agent capability claim.",
            "Documented recipe prefix and train prompt overlap excluded; unknown pretraining exposure remains.",
            "Official character-normalized scoring truncates long prompts to the recorded token limit.",
        ]
    except (
        RuntimeError,
        ValueError,
        OSError,
        AssertionError,
        KeyError,
        TypeError,
    ) as exc:
        result.update(
            status="failed",
            decision="discard_unqualified",
            error_type=type(exc).__name__,
            error=str(exc),
            failed_stage=stage,
        )
        (output / "failure.log").write_text(traceback.format_exc())
        emit(
            "qualification_failure",
            model=config["model"],
            error_type=type(exc).__name__,
            error=str(exc),
        )
    result["finished_at"] = utc_now()
    result_path = output / "result.json"
    result_sha256 = write_once(result_path, result)
    result_receipt = {"path": str(result_path.resolve()), "sha256": result_sha256}
    with (ledger / "qualification-results.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["iteration", "model", "metric", "value", "decision", "status"])
        writer.writerow(
            [
                1,
                config["model"],
                "paired_macro_acc_norm_delta",
                result.get("statistics", {}).get("macro_accuracy_delta"),
                result["decision"],
                result["status"],
            ]
        )
    write_json(
        ledger / "handoff.json",
        {
            "status": result["status"],
            "decision": result["decision"],
            "result": str(output / "result.json"),
            "iterations": 1,
            "max_iterations": 1,
        },
    )
    emit(
        "qualification_complete",
        model=config["model"],
        status=result["status"],
        decision=result["decision"],
    )
    if result["status"] != "qualified":
        channel.send(
            "failed",
            result_receipt,
            failed_stage=stage,
            error_type=result["error_type"],
            reason=result["error"],
        )
        raise SystemExit(1)
    channel.send("evaluation_complete", result_receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    channel = Channel(config["channel_timeout_seconds"])
    try:
        qualify(config, args.output_dir, channel, args.base_path)
    finally:
        channel.close()


if __name__ == "__main__":
    main()
