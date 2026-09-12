import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from pathlib import Path

import torch

from agent_dice import fuse_experts
from data import Reservoir, audit_corpus, build_corpus, digest, scheduled_batch
from learning import (
    capture,
    evaluate,
    event,
    frozen_hash,
    generate,
    layers,
    load_checkpoint,
    load_model,
    load_tokenizer,
    new_optimizer,
    restore,
    save_checkpoint,
    tree_record,
    verify_checkpoint,
    write_json,
)
from learning import train_update as update
from sandbox import FAMILIES

LOGGER = logging.getLogger("skill_transfer")
TRANSPORT = {"config.json", "packages.txt", "execution.json", "task.json", "run.log", "attempts"}


def validate_config(config):
    if config["experiment"] != "varied-tool-workflow-transfer":
        raise ValueError("EXPERIMENT_ID")
    if config["target_family"] not in FAMILIES or config["irrelevant_family"] not in FAMILIES:
        raise ValueError("FAMILY_ID")
    if config["target_family"] == config["irrelevant_family"]:
        raise ValueError("IRRELEVANT_HISTORY_OVERLAP")
    if config["rank"] != 8 or config["lora_alpha"] != 16 or config["target_suffixes"] != ["q_proj", "v_proj"]:
        raise ValueError("FIXED_ADAPTER_CONTRACT")
    if (
        config["model_id"] != "Qwen/Qwen3.5-4B"
        or config["model_revision"] != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ):
        raise ValueError("MODEL_PIN")
    if Path(config["model_path"]).name != config["model_revision"]:
        raise ValueError("SNAPSHOT_PATH_REVISION")
    for name in ("qualification_budgets", "workflow_checkpoints"):
        values = config[name]
        if not values or values != sorted(set(values)) or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError(f"BUDGET_ORDER: {name}")
    if not 0 < config["replay_per_batch"] < config["batch_size"] <= config["buffer_capacity"]:
        raise ValueError("REPLAY_BUDGET")
    for kind in ("primitive", "workflow"):
        for split in ("train", "validation", "test"):
            count = config[f"{kind}_{split}_examples"]
            if count < config["batch_size"] or count % 4:
                raise ValueError("BALANCED_SAMPLE_COUNT")
    for key in (
        "primitive_min_accuracy",
        "primitive_min_per_operation",
        "workflow_min_accuracy",
        "min_baseline_headroom",
        "headroom_fraction",
    ):
        if not 0 < config[key] <= 1:
            raise ValueError("GATE_RANGE")


def prepare_output(config, output):
    output.mkdir(parents=True, exist_ok=True)
    allowed = TRANSPORT - {"attempts"}
    entries = list(output.iterdir())
    for path in entries:
        if path.is_symlink() or not (
            (path.is_file() and path.name in allowed) or (path.is_dir() and path.name == "attempts")
        ):
            raise RuntimeError("OUTPUT_ALREADY_USED: only dispatcher transport files may preexist")
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("DISPATCH_CONFIG_MISMATCH")
    else:
        write_json(config_path, config)
    task_path = output / "task.json"
    if task_path.exists() and json.loads(task_path.read_text())["config"] != config:
        raise RuntimeError("DISPATCH_TASK_MISMATCH")
    write_json(
        output / "dispatcher.json",
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in entries if path.is_file()},
    )


def primitive_gate(initial, final, config):
    competence = final["accuracy"] >= config["primitive_min_accuracy"]
    coverage = (
        len(final["per_pattern"]) == 4 and min(final["per_pattern"].values()) >= config["primitive_min_per_operation"]
    )
    required_gain = config["headroom_fraction"] * (1 - initial["accuracy"])
    previously_competent = initial["accuracy"] >= config["primitive_min_accuracy"]
    sufficient_headroom = 1 - initial["accuracy"] >= config["min_baseline_headroom"]
    acquired = final["accuracy"] - initial["accuracy"] >= required_gain
    return {
        "passed": competence and coverage and acquired and sufficient_headroom,
        "initial": initial["accuracy"],
        "final": final["accuracy"],
        "required_gain": required_gain,
        "gain": final["accuracy"] - initial["accuracy"],
        "all_operations_covered": coverage,
        "baseline_already_competent": previously_competent,
        "sufficient_baseline_headroom": sufficient_headroom,
        "interpretation": "learning_premise_failed"
        if not sufficient_headroom
        else "gain_required_even_if_baseline_competent",
    }


def train_segment(model, optimizer, encoded, rows, start, end, schedule_seed, path, reservoir=None):
    replay_count = encoded.config["replay_per_batch"] if reservoir is not None else 0
    exposed = {}
    totals = {"updates": 0, "current_exposures": 0, "old_exposures": 0, "target_tokens": 0, "input_tokens": 0}
    for step in range(start, end):
        current = scheduled_batch(rows, step, encoded.config["batch_size"], schedule_seed)
        replay = reservoir.sample(replay_count, schedule_seed, step, rows[0].family) if reservoir is not None else []
        if replay:
            current = current[: -len(replay)]
        record = update(model, optimizer, encoded, current + replay)
        for row in current:
            exposed[row.id] = row
        record.update(step=step + 1, current_ids=[row.id for row in current], replay_ids=[row.id for row in replay])
        event(path, "update", **record)
        totals["updates"] += 1
        totals["current_exposures"] += len(current)
        totals["old_exposures"] += len(replay)
        for key in ("target_tokens", "input_tokens"):
            totals[key] += record[key]
    steps = {int(item["step"]) for item in optimizer.state.values()}
    if len(steps) != 1:
        raise RuntimeError("OPTIMIZER_CLOCKS_DIVERGED")
    totals["optimizer_step"] = steps.pop()
    totals["exposed_unique_current_ids"] = sorted(exposed)
    return totals, list(exposed.values())


def load_training_state(model, checkpoint, config):
    optimizer_state = load_checkpoint(model, checkpoint)
    optimizer = new_optimizer(model, config)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        if digest(tree_record(optimizer.state_dict())) != digest(tree_record(optimizer_state)):
            raise RuntimeError("OPTIMIZER_RESTORE_CHANGED")
    return optimizer


def assess_primitives(model, encoded, corpus, families, split, directory):
    return {
        family: evaluate(model, encoded, corpus["primitive"][family][split], directory / f"{family}.{split}.json")
        for family in families
    }


def qualification(model, encoded, spec, corpus, initial, config, output):
    baseline = assess_primitives(model, encoded, corpus, FAMILIES, "validation", output / "baseline")
    workflow_baseline = evaluate(
        model,
        encoded,
        corpus["workflow"][config["target_family"]]["validation"],
        output / "baseline" / "workflow.validation.json",
    )
    saturated = [
        family for family, metrics in baseline.items() if 1 - metrics["accuracy"] < config["min_baseline_headroom"]
    ]
    if 1 - workflow_baseline["accuracy"] < config["min_baseline_headroom"]:
        saturated.append("workflow")
    if saturated:
        return {
            "passed": False,
            "budget": 0,
            "baseline": baseline,
            "workflow_baseline": workflow_baseline,
            "rounds": [],
            "reason": "baseline_saturated",
            "saturated_tasks": saturated,
            "direction": "schemas may disclose operation identity; redesign binding ambiguity using train/dev only under a new protocol",
        }, {}
    checkpoints, rounds = {}, []
    start = 0
    for budget in config["qualification_budgets"]:
        gates = {}
        for family in spec["order"]:
            if family in checkpoints:
                optimizer = load_training_state(model, checkpoints[family], config)
            else:
                restore(model, initial)
                optimizer = new_optimizer(model, config)
            directory = output / "qualification" / family / str(budget)
            directory.mkdir(parents=True)
            schedule_seed = [config["optimization_seed"], "primitive", family]
            exposure, _ = train_segment(
                model,
                optimizer,
                encoded,
                corpus["primitive"][family]["train"],
                start,
                budget,
                schedule_seed,
                directory / "updates.jsonl",
            )
            metrics = assess_primitives(model, encoded, corpus, [family], "validation", directory)[family]
            gate = primitive_gate(baseline[family], metrics, config)
            gates[family] = gate
            checkpoints[family] = directory / "checkpoint"
            save_checkpoint(
                model,
                optimizer,
                checkpoints[family],
                {"kind": "independent_expert", "family": family, "updates": budget, "segment_exposure": exposure},
            )
            write_json(directory / "gate.json", gate)
            event(output / "events.jsonl", "primitive_gate", family=family, budget=budget, gate=gate)
        rounds.append({"budget": budget, "gates": gates})
        if all(gate["passed"] for gate in gates.values()):
            for family in spec["order"]:
                load_training_state(model, checkpoints[family], config)
                assess_primitives(model, encoded, corpus, [family], "test", checkpoints[family].parent)
            return {
                "passed": True,
                "budget": budget,
                "baseline": baseline,
                "workflow_baseline": workflow_baseline,
                "rounds": rounds,
            }, checkpoints
        start = budget
    diagnostics = {}
    for family, checkpoint in checkpoints.items():
        load_training_state(model, checkpoint, config)
        diagnostics[family] = evaluate(
            model, encoded, corpus["primitive"][family]["train"][:16], checkpoint.parent / "train-diagnostic.json"
        )
    return {
        "passed": False,
        "budget": start,
        "baseline": baseline,
        "workflow_baseline": workflow_baseline,
        "rounds": rounds,
        "train_diagnostics": diagnostics,
        "direction": "resolve primitive acquisition using format, EOS, training-fit and execution errors before any new transfer recipe",
    }, checkpoints


def stream_controls(model, encoded, spec, corpus, qualified, experts, config, output):
    budget = qualified["budget"]
    first = spec["order"][0]
    candidates, summaries = {}, {}
    for arm in ("continue", "replay"):
        optimizer = load_training_state(model, experts[first], config)
        reservoir = Reservoir(config["buffer_capacity"], config["optimization_seed"])
        first_rows = []
        seed = [config["optimization_seed"], "primitive", first]
        for step in range(budget):
            first_rows.extend(scheduled_batch(corpus["primitive"][first]["train"], step, config["batch_size"], seed))
        reservoir.observe(first_rows)
        matrices = []
        arm_summary = {
            "status": "complete",
            "first_stage_checkpoint": str(experts[first].relative_to(output)),
            "logical_updates": budget,
            "new_exposures": budget * config["batch_size"],
            "replay_exposures": 0,
            "stages": [],
        }
        for index, family in enumerate(spec["order"][1:], start=1):
            directory = output / "streams" / arm / str(index + 1)
            directory.mkdir(parents=True)
            before_memory = reservoir.record()
            exposure, exposed = train_segment(
                model,
                optimizer,
                encoded,
                corpus["primitive"][family]["train"],
                0,
                budget,
                [config["optimization_seed"], "primitive", family],
                directory / "updates.jsonl",
                reservoir if arm == "replay" else None,
            )
            expected_clock = (index + 1) * budget
            if exposure["optimizer_step"] != expected_clock:
                raise RuntimeError("STREAM_OPTIMIZER_RESET")
            metrics = assess_primitives(model, encoded, corpus, spec["order"][: index + 1], "validation", directory)
            gate = primitive_gate(qualified["baseline"][family], metrics[family], config)
            reservoir.observe(exposed)
            stage = {
                "family": family,
                "gate": gate,
                "exposure": exposure,
                "memory_before": before_memory,
                "memory_after": reservoir.record(),
                "validation": metrics,
            }
            arm_summary["stages"].append(stage)
            arm_summary["logical_updates"] += budget
            arm_summary["new_exposures"] += exposure["current_exposures"]
            arm_summary["replay_exposures"] += exposure["old_exposures"]
            matrices.append(metrics)
            save_checkpoint(model, optimizer, directory / "checkpoint", stage)
            if not gate["passed"]:
                arm_summary.update(status="new_primitive_not_acquired", stopped_before_next_stage=True)
                event(output / "events.jsonl", "stream_stopped", arm=arm, family=family, gate=gate)
                break
        if arm_summary["status"] == "complete":
            candidates[arm] = directory / "checkpoint"
            arm_summary["final_test"] = assess_primitives(
                model, encoded, corpus, FAMILIES, "test", output / "streams" / arm / "final"
            )
        arm_summary["validation_matrices"] = matrices
        summaries[arm] = arm_summary
    return candidates, summaries


def fusion_controls(model, encoded, corpus, experts, budget, config, output):
    expert_states = [
        torch.load(path / "state.pt", map_location="cpu", weights_only=True)["adapter"] for path in experts.values()
    ]
    archive_elements = sum(
        value[key].numel() for state in expert_states for value in state.values() for key in ("A", "B")
    )
    candidates, records = {}, {}
    initial = torch.load(output / "initial" / "state.pt", map_location="cpu", weights_only=True)["adapter"]
    for method in ("arithmetic", "agent_dice"):
        fused, audits = fuse_experts(expert_states, config["lora_alpha"] / config["rank"], method)
        for name in fused:
            fused[name]["A"] = initial[name]["A"].clone()
        restore(model, fused)
        directory = output / "fusion" / method
        resident_elements = sum(value["offset"].numel() for value in fused.values())
        rank_bounds = {
            name: {
                "shape": list(value["offset"].shape),
                "rank_upper_bound": min(*value["offset"].shape, len(experts) * config["rank"])
                if method == "arithmetic"
                else min(value["offset"].shape),
                "numerical_rank_measured": False,
            }
            for name, value in fused.items()
        }
        record = {
            "method": method,
            "task_vectors": len(experts),
            "expert_training_updates": len(experts) * budget,
            "expert_training_example_exposures": len(experts) * budget * config["batch_size"],
            "expert_archive_elements": archive_elements,
            "expert_archive_bytes_fp32": archive_elements * 4,
            "resident_dense_delta_elements": resident_elements,
            "resident_dense_delta_bytes_fp32": resident_elements * 4,
            "resident_delta_rank": rank_bounds,
            "coordinate_audits": audits,
            "rank_truncation": False,
            "learning_recipe": "independent_specialists_from_one_base_then_fusion",
            "fixed_memory_sequential_recipe": False,
            "transfer_trainable_factors": "original shared initialization, zero B; all prior knowledge is in frozen dense offset",
        }
        record["primitive_validation"] = assess_primitives(model, encoded, corpus, FAMILIES, "validation", directory)
        record["primitive_test"] = assess_primitives(model, encoded, corpus, FAMILIES, "test", directory)
        candidates[method] = directory / "checkpoint"
        save_checkpoint(model, None, candidates[method], record)
        write_json(directory / "fusion.json", record)
        records[method] = record
    return candidates, records


def workflow_transfer(model, encoded, corpus, candidates, config, output):
    family = config["target_family"]
    results = {}
    for condition, checkpoint in candidates.items():
        load_training_state(model, checkpoint, config)
        offset_before = digest({name: tree_record(layer.offset) for name, layer in layers(model).items()})
        optimizer = new_optimizer(model, config)
        directory = output / "transfer" / condition
        directory.mkdir(parents=True)
        curve = [
            {
                "updates": 0,
                **evaluate(model, encoded, corpus["workflow"][family]["validation"], directory / "validation-0.json"),
            }
        ]
        primitive_before = assess_primitives(
            model, encoded, corpus, FAMILIES, "validation", directory / "primitive-before"
        )
        start = 0
        exposure = []
        for budget in config["workflow_checkpoints"]:
            segment, _ = train_segment(
                model,
                optimizer,
                encoded,
                corpus["workflow"][family]["train"],
                start,
                budget,
                [config["optimization_seed"], "workflow", family],
                directory / "updates.jsonl",
            )
            if segment["optimizer_step"] != budget:
                raise RuntimeError("TRANSFER_OPTIMIZER_CLOCK")
            exposure.append(segment)
            curve.append(
                {
                    "updates": budget,
                    **evaluate(
                        model,
                        encoded,
                        corpus["workflow"][family]["validation"],
                        directory / f"validation-{budget}.json",
                    ),
                }
            )
            start = budget
        final_test = evaluate(model, encoded, corpus["workflow"][family]["test"], directory / "test.json")
        novel_test = evaluate(model, encoded, corpus["workflow"][family]["novel_test"], directory / "novel-test.json")
        primitive_after = assess_primitives(model, encoded, corpus, FAMILIES, "test", directory / "primitive-after")
        primitive_after_validation = assess_primitives(
            model, encoded, corpus, FAMILIES, "validation", directory / "primitive-after"
        )
        if offset_before != digest({name: tree_record(layer.offset) for name, layer in layers(model).items()}):
            raise RuntimeError("FUSED_OFFSET_CHANGED_DURING_TRANSFER")
        auc = (
            sum(
                (right["updates"] - left["updates"]) * (left["accuracy"] + right["accuracy"]) / 2
                for left, right in zip(curve[:-1], curve[1:], strict=True)
            )
            / start
        )
        reached = [point["updates"] for point in curve if point["accuracy"] >= config["workflow_min_accuracy"]]
        record = {
            "initial_checkpoint": str(checkpoint.relative_to(output)),
            "validation_curve": curve,
            "normalized_validation_auc": auc,
            "first_qualified_checkpoint": min(reached) if reached else None,
            "test": final_test,
            "novel_composition_test": novel_test,
            "primitive_before_validation": primitive_before,
            "primitive_after_test": primitive_after,
            "primitive_after_validation": primitive_after_validation,
            "primitive_validation_change": {
                name: primitive_after_validation[name]["accuracy"] - primitive_before[name]["accuracy"]
                for name in FAMILIES
            },
            "frozen_offset_exact": True,
            "training_exposure": exposure,
            "updates": start,
            "example_exposures": start * config["batch_size"],
            "optimizer_reset_for_every_transfer_condition": True,
            "outcome": "already_competent_before_workflow_training"
            if 0 in reached
            else "workflow_acquired"
            if reached
            else "workflow_not_acquired_at_prespecified_budgets",
        }
        save_checkpoint(model, optimizer, directory / "checkpoint", record)
        verify_checkpoint(
            model,
            encoded,
            directory / "checkpoint",
            corpus["workflow"][family]["validation"][:4],
            directory / "reload.json",
        )
        write_json(directory / "result.json", record)
        results[condition] = record
    fresh = results["fresh"]
    irrelevant = results["irrelevant"]
    for record in results.values():
        record["test_gain_vs_fresh"] = record["test"]["accuracy"] - fresh["test"]["accuracy"]
        record["auc_gain_vs_fresh"] = record["normalized_validation_auc"] - fresh["normalized_validation_auc"]
        record["auc_gain_vs_irrelevant"] = record["normalized_validation_auc"] - irrelevant["normalized_validation_auc"]
    return results


def experiment(config, output):
    validate_config(config)
    prepare_output(config, output)
    protocol_path = Path(__file__).with_name("protocol.json")
    with (output / "protocol.json").open("xb") as handle:
        handle.write(protocol_path.read_bytes())
    spec, corpus = build_corpus(config)
    audit = audit_corpus(spec, corpus)
    write_json(output / "stream.json", spec)
    write_json(output / "dataset-audit.json", audit)
    write_json(
        output / "dataset.json",
        {
            kind: {
                family: {split: [row.record() for row in rows] for split, rows in splits.items()}
                for family, splits in families.items()
            }
            for kind, families in corpus.items()
        },
    )
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if config["mode"] == "prepare":
        write_json(
            output / "result.json",
            {
                "status": "cpu_dataset_prepared",
                "dataset_audit": audit,
                "protocol_sha256": protocol_sha,
                "gpu_result": False,
            },
        )
        return
    encoded = load_tokenizer(config, spec["conventions"])
    lengths = [
        tuple(map(len, encoded.tokens(row)))
        for families in corpus.values()
        for splits in families.values()
        for rows in splits.values()
        for row in rows
    ]
    write_json(
        output / "tokenization.json",
        {
            "max_prompt": max(item[0] for item in lengths),
            "max_answer_with_eos": max(item[1] for item in lengths),
            "examples": len(lengths),
            "native_eos": encoded.eos,
            "truncation": False,
        },
    )
    model = load_model(config)
    base_before = frozen_hash(model)
    initial = capture(model)
    save_checkpoint(model, None, output / "initial", {"kind": "initial", "base_sha256": base_before})
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda_or_hip": torch.version.hip or torch.version.cuda,
        "device": config["device"],
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "adapter_targets": list(layers(model)),
        "load_proof": model.skill_transfer_load_proof,
    }
    write_json(output / "runtime.json", runtime)
    cache_rows = corpus["primitive"][config["target_family"]]["validation"][:4]
    if config["generation_use_cache"]:
        cached = generate(model, encoded, cache_rows, use_cache=True)
        uncached = generate(model, encoded, cache_rows, use_cache=False)
        if cached != uncached:
            raise RuntimeError("CACHE_PARITY: cached and uncached full records differ")
        write_json(output / "cache-parity.json", {"exact_records": True, "ids": [row.id for row in cache_rows]})
    qualified, experts = qualification(model, encoded, spec, corpus, initial, config, output)
    result = {
        "protocol_sha256": protocol_sha,
        "config_sha256": digest(config),
        "data_seed": config["data_seed"],
        "optimization_seed": config["optimization_seed"],
        "target_family": config["target_family"],
        "stream_order": spec["order"],
        "qualification": qualified,
        "status": "prerequisite_failed",
        "transfer_updates": 0,
    }
    if qualified["passed"]:
        if config["generation_use_cache"]:
            load_training_state(model, experts[config["target_family"]], config)
            if generate(model, encoded, cache_rows, True) != generate(model, encoded, cache_rows, False):
                raise RuntimeError("TRAINED_CACHE_PARITY")
            write_json(
                output / "trained-cache-parity.json", {"exact_records": True, "ids": [row.id for row in cache_rows]}
            )
        if config["mode"] == "qualify":
            result["status"] = "primitives_qualified"
        else:
            candidates, streams = stream_controls(model, encoded, spec, corpus, qualified, experts, config, output)
            fused_candidates, fusions = fusion_controls(
                model, encoded, corpus, experts, qualified["budget"], config, output
            )
            candidates = {
                "fresh": output / "initial",
                "relevant": experts[config["target_family"]],
                "irrelevant": experts[config["irrelevant_family"]],
                **candidates,
                **fused_candidates,
            }
            transfers = workflow_transfer(model, encoded, corpus, candidates, config, output)
            result.update(
                status="completed",
                streams=streams,
                fusions=fusions,
                transfer=transfers,
                transfer_updates=sum(row["updates"] for row in transfers.values()),
            )
    base_after = frozen_hash(model)
    if base_before != base_after:
        raise RuntimeError("FROZEN_BASE_CHANGED")
    result["frozen_base"] = {"all_parameter_tensors_exact": True, "before": base_before, "after": base_after}
    executed_updates = 0
    for path in output.rglob("updates.jsonl"):
        executed_updates += len(path.read_text().splitlines())
    result["actual_executed_updates_including_qualification"] = executed_updates
    result["artifact_manifest"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.relative_to(output).parts[0] not in TRANSPORT
    }
    write_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = json.loads(args.config.read_text())
    if config["mode"] not in {"prepare", "qualify", "study"}:
        raise ValueError("UNKNOWN_MODE")
    try:
        experiment(config, args.output_dir)
    except Exception:
        LOGGER.exception("SKILL_TRANSFER_FAILED output=%s", args.output_dir)
        raise


if __name__ == "__main__":
    main()
