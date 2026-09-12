import argparse
import json
import logging
import math
from pathlib import Path

import torch

from data import digest
from followup import (
    comparison,
    evaluation_sets,
    finish,
    fresh_runtime,
    measure,
    prepare_followup,
    restore_source,
)
from learning import capture, event, finite_tensor, load_checkpoint, restore, save_checkpoint, write_json
from run import primitive_gate
from sandbox import FAMILIES, SCHEMAS

LOGGER = logging.getLogger("skill_transfer.compression")


def fusion_gate(baseline, fused, config):
    if set(baseline) != set(FAMILIES) or set(fused) != set(FAMILIES):
        raise ValueError("FUSION_GATE_FAMILY_COVERAGE")
    for family in FAMILIES:
        for metrics in (baseline[family], fused[family]):
            if metrics["n"] != config["primitive_validation_examples"]:
                raise ValueError("FUSION_GATE_VALIDATION_COUNT")
            if set(metrics["per_pattern"]) != set(SCHEMAS[family]):
                raise ValueError("FUSION_GATE_OPERATION_COVERAGE")
    gates = {family: primitive_gate(baseline[family], fused[family], config) for family in FAMILIES}
    return {"passed": all(value["passed"] for value in gates.values()), "families": gates, "split": "validation"}


def state_memory(state):
    factors = sum(value[key].numel() * value[key].element_size() for value in state.values() for key in ("A", "B"))
    offsets = sum(
        value["offset"].numel() * value["offset"].element_size()
        for value in state.values()
        if value["offset"] is not None
    )
    return {
        "factor_bytes": factors,
        "dense_offset_bytes": offsets,
        "resident_adapter_tensor_bytes": factors + offsets,
        "projection_shapes": {name: [value["B"].shape[0], value["A"].shape[1]] for name, value in state.items()},
        "factor_ranks": {name: value["A"].shape[0] for name, value in state.items()},
        "base_weights_included": False,
        "optimizer_bytes": 0,
    }


def bank_archive_storage(source):
    budget = source.result["qualification"]["budget"]
    files = {}
    for family in FAMILIES:
        directory = f"qualification/{family}/{budget}/checkpoint"
        metadata = source.read_json(f"{directory}/metadata.json")
        files[family] = {
            "state_file_bytes": source.verified_path(f"{directory}/state.pt").stat().st_size,
            "metadata_file_bytes": source.verified_path(f"{directory}/metadata.json").stat().st_size,
            "optimizer_state_present": metadata["optimizer_sha256"] != digest(None),
        }
    return {
        "by_family": files,
        "total_file_bytes": sum(value["state_file_bytes"] + value["metadata_file_bytes"] for value in files.values()),
        "resident_inference_memory": False,
    }


def svd_rank8(state, alpha=16):
    if not state or alpha != 16:
        raise ValueError("SVD_FIXED_RANK8_ALPHA16")
    rank, scaling = 8, alpha / 8
    compressed, projections = {}, {}
    for name, values in state.items():
        if set(values) != {"A", "B", "offset"}:
            raise ValueError(f"SVD_STATE_KEYS: {name}")
        a, b, offset = values["A"], values["B"], values["offset"]
        for value in (a, b, offset):
            if value is not None and (value.dtype != torch.float32 or not finite_tensor(value)):
                raise ValueError(f"SVD_FINITE_FP32_REQUIRED: {name}")
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"SVD_FACTOR_SHAPE: {name}")
        if min(b.shape[0], a.shape[1]) < rank or (offset is not None and offset.shape != (b.shape[0], a.shape[1])):
            raise ValueError(f"SVD_MATRIX_SHAPE: {name}")
        LOGGER.info("SVD_START projection=%s shape=%s rank=8 device=cpu dtype=float64", name, (b.shape[0], a.shape[1]))
        delta = scaling * (b.detach().cpu().double() @ a.detach().cpu().double())
        if offset is not None:
            delta.add_(offset.detach().cpu().double())
        u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
        new_a = vh[:rank].float().contiguous()
        new_b = (u[:, :rank] * (singular[:rank] / scaling)).float().contiguous()
        if not finite_tensor(new_a) or not finite_tensor(new_b):
            raise RuntimeError(f"SVD_NONFINITE_FACTORS: {name}")
        actual = scaling * (new_b.double() @ new_a.double())
        energy = float(delta.square().sum())
        kept = float(singular[:rank].square().sum())
        tail = float(singular[rank:].square().sum())
        error = float((delta - actual).square().sum())
        ideal = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        rounding_error = float((ideal - actual).square().sum())
        tolerance = max(delta.shape) * torch.finfo(torch.float64).eps * float(singular[0])
        projections[name] = {
            "shape": list(delta.shape),
            "source_numerical_rank": int((singular > tolerance).sum()),
            "rank_tolerance": tolerance,
            "target_rank": rank,
            "singular_values_retained": singular[:rank].tolist(),
            "total_energy": energy,
            "retained_energy": kept,
            "energy_retained_fraction": kept / energy if energy else 1.0,
            "ideal_tail_energy": tail,
            "stored_factor_squared_error": error,
            "fp32_factor_rounding_squared_error": rounding_error,
            "stored_factor_relative_frobenius_error": math.sqrt(error / energy) if energy else 0.0,
            "source_offset_present": offset is not None,
        }
        compressed[name] = {"A": new_a, "B": new_b, "offset": None}
        LOGGER.info(
            "SVD_DONE projection=%s energy_retained=%.8f relative_error=%.8f",
            name,
            projections[name]["energy_retained_fraction"],
            projections[name]["stored_factor_relative_frobenius_error"],
        )
    total = sum(value["total_energy"] for value in projections.values())
    retained = sum(value["retained_energy"] for value in projections.values())
    errors = sum(value["stored_factor_squared_error"] for value in projections.values())
    return compressed, {
        "method": "dense_reduced_svd_cpu_float64",
        "rank": rank,
        "alpha": alpha,
        "total_energy": total,
        "retained_energy": retained,
        "energy_retained_fraction": retained / total if total else 1.0,
        "stored_factor_squared_error": errors,
        "stored_factor_relative_frobenius_error": math.sqrt(errors / total) if total else 0.0,
        "energy_is_not_a_behavioral_measure": True,
        "projections": projections,
    }


def saved_eligibility(source):
    baseline = {
        family: source.evaluation(
            f"baseline/{family}.validation.json", source.corpus["primitive"][family]["validation"]
        )["metrics"]
        for family in FAMILIES
    }
    references, gates = {}, {}
    for method in ("arithmetic", "agent_dice"):
        references[method] = {
            family: source.evaluation(
                f"fusion/{method}/{family}.validation.json", source.corpus["primitive"][family]["validation"]
            )
            for family in FAMILIES
        }
        gates[method] = fusion_gate(
            baseline, {family: record["metrics"] for family, record in references[method].items()}, source.config
        )
    return baseline, references, gates


def compress(settings, output):
    source, _ = prepare_followup(settings, output, "compress")
    baseline, references, gates = saved_eligibility(source)
    write_json(output / "eligibility.json", gates)
    model = encoded = None
    results = {}
    for method in ("arithmetic", "agent_dice"):
        directory = output / method
        if not gates[method]["passed"]:
            results[method] = {
                "status": "branch_gate_failed",
                "reason": "saved_fusion_dev_unqualified",
                "gate": gates[method],
                "svd_performed": False,
                "teacher_qualified": False,
            }
            write_json(directory / "result.json", results[method])
            event(output / "events.jsonl", "compression_branch_gate", method=method, passed=False)
            continue
        if model is None:
            model, encoded = fresh_runtime(source, output)
        checkpoint = f"fusion/{method}/checkpoint"
        restore_source(model, source, checkpoint)
        original = capture(model)
        if any(value["offset"] is None or bool(torch.count_nonzero(value["B"])) for value in original.values()):
            raise RuntimeError(f"COMPRESSION_EXPECTS_PRE_WORKFLOW_FUSION: {method}")
        original_dev, reload_comparisons = {}, {}
        for family in FAMILIES:
            name = f"primitive-{family}.validation"
            original_dev[name] = measure(
                model,
                encoded,
                source.corpus["primitive"][family]["validation"],
                directory / "original" / f"{name}.json",
            )
            reload_comparisons[family] = comparison(references[method][family], original_dev[name])
        fresh_gate = fusion_gate(
            baseline,
            {family: original_dev[f"primitive-{family}.validation"]["metrics"] for family in FAMILIES},
            source.config,
        )
        reload_exact = all(value["all_records_exact"] for value in reload_comparisons.values())
        write_json(directory / "reload-dev.json", {"gate": fresh_gate, "comparisons": reload_comparisons})
        if not fresh_gate["passed"] or not reload_exact:
            results[method] = {
                "status": "branch_gate_failed",
                "reason": "fresh_load_dev_mismatch_or_unqualified",
                "gate": fresh_gate,
                "reload_exact": reload_exact,
                "svd_performed": False,
                "teacher_qualified": False,
            }
            write_json(directory / "result.json", results[method])
            continue
        event(output / "events.jsonl", "compression_branch_gate", method=method, passed=True)
        compressed, energy = svd_rank8(original, source.config["lora_alpha"])
        write_json(directory / "svd.json", energy)
        restore(model, compressed)
        saved = directory / "checkpoint"
        save_checkpoint(
            model,
            None,
            saved,
            {
                "kind": "rank8_svd_compression",
                "source_checkpoint": checkpoint,
                "source_task_id": settings["source_task_id"],
                "source_result_sha256": source.result_sha256,
                "rank": 8,
                "lora_alpha": 16,
                "training_updates": 0,
                "selected_using_test": False,
            },
        )
        event(output / "events.jsonl", "compressed_checkpoint_saved_before_test", method=method)
        restore(model, original)
        load_checkpoint(model, saved)
        compressed_results = {
            name: measure(model, encoded, rows, directory / "compressed" / f"{name}.json")
            for split in ("validation", "test")
            for name, rows in evaluation_sets(source, split).items()
        }
        restore(model, original)
        original_results = dict(original_dev)
        original_test_audit = {}
        for split in ("validation", "test"):
            for name, rows in evaluation_sets(source, split).items():
                if name not in original_results:
                    original_results[name] = measure(model, encoded, rows, directory / "original" / f"{name}.json")
                if split == "test" and name.startswith("primitive-"):
                    family = rows[0].family
                    expected = source.evaluation(f"fusion/{method}/{family}.test.json", rows)
                    original_test_audit[family] = comparison(expected, original_results[name])
        metadata = source.read_json(f"{checkpoint}/metadata.json")
        memory = {
            "original": state_memory(original),
            "compressed": state_memory(compressed),
            "original_checkpoint_file_bytes": source.verified_path(f"{checkpoint}/state.pt").stat().st_size,
            "compressed_checkpoint_file_bytes": (saved / "state.pt").stat().st_size,
            "retained_source_expert_archive_bytes_fp32": metadata["expert_archive_bytes_fp32"],
            "retained_source_expert_checkpoint_files": bank_archive_storage(source),
            "source_expert_training_updates": metadata["expert_training_updates"],
            "source_expert_training_example_exposures": metadata["expert_training_example_exposures"],
            "compressed_inference_requires_expert_archive": False,
            "source_archive_preserved": True,
            "compression_training_updates": 0,
            "svd_workspace_measured": False,
            "fixed_memory_sequential_training_claim": False,
        }
        source_test_exact = all(value["all_records_exact"] for value in original_test_audit.values())
        results[method] = {
            "status": "compression_evaluated"
            if source_test_exact
            else "compression_evaluated_with_source_test_reload_mismatch",
            "gate": fresh_gate,
            "svd_performed": True,
            "rank": 8,
            "offsets_removed": True,
            "selected_using_test": False,
            "energy": energy,
            "memory": memory,
            "behavior": {
                name: {
                    "original": original_results[name]["metrics"],
                    "compressed": record["metrics"],
                    "accuracy_change": record["metrics"]["accuracy"] - original_results[name]["metrics"]["accuracy"],
                }
                for name, record in compressed_results.items()
            },
            "source_test_reload": original_test_audit,
            "teacher_qualified": False,
            "checkpoint": str(saved.relative_to(output)),
        }
        write_json(directory / "result.json", results[method])
        del original, compressed
    return finish(
        source,
        output,
        {
            "status": "compression_followup_completed",
            "methods": results,
            "checkpoint_scope": "pre_workflow_fusion",
            "teacher_or_bank_distillation_updates": 0,
        },
        model,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    compress(json.loads(args.config.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
