import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import torch
from compare import prediction_index, same_examples
from data import SEQUENCE_TASKS, digest, write_json
from evaluate_targets import (
    checkpoint_files,
    file_sha256,
    fixed_config,
    frozen_fp32,
    load_target_base,
    portal_identity,
    row_identity,
    validate_bootstrap,
    validated_sources,
    verify_primary_files,
    verify_snapshot,
    verify_source_files,
)
from fresh_inputs import load_fresh_rows
from learner import (
    evaluate,
    fp32_forward,
    frozen_base_tensors,
    load_base,
    load_portal,
    save_native,
    tensor_hash,
    transplant,
)
from metrics import paired_group_interval, task_rows
from peft import PeftModel, get_peft_model_state_dict
from portallib import PortalModel
from run import emit, require_gpu

ROOT = Path(__file__).parent
CONFIG_SHA256S = {
    "qwen8": "cb1615cc25250314be0727bd2bd8e960c5775a7388bcf27431646eb8697aabc6",
    "mistral7": "840e0f6cd8409a06b250c6a5b74e3d3b3fe30a7efea796379c94e04fded7412b",
    "qwen4": "71dcc074d328f09eb93b635c255e04befb81d98507694095f3c6d8a6ba6e9b30",
}
TARGET_CONDITIONS = ("raw", "initial", "unchanged", "repair", "mismatch")
SOURCE_CONDITIONS = ("raw", "initial", "native_replay", "lora_replay")


def validate_study_config(config):
    expected = SOURCE_CONDITIONS if config["role"] == "source" else TARGET_CONDITIONS
    if tuple(config["conditions"]) != expected or config["role"] not in (
        "source",
        "target",
    ):
        raise ValueError("REPAIR_EVAL_ROLE_CONDITIONS_MISMATCH")
    if config["evaluation"] != {
        "batch_size": 8,
        "max_prompt": 768,
        "dtype": "float32",
        "autocast": False,
    }:
        raise ValueError("REPAIR_EVAL_FROZEN_ARITHMETIC_MISMATCH")
    if config["role"] == "source":
        spec = config["source_base"]
        if not spec.get("files") or Path(spec["local_path"]).name != spec["revision"]:
            raise ValueError("REPAIR_EVAL_SOURCE_FILE_MANIFEST_REQUIRED")
        for row in spec["files"]:
            if (
                set(row) != {"path", "bytes", "sha256", "git_blob_sha1"}
                or row["bytes"] <= 0
                or not (row["sha256"] or row["git_blob_sha1"])
            ):
                raise ValueError("REPAIR_EVAL_SOURCE_FILE_PIN_REQUIRED")
        names = {row["path"] for row in spec["files"]}
        if not {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        } <= names or not any(name.endswith(".safetensors") for name in names):
            raise ValueError("REPAIR_EVAL_SOURCE_RUNTIME_FILES_REQUIRED")
    elif set(config["carriers"]) != {"repair", "mismatch"}:
        raise ValueError("REPAIR_EVAL_BOTH_CARRIERS_REQUIRED")


def checked_files(spec):
    result = {}
    for name, expected in spec["files"].items():
        path = Path(spec["path"]) / name
        if file_sha256(path) != expected:
            raise ValueError(f"REPAIR_EVAL_INPUT_FILE_CHANGED: {path}")
        result[str(path)] = expected
    return result


def read_primary(config):
    spec = config["primary_config"]
    path = ROOT / spec["path"]
    if file_sha256(path) != spec["sha256"]:
        raise ValueError("REPAIR_EVAL_PRIMARY_CONFIG_CHANGED")
    primary = fixed_config(path)
    verify_primary_files(primary)
    return primary


def geometry_receipt(config):
    spec = config["geometry_audit"]
    path = Path(spec["path"])
    if file_sha256(path) != spec["sha256"]:
        raise ValueError("REPAIR_EVAL_GEOMETRY_AUDIT_CHANGED")
    audit = json.loads(path.read_text())
    if (
        audit["status"] != "passed"
        or not audit["separate_process"]
        or audit["audit_pid"] == audit["fit_pid"]
        or audit["audit_optimizer_updates"] != 0
        or audit["total_fit_updates"] != 600
        or not audit["all_input_pins_unchanged"]
        or not audit["all_new_code_config_pins_unchanged"]
        or not audit["gates"][config["target"]]["passed"]
        or not all(audit["gates"][config["target"]]["checks"].values())
    ):
        raise ValueError("REPAIR_EVAL_GEOMETRY_GATE_NOT_PASSED")
    return audit


def prepare_target_cases(initial, learned, original, carriers, output):
    learned_identity = portal_identity(learned)
    original_identity = portal_identity(original)
    source_before = {"initial": portal_identity(initial), "learned": learned_identity}
    if list(original.config.tasks) != list(learned.config.tasks[:14]):
        raise ValueError("REPAIR_EVAL_OLD_TASK_TABLE_MISMATCH")
    for label, carrier in carriers.items():
        identity = portal_identity(carrier)
        if (
            identity["tasks"] != original_identity["tasks"]
            or identity["core_sha256"] != learned_identity["core_sha256"]
            or identity["published_vectors_sha256"]
            != learned_identity["published_vectors_sha256"]
            or carrier.config.base_model_name_or_path
            != original.config.base_model_name_or_path
            or carrier.config.base_model_revision != original.config.base_model_revision
            or carrier.config.to_dict()["projection_targets"]
            != original.config.to_dict()["projection_targets"]
        ):
            raise ValueError(
                f"REPAIR_EVAL_CARRIER_CORE_VECTOR_OR_TARGET_MISMATCH: {label}"
            )
    cases = {}
    for label, source, carrier in (
        ("initial", initial, original),
        ("unchanged", learned, original),
        ("repair", learned, carriers["repair"]),
        ("mismatch", learned, carriers["mismatch"]),
    ):
        transported = transplant(source, carrier).requires_grad_(False).eval()
        identity = portal_identity(transported)
        source_identity = portal_identity(source)
        if (
            identity["shared_sha256"] != source_identity["shared_sha256"]
            or identity["alignment_sha256"]
            != portal_identity(carrier)["alignment_sha256"]
        ):
            raise ValueError(f"REPAIR_EVAL_ALIGNMENT_ONLY_GRAFT_FAILED: {label}")
        checkpoint = output / "checkpoints" / label
        save_native(transported, checkpoint)
        cases[label] = {
            "path": str(checkpoint),
            "files": checkpoint_files(checkpoint),
            "identity": identity,
        }
    if source_before != {
        "initial": portal_identity(initial),
        "learned": portal_identity(learned),
    } or original_identity != portal_identity(original):
        raise ValueError("REPAIR_EVAL_SOURCE_OR_ORIGINAL_MUTATED")
    return cases


def prepare_study(config, output, primary):
    if config["role"] not in ("target", "source"):
        raise ValueError("REPAIR_EVAL_UNDECLARED_ROLE")
    sources, states, _ = validated_sources(primary)
    rows, fixture = load_fresh_rows(config["fresh_inputs"])
    files, geometry = {}, None
    if config["role"] == "target":
        if tuple(config["conditions"]) != TARGET_CONDITIONS:
            raise ValueError("REPAIR_EVAL_ALL_TARGET_CONDITIONS_REQUIRED")
        geometry = geometry_receipt(config)
        files[config["geometry_audit"]["path"]] = config["geometry_audit"]["sha256"]
        original_spec = primary["target"]["portal"]
        files.update(
            {
                str(Path(original_spec["local_path"]) / name): checksum
                for name, checksum in verify_snapshot(original_spec).items()
            }
        )
        original = load_portal(original_spec).requires_grad_(False).eval()
        carriers = {}
        for label, spec in config["carriers"].items():
            files.update(checked_files(spec))
            carriers[label] = (
                PortalModel.from_pretrained(
                    spec["path"],
                    local_files_only=True,
                    device="cpu",
                    dtype=torch.float32,
                )
                .requires_grad_(False)
                .eval()
            )
            frozen_fp32(carriers[label], label)
        cases = prepare_target_cases(
            states["initial"], states["native_replay"], original, carriers, output
        )
    else:
        if tuple(config["conditions"]) != SOURCE_CONDITIONS:
            raise ValueError("REPAIR_EVAL_ALL_SOURCE_CONDITIONS_REQUIRED")
        cases = {}
        for label in ("initial", "native_replay"):
            checkpoint = output / "checkpoints" / label
            save_native(states[label], checkpoint)
            cases[label] = {
                "path": str(checkpoint),
                "files": checkpoint_files(checkpoint),
                "identity": portal_identity(states[label]),
            }
        files.update(checked_files(config["lora"]))
    prepared = {
        "parent_pid": os.getpid(),
        "config_sha256": digest(config),
        "role": config["role"],
        "target": primary["target"],
        "sources": sources,
        "input_files": files,
        "fresh_inputs_sha256": config["fresh_inputs"]["sha256"],
        "row_identity_sha256": digest(row_identity(rows)),
        "rows": len(rows),
        "groups": len(fixture["unique_inputs"]),
        "cases": cases,
        "geometry": geometry,
    }
    write_json(output / "prepared.json", prepared)
    emit(
        output,
        "repair_evaluation_prepared",
        role=config["role"],
        target=config["target"],
        rows=len(rows),
        cases=list(cases),
        optimizer_updates=0,
    )
    return prepared


def verify_prepared_inputs(prepared):
    verify_source_files(prepared)
    for path, checksum in prepared["input_files"].items():
        if file_sha256(Path(path)) != checksum:
            raise ValueError(f"REPAIR_EVAL_PREPARED_INPUT_CHANGED: {path}")


def measure_panel(base, portal, rows, config, output, label):
    frozen_fp32(base.model, label)
    before_base = tensor_hash(frozen_base_tensors(base.model))
    before_portal = portal_identity(portal) if portal is not None else None
    observations = []

    def inspect_logits(module, inputs, result):
        arithmetic = (
            str(result.logits.dtype),
            torch.is_grad_enabled(),
            torch.is_autocast_enabled(base.device.type),
        )
        if arithmetic != ("torch.float32", False, False):
            raise ValueError(f"REPAIR_EVAL_FP32_FORWARD_CONTRACT: {label}/{arithmetic}")
        observations.append(arithmetic)

    handle = base.model.register_forward_hook(inspect_logits)
    try:
        with torch.inference_mode(), fp32_forward(base):
            panel = evaluate(
                base,
                rows,
                config["evaluation"]["max_prompt"],
                config["evaluation"]["batch_size"],
                portal,
            )
    finally:
        handle.remove()
    same_examples(row_identity(rows), prediction_index(panel["predictions"]))
    after_base = tensor_hash(frozen_base_tensors(base.model))
    frozen_fp32(base.model, label)
    if (
        not observations
        or before_base != after_base
        or (portal is not None and before_portal != portal_identity(portal))
    ):
        raise ValueError(f"REPAIR_EVAL_STATE_OR_FORWARD_PROOF_FAILED: {label}")
    check = {
        "base_sha256": before_base,
        "portal_identity": before_portal,
        "forward_calls": len(observations),
        "forward_dtype": "torch.float32",
        "autocast": False,
        "grad_enabled": False,
        "base_and_portal_unchanged": True,
    }
    write_json(output / f"panel_{label}.json", {"evaluation": panel, "checks": check})
    emit(
        output,
        "repair_evaluation_panel",
        condition=label,
        metrics=panel["metrics"],
        forward_calls=len(observations),
    )
    return panel, check


def comparisons(config, panels, source):
    if config["role"] == "source":
        result = {}
        for condition in ("native_replay", "lora_replay"):
            result[condition] = {}
            for task in SEQUENCE_TASKS:
                accuracy = panels[condition]["metrics"][task]["accuracy"]
                floor = max(
                    panels[label]["metrics"][task]["accuracy"]
                    for label in ("raw", "initial")
                )
                result[condition][task] = {
                    "accuracy": accuracy,
                    "gain_over_floors": accuracy - floor,
                    "fresh_input_competence": accuracy >= 0.90
                    and accuracy - floor >= 0.50,
                }
        return result
    result = {}
    for task in SEQUENCE_TASKS:
        scores = {
            label: panel["metrics"][task]["accuracy"] for label, panel in panels.items()
        }
        original = source["acquisition"]["tasks"][task]
        source_retained = (
            original["acquired"] and original["maximum_forgetting"] <= 0.05 + 1e-12
        )
        result[task] = {
            "accuracies": scores,
            "source_acquired_and_retained_on_original_test": source_retained,
            "repair_gain_over_both_floors": scores["repair"]
            - max(scores["raw"], scores["initial"]),
            "repair_minus_unchanged": scores["repair"] - scores["unchanged"],
            "repair_minus_mismatch": scores["repair"] - scores["mismatch"],
            "target_gain_flag": source_retained
            and scores["repair"] - max(scores["raw"], scores["initial"])
            >= 0.05 - 1e-12,
            "repair_specific_gain_flag": scores["repair"]
            - max(scores["unchanged"], scores["mismatch"])
            >= 0.05 - 1e-12,
            "paired_intervals": {
                label: paired_group_interval(
                    task_rows(panels["repair"], task),
                    task_rows(panels[label], task),
                    config["seed"],
                )
                for label in ("raw", "initial", "unchanged", "mismatch")
            },
        }
    return result


def evaluate_prepared(config, output, prepared_sha256, device="cuda:0"):
    path = output / "prepared.json"
    if file_sha256(path) != prepared_sha256:
        raise ValueError("REPAIR_EVAL_PREPARED_RECEIPT_CHANGED")
    prepared = json.loads(path.read_text())
    if prepared["parent_pid"] == os.getpid() or prepared["config_sha256"] != digest(
        config
    ):
        raise ValueError("REPAIR_EVAL_FRESH_PROCESS_REQUIRED")
    verify_prepared_inputs(prepared)
    rows, _ = load_fresh_rows(config["fresh_inputs"])
    if (
        len(rows) != prepared["rows"]
        or digest(row_identity(rows)) != prepared["row_identity_sha256"]
    ):
        raise ValueError("REPAIR_EVAL_FRESH_INPUT_IDENTITY_CHANGED")
    if device != "cpu":
        require_gpu(output, config["seed"])
    if config["role"] == "target":
        base, base_receipt = load_target_base(prepared["target"], device)
    else:
        base_receipt = {"files": verify_snapshot(config["source_base"])}
        base = load_base(config["source_base"], device)
        if (
            tensor_hash(frozen_base_tensors(base.model))
            != config["source_base_tensor_sha256"]
        ):
            raise ValueError("REPAIR_EVAL_SOURCE_BASE_IDENTITY_MISMATCH")
    panels, checks = {}, {}
    for label in config["conditions"]:
        portal, lora_hash = None, None
        if label in prepared["cases"]:
            spec = prepared["cases"][label]
            checked_files(spec)
            portal = (
                PortalModel.from_pretrained(
                    spec["path"],
                    local_files_only=True,
                    device=device,
                    dtype=torch.float32,
                )
                .requires_grad_(False)
                .eval()
            )
            if portal_identity(portal) != spec["identity"]:
                raise ValueError(f"REPAIR_EVAL_RELOADED_CASE_CHANGED: {label}")
            frozen_fp32(portal, label)
        elif label == "lora_replay":
            model = (
                PeftModel.from_pretrained(
                    base.model,
                    config["lora"]["path"],
                    is_trainable=False,
                    local_files_only=True,
                )
                .requires_grad_(False)
                .eval()
            )
            lora_hash = tensor_hash(
                get_peft_model_state_dict(model, save_embedding_layers=False)
            )
            if lora_hash != config["lora"]["tensor_sha256"]:
                raise ValueError("REPAIR_EVAL_LORA_RELOAD_MISMATCH")
            base = replace(base, model=model)
        elif label != "raw":
            raise ValueError(f"REPAIR_EVAL_MISSING_CASE: {label}")
        panels[label], checks[label] = measure_panel(
            base, portal, rows, config, output, label
        )
        if lora_hash is not None:
            if lora_hash != tensor_hash(
                get_peft_model_state_dict(base.model, save_embedding_layers=False)
            ):
                raise ValueError("REPAIR_EVAL_LORA_MUTATED")
            checks[label]["lora_tensor_sha256"] = lora_hash
        del portal
    verify_prepared_inputs(prepared)
    result = {
        "status": "completed",
        "kind": config["kind"],
        "role": config["role"],
        "target": config["target"],
        "config_sha256": digest(config),
        "prepared_sha256": prepared_sha256,
        "parent_pid": prepared["parent_pid"],
        "evaluator_pid": os.getpid(),
        "fresh_process": True,
        "fresh_inputs_sha256": config["fresh_inputs"]["sha256"],
        "conditions": config["conditions"],
        "paired_rows": len(rows),
        "groups": prepared["groups"],
        "prediction_count": sum(len(panel["predictions"]) for panel in panels.values()),
        "optimizer_updates": 0,
        "geometry_alignment_updates_per_carrier": 100
        if config["role"] == "target"
        else 0,
        "base_receipt": base_receipt,
        "panels": panels,
        "checks": checks,
        "comparisons": comparisons(
            config, panels, prepared["sources"]["native_replay"]
        ),
        "protocol": config["protocol"],
        "claim_boundary": "Fixed fixture303 mappings and unused input triples. Choice-scoring behavior only. Geometry fitting used released old-task adapter functions, never new-task examples. Source fresh-input competence is a separate paired report; no target/model or checkpoint selection.",
    }
    write_json(output / "result.json", result)
    emit(
        output,
        "repair_evaluation_completed",
        role=config["role"],
        target=config["target"],
        paired_rows=len(rows),
        predictions=result["prediction_count"],
        optimizer_updates=0,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("run", "prepare", "evaluate"), default="run"
    )
    parser.add_argument("--prepared-sha256")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if file_sha256(args.config) != CONFIG_SHA256S.get(config.get("target")):
        raise ValueError("REPAIR_EVAL_UNDECLARED_CONFIG")
    validate_study_config(config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase in ("run", "prepare"):
            validate_bootstrap(config, output)
            if (output / "prepared.json").exists():
                raise ValueError("REPAIR_EVAL_OUTPUT_ALREADY_PREPARED")
            prepare_study(config, output, read_primary(config))
        if args.phase == "run":
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--python",
                    sys.executable,
                    "python",
                    str(Path(__file__).resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--output-dir",
                    str(output.resolve()),
                    "--phase",
                    "evaluate",
                    "--prepared-sha256",
                    file_sha256(output / "prepared.json"),
                ],
                check=True,
            )
        elif args.phase == "evaluate":
            if not args.prepared_sha256:
                raise ValueError("REPAIR_EVAL_PREPARED_HASH_REQUIRED")
            evaluate_prepared(config, output, args.prepared_sha256)
    except Exception as error:
        emit(
            output,
            "repair_evaluation_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
