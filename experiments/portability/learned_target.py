import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import follow_through as follow
from calibrate_target import emit, file_pin, input_path, verify_file
from data import SEQUENCE_TASKS, digest, write_json
from learner import frozen_base_tensors, load_base, shared_hash, tensor_hash


def validate(config):
    protocol = config["protocol"]
    if (
        config["kind"] != "portal_example_learned_target_calibration"
        or protocol["mode"] != "calibration"
    ):
        raise ValueError("LEARNED_TARGET_UNKNOWN_EXPERIMENT")
    if digest(protocol) != config["protocol_sha256"]:
        raise ValueError("LEARNED_TARGET_PROTOCOL_HASH_MISMATCH")
    if (
        protocol["scoring"] != "character_normalized_four_choice_gold_answer_nll"
        or config["runtime"]["dtype"] != "float32"
        or config["runtime"]["autocast"]
    ):
        raise ValueError("LEARNED_TARGET_SCORING_OR_PRECISION_CHANGED")
    if "source_constructed" in config["inputs"] or not {
        "source_learned",
        "source_initial",
        "target_portal",
        "source_qualification",
    } <= set(config["inputs"]):
        raise ValueError("LEARNED_TARGET_WRONG_SOURCE_INPUTS")
    checkpoints = protocol["checkpoints"]
    if (
        checkpoints != sorted(set(checkpoints))
        or len(checkpoints) < 3
        or checkpoints[0] != 0
        or checkpoints[-1] <= protocol["gradient_gate_step"]
    ):
        raise ValueError("LEARNED_TARGET_INVALID_CHECKPOINTS")
    expected = [
        ("learned", "source_learned", "alignment", "task_latents"),
        ("untouched", "source_initial", "alignment", "task_latents"),
        ("lora", "source_initial", "lora", "persistent"),
    ]
    if len(protocol["arms"]) != len(expected):
        raise ValueError("LEARNED_TARGET_WRONG_ARMS")
    for arm, identity in zip(protocol["arms"], expected, strict=True):
        if (
            tuple(arm[key] for key in ("name", "source", "train", "routing"))
            != identity
            or arm["train_latents"]
            or arm["initialization"] != "released"
        ):
            raise ValueError("LEARNED_TARGET_CORE_AND_LEARNED_VECTORS_MUST_BE_FROZEN")
    for name, pin in config["code_dependencies"].items():
        verify_file(Path(__file__).parent / name, pin)


def verify_qualification(config):
    spec = config["inputs"]["source_qualification"]
    verify_file(input_path(spec["path"]), spec)
    qualification = json.loads(input_path(spec["path"]).read_text())
    if (
        not qualification["acquisition_qualified"]
        or not qualification["retention_qualified"]
        or qualification["qualification_scope"] != "acquisition_and_retention"
    ):
        raise ValueError("LEARNED_TARGET_SOURCE_NOT_JOINTLY_QUALIFIED")
    if qualification["exported_source"] != config["inputs"]["source_learned"]:
        raise ValueError("LEARNED_TARGET_QUALIFICATION_ARTIFACT_MISMATCH")
    return qualification


def prepare(config, output):
    validate(config)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {
        "config.json",
        "packages.txt",
        "execution.json",
        "task.json",
        "run.log",
        "attempts",
    }
    if any(path.is_symlink() or path.name not in allowed for path in output.iterdir()):
        raise FileExistsError(f"LEARNED_TARGET_OUTPUT_ALREADY_USED: {output}")
    if (output / "config.json").exists() and json.loads(
        (output / "config.json").read_text()
    ) != config:
        raise ValueError("LEARNED_TARGET_DISPATCH_CONFIG_MISMATCH")
    if (
        input_path(config["inputs"]["base"]["path"]).resolve()
        != Path(config["base"]["local_path"]).resolve()
    ):
        raise ValueError("LEARNED_TARGET_BASE_PATH_NOT_PINNED")
    manifest = follow.verify_inputs(config)
    qualification = verify_qualification(config)
    rows = follow.training_rows(config)
    schedule = follow.schedule_for(config, rows)
    adapters = {}
    for arm in config["protocol"]["arms"]:
        source = follow.load_source(config, arm["source"])
        adapter = follow.build_adapter(config, arm)
        if arm["train"] == "alignment":
            if shared_hash(adapter) != shared_hash(source) or not torch.equal(
                adapter.task_latents, source.task_latents
            ):
                raise ValueError("LEARNED_TARGET_TRANSPLANT_CHANGED_SHARED_STATE")
            vector_hashes = {
                task: tensor_hash(
                    {"vector": adapter.task_latents[adapter.config.tasks.index(task)]}
                )
                for task in SEQUENCE_TASKS
            }
            if (
                arm["name"] == "learned"
                and vector_hashes != qualification["ABC_vector_sha256"]
            ):
                raise ValueError(
                    "LEARNED_TARGET_ABC_VECTORS_DIFFER_FROM_QUALIFIED_SOURCE"
                )
        else:
            vector_hashes = None
        named, _, vectors = follow.parameter_groups(adapter, arm)
        if vectors or any(name.startswith(("core.", "latent.")) for name in named):
            raise ValueError("LEARNED_TARGET_ATTEMPTED_SHARED_OR_LATENT_TRAINING")
        adapters[arm["name"]] = {
            "trainable_parameters": sum(
                parameter.numel() for parameter in named.values()
            ),
            "ABC_vector_sha256": vector_hashes,
            "initial_tensor_sha256": tensor_hash(adapter.state_dict()),
        }
    write_json(output / "config.json", config)
    write_json(output / "input_manifest.json", manifest)
    write_json(
        output / "training_rows.json",
        {task: [asdict(row) for row in items] for task, items in rows.items()},
    )
    write_json(output / "schedule.json", schedule)
    write_json(output / "transplant_verification.json", adapters)
    emit(output, "learned_source_and_target_transplant_verified", adapters=adapters)
    return manifest, rows, schedule


def run(config, output, prepare_only=False):
    manifest, rows, schedule = prepare(config, output)
    if prepare_only:
        return {"status": "prepared_not_trained", "inputs": manifest}
    follow.setup_runtime(config)
    base = load_base(config["base"], device=config["runtime"]["device"])
    base_sha = tensor_hash(frozen_base_tensors(base.model))
    receipt = {
        "pid": os.getpid(),
        "config_sha256": digest(config),
        "input_manifest": manifest,
        "base_sha256": base_sha,
        "arms": {},
        "entrypoint": file_pin(Path(__file__)),
        "code_dependencies": config["code_dependencies"],
    }
    for arm in config["protocol"]["arms"]:
        destination = output / arm["name"]
        destination.mkdir()
        adapter = follow.build_adapter(config, arm).to(base.device)
        if (
            adapter.config.base_model_name_or_path != base.model_id
            or adapter.config.base_model_revision != base.revision
        ):
            raise ValueError("LEARNED_TARGET_ADAPTER_BASE_IDENTITY_MISMATCH")
        follow.hook_receipt(base, adapter.config, config["expected_hooks"])
        receipt["arms"][arm["name"]] = follow.fit_arm(
            base, adapter, arm, config, rows, schedule, destination
        )
        if tensor_hash(frozen_base_tensors(base.model)) != base_sha:
            raise ValueError("LEARNED_TARGET_TRAINING_BASE_CHANGED")
        write_json(output / "partial_training_receipt.json", receipt)
        del adapter
        gc.collect()
    write_json(output / "training_receipt.json", receipt)
    write_json(
        output / "training_receipt.pin.json", file_pin(output / "training_receipt.json")
    )
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
            str(output / "config.json"),
            "--output-dir",
            str(output),
            "--reload-evaluate",
            str(os.getpid()),
        ],
        check=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    return json.loads((output / "result.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reload-evaluate", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    try:
        if args.reload_evaluate is not None:
            verify_qualification(config)
            follow.evaluate_saved(
                config, args.output_dir.resolve(), args.reload_evaluate
            )
        else:
            run(config, args.output_dir.resolve(), args.prepare_only)
    except Exception as exc:
        if args.output_dir.is_dir():
            emit(
                args.output_dir,
                "learned_target_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
