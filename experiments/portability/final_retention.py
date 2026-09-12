import argparse
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from portallib import PortalModel
from portallib.evaluation import PortalInjector

import follow_through as follow
from calibrate_target import emit, file_pin, restore_adapter, verify_bundle, verify_file
from data import digest, write_json
from learner import frozen_base_tensors, load_base, tensor_hash

METADATA = (
    "execution.json",
    "task.json",
    "config.json",
    "training_receipt.json",
    "training_receipt.pin.json",
    "input_manifest.json",
    "result.json",
)
SELECTION = "step_zero_and_maximum_numeric_checkpoint_all_arms"


def supervisor_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()


def archived_source_digest(root):
    checksum = hashlib.sha256()
    excluded = {"__pycache__", "tests", "outputs", "runs"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and not any(p.startswith(".") or p in excluded for p in relative.parts)
            and (
                path.suffix in {".py", ".json", ".txt", ".yaml", ".yml", ".sha256"}
                or path.name.startswith(("LICENSE", "NOTICE"))
            )
        ):
            checksum.update(str(relative).encode() + b"\0" + path.read_bytes())
    return checksum.hexdigest()


def inspect_source(config):
    if config["kind"] != "portal_final_retention" or config["selection"] != SELECTION:
        raise ValueError("FINAL_RETENTION_UNKNOWN_CONTRACT")
    source = config["source_run"]
    root = Path(source["path"]).resolve()
    if root.name != source["task_id"]:
        raise ValueError("FINAL_RETENTION_WRONG_SOURCE_DIRECTORY")
    metadata = {name: file_pin(root / name) for name in METADATA}
    values = {name: json.loads((root / name).read_text()) for name in METADATA}
    execution, task = values["execution.json"], values["task.json"]
    original, receipt = values["config.json"], values["training_receipt.json"]
    prior = values["result.json"]
    verify_file(root / "training_receipt.json", values["training_receipt.pin.json"])
    if (
        execution["status"] != "completed"
        or execution["exit_code"] != 0
        or execution["timed_out"]
        or execution["task_id"] != source["task_id"]
        or task["id"] != source["task_id"]
        or task != execution["task"]
        or original != task["config"]
        or execution["task_sha256"] != supervisor_digest(execution["task"])
        or execution["config_sha256"] != supervisor_digest(execution["task"]["config"])
        or digest(original) != source["config_sha256"]
        or receipt["config_sha256"] != source["config_sha256"]
        or execution["source_sha256"] != source["source_sha256"]
        or task["source_sha256"] != source["source_sha256"]
        or prior["status"] != "completed"
        or prior["config_sha256"] != source["config_sha256"]
        or prior["training_pid"] != receipt["pid"]
        or not prior["new_pid_reload"]
        or prior["evaluation_pid"] == receipt["pid"]
    ):
        raise ValueError("FINAL_RETENTION_SOURCE_EXECUTION_RECEIPT_MISMATCH")
    follow.validate(original)
    if original["protocol"]["mode"] not in {"examples", "calibration"}:
        raise ValueError("FINAL_RETENTION_TRAINING_RUN_REQUIRED")
    code = Path(task["code_dir"]).resolve()
    if archived_source_digest(code) != source["source_sha256"]:
        raise ValueError("FINAL_RETENTION_ARCHIVED_SOURCE_HASH_MISMATCH")
    implementation = receipt["implementation"]
    verify_file(
        code / task["entrypoint"], implementation[str(code / task["entrypoint"])]
    )
    for module in (PortalModel, PortalInjector):
        installed = Path(inspect.getfile(module))
        pins = [
            pin
            for name, pin in implementation.items()
            if Path(name).name == installed.name
        ]
        if len(pins) != 1:
            raise ValueError("FINAL_RETENTION_SDK_PIN_MISSING")
        verify_file(installed, pins[0])
    for name in ("learner.py", "data.py", "calibrate_target.py"):
        verify_file(Path(__file__).parent / name, file_pin(code / name))
    bound = copy.deepcopy(original)
    for spec in bound["inputs"].values():
        if not Path(spec["path"]).is_absolute():
            spec["path"] = str(code / spec["path"])
    if (
        Path(bound["inputs"]["base"]["path"]).resolve()
        != Path(bound["base"]["local_path"]).resolve()
    ):
        raise ValueError("FINAL_RETENTION_BASE_PATH_NOT_PINNED")
    if (
        not follow.verify_inputs(bound)
        == receipt["input_manifest"]
        == values["input_manifest.json"]
    ):
        raise ValueError("FINAL_RETENTION_INPUT_MANIFEST_MISMATCH")
    if (
        dict(Counter(row.task for row in follow.retention_rows(bound)))
        != config["probe_counts"]
    ):
        raise ValueError("FINAL_RETENTION_RELEASED_PROBE_COUNTS_CHANGED")
    names = {arm["name"] for arm in original["protocol"]["arms"]}
    if names != set(receipt["arms"]) or names != set(prior["arms"]):
        raise ValueError("FINAL_RETENTION_SOURCE_ARM_SET_MISMATCH")
    endpoints = {}
    for arm in original["protocol"]["arms"]:
        name = arm["name"]
        training = receipt["arms"][name]
        steps = sorted(training["checkpoints"], key=int)
        if (
            [int(step) for step in steps] != original["protocol"]["checkpoints"]
            or steps[0] != "0"
            or int(steps[-1]) != config["expected_final_step"]
            or training["status"] != "completed_budget"
            or training["optimizer_updates"] != int(steps[-1])
            or training["finite_completed_updates"] != int(steps[-1])
        ):
            raise ValueError(f"FINAL_RETENTION_INCOMPLETE_BUDGET: {name}")
        endpoints[name] = [steps[0], steps[-1]]
        for step in endpoints[name]:
            checkpoint = training["checkpoints"][step]
            saved = checkpoint["saved"]
            expected_path = root / name / "checkpoints" / f"step-{int(step):04d}"
            if (
                checkpoint["step"] != int(step)
                or Path(saved["artifact"]["path"]).resolve() != expected_path
                or saved["kind"] not in {"portal", "lora"}
            ):
                raise ValueError(
                    f"FINAL_RETENTION_CHECKPOINT_IDENTITY_MISMATCH: {name}/{step}"
                )
            verify_bundle(saved["artifact"])
    return (
        {
            "root": str(root),
            "metadata": metadata,
            "endpoints": endpoints,
            "source_code_sha256": source["source_sha256"],
            "source_config_sha256": source["config_sha256"],
            "base_tensor_sha256": receipt["base_sha256"],
        },
        bound,
        receipt,
        prior,
    )


def prepare(config, output):
    source = Path(config["source_run"]["path"]).resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("FINAL_RETENTION_OUTPUT_MUST_BE_SEPARATE_FROM_SOURCE")
    output.mkdir(parents=True, exist_ok=True)
    allowed = {
        "config.json",
        "task.json",
        "execution.json",
        "packages.txt",
        "run.log",
        "attempts",
    }
    if any(p.is_symlink() or p.name not in allowed for p in output.iterdir()):
        raise FileExistsError(f"FINAL_RETENTION_OUTPUT_ALREADY_USED: {output}")
    if (output / "config.json").exists() and json.loads(
        (output / "config.json").read_text()
    ) != config:
        raise ValueError("FINAL_RETENTION_DISPATCH_CONFIG_MISMATCH")
    verified, _, _, _ = inspect_source(config)
    write_json(output / "config.json", config)
    write_json(output / "source_verification.json", verified)
    write_json(
        output / "preparation.json",
        {
            "pid": os.getpid(),
            "config_sha256": digest(config),
            "source_verification": file_pin(output / "source_verification.json"),
        },
    )
    emit(output, "final_retention_source_verified", endpoints=verified["endpoints"])
    return verified


def evaluate_saved(config, output, preparation_pid):
    preparation = json.loads((output / "preparation.json").read_text())
    if preparation_pid == os.getpid() or preparation["pid"] != preparation_pid:
        raise ValueError("FINAL_RETENTION_NEW_EVALUATION_PID_REQUIRED")
    if preparation["config_sha256"] != digest(config):
        raise ValueError("FINAL_RETENTION_EVALUATION_CONFIG_CHANGED")
    verify_file(output / "source_verification.json", preparation["source_verification"])
    verified, original, receipt, prior = inspect_source(config)
    if verified != json.loads((output / "source_verification.json").read_text()):
        raise ValueError("FINAL_RETENTION_SOURCE_CHANGED_AFTER_PREPARATION")
    if os.getpid() in {receipt["pid"], prior["evaluation_pid"]}:
        raise ValueError("FINAL_RETENTION_NEW_EVALUATION_PID_REQUIRED")
    follow.setup_runtime(original)
    base = load_base(original["base"], device=original["runtime"]["device"])
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("FINAL_RETENTION_RELOADED_BASE_CHANGED")
    retained, validation = (
        follow.retention_rows(original),
        follow.validation_rows(original),
    )
    raw = follow.measure(base, None, retained, None, original)
    if raw != prior["raw"]["released_tasks"]:
        raise ValueError("FINAL_RETENTION_RAW_PREDICTIONS_DIFFER")
    results = {}
    for arm in original["protocol"]["arms"]:
        name = arm["name"]
        measured = {}
        for step in verified["endpoints"][name]:
            checkpoint = receipt["arms"][name]["checkpoints"][step]
            adapter = restore_adapter(
                checkpoint["saved"], original["runtime"]["device"]
            )
            if (
                adapter.config.base_model_name_or_path != base.model_id
                or adapter.config.base_model_revision != base.revision
            ):
                raise ValueError("FINAL_RETENTION_ADAPTER_BASE_IDENTITY_MISMATCH")
            follow.hook_receipt(base, adapter.config, original["expected_hooks"])
            reloaded = follow.measure(base, adapter, validation, arm, original)
            if reloaded != checkpoint["validation"]:
                raise ValueError(
                    f"FINAL_RETENTION_RELOAD_PREDICTIONS_DIFFER: {name}/{step}"
                )
            panel = follow.measure(base, adapter, retained, arm, original)
            if (
                step == "0"
                and panel != prior["arms"][name]["checkpoints"]["0"]["released_tasks"]
            ):
                raise ValueError(f"FINAL_RETENTION_INITIAL_PREDICTIONS_DIFFER: {name}")
            if (
                tensor_hash(adapter.state_dict())
                != checkpoint["saved"]["tensor_sha256"]
            ):
                raise ValueError("FINAL_RETENTION_EVALUATION_MUTATED_ADAPTER")
            measured[step] = {
                "released_tasks": panel,
                "validation": reloaded,
                "adapter": checkpoint["saved"],
                "reload_predictions_exact": True,
            }
            emit(
                output, "final_retention_checkpoint_evaluated", arm=name, step=int(step)
            )
            del adapter
        first, last = verified["endpoints"][name]
        results[name] = {
            "retention_steps": {"before": int(first), "after": int(last)},
            "checkpoints": measured,
            "retention": follow.retention_comparison(
                measured[first]["released_tasks"], measured[last]["released_tasks"], raw
            ),
            "source_final_qualification": receipt["arms"][name]["checkpoints"][last][
                "qualification"
            ],
        }
    if tensor_hash(frozen_base_tensors(base.model)) != receipt["base_sha256"]:
        raise ValueError("FINAL_RETENTION_EVALUATION_BASE_CHANGED")
    for name, pin in verified["metadata"].items():
        verify_file(Path(verified["root"]) / name, pin)
    result = {
        "status": "completed",
        "kind": config["kind"],
        "source_run": config["source_run"],
        "source_verification": verified,
        "source_training_pid": receipt["pid"],
        "source_evaluation_pid": prior["evaluation_pid"],
        "preparation_pid": preparation_pid,
        "evaluation_pid": os.getpid(),
        "new_pid_reload": True,
        "config_sha256": digest(config),
        "raw": {"released_tasks": raw},
        "arms": results,
        "claim_boundary": "Corrective final-retention evaluation only. Numeric final checkpoints, original task routing and scoring; no training, checkpoint selection, or replacement of archived results.",
    }
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reload-evaluate", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    try:
        if args.reload_evaluate is not None:
            evaluate_saved(config, output, args.reload_evaluate)
        else:
            prepare(config, output)
            if not args.prepare_only:
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
    except Exception as exc:
        source = Path(config["source_run"]["path"]).resolve()
        if (
            output.is_dir()
            and output != source
            and not output.is_relative_to(source)
            and not source.is_relative_to(output)
        ):
            emit(
                output,
                "final_retention_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
