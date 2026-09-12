import argparse
import copy
import json
import shutil
from pathlib import Path

import torch

import final_retention as final
import follow_through as follow
import learned_target
from calibrate_target import (
    file_pin,
    input_path,
    pin_bundle,
    restore_adapter,
    verify_file,
)
from data import SEQUENCE_TASKS, digest, write_json
from learner import tensor_hash
from preserve_tasks import OLD_TASKS


def verify_completed_correction(request):
    spec = request["corrected_retention"]
    root = Path(spec["path"]).resolve()
    for name, pin in spec["files"].items():
        verify_file(root / name, pin)
    verify_file(
        Path(request["source_run"]["path"]) / "training_receipt.json",
        request["expected_source_training_receipt"],
    )
    execution = json.loads((root / "execution.json").read_text())
    config = json.loads((root / "config.json").read_text())
    result = json.loads((root / "result.json").read_text())
    task = execution["task"]
    if (
        root.name != spec["task_id"]
        or execution["task_id"] != spec["task_id"]
        or execution["status"] != "completed"
        or execution["exit_code"] != 0
        or execution["timed_out"]
        or execution["source_sha256"] != spec["source_sha256"]
        or task["source_sha256"] != spec["source_sha256"]
        or execution["task_sha256"] != final.supervisor_digest(task)
        or execution["config_sha256"] != final.supervisor_digest(task["config"])
        or config != task["config"]
        or digest(config) != spec["config_sha256"]
        or result["config_sha256"] != spec["config_sha256"]
        or result["status"] != "completed"
        or not result["new_pid_reload"]
        or result["evaluation_pid"] == result["source_training_pid"]
        or config["source_run"] != request["source_run"]
    ):
        raise ValueError("LEARNED_SOURCE_CORRECTION_RECEIPT_MISMATCH")
    if final.archived_source_digest(Path(task["code_dir"])) != spec["source_sha256"]:
        raise ValueError("LEARNED_SOURCE_CORRECTION_CODE_HASH_MISMATCH")
    verified, original, training, prior = final.inspect_source(config)
    if (
        result["source_verification"] != verified
        or result["source_training_pid"] != training["pid"]
    ):
        raise ValueError("LEARNED_SOURCE_ORIGINAL_RECEIPTS_CHANGED")
    return root, config, result, verified, original, training, prior


def qualify(request, corrected, training):
    name, step = request["arm"], str(request["step"])
    trained = training["arms"][name]
    checkpoint = trained["checkpoints"][step]
    if checkpoint["saved"] != request["expected_checkpoint"]:
        raise ValueError("LEARNED_SOURCE_UNEXPECTED_CHECKPOINT_HASH")
    source_qualified = (
        trained["status"] == "completed_budget"
        and trained["optimizer_updates"] == request["step"]
        and trained["finite_completed_updates"] == request["step"]
        and int(max(trained["checkpoints"], key=int)) == request["step"]
        and checkpoint["qualification"]["all_tasks_qualified"]
        and all(
            checkpoint["train"]["metrics"][task]["accuracy"]
            >= request["minimum_train_accuracy"]
            for task in SEQUENCE_TASKS
        )
        and all(
            checkpoint["validation"]["metrics"][task]["accuracy"]
            >= request["minimum_validation_accuracy"]
            for task in SEQUENCE_TASKS
        )
    )
    arm = corrected["arms"][name]
    if (
        arm["retention_steps"] != {"before": 0, "after": request["step"]}
        or arm["checkpoints"][step]["adapter"] != checkpoint["saved"]
        or arm["checkpoints"][step]["validation"] != checkpoint["validation"]
        or not arm["checkpoints"][step]["reload_predictions_exact"]
    ):
        raise ValueError("LEARNED_SOURCE_FINAL_CHECKPOINT_BINDING_MISMATCH")
    before = arm["checkpoints"]["0"]["released_tasks"]
    after = arm["checkpoints"][step]["released_tasks"]
    retained = follow.retention_comparison(
        before, after, corrected["raw"]["released_tasks"]
    )
    if retained != arm["retention"]:
        raise ValueError("LEARNED_SOURCE_RETENTION_SUMMARY_MISMATCH")
    qualified_retention = all(
        retained[task]["initial_ability_qualified"]
        and retained[task]["examples"] == 32
        and -retained[task]["change"] * retained[task]["examples"]
        <= request["maximum_lost_rows_per_old_task"]
        for task in OLD_TASKS
    )
    return {
        "acquisition_qualified": bool(source_qualified),
        "retention_qualified": qualified_retention,
        "retention": retained,
        "qualification_scope": "acquisition_and_retention",
        "source_checkpoint": checkpoint["saved"],
    }


def prepare(request, output):
    if (
        request["kind"] != "qualify_example_learned_portal_source"
        or request["arm"] != "heads_new_latents"
        or request["qualification_scope"] != "acquisition_and_retention"
    ):
        raise ValueError("LEARNED_SOURCE_UNAUTHORIZED_QUALIFICATION_SCOPE")
    for name, pin in request["code_dependencies"].items():
        verify_file(Path(__file__).parent / name, pin)
    for spec in (request["source_run"], request["corrected_retention"]):
        source = Path(spec["path"]).resolve()
        if (
            source == output
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError("LEARNED_SOURCE_OUTPUT_MUST_BE_SEPARATE")
    if output.exists():
        raise FileExistsError(f"LEARNED_SOURCE_OUTPUT_ALREADY_USED: {output}")
    root, _, corrected, verified, original, training, _ = verify_completed_correction(
        request
    )
    qualification = qualify(request, corrected, training)
    if not qualification["acquisition_qualified"]:
        raise ValueError("LEARNED_SOURCE_ACQUISITION_UNQUALIFIED")
    if not qualification["retention_qualified"]:
        raise ValueError(
            "LEARNED_SOURCE_RETENTION_UNQUALIFIED_REQUIRES_NEW_PARENT_DECISION"
        )
    adapter = restore_adapter(qualification["source_checkpoint"], "cpu")
    initial = follow.load_source(original, "source_initial")
    if adapter.config != initial.config or not set(OLD_TASKS) <= set(
        adapter.config.tasks
    ):
        raise ValueError("LEARNED_SOURCE_NATIVE_CONFIG_CHANGED")
    old_indices = [
        i for i, task in enumerate(adapter.config.tasks) if task not in SEQUENCE_TASKS
    ]
    if not torch.equal(
        adapter.task_latents[old_indices], initial.task_latents[old_indices]
    ):
        raise ValueError("LEARNED_SOURCE_ORIGINAL_VECTORS_CHANGED")
    output.mkdir(parents=True)
    native = output / "native"
    native.mkdir()
    saved = qualification["source_checkpoint"]
    for name, pin in saved["artifact"]["files"].items():
        origin = Path(saved["artifact"]["path"]) / name
        destination = native / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        verify_file(destination, pin)
    exported = {**pin_bundle(native), "tensor_sha256": saved["tensor_sha256"]}
    if exported["files"] != saved["artifact"]["files"]:
        raise ValueError("LEARNED_SOURCE_EXPORT_CHANGED_BYTES")
    qualification.update(
        {
            "request_sha256": digest(request),
            "source_run": request["source_run"],
            "source_arm": request["arm"],
            "source_step": request["step"],
            "corrected_retention_run": request["corrected_retention"],
            "corrected_receipt_files": {
                name: file_pin(root / name)
                for name in ("execution.json", "config.json", "result.json")
            },
            "original_source_verification": verified,
            "exported_source": exported,
            "ABC_vector_sha256": {
                task: tensor_hash(
                    {"vector": adapter.task_latents[adapter.config.tasks.index(task)]}
                )
                for task in SEQUENCE_TASKS
            },
            "ABC_distance_from_original_RTE": {
                task: float(
                    torch.linalg.norm(
                        adapter.task_latents[adapter.config.tasks.index(task)]
                        - initial.task_latents[initial.config.tasks.index("rte")]
                    )
                )
                for task in SEQUENCE_TASKS
            },
            "all_original_task_vectors_frozen": True,
            "claim_boundary": "Example-trained source qualifies acquisition and at most two lost rows per measured old task. Only BoolQ may have qualified adapter gain; retained abilities are not all demonstrated adapter improvements. No unmeasured-skill claim.",
        }
    )
    write_json(output / "source_qualification.json", qualification)
    template = request["calibration_template"]
    verify_file(input_path(template["path"]), template)
    config = copy.deepcopy(json.loads(input_path(template["path"]).read_text()))
    config["kind"] = "portal_example_learned_target_calibration"
    del config["inputs"]["source_constructed"]
    config["inputs"]["source_learned"] = exported
    config["inputs"]["source_qualification"] = {
        "path": str(output / "source_qualification.json"),
        **file_pin(output / "source_qualification.json"),
    }
    config["protocol"]["arms"][0].update(
        name="learned", source="source_learned", routing="task_latents"
    )
    config["protocol"]["arms"][1]["routing"] = "task_latents"
    config["protocol_sha256"] = digest(config["protocol"])
    config["code_dependencies"] = request["code_dependencies"]
    config["method_source"] = {
        "url": "https://labs.ramp.com/research/portal-portable-task-adaptation/",
        "source": "https://github.com/ramp-public/portallib",
        "source_kind": "qualified_example_trained_heads_new_latents_step512",
        "transfer": "Freeze shared core and every task vector, including the learned ABC rows; fit fresh target alignment and layer embeddings using identical calibration examples across arms.",
    }
    config["claim_boundary"] = [
        "Transfer of the actual source heads_new_latents step512 generator learned from examples, with verified final acquisition and retention. No privileged rank8 target weights or constructed generator enter this experiment.",
        "Native ABC evaluation uses its frozen task-specific source vectors. Untouched native uses its original ABC rows, initially RTE copies; LoRA is persistent. No fixed-RTE substitution for the learned source.",
        "Same 64 calibration TRAIN rows per ABC task, batch4 per task, 320 updates, learning rate 1e-3, fixed fresh-alignment seed, FP32, and numeric checkpoints 0/20/80/160/320 as the preceding target64 comparison.",
        "Gold-answer NLL and character-normalized four-choice evaluation stay unchanged. Existing triples, validation/test, and length4 panels have been inspected previously; this is an exploratory transfer comparison.",
        "Heldout evaluation runs in a new PID after all arms finish. Target retention compares step0 with numeric final per arm. Source qualification does not imply target retention, faster learning, or unfamiliar-skill transfer.",
    ]
    learned_target.validate(config)
    learned_target.verify_qualification(config)
    write_json(output / "target64.json", config)
    write_json(output / "request.json", request)
    return {
        "status": "qualified_source_exported",
        "source": exported,
        "ready_config": {
            "path": str(output / "target64.json"),
            **file_pin(output / "target64.json"),
        },
        "ABC_vector_sha256": qualification["ABC_vector_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.config.read_text()), args.output_dir.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
