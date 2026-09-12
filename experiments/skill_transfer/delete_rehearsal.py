import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from compress_fusion import fusion_gate
from compressed_replay import (
    ArtifactRun,
    audit,
    complete,
    program_files,
    rank8_memory,
    saved_gates,
    tensor_bytes,
)
from data import digest, scheduled_batch
from followup import (
    SourceRun,
    comparison,
    evaluation_sets,
    fresh_runtime,
    measure,
    sha256_file,
    verify_code_bundle,
    verify_execution,
)
from learning import (
    capture,
    event,
    finite_tensor,
    load_checkpoint,
    new_optimizer,
    save_checkpoint,
    tree_record,
    write_json,
)
from run import prepare_output
from sandbox import FAMILIES

ROOT = Path(__file__).resolve().parent
ARM = "delete_rehearsal"


def prepare(settings, output):
    if settings["experiment"] != "skill-transfer-delete-rehearsal" or settings["stage"] not in {"train", "audit"}:
        raise ValueError("DELETE_REHEARSAL_EXPERIMENT")
    path = ROOT / settings["protocol"]
    if sha256_file(path) != settings["protocol_sha256"]:
        raise RuntimeError("DELETE_REHEARSAL_PROTOCOL_HASH")
    protocol = json.loads(path.read_text())
    program = {
        **program_files(protocol),
        **{name: sha256_file(ROOT / name) for name in ("delete_rehearsal.py", "delete_rehearsal_protocol.json")},
    }
    source = SourceRun(settings, json.loads((ROOT / "followup_protocol.json").read_text()))
    if source.config["workflow_checkpoints"] != [16, 64, 128] or source.config["batch_size"] != 4:
        raise RuntimeError("DELETE_REHEARSAL_ORIGINAL_BUDGET")
    previous = json.loads((ROOT / "compressed_replay_protocol.json").read_text())
    compressed = ArtifactRun(
        settings["compression_run_dir"],
        settings["compression_task_id"],
        settings["compression_config_sha256"],
        previous["dependency_files"],
        source,
        previous["dependency_source_sha256"],
    )
    reference = ArtifactRun(
        settings["reference_run_dir"],
        settings["reference_task_id"],
        settings["reference_config_sha256"],
        protocol["dependency_files"],
        source,
        protocol["dependency_source_sha256"],
    )
    if (
        compressed.task["entrypoint"] != "compress_fusion.py"
        or compressed.result["status"] != "compression_followup_completed"
        or compressed.result["checkpoint_scope"] != "pre_workflow_fusion"
        or compressed.result["source_result_sha256"] != source.result_sha256
        or compressed.result["source_execution_sha256"] != source.execution_sha256
        or reference.task["entrypoint"] != "compressed_replay.py"
        or reference.task["config"]["stage"] != "train"
        or reference.result["status"] != "compressed_workflow_training_completed"
        or reference.result["source_result_sha256"] != source.result_sha256
        or reference.result["compression_result_sha256"] != compressed.result_sha256
        or reference.result["methods"]["agent_dice"]["status"] != "trained"
        or set(reference.result["methods"]["agent_dice"]["arms"]) != {"compressed_continue", "compressed_replay"}
    ):
        raise RuntimeError("DELETE_REHEARSAL_ANCESTOR_CHAIN")
    training = None
    if settings["stage"] == "audit":
        training = ArtifactRun(
            settings["training_run_dir"],
            settings["training_task_id"],
            settings["training_config_sha256"],
            program,
            source,
        )
        if (
            training.task["entrypoint"] != "delete_rehearsal.py"
            or training.task["config"]["stage"] != "train"
            or training.result["status"] != "compressed_workflow_training_completed"
            or training.result["reference_result_sha256"] != reference.result_sha256
            or training.result["source_result_sha256"] != source.result_sha256
            or training.result["compression_result_sha256"] != compressed.result_sha256
        ):
            raise RuntimeError("DELETE_REHEARSAL_CONTROL_CHAIN")
    ancestors = [source, compressed, reference] + ([training] if training is not None else [])
    if any(output.resolve().is_relative_to(run.root) or run.root.is_relative_to(output.resolve()) for run in ancestors):
        raise RuntimeError("DELETE_REHEARSAL_OUTPUT_OVERLAP")
    task = json.loads((output / "task.json").read_text())
    payload = (output / "execution.json").read_bytes()
    receipt = json.loads(payload)
    verify_execution(receipt, task, "running")
    if (
        task["config"] != settings
        or task["entrypoint"] != "delete_rehearsal.py"
        or Path(task["code_dir"]).resolve() != ROOT
        or Path(sys.argv[0]).resolve() != ROOT / "delete_rehearsal.py"
        or any(
            task["id"] == run.task["id"] or receipt["attempt_id"] == run.execution["attempt_id"] for run in ancestors
        )
        or any(
            datetime.fromisoformat(receipt["started_at"]) < datetime.fromisoformat(run.execution["finished_at"])
            for run in ancestors
        )
    ):
        raise RuntimeError("DELETE_REHEARSAL_NOT_NEW_STANDALONE_EXECUTION")
    code = verify_code_bundle(ROOT, task["source_sha256"])
    source.process_proof = {
        "stage": settings["stage"],
        "task_id": task["id"],
        "attempt_id": receipt["attempt_id"],
        "code_sha256": code["sha256"],
        "entry_execution_sha256": hashlib.sha256(payload).hexdigest(),
        "supervisor": receipt["supervisor"],
        "launcher_pid": receipt["pid"],
        "cli_pid": os.getpid(),
        "cli_parent_pid": os.getppid(),
        "ancestors": [
            {
                "task_id": run.task["id"],
                "attempt_id": run.execution["attempt_id"],
                "execution_sha256": run.execution_sha256,
                "finished_at": run.execution["finished_at"],
                "supervisor": run.execution["supervisor"],
                "launcher_pid": run.execution["pid"],
            }
            for run in ancestors
        ],
        "boundary": "Distinct supervised task/attempt and bound standalone CLI after completed ancestors; numeric PID inequality is not sole proof.",
    }
    prepare_output(settings, output)
    for name, content in [("entry-execution.json", payload), ("delete_rehearsal_protocol.json", path.read_bytes())] + [
        (f"ancestor-{index}-execution.json", run.execution_payload) for index, run in enumerate(ancestors)
    ]:
        with (output / name).open("xb") as handle:
            handle.write(content)
    write_json(output / "process-proof.json", source.process_proof)
    write_json(output / "code-proof.json", code)
    return source, compressed, reference, training, protocol


def current_objective(model, encoded, rows):
    if len(rows) != 2 or any(row.split != "train" or row.kind != "workflow" for row in rows):
        raise ValueError("DELETE_REHEARSAL_REQUIRES_TWO_CURRENT_TRAIN_ROWS")
    inputs, labels, keep = encoded.batch(rows, training=True)
    logits = model(**inputs, use_cache=False, logits_to_keep=keep + 1).logits[:, :-1].float()
    labels = labels[:, -keep:]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view_as(labels)
    mask = labels != -100
    row_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    loss = row_losses.sum() / 4
    return loss, {
        "current_mean_loss": float(row_losses.mean().detach()),
        "eos_loss": float(losses[:, -1].mean().detach()),
        "input_tokens": int(inputs["attention_mask"].sum()),
        "target_tokens": int(mask.sum()),
        "example_ids": [row.id for row in rows],
        "loss_denominator": 4,
    }


def current_update(model, optimizer, encoded, rows):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, report = current_objective(model, encoded, rows)
    if not torch.isfinite(loss):
        raise RuntimeError("DELETE_REHEARSAL_NONFINITE_LOSS")
    loss.backward()
    parameters = [value for value in model.parameters() if value.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, encoded.config["grad_clip"], error_if_nonfinite=True))
    if norm == 0:
        raise RuntimeError("DELETE_REHEARSAL_ZERO_GRADIENT")
    if any(value.grad is not None for value in model.parameters() if not value.requires_grad):
        raise RuntimeError("DELETE_REHEARSAL_FROZEN_GRADIENT")
    optimizer.step()
    if any(not finite_tensor(value) for value in parameters):
        raise RuntimeError("DELETE_REHEARSAL_NONFINITE_ADAPTER")
    return {**report, "loss": float(loss.detach()), "gradient_norm": norm}


def current_schedule(source, reference):
    path = reference.verified_path("agent_dice/compressed_replay/updates.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [row["step"] for row in records] != list(range(1, 129)):
        raise RuntimeError("DELETE_REHEARSAL_REFERENCE_UPDATE_CLOCK")
    schedule = []
    for step, record in enumerate(records):
        rows = scheduled_batch(
            source.corpus["workflow"][source.config["target_family"]]["train"],
            step,
            4,
            [source.config["optimization_seed"], "workflow", source.config["target_family"]],
        )[:2]
        if record["current_ids"] != [row.id for row in rows] or len(record["replay_ids"]) != 2:
            raise RuntimeError(f"DELETE_REHEARSAL_CURRENT_SCHEDULE_MISMATCH: step={step + 1}")
        schedule.append(rows)
    return schedule


def train(source, compressed, reference, protocol, output):
    baseline, _, gates = saved_gates(source, compressed, protocol)
    write_json(output / "saved-dev-gates.json", gates)
    result = {
        "status": "compressed_workflow_training_completed",
        "training_updates": 0,
        "test_evaluated": False,
        "protocol_selected_using_test": False,
        "memory_examples": 0,
        "methods": {"agent_dice": {"status": "branch_gate_failed", "gate": gates["agent_dice"], "training_updates": 0}},
    }
    if not gates["agent_dice"]["passed"]:
        return None, result
    model, encoded = fresh_runtime(source, output)
    if load_checkpoint(model, compressed.checkpoint("agent_dice/checkpoint")) is not None:
        raise RuntimeError("DELETE_REHEARSAL_INITIAL_OPTIMIZER_PRESENT")
    rank8_memory(model)
    identity = digest(tree_record(capture(model)))
    reference_arms = reference.result["methods"]["agent_dice"]["arms"]
    if any(row["initial_adapter_sha256"] != identity for row in reference_arms.values()):
        raise RuntimeError("DELETE_REHEARSAL_INITIAL_STATE_MISMATCH")
    initial, matches = {}, {}
    for name, rows in evaluation_sets(source, "validation").items():
        initial[name] = measure(model, encoded, rows, output / "agent_dice/initial-dev" / f"{name}.json")
        matches[name] = {
            "compression": comparison(compressed.evaluation(f"agent_dice/compressed/{name}.json", rows), initial[name]),
            "reference": comparison(reference.evaluation(f"agent_dice/initial-dev/{name}.json", rows), initial[name]),
        }
    gate = fusion_gate(
        baseline, {family: initial[f"primitive-{family}.validation"]["metrics"] for family in FAMILIES}, source.config
    )
    if not gate["passed"] or not all(
        value["all_records_exact"] for match in matches.values() for value in match.values()
    ):
        result["methods"]["agent_dice"].update(
            reason="fresh_dev_mismatch_or_unqualified", gate=gate, comparisons=matches
        )
        return model, result
    schedule = current_schedule(source, reference)
    optimizer = new_optimizer(model, source.config)
    optimizer_identity = digest(tree_record(optimizer.state_dict()))
    if any(row["initial_optimizer_sha256"] != optimizer_identity for row in reference_arms.values()):
        raise RuntimeError("DELETE_REHEARSAL_INITIAL_OPTIMIZER_MISMATCH")
    directory = output / "agent_dice" / ARM
    directory.mkdir(parents=True)
    curve = [{"updates": 0, "panels": {name: row["metrics"] for name, row in initial.items()}}]
    totals = {"updates": 0, "current_exposures": 0, "old_exposures": 0, "input_tokens": 0, "target_tokens": 0}
    for step, rows in enumerate(schedule, start=1):
        report = current_update(model, optimizer, encoded, rows)
        event(
            directory / "updates.jsonl",
            "update",
            step=step,
            current_ids=[row.id for row in rows],
            replay_ids=[],
            **report,
        )
        totals["updates"] += 1
        totals["current_exposures"] += 2
        for key in ("input_tokens", "target_tokens"):
            totals[key] += report[key]
        if step in (16, 64, 128):
            panels = {
                name: measure(model, encoded, rows_, directory / f"dev-{step}" / f"{name}.json")
                for name, rows_ in evaluation_sets(source, "validation").items()
            }
            curve.append({"updates": step, "panels": {name: row["metrics"] for name, row in panels.items()}})
    if totals["current_exposures"] != 256 or {int(value["step"]) for value in optimizer.state.values()} != {128}:
        raise RuntimeError("DELETE_REHEARSAL_UPDATE_CLOCK")
    resident = rank8_memory(model)
    resident["optimizer_bytes"] = tensor_bytes(optimizer.state_dict())
    record = {
        "updates": 128,
        "initial_adapter_sha256": identity,
        "initial_optimizer_sha256": optimizer_identity,
        "exposures": totals,
        "memory_examples": 0,
        "replay_by_family": {},
        "resident": resident,
        "validation_curve": curve,
        "checkpoint_selected_using_test": False,
        "loss_denominator": 4,
        "actual_batch_size": 2,
        "posthoc_mechanism_control": True,
    }
    save_checkpoint(model, optimizer, directory / "checkpoint", record)
    record.update(
        checkpoint=str((directory / "checkpoint").relative_to(output)),
        checkpoint_file_bytes=(directory / "checkpoint/state.pt").stat().st_size,
        status="trained_waiting_for_fresh_process_audit",
    )
    write_json(directory / "training.json", record)
    result.update(
        training_updates=128,
        methods={
            "agent_dice": {
                "status": "trained",
                "gate": gate,
                "comparisons": matches,
                "arms": {ARM: record},
                "training_updates": 128,
                "ancestral_accounting": reference.result["methods"]["agent_dice"]["ancestral_accounting"],
            }
        },
    )
    return model, result


def experiment(settings, output):
    source, compressed, reference, training, protocol = prepare(settings, output)
    if settings["stage"] == "train":
        model, result = train(source, compressed, reference, protocol, output)
    else:
        model, result = audit(source, compressed, training, protocol, output)
    reference.verify_unchanged()
    write_json(output / "consumed-reference.json", reference.consumed)
    result.update(
        experiment="skill-transfer-delete-rehearsal",
        posthoc_mechanism_control=True,
        original_test_previously_observed=True,
        reference_result_sha256=reference.result_sha256,
        reference_execution_sha256=reference.execution_sha256,
        reference_task_id=reference.task["id"],
    )
    return complete(source, compressed, training, protocol, output, model, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        result = experiment(json.loads(args.config.read_text()), args.output_dir)
    except Exception:
        logging.exception("DELETE_REHEARSAL_FAILED output=%s", args.output_dir)
        raise
    if result.get("all_dev_reloads_exact") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
