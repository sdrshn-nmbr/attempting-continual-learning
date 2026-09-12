import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from compress_fusion import fusion_gate, state_memory
from data import digest, rng_for, scheduled_batch
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
    frozen_hash,
    layers,
    load_checkpoint,
    new_optimizer,
    restore,
    save_checkpoint,
    train_update,
    tree_record,
    write_json,
)
from run import TRANSPORT, prepare_output
from sandbox import FAMILIES, SCHEMAS

ROOT = Path(__file__).resolve().parent


class ArtifactRun(SourceRun):
    def __init__(self, root, task_id, config_sha256, required_files, source, code_sha256=None):
        self.root = Path(root).resolve()
        self.task = json.loads((self.root / "task.json").read_text())
        self.execution_payload = (self.root / "execution.json").read_bytes()
        self.execution_sha256 = hashlib.sha256(self.execution_payload).hexdigest()
        self.execution = json.loads(self.execution_payload)
        verify_execution(self.execution, self.task, "completed")
        settings = json.loads((self.root / "config.json").read_text())
        if self.task["id"] != task_id or self.task["config"] != settings or digest(settings) != config_sha256:
            raise RuntimeError("COMPRESSED_ARTIFACT_TASK_IDENTITY")
        if code_sha256 is not None and self.task["source_sha256"] != code_sha256:
            raise RuntimeError("COMPRESSED_ARTIFACT_SOURCE_IDENTITY")
        self.code_proof = verify_code_bundle(self.task["code_dir"], self.task["source_sha256"])
        for relative, expected in required_files.items():
            if self.code_proof["files"].get(relative) != expected:
                raise RuntimeError(f"COMPRESSED_ARTIFACT_CODE_CHANGED: {relative}")
        self.result = json.loads((self.root / "result.json").read_text())
        self.result_sha256 = sha256_file(self.root / "result.json")
        self.manifest = self.result["artifact_manifest"]
        self.transport_hashes = {
            name: sha256_file(self.root / name) for name in ("task.json", "config.json", "execution.json")
        }
        self.consumed = {}
        self.config, self.spec, self.corpus = source.config, source.spec, source.corpus


def program_files(protocol):
    checksum = hashlib.sha256()
    for relative, expected in sorted(protocol["dependency_files"].items()):
        path = ROOT / relative
        if path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError(f"COMPRESSED_REPLAY_DEPENDENCY_HASH: {relative}")
        checksum.update(relative.encode() + b"\0" + path.read_bytes())
    if checksum.hexdigest() != protocol["dependency_source_sha256"]:
        raise RuntimeError("COMPRESSED_REPLAY_DEPENDENCY_BUNDLE")
    return {
        **protocol["dependency_files"],
        **{name: sha256_file(ROOT / name) for name in ("compressed_replay.py", "compressed_replay_protocol.json")},
    }


def prepare(settings, output):
    if settings["experiment"] != "skill-transfer-compressed-replay" or settings["stage"] not in {"train", "audit"}:
        raise ValueError("COMPRESSED_REPLAY_EXPERIMENT")
    path = ROOT / settings["protocol"]
    if sha256_file(path) != settings["protocol_sha256"]:
        raise RuntimeError("COMPRESSED_REPLAY_PROTOCOL_HASH")
    protocol = json.loads(path.read_text())
    program = program_files(protocol)
    source = SourceRun(settings, json.loads((ROOT / "followup_protocol.json").read_text()))
    if source.config["workflow_checkpoints"] != [16, 64, 128] or source.config["batch_size"] != 4:
        raise RuntimeError("COMPRESSED_REPLAY_ORIGINAL_BUDGET")
    compressed = ArtifactRun(
        settings["compression_run_dir"],
        settings["compression_task_id"],
        settings["compression_config_sha256"],
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
    ):
        raise RuntimeError("COMPRESSED_REPLAY_SOURCE_CHAIN")
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
            training.task["entrypoint"] != "compressed_replay.py"
            or training.task["config"]["stage"] != "train"
            or training.task["config"]["protocol_sha256"] != settings["protocol_sha256"]
            or training.result["status"] != "compressed_workflow_training_completed"
            or training.result["compression_result_sha256"] != compressed.result_sha256
            or training.result["source_result_sha256"] != source.result_sha256
        ):
            raise RuntimeError("COMPRESSED_REPLAY_TRAINING_CHAIN")
    ancestors = [source, compressed] + ([training] if training is not None else [])
    resolved = output.resolve()
    if any(resolved.is_relative_to(run.root) or run.root.is_relative_to(resolved) for run in ancestors):
        raise RuntimeError("COMPRESSED_REPLAY_OUTPUT_OVERLAP")
    task = json.loads((output / "task.json").read_text())
    payload = (output / "execution.json").read_bytes()
    receipt = json.loads(payload)
    verify_execution(receipt, task, "running")
    if (
        task["config"] != settings
        or task["entrypoint"] != "compressed_replay.py"
        or Path(task["code_dir"]).resolve() != ROOT
        or Path(sys.argv[0]).resolve() != ROOT / "compressed_replay.py"
        or any(
            task["id"] == run.task["id"] or receipt["attempt_id"] == run.execution["attempt_id"] for run in ancestors
        )
        or any(
            datetime.fromisoformat(receipt["started_at"]) < datetime.fromisoformat(run.execution["finished_at"])
            for run in ancestors
        )
    ):
        raise RuntimeError("COMPRESSED_REPLAY_NOT_NEW_STANDALONE_EXECUTION")
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
                "supervisor": run.execution["supervisor"],
                "launcher_pid": run.execution["pid"],
                "finished_at": run.execution["finished_at"],
            }
            for run in ancestors
        ],
        "boundary": "Distinct completed ancestor tasks/attempts and bound standalone CLI. Launcher and supervisor PIDs are recorded with their roles; numeric PID inequality is not sole evidence.",
    }
    prepare_output(settings, output)
    for name, content in [("entry-execution.json", payload), ("compressed_replay_protocol.json", path.read_bytes())] + [
        (f"ancestor-{index}-execution.json", run.execution_payload) for index, run in enumerate(ancestors)
    ]:
        with (output / name).open("xb") as handle:
            handle.write(content)
    write_json(output / "process-proof.json", source.process_proof)
    write_json(output / "code-proof.json", code)
    return source, compressed, training, protocol


def rank8_memory(model):
    for name, layer in layers(model).items():
        if layer.offset is not None or layer.A.shape[0] != 8 or layer.B.shape[1] != 8 or layer.scaling != 2:
            raise RuntimeError(f"COMPRESSED_REPLAY_REQUIRES_RANK8_WITHOUT_OFFSET: {name}")
    memory = state_memory(capture(model))
    if not memory["factor_ranks"] or any(rank != 8 for rank in memory["factor_ranks"].values()):
        raise RuntimeError("COMPRESSED_REPLAY_EMPTY_OR_WRONG_RANK")
    return memory


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def replay_memory(source, protocol):
    selected, admission = [], {}
    for family in FAMILIES:
        available = {row.id: row for row in source.corpus["primitive"][family]["train"]}
        records = []
        for round_ in source.result["qualification"]["rounds"]:
            path = source.verified_path(f"qualification/{family}/{round_['budget']}/updates.jsonl")
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        if [row["step"] for row in records] != list(range(1, source.result["qualification"]["budget"] + 1)):
            raise RuntimeError(f"REPLAY_EXPOSURE_CLOCK: {family}")
        exposed = set()
        for record in records:
            if (
                record["replay_ids"]
                or len(record["current_ids"]) != 4
                or not set(record["current_ids"]) <= set(available)
            ):
                raise RuntimeError(f"REPLAY_SOURCE_EXPOSURE: {family}")
            exposed.update(record["current_ids"])
        retained = []
        for operation in SCHEMAS[family]:
            candidates = sorted(
                (available[item] for item in exposed if available[item].pattern == operation), key=lambda row: row.id
            )
            if len(candidates) < 4:
                raise RuntimeError(f"REPLAY_SOURCE_COVERAGE: {family}:{operation}")
            retained.extend(
                rng_for(source.config["optimization_seed"], protocol["id"], "memory", family, operation).sample(
                    candidates, 4
                )
            )
        selected.extend(retained)
        admission[family] = {
            "observed_unique": len(exposed),
            "retained": len(retained),
            "per_operation": dict(Counter(row.pattern for row in retained)),
        }
    if (
        len(selected) != 64
        or len({row.id for row in selected}) != 64
        or any(row.split != "train" or row.kind != "primitive" for row in selected)
    ):
        raise RuntimeError("REPLAY_MEMORY_CONTRACT")
    return selected, {
        "capacity": 64,
        "actual_examples": len(selected),
        "admission": admission,
        "ids": [row.id for row in selected],
        "rows": [row.record() for row in selected],
        "growth": False,
        "selection": "Four previously exposed TRAIN rows per operation, sixteen per family, deterministic sampling without replacement.",
    }


def workflow_batch(source, memory, arm, step, protocol):
    current = scheduled_batch(
        source.corpus["workflow"][source.config["target_family"]]["train"],
        step,
        4,
        [source.config["optimization_seed"], "workflow", source.config["target_family"]],
    )
    if arm == "compressed_continue":
        return current, []
    if arm != "compressed_replay" or len(memory) != 64:
        raise ValueError("COMPRESSED_REPLAY_BATCH_CONTRACT")
    old = rng_for(source.config["optimization_seed"], protocol["id"], "replay", step).sample(memory, 2)
    if any(row.kind != "primitive" or row.split != "train" for row in old):
        raise ValueError("COMPRESSED_REPLAY_OLD_SPLIT")
    return current[:2], old


def saved_gates(source, compressed, protocol):
    baseline = {
        family: source.evaluation(
            f"baseline/{family}.validation.json", source.corpus["primitive"][family]["validation"]
        )["metrics"]
        for family in FAMILIES
    }
    references, gates = {}, {}
    for method in protocol["methods"]:
        branch = compressed.result["methods"][method]
        if not branch["svd_performed"]:
            gates[method] = {"passed": False, "reason": "source_compression_branch_not_created"}
            continue
        references[method] = {
            family: compressed.evaluation(
                f"{method}/compressed/primitive-{family}.validation.json",
                source.corpus["primitive"][family]["validation"],
            )
            for family in FAMILIES
        }
        gates[method] = fusion_gate(
            baseline, {family: row["metrics"] for family, row in references[method].items()}, source.config
        )
    return baseline, references, gates


def train_arm(source, model, encoded, initial, method, arm, memory, initial_dev, protocol, output):
    restore(model, initial)
    optimizer = new_optimizer(model, source.config)
    initial_identity = digest(tree_record(capture(model)))
    optimizer_identity = digest(tree_record(optimizer.state_dict()))
    directory = output / method / arm
    directory.mkdir(parents=True)
    curve = [{"updates": 0, "panels": {name: row["metrics"] for name, row in initial_dev.items()}}]
    totals = {"updates": 0, "current_exposures": 0, "old_exposures": 0, "input_tokens": 0, "target_tokens": 0}
    old_by_family = Counter()
    for step in range(128):
        current, old = workflow_batch(source, memory, arm, step, protocol)
        report = train_update(model, optimizer, encoded, current + old)
        event(
            directory / "updates.jsonl",
            "update",
            step=step + 1,
            current_ids=[row.id for row in current],
            replay_ids=[row.id for row in old],
            **report,
        )
        totals["updates"] += 1
        totals["current_exposures"] += len(current)
        totals["old_exposures"] += len(old)
        old_by_family.update(row.family for row in old)
        for key in ("input_tokens", "target_tokens"):
            totals[key] += report[key]
        if step + 1 in (16, 64, 128):
            panels = {
                name: measure(model, encoded, rows, directory / f"dev-{step + 1}" / f"{name}.json")
                for name, rows in evaluation_sets(source, "validation").items()
            }
            curve.append({"updates": step + 1, "panels": {name: row["metrics"] for name, row in panels.items()}})
    if {int(value["step"]) for value in optimizer.state.values()} != {128} or totals["current_exposures"] + totals[
        "old_exposures"
    ] != 512:
        raise RuntimeError("COMPRESSED_REPLAY_UPDATE_CLOCK_OR_EXPOSURES")
    resident = rank8_memory(model)
    resident["optimizer_bytes"] = tensor_bytes(optimizer.state_dict())
    record = {
        "updates": 128,
        "initial_adapter_sha256": initial_identity,
        "initial_optimizer_sha256": optimizer_identity,
        "exposures": totals,
        "replay_by_family": dict(old_by_family),
        "resident": resident,
        "memory_examples": 64 if arm == "compressed_replay" else 0,
        "validation_curve": curve,
        "checkpoint_selected_using_test": False,
    }
    save_checkpoint(model, optimizer, directory / "checkpoint", record)
    record["checkpoint"] = str((directory / "checkpoint").relative_to(output))
    record["checkpoint_file_bytes"] = (directory / "checkpoint/state.pt").stat().st_size
    record["status"] = "trained_waiting_for_fresh_process_audit"
    write_json(directory / "training.json", record)
    return record


def train(source, compressed, protocol, output):
    baseline, references, gates = saved_gates(source, compressed, protocol)
    write_json(output / "saved-dev-gates.json", gates)
    methods = {}
    model = encoded = None
    memory = None
    memory_storage = {"actual_examples": 0, "file_bytes": 0}
    for method in protocol["methods"]:
        if not gates[method]["passed"]:
            methods[method] = {"status": "branch_gate_failed", "gate": gates[method], "training_updates": 0}
            continue
        if model is None:
            model, encoded = fresh_runtime(source, output)
        if load_checkpoint(model, compressed.checkpoint(f"{method}/checkpoint")) is not None:
            raise RuntimeError("COMPRESSED_SOURCE_CONTAINS_OPTIMIZER")
        initial_memory = rank8_memory(model)
        initial = capture(model)
        initial_dev = {
            name: measure(model, encoded, rows, output / method / "initial-dev" / f"{name}.json")
            for name, rows in evaluation_sets(source, "validation").items()
        }
        matches = {
            family: comparison(references[method][family], initial_dev[f"primitive-{family}.validation"])
            for family in FAMILIES
        }
        gate = fusion_gate(
            baseline,
            {family: initial_dev[f"primitive-{family}.validation"]["metrics"] for family in FAMILIES},
            source.config,
        )
        if not gate["passed"] or not all(row["all_records_exact"] for row in matches.values()):
            methods[method] = {
                "status": "branch_gate_failed",
                "reason": "fresh_dev_mismatch_or_unqualified",
                "gate": gate,
                "comparisons": matches,
                "training_updates": 0,
            }
            continue
        if memory is None:
            memory, record = replay_memory(source, protocol)
            write_json(output / "replay-memory.json", record)
            memory_storage = {
                "actual_examples": len(memory),
                "file_bytes": (output / "replay-memory.json").stat().st_size,
                "admission": record["admission"],
                "whole_process_memory_claim": False,
            }
        arms = {
            arm: train_arm(source, model, encoded, initial, method, arm, memory, initial_dev, protocol, output)
            for arm in protocol["arms"]
        }
        if (
            len({row["initial_adapter_sha256"] for row in arms.values()}) != 1
            or len({row["initial_optimizer_sha256"] for row in arms.values()}) != 1
        ):
            raise RuntimeError("COMPRESSED_REPLAY_FORK_NOT_IDENTICAL")
        methods[method] = {
            "status": "trained",
            "gate": gate,
            "initial_resident": initial_memory,
            "comparisons": matches,
            "arms": arms,
            "ancestral_accounting": compressed.result["methods"][method]["memory"],
            "svd": compressed.read_json(f"{method}/svd.json"),
            "training_updates": sum(row["updates"] for row in arms.values()),
        }
    return model, {
        "status": "compressed_workflow_training_completed",
        "methods": methods,
        "training_updates": sum(row["training_updates"] for row in methods.values()),
        "test_evaluated": False,
        "protocol_selected_using_test": False,
        "rehearsal_memory": memory_storage,
    }


def audit(source, compressed, training, protocol, output):
    methods = {}
    model = encoded = None
    for method in protocol["methods"]:
        trained = training.result["methods"][method]
        if trained["status"] != "trained":
            methods[method] = {"status": "source_branch_gate_failed", "source": trained}
            continue
        if model is None:
            model, encoded = fresh_runtime(source, output)
        arms = {}
        for arm in protocol["arms"]:
            checkpoint = trained["arms"][arm]["checkpoint"]
            metadata = training.read_json(f"{checkpoint}/metadata.json")
            if metadata["updates"] != 128 or metadata["checkpoint_selected_using_test"]:
                raise RuntimeError("COMPRESSED_REPLAY_AUDIT_CHECKPOINT")
            load_checkpoint(model, training.checkpoint(checkpoint))
            resident = rank8_memory(model)
            directory = output / method / arm
            panels, matches = {}, {}
            for name, rows in evaluation_sets(source, "validation").items():
                expected = training.evaluation(f"{method}/{arm}/dev-128/{name}.json", rows)
                panels[name] = measure(model, encoded, rows, directory / f"{name}.json")
                matches[name] = comparison(expected, panels[name])
            if not all(row["all_records_exact"] for row in matches.values()):
                arms[arm] = {"status": "fresh_process_dev_mismatch", "comparisons": matches, "test_evaluated": False}
                continue
            test = {
                name: measure(model, encoded, rows, directory / f"{name}.json")
                for name, rows in evaluation_sets(source, "test").items()
            }
            baseline = {
                name: compressed.evaluation(f"{method}/compressed/{name}.json", rows)["metrics"]
                for name, rows in evaluation_sets(source, "test").items()
            }
            dense = source.result["transfer"][method]
            arms[arm] = {
                "status": "fresh_process_audited",
                "comparisons": matches,
                "test_evaluated": True,
                "resident": resident,
                "dev": {name: row["metrics"] for name, row in panels.items()},
                "test": {name: row["metrics"] for name, row in test.items()},
                "compressed_initial_test": baseline,
                "dense_full_fusion_reference": {
                    "test": dense["test"],
                    "novel_composition_test": dense["novel_composition_test"],
                    "primitive_after_test": dense["primitive_after_test"],
                },
                "training_accounting": trained["arms"][arm],
                "ancestral_accounting": trained["ancestral_accounting"],
            }
            write_json(directory / "audit.json", arms[arm])
        methods[method] = {"status": "audited", "arms": arms}
    audited_arms = [arm for method in methods.values() for arm in method.get("arms", {}).values()]
    exact = all(arm["status"] == "fresh_process_audited" for arm in audited_arms)
    return model, {
        "status": "no_eligible_compressed_arms"
        if not audited_arms
        else "compressed_workflow_audit_completed"
        if exact
        else "compressed_workflow_audit_mismatch",
        "methods": methods,
        "all_dev_reloads_exact": exact if audited_arms else None,
        "evaluated_arms": len(audited_arms),
        "training_updates": 0,
    }


def complete(source, compressed, training, protocol, output, model, result):
    if model is not None and frozen_hash(model) != source.result["frozen_base"]["before"]:
        raise RuntimeError("COMPRESSED_REPLAY_FROZEN_BASE_CHANGED")
    program_files(protocol)
    verify_code_bundle(ROOT, source.process_proof["code_sha256"])
    for name, archive in (("original", source), ("compression", compressed), ("training", training)):
        if archive is not None:
            archive.verify_unchanged()
            write_json(output / f"consumed-{name}.json", archive.consumed)
    result.update(
        source_result_sha256=source.result_sha256,
        compression_result_sha256=compressed.result_sha256,
        process_proof=source.process_proof,
        teacher_qualified=False,
        source_archives_preserved=True,
        no_hidden_dense_offset=True,
    )
    result["artifact_manifest"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.relative_to(output).parts[0] not in TRANSPORT
    }
    write_json(output / "result.json", result)
    return result


def experiment(settings, output):
    source, compressed, training, protocol = prepare(settings, output)
    if settings["stage"] == "train":
        model, result = train(source, compressed, protocol, output)
    else:
        model, result = audit(source, compressed, training, protocol, output)
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
        logging.exception("COMPRESSED_REPLAY_FAILED output=%s", args.output_dir)
        raise
    if result.get("all_dev_reloads_exact") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
