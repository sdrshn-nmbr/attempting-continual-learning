import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import torch

from data import audit_corpus, build_corpus, digest
from learning import (
    file_identity,
    frozen_hash,
    generate,
    load_checkpoint,
    load_model,
    load_tokenizer,
    score,
    write_json,
)
from run import TRANSPORT, prepare_output, validate_config
from sandbox import FAMILIES, grade_text

ROOT = Path(__file__).resolve().parent


def execution_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()


def verify_execution(receipt, task, required_status):
    if (
        receipt["task_id"] != task["id"]
        or receipt["task"] != task
        or receipt["source_sha256"] != task["source_sha256"]
        or receipt["task_sha256"] != execution_digest(receipt["task"])
        or receipt["config_sha256"] != execution_digest(receipt["task"]["config"])
        or not receipt.get("attempt_id")
        or not receipt.get("supervisor", {}).get("id")
    ):
        raise RuntimeError("EXECUTION_IDENTITY_MISMATCH")
    if receipt["status"] != required_status:
        raise RuntimeError(f"EXECUTION_STATUS: expected {required_status}, got {receipt['status']}")
    if required_status == "completed" and (
        type(receipt.get("exit_code")) is not int
        or receipt["exit_code"] != 0
        or receipt.get("timed_out") is not False
        or not receipt.get("finished_at")
    ):
        raise RuntimeError("SOURCE_EXECUTION_NOT_SUCCESSFUL")


def verify_code_bundle(directory, expected_sha256, expected_files=None):
    directory = Path(directory).resolve()
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") or part in {"__pycache__", "tests", "outputs", "runs"}
            for part in path.relative_to(directory).parts
        )
        and (
            path.suffix in {".py", ".json", ".txt", ".yaml", ".yml", ".sha256"}
            or path.name.startswith(("LICENSE", "NOTICE"))
        )
    )
    checksum, manifest = hashlib.sha256(), {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise RuntimeError(f"CODE_BUNDLE_SYMLINK: {path}")
        relative = str(path.relative_to(directory))
        payload = path.read_bytes()
        checksum.update(relative.encode() + b"\0" + payload)
        manifest[relative] = hashlib.sha256(payload).hexdigest()
    if checksum.hexdigest() != expected_sha256 or (expected_files is not None and manifest != expected_files):
        raise RuntimeError(f"CODE_BUNDLE_HASH: {directory}")
    return {"directory": str(directory), "sha256": checksum.hexdigest(), "files": manifest}


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


class SourceRun:
    def __init__(self, settings, protocol):
        self.root = Path(settings["source_run_dir"]).resolve()
        self.consumed = {}
        self.config = json.loads((self.root / "config.json").read_text())
        task = json.loads((self.root / "task.json").read_text())
        self.task = task
        self.execution_payload = (self.root / "execution.json").read_bytes()
        self.execution = json.loads(self.execution_payload)
        self.execution_sha256 = hashlib.sha256(self.execution_payload).hexdigest()
        verify_execution(self.execution, task, "completed")
        self.result = json.loads((self.root / "result.json").read_text())
        self.result_sha256 = sha256_file(self.root / "result.json")
        if self.result["status"] != "completed":
            raise RuntimeError("SOURCE_NOT_COMPLETED: require sealed final source results")
        self.manifest = self.result["artifact_manifest"]
        if (
            task["id"] != settings["source_task_id"]
            or task["source_sha256"] != settings["source_code_sha256"]
            or task["source_sha256"] != protocol["source_code_sha256"]
            or task["config"] != self.config
            or digest(self.config) != settings["source_config_sha256"]
            or self.result["config_sha256"] != digest(self.config)
            or self.result["protocol_sha256"] != protocol["source_protocol_sha256"]
        ):
            raise RuntimeError("SOURCE_IDENTITY_MISMATCH")
        self.transport_hashes = {
            "config.json": sha256_file(self.root / "config.json"),
            "task.json": sha256_file(self.root / "task.json"),
            "execution.json": self.execution_sha256,
        }
        self.code_proof = verify_code_bundle(
            task["code_dir"], settings["source_code_sha256"], protocol["frozen_source_files"]
        )
        validate_config(self.config)
        if sha256_file(self.verified_path("protocol.json")) != protocol["source_protocol_sha256"]:
            raise RuntimeError("SOURCE_PROTOCOL_MISMATCH")
        self.spec, self.corpus = build_corpus(self.config)
        if self.read_json("stream.json") != self.spec:
            raise RuntimeError("SOURCE_STREAM_RECONSTRUCTION")
        if self.read_json("dataset-audit.json") != audit_corpus(self.spec, self.corpus):
            raise RuntimeError("SOURCE_DATASET_RECONSTRUCTION")
        self.runtime = self.read_json("runtime.json")

    def verified_path(self, relative):
        relative = str(relative)
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or str(name) != relative:
            raise RuntimeError(f"SOURCE_PATH_ESCAPE: {relative}")
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"SOURCE_ARTIFACT_MISSING_OR_ESCAPED: {relative}")
        identity = file_identity(path)
        checksum = sha256_file(path)
        if checksum != self.manifest.get(relative) or identity != file_identity(path):
            raise RuntimeError(f"SOURCE_ARTIFACT_HASH: {relative}")
        self.consumed[relative] = {"sha256": checksum, "identity": identity}
        return path

    def read_json(self, relative):
        return json.loads(self.verified_path(relative).read_text())

    def checkpoint(self, relative):
        self.verified_path(f"{relative}/state.pt")
        self.verified_path(f"{relative}/metadata.json")
        return self.root / relative

    def evaluation(self, relative, rows):
        record = self.read_json(relative)
        if [item["id"] for item in record["records"]] != [row.id for row in rows]:
            raise RuntimeError(f"SOURCE_EVALUATION_ROWS: {relative}")
        if record["metrics"] != score(record["records"]):
            raise RuntimeError(f"SOURCE_EVALUATION_METRICS: {relative}")
        for saved, row in zip(record["records"], rows, strict=True):
            regenerated = grade_text(
                saved["text"], row, self.spec["conventions"], saved["native_eos"], saved["padding_only"]
            )
            if {
                key: value for key, value in saved.items() if key not in {"generated_ids", "raw_generation_ids"}
            } != regenerated:
                raise RuntimeError(f"SOURCE_EVALUATION_SCORING: {relative}:{row.id}")
        return record

    def verify_unchanged(self):
        verify_code_bundle(self.code_proof["directory"], self.code_proof["sha256"], self.code_proof["files"])
        if sha256_file(self.root / "result.json") != self.result_sha256:
            raise RuntimeError("SOURCE_RESULT_CHANGED")
        for name, checksum in self.transport_hashes.items():
            if sha256_file(self.root / name) != checksum:
                raise RuntimeError(f"SOURCE_TRANSPORT_CHANGED: {name}")
        for relative, record in self.consumed.items():
            path = self.root / relative
            if file_identity(path) != record["identity"] or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"SOURCE_CHANGED_DURING_FOLLOWUP: {relative}")


def verify_followup_process(settings, output, source, mode):
    task = json.loads((output / "task.json").read_text())
    payload = (output / "execution.json").read_bytes()
    receipt = json.loads(payload)
    verify_execution(receipt, task, "running")
    entrypoint = "audit_persistence.py" if mode == "audit" else "compress_fusion.py"
    if (
        task["id"] == source.task["id"]
        or receipt["attempt_id"] == source.execution["attempt_id"]
        or task["config"] != settings
        or task["entrypoint"] != entrypoint
        or Path(task["code_dir"]).resolve() != ROOT
        or Path(sys.argv[0]).resolve() != ROOT / entrypoint
        or datetime.fromisoformat(receipt["started_at"]) < datetime.fromisoformat(source.execution["finished_at"])
    ):
        raise RuntimeError("FOLLOWUP_NOT_DISTINCT_STANDALONE_EXECUTION")
    new_code = verify_code_bundle(task["code_dir"], task["source_sha256"])
    return {
        "source_task_id": source.task["id"],
        "source_attempt_id": source.execution["attempt_id"],
        "source_execution_sha256": source.execution_sha256,
        "source_supervisor": source.execution["supervisor"],
        "source_launcher_pid": source.execution["pid"],
        "source_pod": source.execution.get("pod"),
        "source_finished_at": source.execution["finished_at"],
        "followup_task_id": task["id"],
        "followup_attempt_id": receipt["attempt_id"],
        "followup_execution_sha256_at_entry": hashlib.sha256(payload).hexdigest(),
        "followup_supervisor": receipt["supervisor"],
        "followup_launcher_pid": receipt["pid"],
        "followup_pod": receipt.get("pod"),
        "followup_started_at": receipt["started_at"],
        "cli_pid": os.getpid(),
        "cli_parent_pid": os.getppid(),
        "standalone_entrypoint": str(ROOT / entrypoint),
        "followup_code_sha256": new_code["sha256"],
        "distinct_task_and_attempt_ids": True,
        "source_completed_before_followup_started": True,
        "pid_evidence_boundary": "Supervisor pid is the queue process; receipt pid identifies its launcher, not necessarily the final run_task/uv Python child. Distinct task and attempt IDs, completed-source receipt and standalone CLI/source binding establish the process boundary; numeric PID inequality alone does not.",
    }


def prepare_followup(settings, output, mode):
    if settings["experiment"] != f"skill-transfer-{mode}":
        raise ValueError("FOLLOWUP_EXPERIMENT")
    path = ROOT / settings["followup_protocol"]
    if sha256_file(path) != settings["followup_protocol_sha256"]:
        raise RuntimeError("FOLLOWUP_PROTOCOL_HASH")
    protocol = json.loads(path.read_text())
    for relative, checksum in protocol["frozen_source_files"].items():
        if sha256_file(ROOT / relative) != checksum:
            raise RuntimeError(f"FROZEN_SOURCE_CHANGED: {relative}")
    source = SourceRun(settings, protocol)
    resolved = output.resolve()
    if resolved.is_relative_to(source.root) or source.root.is_relative_to(resolved):
        raise RuntimeError("FOLLOWUP_OUTPUT_OVERLAPS_SOURCE")
    source.process_proof = verify_followup_process(settings, output, source, mode)
    prepare_output(settings, output)
    with (output / "source-execution.json").open("xb") as handle:
        handle.write(source.execution_payload)
    write_json(output / "source-code.json", source.code_proof)
    write_json(output / "process-proof.json", source.process_proof)
    with (output / "followup_protocol.json").open("xb") as handle:
        handle.write(path.read_bytes())
    write_json(
        output / "source.json",
        {
            "source_task_id": settings["source_task_id"],
            "source_code_sha256": settings["source_code_sha256"],
            "source_result_sha256": source.result_sha256,
            "source_execution_sha256": source.execution_sha256,
            "source_config_sha256": digest(source.config),
            "source_protocol_sha256": protocol["source_protocol_sha256"],
            "source_run_dir": str(source.root),
            "followup_protocol_sha256": settings["followup_protocol_sha256"],
        },
    )
    return source, protocol


def fresh_runtime(source, output):
    actual = {"torch": torch.__version__, "transformers": version("transformers"), "cuda_or_hip": torch.version.hip}
    expected = {key: source.runtime[key] for key in actual}
    if actual != expected:
        raise RuntimeError(f"FOLLOWUP_RUNTIME_MISMATCH: expected={expected}, actual={actual}")
    encoded = load_tokenizer(source.config, source.spec["conventions"])
    model = load_model(source.config)
    base_hash = frozen_hash(model)
    if base_hash != source.result["frozen_base"]["before"]:
        raise RuntimeError("FOLLOWUP_BASE_WEIGHT_MISMATCH")
    write_json(
        output / "runtime.json",
        {
            **actual,
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "fresh_base_loaded_from_disk": True,
            "training_updates": 0,
            "base_sha256": base_hash,
            "load_proof": model.skill_transfer_load_proof,
            "process_proof": source.process_proof,
        },
    )
    return model, encoded


def restore_source(model, source, relative):
    load_checkpoint(model, source.checkpoint(relative))


def comparison(expected, actual):
    before, after = expected["records"], actual["records"]
    if [item["id"] for item in before] != [item["id"] for item in after]:
        raise RuntimeError("AUDIT_ROW_ORDER_OR_COUNT")
    differences = [
        {
            "id": left["id"],
            "fields": sorted(
                key
                for key in left.keys() | right.keys()
                if key not in left or key not in right or left[key] != right[key]
            ),
        }
        for left, right in zip(before, after, strict=True)
        if left != right
    ]
    return {
        "n": len(before),
        "all_records_exact": before == after,
        "records_exact": len(before) - len(differences),
        "all_generated_ids_exact": all(
            a["generated_ids"] == b["generated_ids"] for a, b in zip(before, after, strict=True)
        ),
        "all_raw_generation_ids_exact": all(
            a["raw_generation_ids"] == b["raw_generation_ids"] for a, b in zip(before, after, strict=True)
        ),
        "all_semantic_outcomes_equal": all(a["correct"] == b["correct"] for a, b in zip(before, after, strict=True)),
        "all_format_outcomes_equal": all(
            a["format_valid"] == b["format_valid"] for a, b in zip(before, after, strict=True)
        ),
        "all_execution_outcomes_equal": all(
            a["executable"] == b["executable"] for a, b in zip(before, after, strict=True)
        ),
        "metrics_equal": expected["metrics"] == actual["metrics"],
        "expected_metrics": expected["metrics"],
        "actual_metrics": actual["metrics"],
        "expected_records_sha256": digest(before),
        "actual_records_sha256": digest(after),
        "differences": differences,
    }


def measure(model, encoded, rows, path):
    records = generate(model, encoded, rows)
    result = {"metrics": score(records), "records": records}
    write_json(path, result)
    return result


def evaluation_sets(source, split):
    sets = {f"primitive-{family}.{split}": source.corpus["primitive"][family][split] for family in FAMILIES}
    sets[f"workflow.{split}"] = source.corpus["workflow"][source.config["target_family"]][split]
    if split == "test":
        sets["workflow.novel-test"] = source.corpus["workflow"][source.config["target_family"]]["novel_test"]
    return sets


def finish(source, output, result, model=None):
    if model is not None and frozen_hash(model) != source.result["frozen_base"]["before"]:
        raise RuntimeError("FOLLOWUP_BASE_CHANGED")
    source.verify_unchanged()
    result.update(
        training_updates=0,
        source_unchanged=True,
        source_result_sha256=source.result_sha256,
        source_execution_sha256=source.execution_sha256,
        process_proof=source.process_proof,
    )
    write_json(output / "consumed-source.json", source.consumed)
    result["artifact_manifest"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.relative_to(output).parts[0] not in TRANSPORT
    }
    write_json(output / "result.json", result)
    return result
