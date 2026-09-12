import hashlib
import json
import math
import os
import re
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
CONTRACT = "onpolicy303_fp32_native_weight_export_20260912"
TASKS = ("sequence_a", "sequence_b", "sequence_c")
SOURCE_FILES = ("native303_export.py", "native303_contract.py")
TOLERANCES = {
    "source_scores": {"atol": 1e-5, "rtol": 1e-6},
    "merged_scores": {"atol": 1e-4, "rtol": 1e-5},
    "prefix_logits": {"atol": 1e-3, "rtol": 1e-5},
    "merged_weights": {"atol": 1e-6, "rtol": 1e-6},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, label, detail=None):
    if not condition:
        raise ValueError(f"NATIVE303_{label}: {detail}")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())


def file_pin(path, git_blob=False):
    path = Path(path)
    start = path.stat()
    sha = hashlib.sha256()
    blob = (
        hashlib.sha1(b"blob " + str(start.st_size).encode() + b"\0")
        if git_blob
        else None
    )
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            if blob is not None:
                blob.update(chunk)
    end = path.stat()
    require(
        (start.st_size, start.st_mtime_ns, start.st_ino)
        == (end.st_size, end.st_mtime_ns, end.st_ino),
        "FILE_CHANGED_WHILE_HASHING",
        str(path),
    )
    result = {"sha256": sha.hexdigest(), "bytes": start.st_size}
    if blob is not None:
        result["git_blob_sha1"] = blob.hexdigest()
    return result


def checked_pin(path, expected):
    actual = file_pin(path, bool(expected.get("git_blob_sha1")))
    for key in ("bytes", "sha256", "git_blob_sha1"):
        if expected.get(key) is not None:
            require(
                actual.get(key) == expected[key],
                "FILE_PIN_MISMATCH",
                {
                    "path": str(path),
                    "field": key,
                    "expected": expected[key],
                    "actual": actual.get(key),
                },
            )
    return actual


def tensor_digest(values):
    combined, metadata = hashlib.sha256(), {}
    for name, value in sorted(values.items()):
        require(
            value.grad is None
            and (not value.is_floating_point() or value.dtype == torch.float32),
            "FP32_FROZEN_TENSOR",
            name,
        )
        combined.update(name.encode())
        combined.update(str((tuple(value.shape), value.dtype)).encode())
        raw_hash, finite = hashlib.sha256(), True
        for chunk in value.detach().contiguous().reshape(-1).split(1024 * 1024):
            finite = finite and bool(torch.isfinite(chunk).all())
            raw = chunk.view(torch.uint8).cpu().numpy().tobytes()
            combined.update(raw)
            raw_hash.update(raw)
        require(finite, "NONFINITE_TENSOR", name)
        metadata[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": raw_hash.hexdigest(),
            "elements": value.numel(),
        }
    require(bool(metadata), "EMPTY_TENSOR_SET")
    return {
        "tensor_sha256": combined.hexdigest(),
        "tensors": metadata,
        "elements": sum(row["elements"] for row in metadata.values()),
    }


def numeric_difference(actual, expected, tolerance):
    require(
        actual.shape == expected.shape,
        "NUMERIC_SHAPE",
        [list(actual.shape), list(expected.shape)],
    )
    a, b = actual.detach().double(), expected.detach().double()
    require(
        bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "NONFINITE_PARITY_VALUE",
    )
    diff = a - b
    absolute = diff.abs()
    allowed = tolerance["atol"] + tolerance["rtol"] * b.abs()
    flat = absolute.flatten()
    index = int(flat.argmax()) if flat.numel() else None
    return {
        "passed": bool((absolute <= allowed).all()),
        "elements": a.numel(),
        "changed_elements": int(torch.count_nonzero(diff)),
        "outside_tolerance": int((absolute > allowed).sum()),
        "max_absolute_difference": float(flat.max()) if index is not None else 0.0,
        "rms_difference": float(diff.square().mean().sqrt())
        if index is not None
        else 0.0,
        "maximum_flat_index": index,
        "actual_at_maximum": float(a.flatten()[index]) if index is not None else None,
        "expected_at_maximum": float(b.flatten()[index]) if index is not None else None,
        "tolerance": tolerance,
    }


def grade(row, tokens, body, eos, special_ids, cap):
    ended = bool(tokens) and tokens[-1] == eos
    content = tokens[:-1] if ended else tokens
    valid = re.fullmatch(r" [0-7] [0-7] [0-7]", body) is not None and not any(
        token in special_ids for token in content
    )
    gold = row["choices"][row["gold_idx"]]
    return {
        "correct": len(tokens) <= cap and ended and valid and body == gold,
        "format_valid": valid,
        "terminated": ended,
        "cap_without_eos": len(tokens) >= cap and not ended,
        "extra_body_characters": len(body) > 6,
        "correct_prefix_with_extra_text": body.startswith(gold) and body != gold,
        "exact_body_without_valid_stop": body == gold and not ended,
        "digit_position_correct": [
            len(body) > index and body[index] == gold[index] for index in (1, 3, 5)
        ],
    }


def validate_native(rows, records, token_contract):
    require(len(rows) == len(records), "NATIVE_ROW_COUNT")
    for row, record in zip(rows, records, strict=True):
        require(
            {key: record[key] for key in ("id", "task", "group", "row_sha256")}
            == {
                "id": row["id"],
                "task": row["task"],
                "group": row["group"],
                "row_sha256": digest(row),
            },
            "NATIVE_ROW_BINDING",
            row["id"],
        )
        output = record["generation"]
        expected = grade(
            row,
            output["token_ids"],
            output["body_text"],
            token_contract["eos_token_id"],
            token_contract["special_token_ids"],
            16,
        )
        require(
            all(output[key] == value for key, value in expected.items()),
            "NATIVE_GRADING",
            row["id"],
        )


def compare_native(actual, expected):
    require(len(actual) == len(expected), "NATIVE_PARITY_COUNT")
    differences = [
        {"index": i, "id": old["id"], "expected": old, "actual": new}
        for i, (new, old) in enumerate(zip(actual, expected, strict=True))
        if new != old
    ]
    return {
        "passed": not differences,
        "count": len(actual),
        "identical_records": len(actual) - len(differences),
        "differences": differences,
        "correct": sum(row["generation"]["correct"] for row in actual),
        "native_eos": sum(row["generation"]["terminated"] for row in actual),
    }


def validate_generic(rows, records):
    require(len(rows) == len(records), "GENERIC_ROW_COUNT")
    for row, record in zip(rows, records, strict=True):
        scores = record["scores"]
        require(
            len(scores) == len(row["choices"])
            and all(math.isfinite(score) for score in scores),
            "GENERIC_SCORES",
        )
        winner = max(range(len(scores)), key=scores.__getitem__)
        expected = {
            "row_sha256": digest(row),
            "task": row["task"],
            "scores": scores,
            "choice_count": len(scores),
            "gold": row["gold_idx"],
            "prediction": winner,
            "correct": winner == row["gold_idx"],
        }
        require(record == expected, "GENERIC_RECORD_BINDING", digest(row))


def compare_generic(actual, expected, tolerance):
    require(len(actual) == len(expected), "GENERIC_PARITY_COUNT")
    rows, passed = [], True
    for new, old in zip(actual, expected, strict=True):
        require(
            {key: new[key] for key in ("row_sha256", "task", "gold", "choice_count")}
            == {
                key: old[key] for key in ("row_sha256", "task", "gold", "choice_count")
            },
            "GENERIC_PARITY_ROW",
        )
        numeric = numeric_difference(
            torch.tensor(new["scores"], dtype=torch.float64),
            torch.tensor(old["scores"], dtype=torch.float64),
            tolerance,
        )
        exact_decision = (
            new["prediction"] == old["prediction"] and new["correct"] == old["correct"]
        )
        rows.append(
            {
                "row_sha256": old["row_sha256"],
                "task": old["task"],
                "prediction_equal": exact_decision,
                "expected_scores": old["scores"],
                "actual_scores": new["scores"],
                "score_differences": [
                    a - b for a, b in zip(new["scores"], old["scores"], strict=True)
                ],
                "numeric": numeric,
                "expected_prediction": old["prediction"],
                "actual_prediction": new["prediction"],
            }
        )
        passed = passed and exact_decision and numeric["passed"]
    return {
        "passed": passed,
        "count": len(rows),
        "prediction_changes": sum(not row["prediction_equal"] for row in rows),
        "rows": rows,
    }


def compare_prefixes(actual, expected, references, tolerance):
    require(set(actual) == set(expected), "PREFIX_WITNESS_KEYS")
    rows = []
    for i, record in enumerate(references):
        key = f"row_{i:04d}"
        require(
            actual[key].shape == expected[key].shape
            and actual[key].shape[0] == len(record["generation"]["token_ids"]),
            "PREFIX_WITNESS_SHAPE",
            key,
        )
        for step in range(actual[key].shape[0]):
            rows.append(
                {
                    "id": record["id"],
                    "step": step,
                    "prefix_token_ids": record["generation"]["prompt_token_ids"]
                    + record["generation"]["token_ids"][:step],
                    "actual_argmax": int(actual[key][step].argmax()),
                    "expected_argmax": int(expected[key][step].argmax()),
                    **numeric_difference(
                        actual[key][step], expected[key][step], tolerance
                    ),
                }
            )
    return {
        "passed": all(row["passed"] for row in rows),
        "prefixes": len(rows),
        "rows": rows,
        "comparison_inputs": "Identical fixed prefixes from the original student's 24 saved TRAIN generations, including the prefix predicting EOS.",
    }


def validated_corpus(design):
    fixtures = {}
    for name, pin in design["fixtures"].items():
        path = ROOT / pin["path"]
        checked_pin(path, pin)
        fixtures[name] = read_json(path)
    original = fixtures["original"]
    require(original["provenance"]["fixture_seed"] == 303, "WRONG_FIXTURE")
    corpus = {
        split: [row for task in TASKS for row in original["splits"][f"{task}_{split}"]]
        for split in ("train", "validation", "test")
    }
    corpus["unused288"] = fixtures["unused"]["rows"]
    ids, groups = set(), {}
    for split, count in {
        "train": 128,
        "validation": 32,
        "test": 64,
        "unused288": 288,
    }.items():
        rows = corpus[split]
        require(
            Counter(row["task"] for row in rows) == dict.fromkeys(TASKS, count),
            "FIXED_SPLIT_COUNTS",
            split,
        )
        groups[split] = {row["group"] for row in rows}
        require(len(groups[split]) == count, "FIXED_INPUT_COUNTS", split)
        for row in rows:
            inputs = tuple(map(int, row["group"].split()))
            require(
                len(inputs) == 3 and all(i in range(8) for i in inputs),
                "FIXED_INPUT_DOMAIN",
            )
            require(
                row["prompt"]
                == f"Apply the {row['task']} code.\nInput: {row['group']}\nOutput:"
                and row["id"] == digest({"task": row["task"], "input": inputs})
                and row["id"] not in ids,
                "FIXED_PROMPT_UID",
            )
            gold = " " + " ".join(
                str(original["provenance"]["rules"][row["task"]][i]) for i in inputs
            )
            require(row["choices"][row["gold_idx"]] == gold, "FIXED_ORACLE")
            ids.add(row["id"])
    require(len(set.union(*groups.values())) == 512, "CROSS_SPLIT_INPUT_OVERLAP")
    probes = [
        row
        for task in TASKS
        for row in [r for r in corpus["train"] if r["task"] == task][:8]
    ]
    generic = fixtures["generic"]["validation"]
    require(
        len(generic) == 128
        and Counter(row["task"] for row in generic)
        == dict.fromkeys(design["generic_tasks"], 32),
        "FIXED_GENERIC_PROBES",
    )
    return {
        "train": probes,
        "generic": generic,
        "test": corpus["test"],
        "unused288": corpus["unused288"],
    }, {split: digest(rows) for split, rows in corpus.items()}


def verify_source_archive(spec):
    checked_pin(spec["path"], spec["archive"])
    files = {}
    with tarfile.open(spec["path"]) as archive:
        for member in archive.getmembers():
            if member.isfile():
                require(
                    not Path(member.name).is_absolute()
                    and ".." not in Path(member.name).parts
                    and member.name not in files,
                    "SOURCE_ARCHIVE_PATH",
                )
                files[member.name] = archive.extractfile(member).read()
    sha = hashlib.sha256()
    for name in sorted(files, key=Path):
        sha.update(name.encode() + b"\0" + files[name])
    require(
        sha.hexdigest() == spec["source_sha256"]
        and ("files" not in spec or len(files) == spec["files"]),
        "SOURCE_ARCHIVE_CLOSURE",
    )
    return {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }


def verify_execution(directory, task_id, source=None, completed=True, config=None):
    directory = Path(directory)
    execution = read_json(directory / "execution.json")
    task = read_json(directory / "task.json")
    actual_config = read_json(directory / "config.json")
    require(
        execution["task_id"] == task_id
        and task["id"] == task_id
        and execution["task"] == task,
        "EXECUTION_TASK",
    )
    require(
        execution["task_sha256"] == file_pin(directory / "task.json")["sha256"]
        and execution["config_sha256"] == file_pin(directory / "config.json")["sha256"]
        and task["config"] == actual_config,
        "EXECUTION_CONFIG_HASH",
    )
    if config is not None:
        require(config == actual_config, "ACTUAL_DISPATCH_CONFIG")
    if source is not None:
        require(
            execution["source_sha256"] == source and task["source_sha256"] == source,
            "EXECUTION_SOURCE",
        )
    require(
        execution["status"] == ("completed" if completed else "running"),
        "EXECUTION_STATE",
    )
    if completed:
        require(
            execution["exit_code"] == 0 and not execution["timed_out"],
            "EXECUTION_COMPLETION",
        )
    return execution


def source_inputs(design):
    archives = verify_source_archive(design["source_archive"])
    loaded, pins, executions = {}, {}, {}
    for label, source in design["source_runs"].items():
        directory = Path(source["directory"])
        pins[label], loaded[label] = {}, {}
        for name, expected in source["files"].items():
            pins[label][name] = checked_pin(directory / name, expected)
            if name.endswith(".json"):
                loaded[label][name] = read_json(directory / name)
        executions[label] = verify_execution(
            directory, source["task_id"], design["source_archive"]["source_sha256"]
        )
    train, evaluation = (
        loaded["train"]["training.json"],
        loaded["evaluate"]["result.json"],
    )
    require(
        train["method"] == evaluation["method"] == "onpolicy_kl"
        and train["arm"]["selected_checkpoint"] == 384
        and train["arm"]["updates"] == 384,
        "FIXED_STUDENT_CHECKPOINT",
    )
    require(
        train["base_immutable"]
        and train["resident_learner_rank"] == 8
        and train["trainable_parameters"] == 3833856,
        "SOURCE_STUDENT_ARCHITECTURE",
    )
    require(
        train["base_tensor_sha256"] == design["base_tensor_sha256"],
        "SOURCE_BASE_TENSORS",
    )
    adapter = train["arm"]["checkpoints"]["384"]["adapter"]
    require(adapter == design["adapter"], "SOURCE_ADAPTER_IDENTITY")
    require(
        evaluation["status"] == "completed"
        and evaluation["new_process_persistence_evaluation_completed"]
        and evaluation["initial_and_final_exact_reload_parity"]
        and not evaluation["teacher_model_loaded"]
        and not evaluation["teacher_artifact_files_read_during_evaluation"]
        and evaluation["optimizer_updates"] == 0,
        "SOURCE_TEACHER_ABSENT_EVALUATION",
    )
    require(
        evaluation["training_sha256"] == pins["train"]["training.json"]["sha256"]
        and evaluation["panels_sha256"] == pins["evaluate"]["panels.json"]["sha256"],
        "SOURCE_EVALUATION_LINK",
    )
    require(train["pid"] != evaluation["pid"], "SOURCE_EVALUATION_FRESH_PID")
    for receipt in (train, evaluation):
        for name, sha in receipt["source_sha256"].items():
            require(archives[name] == sha, "ORIGINAL_IMPLEMENTATION_PIN", name)
    for label, receipt in (("train", train), ("evaluate", evaluation)):
        historical = receipt["execution"]
        actual = executions[label]
        for key in ("task_id", "attempt_id", "task_sha256", "config_sha256"):
            require(
                historical[key] == actual[key],
                "SOURCE_RECEIPT_ATTEMPT_BINDING",
                [label, key],
            )
        require(
            historical["source"]["source_sha256"] == actual["source_sha256"]
            and historical["code_dir"] == actual["task"]["code_dir"],
            "SOURCE_RECEIPT_CODE_BINDING",
            label,
        )
    rows, row_hashes = validated_corpus(design)
    require(
        train["dataset"]["rows_sha256"] == row_hashes
        and evaluation["dataset"]["rows_sha256"] == row_hashes,
        "SOURCE_DATASET_IDENTITY",
    )
    reference = {
        "train": loaded["train"]["onpolicy_kl/checkpoint384/train_probes.json"],
        "generic": loaded["train"]["final_retention.json"],
        **{
            key: loaded["evaluate"]["panels.json"][key] for key in ("test", "unused288")
        },
    }
    require(
        reference["train"] == loaded["evaluate"]["final_train_probes.json"]
        and reference["generic"] == loaded["evaluate"]["final_retention.json"],
        "SOURCE_PREMERGE_REFERENCES",
    )
    for split in ("train", "test", "unused288"):
        validate_native(rows[split], reference[split], design["tokens"])
        require(
            all(record["generation"]["correct"] for record in reference[split]),
            "SUCCESSFUL_SOURCE_PREREQUISITE",
            split,
        )
    validate_generic(rows["generic"], reference["generic"])
    return {
        "rows": rows,
        "reference": reference,
        "tokens": design["tokens"],
        "source_pins": pins,
        "source_execution": executions,
        "source_checkpoint": 384,
        "source_adapter_tensor_sha256": adapter["tensor_sha256"],
        "source_training_pid": train["pid"],
        "source_evaluation_pid": evaluation["pid"],
        "source_dataset_sha256": row_hashes,
        "interpretation": "Fixed previously evaluated student; unused288 is the original fixture name, not a new prospective evaluation split.",
    }


def source_base_pins(spec):
    root = Path(spec["local_path"])
    result = {
        item["path"]: checked_pin(root / item["path"], item) for item in spec["files"]
    }
    names = {path.name for path in root.iterdir() if path.is_file()}
    require(
        names <= set(result) | {"README.md", ".gitattributes"},
        "UNPINNED_BASE_FILES",
        sorted(names - set(result)),
    )
    return result


def check_protocol(design):
    require(
        design["contract"] == CONTRACT
        and design["optimizer_updates"] == 0
        and design["tolerances"] == TOLERANCES,
        "PROTOCOL_OR_TOLERANCE",
    )
    require(
        design["source_runs"]["train"]["task_id"]
        == "followthrough-20260912-onpolicy303-onpolicy-kl-train",
        "FIXED_SOURCE_RUN",
    )
    require(
        design["source_base"]["repo_id"] == "Qwen/Qwen3-8B"
        and design["checkpoint"] == 384,
        "FIXED_MODEL_AND_STEP",
    )
    require(
        design["artifact_root"]
        == "/mnt/shared/cl-portfolio/artifacts/followthrough-20260912-native-consolidated303",
        "EXTERNAL_ARTIFACT_ROOT",
    )
    for name, sha in design["code_sha256"].items():
        require(
            name in SOURCE_FILES and file_pin(ROOT / name)["sha256"] == sha,
            "EXECUTABLE_SOURCE_CHANGED",
            name,
        )
    require(
        set(design["code_sha256"]) == set(SOURCE_FILES), "EXECUTABLE_SOURCE_CLOSURE"
    )


def manifest_files(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "ARTIFACT_SYMLINK", str(path))
        if path.is_file() and path != root / "native303_manifest.json":
            files[str(path.relative_to(root))] = file_pin(path)
    return files


def verify_export(root, manifest_pin):
    root = Path(root).resolve()
    manifest_path = root / "native303_manifest.json"
    checked_pin(manifest_path, manifest_pin)
    manifest = read_json(manifest_path)
    require(
        manifest["contract"] == CONTRACT and manifest["optimizer_updates"] == 0,
        "EXPORT_CONTRACT",
    )
    for name in manifest["files"]:
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "MANIFEST_RELATIVE_PATH",
            name,
        )
        require(
            "adapter" not in Path(name).name.lower() and not name.endswith(".py"),
            "NONSTANDARD_EXPORTED_FILE",
            name,
        )
    actual = manifest_files(root)
    require(
        set(actual) == set(manifest["files"]),
        "EXPORT_FILE_SET",
        sorted(set(actual) ^ set(manifest["files"])),
    )
    for name, pin in actual.items():
        require(
            pin == manifest["files"][name],
            "EXPORT_FILE_HASH",
            {"path": name, "expected": manifest["files"][name], "actual": pin},
        )
    index = read_json(root / "model/model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    require(
        bool(shards)
        and all(
            Path(name).name == name and name.endswith(".safetensors") for name in shards
        ),
        "STANDARD_SHARD_INDEX",
    )
    require(
        shards
        == {
            Path(name).name
            for name in actual
            if name.startswith("model/") and name.endswith(".safetensors")
        },
        "SHARD_INDEX_CLOSURE",
    )
    seen, elements, payload_bytes = set(), 0, 0
    for shard in sorted(shards):
        with safe_open(root / "model" / shard, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for key in keys:
                require(
                    key not in seen
                    and index["weight_map"].get(key) == shard
                    and "lora_" not in key
                    and ".base_layer." not in key,
                    "STANDARD_WEIGHT_KEY",
                    key,
                )
                view = handle.get_slice(key)
                require(view.get_dtype() == "F32", "EXPORTED_WEIGHT_NOT_FP32", key)
                number = math.prod(view.get_shape())
                elements += number
                payload_bytes += number * 4
                seen.add(key)
    require(seen == set(index["weight_map"]), "ALL_INDEXED_WEIGHTS_PRESENT")
    require(index["metadata"]["total_size"] == payload_bytes, "SHARDED_PAYLOAD_SIZE")
    require(
        read_json(root / "model/config.json")["model_type"] == "qwen3",
        "ORDINARY_QWEN3_CONFIG",
    )
    require(
        (root / "model/generation_config.json").is_file(), "STANDARD_GENERATION_CONFIG"
    )
    return manifest, {
        "manifest": file_pin(manifest_path),
        "files": actual,
        "shards": {name: actual["model/" + name] for name in sorted(shards)},
        "weight_elements": elements,
        "tensor_payload_bytes": payload_bytes,
    }


class ReadBoundary:
    def __init__(self, forbidden_roots, shared_allowed=()):
        self.forbidden = [str(Path(path).resolve()) for path in forbidden_roots]
        self.shared_allowed = [str(Path(path).resolve()) for path in shared_allowed]
        self.active = False
        self.denied = []
        self.reads = Counter()

    def check(self, path):
        if not self.active or not isinstance(path, (str, bytes, os.PathLike)):
            return
        name = os.path.realpath(os.fsdecode(path))
        forbidden = any(
            name == root or name.startswith(root + os.sep) for root in self.forbidden
        )
        base = Path(name).name.lower()
        forbidden = forbidden or base.startswith(("adapter_", "teacher_"))
        if name.startswith("/mnt/shared/"):
            forbidden = forbidden or not any(
                name == root or name.startswith(root + os.sep)
                for root in self.shared_allowed
            )
        if forbidden:
            self.denied.append(name)
            raise PermissionError(f"NATIVE303_FORBIDDEN_ARTIFACT_READ: {name}")
        self.reads[name] += 1

    def audit(self, event, args):
        if event == "open":
            path, mode, flags = args
            writing = (
                bool(flags & os.O_WRONLY)
                if isinstance(flags, int)
                else bool(
                    mode and "+" not in mode and any(char in mode for char in "wax")
                )
            )
            if not writing:
                self.check(path)

    def __enter__(self):
        sys.addaudithook(self.audit)
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False

    def proof(self):
        return {
            "forbidden_roots": self.forbidden,
            "shared_read_allowlist": self.shared_allowed,
            "denied_reads": self.denied,
            "python_audit_read_paths": dict(sorted(self.reads.items())),
            "boundary": "Python open auditing plus a closed, hash-verified local shard index and direct standard loader. This is not an operating-system sandbox against arbitrary hostile native code.",
        }
