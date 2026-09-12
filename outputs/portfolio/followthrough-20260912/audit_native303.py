import argparse
import ast
import base64
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

ORIGINAL_SOURCE = "0498be8531bfd0cd9dea2dab4e74946345963ac4a7bf4f40aba867207f6b8bef"
ORIGINAL_PROTOCOL = "732e9539d2f178896478afda386d9297ca363b2a1c70daf6f6cb5593102bebc9"
SOURCE = "0dfad5e45ba37d6077163f56f8158ba1d39502d87f97fdf63f569756cdbadf50"
PROTOCOL = "bb39ddb5dd8585eaee0561f5c6b4742ce8ea6bc45ecb509d7efb12affcb02184"
HANDOFF = "94741e23b27e88a07153b5ea8d8ba3d3beca7aadcb1239ac28e03bdff5a0dab9"
EXPORT = "followthrough-20260912-native303-export-runtime"
EVALUATE = "followthrough-20260912-native303-evaluate-runtime"
TASKS = ("sequence_a", "sequence_b", "sequence_c")
TOLERANCES = {
    "source_scores": {"atol": 1e-5, "rtol": 1e-6},
    "merged_scores": {"atol": 1e-4, "rtol": 1e-5},
    "merged_weights": {"atol": 1e-6, "rtol": 1e-6},
    "prefix_logits": {"atol": 1e-3, "rtol": 1e-5},
}


def require(condition, label, detail=None):
    if not condition:
        raise ValueError(f"NATIVE303_INDEPENDENT_AUDIT_{label}: {detail}")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while part := stream.read(8 * 1024 * 1024):
            sha.update(part)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ino),
        "FILE_CHANGED",
        str(path),
    )
    return {"bytes": before.st_size, "sha256": sha.hexdigest()}


def check_pin(path, expected):
    actual = pin(path)
    require(
        all(
            actual[key] == expected[key]
            for key in ("bytes", "sha256")
            if key in expected
        ),
        "FILE_PIN",
        str(path),
    )
    return actual


def relative(name):
    path = Path(name)
    require(
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == name,
        "RELATIVE_PATH",
        name,
    )
    return path


def archive_files(path, expected_source, count):
    files = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            relative(member.name)
            require(
                member.isfile() and member.name not in files,
                "ARCHIVE_MEMBER",
                member.name,
            )
            files[member.name] = archive.extractfile(member).read()
    combined = hashlib.sha256()
    for name in sorted(files, key=Path):
        combined.update(name.encode() + b"\0" + files[name])
    require(
        combined.hexdigest() == expected_source and len(files) == count,
        "ARCHIVE_CLOSURE",
    )
    return files


def collection(folder, expected_status="completed"):
    manifest = read(folder / "collection.json")
    require(
        manifest["execution_status"] == expected_status,
        "COLLECTION_STATUS",
        str(folder),
    )
    files = manifest["files"]
    actual = {
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and p != folder / "collection.json"
    }
    require(actual == set(files), "COLLECTION_CLOSURE", sorted(actual ^ set(files)))
    require(not any(p.is_symlink() for p in folder.rglob("*")), "COLLECTION_SYMLINK")
    for name, expected in files.items():
        check_pin(folder / relative(name), expected)
    return {
        "directory": str(folder),
        "remote": manifest["remote"],
        "collection": pin(folder / "collection.json"),
        "files": len(files),
        "bytes": sum(p["bytes"] for p in files.values()),
        "files_sha256": digest(files),
    }


def execution(folder, task_id, source, expected_status="completed"):
    receipt, task, config = (
        read(folder / name) for name in ("execution.json", "task.json", "config.json")
    )
    require(
        receipt["status"] == expected_status
        and (
            receipt["exit_code"] == 0
            if expected_status == "completed"
            else receipt["exit_code"] != 0
        )
        and receipt["timed_out"] is False,
        "EXECUTION_STATE",
        task_id,
    )
    require(
        receipt["task_id"] == task["id"] == task_id and receipt["task"] == task,
        "TASK_BINDING",
        task_id,
    )
    require(
        receipt["source_sha256"] == task["source_sha256"] == source,
        "TASK_SOURCE",
        task_id,
    )
    require(
        receipt["task_sha256"] == pin(folder / "task.json")["sha256"]
        and receipt["config_sha256"] == pin(folder / "config.json")["sha256"]
        and task["config"] == config,
        "TASK_CONFIG",
        task_id,
    )
    require(
        datetime.fromisoformat(receipt["started_at"])
        < datetime.fromisoformat(receipt["finished_at"]),
        "EXECUTION_CLOCK",
        task_id,
    )
    return receipt


def corpus(files, design):
    fixtures = {}
    for label, spec in design["fixtures"].items():
        payload = files[spec["path"]]
        require(
            len(payload) == spec["bytes"]
            and hashlib.sha256(payload).hexdigest() == spec["sha256"],
            "FIXTURE_PIN",
            label,
        )
        fixtures[label] = json.loads(payload)
    original = fixtures["original"]
    require(original["provenance"]["fixture_seed"] == 303, "FIXTURE_SEED")
    rows = {
        split: [row for task in TASKS for row in original["splits"][f"{task}_{split}"]]
        for split in ("train", "validation", "test")
    }
    rows["unused288"] = fixtures["unused"]["rows"]
    groups, ids = {}, set()
    for split, per_task in {
        "train": 128,
        "validation": 32,
        "test": 64,
        "unused288": 288,
    }.items():
        require(
            Counter(row["task"] for row in rows[split])
            == dict.fromkeys(TASKS, per_task),
            "CORPUS_COUNT",
            split,
        )
        groups[split] = {row["group"] for row in rows[split]}
        require(len(groups[split]) == per_task, "GROUP_COUNT", split)
        for row in rows[split]:
            digits = tuple(map(int, row["group"].split()))
            require(
                len(digits) == 3 and all(0 <= x < 8 for x in digits), "INPUT_DIGITS"
            )
            require(
                row["id"] == digest({"task": row["task"], "input": digits})
                and row["id"] not in ids,
                "ROW_ID",
                row["id"],
            )
            require(
                row["prompt"]
                == f"Apply the {row['task']} code.\nInput: {row['group']}\nOutput:",
                "ROW_PROMPT",
                row["id"],
            )
            expected = " " + " ".join(
                str(original["provenance"]["rules"][row["task"]][i]) for i in digits
            )
            require(
                row["choices"][row["gold_idx"]] == expected,
                "INDEPENDENT_ORACLE",
                row["id"],
            )
            ids.add(row["id"])
    require(
        sum(map(len, groups.values())) == len(set.union(*groups.values())) == 512,
        "SPLIT_LEAKAGE",
    )
    row_hashes = {name: digest(values) for name, values in rows.items()}
    rows["train"] = [
        row
        for task in TASKS
        for row in [r for r in rows["train"] if r["task"] == task][:8]
    ]
    rows.pop("validation")
    rows["generic"] = fixtures["generic"]["validation"]
    require(
        Counter(row["task"] for row in rows["generic"])
        == dict.fromkeys(design["generic_tasks"], 32),
        "GENERIC_COUNT",
    )
    return rows, row_hashes


def native_flags(row, tokens, tokenizer, contract):
    ended = bool(tokens) and tokens[-1] == contract["eos_token_id"]
    content = tokens[:-1] if ended else tokens
    body = tokenizer.decode(content, skip_special_tokens=False)
    gold = row["choices"][row["gold_idx"]]
    valid = bool(re.fullmatch(r" [0-7] [0-7] [0-7]", body)) and not set(
        content
    ).intersection(contract["special_token_ids"])
    return {
        "body_text": body,
        "raw_text": tokenizer.decode(tokens, skip_special_tokens=False),
        "prompt_token_ids": tokenizer.encode(
            row["prompt"], add_special_tokens=True
        ).ids,
        "correct": len(tokens) <= 16 and ended and valid and body == gold,
        "format_valid": valid,
        "terminated": ended,
        "cap_without_eos": len(tokens) >= 16 and not ended,
        "extra_body_characters": len(body) > 6,
        "correct_prefix_with_extra_text": body.startswith(gold) and body != gold,
        "exact_body_without_valid_stop": body == gold and not ended,
        "digit_position_correct": [
            len(body) > i and body[i] == gold[i] for i in (1, 3, 5)
        ],
    }


def native_panel(rows, records, tokenizer, contract):
    require(len(rows) == len(records), "NATIVE_COUNT")
    flags = []
    for row, record in zip(rows, records, strict=True):
        require(
            {k: record[k] for k in ("id", "task", "group", "row_sha256")}
            == {
                "id": row["id"],
                "task": row["task"],
                "group": row["group"],
                "row_sha256": digest(row),
            },
            "NATIVE_ROW_BINDING",
            row["id"],
        )
        tokens = record["generation"]["token_ids"]
        require(
            all(
                type(token) is int and 0 <= token < tokenizer.get_vocab_size()
                for token in tokens
            ),
            "TOKEN_IDS",
            row["id"],
        )
        result = native_flags(row, tokens, tokenizer, contract)
        require(
            all(record["generation"][key] == value for key, value in result.items()),
            "NATIVE_DECODE_OR_GRADE",
            row["id"],
        )
        flags.append((row["task"], result))

    def counts(selected):
        return {
            "count": len(selected),
            **{
                key: sum(value[key] for value in selected)
                for key in (
                    "correct",
                    "format_valid",
                    "terminated",
                    "cap_without_eos",
                    "correct_prefix_with_extra_text",
                    "exact_body_without_valid_stop",
                )
            },
            "digit_position_correct": [
                sum(value["digit_position_correct"][i] for value in selected)
                for i in range(3)
            ],
        }

    return {
        **counts([f for _, f in flags]),
        "by_task": {task: counts([f for t, f in flags if t == task]) for task in TASKS},
        "records_sha256": digest(records),
    }


def generic_panel(rows, records, tokenizer):
    require(len(rows) == len(records), "GENERIC_COUNT")
    tasks, ties, encodings = {}, 0, []
    for row, record in zip(rows, records, strict=True):
        scores = record["scores"]
        require(
            len(scores) == len(row["choices"])
            and all(math.isfinite(s) for s in scores),
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
        require(record == expected, "GENERIC_DECISION", digest(row))
        tally = tasks.setdefault(row["task"], {"count": 0, "correct": 0})
        tally["count"] += 1
        tally["correct"] += expected["correct"]
        ties += scores.count(max(scores)) > 1
        text = row["prompt"].rstrip()
        prompt = tokenizer.encode(text, add_special_tokens=True).ids
        require(0 < len(prompt) <= 768, "GENERIC_PROMPT_BOUNDARY")
        boundary = row["prompt"][len(text) :]
        answers = [
            tokenizer.encode(boundary + choice, add_special_tokens=False).ids
            for choice in row["choices"]
        ]
        require(all(answers), "GENERIC_ANSWER_TOKENS")
        encodings.append(
            {
                "row_sha256": digest(row),
                "prompt_ids": prompt,
                "answer_ids": answers,
                "character_denominators": [
                    len(boundary + choice) for choice in row["choices"]
                ],
            }
        )
    return {
        "count": len(records),
        "correct": sum(t["correct"] for t in tasks.values()),
        "by_task": tasks,
        "argmax_ties": ties,
        "encoded_scoring_inputs_sha256": digest(encodings),
        "boundary": "Decisions independently recomputed from all saved choice scores; tokenizer boundaries verified. Per-choice model likelihoods were not independently rerun.",
    }


def numeric(actual, expected, tolerance):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(
        a.shape == b.shape
        and a.size > 0
        and np.isfinite(a).all()
        and np.isfinite(b).all(),
        "NUMERIC_INPUTS",
    )
    delta = a - b
    absolute = np.abs(delta)
    index = int(np.argmax(absolute))
    outside = int(
        np.count_nonzero(absolute > tolerance["atol"] + tolerance["rtol"] * np.abs(b))
    )
    return {
        "passed": outside == 0,
        "elements": int(a.size),
        "changed_elements": int(np.count_nonzero(delta)),
        "outside_tolerance": outside,
        "max_absolute_difference": float(absolute.flat[index]),
        "rms_difference": float(np.sqrt(np.mean(delta * delta))),
        "maximum_flat_index": index,
        "actual_at_maximum": float(a.flat[index]),
        "expected_at_maximum": float(b.flat[index]),
        "tolerance": tolerance,
    }


def check_numeric_claim(observed, claimed):
    for key in (
        "passed",
        "elements",
        "changed_elements",
        "outside_tolerance",
        "maximum_flat_index",
        "actual_at_maximum",
        "expected_at_maximum",
        "tolerance",
    ):
        require(observed[key] == claimed[key], "NUMERIC_RECOUNT", key)
    for key in ("max_absolute_difference", "rms_difference"):
        require(
            math.isclose(observed[key], claimed[key], rel_tol=1e-10, abs_tol=1e-13),
            "NUMERIC_REDUCTION",
            key,
        )


def compare_probes(actual, expected, tolerance, claimed):
    old, new = expected["train"], actual["train"]
    require(len(old) == len(new), "PROBE_LENGTH")
    changed = [r["id"] for r, s in zip(old, new, strict=True) if r != s]
    native = claimed["train"]
    require(
        native["count"] == len(new)
        and native["identical_records"] == len(new) - len(changed)
        and native["passed"] == (not changed),
        "NATIVE_PARITY_RECOUNT",
    )
    require(
        native["correct"] == sum(r["generation"]["correct"] for r in new)
        and native["native_eos"] == sum(r["generation"]["terminated"] for r in new),
        "NATIVE_PARITY_COUNTS",
    )
    require(
        [d["id"] for d in native["differences"]] == changed, "NATIVE_PARITY_DIFFERENCES"
    )
    report, changes, maximum = [], 0, 0.0
    require(
        len(actual["generic"])
        == len(expected["generic"])
        == len(claimed["generic"]["rows"]),
        "GENERIC_PARITY_LENGTH",
    )
    for a, b, saved in zip(
        actual["generic"], expected["generic"], claimed["generic"]["rows"], strict=True
    ):
        require(
            a["row_sha256"] == b["row_sha256"] == saved["row_sha256"],
            "GENERIC_PARITY_BINDING",
        )
        observed = numeric(a["scores"], b["scores"], tolerance)
        check_numeric_claim(observed, saved["numeric"])
        equal = a["prediction"] == b["prediction"] and a["correct"] == b["correct"]
        require(
            saved["prediction_equal"] == equal
            and saved["actual_scores"] == a["scores"]
            and saved["expected_scores"] == b["scores"],
            "GENERIC_PARITY_RECORD",
        )
        require(
            saved["score_differences"]
            == [x - y for x, y in zip(a["scores"], b["scores"], strict=True)],
            "GENERIC_SCORE_DIFFERENCES",
        )
        changes += not equal
        maximum = max(maximum, observed["max_absolute_difference"])
        report.append(observed)
    passed = not changed and changes == 0 and all(item["passed"] for item in report)
    require(
        claimed["generic"]["count"] == len(report)
        and claimed["generic"]["prediction_changes"] == changes
        and claimed["generic"]["passed"]
        == (changes == 0 and all(item["passed"] for item in report))
        and claimed["passed"] == passed,
        "PROBE_PARITY_GATE",
    )
    return {
        "passed": passed,
        "native_record_changes": changed,
        "generic_decision_changes": changes,
        "generic_score_elements": sum(r["elements"] for r in report),
        "generic_scores_changed": sum(r["changed_elements"] for r in report),
        "generic_scores_outside_tolerance": sum(r["outside_tolerance"] for r in report),
        "maximum_generic_score_difference": maximum,
        "tolerance": tolerance,
    }


def source_adapter(folder, design):
    location = folder / "onpolicy_kl/checkpoint384/learner"
    for name, sha in design["adapter"]["files"].items():
        check_pin(location / name, {"sha256": sha})
    config = read(location / "adapter_config.json")
    require(
        config["r"] == 8
        and config["lora_alpha"] == 16
        and set(config["target_modules"]) == {"q_proj", "v_proj"}
        and config["bias"] == "none"
        and config["lora_dropout"] == 0,
        "FIXED_ADAPTER_ARCHITECTURE",
    )
    tensors = load_file(location / "adapter_model.safetensors")
    combined, total = hashlib.sha256(), 0
    for name, tensor in sorted(tensors.items()):
        require(
            tensor.dtype == np.float32 and np.isfinite(tensor).all(),
            "ADAPTER_FINITE_FP32",
            name,
        )
        require(
            any(
                f".{projection}.lora_{factor}.weight" in name
                for projection in ("q_proj", "v_proj")
                for factor in ("A", "B")
            ),
            "ADAPTER_KEY",
            name,
        )
        combined.update(name.encode())
        combined.update(f"({tuple(tensor.shape)}, torch.float32)".encode())
        combined.update(tensor.tobytes())
        total += tensor.size
    require(
        len(tensors) == 144
        and total == 3833856
        and combined.hexdigest() == design["adapter"]["tensor_sha256"],
        "SOURCE_ADAPTER_TENSORS",
    )
    return {
        "checkpoint": 384,
        "rank": 8,
        "alpha": 16,
        "tensor_count": len(tensors),
        "parameters": total,
        "tensor_sha256": combined.hexdigest(),
        "files": {name: pin(location / name) for name in design["adapter"]["files"]},
    }


def inputs(wave, require_collected_archive=True):
    staging = read(wave / "native303-runtime-staging.json")
    require(
        staging["source_sha256"] == SOURCE
        and staging["source_files"] == 134
        and staging["native_handoff_sha256"] == HANDOFF,
        "STAGING_BINDING",
    )
    archive = wave / "code" / f"{SOURCE}.tar"
    if require_collected_archive:
        metadata = read(archive.with_suffix(".json"))
        check_pin(archive, metadata["archive"])
        files = archive_files(archive, SOURCE, 134)
    else:
        snapshot = Path(staging["snapshot_directory"])
        files = {
            name: (snapshot / relative(name)).read_bytes() for name in staging["files"]
        }
        combined = hashlib.sha256()
        for name in sorted(files, key=Path):
            combined.update(name.encode() + b"\0" + files[name])
        require(combined.hexdigest() == SOURCE, "PRECOLLECTION_SNAPSHOT")
    require(set(files) == set(staging["files"]), "STAGING_FILE_CLOSURE")
    for name, payload in files.items():
        require(
            {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            == staging["files"][name],
            "STAGED_FILE",
            name,
        )
    require(
        hashlib.sha256(files["native303_handoff.json"]).hexdigest() == HANDOFF,
        "FROZEN_HANDOFF",
    )
    handoff = json.loads(files["native303_handoff.json"])
    design = json.loads(files["configs/native303_protocol.json"])
    require(
        digest(design) == PROTOCOL
        and design["tolerances"] == TOLERANCES
        and design["optimizer_updates"] == 0,
        "SEALED_PROTOCOL",
    )
    require(
        all(
            hashlib.sha256(files[name]).hexdigest() == sha
            for name, sha in design["code_sha256"].items()
        ),
        "RUNTIME_SOURCE_PINS",
    )
    rows, row_hashes = corpus(files, design)
    tokenizer_path = wave / "tokenizer/tokenizer.json"
    token_spec = next(
        x for x in design["source_base"]["files"] if x["path"] == "tokenizer.json"
    )
    check_pin(tokenizer_path, token_spec)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    require(
        tokenizer.token_to_id("<|im_end|>")
        == design["tokens"]["eos_token_id"]
        == 151645,
        "EOS_IDENTITY",
    )
    references, sources, source_files = {}, {}, {}
    original_archive = wave / "code" / Path(design["source_archive"]["path"]).name
    check_pin(original_archive, design["source_archive"]["archive"])
    old_files = archive_files(
        original_archive,
        design["source_archive"]["source_sha256"],
        design["source_archive"]["files"],
    )
    for label, spec in design["source_runs"].items():
        folder = wave / "runs" / spec["task_id"]
        for name, expected in spec["files"].items():
            check_pin(folder / relative(name), expected)
        sources[label] = execution(
            folder, spec["task_id"], design["source_archive"]["source_sha256"]
        )
        source_files[label] = {
            name: read(folder / name)
            for name in spec["files"]
            if name.endswith(".json")
        }
    train, evaluation = (
        source_files["train"]["training.json"],
        source_files["evaluate"]["result.json"],
    )
    require(
        train["method"] == evaluation["method"] == "onpolicy_kl"
        and train["arm"]["updates"] == train["arm"]["selected_checkpoint"] == 384,
        "FIXED_SOURCE_METHOD",
    )
    require(
        train["base_immutable"]
        and train["base_tensor_sha256"] == design["base_tensor_sha256"]
        and train["arm"]["checkpoints"]["384"]["adapter"] == design["adapter"],
        "SOURCE_MODEL_IDENTITY",
    )
    require(
        train["pid"] != evaluation["pid"]
        and evaluation["new_process_persistence_evaluation_completed"]
        and evaluation["initial_and_final_exact_reload_parity"]
        and evaluation["optimizer_updates"] == 0
        and not evaluation["teacher_model_loaded"]
        and not evaluation["teacher_artifact_files_read_during_evaluation"],
        "SOURCE_PERSISTENCE",
    )
    require(
        evaluation["training_sha256"]
        == design["source_runs"]["train"]["files"]["training.json"]["sha256"]
        and evaluation["panels_sha256"]
        == design["source_runs"]["evaluate"]["files"]["panels.json"]["sha256"],
        "SOURCE_RECORD_LINK",
    )
    for label, receipt in (("train", train), ("evaluate", evaluation)):
        require(
            receipt["dataset"]["rows_sha256"] == row_hashes,
            "SOURCE_CORPUS_BINDING",
            label,
        )
        for name, sha in receipt["source_sha256"].items():
            require(
                hashlib.sha256(old_files[name]).hexdigest() == sha,
                "OLD_SOURCE_CLOSURE",
                name,
            )
        for key in ("task_id", "attempt_id", "task_sha256", "config_sha256"):
            require(
                receipt["execution"][key] == sources[label][key],
                "OLD_EXECUTION_LINK",
                key,
            )
    references["train"] = source_files["train"][
        "onpolicy_kl/checkpoint384/train_probes.json"
    ]
    references["generic"] = source_files["train"]["final_retention.json"]
    for split in ("test", "unused288"):
        references[split] = source_files["evaluate"]["panels.json"][split]
    require(
        references["train"] == source_files["evaluate"]["final_train_probes.json"]
        and references["generic"] == source_files["evaluate"]["final_retention.json"],
        "ORIGINAL_RELOAD_RECORDS",
    )
    source_grades = {
        name: native_panel(rows[name], references[name], tokenizer, design["tokens"])
        for name in ("train", "test", "unused288")
    }
    require(
        all(g["count"] == g["correct"] for g in source_grades.values()),
        "SOURCE_NATIVE_ACQUISITION",
    )
    source_grades["generic"] = generic_panel(
        rows["generic"], references["generic"], tokenizer
    )
    adapter = source_adapter(
        wave / "runs" / design["source_runs"]["train"]["task_id"], design
    )
    return {
        "design": design,
        "handoff": handoff,
        "files": files,
        "rows": rows,
        "row_hashes": row_hashes,
        "tokenizer": tokenizer,
        "references": references,
        "sources": sources,
        "train": train,
        "evaluation": evaluation,
        "source_grades": source_grades,
        "adapter": adapter,
        "staging": staging,
    }


REMOTE = r"""
import array
import base64
import hashlib
import importlib.util
import json
import marshal
import math
import os
import struct
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path


def need(condition, label):
    if not condition:
        raise ValueError('NATIVE303_REMOTE_AUDIT_' + label)


def load(path):
    return json.loads(Path(path).read_text())


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable(before, after):
    need((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'FILE_CHANGED_DURING_READ')


def file_pin(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        while part := f.read(8 * 1024 * 1024):
            h.update(part)
    stable(before, path.stat())
    return {'bytes': before.st_size, 'sha256': h.hexdigest()}


def header(path):
    with path.open('rb') as f:
        prefix = f.read(8)
        need(len(prefix) == 8, 'SHORT_HEADER')
        size = int.from_bytes(prefix, 'little')
        need(0 < size <= 32 * 1024 * 1024, 'HEADER_SIZE')
        payload = f.read(size)
    need(len(payload) == size, 'SHORT_HEADER_JSON')
    fields = json.loads(payload)
    fields.pop('__metadata__', None)
    position = 0
    for key, meta in sorted(fields.items(), key=lambda pair: pair[1]['data_offsets'][0]):
        shape, (start, stop) = meta['shape'], meta['data_offsets']
        need(meta['dtype'] == 'F32' and all(type(x) is int and x >= 0 for x in shape), 'NON_FP32_OR_SHAPE')
        need(start == position and stop >= start and stop-start == math.prod(shape)*4, 'TENSOR_PAYLOAD_LAYOUT')
        position = stop
    need(8+size+position == path.stat().st_size, 'TENSOR_FILE_CLOSURE')
    return fields, prefix + payload


def shard_inventory(path):
    before = path.stat()
    fields, prefix = header(path)
    total, tensors = hashlib.sha256(prefix), {}
    with path.open('rb') as f:
        f.seek(len(prefix))
        for key, meta in sorted(fields.items(), key=lambda pair: pair[1]['data_offsets'][0]):
            need('lora_' not in key and '.base_layer.' not in key and 'adapter' not in key.lower(), 'ADAPTER_WEIGHT_KEY')
            remaining = meta['data_offsets'][1] - meta['data_offsets'][0]
            h = hashlib.sha256()
            while remaining:
                part = f.read(min(8 * 1024 * 1024, remaining))
                need(bool(part), 'TRUNCATED_TENSOR')
                remaining -= len(part)
                total.update(part)
                h.update(part)
            tensors[key] = {'shape': meta['shape'], 'dtype': 'torch.float32', 'sha256': h.hexdigest(), 'elements': math.prod(meta['shape'])}
    stable(before, path.stat())
    return {'bytes': before.st_size, 'sha256': total.hexdigest()}, tensors


def values(path, fields, key):
    meta = fields[key]
    start, stop = meta['data_offsets']
    with path.open('rb') as f:
        length = int.from_bytes(f.read(8), 'little')
        f.seek(8 + length + start)
        result = array.array('f')
        result.frombytes(f.read(stop-start))
    if sys.byteorder != 'little':
        result.byteswap()
    need(len(result) == math.prod(meta['shape']), 'PREFIX_VALUES_COUNT')
    return result


def differences(a, b, tolerance):
    need(len(a) == len(b) and len(a) > 0, 'NUMERIC_SHAPE')
    outside, changed, square = 0, 0, 0.0
    maximum, at_maximum = -1.0, 0
    a_arg, b_arg = 0, 0
    a_max, b_max = -math.inf, -math.inf
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        need(math.isfinite(x) and math.isfinite(y), 'NONFINITE_PREFIX')
        difference = x-y
        absolute = abs(difference)
        changed += difference != 0
        outside += absolute > tolerance['atol'] + tolerance['rtol']*abs(y)
        square += difference*difference
        if absolute > maximum:
            maximum, at_maximum = absolute, i
        if x > a_max:
            a_max, a_arg = x, i
        if y > b_max:
            b_max, b_arg = y, i
    return {'passed': outside == 0, 'elements': len(a), 'changed_elements': changed, 'outside_tolerance': outside,
            'max_absolute_difference': maximum, 'rms_difference': math.sqrt(square/len(a)),
            'maximum_flat_index': at_maximum, 'actual_at_maximum': a[at_maximum], 'expected_at_maximum': b[at_maximum],
            'tolerance': tolerance, 'actual_argmax': a_arg, 'expected_argmax': b_arg}


def compare_prefix_files(paths, references, tolerance, vocabulary):
    headers = {name: header(path)[0] for name, path in paths.items()}
    keys = {f'row_{i:04d}' for i in range(len(references))}
    need(all(set(h) == keys for h in headers.values()), 'PREFIX_KEY_CLOSURE')
    pairs = {'merged_source': ('merged', 'source'), 'fresh_source': ('fresh', 'source'), 'fresh_merged': ('fresh', 'merged')}
    report = {name: [] for name in pairs}
    for i, ref in enumerate(references):
        key = f'row_{i:04d}'
        expected = [len(ref['generation']['token_ids']), vocabulary]
        need(all(h[key]['shape'] == expected for h in headers.values()), 'FULL_VOCAB_PREFIX_SHAPE')
        arrays = {name: values(path, headers[name], key) for name, path in paths.items()}
        for step, token in enumerate(ref['generation']['token_ids']):
            prefix = ref['generation']['prompt_token_ids'] + ref['generation']['token_ids'][:step]
            vectors = {name: data[step*vocabulary:(step+1)*vocabulary] for name, data in arrays.items()}
            for label, (actual, expected) in pairs.items():
                observed = differences(vectors[actual], vectors[expected], tolerance)
                report[label].append({'id': ref['id'], 'step': step, 'prefix_sha256': sha_object(prefix), 'recorded_next_token': token, **observed})
        print('NATIVE303_REMOTE_PREFIX_ROW ' + str(i+1), file=sys.stderr, flush=True)
    return {label: {'passed': all(row['passed'] for row in rows), 'prefixes': len(rows), 'rows': rows} for label, rows in report.items()}


def source_cache_proof(request):
    need(sys.version.split()[0] == request['compiler_version'], 'MATCHED_CACHE_COMPILER')
    root = Path(request['source_directory'])
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    need(set(request['source_files']) <= actual, 'MISSING_STAGED_SOURCE')
    caches = {}
    for name in sorted(actual - set(request['source_files'])):
        path = root / name
        need(path.parent == root / '__pycache__' and path.name.endswith('.cpython-312.pyc'), 'UNMANIFESTED_SOURCE_FILE:' + name)
        source_name = path.name.removesuffix('.cpython-312.pyc') + '.py'
        need(source_name in request['source_files'], 'UNMANIFESTED_CACHED_MODULE')
        need(file_pin(root / source_name) == request['source_files'][source_name], 'CACHED_SOURCE_FILE_PIN')
        raw = path.read_bytes()
        need(len(raw) > 16 and raw[:4] == importlib.util.MAGIC_NUMBER, 'BYTECODE_INTERPRETER')
        cached = marshal.loads(raw[16:])
        need(isinstance(cached, types.CodeType) and cached.co_filename == str(root / source_name), 'BYTECODE_SOURCE_PATH')
        compiled = compile((root / source_name).read_bytes(), str(root / source_name), 'exec', dont_inherit=True, optimize=0)
        need(cached == compiled, 'BYTECODE_DIFFERS_FROM_FROZEN_SOURCE')
        caches[name] = {'file': file_pin(path), 'source': source_name, 'source_sha256': request['source_files'][source_name]['sha256'],
                        'code_object_equal_to_frozen_source_compilation': True, 'executed_by_auditor': False}
    return {'compiler_version': sys.version, 'compiler_executable': sys.executable, 'caches': caches,
            'boundary': 'Code objects compared using the exact runtime Python patch version; frozen experiment source was compiled but never executed by the auditor.'}


def inspect_export(request):
    root = Path(request['root'])
    need(root.is_dir() and not root.is_symlink(), 'PROMOTED_EXPORT_REQUIRED')
    manifest_path = root / 'native303_manifest.json'
    need(file_pin(manifest_path) == request['manifest_pin'], 'MANIFEST_PIN')
    manifest = load(manifest_path)
    actual = {}
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'ARTIFACT_SYMLINK')
        if path.is_file() and path != manifest_path:
            actual[str(path.relative_to(root))] = path
    need(set(actual) == set(manifest['files']), 'MANIFEST_CLOSURE')
    pins, tensors = {}, {}
    for name, path in sorted(actual.items()):
        need('adapter' not in path.name.lower() and not name.endswith('.py'), 'NONSTANDARD_FILE')
        if name.startswith('model/') and name.endswith('.safetensors'):
            pins[name], observed = shard_inventory(path)
            need(not set(tensors).intersection(observed), 'DUPLICATE_SHARD_KEYS')
            tensors.update({key: {**value, 'shard': path.name} for key, value in observed.items()})
        else:
            pins[name] = file_pin(path)
        need(pins[name] == manifest['files'][name], 'EXTERNAL_FILE_PIN:' + name)
        print('NATIVE303_REMOTE_FILE ' + name, file=sys.stderr, flush=True)
    index = load(root / 'model/model.safetensors.index.json')
    need(set(index['weight_map']) == set(tensors), 'WEIGHT_INDEX_CLOSURE')
    need(all(index['weight_map'][key] == item['shard'] for key, item in tensors.items()), 'WRONG_SHARD_ASSIGNMENT')
    shards = {Path(name).name for name in pins if name.startswith('model/') and name.endswith('.safetensors')}
    need(shards == set(index['weight_map'].values()), 'SHARD_INDEX_CLOSURE')
    payload_bytes = 4 * sum(t['elements'] for t in tensors.values())
    need(index['metadata']['total_size'] == payload_bytes, 'SHARD_PAYLOAD_BYTES')
    weights = manifest['ordinary_weight_digest']['tensors']
    parameter_names = {key for key in weights if key.startswith('parameter:')}
    need(parameter_names == {'parameter:' + key for key in tensors}, 'ALL_SERIALIZED_PARAMETERS')
    for key, observed in tensors.items():
        need({name: observed[name] for name in ('shape', 'dtype', 'sha256', 'elements')} == weights['parameter:' + key], 'SERIALIZED_TENSOR_HASH:' + key)
    nonserialized = {key: val for key, val in weights.items() if key not in parameter_names}
    need(all(key.startswith('buffer:') for key in nonserialized), 'UNKNOWN_NONSERIALIZED_TENSOR')
    source_root = Path(request['source_directory'])
    for name, expected in request['source_files'].items():
        path = source_root / name
        need(file_pin(path) == expected, 'STAGED_SOURCE_CHANGED:' + name)
    actual_source = {str(path.relative_to(source_root)) for path in source_root.rglob('*') if path.is_file()}
    cache_proof = request['source_cache_proof']
    caches = cache_proof['caches']
    need(actual_source - set(request['source_files']) == set(caches), 'SOURCE_AND_GENERATED_CACHE_CLOSURE')
    for name, expected in caches.items():
        need(file_pin(source_root / name) == expected['file'], 'CACHE_CHANGED_AFTER_MATCHED_COMPILATION')
        need(expected['source_sha256'] == request['source_files'][expected['source']]['sha256'], 'CACHE_PROOF_SOURCE_BINDING')
    need(set(request['source_files']) <= actual_source, 'MISSING_STAGED_SOURCE')
    need(file_pin(Path(request['source_directory'] + '.tar')) == request['source_archive_pin'], 'REMOTE_SOURCE_ARCHIVE')
    evidence = load(root / 'evidence/inputs.json')
    prefixes = {'source': root / 'evidence/source_prefix_logits.safetensors',
                'merged': root / 'evidence/merged_prefix_logits.safetensors', 'fresh': Path(request['fresh_prefix'])}
    fresh_pin = file_pin(prefixes['fresh'])
    need(fresh_pin == request['fresh_prefix_pin'], 'FRESH_PREFIX_BINDING')
    config = load(root / 'model/config.json')
    prefix_comparisons = compare_prefix_files(prefixes, evidence['reference']['train'], request['prefix_tolerance'], config['vocab_size'])
    return {'at': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'manifest_pin': file_pin(manifest_path),
            'manifest': manifest, 'files': pins, 'tensor_inventory': tensors, 'tensor_payload_bytes': payload_bytes,
            'nonserialized_buffers': nonserialized, 'source_files_verified': len(request['source_files']),
            'source_files_sha256': sha_object(request['source_files']), 'source_archive': request['source_archive_pin'], 'generated_bytecode_caches': cache_proof,
            'inputs': evidence, 'merged_probes': load(root / 'evidence/merged_probes.json'),
            'weight_audit': load(root / 'evidence/weight_audit.json'), 'merge_parity': load(root / 'evidence/merge_parity.json'),
            'config': config, 'generation_config': load(root / 'model/generation_config.json'),
            'tokenizer_config': load(root / 'model/tokenizer_config.json'),
            'tokenizer_base64': base64.b64encode((root / 'model/tokenizer.json').read_bytes()).decode(),
            'fresh_prefix_pin': fresh_pin, 'prefix_comparisons': prefix_comparisons,
            'boundary': 'Read-only independent stdlib file/tensor hashes and full-vocabulary prefix arithmetic. No model load, optimization, provider mutation or local model shard copy. Nonpersistent buffers and module construction remain tied to the sealed runtime evidence.'}


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'test.safetensors'
        data = struct.pack('<ff', 1.0, 2.0)
        meta = {'a.weight': {'shape': [2], 'dtype': 'F32', 'data_offsets': [0, len(data)]}}
        encoded = json.dumps(meta).encode()
        path.write_bytes(len(encoded).to_bytes(8, 'little') + encoded + data)
        found, tensors = shard_inventory(path)
        need(found == file_pin(path) and tensors['a.weight']['sha256'] == hashlib.sha256(data).hexdigest(), 'SELF_TEST_HASH')
        path.write_bytes(path.read_bytes() + b'0')
        try:
            header(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Trailing unindexed bytes accepted')
    near = differences([1.0005, 2.0], [1.0, 2.0], {'atol': .001, 'rtol': 0})
    need(near['passed'] and near['changed_elements'] == 1, 'SELF_TEST_TOLERANCE')
    need(not differences([1.002], [1.0], {'atol': .001, 'rtol': 0})['passed'], 'SELF_TEST_REJECTION')
    try:
        differences([math.nan], [1.0], {'atol': 1, 'rtol': 0})
    except ValueError:
        pass
    else:
        raise AssertionError('NaN accepted')
    print(json.dumps({'remote_cpu_checks': 5, 'passed': True}))
"""


def remote_artifact(ctx, wave, export_receipt):
    evaluation = wave / "runs" / EVALUATE
    request = {
        "root": export_receipt["artifact"]["directory"],
        "manifest_pin": export_receipt["artifact"]["manifest"],
        "source_files": ctx["staging"]["files"],
        "source_directory": f"/mnt/shared/cl-portfolio/code/{SOURCE}",
        "source_archive_pin": pin(wave / "code" / f"{SOURCE}.tar"),
        "fresh_prefix": f"/mnt/shared/cl-portfolio/runs/{EVALUATE}/reloaded_prefix_logits.safetensors",
        "fresh_prefix_pin": pin(evaluation / "reloaded_prefix_logits.safetensors"),
        "prefix_tolerance": TOLERANCES["prefix_logits"],
    }
    runtime = read(evaluation / "runtime.json")
    cache_request = {
        "source_files": request["source_files"],
        "source_directory": request["source_directory"],
        "compiler_version": runtime["python"].split()[0],
    }
    cache_code = (
        REMOTE
        + "\nprint(json.dumps(source_cache_proof("
        + repr(cache_request)
        + "), allow_nan=False))\n"
    )
    cache_command = [
        "kubectl",
        "--context",
        "us-mi355x-nambiar-k8s",
        "-n",
        "default",
        "exec",
        "-i",
        runtime["hostname"],
        "--",
        "/tools/uv",
        "run",
        "--no-project",
        "--python",
        runtime["executable"],
        "python",
        "-B",
        "-",
    ]
    cache_process = subprocess.run(
        cache_command, input=cache_code, text=True, stdout=subprocess.PIPE, check=True
    )
    request["source_cache_proof"] = json.loads(cache_process.stdout)
    request["source_cache_proof"]["audit_program_sha256"] = hashlib.sha256(
        cache_code.encode()
    ).hexdigest()
    code = (
        REMOTE
        + "\nprint(json.dumps(inspect_export("
        + repr(request)
        + "), allow_nan=False))\n"
    )
    command = [
        "kubectl",
        "--context",
        "us-mi355x-nambiar-k8s",
        "-n",
        "default",
        "exec",
        "-i",
        "cl-portfolio-control-20260912",
        "--",
        "uv",
        "run",
        "--no-project",
        "python",
        "-B",
        "-",
    ]
    process = subprocess.run(
        command, input=code, text=True, stdout=subprocess.PIPE, check=True
    )
    result = json.loads(process.stdout)
    result["independent_audit_program_sha256"] = hashlib.sha256(
        code.encode()
    ).hexdigest()
    return result


def original_attempts(ctx, wave):
    old_export = "followthrough-20260912-native303-export"
    old_evaluate = "followthrough-20260912-native303-evaluate"
    root = wave / "runs" / old_export
    failed_collection = collection(root, "failed")
    actual = execution(root, old_export, ORIGINAL_SOURCE, "failed")
    failure, runtime = read(root / "failure.json"), read(root / "runtime.json")
    require(
        failure["message"] == "NATIVE303_PINNED_WORKER_TORCH: None"
        and failure["optimizer_updates"] == 0
        and failure["pid"] == runtime["pid"],
        "ORIGINAL_FAILURE",
    )
    require(runtime["torch"] == "2.10.0+rocm7.2.4.git3d3aa833", "ORIGINAL_RUNTIME")
    require(
        "torch==2.10.0+rocm7.2.4.lw.git3d3aa833"
        in (root / "packages.txt").read_text().splitlines(),
        "ORIGINAL_DISTRIBUTION_VERSION",
    )
    absent = [
        "input_manifest.json",
        "premerge.json",
        "merge_parity.json",
        "weight_audit.json",
        "export_receipt.json",
        "result.json",
    ]
    require(
        not any((root / name).exists() for name in absent), "PREMODEL_FAILURE_ARTIFACTS"
    )
    archive = wave / "code" / f"{ORIGINAL_SOURCE}.tar"
    check_pin(archive, read(archive.with_suffix(".json"))["archive"])
    files = archive_files(archive, ORIGINAL_SOURCE, 134)
    old_design = json.loads(files["configs/native303_protocol.json"])
    require(
        digest(old_design) == ORIGINAL_PROTOCOL
        and old_design["worker_torch"] == "2.10.0+rocm7.2.4.lw.git3d3aa833",
        "ORIGINAL_PROTOCOL",
    )
    allowed = {"worker_torch", "export_task_id", "evaluate_task_id", "code_sha256"}
    changes = {
        key
        for key in set(old_design) | set(ctx["design"])
        if old_design.get(key) != ctx["design"].get(key)
    }
    require(changes == allowed, "RUNTIME_RETRY_SCIENTIFIC_DRIFT", sorted(changes))
    tree = ast.parse(files["native303_export.py"])
    new_tree = ast.parse(ctx["files"]["native303_export.py"])
    old_functions = {
        node.name: ast.dump(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    new_functions = {
        node.name: ast.dump(node)
        for node in new_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    require(
        set(new_functions) - set(old_functions) == {"validate_runtime"}
        and set(old_functions) <= set(new_functions),
        "RETRY_FUNCTION_SCOPE",
    )
    unchanged = set(old_functions) - {"main", "runtime"}
    require(
        all(old_functions[name] == new_functions[name] for name in unchanged),
        "RETRY_MODEL_OR_SCORING_CODE_CHANGED",
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    checks = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "require"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "PINNED_WORKER_TORCH"
            for arg in node.args
        )
    ]
    stage_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "export_stage"
    ]
    require(
        len(checks) == len(stage_calls) == 1
        and checks[0].lineno < stage_calls[0].lineno,
        "FAILURE_BEFORE_EXPORT_STAGE",
    )
    blocked_root = wave / "runs" / old_evaluate
    blocked_collection = collection(blocked_root, "blocked")
    blocked = read(blocked_root / "execution.json")
    require(
        blocked["task_id"] == old_evaluate
        and blocked["status"] == "blocked"
        and blocked["dependencies"][old_export] == "failed",
        "BLOCKED_DEPENDENCY",
    )
    require(
        not any(
            key in blocked
            for key in ("task", "source_sha256", "pid", "attempt_id", "started_at")
        ),
        "BLOCKED_NEVER_EXECUTED",
    )
    binding = read(wave / "native303-blocked-dispatch-binding.json")
    require(
        binding["execution_sha256"] == pin(blocked_root / "execution.json")["sha256"]
        and binding["remote_bytes_equal_local_submitted_manifest"] is True,
        "BLOCKED_EXTERNAL_BINDING",
    )
    check_pin(
        Path(binding["local_submitted_manifest_path"]),
        {"sha256": binding["dispatch_sha256"], "bytes": binding["dispatch_bytes"]},
    )
    task = read(binding["local_submitted_manifest_path"])
    require(
        task == binding["task"]
        and task["id"] == old_evaluate
        and task["source_sha256"] == ORIGINAL_SOURCE
        and task["config"]["protocol_sha256"] == ORIGINAL_PROTOCOL,
        "BLOCKED_SUBMITTED_SOURCE",
    )
    dispatch = read(wave / "native303-dispatch.json")
    entry = next(job for job in dispatch["jobs"] if job["id"] == old_evaluate)
    require(
        entry["manifest_sha256"] == binding["dispatch_sha256"]
        and entry["depends_on"] == task["depends_on"],
        "BLOCKED_DISPATCH_CHAIN",
    )
    return {
        "export": {
            "classification": "infrastructure_runtime_identity_check_failed_before_model_load_or_merge",
            "execution": pin(root / "execution.json"),
            "attempt_id": actual["attempt_id"],
            "process_pid": failure["pid"],
            "failure": failure,
            "collection": failed_collection,
            "absent_model_stage_artifacts": absent,
            "source_sha256": ORIGINAL_SOURCE,
            "source_archive": pin(archive),
            "torch_version_namespaces": {
                "module": runtime["torch"],
                "installed_distribution_from_packages_txt": "2.10.0+rocm7.2.4.lw.git3d3aa833",
            },
        },
        "evaluate": {
            "classification": "never_executed_dependency_blocked",
            "execution": pin(blocked_root / "execution.json"),
            "collection": blocked_collection,
            "embedded_execution_source_or_pid": False,
            "separate_dispatch_evidence": pin(
                wave / "native303-blocked-dispatch-binding.json"
            ),
            "submitted_manifest": {
                "bytes": binding["dispatch_bytes"],
                "sha256": binding["dispatch_sha256"],
            },
            "boundary": "Source identity belongs to the separately verified submitted dispatch, not to a nonexistent runtime execution. The parent captured NFS queue bytes equal to the local manifest; this auditor verifies that saved binding.",
        },
        "retry_protocol_changes": sorted(changes),
        "unchanged_export_and_evaluation_functions": sorted(unchanged),
        "scientific_contract_unchanged": True,
    }


def identity(receipt, actual, runtime, archive_pin):
    value = receipt["identity"]
    for key in (
        "task_id",
        "attempt_id",
        "task",
        "task_sha256",
        "config_sha256",
        "source_sha256",
        "started_at",
        "pid",
    ):
        require(value["execution"][key] == actual[key], "ATTEMPT_IDENTITY", key)
    require(
        value["source_archive"] == archive_pin
        and value["pid"] == receipt["pid"] == runtime["pid"]
        and value["parent_pid"] == runtime["parent_pid"],
        "PROCESS_IDENTITY",
    )
    require(
        receipt["optimizer_updates"] == 0 and receipt["protocol_sha256"] == PROTOCOL,
        "ZERO_UPDATE_PROTOCOL",
    )


def exported_weights(ctx, remote, saved):
    manifest, model = remote["manifest"], remote["config"]
    require(
        remote["weight_audit"] == saved
        and manifest["ordinary_weight_digest"] == saved["after"],
        "WEIGHT_MANIFEST_BINDING",
    )
    require(
        saved["before"]["tensor_sha256"] == ctx["design"]["base_tensor_sha256"]
        and saved["after"]["elements"] == saved["before"]["elements"],
        "BASE_WEIGHT_IDENTITY",
    )
    require(
        model["model_type"] == "qwen3"
        and model["architectures"] == ["Qwen3ForCausalLM"]
        and not model.get("auto_map")
        and not model.get("quantization_config"),
        "ORDINARY_MODEL_CONFIG",
    )
    targets = {
        f"parameter:model.layers.{i}.self_attn.{projection}.weight"
        for i in range(model["num_hidden_layers"])
        for projection in ("q_proj", "v_proj")
    }
    before, after = saved["before"]["tensors"], saved["after"]["tensors"]
    require(
        set(before) == set(after)
        and targets <= set(before)
        and set(saved["target_weight_comparisons"]) == targets,
        "QV_MERGE_TARGETS",
    )
    untouched = set(before) - targets
    require(
        saved["target_matrices"] == len(targets) == 72
        and saved["untargeted_tensors"] == len(untouched)
        and saved["changed_untargeted"] == []
        and all(before[key] == after[key] for key in untouched),
        "UNTARGETED_CONSERVATION",
    )
    for name, comparison in saved["target_weight_comparisons"].items():
        require(
            comparison["passed"]
            and comparison["outside_tolerance"] == 0
            and comparison["tolerance"] == TOLERANCES["merged_weights"]
            and comparison["elements"] == after[name]["elements"],
            "MERGE_EQUATION_WITNESS",
            name,
        )
        require(
            math.isfinite(comparison["max_absolute_difference"]), "MERGE_FINITE_WITNESS"
        )
    require(
        saved["passed"]
        and saved["ordinary_module_classes"] == manifest["ordinary_module_classes"],
        "ORDINARY_MODULES_BINDING",
    )
    classes = manifest["ordinary_module_classes"]
    require(
        classes[""] == "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM",
        "MODEL_ROOT_CLASS",
    )
    require(
        all(
            name.startswith(
                ("torch.nn.", "transformers.models.qwen3.", "transformers.activations.")
            )
            for name in classes.values()
        )
        and not any(
            "lora" in name.lower() or "adapter" in name.lower() for name in classes
        ),
        "ORDINARY_MODULE_INVENTORY",
    )
    generation = remote["generation_config"]
    require(
        generation.get("do_sample", False) is False
        and generation["max_new_tokens"] == 16
        and generation["eos_token_id"] == 151645
        and generation["pad_token_id"] == 151645
        and generation.get("num_beams", 1) == 1
        and generation.get("repetition_penalty", 1.0) == 1.0,
        "SAVED_GREEDY_EOS_CONTRACT",
    )
    return {
        "ordinary_modules": len(classes),
        "targeted_qv_matrices": len(targets),
        "changed_targeted_matrices": sum(
            before[k]["sha256"] != after[k]["sha256"] for k in targets
        ),
        "untargeted_tensors_exact": len(untouched),
        "serialized_parameters_independently_hashed": len(remote["tensor_inventory"]),
        "serialized_parameter_elements": sum(
            t["elements"] for t in remote["tensor_inventory"].values()
        ),
        "serialized_fp32_payload_bytes": remote["tensor_payload_bytes"],
        "nonserialized_buffers": remote["nonserialized_buffers"],
        "ordinary_weight_tensor_sha256": saved["after"]["tensor_sha256"],
        "maximum_reported_weight_equation_difference": max(
            x["max_absolute_difference"]
            for x in saved["target_weight_comparisons"].values()
        ),
        "proof_boundary": "Every serialized parameter is independently hashed and matched to the in-process after-merge tensor inventory. The 72 FP32 W+2*(B@A) arithmetic comparisons, finite full-model values, ordinary runtime module construction and nonpersistent buffers are verified from pinned runtime witnesses; this audit does not recompute those matrix products or instantiate a second CPU model.",
    }


def prefix_comparison(remote, saved, references):
    require(
        remote["prefixes"] == saved["prefixes"] == 168
        and len(remote["rows"]) == len(saved["rows"]) == 168,
        "PREFIX_COUNT",
    )
    ref = {record["id"]: record for record in references}
    for observed, claimed in zip(remote["rows"], saved["rows"], strict=True):
        record = ref[observed["id"]]
        step = observed["step"]
        prefix = (
            record["generation"]["prompt_token_ids"]
            + record["generation"]["token_ids"][:step]
        )
        require(
            observed["id"] == claimed["id"]
            and step == claimed["step"]
            and claimed["prefix_token_ids"] == prefix
            and observed["prefix_sha256"] == digest(prefix),
            "PREFIX_INPUT_BINDING",
        )
        check_numeric_claim(observed, claimed)
        require(
            observed["actual_argmax"] == claimed["actual_argmax"]
            and observed["expected_argmax"] == claimed["expected_argmax"],
            "FULL_VOCAB_ARGMAX",
        )
        require(
            observed["recorded_next_token"] == record["generation"]["token_ids"][step],
            "PREFIX_NEXT_TOKEN",
        )
    passed = all(row["passed"] for row in remote["rows"])
    require(remote["passed"] == saved["passed"] == passed, "PREFIX_GATE")
    return {
        "passed": passed,
        "prefixes": 168,
        "full_vocabulary_elements": sum(r["elements"] for r in remote["rows"]),
        "changed_elements": sum(r["changed_elements"] for r in remote["rows"]),
        "outside_tolerance": sum(r["outside_tolerance"] for r in remote["rows"]),
        "maximum_absolute_difference": max(
            r["max_absolute_difference"] for r in remote["rows"]
        ),
        "argmax_changes": sum(
            r["actual_argmax"] != r["expected_argmax"] for r in remote["rows"]
        ),
        "recorded_next_token_mismatches": sum(
            r["actual_argmax"] != r["recorded_next_token"]
            or r["expected_argmax"] != r["recorded_next_token"]
            for r in remote["rows"]
        ),
        "all_prefix_rows_sha256": digest(remote["rows"]),
        "tolerance": TOLERANCES["prefix_logits"],
    }


def read_boundary(ctx, result):
    proof = result["read_boundary"]
    design = ctx["design"]
    forbidden = {
        design["source_base"]["local_path"],
        *[x["directory"] for x in design["source_runs"].values()],
        *design["teacher_forbidden_roots"],
    }
    allowed = {
        f"{design['artifact_root']}/export",
        f"/mnt/shared/cl-portfolio/code/{SOURCE}",
        f"/mnt/shared/cl-portfolio/runs/{EVALUATE}",
    }
    require(
        set(proof["forbidden_roots"]) == forbidden
        and set(proof["shared_read_allowlist"]) == allowed
        and not proof["denied_reads"],
        "PYTHON_READ_BOUNDARY",
    )
    paths = proof["python_audit_read_paths"]
    require(bool(paths), "READ_AUDIT_NOT_EMPTY")
    for path, count in paths.items():
        require(type(count) is int and count > 0, "READ_COUNT")
        require(
            not any(path == root or path.startswith(root + "/") for root in forbidden)
            and not Path(path).name.lower().startswith(("teacher_", "adapter_")),
            "FORBIDDEN_READ",
            path,
        )
        if path.startswith("/mnt/shared/"):
            require(
                any(path == root or path.startswith(root + "/") for root in allowed),
                "UNLISTED_SHARED_READ",
                path,
            )
    require(
        not result["teacher_model_loaded"]
        and not result["adapter_model_loaded"]
        and not result["teacher_or_adapter_artifact_read"],
        "TEACHER_OR_ADAPTER_PRESENT",
    )
    return {
        "python_open_paths": len(paths),
        "python_open_events": sum(paths.values()),
        "read_paths_sha256": digest(paths),
        "denied_reads": [],
        "forbidden_roots": sorted(forbidden),
        "allowed_shared_roots": sorted(allowed),
        "teacher_or_adapter_artifact_read_observed": False,
        "boundary": "The sealed evaluation path loads only the ordinary export; the captured Python open audit reports no teacher, original base or adapter reads. Python audit hooks do not intercept every native-library read and are not an OS sandbox. No adversarial native-code isolation is claimed.",
    }


def audit(wave):
    ctx = inputs(wave)
    prior = original_attempts(ctx, wave)
    folders = {"export": wave / "runs" / EXPORT, "evaluate": wave / "runs" / EVALUATE}
    collections = {name: collection(path) for name, path in folders.items()}
    executions = {
        name: execution(path, EXPORT if name == "export" else EVALUATE, SOURCE)
        for name, path in folders.items()
    }
    for name, value in executions.items():
        task = value["task"]
        expected_config = json.loads(ctx["files"][f"configs/native303_{name}.json"])
        require(
            task["entrypoint"] == "native303_export.py"
            and task["code_dir"] == f"/mnt/shared/cl-portfolio/code/{SOURCE}"
            and task["gpus"] == 1
            and len(value["gpus"]) == 1
            and task["config"] == expected_config,
            "FIXED_JOB_EXECUTION_CONTRACT",
            name,
        )
    exported, evaluated = (
        read(folders["export"] / "export_receipt.json"),
        read(folders["evaluate"] / "result.json"),
    )
    archive_pin = pin(wave / "code" / f"{SOURCE}.tar")
    for label, receipt in (("export", exported), ("evaluate", evaluated)):
        runtime = read(folders[label] / "runtime.json")
        identity(receipt, executions[label], runtime, archive_pin)
        require(
            runtime["torch"] == ctx["design"]["worker_torch"]
            and runtime["packages"] == ctx["design"]["packages"],
            "DUAL_EXACT_RUNTIME_PIN",
        )
    require(
        exported["pid"] != evaluated["pid"]
        and executions["export"]["attempt_id"] != executions["evaluate"]["attempt_id"]
        and executions["export"]["task_id"] != executions["evaluate"]["task_id"],
        "SEPARATE_JOB_PID",
    )
    require(
        len(
            {
                ctx["train"]["pid"],
                ctx["evaluation"]["pid"],
                exported["pid"],
                evaluated["pid"],
            }
        )
        == 4,
        "ALL_FRESH_PIDS",
    )
    require(
        datetime.fromisoformat(executions["export"]["finished_at"])
        <= datetime.fromisoformat(executions["evaluate"]["started_at"]),
        "COMPLETED_EXPORT_BEFORE_RELOAD",
    )
    dependency = evaluated["export_dependency"]
    require(
        dependency["execution"] == executions["export"]
        and dependency["receipt_pin"] == pin(folders["export"] / "export_receipt.json")
        and dependency["execution_pin"] == pin(folders["export"] / "execution.json"),
        "EXPORT_DEPENDENCY_RECEIPT",
    )
    require(
        exported["source_adapter"] == ctx["design"]["adapter"]
        and exported["source_base_tensor_sha256"] == ctx["design"]["base_tensor_sha256"]
        and not exported["teacher_model_loaded"],
        "EXPORTED_SOURCE",
    )
    require(
        exported["status"] == "exported_pending_separate_job_parity", "EXPORT_STATUS"
    )
    require(
        exported["artifact"]["directory"] == ctx["design"]["artifact_root"] + "/export",
        "FIXED_EXTERNAL_EXPORT_LOCATION",
    )
    input_manifest = read(folders["export"] / "input_manifest.json")
    require(
        input_manifest["protocol"] == PROTOCOL
        and input_manifest["source"]
        == {name: spec["files"] for name, spec in ctx["design"]["source_runs"].items()},
        "EXPORT_INPUT_MANIFEST",
    )
    base_pins = input_manifest["base"]
    require(
        set(base_pins)
        == {spec["path"] for spec in ctx["design"]["source_base"]["files"]},
        "BASE_INPUT_FILE_CLOSURE",
    )
    for spec in ctx["design"]["source_base"]["files"]:
        require(
            all(
                base_pins[spec["path"]][key] == spec[key]
                for key in ("sha256", "bytes", "git_blob_sha1")
                if spec.get(key) is not None
            ),
            "BASE_INPUT_FILE_PIN",
            spec["path"],
        )
    remote = remote_artifact(ctx, wave, exported)
    manifest, evidence = remote["manifest"], remote["inputs"]
    require(
        manifest["protocol_sha256"] == PROTOCOL
        and manifest["identity"] == exported["identity"]
        and manifest["optimizer_updates"] == 0
        and manifest["source_checkpoint"] == 384,
        "PROMOTED_MANIFEST_IDENTITY",
    )
    require(manifest["base_file_pins"] == base_pins, "PROMOTED_BASE_INPUT_PINS")
    require(
        evidence["rows"] == ctx["rows"]
        and evidence["reference"] == ctx["references"]
        and evidence["tokens"] == ctx["design"]["tokens"]
        and evidence["source_dataset_sha256"] == ctx["row_hashes"],
        "EXPORT_RECORD_AND_FIXTURE_BINDING",
    )
    require(
        evidence["source_execution"] == ctx["sources"]
        and evidence["source_checkpoint"] == 384
        and evidence["source_adapter_tensor_sha256"] == ctx["adapter"]["tensor_sha256"]
        and evidence["source_training_pid"] == ctx["train"]["pid"]
        and evidence["source_evaluation_pid"] == ctx["evaluation"]["pid"],
        "EXPORT_SOURCE_RECEIPT_BINDING",
    )
    require(
        evidence["source_pins"]
        == {name: spec["files"] for name, spec in ctx["design"]["source_runs"].items()},
        "EXPORT_SOURCE_FILES_BINDING",
    )
    for artifact in (exported["artifact"], evaluated["artifact"]):
        require(
            artifact["manifest"] == remote["manifest_pin"]
            and artifact["files"] == remote["files"]
            and artifact["tensor_payload_bytes"] == remote["tensor_payload_bytes"],
            "EXTERNAL_HASH_PROOF_BINDING",
        )
    token_bytes = base64.b64decode(remote.pop("tokenizer_base64"), validate=True)
    require(
        {"bytes": len(token_bytes), "sha256": hashlib.sha256(token_bytes).hexdigest()}
        == remote["files"]["model/tokenizer.json"],
        "EXPORTED_TOKENIZER_PIN",
    )
    tokenizer = Tokenizer.from_str(token_bytes.decode())
    require(
        tokenizer.get_vocab() == ctx["tokenizer"].get_vocab()
        and tokenizer.token_to_id("<|im_end|>") == 151645,
        "EXPORTED_TOKENIZER_VOCAB",
    )
    premerge, merge = (
        read(folders["export"] / "premerge.json"),
        read(folders["export"] / "merge_parity.json"),
    )
    reload = read(folders["evaluate"] / "reload_probe_parity.json")
    require(
        remote["merged_probes"] == merge["panel"]
        and remote["merge_parity"] == merge["comparisons"]
        and evaluated["probe_comparisons"] == reload["comparisons"],
        "PARITY_WITNESS_BINDING",
    )
    panels = {
        "source": {
            "train": ctx["references"]["train"],
            "generic": ctx["references"]["generic"],
        },
        "premerge": premerge["panel"],
        "merged": merge["panel"],
        "reloaded": reload["panel"],
    }
    grades = {
        stage: {
            "train": native_panel(
                ctx["rows"]["train"], panel["train"], tokenizer, ctx["design"]["tokens"]
            ),
            "generic": generic_panel(
                ctx["rows"]["generic"], panel["generic"], tokenizer
            ),
        }
        for stage, panel in panels.items()
    }
    comparisons = {
        "premerge_original": compare_probes(
            panels["premerge"],
            panels["source"],
            TOLERANCES["source_scores"],
            premerge["comparison"],
        ),
        "merged_original": compare_probes(
            panels["merged"],
            panels["source"],
            TOLERANCES["merged_scores"],
            merge["comparisons"]["merged"],
        ),
        "fresh_original": compare_probes(
            panels["reloaded"],
            panels["source"],
            TOLERANCES["merged_scores"],
            reload["comparisons"]["original_student"],
        ),
        "fresh_merged": compare_probes(
            panels["reloaded"],
            panels["merged"],
            TOLERANCES["source_scores"],
            reload["comparisons"]["exported_student"],
        ),
    }
    require(
        merge["comparisons"]["source"] == premerge["comparison"],
        "PREMERGE_COMPARISON_COPY",
    )
    native = {}
    for split in ("test", "unused288"):
        saved = read(folders["evaluate"] / f"{split}.json")
        native[split] = native_panel(
            ctx["rows"][split], saved["records"], tokenizer, ctx["design"]["tokens"]
        )
        changes = [
            old["id"]
            for old, new in zip(ctx["references"][split], saved["records"], strict=True)
            if old != new
        ]
        require(
            saved["comparison"] == evaluated["native_panels"][split],
            "NATIVE_RESULT_BINDING",
        )
        claimed = saved["comparison"]
        require(
            claimed["passed"] == (not changes)
            and claimed["count"] == native[split]["count"]
            and claimed["identical_records"] == claimed["count"] - len(changes)
            and claimed["correct"] == native[split]["correct"]
            and claimed["native_eos"] == native[split]["terminated"]
            and [r["id"] for r in claimed["differences"]] == changes,
            "NATIVE_HELDOUT_PARITY",
        )
        native[split]["source_record_changes"] = changes
    prefix = {
        "merged_source": prefix_comparison(
            remote["prefix_comparisons"]["merged_source"],
            merge["comparisons"]["prefix_logits"],
            ctx["references"]["train"],
        ),
        "fresh_source": prefix_comparison(
            remote["prefix_comparisons"]["fresh_source"],
            reload["comparisons"]["source_prefix_logits"],
            ctx["references"]["train"],
        ),
        "fresh_merged": prefix_comparison(
            remote["prefix_comparisons"]["fresh_merged"],
            reload["comparisons"]["merged_prefix_logits"],
            ctx["references"]["train"],
        ),
    }
    require(
        exported["probe_qualification"]
        == {
            "source": comparisons["premerge_original"]["passed"],
            "merged": comparisons["merged_original"]["passed"],
            "prefix_logits": prefix["merged_source"]["passed"],
        },
        "EXPORT_QUALIFICATION_RECOUNT",
    )
    weights = exported_weights(
        ctx, remote, read(folders["export"] / "weight_audit.json")
    )
    require(
        evaluated["reloaded_weight_tensor_sha256"]
        == weights["ordinary_weight_tensor_sha256"]
        and evaluated["prefix_witness"] == remote["fresh_prefix_pin"],
        "FINAL_RELOAD_WEIGHT_AND_PREFIX_BINDING",
    )
    boundary = read_boundary(ctx, evaluated)
    passed = (
        all(x["passed"] for x in comparisons.values())
        and all(
            x["passed"] and x["recorded_next_token_mismatches"] == 0
            for x in prefix.values()
        )
        and all(not v["source_record_changes"] for v in native.values())
    )
    require(
        evaluated["passed"] == passed
        and evaluated["status"]
        == (
            "completed_parity_verified"
            if passed
            else "completed_with_prediction_differences"
        ),
        "FINAL_PARITY_STATUS",
    )
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "status": "independently_verified"
        if passed
        else "independently_verified_prediction_differences",
        "passed": passed,
        "auditor": {"file": str(Path(__file__).resolve()), **pin(Path(__file__))},
        "source": {
            "sha256": SOURCE,
            "files": 134,
            "archive": archive_pin,
            "handoff_sha256": HANDOFF,
            "protocol_sha256": PROTOCOL,
            "staging": pin(wave / "native303-runtime-staging.json"),
            "original_student": ctx["adapter"],
        },
        "original_attempts": prior,
        "collections": collections,
        "execution": {
            name: {
                "task_id": value["task_id"],
                "attempt_id": value["attempt_id"],
                "supervisor_child_pid": value["pid"],
                "model_process_pid": exported["pid"]
                if name == "export"
                else evaluated["pid"],
                "started_at": value["started_at"],
                "finished_at": value["finished_at"],
                "pod": value["pod"],
                "execution": pin(folders[name] / "execution.json"),
            }
            for name, value in executions.items()
        },
        "optimizer_updates": 0,
        "source_panel_grades": ctx["source_grades"],
        "probe_panel_grades": grades,
        "native_reloaded_panels": native,
        "probe_comparisons": comparisons,
        "full_vocabulary_prefix_comparisons": prefix,
        "standard_weights": weights,
        "read_boundary": boundary,
        "external_artifact": {
            "root": remote["root"],
            "observed_at": remote["at"],
            "manifest": remote["manifest_pin"],
            "files": remote["files"],
            "tensor_inventory_sha256": digest(remote["tensor_inventory"]),
            "independent_audit_program_sha256": remote[
                "independent_audit_program_sha256"
            ],
            "source_files_independently_verified_on_nfs": remote[
                "source_files_verified"
            ],
            "source_files_sha256": remote["source_files_sha256"],
            "generated_staged_source_bytecode_caches": remote[
                "generated_bytecode_caches"
            ],
            "full_model_shards_copied_locally": False,
            "prefix_witnesses": {
                "source": remote["files"]["evidence/source_prefix_logits.safetensors"],
                "merged": remote["files"]["evidence/merged_prefix_logits.safetensors"],
                "fresh": remote["fresh_prefix_pin"],
            },
        },
        "interpretation": {
            "outcome": "The fixed trained rank8 learner is represented by ordinary FP32 Qwen3 weights and independently reloaded in a separate job with zero additional updates. Numerical parity uses sealed tolerances; fusion is not assumed bitwise identical.",
            "native_contract": "Exactly three space-separated digits in 0..7 with the original leading space and native EOS, cap16, full-vocabulary greedy decoding.",
            "generic_contract": "128 original multiple-choice decisions, not native generation.",
            "evaluation_reuse": "24 TRAIN probes and the previously observed 192 TEST plus 864 unused source panels; no new confirmatory test or checkpoint selection.",
            "learning_claim": "Export/persistence engineering evidence. No new learning, direct base-weight optimization, new consolidation objective, SDFT/SDPO reproduction or general continual-learning claim.",
        },
    }


def self_test(wave):
    ctx = inputs(
        wave, require_collected_archive=(wave / "code" / f"{SOURCE}.tar").exists()
    )
    original_attempts(ctx, wave)
    names = [
        "fixed_source_and_raw_native_generic_rescoring",
        "original_failure_and_never_executed_child_binding",
    ]
    row, tokenizer, contract = (
        ctx["rows"]["train"][0],
        ctx["tokenizer"],
        ctx["design"]["tokens"],
    )
    record = ctx["references"]["train"][0]
    wrong = json.loads(json.dumps(record))
    wrong["generation"]["token_ids"][1] = tokenizer.encode(
        "7" if row["choices"][row["gold_idx"]][1] != "7" else "6",
        add_special_tokens=False,
    ).ids[0]
    try:
        native_panel([row], [wrong], tokenizer, contract)
    except ValueError:
        names.append("incorrect_tokens_cannot_reuse_correct_record")
    else:
        raise AssertionError("Incorrect token record accepted")
    flags = native_flags(
        row, record["generation"]["token_ids"][:-1], tokenizer, contract
    )
    require(
        not flags["correct"] and flags["exact_body_without_valid_stop"],
        "SELF_TEST_NO_PREFIX_RESCUE",
    )
    names.append("correct_body_without_eos_rejected")
    probe = json.loads(json.dumps(ctx["references"]["generic"][0]))
    probe["prediction"] = (probe["prediction"] + 1) % probe["choice_count"]
    try:
        generic_panel(ctx["rows"]["generic"][:1], [probe], tokenizer)
    except ValueError:
        names.append("wrong_generic_argmax_rejected")
    else:
        raise AssertionError("Wrong generic argmax accepted")
    require(
        numeric([1.0005], [1], TOLERANCES["prefix_logits"])["passed"]
        and not numeric([1.002], [1], TOLERANCES["prefix_logits"])["passed"],
        "SELF_TEST_NUMERICAL_BOUNDARY",
    )
    names.append("tolerated_fusion_difference_and_outside_tolerance_rejection")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.tar"
        with tarfile.open(path, "w") as archive:
            item = tarfile.TarInfo("../escape.py")
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
        try:
            archive_files(path, "invalid", 1)
        except ValueError:
            names.append("archive_path_escape_rejected")
        else:
            raise AssertionError("Archive traversal accepted")
    code = REMOTE + "\nself_test()\n"
    process = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-B",
            "-c",
            code,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    remote = json.loads(process.stdout)
    return {
        "status": "cpu_verified",
        "checks": names,
        "remote_stdlib_checks": remote,
        "experiment_imports": False,
        "model_loads": 0,
        "optimizer_updates": 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Independent native303 evidence audit; no model loads or optimization."
    )
    parser.add_argument("--wave", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        report = self_test(args.wave)
    elif args.source_check:
        ctx = inputs(
            args.wave,
            require_collected_archive=(args.wave / "code" / f"{SOURCE}.tar").exists(),
        )
        report = {
            "status": "source_checked_no_new_model_execution_claim",
            "source": SOURCE,
            "source_adapter": ctx["adapter"],
            "source_raw_panel_grades": ctx["source_grades"],
            "original_attempts": original_attempts(ctx, args.wave),
        }
    else:
        require(args.output is not None, "OUTPUT_REQUIRED_AFTER_PARENT_COLLECTION")
        report = audit(args.wave)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    **pin(args.output),
                },
                allow_nan=False,
            )
        )
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
