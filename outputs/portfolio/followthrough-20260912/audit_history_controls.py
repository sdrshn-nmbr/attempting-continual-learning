import argparse
import ast
import hashlib
import json
import math
import random
import sys
import tarfile
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

WAVE = Path(__file__).resolve().parent
ROOT = WAVE.parents[2]
HISTORY_SOURCE = "d93d4bad5c2fe9f70e7219a00f636fff427ef119681c475e260764852b70d658"
ORIGINAL_SOURCE = "99ac3030bb56b44579b07e0f672ce098b0831f9c871b350d087201ad546ba8d1"
PREFIX = "followthrough-20260912-recovery-"
BUDGETS = (2, 4, 8, 16)
INPUT_KEYS = ("intent", "text_sha256", "source_split", "source_index", "input_ids")
METRICS = (
    "accuracy",
    "gain",
    "guard",
    "guard_change",
    "guard_loss",
    "utility",
    "qualified",
    "accuracy_passed",
    "guard_passed",
    "updates",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def compact(value):
    if isinstance(value, dict):
        value = {str(key): compact(item) for key, item in value.items()}
    elif isinstance(value, (set, tuple, list)):
        values = sorted(value, key=repr) if isinstance(value, set) else value
        value = [compact(item) for item in values]
    elif not isinstance(value, (str, int, float, bool, type(None))):
        value = repr(value)
    encoded = canonical(value)
    return (
        value
        if len(encoded) < 1400
        else {
            "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        }
    )


class AuditFailure(Exception):
    def __init__(self, code, expected=None, actual=None):
        super().__init__(code)
        self.detail = {
            "check": code,
            "expected": compact(expected),
            "actual": compact(actual),
        }


def equal(actual, expected, code):
    if actual != expected:
        raise AuditFailure(code, expected, actual)


def require(value, code, actual=None):
    if not value:
        raise AuditFailure(code, True, actual)


def before(earlier, later, code):
    require(
        datetime.fromisoformat(earlier) <= datetime.fromisoformat(later),
        code,
        [earlier, later],
    )


def flatten(stream, kind, role):
    return [row for intent in stream[kind] for row in stream["units"][intent][role]]


def alternate(stream):
    old = stream["old"]
    original = {intent: stream["units"][intent]["learn"][0]["code"] for intent in old}
    rotated = {
        intent: original[old[(i + 1) % len(old)]] for i, intent in enumerate(old)
    }
    equal(len(set(original.values())), 8, "eight_distinct_old_codes")
    equal(
        set(rotated.values()),
        set(original.values()),
        "derangement_preserves_old_code_set",
    )
    require(
        all(rotated[intent] != original[intent] for intent in old),
        "derangement_has_no_fixed_points",
    )
    return original, rotated


def relabeled(rows, mapping, codes):
    return [
        row | {"code": mapping[row["intent"]], "target": codes[mapping[row["intent"]]]}
        for row in rows
    ]


def seed(seed_value, *labels):
    return int(digest([seed_value, *labels])[:15], 16)


def learning_batches(rows, steps, batch_size, seed_value):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["code"]].append(index)
    require(
        bool(rows) and batch_size % len(groups) == 0, "balanced_learning_batch_size"
    )
    queues = {code: [] for code in groups}
    rng, result = random.Random(seed_value), []
    for _ in range(steps):
        batch = []
        for code in sorted(groups):
            for _ in range(batch_size // len(groups)):
                if not queues[code]:
                    queues[code] = rng.sample(groups[code], len(groups[code]))
                batch.append(queues[code].pop())
        rng.shuffle(batch)
        result.append(batch)
    return result


def repair_schedule(stream, intent):
    old = stream["units"][intent]["repair"]
    new = flatten(stream, "new", "repair")
    rows, rng = old + new, random.Random(seed(stream["seed"], intent, "repair"))
    queues, batches = [[], []], []
    populations = [list(range(len(old))), list(range(len(old), len(rows)))]
    for _ in range(16):
        batch = []
        for _ in range(4):
            for side in (0, 1):
                if not queues[side]:
                    queues[side] = rng.sample(populations[side], len(populations[side]))
                batch.append(queues[side].pop())
        rng.shuffle(batch)
        batches.append(batch)
    return rows, batches


def input_schedule(rows, batches):
    return [
        [{key: rows[index][key] for key in INPUT_KEYS} for index in batch]
        for batch in batches
    ]


def verify_trace(trace, batches):
    equal(
        [row["step"] for row in trace],
        list(range(1, len(batches) + 1)),
        "actual_update_numbers",
    )
    equal([row["rows"] for row in trace], batches, "actual_batch_indices")
    require(
        all(
            math.isfinite(row[key])
            for row in trace
            for key in ("loss", "gradient_norm")
        ),
        "finite_learning_or_repair_trace",
    )
    require(
        all(row["gradient_norm"] > 0 and row["loss"] >= 0 for row in trace),
        "positive_gradient_nonnegative_loss",
    )


def native_scores(records, rows, codes):
    equal(len(codes), 16, "native_16_code_contract")
    equal(len(set(codes)), 16, "unique_code_tokens")
    equal(len(records), len(rows), "evaluation_row_count")
    counts = defaultdict(lambda: [0, 0])
    max_error = defaultdict(float)
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        equal(
            {key: record[key] for key in ("intent", "target", "text_sha256")},
            {key: row[key] for key in ("intent", "target", "text_sha256")},
            f"evaluation_identity_row_{index}",
        )
        output = record["output"]
        logits = output["code_logits"]
        require(
            len(logits) == 16 and all(math.isfinite(x) for x in logits),
            "finite_16_logits",
            index,
        )
        winner = max(range(16), key=logits.__getitem__)
        equal(output["prediction"], codes[winner], "prediction_recomputed_from_logits")
        target = codes.index(row["target"])
        shifted = [x - max(logits) for x in logits]
        log_total = math.log(math.fsum(math.exp(x) for x in shifted))
        logp = [x - log_total for x in shifted]
        probabilities = [math.exp(x) for x in logp]
        largest = sorted(probabilities, reverse=True)
        expected = {
            "target_logp": logp[target],
            "target_margin": logits[target]
            - max(x for i, x in enumerate(logits) if i != target),
            "top_probability": largest[0],
            "top_margin": largest[0] - largest[1],
            "entropy": -math.fsum(
                p * lp for p, lp in zip(probabilities, logp, strict=True)
            ),
        }
        for key, value in expected.items():
            error = abs(output[key] - value)
            max_error[key] = max(max_error[key], error)
            require(
                math.isfinite(output[key]) and error <= 3e-5 + 3e-6 * abs(value),
                "saved_native_statistic_matches_logits",
                {"row": index, "metric": key, "error": error},
            )
        counts[row["intent"]][0] += int(winner == target)
        counts[row["intent"]][1] += 1
    require(bool(counts), "nonempty_scoring_panel")
    return {
        name: {"correct": c, "total": n, "accuracy": c / n}
        for name, (c, n) in sorted(counts.items())
    }, dict(max_error)


def fraction(count):
    return Fraction(count["correct"], count["total"])


def acquired(before_counts, after_counts, rule):
    equal(set(before_counts), set(after_counts), "acquisition_intent_set")
    return sorted(
        intent
        for intent in after_counts
        if fraction(after_counts[intent]) >= Fraction(str(rule["acquired_min"]))
        and fraction(after_counts[intent]) - fraction(before_counts[intent])
        >= Fraction(str(rule["acquisition_gain_min"]))
    )


def aggregate_counts(counts):
    correct = sum(row["correct"] for row in counts.values())
    total = sum(row["total"] for row in counts.values())
    return {"correct": correct, "total": total, "accuracy": correct / total}


def outcome(counts, baseline, budget, rule):
    accuracy = fraction(counts["test"])
    guard = fraction(counts["guard"])
    gain = accuracy - fraction(baseline["test"])
    guard_change = guard - fraction(baseline["guard"])
    loss = max(Fraction(0), -guard_change)
    accuracy_passed = accuracy >= Fraction(str(rule["accuracy_min"]))
    guard_passed = loss <= Fraction(str(rule["guard_drop_max"]))
    return {
        "accuracy": float(accuracy),
        "gain": float(gain),
        "guard": float(guard),
        "guard_change": float(guard_change),
        "guard_loss": float(loss),
        "utility": float(gain - loss),
        "qualified": accuracy_passed
        and gain >= Fraction(str(rule["gain_min"]))
        and guard_passed,
        "accuracy_passed": accuracy_passed,
        "guard_passed": guard_passed,
        "updates": budget,
    }


def budget_label(values):
    qualifying = [b for b in BUDGETS if values[str(b)]["qualified"]]
    return {
        "minimum_budget": str(min(qualifying)) if qualifying else "never",
        "qualified_budgets": qualifying,
        "later_qualification_loss": bool(qualifying)
        and any(
            not values[str(b)]["qualified"] for b in BUDGETS if b > min(qualifying)
        ),
    }


def tensor_sha(tensor):
    flat, h = tensor.detach().contiguous().view(-1), hashlib.sha256()
    for chunk in flat.split(1024 * 1024):
        h.update(chunk.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def second_moment_tail(old, new, beta, updates):
    carried = old.double() * beta**updates
    final = new.double()
    tolerance = 4 * updates * 2**-24 * torch.maximum(carried.abs(), final.abs()) + 1e-30
    violations = int(((carried - final) > tolerance).sum())
    return {
        "violations": violations,
        "elements": old.numel(),
        "minimum_new_contribution": float((final - carried).min()),
        "maximum_new_contribution": float((final - carried).max()),
    }


def optimizer_clock(state):
    values = [float(row["step"]) for row in state["state"].values()]
    require(
        all(math.isfinite(x) and x == int(x) for x in values),
        "finite_integer_optimizer_clocks",
    )
    return sorted({int(x) for x in values})


def source_optimizer_flow(history_text, reset_text):
    tree = ast.parse(history_text)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "learn_control"
    )
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "optimizer"
            for target in node.targets
        )
    ]
    factories = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "optimizer"
    ]
    updates = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
    ]
    equal(len(assignments), 1, "single_learning_optimizer_assignment")
    equal(len(factories), 1, "single_learning_optimizer_construction")
    equal(len(updates), 3, "old_and_two_new_segments")
    require(
        all(
            len(node.args) >= 3
            and isinstance(node.args[2], ast.Name)
            and node.args[2].id == "optimizer"
            for node in updates
        ),
        "all_learning_segments_receive_same_optimizer_variable",
    )
    reset_tree = ast.parse(reset_text)
    reset = next(
        node
        for node in ast.walk(reset_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reset"
    )
    require(
        not any(
            isinstance(node, ast.Name) and node.id == "optimizer"
            for node in ast.walk(reset)
        ),
        "adapter_reset_does_not_reference_optimizer",
    )
    return {
        "optimizer_constructions": len(factories),
        "optimizer_assignments": len(assignments),
        "update_calls_using_same_variable": len(updates),
        "adapter_reset_has_no_optimizer_reference": True,
        "learn_function_lines": [function.lineno, function.end_lineno],
        "reset_function_lines": [reset.lineno, reset.end_lineno],
        "evidence_kind": "AST inspection of the actual archived source, not a dynamic object-identity trace",
    }


class Auditor:
    def __init__(self, wave):
        self.wave = wave
        self.started = now()
        self.hash_cache = {}
        self.tensor_cache = {}
        self.loaded = {}
        self.snapshots = {}
        self.archives = {}
        self.issues = []
        self.pending = []
        self.raw_records = 0
        self.raw_panels = 0
        self.native_errors = defaultdict(float)
        self.report = {
            "started_at": self.started,
            "scope": "Independent CPU audit of matched history controls; no experiment imports or execution, provider calls, or artifact edits.",
        }

    def pin(self, path):
        path = Path(path)
        st = path.stat()
        identity = (st.st_size, st.st_mtime_ns, st.st_ino)
        if path in self.hash_cache:
            equal(
                identity,
                self.hash_cache[path][0],
                "input_changed_since_snapshot:" + str(path),
            )
            return self.hash_cache[path][1]
        h = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                h.update(chunk)
        end = path.stat()
        equal(
            (end.st_size, end.st_mtime_ns, end.st_ino),
            identity,
            "input_changed_while_hashing:" + str(path),
        )
        result = {"bytes": st.st_size, "sha256": h.hexdigest()}
        self.hash_cache[path] = (identity, result)
        return result

    def read(self, path):
        self.pin(path)
        return json.loads(Path(path).read_text())

    def envelope(self, path):
        value = self.read(path)
        equal(
            digest(value["payload"]),
            value["sha256"],
            "envelope_payload_sha256:" + str(path),
        )
        return value["payload"]

    def attempt(self, scope, operation):
        try:
            return operation()
        except AuditFailure as error:
            issue = {"scope": scope, "kind": "evidence_discrepancy", **error.detail}
        except (
            KeyError,
            ValueError,
            TypeError,
            OSError,
            StopIteration,
            RuntimeError,
        ) as error:
            issue = {
                "scope": scope,
                "kind": "auditor_or_schema_error",
                "exception": type(error).__name__,
                "detail": str(error),
            }
        self.issues.append(issue)
        print(
            json.dumps({"AUDIT_CHECK_FAILED": issue}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return None

    def snapshot(self, ids):
        for run_id in sorted(set(ids)):
            directory = self.wave / "runs" / run_id
            marker = directory / "collection.json"
            if not marker.is_file():
                self.pending.append(
                    {"run_id": run_id, "status": "not_locally_collected_at_snapshot"}
                )
                continue
            collection = self.read(marker)
            self.snapshots[run_id] = {
                "directory": directory,
                "collection": collection,
                "collection_pin": self.pin(marker),
                "snapshot_at": now(),
                "execution_status": collection["execution_status"],
            }

    def archive(self, source):
        path = self.wave / "code" / (source + ".tar")
        receipt = self.read(path.with_suffix(".json"))
        equal(self.pin(path), receipt["archive"], "archive_file_bytes:" + str(path))
        equal(receipt["source_sha256"], source, "archive_source_identity")
        files = {}
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                name = member.name
                require(
                    not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "archive_relative_name",
                    name,
                )
                require(
                    member.isfile() or member.isdir(),
                    "archive_regular_source_only",
                    name,
                )
                if member.isfile():
                    require(name not in files, "archive_duplicate_file", name)
                    files[name] = archive.extractfile(member).read()
        closure = hashlib.sha256()
        for name in sorted(files, key=Path):
            closure.update(name.encode() + b"\0" + files[name])
        equal(closure.hexdigest(), source, "archive_source_closure")
        equal(len(files), receipt["files"], "archive_file_count")
        self.archives[source] = files
        return {
            "source_sha256": source,
            "archive": self.pin(path),
            "files": len(files),
            "closure_verified": True,
        }

    def documents(self):
        archive_reports = [
            self.archive(source) for source in (ORIGINAL_SOURCE, HISTORY_SOURCE)
        ]
        files, original = self.archives[HISTORY_SOURCE], self.archives[ORIGINAL_SOURCE]
        self.protocol = json.loads(files["history-control-protocol.json"])
        self.design = json.loads(files["history-control-design.json"])
        self.interface = json.loads(files["history-control-interface.json"])
        self.config = json.loads(files[self.protocol["original_config"]])
        self.spec = json.loads(files[self.config["protocol"]])
        self.data = json.loads(files[self.config["dataset"]])
        for name in ("protocol", "design", "interface"):
            filename = f"history-control-{name}.json"
            equal(
                hashlib.sha256(files[filename]).hexdigest(),
                files[filename.replace(".json", ".sha256")].decode().strip(),
                "archived_document_pin:" + filename,
            )
            equal(
                self.pin(ROOT / "experiments/recoverability" / filename)["sha256"],
                hashlib.sha256(files[filename]).hexdigest(),
                "local_document_matches_executed_source:" + filename,
            )
        for name, expected in self.protocol["original_file_sha256"].items():
            equal(
                hashlib.sha256(original[name]).hexdigest(),
                expected,
                "underlying_original_source_pin:" + name,
            )
            equal(
                files[name],
                original[name],
                "frozen_dependency_same_original_bytes:" + name,
            )
        interface_pins = {}
        local_test_pins = {}
        for name in sorted(self.interface["file_sha256"], key=Path):
            if name == "tests/test_history_control.py" and name not in files:
                local = ROOT / "experiments/recoverability" / name
                content = local.read_bytes()
                local_test_pins[name] = self.pin(local) | {
                    "status": "same_sealed_bytes_in_local_test; not_in_executed_archive"
                }
            else:
                content = files[name]
            interface_pins[name] = hashlib.sha256(content).hexdigest()
            equal(
                interface_pins[name],
                self.interface["file_sha256"][name],
                "interface_dependency_pin:" + name,
            )
        equal(
            len(self.interface["file_sha256"]),
            self.interface["file_closure_count"],
            "interface_subset_count",
        )
        equal(
            digest(interface_pins),
            self.interface["file_closure_sha256"],
            "interface_subset_closure",
        )
        self.implementation = {
            name: hashlib.sha256(files[name]).hexdigest()
            for name in (
                "history_control.py",
                "history_control_sources.py",
                "history_control_model.py",
                "history_control_analysis.py",
            )
        }
        self.identity = digest(
            {
                "protocol": self.protocol,
                "design": self.design,
                "implementation": self.implementation,
                "original_prediction_identity": self.protocol[
                    "original_prediction_identity"
                ],
            }
        )
        equal(self.identity, self.interface["history_identity"], "history_identity")
        equal(
            self.implementation["history_control.py"],
            self.interface["entrypoint_sha256"],
            "history_entrypoint",
        )
        parent_path = self.wave / "history-control-parent-decision.json"
        parent = self.envelope(parent_path)
        equal(
            self.pin(parent_path)["sha256"],
            self.design["parent_decision_file_sha256"],
            "parent_decision_file_pin",
        )
        equal(
            self.design["parent_decision"],
            {"payload": parent, "sha256": digest(parent)},
            "parent_decision_embedded_binding",
        )
        before(
            parent["at"],
            self.protocol["sealed_at_utc"],
            "parent_decision_before_protocol",
        )
        equal(
            self.protocol["training"]["settings"],
            self.spec["optimizer"],
            "unchanged_optimizer_contract",
        )
        equal(
            self.protocol["repair"]["budgets"],
            list(BUDGETS),
            "unchanged_repair_budgets",
        )
        equal(
            self.protocol["repair"]["qualification"],
            self.spec["recovered"],
            "unchanged_target_qualification",
        )
        equal(self.config["acquisition_updates"], 128, "old_learning_budget")
        equal(self.config["forgetting_updates"], 128, "new_learning_budget")
        mappings = {}
        for name, stream in self.data["streams"].items():
            old, rotated = alternate(stream)
            new = {
                intent: stream["units"][intent]["learn"][0]["code"]
                for intent in stream["new"]
            }
            equal(set(old) & set(new), set(), "old_new_intents_disjoint")
            equal(
                set(old.values()) & set(new.values()),
                set(),
                "old_new_code_sets_disjoint",
            )
            equal(
                set(old.values()) | set(new.values()),
                set(range(16)),
                "all_sixteen_codes_covered",
            )
            fixed_mapping = old | new
            declared = self.protocol["mappings"][name]
            equal(declared["original"], old, "declared_original_mapping:" + name)
            equal(declared["alternate"], rotated, "declared_cyclic_mapping:" + name)
            equal(
                declared["old_intent_order"],
                stream["old"],
                "declared_old_order:" + name,
            )
            all_roles = {
                role: flatten(stream, "old", role) + flatten(stream, "new", role)
                for role in ("learn", "repair", "gate", "probe", "test")
            }
            uid_sets = {
                role: {row["text_sha256"] for row in rows}
                for role, rows in all_roles.items()
            }
            for i, left in enumerate(uid_sets):
                for right in list(uid_sets)[i + 1 :]:
                    require(
                        not uid_sets[left] & uid_sets[right],
                        "disjoint_learning_gate_repair_test_uids",
                        [name, left, right],
                    )
            for rows in all_roles.values():
                for row in rows:
                    equal(
                        row["code"],
                        fixed_mapping[row["intent"]],
                        "consistent_original_binding_across_data_roles",
                    )
                    equal(
                        row["target"],
                        self.data["codes"][row["code"]],
                        "dataset_code_target_consistency",
                    )
            mappings[name] = {
                "original": old,
                "alternate": rotated,
                "old_fixed_points": 0,
                "old_code_set_unchanged": True,
                "new_mapping_unchanged_required": True,
                "role_rows": {key: len(rows) for key, rows in all_roles.items()},
                "role_uid_disjointness_verified": True,
            }
        flow = source_optimizer_flow(
            files["history_control_model.py"].decode(), files["model.py"].decode()
        )
        self.report["sources"] = {
            "archives": archive_reports,
            "protocol_sha256": hashlib.sha256(
                files["history-control-protocol.json"]
            ).hexdigest(),
            "design_sha256": hashlib.sha256(
                files["history-control-design.json"]
            ).hexdigest(),
            "history_identity": self.identity,
            "original_prediction_identity": self.protocol[
                "original_prediction_identity"
            ],
            "original_dependencies_verified": len(
                self.protocol["original_file_sha256"]
            ),
            "interface_dependencies_verified": len(self.interface["file_sha256"]),
            "interface_nonexecuted_local_test_pins": local_test_pins,
            "interface_closure_sha256": digest(interface_pins),
            "interface_closure_format": "SHA256 of canonical JSON filename-to-file-SHA256 map; distinct from provider source archive byte closure",
            "dataset_file_sha256": hashlib.sha256(
                files[self.config["dataset"]]
            ).hexdigest(),
            "dataset_canonical_sha256": digest(self.data),
            "parent_decision": self.pin(parent_path),
            "temporal_boundary": self.design["temporal_interpretation"],
            "source_optimizer_flow": flow,
            "mappings": mappings,
        }

    def load_run(self, run_id, source):
        if run_id in self.loaded:
            return self.loaded[run_id]
        snapshot = self.snapshots.get(run_id)
        if snapshot is None or snapshot["execution_status"] != "completed":
            return None
        directory, collection = snapshot["directory"], snapshot["collection"]
        for name, expected in collection["files"].items():
            equal(
                self.pin(directory / name),
                expected,
                "collected_file_bytes:" + str(directory / name),
            )
        execution = self.read(directory / "execution.json")
        equal(execution["status"], "completed", "completed_execution")
        equal(execution["exit_code"], 0, "completed_exit_code")
        equal(execution["task_id"], run_id, "execution_run_id")
        equal(execution["source_sha256"], source, "execution_source")
        equal(
            execution["task"],
            self.read(directory / "task.json"),
            "execution_task_content",
        )
        equal(
            execution["task_sha256"],
            self.pin(directory / "task.json")["sha256"],
            "execution_task_file_sha",
        )
        equal(
            execution["config_sha256"],
            self.pin(directory / "config.json")["sha256"],
            "execution_config_file_sha",
        )
        equal(
            execution["task"]["config"],
            self.read(directory / "config.json"),
            "task_config_content",
        )
        equal(execution["task"]["source_sha256"], source, "task_source")
        equal(Path(execution["task"]["code_dir"]).name, source, "code_directory_source")
        study = directory / "study"
        receipt, seal = (
            self.envelope(study / "receipt.json"),
            self.envelope(study / "seal.json"),
        )
        equal(
            seal["dispatch_manifest"],
            execution["task"]["config"],
            "scientific_seal_dispatch_config",
        )
        equal(seal["config"], self.config, "sealed_original_learning_config")
        equal(seal["protocol"], self.spec, "sealed_original_learning_spec")
        equal(seal["dataset_sha256"], digest(self.data), "sealed_dataset_identity")
        study_members = {
            name.removeprefix("study/")
            for name in collection["files"]
            if name.startswith("study/") and name != "study/receipt.json"
        }
        equal(set(receipt["files"]), study_members, "scientific_receipt_file_closure")
        for name, expected in receipt["files"].items():
            equal(
                self.pin(study / name)["sha256"],
                expected,
                "scientific_receipt_file_sha:" + str(study / name),
            )
        for key in (
            "implementation",
            "learning_implementation",
            "prediction_implementation",
            "history_implementation",
        ):
            for name, expected in seal.get(key, {}).items():
                equal(
                    hashlib.sha256(self.archives[source][name]).hexdigest(),
                    expected,
                    "executed_implementation_pin:" + name,
                )
        if source == HISTORY_SOURCE:
            equal(seal["history_identity"], self.identity, "sealed_history_identity")
            equal(seal["history_protocol"], self.protocol, "sealed_history_protocol")
            equal(seal["history_design"], self.design, "sealed_history_design")
            equal(
                receipt["history_identity"], self.identity, "receipt_history_identity"
            )
            job = self.interface["jobs"][run_id]
            stage = execution["task"]["config"]["stage"]
            equal(receipt["stage"], "history-control-" + stage, "receipt_history_stage")
            equal(seal["stage"], receipt["stage"], "seal_receipt_history_stage")
            for key, expected in {
                "protocol_sha256": self.report["sources"]["protocol_sha256"],
                "design_sha256": self.report["sources"]["design_sha256"],
                "parent_decision_sha256": self.design["parent_decision"]["sha256"],
                "parent_decision_at": self.design["parent_decision_at"],
                "lane_protocol_at": self.protocol["sealed_at_utc"],
                "historical_cohort_sha256": self.protocol["original_cohort"][
                    "receipt_sha256"
                ],
            }.items():
                equal(receipt[key], expected, "receipt_prerequisite_binding:" + key)
            if stage in ("learn", "repair"):
                for key in ("stream", "condition"):
                    equal(
                        receipt[key],
                        execution["task"]["config"][key],
                        "receipt_control_identity:" + key,
                    )
            equal(
                execution["task"]["config"],
                json.loads(self.archives[source][job["config"]]),
                "executed_archived_job_config",
            )
            equal(
                execution["task"]["entrypoint"],
                job["entrypoint"],
                "executed_history_entrypoint",
            )
            equal(
                execution["task"]["depends_on"],
                job["depends_on"],
                "sealed_job_dependencies",
            )
            before(
                execution["started_at"],
                seal["stage_started_at"],
                "process_before_scientific_stage",
            )
            before(
                seal["stage_started_at"],
                receipt["completed_at"],
                "scientific_stage_time",
            )
            before(
                receipt["completed_at"],
                execution["finished_at"],
                "science_before_process_completion",
            )
        else:
            equal(
                seal["prediction_identity"],
                self.protocol["original_prediction_identity"],
                "historical_prediction_identity",
            )
        value = {
            "run_id": run_id,
            "study": study,
            "receipt": receipt,
            "seal": seal,
            "execution": execution,
        }
        self.loaded[run_id] = value
        return value

    def score(self, records, rows):
        counts, errors = native_scores(records, rows, self.data["codes"])
        self.raw_panels += 1
        self.raw_records += len(records)
        for key, value in errors.items():
            self.native_errors[key] = max(self.native_errors[key], value)
        return counts

    def tensors(self, path):
        sha = self.pin(path)["sha256"]
        if sha in self.tensor_cache:
            return self.tensor_cache[sha]
        metadata, sumsq, elements, nonzero = {}, 0.0, 0, 0
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for key in keys:
                value = handle.get_tensor(key)
                require(
                    bool(torch.isfinite(value).all()),
                    "finite_saved_tensor",
                    [str(path), key],
                )
                metadata[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": tensor_sha(value),
                }
                sumsq += float(value.double().square().sum())
                elements += value.numel()
                nonzero += int(torch.count_nonzero(value))
        require(bool(metadata), "nonempty_tensor_artifact", str(path))
        result = {
            "metadata": metadata,
            "tensors": len(metadata),
            "elements": elements,
            "nonzero": nonzero,
            "l2": math.sqrt(sumsq),
        }
        self.tensor_cache[sha] = result
        return result

    def optimizer(self, path, expected_clock, adapter):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        equal(len(state["param_groups"]), 1, "single_optimizer_parameter_group")
        group = state["param_groups"][0]
        for key, expected in {
            "lr": 0.0003,
            "weight_decay": 0,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "fused": False,
        }.items():
            equal(group[key], expected, "actual_optimizer_setting:" + key)
        equal(
            len(group["params"]),
            adapter["tensors"],
            "optimizer_covers_adapter_tensor_count",
        )
        equal(optimizer_clock(state), expected_clock, "actual_saved_optimizer_clock")
        if not expected_clock:
            equal(state["state"], {}, "actual_empty_initial_optimizer")
        else:
            equal(
                set(state["state"]),
                set(group["params"]),
                "all_optimizer_parameters_have_state",
            )
            equal(
                Counter(tuple(v["exp_avg"].shape) for v in state["state"].values()),
                Counter(tuple(v["shape"]) for v in adapter["metadata"].values()),
                "moment_and_adapter_shape_multisets",
            )
        hashes, sizes, norms, zeros = {}, Counter(), defaultdict(float), Counter()
        for index, values in state["state"].items():
            equal(set(values), {"step", "exp_avg", "exp_avg_sq"}, "adam_state_fields")
            hashes[str(index)] = {}
            equal(
                list(values["exp_avg"].shape),
                list(values["exp_avg_sq"].shape),
                "adam_moment_shape_pair",
            )
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = values[name]
                require(
                    bool(torch.isfinite(tensor).all()),
                    "finite_saved_adam_moments",
                    [str(path), index, name],
                )
                if name == "exp_avg_sq":
                    require(bool((tensor >= 0).all()), "nonnegative_adam_second_moment")
                hashes[str(index)][name] = tensor_sha(tensor)
                sizes[name] += tensor.numel()
                norms[name] += float(tensor.double().square().sum())
                zeros[name] += int(not bool(torch.count_nonzero(tensor)))
        require(
            not expected_clock or not any(zeros.values()),
            "nonzero_moment_tensor_for_each_parameter",
            dict(zeros),
        )
        return {
            "file": self.pin(path),
            "clocks": optimizer_clock(state),
            "parameter_tensors": len(group["params"]),
            "state_entries": len(state["state"]),
            "settings": {key: value for key, value in group.items() if key != "params"},
            "moment_elements": dict(sizes),
            "moment_l2": {key: math.sqrt(value) for key, value in norms.items()},
            "moment_hashes_canonical_sha256": digest(hashes),
            "finite": True,
            "zero_moment_tensors": dict(zeros),
        }

    def carried_moments(self, first_path, last_path):
        old = torch.load(first_path, map_location="cpu", weights_only=True, mmap=True)
        new = torch.load(last_path, map_location="cpu", weights_only=True, mmap=True)
        equal(
            old["param_groups"],
            new["param_groups"],
            "same_optimizer_groups_and_parameter_ids_across_boundary",
        )
        equal(set(old["state"]), set(new["state"]), "same_adam_state_parameter_ids")
        checks = [
            second_moment_tail(
                old["state"][key]["exp_avg_sq"],
                new["state"][key]["exp_avg_sq"],
                0.999,
                128,
            )
            for key in old["state"]
        ]
        violations = sum(row["violations"] for row in checks)
        equal(violations, 0, "carried_old_second_moment_lower_bound")
        return {
            "verified": True,
            "elements": sum(row["elements"] for row in checks),
            "violations": violations,
            "minimum_new_contribution": min(
                row["minimum_new_contribution"] for row in checks
            ),
            "maximum_new_contribution": max(
                row["maximum_new_contribution"] for row in checks
            ),
            "beta2": 0.999,
            "new_updates": 128,
            "old_second_moment_decay": 0.999**128,
            "necessary_bound": "v_final >= beta2**128 * v_acquired, because squared-gradient contributions are nonnegative",
            "fp32_absolute_tolerance": "4*128*2**-24*max(abs(carried),abs(final)) + 1e-30 per element",
            "proof_limit": "Necessary condition for carried moments, not sufficient to prove the full Adam recurrence or every intermediate state.",
        }

    def checkpoints(self, run, phase_clocks):
        study = run["study"]
        declared = self.read(study / "checkpoints.json")
        equal(set(declared), set(phase_clocks), "saved_checkpoint_phase_set")
        result = {}
        for phase, clock in phase_clocks.items():
            for name, sha in declared[phase].items():
                equal(
                    self.pin(study / phase / name)["sha256"],
                    sha,
                    "checkpoint_file_sha:" + str(study / phase / name),
                )
            adapter_path = study / phase / "adapter_model.safetensors"
            tensors = self.tensors(adapter_path)
            cfg = self.read(study / phase / "adapter_config.json")
            equal(cfg["r"], 8, "rank8_capacity")
            equal(cfg["lora_alpha"], 16, "rank8_alpha")
            equal(
                cfg["target_modules"],
                r"model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
                "same_adapter_target_modules",
            )
            result[phase] = {
                "adapter": self.pin(adapter_path),
                "parameter_elements": tensors["elements"],
                "parameter_tensors": tensors["tensors"],
                "finite_weights": True,
                "optimizer": self.optimizer(
                    study / phase / "optimizer.pt", clock, tensors
                ),
            }
        return result

    def historical_learning(self, name):
        rule, stream = (
            self.protocol["learning_sources"][name],
            self.data["streams"][name],
        )
        run = self.load_run(rule["run_id"], ORIGINAL_SOURCE)
        if run is None:
            return None
        study, receipt = run["study"], run["receipt"]
        equal(
            self.pin(study / "receipt.json")["sha256"],
            rule["receipt_sha256"],
            "historical_learning_receipt_pin",
        )
        for filename, key in (
            ("initial/adapter_model.safetensors", "initial_adapter_sha256"),
            ("acquisition-updates.json", "acquisition_trace_sha256"),
            ("forgetting-updates.json", "new_trace_sha256"),
        ):
            equal(
                self.pin(study / filename)["sha256"],
                rule[key],
                "historical_learning_file_pin:" + filename,
            )
        observations = self.read(study / "observations.json")
        counts, schedules = {}, {}
        for phase, panels in observations.items():
            counts[phase] = {}
            for kind in ("old", "new"):
                counts[phase][kind] = self.score(
                    panels[kind + "_gate"], flatten(stream, kind, "gate")
                )
            if "train" in panels:
                counts[phase]["train"] = self.score(
                    panels["train"],
                    flatten(stream, "old" if phase == "acquired" else "new", "learn"),
                )
        rule_gate = self.spec["gates"]
        old_acquired = acquired(
            counts["initial"]["old"], counts["acquired"]["old"], rule_gate
        )
        new_acquired = acquired(
            counts["acquired"]["new"], counts["forgotten"]["new"], rule_gate
        )
        eligible = sorted(
            intent
            for intent in old_acquired
            if fraction(counts["forgotten"]["old"][intent])
            <= Fraction(str(rule_gate["after_max"]))
            and fraction(counts["acquired"]["old"][intent])
            - fraction(counts["forgotten"]["old"][intent])
            >= Fraction(str(rule_gate["drop_min"]))
        )
        equal(
            receipt["acquired"], old_acquired, "historical_old_acquisition_recomputed"
        )
        equal(
            receipt["new_acquired"],
            new_acquired,
            "historical_new_acquisition_recomputed",
        )
        equal(receipt["eligible"], eligible, "historical_eligibility_recomputed")
        equal(eligible, rule["eligible"], "sealed_historical_eligible_cohort")
        qualified = (
            len(old_acquired) >= rule_gate["minimum_acquired_per_stream"]
            and len(new_acquired) >= rule_gate["minimum_new_acquired_per_stream"]
            and len(eligible) >= rule_gate["minimum_forgotten_per_stream"]
        )
        equal(receipt["qualified"], qualified, "historical_source_qualification")
        for kind, filename, label in (
            ("old", "acquisition-updates.json", "acquisition"),
            ("new", "forgetting-updates.json", "forgetting"),
        ):
            rows = flatten(stream, kind, "learn")
            trace = self.read(study / filename)
            batches = learning_batches(rows, 128, 8, seed(stream["seed"], label))
            verify_trace(trace, batches)
            schedules[kind] = {
                "rows": rows,
                "batches": batches,
                "historical_trace_sha256": digest(trace),
                "input_schedule_sha256": digest(input_schedule(rows, batches)),
            }
        checkpoints = self.checkpoints(
            run, {"initial": [], "acquired": [128], "forgotten": [256]}
        )
        equal(
            receipt["optimizer_steps"],
            {"acquired": [128], "final": [256]},
            "historical_recorded_clocks",
        )
        return {
            "run": run,
            "counts": counts,
            "observations": observations,
            "schedules": schedules,
            "summary": {
                "run_id": run["run_id"],
                "receipt": self.pin(study / "receipt.json"),
                "qualified": qualified,
                "old_acquired": old_acquired,
                "new_acquired": new_acquired,
                "eligible": eligible,
                "scores": counts,
                "checkpoints": checkpoints,
                "carried_moments": self.carried_moments(
                    study / "acquired/optimizer.pt", study / "forgotten/optimizer.pt"
                ),
                "input_schedules": {
                    kind: {
                        key: value
                        for key, value in schedule.items()
                        if key not in ("rows", "batches")
                    }
                    for kind, schedule in schedules.items()
                },
            },
        }

    def learning_gate(self, counts, condition):
        rule = self.protocol["source_qualification"]
        old = (
            acquired(counts["initial"]["old_own"], counts["acquired"]["old_own"], rule)
            if condition == "alternate_binding"
            else []
        )
        old_passed = (
            condition == "new_only"
            or len(old) >= rule["minimum_alternate_old_acquired"]
        )
        baseline = "acquired" if condition == "alternate_binding" else "initial"
        new = (
            acquired(counts[baseline]["new"], counts["post_stream"]["new"], rule)
            if "post_stream" in counts
            else []
        )
        new_passed = len(new) >= rule["minimum_new_acquired"]
        passed = old_passed and new_passed
        return {
            "qualified": passed,
            "status": "qualified"
            if passed
            else (
                "alternate_acquisition_failed"
                if not old_passed
                else "new_acquisition_failed"
            ),
            "alternate_acquired": old,
            "old_acquisition_required": condition == "alternate_binding",
            "old_source_passed": old_passed,
            "new_acquired": new,
            "new_source_passed": new_passed,
            "new_baseline_phase": baseline,
            "scores": {
                phase: {
                    panel: {intent: row["accuracy"] for intent, row in scores.items()}
                    for panel, scores in panels.items()
                }
                for phase, panels in counts.items()
            },
            "repair_updates": 0,
            "test_rows_evaluated": 0,
        }

    def control_learning(self, name, condition, historical):
        run_id = (
            PREFIX
            + "history-learn-"
            + condition.replace("_", "-")
            + "-"
            + name.removeprefix("fresh-")
        )
        run = self.load_run(run_id, HISTORY_SOURCE)
        if run is None or historical is None:
            return None
        study, receipt, stream = (
            run["study"],
            run["receipt"],
            self.data["streams"][name],
        )
        original_study = historical["run"]["study"]
        equal(receipt["stage"], "history-control-learn", "control_learning_stage")
        equal(receipt["condition"], condition, "control_learning_condition")
        equal(receipt["stream"], name, "control_learning_stream")
        initial = self.pin(study / "initial/adapter_model.safetensors")
        equal(
            initial,
            self.pin(original_study / "initial/adapter_model.safetensors"),
            "exact_initial_adapter_bytes",
        )
        equal(
            initial["sha256"],
            self.protocol["learning_sources"][name]["initial_adapter_sha256"],
            "sealed_initial_source_sha",
        )
        equal(
            receipt["initial_adapter_sha256"], initial["sha256"], "receipt_initial_sha"
        )
        equal(
            self.read(study / "frozen-backbone.json"),
            self.read(original_study / "frozen-backbone.json"),
            "same_original_frozen_backbone_hash_manifest",
        )
        equal(
            receipt["frozen_backbone_unchanged"], True, "reported_backbone_conservation"
        )
        observations, counts = self.read(study / "observations.json"), {}
        original_map, mapping = alternate(stream)
        old = flatten(stream, "old", "gate")
        own = (
            relabeled(old, mapping, self.data["codes"])
            if condition == "alternate_binding"
            else old
        )
        for phase, panel in observations.items():
            counts[phase] = {
                key: self.score(panel[key], rows)
                for key, rows in (
                    ("old_original", old),
                    ("old_own", own),
                    ("new", flatten(stream, "new", "gate")),
                )
            }
            equal(
                [row["output"]["code_logits"] for row in panel["old_original"]],
                [row["output"]["code_logits"] for row in panel["old_own"]],
                "label_change_leaves_logits_identical",
            )
        equal(
            observations["initial"]["old_original"],
            historical["observations"]["initial"]["old_gate"],
            "exact_initial_old_raw_outputs",
        )
        equal(
            observations["initial"]["new"],
            historical["observations"]["initial"]["new_gate"],
            "exact_initial_new_raw_outputs",
        )
        gate = self.learning_gate(counts, condition)
        equal(
            self.read(study / "learning-gate.json"),
            gate,
            "control_gate_independent_thresholds",
        )
        equal(
            {key: receipt[key] for key in gate},
            gate,
            "receipt_gate_independent_thresholds",
        )
        early_counts = {
            key: value
            for key, value in counts.items()
            if key in ("initial", "acquired")
        }
        equal(
            self.read(study / "old-acquisition-gate.json"),
            self.learning_gate(early_counts, condition),
            "early_old_only_gate_before_new_acquisition",
        )
        schedules = historical["schedules"]
        equal(
            self.read(study / "schedules-before-updates.json"),
            schedules,
            "preupdate_schedule_seal_matches_actual_history",
        )
        old_steps = 128 if condition == "alternate_binding" else 0
        new_steps = 128 if gate["old_source_passed"] else 0
        trace_reports = {}
        for kind, steps in (("old", old_steps), ("new", new_steps)):
            path = study / f"{kind}-updates.json"
            if not steps:
                require(
                    not path.exists(),
                    "no_updates_for_skipped_or_failed_phase",
                    str(path),
                )
                continue
            trace = self.read(path)
            expected_rows = schedules[kind]["rows"]
            if kind == "old":
                expected_rows = relabeled(expected_rows, mapping, self.data["codes"])
            equal(
                trace["rows"],
                expected_rows,
                "exact_rows_with_only_permitted_old_label_changes",
            )
            batches = schedules[kind]["batches"]
            verify_trace(trace["updates"], batches)
            actual_input_hash = digest(input_schedule(trace["rows"], batches))
            equal(
                actual_input_hash,
                schedules[kind]["input_schedule_sha256"],
                "actual_input_uid_token_batch_identity",
            )
            changed = sum(
                trace["rows"][i]["target"] != schedules[kind]["rows"][i]["target"]
                for i in range(len(expected_rows))
            )
            equal(changed, 256 if kind == "old" else 0, "changed_label_count")
            trace_reports[kind] = {
                "updates": steps,
                "exposures": steps * 8,
                "input_schedule_sha256": actual_input_hash,
                "actual_trace": self.pin(path),
                "changed_unique_row_labels": changed,
                "original_vs_alternate_codes": [original_map, mapping]
                if kind == "old"
                else None,
                "finite_positive_gradient_norms": True,
                "first_loss": trace["updates"][0]["loss"],
                "final_loss": trace["updates"][-1]["loss"],
            }
        expected_clocks = {"initial": [], "final": [old_steps + new_steps]}
        phases = {"initial": []}
        if old_steps:
            expected_clocks["acquired"] = phases["acquired"] = [128]
        if new_steps:
            expected_clocks["new_midpoint"] = [old_steps + 64]
            phases["post_stream"] = [old_steps + new_steps]
        equal(
            receipt["optimizer_steps"],
            expected_clocks,
            "reported_control_optimizer_clocks",
        )
        equal(
            {
                key: receipt[key]
                for key in (
                    "old_updates",
                    "new_updates",
                    "old_exposures",
                    "new_exposures",
                )
            },
            {
                "old_updates": old_steps,
                "new_updates": new_steps,
                "old_exposures": old_steps * 8,
                "new_exposures": new_steps * 8,
            },
            "control_exposure_accounting",
        )
        for key in (
            "lens_updates",
            "predictor_fits",
            "repair_updates",
            "test_rows_evaluated",
        ):
            equal(receipt[key], 0, "no_learning_stage_downstream_work:" + key)
        prerequisites = self.read(study / "learning-prerequisites.json")
        equal(
            prerequisites["historical_learning_sha256"],
            self.pin(original_study / "receipt.json")["sha256"],
            "learning_prerequisite_source_receipt",
        )
        equal(
            prerequisites["qualified"], True, "historical_gate_before_control_learning"
        )
        before(
            historical["run"]["execution"]["finished_at"],
            prerequisites["at"],
            "historical_learning_before_control_prerequisite",
        )
        before(
            run["seal"]["stage_started_at"],
            prerequisites["at"],
            "prerequisite_after_stage_seal",
        )
        checkpoints = self.checkpoints(run, phases)
        return {
            "run": run,
            "gate": gate,
            "summary": {
                "run_id": run_id,
                "qualified": gate["qualified"],
                "condition": condition,
                "receipt": self.pin(study / "receipt.json"),
                "alternate_acquired": gate["alternate_acquired"],
                "new_acquired": gate["new_acquired"],
                "new_baseline_phase": gate["new_baseline_phase"],
                "scores": counts,
                "exact_initial_adapter_bytes": initial,
                "initial_outputs_exact": True,
                "frozen_backbone_manifest": self.pin(study / "frozen-backbone.json"),
                "traces": trace_reports,
                "checkpoints": checkpoints,
                "reported_midpoint_clock": expected_clocks.get("new_midpoint"),
                "midpoint_optimizer_checkpoint_saved": False,
                "carried_moments": self.carried_moments(
                    study / "acquired/optimizer.pt", study / "post_stream/optimizer.pt"
                )
                if old_steps and new_steps
                else None,
                "own_binding_before_new_gate_verified": True,
                "learning_prerequisites": self.pin(
                    study / "learning-prerequisites.json"
                ),
            },
        }

    def common(self, histories, controls):
        cohort = self.load_run(
            self.protocol["original_cohort"]["run_id"], ORIGINAL_SOURCE
        )
        gate_run = self.load_run(PREFIX + "history-gate", HISTORY_SOURCE)
        if (
            cohort is None
            or gate_run is None
            or any(
                histories[name] is None
                or any(value is None for value in controls[name].values())
                for name in self.data["streams"]
            )
        ):
            return None
        equal(
            self.pin(cohort["study"] / "receipt.json")["sha256"],
            self.protocol["original_cohort"]["receipt_sha256"],
            "original_cohort_file_pin",
        )
        groups, failures, source_paths, hashes = {}, [], {}, {}
        for name, stream in self.data["streams"].items():
            eligible = histories[name]["summary"]["eligible"]
            equal(
                cohort["receipt"]["streams"][name]["eligible"],
                eligible,
                "original_cohort_eligible_recomputed",
            )
            own_acquired = controls[name]["alternate_binding"]["gate"][
                "alternate_acquired"
            ]
            common = [intent for intent in eligible if intent in own_acquired]
            failed = [
                condition
                for condition, value in controls[name].items()
                if not value["gate"]["qualified"]
            ]
            if failed:
                failures.append({"stream": name, "failed_conditions": failed})
            groups[name] = {
                "split": stream["split"],
                "historical_eligible": eligible,
                "common": common,
                "exclusions": [
                    {
                        "intent": intent,
                        "reason": "alternate_own_binding_acquisition_failed",
                    }
                    for intent in eligible
                    if intent not in own_acquired
                ],
                "source_gates_passed": not failed,
                "enough_common": len(common)
                >= self.protocol["common_gate"]["minimum_per_stream"],
            }
            source_paths[name], hashes[name] = {}, {}
            for condition, value in controls[name].items():
                source_paths[name][condition] = (
                    "/mnt/shared/cl-portfolio/runs/" + value["run"]["run_id"] + "/study"
                )
                hashes[name][condition] = self.pin(
                    value["run"]["study"] / "receipt.json"
                )["sha256"]
                before(
                    value["run"]["execution"]["finished_at"],
                    gate_run["execution"]["started_at"],
                    "all_control_learning_complete_before_common_gate",
                )
        counts = {
            split: sum(
                len(group["common"])
                for group in groups.values()
                if group["split"] == split
            )
            for split in ("train", "test")
        }
        clusters = {
            split: sum(
                group["source_gates_passed"] and group["enough_common"]
                for group in groups.values()
                if group["split"] == split
            )
            for split in counts
        }
        passed = (
            not failures
            and all(group["enough_common"] for group in groups.values())
            and min(counts.values())
            >= self.protocol["common_gate"]["minimum_per_split"]
            and min(clusters.values())
            >= self.protocol["common_gate"]["source_clusters_per_split"]
        )
        expected = {
            "qualified": passed,
            "status": "qualified" if passed else "common_eligibility_failed",
            "streams": groups,
            "counts": counts,
            "source_clusters": clusters,
            "source_failures": failures,
            "repair_updates": 0,
            "predictor_fits": 0,
            "decision": "matched_control_repairs_allowed"
            if passed
            else "stop_all_control_repairs; new_TRAIN_protocol_required",
            "control_sources": source_paths,
            "control_receipts": hashes,
            "historical_cohort_sha256": self.protocol["original_cohort"][
                "receipt_sha256"
            ],
        }
        equal(
            self.read(gate_run["study"] / "common-gate.json"),
            expected,
            "common_gate_independent_recomputation",
        )
        equal(
            {key: gate_run["receipt"][key] for key in expected},
            expected,
            "common_gate_receipt_independent_recomputation",
        )
        self.report["common_gate"] = {
            "recomputed": expected,
            "receipt": self.pin(gate_run["study"] / "receipt.json"),
            "execution_finished_at": gate_run["execution"]["finished_at"],
            "all_learning_precedes_gate": True,
            "selection_inputs": "Historical acquisition/forgetting gate plus alternate own-binding acquisition and all new-task source gates; no repair outcomes.",
        }
        return {"run": gate_run, "gate": expected}

    def gradient_records(self, records, budget, anchor):
        equal(len(records), budget, "gradient_witness_step_count")
        equal(records, anchor[:budget], "independent_restart_gradient_record_prefix")
        for step, row in enumerate(records, 1):
            equal(row["step"], step, "gradient_witness_step")
            equal(
                row["optimizer_before"],
                [] if step == 1 else [step - 1],
                "actual_pre_step_clock_witness",
            )
            require(bool(row["tensors"]), "nonempty_gradient_witness")
            equal(
                digest(row["tensors"]), row["sha256"], "gradient_witness_tensor_digest"
            )

    def repair_unit(self, run, unit, stream, condition, start_sha, baseline_at=None):
        study, intent = run["study"], unit["intent"]
        base = study / "actions" / intent
        equal(unit["id"], stream["id"] + "/" + intent, "repair_unit_id")
        equal(unit["stream"], stream["id"], "repair_unit_stream")
        equal(unit["split"], stream["split"], "repair_unit_split")
        control = condition != "original_history"
        if control:
            equal(unit["condition"], condition, "repair_unit_condition")
        equal(set(unit["actions"]), {str(b) for b in BUDGETS}, "actual_budget_arm_set")
        baseline = unit["baseline"] if control else unit["baseline"]["after"]
        panels = {"0": baseline} | unit["actions"]
        panel_counts = {
            key: {
                kind: aggregate_counts(
                    self.score(
                        panel[kind],
                        stream["units"][intent]["test"]
                        if kind == "test"
                        else flatten(stream, "new", "test"),
                    )
                )
                for kind in ("test", "guard")
            }
            for key, panel in panels.items()
        }
        original_before_counts = None
        if not control:
            original_before_counts = {}
            for kind in ("test", "guard"):
                original_before_counts[kind] = aggregate_counts(
                    self.score(
                        unit["baseline"]["before"][kind],
                        stream["units"][intent]["test"]
                        if kind == "test"
                        else flatten(stream, "new", "test"),
                    )
                )
        rows, batches = repair_schedule(stream, intent)
        anchor = self.read(base / "16-updates.json")
        equal(anchor["rows"], rows, "repair16_original_support_rows")
        verify_trace(anchor["updates"], batches)
        gradient_anchor = self.read(base / "16-gradients.json") if control else None
        proof_rows, timings, saved_gradients = [], [], 0
        for budget in BUDGETS:
            action = unit["actions"][str(budget)]
            trace = self.read(base / f"{budget}-updates.json")
            equal(trace["rows"], rows, "repair_original_rows_all_histories")
            equal(
                trace["updates"],
                anchor["updates"][:budget],
                "actual_loss_gradient_norm_batch_prefix",
            )
            verify_trace(trace["updates"], batches[:budget])
            equal(
                self.read(base / f"{budget}-outcome.json"),
                action,
                "saved_action_matches_unit",
            )
            equal(action["updates"], budget, "repair_update_accounting")
            equal(action["old_exposures"], budget * 4, "old_repair_exposure_count")
            equal(action["new_exposures"], budget * 4, "new_repair_exposure_count")
            equal(
                action["start_adapter_sha256"],
                start_sha,
                "own_poststream_start_for_every_restart",
            )
            equal(
                action["schedule_sha256"],
                digest({"rows": rows, "batches": batches[:budget]}),
                "actual_original_repair_schedule",
            )
            equal(
                action["trace_sha256"], digest(trace["updates"]), "repair_trace_digest"
            )
            equal(action["fresh_optimizer"], True, "reported_fresh_repair_optimizer")
            equal(action["reload_exact"], True, "reported_repair_reload_parity")
            if control:
                equal(
                    action["optimizer_steps"], [budget], "reported_final_repair_clock"
                )
                equal(
                    action["input_schedule_sha256"],
                    digest(input_schedule(rows, batches[:budget])),
                    "repair_input_uid_schedule",
                )
                gradients = self.read(base / f"{budget}-gradients.json")
                self.gradient_records(gradients, budget, gradient_anchor)
                equal(
                    action["gradients_sha256"],
                    digest(gradients),
                    "action_gradient_records_digest",
                )
                timing = self.read(base / f"{budget}-timing.json")
                before(
                    baseline_at,
                    timing["started_at"],
                    "all_original_code_baselines_before_any_repair",
                )
                before(
                    timing["started_at"],
                    timing["finished_at"],
                    "repair_arm_start_finish",
                )
                timings.append({"budget": budget, **timing})
            for stop in [b for b in BUDGETS if b <= budget]:
                path = (
                    base / str(budget) / f"prefix-{stop}" / "adapter_model.safetensors"
                )
                anchor_path = (
                    base / "16" / f"prefix-{stop}" / "adapter_model.safetensors"
                )
                equal(
                    self.pin(path),
                    self.pin(anchor_path),
                    "actual_independent_restart_prefix_weight_bytes",
                )
                adapter = self.tensors(path)
                proof = {
                    "adapter_sha256": self.pin(path)["sha256"],
                    "trace_sha256": digest(trace["updates"][:stop]),
                }
                if control:
                    gradient_path = base / str(budget) / f"gradient-{stop}.safetensors"
                    actual = self.tensors(gradient_path)
                    equal(
                        actual["metadata"],
                        gradients[stop - 1]["tensors"],
                        "saved_postclip_gradient_tensor_bytes",
                    )
                    shape_mapping = {
                        key.replace(".lora_A.default.weight", ".lora_A.weight").replace(
                            ".lora_B.default.weight", ".lora_B.weight"
                        ): value["shape"]
                        for key, value in actual["metadata"].items()
                    }
                    equal(
                        shape_mapping,
                        {
                            key: value["shape"]
                            for key, value in adapter["metadata"].items()
                        },
                        "gradient_witness_covers_every_adapter_parameter",
                    )
                    norm = trace["updates"][stop - 1]["gradient_norm"]
                    expected_postclip = norm * min(1.0, 1.0 / (norm + 1e-6))
                    require(
                        abs(actual["l2"] - expected_postclip)
                        <= 2e-5 * max(1.0, expected_postclip),
                        "saved_postclip_gradient_norm",
                        {"actual": actual["l2"], "expected": expected_postclip},
                    )
                    proof.update(
                        gradients_sha256=digest(gradients[:stop]),
                        gradient_tensors_sha256=digest(actual["metadata"]),
                        optimizer_steps=[stop],
                    )
                    equal(
                        action["prefix_proofs"][str(stop)],
                        proof,
                        "prefix_weight_gradient_clock_proof",
                    )
                    saved_gradients += 1
                proof_rows.append({"budget": budget, "prefix": stop, **proof})
            equal(
                action["adapter_sha256"],
                self.pin(
                    base
                    / str(budget)
                    / f"prefix-{budget}"
                    / "adapter_model.safetensors"
                )["sha256"],
                "final_action_checkpoint_binding",
            )
        if control:
            ordered = {row["budget"]: row for row in timings}
            for first, second in ((16, 2), (2, 4), (4, 8)):
                before(
                    ordered[first]["finished_at"],
                    ordered[second]["started_at"],
                    "predeclared_repair16_then_2_4_8_execution_order",
                )
        values = {
            str(b): outcome(
                panel_counts[str(b)],
                panel_counts["0"],
                b,
                self.protocol["repair"]["qualification"],
            )
            for b in (0, *BUDGETS)
        }
        return {
            "id": unit["id"],
            "intent": intent,
            "stream": stream["id"],
            "split": stream["split"],
            "condition": condition,
            "counts": panel_counts,
            "historical_before_counts": original_before_counts,
            "curve": values,
            **budget_label(values),
            "proof": {
                "repair_updates": 30,
                "old_exposures": 120,
                "new_exposures": 120,
                "prefix_checkpoints": len(proof_rows),
                "independent_prefix_equalities": 6,
                "saved_gradient_tensor_sets_verified": saved_gradients,
                "step_gradient_hash_records": 30 if control else 0,
                "prefixes": proof_rows,
                "baseline_before_updates_at": baseline_at,
                "timings": timings,
                "cross_history_numerical_weights_or_gradients_equality_required": False,
            },
        }

    def repairs(self, name, condition, historical, controls, common):
        control = condition != "original_history"
        run_id = (
            PREFIX
            + (
                "history-repair-" + condition.replace("_", "-") + "-"
                if control
                else "predict-repair-"
            )
            + name.removeprefix("fresh-")
        )
        run = self.load_run(run_id, HISTORY_SOURCE if control else ORIGINAL_SOURCE)
        if run is None or historical is None or common is None:
            return None
        stream, study, receipt = (
            self.data["streams"][name],
            run["study"],
            run["receipt"],
        )
        units = self.read(study / "units.json")
        common_intents = common["gate"]["streams"][name]["common"]
        expected_intents = (
            common_intents if control else historical["summary"]["eligible"]
        )
        equal(
            [unit["intent"] for unit in units],
            expected_intents,
            "repair_eligible_order",
        )
        equal(
            receipt["unit_ids"],
            [unit["id"] for unit in units],
            "repair_receipt_unit_ids",
        )
        equal(receipt["repair_updates"], len(units) * 30, "repair_grid_total_updates")
        equal(receipt["lens_updates"], 0, "no_repair_stage_lens_updates")
        baseline_at = None
        source = controls[name][condition]["run"] if control else historical["run"]
        phase = "post_stream" if control else "forgotten"
        start_sha = self.pin(source["study"] / phase / "adapter_model.safetensors")[
            "sha256"
        ]
        equal(
            self.read(study / "frozen-backbone.json"),
            self.read(source["study"] / "frozen-backbone.json"),
            "repair_backbone_matches_own_learning_source",
        )
        if control:
            require(common["gate"]["qualified"], "no_repairs_after_failed_common_gate")
            equal(
                receipt["predictor_fits"], 0, "controls_do_not_fit_original_predictor"
            )
            equal(
                receipt["common_gate_sha256"],
                self.pin(common["run"]["study"] / "receipt.json")["sha256"],
                "repair_common_gate_receipt",
            )
            equal(
                receipt["control_learning_sha256"],
                self.pin(source["study"] / "receipt.json")["sha256"],
                "repair_own_source_receipt",
            )
            before(
                common["run"]["execution"]["finished_at"],
                run["execution"]["started_at"],
                "common_gate_completed_before_repair_execution",
            )
            eligibility = self.read(study / "eligibility-before-updates.json")
            equal(
                eligibility["gate"],
                common["gate"],
                "repair_preupdate_eligibility_exact",
            )
            equal(
                eligibility["gate_sha256"],
                self.pin(common["run"]["study"] / "receipt.json")["sha256"],
                "repair_preupdate_eligibility_receipt",
            )
            prerequisites = self.read(study / "repair-prerequisites.json")
            equal(prerequisites["common"], common_intents, "preupdate_common_intents")
            equal(
                prerequisites["prediction_inputs_changed"],
                False,
                "no_predictor_input_changes_reported",
            )
            historical_repair = self.load_run(
                PREFIX + "predict-repair-" + name.removeprefix("fresh-"),
                ORIGINAL_SOURCE,
            )
            require(
                historical_repair is not None,
                "historical_repair_collected_before_control_audit",
            )
            equal(
                receipt["historical_repair_sha256"],
                self.pin(historical_repair["study"] / "receipt.json")["sha256"],
                "original_repair_dependency_pin",
            )
            equal(
                prerequisites["historical_repair_sha256"],
                receipt["historical_repair_sha256"],
                "preupdate_historical_dependency_pin",
            )
            before(
                historical_repair["execution"]["finished_at"],
                run["execution"]["started_at"],
                "historical_repair_before_control",
            )
            frozen = self.read(study / "baselines-before-updates.json")
            baseline_at = frozen["at"]
            equal(
                frozen["units"],
                [unit | {"actions": {}} for unit in units],
                "all_control_baselines_frozen_before_any_action",
            )
            equal(
                self.pin(study / "baselines-before-updates.json")["sha256"],
                receipt["baselines_sha256"],
                "frozen_control_baseline_file_sha",
            )
            before(
                eligibility["at"],
                prerequisites["at"],
                "eligibility_before_repair_prerequisites",
            )
            before(
                prerequisites["at"],
                baseline_at,
                "all_prerequisites_before_baseline_and_actions",
            )
        else:
            frozen = self.read(study / "features-frozen.json")
            equal(
                frozen,
                [unit | {"actions": {}} for unit in units],
                "historical_features_and_baselines_frozen",
            )
            equal(
                self.pin(study / "features-frozen.json")["sha256"],
                receipt["features_sha256"],
                "historical_frozen_feature_file_sha",
            )
        if stream["split"] == "test":
            forecast = self.load_run(
                self.protocol["test_ordering"]["require_forecast"], ORIGINAL_SOURCE
            )
            require(forecast is not None, "completed_original_test_forecast_available")
            before(
                forecast["execution"]["finished_at"],
                run["execution"]["started_at"],
                "original_forecast_complete_before_test_repair",
            )
            equal(
                receipt["forecast_sha256" if control else "forecasts_sha256"],
                self.pin(forecast["study"] / "receipt.json")["sha256"],
                "test_forecast_receipt_pin",
            )
        curves = []
        historical_outcomes = (
            {row["id"]: row for row in self.read(study / "outcomes.json")}
            if not control
            else {}
        )
        if not control:
            equal(
                set(historical_outcomes),
                {unit["id"] for unit in units},
                "historical_outcome_unit_set",
            )
        for unit in units:
            equal(
                self.read(study / "units" / (unit["intent"] + ".json")),
                unit,
                "unit_file_equals_grid_member",
            )
            result = self.repair_unit(
                run, unit, stream, condition, start_sha, baseline_at
            )
            if not control:
                expected_outcome = {
                    key: result[key]
                    for key in (
                        "id",
                        "intent",
                        "stream",
                        "split",
                        "minimum_budget",
                        "qualified_budgets",
                        "later_qualification_loss",
                    )
                }
                expected_outcome.update(
                    before=result["historical_before_counts"]["test"]["accuracy"],
                    after=result["counts"]["0"]["test"]["accuracy"],
                    actions={
                        str(b): {
                            "accuracy": result["curve"][str(b)]["accuracy"],
                            "gain": result["curve"][str(b)]["gain"],
                            "guard_accuracy": result["curve"][str(b)]["guard"],
                            "utility": result["curve"][str(b)]["utility"],
                            "recovered": result["curve"][str(b)]["qualified"],
                        }
                        for b in BUDGETS
                    },
                )
                self.approx_tree(
                    historical_outcomes[unit["id"]],
                    expected_outcome,
                    "historical_saved_outcome_recomputed",
                )
            if unit["intent"] in common_intents:
                curves.append(result)
        equal(
            [unit["intent"] for unit in curves],
            common_intents,
            "paired_common_set_after_raw_scoring",
        )
        return {
            "run_id": run_id,
            "receipt": self.pin(study / "receipt.json"),
            "execution_finished_at": run["execution"]["finished_at"],
            "scientific_status": receipt["status"],
            "qualified": receipt["qualified"],
            "units": curves,
        }

    def paired(self, grids):
        per_source = {}
        for name, conditions in grids.items():
            good = {
                condition: value
                for condition, value in conditions.items()
                if value is not None
            }
            curves = {condition: value["units"] for condition, value in good.items()}
            means = {
                condition: {
                    str(b): {
                        key: math.fsum(unit["curve"][str(b)][key] for unit in units)
                        / len(units)
                        for key in METRICS
                    }
                    for b in (0, *BUDGETS)
                }
                for condition, units in curves.items()
                if units
            }
            pairs = {}
            if "original_history" in curves:
                for condition in self.protocol["conditions"]:
                    if condition not in curves:
                        continue
                    pairs[condition] = {}
                    for budget in (0, *BUDGETS):
                        key = str(budget)
                        rows = []
                        for historical, control in zip(
                            curves["original_history"], curves[condition], strict=True
                        ):
                            equal(
                                historical["id"],
                                control["id"],
                                "paired_curve_unit_identity",
                            )
                            left, right = (
                                historical["curve"][key],
                                control["curve"][key],
                            )
                            rows.append(
                                {
                                    "id": historical["id"],
                                    "excess": {
                                        metric: float(left[metric])
                                        - float(right[metric])
                                        for metric in METRICS
                                    },
                                    "only_historical_qualified": left["qualified"]
                                    and not right["qualified"],
                                    "only_control_qualified": right["qualified"]
                                    and not left["qualified"],
                                }
                            )
                        pairs[condition][key] = {
                            "pairs": rows,
                            "mean_excess": {
                                metric: math.fsum(row["excess"][metric] for row in rows)
                                / len(rows)
                                for metric in METRICS
                            },
                            "only_historical_qualified": sum(
                                row["only_historical_qualified"] for row in rows
                            ),
                            "only_control_qualified": sum(
                                row["only_control_qualified"] for row in rows
                            ),
                        }
            per_source[name] = {
                "split": self.data["streams"][name]["split"],
                "available_conditions": list(good),
                "means": means,
                "paired_historical_excess": pairs,
                "minimum_budget_counts": {
                    condition: {
                        label: Counter(unit["minimum_budget"] for unit in units)[label]
                        for label in ("2", "4", "8", "16", "never")
                    }
                    for condition, units in curves.items()
                },
                "later_qualification_losses": {
                    condition: [
                        unit["id"] for unit in units if unit["later_qualification_loss"]
                    ]
                    for condition, units in curves.items()
                },
            }
        aggregate = {}
        for split in ("train", "test"):
            names = [
                name
                for name in per_source
                if self.data["streams"][name]["split"] == split
            ]
            aggregate[split] = {"source_clusters_required": 2, "conditions": {}}
            for condition in ("original_history", *self.protocol["conditions"]):
                available = [
                    name for name in names if grids[name][condition] is not None
                ]
                missing = [name for name in names if name not in available]
                result = {
                    "available_sources": available,
                    "missing_sources": missing,
                    "complete_split": not missing,
                }
                if available:
                    for method in ("equal_source", "pooled_intents_descriptive"):
                        weights = {
                            name: 1
                            if method == "equal_source"
                            else len(grids[name][condition]["units"])
                            for name in available
                        }
                        result[method] = {
                            str(b): {
                                metric: math.fsum(
                                    weights[name]
                                    * per_source[name]["means"][condition][str(b)][
                                        metric
                                    ]
                                    for name in available
                                )
                                / sum(weights.values())
                                for metric in METRICS
                            }
                            for b in (0, *BUDGETS)
                        }
                aggregate[split]["conditions"][condition] = result
            aggregate[split]["paired_available_subset"] = {}
            for condition in self.protocol["conditions"]:
                available = [
                    name
                    for name in names
                    if condition in per_source[name]["paired_historical_excess"]
                ]
                comparison = {
                    "available_sources": available,
                    "missing_sources": [
                        name for name in names if name not in available
                    ],
                    "complete_split": len(available) == len(names),
                }
                if available:
                    for method in ("equal_source", "pooled_intents_descriptive"):
                        weights = {
                            name: 1
                            if method == "equal_source"
                            else len(grids[name][condition]["units"])
                            for name in available
                        }
                        comparison[method] = {
                            str(b): {
                                metric: math.fsum(
                                    weights[name]
                                    * per_source[name]["paired_historical_excess"][
                                        condition
                                    ][str(b)]["mean_excess"][metric]
                                    for name in available
                                )
                                / sum(weights.values())
                                for metric in METRICS
                            }
                            for b in (0, *BUDGETS)
                        }
                    comparison["discordant_pairs"] = {
                        str(b): {
                            key: sum(
                                per_source[name]["paired_historical_excess"][condition][
                                    str(b)
                                ][key]
                                for name in available
                            )
                            for key in (
                                "only_historical_qualified",
                                "only_control_qualified",
                            )
                        }
                        for b in (0, *BUDGETS)
                    }
                aggregate[split]["paired_available_subset"][condition] = comparison
        complete_aggregates = {}
        for split, groups in aggregate.items():
            if not all(
                value["complete_split"] for value in groups["conditions"].values()
            ):
                continue
            names = [
                name
                for name in per_source
                if self.data["streams"][name]["split"] == split
            ]
            complete_aggregates[split] = {
                "source_clusters": len(names),
                "intents": sum(
                    len(grids[name]["original_history"]["units"]) for name in names
                ),
            }
            for method in ("equal_source", "pooled_intents_descriptive"):
                means = {
                    condition: value[method]
                    for condition, value in groups["conditions"].items()
                }
                complete_aggregates[split][method] = {
                    "means": means,
                    "paired_historical_excess": {
                        condition: {
                            str(b): {
                                metric: means["original_history"][str(b)][metric]
                                - means[condition][str(b)][metric]
                                for metric in METRICS
                            }
                            for b in (0, *BUDGETS)
                        }
                        for condition in self.protocol["conditions"]
                    },
                }
        return {
            "per_source": per_source,
            "by_split": aggregate,
            "complete_split_aggregates": complete_aggregates,
            "interpretation": {
                "original_vs_new_only": "Same initialization and new data; original history has 128 additional old updates, extra text exposure and older optimizer moments. This comparison alone does not isolate a specific old binding.",
                "original_vs_alternate": "Matches old/new input schedules, update budget and optimizer age; changes old intent-to-code assignments. A difference can also reflect interference from conflicting learned bindings.",
                "control_qualification": "Original-code target acquisition under the same thresholds; not recovery of an original binding that a control never learned.",
                "dependency_unit": "Two independent source models per split; intents are paired observations within a model. Pooled intent means are descriptive. No significance tests or intent bootstrap.",
                "limits": "Behavioral training-history effects only; no circuit localization, physical-memory proof, semantic erasure, free-generation result or universal recoverability claim.",
            },
        }

    def analysis_receipt(self, grids, paired):
        run = self.load_run(PREFIX + "history-analyze", HISTORY_SOURCE)
        if run is None:
            return {"status": "not_locally_collected_at_snapshot"}
        missing = [
            {"stream": name, "condition": condition}
            for name, conditions in grids.items()
            for condition, value in conditions.items()
            if value is None
        ]
        if missing:
            return {
                "status": "collected_analysis_pending_local_repair_verification",
                "receipt": self.pin(run["study"] / "receipt.json"),
                "missing_or_unverified_repair_grids": missing,
                "scope": "The terminal analysis was collected before its large dependencies. This is a local collection boundary, not a scientific integrity discrepancy.",
            }
        saved = self.read(run["study"] / "analysis.json")
        for conditions in grids.values():
            for grid in conditions.values():
                before(
                    grid["execution_finished_at"],
                    run["execution"]["started_at"],
                    "all_repair_executions_complete_before_analysis",
                )
        equal(
            saved["common_eligibility"],
            self.report["common_gate"]["recomputed"],
            "analysis_exact_common_gate",
        )
        equal(
            self.pin(run["study"] / "analysis.json")["sha256"],
            run["receipt"]["analysis_sha256"],
            "analysis_file_receipt_sha",
        )
        for name, source in paired["per_source"].items():
            for condition, grid in grids[name].items():
                expected = [
                    {
                        key: unit[key]
                        for key in (
                            "id",
                            "intent",
                            "stream",
                            "split",
                            "curve",
                            "minimum_budget",
                            "qualified_budgets",
                            "later_qualification_loss",
                        )
                    }
                    for unit in grid["units"]
                ]
                self.approx_tree(
                    saved["per_source"][name]["curves"][condition],
                    expected,
                    "saved_analysis_unit_curves",
                )
            for key in ("means", "paired_historical_excess", "minimum_budget_counts"):
                self.approx_tree(
                    saved["per_source"][name][key], source[key], "saved_analysis_" + key
                )
        self.approx_tree(
            saved["aggregate"],
            paired["complete_split_aggregates"],
            "saved_analysis_aggregate",
        )
        return {
            "status": "verified_collected_analysis",
            "receipt": self.pin(run["study"] / "receipt.json"),
            "per_unit_curves_and_paired_results_recomputed": True,
        }

    def approx_tree(self, actual, expected, label):
        if isinstance(expected, dict):
            equal(set(actual), set(expected), label + ":keys")
            for key, value in expected.items():
                self.approx_tree(actual[key], value, label + ":" + str(key))
        elif isinstance(expected, list):
            equal(len(actual), len(expected), label + ":length")
            for index, value in enumerate(expected):
                self.approx_tree(actual[index], value, label + ":" + str(index))
        elif isinstance(expected, float):
            require(
                math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                label,
                {"actual": actual, "expected": expected},
            )
        else:
            equal(actual, expected, label)

    def run(self):
        if (
            self.attempt("source_documents", self.documents) is None
            and "sources" not in self.report
        ):
            return self.finish()
        ids = [
            *self.interface["jobs"],
            self.protocol["original_cohort"]["run_id"],
            self.protocol["test_ordering"]["require_forecast"],
        ]
        ids.extend(
            rule["run_id"] for rule in self.protocol["learning_sources"].values()
        )
        ids.extend(
            PREFIX + "predict-repair-" + name.removeprefix("fresh-")
            for name in self.data["streams"]
        )
        self.snapshot(ids)
        histories, controls = {}, {}
        for name in self.data["streams"]:
            histories[name] = self.attempt(
                "historical_learning:" + name,
                lambda name=name: self.historical_learning(name),
            )
            controls[name] = {
                condition: self.attempt(
                    "control_learning:" + name + ":" + condition,
                    lambda name=name, condition=condition: self.control_learning(
                        name, condition, histories[name]
                    ),
                )
                for condition in self.protocol["conditions"]
            }
        self.report["historical_learning"] = {
            name: value["summary"] if value else {"status": "unavailable_or_unverified"}
            for name, value in histories.items()
        }
        self.report["control_learning"] = {
            name: {
                condition: value["summary"]
                if value
                else {"status": "unavailable_or_unverified"}
                for condition, value in conditions.items()
            }
            for name, conditions in controls.items()
        }
        common = self.attempt("common_gate", lambda: self.common(histories, controls))
        grids = {
            name: {
                condition: self.attempt(
                    "repair:" + name + ":" + condition,
                    lambda name=name, condition=condition: self.repairs(
                        name, condition, histories[name], controls, common
                    ),
                )
                for condition in ("original_history", *self.protocol["conditions"])
            }
            for name in self.data["streams"]
        }
        self.report["repair_grids"] = grids
        paired = self.attempt("paired_analysis", lambda: self.paired(grids))
        self.report["paired_analysis"] = paired
        self.report["collected_analysis"] = self.attempt(
            "analysis_receipt", lambda: self.analysis_receipt(grids, paired)
        )
        qa = []
        for path in sorted(self.wave.glob("recovery-history-runtime*.json")):
            value = self.read(path)
            qa.append(
                {
                    "path": str(path),
                    **self.pin(path),
                    "at": value.get("at"),
                    "recorded_status_counts": dict(
                        Counter(row["status"] for row in value.get("reports", []))
                    ),
                    "scope": "Parent runtime snapshot as collected, not a current provider-state query or a substitute for terminal scientific receipts.",
                }
            )
        self.report["runtime_qa_as_collected"] = qa
        return self.finish()

    def finish(self):
        changed = []
        for path, (identity, _) in self.hash_cache.items():
            st = path.stat()
            if (st.st_size, st.st_mtime_ns, st.st_ino) != identity:
                changed.append(str(path))
        self.report["snapshot"] = {
            "pending": self.pending,
            "changed_during_audit": changed,
            "runs": {
                name: {
                    "collection": value["collection_pin"],
                    "snapshot_at": value["snapshot_at"],
                    "execution_status": value["execution_status"],
                    "manifest_files": len(value["collection"]["files"]),
                    "provenance_files_verified": name in self.loaded,
                }
                for name, value in self.snapshots.items()
            },
            "policy": "Only terminal locally collected run directories enter each invocation's frozen snapshot. cl-collect temporary directories and later collections are excluded. A missing collection is pending, not byte corruption.",
        }
        self.report["proof_boundaries"] = {
            "scoring": "Native argmax among the same 16 trained output codes; code logits, targets, input UIDs, counts and threshold arithmetic recomputed independently. Not full-vocabulary generation.",
            "arithmetic": "Gates use integer correct/total counts and exact rational differences. Saved ancillary logit statistics allow FP32 error 3e-5 + 3e-6*abs(value).",
            "learning_optimizer": "Saved initial/acquired/final optimizer clocks, full moments, parameter IDs/settings and old-second-moment decay bounds are checked. Archived source uses one optimizer for old and both new segments. Midpoint clocks are receipt-only; learning gradients and the immediately-before-first-new moment state are not saved, so full numerical recurrence and dynamic object identity are not independently reproducible.",
            "repair_optimizer": "Control per-step pre-hook clocks and gradient hashes plus saved postclip gradients at 2/4/8/16 and actual prefix weight equality are checked. No repair optimizer state files or full intervening gradient tensor sequence are stored; a full Adam replay is unavailable. Historical repairs retain weights/traces and reported fresh starts, but no gradient witness files.",
            "backbone": "Local frozen-backbone hash manifests are compared exactly with each original learner; base tensors are not reloaded or model outputs regenerated by this CPU audit. Model-file pins belong to the frozen portfolio provenance audit.",
            "timing": "Execution/receipt/action timestamps and archived operation order establish the recorded prerequisite sequence; local copied-file mtimes are not treated as experiment timestamps.",
            "independence": "No experiment modules are imported. Scientific scoring, thresholds, schedules, moment constraints and paired statistics are implemented here. No scientific artifacts or previous audit snapshots are changed.",
        }
        self.report["verification"] = {
            "raw_native_records_checked": self.raw_records,
            "raw_native_panels_checked": self.raw_panels,
            "record_count_scope": "All scored panel entries, including repeated guard panels; not unique utterances or independent samples.",
            "maximum_saved_statistic_absolute_error": dict(self.native_errors),
            "files_hashed_once_this_invocation": len(self.hash_cache),
            "bytes_stream_hashed": sum(
                pin["bytes"] for _, pin in self.hash_cache.values()
            ),
            "hash_policy": "8 MiB streaming file hashes; per-path metadata cache for this invocation; tensor decoding deduplicated by verified file SHA256.",
        }
        self.report["issues"] = self.issues
        learners = [
            value
            for conditions in self.report.get("control_learning", {}).values()
            for value in conditions.values()
            if "qualified" in value
        ]
        grids = self.report.get("repair_grids", {})
        verified_controls = [
            grid
            for conditions in grids.values()
            for condition, grid in conditions.items()
            if condition != "original_history" and grid is not None
        ]
        units = [unit for grid in verified_controls for unit in grid["units"]]
        early_pairs = []
        for name, conditions in grids.items():
            historical = conditions.get("original_history")
            if historical is None:
                continue
            for condition, control in conditions.items():
                if condition == "original_history" or control is None:
                    continue
                panel = {
                    "stream": name,
                    "control_condition": condition,
                    "common_intents": len(control["units"]),
                }
                for label, grid in (
                    ("original_history", historical),
                    ("control", control),
                ):
                    values = grid["units"]
                    panel[label] = {
                        "target_correct": sum(
                            unit["counts"]["2"]["test"]["correct"] for unit in values
                        ),
                        "target_total": sum(
                            unit["counts"]["2"]["test"]["total"] for unit in values
                        ),
                        "jointly_qualified": sum(
                            unit["curve"]["2"]["qualified"] for unit in values
                        ),
                        "mean_new_guard_accuracy": math.fsum(
                            unit["curve"]["2"]["guard"] for unit in values
                        )
                        / len(values),
                    }
                early_pairs.append(panel)
        self.report["summary"] = {
            "verified_control_learners": len(learners),
            "qualified_control_learners": sum(value["qualified"] for value in learners),
            "alternate_own_bindings_acquired": sum(
                len(value["alternate_acquired"]) for value in learners
            ),
            "new_bindings_acquired_across_controls": sum(
                len(value["new_acquired"]) for value in learners
            ),
            "common_cohort": self.report.get("common_gate", {})
            .get("recomputed", {})
            .get("counts"),
            "verified_control_repair_grids": len(verified_controls),
            "required_control_repair_grids": 8,
            "verified_control_repair_units": len(units),
            "control_prefix_weight_checkpoints": sum(
                unit["proof"]["prefix_checkpoints"] for unit in units
            ),
            "control_independent_prefix_equalities": sum(
                unit["proof"]["independent_prefix_equalities"] for unit in units
            ),
            "control_saved_gradient_tensor_sets": sum(
                unit["proof"]["saved_gradient_tensor_sets_verified"] for unit in units
            ),
            "control_per_step_gradient_and_clock_records": sum(
                unit["proof"]["step_gradient_hash_records"] for unit in units
            ),
            "two_update_raw_paired_results_available": early_pairs,
            "paired_result_boundary": "Available source pairs only. Target accuracy, gain and new-task guard are distinct outcomes; source-specific reversals and later qualification loss remain visible in full curves.",
        }
        complete = bool(
            (self.report.get("collected_analysis") or {}).get("status")
            == "verified_collected_analysis"
        )
        noncompleted = [
            name
            for name, value in self.snapshots.items()
            if value["execution_status"] != "completed"
        ]
        self.report["execution_failed_or_interrupted"] = noncompleted
        self.report["status"] = (
            "audit_errors"
            if self.issues
            else "snapshot_changed_requires_rerun"
            if changed
            else "complete_verified"
            if complete and not self.pending and not noncompleted
            else "collected_subset_verified_pending_remaining_results"
        )
        self.report["finished_at"] = now()
        return self.report


class AuditTests(unittest.TestCase):
    def test_analysis_collected_before_repairs_is_pending_not_corruption(self):
        auditor = Auditor(WAVE)
        run = {"study": Path("/not-read-by-this-test")}
        with (
            patch.object(auditor, "load_run", return_value=run),
            patch.object(
                auditor, "pin", return_value={"bytes": 1, "sha256": "test-receipt"}
            ),
        ):
            result = auditor.analysis_receipt({"stream": {"new_only": None}}, None)
        self.assertEqual(
            result["status"], "collected_analysis_pending_local_repair_verification"
        )
        self.assertEqual(
            result["missing_or_unverified_repair_grids"],
            [{"stream": "stream", "condition": "new_only"}],
        )
        self.assertEqual(auditor.issues, [])

    def test_exact_guard_and_gain_thresholds(self):
        base = {
            "test": {"correct": 2, "total": 20},
            "guard": {"correct": 151, "total": 160},
        }
        rule = {"accuracy_min": 0.8, "gain_min": 0.3, "guard_drop_max": 0.1}
        panel = {
            "test": {"correct": 16, "total": 20},
            "guard": {"correct": 135, "total": 160},
        }
        self.assertTrue(outcome(panel, base, 2, rule)["qualified"])
        panel["guard"]["correct"] = 134
        self.assertFalse(outcome(panel, base, 2, rule)["qualified"])

    def test_later_loss_not_monotone(self):
        values = {str(b): {"qualified": b in (2, 8)} for b in BUDGETS}
        self.assertEqual(
            budget_label(values),
            {
                "minimum_budget": "2",
                "qualified_budgets": [2, 8],
                "later_qualification_loss": True,
            },
        )
        self.assertEqual(
            budget_label({str(b): {"qualified": False} for b in BUDGETS})[
                "minimum_budget"
            ],
            "never",
        )

    def test_wrong_prediction_and_uid_rejected(self):
        codes = list(range(32, 48))
        row = {"intent": "a", "target": 32, "text_sha256": "uid"}
        logits = [0.0] * 16
        output = {
            "code_logits": logits,
            "prediction": 32,
            "target_logp": -math.log(16),
            "target_margin": 0.0,
            "top_probability": 1 / 16,
            "top_margin": 0.0,
            "entropy": math.log(16),
        }
        record = row | {"output": output}
        counts, _ = native_scores([record], [row], codes)
        self.assertEqual(counts["a"]["correct"], 1)
        with self.assertRaisesRegex(AuditFailure, "prediction_recomputed"):
            native_scores(
                [record | {"output": output | {"prediction": 33}}], [row], codes
            )
        with self.assertRaisesRegex(AuditFailure, "evaluation_identity"):
            native_scores([record | {"text_sha256": "wrong"}], [row], codes)

    def test_relabel_preserves_inputs_but_rebalancing_changes_schedule(self):
        stream = {
            "old": list("abcdefgh"),
            "units": {
                name: {"learn": [{"code": i}]} for i, name in enumerate("abcdefgh")
            },
        }
        _, mapping = alternate(stream)
        rows = [
            {
                "intent": name,
                "code": i,
                "target": 32 + i,
                "text_sha256": name,
                "source_split": "train",
                "source_index": i,
                "input_ids": [i],
            }
            for i, name in enumerate("abcdefgh")
        ]
        rotated = relabeled(rows, mapping, list(range(32, 48)))
        batches = learning_batches(rows, 3, 8, 99)
        self.assertEqual(
            input_schedule(rows, batches), input_schedule(rotated, batches)
        )
        self.assertNotEqual(batches, learning_batches(rotated, 3, 8, 99))
        self.assertTrue(
            all(x["target"] != y["target"] for x, y in zip(rows, rotated, strict=True))
        )

    def test_moment_reset_with_forged_clock_fails_decay_bound(self):
        old = torch.tensor([1.0, 2.0])
        carried = old * (0.999**128) + 0.01
        self.assertEqual(second_moment_tail(old, carried, 0.999, 128)["violations"], 0)
        forged = {
            "state": {0: {"step": torch.tensor(256.0), "exp_avg_sq": torch.zeros(2)}}
        }
        self.assertEqual(optimizer_clock(forged), [256])
        self.assertEqual(
            second_moment_tail(old, forged["state"][0]["exp_avg_sq"], 0.999, 128)[
                "violations"
            ],
            2,
        )

    def test_fractional_clock_and_prefix_change_rejected(self):
        with self.assertRaisesRegex(AuditFailure, "finite_integer_optimizer"):
            optimizer_clock({"state": {0: {"step": torch.tensor(128.5)}}})
        trace = [{"step": 1, "rows": [0, 1], "loss": 1.0, "gradient_norm": 0.2}]
        with self.assertRaisesRegex(AuditFailure, "actual_batch_indices"):
            verify_trace(trace, [[1, 0]])
        tensors = {
            "parameter": {"shape": [2], "dtype": "torch.float32", "sha256": "pinned"}
        }
        anchor = [
            {
                "step": step,
                "optimizer_before": [] if step == 1 else [step - 1],
                "tensors": tensors,
                "sha256": digest(tensors),
            }
            for step in (1, 2)
        ]
        auditor = Auditor(WAVE)
        auditor.gradient_records(anchor, 2, anchor)
        altered = [anchor[0], anchor[1] | {"optimizer_before": []}]
        with self.assertRaisesRegex(AuditFailure, "pre_step_clock"):
            auditor.gradient_records(altered, 2, altered)
        altered = [anchor[0], anchor[1] | {"sha256": "wrong-gradient-digest"}]
        with self.assertRaisesRegex(AuditFailure, "tensor_digest"):
            auditor.gradient_records(altered, 2, altered)
        with self.assertRaisesRegex(AuditFailure, "record_prefix"):
            auditor.gradient_records(altered, 2, anchor)

    def test_duplicate_optimizer_construction_rejected(self):
        source = "def learn_control():\n    optimizer = observer.optimizer()\n    observer.update(rows, old, optimizer)\n    observer.update(rows, first, optimizer)\n    observer.update(rows, second, optimizer)\n"
        reset = "def reset(self, checkpoint):\n    self.model.zero_grad()\n"
        self.assertEqual(
            source_optimizer_flow(source, reset)["optimizer_constructions"], 1
        )
        with self.assertRaisesRegex(
            AuditFailure, "single_learning_optimizer_assignment"
        ):
            source_optimizer_flow(
                source + "    optimizer = observer.optimizer()\n", reset
            )

    def test_equal_source_and_pooled_pairing_differ(self):
        auditor = Auditor(WAVE)
        auditor.protocol = {"conditions": ["new_only", "alternate_binding"]}
        auditor.data = {
            "streams": {
                name: {"split": split}
                for name, split in (
                    ("a", "train"),
                    ("b", "train"),
                    ("c", "test"),
                    ("d", "test"),
                )
            }
        }
        grids = {}
        for name in auditor.data["streams"]:
            count, score = (1, 1.0) if name in ("a", "c") else (3, 0.5)
            grids[name] = {}
            for condition in ("original_history", "new_only", "alternate_binding"):
                value = score if condition == "original_history" else 0.0
                curve = {
                    str(b): {metric: value for metric in METRICS} for b in (0, *BUDGETS)
                }
                for row in curve.values():
                    row.update(
                        qualified=False, accuracy_passed=False, guard_passed=True
                    )
                units = [
                    {
                        "id": name + str(i),
                        "curve": curve,
                        "minimum_budget": "never",
                        "later_qualification_loss": False,
                    }
                    for i in range(count)
                ]
                grids[name][condition] = {"units": units}
        paired = auditor.paired(grids)
        train = paired["complete_split_aggregates"]["train"]
        self.assertEqual(
            train["equal_source"]["means"]["original_history"]["2"]["accuracy"], 0.75
        )
        self.assertEqual(
            train["pooled_intents_descriptive"]["means"]["original_history"]["2"][
                "accuracy"
            ],
            0.625,
        )
        self.assertEqual(
            train["equal_source"]["paired_historical_excess"]["new_only"]["2"][
                "accuracy"
            ],
            0.75,
        )
        grids["b"]["new_only"] = None
        partial = auditor.paired(grids)
        self.assertNotIn("train", partial["complete_split_aggregates"])
        self.assertEqual(
            partial["by_split"]["train"]["paired_available_subset"]["new_only"][
                "available_sources"
            ],
            ["a"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = unittest.TextTestRunner(verbosity=2 if args.self_test else 1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    )
    if args.self_test or not result.wasSuccessful():
        return 0 if result.wasSuccessful() else 1
    auditor = Auditor(WAVE)
    report = auditor.run()
    report["self_tests"] = {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
    }
    output = Path(__file__).with_suffix(".json")
    report["auditor"] = {
        "path": str(Path(__file__).resolve()),
        **auditor.pin(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "no_experiment_imports": True,
    }
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "issues": len(report["issues"]),
                "pending": len(report["snapshot"]["pending"]),
                "raw_records": report["verification"]["raw_native_records_checked"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 1 if report["issues"] or report["snapshot"]["changed_during_audit"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
