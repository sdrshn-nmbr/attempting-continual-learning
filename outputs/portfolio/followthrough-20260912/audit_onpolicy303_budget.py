import argparse
import ast
import copy
import hashlib
import io
import json
import math
import random
import tarfile
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import audit_onpolicy303 as original_audit
import torch
from safetensors.torch import load_file

METHODS = ("cached_teacher_sft", "cached_teacher_kl")
FAMILIES = ("sequence_a", "sequence_b", "sequence_c")
REMOTE_RUNS = Path("/mnt/shared/cl-portfolio/runs")
PROTOCOL_PATH = "configs/onpolicy303-budget-protocol.json"
CONTRACT = "onpolicy303_cached_controls_384_plus_21_post_observation"
BUDGET_HANDOFF_SHA256 = "1fbde94e44d4ff99fd4b81cef1e53d614095ab94a9735f1bfbf1df858232c333"
BUDGET_PROTOCOL_SHA256 = "97300f7fd7fda6f209f829708995accaea34381faef29f603ac25e028416dc92"
RECOVERY_HANDOFF_SHA256 = "323a4030abb91bdcd41c60f91fadea183505362d7098ae08166ce484819ddb6a"
RECOVERY_CLOSURE_SHA256 = "6e852da54e364c55c28f5108a454e1ad175e3d6ae2e242a1485ee79994436f69"
RECOVERY_PROTOCOL_PATH = "configs/onpolicy303-budget-evaluate-protocol.json"
RECOVERY_COPIES = {
    "original_training.json": "study/training.json",
    "parameter_mapping.json": "study/parameter_mapping.json",
    "checkpoint405/train_probes.json": "study/checkpoint405/train_probes.json",
    "checkpoint405/retention.json": "study/checkpoint405/retention.json",
    "checkpoint405/learner/adapter_config.json": "study/checkpoint405/learner/adapter_config.json",
    "checkpoint405/learner/adapter_model.safetensors": "study/checkpoint405/learner/adapter_model.safetensors",
}
REFERENCE_REPORT = "onpolicy303-independent-audit/final-audit-20260912T074811993684Z.json"
REFERENCE_HASHES = {
    "audit_onpolicy303.py": "3a1109bd272d51ab309b80c663cbaaf42a414c94c8049635a6619f81c74b17a2",
    "audit_consolidation.py": "915b4b707311e13adac56b8fc2018fd46d5bdf090e3645a111b02089be87fdf6",
    REFERENCE_REPORT: "43867c405c8276c0daec0c047bc039c31f8246f69a35bad787036cebffa96d8e",
}
BOUNDARY = (
    "Post-observation control using the same previously observed TEST192, unused864 and generic128. "
    "Each cached control receives 405 updates, 1620 row exposures and 11340 loss tokens; own-prefix "
    "KL remains at 384 updates, 1536 rows and 11314 tokens. The controls have 21 more updates, "
    "84 more row exposures and 26 more loss tokens than own-prefix KL. This is not an exact "
    "token, update or compute match and does not uniquely identify a causal objective or prefix effect. "
    "One learner seed and one existing Sequence303 mapping; fixed qualified expert hybrid, not a "
    "full SDFT/SDPO reproduction. Generic option-likelihood scores are separate from native generation."
)
PROOF_LIMIT = (
    "Independent local hashes, saved tensors, Adam states, ledgers and token decoding are checked. "
    "Per-step gradients and full vocabulary logits are not archived, so the auditor cannot independently "
    "recompute GPU updates, full KL distributions or SFT losses. Frozen executable source and bound "
    "runtime assertions support restoration, teacher detachment and base/expert immutability. "
    "The recovery child is bound by distinct supervisor task/attempt, nonce, parent chain, receipts and "
    "the archived launch code, not numeric PID inequality alone; its exact argv is not recorded. "
    "Its Python file-open guard is not an OS syscall sandbox."
)


def require(condition, detail):
    if not condition:
        raise ValueError(f"ONPOLICY303_BUDGET_INDEPENDENT_AUDIT: {detail}")


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def bound_files(root, hashes):
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"relative artifact path {name}")
        path = root / relative
        require(path.is_file() and not path.is_symlink() and file_hash(path) == expected, f"artifact bytes {path}")


def collection_audit(root, expected_status="completed"):
    receipt = read(root / "collection.json")
    require(receipt["execution_status"] == expected_status, f"collection execution state {root.name}")
    bound_files(root, {name: spec["sha256"] for name, spec in receipt["files"].items()})
    for name, spec in receipt["files"].items():
        require((root / name).stat().st_size == spec["bytes"], f"collected byte count {name}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} - {"collection.json"}
    require(actual == set(receipt["files"]), f"complete local collection closure {root.name}")
    return {"files": len(actual), "bytes": sum(s["bytes"] for s in receipt["files"].values()),
            "collection_sha256": file_hash(root / "collection.json")}


def execution_identity_audit(root):
    execution, task, config = [read(root / name) for name in ("execution.json", "task.json", "config.json")]
    require(execution["task"] == task and execution["task_id"] == task["id"] == root.name,
            "supervisor task identity")
    require(task["config"] == config and execution["source_sha256"] == task["source_sha256"],
            "supervisor config/source identity")
    require(bool(execution["attempt_id"]), "execution attempt identity")
    for name, value in (("task", task), ("config", config)):
        expected = hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()
        require(execution[name + "_sha256"] == expected, f"supervisor {name} byte hash")
    return execution, task, config


def execution_audit(root):
    execution, task, config = execution_identity_audit(root)
    require(execution["status"] == "completed" and execution["exit_code"] == 0
            and execution["timed_out"] is False, f"completed execution {root.name}")
    return execution, task, config


def training_envelope_audit(root):
    execution, task, config = execution_identity_audit(root)
    if execution["status"] == "completed":
        require(execution["exit_code"] == 0 and execution["timed_out"] is False, "successful training envelope")
        return execution, task, config, None
    require(execution["status"] == "failed" and execution["exit_code"] == 1
            and execution["timed_out"] is False, "only specific terminal post-training failure accepted")
    output = root / "study"
    failure = read(output / "failure.json")
    training = read(output / "training.json")
    require(failure == {"exception": "ValueError", "detail": "ONPOLICY303_BUDGET release all training models before fresh evaluator",
                        "pid": training["pid"]}, "exact post-training allocation guard failure")
    require(training["status"] == "trained_evaluation_pending" and training["updates_this_run"] == 21
            and training["selected_checkpoint"] == 405, "complete training receipt before teardown failure")
    require(not any((output / name).exists() for name in ("evaluation_launch.json", "evaluation_process.json", "evaluation", "result.json")),
            "failed before any evaluator launch")
    return execution, task, config, {
        "status": "failed_after_saved_training_before_evaluation", "failure": failure,
        "failure_sha256": file_hash(output / "failure.json"), "execution_sha256": file_hash(root / "execution.json"),
        "evaluation_started": False, "scientific_failure_inferred": False,
        "remaining_allocation_bytes_or_object_identity_recorded": False,
        "interpretation": "The original job failed its zero-allocated-memory guard after saving training. Completed training must pass artifact/ledger checks independently. No heldout prediction or capability failure is inferred; a separately authorized zero-update evaluation can consume the immutable405 checkpoint.",
    }


def archive_audit(wave, identity):
    path = wave / "code" / (identity + ".tar")
    metadata = read(path.with_suffix(".json"))
    require(file_hash(path) == metadata["archive"]["sha256"]
            and path.stat().st_size == metadata["archive"]["bytes"], "collected source archive bytes")
    checksum, files, payloads = hashlib.sha256(), {}, {}
    with tarfile.open(path) as stream:
        for member in sorted(stream.getmembers(), key=lambda row: Path(row.name)):
            relative = Path(member.name)
            require(member.isfile() and not relative.is_absolute() and ".." not in relative.parts
                    and member.name not in files, "source archive regular unique relative members")
            payload = stream.extractfile(member).read()
            checksum.update(member.name.encode() + b"\0" + payload)
            files[member.name] = hashlib.sha256(payload).hexdigest()
            payloads[member.name] = payload
    require(checksum.hexdigest() == identity == metadata["source_sha256"], "actual source closure digest")
    require(len(files) == metadata["files"], "source archive member count")
    return {"source_sha256": identity, "archive_sha256": file_hash(path), "files_sha256": files}, payloads


def check_execution_snapshot(snapshot, execution, task, source, completed):
    for key, expected in {
        "task_id": execution["task_id"], "attempt_id": execution["attempt_id"],
        "task_sha256": execution["task_sha256"], "config_sha256": execution["config_sha256"],
        "supervisor_child_pid": execution["pid"], "code_dir": task["code_dir"],
        "entrypoint": task["entrypoint"], "config": task["config"],
        "status": "completed" if completed else "running", "source": source,
    }.items():
        require(snapshot[key] == expected, f"embedded execution {key}")


def independent_schedule(rows, order_seed):
    batches = []
    for epoch in range(5):
        order = list(range(len(rows)))
        random.Random(order_seed + epoch).shuffle(order)
        batches.extend(order[offset:offset + 4] for offset in range(0, len(order), 4))
    return batches[:405]


def schedule_audit(design, ctx):
    rows = ctx["corpus"]["train"]
    batches = independent_schedule(rows, ctx["design"]["training"]["order_seed"])
    ids = [[rows[index]["id"] for index in batch] for batch in batches]
    require(ids[:384] == ctx["schedule"], "independently reconstructed original384 schedule")
    expected = {
        "mode": "next_epoch_prefix", "order_seed": 91230431, "continuation_epoch_seed": 91230435,
        "original_prefix_equal": True, "original_batch_ids_sha256": digest(ids[:384]),
        "full405_batch_ids_sha256": digest(ids), "continuation_indices": batches[384:],
        "continuation_batch_ids": ids[384:], "continuation_batch_ids_sha256": digest(ids[384:]),
        "repeated_original_prefix": False,
    }
    require(all(design["schedule"][key] == value for key, value in expected.items()), "sealed next-epoch suffix")
    suffix = [index for batch in batches[384:] for index in batch]
    require(len(suffix) == len(set(suffix)) == 84 and len(batches) == 405, "84 distinct TRAIN rows in suffix")
    return batches, {**expected, "additional_exposures_per_family": dict(Counter(rows[i]["task"] for i in suffix)),
                     "unique_suffix_training_rows": 84, "total_unique_training_rows": 384,
                     "total_loss_tokens": 11340, "additional_loss_tokens": 588,
                     "original_prefix_exposures_per_row": 4, "suffix_row_exposures": 1}


def tensor_hash(tensors):
    checksum = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        checksum.update(name.encode())
        checksum.update(str((tuple(value.shape), value.dtype)).encode())
        checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def adapter_mapping(tensors):
    expected = []
    for layer in range(36):
        for projection, width in (("q_proj", 4096), ("v_proj", 1024)):
            for factor, shape in (("A", [8, 4096]), ("B", [width, 8])):
                key = f"base_model.model.model.layers.{layer}.self_attn.{projection}.lora_{factor}.weight"
                require(key in tensors, f"missing q/v rank8 factor {key}")
                tensor = tensors[key]
                require(list(tensor.shape) == shape and tensor.dtype == torch.float32
                        and bool(torch.isfinite(tensor).all()), f"factor shape/precision/finite {key}")
                expected.append({"optimizer_id": len(expected), "adapter_key": key,
                                 "parameter_name": key.removesuffix(".weight") + ".learner.weight",
                                 "shape": shape, "dtype": "torch.float32", "numel": tensor.numel()})
    require(set(tensors) == {row["adapter_key"] for row in expected}, "only q/v rank8 factors, no offset")
    require(sum(t.numel() for t in tensors.values()) == 3833856, "fixed adapter parameter count")
    return expected


def adapter_audit(folder, spec):
    bound_files(folder, spec["files"])
    config = read(folder / "adapter_config.json")
    require(config["r"] == 8 and config["lora_alpha"] == 16 and config["lora_dropout"] == 0
            and set(config["target_modules"]) == {"q_proj", "v_proj"}
            and config["bias"] == "none" and not config["use_dora"], "original adapter parameterization")
    tensors = load_file(str(folder / "adapter_model.safetensors"), device="cpu")
    mapping = adapter_mapping(tensors)
    checksum = tensor_hash(tensors)
    require(checksum == spec["tensor_sha256"], "actual adapter tensor hash")
    return tensors, mapping, {"tensor_sha256": checksum, "rank": 8, "parameters": 3833856,
                              "tensor_payload_bytes": 15335424, "dense_offset": None,
                              "files_sha256": spec["files"]}


def optimizer_audit(state, mapping, clock, options):
    require(set(state) == {"state", "param_groups"} and len(state["param_groups"]) == 1, "single AdamW group")
    group = state["param_groups"][0]
    actual_options = normalized({key: value for key, value in group.items() if key != "params"})
    require(actual_options == options, "unchanged original AdamW options")
    ids = list(range(len(mapping)))
    require(group["params"] == ids and set(state["state"]) == set(ids)
            and [row["optimizer_id"] for row in mapping] == ids, "ordered Adam parameter IDs")
    require(len({row["parameter_name"] for row in mapping}) == len(ids), "unique parameter ordering")
    for index, expected in enumerate(mapping):
        slot = state["state"][index]
        require(set(slot) == {"step", "exp_avg", "exp_avg_sq"}, f"Adam state slots {index}")
        step = slot["step"]
        require(step.numel() == 1 and step.dtype == torch.float32 and float(step) == clock, f"Adam actual clock {index}")
        for name in ("exp_avg", "exp_avg_sq"):
            value = slot[name]
            require(list(value.shape) == expected["shape"] and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()), f"Adam finite moment shape {index}/{name}")
        require(bool((slot["exp_avg_sq"] >= 0).all()), f"Adam nonnegative second moment {index}")
    checksum = tensor_hash({f"{index}/{key}": value for index, slot in state["state"].items() for key, value in slot.items()})
    return {"clock": clock, "parameter_states": len(ids), "parameter_order_sha256": digest(mapping),
            "tensor_sha256": checksum, "options": actual_options, "all_moments_finite": True}


def cached_trajectory_audit(ctx, records, ledger, events, batches, method):
    require(method in METHODS and len(records) == 84 and len(ledger) == 21, "fixed cached suffix dimensions")
    updates = [row for row in events if row["event"] == "onpolicy303_budget_update"]
    require(len(updates) == 21, "21 actual update events")
    rows, canonical = ctx["corpus"]["train"], ctx["canonical"]["train"]["generated"]
    tokens, families, losses, teachers_correct = 0, Counter(), [], 0
    for number, record in enumerate(records):
        offset, microstep = divmod(number, 4)
        step = offset + 385
        index = batches[step - 1][microstep]
        row = rows[index]
        require(record["step"] == step and record["microstep"] == microstep
                and record["row_index"] == index and record["id"] == row["id"]
                and record["task"] == row["task"] and record["row_sha256"] == digest(row), "suffix trajectory order/row")
        require(record["method"] == method and record["sampling"] is None
                and record["student_optimizer_clock_before_update"] == step - 1, "cached trajectory update clock")
        generated, diagnostic = record["generation"], record["diagnostics"]
        native = {"id": row["id"], "task": row["task"], "group": row["group"],
                  "row_sha256": digest(row), "generation": generated}
        original_audit.verify_generations(ctx, [row], [native])
        target = canonical[index]["generation"]["token_ids"]
        require(len(target) == 7 and target[-1] == 151645 and 151645 not in target[:-1], "seven native target tokens")
        require(generated["token_ids"] == target and record["canonical_teacher_response_sha256"] == digest(target),
                "verbatim original qualified teacher target")
        require(diagnostic["prediction_mask"] == [False] * (len(generated["prompt_token_ids"]) - 1) + [True] * 7,
                "response-only prediction mask including EOS")
        require(diagnostic["loss_tokens"] == 7 and diagnostic["loss_token_ids"] == target
                and diagnostic["vocabulary_size"] == ctx["vocabulary_size"], "full vocabulary token accounting")
        require(math.isfinite(diagnostic["row_loss"]) and diagnostic["row_loss"] >= -1e-5, "finite nonnegative row loss")
        require(len(diagnostic["student_argmax_token_ids"]) == 7
                and all(0 <= t < ctx["vocabulary_size"] for t in diagnostic["student_argmax_token_ids"]), "student per-token predictions")
        if method == "cached_teacher_kl":
            fields = ("teacher_argmax_token_ids", "student_sampled_token_logp", "teacher_sampled_token_logp",
                      "teacher_entropy", "teacher_eos_probability", "forward_kl_per_token")
            require(all(len(diagnostic[key]) == 7 for key in fields), "seven KL diagnostics per row")
            for key in fields[1:]:
                require(all(math.isfinite(v) for v in diagnostic[key]), f"finite KL diagnostic {key}")
            require(all(value <= 1e-5 for key in fields[1:3] for value in diagnostic[key]), "valid token log probabilities")
            require(all(-1e-5 <= h <= math.log(ctx["vocabulary_size"]) + 1e-5 for h in diagnostic["teacher_entropy"]), "teacher entropy range")
            require(all(0 <= p <= 1 for p in diagnostic["teacher_eos_probability"]), "teacher EOS probability range")
            require(all(k >= -1e-5 for k in diagnostic["forward_kl_per_token"]), "nonnegative forward KL terms")
            require(math.isclose(sum(diagnostic["forward_kl_per_token"]) / 7, diagnostic["row_loss"], abs_tol=2e-5, rel_tol=2e-5),
                    "forward KL token mean matches row loss")
            matches = [a == b for a, b in zip(diagnostic["teacher_argmax_token_ids"], target, strict=True)]
            require(diagnostic["prefix_matches_qualified_answer"] == [True] * 7
                    and diagnostic["qualified_next_token_ids"] == target
                    and diagnostic["teacher_next_token_matches_on_qualified_prefix"] == matches
                    and diagnostic["off_answer_prefix_competence_certified"] is False, "qualified prefix diagnostics")
            teachers_correct += sum(matches)
        else:
            require(diagnostic["teacher_forward_performed"] is False and "forward_kl_per_token" not in diagnostic,
                    "SFT has no teacher forward or KL objective")
        tokens += 7
        families[row["task"]] += 1
        losses.append(diagnostic["row_loss"])
    for offset, (entry, event) in enumerate(zip(ledger, updates, strict=True)):
        step = offset + 385
        block = records[offset * 4:(offset + 1) * 4]
        require(entry["step"] == step and entry["indices"] == batches[step - 1]
                and entry["ids"] == [r["id"] for r in block], "suffix ledger row order")
        require(entry["trajectory_sha256"] == [digest(row) for row in block], "raw trajectory hashes bound to ledger")
        require(entry["loss_tokens"] == 28 and entry["optimizer_clocks"] == {
            "minimum": step, "maximum": step, "parameters_with_state": 144}, "actual per-update clocks and tokens")
        require(math.isfinite(entry["gradient_norm"]) and entry["gradient_norm"] >= 0, "finite gradient norm")
        require(math.isclose(entry["loss"], sum(losses[offset * 4:(offset + 1) * 4]) / 4, rel_tol=1e-6, abs_tol=1e-6),
                "four-row mean update loss")
        require(all(event[key] == value for key, value in entry.items()), "event/ledger exact identity")
    return {"actual_updates": 21, "optimizer_start": 384, "optimizer_final": 405,
            "trajectory_rows": 84, "loss_tokens": tokens, "exposures_per_family": dict(families),
            "loss_tokens_per_family": {task: 7 * count for task, count in families.items()},
            "loss_reduction": "mean of four response-token means, all seven response tokens including native EOS",
            "objective": "forward KL teacher||student, full vocabulary, fixed detached expert" if method.endswith("kl") else "cached teacher answer SFT",
            "teacher_next_tokens_correct_on_qualified_prefix": teachers_correct if method.endswith("kl") else None,
            "cached_target_accuracy_is_not_student_generated_accuracy": True}


def paired_native(rows, before, after):
    counts = Counter(dict.fromkeys(("before_correct", "after_correct", "gained", "lost", "still_wrong", "both_correct", "changed_answers"), 0))
    changed, remaining_wrong, per_family = [], [], {}
    fields = ("token_ids", "raw_text", "body_text", "terminated", "format_valid", "correct", "digit_position_correct")
    for row, left, right in zip(rows, before, after, strict=True):
        require(row["id"] == left["id"] == right["id"], "paired native row IDs")
        a, b = left["generation"], right["generation"]
        transition = "both_correct" if a["correct"] and b["correct"] else "gained" if b["correct"] else "lost" if a["correct"] else "still_wrong"
        difference = a["token_ids"] != b["token_ids"]
        cell = per_family.setdefault(row["task"], Counter())
        for key, value in (("count", 1), ("before_correct", int(a["correct"])), ("after_correct", int(b["correct"])),
                           (transition, 1), ("changed_answers", int(difference))):
            counts[key] += value
            cell[key] += value
        record = {"id": row["id"], "task": row["task"], "input": row["group"], "row_sha256": digest(row),
                  "gold": row["choices"][row["gold_idx"]], "transition": transition,
                  "before": {key: a[key] for key in fields}, "after": {key: b[key] for key in fields}}
        if difference:
            changed.append(record)
        if not b["correct"]:
            remaining_wrong.append(record)
    counts["net_gain"] = counts["gained"] - counts["lost"]
    require(counts["after_correct"] - counts["before_correct"] == counts["net_gain"], "paired count conservation")
    return {"counts": dict(counts), "per_family": {key: dict(value) for key, value in per_family.items()},
            "every_changed_native_answer": changed, "every_remaining_wrong_answer": remaining_wrong,
            "change_definition": "different complete emitted token sequence; also records changes with unchanged correctness"}


def guard_audit(guard):
    require(guard["devices"] == ["cuda:0"] and guard["dtype"] == "float32" and guard["finite"] is True
            and guard["visible_rocm_gpus"] == 1 and guard["offload_hooks"] is False, "qualified FP32 one-GPU path")
    mapping = guard["hf_device_map"]
    require(mapping is None or all(str(value) in {"0", "cuda", "cuda:0"} for value in mapping.values()), "optional device map")


def runtime_audit(runtime, previous):
    for key in ("base_files", "base_tensor_sha256", "dtype", "autocast", "packages", "hip"):
        require(runtime[key] == previous[key], f"unchanged original runtime {key}")
    require(runtime["dtype"] == "float32" and runtime["autocast"] is False and bool(runtime["hip"]), "native FP32 ROCm runtime")


def reported_native(audited, reported):
    require(set(audited["families"]) == set(reported), "reported native families")
    for task, cell in audited["families"].items():
        require(reported[task]["count"] == cell["count"]
                and reported[task]["correct"] == cell["correct"] / cell["count"], f"reported native score {task}")


def fresh_child_audit(training, result, launch, process, training_hash, config_path, output_path, code_dir):
    require(launch["training_pid"] == process["training_pid"] == result["training_pid"] == training["pid"], "training process chain")
    require(launch["training_sha256"] == result["training_sha256"] == training_hash, "fresh child exact training receipt")
    require(result["child_token"] == launch["child_token"] and len(launch["child_token"]) == 32, "fresh child launch nonce")
    require(result["parent_pid"] == process["uv_launcher_pid"]
            and len({result["pid"], process["uv_launcher_pid"], training["pid"]}) == 3, "fresh child parent chain")
    require(process["returncode"] == 0 and result["status"] == "completed"
            and process["completed_unix_ns"] >= launch["started_unix_ns"], "completed child launch")
    command = launch["command"]
    require(command[:4] == ["uv", "run", "--no-project", "--python"] and Path(command[4]).is_absolute(), "explicit Python uv launcher")
    require(command[5:] == ["python", "-B", str(Path(code_dir) / "onpolicy303_budget.py"),
                           "--config", str(config_path), "--output-dir", str(output_path), "--evaluate-child"], "actual standalone evaluator CLI")
    require(launch["method"] == result["method"] == training["method"], "child method identity")
    require(result["teacher_model_loaded"] is False and result["active_adapters"] == ["learner"]
            and result["learner_rank"] == 8 and result["optimizer_updates"] == 0
            and result["train24_reload_parity"] is True and result["generic128_reload_parity"] is True, "teacher-absent frozen fresh learner")
    return {"training_pid": training["pid"], "uv_launcher_pid": process["uv_launcher_pid"],
            "evaluation_pid": result["pid"], "evaluation_parent_pid": result["parent_pid"],
            "child_token": launch["child_token"], "training_sha256": training_hash,
            "standalone_cli": command, "separate_execution_task": False,
            "fresh_child_under_same_supervisor_task": True, "numeric_pid_inequality_is_not_sole_proof": True}


def reference_audit(repo, wave):
    bound_files(wave, REFERENCE_HASHES)
    for name in ("audit_onpolicy303.py", "audit_consolidation.py", Path(__file__).name):
        tree = ast.parse((wave / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                require(node.module is not None and not node.module.startswith("experiments"), "no experiment imports")
                require(node.module not in {"onpolicy303", "onpolicy303_contract", "onpolicy303_budget", "onpolicy303_budget_contract",
                                            "choice_contract", "choice_consolidation", "generation_teacher", "generation_consolidation"},
                        "independent audit import closure")
            elif isinstance(node, ast.Import):
                require(all(not item.name.startswith(("experiments", "onpolicy303")) for item in node.names), "no experiment imports")
    ctx = original_audit.context(repo, wave)
    repeated = original_audit.final_audit(ctx, wave)
    previous = read(wave / REFERENCE_REPORT)
    for key, value in repeated.items():
        require(value == previous[key], f"frozen original six-run audit recheck {key}")
    require(ctx["vocabulary_size"] == 151936, "original full model vocabulary")
    panels = {}
    for method in (*METHODS, "onpolicy_kl"):
        task_id = "followthrough-20260912-onpolicy303-" + method.replace("_", "-") + "-evaluate"
        panels[method] = read(wave / "runs" / task_id / "panels.json")
    return ctx, panels, {
        "status": "original_six_runs_reaudited_exactly_equal_frozen_report",
        "frozen_hashes": REFERENCE_HASHES, "source_sha256": original_audit.SOURCE,
        "training": repeated["training"], "evaluation": repeated["evaluation"],
    }


def design_audit(repo, wave, ctx, design):
    lane = repo / "experiments/distillation"
    require(file_hash(lane / "onpolicy303_budget_handoff.json") == BUDGET_HANDOFF_SHA256
            and digest(design) == BUDGET_PROTOCOL_SHA256, "parent-confirmed budget handoff/protocol")
    handoff = read(lane / "onpolicy303_budget_handoff.json")
    closure = handoff["hash_closure"]
    bound_files(lane, closure["files_sha256"])
    require(len(closure["files_sha256"]) == closure["files"]
            and digest(closure["files_sha256"]) == closure["files_mapping_sha256"], "sealed handoff closure")
    require(handoff["protocol"]["payload_sha256"] == digest(design)
            and handoff["protocol"]["file_sha256"] == file_hash(lane / PROTOCOL_PATH), "sealed handoff protocol")
    require(design["contract"] == CONTRACT and set(design["methods"]) == set(METHODS), "only two cached continuation methods")
    bound_files(lane, design["scientific_files_sha256"])
    for name, checksum in ctx["handoff"]["scientific_closure_sha256"].items():
        if name in design["scientific_files_sha256"]:
            require(design["scientific_files_sha256"][name] == checksum, f"frozen scientific dependency {name}")
    original = design["original_design"]
    require(original["path"] == "configs/onpolicy303_protocol.json"
            and original["file_sha256"] == file_hash(lane / original["path"])
            and original["payload_sha256"] == original_audit.PROTOCOL, "exact original protocol identity")
    require(design["training"] == ctx["design"]["training"] and design["evaluation"] == ctx["design"]["evaluation"],
            "unchanged original optimizer/evaluation recipe")
    require(design["runtime_versions"] == handoff["runtime_versions"], "sealed exact module/distribution versions")
    require((design["start_update"], design["additional_updates"], design["final_update"]) == (384, 21, 405), "fixed continuation budget")
    require(design["tokens"] == {"cached_at384": 10752, "additional": 588, "cached_at405": 11340,
                                  "ownprefix_at384": 11314, "excess_over_ownprefix": 26}, "conservative token budget")
    require(design["native_export"] == {"method": "onpolicy_kl", "checkpoint": 384, "selection_changed_by_control": False}, "no export selection change")
    require(set(design["sources"]) == set(METHODS), "no own-prefix continuation")
    batches, schedule = schedule_audit(design, ctx)
    sources = {}
    for method, descriptor in design["sources"].items():
        root = wave / "runs" / descriptor["task_id"]
        expected_id = "followthrough-20260912-onpolicy303-" + method.replace("_", "-") + "-train"
        require(descriptor["task_id"] == expected_id and descriptor["run_dir"] == str(REMOTE_RUNS / expected_id)
                and descriptor["method"] == method and descriptor["source_archive_sha256"] == original_audit.SOURCE, "original source path/method")
        bound_files(root, descriptor["files_sha256"])
        training = read(root / "training.json")
        require(training["arm"]["checkpoints"]["384"] == descriptor["checkpoint"], "original checkpoint384 descriptor")
        tensors, mapping, proof = adapter_audit(root / method / "checkpoint384/learner", descriptor["checkpoint"]["adapter"])
        require(mapping == descriptor["parameter_mapping"], "actual original parameter ordering")
        state = torch.load(root / method / "checkpoint384/optimizer.pt", map_location="cpu", weights_only=True)
        optimizer = optimizer_audit(state, mapping, 384, descriptor["optimizer_options"])
        require(optimizer["tensor_sha256"] == descriptor["optimizer_tensor_sha256"], "original actual Adam moments hash")
        require(optimizer["options"]["lr"] == 0.0003 and optimizer["options"]["weight_decay"] == 0
                and optimizer["options"]["betas"] == [0.9, 0.999] and optimizer["options"]["eps"] == 1e-8, "original Adam recipe")
        sources[method] = {"root": root, "training": training, "adapter_tensors": tensors,
                           "adapter": proof, "optimizer": optimizer, "mapping": mapping}
    return batches, schedule, sources


def dispatch_audit(ctx, wave, design):
    path = wave / "onpolicy303-budget-dispatch.json"
    dispatch = read(path)
    handoff = read(ctx["lane"] / "onpolicy303_budget_handoff.json")
    require(dispatch["handoff_sha256"] == BUDGET_HANDOFF_SHA256
            and dispatch["protocol_sha256"] == BUDGET_PROTOCOL_SHA256
            and dispatch["scientific_files_mapping_sha256"] == handoff["hash_closure"]["files_mapping_sha256"],
            "parent dispatch binds sealed handoff, protocol and closure")
    require(dispatch["tokens"] == design["tokens"] and dispatch["frozen_files_verified"] == 24
            and dispatch["parent_cpu_tests_passed"] == 32, "parent sealed preparation evidence")
    jobs = {}
    for spec in dispatch["jobs"]:
        require(spec["method"] in METHODS and spec["method"] not in jobs, "exact two distinct dispatched cached methods")
        manifest = Path(spec["manifest"])
        require(file_hash(manifest) == spec["manifest_sha256"], "actual dispatch manifest bytes")
        task = read(manifest)
        config = read(ctx["lane"] / ("configs/onpolicy303-budget-" + spec["method"].replace("_", "-") + ".json"))
        require(task["id"] == spec["id"] == config["task_id"] and task["config"] == config
                and task["source_sha256"] == dispatch["source_sha256"]
                and task["depends_on"] == spec["depends_on"], "dispatched task source/config/dependencies")
        require(task["entrypoint"] == "onpolicy303_budget.py" and task["gpus"] == 1, "dispatched bounded entrypoint")
        jobs[spec["method"]] = {"task": task, "manifest_sha256": spec["manifest_sha256"]}
    require(set(jobs) == set(METHODS), "both cached controls dispatched")
    return {"dispatch_file_sha256": file_hash(path), "source_sha256": dispatch["source_sha256"],
            "handoff_sha256": dispatch["handoff_sha256"], "protocol_sha256": dispatch["protocol_sha256"],
            "jobs": jobs, "status_boundary": "Manifest and source identity binding only; actual completion requires collected supervisor and child receipts."}


def generation_events(events, condition, records):
    selected = [event for event in events if event.get("condition") == condition]
    require(len(selected) == len(records), f"generation event count {condition}")
    require(all(event["id"] == row["id"] and event["correct"] == row["generation"]["correct"]
                for event, row in zip(selected, records, strict=True)), f"generation event/raw panel identity {condition}")


def training_audit(ctx, root, design, source, execution, task, archive, batches):
    output = root / "study"
    training = read(output / "training.json")
    method = training["method"]
    require(training["contract"] == CONTRACT and training["status"] == "trained_evaluation_pending"
            and method == task["config"]["method"], "continuation training identity")
    require(training["design_sha256"] == digest(design) and training["source_sha256"] == design["scientific_files_sha256"], "training design/source")
    require(training["dataset"] == source["training"]["dataset"] == design["dataset"], "unchanged actual dataset")
    require(training["original_source"] == design["sources"][method], "original artifact descriptor retained")
    check_execution_snapshot(training["execution"], execution, task, archive, completed=False)
    old_execution, old_task, _ = execution_audit(source["root"])
    check_execution_snapshot(training["original_execution"], old_execution, old_task, source["training"]["execution"]["source"], completed=True)
    require(training["original_execution"]["execution_sha256"] == file_hash(source["root"] / "execution.json"), "completed original execution bytes")
    require(execution["attempt_id"] != old_execution["attempt_id"] and execution["task_id"] != old_execution["task_id"], "distinct continuation execution")
    bound_files(output, training["files_sha256"])
    require(read(output / "parameter_mapping.json") == source["mapping"], "actual saved parameter ordering")
    require(read(output / "optimizer_restore.json") == training["optimizer_restored"] == source["optimizer"], "exact384 Adam restore proof")
    require(training["runtime_versions"] == read(output / "runtime_versions.json") == design["runtime_versions"],
            "actual exact module and distribution versions before training")
    require(training["native384_reload_parity"] == {"train24_exact_native_parity": True, "generic128_exact_score_parity": True},
            "recorded384 reload prerequisite identity")
    require(read(output / "schedule.json") == design["schedule"], "actual continuation schedule")
    require(file_hash(output / "teacher_train_cache.json") == file_hash(source["root"] / "teacher_train_cache.json"), "exact original teacher cache bytes")
    cache = read(output / "teacher_train_cache.json")
    require(cache["split"] == "train" and len(cache["rows"]) == 384 and cache["rows_sha256"] == digest(ctx["corpus"]["train"]), "TRAIN-only cache population")
    for row, cached, canonical in zip(ctx["corpus"]["train"], cache["rows"], ctx["canonical"]["train"]["generated"], strict=True):
        require(cached["row_sha256"] == digest(row) and cached["id"] == row["id"]
                and cached["response_token_ids"] == canonical["generation"]["token_ids"]
                and len(cached["response_token_ids"]) == 7, "same seven-token qualified target for every TRAIN row")
    probes = [r for task_name in FAMILIES for r in [row for row in ctx["corpus"]["train"] if row["task"] == task_name][:8]]
    restored = read(output / "restored384_train_probes.json")
    require(restored == read(source["root"] / method / "checkpoint384/train_probes.json"), "actual384 native probe reload")
    old_generic = read(output / "restored384_retention.json")
    require(old_generic == read(source["root"] / "final_retention.json"), "actual384 generic score-vector reload")
    restored_native = original_audit.verify_generations(ctx, probes, restored)
    restored_generic = original_audit.retention_audit(ctx, old_generic)
    prerequisites = read(output / "prerequisites.json")
    require(prerequisites["passed"] is True and prerequisites["updates_this_run"] == 0
            and prerequisites["optimizer_clock"] == 384 and prerequisites["train24_exact_native_parity"] is True
            and prerequisites["generic128_exact_score_parity"] is True
            and prerequisites["source_execution"] == training["original_execution"]
            and prerequisites["schedule"] == design["schedule"], "pre-update prerequisites")
    events = jsonl(output / "events.jsonl")
    require(all(event["pid"] == training["pid"] for event in events), "training event process identity")
    first = next(i for i, event in enumerate(events) if event["event"] == "onpolicy303_budget_update")
    before = events[:first]
    gates = [event for event in before if event["event"] == "budget_prerequisites_passed"]
    require(len(gates) == 1 and gates[0]["updates_this_run"] == 0 and gates[0]["optimizer_clock"] == 384, "eligibility before first new update")
    generation_events(before, "restored384/train_only", restored)
    require(not any(event.get("condition", "").endswith(("/test", "/unused288")) for event in events), "no heldout generation during continuation training")
    teacher = {"reuses_original_qualified_cache": True, "fresh_qualification_this_run": method == "cached_teacher_kl"}
    require(training["teacher_loaded_in_training"] == training["teacher_fresh_qualified_before_student_load"]
            == prerequisites["teacher_fresh_before_student_load"] == (method == "cached_teacher_kl"), "actual teacher residency by objective")
    if method == "cached_teacher_kl":
        certificate = read(output / "teacher_fresh_qualification.json")
        require(certificate == read(source["root"] / "teacher_fresh_qualification.json"), "same fresh teacher qualification certificate")
        teacher["panels"] = {}
        for split in ("train", "validation"):
            records = read(output / f"teacher_fresh_{split}.json")
            audit = original_audit.verify_generations(ctx, ctx["corpus"][split], records)
            require(audit["correct"] == audit["count"] and records == read(source["root"] / f"teacher_fresh_{split}.json"), "actual complete fresh teacher native qualification")
            generation_events(before, "teacher_fresh/" + split, records)
            teacher["panels"][split] = audit
        qualified = [event for event in before if event["event"] == "budget_teacher_qualified_before_student_load"]
        require(len(qualified) == 1 and qualified[0]["learner_loaded"] is False and qualified[0]["teacher_updates"] == 0, "fresh teacher before learner load")
    ledger, trajectories = read(output / "ledger.json"), jsonl(output / "trajectories.jsonl")
    trajectory = cached_trajectory_audit(ctx, trajectories, ledger, events, batches, method)
    expected_tokens = {
        "original_updates": 384, "additional_updates": 21, "total_updates": 405,
        "additional_row_exposures": 84, "total_row_exposures": 1620, "unique_training_rows": 384,
        "original_loss_tokens": 10752, "additional_loss_tokens": 588, "total_loss_tokens": 11340,
        "ownprefix384_loss_tokens": 11314, "excess_over_ownprefix": 26,
        "loss_tokens_by_continuation_update": [28] * 21, "exact_token_update_or_flop_match": False,
    }
    require(training["tokens"] == expected_tokens and training["updates_this_run"] == 21
            and training["selected_checkpoint"] == 405, "actual total exposure accounting")
    checkpoint = training["checkpoint405"]
    require(checkpoint["path"] == str(REMOTE_RUNS / root.name / "study/checkpoint405/learner"), "own final checkpoint path")
    tensors, mapping, adapter = adapter_audit(output / "checkpoint405/learner", checkpoint)
    require(mapping == source["mapping"], "unchanged405 parameter order")
    changed = {key: int(torch.count_nonzero(value != source["adapter_tensors"][key])) for key, value in tensors.items()}
    require(sum(changed.values()) > 0 and adapter["tensor_sha256"] != source["adapter"]["tensor_sha256"], "actual persistent405 weight change")
    state = torch.load(output / "checkpoint405/optimizer.pt", map_location="cpu", weights_only=True)
    optimizer = optimizer_audit(state, mapping, 405, source["optimizer"]["options"])
    require(optimizer == training["optimizer_final"] and optimizer["tensor_sha256"] != source["optimizer"]["tensor_sha256"], "actual405 Adam clocks/moments")
    native405 = read(output / "checkpoint405/train_probes.json")
    generic405 = read(output / "checkpoint405/retention.json")
    native = original_audit.verify_generations(ctx, probes, native405)
    generic = original_audit.retention_audit(ctx, generic405)
    reported_native(native, training["native405_train_summary"])
    require(generic["families"] == training["generic405_summary"], "training generic405 summary")
    generation_events(events, "final405/train_only", native405)
    guard_audit(training["model_guard"])
    runtime_audit(read(output / "runtime.json"), read(source["root"] / "runtime.json"))
    require(training["base_tensor_sha256"] == ctx["choice"]["source_base_tensor_sha256"]
            and training["teacher_tensor_sha256"] == ctx["teacher"]["teacher_tensor_sha256"]
            and training["teacher_optimizer_updates"] == 0, "fixed base/teacher identities")
    require(training["trainable_parameters"] == 3833856 and training["resident_learner_rank"] == 8, "only fixed LoRA8 capacity")
    require(training["heldout_predictions_observed_during_this_training"] is False
            and training["post_observation_control"] is True and training["heldouts_previously_observed"] is True
            and training["native_export"] == design["native_export"] and training["claim_boundary"] == design["claim_boundary"], "training claim boundary")
    return training, {"status": "verified", "training_sha256": file_hash(output / "training.json"),
                      "restored_adapter384": source["adapter"], "optimizer_restored384": source["optimizer"],
                      "final_adapter405": adapter, "optimizer_final405": optimizer,
                      "changed_adapter_elements_by_tensor": changed, "trajectory_audit": trajectory,
                      "tokens": expected_tokens, "teacher": teacher,
                      "native_reload384": restored_native, "generic_reload384": restored_generic,
                      "native_train405": native, "generic405": generic,
                      "fixed_weights": {"base_tensor_sha256": training["base_tensor_sha256"],
                                        "teacher_tensor_sha256": training["teacher_tensor_sha256"],
                                        "zero_teacher_updates": True, "runtime_assertions_bound_to_archived_source": True,
                                        "independent_live_memory_remeasurement": False}}


def read_ban_audit(ban, forbidden, allowed):
    require(ban["forbidden_roots"] == forbidden, "exact evaluator forbidden paths including duplicates")
    require(ban["allowed_model_roots"] == allowed, "exact ordered evaluator allowed paths")
    require(ban["denied_reads"] == [] and ban["installed_before_model_load"] is True,
            "evaluator read guard installed before loading without denied reads")
    for name in ban["observed_weight_paths"]:
        path = Path(name)
        require(any(path.is_relative_to(parent) for parent in map(Path, allowed))
                and not any(path.is_relative_to(parent) for parent in map(Path, forbidden)),
                "observed evaluator weight read")


def evaluation_behavior_audit(ctx, root, design, training, source):
    output, child = root / "study", root / "study/evaluation"
    result = read(child / "result.json")
    require(result["design_sha256"] == digest(design) and result["source_sha256"] == design["scientific_files_sha256"]
            and result["dataset"] == training["dataset"] and result["checkpoint405"] == training["checkpoint405"], "fresh evaluation identity")
    require(result["runtime_versions"] == design["runtime_versions"], "fresh child exact module/distribution versions")
    bound_files(child, result["files_sha256"])
    native_reload, generic = read(child / "train_probes.json"), read(child / "retention.json")
    require(native_reload == read(output / "checkpoint405/train_probes.json")
            and generic == read(output / "checkpoint405/retention.json"), "actual exact405 native/generic fresh reload")
    probes = [r for task in FAMILIES for r in [row for row in ctx["corpus"]["train"] if row["task"] == task][:8]]
    native_probe = original_audit.verify_generations(ctx, probes, native_reload)
    generic_audit = original_audit.retention_audit(ctx, generic)
    require(result["generic_retention"] == generic_audit["families"], "fresh generic128 reported scores")
    panels = read(child / "panels.json")
    require(set(panels) == {"test", "unused288"} and len(panels["test"]) == 192
            and len(panels["unused288"]) == 864, "fixed native heldout denominators")
    native = {}
    events = jsonl(child / "events.jsonl")
    require(all(event["pid"] == result["pid"] for event in events)
            and not any("update" in event["event"] for event in events), "fresh evaluator event PID and zero updates")
    generation_events(events, "reload405/train_only", native_reload)
    first = next(i for i, event in enumerate(events) if event.get("condition") == "final405/test")
    generation_events(events[:first], "reload405/train_only", native_reload)
    for split in ("test", "unused288"):
        native[split] = original_audit.verify_generations(ctx, ctx["corpus"][split], panels[split])
        reported_native(native[split], result["native"][split])
        generation_events(events, "final405/" + split, panels[split])
    guard_audit(result["model_guard"])
    runtime_audit(read(child / "runtime.json"), read(source["root"] / "runtime.json"))
    ban = result["read_ban"]
    expected_forbidden = design["evaluation_forbidden_paths"] + [str(REMOTE_RUNS / root.name / "study" / name) for name in
                         ("teacher_train_cache.json", "checkpoint405/optimizer.pt", "trajectories.jsonl", "ledger.json")]
    expected_forbidden += [str(REMOTE_RUNS / root.name / "study" / name) for name in training["files_sha256"] if name.startswith("teacher_")]
    base = ctx["choice"]["source_base"]
    allowed = [base["local_path"], training["checkpoint405"]["path"],
               *[str(Path(base["local_path"]) / spec["path"]) for spec in base["files"]]]
    read_ban_audit(ban, expected_forbidden, allowed)
    require(result["post_observation_control"] is True and result["heldouts_previously_observed"] is True
            and result["native_export"] == design["native_export"] and result["claim_boundary"] == design["claim_boundary"], "evaluation claim boundary")
    return {"status": "verified", "result_sha256": file_hash(child / "result.json"),
            "panels_sha256": file_hash(child / "panels.json"),
            "native": native, "native_train24_reload": native_probe, "generic_retention": generic_audit,
            "teacher_absent": True, "optimizer_updates": 0, "read_ban": ban}, {**panels, "final_retention": generic}


def evaluation_audit(ctx, root, design, training, source):
    output, child = root / "study", root / "study/evaluation"
    checked, panels = evaluation_behavior_audit(ctx, root, design, training, source)
    result = read(child / "result.json")
    launch, process = read(output / "evaluation_launch.json"), read(output / "evaluation_process.json")
    checked["process_proof"] = fresh_child_audit(training, result, launch, process, file_hash(output / "training.json"),
                                               REMOTE_RUNS / root.name / "config.json", REMOTE_RUNS / root.name / "study",
                                               training["execution"]["code_dir"])
    overall = read(output / "result.json")
    require(overall["status"] == "completed" and overall["method"] == training["method"]
            and overall["training_sha256"] == file_hash(output / "training.json")
            and overall["evaluation_sha256"] == file_hash(child / "result.json")
            and overall["tokens"] == training["tokens"] and overall["native"] == result["native"]
            and overall["generic_retention"] == result["generic_retention"]
            and overall["teacher_absent_fresh_process_completed"] is True
            and overall["native_export"] == design["native_export"] and overall["claim_boundary"] == design["claim_boundary"], "completed run summary binds actual child")
    return checked, panels


def run_audit(ctx, wave, root, design, source, batches):
    execution, task, config, envelope_failure = training_envelope_audit(root)
    collection = collection_audit(root, execution["status"])
    require(task == ctx["dispatch"]["jobs"][config["method"]]["task"], "actual execution equals dispatched manifest")
    require(config == read(ctx["lane"] / ("configs/onpolicy303-budget-" + config["method"].replace("_", "-") + ".json")), "sealed dispatch config bytes")
    require(config["protocol"] == PROTOCOL_PATH and config["protocol_sha256"] == digest(design)
            and config["stage"] == "continue_and_evaluate" and config["task_id"] == root.name, "dispatched budget contract")
    require(task["entrypoint"] == "onpolicy303_budget.py" and task["gpus"] == 1 and len(execution["gpus"]) == 1, "one-GPU bounded entrypoint")
    require(ctx["design"]["runtime_dependency"]["task_id"] in task["depends_on"]
            and source["root"].name in task["depends_on"], "original training and runtime dependencies")
    if config["method"] == "cached_teacher_kl":
        require("consolidation-source303-coverage-20260912" in task["depends_on"], "qualified teacher source dependency")
    archive, payloads = archive_audit(wave, task["source_sha256"])
    require(task["code_dir"] == "/mnt/shared/cl-portfolio/code/" + task["source_sha256"], "content-bound code directory")
    require(json.loads(payloads[PROTOCOL_PATH]) == design, "actual archived budget design")
    for name, checksum in design["scientific_files_sha256"].items():
        require(archive["files_sha256"][name] == checksum, f"archived scientific dependency {name}")
    handoff = read(ctx["lane"] / "onpolicy303_budget_handoff.json")
    for name, checksum in handoff["hash_closure"]["files_sha256"].items():
        require(archive["files_sha256"][name] == checksum, f"archived sealed source/config/test {name}")
    seal = read(root / "study/seal.json")
    require(seal["design"] == design and seal["dispatch"] == config
            and seal["source_sha256"] == design["scientific_files_sha256"]
            and seal["dataset"] == design["dataset"], "pre-training sealed run contract")
    training, training_result = training_audit(ctx, root, design, source, execution, task, archive, batches)
    require(seal["pid"] == training["pid"], "sealed actual training PID")
    if envelope_failure is None:
        evaluation, panels = evaluation_audit(ctx, root, design, training, source)
    else:
        evaluation, panels = {"status": "not_started", "heldout_predictions": 0, "fresh_process_persistence_verified": False}, None
    return {"status": "complete_verified" if envelope_failure is None else "training_verified_evaluation_pending",
            "envelope_failure": envelope_failure, "task_id": root.name, "method": config["method"],
            "attempt_id": execution["attempt_id"], "execution_sha256": file_hash(root / "execution.json"),
            "source": {"sha256": archive["source_sha256"], "archive_sha256": archive["archive_sha256"],
                       "full_source_files_checked": len(archive["files_sha256"]),
                       "scientific_dependencies_checked": len(design["scientific_files_sha256"])},
            "collection": collection, "training": training_result, "evaluation": evaluation}, panels


def recovery_context(ctx, wave, budget):
    lane = ctx["lane"]
    handoff_path = lane / "onpolicy303_budget_evaluate_handoff.json"
    require(file_hash(handoff_path) == RECOVERY_HANDOFF_SHA256, "parent-confirmed recovery handoff")
    handoff = read(handoff_path)
    closure = handoff["hash_closure"]
    require(closure["files"] == len(closure["files_sha256"]) == 26
            and digest(closure["files_sha256"]) == closure["files_mapping_sha256"] == RECOVERY_CLOSURE_SHA256,
            "parent-confirmed recovery source closure")
    bound_files(lane, closure["files_sha256"])
    design = read(lane / RECOVERY_PROTOCOL_PATH)
    require(file_hash(lane / RECOVERY_PROTOCOL_PATH) == handoff["protocol"]["file_sha256"]
            and digest(design) == handoff["protocol"]["payload_sha256"], "sealed recovery protocol")
    require(design["contract"] == "onpolicy303_budget405_zero_update_evaluation_recovery"
            and design["new_updates"] == 0 and design["native_export"] == budget["native_export"], "zero-update recovery contract")
    require(design["budget_protocol"] == {"path": PROTOCOL_PATH, "file_sha256": file_hash(lane / PROTOCOL_PATH),
                                         "payload_sha256": BUDGET_PROTOCOL_SHA256}, "frozen405 training protocol")
    for name, expected in design["files_sha256"].items():
        require(closure["files_sha256"][name] == expected, "recovery runtime dependency closure")
        if name in budget["scientific_files_sha256"]:
            require(expected == budget["scientific_files_sha256"][name], "recovery retains original scientific implementation")
    require(set(design["sources"]) == set(METHODS), "recovery original arm set")
    jobs = {job["method"]: job for job in handoff["jobs"]}
    require(set(jobs) == set(METHODS), "two recovery jobs")
    for method, spec in design["sources"].items():
        root = wave / "runs" / spec["task_id"]
        bound_files(root, spec["files_sha256"])
        execution, _, _, failure = training_envelope_audit(root)
        training = read(root / "study/training.json")
        require(failure is not None and spec["method"] == method and spec["run_dir"] == str(REMOTE_RUNS / root.name)
                and spec["failed_source_sha256"] == execution["source_sha256"] == ctx["dispatch"]["source_sha256"]
                and spec["finished_at"] == execution["finished_at"] and spec["training_pid"] == training["pid"]
                and spec["checkpoint405"] == training["checkpoint405"], "pinned failed-source recovery prerequisites")
        job = jobs[method]
        require(job["strict_failed_source_data_gate"] == root.name and root.name not in job["depends_on"],
                "failed405 is a data prerequisite, not a successful task dependency")
        config_path = Path(job["config"])
        require(file_hash(config_path) == job["config_sha256"], "recovery dispatch config hash")
        config = read(config_path)
        require(config["method"] == method and config["task_id"] == job["task_id"]
                and config["stage"] == "evaluate_only" and config["protocol"] == RECOVERY_PROTOCOL_PATH
                and config["protocol_sha256"] == digest(design), "recovery config contract")
    dispatch_path = wave / "onpolicy303-budget-evaluation-recovery-dispatch.json"
    dispatch = read(dispatch_path)
    require(dispatch["handoff_sha256"] == RECOVERY_HANDOFF_SHA256
            and dispatch["source_mapping_sha256"] == RECOVERY_CLOSURE_SHA256
            and dispatch["parent_cpu_tests_passed"] == 35 and dispatch["frozen_source_files_verified"] == 26
            and dispatch["new_optimizer_updates"] == 0 and dispatch["original_training_runs_unchanged"] is True,
            "parent recovery dispatch binding")
    tasks = {}
    for spec in dispatch["jobs"]:
        manifest = Path(spec["manifest"])
        require(file_hash(manifest) == spec["manifest_sha256"], "recovery actual manifest bytes")
        task = read(manifest)
        method = task["config"]["method"]
        require(method in jobs and method not in tasks, "unique dispatched recovery method")
        job = jobs[method]
        require(task["id"] == spec["id"] == job["task_id"] and task["config"] == read(Path(job["config"]))
                and task["depends_on"] == spec["depends_on"] == job["depends_on"]
                and task["source_sha256"] == dispatch["source_sha256"]
                and task["entrypoint"] == "onpolicy303_budget_evaluate.py" and task["gpus"] == 1,
                "actual recovery task identity")
        tasks[method] = task
    require(set(tasks) == set(METHODS), "both recovery tasks dispatched")
    return {"design": design, "handoff_sha256": RECOVERY_HANDOFF_SHA256,
            "protocol_sha256": digest(design), "closure": closure["files_sha256"],
            "dispatch_sha256": file_hash(dispatch_path), "source_sha256": dispatch["source_sha256"], "tasks": tasks}


def receipt_projection_audit(origin, projected, origin_hash, remote_output):
    require("evaluation_only_origin_sha256" not in origin, "original receipt has no derived origin field")
    expected = copy.deepcopy(origin)
    expected["checkpoint405"]["path"] = str(remote_output / "checkpoint405/learner")
    expected["evaluation_only_origin_sha256"] = origin_hash
    require(projected == expected and digest(projected) == digest(expected), "derived receipt changes exactly path and origin hash")
    return {"allowed_changes": ["/checkpoint405/path", "/evaluation_only_origin_sha256"],
            "origin_training_sha256": origin_hash, "original_checkpoint_path": origin["checkpoint405"]["path"],
            "derived_checkpoint_path": expected["checkpoint405"]["path"], "all_other_values_exactly_preserved": True}


def recovery_process_audit(origin_execution, execution, origin, result, launch, process, wrapper_pid, projected_hash, copy_hash):
    require(execution["task_id"] != origin_execution["task_id"]
            and execution["attempt_id"] != origin_execution["attempt_id"], "distinct recovery supervisor task and attempt")
    require(datetime.fromisoformat(origin_execution["finished_at"]) < datetime.fromisoformat(execution["started_at"]),
            "original training process terminated before recovery task")
    pids = (origin["pid"], wrapper_pid, process["uv_launcher_pid"], result["pid"])
    require(all(isinstance(pid, int) and pid > 0 for pid in pids) and len(set(pids)) == 4, "original/wrapper/uv/child process chain")
    require(launch["training_pid"] == process["original_training_pid"] == result["training_pid"] == origin["pid"]
            and launch["evaluation_only_parent_pid"] == process["evaluation_only_parent_pid"] == wrapper_pid
            and process["evaluation_pid"] == result["pid"] and result["parent_pid"] == process["uv_launcher_pid"], "recorded recovery parent links")
    require(launch["training_sha256"] == result["training_sha256"] == projected_hash
            and launch["copy_sha256"] == copy_hash and launch["child_token"] == result["child_token"]
            and len(launch["child_token"]) == 32, "recovery nonce and exact input receipts")
    require(launch["method"] == result["method"] == origin["method"] and process["exit_code"] == 0
            and process["fresh_process"] is True and result["status"] == "completed", "completed recovery process")
    require(launch["new_optimizer_updates"] == process["new_optimizer_updates"] == result["optimizer_updates"] == 0
            and result["teacher_model_loaded"] is False and result["active_adapters"] == ["learner"]
            and result["learner_rank"] == 8 and result["train24_reload_parity"] is True
            and result["generic128_reload_parity"] is True, "teacher-absent zero-update rank8 recovery")
    return {"original_task_id": origin_execution["task_id"], "original_attempt_id": origin_execution["attempt_id"],
            "recovery_task_id": execution["task_id"], "recovery_attempt_id": execution["attempt_id"],
            "original_training_pid": origin["pid"], "recovery_wrapper_pid": wrapper_pid,
            "uv_launcher_pid": process["uv_launcher_pid"], "evaluation_pid": result["pid"],
            "evaluation_parent_pid": result["parent_pid"], "child_token": launch["child_token"],
            "distinct_tasks_and_attempts": True, "original_ended_before_recovery_started": True,
            "exact_child_argv_recorded": False, "fresh_cli_launch_checked_in_pinned_source": True,
            "numeric_pid_inequality_is_not_sole_proof": True}


def recovery_audit(ctx, wave, root, budget, recovery, source384, source405_result):
    execution, task, config = execution_audit(root)
    method = config["method"]
    require(task == recovery["tasks"][method], "collected recovery equals parent dispatched task")
    collection = collection_audit(root)
    archive, payloads = archive_audit(wave, recovery["source_sha256"])
    require(task["code_dir"] == "/mnt/shared/cl-portfolio/code/" + archive["source_sha256"], "recovery content-bound code directory")
    require(json.loads(payloads[RECOVERY_PROTOCOL_PATH]) == recovery["design"], "actual archived recovery protocol")
    for name, expected in recovery["closure"].items():
        require(archive["files_sha256"][name] == expected, f"actual recovery closure member {name}")
    require(len(execution["gpus"]) == 1, "one GPU assigned to evaluation-only task")
    spec = recovery["design"]["sources"][method]
    original_root = wave / "runs" / spec["task_id"]
    origin_execution, origin_task, _, failure = training_envelope_audit(original_root)
    require(failure is not None and source405_result["status"] == "training_verified_evaluation_pending", "independently verified failed training source")
    bound_files(original_root, spec["files_sha256"])
    origin = read(original_root / "study/training.json")
    output, child = root / "study", root / "study/evaluation"
    original_hash = file_hash(original_root / "study/training.json")
    copies = {name: spec["files_sha256"][original] for name, original in RECOVERY_COPIES.items()}
    bound_files(output, copies)
    for name, original in RECOVERY_COPIES.items():
        require((output / name).read_bytes() == (original_root / original).read_bytes(), f"byte-exact evaluator copy {name}")
    require(read(output / "original_training.json") == origin, "immutable original receipt copied intact")
    projected = read(output / "training.json")
    projection = receipt_projection_audit(origin, projected, original_hash, REMOTE_RUNS / root.name / "study")
    copied, gate = read(output / "copy.json"), read(output / "source_gate.json")
    require(copied["kind"] == "zero_update_evaluator_input_copy_not_a_new_training_receipt"
            and copied["source_run"] == spec["run_dir"] and copied["origin_training_sha256"] == original_hash
            and copied["projected_training_sha256"] == file_hash(output / "training.json")
            and copied["exact_allowed_json_changes"] == projection["allowed_changes"]
            and copied["copies_sha256"] == copies and copied["source_gate"] == gate
            and copied["original_optimizer_copied"] is False and copied["new_optimizer_updates"] == 0,
            "exact copy contract and derived receipt provenance")
    require(gate["terminal"] == {
        "task_id": original_root.name, "attempt_id": origin_execution["attempt_id"], "training_pid": origin["pid"],
        "supervisor_child_pid": origin_execution["pid"], "finished_at": origin_execution["finished_at"],
        "status": "failed_after_complete_training", "cleanup_failure": failure["failure"]["detail"],
        "cleanup_cause": "unknown residual GPU allocations; no claim about teacher residency",
        "source_sha256": origin_task["source_sha256"]}, "exact failed source execution gate")
    require(gate["ledger_updates"] == 21 and gate["loss_tokens"] == 588 and gate["total_tokens"] == 11340
            and gate["optimizer_initial_clock"] == 384 and gate["optimizer_final"] == source405_result["training"]["optimizer_final405"]
            and gate["checkpoint405"] == origin["checkpoint405"] and gate["cuda_initialized"] is False
            and gate["new_optimizer_updates"] == 0 and gate["archive"] == origin["execution"]["source"], "recovery preflight matches independently audited source")
    tensors, mapping, adapter = adapter_audit(output / "checkpoint405/learner", projected["checkpoint405"])
    require(adapter == source405_result["training"]["final_adapter405"]
            and mapping == source384["mapping"] and read(output / "parameter_mapping.json") == mapping,
            "exact resident405 adapter and parameter map copied")
    del tensors
    require(not any((output / name).exists() for name in ("checkpoint405/optimizer.pt", "ledger.json", "trajectories.jsonl", "teacher_train_cache.json")),
            "no optimizer, training ledger, sampled trajectories or teacher cache copied into recovery")
    checked, panels = evaluation_behavior_audit(ctx, root, budget, projected, source384)
    result = read(child / "result.json")
    launch, process, seal = [read(output / name) for name in ("evaluation_launch.json", "evaluation_process.json", "seal.json")]
    require(seal["design"] == recovery["design"] and seal["dispatch"] == config and seal["new_optimizer_updates"] == 0, "sealed recovery wrapper")
    require(launch["origin_training_sha256"] == original_hash, "child launch binds immutable original receipt")
    proof = recovery_process_audit(origin_execution, execution, origin, result, launch, process, seal["pid"],
                                   file_hash(output / "training.json"), file_hash(output / "copy.json"))
    proof["launcher_source_sha256"] = archive["files_sha256"]["onpolicy303_budget_evaluate.py"]
    origin_ban = read(child / "origin_read_ban.json")
    expected_forbidden = [value["run_dir"] for value in recovery["design"]["sources"].values()]
    base = ctx["choice"]["source_base"]
    expected_allowed = [base["local_path"], *[str(Path(base["local_path"]) / value["path"]) for value in base["files"]],
                        projected["checkpoint405"]["path"]]
    read_ban_audit(origin_ban, expected_forbidden, expected_allowed)
    overall = read(output / "result.json")
    check_execution_snapshot(overall["execution"], execution, task, archive, completed=False)
    require(overall["status"] == "completed" and overall["method"] == method and overall["source_execution"] == gate["terminal"]
            and overall["origin_training_sha256"] == original_hash and overall["copy_sha256"] == file_hash(output / "copy.json")
            and overall["evaluation_sha256"] == file_hash(child / "result.json") and overall["new_optimizer_updates"] == 0
            and overall["original_files_unchanged"] is True and overall["native"] == result["native"]
            and overall["generic_retention"] == result["generic_retention"] and overall["native_export"] == budget["native_export"]
            and overall["claim_boundary"] == budget["claim_boundary"], "completed recovery wrapper binds original, copy and evaluator")
    wrapper_events = jsonl(output / "events.jsonl")
    require(len(wrapper_events) == 1 and wrapper_events[0]["pid"] == seal["pid"]
            and wrapper_events[0]["event"] == "onpolicy303_budget_evaluation_recovered"
            and wrapper_events[0]["new_optimizer_updates"] == 0, "wrapper records evaluation only")
    bound_files(original_root, spec["files_sha256"])
    checked.update({"task_id": root.name, "attempt_id": execution["attempt_id"], "collection": collection,
                    "execution_sha256": file_hash(root / "execution.json"), "process_proof": proof,
                    "copy": {"files_byte_identical": len(copies), "files_sha256": copies, "receipt_projection": projection,
                             "files_sha256_scope": "The six copied input files under the recovery study directory.",
                             "projected_training_files_sha256_scope": {
                                 "root": str(REMOTE_RUNS / original_root.name / "study"),
                                 "sha256": digest(origin["files_sha256"]),
                                 "files": len(origin["files_sha256"]),
                                 "unchanged_original_manifest": True,
                                 "is_recovery_copy_inventory": False,
                                 "verified_against_original_training_files": True},
                             "copy_sha256": file_hash(output / "copy.json"), "projected_receipt_sha256": file_hash(output / "training.json"),
                             "origin_read_ban_sha256": file_hash(child / "origin_read_ban.json")},
                    "original_source_gate": gate["terminal"], "original_source_files_unchanged": True,
                    "origin_read_ban": origin_ban, "copied_adapter": adapter,
                    "source": {"sha256": archive["source_sha256"], "archive_sha256": archive["archive_sha256"],
                               "full_source_files_checked": len(archive["files_sha256"]), "scientific_closure_checked": 26},
                    "original_failed_envelope_preserved": True, "additional_training_updates": 0})
    return checked, panels


def audit_controls(ctx, design, sources, batches, original_panels):
    method = "cached_teacher_kl"
    originals = jsonl(sources[method]["root"] / method / "trajectories.jsonl")
    by_index = {record["row_index"]: record for record in originals}
    records, ledger = [], []
    for step, indices in enumerate(batches[384:], 385):
        block = []
        for microstep, index in enumerate(indices):
            record = copy.deepcopy(by_index[index])
            record.pop("student_optimizer_clock_before_sampling")
            record.update(step=step, microstep=microstep, student_optimizer_clock_before_update=step - 1)
            block.append(record)
        ledger.append({"step": step, "indices": indices, "ids": [record["id"] for record in block],
                       "trajectory_sha256": [digest(record) for record in block], "loss_tokens": 28,
                       "loss": sum(record["diagnostics"]["row_loss"] for record in block) / 4,
                       "gradient_norm": 0.1,
                       "optimizer_clocks": {"minimum": step, "maximum": step, "parameters_with_state": 144}})
        records.extend(block)
    events = [{"event": "onpolicy303_budget_update", **entry} for entry in ledger]

    class AuditControls(unittest.TestCase):
        def test_original_prefix_and_independent_next_epoch(self):
            _, proof = schedule_audit(design, ctx)
            self.assertEqual(proof["unique_suffix_training_rows"], 84)
            damaged = copy.deepcopy(design)
            damaged["schedule"]["continuation_indices"][0].reverse()
            with self.assertRaisesRegex(ValueError, "sealed next-epoch suffix"):
                schedule_audit(damaged, ctx)

        def test_ledger_and_full_vocabulary_mask_control(self):
            result = cached_trajectory_audit(ctx, records, ledger, events, batches, method)
            self.assertEqual(result["loss_tokens"], 588)
            for mutation in ("mask", "vocabulary", "clock", "target", "loss", "order"):
                with self.subTest(mutation=mutation):
                    damaged = copy.deepcopy(records)
                    record = damaged[0]
                    if mutation == "mask":
                        record["diagnostics"]["prediction_mask"][0] = True
                    elif mutation == "vocabulary":
                        record["diagnostics"]["vocabulary_size"] = 4
                    elif mutation == "clock":
                        record["student_optimizer_clock_before_update"] = 0
                    elif mutation == "target":
                        record["canonical_teacher_response_sha256"] = "0" * 64
                    elif mutation == "loss":
                        record["diagnostics"]["row_loss"] += 1
                    else:
                        record["row_index"] = (record["row_index"] + 1) % 384
                    with self.assertRaises(ValueError):
                        cached_trajectory_audit(ctx, damaged, ledger, events, batches, method)

        def test_missing_update_and_fabricated_clocks_rejected(self):
            with self.assertRaisesRegex(ValueError, "21 actual update events"):
                cached_trajectory_audit(ctx, records, ledger, events[:-1], batches, method)
            damaged = copy.deepcopy(ledger)
            damaged[-1]["optimizer_clocks"]["maximum"] = 384
            with self.assertRaisesRegex(ValueError, "actual per-update clocks"):
                cached_trajectory_audit(ctx, records, damaged, events, batches, method)

        def test_actual_original_optimizer_tensor_binding(self):
            source = sources[method]
            state = torch.load(source["root"] / method / "checkpoint384/optimizer.pt", map_location="cpu", weights_only=True)
            self.assertEqual(optimizer_audit(state, source["mapping"], 384, source["optimizer"]["options"]), source["optimizer"])
            for mutation in ("clock", "moment", "order", "lr"):
                with self.subTest(mutation=mutation):
                    damaged = copy.deepcopy(state)
                    if mutation == "clock":
                        damaged["state"][0]["step"].zero_()
                    elif mutation == "moment":
                        damaged["state"][0]["exp_avg"].flatten()[0] = float("nan")
                    elif mutation == "order":
                        damaged["param_groups"][0]["params"][0:2] = [1, 0]
                    else:
                        damaged["param_groups"][0]["lr"] = 0.1
                    with self.assertRaises(ValueError):
                        optimizer_audit(damaged, source["mapping"], 384, source["optimizer"]["options"])

        def test_finite_moment_change_changes_tensor_digest(self):
            source = sources[method]
            state = torch.load(source["root"] / method / "checkpoint384/optimizer.pt", map_location="cpu", weights_only=True)
            state["state"][0]["exp_avg"].flatten()[0] += 0.125
            proof = optimizer_audit(state, source["mapping"], 384, source["optimizer"]["options"])
            self.assertNotEqual(proof["tensor_sha256"], source["optimizer"]["tensor_sha256"])

        def test_rank_and_hidden_offset_rejected(self):
            tensors = dict(sources[method]["adapter_tensors"])
            tensors["hidden_dense_offset"] = torch.zeros((8, 8))
            with self.assertRaisesRegex(ValueError, "only q/v rank8 factors"):
                adapter_mapping(tensors)

        def test_every_changed_answer_including_wrong_to_wrong(self):
            rows = ctx["corpus"]["test"]
            left = original_panels["cached_teacher_sft"]["test"]
            right = original_panels["cached_teacher_kl"]["test"]
            result = paired_native(rows, left, right)
            self.assertEqual((result["counts"]["gained"], result["counts"]["lost"]), (5, 1))
            wrong = next(index for index, row in enumerate(left) if not row["generation"]["correct"])
            altered = copy.deepcopy(left)
            altered[wrong]["generation"]["token_ids"] = [151645]
            altered[wrong]["generation"]["body_text"] = ""
            altered[wrong]["generation"]["raw_text"] = "<|im_end|>"
            changed = paired_native(rows, left, altered)
            self.assertEqual(changed["counts"]["changed_answers"], 1)
            self.assertEqual(changed["counts"]["net_gain"], 0)
            self.assertEqual(changed["every_changed_native_answer"][0]["transition"], "still_wrong")

        def test_standalone_child_not_pid_inequality_alone(self):
            training = {"pid": 101, "method": method}
            command = ["uv", "run", "--no-project", "--python", "/usr/bin/python", "python", "-B",
                       "/code/onpolicy303_budget.py", "--config", "/runs/task/config.json",
                       "--output-dir", "/runs/task/study", "--evaluate-child"]
            launch = {"training_pid": 101, "training_sha256": "h", "child_token": "a" * 32,
                      "method": method, "started_unix_ns": 10, "command": command}
            process = {"training_pid": 101, "uv_launcher_pid": 102, "returncode": 0, "completed_unix_ns": 20}
            result = {"status": "completed", "pid": 103, "parent_pid": 102, "training_pid": 101,
                      "training_sha256": "h", "child_token": "a" * 32, "method": method,
                      "teacher_model_loaded": False, "active_adapters": ["learner"], "learner_rank": 8,
                      "optimizer_updates": 0, "train24_reload_parity": True, "generic128_reload_parity": True}
            arguments = (training, result, launch, process, "h", Path("/runs/task/config.json"), Path("/runs/task/study"), "/code")
            self.assertTrue(fresh_child_audit(*arguments)["fresh_child_under_same_supervisor_task"])
            for key, value in (("parent_pid", 999), ("child_token", "b" * 32), ("teacher_model_loaded", True)):
                previous = result[key]
                result[key] = value
                with self.assertRaises(ValueError):
                    fresh_child_audit(*arguments)
                result[key] = previous
            command[-1] = "--train"
            with self.assertRaisesRegex(ValueError, "actual standalone evaluator CLI"):
                fresh_child_audit(*arguments)

        def test_execution_identity_and_incomplete_status(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "task"
                root.mkdir()
                config = {"method": method}
                task = {"id": "task", "config": config, "source_sha256": "a" * 64}
                execution = {"task": task, "task_id": "task", "source_sha256": "a" * 64, "attempt_id": "attempt",
                             "status": "completed", "exit_code": 0, "timed_out": False}
                for name, value in (("config", config), ("task", task)):
                    payload = json.dumps(value, indent=2) + "\n"
                    (root / (name + ".json")).write_text(payload)
                    execution[name + "_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
                path = root / "execution.json"
                path.write_text(json.dumps(execution))
                execution_audit(root)
                for key, value in (("status", "running"), ("task_sha256", "0" * 64), ("timed_out", True)):
                    damaged = {**execution, key: value}
                    path.write_text(json.dumps(damaged))
                    with self.assertRaises(ValueError):
                        execution_audit(root)

        def test_collected_byte_drift_rejected(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / "artifact.json"
                data.write_text("{}\n")
                manifest = {"execution_status": "completed", "files": {data.name: {"bytes": data.stat().st_size, "sha256": file_hash(data)}}}
                (root / "collection.json").write_text(json.dumps(manifest))
                self.assertEqual(collection_audit(root)["files"], 1)
                data.write_text("[]\n")
                with self.assertRaisesRegex(ValueError, "artifact bytes"):
                    collection_audit(root)

        def test_failed_teardown_preserves_training_without_inventing_evaluation(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "task"
                output = root / "study"
                output.mkdir(parents=True)
                config = {"method": method}
                task = {"id": "task", "config": config, "source_sha256": "a" * 64}
                execution = {"task": task, "task_id": "task", "source_sha256": "a" * 64, "attempt_id": "attempt",
                             "status": "failed", "exit_code": 1, "timed_out": False}
                for name, value in (("config", config), ("task", task)):
                    payload = json.dumps(value, indent=2) + "\n"
                    (root / (name + ".json")).write_text(payload)
                    execution[name + "_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
                (root / "execution.json").write_text(json.dumps(execution))
                training = {"pid": 100, "status": "trained_evaluation_pending", "updates_this_run": 21, "selected_checkpoint": 405}
                failure = {"pid": 100, "exception": "ValueError", "detail": "ONPOLICY303_BUDGET release all training models before fresh evaluator"}
                (output / "training.json").write_text(json.dumps(training))
                (output / "failure.json").write_text(json.dumps(failure))
                proof = training_envelope_audit(root)[3]
                self.assertFalse(proof["evaluation_started"])
                self.assertFalse(proof["scientific_failure_inferred"])
                with self.assertRaises(ValueError):
                    execution_audit(root)
                (output / "evaluation_launch.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "failed before any evaluator launch"):
                    training_envelope_audit(root)
                (output / "evaluation_launch.json").unlink()
                failure["detail"] = "teacher qualification failed"
                (output / "failure.json").write_text(json.dumps(failure))
                with self.assertRaisesRegex(ValueError, "exact post-training allocation guard failure"):
                    training_envelope_audit(root)

        def test_archive_rehash_not_basename_only(self):
            with tempfile.TemporaryDirectory() as temporary:
                wave = Path(temporary)
                (wave / "code").mkdir()
                payload = b"value = 1\n"
                identity = hashlib.sha256(b"main.py\0" + payload).hexdigest()
                path = wave / "code" / (identity + ".tar")
                for content, accepted in ((payload, True), (b"value = 2\n", False)):
                    with tarfile.open(path, "w") as stream:
                        member = tarfile.TarInfo("main.py")
                        member.size = len(content)
                        stream.addfile(member, io.BytesIO(content))
                    metadata = {"archive": {"bytes": path.stat().st_size, "sha256": file_hash(path)}, "files": 1, "source_sha256": identity}
                    path.with_suffix(".json").write_text(json.dumps(metadata))
                    if accepted:
                        self.assertEqual(archive_audit(wave, identity)[0]["source_sha256"], identity)
                    else:
                        with self.assertRaisesRegex(ValueError, "actual source closure digest"):
                            archive_audit(wave, identity)

        def test_read_ban_requires_exact_pinned_paths_and_order(self):
            base = ctx["choice"]["source_base"]
            checkpoint = "/audit/recovery/checkpoint405/learner"
            pinned = [str(Path(base["local_path"]) / spec["path"]) for spec in base["files"]]
            allowed = [base["local_path"], checkpoint, *pinned]
            forbidden = ["/audit/original405", "/audit/recovery/teacher_train_cache.json",
                         "/audit/recovery/teacher_train_cache.json"]
            ban = {"allowed_model_roots": allowed, "forbidden_roots": forbidden,
                   "observed_weight_paths": [*pinned, checkpoint + "/adapter_model.safetensors"],
                   "denied_reads": [], "installed_before_model_load": True}
            read_ban_audit(ban, forbidden, allowed)
            for mutation in ("extra_root", "extra_base_file", "missing_pin", "order", "forbidden_duplicate",
                             "outside_read", "forbidden_read", "denied_read", "late_install"):
                with self.subTest(mutation=mutation):
                    damaged = copy.deepcopy(ban)
                    if mutation == "extra_root":
                        damaged["allowed_model_roots"].append("/audit/teacher")
                    elif mutation == "extra_base_file":
                        damaged["allowed_model_roots"].append(str(Path(base["local_path"]) / "unlisted.safetensors"))
                    elif mutation == "missing_pin":
                        damaged["allowed_model_roots"].pop()
                    elif mutation == "order":
                        damaged["allowed_model_roots"][1:3] = damaged["allowed_model_roots"][1:3][::-1]
                    elif mutation == "forbidden_duplicate":
                        damaged["forbidden_roots"].pop()
                    elif mutation == "outside_read":
                        damaged["observed_weight_paths"].append("/audit/teacher/adapter_model.safetensors")
                    elif mutation == "forbidden_read":
                        damaged["observed_weight_paths"].append("/audit/original405/adapter_model.safetensors")
                    elif mutation == "denied_read":
                        damaged["denied_reads"].append("/audit/original405")
                    else:
                        damaged["installed_before_model_load"] = False
                    with self.assertRaises(ValueError):
                        read_ban_audit(damaged, forbidden, allowed)

        def test_recovery_receipt_allows_only_path_and_origin(self):
            original = {"checkpoint405": {"path": "/original/learner", "tensor_sha256": "tensor"},
                        "optimizer_final": {"clock": 405}, "tokens": {"total_loss_tokens": 11340}, "method": method}
            projected = copy.deepcopy(original)
            projected["checkpoint405"]["path"] = "/new/study/checkpoint405/learner"
            projected["evaluation_only_origin_sha256"] = "origin"
            proof = receipt_projection_audit(original, projected, "origin", Path("/new/study"))
            self.assertTrue(proof["all_other_values_exactly_preserved"])
            for mutation in ("clock", "clock_type", "tokens", "tensor", "method", "origin", "path"):
                with self.subTest(mutation=mutation):
                    damaged = copy.deepcopy(projected)
                    if mutation == "clock":
                        damaged["optimizer_final"]["clock"] = 406
                    elif mutation == "clock_type":
                        damaged["optimizer_final"]["clock"] = 405.0
                    elif mutation == "tokens":
                        damaged["tokens"]["total_loss_tokens"] = 11314
                    elif mutation == "tensor":
                        damaged["checkpoint405"]["tensor_sha256"] = "another"
                    elif mutation == "method":
                        damaged["method"] = "onpolicy_kl"
                    elif mutation == "origin":
                        damaged["evaluation_only_origin_sha256"] = "different"
                    else:
                        damaged["checkpoint405"]["path"] = "/another/learner"
                    with self.assertRaisesRegex(ValueError, "derived receipt changes exactly"):
                        receipt_projection_audit(original, damaged, "origin", Path("/new/study"))

        def test_recovery_needs_new_task_nonce_parent_and_zero_updates(self):
            origin_execution = {"task_id": "original", "attempt_id": "attempt-original", "finished_at": "2026-09-12T08:51:00+00:00"}
            execution = {"task_id": "recovery", "attempt_id": "attempt-recovery", "started_at": "2026-09-12T09:04:00+00:00"}
            origin = {"pid": 101, "method": method}
            result = {"status": "completed", "pid": 104, "parent_pid": 103, "training_pid": 101, "method": method,
                      "training_sha256": "derived", "child_token": "a" * 32, "optimizer_updates": 0,
                      "teacher_model_loaded": False, "active_adapters": ["learner"], "learner_rank": 8,
                      "train24_reload_parity": True, "generic128_reload_parity": True}
            launch = {"training_pid": 101, "evaluation_only_parent_pid": 102, "training_sha256": "derived",
                      "copy_sha256": "copy", "child_token": "a" * 32, "method": method, "new_optimizer_updates": 0}
            process = {"original_training_pid": 101, "evaluation_only_parent_pid": 102, "evaluation_pid": 104,
                       "uv_launcher_pid": 103, "exit_code": 0, "fresh_process": True, "new_optimizer_updates": 0}
            arguments = (origin_execution, execution, origin, result, launch, process, 102, "derived", "copy")
            self.assertTrue(recovery_process_audit(*arguments)["distinct_tasks_and_attempts"])
            for target, key, value in ((execution, "task_id", "original"), (execution, "attempt_id", "attempt-original"),
                                       (execution, "started_at", "2026-09-12T08:50:00+00:00"),
                                       (result, "parent_pid", 999), (launch, "child_token", "b" * 32),
                                       (result, "training_sha256", "wrong"), (process, "new_optimizer_updates", 1),
                                       (result, "teacher_model_loaded", True)):
                previous = target[key]
                target[key] = value
                with self.assertRaises(ValueError):
                    recovery_process_audit(*arguments)
                target[key] = previous

    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditControls))
    require(result.wasSuccessful(), "CPU audit controls failed\n" + buffer.getvalue())
    return {"passed": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "log": buffer.getvalue(), "scope": "CPU auditor rejection controls; synthetic ledgers reuse historical diagnostic rows solely to test checks, not evidence of new GPU updates."}


def build_report(repo, wave):
    ctx, original_panels, references = reference_audit(repo, wave)
    ctx["lane"] = repo / "experiments/distillation"
    design = read(ctx["lane"] / PROTOCOL_PATH)
    batches, schedule, sources = design_audit(repo, wave, ctx, design)
    ctx["dispatch"] = dispatch_audit(ctx, wave, design)
    recovery = recovery_context(ctx, wave, design)
    tests = audit_controls(ctx, design, sources, batches, original_panels)
    report = {
        "status": "prepared_awaiting_collected_continuations", "observed_at": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": file_hash(Path(__file__)), "protocol_sha256": digest(design),
        "protocol_file_sha256": file_hash(ctx["lane"] / PROTOCOL_PATH), "scientific_files_sha256": design["scientific_files_sha256"],
        "handoff_sha256": file_hash(ctx["lane"] / "onpolicy303_budget_handoff.json"),
        "dispatch": ctx["dispatch"],
        "recovery": {key: value for key, value in recovery.items() if key != "design"},
        "original_references": references, "independent_suffix_schedule": schedule,
        "cpu_auditor_controls": tests, "runs": {}, "pending": [], "issues": [], "paired_comparisons": {},
        "claim_boundary": BOUNDARY, "proof_limit": PROOF_LIMIT, "native_export": design["native_export"],
        "science_or_recipe_changes_by_auditor": [], "provider_actions_by_auditor": [],
        "auditor_assumption_corrections": [{
            "check": "evaluation read-ban allowed_model_roots",
            "previous_expectation": "Only pinned base root and copied learner root.",
            "frozen_implementation": "Those two roots followed by every source_base.files path in its pinned order.",
            "resolution": "Require the complete exact ordered list, retain exact forbidden paths including duplicates, and reject missing, reordered or arbitrary extra roots/files in CPU controls.",
            "experiment_artifacts_modified": False}],
    }
    conditions = {method + "384": panels for method, panels in original_panels.items()}
    for method in METHODS:
        config = read(ctx["lane"] / ("configs/onpolicy303-budget-" + method.replace("_", "-") + ".json"))
        root = wave / "runs" / config["task_id"]
        if not (root / "collection.json").exists():
            report["pending"].append({"task_id": config["task_id"], "reason": "canonical terminal collection not available; no partial staging is read"})
            continue
        archive = wave / "code" / (ctx["dispatch"]["source_sha256"] + ".tar")
        if not archive.is_file() or not archive.with_suffix(".json").is_file():
            report["pending"].append({"task_id": config["task_id"], "reason": "run collected; final source archive collection not yet available"})
            continue
        try:
            result, panels = run_audit(ctx, wave, root, design, sources[method], batches)
            report["runs"][method] = result
            if panels is not None:
                conditions[method + "405"] = panels
            else:
                report["pending"].append({"task_id": root.name, "reason": "all training verified; original envelope failed before evaluation; separate zero-update recovery evaluation required"})
        except (ValueError, KeyError, OSError, AssertionError, TypeError) as error:
            report["issues"].append({"task_id": root.name, "exception": type(error).__name__, "detail": str(error)})
    for method in METHODS:
        if method not in report["runs"] or report["runs"][method]["status"] != "training_verified_evaluation_pending":
            continue
        training_result = report["runs"][method]
        report["pending"] = [item for item in report["pending"] if item["task_id"] != training_result["task_id"]]
        root = wave / "runs" / recovery["tasks"][method]["id"]
        archive = wave / "code" / (recovery["source_sha256"] + ".tar")
        if not (root / "collection.json").exists():
            report["pending"].append({"task_id": root.name, "reason": "training verified; fresh zero-update recovery evaluation not yet collected"})
            continue
        if not archive.is_file() or not archive.with_suffix(".json").is_file():
            report["pending"].append({"task_id": root.name, "reason": "recovery run collected; its final source archive collection is pending"})
            continue
        try:
            checked, panels = recovery_audit(ctx, wave, root, design, recovery, sources[method], training_result)
            training_result["original_evaluation"] = training_result["evaluation"]
            training_result["evaluation"] = checked
            training_result["status"] = "training_and_recovery_evaluation_verified"
            conditions[method + "405"] = panels
        except (ValueError, KeyError, OSError, AssertionError, TypeError) as error:
            report["issues"].append({"task_id": root.name, "exception": type(error).__name__, "detail": str(error)})
    retention_rows = read(ctx["lane"] / "configs/onpolicy303_retention.json")["validation"]
    for method in METHODS:
        final = method + "405"
        if final not in conditions:
            continue
        for reference in (method + "384", "onpolicy_kl384"):
            report["paired_comparisons"][final + "_versus_" + reference] = {
                **{split: paired_native(ctx["corpus"][split], conditions[reference][split], conditions[final][split])
                   for split in ("test", "unused288")},
                "generic_retention": original_audit.paired_generic(retention_rows, conditions[reference]["final_retention"], conditions[final]["final_retention"]),
            }
        report["runs"][method]["generic_initial_to405"] = original_audit.paired_generic(
            retention_rows, original_panels[method]["initial_retention"], conditions[final]["final_retention"])
    if all(method + "405" in conditions for method in METHODS):
        report["paired_comparisons"]["cached_teacher_kl405_versus_cached_teacher_sft405"] = {
            **{split: paired_native(ctx["corpus"][split], conditions["cached_teacher_sft405"][split], conditions["cached_teacher_kl405"][split])
               for split in ("test", "unused288")},
            "generic_retention": original_audit.paired_generic(retention_rows, conditions["cached_teacher_sft405"]["final_retention"], conditions["cached_teacher_kl405"]["final_retention"]),
        }
    report["condition_counts"] = {
        name: {"native": {split: original_audit.verify_generations(ctx, ctx["corpus"][split], panels[split]) for split in ("test", "unused288")},
               "generic": original_audit.retention_audit(ctx, panels["final_retention"]),
               "updates": 405 if name.endswith("405") else 384,
               "loss_tokens": 11340 if name.endswith("405") else 11314 if name == "onpolicy_kl384" else 10752}
        for name, panels in conditions.items()
    }
    report["raw_counts_new_continuations"] = {
        "training_trajectory_rows": sum(run["training"]["trajectory_audit"]["trajectory_rows"] for run in report["runs"].values()),
        "native_heldout_records": sum(sum(panel["count"] for panel in run["evaluation"]["native"].values()) for run in report["runs"].values() if run["evaluation"]["status"] == "verified"),
        "native_train_probe_records": sum(72 if run["evaluation"]["status"] == "verified" else 48 for run in report["runs"].values()),
        "generic_score_records": sum(384 if run["evaluation"]["status"] == "verified" else 256 for run in report["runs"].values()),
        "fresh_teacher_records": 480 if "cached_teacher_kl" in report["runs"] else 0,
    }
    if report["issues"]:
        report["status"] = "audit_failed"
    elif not report["pending"]:
        report["status"] = "complete_verified"
    elif len(report["runs"]) == len(METHODS):
        report["status"] = "training_verified_evaluation_pending"
    return report


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--wave", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = build_report(args.repo.resolve(), args.wave.resolve())
    path = args.wave.resolve() / "audit_onpolicy303_budget.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".onpolicy303-budget-audit-", delete=False) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)
    print(json.dumps({"audit": str(path), "status": report["status"],
                      "cpu_tests": report["cpu_auditor_controls"]["passed"], "pending": report["pending"], "issues": report["issues"],
                      "counts": {name: {"native": {split: score["correct"] for split, score in values["native"].items()},
                                        "generic": values["generic"]["correct"]} for name, values in report["condition_counts"].items()}}, indent=2))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
