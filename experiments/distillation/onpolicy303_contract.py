import hashlib
import json
import math
import os
import tarfile
from collections import Counter
from pathlib import Path

import torch

from choice_consolidation import candidate_logps, encode_choices
from choice_contract import ROOT, SCORING, digest, file_hash
from coverage_learner import validate_design as validate_coverage_design

CONTRACT = "sequence303_fixed_expert_onpolicy_forward_kl_20260912"
METHODS = ("onpolicy_kl", "cached_teacher_kl", "cached_teacher_sft")
SAMPLING = {
    "temperature": 1.0,
    "distribution": "full_vocabulary_student_softmax_float64",
    "seed": 91230503,
    "max_new_tokens": 16,
    "stop": "native_eos_or_cap",
    "forced_eos": False,
    "token_filtering": False,
    "trajectory_gradient": False,
}
OBJECTIVE = {
    "kl_direction": "teacher_to_student",
    "support": "every_model_output_vocabulary_id",
    "teacher": "fixed_detached_qualified_expert_same_prompt_no_demonstration_no_ema",
    "reduction": "mean_response_tokens_per_row_then_mean_four_rows",
    "mask": "response_predictions_including_sampled_eos_excluding_prompt_padding_and_post_eos",
    "trajectory_distribution_gradient": False,
    "importance_weights": False,
}


def checked_file(spec):
    path = ROOT / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError(f"ONPOLICY_INPUT_HASH: {path}")
    return json.loads(path.read_text())


def source_hashes(design):
    return {name: file_hash(ROOT / name) for name in design["scientific_files_sha256"]}


def validate_design(design):
    old = checked_file(design["coverage_learner_design"])
    coverage, previous, teacher, choice = validate_coverage_design(old)
    if design["contract"] != CONTRACT or source_hashes(design) != design["scientific_files_sha256"]:
        raise ValueError("ONPOLICY_SCIENTIFIC_SOURCE_CHANGED")
    if design["training"] != {**old["training"], "methods": list(METHODS)}:
        raise ValueError("ONPOLICY_MATCHED_RECIPE_CHANGED")
    if design["sampling"] != SAMPLING or design["objective"] != OBJECTIVE:
        raise ValueError("ONPOLICY_LOSS_OR_SAMPLING_CHANGED")
    if design["evaluation"] != old["evaluation"]:
        raise ValueError("ONPOLICY_NATIVE_EVALUATION_CHANGED")
    if design["teacher"]["qualification_sha256"] != "cf9b1ab01ee71e0e9265d2f73a7795e86b8a9f9e83a806d8fec47f3e3f2af91c":
        raise ValueError("ONPOLICY_WRONG_QUALIFIED_TEACHER")
    return old, coverage, previous, teacher, choice


def response_batch(prompts, responses, pad, eos, cap, device):
    if not prompts or len(prompts) != len(responses):
        raise ValueError("ONPOLICY_EMPTY_OR_UNPAIRED_BATCH")
    for prompt, response in zip(prompts, responses, strict=True):
        if not prompt or not response or len(response) > cap or eos in response[:-1]:
            raise ValueError("ONPOLICY_RESPONSE_BOUNDARY_OR_POST_EOS")
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses, strict=True)]
    ids = torch.full((len(prompts), max(lengths)), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    prediction_mask = torch.zeros((len(prompts), max(lengths) - 1), dtype=torch.bool, device=device)
    for i, (prompt, response, length) in enumerate(zip(prompts, responses, lengths, strict=True)):
        ids[i, :length] = torch.tensor(prompt + response, device=device)
        attention[i, :length] = 1
        prediction_mask[i, len(prompt) - 1:length - 1] = True
    return ids, attention, prediction_mask


def forward_kl(student_logits, teacher_logits, mask):
    if (
        student_logits.shape != teacher_logits.shape or student_logits.ndim != 3
        or mask.shape != student_logits.shape[:-1] or mask.dtype != torch.bool
        or not bool(mask.any(-1).all()) or student_logits.dtype != torch.float32
        or teacher_logits.dtype != torch.float32
    ):
        raise ValueError("ONPOLICY_FULL_VOCAB_MASK_OR_PRECISION")
    if not bool(torch.isfinite(student_logits).all() & torch.isfinite(teacher_logits).all()):
        raise ValueError("ONPOLICY_NONFINITE_LOGITS")
    student_logp = student_logits.log_softmax(-1)
    teacher_logp = teacher_logits.detach().log_softmax(-1)
    per_token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    loss = ((per_token * mask).sum(-1) / mask.sum(-1)).mean()
    return loss, per_token.detach()


def full_support_probabilities(logits):
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("ONPOLICY_SAMPLER_NONFINITE_OR_SHAPE")
    probabilities = logits.detach().double().softmax(-1)
    if not bool(torch.isfinite(probabilities).all() & (probabilities > 0).all()):
        raise ValueError("ONPOLICY_SAMPLER_NUMERICAL_SUPPORT_LOSS")
    return probabilities


def sample_response(model, prompt, eos, cap, generator):
    if not prompt or cap <= 0:
        raise ValueError("ONPOLICY_SAMPLER_EMPTY_PROMPT_OR_CAP")
    device = next(model.parameters()).device
    response, probabilities, minima = [], [], []
    cache = None
    with torch.inference_mode():
        for _ in range(cap):
            inputs = prompt if cache is None else [response[-1]]
            ids = torch.tensor([inputs], dtype=torch.long, device=device)
            attention = torch.ones((1, len(prompt) + len(response)), dtype=torch.long, device=device)
            output = model(input_ids=ids, attention_mask=attention, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if cache is None:
                raise ValueError("ONPOLICY_SAMPLER_MISSING_KV_CACHE")
            logits = output.logits[0, -1]
            if logits.dtype != torch.float32:
                raise ValueError("ONPOLICY_SAMPLER_NOT_FP32")
            distribution = full_support_probabilities(logits)
            token = int(torch.multinomial(distribution, 1, generator=generator).item())
            response.append(token)
            probabilities.append(float(distribution[token]))
            minima.append(float(distribution.min()))
            if token == eos:
                break
    return response, {
        "sampled_token_probabilities": probabilities,
        "minimum_vocabulary_probabilities": minima,
        "vocabulary_size": distribution.numel(),
        "positive_support_count": distribution.numel(),
        "temperature": 1.0, "forced_eos": False,
        "ended_with_native_eos": response[-1] == eos,
        "cap_without_eos": len(response) == cap and response[-1] != eos,
    }


def prefix_diagnostics(student, teacher, response, canonical, eos, token_kl):
    with torch.no_grad():
        slog = student.detach().log_softmax(-1)
        tlog = teacher.detach().log_softmax(-1)
        target = torch.tensor(response, device=student.device).unsqueeze(-1)
        matched = [response[:i] == canonical[:i] and i < len(canonical) for i in range(len(response))]
        tpred = teacher.argmax(-1).tolist()
        return {
            "student_argmax_token_ids": student.argmax(-1).tolist(),
            "teacher_argmax_token_ids": tpred,
            "student_sampled_token_logp": slog.gather(-1, target).squeeze(-1).tolist(),
            "teacher_sampled_token_logp": tlog.gather(-1, target).squeeze(-1).tolist(),
            "teacher_entropy": (-(tlog.exp() * tlog).sum(-1)).tolist(),
            "teacher_eos_probability": tlog[:, eos].exp().tolist(),
            "forward_kl_per_token": token_kl.tolist(),
            "prefix_matches_qualified_answer": matched,
            "qualified_next_token_ids": [canonical[i] if ok else None for i, ok in enumerate(matched)],
            "teacher_next_token_matches_on_qualified_prefix": [tpred[i] == canonical[i] if ok else None for i, ok in enumerate(matched)],
            "off_answer_prefix_competence_certified": False,
        }


def retention_rows(design, corpus):
    artifact = checked_file(design["retention"]["artifact"])
    rows = artifact["validation"]
    if len(rows) != 128 or Counter(row["task"] for row in rows) != dict.fromkeys(design["retention"]["tasks"], 32):
        raise ValueError("ONPOLICY_RETENTION_COUNTS")
    if any(row["prompt"] in {r["prompt"] for split in corpus.values() for r in split} for row in rows):
        raise ValueError("ONPOLICY_RETENTION_NATIVE_OVERLAP")
    if len({digest(row) for row in rows}) != 128 or any(not 0 <= row["gold_idx"] < len(row["choices"]) for row in rows):
        raise ValueError("ONPOLICY_RETENTION_INVALID_ROWS")
    return rows


def retention_prediction(row, scores):
    if len(scores) != len(row["choices"]) or any(not math.isfinite(x) for x in scores):
        raise ValueError("ONPOLICY_RETENTION_SCORES")
    prediction = max(range(len(scores)), key=scores.__getitem__)
    return {"row_sha256": digest(row), "task": row["task"], "scores": scores,
            "choice_count": len(scores), "gold": row["gold_idx"], "prediction": prediction,
            "correct": prediction == row["gold_idx"]}


def retention_evaluate(model, tokenizer, rows):
    encoded = [item for row in rows for item in encode_choices(tokenizer, row, range(len(row["choices"])))]
    scores = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(encoded), SCORING["candidate_batch_size"]):
            batch = encoded[offset:offset + SCORING["candidate_batch_size"]]
            totals = candidate_logps(model, tokenizer, batch).tolist()
            scores.extend(total / item[2] for total, item in zip(totals, batch, strict=True))
    records, start = [], 0
    for row in rows:
        end = start + len(row["choices"])
        records.append(retention_prediction(row, scores[start:end]))
        start = end
    return records


def retention_summary(rows):
    return {task: {"correct": sum(r["correct"] for r in rows if r["task"] == task),
                   "count": sum(r["task"] == task for r in rows)} for task in sorted({r["task"] for r in rows})}


def trajectory_summary(path, checkpoints):
    panels = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        stop = next(step for step in checkpoints if record["step"] <= step)
        start = 1 + max((step for step in checkpoints if step < stop), default=0)
        key = f"{start}-{stop}/{record['task']}"
        panel = panels.setdefault(key, {
            "trajectory_rows": 0, "loss_tokens": 0, "student_sampled_rows": 0,
            "trajectory_whole_answer_correct": 0, "trajectory_format_valid": 0, "trajectory_native_eos": 0,
            "qualified_prefix_positions": 0, "teacher_agrees_on_qualified_prefix_positions": 0,
            "divergent_prefix_positions": 0, "divergent_teacher_entropy_sum": 0.0,
            "divergent_teacher_eos_probability_sum": 0.0, "divergent_teacher_sampled_token_logp_sum": 0.0,
            "full_vocabulary_forward_kl_sum": 0.0, "kl_positions": 0,
        })
        panel["trajectory_rows"] += 1
        panel["student_sampled_rows"] += record["sampling"] is not None
        panel["loss_tokens"] += record["diagnostics"]["loss_tokens"]
        for destination, field in (("trajectory_whole_answer_correct", "correct"), ("trajectory_format_valid", "format_valid"), ("trajectory_native_eos", "terminated")):
            panel[destination] += record["generation"][field]
        diagnostics = record["diagnostics"]
        if "prefix_matches_qualified_answer" in diagnostics:
            for i, matched in enumerate(diagnostics["prefix_matches_qualified_answer"]):
                panel["kl_positions"] += 1
                panel["full_vocabulary_forward_kl_sum"] += diagnostics["forward_kl_per_token"][i]
                if matched:
                    panel["qualified_prefix_positions"] += 1
                    panel["teacher_agrees_on_qualified_prefix_positions"] += diagnostics["teacher_next_token_matches_on_qualified_prefix"][i]
                else:
                    panel["divergent_prefix_positions"] += 1
                    panel["divergent_teacher_entropy_sum"] += diagnostics["teacher_entropy"][i]
                    panel["divergent_teacher_eos_probability_sum"] += diagnostics["teacher_eos_probability"][i]
                    panel["divergent_teacher_sampled_token_logp_sum"] += diagnostics["teacher_sampled_token_logp"][i]
    return {"panels": panels, "off_answer_prefix_competence_certified": False,
            "cached_trajectory_grades_are_teacher_targets_not_student_generated_performance": True,
            "selection_from_diagnostics": False}


def archive_identity(directory, expected):
    directory = Path(directory)
    archive = directory.with_suffix(".tar")
    checksum, files = hashlib.sha256(), {}
    with tarfile.open(archive) as handle:
        for member in sorted(handle.getmembers(), key=lambda entry: Path(entry.name)):
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or member.name in files:
                raise ValueError("ONPOLICY_SOURCE_ARCHIVE_MEMBER")
            content = handle.extractfile(member).read()
            actual = directory / path
            if actual.is_symlink() or actual.read_bytes() != content:
                raise ValueError(f"ONPOLICY_SOURCE_ARCHIVE_BYTES: {actual}")
            checksum.update(member.name.encode() + b"\0" + content)
            files[member.name] = hashlib.sha256(content).hexdigest()
    if not files or checksum.hexdigest() != expected:
        raise ValueError("ONPOLICY_SOURCE_ARCHIVE_HASH")
    return {"source_sha256": expected, "archive_sha256": file_hash(archive), "files_sha256": files}


def supervisor_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()


def execution_identity(root, task_id, completed, config=None):
    root = Path(root)
    receipt = json.loads((root / "execution.json").read_text())
    task = json.loads((root / "task.json").read_text())
    actual_config = json.loads((root / "config.json").read_text())
    if (
        receipt["task_id"] != task_id or task["id"] != task_id or receipt["task"] != task
        or actual_config != task["config"] or (config is not None and actual_config != config)
        or receipt["task_sha256"] != supervisor_digest(receipt["task"])
        or receipt["config_sha256"] != supervisor_digest(receipt["task"]["config"])
        or not receipt["attempt_id"]
        or (completed and (receipt["status"] != "completed" or receipt["exit_code"] != 0 or receipt["timed_out"] is not False))
        or (not completed and receipt["status"] != "running")
    ):
        raise ValueError(f"ONPOLICY_EXECUTION_BINDING: {root}")
    source = archive_identity(task["code_dir"], task["source_sha256"])
    return {"task_id": task_id, "attempt_id": receipt["attempt_id"], "task_sha256": receipt["task_sha256"],
            "config_sha256": receipt["config_sha256"], "execution_sha256": file_hash(root / "execution.json"),
            "supervisor_child_pid": receipt["pid"], "source": source, "code_dir": task["code_dir"],
            "entrypoint": task["entrypoint"], "config": actual_config, "status": receipt["status"]}


def require_standalone(current, training):
    previous = training["execution"]
    if (
        current["task_id"] == previous["task_id"] or current["attempt_id"] == previous["attempt_id"]
        or current["entrypoint"] != "onpolicy303.py" or current["config"]["stage"] != "evaluate"
        or current["config"]["method"] != training["method"]
    ):
        raise ValueError("ONPOLICY_SEPARATE_EVALUATION_TASK_REQUIRED")
    return {"distinct_task_ids": True, "distinct_attempt_ids": True, "standalone_cli": True,
            "training_pid": training["pid"], "evaluation_pid": os.getpid(),
            "evaluation_parent_pid": os.getppid(),
            "numeric_pid_inequality_is_not_the_process_proof": True}
