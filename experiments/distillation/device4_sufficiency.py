import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import device4_chain as chain
import device4_recovery as recovery
import torch
from device4_sufficiency_contract import (
    ARMS,
    CONTRACT,
    RECIPE,
    acquisition,
    schedule_for,
    verify_cached_targets,
    verify_prefix,
)
from generated_contract import digest, file_digest, learner_prompt
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn import functional

from run import adapter_parameters, append_json, tensor_digest

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("device4_sufficiency.py", "device4_sufficiency_contract.py", *chain.SOURCE_FILES)
write_new = chain.write_new
LOGGER = logging.getLogger("distillation.train_sufficiency")


def source_hashes():
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    if design["contract"] != CONTRACT or design["training"] != RECIPE or design["frozen_source_sha256"] != chain.source_hashes():
        raise ValueError("SUFFICIENCY_FROZEN_SCIENTIFIC_CONTRACT_CHANGED")
    spec = design["original_design"]
    if file_digest(ROOT / spec["path"]) != spec["sha256"]:
        raise ValueError("SUFFICIENCY_ORIGINAL_PROTOCOL_CHANGED")
    old = json.loads((ROOT / spec["path"]).read_text())
    native_design, _, original = chain.validate_design(old)
    if set(design["arms"]) != set(ARMS) or design["acquisition"] != {"overall": 0.95, "each_depth": 0.9, "score": "gold_whole_answer_native_eos", "selection": "fixed_final1024_even_if_gate_fails"}:
        raise ValueError("SUFFICIENCY_ARMS_OR_MASTERY_GATE_CHANGED")
    for name, (family, method) in ARMS.items():
        if (design["arms"][name]["family"], design["arms"][name]["method"]) != (family, method):
            raise ValueError("SUFFICIENCY_ARM_IDENTITY_CHANGED")
    return old, original, *chain.native.load_data(native_design, original)


def student_only(runner):
    if set(runner.model.peft_config) != {"student"}:
        raise ValueError("SUFFICIENCY_TEACHER_ADAPTER_PRESENT")


def load_student(original, output):
    runner = chain.native.load_evaluator(original, output)
    student_only(runner)
    return runner


def restore(runner, checkpoint, optimizer_path, step, names, recipe):
    student_only(runner)
    state = load_file(str(Path(checkpoint["path"]) / "adapter_model.safetensors"), device="cpu")
    set_peft_model_state_dict(runner.model, state, adapter_name="student")
    runner.model.set_adapter("student")
    runner.model.eval()
    parameters = adapter_parameters(runner.model, "student")
    if list(parameters) != names or tensor_digest(parameters) != checkpoint["tensor_sha256"]:
        raise ValueError("SUFFICIENCY_RESTORE_PARAMETER_IDENTITY_CHANGED")
    saved = recovery.inspect_optimizer(optimizer_path, step, names, parameters, recipe)
    runner.optimizer = torch.optim.AdamW(runner.student_parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    runner.optimizer.load_state_dict(saved)
    runner.step = step


def probe_parity(runner, tasks, expected, output):
    actual = chain.measure(runner, tasks, output, "source_native_probe_parity_before_updates", privileged=False, forced=False)
    if actual != expected:
        raise ValueError("SUFFICIENCY_SOURCE128_PROBE_PARITY_BEFORE_UPDATES")
    return actual


def response_for(runner, task, cache):
    return runner.target_tokens(task) if cache is None else torch.tensor(cache[task.uid]["response_token_ids"], device=runner.device, dtype=torch.long)


def update(runner, tasks, batch, step, cache, recipe, output):
    student_only(runner)
    runner.model.set_adapter("student")
    runner.optimizer.zero_grad(set_to_none=True)
    examples = []
    for uid in batch:
        task = tasks[uid]
        inputs = runner.prompt_ids(learner_prompt(task))
        response = response_for(runner, task, cache)
        logits = runner.logits(inputs, response)
        loss = functional.cross_entropy(logits.float(), response)
        if not torch.isfinite(loss):
            raise ValueError(f"SUFFICIENCY_NONFINITE_LOSS: {step}/{uid}")
        (loss / len(batch)).backward()
        row = {
            "step": step, "uid": uid, "family": task.family,
            "task_sha256": digest(asdict(task)),
            "prompt_token_ids": inputs[0].tolist(), "response_token_ids": response.tolist(),
            "loss": float(loss.detach()), "chain_sha256": cache[uid]["chain_sha256"] if cache is not None else None,
        }
        examples.append(row)
        if output is not None:
            append_json(output / "rows.partial.jsonl", row)
        del logits, loss
    norm = float(torch.nn.utils.clip_grad_norm_(runner.student_parameters, recipe["max_grad_norm"]))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"SUFFICIENCY_INVALID_GRADIENT: {step}")
    runner.optimizer.step()
    runner.step = step
    runner.assert_frozen()
    record = {"step": step, "ids": batch, "response_sha256": [digest(r["response_token_ids"]) for r in examples], "loss_tokens": sum(len(r["response_token_ids"]) for r in examples), "loss": sum(r["loss"] for r in examples) / len(examples), "gradient_norm": norm}
    runner.event("sufficiency_update", **record)
    return record, examples


def diagnose(runner, tasks, cache, output, label):
    student_only(runner)
    records = chain.measure(runner, tasks, output, label, privileged=False, forced=True)
    for task, row in zip(tasks, records, strict=True):
        gold = runner.target_tokens(task)
        target = response_for(runner, task, cache)
        gold_loss = -row["forced"]["gold_sequence_log_probability_including_eos"] / len(gold)
        if torch.equal(gold, target):
            target_loss = gold_loss
        else:
            with torch.no_grad():
                target_loss = float(functional.cross_entropy(runner.logits(runner.prompt_ids(learner_prompt(task)), target).float(), target))
        row.update(supplied_target_token_ids=target.tolist(), supplied_target_match=row["generation"]["token_ids"] == target.tolist(), gold_teacher_forced_token_loss=gold_loss, supplied_teacher_forced_token_loss=target_loss)
    return records


def save_point(runner, tasks, cache, recipe, output, step):
    directory = output / f"checkpoint{step}"
    runner.optimizer.zero_grad(set_to_none=True)
    adapter = chain.native.save_student(runner, directory)
    torch.save(runner.optimizer.state_dict(), directory / "optimizer.pt")
    runner.model.set_adapter("student", inference_mode=True)
    records = diagnose(runner, tasks, cache, output, f"TRAIN/{step}")
    write_new(directory / "train_diagnostics.json", records)
    return {"adapter": adapter, "optimizer_sha256": file_digest(directory / "optimizer.pt"), "train_diagnostics_sha256": file_digest(directory / "train_diagnostics.json"), "acquisition": acquisition(records)}


def copy_inputs(root, output, files):
    for name, checksum in files.items():
        src, dst = root / name, output / "origin" / name
        if file_digest(src) != checksum:
            raise ValueError(f"SUFFICIENCY_ORIGIN_FILE_CHANGED: {name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        shutil.copyfile(src, dst)
        if file_digest(dst) != checksum:
            raise ValueError(f"SUFFICIENCY_ORIGIN_COPY_CHANGED: {name}")


def origin_inputs(design, arm, old, original, corpus, audit):
    spec, root = design["arms"][arm], Path(design["arms"][arm]["training_dir"])
    start = design["training"]["start_update"]
    for name, expected in spec["receipt_files_sha256"].items():
        if not (root / name).is_file() or file_digest(root / name) != expected:
            raise ValueError(f"SUFFICIENCY_PARENT_RECEIPT_CHANGED: {name}")
    execution = json.loads((root / "execution.json").read_text())
    evaluation = json.loads((root / "evaluation/result.json").read_text())
    training, qualified, _ = chain.verify_training(root, old, original, corpus, audit, spec["family"])
    if execution["status"] != "completed" or evaluation["status"] != "completed" or evaluation["training_sha256"] != file_digest(root / "training.json") or evaluation["pid"] == training["pid"]:
        raise ValueError("SUFFICIENCY_COMPLETED_PARENT_REQUIRED")
    method = spec["method"]
    checkpoint = training["methods"][method]["checkpoints"][str(start)]
    if checkpoint["adapter"]["tensor_sha256"] != spec["checkpoint_tensor_sha256"] or checkpoint["optimizer_sha256"] != spec["optimizer_sha256"]:
        raise ValueError("SUFFICIENCY_PARENT_CHECKPOINT_CHANGED")
    files = {**spec["receipt_files_sha256"], "qualified/qualification.json": training["qualification_files_sha256"]["qualification.json"], f"{method}/ledger.json": training["methods"][method]["ledger_sha256"], f"{method}/rows.json": training["methods"][method]["rows_sha256"], f"{method}/checkpoint{start}/train_probes.json": checkpoint["train_probes_sha256"]}
    candidate = training["method_candidates"][method]
    cache = None
    if candidate is not None:
        path = f"qualified/caches/{spec['family']}/{candidate}.json"
        files[path] = qualified["cache_files_sha256"][path.removeprefix("qualified/")]
        cache = json.loads((root / path).read_text())
        verify_cached_targets(chain.family_rows(corpus, "train", spec["family"]), cache, spec["cache_canonical_sha256"])
        if sum(not row["teacher_correct"] for row in cache) != spec["expected_wrong_cached_targets"]:
            raise ValueError("SUFFICIENCY_ORIGINAL_TEACHER_ERROR_COUNT_CHANGED")
    elif spec["cache_canonical_sha256"] is not None:
        raise ValueError("SUFFICIENCY_ORACLE_CANNOT_USE_TEACHER_TARGETS")
    return root, training, qualified, checkpoint, files, cache


def train(design, dispatch, old, original, corpus, audit, output):
    arm, recipe = dispatch["arm"], design["training"]
    spec = design["arms"][arm]
    root, origin, qualified, checkpoint, files, cached = origin_inputs(design, arm, old, original, corpus, audit)
    copy_inputs(root, output, files)
    runner = load_student(original, output)
    runner.corpus = corpus
    student_only(runner)
    chain.native.check_runtime(runner, qualified["upstream"])
    if sum(p.numel() for p in runner.student_parameters) != recipe["expected_trainable_parameters"]:
        raise ValueError("SUFFICIENCY_ADAPTER_CAPACITY_CHANGED")
    chain.verify_training_tokens(runner, root, origin, corpus, old["training"])
    tasks = chain.family_rows(corpus, "train", spec["family"])
    cache = verify_cached_targets(tasks, cached, spec["cache_canonical_sha256"]) if cached is not None else None
    names = list(adapter_parameters(runner.model, "student"))
    runtime = json.loads((root / "runtime.json").read_text())
    if names != runtime["targeted_parameter_names"]:
        raise ValueError("SUFFICIENCY_ORIGINAL_PARAMETER_ORDER_CHANGED")
    source_dir = root / spec["method"] / f"checkpoint{recipe['start_update']}"
    restore(runner, checkpoint["adapter"], source_dir / "optimizer.pt", recipe["start_update"], names, recipe)
    probes = chain.train_probes(corpus, spec["family"], old["training"])
    actual = probe_parity(runner, probes, json.loads((source_dir / "train_probes.json").read_text()), output)
    write_new(output / "source_probe_parity.json", {"exact": True, "rows": actual, "source_optimizer_step": recipe["start_update"]})
    schedule = schedule_for(corpus["train"], spec["family"], recipe)
    ledger = json.loads((root / spec["method"] / "ledger.json").read_text())
    traces = json.loads((root / spec["method"] / "rows.json").read_text())
    verify_prefix(schedule, json.loads((root / "schedule.json").read_text()), ledger, recipe["start_update"])
    write_new(output / "schedule.json", schedule)
    points = {str(recipe["start_update"]): save_point(runner, tasks, cache, recipe, output, recipe["start_update"])}
    lookup = {task.uid: task for task in tasks}
    for step in range(recipe["start_update"] + 1, recipe["final_update"] + 1):
        row, examples = update(runner, lookup, schedule[step - 1], step, cache, recipe, output)
        ledger.append(row)
        traces.extend(examples)
        if step in recipe["checkpoints"]:
            points[str(step)] = save_point(runner, tasks, cache, recipe, output, step)
    if chain.native.frozen_digest(runner) != origin["base_tensor_sha256"]:
        raise ValueError("SUFFICIENCY_FROZEN_BASE_CHANGED")
    write_new(output / "ledger.json", ledger)
    write_new(output / "rows.json", traces)
    result = {"status": "trained_evaluation_pending", "pid": os.getpid(), "arm": arm, "family": spec["family"], "method": spec["method"], "source_sha256": source_hashes(), "design_sha256": digest(design), "dataset": audit, "origin_files_sha256": files, "origin_training_pid": origin["pid"], "base_tensor_sha256": origin["base_tensor_sha256"], "checkpoints": points, "retained_updates": recipe["start_update"], "new_updates": recipe["new_updates"], "total_updates": recipe["final_update"], "new_example_exposures": recipe["new_example_exposures"], "total_example_exposures": len(traces), "schedule_sha256": file_digest(output / "schedule.json"), "ledger_sha256": file_digest(output / "ledger.json"), "rows_sha256": file_digest(output / "rows.json"), "source_probe_parity_sha256": file_digest(output / "source_probe_parity.json"), "teacher_model_loaded": False, "external_chain_calls": 0, "test_predictions_during_learning": 0, "selection": "fixed_final"}
    write_new(output / "training.json", result)
    return result


def local_cache(root, spec, tasks):
    if spec["cache_canonical_sha256"] is None:
        return None
    path = root / "origin/qualified/caches/A/flat_mixed.json"
    return verify_cached_targets(tasks, json.loads(path.read_text()), spec["cache_canonical_sha256"])


def verify_training(root, design, arm, corpus, audit):
    training = json.loads((root / "training.json").read_text())
    spec, recipe = design["arms"][arm], design["training"]
    if (
        training["status"] != "trained_evaluation_pending" or training["pid"] == os.getpid()
        or training["arm"] != arm or training["family"] != spec["family"] or training["method"] != spec["method"]
        or training["source_sha256"] != source_hashes() or training["design_sha256"] != digest(design)
        or training["dataset"] != audit or training["selection"] != "fixed_final"
        or training["retained_updates"] != recipe["start_update"] or training["new_updates"] != recipe["new_updates"]
        or training["total_updates"] != recipe["final_update"]
        or training["new_example_exposures"] != recipe["new_example_exposures"]
        or training["total_example_exposures"] != recipe["final_update"] * recipe["examples_per_update"]
        or training["teacher_model_loaded"] or training["external_chain_calls"] or training["test_predictions_during_learning"]
    ):
        raise ValueError("SUFFICIENCY_TRAINING_IDENTITY_OR_FIXED_BUDGET_CHANGED")
    files = training["origin_files_sha256"]
    for name, expected in files.items():
        if file_digest(root / "origin" / name) != expected:
            raise ValueError(f"SUFFICIENCY_LOCAL_ORIGIN_CHANGED: {name}")
    if any(files.get(name) != value for name, value in spec["receipt_files_sha256"].items()):
        raise ValueError("SUFFICIENCY_ORIGIN_RECEIPTS_NOT_BOUND_TO_PROTOCOL")
    origin = json.loads((root / "origin/training.json").read_text())
    parent = origin["methods"][spec["method"]]
    selected = parent["checkpoints"][str(recipe["start_update"])]["adapter"]
    if selected["tensor_sha256"] != spec["checkpoint_tensor_sha256"] or training["base_tensor_sha256"] != origin["base_tensor_sha256"] or training["origin_training_pid"] != origin["pid"]:
        raise ValueError("SUFFICIENCY_ORIGINAL_ADAPTER_IDENTITY_CHANGED")
    expected_extra = {
        "qualified/qualification.json": origin["qualification_files_sha256"]["qualification.json"],
        f"{spec['method']}/ledger.json": parent["ledger_sha256"],
        f"{spec['method']}/rows.json": parent["rows_sha256"],
        f"{spec['method']}/checkpoint{recipe['start_update']}/train_probes.json": parent["checkpoints"][str(recipe["start_update"])]["train_probes_sha256"],
    }
    if spec["cache_canonical_sha256"] is not None:
        key = "qualified/caches/A/flat_mixed.json"
        expected_extra[key] = origin["qualification_files_sha256"][key.removeprefix("qualified/")]
    if files != {**spec["receipt_files_sha256"], **expected_extra}:
        raise ValueError("SUFFICIENCY_ORIGIN_FILE_SET_CHANGED")
    schedule = schedule_for(corpus["train"], spec["family"], recipe)
    old_schedule = json.loads((root / "origin/schedule.json").read_text())
    old_ledger = json.loads((root / "origin" / spec["method"] / "ledger.json").read_text())
    old_rows = json.loads((root / "origin" / spec["method"] / "rows.json").read_text())
    verify_prefix(schedule, old_schedule, old_ledger, recipe["start_update"])
    ledger = json.loads((root / "ledger.json").read_text())
    rows = json.loads((root / "rows.json").read_text())
    for filename in ("ledger.json", "rows.json", "schedule.json", "source_probe_parity.json"):
        if file_digest(root / filename) != training[filename.removesuffix(".json") + "_sha256"]:
            raise ValueError(f"SUFFICIENCY_TRACE_CHANGED: {filename}")
    if (
        json.loads((root / "schedule.json").read_text()) != schedule
        or ledger[:recipe["start_update"]] != old_ledger or rows[:len(old_rows)] != old_rows
        or [r["ids"] for r in ledger] != schedule or len(rows) != recipe["final_update"] * recipe["examples_per_update"]
    ):
        raise ValueError("SUFFICIENCY_RETAINED_PREFIX_OR_EXPOSURES_CHANGED")
    parity = json.loads((root / "source_probe_parity.json").read_text())
    if not parity["exact"] or parity["source_optimizer_step"] != recipe["start_update"] or parity["rows"] != json.loads((root / "origin" / spec["method"] / f"checkpoint{recipe['start_update']}/train_probes.json").read_text()):
        raise ValueError("SUFFICIENCY_SOURCE_NATIVE_PARITY_CHANGED")
    names = json.loads((root / "origin/runtime.json").read_text())["targeted_parameter_names"]
    if set(training["checkpoints"]) != {str(s) for s in recipe["checkpoints"]}:
        raise ValueError("SUFFICIENCY_CHECKPOINT_SET_CHANGED")
    for step, point in training["checkpoints"].items():
        folder = root / f"checkpoint{step}"
        if Path(point["adapter"]["path"]) != (folder / "student").resolve():
            raise ValueError("SUFFICIENCY_CHECKPOINT_OUTSIDE_OWN_RUN")
        chain.native.verify_adapter(point["adapter"])
        if file_digest(folder / "optimizer.pt") != point["optimizer_sha256"] or file_digest(folder / "train_diagnostics.json") != point["train_diagnostics_sha256"]:
            raise ValueError("SUFFICIENCY_CHECKPOINT_OR_DIAGNOSTICS_CHANGED")
        parameters = recovery.saved_parameters(folder / "student/adapter_model.safetensors")
        recovery.inspect_optimizer(folder / "optimizer.pt", int(step), names, parameters, recipe)
        records = json.loads((folder / "train_diagnostics.json").read_text())
        if acquisition(records) != point["acquisition"]:
            raise ValueError("SUFFICIENCY_MASTERY_RECOMPUTATION_CHANGED")
    if training["checkpoints"][str(recipe["start_update"])]["adapter"]["tensor_sha256"] != selected["tensor_sha256"] or training["checkpoints"][str(recipe["final_update"])]["adapter"]["tensor_sha256"] == selected["tensor_sha256"]:
        raise ValueError("SUFFICIENCY_RESUMED_LEARNER_DID_NOT_CHANGE")
    return training, ledger, rows


def verify_tokens(runner, root, design, arm, corpus, ledger, rows):
    spec, recipe = design["arms"][arm], design["training"]
    tasks = chain.family_rows(corpus, "train", spec["family"])
    lookup = {t.uid: t for t in tasks}
    cache = local_cache(root, spec, tasks)
    batch = recipe["examples_per_update"]
    for step, update_row in enumerate(ledger, 1):
        examples = rows[(step - 1) * batch:step * batch]
        for uid, row in zip(update_row["ids"], examples, strict=True):
            task = lookup[uid]
            if (
                row["uid"] != uid or row["step"] != step or row["task_sha256"] != digest(asdict(task))
                or row["response_token_ids"] != response_for(runner, task, cache).tolist()
                or row["prompt_token_ids"] != runner.prompt_ids(learner_prompt(task))[0].tolist()
                or row["chain_sha256"] != (cache[uid]["chain_sha256"] if cache is not None else None)
                or not math.isfinite(row["loss"])
            ):
                raise ValueError(f"SUFFICIENCY_TRAIN_ONLY_TARGET_OR_PROMPT_CHANGED: {step}/{uid}")
        if (
            update_row["step"] != step or update_row["response_sha256"] != [digest(r["response_token_ids"]) for r in examples]
            or update_row["loss_tokens"] != sum(len(r["response_token_ids"]) for r in examples)
            or not math.isclose(update_row["loss"], sum(r["loss"] for r in examples) / batch, abs_tol=1e-12)
            or not math.isfinite(update_row["gradient_norm"]) or update_row["gradient_norm"] <= 0
        ):
            raise ValueError(f"SUFFICIENCY_UPDATE_LEDGER_CHANGED: {step}")
    for step in recipe["checkpoints"]:
        records = json.loads((root / f"checkpoint{step}/train_diagnostics.json").read_text())
        chain.native.verify_records(tasks, records, corpus, runner.config, runner.eos, runner.special_ids, privileged=False)
        for task, row in zip(tasks, records, strict=True):
            chain.verify_generation_tokens(runner, row["generation"])
            expected = response_for(runner, task, cache).tolist()
            if row["supplied_target_token_ids"] != expected or row["supplied_target_match"] != (row["generation"]["token_ids"] == expected) or not all(math.isfinite(row[k]) and row[k] >= 0 for k in ("gold_teacher_forced_token_loss", "supplied_teacher_forced_token_loss")):
                raise ValueError("SUFFICIENCY_TRAIN_DIAGNOSTIC_TARGET_CHANGED")
    return tasks, cache


def evaluate(design, dispatch, original, corpus, audit, output):
    root, arm = Path(dispatch["training_dir"]), dispatch["arm"]
    training, ledger, rows = verify_training(root, design, arm, corpus, audit)
    runner = load_student(original, output)
    runner.corpus = corpus
    student_only(runner)
    qualified = json.loads((root / "origin/qualified/qualification.json").read_text())
    chain.native.check_runtime(runner, qualified["upstream"])
    tasks, cache = verify_tokens(runner, root, design, arm, corpus, ledger, rows)
    final = design["training"]["final_update"]
    point = training["checkpoints"][str(final)]
    state = load_file(str(Path(point["adapter"]["path"]) / "adapter_model.safetensors"), device="cpu")
    set_peft_model_state_dict(runner.model, state, adapter_name="student")
    runner.model.set_adapter("student", inference_mode=True)
    if tensor_digest(adapter_parameters(runner.model, "student")) != point["adapter"]["tensor_sha256"]:
        raise ValueError("SUFFICIENCY_FINAL_RELOAD_IDENTITY")
    actual = diagnose(runner, tasks, cache, output, "full128_TRAIN_reload_before_reused_TEST")
    saved = json.loads((root / f"checkpoint{final}/train_diagnostics.json").read_text())
    if [r["generation"] for r in actual] != [r["generation"] for r in saved] or [r["supplied_target_match"] for r in actual] != [r["supplied_target_match"] for r in saved]:
        raise ValueError("SUFFICIENCY_FULL_TRAIN_NATIVE_RELOAD_PARITY_FAILED")
    write_new(output / "train_reload.json", actual)
    mastery = acquisition(actual)
    test = chain.family_rows(corpus, "test", design["arms"][arm]["family"])
    records = chain.measure(runner, test, output, "reused_observed_TEST/fixed_final", privileged=False, forced=False)
    chain.native.verify_records(test, records, corpus, original, runner.eos, runner.special_ids, privileged=False)
    if tensor_digest(adapter_parameters(runner.model, "student")) != point["adapter"]["tensor_sha256"] or chain.native.frozen_digest(runner) != training["base_tensor_sha256"]:
        raise ValueError("SUFFICIENCY_EVALUATION_MUTATED_WEIGHTS")
    write_new(output / "test.json", records)
    result = {"status": "completed", "pid": os.getpid(), "training_pid": training["pid"], "arm": arm, "design_sha256": digest(design), "source_sha256": source_hashes(), "training_sha256": file_digest(root / "training.json"), "train_reload_sha256": file_digest(output / "train_reload.json"), "test_sha256": file_digest(output / "test.json"), "mastery": mastery, "test": chain.summarize(records), "new_updates": design["training"]["new_updates"], "total_updates": final, "full_train_native_reload_exact": True, "train_reload_rows": len(actual), "resident_adapters": ["student"], "teacher_weights_loaded": False, "external_chain_calls": 0, "external_origin_artifacts_read": False, "evaluation_updates": 0, "post_observation_reused_test": True, "confirmatory_claim": False, "generalization_after_mastery_interpretable": mastery["passed"], "claim_boundary": design["interpretation"]}
    write_new(output / "result.json", result)
    return result


def evaluate_child(output, dispatch):
    config = {"stage": "evaluate", "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "arm": dispatch["arm"], "training_dir": str(output.resolve())}
    path = output / "evaluate_config.json"
    write_new(path, config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "device4_sufficiency.py"), "--config", str(path.resolve()), "--output-dir", str((output / "evaluation").resolve())], cwd=ROOT, check=True)
    result = json.loads((output / "evaluation/result.json").read_text())
    if result["status"] != "completed" or result["pid"] == os.getpid() or result["training_pid"] != os.getpid() or result["arm"] != dispatch["arm"]:
        raise ValueError("SUFFICIENCY_SEPARATE_EVALUATOR_REQUIRED")
    write_new(output / "result.json", {"status": "completed", "arm": dispatch["arm"], "training_pid": os.getpid(), "evaluation_pid": result["pid"], "evaluation_result_sha256": file_digest(output / "evaluation/result.json"), "new_updates": result["new_updates"], "total_updates": result["total_updates"], "mastery": result["mastery"], "test": result["test"], "post_observation_reused_test": True, "claim_boundary": result["claim_boundary"]})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [train-sufficiency] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if dispatch["protocol_sha256"] != digest(design) or dispatch["stage"] not in {"learn", "evaluate"} or dispatch["arm"] not in ARMS:
        raise ValueError("SUFFICIENCY_DISPATCH_CONTRACT_CHANGED")
    old, original, corpus, audit = validate_design(design)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(p.name not in allowed for p in output.iterdir()):
        raise FileExistsError("SUFFICIENCY_OUTPUT_ALREADY_USED")
    sources = source_hashes()
    write_new(output / "seal.json", {"design": design, "design_sha256": digest(design), "source_sha256": sources, "dispatch": dispatch, "dataset": audit, "pid": os.getpid(), "new_gpu_work_observed": False})
    if args.validate_only:
        write_new(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        if dispatch["stage"] == "learn":
            train(design, dispatch, old, original, corpus, audit, output)
            evaluate_child(output, dispatch)
        else:
            evaluate(design, dispatch, original, corpus, audit, output)
        if source_hashes() != sources:
            raise ValueError("SUFFICIENCY_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_new(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("SUFFICIENCY_FAILED")
        raise


if __name__ == "__main__":
    main()
