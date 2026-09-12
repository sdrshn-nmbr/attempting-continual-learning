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

import device4_depth34 as native
import device4_recovered_source as recovered
import torch
from device4_chain_contract import (
    CANDIDATES,
    CONTRACT,
    GATE,
    TRAINING,
    cache_outputs,
    chain_record,
    family_feasibility,
    family_gate,
    family_rows,
    gates_by_family,
    methods_for,
    primitive_rows,
    state_from_generation,
    step_prompt,
    train_probes,
    train_schedule,
)
from generated_contract import digest, file_digest, learner_prompt
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn import functional

from run import adapter_parameters, append_json, tensor_digest
from tasks import FAMILIES

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = tuple(dict.fromkeys(("device4_chain.py", "device4_chain_contract.py", *native.SOURCE_FILES, *recovered.SOURCE_FILES)))
LOGGER = logging.getLogger("distillation.external_chain")
write_new = native.write_new
measure = native.measure
summarize = native.summarize


def source_hashes():
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def validate_design(design):
    spec = design["native_design"]
    if design["contract"] != CONTRACT or file_digest(ROOT / spec["path"]) != spec["sha256"]:
        raise ValueError("CHAIN_NATIVE_PROTOCOL_CHANGED")
    old = json.loads((ROOT / spec["path"]).read_text())
    upstream, original = native.validate_design(old)
    _, recovered_upstream, recovered_original, _ = recovered.validate_source(design["teacher_source"])
    if upstream != recovered_upstream or original != recovered_original:
        raise ValueError("CHAIN_RECOVERED_SCIENTIFIC_RECIPE_CHANGED")
    if design["frozen_source_sha256"] != native.source_hashes():
        raise ValueError("CHAIN_FROZEN_DEPENDENCY_CHANGED")
    if design["candidates"] != list(CANDIDATES) or design["qualification"] != GATE or design["training"] != TRAINING:
        raise ValueError("CHAIN_CANDIDATES_GATES_OR_BUDGET_CHANGED")
    if design["primitive_counts_per_family"] != {"train": 24, "fallback_validation": 6}:
        raise ValueError("CHAIN_PRIMITIVE_ADMISSION_POOL_CHANGED")
    if design["execution"] != {
        "qualification_jobs": 1, "learner_jobs": "one_independent_job_per_family",
        "teacher_or_chain_loaded_during_learning": False, "fresh_process_evaluation": True,
        "evaluation_split": "test", "carry_all_qualified_candidates": True,
        "oracle_reference_every_family": True,
    }:
        raise ValueError("CHAIN_EXECUTION_BOUNDARY_CHANGED")
    return old, upstream, original


def verify_source(spec, upstream, original):
    proof, corpus, history = recovered.verify_source(spec)
    tasks = primitive_rows(corpus)
    panels = json.loads((Path(spec["qualification_dir"]) / "panels.json").read_text())
    candidates = {}
    for arm in native.upstream_lane.ARMS:
        train = json.loads(Path(proof["train_diagnostics"][arm]["path"]).read_text())
        lookup = {row["uid"]: row for row in train + panels[arm]["diagnostics"]}
        candidates[arm] = {split: [lookup[task.uid] for task in rows] for split, rows in tasks.items()}
        for split, rows in tasks.items():
            native.verify_records(rows, candidates[arm][split], corpus, original, proof["eos_token_id"], proof["special_token_ids"])
    return proof, corpus, history, candidates


def emitted_generation(runner, prompt):
    inputs, response, body, _ = runner.generate(prompt, sample=False)
    tokens = response.tolist()
    return {
        "prompt": prompt, "prompt_token_ids": inputs[0].tolist(), "token_ids": tokens,
        "body_text": body,
        "raw_text": runner.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
    }


def execute_chain(runner, task):
    state, steps = task.initial, []
    for index, command in enumerate(task.program):
        prompt = step_prompt(task, state, index, runner.corpus["demonstration"])
        generated = emitted_generation(runner, prompt) if prompt is not None else None
        steps.append({"index": index, "command": command, "input_state": list(state), "generation": generated})
        state = state_from_generation(generated, runner.eos, runner.special_ids, runner.config["max_new_tokens"]) if generated is not None else None
        if state is None:
            break
    return chain_record(task, steps, runner.corpus["demonstration"], runner.config, runner.eos, runner.special_ids)


def pair_result(admitted, feasibility, gate):
    if not admitted:
        status = "rejected_primitive_gate"
    elif not feasibility["passed"]:
        status = "rejected_infeasible_deep_gain_or_significance"
    elif not gate["whole_chain_gate_passed"]:
        status = "rejected_whole_chain_gate"
    elif not gate["complete_train_cache"]:
        status = "rejected_incomplete_train_cache"
    else:
        status = "qualified"
    return {"status": status, "gate": gate}


def qualify(design, old, upstream, original, corpus, audit, output):
    proof, old_corpus, history, candidates = verify_source(design["teacher_source"], upstream, original)
    tasks = primitive_rows(old_corpus)
    for split, rows in tasks.items():
        for family in FAMILIES:
            if len([task for task in rows if task.family == family]) != design["primitive_counts_per_family"][split]:
                raise ValueError("CHAIN_ORIGINAL_PRIMITIVE_COUNTS")
    runner = native.load_runner(original, upstream, old_corpus, history, output)
    native.check_runtime(runner, proof)
    runner.model.set_adapter("student", inference_mode=True)
    primitive_raw = {split: measure(runner, rows, output, f"primitive/raw_learner/{split}", privileged=False, forced=False) for split, rows in tasks.items()}
    candidates["raw_privileged"] = {split: measure(runner, rows, output, f"primitive/raw_privileged/{split}", privileged=True, forced=False) for split, rows in tasks.items()}
    primitive = gates_by_family(tasks, primitive_raw, candidates, original, old_corpus, runner.eos, runner.special_ids, design["qualification"])
    panels = {"primitive_raw": primitive_raw, "primitive_candidates": candidates, "raw_full_program": {}, "chains": {}}
    feasibility, pairs, eligible, cache_files = {}, {}, {}, {}
    runner.corpus = corpus
    for family in FAMILIES:
        admitted = [candidate for candidate in CANDIDATES if primitive[candidate][family]["passed"]]
        raw = None
        feasibility[family] = None
        if admitted:
            runner.model.set_adapter("student", inference_mode=True)
            raw = {split: measure(runner, family_rows(corpus, split, family), output, f"{family}/raw_full_program/{split}", privileged=False, forced=False) for split in ("train", "validation")}
            panels["raw_full_program"][family] = raw
            feasibility[family] = family_feasibility(corpus, family, raw, original, runner.eos, runner.special_ids, design["qualification"])
        pairs[family], eligible[family], panels["chains"][family] = {}, [], {}
        for candidate in CANDIDATES:
            gate = None
            if candidate in admitted and feasibility[family]["passed"]:
                if candidate == "raw_privileged":
                    runner.model.set_adapter("student", inference_mode=True)
                else:
                    native.upstream_lane.restore(runner, proof["checkpoints"][candidate])
                records = {}
                for split in ("train", "validation"):
                    records[split] = []
                    for task in family_rows(corpus, split, family):
                        record = execute_chain(runner, task)
                        records[split].append(record)
                        append_json(output / "chains.partial.jsonl", {"candidate": candidate, **record})
                        runner.event("external_chain_answer", candidate=candidate, family=family, split=split, uid=task.uid, correct=record["correct"], aborted=record["aborted"], model_calls=record["model_calls"])
                panels["chains"][family][candidate] = records
                gate = family_gate(corpus, family, raw, records, original, runner.eos, runner.special_ids, design["qualification"])
                if candidate != "raw_privileged" and tensor_digest(adapter_parameters(runner.model, "teacher")) != proof["checkpoints"][candidate]["tensor_sha256"]:
                    raise ValueError("CHAIN_QUALIFICATION_MUTATED_TEACHER")
            result = pair_result(candidate in admitted, feasibility[family], gate)
            pairs[family][candidate] = result
            if result["status"] == "qualified":
                eligible[family].append(candidate)
                cache = cache_outputs(family_rows(corpus, "train", family), panels["chains"][family][candidate]["train"])
                path = f"caches/{family}/{candidate}.json"
                write_new(output / path, cache)
                cache_files[path] = file_digest(output / path)
            runner.event("external_chain_pair_gate", family=family, candidate=candidate, **result)
    runner.assert_frozen()
    if native.frozen_digest(runner) != proof["base_tensor_sha256"] or tensor_digest(adapter_parameters(runner.model, "student")) != proof["initial_tensor_sha256"]:
        raise ValueError("CHAIN_QUALIFICATION_MUTATED_BASE_OR_RAW_ADAPTER")
    write_new(output / "qualification_panels.json", panels)
    traces = [row for by_candidate in panels["chains"].values() for splits in by_candidate.values() for rows in splits.values() for row in rows]
    receipt = {
        "status": "qualified" if any(eligible.values()) else "rejected", "pid": os.getpid(),
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "upstream": proof, "primitive": primitive, "feasibility": feasibility, "pairs": pairs,
        "eligible": eligible, "cache_files_sha256": cache_files,
        "panels_sha256": file_digest(output / "qualification_panels.json"),
        "teacher_optimizer_updates": 0, "learner_updates": 0, "test_predictions": 0,
        "primitive_fresh_predictions": sum(map(len, tasks.values())) * 2,
        "primitive_archived_predictions": sum(map(len, tasks.values())) * len(native.upstream_lane.ARMS),
        "chain_model_calls": sum(r["model_calls"] for r in traces),
        "chain_prompt_tokens": sum(r["prompt_tokens"] for r in traces),
        "chain_response_tokens": sum(r["response_tokens"] for r in traces),
    }
    write_new(output / "qualification.json", receipt)
    return receipt


def read_certificate(root, design, original, corpus, audit):
    root = Path(root)
    receipt = json.loads((root / "qualification.json").read_text())
    if (
        receipt["status"] not in {"qualified", "rejected"} or receipt["design_sha256"] != digest(design)
        or receipt["source_sha256"] != source_hashes() or receipt["dataset"] != audit
        or any(receipt[key] != 0 for key in ("teacher_optimizer_updates", "learner_updates", "test_predictions"))
        or file_digest(root / "qualification_panels.json") != receipt["panels_sha256"]
    ):
        raise ValueError("CHAIN_QUALIFICATION_CERTIFICATE_CHANGED")
    panels = json.loads((root / "qualification_panels.json").read_text())
    old_corpus = native.make_corpus(original, native.upstream_lane.SPLITS)
    proof = receipt["upstream"]
    eos, specials = proof["eos_token_id"], proof["special_token_ids"]
    primitive = gates_by_family(primitive_rows(old_corpus), panels["primitive_raw"], panels["primitive_candidates"], original, old_corpus, eos, specials, design["qualification"])
    if primitive != receipt["primitive"]:
        raise ValueError("CHAIN_PRIMITIVE_ADMISSION_RECOMPUTATION_CHANGED")
    expected_caches, eligible, pairs, feasibility = {}, {}, {}, {}
    for family in FAMILIES:
        admitted = [name for name in CANDIDATES if primitive[name][family]["passed"]]
        raw = panels["raw_full_program"].get(family)
        feasibility[family] = family_feasibility(corpus, family, raw, original, eos, specials, design["qualification"]) if admitted else None
        if not admitted and raw is not None:
            raise ValueError("CHAIN_UNADMITTED_FAMILY_DEEP_EVALUATION")
        measured = panels["chains"][family]
        required = admitted if admitted and feasibility[family]["passed"] else []
        if set(measured) != set(required):
            raise ValueError("CHAIN_EVERY_ADMITTED_CANDIDATE_REQUIRED")
        eligible[family], pairs[family] = [], {}
        for name in CANDIDATES:
            gate = family_gate(corpus, family, raw, measured[name], original, eos, specials, design["qualification"]) if name in measured else None
            result = pair_result(name in admitted, feasibility[family], gate)
            pairs[family][name] = result
            if result["status"] == "qualified":
                eligible[family].append(name)
                cache = cache_outputs(family_rows(corpus, "train", family), measured[name]["train"])
                path = f"caches/{family}/{name}.json"
                if json.loads((root / path).read_text()) != cache:
                    raise ValueError("CHAIN_CACHED_EMISSIONS_CHANGED_OR_GOLD_SUBSTITUTED")
                expected_caches[path] = file_digest(root / path)
    if (
        eligible != receipt["eligible"] or pairs != receipt["pairs"] or feasibility != receipt["feasibility"]
        or expected_caches != receipt["cache_files_sha256"]
        or receipt["status"] != ("qualified" if any(eligible.values()) else "rejected")
    ):
        raise ValueError("CHAIN_ALL_QUALIFIERS_AND_REJECTIONS_MUST_BE_PRESERVED")
    return receipt, panels


def verify_generation_tokens(runner, generation):
    ids = generation["token_ids"]
    body = ids[:-1] if ids and ids[-1] == runner.eos else ids
    if (
        runner.prompt_ids(generation["prompt"])[0].tolist() != generation["prompt_token_ids"]
        or runner.tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False) != generation["body_text"]
        or runner.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != generation["raw_text"]
    ):
        raise ValueError("CHAIN_GENERATION_TOKEN_OR_PROMPT_MISMATCH")


def verify_certificate_tokens(runner, panels, family):
    primitive = [r["generation"] for rows in panels["primitive_raw"].values() for r in rows if r["family"] == family]
    primitive.extend(r["generation"] for splits in panels["primitive_candidates"].values() for rows in splits.values() for r in rows if r["family"] == family)
    for generation in primitive:
        verify_generation_tokens(runner, generation)
    for rows in panels["raw_full_program"].get(family, {}).values():
        for record in rows:
            verify_generation_tokens(runner, record["generation"])
    for splits in panels["chains"][family].values():
        for records in splits.values():
            for record in records:
                for step in record["steps"]:
                    if step["generation"] is not None:
                        verify_generation_tokens(runner, step["generation"])


def reset_student(runner, recipe):
    parameters = adapter_parameters(runner.model, "student")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(runner.initial_adapter[name])
    runner.model.set_adapter("student")
    runner.model.eval()
    runner.step = 0
    torch.manual_seed(runner.config["seed"])
    runner.optimizer = torch.optim.AdamW(runner.student_parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    if tensor_digest(parameters) != tensor_digest(runner.initial_adapter):
        raise ValueError("CHAIN_FRESH_LEARNER_INITIALIZATION_CHANGED")


def train_method(runner, design, family, method, cache, schedule, output):
    recipe = design["training"]
    reset_student(runner, recipe)
    folder = output / method
    folder.mkdir()
    rows = family_rows(runner.corpus, "train", family)
    lookup = {task.uid: task for task in rows}
    cached = {row["uid"]: row for row in cache} if cache is not None else None
    probes = train_probes(runner.corpus, family, recipe)
    initial = tensor_digest(adapter_parameters(runner.model, "student"))
    ledger, traces, checkpoints = [], [], {}
    for step, batch in enumerate(schedule, 1):
        runner.model.set_adapter("student")
        runner.optimizer.zero_grad(set_to_none=True)
        examples = []
        for uid in batch:
            task = lookup[uid]
            prompt = learner_prompt(task)
            inputs = runner.prompt_ids(prompt)
            response = runner.target_tokens(task) if cached is None else torch.tensor(cached[uid]["response_token_ids"], device=runner.device, dtype=torch.long)
            logits = runner.logits(inputs, response)
            loss = functional.cross_entropy(logits.float(), response)
            if not torch.isfinite(loss):
                raise ValueError(f"CHAIN_NONFINITE_LOSS: {family}/{method}/{step}")
            (loss / len(batch)).backward()
            trace = {
                "step": step, "uid": uid, "family": family, "task_sha256": digest(asdict(task)),
                "prompt_token_ids": inputs[0].tolist(), "response_token_ids": response.tolist(),
                "loss": float(loss.detach()), "chain_sha256": cached[uid]["chain_sha256"] if cached is not None else None,
            }
            examples.append(trace)
            traces.append(trace)
            append_json(folder / "rows.partial.jsonl", trace)
            del logits, loss
        norm = float(torch.nn.utils.clip_grad_norm_(runner.student_parameters, recipe["max_grad_norm"]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"CHAIN_INVALID_GRADIENT: {family}/{method}/{step}")
        runner.optimizer.step()
        runner.step += 1
        runner.assert_frozen()
        update = {
            "step": step, "ids": batch, "response_sha256": [digest(r["response_token_ids"]) for r in examples],
            "loss_tokens": sum(len(r["response_token_ids"]) for r in examples),
            "loss": sum(r["loss"] for r in examples) / len(examples), "gradient_norm": norm,
        }
        ledger.append(update)
        runner.event("chain_internalization_update", family=family, method=method, **update)
        if step in recipe["checkpoint_updates"]:
            runner.optimizer.zero_grad(set_to_none=True)
            destination = folder / f"checkpoint{step}"
            adapter = native.save_student(runner, destination)
            torch.save(runner.optimizer.state_dict(), destination / "optimizer.pt")
            runner.model.set_adapter("student", inference_mode=True)
            measured = measure(runner, rows if step == recipe["updates_per_method"] else probes, output, f"{family}/{method}/train_only/{step}", privileged=False, forced=False)
            if step == recipe["updates_per_method"]:
                write_new(destination / "train_diagnostics.json", measured)
                by_id = {r["uid"]: r for r in measured}
                measured = [by_id[t.uid] for t in probes]
            write_new(destination / "train_probes.json", measured)
            checkpoints[str(step)] = {
                "adapter": adapter, "optimizer_sha256": file_digest(destination / "optimizer.pt"),
                "train_probes_sha256": file_digest(destination / "train_probes.json"), "probe_summary": summarize(measured),
                "train_diagnostics_sha256": file_digest(destination / "train_diagnostics.json") if step == recipe["updates_per_method"] else None,
            }
    if runner.step != recipe["updates_per_method"] or tensor_digest(adapter_parameters(runner.model, "student")) == initial:
        raise ValueError("CHAIN_LEARNER_DID_NOT_PERSISTENTLY_UPDATE")
    write_new(folder / "rows.json", traces)
    write_new(folder / "ledger.json", ledger)
    return {
        "updates": len(ledger), "example_exposures": len(traces), "loss_tokens": sum(r["loss_tokens"] for r in ledger),
        "selected_checkpoint": recipe["updates_per_method"], "checkpoints": checkpoints,
        "initial_tensor_sha256": initial, "rows_sha256": file_digest(folder / "rows.json"),
        "ledger_sha256": file_digest(folder / "ledger.json"), "teacher_model_loaded": False, "external_chain_calls": 0,
    }


def copy_certificate(source, destination, receipt):
    files = {"qualification.json": file_digest(source / "qualification.json"), "qualification_panels.json": receipt["panels_sha256"], **receipt["cache_files_sha256"]}
    for name, checksum in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"CHAIN_PROVENANCE_ALREADY_EXISTS: {path}")
        shutil.copyfile(source / name, path)
        if file_digest(path) != checksum:
            raise ValueError("CHAIN_PROVENANCE_COPY_CHANGED")
    return files


def train(design, dispatch, old, upstream, original, corpus, audit, output):
    family, source = dispatch["family"], Path(dispatch["qualification_dir"])
    receipt, panels = read_certificate(source, design, original, corpus, audit)
    methods = methods_for(receipt, family)
    if receipt["pid"] == os.getpid():
        raise ValueError("CHAIN_LEARNING_REQUIRES_SEPARATE_QUALIFICATION_PROCESS")
    proof, _, _, candidates = verify_source(design["teacher_source"], upstream, original)
    if proof != receipt["upstream"] or any(candidates[name] != panels["primitive_candidates"][name] for name in native.upstream_lane.ARMS):
        raise ValueError("CHAIN_TEACHER_SOURCE_CHANGED_SINCE_QUALIFICATION")
    copied = copy_certificate(source, output / "qualified", receipt)
    runner = native.load_evaluator(original, output)
    runner.corpus = corpus
    native.check_runtime(runner, proof)
    if set(runner.model.peft_config) != {"student"} or sum(p.numel() for p in runner.student_parameters) != design["training"]["expected_trainable_parameters"]:
        raise ValueError("CHAIN_STUDENT_ONLY_CAPACITY_REQUIRED")
    verify_certificate_tokens(runner, panels, family)
    initial = native.save_student(runner, output / "initial")
    probes = train_probes(corpus, family, design["training"])
    write_new(output / "initial_train_probes.json", measure(runner, probes, output, f"{family}/initial/train", privileged=False, forced=False))
    schedule = train_schedule(corpus, family, design["training"])
    write_new(output / "schedule.json", schedule)
    results, equivalence = {}, {}
    for method, candidate in methods.items():
        cache = json.loads((source / f"caches/{family}/{candidate}.json").read_text()) if candidate is not None else None
        if cache is not None:
            equal = sum(runner.target_tokens(task).tolist() == row["response_token_ids"] for task, row in zip(family_rows(corpus, "train", family), cache, strict=True))
            equivalence[method] = {"equal_oracle_count": equal, "rows": len(cache), "token_identical_to_oracle": equal == len(cache), "wrong_teacher_targets_retained": sum(not row["teacher_correct"] for row in cache)}
        results[method] = train_method(runner, design, family, method, cache, schedule, output)
        if native.frozen_digest(runner) != proof["base_tensor_sha256"]:
            raise ValueError("CHAIN_LEARNING_MUTATED_BASE")
    result = {
        "status": "trained_evaluation_pending", "pid": os.getpid(), "family": family,
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "initial": initial, "base_tensor_sha256": proof["base_tensor_sha256"], "methods": results,
        "method_candidates": methods, "qualified_candidates": receipt["eligible"][family],
        "qualification_files_sha256": copied, "target_equivalence": equivalence,
        "schedule_sha256": file_digest(output / "schedule.json"), "initial_train_probes_sha256": file_digest(output / "initial_train_probes.json"),
        "total_learner_updates": sum(r["updates"] for r in results.values()),
        "oracle_reference_updates": results["oracle_sft"]["updates"],
        "teacher_output_updates": sum(r["updates"] for name, r in results.items() if name != "oracle_sft"),
        "oracle_only": not receipt["eligible"][family],
        "teacher_model_loaded": False, "external_chain_calls": 0, "test_predictions_during_learning": 0,
        "trainable_parameters": sum(p.numel() for p in runner.student_parameters),
    }
    write_new(output / "training.json", result)
    return result


def verify_student(spec, folder):
    if Path(spec["path"]).resolve() != (folder / "student").resolve():
        raise ValueError("CHAIN_LEARNER_CHECKPOINT_OUTSIDE_OWN_RUN")
    native.verify_adapter(spec)


def verify_training(root, design, original, corpus, audit, family):
    receipt, panels = read_certificate(root / "qualified", design, original, corpus, audit)
    training = json.loads((root / "training.json").read_text())
    methods = methods_for(receipt, family)
    recipe = design["training"]
    if (
        not methods or training["status"] != "trained_evaluation_pending" or training["pid"] == os.getpid()
        or training["family"] != family or training["dataset"] != audit or training["design_sha256"] != digest(design)
        or training["source_sha256"] != source_hashes() or training["method_candidates"] != methods
        or set(training["methods"]) != set(methods) or training["qualified_candidates"] != receipt["eligible"][family]
        or training["teacher_model_loaded"] or training["external_chain_calls"] != 0 or training["test_predictions_during_learning"] != 0
        or training["initial"]["tensor_sha256"] != receipt["upstream"]["initial_tensor_sha256"]
        or training["trainable_parameters"] != recipe["expected_trainable_parameters"]
        or training["base_tensor_sha256"] != receipt["upstream"]["base_tensor_sha256"]
        or training["total_learner_updates"] != len(methods) * recipe["updates_per_method"]
        or training["oracle_reference_updates"] != recipe["updates_per_method"]
        or training["teacher_output_updates"] != (len(methods) - 1) * recipe["updates_per_method"]
        or training["oracle_only"] != (not receipt["eligible"][family])
    ):
        raise ValueError("CHAIN_TRAINING_METHOD_IDENTITY_OR_BUDGET_CHANGED")
    expected_copies = {"qualification.json": file_digest(root / "qualified/qualification.json"), "qualification_panels.json": receipt["panels_sha256"], **receipt["cache_files_sha256"]}
    if expected_copies != training["qualification_files_sha256"]:
        raise ValueError("CHAIN_LOCAL_QUALIFICATION_PROVENANCE_CHANGED")
    for name, checksum in expected_copies.items():
        if file_digest(root / "qualified" / name) != checksum:
            raise ValueError("CHAIN_LOCAL_PROVENANCE_FILE_CHANGED")
    schedule = train_schedule(corpus, family, recipe)
    if json.loads((root / "schedule.json").read_text()) != schedule or file_digest(root / "schedule.json") != training["schedule_sha256"]:
        raise ValueError("CHAIN_MATCHED_FAMILY_SCHEDULE_CHANGED")
    verify_student(training["initial"], root / "initial")
    probes = train_probes(corpus, family, recipe)
    if file_digest(root / "initial_train_probes.json") != training["initial_train_probes_sha256"]:
        raise ValueError("CHAIN_INITIAL_PROBES_CHANGED")
    native.verify_records(probes, json.loads((root / "initial_train_probes.json").read_text()), corpus, original, receipt["upstream"]["eos_token_id"], receipt["upstream"]["special_token_ids"], privileged=False)
    for method, arm in training["methods"].items():
        folder = root / method
        if (
            arm["updates"] != recipe["updates_per_method"] or arm["example_exposures"] != recipe["example_exposures_per_method"]
            or arm["selected_checkpoint"] != recipe["updates_per_method"]
            or arm["initial_tensor_sha256"] != training["initial"]["tensor_sha256"]
            or arm["teacher_model_loaded"] or arm["external_chain_calls"] != 0
            or set(arm["checkpoints"]) != {str(step) for step in recipe["checkpoint_updates"]}
            or file_digest(folder / "rows.json") != arm["rows_sha256"] or file_digest(folder / "ledger.json") != arm["ledger_sha256"]
        ):
            raise ValueError("CHAIN_ARM_BUDGET_OR_FINAL_CHECKPOINT_CHANGED")
        ledger, traces = json.loads((folder / "ledger.json").read_text()), json.loads((folder / "rows.json").read_text())
        if [r["ids"] for r in ledger] != schedule or len(traces) != arm["example_exposures"] or sum(r["loss_tokens"] for r in ledger) != arm["loss_tokens"]:
            raise ValueError("CHAIN_ACTUAL_EXPOSURES_CHANGED")
        for step, checkpoint in arm["checkpoints"].items():
            destination = folder / f"checkpoint{step}"
            verify_student(checkpoint["adapter"], destination)
            if file_digest(destination / "optimizer.pt") != checkpoint["optimizer_sha256"] or file_digest(destination / "train_probes.json") != checkpoint["train_probes_sha256"]:
                raise ValueError("CHAIN_SAVED_OPTIMIZER_OR_PROBES_CHANGED")
            if int(step) == recipe["updates_per_method"]:
                records = json.loads((destination / "train_diagnostics.json").read_text())
                if file_digest(destination / "train_diagnostics.json") != checkpoint["train_diagnostics_sha256"]:
                    raise ValueError("CHAIN_FINAL_TRAIN_DIAGNOSTICS_CHANGED")
                native.verify_records(family_rows(corpus, "train", family), records, corpus, original, receipt["upstream"]["eos_token_id"], receipt["upstream"]["special_token_ids"], privileged=False)
                if checkpoint["adapter"]["tensor_sha256"] == training["initial"]["tensor_sha256"]:
                    raise ValueError("CHAIN_FINAL_LEARNER_UNCHANGED")
    return training, receipt, panels


def verify_training_tokens(runner, root, training, corpus, recipe):
    family = training["family"]
    lookup = {t.uid: t for t in family_rows(corpus, "train", family)}
    schedule = train_schedule(corpus, family, recipe)
    equivalence = {}
    for method, candidate in training["method_candidates"].items():
        cache = json.loads((root / f"qualified/caches/{family}/{candidate}.json").read_text()) if candidate is not None else None
        by_id = {row["uid"]: row for row in cache} if cache is not None else None
        if cache is not None:
            equal = sum(runner.target_tokens(task).tolist() == by_id[task.uid]["response_token_ids"] for task in lookup.values())
            equivalence[method] = {"equal_oracle_count": equal, "rows": len(cache), "token_identical_to_oracle": equal == len(cache), "wrong_teacher_targets_retained": sum(not row["teacher_correct"] for row in cache)}
        traces = json.loads((root / method / "rows.json").read_text())
        ledger = json.loads((root / method / "ledger.json").read_text())
        offset = 0
        for step, (batch, update) in enumerate(zip(schedule, ledger, strict=True), 1):
            expected = []
            for uid in batch:
                task, trace = lookup[uid], traces[offset]
                response = runner.target_tokens(task).tolist() if by_id is None else by_id[uid]["response_token_ids"]
                if (
                    trace["step"] != step or trace["uid"] != uid or trace["family"] != family
                    or trace["task_sha256"] != digest(asdict(task)) or trace["response_token_ids"] != response
                    or trace["prompt_token_ids"] != runner.prompt_ids(learner_prompt(task))[0].tolist()
                    or trace["chain_sha256"] != (by_id[uid]["chain_sha256"] if by_id is not None else None)
                    or not math.isfinite(trace["loss"])
                ):
                    raise ValueError("CHAIN_ONLY_CACHED_TRAIN_FINAL_ANSWER_ALLOWED")
                expected.append(response)
                offset += 1
            if (
                update["step"] != step or update["response_sha256"] != [digest(r) for r in expected]
                or update["loss_tokens"] != sum(map(len, expected))
                or not math.isfinite(update["gradient_norm"]) or update["gradient_norm"] <= 0 or not math.isfinite(update["loss"])
            ):
                raise ValueError("CHAIN_TARGET_TOKEN_OR_OPTIMIZER_LEDGER_CHANGED")
    if equivalence != training["target_equivalence"]:
        raise ValueError("CHAIN_TEACHER_ORACLE_EQUIVALENCE_REPORT_CHANGED")


def evaluate(design, dispatch, original, corpus, audit, output):
    root, family = Path(dispatch["training_dir"]), dispatch["family"]
    training, receipt, qualification_panels = verify_training(root, design, original, corpus, audit, family)
    runner = native.load_evaluator(original, output)
    runner.corpus = corpus
    native.check_runtime(runner, receipt["upstream"])
    if set(runner.model.peft_config) != {"student"}:
        raise ValueError("CHAIN_EVALUATION_REQUIRES_SINGLE_STUDENT_ADAPTER")
    verify_certificate_tokens(runner, qualification_panels, family)
    verify_training_tokens(runner, root, training, corpus, design["training"])
    conditions = {"untouched": training["initial"], **{method: arm["checkpoints"][str(arm["selected_checkpoint"])]["adapter"] for method, arm in training["methods"].items()}}
    panels, diagnostics = {}, {}
    probes = train_probes(corpus, family, design["training"])
    for condition, spec in conditions.items():
        state = load_file(str(Path(spec["path"]) / "adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(runner.model, state, adapter_name="student")
        runner.model.set_adapter("student", inference_mode=True)
        if tensor_digest(adapter_parameters(runner.model, "student")) != spec["tensor_sha256"]:
            raise ValueError("CHAIN_PERSISTENT_LEARNER_RELOAD_IDENTITY")
        actual = measure(runner, probes, output, f"{family}/{condition}/reload_probe", privileged=False, forced=False)
        saved = root / "initial_train_probes.json" if condition == "untouched" else root / condition / f"checkpoint{design['training']['updates_per_method']}" / "train_probes.json"
        if actual != json.loads(saved.read_text()):
            raise ValueError("CHAIN_FRESH_PROCESS_RELOAD_GENERATION_PARITY")
        panels[condition] = measure(runner, family_rows(corpus, "test", family), output, f"{family}/{condition}/test", privileged=False, forced=False)
        native.verify_records(family_rows(corpus, "test", family), panels[condition], corpus, original, runner.eos, runner.special_ids, privileged=False)
        if condition != "untouched":
            records = json.loads((root / condition / f"checkpoint{design['training']['updates_per_method']}" / "train_diagnostics.json").read_text())
            diagnostics[condition] = summarize(records)
        if tensor_digest(adapter_parameters(runner.model, "student")) != spec["tensor_sha256"] or native.frozen_digest(runner) != training["base_tensor_sha256"]:
            raise ValueError("CHAIN_EVALUATION_MUTATED_WEIGHTS")
    comparison = {method: summarize(records) for method, records in panels.items()}
    differences = {method: {
        "minus_untouched": comparison[method]["all"]["accuracy"] - comparison["untouched"]["all"]["accuracy"],
        "minus_oracle": comparison[method]["all"]["accuracy"] - comparison["oracle_sft"]["all"]["accuracy"],
        "tensor_identical_to_oracle": conditions[method]["tensor_sha256"] == conditions["oracle_sft"]["tensor_sha256"],
    } for method in training["methods"]}
    write_new(output / "panels.json", panels)
    result = {
        "status": "completed", "pid": os.getpid(), "training_pid": training["pid"], "family": family,
        "design_sha256": digest(design), "source_sha256": source_hashes(), "dataset": audit,
        "training_sha256": file_digest(root / "training.json"), "panels_sha256": file_digest(output / "panels.json"),
        "comparisons": comparison, "paired_differences": differences, "full_train_diagnostics": diagnostics,
        "qualified_candidates": training["qualified_candidates"], "target_equivalence": training["target_equivalence"],
        "learner_updates": training["total_learner_updates"], "updates_per_method": design["training"]["updates_per_method"],
        "oracle_reference_updates": training["oracle_reference_updates"],
        "teacher_output_updates": training["teacher_output_updates"], "oracle_only": training["oracle_only"],
        "evaluation_optimizer_updates": 0, "teacher_weights_loaded": False, "external_chain_calls": 0,
        "external_teacher_or_qualification_artifacts_read": False, "resident_adapters": ["student"],
        "unused_initial_teacher_adapter_removed_before_predictions": True, "fresh_process_reload_generation_parity": True,
        "claim_boundary": design["scope"],
    }
    write_new(output / "result.json", result)
    return result


def evaluate_child(output, dispatch):
    config = {"stage": "evaluate", "protocol": dispatch["protocol"], "protocol_sha256": dispatch["protocol_sha256"], "family": dispatch["family"], "training_dir": str(output.resolve())}
    path = output / "evaluate_config.json"
    write_new(path, config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run([
        "uv", "run", "--no-project", "--python", sys.executable, "python", str(ROOT / "device4_chain.py"),
        "--config", str(path.resolve()), "--output-dir", str((output / "evaluation").resolve()),
    ], cwd=ROOT, check=True)
    result = json.loads((output / "evaluation/result.json").read_text())
    if result["status"] != "completed" or result["pid"] == os.getpid() or result["training_pid"] != os.getpid() or result["family"] != dispatch["family"]:
        raise ValueError("CHAIN_FRESH_EVALUATION_CHILD_NOT_COMPLETED")
    write_new(output / "result.json", {
        "status": "completed", "family": result["family"], "training_pid": os.getpid(), "evaluation_pid": result["pid"],
        "evaluation_result_sha256": file_digest(output / "evaluation/result.json"), "comparisons": result["comparisons"],
        "qualified_candidates": result["qualified_candidates"], "learner_updates": result["learner_updates"],
        "oracle_reference_updates": result["oracle_reference_updates"],
        "teacher_output_updates": result["teacher_output_updates"], "oracle_only": result["oracle_only"],
        "claim_boundary": result["claim_boundary"],
    })


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [external-chain] %(message)s")
    dispatch = json.loads(args.config.read_text())
    design = json.loads((ROOT / dispatch["protocol"]).read_text())
    if digest(design) != dispatch["protocol_sha256"] or dispatch["stage"] not in {"qualify", "learn", "evaluate"}:
        raise ValueError("CHAIN_DISPATCH_SEAL_CHANGED")
    if dispatch["stage"] != "qualify" and dispatch["family"] not in FAMILIES:
        raise ValueError("CHAIN_EXPLICIT_SINGLE_FAMILY_JOB_REQUIRED")
    old, upstream, original = validate_design(design)
    corpus, audit = native.load_data(old, original)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", "task.json", "execution.json", "packages.txt", "run.log", "stdout.log", "stderr.log", "attempts"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("CHAIN_OUTPUT_ALREADY_USED")
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != dispatch:
        raise ValueError("CHAIN_OUTPUT_DISPATCH_CHANGED")
    sources = source_hashes()
    write_new(output / "seal.json", {"design": design, "source_sha256": sources, "dataset": audit, "dispatch": dispatch, "pid": os.getpid(), "predictions_observed": False})
    if args.validate_only:
        write_new(output / "validation.json", {"status": "cpu_contract_only", "gpu_qualified": False})
        return
    try:
        if dispatch["stage"] == "qualify":
            result = qualify(design, old, upstream, original, corpus, audit, output)
            write_new(output / "result.json", {"status": result["status"], "qualified_by_family": result["eligible"], "learner_updates": 0, "teacher_optimizer_updates": 0, "qualification_sha256": file_digest(output / "qualification.json")})
        elif dispatch["stage"] == "learn":
            result = train(design, dispatch, old, upstream, original, corpus, audit, output)
            if source_hashes() != sources:
                raise ValueError("CHAIN_SOURCE_CHANGED_DURING_LEARNING")
            if result["status"] == "trained_evaluation_pending":
                evaluate_child(output, dispatch)
        else:
            evaluate(design, dispatch, original, corpus, audit, output)
        if source_hashes() != sources:
            raise ValueError("CHAIN_SOURCE_CHANGED_DURING_RUN")
    except Exception as error:
        write_new(output / "failure.json", {"exception": type(error).__name__, "detail": str(error), "pid": os.getpid()})
        LOGGER.exception("CHAIN_FAILED")
        raise


if __name__ == "__main__":
    main()
