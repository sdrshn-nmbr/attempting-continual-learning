import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from audit_persistence import test_surfaces
from data import digest, make_example
from followup import (
    SourceRun,
    comparison,
    finish,
    fresh_runtime,
    measure,
    restore_source,
    sha256_file,
    verify_code_bundle,
    verify_execution,
)
from learning import event, write_json
from run import prepare_output
from sandbox import FAMILIES, execute, grade_text, parse_calls, render

ROOT = Path(__file__).resolve().parent


def verify_dependencies(protocol):
    checksum = hashlib.sha256()
    for relative, expected in sorted(protocol["dependency_files"].items()):
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"DIAGNOSIS_DEPENDENCY_PATH: {relative}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f"DIAGNOSIS_DEPENDENCY_HASH: {relative}")
        checksum.update(relative.encode() + b"\0" + payload)
    if checksum.hexdigest() != protocol["dependency_source_sha256"]:
        raise RuntimeError("DIAGNOSIS_DEPENDENCY_BUNDLE")


def prepare(settings, output):
    if settings["experiment"] != "skill-transfer-initial-diagnosis":
        raise ValueError("DIAGNOSIS_EXPERIMENT")
    protocol_path = ROOT / settings["diagnosis_protocol"]
    if sha256_file(protocol_path) != settings["diagnosis_protocol_sha256"]:
        raise RuntimeError("DIAGNOSIS_PROTOCOL_HASH")
    protocol = json.loads(protocol_path.read_text())
    verify_dependencies(protocol)
    original = json.loads((ROOT / "followup_protocol.json").read_text())
    source = SourceRun(settings, original)
    if set(source.result["transfer"]) != set(protocol["conditions"]):
        raise RuntimeError("DIAGNOSIS_REQUIRES_ALL_SEVEN_CONDITIONS")
    if not source.result["qualification"]["passed"]:
        raise RuntimeError("DIAGNOSIS_PRIMITIVES_NEVER_QUALIFIED")
    resolved = output.resolve()
    if resolved.is_relative_to(source.root) or source.root.is_relative_to(resolved):
        raise RuntimeError("DIAGNOSIS_OUTPUT_OVERLAPS_SOURCE")
    task = json.loads((output / "task.json").read_text())
    payload = (output / "execution.json").read_bytes()
    receipt = json.loads(payload)
    verify_execution(receipt, task, "running")
    if (
        task["id"] == source.task["id"]
        or receipt["attempt_id"] == source.execution["attempt_id"]
        or task["config"] != settings
        or task["entrypoint"] != "diagnose_initial.py"
        or Path(task["code_dir"]).resolve() != ROOT
        or Path(sys.argv[0]).resolve() != ROOT / "diagnose_initial.py"
        or datetime.fromisoformat(receipt["started_at"]) < datetime.fromisoformat(source.execution["finished_at"])
    ):
        raise RuntimeError("DIAGNOSIS_NOT_DISTINCT_STANDALONE_EXECUTION")
    code = verify_code_bundle(ROOT, task["source_sha256"])
    source.process_proof = {
        "source_task_id": source.task["id"],
        "source_attempt_id": source.execution["attempt_id"],
        "source_execution_sha256": source.execution_sha256,
        "source_supervisor": source.execution["supervisor"],
        "source_launcher_pid": source.execution["pid"],
        "source_finished_at": source.execution["finished_at"],
        "diagnosis_task_id": task["id"],
        "diagnosis_attempt_id": receipt["attempt_id"],
        "diagnosis_supervisor": receipt["supervisor"],
        "diagnosis_launcher_pid": receipt["pid"],
        "diagnosis_started_at": receipt["started_at"],
        "entry_execution_sha256": hashlib.sha256(payload).hexdigest(),
        "cli_pid": os.getpid(),
        "cli_parent_pid": os.getppid(),
        "standalone_entrypoint": str(ROOT / "diagnose_initial.py"),
        "code_sha256": code["sha256"],
        "pid_boundary": "Supervisor is the queue process; receipt PID is its launcher and may differ from the final Python child. Distinct supervised task/attempt IDs, completed source and bound standalone CLI establish the recorded process boundary, not numeric PID inequality.",
    }
    prepare_output(settings, output)
    for name, content in (
        ("source-execution.json", source.execution_payload),
        ("entry-execution.json", payload),
        ("diagnosis_protocol.json", protocol_path.read_bytes()),
    ):
        with (output / name).open("xb") as handle:
            handle.write(content)
    write_json(output / "process-proof.json", source.process_proof)
    write_json(output / "source-code.json", source.code_proof)
    write_json(output / "diagnosis-code.json", code)
    return source, protocol


def validation_surfaces(source, condition, initial):
    directory = f"transfer/{condition}"
    budget = 0 if initial else source.config["workflow_checkpoints"][-1]
    subdir = "primitive-before" if initial else "primitive-after"
    surfaces = {
        "workflow.validation": (
            f"{directory}/validation-{budget}.json",
            source.corpus["workflow"][source.config["target_family"]]["validation"],
        )
    }
    surfaces.update(
        {
            f"primitive-{family}.validation": (
                f"{directory}/{subdir}/{family}.validation.json",
                source.corpus["primitive"][family]["validation"],
            )
            for family in FAMILIES
        }
    )
    return surfaces


def fresh_panels(source, protocol):
    existing = [
        row
        for families in source.corpus.values()
        for splits in families.values()
        for rows in splits.values()
        for row in rows
    ]
    groups = {row.group_id for row in existing}
    prompts = {render(row, source.spec["conventions"]) for row in existing}
    spec = {**source.spec, "data_seed": source.config["data_seed"] + protocol["fresh_inputs"]["seed_offset"]}
    panels = {}
    for label, split in (("ordinary", "validation"), ("unseen-combinations", "novel_test")):
        rows = []
        for index in range(protocol["fresh_inputs"]["rows_per_panel"]):
            row = make_example(spec, source.config["target_family"], "workflow", split, index)
            row = replace(row, id=digest([protocol["id"], row.id]), split="posthoc_diagnostic")
            prompt = render(row, source.spec["conventions"])
            if row.group_id in groups or prompt in prompts:
                raise RuntimeError("DIAGNOSIS_FRESH_INPUT_OVERLAP")
            if not grade_text(json.dumps(row.calls), row, source.spec["conventions"])["correct"]:
                raise RuntimeError("DIAGNOSIS_FRESH_ORACLE")
            groups.add(row.group_id)
            prompts.add(prompt)
            rows.append(row)
        panels[label] = rows
    return panels


def operations(calls, conventions, family):
    reverse = {alias: operation for operation, alias in conventions[family].items()}
    return [reverse.get(call["tool"], f"unknown:{call['tool']}") for call in calls]


def record_details(row, record, conventions, training_plans):
    expected = operations(row.calls, conventions, row.family)
    calls = parse_calls(record["text"]) if record["format_valid"] else []
    emitted = operations(calls, conventions, row.family)
    state_divergence = None
    if record["executable"]:
        _, oracle_trace = execute(row.calls, row.state, row.family, conventions)
        for index, (oracle, actual) in enumerate(zip(oracle_trace, record["trace"], strict=False), start=1):
            if oracle["state"] != actual["state"]:
                state_divergence = index
                break
    return {
        "id": row.id,
        "correct": record["correct"],
        "error": record["error"] or ("CORRECT" if record["correct"] else "WRONG_STATE"),
        "expected_operations": expected,
        "emitted_operations": emitted,
        "emitted_plan_in_workflow_training": tuple(emitted) in training_plans,
        "same_operations": emitted == expected,
        "argument_differences": [
            {"step": index + 1, "operation": operation, "expected": oracle["args"], "emitted": actual["args"]}
            for index, (operation, observed, oracle, actual) in enumerate(
                zip(expected, emitted, row.calls, calls, strict=False)
            )
            if operation == observed and oracle["args"] != actual["args"]
        ],
        "first_state_divergence": state_divergence,
        "request": row.request,
        "initial_state": row.state,
        "expected_calls": row.calls,
        "emitted_calls": calls,
        "expected_state": row.expected,
        "final_state": record.get("final_state"),
        "text": record["text"],
        "native_eos": record["native_eos"],
        "padding_only": record["padding_only"],
    }


def explain(result, rows, source, protocol):
    if [record["id"] for record in result["records"]] != [row.id for row in rows]:
        raise RuntimeError("DIAGNOSIS_EXPLANATION_ROW_ORDER")
    family = source.config["target_family"]
    plans = {
        tuple(operations(row.calls, source.spec["conventions"], family))
        for row in source.corpus["workflow"][family]["train"]
    }
    details = [
        record_details(row, record, source.spec["conventions"], plans)
        for row, record in zip(rows, result["records"], strict=True)
    ]
    patterns = {}
    for pattern in sorted({row.pattern for row in rows}):
        selected = [detail for row, detail in zip(rows, details, strict=True) if row.pattern == pattern]
        wrong = [detail for detail in selected if not detail["correct"]]
        patterns[pattern] = {
            "n": len(selected),
            "correct": len(selected) - len(wrong),
            "errors": dict(Counter(detail["error"] for detail in selected)),
            "emitted_plans": dict(Counter(">".join(detail["emitted_operations"]) or "unparsed" for detail in selected)),
            "wrong_answers_using_training_plan": sum(detail["emitted_plan_in_workflow_training"] for detail in wrong),
            "wrong_answers_with_same_operations": sum(detail["same_operations"] for detail in wrong),
            "wrong_answers_with_argument_differences": sum(bool(detail["argument_differences"]) for detail in wrong),
            "examples": wrong[: protocol["failure_examples_per_pattern"]],
        }
    return {"metrics": result["metrics"], "per_pattern": patterns, "all_records": details}


def paired_change(initial, final, rows):
    ids = [row.id for row in rows]
    if [row["id"] for row in initial["records"]] != ids or [row["id"] for row in final["records"]] != ids:
        raise RuntimeError("DIAGNOSIS_PAIRED_ROW_ORDER")
    groups = {}
    for pattern in ["all", *sorted({row.pattern for row in rows})]:
        selected = [
            (row, before, after)
            for row, before, after in zip(rows, initial["records"], final["records"], strict=True)
            if pattern == "all" or row.pattern == pattern
        ]
        transitions = Counter(
            "retained"
            if before["correct"] and after["correct"]
            else "lost"
            if before["correct"]
            else "gained"
            if after["correct"]
            else "wrong_at_both"
            for _, before, after in selected
        )
        groups[pattern] = {
            "n": len(selected),
            "counts": {key: transitions[key] for key in ("retained", "lost", "gained", "wrong_at_both")},
            "lost_ids": [row.id for row, before, after in selected if before["correct"] and not after["correct"]],
            "gained_ids": [row.id for row, before, after in selected if not before["correct"] and after["correct"]],
        }
    return {"initial_metrics": initial["metrics"], "final_metrics": final["metrics"], "paired": groups}


def diagnose(settings, output):
    source, protocol = prepare(settings, output)
    panels = fresh_panels(source, protocol)
    write_json(output / "fresh-inputs.json", {name: [row.record() for row in rows] for name, rows in panels.items()})
    model, encoded = fresh_runtime(source, output)
    results = {}
    for condition in protocol["conditions"]:
        initial_path = source.result["transfer"][condition]["initial_checkpoint"]
        restore_source(model, source, initial_path)
        directory = output / condition
        anchors, initial = {}, {}
        for name, (reference, rows) in validation_surfaces(source, condition, True).items():
            expected = source.evaluation(reference, rows)
            initial[name] = measure(model, encoded, rows, directory / "initial" / f"{name}.json")
            anchors[name] = comparison(expected, initial[name])
        record = {"initial_checkpoint": initial_path, "initial_dev_reload": anchors}
        if not all(value["all_records_exact"] for value in anchors.values()):
            record.update(status="initial_reload_mismatch", original_test_evaluated=False)
            results[condition] = record
            write_json(directory / "diagnosis.json", record)
            continue
        changes = {}
        surfaces = {**validation_surfaces(source, condition, False), **test_surfaces(source, condition)}
        for name, (reference, rows) in surfaces.items():
            if name not in initial:
                initial[name] = measure(model, encoded, rows, directory / "initial" / f"{name}.json")
            final = source.evaluation(reference, rows)
            changes[name] = paired_change(initial[name], final, rows)
            for phase, evaluation in (("initial", initial[name]), ("final-saved", final)):
                write_json(
                    directory / "explanations" / phase / f"{name}.json", explain(evaluation, rows, source, protocol)
                )
        fresh_initial = {
            name: measure(model, encoded, rows, directory / "fresh-initial" / f"{name}.json")
            for name, rows in panels.items()
        }
        final_path = f"transfer/{condition}/checkpoint"
        metadata = source.read_json(f"{final_path}/metadata.json")
        if metadata["updates"] != source.config["workflow_checkpoints"][-1]:
            raise RuntimeError(f"DIAGNOSIS_NOT_FINAL_CHECKPOINT: {condition}")
        restore_source(model, source, final_path)
        fresh_changes = {}
        for name, rows in panels.items():
            final = measure(model, encoded, rows, directory / "fresh-final" / f"{name}.json")
            fresh_changes[name] = paired_change(fresh_initial[name], final, rows)
            for phase, evaluation in (("fresh-initial", fresh_initial[name]), ("fresh-final", final)):
                write_json(
                    directory / "explanations" / phase / f"{name}.json", explain(evaluation, rows, source, protocol)
                )
        record.update(
            status="posthoc_diagnosed",
            original_test_evaluated=True,
            final_checkpoint=final_path,
            original_panels=changes,
            fresh_diagnostic_panels=fresh_changes,
        )
        results[condition] = record
        write_json(directory / "diagnosis.json", record)
        event(output / "events.jsonl", "posthoc_initial_diagnosis", condition=condition, training_updates=0)
    verify_dependencies(protocol)
    exact = all(row["status"] == "posthoc_diagnosed" for row in results.values())
    return finish(
        source,
        output,
        {
            "status": "posthoc_initial_diagnosis_complete" if exact else "posthoc_initial_reload_mismatch",
            "all_initial_dev_reloads_exact": exact,
            "conditions": results,
            "posthoc": True,
            "recipe_selection": False,
            "teacher_qualified": False,
            "interpretation": "Original panels compare newly generated initial-checkpoint records with immutable final records. Separate dispatched audits test final persistence. Fresh diagnostic scenes compare both checkpoints in this process and are not a new unbiased held-out result.",
        },
        model,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        result = diagnose(json.loads(args.config.read_text()), args.output_dir)
    except Exception:
        logging.exception("INITIAL_DIAGNOSIS_FAILED output=%s", args.output_dir)
        raise
    if not result["all_initial_dev_reloads_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
