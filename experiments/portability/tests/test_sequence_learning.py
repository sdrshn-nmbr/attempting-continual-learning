from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from portallib.evaluation import PortalInjector
from safetensors.torch import load_file
from test_contract import ROOT, tiny_model

import run
from data import SEQUENCE_TASKS, prepare_data, read_data, write_json
from learner import (
    SequenceLearner,
    evaluate,
    extend_portal,
    initialization_parity,
    load_base,
    make_persistent_lora,
    parameter_plan,
    probe_logits,
    state_hash,
    tensor_hash,
)
from metrics import sequence_metrics, transport_metrics


def configuration(method="native"):
    return json.loads((ROOT / f"configs/sequence_{method}_no_replay.json").read_text())


def make_learner(method, directory, config, dtype=torch.float32):
    base, initial = tiny_model(dtype=dtype)
    portal = extend_portal(initial, SEQUENCE_TASKS)
    plan = parameter_plan(portal)
    if method == "lora":
        model, receipt = make_persistent_lora(base, portal, directory / "rte_export")
        assert receipt["algebraic_effective_delta_equal"]
        assert all(all(checks.values()) for checks in receipt["factor_checks"].values())
        base = replace(base, model=model)
        portal = None
    return SequenceLearner(
        base, portal, SEQUENCE_TASKS, 0, run.learner_recipe(config, plan)
    ), plan


def fixture_rows(config, directory):
    prepare_data(config, directory)
    return read_data(
        directory,
        tuple(
            f"{task}_{split}"
            for task in SEQUENCE_TASKS
            for split in ("train", "validation", "test")
        ),
    )


def initial_checkpoint(config, directory):
    rows = fixture_rows(config, directory)
    learner, plan = make_learner(config["method"], directory, config)
    probes = [
        rows[f"{task}_validation"][index] for task in SEQUENCE_TASKS for index in (0, 1)
    ]
    run.save_checkpoint(learner, run.initial_paths(directory, config["method"]), probes)
    write_json(
        directory / "initialization_receipt.json",
        {"parity": {"passed": True}, "parameter_match": plan},
    )
    write_json(directory / "config.json", config)
    learner.close()
    return rows


@pytest.mark.parametrize("method", ["native", "lora"])
def test_reference_off_matches_current_only_with_rng_and_on_changes_update(
    method, tmp_path, monkeypatch
):
    config = configuration(method)
    rows = initial_checkpoint(config, tmp_path)
    monkeypatch.setattr(run, "load_base", lambda _: tiny_model()[0])
    learner = run.restore_learner(
        config, tmp_path, run.initial_paths(tmp_path, method), 0
    )
    for _ in range(3):
        learner.step(rows["sequence_a_train"][:4], [], 0.0)
    seed_checkpoint = tmp_path / "after_a"
    run.save_checkpoint(learner, seed_checkpoint, rows["sequence_a_validation"][:2])
    learner.close()
    outcomes = {}
    reference_norm = {}
    for mode, reference, weight in (
        ("current_only", [], 0.0),
        ("reference_off", rows["sequence_a_train"][4:8], 0.0),
        ("reference_on", rows["sequence_a_train"][4:8], 0.25),
    ):
        candidate = run.restore_learner(config, tmp_path, seed_checkpoint, 0)
        candidate.advance_stage()
        torch.manual_seed(999)
        for _ in range(3):
            result = candidate.step(rows["sequence_b_train"][:4], reference, weight)
        outcomes[mode] = (
            tensor_hash(candidate.named_parameters),
            state_hash(candidate.optimizer.state_dict()),
            tensor_hash({"rng": torch.get_rng_state()}),
        )
        reference_norm[mode] = result["reference_gradient_norm"]
        candidate.close()
    assert outcomes["current_only"] == outcomes["reference_off"]
    assert outcomes["reference_on"][:2] != outcomes["reference_off"][:2]
    assert outcomes["reference_on"][2] == outcomes["current_only"][2]
    assert reference_norm["reference_off"] > 0


@pytest.mark.parametrize("method", ["native", "lora"])
def test_96_step_optimizer_continuity_frozen_momentum_and_fresh_process_reload(
    method, tmp_path
):
    config = configuration(method)
    rows = fixture_rows(config, tmp_path)
    learner, plan = make_learner(method, tmp_path, config)
    write_json(
        tmp_path / "initialization_receipt.json",
        {"parity": {"passed": True}, "parameter_match": plan},
    )
    write_json(tmp_path / "config.json", config)
    old_vectors = {}
    old_moments = {}
    optimizer_identity = id(learner.optimizer)
    for stage, task in enumerate(SEQUENCE_TASKS):
        if stage:
            transition = learner.advance_stage()
            assert transition["optimizer_state_unchanged"]
            assert id(learner.optimizer) == optimizer_identity
            if method == "native":
                assert not learner.optimizer.state.get(learner.vectors[task])
        frozen_before = run.frozen_snapshot(learner)
        seen_extra = {}
        for current, reference, _ in run.stage_schedule(
            rows, stage, config["seed"], config["training"]
        ):
            result = learner.step(current, reference, 0.0)
            for name, count in result["extra_rank_gradient_nonzero_elements"].items():
                seen_extra[name] = seen_extra.get(name, False) or count > 0
        assert run.frozen_snapshot(learner) == frozen_before
        assert run.check_optimizer_steps(learner, 96)
        if method == "native":
            for previous in SEQUENCE_TASKS[:stage]:
                assert learner.vectors[previous].grad is None
                assert (
                    tensor_hash({"vector": learner.vectors[previous]})
                    == old_vectors[previous]
                )
                assert (
                    state_hash(learner.optimizer.state[learner.vectors[previous]])
                    == old_moments[previous]
                )
            vector = learner.vectors[task]
            assert torch.count_nonzero(learner.optimizer.state[vector]["exp_avg"])
            old_vectors[task] = tensor_hash({"vector": vector})
            old_moments[task] = state_hash(learner.optimizer.state[vector])
        else:
            assert len(seen_extra) == len(learner.active)
            assert all(seen_extra.values())
        checkpoint = tmp_path / "stages" / run.BOUNDARIES[stage + 1]
        probes = [
            rows[f"{name}_validation"][index]
            for name in SEQUENCE_TASKS[: stage + 1]
            for index in (0, 1)
        ]
        saved = run.save_checkpoint(learner, checkpoint, probes)
        assert saved["optimizer"]["storage"]["tensor_bytes"] > 0
        assert saved["storage"]["serialized_checkpoint_bytes"] > 0
        environment = {**os.environ, "PYTHONPATH": str(ROOT)}
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(Path(__file__).resolve()),
                "--reload",
                str(tmp_path),
                "--stage",
                str(stage),
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        reloaded = json.loads((checkpoint / "cpu_reload.json").read_text())
        assert reloaded["passed"] and reloaded["fresh_process"]
        assert reloaded["probe_logits"]["bitwise_equal"]
    learner.close()


@pytest.mark.parametrize("method", ["native", "lora"])
def test_full_runner_stage_reads_and_zero_reference_backward_are_valid(
    method, tmp_path, monkeypatch
):
    config = configuration(method)
    config["training"].update(steps=8, eval_every=8)
    initial_checkpoint(config, tmp_path)
    monkeypatch.setattr(run, "require_gpu", lambda *_: None)
    monkeypatch.setattr(run, "load_base", lambda _: tiny_model()[0])
    original_loss = SequenceLearner.loss
    original_read_text = Path.read_text

    def restricted_read_text(path, *args, **kwargs):
        if path.name in {"initialization.json", "baselines.json"}:
            raise AssertionError("TRAINER_OPENED_HELDOUT_INITIALIZATION_REPORT")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", restricted_read_text)

    def zero_reference(self, rows):
        result = original_loss(self, rows)
        if all(row.task != self.tasks[self.stage] for row in rows):
            return replace(result, loss=result.loss * 0)
        return result

    monkeypatch.setattr(SequenceLearner, "loss", zero_reference)
    original_read = run.read_data
    reads = []

    def record_read(output, splits):
        reads.append(splits)
        assert all(not name.endswith("_test") for name in splits)
        return original_read(output, splits)

    monkeypatch.setattr(run, "read_data", record_read)
    run.train(config, tmp_path)
    receipt = json.loads((tmp_path / "training_receipt.json").read_text())
    assert all(receipt["checks"].values())
    assert receipt["budget"]["reference_batches_with_nonzero_gradient"] == 0
    assert receipt["budget"]["reference_backward_calls"] == 4
    assert receipt["budget"]["reference_forward_backward_groups"] == 6
    assert receipt["budget"]["optimizer_updates"] == 24
    assert len(reads) == 3
    for stage, names in enumerate(reads):
        assert set(names) == {
            f"{task}_{split}"
            for task in SEQUENCE_TASKS[: stage + 1]
            for split in ("train", "validation")
        }
    assert len({stage["pid"] for stage in receipt["stages"].values()}) == 1


def test_fp32_generation_forwards_and_gradients_ignore_outer_autocast(tmp_path):
    config = configuration()
    rows = fixture_rows(config, tmp_path)
    learner, _ = make_learner("native", tmp_path / "native", config)
    base_contexts = []
    generation_contexts = []
    base_hook = learner.base.model.register_forward_pre_hook(
        lambda *_: base_contexts.append(torch.is_autocast_enabled("cpu"))
    )
    core_hook = learner.portal.core.l1.register_forward_pre_hook(
        lambda *_: generation_contexts.append(torch.is_autocast_enabled("cpu"))
    )
    validation = rows["sequence_a_validation"][:4]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        native = evaluate(learner.base, validation, 768, 4, learner.portal)
        native_probes = probe_logits(learner.base, validation[:2], learner.portal)
        result = learner.step(rows["sequence_a_train"][:4], [], 0.0)
    assert result["current_loss"] > 0
    assert base_contexts and not any(base_contexts)
    assert generation_contexts and not any(generation_contexts)
    assert all(value.dtype == torch.float32 for value in learner.active.values())
    base_hook.remove()
    core_hook.remove()
    learner.close()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lora, _ = make_learner("lora", tmp_path / "lora", config)
        initial = evaluate(lora.base, validation, 768, 4, None)
        probes = probe_logits(lora.base, validation[:2], None)
    parity = initialization_parity(native, initial, native_probes, probes)
    assert parity["probe"]["overall_max_abs"] <= 0.125
    assert parity["probe"]["overall_relative_l2"] <= 0.01
    observed = {}
    for _ in range(3):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            step = lora.step(rows["sequence_a_train"][:4], [], 0.0)
        for name, count in step["extra_rank_gradient_nonzero_elements"].items():
            observed[name] = observed.get(name, False) or count > 0
    assert all(observed.values())
    assert all(value.dtype == torch.float32 for value in lora.active.values())
    lora.close()


def test_cpu_autocast_counterexample_and_corrected_projection_parity(tmp_path):
    config = configuration()
    rows = fixture_rows(config, tmp_path)
    probes = [
        rows[f"{task}_validation"][index] for task in SEQUENCE_TASKS for index in (0, 1)
    ]
    base, initial = tiny_model()
    base.model.eval()
    portal = extend_portal(initial, SEQUENCE_TASKS)
    old_native = {}
    with PortalInjector(base.model, portal.config) as injector:
        for row in probes:
            inputs = base.tokenizer(row.prompt, return_tensors="pt")
            with (
                torch.no_grad(),
                torch.autocast("cpu", dtype=torch.bfloat16),
                injector.activate(portal.generate(row.task)),
            ):
                old_native[row.id] = base.model(**inputs).logits[0, -1].float()
    native_arithmetic = {}
    native = probe_logits(base, probes, portal, arithmetic=native_arithmetic)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        native_nested = probe_logits(base, probes, portal)
    model, embedding = make_persistent_lora(base, portal, tmp_path / "exported_rte")
    adapted = replace(base, model=model)
    old_lora = {}
    for row in probes:
        inputs = base.tokenizer(row.prompt, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            old_lora[row.id] = model(**inputs).logits[0, -1].float()
    projection_dtypes = []
    handles = [
        model.get_base_model()
        .get_submodule(path)
        .register_forward_hook(
            lambda module, inputs, output: projection_dtypes.append(
                (inputs[0].dtype, output.dtype, torch.is_autocast_enabled("cpu"))
            )
        )
        for _, path in portal.config.resolved_targets()
    ]
    lora_arithmetic = {}
    lora = probe_logits(adapted, probes, None, arithmetic=lora_arithmetic)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lora_nested = probe_logits(adapted, probes, None)
    for handle in handles:
        handle.remove()
    old_gap = max(
        float((old_native[key] - old_lora[key]).abs().max()) for key in native
    )
    gap = max(float((native[key] - lora[key]).abs().max()) for key in native)
    assert old_gap > 1e-4
    assert gap < 1e-6
    assert all(torch.equal(native[key], native_nested[key]) for key in native)
    assert all(torch.equal(lora[key], lora_nested[key]) for key in lora)
    assert projection_dtypes and set(projection_dtypes) == {
        (torch.float32, torch.float32, False)
    }
    for receipts in (native_arithmetic, lora_arithmetic):
        assert set(receipts) == set(native)
        assert all(
            row == {"logits_dtype": "torch.float32", "autocast_enabled": False}
            for row in receipts.values()
        )
    write_json(
        tmp_path / "counterexample.json",
        {
            "scope": "CPU counterexample to FP32 storage alone; not an AMD-kernel reproduction",
            "old_outer_autocast_max_abs": old_gap,
            "corrected_max_abs": gap,
            "projection_embedding_max_abs": embedding[
                "projection_probe_overall_max_abs"
            ],
            "outer_autocast_does_not_change_corrected_probes": True,
            "lora_projection_input_output_dtypes": ["torch.float32", "torch.float32"],
        },
    )


def test_load_base_explicitly_promotes_bf16_snapshot_to_fp32(tmp_path):
    original, _ = tiny_model(dtype=torch.bfloat16)
    snapshot = tmp_path / ("a" * 40)
    original.model.save_pretrained(snapshot)
    original.tokenizer.save_pretrained(snapshot)
    spec = {
        "local_path": str(snapshot),
        "revision": snapshot.name,
        "repo_id": original.model_id,
        "cache_dir": str(tmp_path / "cache"),
    }
    loaded = load_base(spec, device="cpu")
    assert not any(value.requires_grad for value in loaded.model.parameters())
    assert all(value.dtype == torch.float32 for value in loaded.model.parameters())
    for name, value in original.model.named_parameters():
        assert torch.equal(loaded.model.get_parameter(name), value.float())


def panel(counts):
    predictions = []
    for task, correct in zip(SEQUENCE_TASKS, counts, strict=True):
        for index in range(64):
            predictions.append(
                {
                    "id": f"{task}:{index}",
                    "task": task,
                    "group": str(index),
                    "gold": 0,
                    "correct": int(index < correct),
                    "prediction": int(index >= correct),
                    "scores": [0.0, -1.0, -2.0, -3.0],
                    "prompt_sha256": f"{task}:{index}",
                }
            )
    return {
        "metrics": {
            task: {"accuracy": count / 64, "examples": 64}
            for task, count in zip(SEQUENCE_TASKS, counts, strict=True)
        },
        "predictions": predictions,
    }


def matrix(values):
    return {
        boundary: {"evaluation": panel(counts)}
        for boundary, counts in zip(run.BOUNDARIES, values, strict=True)
    }


def test_acquisition_credits_forward_transfer_and_reports_own_stage_increment():
    timeline = matrix([(16, 16, 16), (32, 32, 16), (32, 33, 32), (32, 33, 33)])
    result = sequence_metrics(timeline, panel((16, 16, 16)), 1)
    assert result["sequence_learning_signal"]
    b = result["tasks"]["sequence_b"]
    assert b["acquired"]
    assert b["stage_acquisition_gain"] == 1 / 64
    assert b["threshold_reached_before_arrival"]
    assert b["acquisition_timing"] == "before_arrival_via_forward_transfer"
    assert result["tasks"]["sequence_a"]["acquisition_timing"] == "during_own_stage"


def test_no_retention_success_for_unacquired_tasks_or_failed_current_task():
    result = sequence_metrics(matrix([(16, 16, 16)] * 4), panel((16, 16, 16)), 1)
    assert not result["sequence_learning_signal"]
    assert not result["tasks"]["sequence_a"]["acquired_and_retained"]
    failed_c = sequence_metrics(
        matrix([(16, 16, 16), (32, 16, 16), (32, 32, 16), (32, 32, 16)]),
        panel((16, 16, 16)),
        1,
    )
    assert not failed_c["sequence_learning_signal"]


def test_every_later_boundary_is_checked_without_macro_masking():
    result = sequence_metrics(
        matrix([(16, 16, 16), (32, 16, 16), (20, 32, 16), (32, 40, 32)]),
        panel((16, 16, 16)),
        1,
    )
    assert not result["sequence_learning_signal"]
    assert result["tasks"]["sequence_a"]["maximum_forgetting"] == 12 / 64
    assert (
        result["tasks"]["sequence_a"]["later_boundaries"]["after_c"]["forgetting"] == 0
    )
    raw_floor = sequence_metrics(
        matrix([(16, 16, 16), (32, 16, 16), (32, 32, 16), (32, 32, 32)]),
        panel((32, 16, 16)),
        1,
    )
    assert not raw_floor["tasks"]["sequence_a"]["acquired"]


def test_target_gain_threshold_is_separate_and_uses_both_target_floors():
    result = transport_metrics(
        matrix([(20, 20, 20), (24, 20, 20), (24, 24, 20), (24, 24, 24)]),
        panel((16, 16, 16)),
        1,
    )
    assert result["after_c_target_gain_signal"]
    assert result["sequence_learning_signal"] is None
    assert (
        result["target_gain_by_boundary"]["after_c"]["tasks"]["sequence_a"][
            "gain_over_both_target_floors"
        ]
        == 4 / 64
    )
    stronger_raw = transport_metrics(
        matrix([(16, 16, 16), (24, 16, 16), (24, 24, 16), (24, 24, 24)]),
        panel((24, 16, 16)),
        1,
    )
    assert not stronger_raw["after_c_target_gain_signal"]


@pytest.mark.parametrize("method", ["native", "lora"])
def test_complete_cpu_runner_initialization_evaluation_and_result_schema(
    method, tmp_path, monkeypatch
):
    config = configuration(method)
    config["training"].update(steps=8, eval_every=8)
    fixture_rows(config, tmp_path)
    write_json(tmp_path / "config.json", config)
    monkeypatch.setattr(run, "require_gpu", lambda *_: None)
    monkeypatch.setattr(run, "load_base", lambda _: tiny_model()[0])
    monkeypatch.setattr(run, "load_portal", lambda _: tiny_model()[1])
    original_read_text = Path.read_text

    def forbid_heldout_read(path, *args, **kwargs):
        if path.name.endswith("_test.json") or path.name == "baselines.json":
            raise AssertionError("HELDOUT_READ_BEFORE_POSTTRAINING_EVALUATION")
        return original_read_text(path, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "read_text", forbid_heldout_read)
        run.initialize(config, tmp_path)
        run.train(config, tmp_path)
    initialization = json.loads((tmp_path / "initialization.json").read_text())
    assert initialization["parity"]["passed"]
    assert initialization["heldout_reads"] == []
    assert "baselines" not in initialization
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(Path(__file__).resolve()),
            "--baselines",
            str(tmp_path),
            "--stage",
            "0",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    baselines = json.loads((tmp_path / "baselines.json").read_text())
    identity = baselines["initial_checkpoint_identity"]
    for candidate in ("native", "lora"):
        assert (
            identity[f"{candidate}_sha256"]
            == initialization["checkpoints"][candidate]["checkpoint_sha256"]
        )
    original_vectors = load_file(
        tmp_path / "initial/native/checkpoint/model.safetensors"
    )["task_latents"]
    assert identity["native_vectors_sha256"] == tensor_hash(
        {"task_vectors": original_vectors}
    )
    assert (
        identity["raw_base_sha256"]
        == initialization["checkpoints"]["native"]["frozen"]["base_sha256"]
    )
    final_checkpoint = json.loads(
        (tmp_path / "stages/after_c/checkpoint_receipt.json").read_text()
    )
    assert final_checkpoint["checkpoint_sha256"] != identity[f"{method}_sha256"]
    for stage in range(3):
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(Path(__file__).resolve()),
                "--evaluate",
                str(tmp_path),
                "--stage",
                str(stage),
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
            capture_output=True,
            text=True,
        )
    run.finish(config, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["qualified"]
    assert result["parent_paired_review_required"]
    assert set(evaluation["timeline"]) == set(run.BOUNDARIES)
    for boundary in run.BOUNDARIES:
        assert set(evaluation["timeline"][boundary]["evaluation"]["metrics"]) == set(
            SEQUENCE_TASKS
        )
        if method == "native":
            assert set(
                evaluation["timeline"][boundary]["transport"]["evaluation"]["metrics"]
            ) == set(SEQUENCE_TASKS)
    if method == "lora":
        assert not result["transport"]["measured"]


@pytest.mark.parametrize("method", ["native", "lora"])
def test_mutated_heldout_gold_prompts_and_predictions_cannot_change_training(
    method, tmp_path, monkeypatch
):
    config = configuration(method)
    config["training"].update(steps=8, eval_every=8)
    clean = tmp_path / "clean"
    mutated = tmp_path / "mutated"
    initial_checkpoint(config, clean)
    shutil.copytree(clean, mutated)
    for task in SEQUENCE_TASKS:
        path = mutated / "data" / f"{task}_test.json"
        rows = json.loads(path.read_text())
        for row in rows:
            row["gold_idx"] = (row["gold_idx"] + 1) % 4
            row["prompt"] = "THIS HELDOUT INPUT MUST NEVER ENTER TRAINING"
        write_json(path, rows)
    write_json(
        mutated / "baselines.json", {"predictions": [{"gold": 999, "scores": [999.0]}]}
    )
    write_json(
        mutated / "initialization.json",
        {"predictions": [{"gold": 999, "scores": [999.0]}]},
    )
    monkeypatch.setattr(run, "require_gpu", lambda *_: None)
    monkeypatch.setattr(run, "load_base", lambda _: tiny_model()[0])
    original_read_text = Path.read_text

    def guard(path, *args, **kwargs):
        if path.name.endswith("_test.json") or path.name in {
            "baselines.json",
            "initialization.json",
        }:
            raise AssertionError("TRAINER_DESERIALIZED_HELDOUT_PAYLOAD")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard)
    for directory in (clean, mutated):
        run.train(config, directory)
    left = json.loads((clean / "training_receipt.json").read_text())
    right = json.loads((mutated / "training_receipt.json").read_text())
    assert left == right
    for boundary in run.BOUNDARIES[1:]:
        before = left["stages"][boundary]["checkpoint"]
        after = right["stages"][boundary]["checkpoint"]
        assert before["checkpoint_sha256"] == after["checkpoint_sha256"]
        assert before["optimizer"]["state_sha256"] == after["optimizer"]["state_sha256"]


def test_metrics_reject_identity_or_panel_drift():
    timeline = matrix([(16, 16, 16), (32, 16, 16), (32, 32, 16), (32, 32, 32)])
    changed = copy.deepcopy(timeline)
    changed["after_c"]["evaluation"]["predictions"][0]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="COMPARISON_IDENTITY_MISMATCH"):
        sequence_metrics(changed, panel((16, 16, 16)), 1)
    changed = copy.deepcopy(timeline)
    changed["after_c"]["evaluation"]["metrics"].pop("sequence_b")
    with pytest.raises(ValueError, match="SEQUENCE_TASK_MATRIX_INCOMPLETE"):
        sequence_metrics(changed, panel((16, 16, 16)), 1)


def test_parity_gate_rejects_validation_disagreement_and_large_logit_difference():
    native = panel((32, 32, 32))
    identical_probes = {"x": torch.ones(100)}
    assert initialization_parity(native, native, identical_probes, identical_probes)[
        "passed"
    ]
    changed = copy.deepcopy(native)
    for row in changed["predictions"][:3]:
        row["prediction"] = 1 - row["prediction"]
        row["correct"] = 1 - row["correct"]
    assert not initialization_parity(
        native, changed, identical_probes, identical_probes
    )["passed"]
    assert not initialization_parity(
        native, native, identical_probes, {"x": torch.ones(100) + 0.2}
    )["passed"]


def reload_cpu(directory, stage):
    config = json.loads((directory / "config.json").read_text())
    run.load_base = lambda _: tiny_model()[0]
    checkpoint = directory / "stages" / run.BOUNDARIES[stage + 1]
    learner = run.restore_learner(config, directory, checkpoint, stage)
    try:
        receipt = run.reload_probes(
            learner, directory, checkpoint, SEQUENCE_TASKS[: stage + 1]
        )
    finally:
        learner.close()
    write_json(checkpoint / "cpu_reload.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--reload", type=Path)
    phases.add_argument("--evaluate", type=Path)
    phases.add_argument("--baselines", type=Path)
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()
    if args.reload is not None:
        reload_cpu(args.reload, args.stage)
    else:
        run.require_gpu = lambda *_: None
        run.load_base = lambda _: tiny_model()[0]
        run.load_portal = lambda _: tiny_model()[1]
        if args.baselines is not None:
            run.evaluate_baselines(
                json.loads((args.baselines / "config.json").read_text()), args.baselines
            )
        else:
            run.evaluate_checkpoint(
                json.loads((args.evaluate / "config.json").read_text()),
                args.evaluate,
                args.stage,
            )
