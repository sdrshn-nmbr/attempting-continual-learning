from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from portallib import PortalConfig, PortalModel, PortalProjectionTarget
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

import repair_alignment as repair

CANONICAL = (
    Path(repair.__file__).parent / "configs/repair/sequence303_native_replay.json"
)


def canonical_config():
    return json.loads(CANONICAL.read_text())


def tiny_configuration(name, layers, width):
    projections = tuple(
        PortalProjectionTarget(
            layer, module, f"self_attn.{module}_proj", width, output, module, module
        )
        for layer in range(layers)
        for module, output in (("q", width), ("v", width // 2))
    )
    return PortalConfig(
        base_model_name_or_path=name,
        tasks=repair.PUBLISHED_TASKS,
        n_layers=layers,
        projection_targets=projections,
        rank=2,
        alpha=4,
        d_z=5,
        d_layer=3,
        hidden=6,
        d_core=4,
    )


def fixture_config(directory):
    config = canonical_config()
    config["fit"].update(steps=10, cpu_threads=1)
    torch.manual_seed(709)
    latents = torch.randn(14, 5)
    source = PortalModel(tiny_configuration("source", 3, 12), latents)
    with torch.no_grad():
        for parameter in source.core.parameters():
            parameter.normal_(0, 0.2)
    initial = PortalModel(
        replace(
            source.config,
            tasks=repair.PUBLISHED_TASKS + ("sequence_a", "sequence_b", "sequence_c"),
        ),
        torch.cat((latents, torch.randn(3, 5))),
    )
    initial.core.load_state_dict(source.core.state_dict())
    initial.alignment.load_state_dict(source.alignment.state_dict())
    learned = copy.deepcopy(initial)
    with torch.no_grad():
        for parameter in learned.core.B.parameters():
            parameter.mul_(1.3)
        learned.task_latents[-3:].add_(3)
    config["source_initial"] = repair.save_checkpoint(
        initial, directory / "source_initial"
    )
    config["source_learned"] = repair.save_checkpoint(
        learned, directory / "source_learned"
    )
    for name, layers, width in (("qwen4", 2, 8), ("mistral7", 3, 10)):
        target = PortalModel(tiny_configuration(name, layers, width), latents)
        target.core.load_state_dict(source.core.state_dict())
        config["targets"][name] = repair.save_checkpoint(target, directory / name)
    return config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_implicit_value_and_all_four_factor_gradients_match_dense(dtype):
    torch.manual_seed(91)
    factors = tuple(
        torch.randn(*shape, dtype=dtype, requires_grad=True)
        for shape in ((2, 5), (7, 2), (2, 5), (7, 2))
    )
    a, b, reference_a, reference_b = factors
    dense = (
        (b.double() @ a.double() - reference_b.double() @ reference_a.double())
        .square()
        .sum()
    )
    implicit = repair.product_squared_error(*factors)
    torch.testing.assert_close(implicit, dense, atol=1e-10, rtol=1e-12)
    dense_gradients = torch.autograd.grad(dense, factors)
    implicit_gradients = torch.autograd.grad(implicit, factors)
    for observed, expected in zip(implicit_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(
            observed,
            expected,
            atol=1e-10 if dtype == torch.float64 else 1e-5,
            rtol=1e-10 if dtype == torch.float64 else 1e-6,
        )


def test_independent_nonsingular_factor_gauges_preserve_value_and_gradients():
    torch.manual_seed(19)
    factors = tuple(
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in ((2, 5), (7, 2), (2, 5), (7, 2))
    )
    a, b, reference_a, reference_b = factors
    left = torch.tensor([[1.3, 0.2], [-0.1, 0.7]], dtype=torch.float64)
    right = torch.tensor([[0.9, -0.4], [0.2, 1.1]], dtype=torch.float64)
    original = repair.product_squared_error(*factors)
    changed = repair.product_squared_error(
        left @ a,
        b @ torch.linalg.inv(left),
        right @ reference_a,
        reference_b @ torch.linalg.inv(right),
    )
    torch.testing.assert_close(original, changed, rtol=1e-12, atol=1e-10)
    for expected, observed in zip(
        torch.autograd.grad(original, factors),
        torch.autograd.grad(changed, factors),
        strict=True,
    ):
        torch.testing.assert_close(expected, observed, rtol=1e-11, atol=1e-10)


def test_identical_fp32_factors_have_exact_zero_value_and_all_gradients():
    torch.manual_seed(10)
    a, b = torch.randn(2, 5), torch.randn(7, 2)
    factors = tuple(value.clone().requires_grad_() for value in (a, b, a, b))
    loss = repair.product_squared_error(*factors)
    assert loss.item() == 0
    assert all(
        torch.count_nonzero(gradient).item() == 0
        for gradient in torch.autograd.grad(loss, factors)
    )


@pytest.mark.parametrize("epsilon", [1e-12, 1.0])
def test_projection_normalization_and_equal_projection_weighting(epsilon):
    a = torch.tensor([[2.0, 1.0]], requires_grad=True)
    b = torch.tensor([[3.0], [1.0]], requires_grad=True)
    generated = {"small": (a, b), "large": (a * 100, b * 100)}
    reference = {
        "small": (a.detach() * 0.5, b.detach()),
        "large": (a.detach() * 100, b.detach() * 100),
    }
    dense_losses = []
    for key, (left, right) in generated.items():
        teacher_left, teacher_right = reference[key]
        teacher = 2 * teacher_right.double() @ teacher_left.double()
        dense_losses.append(
            ((2 * right.double() @ left.double() - teacher).square().sum())
            / (teacher.square().sum() + epsilon)
        )
    torch.testing.assert_close(
        repair.product_loss(generated, reference, epsilon, scale=2),
        torch.stack(dense_losses).mean(),
    )


def test_canonical_nested_config_is_the_only_interface():
    assert (
        repair.file_hash(CANONICAL)
        == "0c895276ae8bcd45ea2309e3dd0688e766a2156b4694206a75afe8eeef82ab97"
    )
    config = canonical_config()
    repair.validate_config(config)
    with pytest.raises(ValueError, match="CONFIG_KEYS"):
        repair.validate_config({"target_initial_checkpoint": "/tmp/old-flat-schema"})
    invalid = copy.deepcopy(config)
    invalid["mismatch_task_map"]["rte"] = "truthfulqa"
    with pytest.raises(ValueError, match="CYCLIC_TRAIN_ONLY"):
        repair.validate_config(invalid)
    invalid = copy.deepcopy(config)
    invalid["train_tasks"][0] = "sequence_a"
    with pytest.raises(ValueError, match="PUBLISHED_TASKS_ONLY"):
        repair.validate_config(invalid)
    invalid = copy.deepcopy(config)
    invalid["fit"]["device"] = "cuda:0"
    with pytest.raises(ValueError, match="FIXED_FIT_CONTRACT"):
        repair.validate_config(invalid)


def test_frozen_schedule_balances_every_cycle_and_every_control():
    config = canonical_config()
    schedule = repair.make_schedule(config["train_tasks"], config["fit"])
    assert len(schedule) == 100
    for start in range(0, 100, 10):
        assert Counter(
            task for batch in schedule[start : start + 10] for task in batch
        ) == Counter(config["train_tasks"])
    assert Counter(task for batch in schedule for task in batch) == {
        task: 10 for task in config["train_tasks"]
    }
    assert schedule == repair.make_schedule(config["train_tasks"], config["fit"])


def test_complete_cpu_run_identity_freeze_isolation_and_actual_disk_reload(
    tmp_path, monkeypatch
):
    config = fixture_config(tmp_path / "inputs")
    config["fit"]["steps"] = 100
    output = tmp_path / "output"
    original_open = Path.open
    original_safe_open = repair.safe_open
    original_train = repair.train_alignment
    original_factors = repair.task_factors
    active_training = False
    reads = []
    slices = []
    optimizer_calls = []
    original_optimizer = torch.optim.AdamW

    def forbid(*args, **kwargs):
        raise AssertionError("NO_LLM_OR_TOKENIZER_LOAD_ALLOWED")

    for cls in (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoTokenizer,
        PortalModel,
    ):
        monkeypatch.setattr(cls, "from_pretrained", forbid)

    def checked_open(path, mode="r", *args, **kwargs):
        if "r" in mode:
            metadata = (
                path.name == "METADATA"
                and path.parent.name.startswith(("portallib-", "safetensors-"))
                and path.parent.name.endswith(".dist-info")
            )
            assert metadata or path.name in {
                "config.json",
                "model.safetensors",
                "repair_alignment.py",
            }
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    class CheckedSlice:
        def __init__(self, inner):
            self.inner = inner

        def get_shape(self):
            return self.inner.get_shape()

        def __getitem__(self, index):
            assert index == slice(None, 14)
            slices.append(index)
            return self.inner[index]

    class CheckedArtifact:
        def __init__(self, *args, **kwargs):
            self.inner = original_safe_open(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def metadata(self):
            return self.inner.metadata()

        def keys(self):
            return self.inner.keys()

        def get_slice(self, name):
            assert name == "task_latents"
            return CheckedSlice(self.inner.get_slice(name))

        def get_tensor(self, name):
            assert name.startswith(("core.", "alignment."))
            return self.inner.get_tensor(name)

    def checked_factors(model, task):
        if active_training:
            assert task in config["train_tasks"]
        return original_factors(model, task)

    def checked_train(*args, **kwargs):
        nonlocal active_training
        active_training = True
        try:
            return original_train(*args, **kwargs)
        finally:
            active_training = False

    def checked_optimizer(parameters, **kwargs):
        optimizer_calls.append(kwargs)
        return original_optimizer(parameters, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    monkeypatch.setattr(repair, "safe_open", CheckedArtifact)
    monkeypatch.setattr(repair, "task_factors", checked_factors)
    monkeypatch.setattr(repair, "train_alignment", checked_train)
    monkeypatch.setattr(torch.optim, "AdamW", checked_optimizer)
    result = repair.run(config, output)
    assert result["status"] == "completed" and result["input_files_reverified_exact"]
    assert (
        result["target_base_loaded"] is False
        and result["added_latent_rows_loaded"] == 0
    )
    assert slices and reads
    assert len(optimizer_calls) == 6
    assert all(
        call == {"lr": 1e-4, "weight_decay": 0.0, "eps": 1e-8, "foreach": False}
        for call in optimizer_calls
    )
    for target in result["targets"].values():
        for name, arm in target["arms"].items():
            assert arm["optimizer_updates"] == 100
            assert arm["schedule_sha256"] == result["schedule_sha256"]
            assert arm["frozen_before"] == arm["frozen_after"]
            assert arm["reload"]["state_and_all_old_task_metrics_exact"]
            assert all(
                parameter.startswith("alignment.")
                for parameter in arm["trainable_names"]
            )
            assert arm["alignment_carrier"]["tasks"] == list(repair.PUBLISHED_TASKS)
            assert arm["task_mapping"] == (
                config["mismatch_task_map"]
                if name == "mismatch"
                else {task: task for task in config["train_tasks"]}
            )
            assert arm["optimizer"]["gradient_clip"] == 1
        identity = target["arms"]["identity"]
        assert identity["alignment_max_abs_delta"] == 0
        assert identity["alignment_before_sha256"] == identity["alignment_after_sha256"]
        assert all(
            row["loss"] == row["maximum_alignment_gradient"] == 0
            for row in identity["training"]
        )
        assert identity["evaluation"]["holdout"]["max_normalized_squared_error"] == 0
        repaired = target["arms"]["repair"]
        assert (
            repaired["evaluation"]["train"]["mean_normalized_squared_error"]
            < repaired["initial_evaluation"]["train"]["mean_normalized_squared_error"]
        )
        assert (
            repaired["initial_evaluation"]
            == target["arms"]["mismatch"]["initial_evaluation"]
        )


def test_poisoned_new_vectors_do_not_change_fit_or_output_carrier(tmp_path):
    config = fixture_config(tmp_path / "inputs")
    original = repair.run(config, tmp_path / "original")
    for key in ("source_initial", "source_learned"):
        spec = config[key]
        weights = Path(spec["path"]) / "model.safetensors"
        state = load_file(weights)
        state["task_latents"][14:] = float("nan")
        save_file(
            state, weights, metadata={"format": "portallib", "format_version": "1"}
        )
        spec["files_sha256"]["model.safetensors"] = repair.file_hash(weights)
    poisoned = repair.run(config, tmp_path / "poisoned")
    assert (
        original["inputs"]["source_learned"]["files_sha256"]
        != poisoned["inputs"]["source_learned"]["files_sha256"]
    )
    assert (
        original["inputs"]["source_learned"]["used_tensor_sha256"]
        == poisoned["inputs"]["source_learned"]["used_tensor_sha256"]
    )
    for target in ("qwen4", "mistral7"):
        for arm in repair.ARMS:
            before, after = (
                result["targets"][target]["arms"][arm]
                for result in (original, poisoned)
            )
            for key in (
                "training",
                "initial_evaluation",
                "evaluation",
                "frozen_before",
                "frozen_after",
                "alignment_after_sha256",
            ):
                assert before[key] == after[key]
            assert (
                before["checkpoint"]["files_sha256"]["config.json"]
                == after["checkpoint"]["files_sha256"]["config.json"]
            )
            states = [
                load_file(Path(arm_result["checkpoint"]["path"]) / "model.safetensors")
                for arm_result in (before, after)
            ]
            assert repair.tensor_hash(states[0]) == repair.tensor_hash(states[1])


@pytest.mark.parametrize(
    "spec_name", ["source_initial", "source_learned", "qwen4", "mistral7"]
)
@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_every_supplied_file_pin_is_enforced_before_training(
    tmp_path, spec_name, filename
):
    config = fixture_config(tmp_path / "inputs")
    repair.checkpoint_specs(config)[spec_name]["files_sha256"][filename] = "0" * 64
    with pytest.raises(ValueError, match="FILE_SHA256_MISMATCH"):
        repair.run(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("tensor", ["core.l1.weight", "task_latents"])
def test_original_core_and_published_vector_equality_are_required(tmp_path, tensor):
    config = fixture_config(tmp_path / "inputs")
    spec = config["targets"]["qwen4"]
    weights = Path(spec["path"]) / "model.safetensors"
    state = load_file(weights)
    state[tensor].add_(0.1)
    save_file(state, weights, metadata={"format": "portallib", "format_version": "1"})
    spec["files_sha256"]["model.safetensors"] = repair.file_hash(weights)
    with pytest.raises(ValueError, match="CORE_MISMATCH|PUBLISHED_LATENTS_CHANGED"):
        repair.run(config, tmp_path / "output")


def test_reload_corruption_fails_without_completed_result(tmp_path, monkeypatch):
    config = fixture_config(tmp_path / "inputs")
    output = tmp_path / "output"
    original = repair.make_model

    def corrupt_reload(target, shared):
        model = original(target, shared)
        if str(output) in shared.receipt["path"]:
            with torch.no_grad():
                next(model.alignment.parameters()).add_(0.1)
        return model

    monkeypatch.setattr(repair, "make_model", corrupt_reload)
    with pytest.raises(ValueError, match="CHECKPOINT_RELOAD_MISMATCH"):
        repair.run(config, output)
    assert not (output / "result.json").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"


def gate_arm(train_mean, holdout_mean, task_errors):
    return {
        "initial_evaluation": {
            "train": {"mean_normalized_squared_error": 1.0},
            "holdout": {
                "mean_normalized_squared_error": 1.0,
                "tasks": {
                    "a": {"mean_normalized_squared_error": 0.5},
                    "b": {"mean_normalized_squared_error": 1.5},
                },
            },
        },
        "evaluation": {
            "train": {
                "mean_normalized_squared_error": train_mean,
                "max_normalized_squared_error": train_mean,
            },
            "holdout": {
                "mean_normalized_squared_error": holdout_mean,
                "max_normalized_squared_error": max(task_errors),
                "tasks": {
                    task: {"mean_normalized_squared_error": value}
                    for task, value in zip(("a", "b"), task_errors, strict=True)
                },
            },
        },
        "alignment_max_abs_delta": 0,
        "frozen_before": {"hash": "same"},
        "frozen_after": {"hash": "same"},
        "reload": {"state_and_all_old_task_metrics_exact": True},
    }


def test_mean_improvement_cannot_hide_a_worsened_heldout_task():
    arms = {
        "identity": gate_arm(0, 0, [0, 0]),
        "repair": gate_arm(0.4, 0.4, [0.6, 0.2]),
        "mismatch": gate_arm(1, 1, [1, 1]),
    }
    result = repair.apply_gates(arms, canonical_config()["gates"])
    assert (
        result["checks"]["training_improvement"]
        and result["checks"]["heldout_improvement"]
    )
    assert not result["checks"]["every_heldout_task_nonworsening"]
    assert not result["passed"]


def test_negligible_baseline_is_inconclusive_and_never_nan():
    arms = {name: gate_arm(0, 0, [0, 0]) for name in repair.ARMS}
    for split in ("train", "holdout"):
        arms["repair"]["initial_evaluation"][split]["mean_normalized_squared_error"] = 0
    result = repair.apply_gates(arms, canonical_config()["gates"])
    assert result["status"] == "no_substantive_geometry_error" and not result["passed"]
    assert result["final_over_initial"] == {"train": None, "holdout": None}
    json.dumps(result, allow_nan=False)


def test_cli_runs_nested_two_target_config_and_records_exact_config_bytes(tmp_path):
    config = fixture_config(tmp_path / "inputs")
    path = tmp_path / "config.json"
    repair.write_json(path, config)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--python",
            sys.executable,
            repair.__file__,
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        env={
            **os.environ,
            "UV_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "output/result.json").read_text())
    assert receipt["config_file_sha256"] == repair.file_hash(path)
    assert set(receipt["targets"]) == {"qwen4", "mistral7"}
    assert all(
        list(target["arms"]) == sorted(repair.ARMS)
        for target in receipt["targets"].values()
    )
