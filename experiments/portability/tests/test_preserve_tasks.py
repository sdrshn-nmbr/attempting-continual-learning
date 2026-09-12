import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from portallib import PortalModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_target_calibration import tiny_models

import follow_through as follow
import preserve_tasks as preservation
from calibrate_target import file_pin, pin_bundle
from data import SEQUENCE_TASKS, digest, write_json
from learner import frozen_base_tensors, save_native, state_hash, tensor_hash
from repair_alignment import product_loss

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preservation/heads_new_latents.json"


def configured():
    return json.loads(CONFIG.read_text())


def tiny_source(tmp_path):
    base, initial, _, _ = tiny_models()
    tasks = ("rte", *preservation.OLD_TASKS, *SEQUENCE_TASKS)
    z = initial.task_latents[0].detach()
    gen = torch.Generator().manual_seed(91209)
    vectors = torch.stack(
        [
            z + 0.2 * torch.randn(z.shape, generator=gen)
            if task in preservation.OLD_TASKS
            else z
            for task in tasks
        ]
    )
    model = PortalModel(replace(initial.config, tasks=tasks), vectors)
    model.core.load_state_dict(initial.core.state_dict())
    model.alignment.load_state_dict(initial.alignment.state_dict())
    model.requires_grad_(False).eval()
    path = tmp_path / "native"
    save_native(model, path)
    config = configured()
    config["inputs"]["source_initial"] = {
        **pin_bundle(path),
        "tensor_sha256": tensor_hash(model.state_dict()),
    }
    config["runtime"] = {
        "device": "cpu",
        "dtype": "float32",
        "autocast": False,
        "cpu_threads": 1,
    }
    config["role"] = "cpu_contract_test"
    config["expected_hooks"] = len(model.config.projection_targets)
    config["protocol"]["checkpoints"] = [0, 4, 8]
    config["protocol"]["gradient_gate_step"] = 4
    config["protocol"]["training_ids"] = {
        task: ids[:4] for task, ids in config["protocol"]["training_ids"].items()
    }
    config["protocol_sha256"] = digest(config["protocol"])
    return config, base, model


@pytest.mark.parametrize("gauge", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_product_loss_and_gradients_equal_dense_matrix_control(gauge, dtype):
    generator = torch.Generator().manual_seed(91210)
    reference_a = torch.randn((3, 11), generator=generator, dtype=dtype)
    reference_b = torch.randn((7, 3), generator=generator, dtype=dtype)
    if gauge:
        transform = torch.tensor(
            [[2.0, 0.2, 0.0], [0.0, 0.5, 0.1], [0.1, 0.0, 1.5]], dtype=dtype
        )
        a = torch.linalg.solve(transform, reference_a).detach().requires_grad_(True)
        b = (reference_b @ transform).detach().requires_grad_(True)
    else:
        a = torch.randn((3, 11), generator=generator, dtype=dtype, requires_grad=True)
        b = torch.randn((7, 3), generator=generator, dtype=dtype, requires_grad=True)
    scale, epsilon = 2.0, 1e-12
    implicit = product_loss(
        {(0, "q"): (a, b)}, {(0, "q"): (reference_a, reference_b)}, epsilon, scale
    )
    actual = scale * b.double() @ a.double()
    expected = scale * reference_b.double() @ reference_a.double()
    dense = (actual - expected).square().sum() / (expected.square().sum() + epsilon)
    implicit_gradients = torch.autograd.grad(implicit, (a, b))
    dense_gradients = torch.autograd.grad(dense, (a, b))
    torch.testing.assert_close(implicit, dense, atol=1e-12, rtol=1e-11)
    for actual_gradient, expected_gradient in zip(
        implicit_gradients, dense_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=1e-12,
            rtol=1e-6 if dtype == torch.float32 else 1e-10,
        )
    if gauge:
        assert (
            float(
                (
                    (a - reference_a).square().sum() + (b - reference_b).square().sum()
                ).detach()
            )
            > 1
        )
        assert float(implicit.detach()) < 1e-12


def test_preservation_config_has_exact_matched_arms_and_existing_inputs():
    config = configured()
    preservation.validate(config)
    previous = json.loads(
        (ROOT / "configs/follow_through/examples_released.json").read_text()
    )
    base_arm = next(
        arm
        for arm in previous["protocol"]["arms"]
        if arm["name"] == "heads_new_latents"
    )
    assert config["inputs"] == previous["inputs"]
    assert config["protocol"]["training_ids"] == previous["protocol"]["training_ids"]
    assert config["protocol"]["checkpoints"] == [0, 16, 64, 128, 256, 512]
    for arm in config["protocol"]["arms"]:
        assert {
            key: value
            for key, value in arm.items()
            if key not in {"name", "preservation_weight"}
        } == {key: value for key, value in base_arm.items() if key != "name"}


def test_reference_exact_identity_freezing_and_tamper_checks(tmp_path):
    config, _, model = tiny_source(tmp_path)
    references = preservation.MatrixReferences(model, config["expected_hooks"])
    arm = config["protocol"]["arms"][1]
    trainable, _, _ = follow.parameter_groups(model, arm)
    for task in preservation.OLD_TASKS:
        loss = references.loss(model, task)
        assert float(loss.detach()) == 0
        gradients = torch.autograd.grad(
            loss, tuple(trainable.values()), allow_unused=True
        )
        assert all(
            gradient is None or torch.count_nonzero(gradient) == 0
            for gradient in gradients
        )
    references.verify()
    assert all(not value.requires_grad for value in references.tensors().values())
    reference_a = next(iter(references.factors[preservation.OLD_TASKS[0]].values()))[0]
    reference_a.add_(0.01)
    with pytest.raises(ValueError, match="REFERENCE_TENSORS_CHANGED"):
        references.verify()


def test_zero_references_fail_before_training(tmp_path):
    config, _, model = tiny_source(tmp_path)
    with torch.no_grad():
        for head in model.core.B.values():
            head.weight.zero_()
            head.bias.zero_()
    with pytest.raises(ValueError, match="DEGENERATE_REFERENCE_MATRIX"):
        preservation.MatrixReferences(model, config["expected_hooks"])


def test_zero_weight_matches_existing_trainer_weights_optimizer_rng_and_base_calls(
    tmp_path, monkeypatch
):
    config, original_base, original_model = tiny_source(tmp_path)
    rows = follow.training_rows(config)
    monkeypatch.setattr(
        follow,
        "validation_rows",
        lambda config: [r for values in rows.values() for r in values],
    )
    arm = config["protocol"]["arms"][0]
    outputs = {}
    for name in ("existing", "preservation_control"):
        base, model = copy.deepcopy(original_base), copy.deepcopy(original_model)
        references = preservation.MatrixReferences(model, config["expected_hooks"])
        calls = [0]
        handle = base.model.register_forward_pre_hook(
            lambda *args, count=calls: count.__setitem__(0, count[0] + 1)
        )
        output = tmp_path / name
        output.mkdir()
        torch.manual_seed(9912)
        if name == "existing":
            result = follow.fit_arm(
                base,
                model,
                arm,
                config,
                rows,
                follow.schedule_for(config, rows),
                output,
            )
        else:
            result = preservation.fit_arm(
                base,
                model,
                arm,
                config,
                rows,
                follow.schedule_for(config, rows),
                output,
                references,
            )
        handle.remove()
        assert result["status"] == "completed_budget"
        outputs[name] = {
            "tensor": tensor_hash(model.state_dict()),
            "optimizer": state_hash(
                torch.load(output / "checkpoints/optimizer-0008.pt", weights_only=True)
            ),
            "rng": tensor_hash({"rng": torch.get_rng_state()}),
            "calls": calls[0],
            "base": tensor_hash(frozen_base_tensors(base.model)),
        }
        if name == "preservation_control":
            assert result["preservation_training_forward_calls"] == 32
            assert result["preservation_training_backward_calls"] == 0
    assert outputs["existing"] == outputs["preservation_control"]


def test_protected_training_reduces_matrix_drift_without_old_behavior_calls(
    tmp_path, monkeypatch
):
    config, original_base, original_model = tiny_source(tmp_path)
    rows = follow.training_rows(config)
    monkeypatch.setattr(
        follow,
        "validation_rows",
        lambda config: [r for values in rows.values() for r in values],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("old behavioral examples must not enter training")

    monkeypatch.setattr(follow, "retention_rows", forbidden)
    references = preservation.MatrixReferences(original_model, config["expected_hooks"])
    outcomes = {}
    for arm in config["protocol"]["arms"]:
        base, model = copy.deepcopy(original_base), copy.deepcopy(original_model)
        before = tensor_hash(frozen_base_tensors(base.model))
        before_vectors = model.task_latents.detach().clone()
        output = tmp_path / arm["name"]
        output.mkdir()
        trained = preservation.fit_arm(
            base,
            model,
            arm,
            config,
            rows,
            follow.schedule_for(config, rows),
            output,
            references,
        )
        assert trained["status"] == "completed_budget"
        assert trained["optimizer_updates"] == 8
        assert trained["old_behavioral_training_examples"] == 0
        assert trained["example_exposures_per_task"] == dict.fromkeys(
            SEQUENCE_TASKS, 32
        )
        assert tensor_hash(frozen_base_tensors(base.model)) == before
        for index, task in enumerate(model.config.tasks):
            if task not in SEQUENCE_TASKS:
                assert torch.equal(model.task_latents[index], before_vectors[index])
        outcomes[arm["name"]] = references.diagnostics(model)["mean"]
        references.verify()
    assert outcomes["preserved"] < outcomes["control"]


def test_complete_cpu_run_both_arms_new_pid_and_separate_retention_result(tmp_path):
    config, base, _ = tiny_source(tmp_path)
    config["protocol"]["checkpoints"] = [0, 4, 12]
    snapshot = tmp_path / "base" / base.revision
    snapshot.mkdir(parents=True)
    base.model.save_pretrained(snapshot)
    base.tokenizer.save_pretrained(snapshot)
    config["inputs"]["base"] = pin_bundle(snapshot)
    config["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "cache_dir": str(tmp_path / "cache"),
        "local_path": str(snapshot),
    }
    old = {"validation": [], "indices": []}
    examples = follow.validation_rows(config)[:4]
    for task, row in zip(preservation.OLD_TASKS, examples, strict=True):
        old["validation"].append(
            {
                "task": task,
                "prompt": row.prompt,
                "choices": list(row.choices),
                "gold_idx": row.gold_idx,
            }
        )
        old["indices"].append(
            {
                "task": task,
                "prompt_sha256": hashlib.sha256(
                    " ".join(row.prompt.split()).casefold().encode()
                ).hexdigest(),
            }
        )
    path = tmp_path / "retention.json"
    write_json(path, old)
    config["inputs"]["retention_probes"] = {"path": str(path), **file_pin(path)}
    config["protocol"]["retention_indices_sha256"] = digest(old["indices"])
    config["protocol_sha256"] = digest(config["protocol"])
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    output = tmp_path / "experiment"
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "preserve_tasks.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    recorded = json.loads((output / "result.json").read_text())
    assert recorded["training_pid"] != recorded["evaluation_pid"]
    assert set(recorded["arms"]) == {"control", "preserved"}
    assert all(arm["reload_predictions_exact"] for arm in recorded["arms"].values())
    assert all(
        arm["retention_steps"] == {"before": 0, "after": 12}
        for arm in recorded["arms"].values()
    )
    comparison = recorded["native_adapter_retention_regularization"]
    assert comparison["control"]["completed_planned_budget"]
    assert comparison["preserved"]["completed_planned_budget"]
    assert not comparison["joint_acquisition_and_retention_qualified"]
