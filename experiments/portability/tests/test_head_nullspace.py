import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from portallib import PortalModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_preserve_tasks import tiny_source

import follow_through as follow
import head_nullspace as head
import train_head_nullspace as training
from calibrate_target import pin_bundle
from data import SEQUENCE_TASKS, digest, write_json
from learner import tensor_hash
from preserve_tasks import OLD_TASKS, MatrixReferences

ROOT = Path(__file__).resolve().parents[1]


def setup_source(tmp_path):
    original, base, source = tiny_source(tmp_path)
    config = json.loads((ROOT / "configs/head_nullspace/source.json").read_text())
    config["inputs"]["source_initial"] = original["inputs"]["source_initial"]
    config["runtime"] = original["runtime"]
    config["role"] = "cpu_contract_test"
    config["expected_hooks"] = len(source.config.projection_targets)
    config["protocol"]["checkpoints"] = [0, 4, 12]
    config["protocol"]["gradient_gate_step"] = 4
    config["protocol"]["training_ids"] = {
        task: ids[:4] for task, ids in config["protocol"]["training_ids"].items()
    }
    config["protocol_sha256"] = digest(config["protocol"])
    return config, base, source


def test_augmented_basis_rank_and_identity_start(tmp_path):
    _, _, source = setup_source(tmp_path)
    spaces, proof = head.feature_space(source)
    design, basis = spaces["old_design"], spaces["null_basis"]
    assert design.shape == (4 * source.config.n_layers, source.config.hidden + 1)
    assert torch.equal(design[:, -1], torch.ones(design.shape[0], dtype=torch.float64))
    assert proof["rank"] + proof["null_dimension"] == source.config.hidden + 1
    torch.testing.assert_close(
        design @ basis,
        torch.zeros(design.shape[0], basis.shape[1], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert all(
        row["rank"] == source.config.n_layers
        for row in proof["ABC_projected_features"].values()
    )
    reference = MatrixReferences(source, len(source.config.projection_targets))
    for name, space in (
        ("head_only_control", spaces["full_basis"]),
        ("head_nullspace", basis),
    ):
        adapter = head.HeadSpacePortal(copy.deepcopy(source), space)
        named, _, vectors = head.trainable_plan(
            adapter, {"learning_rate": 1e-4, "latent_learning_rate": 2e-4}
        )
        assert (
            sum(p.numel() for p in named.values())
            == proof["trainable_parameters"][name]
        )
        assert adapter.head_basis.numel() * 8 == proof["basis_bytes_per_learner"][name]
        audit = head.conservation(adapter, reference)
        assert audit["passed"]
        assert all(row["all_factors_bitwise_equal"] for row in audit["tasks"].values())
        assert set(vectors) == set(SEQUENCE_TASKS)


def test_factorized_affine_update_equals_dense_value_and_gradients_including_bias():
    generator = torch.Generator().manual_seed(714)
    hidden = torch.randn(
        5, 9, generator=generator, dtype=torch.float64, requires_grad=True
    )
    basis = torch.linalg.qr(
        torch.randn(10, 6, generator=generator, dtype=torch.float64)
    ).Q
    coordinates = torch.randn(
        7, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    original = torch.randn(7, 10, generator=generator, dtype=torch.float64)
    augmented = F.pad(hidden, (0, 1), value=1)
    residual = F.linear(augmented, original) + F.linear(augmented @ basis, coordinates)
    dense = F.linear(augmented, original + coordinates @ basis.T)
    torch.testing.assert_close(residual, dense, atol=1e-12, rtol=1e-12)
    left = torch.autograd.grad(
        residual.square().sum(), (hidden, coordinates), retain_graph=True
    )
    right = torch.autograd.grad(dense.square().sum(), (hidden, coordinates))
    for actual, expected in zip(left, right, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)
    assert (coordinates @ basis.T)[:, -1].abs().sum() > 0


def test_adam_coordinates_preserve_old_factors_and_exact_effective_matrices(tmp_path):
    config, _, source = setup_source(tmp_path)
    spaces, _ = head.feature_space(source)
    reference = MatrixReferences(source, config["expected_hooks"])
    outcomes = {}
    for name, space in (
        ("head_only_control", spaces["full_basis"]),
        ("head_nullspace", spaces["null_basis"]),
    ):
        adapter = head.HeadSpacePortal(copy.deepcopy(source), space)
        arm = next(arm for arm in config["protocol"]["arms"] if arm["name"] == name)
        named, groups, vectors = head.trainable_plan(
            adapter, {**arm, "learning_rate": 0.01, "latent_learning_rate": 0.002}
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0, foreach=False)
        frozen = training.frozen_signature(adapter)
        before = source.generate(SEQUENCE_TASKS[0])
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            generated = follow.factors(adapter, SEQUENCE_TASKS[0], arm, vectors)
            loss = sum(
                (value - (target + 0.02)).square().mean()
                for key in generated
                for value, target in zip(generated[key], before[key], strict=True)
            )
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in named.values()
                if p.grad is not None
            )
            optimizer.step()
        follow.sync_vectors(adapter, vectors)
        assert training.frozen_signature(adapter) == frozen
        assert any(
            parameter.abs().sum() > 0 for parameter in adapter.coordinates.values()
        )
        audit = head.conservation(adapter, reference)
        outcomes[name] = audit
        if name == "head_nullspace":
            assert audit["passed"]
            for task in OLD_TASKS:
                actual = adapter.generate(task)
                dense_errors = []
                for key, (a, b) in actual.items():
                    ra, rb = reference.factors[task][key]
                    delta = reference.scale * b.double() @ a.double()
                    original = reference.scale * rb.double() @ ra.double()
                    dense_errors.append(
                        float(
                            torch.linalg.norm(delta - original)
                            / torch.linalg.norm(original)
                        )
                    )
                assert audit["tasks"][task][
                    "matrix_relative_frobenius_max"
                ] == pytest.approx(max(dense_errors), abs=1e-12)
            saved = head.save_adapter(adapter, tmp_path / "saved")
            restored = head.restore_adapter(saved, "cpu")
            assert tensor_hash(adapter.state_dict()) == tensor_hash(
                restored.state_dict()
            )
            assert head.conservation(restored, reference) == audit
    assert not outcomes["head_only_control"]["passed"]


def test_degenerate_feature_prerequisites_reject_before_training(tmp_path):
    _, _, source = setup_source(tmp_path)
    small = PortalModel(replace(source.config, hidden=2), source.task_latents)
    with pytest.raises(ValueError, match="INSUFFICIENT_NULL_DIMENSION"):
        head.feature_space(small)
    with torch.no_grad():
        source.task_latents[:] = source.task_latents[0].clone()
    with pytest.raises(ValueError, match="INSUFFICIENT_NEW_TASK_FEATURES"):
        head.feature_space(source)


@pytest.mark.parametrize("eligible", [False, True])
def test_new_pid_pipeline_and_control_acquisition_gate(tmp_path, eligible):
    config, base, _ = setup_source(tmp_path)
    snapshot = tmp_path / "base" / base.revision
    snapshot.mkdir(parents=True)
    base.model.save_pretrained(snapshot)
    base.tokenizer.save_pretrained(snapshot)
    config["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "local_path": str(snapshot),
        "cache_dir": str(tmp_path / "cache"),
    }
    config["inputs"]["base"] = pin_bundle(snapshot)
    if eligible:
        config["protocol"]["acquisition_train_floor"] = 0.0
        config["protocol"]["acquisition_validation_floor"] = 0.0
    config["protocol_sha256"] = digest(config["protocol"])
    write_json(tmp_path / "config.json", config)
    output = tmp_path / "experiment"
    executed = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "train_head_nullspace.py"),
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert executed.returncode == 0, executed.stdout[-4000:] + executed.stderr[-4000:]
    result = json.loads((output / "result.json").read_text())
    receipt = json.loads((output / "training_receipt.json").read_text())
    assert result["training_pid"] != result["evaluation_pid"]
    protected = result["arms"]["head_nullspace"]
    if eligible:
        assert protected["status"] == "completed_budget"
        assert protected["old_choice_predictions_all_preserved"]
        assert protected["retention_steps"] == {"before": 0, "after": 12}
        assert protected["checkpoints"]["12"]["conservation"]["passed"]
        saved = receipt["arms"]["head_nullspace"]["checkpoints"]["12"]["saved"]
        adapter = head.restore_adapter(saved, "cpu")
        original = follow.load_source(config, "source_initial")
        indices = [
            i
            for i, task in enumerate(adapter.config.tasks)
            if task not in SEQUENCE_TASKS
        ]
        assert torch.equal(
            adapter.task_latents[indices], original.task_latents[indices]
        )
    else:
        assert protected["status"] == "skipped_control_acquisition_unqualified"
        assert protected["optimizer_updates"] == 0
        assert not (output / "head_nullspace").exists()


def test_sealed_production_protocol_and_frozen_dependency_pins():
    config = json.loads((ROOT / "configs/head_nullspace/source.json").read_text())
    training.validate(config)
    assert config["protocol"]["acquisition_train_floor"] == 0.9
    assert config["protocol"]["acquisition_validation_floor"] == 0.85
    assert config["protocol"]["checkpoints"][-1] == 512
