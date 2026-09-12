from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import compression_spectrum as spectrum


def random_term(out_features, in_features, rank, scale, seed):
    rng = torch.Generator().manual_seed(seed)
    return spectrum.FactorTerm(
        torch.randn(rank, in_features, generator=rng, dtype=torch.float64),
        torch.randn(out_features, rank, generator=rng, dtype=torch.float64),
        scale,
    )


@pytest.mark.parametrize("shape", [(17, 13, 4), (4, 6, 9), (8, 8, 8)])
def test_factor_spectrum_matches_dense_svd(shape):
    term = random_term(*shape, 2.75, 93)
    actual = spectrum.factor_singular_values(term)
    expected = torch.linalg.svdvals(term.scale * (term.b @ term.a))
    torch.testing.assert_close(actual, expected[: len(actual)], rtol=1e-12, atol=1e-12)
    assert torch.linalg.vector_norm(expected[len(actual) :]) < 1e-12


def test_learned_delta_includes_asymmetric_ranks_and_scales():
    final = random_term(19, 23, 5, 2.0, 13)
    initial = random_term(19, 23, 3, 0.75, 29)
    actual = spectrum.factor_singular_values(
        final, spectrum.FactorTerm(initial.a, initial.b, -initial.scale)
    )
    dense = final.scale * (final.b @ final.a) - initial.scale * (initial.b @ initial.a)
    expected = torch.linalg.svdvals(dense)
    torch.testing.assert_close(actual, expected[: len(actual)], rtol=1e-12, atol=1e-12)


def test_nonsingular_gauge_preserves_spectrum():
    term = random_term(18, 15, 4, 2.0, 27)
    gauge = torch.tensor(
        [
            [2.0, 0.3, 0.0, 0.2],
            [0.0, 0.7, 0.4, 0.0],
            [0.0, 0.0, 1.2, -0.1],
            [0.0, 0.0, 0.0, 0.4],
        ],
        dtype=torch.float64,
    )
    transformed = spectrum.FactorTerm(
        torch.linalg.solve(gauge, term.a), term.b @ gauge, term.scale
    )
    torch.testing.assert_close(
        spectrum.factor_singular_values(term),
        spectrum.factor_singular_values(transformed),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rank8_residual_matches_dense_truncated_svd():
    term = random_term(21, 19, 12, 1.5, 21)
    dense = term.scale * (term.b @ term.a)
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    rank8 = (u[:, :8] * s[:8]) @ vh[:8]
    stats = spectrum.spectrum_statistics(
        spectrum.factor_singular_values(term), (21, 19), 8
    )
    assert stats["numerical_rank"] == 12
    assert stats["residual_frobenius_squared"] == pytest.approx(
        float((dense - rank8).square().sum()), rel=1e-12
    )
    assert stats["relative_frobenius_error"] == pytest.approx(
        float(
            torch.linalg.vector_norm(dense - rank8) / torch.linalg.vector_norm(dense)
        ),
        rel=1e-12,
    )


@pytest.mark.parametrize("zero_scale", [False, True])
def test_zero_update_has_finite_zero_residual(zero_scale):
    term = random_term(11, 14, 4, 0.0 if zero_scale else 2.0, 37)
    if not zero_scale:
        term.b.zero_()
    stats = spectrum.spectrum_statistics(
        spectrum.factor_singular_values(term), (11, 14), 8
    )
    assert stats["numerical_rank"] == 0
    assert stats["zero_energy"]
    assert stats["relative_frobenius_error"] == 0
    assert stats["residual_energy_fraction"] == 0
    json.dumps(stats, allow_nan=False)


def test_equal_updates_cancel_without_dense_materialization():
    term = random_term(17, 13, 4, 2.0, 91)
    values = spectrum.factor_singular_values(
        term, spectrum.FactorTerm(term.a, term.b, -term.scale)
    )
    assert float(torch.linalg.vector_norm(values)) < 1e-12


def test_weighted_residual_uses_energy_not_average_projection_error():
    rows = [
        {
            "final_adapter": {
                "numerical_rank": 2,
                **spectrum.residual_statistics(100.0, 1.0),
            }
        },
        {
            "final_adapter": {
                "numerical_rank": 3,
                **spectrum.residual_statistics(1.0, 1.0),
            }
        },
    ]
    result = spectrum.aggregate(rows, "final_adapter")
    assert result["residual_energy_fraction"] == pytest.approx(2 / 101)
    assert result["relative_frobenius_error"] == pytest.approx((2 / 101) ** 0.5)


def test_rejects_mismatched_and_nonfinite_factors():
    term = random_term(7, 9, 2, 2.0, 3)
    with pytest.raises(ValueError, match="TERM_SHAPE_MISMATCH"):
        spectrum.factor_singular_values(term, random_term(8, 9, 2, 2.0, 4))
    term.a[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE_FACTORS"):
        spectrum.factor_singular_values(term)


def checkpoint_fixture(root: Path) -> Path:
    config = {
        "name": "cpu_fixture",
        "native_rank": 2,
        "expected_projection_count": 2,
        "cpu_threads": 1,
        "checkpoints": {},
    }
    for label, seed in (("initial", 10), ("final", 20)):
        directory = root / label
        directory.mkdir(parents=True)
        ranks = {
            "model.layers.0.self_attn.q_proj": 3,
            "model.layers.0.self_attn.v_proj": 4,
        }
        if label == "initial":
            ranks = {path: rank - 1 for path, rank in ranks.items()}
        alphas = {
            path: rank * (2 if label == "final" else 3) for path, rank in ranks.items()
        }
        adapter_config = {
            "peft_type": "LORA",
            "bias": "none",
            "rank_pattern": ranks,
            "alpha_pattern": alphas,
        }
        (directory / "adapter_config.json").write_text(json.dumps(adapter_config))
        tensors = {}
        for path, rank in ranks.items():
            term = random_term(11, 13, rank, alphas[path] / rank, seed + rank)
            tensors[f"base_model.model.{path}.lora_A.weight"] = term.a
            tensors[f"base_model.model.{path}.lora_B.weight"] = term.b
        save_file(tensors, directory / "adapter_model.safetensors")
        config["checkpoints"][label] = {
            "origin": {"path": str(directory)},
            "local_path": label,
            "sha256": {
                name: spectrum.file_hash(directory / name)
                for name in ("adapter_config.json", "adapter_model.safetensors")
            },
        }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_run_freezes_inputs_before_measurement_and_preserves_scaling(
    tmp_path, monkeypatch
):
    config_path = checkpoint_fixture(tmp_path)
    output = tmp_path / "result"
    original = spectrum.factor_singular_values

    def checked(*terms):
        frozen = json.loads((output / "frozen-inputs.json").read_text())
        assert frozen["pins"][str(config_path)] == spectrum.file_hash(config_path)
        return original(*terms)

    monkeypatch.setattr(spectrum, "factor_singular_values", checked)
    result = spectrum.run(config_path, output)
    assert result == json.loads((output / "spectrum.json").read_text())
    assert result["frozen_inputs_sha256"] == spectrum.file_hash(
        output / "frozen-inputs.json"
    )
    assert result["inputs_unchanged_after_measurement"]
    assert len(result["projections"]) == 2
    for row in result["projections"]:
        assert row["initial"]["scale"] == 3
        assert row["final"]["scale"] == 2
        assert row["initial"]["factor_rank"] + 1 == row["final"]["factor_rank"]


def test_hash_mismatch_stops_before_loading_or_measurement(tmp_path, monkeypatch):
    config_path = checkpoint_fixture(tmp_path)
    (tmp_path / "final/adapter_config.json").write_text("{}")

    def forbidden(*args, **kwargs):
        pytest.fail("unverified input reached tensor loading")

    monkeypatch.setattr(spectrum, "checkpoint_factors", forbidden)
    with pytest.raises(ValueError, match="INPUT_HASH_MISMATCH"):
        spectrum.run(config_path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_input_mutation_during_measurement_prevents_completed_report(
    tmp_path, monkeypatch
):
    config_path = checkpoint_fixture(tmp_path)
    original = spectrum.factor_singular_values

    def mutate(*terms):
        config_path.write_text(config_path.read_text() + "\n")
        return original(*terms)

    monkeypatch.setattr(spectrum, "factor_singular_values", mutate)
    with pytest.raises(ValueError, match="INPUT_CHANGED_DURING_RUN"):
        spectrum.run(config_path, tmp_path / "result")
    assert not (tmp_path / "result/spectrum.json").exists()
