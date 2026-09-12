from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

import compression_spectrum as spectrum
import rank_control as control


def random_term(shape, scale=2.0, seed=83):
    out_features, in_features, rank = shape
    rng = torch.Generator().manual_seed(seed)
    return spectrum.FactorTerm(
        torch.randn(rank, in_features, generator=rng, dtype=torch.float64),
        torch.randn(out_features, rank, generator=rng, dtype=torch.float64),
        scale,
    )


def dense_best8(term):
    dense = term.scale * (term.b.double() @ term.a.double())
    u, values, vh = torch.linalg.svd(dense, full_matrices=False)
    return (u[:, :8] * values[:8]) @ vh[:8]


@pytest.mark.parametrize("shape", [(23, 19, 12), (5, 7, 10), (15, 12, 4)])
@pytest.mark.parametrize("scale", [0.75, 2.0])
def test_thin_fit_matches_independent_dense_svd(shape, scale):
    term = random_term(shape, scale)
    fitted, values = control.best_rank8(term)
    assert fitted.a.shape == (8, shape[1])
    assert fitted.b.shape == (shape[0], 8)
    assert fitted.scale == 2
    torch.testing.assert_close(
        fitted.scale * (fitted.b @ fitted.a), dense_best8(term), rtol=1e-11, atol=1e-11
    )
    assert values.dtype == torch.float64


def test_nonorthogonal_gauge_preserves_best_rank8_product():
    term = random_term((22, 25, 12), seed=21)
    gauge = torch.diag(torch.linspace(0.3, 2.5, 12, dtype=torch.float64))
    gauge[0, 1] = 0.4
    gauge[4, 7] = -0.3
    transformed = spectrum.FactorTerm(
        torch.linalg.solve(gauge, term.a), term.b @ gauge, term.scale
    )
    original, _ = control.best_rank8(term)
    equivalent, _ = control.best_rank8(transformed)
    torch.testing.assert_close(
        original.b @ original.a, equivalent.b @ equivalent.a, rtol=1e-11, atol=1e-11
    )


def test_embedded_initial_rank8_is_preserved():
    term = random_term((23, 19, 12))
    term.b[:, 8:].zero_()
    fitted, _ = control.best_rank8(term)
    torch.testing.assert_close(
        fitted.scale * (fitted.b @ fitted.a),
        term.scale * (term.b @ term.a),
        rtol=1e-11,
        atol=1e-11,
    )


def test_zero_update_stays_zero_with_rank8_shapes():
    term = random_term((17, 21, 12))
    term.b.zero_()
    fitted, values = control.best_rank8(term)
    assert torch.count_nonzero(fitted.a) == 0
    assert torch.count_nonzero(fitted.b) == 0
    assert torch.count_nonzero(values) == 0


def test_nonfinite_input_is_rejected():
    term = random_term((12, 14, 10))
    term.a[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE_FACTORS"):
        control.best_rank8(term)


def test_tensor_hash_matches_existing_learner_contract():
    state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
            [[1.0, -2.0], [0.5, 0.0]], dtype=torch.float32
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
            [[3.0, 4.0]], dtype=torch.float32
        ),
    }
    assert (
        control.tensor_hash(state)
        == "f70ab0020fd7600a01b017436b2a96efcdd7e60a1c28715b2bd9e18ef19dec62"
    )


def source_fixture(root: Path) -> tuple[dict, dict, Path]:
    config = json.loads(
        (
            Path(control.__file__).parent / "configs/consolidation/rank_control303.json"
        ).read_text()
    )
    config["expected_projection_count"] = 4
    config["cpu_threads"] = 1
    config["base_identity"] = {
        "base_model_name_or_path": "frozen-tiny-qwen",
        "revision": "fixed-revision",
    }
    config["checkpoints"] = {}
    ranks = {
        f"model.layers.{layer}.self_attn.{projection}": rank
        for layer in range(2)
        for projection, rank in (("q_proj", 12), ("v_proj", 10))
    }
    sealed = {}
    for label, seed in (("initial", 71), ("final", 99)):
        path = root / label
        path.mkdir(parents=True)
        adapter_config = LoraConfig(
            r=12,
            lora_alpha=24,
            target_modules=["q_proj", "v_proj"],
            rank_pattern=ranks,
            alpha_pattern={key: 2 * rank for key, rank in ranks.items()},
            task_type="CAUSAL_LM",
            inference_mode=True,
            base_model_name_or_path="frozen-tiny-qwen",
            revision="fixed-revision",
        )
        adapter_config.save_pretrained(path)
        weights = {}
        for index, (name, rank) in enumerate(ranks.items()):
            term = random_term(
                (32 if name.endswith("q_proj") else 16, 32, rank), seed=seed + index
            )
            a, b = (term.a * 0.05).float(), (term.b * 0.05).float()
            if label == "initial":
                b[:, 8:].zero_()
            weights[f"base_model.model.{name}.lora_A.weight"] = a
            weights[f"base_model.model.{name}.lora_B.weight"] = b
            if label == "final":
                values = torch.linalg.svdvals(2 * (b.double() @ a.double()))
                sealed[name] = {
                    "path": name,
                    "final_adapter": {
                        "frobenius_squared": float(values.square().sum()),
                        "residual_frobenius_squared": float(values[8:].square().sum()),
                    },
                }
        save_file(
            weights, path / "adapter_model.safetensors", metadata={"format": "pt"}
        )
        config["checkpoints"][label] = {
            "local_path": label,
            "origin": str(path),
            "sha256": {
                name: spectrum.file_hash(path / name)
                for name in ("adapter_config.json", "adapter_model.safetensors")
            },
        }
    reference_path = root / "sealed.json"
    reference_path.write_text(json.dumps({"projections": list(sealed.values())}))
    config["frozen_references"] = {
        "spectrum_implementation": {
            "path": str(Path(spectrum.__file__).resolve()),
            "sha256": spectrum.file_hash(Path(spectrum.__file__)),
        },
        "spectrum": {
            "path": "sealed.json",
            "sha256": spectrum.file_hash(reference_path),
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config, sealed, config_path


def test_real_peft_reload_named_adapters_matches_dense_merged_tiny_qwen(tmp_path):
    config, _, _ = source_fixture(tmp_path)
    fitted = {}
    originals = {}
    for label in ("initial", "final"):
        _, originals[label] = spectrum.checkpoint_factors(tmp_path / label)
        fitted[label] = {
            name: control.best_rank8(term)[0] for name, term in originals[label].items()
        }
        control.save_adapter(
            LoraConfig.from_pretrained(tmp_path / label),
            fitted[label],
            tmp_path / f"rank8_{label}",
        )
    torch.manual_seed(33)
    base = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
                use_cache=False,
            )
        )
        .float()
        .eval()
    )
    reloaded = PeftModel.from_pretrained(
        copy.deepcopy(base),
        tmp_path / "rank8_final",
        adapter_name="rank8_final",
        is_trainable=False,
        local_files_only=True,
    )
    reloaded.load_adapter(
        tmp_path / "rank8_initial",
        adapter_name="rank8_initial",
        is_trainable=False,
        torch_device="cpu",
    )
    inputs = torch.tensor([[2, 5, 8, 11], [4, 9, 3, 7]])
    for label in ("initial", "final"):
        reloaded.set_adapter(f"rank8_{label}", inference_mode=True)
        reloaded.requires_grad_(False).eval()
        expected = copy.deepcopy(base)
        with torch.no_grad():
            for name, term in originals[label].items():
                expected.get_submodule(name).weight.add_(dense_best8(term).float())
            torch.testing.assert_close(
                reloaded(inputs).logits, expected(inputs).logits, rtol=2e-5, atol=2e-6
            )
        state = get_peft_model_state_dict(
            reloaded, adapter_name=f"rank8_{label}", save_embedding_layers=False
        )
        saved = load_file(tmp_path / f"rank8_{label}/adapter_model.safetensors")
        assert control.tensor_hash(state) == control.tensor_hash(saved)
        assert all(torch.equal(state[name], value) for name, value in saved.items())
        assert not any(parameter.requires_grad for parameter in reloaded.parameters())
        saved_config = LoraConfig.from_pretrained(tmp_path / f"rank8_{label}")
        assert (
            saved_config.base_model_name_or_path
            == config["base_identity"]["base_model_name_or_path"]
        )
        assert saved_config.revision == "fixed-revision"


def test_complete_proof_uses_peft_reloaded_state_and_preserves_inputs(
    tmp_path, monkeypatch
):
    config, _, path = source_fixture(tmp_path)
    output = tmp_path / "actual"
    original_fit = control.best_rank8

    def checked(term):
        assert (output / "frozen-inputs.json").exists()
        return original_fit(term)

    monkeypatch.setattr(control, "best_rank8", checked)
    receipt = control.run(path, output)
    proof = json.loads((output / "proof.json").read_text())
    assert receipt["passed"] and proof["passed"]
    assert proof["all_inputs_unchanged"]
    for label in ("initial", "final"):
        artifact = receipt["artifacts"][f"rank8_{label}"]
        saved = load_file(output / label / "adapter/adapter_model.safetensors")
        assert artifact["tensor_sha256"] == control.tensor_hash(saved)
        assert artifact["tensor_count"] == 8
        assert artifact["rank"] == 8 and artifact["scaling"] == 2
        assert proof["checkpoints"][label]["peft_reload_exact"]
        for row in proof["checkpoints"][label]["projections"]:
            assert row["peft_probe_max_abs_error"] == 0
            assert row["fp32_vs_fp64_fit_probe_max_abs"] >= 0
        for name, digest in config["checkpoints"][label]["sha256"].items():
            assert spectrum.file_hash(tmp_path / label / name) == digest


def test_projection_carrier_stores_no_dense_base_weights(tmp_path):
    source_fixture(tmp_path)
    _, terms = spectrum.checkpoint_factors(tmp_path / "final")
    fitted = {name: control.best_rank8(term)[0] for name, term in terms.items()}
    control.save_adapter(
        LoraConfig.from_pretrained(tmp_path / "final"), fitted, tmp_path / "rank8"
    )
    model = control.reload_projection_carrier(tmp_path / "rank8", fitted)
    for name in terms:
        layer = model.get_submodule(f"base_model.model.{name}")
        assert layer.base_layer.weight.is_sparse
        assert layer.base_layer.weight._nnz() == 0
        assert layer.lora_A.default.weight.dtype == torch.float32


def test_bad_sealed_residual_is_rejected_before_save(tmp_path):
    config, sealed, _ = source_fixture(tmp_path)
    next(iter(sealed.values()))["final_adapter"]["residual_frobenius_squared"] += 1
    with pytest.raises(ValueError, match="SEALED_SPECTRUM_MISMATCH"):
        control.project_checkpoint(
            tmp_path / "final", tmp_path / "result", config, "final", sealed
        )
    assert not (tmp_path / "result").exists()


def test_input_hash_mismatch_stops_before_svd(tmp_path, monkeypatch):
    _, _, path = source_fixture(tmp_path)
    (tmp_path / "initial/adapter_config.json").write_text("{}")

    def forbidden(*args):
        pytest.fail("unverified checkpoint reached SVD")

    monkeypatch.setattr(control, "best_rank8", forbidden)
    with pytest.raises(ValueError, match="INPUT_HASH_MISMATCH"):
        control.run(path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_changed_input_prevents_ready_receipt(tmp_path, monkeypatch):
    _, _, path = source_fixture(tmp_path)
    original_fit = control.best_rank8

    def mutate(term):
        path.write_text(path.read_text() + "\n")
        return original_fit(term)

    monkeypatch.setattr(control, "best_rank8", mutate)
    with pytest.raises(ValueError, match="INPUT_CHANGED_DURING_RUN"):
        control.run(path, tmp_path / "result")
    assert not (tmp_path / "result/receipt.json").exists()


def test_rank_selection_is_fixed_at_eight(tmp_path):
    config, _, path = source_fixture(tmp_path)
    config["rank"] = 4
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="FIXED_RANK8_ALPHA16_REQUIRED"):
        control.run(path, tmp_path / "result")
