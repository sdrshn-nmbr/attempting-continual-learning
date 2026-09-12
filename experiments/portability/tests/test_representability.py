from __future__ import annotations

import copy
import json
from pathlib import Path

import compression_spectrum as spectrum
import pytest
import rank_control
import representability as study
import torch
from peft import LoraConfig
from portallib import PortalConfig, PortalModel, PortalProjectionTarget
from safetensors.torch import load_file


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(117)


def random_matrix(*shape):
    return torch.randn(shape, dtype=torch.float64)


def canonical_config():
    return json.loads(
        (
            Path(study.__file__).parent
            / "configs/representability/source_rank8_303.json"
        ).read_text()
    )


def tiny_portal(layers=3):
    config = PortalConfig(
        base_model_name_or_path="synthetic/no-base-model",
        tasks=("rte", "control"),
        n_layers=layers,
        projection_targets=tuple(
            PortalProjectionTarget(
                layer, name, f"self_attn.{name}_proj", 13, width, name, name
            )
            for layer in range(layers)
            for name, width in (("q", 13), ("v", 7))
        ),
        rank=8,
        alpha=16,
        d_z=5,
        d_layer=4,
        hidden=12,
        d_core=6,
    )
    model = PortalModel(config, random_matrix(2, 5)).double().requires_grad_(False)
    for parameter in model.parameters():
        parameter.normal_(0, 0.25)
    return model


def dense(term):
    return term.scale * (term.b.double() @ term.a.double())


@pytest.mark.parametrize("source_rank,target_rank", [(3, 3), (5, 2), (5, 8)])
@pytest.mark.parametrize("source_scale", [0.7, -2.0])
def test_projection_matches_independent_dense_pseudoinverse_and_svd(
    source_rank, target_rank, source_scale
):
    inputs, outputs = random_matrix(4, 11), random_matrix(9, 5)
    term = spectrum.FactorTerm(
        random_matrix(source_rank, 11), random_matrix(9, source_rank), source_scale
    )
    result = study.project(
        term,
        study.alignment_span(inputs.T),
        study.alignment_span(outputs),
        target_rank,
        2.0,
    )
    expected = (
        outputs
        @ torch.linalg.pinv(outputs)
        @ dense(term)
        @ torch.linalg.pinv(inputs)
        @ inputs
    )
    u, values, vh = torch.linalg.svd(expected, full_matrices=False)
    expected = (u[:, :target_rank] * values[:target_rank]) @ vh[:target_rank]
    oracle_error = float((dense(term) - expected).square().sum())
    torch.testing.assert_close(
        dense(result.effective), expected, rtol=1e-11, atol=1e-11
    )
    lifted = 2 * outputs @ result.canonical_b @ result.canonical_a @ inputs
    torch.testing.assert_close(lifted, expected, rtol=1e-11, atol=1e-11)
    assert result.metrics["residual_frobenius_squared"] == pytest.approx(
        oracle_error, rel=1e-11, abs=1e-11
    )
    assert result.metrics["attained_residual_squared"] == pytest.approx(
        oracle_error, rel=1e-11, abs=1e-11
    )
    assert result.metrics["energy_partition_relative_error"] < 1e-12
    assert result.canonical_a.shape == (target_rank, inputs.shape[0])
    assert result.canonical_b.shape == (outputs.shape[1], target_rank)


def test_nonorthogonal_source_gauge_preserves_minimum_and_product():
    inputs, outputs = (
        study.alignment_span(random_matrix(11, 4)),
        study.alignment_span(random_matrix(9, 5)),
    )
    term = spectrum.FactorTerm(random_matrix(3, 11), random_matrix(9, 3), 1.7)
    gauge = torch.tensor(
        [[0.5, 0.3, 0.0], [0.0, 2.0, -0.7], [0.0, 0.0, 1.5]], dtype=torch.float64
    )
    equivalent = spectrum.FactorTerm(
        torch.linalg.solve(gauge, term.a), term.b @ gauge, term.scale
    )
    before, after = (
        study.project(item, inputs, outputs, 2, 2.0) for item in (term, equivalent)
    )
    torch.testing.assert_close(
        dense(before.effective), dense(after.effective), rtol=1e-11, atol=1e-11
    )
    assert before.metrics["residual_frobenius_squared"] == pytest.approx(
        after.metrics["residual_frobenius_squared"], rel=1e-12
    )


def test_alignment_coordinate_gauge_cannot_change_feasible_subspace():
    inputs, outputs = random_matrix(4, 11), random_matrix(9, 5)
    term = spectrum.FactorTerm(random_matrix(3, 11), random_matrix(9, 3), 2)
    input_change = random_matrix(4, 4) + 5 * torch.eye(4, dtype=torch.float64)
    output_change = random_matrix(5, 5) + 5 * torch.eye(5, dtype=torch.float64)
    first = study.project(
        term, study.alignment_span(inputs.T), study.alignment_span(outputs), 3, 2
    )
    changed = study.project(
        term,
        study.alignment_span((input_change @ inputs).T),
        study.alignment_span(outputs @ output_change),
        3,
        2,
    )
    torch.testing.assert_close(
        dense(first.effective), dense(changed.effective), rtol=1e-11, atol=1e-11
    )


def test_zero_adapter_and_full_spans_have_correct_minima():
    identity = study.alignment_span(torch.eye(7, dtype=torch.float64))
    term = spectrum.FactorTerm(random_matrix(3, 7), random_matrix(7, 3), 2)
    projected = study.project(term, identity, identity, 3, 2)
    assert projected.metrics["residual_frobenius_squared"] == 0
    zero = spectrum.FactorTerm(term.a, torch.zeros_like(term.b), 2)
    result = study.project(zero, identity, identity, 3, 2)
    assert result.metrics["zero_energy"]
    assert torch.count_nonzero(dense(result.effective)) == 0


def test_exact_outside_span_control_has_one_hundred_percent_residual():
    inputs = torch.eye(5, dtype=torch.float64)[:2]
    outputs = torch.eye(6, dtype=torch.float64)[:, :2]
    a = torch.eye(5, dtype=torch.float64)[4:5]
    b = torch.eye(6, dtype=torch.float64)[:, 5:6]
    result = study.project(
        spectrum.FactorTerm(a, b, 2),
        study.alignment_span(inputs.T),
        study.alignment_span(outputs),
        1,
        2,
    )
    assert result.metrics["residual_energy_fraction"] == 1
    assert result.metrics["output_span_residual_squared"] == 4
    assert result.metrics["additional_input_span_residual_squared"] == 0


def test_rank_deficiency_and_nonfinite_alignments_cannot_support_impossibility_claim():
    deficient = torch.zeros(6, 3, dtype=torch.float64)
    deficient[:2, :2] = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="RANK_UNRESOLVED_NO_IMPOSSIBILITY_CLAIM"):
        study.alignment_span(deficient)
    ambiguous = torch.diag(torch.tensor([1.0, 0.5, 1e-18], dtype=torch.float64))
    with pytest.raises(ValueError, match="RANK_UNRESOLVED_NO_IMPOSSIBILITY_CLAIM"):
        study.alignment_span(ambiguous)
    with pytest.raises(ValueError, match="NONFINITE"):
        study.alignment_span(torch.full((3, 3), float("nan")))


def test_independent_canonical_optimum_is_lower_than_arbitrary_feasible_candidates():
    inputs, outputs = random_matrix(4, 11), random_matrix(9, 5)
    term = spectrum.FactorTerm(random_matrix(3, 11), random_matrix(9, 3), 2)
    result = study.project(
        term, study.alignment_span(inputs.T), study.alignment_span(outputs), 2, 2
    )
    for _ in range(5):
        candidate = 2 * outputs @ random_matrix(5, 2) @ random_matrix(2, 4) @ inputs
        assert (
            float((dense(term) - candidate).square().sum())
            >= result.metrics["residual_frobenius_squared"]
        )


def test_shared_heads_construct_all_layers_with_exact_frozen_state_and_native_reload(
    tmp_path,
):
    model = tiny_portal()
    before = copy.deepcopy(model.state_dict())
    targets = {
        path: (random_matrix(8, 6), random_matrix(6, 8))
        for _, path in model.config.resolved_targets()
    }
    result = study.fit_shared_heads(model, targets, canonical_config())
    assert result["status"] == "constructed"
    assert result["linear_solves"] == 4
    assert result["optimizer_steps"] == 0
    changed = {
        key
        for key, value in model.state_dict().items()
        if not torch.equal(value, before[key])
    }
    assert changed == set(canonical_config()["fit"]["trainable"])
    native_path = tmp_path / "native"
    assert study.save_native(model, native_path)["reload_exact"]
    restored = study.load_native(native_path)
    assert all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    assert torch.get_default_dtype() == torch.float32
    generated = study.generated_terms(restored, "rte")
    for target, path in restored.config.resolved_targets():
        a, b = targets[path]
        oracle = (
            2
            * restored.alignment.output[target.output_group]
            @ b
            @ a
            @ restored.alignment.input[target.input_group]
        )
        torch.testing.assert_close(dense(generated[path]), oracle, rtol=1e-9, atol=1e-9)


def test_shared_head_rank_failure_is_inconclusive_and_does_not_mutate_model():
    model = tiny_portal()
    model.alignment.layer_embeddings.weight.zero_()
    before = rank_control.tensor_hash(model.state_dict())
    targets = {
        path: (random_matrix(8, 6), random_matrix(6, 8))
        for _, path in model.config.resolved_targets()
    }
    result = study.fit_shared_heads(model, targets, canonical_config())
    assert result["status"] == "inconclusive_hidden_design"
    assert result["linear_solves"] == 0
    assert result["impossibility_claim"] is False
    assert rank_control.tensor_hash(model.state_dict()) == before


def test_separate_initial_generator_positive_control_with_nonzero_adapter():
    model = tiny_portal()
    generated = study.generated_terms(model, "rte")
    assert all(study.energy(term) > 0 for term in generated.values())
    for target, path in model.config.resolved_targets():
        result = study.project(
            generated[path],
            study.alignment_span(model.alignment.input[target.input_group].T),
            study.alignment_span(model.alignment.output[target.output_group]),
            8,
            2,
        )
        assert result.metrics["relative_frobenius_error"] < 1e-12


def test_serialized_native_and_peft_fp32_injection_cover_every_exact_path(tmp_path):
    model = tiny_portal()
    terms = study.generated_terms(model, "rte")
    config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"
    )
    receipt = study.serialization_probe(
        model.config, config, terms, tmp_path / "adapter", canonical_config()
    )
    assert receipt["passed"]
    assert receipt["matrix_count"] == 6
    assert {row["path"] for row in receipt["rows"]} == set(terms)
    assert max(row["native_relative_l2"] for row in receipt["rows"]) < 1e-6
    assert max(row["peft_relative_l2"] for row in receipt["rows"]) < 1e-6
    assert all(row["serialization_relative_frobenius"] > 0 for row in receipt["rows"])


def test_missing_projection_is_rejected_before_serialization(tmp_path):
    model = tiny_portal()
    terms = study.generated_terms(model, "rte")
    terms.pop(next(iter(terms)))
    with pytest.raises(ValueError, match="INJECTION_MATRIX_COVERAGE"):
        study.serialization_probe(
            model.config,
            LoraConfig(r=8),
            terms,
            tmp_path / "adapter",
            canonical_config(),
        )
    assert not (tmp_path / "adapter").exists()


def test_fit_gate_requires_both_aggregate_and_every_matrix():
    gates = canonical_config()["gates"]
    assert study.fit_gate(
        {
            "relative_frobenius_error": 0.005,
            "maximum_matrix_relative_frobenius_error": 0.02,
        },
        gates,
    )
    assert not study.fit_gate(
        {
            "relative_frobenius_error": 0.005,
            "maximum_matrix_relative_frobenius_error": 0.06,
        },
        gates,
    )
    assert not study.fit_gate(
        {
            "relative_frobenius_error": 0.02,
            "maximum_matrix_relative_frobenius_error": 0.02,
        },
        gates,
    )


def test_protocol_and_input_tampering_rejected_before_study_or_start_receipt(
    tmp_path, monkeypatch
):
    inputs = tmp_path / "input.bin"
    inputs.write_bytes(b"frozen-before-outcomes")
    protocol_path = tmp_path / "protocol.json"
    study.write_json(
        protocol_path, {"pins": [study.pin(inputs)], "runtime": study.runtime()}
    )
    sealed_hash = study.file_hash(protocol_path)
    calls = []
    monkeypatch.setattr(study, "study", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="PROTOCOL_HASH_MISMATCH"):
        study.run(protocol_path, "0" * 64)
    inputs.write_bytes(b"mutated-after-freeze")
    with pytest.raises(ValueError, match="PIN_MISMATCH"):
        study.run(protocol_path, sealed_hash)
    assert not calls
    assert not (tmp_path / "started.json").exists()


def test_existing_output_receipt_cannot_be_overwritten(tmp_path):
    path = tmp_path / "protocol.json"
    study.write_json(path, {"frozen": True})
    with pytest.raises(FileExistsError):
        study.write_json(path, {"frozen": False})
    assert json.loads(path.read_text()) == {"frozen": True}


def test_learned_change_concat_matches_dense_with_independent_scales_and_gauges():
    initial = spectrum.FactorTerm(random_matrix(3, 11), random_matrix(9, 3), 0.7)
    final = spectrum.FactorTerm(random_matrix(3, 11), random_matrix(9, 3), 2.0)
    change = study.difference_term(final, initial)
    oracle = dense(final) - dense(initial)
    torch.testing.assert_close(dense(change), oracle, rtol=1e-12, atol=1e-12)
    assert change.a.shape == (6, 11) and change.b.shape == (9, 6)
    inputs, outputs = random_matrix(5, 11), random_matrix(9, 4)
    projection = study.project(
        change, study.alignment_span(inputs.T), study.alignment_span(outputs), 6, 1.0
    )
    expected = (
        outputs
        @ torch.linalg.pinv(outputs)
        @ oracle
        @ torch.linalg.pinv(inputs)
        @ inputs
    )
    torch.testing.assert_close(
        dense(projection.effective), expected, rtol=1e-11, atol=1e-11
    )
    assert projection.metrics["residual_frobenius_squared"] == pytest.approx(
        float((oracle - expected).square().sum()), rel=1e-12
    )
    gauge = torch.diag(torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64))
    initial_gauged = spectrum.FactorTerm(
        torch.linalg.solve(gauge, initial.a), initial.b @ gauge, initial.scale
    )
    final_gauged = spectrum.FactorTerm(
        torch.linalg.solve(gauge.T, final.a), final.b @ gauge.T, final.scale
    )
    torch.testing.assert_close(
        dense(study.difference_term(final_gauged, initial_gauged)),
        oracle,
        rtol=1e-12,
        atol=1e-12,
    )


def test_dominant_initial_adapter_does_not_hide_total_loss_of_learned_change():
    inputs = torch.eye(5, dtype=torch.float64)[:2]
    outputs = torch.eye(5, dtype=torch.float64)[:, :2]
    initial = spectrum.FactorTerm(
        torch.tensor([[1000.0, 0, 0, 0, 0]], dtype=torch.float64),
        torch.tensor([[1.0], [0], [0], [0], [0]], dtype=torch.float64),
        1.0,
    )
    final = spectrum.FactorTerm(
        torch.tensor([[1000.0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0], [0, 0], [0, 0], [0, 0], [0, 1.0]], dtype=torch.float64),
        1.0,
    )
    ui, uo = study.alignment_span(inputs.T), study.alignment_span(outputs)
    total = study.project(final, ui, uo, 2, 1)
    learned = study.project(study.difference_term(final, initial), ui, uo, 3, 1)
    assert total.metrics["relative_frobenius_error"] < 0.0011
    assert learned.metrics["residual_energy_fraction"] == pytest.approx(1.0)
    assert learned.metrics["frobenius_squared"] == pytest.approx(1.0)


def test_native_loader_restores_default_dtype_even_on_missing_input(tmp_path):
    with pytest.raises(ValueError, match="MISSING_NATIVE_CHECKPOINT"):
        study.load_native(tmp_path)
    assert torch.get_default_dtype() == torch.float32


def test_historical_binding_distinguishes_canonical_config_hash_from_file_bytes(
    tmp_path,
):
    model = tiny_portal()
    config = canonical_config()
    config["source_base"] = {
        "repo_id": model.config.base_model_name_or_path,
        "revision": model.config.base_model_revision,
    }
    source_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"
    )
    behavior, tensor_hashes = {"adapters": {}}, {}
    for label in ("rank8_initial", "rank8_final"):
        directory = tmp_path / label
        rank_control.save_adapter(
            source_config, study.generated_terms(model, "rte"), directory
        )
        specs = {
            name: study.pin(directory / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        config[label] = {"path": str(directory), "files": specs}
        behavior["adapters"][label] = {
            "files": {name: spec["sha256"] for name, spec in specs.items()}
        }
        tensor_hashes[label] = rank_control.tensor_hash(
            load_file(directory / "adapter_model.safetensors")
        )
    study.write_json(tmp_path / "behavior.json", behavior)
    assert study.file_hash(tmp_path / "behavior.json") != study.canonical_json_hash(
        behavior
    )
    study.write_json(
        tmp_path / "result.json",
        {
            "config_sha256": study.canonical_json_hash(behavior),
            "adapter_tensor_hashes": tensor_hashes,
            "rows": 864,
        },
    )
    study.write_json(
        tmp_path / "audit.json",
        {
            "passed": True,
            "rank8_final_correct": 850,
            "result_sha256": study.file_hash(tmp_path / "result.json"),
        },
    )
    config["historical"] = {
        name: study.pin(tmp_path / filename)
        for name, filename in (
            ("behavior_config", "behavior.json"),
            ("behavior_result", "result.json"),
            ("behavior_audit", "audit.json"),
        )
    }
    assert study.verify_source_identity(config, model)[
        "historical_behavior_binding_verified"
    ]
    behavior["semantic_mutation"] = True
    (tmp_path / "behavior.json").write_text(json.dumps(behavior))
    with pytest.raises(ValueError, match="HISTORICAL_CONFIG_BINDING"):
        study.verify_source_identity(config, model)
