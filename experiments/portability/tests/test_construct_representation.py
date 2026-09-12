import compression_spectrum as spectrum
import construct_representation as study
import pytest
import representability as geometry
import torch
from portallib import PortalConfig, PortalModel, PortalProjectionTarget
from test_representability import canonical_config


def test_constructs_independent_updates_with_same_shared_generator_architecture(
    tmp_path,
):
    torch.set_num_threads(1)
    torch.manual_seed(711)
    config = PortalConfig(
        base_model_name_or_path="synthetic/no-base-model",
        tasks=("rte", "control"),
        n_layers=3,
        projection_targets=tuple(
            PortalProjectionTarget(
                layer, name, f"self_attn.{name}_proj", 17, width, name, name
            )
            for layer in range(3)
            for name, width in (("q", 17), ("v", 11))
        ),
        rank=2,
        alpha=4,
        d_z=5,
        d_layer=4,
        hidden=12,
        d_core=8,
    )
    model = PortalModel(config, torch.randn(2, 5)).double().requires_grad_(False)
    for parameter in model.parameters():
        parameter.normal_(0, 0.25)
    source = {
        path: spectrum.FactorTerm(
            torch.randn(2, target.in_features),
            torch.randn(target.out_features, 2),
            2.0,
        )
        for target, path in model.config.resolved_targets()
    }
    original = geometry.generated_terms(model, "rte")
    assert study.errors(source, original)["relative_frobenius_error"] > 0.5
    result = study.construct(model, source, canonical_config(), 711)
    assert result["architecture_unchanged"] and result["optimizer_steps"] == 0
    assert result["shared_heads"]["linear_solves"] == 4
    assert result["fp64_weight_error"]["relative_frobenius_error"] < 1e-10
    geometry.save_native(model.float(), tmp_path / "native")
    restored = geometry.load_native(tmp_path / "native", torch.float32)
    actual = geometry.generated_terms(restored, "rte")
    assert study.errors(source, actual)["relative_frobenius_error"] < 1e-5


def test_dependent_factor_columns_still_preserve_the_complete_source_span():
    generator = torch.Generator().manual_seed(17)
    vectors = torch.randn(17, 2, generator=generator)
    vectors = torch.cat((vectors, vectors), dim=1)
    basis, result = study.complete_span(vectors, 8, generator)
    assert basis.shape == (17, 8)
    assert result["relative_span_error"] < 1e-12


def test_union_larger_than_canonical_width_does_not_make_capacity_claim():
    with pytest.raises(ValueError, match="SPAN_DIMENSIONS"):
        study.complete_span(torch.ones(17, 9), 8, torch.Generator())
