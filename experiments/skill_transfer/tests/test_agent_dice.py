import math

import pytest
import torch

from agent_dice import consensus_fusion, effective_delta, fuse_experts


def scalar_reference(values):
    positives = sum(value >= 0 for value in values)
    if positives * 2 > len(values):
        values = [value for value in values if value >= 0]
    elif positives * 2 < len(values):
        values = [value for value in values if value < 0]
    maximum = max(abs(value) for value in values)
    weights = [math.exp(abs(value) - maximum) for value in values]
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_paper_equations_against_independent_scalar_reference(count):
    generator = torch.Generator().manual_seed(91)
    values = torch.randn(count, 7, 11, generator=generator, dtype=torch.float64)
    values[0, 0] = 0
    actual, audit = consensus_fusion(list(values))
    expected = torch.tensor(
        [[scalar_reference(values[:, row, column].tolist()) for column in range(11)] for row in range(7)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert audit["max_weight_sum_error"] < 1e-14
    permuted, _ = consensus_fusion(list(values.flip(0)))
    torch.testing.assert_close(permuted, actual)


def test_large_minority_is_rejected_per_coordinate_not_tensor_any():
    vectors = [
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([-100.0, 100.0]),
    ]
    actual, audit = consensus_fusion(vectors)
    torch.testing.assert_close(actual, torch.tensor([1.0, -1.0]))
    assert audit["pruned_contributions"] == 2
    unfiltered = (torch.softmax(torch.stack(vectors).abs(), 0) * torch.stack(vectors)).sum(0)
    assert (unfiltered - actual).abs().min() > 90


def test_tie_keeps_both_signs_and_zero_votes_positive():
    vectors = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([-1.0, -10.0]),
    ]
    actual, audit = consensus_fusion(vectors)
    torch.testing.assert_close(actual, torch.zeros(2))
    assert audit["tie_coordinates"] == 1


def test_dense_fusion_is_factor_gauge_invariant_and_has_arithmetic_control():
    generator = torch.Generator().manual_seed(33)
    experts = [
        {
            "layer": {
                "A": torch.randn(2, 5, generator=generator),
                "B": torch.randn(4, 2, generator=generator),
                "offset": None,
            }
        }
        for _ in range(4)
    ]
    rotated = [
        {"layer": {"A": state["layer"]["A"] * 2, "B": state["layer"]["B"] / 2, "offset": None}} for state in experts
    ]
    first, _ = fuse_experts(experts, 2)
    second, _ = fuse_experts(rotated, 2)
    torch.testing.assert_close(effective_delta(first, "layer", 2), effective_delta(second, "layer", 2))
    mean, _ = fuse_experts(experts, 2, "arithmetic")
    expected = sum(effective_delta(state, "layer", 2) for state in experts) / 4
    torch.testing.assert_close(effective_delta(mean, "layer", 2), expected)
    assert mean["layer"]["offset"].numel() == first["layer"]["offset"].numel()
    assert not torch.count_nonzero(first["layer"]["B"])


def test_invalid_fusion_is_rejected():
    with pytest.raises(ValueError, match="DICE_NONFINITE"):
        consensus_fusion([torch.tensor([float("nan")])])
    with pytest.raises(ValueError, match="DICE_PRECISION"):
        consensus_fusion([torch.zeros(2, dtype=torch.bfloat16)])
    with pytest.raises(ValueError, match="DICE_SHAPES"):
        consensus_fusion([torch.zeros(2), torch.zeros(3)])
    with pytest.raises(ValueError, match="DICE_REFERENCE"):
        fuse_experts([{"x": {"A": torch.ones(1, 2), "B": torch.ones(2, 1), "offset": torch.zeros(2, 2)}}], 2)
