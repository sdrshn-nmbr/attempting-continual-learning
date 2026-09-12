import copy
from pathlib import Path

import evaluate_representation as behavior
import pytest
import torch
from test_rank_eval import fixture_config


def config_for(directory):
    original = fixture_config(directory)
    config = {
        key: original[key]
        for key in (
            "source_base",
            "source_base_tensor_sha256",
            "fresh_inputs",
            "evaluation",
            "seed",
            "claim_boundary",
        )
    }
    config.update(
        kind="fixed_alignment_projection_behavior",
        conditions=list(behavior.CONDITIONS),
        geometry={"result": original["projection_proof"]},
        adapters={
            name: copy.deepcopy(original["adapters"][source])
            for name, source in (
                ("rank8_initial", "rank8_initial"),
                ("initial_exact", "rank8_initial"),
                ("rank8_final", "rank8_final"),
                ("projection", "rank8_initial"),
            )
        },
    )
    return config


def test_actual_reloaded_panels_without_optimizer_or_backward(tmp_path, monkeypatch):
    config = config_for(tmp_path / "assets")

    def forbidden(*args, **kwargs):
        pytest.fail("inference evaluator attempted to train")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    output = tmp_path / "run"
    output.mkdir()
    result = behavior.run(config, output, "cpu")
    assert result["status"] == "completed" and result["optimizer_updates"] == 0
    assert result["comparisons"]["rows_per_condition"] == 864
    assert result["comparisons"]["initial_exact_control"]["passed"]
    assert result["panels"]["projection"] == result["panels"]["rank8_initial"]
    assert result["panels"]["projection"] != result["panels"]["rank8_final"]
    assert all(
        check["base_and_portal_unchanged"] for check in result["checks"].values()
    )
    with pytest.raises(FileExistsError):
        behavior.run(config, output, "cpu")


def test_changed_adapter_stops_before_loading_model(tmp_path, monkeypatch):
    config = config_for(tmp_path / "assets")
    path = Path(config["adapters"]["projection"]["path"]) / "adapter_model.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("model loaded before input validation")

    monkeypatch.setattr(behavior, "evaluate_adapter_panels", forbidden)
    with pytest.raises(ValueError, match="PROJECTION_INPUT_CHANGED"):
        behavior.run(config, tmp_path / "run", "cpu")
