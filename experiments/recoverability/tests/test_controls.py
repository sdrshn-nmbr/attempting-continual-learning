import copy
import json

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig

import protocol as protocol_module
from analysis import accuracy, analyze, features, forgotten, recovered, ridge_predict
from model import Observer, distribution_features, runtime_info
from protocol import (
    ACTIONS,
    PROTOCOL,
    ROOT,
    digest,
    file_hash,
    normalized_hash,
    read_json,
    seal,
    validate_dataset,
    verify_seal,
    write_json,
)
from run import analyze_output, execute


def row(text, target=2, ids=None):
    return {"text": text, "text_sha256": normalized_hash(text), "target": target, "input_ids": ids or [1, 4, 5]}


@pytest.fixture
def observer(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        patch_size=2,
    )
    cfg = Qwen3_5Config(text_config=text, vision_config=vision)
    cfg._attn_implementation = "eager"
    base = Qwen3_5ForConditionalGeneration(cfg)
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r"model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
        ),
    )
    spec = read_json(PROTOCOL)
    spec["runtime"] = spec["runtime"] | {"device": "cpu", "attention": "eager"}
    spec["layers"] = [1, 2, 3]
    spec["actions"] = spec["actions"] | {"replay_short": 2, "replay_long": 3, "sham": 3}
    spec["optimizer"] = spec["optimizer"] | {"learning_rate": 1.0, "batch_size": 1}
    obs = Observer(model, spec, list(range(64)))
    before, after = tmp_path / "before", tmp_path / "after"
    for path in (before, after):
        path.mkdir()
        model.peft_config["default"].save_pretrained(str(path))
    obs.save_adapter(before / "adapter_model.safetensors")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 10)
    obs.save_adapter(after / "adapter_model.safetensors")
    return obs, before, after


def test_real_hybrid_qwen_terminal_lens_control(observer):
    obs, before, _ = observer
    obs.reset(before)
    measured = obs.observe([row("a fixture")], lens=True)[0]
    assert measured["terminal_max_error"] <= 1e-5
    assert set(measured["lens"]) == {"1", "2", "3"}
    assert all(not layer._forward_hooks for layer in obs.text.layers)


def test_restoration_is_measured_behavior_and_reset_is_exact(observer):
    obs, before, after = observer
    prompts = [row("control", ids=[1, 12, 19])]
    obs.reset(before)
    baseline = obs.observe(prompts)
    prompts[0]["target"] = baseline[0]["output"]["prediction"]
    baseline = obs.observe(prompts)
    obs.reset(after)
    damaged = obs.observe(prompts)
    assert accuracy(baseline, prompts) == 1
    assert accuracy(damaged, prompts) == 0
    obs.reset(before)
    restored = obs.observe(prompts)
    assert restored == baseline
    assert accuracy(restored, prompts) == 1


def test_repair_is_repeatable_and_does_not_change_base_or_checkpoints(observer):
    obs, before, after = observer
    old_hashes = {str(p): file_hash(p) for d in (before, after) for p in d.iterdir()}
    frozen = {n: p.detach().clone() for n, p in obs.model.named_parameters() if not p.requires_grad}
    rows = [row("repair input", ids=[2, 3, 7])]
    obs.reset(after)
    obs.repair(rows, "sham")
    obs.reset(after)
    loss1 = obs.repair(rows, "replay_long")
    state1 = {k: v.detach().clone() for k, v in get_peft_model_state_dict(obs.model, save_embedding_layers=False).items()}
    obs.reset(after)
    loss2 = obs.repair(rows, "replay_long")
    state2 = get_peft_model_state_dict(obs.model, save_embedding_layers=False)
    assert loss1 == loss2
    assert all(torch.equal(state1[key], state2[key]) for key in state1)
    original = load_file(str(after / "adapter_model.safetensors"))
    assert any(not torch.equal(state1[key], original[key]) for key in state1)
    assert all(torch.equal(p, frozen[n]) for n, p in obs.model.named_parameters() if n in frozen)
    assert old_hashes == {str(p): file_hash(p) for d in (before, after) for p in d.iterdir()}


def test_output_confidence_uses_full_vocabulary_not_answer_subset():
    score = distribution_features(torch.tensor([0.0, 10.0, 2.0, 3.0]), 3, [2, 3])
    assert score["prediction"] == 1
    assert score["code_prediction"] == 3
    assert score["top_probability"] > 0.99


def test_recovery_requires_measured_behavior_and_guard():
    spec = read_json(PROTOCOL)
    assert forgotten(0.8, 0.4, spec)
    assert not forgotten(0.7, 0.4, spec)
    assert not recovered(0.8, 0.4, 0.7, 0.9, 0.9, spec)
    assert not recovered(0.8, 0.4, 0.8, 0.9, 0.6, spec)
    assert recovered(0.8, 0.4, 0.8, 0.9, 0.9, spec)


def test_prepared_data_counts_identity_and_role_separation():
    data, spec = read_json(ROOT / "inputs/cohort.json"), read_json(PROTOCOL)
    validate_dataset(data, spec)
    assert len([u for u in data["units"] if u["split"] == "train"]) == 12
    assert len([u for u in data["units"] if u["split"] == "test"]) == 9
    assert spec["budget"]["max_optimizer_steps"] == len(data["units"]) * 28


@pytest.mark.parametrize("mutation", ["group", "text", "guard", "run", "missing_row", "budget"])
def test_leakage_and_budget_fail_closed(mutation):
    data, spec = read_json(ROOT / "inputs/cohort.json"), read_json(PROTOCOL)
    train = next(u for u in data["units"] if u["split"] == "train")
    test = next(u for u in data["units"] if u["split"] == "test")
    if mutation == "group":
        test["skill"] = train["skill"]
    elif mutation == "text":
        test["label"][0] = copy.deepcopy(test["repair"][0])
    elif mutation == "guard":
        data["guards"][test["run"]][0] = copy.deepcopy(test["label"][0])
    elif mutation == "run":
        test["run"] = train["run"]
    elif mutation == "missing_row":
        test["probe"].pop()
    else:
        spec["budget"]["max_intents"] = 2
    with pytest.raises(ValueError, match="RECOVERY_"):
        validate_dataset(data, spec)


def test_scaler_fits_only_training_set():
    train_x = [[0.0], [1.0], [2.0]]
    y = [0.0, 0.5, 1.0]
    one = ridge_predict(train_x, y, [[0.3]], 10)
    with_outlier = ridge_predict(train_x, y, [[0.3], [1e12]], 10)
    np.testing.assert_array_equal(one, with_outlier[:1])


def test_features_never_accept_post_action_or_label_cohort():
    metric = {"top_probability": 0.5, "entropy": 1.0, "top_margin": 0.1, "target_logp": -1.0, "target_code_margin": 0.2}
    probe = [{"output": metric, "lens": {"8": metric, "16": metric, "24": metric}}]
    source = {"before": probe, "after": probe}
    expected = features(source)
    source["label"] = [{"recovered": 1234}]
    source["after_repair"] = [{"target_logp": 999}]
    assert features(source) == expected


def analysis_fixture():
    data = {"units": [], "guards": {"train-run": [row("train guard")], "test-run": [row("test guard")]}}
    observations = {"units": {}, "guards": {}}
    outcomes, table = {}, {}

    def measured(rows, correct):
        return [
            {
                "text_sha256": r["text_sha256"],
                "output": {
                    "prediction": r["target"] if correct else 0,
                    "code_prediction": r["target"] if correct else 0,
                },
            }
            for r in rows
        ]

    for split in ("train", "test"):
        run = f"{split}-run"
        observations["guards"][run] = {
            "before": measured(data["guards"][run], True),
            "after": measured(data["guards"][run], True),
        }
        for i in range(8):
            name = f"{split}/{i}"
            unit = {"id": name, "run": run, "split": split, "skill": name, "label": [row(name)]}
            data["units"].append(unit)
            observations["units"][name] = {
                phase: {"label": measured(unit["label"], phase == "before")} for phase in ("before", "after")
            }
            signal = float(i % 2)
            table[name] = {"confidence": [1.0], "output": [1.0], "lens": [1.0, signal]}
            outcomes[name] = {
                action: {
                    "label": measured(
                        unit["label"], action == "restore" or (action.startswith("replay") and signal == 1)
                    ),
                    "guard": measured(data["guards"][run], True),
                }
                for action in ACTIONS
            }
    return data, observations, outcomes, table


def test_predictive_metrics_are_descriptive_and_skill_weighted():
    data, observations, outcomes, table = analysis_fixture()
    result = analyze(data, observations, outcomes, table, read_json(PROTOCOL))
    assert result["status"] == "descriptive_pilot"
    assert result["feasibility"] == "insufficient_independent_runs"
    assert result["counts"]["forgotten_test"] == 8
    assert result["counts"]["raw_clusters_by_split"] == {"train": 1, "test": 1}
    assert "gates_failed" not in result
    score = result["recovery_prediction"]["replay_short"]
    assert len(score["labels"]) == 8
    assert score["brier"]["lens"] < score["brier"]["output"]
    assert result["controls"] == {"restore": 1.0, "sham": 0.0}


def test_test_outcomes_cannot_change_fitted_predictor():
    data, observations, outcomes, table = analysis_fixture()
    first = analyze(data, observations, outcomes, table, read_json(PROTOCOL))
    for unit in data["units"]:
        if unit["split"] == "test":
            for action in ("replay_short", "replay_long"):
                record = outcomes[unit["id"]][action]["label"][0]["output"]
                record["prediction"] = 2 if record["prediction"] == 0 else 0
    second = analyze(data, observations, outcomes, table, read_json(PROTOCOL))
    for action in ("replay_short", "replay_long"):
        assert (
            first["recovery_prediction"][action]["predictions"] == second["recovery_prediction"][action]["predictions"]
        )
    for key in ("output", "lens"):
        assert first["policies"][key]["choices"] == second["policies"][key]["choices"]


def test_no_forgotten_skills_is_infeasibility_not_predictor_failure():
    data, observations, outcomes, table = analysis_fixture()
    for value in observations["units"].values():
        value["after"] = copy.deepcopy(value["before"])
    result = analyze(data, observations, outcomes, table, read_json(PROTOCOL))
    assert result["status"] == "descriptive_pilot"
    assert result["predictor_feasibility"] == "insufficient_eligible_skills"
    assert result["counts"]["forgotten_train"] == result["counts"]["forgotten_test"] == 0
    assert result["predictor_estimate"] is None


def test_seal_detects_payload_change_before_model_loading():
    payload = {"runtime": {"device": "cpu"}, "data": [1]}
    sealed = {"payload": payload, "sha256": digest(payload)}
    payload["data"].append(2)
    with pytest.raises(ValueError, match="RECOVERY_SEAL_OR_RUNTIME_CHANGED"):
        verify_seal(sealed, {"device": "cpu"})


def test_protocol_is_sealed_before_real_outcomes():
    assert file_hash(PROTOCOL) == (ROOT / "protocol.sha256").read_text().strip()
    spec = json.loads(PROTOCOL.read_text())
    assert spec["runtime"] == {
        "device": "cuda:0",
        "dtype": "float32",
        "attention": "sdpa",
        "threads": 2,
        "max_length": 128,
        "seed": 20260911,
    }
    assert spec["study_type"] == "descriptive_pilot"


def test_complete_sealed_pipeline_on_cpu_hybrid_fixture(observer, tmp_path, monkeypatch):
    obs, before, after = observer
    spec = copy.deepcopy(obs.spec)
    spec["rows"] = {"repair": 1, "probe": 1, "label": 1}
    spec["runtime"]["threads"] = 1
    model_path = tmp_path / "base"
    data = {"units": [], "runs": {}, "guards": {}, "sources": {}, "codes": list(range(64)),
            "base_manifest": [{"path": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)}
                              for p in model_path.iterdir()]}
    for i, split in enumerate(("train", "test")):
        name = f"{split}-fixture"
        data["runs"][name] = {phase: {"path": str(path), "files": {p.name: file_hash(p) for p in path.iterdir()}}
                              for phase, path in (("before", before), ("after", after))}
        unit = {"id": name, "run": name, "split": split, "skill": name}
        for j, role in enumerate(("repair", "probe", "label")):
            unit[role] = [row(f"{name}-{role}", target=2 + i, ids=[1, 10 * i + j + 2])]
        data["units"].append(unit)
        data["guards"][name] = [row(f"{name}-guard", target=4, ids=[1, 10 * i + 8])]
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, spec)
    (tmp_path / "protocol.sha256").write_text(file_hash(protocol_path) + "\n")
    dataset = tmp_path / "cohort.json"
    write_json(dataset, data)
    config = tmp_path / "config.json"
    write_json(config, {"model_id": spec["backbone"], "revision": spec["revision"],
                       "model_path": str(model_path), "dataset": str(dataset)})
    monkeypatch.setattr(protocol_module, "ROOT", tmp_path)
    monkeypatch.setattr(protocol_module, "PROTOCOL", protocol_path)
    sealed = seal(config, runtime_info())
    output = tmp_path / "run"
    execute(sealed, output)
    result = analyze_output(output)
    receipt = read_json(output / "receipt.json")
    assert receipt["status"] == "complete" and receipt["device"] == "cpu"
    assert len(list((output / "adapters").glob("*.safetensors"))) == 6
    assert result["status"] == "descriptive_pilot"
    assert result["counts"]["all_skills"] == 2
    with (output / "features.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="RECOVERY_OUTPUT_CHANGED"):
        analyze_output(output)
