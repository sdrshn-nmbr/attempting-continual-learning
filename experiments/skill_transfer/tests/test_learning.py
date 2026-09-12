import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    GenerationConfig,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from agent_dice import effective_delta
from data import build_corpus, digest
from learning import (
    LowRankLinear,
    capture,
    frozen_hash,
    install_adapters,
    new_optimizer,
    restore,
    save_checkpoint,
    score,
    train_update,
    tree_record,
    write_json,
)
from run import experiment, load_training_state, prepare_output, primitive_gate, qualification, train_segment


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skill_transfer_load_proof = {"test_fixture": True}
        self.embedding = torch.nn.Embedding(11, 8)
        self.embedding.requires_grad_(False)
        self.projection = LowRankLinear(torch.nn.Linear(8, 11, bias=False), 8, 16)

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        logits = self.projection(self.embedding(input_ids))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


class EncodedFixture:
    def __init__(self, config):
        self.config = config

    def batch(self, rows, training):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5] for _ in rows]),
            "attention_mask": torch.ones(len(rows), 5, dtype=torch.long),
        }
        labels = torch.tensor([[-100, -100, -100, 4, 5] for _ in rows])
        return inputs, labels, 2


def config():
    return json.loads((Path(__file__).parents[1] / "configs/stream_1103.json").read_text())


def test_adapter_equals_effective_dense_update_and_base_is_frozen():
    torch.manual_seed(13)
    model = TinyLanguageModel()
    with torch.no_grad():
        model.projection.B.normal_()
        model.projection.offset = torch.randn(11, 8)
    value = torch.randn(3, 8)
    delta = effective_delta(capture(model), "projection", 2)
    expected = torch.nn.functional.linear(value, model.projection.base.weight + delta)
    torch.testing.assert_close(model.projection(value), expected, rtol=2e-6, atol=2e-6)
    assert not model.projection.base.weight.requires_grad


def test_real_cpu_updates_reduce_answer_loss_preserve_base_and_resume_exactly(tmp_path):
    torch.manual_seed(41)
    settings = {**config(), "learning_rate": 0.03, "device": "cpu"}
    _, corpus = build_corpus(settings)
    rows = corpus["primitive"]["records"]["train"][:4]
    encoded = EncodedFixture(settings)
    model = TinyLanguageModel()
    optimizer = new_optimizer(model, settings)
    base = frozen_hash(model)
    losses = [train_update(model, optimizer, encoded, rows)["loss"] for _ in range(12)]
    assert losses[-1] < losses[0] / 3
    assert frozen_hash(model) == base
    assert torch.count_nonzero(model.projection.B)
    saved = save_checkpoint(model, optimizer, tmp_path / "checkpoint", {"test": True})
    train_update(model, optimizer, encoded, rows)
    expected_after_one = capture(model)
    resumed = load_training_state(model, tmp_path / "checkpoint", settings)
    assert tree_record(capture(model)) == tree_record(saved)
    train_update(model, resumed, encoded, rows)
    assert tree_record(capture(model)) == tree_record(expected_after_one)
    assert {int(value["step"]) for value in resumed.state.values()} == {13}


def test_training_rejects_held_out_rows():
    settings = config()
    _, corpus = build_corpus(settings)
    model = TinyLanguageModel()
    with pytest.raises(ValueError, match="TRAIN_SPLIT_REQUIRED"):
        train_update(
            model, new_optimizer(model, settings), EncodedFixture(settings), corpus["primitive"]["text"]["test"][:4]
        )


def test_capture_restore_does_not_leave_an_old_dense_offset():
    model = TinyLanguageModel()
    initial = capture(model)
    model.projection.offset = torch.ones(11, 8)
    restore(model, initial)
    assert model.projection.offset is None
    assert digest(tree_record(capture(model))) == digest(tree_record(initial))


def test_real_segment_update_and_exposure_accounting(tmp_path):
    settings = {**config(), "learning_rate": 0.01, "device": "cpu"}
    _, corpus = build_corpus(settings)
    model = TinyLanguageModel()
    optimizer = new_optimizer(model, settings)
    counts, rows = train_segment(
        model,
        optimizer,
        EncodedFixture(settings),
        corpus["primitive"]["text"]["train"],
        0,
        3,
        91,
        tmp_path / "updates.jsonl",
    )
    assert counts["current_exposures"] == 12 and counts["old_exposures"] == 0
    assert counts["optimizer_step"] == 3 and counts["target_tokens"] == 24
    assert len(rows) == len(counts["exposed_unique_current_ids"])
    assert len((tmp_path / "updates.jsonl").read_text().splitlines()) == 3


def test_qualification_requires_each_operation_and_feasible_gain():
    settings = config()
    initial = {"accuracy": 0.25}
    good = {"accuracy": 0.875, "per_pattern": dict(a=1, b=1, c=0.75, d=0.75)}
    assert primitive_gate(initial, good, settings)["passed"]
    bad = {"accuracy": 0.875, "per_pattern": dict(a=1, b=1, c=1, d=0.5)}
    assert not primitive_gate(initial, bad, settings)["passed"]
    saturated = primitive_gate({"accuracy": 1}, {"accuracy": 1, "per_pattern": dict(a=1, b=1, c=1, d=1)}, settings)
    assert not saturated["passed"] and not saturated["sufficient_baseline_headroom"]


def test_saturated_baseline_stops_before_any_training(monkeypatch, tmp_path):
    settings = config()
    spec, corpus = build_corpus(settings)
    metrics = {"accuracy": 1.0, "per_pattern": dict(a=1, b=1, c=1, d=1)}
    monkeypatch.setattr("run.assess_primitives", lambda *args, **kwargs: {family: metrics for family in spec["order"]})
    monkeypatch.setattr("run.evaluate", lambda *args, **kwargs: metrics)

    def forbidden(*args, **kwargs):
        raise AssertionError("training should not run")

    monkeypatch.setattr("run.train_segment", forbidden)
    result, checkpoints = qualification(None, None, spec, corpus, {}, settings, tmp_path)
    assert result["reason"] == "baseline_saturated" and result["budget"] == 0
    assert not checkpoints


def test_failed_acquisition_never_enters_stream_or_fusion(monkeypatch, tmp_path):
    settings = {**config(), "mode": "study", "generation_use_cache": False}
    model = TinyLanguageModel()
    encoded = SimpleNamespace(tokens=lambda row: ([1, 2], [3, 4]), eos=4)
    monkeypatch.setattr("run.load_tokenizer", lambda *args: encoded)
    monkeypatch.setattr("run.load_model", lambda *args: model)
    monkeypatch.setattr("run.qualification", lambda *args: ({"passed": False, "budget": 512}, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("no intervention before acquisition")

    monkeypatch.setattr("run.stream_controls", forbidden)
    monkeypatch.setattr("run.fusion_controls", forbidden)
    monkeypatch.setattr("run.workflow_transfer", forbidden)
    experiment(settings, tmp_path / "study")
    result = json.loads((tmp_path / "study/result.json").read_text())
    assert result["status"] == "prerequisite_failed" and result["transfer_updates"] == 0
    assert result["actual_executed_updates_including_qualification"] == 0


def test_dispatch_transport_can_preexist_but_results_cannot(tmp_path):
    settings = config()
    (tmp_path / "config.json").write_text(json.dumps(settings))
    (tmp_path / "task.json").write_text(json.dumps({"config": settings}))
    (tmp_path / "execution.json").write_text("{}")
    (tmp_path / "attempts").mkdir()
    prepare_output(settings, tmp_path)
    assert (tmp_path / "dispatcher.json").exists()
    with pytest.raises(RuntimeError, match="OUTPUT_ALREADY_USED"):
        prepare_output(settings, tmp_path)


def test_full_controller_preserves_comparator_and_exposure_contracts(monkeypatch, tmp_path):
    settings = {
        **config(),
        "generation_use_cache": False,
        "qualification_budgets": [1],
        "workflow_checkpoints": [1],
        "device": "cpu",
        "buffer_capacity": 4,
    }
    for kind in ("primitive", "workflow"):
        for split in ("train", "validation", "test"):
            settings[f"{kind}_{split}_examples"] = 4
    model = TinyLanguageModel()
    encoded = EncodedFixture(settings)
    encoded.tokens = lambda row: ([1, 2], [3, 4])
    encoded.eos = 4

    def observations(model, encoded, rows):
        competent = bool(torch.count_nonzero(model.projection.B)) or model.projection.offset is not None
        return [
            {
                "id": row.id,
                "pattern": row.pattern,
                "correct": competent,
                "native_eos": True,
                "format_valid": True,
                "executable": True,
                "error": None,
            }
            for row in rows
        ]

    def evaluation(model, encoded, rows, path):
        records = observations(model, encoded, rows)
        metrics = score(records)
        write_json(path, {"metrics": metrics, "records": records})
        return metrics

    monkeypatch.setattr("run.load_model", lambda *args: model)
    monkeypatch.setattr("run.load_tokenizer", lambda *args: encoded)
    monkeypatch.setattr("run.evaluate", evaluation)
    monkeypatch.setattr("learning.generate", observations)
    experiment(settings, tmp_path / "study")
    result = json.loads((tmp_path / "study/result.json").read_text())
    assert result["status"] == "completed"
    assert result["actual_executed_updates_including_qualification"] == 17
    assert set(result["transfer"]) == {
        "fresh",
        "relevant",
        "irrelevant",
        "continue",
        "replay",
        "arithmetic",
        "agent_dice",
    }
    assert (
        result["fusions"]["agent_dice"]["resident_dense_delta_elements"]
        == result["fusions"]["arithmetic"]["resident_dense_delta_elements"]
    )
    assert result["fusions"]["agent_dice"]["expert_training_updates"] == 4
    sequences = []
    for condition in result["transfer"]:
        updates = [
            json.loads(line)
            for line in (tmp_path / "study/transfer" / condition / "updates.jsonl").read_text().splitlines()
        ]
        sequences.append([record["example_ids"] for record in updates])
    assert all(sequence == sequences[0] for sequence in sequences)
    assert (tmp_path / "study/protocol.json").read_bytes() == (Path(__file__).parents[1] / "protocol.json").read_bytes()


def test_native_qwen_model_adapter_backward_and_cached_generation():
    torch.manual_seed(37)
    text = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "mrope_section": [2, 2, 4]},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision))
    install_adapters(model, config())
    before = frozen_hash(model)
    inputs = {
        "input_ids": torch.tensor([[0, 1, 4, 5], [1, 3, 4, 5]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
    }
    logits = model(**inputs, use_cache=False, logits_to_keep=3).logits
    assert tuple(logits.shape) == (2, 3, 32)
    logits.sum().backward()
    assert any(
        value.grad is not None and value.grad.abs().sum() > 0 for value in model.parameters() if value.requires_grad
    )
    assert frozen_hash(model) == before
    results = []
    for cached in (False, True):
        generation = GenerationConfig(
            do_sample=False, max_new_tokens=4, pad_token_id=0, eos_token_id=2, bos_token_id=1, use_cache=cached
        )
        model.generation_config = generation
        results.append(model.generate(**inputs, generation_config=generation))
    assert torch.equal(*results)
