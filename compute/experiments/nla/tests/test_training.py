import json
import signal
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

import run as runner
from native import generation_diagnostics, inject_activation, norm_matched_random
from protocol import make_records
from run import adapter_digest, adapter_state, train_batch, train_step
from storage import StopRequested, Store, atomic_torch, read_torch


@pytest.mark.parametrize("gate_passed", [False, True])
def test_native_run_never_enters_training_and_persists_all_controls(
    diagnostic_config, native_tokenizers, tiny_model, tokenizer, tmp_path, monkeypatch, gate_passed
):
    original_weights = {name: tensor.detach().clone() for name, tensor in tiny_model.state_dict().items()}
    calls = []
    av_tokenizer = native_tokenizers["av"]

    class CalibrationAV:
        def __init__(self, path):
            self.model = torch.nn.Identity()
            self.sidecar = SimpleNamespace(layer=2, injection_scale=150)

        def describe(self, vector, max_new_tokens):
            calls.append(vector.clone())
            text = "<explanation>bread" + ("</explanation>" if vector.norm() > 0 else "")
            ids = av_tokenizer.encode(text, add_special_tokens=False) + [av_tokenizer.eos_token_id]
            slot = inject_activation(torch.zeros(1, 1, 32, dtype=torch.bfloat16), vector, 0, 150)[0, 0]
            return {
                **generation_diagnostics(av_tokenizer, ids, max_new_tokens),
                "injection_slot_norm": float(slot.float().norm()),
                "injection_scale": 150,
            }

    class CalibrationAR:
        def __init__(self, path):
            self.model = self.head = torch.nn.Identity()
            self.sidecar = SimpleNamespace(mse_scale=32**0.5)

        def reconstruct(self, explanation):
            return torch.ones(32)

    def forbidden_training(*args, **kwargs):
        pytest.fail("Native calibration entered the training or adapter setup path")

    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"test_boundary": "CPU orchestration only"})
    monkeypatch.setattr(runner, "verify_model_files", lambda *args: {})
    monkeypatch.setattr(runner, "gpu_memory", dict)
    monkeypatch.setattr(runner, "load_source", lambda path: tiny_model)
    monkeypatch.setattr(runner, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(runner.Sidecar, "load", lambda path, role: SimpleNamespace(layer=2))
    monkeypatch.setattr(runner, "NativeAV", CalibrationAV)
    monkeypatch.setattr(runner, "NativeAR", CalibrationAR)
    monkeypatch.setattr(runner, "fidelity_gate", lambda *args: {"passed": gate_passed})
    for name in ("make_records", "get_peft_model", "LoraConfig", "update_weights", "train_step", "train_batch"):
        monkeypatch.setattr(runner, name, forbidden_training)
    store = Store(tmp_path, diagnostic_config, "cpu-orchestration-test")
    try:
        assert runner.run(diagnostic_config, store, {}) == (0 if gate_passed else 2)
    finally:
        store.close()
    assert len(calls) == 64
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in tiny_model.parameters())
    for name, tensor in tiny_model.state_dict().items():
        torch.testing.assert_close(tensor, original_weights[name], rtol=0, atol=0)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["status"] == ("native_calibration_complete" if gate_passed else "native_fidelity_gate_failed")
    assert metrics["weight_updates_executed"] == 0
    assert metrics["native_trainable_parameter_counts"] == {"source": 0, "av": 0, "ar": 0, "ar_head": 0}
    diagnostics = metrics["measurements"]["native_diagnostics"]["conditions"]
    assert all(diagnostics[name]["output"]["n_total"] == 16 for name in ("true", "empty", "shuffled", "random"))
    assert diagnostics["empty"]["output"]["n_scorable"] == 0
    assert all(diagnostics[name]["output"]["n_scorable"] == 16 for name in ("true", "shuffled", "random"))
    data = json.loads((tmp_path / "data.json").read_text())
    assert data["records"] == data["training_examples"] == []
    assert not (tmp_path / "checkpoints").exists()
    for i, record in enumerate(data["calibration"]):
        state = read_torch(tmp_path / "states" / "native_calibration" / f"{record['id']}.pt")
        random = read_torch(tmp_path / "readouts" / "native_calibration" / record["id"] / "random.pt")
        vector, provenance = norm_matched_random(state["activation"], 2718, record["id"])
        assert random["random_control"] == provenance
        torch.testing.assert_close(calls[4 * i], state["activation"], rtol=0, atol=0)
        assert torch.count_nonzero(calls[4 * i + 1]) == 0
        torch.testing.assert_close(calls[4 * i + 3], vector, rtol=0, atol=0)
        assert random["injected_raw_norm"] == pytest.approx(float(state["activation"].norm()), rel=2e-7)
        assert random["injection_slot_norm"] == pytest.approx(150, abs=0.5)


def test_training_masks_all_prompt_and_padding_tokens(native_tokenizers, config):
    records = [r for r in make_records(config) if r.learning_stage == "a"]
    tokenizer = native_tokenizers["base"]
    batch = train_batch(records, tokenizer, config, config["updates"][0], 0)
    for ids, mask, labels in zip(batch["input_ids"], batch["attention_mask"], batch["labels"], strict=True):
        selected = labels != -100
        assert int(selected.sum()) >= 2
        assert torch.equal(ids[selected], labels[selected])
        assert torch.all(labels[mask == 0] == -100)
        decoded = tokenizer.decode(labels[selected].tolist(), skip_special_tokens=False)
        assert decoded.endswith(tokenizer.eos_token)
        assert "Neral" not in decoded
        assert not torch.any(selected[:10])


def test_lora_changes_weights_and_disabled_repeat_preserves_base(tiny_model):
    model = get_peft_model(
        tiny_model,
        LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none"
        ),
    )
    ids = torch.tensor([[3, 4, 11, 12, 13, 14, 5, 6, 7, 2]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    batch["labels"][:, :-2] = -100
    model.eval()
    with model.disable_adapter(), torch.inference_mode():
        before = model(input_ids=ids).logits.clone()
    initial = adapter_digest(adapter_state(model))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(3):
        loss, grad = train_step(model, optimizer, batch)
        assert loss > 0 and grad > 0
    model.eval()
    with model.disable_adapter(), torch.inference_mode():
        frozen = model(input_ids=ids).logits.clone()
    with torch.inference_mode():
        after = model(input_ids=ids).logits.clone()
    assert initial != adapter_digest(adapter_state(model))
    torch.testing.assert_close(before, frozen, rtol=0, atol=0)
    assert not torch.equal(before, after)


def test_adapter_optimizer_rng_checkpoint_resumes_exactly(tiny_model, tmp_path):
    model = get_peft_model(
        tiny_model, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["v_proj"], lora_dropout=0.0)
    )
    ids = torch.tensor([[3, 4, 11, 12, 5, 6, 7, 2]])
    batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
    batch["labels"][:, :-2] = -100
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0.0)
    train_step(model, optimizer, batch)
    path = tmp_path / "checkpoint.pt"
    atomic_torch(
        path, {"adapter": adapter_state(model), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
    )
    train_step(model, optimizer, batch)
    expected = adapter_state(model)
    checkpoint = read_torch(path)
    result = set_peft_model_state_dict(model, checkpoint["adapter"])
    assert not result.unexpected_keys
    assert not any("lora_" in name for name in result.missing_keys)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    train_step(model, optimizer, batch)
    for key, value in adapter_state(model).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_events_append_and_interruption_does_not_claim_completion(config, tmp_path):
    store = Store(tmp_path, config, "code")
    store.request_stop(signal.SIGTERM, None)
    with pytest.raises(StopRequested):
        store.check_stop()
    store.finish("interrupted", resumable=True)
    store.close()
    first = (tmp_path / "events.jsonl").read_bytes()
    resumed = Store(tmp_path, config, "code")
    resumed.finish("native_calibration_complete", weight_updates_executed=0)
    resumed.close()
    assert (tmp_path / "events.jsonl").read_bytes().startswith(first)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["status"] == "native_calibration_complete"
    assert metrics["measurements"] == {}


def test_resume_rejects_changed_config(config, tmp_path):
    store = Store(tmp_path, config, "code")
    store.close()
    config["seed"] += 1
    with pytest.raises(ValueError, match="RESUME_IDENTITY"):
        Store(tmp_path, config, "code")
