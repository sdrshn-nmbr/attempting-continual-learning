import json
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from portallib import PortalBase, PortalConfig, PortalModel, collate_gold_batch
from transformers import Qwen3Config, Qwen3ForCausalLM

import calibration
import engine
import tasks
from calibration import CalibrationPlan, qualify, score_split, source_calibration
from engine import (
    Adaptation,
    Pilot,
    batch_schedule,
    evaluate,
    fit_stage,
    learning_deltas,
    run_arm,
    verify_shared_core,
)
from inventory import tensor_digest
from runtime import (
    CapabilityBlocked,
    InterruptedRun,
    OutputDirectoryLocked,
    RunLog,
    registry_from,
    select_port,
)
from tasks import (
    CalibrationData,
    make_calibration_tasks,
    make_tasks,
    oracle,
    rows_digest,
)

ROOT = Path(__file__).resolve().parents[1]


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(
            input_ids=([self.bos_token_id] if add_special_tokens else [])
            + [ord(char) for char in text]
        )


def tiny_base(width=32, model_id="test/source"):
    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=width,
        intermediate_size=width * 2,
        num_hidden_layers=2,
        num_attention_heads=width // 16,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=256,
        attention_dropout=0,
    )
    model = Qwen3ForCausalLM(config)
    base = PortalBase(model_id, model, CharTokenizer(), revision="tiny-test-only")
    base.freeze(gradient_checkpointing=True)
    return base


def tiny_portal(base, source=None):
    config = PortalConfig.from_model(
        base.model,
        tasks=("released_a", "released_b"),
        base_model_name_or_path=base.model_id,
        base_model_revision=base.revision,
        rank=2,
        alpha=4,
        d_z=8,
        d_layer=4,
        hidden=16,
        d_core=8,
    )
    torch.manual_seed(23)
    portal = PortalModel(config, torch.randn(2, 8))
    if source:
        portal.core.load_state_dict(source.core.state_dict())
        portal.task_latents.data.copy_(source.task_latents.data)
    else:
        for head in portal.core.B.values():
            torch.nn.init.normal_(head.weight, std=0.04)
            torch.nn.init.normal_(head.bias, std=0.04)
    portal.requires_grad_(False)
    return portal


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def test_task_oracle_and_semantic_splits():
    data = make_tasks()
    assert data.provenance == make_tasks().provenance
    assert data.provenance["sha256"] != make_tasks(seed=18).provenance["sha256"]
    split_inputs = {}
    a = data.provenance["rules"]["amber"]
    b = data.provenance["rules"]["violet_reverse_then_map"]
    for split in ("train", "validation", "test"):
        split_inputs[split] = set()
        for task, rows in getattr(data, split).items():
            positions = [0, 0, 0, 0]
            for row in rows:
                digits = tuple(
                    map(int, row.prompt.split("Input: ")[1].split("\n")[0].split())
                )
                split_inputs[split].add(digits)
                assert row.choices[row.gold_idx] == oracle(task, digits, a, b)
                assert len(set(row.choices)) == 4
                assert len(set(map(len, row.choices))) == 1
                positions[row.gold_idx] += 1
            assert len(set(positions)) == 1
    assert not split_inputs["train"] & split_inputs["test"]
    assert not split_inputs["train"] & split_inputs["validation"]
    assert not split_inputs["validation"] & split_inputs["test"]
    with pytest.raises(ValueError):
        make_tasks(train_examples=500, test_examples=100)


def test_matched_budget_schedule():
    rows = make_tasks(train_examples=4, validation_examples=4, test_examples=4).train[
        "acquisition_a"
    ]
    schedule = batch_schedule(rows, 7, 2, 17)
    assert schedule == batch_schedule(rows, 7, 2, 17)
    assert sum(map(len, schedule)) == 14
    assert set(schedule[0] + schedule[1]) == set(range(4))


def test_portable_latent_different_width_and_frozen_bases(tmp_path):
    source = tiny_base()
    target = tiny_base(width=48, model_id="test/target")
    source_portal = tiny_portal(source)
    target_portal = tiny_portal(target, source_portal)
    verify_shared_core(source_portal, target_portal)
    source_weights = tensor_digest(source.model.state_dict())
    target_weights = tensor_digest(target.model.state_dict())
    data = make_tasks(train_examples=4, validation_examples=4, test_examples=4)
    pilot = Pilot(
        steps_per_task=2,
        batch_size=2,
        train_examples=4,
        validation_examples=4,
        test_examples=4,
    )
    log = RunLog(tmp_path, {"test": "different-width"})
    initial = source_portal.task_latents.mean(dim=0)
    with Adaptation(source, source_portal, "latent", initial, seed=17) as adaptation:
        stats = fit_stage(
            adaptation,
            data.train["acquisition_a"],
            "acquisition_a",
            "source",
            pilot,
            log,
            17,
        )
        assert stats["examples_seen"] == 4
        assert stats["trainable_parameters"] == 8
        learned = adaptation.state()
        assert not torch.equal(learned["latent"], initial)
    with Adaptation(target, target_portal, "latent", initial, seed=17) as adaptation:
        adaptation.restore(learned)
        factors = target_portal(adaptation.latent)
        source_factors = source_portal(learned["latent"])
        assert factors[(0, "q")][0].shape[-1] == 48
        assert source_factors[(0, "q")][0].shape[-1] == 32
        measured = evaluate(adaptation, target, data, "test", pilot)
        assert measured["tasks"]["acquisition_a"]["examples"] == 4
        adaptation.verify_frozen()
    assert tensor_digest(source.model.state_dict()) == source_weights
    assert tensor_digest(target.model.state_dict()) == target_weights
    assert not source.model.model.layers[0].self_attn.q_proj._forward_hooks
    assert not target.model.model.layers[0].self_attn.q_proj._forward_hooks
    with torch.no_grad():
        target_portal.core.l1.weight[0, 0] += 1
    with pytest.raises(CapabilityBlocked, match="core_weights_differ"):
        verify_shared_core(source_portal, target_portal)


def test_native_lora_updates_and_unloads_without_changing_base(tmp_path):
    base = tiny_base()
    portal = tiny_portal(base)
    before = tensor_digest(base.model.state_dict())
    data = make_tasks(train_examples=4, validation_examples=4, test_examples=4)
    pilot = Pilot(
        steps_per_task=2,
        batch_size=2,
        train_examples=4,
        validation_examples=4,
        test_examples=4,
    )
    log = RunLog(tmp_path, {"test": "lora"})
    with Adaptation(
        base, portal, "native_lora", portal.task_latents.mean(dim=0), seed=17
    ) as adaptation:
        stats = fit_stage(
            adaptation,
            data.train["acquisition_a"],
            "acquisition_a",
            "lora",
            pilot,
            log,
            17,
        )
        assert stats["trainable_parameters"] > portal.config.d_z
        assert all("lora_" in key for key in adaptation.state())
        result = evaluate(adaptation, adaptation.base, data, "test", pilot)
        assert 0 <= result["macro_accuracy"] <= 1
    assert tensor_digest(base.model.state_dict()) == before
    assert isinstance(base.model.model.layers[0].self_attn.q_proj, torch.nn.Linear)


@pytest.mark.parametrize("kind", ["latent", "native_lora"])
def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, kind):
    data = make_tasks(train_examples=4, validation_examples=4, test_examples=4)
    pilot = Pilot(
        steps_per_task=4,
        batch_size=2,
        train_examples=4,
        validation_examples=4,
        test_examples=4,
        checkpoint_every=1,
    )
    base = tiny_base()
    portal = tiny_portal(base)
    initial = portal.task_latents.mean(dim=0)
    continuous_log = RunLog(tmp_path / "continuous", {"test": "resume", "kind": kind})
    with Adaptation(base, portal, kind, initial, seed=17) as adaptation:
        continuous_stats = fit_stage(
            adaptation,
            data.train["acquisition_a"],
            "acquisition_a",
            "arm",
            pilot,
            continuous_log,
            17,
        )
        expected = adaptation.state()
    split_log = RunLog(tmp_path / "split", {"test": "resume", "kind": kind})
    original_event = split_log.event

    def stop_after_second_step(event, **values):
        original_event(event, **values)
        if event == "optimizer_step" and values["step"] == 2:
            split_log.request_stop(signal.SIGTERM, None)

    split_log.event = stop_after_second_step
    with (
        Adaptation(base, portal, kind, initial, seed=17) as adaptation,
        pytest.raises(InterruptedRun),
    ):
        fit_stage(
            adaptation,
            data.train["acquisition_a"],
            "acquisition_a",
            "arm",
            pilot,
            split_log,
            17,
        )
    split_log.close()
    resumed_log = RunLog(tmp_path / "split", {"test": "resume", "kind": kind})
    with Adaptation(base, portal, kind, initial, seed=17) as adaptation:
        resumed_stats = fit_stage(
            adaptation,
            data.train["acquisition_a"],
            "acquisition_a",
            "arm",
            pilot,
            resumed_log,
            17,
        )
        actual = adaptation.state()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert resumed_stats["losses"] == continuous_stats["losses"]
    assert resumed_stats["examples_seen"] == continuous_stats["examples_seen"] == 8
    assert resumed_stats["answer_tokens_seen"] == continuous_stats["answer_tokens_seen"]
    events = [
        json.loads(line)
        for line in (tmp_path / "split" / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event"] == "training_resumed" and event["step"] == 2 for event in events
    )


def test_sequential_arm_and_deltas(tmp_path):
    base = tiny_base()
    portal = tiny_portal(base)
    data = make_tasks(train_examples=4, validation_examples=4, test_examples=4)
    pilot = Pilot(
        steps_per_task=1,
        batch_size=2,
        train_examples=4,
        validation_examples=4,
        test_examples=4,
    )
    log = RunLog(tmp_path, {"test": "arm"})
    raw = {"test": evaluate(None, base, data, "test", pilot)}
    result = run_arm(
        base,
        portal,
        "latent",
        portal.task_latents.mean(dim=0),
        data,
        pilot,
        log,
        "arm",
        17,
    )
    assert set(result["snapshots"]) == {"initial", "after_a", "after_b"}
    assert (tmp_path / "arm" / "after_a.safetensors").is_file()
    delta = learning_deltas(result, raw)
    a = result["snapshots"]["after_a"]["test"]["tasks"]["acquisition_a"]["accuracy"]
    b = result["snapshots"]["after_b"]["test"]["tasks"]["acquisition_a"]["accuracy"]
    assert delta["a_retention_accuracy_delta_after_b"] == b - a


def test_registry_all_ports_and_exact_base_identity():
    registry = json.loads((ROOT.parent.parent / "portal-models.json").read_text())
    assert registry["release_count"] == len(registry["ports"]) == 7
    assert registry["shared_core_identical_across_all_releases"]
    assert registry["released_task_latents_identical_across_all_releases"]
    for row in registry["ports"]:
        assert row["inspection"]["weight_download_and_hash_verified"]
        assert row["base"]["revision"] == row["base"]["resolved_revision"]
        assert select_port(registry, row["base"]["id"], row["base"]["revision"]) == row
        assert (ROOT / "configs" / Path(row["status"]["ready_config"]).name).exists()
    with pytest.raises(CapabilityBlocked, match="no_official_port"):
        select_port(registry, "Qwen/Qwen3.5-4B", "anything")
    with pytest.raises(CapabilityBlocked, match="base_revision_mismatch"):
        select_port(registry, "Qwen/Qwen3-8B", "main")
    inkling = next(row for row in registry["ports"] if row["key"] == "inkling")
    assert inkling["status"]["run_eligibility"] == "blocked"
    assert registry_from({"registry_path": "registry.json"}) == registry


def test_resume_rejects_config_change(tmp_path):
    with RunLog(tmp_path, {"seed": 1}):
        pass
    with pytest.raises(ValueError, match="resume_config_mismatch"):
        RunLog(tmp_path, {"seed": 2})
    with RunLog(tmp_path, {"seed": 1}):
        pass


def test_resume_refreshes_runtime_and_clears_resolved_error(tmp_path):
    previous = RunLog(tmp_path, {"seed": 1})
    previous.metrics["runtime"]["platform"] = "previous invocation marker"
    previous.finish("blocked", blockers=[{"code": "previous_blocker"}])
    previous.close()
    current = RunLog(tmp_path, {"seed": 1})
    assert "blockers" not in current.metrics
    assert current.metrics["runtime"]["platform"] != "previous invocation marker"
    assert len(current.metrics["invocations"]) == 2
    assert (
        current.metrics["invocations"][0]["runtime"]["platform"]
        == "previous invocation marker"
    )


def test_output_lock_rejects_duplicate_worker_without_mutating_owner(tmp_path):
    output = tmp_path / "output"
    config = {
        "mode": "smoke",
        "model_id": "test/no-released-port",
        "revision": "test-only",
        "model_path": str(tmp_path),
        "seed": 17,
        "registry_path": str(ROOT / "registry.json"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(ROOT / "run.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    with RunLog(output, config):
        before = {
            path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
        }
        duplicate = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        assert duplicate.returncode == 2
        assert '"code": "output_directory_locked"' in duplicate.stderr
        assert before == {
            path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
        }
    after_release = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    assert after_release.returncode == 2
    assert "output_directory_lock_rejected" not in after_release.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert "no_official_port" in metrics["blockers"][0]["detail"]
    with RunLog(output, config):
        pass


def test_output_lock_lifetime_exception_cleanup_and_safe_close(tmp_path):
    config = {"seed": 17}
    with (
        pytest.raises(RuntimeError, match="body_failed"),
        RunLog(tmp_path, config) as owner,
    ):
        with pytest.raises(OutputDirectoryLocked):
            RunLog(tmp_path, config)
        owner.event("owner_still_holds_lock")
        raise RuntimeError("body_failed")
    owner.close()
    with pytest.raises(RuntimeError, match="run_log_closed"):
        owner.event("must_not_write")
    with RunLog(tmp_path, config):
        assert (tmp_path / ".portal-run.lock").exists()


def test_pilot_budget_validation():
    with pytest.raises(ValueError):
        Pilot(steps_per_task=0)
    with pytest.raises(ValueError):
        replace(Pilot(), latent_learning_rate=float("nan"))


def test_transfer_orchestration_with_real_tiny_models_at_loading_boundary(
    tmp_path, monkeypatch
):
    source_base = tiny_base(model_id="test/source")
    target_base = tiny_base(width=48, model_id="test/target")
    source_portal = tiny_portal(source_base)
    target_portal = tiny_portal(target_base, source_portal)
    models = {"source": source_base, "target": target_base}
    portals = {"source": source_portal, "target": target_portal}
    registry = {
        "all_official_ports_accounted_for": True,
        "shared_core_identical_across_all_releases": True,
        "ports": [
            {
                "key": key,
                "base": {"id": base.model_id, "revision": base.revision},
                "status": {"blockers": []},
            }
            for key, base in models.items()
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    config = {
        "mode": "transfer",
        "seed": 17,
        "model_id": target_base.model_id,
        "revision": target_base.revision,
        "model_path": "fixture-only",
        "registry_path": str(registry_path),
        "source": {
            "model_id": source_base.model_id,
            "revision": source_base.revision,
            "model_path": "fixture-only",
        },
        "pilot": {
            "steps_per_task": 1,
            "batch_size": 2,
            "train_examples": 4,
            "validation_examples": 4,
            "test_examples": 4,
            "source_native_lora": True,
        },
    }
    monkeypatch.setattr(
        engine, "load_port", lambda row, _config, _log: portals[row["key"]]
    )
    monkeypatch.setattr(
        engine,
        "load_local_base",
        lambda row, _path, _device, _log, _checkpointing: models[row["key"]],
    )
    monkeypatch.setattr(engine, "ensure_gpu", lambda _config, _log: torch.device("cpu"))
    log = RunLog(tmp_path / "run", config)
    engine.transfer(config, log)
    assert (
        log.metrics["target"]["portable_transfer"]["new_task_target_optimizer_steps"]
        == 0
    )
    assert log.metrics["target"]["native_latent"]["trainable_parameters"] == 8
    assert (
        log.metrics["source"]["source_latent"]["training"]["acquisition_a"][
            "examples_seen"
        ]
        == 2
    )
    assert (
        log.metrics["target"]["native_lora"]["training"]["acquisition_a"][
            "examples_seen"
        ]
        == 2
    )
    assert not log.metrics["runtime"]["gpu_execution"]
    assert set(log.metrics["target"]["transfer_vs_native"]) == {"after_a", "after_b"}


def test_calibration_preserves_training_and_excludes_all_prior_evaluation_inputs():
    original = make_tasks()
    data = make_calibration_tasks()
    assert (
        original.provenance["sha256"]
        == "aecf32ff0a071de00aabe692746deafa36de5b7c17fce5e425c59442bbf76c20"
    )
    assert list(data.train) == original.train["acquisition_a"]
    assert (
        rows_digest(data.train)
        == "0c6115ef34862112f8c5ecc283d958bc6dd4b6aa56806d9d010f26961d537157"
    )
    assert (
        rows_digest(data.validation)
        == "01e2a90ca119f8777ce5773a92f8528f020f92545a4362486990f8d9002fb4c0"
    )
    prior_prompts = {
        row.prompt
        for split in ("train", "validation", "test")
        for row in getattr(original, split)["acquisition_a"]
    }
    assert not prior_prompts & {row.prompt for row in data.validation}
    assert len({row.prompt for row in data.validation}) == 128
    changed_validation = make_calibration_tasks(validation_seed=2719)
    assert changed_validation.train == data.train
    assert changed_validation.validation != data.validation
    assert set(data.as_dict()["splits"]) == {"train", "validation"}
    assert not hasattr(data, "test")
    for split in ("test", "heldout", "target"):
        with pytest.raises(ValueError, match="calibration_split_forbidden"):
            data.rows(split)
        with pytest.raises(ValueError, match="calibration_split_forbidden"):
            score_split(None, None, data, split, CalibrationPlan(), None)


def test_checkpointing_preserves_latent_gradient_through_recomputation():
    data = make_calibration_tasks(train_examples=4, validation_examples=4)
    gradients, losses = [], []
    for checkpointing in (False, True):
        base = tiny_base()
        if not checkpointing:
            base.model.gradient_checkpointing_disable()
        assert base.model.is_gradient_checkpointing == checkpointing
        portal = tiny_portal(base)
        with Adaptation(
            base, portal, "latent", portal.task_latents.mean(dim=0), seed=17
        ) as adaptation:
            ids, mask, labels = collate_gold_batch(
                base.tokenizer, list(data.train), max_prompt=128, device=base.device
            )
            base.model.train()
            with adaptation.activation():
                loss = base.model(
                    input_ids=ids, attention_mask=mask, labels=labels, use_cache=False
                ).loss
                loss.backward()
            gradients.append(adaptation.latent.grad.detach().clone())
            losses.append(loss.detach().clone())
            adaptation.verify_frozen()
    assert torch.isfinite(gradients[1]).all()
    assert gradients[1].norm() > 0
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


@pytest.fixture
def calibration_fixture(tmp_path, monkeypatch):
    plan = CalibrationPlan(
        max_steps=4, checkpoint_steps=(2, 4), train_examples=4, validation_examples=4
    )
    data = make_calibration_tasks(train_examples=4, validation_examples=4)
    row = {
        "key": "tiny",
        "base": {"id": "test/source", "revision": "tiny-test-only"},
        "artifact": {"id": "test/portal", "revision": "tiny-test-only"},
        "status": {"blockers": []},
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"all_official_ports_accounted_for": True, "ports": [row]})
    )
    config = {
        "mode": "source_calibration",
        "model_id": row["base"]["id"],
        "revision": row["base"]["revision"],
        "model_path": "fixture-only",
        "seed": 17,
        "device": "cuda:0",
        "registry_path": str(registry_path),
        "source_training_rows_sha256": rows_digest(data.train),
        "calibration": asdict(plan),
    }
    accesses, loads = [], []
    original_rows = CalibrationData.rows

    def guarded_rows(self, split):
        accesses.append(split)
        assert split in ("train", "validation")
        return original_rows(self, split)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("source_calibration_accessed_legacy_test_or_target_path")

    def load_base(row, _path, _device, log, _checkpointing):
        loads.append(row["base"]["id"])
        base = tiny_base()
        log.metrics["provenance"][base.model_id] = {
            "revision": base.revision,
            "weights_sha256": tensor_digest(base.model.state_dict()),
        }
        return base

    def load_port(row, _config, log):
        portal = tiny_portal(tiny_base())
        log.metrics["provenance"][row["artifact"]["id"]] = {
            "revision": row["artifact"]["revision"],
            "weights_sha256": tensor_digest(portal.state_dict()),
        }
        return portal

    monkeypatch.setattr(CalibrationData, "rows", guarded_rows)
    monkeypatch.setattr(tasks, "make_tasks", forbidden)
    monkeypatch.setattr(engine, "make_tasks", forbidden)
    monkeypatch.setattr(engine, "measure", forbidden)
    monkeypatch.setattr(engine, "target_phase", forbidden)
    monkeypatch.setattr(calibration, "load_local_base", load_base)
    monkeypatch.setattr(calibration, "load_port", load_port)
    monkeypatch.setattr(
        calibration, "ensure_gpu", lambda _config, _log: torch.device("cpu")
    )
    return config, accesses, loads


def test_calibration_runs_only_source_train_dev_with_exact_prediction_receipts(
    tmp_path, calibration_fixture
):
    config, accesses, loads = calibration_fixture
    output = tmp_path / "run"
    with RunLog(output, config) as log:
        source_calibration(config, log)
        results = log.metrics["source_calibration"]
        assert set(results) == {"latent_lr_0.002", "latent_lr_0.02", "native_lora"}
        assert log.metrics["qualification"]["status"] == "source_fit_not_qualified"
        assert not log.metrics["qualification"]["gpu_execution_observed"]
        assert log.metrics["scientific_contract"]["test_access"] is False
        assert log.metrics["scientific_contract"]["target_model_loaded"] is False
        assert "not a research conclusion" in log.metrics["claim_scope"]
    assert loads == ["test/source"]
    assert set(accesses) == {"train", "validation"}
    assert set(json.loads((output / "data.json").read_text())["splits"]) == {
        "train",
        "validation",
    }
    for arm, result in results.items():
        for steps, milestone in result["milestones"].items():
            assert milestone["training"]["steps"] == int(steps)
            assert milestone["training"]["examples_seen"] == int(steps) * 4
            assert milestone["all_gradients_finite_and_nonzero"]
            assert milestone["parameter_change_l2"] > 0
            assert milestone["reload"]["maximum_nll_or_choice_score_error"] == 0
            directory = output / arm / f"step-{steps}"
            assert (directory / "checkpoint.pt").is_file()
            measured = json.loads((directory / "validation.json").read_text())
            assert len(measured["rows"]) == measured["examples"] == 4
            assert measured["gold_tokens"] == sum(
                row["gold_tokens"] for row in measured["rows"]
            )
            assert measured["gold_nll"] == pytest.approx(
                sum(row["gold_nll_sum"] for row in measured["rows"])
                / measured["gold_tokens"]
            )
            for row in measured["rows"]:
                assert len(row["choice_logprob_per_character"]) == 4
                assert row["predicted_answer"] == row["choices"][row["predicted_index"]]
                assert (
                    row["gold_nll_per_token"]
                    == row["gold_nll_sum"] / row["gold_tokens"]
                )
                assert row["correct"] == (row["gold_idx"] == row["predicted_index"])
    assert not list(output.rglob("test.json"))


def test_calibration_interrupted_milestone_resume_is_identical(
    tmp_path, calibration_fixture
):
    config, _accesses, loads = calibration_fixture
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    with RunLog(continuous, config) as log:
        source_calibration(config, log)
        expected = log.metrics["source_calibration"]
    with RunLog(interrupted, config) as log:
        original_event = log.event

        def interrupt(event, **values):
            original_event(event, **values)
            if (
                event == "optimizer_step"
                and values["arm"] == "latent_lr_0.002"
                and values["step"] == 3
            ):
                log.request_stop(signal.SIGTERM, None)

        log.event = interrupt
        with pytest.raises(InterruptedRun):
            source_calibration(config, log)
    cached_path = interrupted / "latent_lr_0.002" / "step-2" / "result.json"
    cached_mtime = cached_path.stat().st_mtime_ns
    checkpoint = torch.load(
        interrupted / "latent_lr_0.002" / "acquisition_a" / "checkpoint.pt",
        weights_only=True,
    )
    assert checkpoint["step"] == 3
    with RunLog(interrupted, config) as log:
        source_calibration(config, log)
        actual = log.metrics["source_calibration"]
    assert cached_path.stat().st_mtime_ns == cached_mtime
    assert loads == ["test/source"] * 3
    for arm, result in expected.items():
        assert actual[arm]["qualification"] == result["qualification"]
        for steps, milestone in result["milestones"].items():
            resumed = actual[arm]["milestones"][steps]
            assert (
                resumed["trainable_state_sha256"] == milestone["trainable_state_sha256"]
            )
            assert resumed["training"]["losses"] == milestone["training"]["losses"]
            assert (
                resumed["training"]["gradient_norms"]
                == milestone["training"]["gradient_norms"]
            )
            assert (
                resumed["training"]["examples_seen"]
                == milestone["training"]["examples_seen"]
            )
            for split in ("train", "validation"):
                relative = Path(arm) / f"step-{steps}" / f"{split}.json"
                assert (interrupted / relative).read_bytes() == (
                    continuous / relative
                ).read_bytes()
    events = [
        json.loads(line)
        for line in (interrupted / "events.jsonl").read_text().splitlines()
    ]
    for arm in actual:
        assert [
            event["step"]
            for event in events
            if event["event"] == "optimizer_step" and event["arm"] == arm
        ] == [1, 2, 3, 4]
    assert any(
        event["event"] == "training_resumed" and event["step"] == 3 for event in events
    )


def test_calibration_validation_seed_cannot_change_training(
    tmp_path, calibration_fixture
):
    config, _accesses, _loads = calibration_fixture
    results = []
    for seed in (2718, 2719):
        current = {
            **config,
            "calibration": {**config["calibration"], "validation_seed": seed},
        }
        with RunLog(tmp_path / str(seed), current) as log:
            source_calibration(current, log)
            results.append(log.metrics["source_calibration"])
    for arm in results[0]:
        left = results[0][arm]["milestones"]["4"]
        right = results[1][arm]["milestones"]["4"]
        assert left["trainable_state_sha256"] == right["trainable_state_sha256"]
        assert left["training"]["losses"] == right["training"]["losses"]
        assert left["training"]["gradient_norms"] == right["training"]["gradient_norms"]
        assert (
            left["validation"]["prediction_sha256"]
            != right["validation"]["prediction_sha256"]
        )


@pytest.mark.parametrize(
    "relative,error",
    [
        ("anchors.json", "calibration_receipt_hash_mismatch"),
        ("latent_lr_0.002/step-2/result.json", "calibration_receipt_hash_mismatch"),
        (
            "latent_lr_0.002/step-2/validation.json",
            "calibration_artifact_hash_mismatch",
        ),
    ],
)
def test_calibration_resume_rejects_changed_receipts(
    tmp_path, calibration_fixture, relative, error
):
    config, _accesses, _loads = calibration_fixture
    output = tmp_path / "run"
    with RunLog(output, config) as log:
        source_calibration(config, log)
    changed = output / relative
    changed.write_text(changed.read_text() + " ")
    with RunLog(output, config) as log, pytest.raises(ValueError, match=error):
        source_calibration(config, log)


def test_calibration_resume_rejects_same_metadata_changed_base_weights(
    tmp_path, calibration_fixture, monkeypatch
):
    config, _accesses, _loads = calibration_fixture
    output = tmp_path / "run"
    with RunLog(output, config) as log:
        source_calibration(config, log)
    original_load = calibration.load_local_base

    def changed_base(*args):
        base = original_load(*args)
        with torch.no_grad():
            next(base.model.parameters()).view(-1)[0].add_(1.0)
        return base

    monkeypatch.setattr(calibration, "load_local_base", changed_base)
    with (
        RunLog(output, config) as log,
        pytest.raises(ValueError, match="source_calibration_resume_provenance_changed"),
    ):
        source_calibration(config, log)


@pytest.mark.parametrize(
    "changed,error",
    [
        (
            {"target": {"model_id": "forbidden"}},
            "source_calibration_config_must_not_include_target",
        ),
        (
            {"source_training_rows_sha256": "different"},
            "source_calibration_training_rows_changed",
        ),
    ],
)
def test_calibration_rejects_target_and_changed_training_before_model_load(
    tmp_path, calibration_fixture, changed, error
):
    config, _accesses, loads = calibration_fixture
    config = {**config, **changed}
    with (
        RunLog(tmp_path / "run", config) as log,
        pytest.raises(ValueError, match=error),
    ):
        source_calibration(config, log)
    assert not loads


def test_calibration_cli_rejects_cpu_before_any_measurement(tmp_path):
    config = json.loads((ROOT / "configs" / "calibrate-qwen3-1.7b.json").read_text())
    config["device"] = "cpu"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "run.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "blocked"
    assert "production_lane_requires_cuda_0" in metrics["blockers"][0]["detail"]
    assert not metrics["measurements"]
    assert not metrics["runtime"]["gpu_execution"]
    assert not (output / "data.json").exists()


@pytest.mark.parametrize(
    "changed",
    [
        {"max_steps": 257},
        {"batch_size": 8},
        {"checkpoint_steps": (128, 32, 256)},
        {"latent_learning_rates": (0.001, 0.02)},
        {"lora_learning_rate": 0.002},
        {"min_train_accuracy": 0.8},
        {"min_validation_accuracy": 0.7},
        {"min_validation_accuracy_gain": 0.1},
        {"min_validation_nll_drop": 0.0},
        {"reload_atol": 1e-5},
    ],
)
def test_calibration_rejects_weakened_or_changed_preregistration(changed):
    with pytest.raises(ValueError):
        CalibrationPlan(**changed)


def test_calibration_qualification_requires_final_step_and_both_anchor_improvements():
    plan = CalibrationPlan()
    references = {
        "no_adapter": {"validation": {"accuracy": 0.4, "gold_nll": 2.0}},
        "initial_latent": {"validation": {"accuracy": 0.5, "gold_nll": 1.8}},
    }
    passed = {
        "train": {"accuracy": 0.95},
        "validation": {"accuracy": 0.9, "gold_nll": 1.5},
        "all_gradients_finite_and_nonzero": True,
        "parameter_change_l2": 0.01,
        "reload": {"passed": True},
        "training": {"examples_seen": 1024, "steps": 256},
    }
    result = {
        "kind": "latent",
        "milestones": {"32": passed, "128": passed, "256": passed},
    }
    assert qualify(result, references, plan)["passed"]
    result["milestones"]["256"] = {
        **passed,
        "validation": {"accuracy": 0.6, "gold_nll": 1.9},
    }
    assert not qualify(result, references, plan)["passed"]
    result["milestones"]["256"] = passed
    references["initial_latent"]["validation"]["accuracy"] = 0.8
    assert not qualify(result, references, plan)["checks"]["validation_gain_vs_initial"]
