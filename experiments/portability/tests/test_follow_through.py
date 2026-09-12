import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from portallib import PortalModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import follow_through as follow
from calibrate_target import pin_bundle
from data import SEQUENCE_TASKS, digest, write_json
from learner import frozen_base_tensors, save_native, tensor_hash
from prepare_follow_through import sealed_length_holdout
from test_target_calibration import tiny_models

ROOT = Path(__file__).resolve().parents[1]


def configured(name="constructed_target16"):
    return json.loads((ROOT / f"configs/follow_through/{name}.json").read_text())


def make_sources(tmp_path, monkeypatch, name="constructed_target16"):
    config = configured(name)
    config["runtime"] = {
        "device": "cpu",
        "dtype": "float32",
        "autocast": False,
        "cpu_threads": 1,
    }
    base, initial, constructed, target = tiny_models()
    for source_name, portal in (
        ("source_initial", initial),
        ("source_constructed", constructed),
        ("target_portal", target),
    ):
        if source_name in config["inputs"]:
            path = tmp_path / source_name
            save_native(portal, path)
            config["inputs"][source_name] = {
                **pin_bundle(path),
                "tensor_sha256": tensor_hash(portal.state_dict()),
            }
    protocol = config["protocol"]
    protocol["checkpoints"] = [0, 4, 8]
    protocol["gradient_gate_step"] = 4
    protocol["eval_batch"] = 8
    protocol["training_ids"] = {
        task: ids[:4] for task, ids in protocol["training_ids"].items()
    }
    config["protocol_sha256"] = digest(protocol)
    config["expected_hooks"] = len(target.config.projection_targets)
    monkeypatch.setattr(
        follow,
        "validation_rows",
        lambda config: [
            r for values in follow.training_rows(config).values() for r in values
        ],
    )
    return config, base


def test_configs_seal_different_questions_and_actual_construction():
    for path in sorted((ROOT / "configs/follow_through").glob("*.json")):
        config = json.loads(path.read_text())
        follow.validate(config)
        rows = follow.training_rows(config)
        schedule = follow.schedule_for(config, rows)
        assert len(schedule) == config["protocol"]["checkpoints"][-1]
        assert len(follow.retention_rows(config)) == 128
        follow.final_panels(config)
    source = configured("examples_released")
    assert "source_constructed" not in source["inputs"]
    assert source["protocol"]["checkpoints"][-1] == 512
    target = configured()
    assert (
        target["inputs"]["source_constructed"]["tensor_sha256"]
        == "529e0fa922245ec16457ff7b057f4668a4b123c045c75c11490cd5abf736a619"
    )
    assert (
        target["inputs"]["source_constructed"]["files"]["model.safetensors"]["sha256"]
        == "e69e16ed9b0fd02e709dda37d99eba93e861567053af4cd145aced3190e84500"
    )
    assert target["protocol"]["checkpoints"] == [0, 20, 80, 160, 320]
    assert (
        configured("constructed_target64")["protocol"]["training_ids"][
            SEQUENCE_TASKS[0]
        ][:16]
        == target["protocol"]["training_ids"][SEQUENCE_TASKS[0]]
    )


def test_new_length_holdout_is_reproducible_balanced_and_oracle_correct():
    fixture = json.loads((ROOT / "data/sequence303.json").read_text())
    stored = json.loads((ROOT / "data/sequence303_length4_holdout.json").read_text())
    assert stored == sealed_length_holdout(fixture)
    assert len(stored["rows"]) == 384
    for task in SEQUENCE_TASKS:
        rows = [r for r in stored["rows"] if r["task"] == task]
        assert {
            position: sum(r["gold_idx"] == position for r in rows)
            for position in range(4)
        } == dict.fromkeys(range(4), 32)
        assert all(len(r["group"].split()) == 4 for r in rows)


def test_conservative_target64_changes_only_shared_learning_rate():
    previous = configured("constructed_target64")
    conservative = configured("constructed_target64_conservative")
    follow.validate(conservative)
    for arm in conservative["protocol"]["arms"]:
        assert arm["learning_rate"] == 1e-4
        arm["learning_rate"] = 1e-3
    conservative.pop("optimizer_control")
    conservative["protocol_sha256"] = digest(conservative["protocol"])
    assert conservative == previous


def test_pinned_input_and_example_only_controls_reject_swaps(tmp_path, monkeypatch):
    config, _ = make_sources(tmp_path, monkeypatch, "examples_released")
    config["inputs"]["source_constructed"] = config["inputs"]["source_initial"]
    with pytest.raises(ValueError, match="MUST_NOT_ACCESS"):
        follow.validate(config)
    del config["inputs"]["source_constructed"]
    config["inputs"]["source_initial"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="WRONG_SOURCE_TENSORS"):
        follow.load_source(config, "source_initial")


def test_schedule_matches_all_arms_and_never_sees_validation():
    config = configured()
    rows = follow.training_rows(config)
    schedule = follow.schedule_for(config, rows)
    allowed = {r.id for values in rows.values() for r in values}
    assert all(
        len(ids) == 4 and set(ids) <= allowed
        for step in schedule
        for ids in step["ids"].values()
    )
    assert schedule == follow.schedule_for(config, rows)
    assert all(
        sum(len(step["ids"][task]) for step in schedule) == 1280
        for task in SEQUENCE_TASKS
    )
    for offset in range(0, len(schedule), 4):
        assert all(
            {key for step in schedule[offset : offset + 4] for key in step["ids"][task]}
            == {r.id for r in rows[task]}
            for task in SEQUENCE_TASKS
        )


def test_rte_start_lora_is_same_actual_adapter(tmp_path, monkeypatch):
    config, base = make_sources(tmp_path, monkeypatch, "examples_released")
    native_arm, lora_arm = config["protocol"]["arms"][0], config["protocol"]["arms"][-1]
    native, lora = (
        follow.build_adapter(config, native_arm),
        follow.build_adapter(config, lora_arm),
    )
    rows = [r for values in follow.training_rows(config).values() for r in values]
    assert follow.measure(base, native, rows, native_arm, config) == follow.measure(
        base, lora, rows, lora_arm, config
    )


@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_target_training_reaches_alignment_and_freezes_actual_core(
    tmp_path, monkeypatch, arm_index
):
    config, base = make_sources(tmp_path, monkeypatch)
    arm = config["protocol"]["arms"][arm_index]
    adapter = follow.build_adapter(config, arm)
    original = {k: v.clone() for k, v in adapter.state_dict().items()}
    base_before = tensor_hash(frozen_base_tensors(base.model))
    rows = follow.training_rows(config)
    output = tmp_path / "run"
    output.mkdir()
    trained = follow.fit_arm(
        base, adapter, arm, config, rows, follow.schedule_for(config, rows), output
    )
    assert trained["status"] == "completed_budget"
    assert trained["optimizer_updates"] == 8
    assert len(trained["checkpoints"]) == 3
    if isinstance(adapter, PortalModel):
        assert all(
            torch.equal(value, adapter.state_dict()[key])
            for key, value in original.items()
            if not key.startswith("alignment.")
        )
        assert set(trained["active_gradient_families"]) == {
            "alignment.input",
            "alignment.output",
            "alignment.layer_embeddings",
        }
    assert base_before == tensor_hash(frozen_base_tensors(base.model))
    restored = follow.restore_adapter(trained["checkpoints"]["8"]["saved"], "cpu")
    assert (
        follow.measure(base, restored, follow.validation_rows(config), arm, config)
        == trained["checkpoints"]["8"]["validation"]
    )


@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_source_examples_train_exact_allowed_components_and_keep_old_vectors(
    tmp_path, monkeypatch, arm_index
):
    config, base = make_sources(tmp_path, monkeypatch, "examples_released")
    arm = config["protocol"]["arms"][arm_index]
    adapter = follow.build_adapter(config, arm)
    initial = {k: v.clone() for k, v in adapter.state_dict().items()}
    rows = follow.training_rows(config)
    output = tmp_path / "run"
    output.mkdir()
    trained = follow.fit_arm(
        base, adapter, arm, config, rows, follow.schedule_for(config, rows), output
    )
    assert trained["status"] == "completed_budget"
    assert torch.equal(initial["task_latents"][0], adapter.task_latents[0])
    assert any(
        not torch.equal(v, adapter.state_dict()[k])
        for k, v in initial.items()
        if k.startswith("alignment.input.")
    )
    if arm["train"] == "heads":
        assert all(
            torch.equal(v, adapter.state_dict()[k])
            for k, v in initial.items()
            if k.startswith(("core.l1.", "core.l2.", "core.film."))
        )
    if arm["train_latents"]:
        assert not torch.equal(initial["task_latents"][1:], adapter.task_latents[1:])
    else:
        assert torch.equal(initial["task_latents"], adapter.task_latents)
        assert torch.equal(
            initial["alignment.layer_embeddings.weight"],
            adapter.alignment.layer_embeddings.weight,
        )


def test_zero_initialization_has_one_zero_path_and_can_learn(tmp_path, monkeypatch):
    config, base = make_sources(tmp_path, monkeypatch, "examples_zero")
    arm = config["protocol"]["arms"][0]
    adapter = follow.build_adapter(config, arm)
    assert all(torch.count_nonzero(p) for p in adapter.alignment.output.values())
    assert all(
        torch.count_nonzero(b) == 0
        for _, b in follow.factors(adapter, SEQUENCE_TASKS[0], arm).values()
    )
    rows = follow.training_rows(config)
    output = tmp_path / "run"
    output.mkdir()
    trained = follow.fit_arm(
        base, adapter, arm, config, rows, follow.schedule_for(config, rows), output
    )
    assert trained["status"] == "completed_budget"
    assert "core_hidden" in trained["active_gradient_families"]


def test_double_zero_stops_before_useless_updates(tmp_path, monkeypatch):
    config, base = make_sources(tmp_path, monkeypatch, "examples_zero")
    arm = config["protocol"]["arms"][0]
    adapter = follow.build_adapter(config, arm)
    with torch.no_grad():
        for p in adapter.alignment.output.values():
            p.zero_()
    rows = follow.training_rows(config)
    output = tmp_path / "run"
    output.mkdir()
    trained = follow.fit_arm(
        base, adapter, arm, config, rows, follow.schedule_for(config, rows), output
    )
    assert trained["status"] == "failed_zero_gradient"
    assert trained["optimizer_updates"] == 0
    assert list(trained["checkpoints"]) == ["0"]


def test_nonfinite_update_is_counted_and_not_saved(tmp_path, monkeypatch):
    config, base = make_sources(tmp_path, monkeypatch)
    arm = config["protocol"]["arms"][0]
    adapter = follow.build_adapter(config, arm)
    original_step = torch.optim.AdamW.step

    def corrupt_update(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_update)
    rows = follow.training_rows(config)
    output = tmp_path / "run"
    output.mkdir()
    trained = follow.fit_arm(
        base, adapter, arm, config, rows, follow.schedule_for(config, rows), output
    )
    assert trained["status"] == "failed_nonfinite_optimization"
    assert trained["optimizer_updates"] == 1
    assert trained["finite_completed_updates"] == 0
    assert trained["example_exposures_per_task"] == dict.fromkeys(SEQUENCE_TASKS, 4)
    assert list(trained["checkpoints"]) == ["0"]


def test_scale_uses_actual_full_update_norm(tmp_path, monkeypatch):
    config, _ = make_sources(tmp_path, monkeypatch, "examples_released")
    arm = config["protocol"]["arms"][0]
    adapter = follow.build_adapter(config, arm)
    actual = follow.scale_diagnostics(adapter, arm)[SEQUENCE_TASKS[0]]
    generated = follow.factors(adapter, SEQUENCE_TASKS[0], arm)
    for row in actual:
        a, b = generated[row["layer"], row["module"]]
        assert row["delta_frobenius"] == pytest.approx(
            float(torch.linalg.norm((b.double() @ a.double()) * adapter.config.scaling))
        )


def test_release_probes_are_not_used_for_training_or_validation():
    config = configured("examples_released")
    saved = follow.retention_rows(config)
    train = [r for values in follow.training_rows(config).values() for r in values]
    validation = follow.validation_rows(config)
    assert not {r.id for r in saved} & {r.id for r in train + validation}
    assert {r.task for r in saved} == {
        "boolq",
        "hellaswag",
        "winogrande",
        "commonsense_qa",
    }


def test_insertion_compares_constructed_to_untouched_source():
    baseline = {"metrics": {"boolq": {"accuracy": 0.875, "examples": 32}}}
    inserted = {"metrics": {"boolq": {"accuracy": 0.5, "examples": 32}}}
    raw = {"released_tasks": {"metrics": {"boolq": {"accuracy": 0.625}}}}
    results = {
        "untouched": {"checkpoints": {"0": {"released_tasks": baseline}}},
        "constructed": {"checkpoints": {"0": {"released_tasks": inserted}}},
    }
    result = follow.insertion_comparison(results, raw)["boolq"]
    assert result["before"] == 0.875
    assert result["after"] == 0.5
    assert result["change"] == -0.375
    assert result["initial_adapter_gain_qualified"]


def test_new_pid_is_required_before_heldout_or_model_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must fail before loading or evaluating")

    monkeypatch.setattr(follow, "load_base", forbidden)
    monkeypatch.setattr(follow, "final_panels", forbidden)
    with pytest.raises(ValueError, match="NEW_EVALUATION_PID"):
        follow.evaluate_saved(configured(), tmp_path, os.getpid())


@pytest.mark.parametrize("steps", [[0, 20, 80, 160, 320], [0, 16, 64, 128, 256, 512]])
def test_saved_json_evaluates_retention_at_numeric_final_step(
    tmp_path, monkeypatch, steps
):
    config = configured()
    config["protocol"]["arms"] = config["protocol"]["arms"][:1]
    base, _, _, _ = tiny_models()
    adapters = {step: torch.nn.Linear(1, 1) for step in steps}
    for step, adapter in adapters.items():
        adapter.step = step
    checkpoints = {
        str(step): {
            "saved": {
                "step": step,
                "tensor_sha256": tensor_hash(adapters[step].state_dict()),
            },
            "validation": {"step": step},
        }
        for step in steps
    }
    receipt = {
        "pid": -1,
        "config_sha256": digest(config),
        "input_manifest": {},
        "base_sha256": tensor_hash(frozen_base_tensors(base.model)),
        "arms": {
            "constructed": {"checkpoints": checkpoints, "status": "completed_budget"}
        },
    }
    write_json(tmp_path / "training_receipt.json", receipt)
    write_json(
        tmp_path / "training_receipt.pin.json",
        follow.file_pin(tmp_path / "training_receipt.json"),
    )
    serialized = json.loads((tmp_path / "training_receipt.json").read_text())
    assert list(serialized["arms"]["constructed"]["checkpoints"])[-1] != str(steps[-1])
    retained_steps = []

    def measure(base, adapter, rows, arm, config):
        step = adapter.step if adapter is not None else -1
        if rows == "validation":
            return {"step": step}
        if rows == "retained":
            retained_steps.append(step)
        accuracy = 1.0 if step == 0 else 0.75 if step == steps[-1] else 0.25
        return {"metrics": {"boolq": {"accuracy": accuracy, "examples": 32}}}

    monkeypatch.setattr(follow, "verify_inputs", lambda config: {})
    monkeypatch.setattr(follow, "load_base", lambda *args, **kwargs: base)
    monkeypatch.setattr(
        follow, "restore_adapter", lambda saved, device: adapters[saved["step"]]
    )
    monkeypatch.setattr(follow, "final_panels", lambda config: {"abc": "abc"})
    monkeypatch.setattr(follow, "retention_rows", lambda config: "retained")
    monkeypatch.setattr(follow, "validation_rows", lambda config: "validation")
    monkeypatch.setattr(follow, "measure", measure)
    result = follow.evaluate_saved(config, tmp_path, -1)["arms"]["constructed"]
    assert retained_steps == [-1, 0, steps[-1]]
    assert result["retention_steps"] == {"before": 0, "after": steps[-1]}
    assert result["retention"]["boolq"]["change"] == -0.25
    assert [
        int(step)
        for step, panel in result["checkpoints"].items()
        if "released_tasks" in panel
    ] == [0, steps[-1]]


def test_complete_cpu_run_reloads_in_new_process_before_holdout(tmp_path, monkeypatch):
    config, base = make_sources(tmp_path, monkeypatch)
    monkeypatch.undo()
    snapshot = tmp_path / "base" / base.revision
    snapshot.mkdir(parents=True)
    base.model.save_pretrained(snapshot)
    base.tokenizer.save_pretrained(snapshot)
    config["inputs"]["base"] = pin_bundle(snapshot)
    config["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "cache_dir": str(tmp_path / "cache"),
        "local_path": str(snapshot),
    }
    examples = follow.validation_rows(configured())[:2]
    retention = {"validation": [], "indices": []}
    for row in examples:
        retention["validation"].append(
            {
                "task": "rte",
                "prompt": row.prompt,
                "choices": list(row.choices),
                "gold_idx": row.gold_idx,
            }
        )
        retention["indices"].append(
            {
                "task": "rte",
                "prompt_sha256": hashlib.sha256(
                    " ".join(row.prompt.split()).casefold().encode()
                ).hexdigest(),
            }
        )
    retention_path = tmp_path / "retention.json"
    write_json(retention_path, retention)
    config["inputs"]["retention_probes"] = {
        "path": str(retention_path),
        **follow.file_pin(retention_path),
    }
    config["protocol"]["retention_indices_sha256"] = digest(retention["indices"])
    config["protocol_sha256"] = digest(config["protocol"])
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    output = tmp_path / "experiment"
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(ROOT / "follow_through.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    executed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert executed.returncode == 0, executed.stdout[-3000:] + executed.stderr[-3000:]
    result = json.loads((output / "result.json").read_text())
    assert result["training_pid"] != result["evaluation_pid"]
    assert all(arm["reload_predictions_exact"] for arm in result["arms"].values())
    assert set(result["arms"]) == {"constructed", "untouched", "lora"}
    assert all(
        set(arm["checkpoints"]) == {"0", "4", "8"} for arm in result["arms"].values()
    )
