from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_anchoring import ROOT, source_config, tiny_assets

import run
import train_anchored as runner
from data import SEQUENCE_TASKS, write_json
from learner import load_base
from repair_alignment import file_hash, save_checkpoint


def canonical(weight=0):
    return json.loads(
        (ROOT / f"configs/anchored/native_replay_anchor{weight}.json").read_text()
    )


def cpu_fixture(directory, weight=0):
    base, portal, targets = tiny_assets()
    base_path = directory / "base" / base.revision
    base.model.save_pretrained(base_path)
    base.tokenizer.save_pretrained(base_path)
    source = source_config()
    source["models"]["qwen8"]["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "local_path": str(base_path),
        "cache_dir": None,
    }
    source_path = directory / "cpu_source_config.json"
    write_json(source_path, source)
    config = canonical(weight)
    config["source_initial"] = save_checkpoint(portal, directory / "initial")
    config["targets"] = {
        name: save_checkpoint(model, directory / name)
        for name, model in targets.items()
    }
    return config, source, source_path


def use_cpu(source, monkeypatch):
    def require_cpu(_output, seed):
        torch.manual_seed(seed)
        torch.set_num_threads(1)

    monkeypatch.setattr(runner, "read_source_config", lambda _config: source)
    monkeypatch.setattr(runner, "load_base", lambda spec: load_base(spec, device="cpu"))
    monkeypatch.setattr(run, "require_gpu", require_cpu)


def test_exact_frozen_configs_only_differ_by_declared_arm_and_weight():
    configs = [canonical(weight) for weight in (0, 1)]
    for weight, config in enumerate(configs):
        runner.validate_config(config)
        assert runner.read_source_config(config)["training"]["steps"] == 96
        assert config["gates"]["max_later_validation_forgetting"] == 0.05
        assert (
            config["target_confirmation"]["sha256"]
            == "4f298979c5f6e76ed4a1ca8d07d1880a20499bd5911d8991591ece3934e8ce2e"
        )
        assert file_hash(
            ROOT / f"configs/anchored/native_replay_anchor{weight}.json"
        ) == (
            "fe1ac9f0b341f22fd79725327b960dd0488fa5f1ba3744d1a45b484268bbb254"
            if weight == 0
            else "72613cbef590a1c948a46581c4011ccba052c414c130b3f69fb933d254e7f12a"
        )
    assert {
        key: value
        for key, value in configs[0].items()
        if key not in ("arm", "anchor_weight")
    } == {
        key: value
        for key, value in configs[1].items()
        if key not in ("arm", "anchor_weight")
    }
    for changed in ("anchors", "gates", "target_confirmation"):
        value = copy.deepcopy(configs[0])
        value[changed] = {}
        with pytest.raises(ValueError, match="FROZEN_PROTOCOL_CHANGED"):
            runner.validate_config(value)


def successful_timeline():
    result = {}
    for index, boundary in enumerate(run.BOUNDARIES):
        tasks = SEQUENCE_TASKS if index == 0 else SEQUENCE_TASKS[:index]
        value = 0.25 if index == 0 else 0.95
        result[boundary] = {
            split: {"metrics": {task: {"accuracy": value} for task in tasks}}
            for split in ("train", "validation")
        }
    return result


def test_acquisition_and_every_later_boundary_gates_never_change_budget():
    good = successful_timeline()
    assert runner.source_gates(good, runner.SOURCE_GATES)["passed"]
    bad_middle = copy.deepcopy(good)
    bad_middle["after_b"]["validation"]["metrics"]["sequence_a"]["accuracy"] = 0.75
    result = runner.source_gates(bad_middle, runner.SOURCE_GATES)
    assert not result["passed"]
    assert (
        result["tasks"]["sequence_a"]["later"]["after_b"]["forgetting_from_acquisition"]
        > 0.05
    )
    assert (
        result["tasks"]["sequence_a"]["later"]["after_c"]["forgetting_from_acquisition"]
        == 0
    )
    assert result["training_budget_action"] == "all288updates_already_completed"
    failed_acquisition = copy.deepcopy(good)
    failed_acquisition["after_a"]["train"]["metrics"]["sequence_a"]["accuracy"] = 0.89
    assert not runner.source_gates(failed_acquisition, runner.SOURCE_GATES)[
        "target_evaluation_eligible"
    ]
    failed_gain = copy.deepcopy(good)
    failed_gain["initial"]["validation"]["metrics"]["sequence_c"]["accuracy"] = 0.75
    assert not runner.source_gates(failed_gain, runner.SOURCE_GATES)["passed"]
    incomplete = copy.deepcopy(good)
    del incomplete["after_b"]
    with pytest.raises(ValueError, match="ALL_SOURCE_BOUNDARIES"):
        runner.source_gates(incomplete, runner.SOURCE_GATES)


def test_supervisor_bootstrap_is_preserved_and_existing_research_rejected(
    tmp_path, monkeypatch
):
    config, source, _ = cpu_fixture(tmp_path / "assets")
    use_cpu(source, monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    task = {
        "id": output.name,
        "config": config,
        "entrypoint": "train_anchored.py",
        "source_sha256": "0" * 64,
    }
    execution = {"task_id": task["id"], "task": task, "status": "running"}
    bootstrap = {
        "config.json": json.dumps(config).encode(),
        "task.json": json.dumps(task).encode(),
        "execution.json": json.dumps(execution).encode(),
        "run.log": b"bootstrap\n",
        "packages.txt": b"pinned packages\n",
    }
    for name, content in bootstrap.items():
        (output / name).write_bytes(content)
    runner.prepare(config, output)
    assert all(
        (output / name).read_bytes() == content for name, content in bootstrap.items()
    )
    runner.verify_prepared(config, output)
    with pytest.raises(ValueError, match="OUTPUT_CONTAINS_RESEARCH"):
        runner.prepare(config, output)


def test_full_fixed_288_training_then_real_fresh_process_reload_and_source_gates(
    tmp_path, monkeypatch
):
    config, source, source_path = cpu_fixture(tmp_path / "assets", weight=1)
    use_cpu(source, monkeypatch)
    output = tmp_path / "run"
    runner.prepare(config, output)
    forbidden_calls = []

    def forbidden_gate(*_args, **_kwargs):
        forbidden_calls.append(True)
        raise AssertionError("ACQUISITION_GATES_MUST_NOT_RUN_DURING_TRAINING")

    monkeypatch.setattr(runner, "source_gates", forbidden_gate)
    original_read_data = runner.read_data
    reads = []

    def restricted_data(directory, names):
        assert all(name.endswith(("_train", "_validation")) for name in names)
        reads.extend(names)
        return original_read_data(directory, names)

    monkeypatch.setattr(runner, "read_data", restricted_data)
    original_json_loads = json.loads

    def no_confirmation_rows(value, *args, **kwargs):
        raw = value.encode() if isinstance(value, str) else value
        assert hashlib.sha256(raw).hexdigest() != runner.CONFIRMATION["sha256"]
        parsed = original_json_loads(value, *args, **kwargs)
        return parsed

    monkeypatch.setattr(json, "loads", no_confirmation_rows)
    runner.train(config, output)
    assert not forbidden_calls
    training = original_json_loads((output / "training_receipt.json").read_text())
    assert (
        training["status"] == "completed"
        and training["budget"]["optimizer_updates"] == 288
    )
    assert training["actual_llm_training_forward_calls"] == 360
    assert training["actual_anchor_native_forward_calls"] == 1440
    assert training["native_generator_calls"]["reference"] == 48
    assert set(training["stages"]) == {"after_a", "after_b", "after_c"}
    assert training["base_sha256_before"] == training["base_sha256_after"]
    assert set(training["reference_files"]) == set(
        runner.PUBLISHED_TASKS + SEQUENCE_TASKS[:2]
    )
    for stage in training["stages"].values():
        assert stage["budget"]["optimizer_updates"] == 96
        assert stage["frozen_before"] == stage["frozen_after"]
        assert stage["reference_hashes_before"] == stage["reference_hashes_after"]
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(Path(__file__).resolve()),
        "--reload",
        str(output),
        "--source",
        str(source_path),
    ]
    process = subprocess.run(
        command,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = original_json_loads((output / "result.json").read_text())
    assert result["status"] == "completed" and result["mechanically_qualified"]
    assert result["source_evaluation_pid"] != training["pid"]
    assert (
        result["separate_process_reload"]
        and result["evaluation_optimizer_updates"] == 0
    )
    assert (
        not result["source_gates"]["passed"]
        and not result["target_evaluation_eligible"]
    )
    assert result["budget"]["optimizer_updates"] == 288
    evaluations = original_json_loads((output / "source_evaluation.json").read_text())
    assert all(row["reload"]["passed"] for row in evaluations["timeline"].values())
    assert all(
        row["reload"]["probe_logits"]["bitwise_equal"]
        for name, row in evaluations["timeline"].items()
        if name != "initial"
    )
    damaged = Path(training["reference_files"]["sequence_a"]["path"])
    states = load_file(damaged)
    states["vector"].add_(1)
    save_file(states, damaged)
    rerun = subprocess.run(
        command,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert rerun.returncode != 0 and "ANCHOR_SAVED_REFERENCE_CHANGED" in rerun.stderr


def reload_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.reload / "config.json").read_text())
    source = json.loads(args.source.read_text())
    with pytest.MonkeyPatch.context() as patch:
        use_cpu(source, patch)
        runner.verify_prepared(config, args.reload)
        runner.evaluate_sources(config, args.reload)


if __name__ == "__main__":
    reload_main()
