from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from portallib import PortalConfig, PortalModel
from test_contract import tiny_model
from test_sequence_learning import panel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    Gemma3Config,
    Gemma3TextConfig,
    Gemma4Config,
    Gemma4TextConfig,
    MistralConfig,
    SiglipVisionConfig,
)

import evaluate_targets as target
from compare import load_run
from data import SEQUENCE_TASKS, digest, write_json
from learner import extend_portal, save_native, tensor_hash
from metrics import sequence_metrics

ROOT = target.ROOT


def configuration(name="qwen4", fixture=303):
    directory = ROOT / "configs" / {303: "", 419: "confirmation419"}[fixture]
    return target.fixed_config(directory / f"sequence_target_{name}_s{fixture}01.json")


def snapshot(directory, spec):
    return {
        **spec,
        "local_path": str(directory),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": target.file_sha256(path),
                "git_blob_sha1": None,
            }
            for path in sorted(directory.iterdir())
            if path.is_file()
        ],
    }


def tiny_sources(directory, config):
    base, original = tiny_model()
    primary = target.read_json(ROOT / "configs/sequence_native_no_replay.json")
    spec = primary["models"]["qwen8"]["base"]
    published = PortalModel(
        replace(
            original.config,
            tasks=tuple(config["published_tasks"]),
            base_model_name_or_path=spec["repo_id"],
            base_model_revision=spec["revision"],
        ),
        torch.randn(14, original.config.d_z),
    )
    published.core.load_state_dict(original.core.state_dict())
    published.alignment.load_state_dict(original.alignment.state_dict())
    initial = extend_portal(published, SEQUENCE_TASKS)
    states = {"initial": initial}
    for index, condition in enumerate(target.CONDITIONS[2:], 1):
        learned = copy.deepcopy(initial)
        with torch.no_grad():
            for value in learned.core.parameters():
                value.add_(index * 0.1)
            learned.task_latents[-3:].add_(index * 0.2)
        states[condition] = learned.requires_grad_(False).eval()
    files = {}
    for condition, state in states.items():
        checkpoint = directory / condition
        save_native(state, checkpoint)
        files.update(
            {
                str(checkpoint / name): checksum
                for name, checksum in target.checkpoint_files(checkpoint).items()
            }
        )
    receipts = {
        condition: {
            "task_id": f"cpu-fixture-{condition}",
            "source_bundle_sha256": target.PRIMARY_BUNDLE,
            "runtime_sha256": config["runtime_sha256"],
            "initial": target.portal_identity(initial),
            "final": target.portal_identity(states[condition]),
            "files": files,
            "acquisition": {
                "sequence_learning_signal": False,
                "tasks": {
                    task: {"acquired": False, "maximum_forgetting": 0.0}
                    for task in SEQUENCE_TASKS
                },
            },
        }
        for condition in target.CONDITIONS[2:]
    }
    return base, states, receipts


def tiny_target(directory, config, tokenizer, initial):
    spec = copy.deepcopy(config["target"])
    name = spec["name"]
    common = {
        "vocab_size": len(tokenizer),
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "use_cache": False,
    }
    if name == "qwen4":
        model = tiny_model()[0].model
    elif name == "mistral7":
        model = AutoModelForCausalLM.from_config(MistralConfig(**common))
    elif name == "gemma3":
        model = AutoModelForMultimodalLM.from_config(
            Gemma3Config(
                text_config=Gemma3TextConfig(
                    **common,
                    query_pre_attn_scalar=8,
                    sliding_window=32,
                    layer_types=["sliding_attention", "full_attention"],
                ),
                vision_config=SiglipVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=8,
                    patch_size=4,
                ),
                mm_tokens_per_image=4,
            )
        )
    else:
        model = AutoModelForMultimodalLM.from_config(
            Gemma4Config(
                text_config=Gemma4TextConfig(
                    **common,
                    global_head_dim=16,
                    vocab_size_per_layer_input=len(tokenizer),
                    hidden_size_per_layer_input=8,
                    layer_types=["sliding_attention", "full_attention"],
                    sliding_window=32,
                )
            )
        )
    portal_config = PortalConfig.from_model(
        model,
        tasks=tuple(config["published_tasks"]),
        base_model_name_or_path=spec["base"]["repo_id"],
        base_model_revision=spec["base"]["revision"],
        layer_path=spec["layer_path"],
        allow_heterogeneous_targets=spec["heterogeneous"],
        **initial.config.architecture_kwargs(),
    )
    original = PortalModel(portal_config, initial.task_latents[:14].detach().clone())
    original.core.load_state_dict(initial.core.state_dict())
    original.requires_grad_(False).eval()
    spec["projection_targets"] = list(portal_config.to_dict()["projection_targets"])
    spec["pad_token"] = "[PAD]"
    base_directory = directory / "base" / spec["base"]["revision"]
    model.to(dtype=torch.bfloat16).save_pretrained(base_directory)
    tokenizer.save_pretrained(base_directory)
    portal_directory = directory / "portal" / spec["portal"]["revision"]
    save_native(original, portal_directory)
    spec["base"] = snapshot(base_directory, spec["base"])
    spec["portal"] = snapshot(portal_directory, spec["portal"])
    return spec


def bootstrap(directory, config):
    task = {"id": directory.name, "config": config, "entrypoint": "evaluate_targets.py"}
    write_json(directory / "config.json", config)
    write_json(directory / "task.json", task)
    write_json(
        directory / "execution.json",
        {"task_id": task["id"], "task": task, "status": "running"},
    )
    (directory / "packages.txt").write_bytes(b"torch==cpu-test\n")
    (directory / "run.log").write_bytes(b"supervisor bootstrap\n")
    archive = directory / "attempts" / "previous" / "receipt.json"
    write_json(archive, {"status": "running", "pid": None})
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("name", "fixture"),
    [*((name, 303) for name in target.TARGETS), ("qwen4", 419)],
)
def test_saved_bf16_target_fresh_process_four_fp32_panels_zero_updates(
    name, fixture, tmp_path, monkeypatch
):
    config = configuration(name, fixture)
    base, states, receipts = tiny_sources(tmp_path / "sources", config)
    config["target"] = tiny_target(
        tmp_path / "assets", config, base.tokenizer, states["initial"]
    )
    config["evaluation"]["max_prompt"] = 64
    output = tmp_path / "evaluation"
    output.mkdir()
    preserved = bootstrap(output, config)
    rows = target.fixture_test_rows(config)
    monkeypatch.setattr(target, "validated_sources", lambda _: (receipts, states, rows))
    prepared = target.prepare_target(config, output)
    assert prepared["target_optimizer_updates"] == 0
    initial_identity = prepared["checkpoints"]["initial"]["identity"]
    for condition in target.CONDITIONS[1:]:
        actual = prepared["checkpoints"][condition]["identity"]
        source = target.portal_identity(states[condition])
        assert actual["core_sha256"] == source["core_sha256"]
        assert actual["task_table_sha256"] == source["task_table_sha256"]
        assert actual["sequence_vectors_sha256"] == source["sequence_vectors_sha256"]
        assert (
            actual["published_vectors_sha256"]
            == initial_identity["published_vectors_sha256"]
        )
        assert (
            actual["alignment_sha256"]
            == prepared["target_original"]["alignment_sha256"]
        )
    assert (
        len(
            {
                row["identity"]["shared_sha256"]
                for row in prepared["checkpoints"].values()
            }
        )
        == 3
    )
    with pytest.raises(ValueError, match="FRESH_TARGET_EVALUATOR_PROCESS_REQUIRED"):
        target.evaluate_prepared(
            config, output, target.file_sha256(output / "prepared.json"), device="cpu"
        )
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(Path(__file__).resolve()),
        "--evaluate",
        str(output),
    ]
    with (output / "child.log").open("w") as stream:
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            },
        )
    report = target.read_json(output / "result.json")
    assert report["fixture_seed"] == fixture
    assert report["fixture_sha256"] == config["fixture"]["sha256"]
    assert report["kind"] == f"sequence{fixture}_single_target_evaluation"
    assert f"fixture{fixture}." in report["interpretation"]["claim_boundary"]
    assert report["evaluator_pid"] != os.getpid() == report["parent_pid"]
    assert report["panel_count"] == 4
    assert (
        report["prediction_count"]
        == sum(len(row["predictions"]) for row in report["panels"].values())
        == 768
    )
    assert set(report["panels"]) == set(target.CONDITIONS)
    assert report["target_optimizer_updates"] == report["calibration_examples"] == 0
    for condition, check in report["checks"].items():
        assert check["base_before_sha256"] == check["base_after_sha256"]
        assert check["frozen"]
        assert check["reload_from_disk"] is (condition != "raw")
        assert report["arithmetic"][condition]["logits_dtype"] == "torch.float32"
        assert not report["arithmetic"][condition]["autocast_enabled"]
        assert not report["arithmetic"][condition]["grad_enabled"]
    assert target.read_json(output / "zero-update-guard.json") == {
        "optimizer_constructions": 0,
        "backward_calls": 0,
        "passed": True,
    }
    assert all((output / path).read_bytes() == data for path, data in preserved.items())
    assert report["target_loader"]["layer_path"] == config["target"]["layer_path"]
    if name == "gemma4":
        assert report["target_loader"]["heterogeneous"]
        assert (
            len(
                {
                    row["out_features"]
                    for row in config["target"]["projection_targets"]
                    if row["module_name"] == "q"
                }
            )
            == 2
        )
    for arm in report["interpretation"]["arms"].values():
        assert not any(
            row["acquired_source_target_gain"] or row["retained_source_target_gain"]
            for row in arm["tasks"].values()
        )


def test_real_source_checkpoint_rejects_bad_hash_and_task_table(tmp_path):
    config = configuration()
    _, states, _ = tiny_sources(tmp_path, config)
    primary = target.read_json(ROOT / "configs/sequence_native_no_replay.json")
    checkpoint = tmp_path / "initial"
    expected = tensor_hash(states["initial"].state_dict())
    restored = target.load_source_checkpoint(checkpoint, expected, primary, config)
    assert target.portal_identity(restored) == target.portal_identity(states["initial"])
    with pytest.raises(ValueError, match="SOURCE_CHECKPOINT_HASH_MISMATCH"):
        target.load_source_checkpoint(checkpoint, "0" * 64, primary, config)
    changed = target.read_json(checkpoint / "config.json")
    changed["tasks"][-1] = "unregistered_task"
    write_json(checkpoint / "config.json", changed)
    with pytest.raises(ValueError, match="SOURCE_TASK_TABLE_MISMATCH"):
        target.load_source_checkpoint(checkpoint, expected, primary, config)


@pytest.mark.parametrize("fixture", (303, 419))
def test_acquisition_and_retention_condition_target_gain_independently(fixture):
    panels = {
        "raw": panel((16, 16, 16)),
        "initial": panel((20, 20, 20)),
        "native_no_replay": panel((28, 28, 28)),
        "native_replay": panel((28, 28, 28)),
    }
    sources = {
        condition: {
            "acquisition": {
                "sequence_learning_signal": False,
                "tasks": {
                    "sequence_a": {"acquired": True, "maximum_forgetting": 5 / 64},
                    "sequence_b": {"acquired": False, "maximum_forgetting": 0},
                    "sequence_c": {"acquired": True, "maximum_forgetting": 0.05},
                },
            }
        }
        for condition in target.CONDITIONS[2:]
    }
    result = target.interpretation(panels, sources, fixture * 100 + 1, fixture)
    assert f"fixture{fixture}." in result["claim_boundary"]
    for arm in result["arms"].values():
        a, b, c = (arm["tasks"][task] for task in SEQUENCE_TASKS)
        assert a["acquired_source_target_gain"] and not a["retained_source_target_gain"]
        assert (
            b["clears_target_gain_threshold"]
            and not b["acquired_source_target_gain"]
            and not b["retained_source_target_gain"]
        )
        assert c["acquired_source_target_gain"] and c["retained_source_target_gain"]
        assert a["gain_over_raw"] == 12 / 64
        assert a["gain_over_initial"] == a["gain_over_both_floors"] == 8 / 64
    panels["raw"] = panel((32, 32, 32))
    stronger_floor = target.interpretation(panels, sources, fixture * 100 + 1, fixture)
    assert not any(
        row["acquired_source_target_gain"] or row["retained_source_target_gain"]
        for arm in stronger_floor["arms"].values()
        for row in arm["tasks"].values()
    )


def test_eight_fixed_configs_and_actual_twenty_primary_files(tmp_path):
    manifest = target.read_json(
        ROOT / "outputs/sequence-fp32-integration/source-freeze.json"
    )
    assert set(target.CONFIG_DIGESTS) == {
        (name, fixture) for name in target.TARGETS for fixture in (303, 419)
    }
    identities = {303: set(), 419: set()}
    for name, fixture in target.CONFIG_DIGESTS:
        config = configuration(name, fixture)
        checked = target.verify_primary_files(config)
        assert len(checked) == 20
        assert checked == manifest["source_files"]
        rows = target.fixture_test_rows(config)
        assert len(rows) == 192
        identities[fixture].add(digest(target.row_identity(rows)))
        assert config["kind"] == f"sequence{fixture}_fixed_target_evaluation"
        assert config["target"] == configuration(name)["target"]
        assert config["evaluation"] == configuration(name)["evaluation"]
        changed = copy.deepcopy(config)
        changed["source_runs"]["native_replay"]["id"] = "latest419"
        path = tmp_path / f"{name}-{fixture}.json"
        write_json(path, changed)
        with pytest.raises(ValueError, match="UNDECLARED_TARGET_EVALUATION_CONFIG"):
            target.fixed_config(path)
        changed = copy.deepcopy(config)
        changed["fixture"] = configuration(name, {303: 419, 419: 303}[fixture])[
            "fixture"
        ]
        write_json(path, changed)
        with pytest.raises(ValueError, match="UNDECLARED_TARGET_EVALUATION_CONFIG"):
            target.fixed_config(path)
    assert len(identities[303]) == len(identities[419]) == 1
    assert identities[303].isdisjoint(identities[419])


def test_all_four_419_configs_accept_both_actual_source_receipts():
    root = ROOT / "outputs/sequence-fp32-confirmation419"
    checks = target.read_json(root / "checkpoint-byte-check.json")
    assert checks["all_matched"]
    for name in target.TARGETS:
        config = configuration(name, 419)
        config["source_runs_root"] = str(root / "runs")
        directories = target.source_readiness(config)
        assert tuple(directories) == target.CONDITIONS[2:]
    for condition, directory in directories.items():
        record = load_run(directory)
        source = sequence_metrics(
            record["evaluation"]["timeline"],
            record["evaluation"]["baselines"]["train"]["raw"],
            record["config"]["seed"],
        )
        assert source == record["result"]["source"]
        assert all(row["acquired"] for row in source["tasks"].values())
        assert source["sequence_learning_signal"] is False
        assert record["config"]["sequence"] == config["fixture"]
        for suffix, expected in (
            (
                "initial/native/checkpoint/model.safetensors",
                config["initial_checkpoint_sha256"],
            ),
            (
                "stages/after_c/checkpoint/model.safetensors",
                config["source_runs"][condition]["final_checkpoint_sha256"],
            ),
        ):
            checked = checks["checkpoint_checks"][f"{directory.name}/{suffix}"]
            assert checked["matched"]
            assert checked["tensor_hash"] == checked["receipt_tensor_hash"] == expected


def test_419_aggregate_requires_all_four_targets_from_one_fixed_fixture():
    reports = {}
    for fixture in (303, 419):
        group = []
        for name in target.TARGETS:
            config = configuration(name, fixture)
            rows = target.row_identity(target.fixture_test_rows(config))
            predictions = [
                {"id": key, **row, "correct": 0} for key, row in rows.items()
            ]
            group.append(
                {
                    "kind": f"sequence{fixture}_single_target_evaluation",
                    "target": name,
                    "fixture_seed": fixture,
                    "fixture_sha256": config["fixture"]["sha256"],
                    "config_sha256": digest(config),
                    "mechanically_qualified": True,
                    "panel_count": 4,
                    "prediction_count": 768,
                    "fresh_process": True,
                    "parent_pid": 1,
                    "evaluator_pid": 2,
                    "conditions": list(target.CONDITIONS),
                    "panels": {
                        condition: {"predictions": predictions}
                        for condition in target.CONDITIONS
                    },
                    "source_identity": config["source_runs"],
                    "target_optimizer_updates": 0,
                    "calibration_examples": 0,
                }
            )
        aggregate = target.aggregate_results(group)
        assert aggregate["fixture_seed"] == fixture
        assert aggregate["panel_count"] == 16
        assert aggregate["prediction_count"] == 3072
        assert aggregate["selected_target"] is None
        reports[fixture] = group
    with pytest.raises(ValueError, match="TARGET_RESULT_MATRIX_MISMATCH"):
        target.aggregate_results([*reports[419][:-1], reports[303][-1]])
    with pytest.raises(ValueError, match="EXACT_FOUR_TARGET_RESULTS_REQUIRED"):
        target.aggregate_results(reports[419][:-1])


def evaluate_child(output):
    torch.set_num_threads(1)
    counts = {"optimizer_constructions": 0, "backward_calls": 0}

    def forbid_optimizer(*args, **kwargs):
        counts["optimizer_constructions"] += 1
        raise AssertionError("TARGET_CONSTRUCTED_OPTIMIZER")

    def forbid_backward(*args, **kwargs):
        counts["backward_calls"] += 1
        raise AssertionError("TARGET_CALLED_BACKWARD")

    torch.optim.Optimizer.__init__ = forbid_optimizer
    torch.autograd.backward = forbid_backward
    config = target.read_json(output / "config.json")
    target.evaluate_prepared(
        config, output, target.file_sha256(output / "prepared.json"), device="cpu"
    )
    write_json(output / "zero-update-guard.json", {**counts, "passed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", type=Path, required=True)
    evaluate_child(parser.parse_args().evaluate)
