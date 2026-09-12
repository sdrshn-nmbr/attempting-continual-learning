from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from portallib import PortalBase, PortalConfig, PortalModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from data import SEQUENCE_TASKS, prepare_data, read_data, write_json
from learner import extend_portal, parameter_plan, state_hash, tensor_hash, transplant
from run import prepare, stage_schedule, validate_config

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(seed=71, dtype=torch.float32):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        *map(str, range(8)),
        "Apply",
        "the",
        *SEQUENCE_TASKS,
        "code.",
        "Input:",
        "Output:",
    ]
    backend = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(words),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
            attention_dropout=0.1,
        )
    ).to(dtype=dtype)
    base = PortalBase("test/qwen", model, tokenizer, revision="0" * 40)
    base.freeze(gradient_checkpointing=False)
    config = PortalConfig.from_model(
        model,
        tasks=("rte",),
        base_model_name_or_path=base.model_id,
        base_model_revision=base.revision,
        rank=2,
        alpha=4,
        d_z=8,
        d_layer=4,
        hidden=16,
        d_core=20,
    )
    portal = PortalModel(config, torch.randn(1, 8))
    with torch.no_grad():
        for parameter in portal.core.parameters():
            parameter.normal_(mean=0, std=0.05)
    return base, portal.requires_grad_(False)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/sequence_native_no_replay.json").read_text())


@pytest.fixture
def rows(config, tmp_path):
    prepare_data(config, tmp_path)
    return read_data(
        tmp_path,
        tuple(
            f"{task}_{split}"
            for task in SEQUENCE_TASKS
            for split in ("train", "validation", "test")
        ),
    )


@pytest.fixture
def tiny():
    return tiny_model()


def test_all_frozen_training_configs_validate_and_reject_drift():
    configs = sorted(
        path
        for path in (ROOT / "configs").rglob("*.json")
        if path.name.startswith(("sequence_native_", "sequence_lora_"))
    )
    assert len(configs) == 8
    for path in configs:
        config = json.loads(path.read_text())
        validate_config(config)
        moved = copy.deepcopy(config)
        moved["models"]["qwen4"]["base"]["revision"] = "main"
        with pytest.raises(ValueError, match="UNPINNED_MODEL"):
            validate_config(moved)
        changed = copy.deepcopy(config)
        changed["training"]["steps"] = 95
        with pytest.raises(ValueError, match="UNDECLARED_SEQUENCE_TRAINING_RECIPE"):
            validate_config(changed)


def test_supervisor_bootstrap_is_preserved_by_real_cli(config, tmp_path):
    task = {
        "id": tmp_path.name,
        "config": config,
        "source_sha256": "0" * 64,
        "entrypoint": "run.py",
    }
    execution = {
        "task_id": task["id"],
        "task": task,
        "status": "running",
        "attempt_id": "abc",
        "pid": 123,
    }
    bootstrap = {
        "config.json": (json.dumps(config, indent=2) + "\n").encode(),
        "task.json": (json.dumps(task, indent=2) + "\n").encode(),
        "execution.json": (json.dumps(execution, indent=2) + "\n").encode(),
        "packages.txt": b"torch==vendor-rocm\n",
        "run.log": b"wrapper initialized\n",
    }
    for name, value in bootstrap.items():
        (tmp_path / name).write_bytes(value)
    archive = tmp_path / "attempts" / "abc" / "prior-receipt.json"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'{"status":"running","pid":null}\n')
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "run.py"),
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(tmp_path),
            "--phase",
            "prepare",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert all(
        (tmp_path / name).read_bytes() == value for name, value in bootstrap.items()
    )
    assert archive.read_bytes() == b'{"status":"running","pid":null}\n'
    assert (tmp_path / "data_manifest.json").exists()
    with pytest.raises(ValueError, match="OUTPUT_CONTAINS_RESEARCH"):
        prepare(config, tmp_path)


@pytest.mark.parametrize("name", ["data", "result.json", "checkpoint"])
def test_prepare_rejects_previous_research(config, tmp_path, name):
    (tmp_path / name).mkdir() if "." not in name else (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="OUTPUT_CONTAINS_RESEARCH"):
        prepare(config, tmp_path)
    assert not (tmp_path / "data_manifest.json").exists()


def test_prepare_rejects_bootstrap_configuration_mismatch(config, tmp_path):
    task = {"id": tmp_path.name, "config": {**config, "seed": 99}}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "task.json", task)
    write_json(
        tmp_path / "execution.json",
        {"task_id": tmp_path.name, "task": task, "status": "running"},
    )
    (tmp_path / "run.log").touch()
    (tmp_path / "packages.txt").touch()
    with pytest.raises(ValueError, match="BOOTSTRAP_CONFIG_MISMATCH"):
        prepare(config, tmp_path)


def test_schedule_matches_all_four_arms_and_never_replays_future(config, rows):
    schedules = []
    for method in ("native", "lora"):
        for weight in (0.0, 0.25):
            recipe = {
                **config["training"],
                "train_core": method == "native",
                "replay_weight": weight,
            }
            schedule = []
            current_count = reference_count = 0
            for stage in range(3):
                permitted = {
                    key: value
                    for key, value in rows.items()
                    if key in {f"{task}_train" for task in SEQUENCE_TASKS[: stage + 1]}
                }
                for current, reference, identity in stage_schedule(
                    permitted, stage, config["seed"], recipe
                ):
                    assert len(current) == 4
                    assert {row.task for row in current} == {SEQUENCE_TASKS[stage]}
                    assert {row.task for row in reference} <= set(
                        SEQUENCE_TASKS[:stage]
                    )
                    if stage == 2 and reference:
                        assert [row.task for row in reference] == ["sequence_a"] * 2 + [
                            "sequence_b"
                        ] * 2
                    current_count += len(current)
                    reference_count += len(reference)
                    schedule.append(identity)
            assert (current_count, reference_count, len(schedule)) == (1152, 192, 288)
            schedules.append(schedule)
    assert all(schedule == schedules[0] for schedule in schedules)


def test_transplant_preserves_full_shared_state_and_original_target_alignment(tiny):
    _, initial = tiny
    learned = extend_portal(initial, SEQUENCE_TASKS)
    with torch.no_grad():
        learned.core.l1.weight.add_(0.03)
        learned.task_latents[-3:].add_(0.2)
    target = copy.deepcopy(initial)
    with torch.no_grad():
        target.alignment.layer_embeddings.weight.add_(0.05)
    result = transplant(learned, target)
    assert tensor_hash(result.core.state_dict()) == tensor_hash(
        learned.core.state_dict()
    )
    assert torch.equal(result.task_latents, learned.task_latents)
    assert tensor_hash(result.alignment.state_dict()) == tensor_hash(
        target.alignment.state_dict()
    )
    assert all(not value.requires_grad for value in result.parameters())


def test_parameter_plan_uses_live_capacity_and_uniform_projection_ranks(tiny):
    _, portal = tiny
    plan = parameter_plan(portal)
    assert plan["native_active_parameters"] == 3496
    assert plan["lora_parameters"] == 3488
    assert plan["projection_ranks"] == {"q": 16, "v": 15}
    assert plan["relative_difference"] <= 0.005
    assert all(
        plan["alpha_pattern"][name] / rank == 2
        for name, rank in plan["rank_pattern"].items()
    )


def test_hash_covers_tensor_dtype_shape_and_nested_optimizer_state():
    value = torch.tensor([1.0, 2.0])
    assert tensor_hash({"x": value}) != tensor_hash({"x": value.reshape(1, 2)})
    assert tensor_hash({"x": value}) != tensor_hash({"x": value.double()})
    assert state_hash(
        {"state": {1: {"exp_avg": value}}, "param_groups": [1]}
    ) != state_hash({"state": {1: {"exp_avg": value + 1}}, "param_groups": [1]})
