from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from portallib import PortalBase, PortalConfig, PortalModel, collate_gold_batch
from portallib.evaluation import PortalInjector
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibrate_target as calibration
from data import SEQUENCE_TASKS, digest, write_json
from learner import (
    frozen_base_tensors,
    save_native,
    shared_hash,
    supervised_loss,
    tensor_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/target_calibration/qwen4_sequence303.json"
CPU_PYTHON = Path(
    "/Users/sudarshan/Documents/Codex/2026-09-05/i-want-you-to-run-extensive/experiments/portability/.venv/bin/python"
)


def tiny_models(layers=2):
    torch.manual_seed(20260911)
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
        WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(words),
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=layers,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=128,
                pad_token_id=0,
                eos_token_id=1,
                use_cache=False,
                attention_dropout=0.2,
            )
        )
        .float()
        .eval()
    )
    base = PortalBase("test/qwen", model, tokenizer, revision="1" * 40)
    base.freeze(gradient_checkpointing=False)
    config = PortalConfig.from_model(
        model,
        tasks=("rte",),
        base_model_name_or_path=base.model_id,
        base_model_revision=base.revision,
        rank=8,
        alpha=16,
        d_z=8,
        d_layer=4,
        hidden=16,
        d_core=12,
    )
    target = PortalModel(config, torch.randn(1, 8))
    with torch.no_grad():
        for parameter in target.core.parameters():
            parameter.normal_(0, 0.15)
    initial = PortalModel(
        replace(config, tasks=("rte", *SEQUENCE_TASKS)),
        target.task_latents.detach().repeat(4, 1),
    )
    initial.core.load_state_dict(target.core.state_dict())
    initial.alignment.load_state_dict(target.alignment.state_dict())
    learned = copy.deepcopy(initial)
    with torch.no_grad():
        for parameter in learned.core.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
        learned.task_latents[1:].add_(torch.randn_like(learned.task_latents[1:]) * 0.3)
    return (
        base,
        initial.requires_grad_(False),
        learned.requires_grad_(False),
        target.requires_grad_(False),
    )


def actual_rows():
    return calibration.calibration_rows(json.loads(CONFIG.read_text()))


def fresh_candidate(initial=False, layers=2):
    base, old, learned, target = tiny_models(layers)
    adapter = calibration.make_target(
        old if initial else learned, target, calibration.fresh_alignment(target)
    )
    return base, adapter


def fixture_config(directory: Path):
    directory.mkdir(parents=True)
    config = json.loads(CONFIG.read_text())
    base, initial, learned, target = tiny_models()
    for key, adapter in (
        ("source_initial", initial),
        ("source_learned", learned),
        ("target_portal", target),
    ):
        path = directory / key
        save_native(adapter, path)
        config["inputs"][key] = calibration.pin_bundle(path)
    base_path = directory / "base" / base.revision
    base.model.save_pretrained(base_path)
    base.tokenizer.save_pretrained(base_path)
    config["inputs"]["target_base"] = calibration.pin_bundle(base_path)
    config["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "local_path": str(base_path),
        "cache_dir": str(directory / "cache"),
    }
    original = json.loads(
        calibration.input_path(config["inputs"]["fixture"]["path"]).read_text()
    )
    for task in SEQUENCE_TASKS:
        for split in ("validation", "test"):
            original["splits"][f"{task}_{split}"] = original["splits"][
                f"{task}_{split}"
            ][:1]
    fresh = json.loads(
        calibration.input_path(config["inputs"]["fresh_fixture"]["path"]).read_text()
    )
    fresh["rows"] = [
        row
        for task in SEQUENCE_TASKS
        for row in [r for r in fresh["rows"] if r["task"] == task][:2]
    ]
    for key, value in (("fixture", original), ("fresh_fixture", fresh)):
        path = directory / f"{key}.json"
        write_json(path, value)
        config["inputs"][key] = {"path": str(path), **calibration.file_pin(path)}
    config.update(
        role="cpu_contract_test",
        expected_source_hooks=4,
        expected_target_hooks=4,
        runtime={
            "device": "cpu",
            "dtype": "float32",
            "autocast": False,
            "cpu_threads": 1,
        },
        target_hooks=calibration.hook_receipt(base, target.config, 4),
    )
    config["source_provenance"]["initial_tensor_sha256"] = tensor_hash(
        initial.state_dict()
    )
    config["source_provenance"]["learned_tensor_sha256"] = tensor_hash(
        learned.state_dict()
    )
    config["claim_boundary"] = [
        "Tiny random Qwen3 CPU correctness test; no Qwen3-4B behavioral result."
    ]
    calibration.validate_config(config)
    return config


def run_cli(config: dict, config_path: Path, output: Path):
    write_json(config_path, config)
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        str(CPU_PYTHON),
        "python",
        str(ROOT / "calibrate_target.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    write_json(
        config_path.with_suffix(".execution.json"),
        {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "result.json").read_text()), json.loads(
        (output / "training_receipt.json").read_text()
    )


def test_frozen_recipe_48_train_rows_20_updates_five_balanced_epochs():
    config = json.loads(CONFIG.read_text())
    calibration.validate_config(config)
    rows = calibration.calibration_rows(config)
    schedule = calibration.balanced_schedule(rows)
    assert len(schedule) == 20
    assert (
        config["protocol"]["alignment_seed"]
        != config["protocol"]["source_training_seed"]
    )
    for epoch in range(1, 6):
        for task in SEQUENCE_TASKS:
            ids = [
                row_id
                for step in schedule
                if step["epoch"] == epoch
                for row_id in step["tasks"][task]
            ]
            assert len(ids) == len(set(ids)) == 16
            assert set(ids) == set(config["calibration_ids"][task])
    for field, changed in (
        ("epochs", 4),
        ("examples_per_task", 32),
        ("selection", "best_heldout"),
    ):
        wrong = copy.deepcopy(config)
        wrong["protocol"]["training"][field] = changed
        wrong["protocol_sha256"] = digest(wrong["protocol"])
        with pytest.raises(ValueError, match="FROZEN_PROTOCOL"):
            calibration.validate_config(wrong)
    wrong = copy.deepcopy(config)
    wrong["protocol"]["gates"]["minimum_lora_gain_over_raw"] = 0
    with pytest.raises(ValueError, match="FROZEN_PROTOCOL"):
        calibration.validate_config(wrong)


def test_headroom_cannot_silently_become_impossible_failure():
    feasible = calibration.headroom_result(19 / 32, 19 / 32, 0.75, 0.8)
    assert feasible["source_benefit_gate_feasible"]
    assert feasible["source_benefit_gate"]
    for values in ((0.95, 0.97, 0.99, 0.97), (0.6, 0.7, 0.9, 0.62)):
        result = calibration.headroom_result(*values)
        assert result["status"] == "inconclusive_insufficient_headroom"
        assert result["source_benefit_gate"] is None
        assert result["recovered_lift"] is None


@pytest.mark.parametrize("kind", ["portal", "lora"])
def test_tiny_qwen_injected_loss_and_gradients_match_dense_gold_only_control(kind):
    base, adapter = fresh_candidate()
    if kind == "portal":
        adapter.alignment.requires_grad_(True)
        with torch.no_grad():
            for parameter in adapter.alignment.output.values():
                parameter.normal_(0, 0.03)
        parameters = list(adapter.alignment.parameters())
    else:
        adapter = calibration.ZeroLora(adapter.config)
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if name.endswith("_b"):
                    parameter.normal_(0, 0.03)
        parameters = list(adapter.parameters())
    rows = actual_rows()[SEQUENCE_TASKS[0]][:4]
    base_hash = tensor_hash(frozen_base_tensors(base.model))
    factors = calibration.factors_for(adapter, rows[0].task)
    with (
        PortalInjector(base.model, adapter.config) as injector,
        injector.activate(factors),
    ):
        loss = supervised_loss(base, rows, 768).loss
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
    merged = dict(base.model.named_parameters())
    for target, path in adapter.config.resolved_targets():
        a, b = factors[target.key]
        merged[f"{path}.weight"] = (
            merged[f"{path}.weight"] + adapter.config.scaling * b @ a
        )
    ids, mask, labels = collate_gold_batch(
        base.tokenizer, [r.choice() for r in rows], max_prompt=768, device=base.device
    )
    dense = torch.func.functional_call(
        base.model,
        merged,
        (),
        {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "use_cache": False,
        },
    )
    assert torch.allclose(loss, dense.loss, atol=5e-7, rtol=1e-6)
    dense_gradients = torch.autograd.grad(dense.loss, parameters)
    for generated, dense_gradient in zip(gradients, dense_gradients, strict=True):
        assert torch.isfinite(generated).all()
        assert torch.allclose(generated, dense_gradient, atol=2e-7, rtol=2e-5)
    manual = F.cross_entropy(
        dense.logits[:, :-1].reshape(-1, dense.logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.equal(manual, dense.loss)
    assert (labels == -100).any() and (labels != -100).any()
    assert tensor_hash(frozen_base_tensors(base.model)) == base_hash
    assert all(p.grad is None for p in base.model.parameters())


def test_generate_is_inference_only_but_training_forward_retains_alignment_gradient():
    base, adapter = fresh_candidate()
    adapter.alignment.requires_grad_(True)
    task = SEQUENCE_TASKS[0]
    assert not any(
        t.requires_grad for factors in adapter.generate(task).values() for t in factors
    )
    assert all(
        t.requires_grad
        for factors in calibration.factors_for(adapter, task).values()
        for t in factors
    )
    with (
        PortalInjector(base.model, adapter.config) as injector,
        injector.activate(calibration.factors_for(adapter, task)),
    ):
        supervised_loss(base, actual_rows()[task][:4], 768).loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.alignment.parameters()
    )
    assert all(p.grad is None for p in adapter.core.parameters())
    assert adapter.task_latents.grad is None


@pytest.mark.parametrize("kind", ["portal", "lora"])
def test_balanced_calibration_updates_every_alignment_and_lowers_training_loss_fp32(
    kind,
):
    base, adapter = fresh_candidate()
    if kind == "lora":
        adapter = calibration.ZeroLora(adapter.config)
    rows = actual_rows()
    evaluation_rows = [r for task in SEQUENCE_TASKS for r in rows[task]]
    before = calibration.evaluate_adapter(base, adapter, evaluation_rows)
    observed = []
    handle = base.model.register_forward_pre_hook(
        lambda module, args: observed.append(torch.is_autocast_enabled("cpu"))
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        receipt = calibration.fit(
            base, adapter, rows, calibration.balanced_schedule(rows)
        )
    handle.remove()
    after = calibration.evaluate_adapter(base, adapter, evaluation_rows)
    assert not any(observed)
    assert receipt["optimizer_updates"] == 20
    assert receipt["task_microbatches"] == 60
    assert receipt["frozen_before"] == receipt["frozen_after"]
    assert receipt["all_trainables_received_finite_nonzero_gradient"]
    assert sum(m["gold_nll"] for m in after["metrics"].values()) < sum(
        m["gold_nll"] for m in before["metrics"].values()
    )
    if kind == "portal":
        assert "layer_embeddings.weight" in receipt["trainable_names"]
        assert not any(n.startswith("core.") for n in receipt["trainable_names"])


def test_all_72_qwen_hooks_receive_declared_shapes_and_dimension_mismatch_fails():
    base, adapter = fresh_candidate(layers=36)
    rows = calibration.hook_receipt(base, adapter.config, 72)
    assert len(rows) == 72
    seen = []
    handles = [
        base.model.get_submodule(row["path"]).register_forward_hook(
            lambda module, inputs, output: seen.append(
                (inputs[0].shape[-1], output.shape[-1])
            )
        )
        for row in rows
    ]
    with (
        PortalInjector(base.model, adapter.config) as injector,
        injector.activate(calibration.factors_for(adapter, SEQUENCE_TASKS[0])),
    ):
        supervised_loss(base, actual_rows()[SEQUENCE_TASKS[0]][:1], 768)
    for handle in handles:
        handle.remove()
    assert seen == [(r["A"][1], r["B"][0]) for r in rows]
    wrong = replace(
        adapter.config,
        projection_targets=tuple(
            replace(target, out_features=99) if target.module_name == "q" else target
            for target in adapter.config.projection_targets
        ),
    )
    with pytest.raises(ValueError):
        calibration.hook_receipt(base, wrong, 72)
    with pytest.raises(ValueError, match="HOOK_COUNT"):
        calibration.hook_receipt(base, adapter.config, 70)


def test_full_base_hash_catches_mutation_outside_old_sparse_samples():
    base, adapter = fresh_candidate()
    rows = actual_rows()
    changed = False

    def corrupt(module, inputs):
        nonlocal changed
        if not changed:
            with torch.no_grad():
                base.model.lm_head.weight[-1, -1].add_(1)
            changed = True

    handle = base.model.register_forward_pre_hook(corrupt)
    with pytest.raises(ValueError, match="FROZEN_FULL_TENSOR_STATE_CHANGED"):
        calibration.fit(base, adapter, rows, calibration.balanced_schedule(rows))
    handle.remove()


def test_download_metadata_is_ignored_but_extra_model_inputs_are_rejected(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text("{}")
    spec = calibration.pin_bundle(root)
    metadata = root / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("download bookkeeping")
    (root / "README.md").write_text("model card")
    assert set(calibration.verify_bundle(spec)) == {"config.json"}
    (root / "chat_template.jinja").write_text("unsealed prompt behavior")
    with pytest.raises(ValueError, match="FILE_SET_MISMATCH"):
        calibration.verify_bundle(spec)


def test_artifacts_verified_before_any_model_load(tmp_path, monkeypatch):
    config = fixture_config(tmp_path / "assets")
    path = Path(config["inputs"]["source_learned"]["path"]) / "config.json"
    path.write_text(path.read_text() + " ")

    def forbidden(*args, **kwargs):
        pytest.fail("model loading happened before all input verification")

    monkeypatch.setattr(calibration, "load_native_bundle", forbidden)
    monkeypatch.setattr(calibration, "load_base", forbidden)
    with pytest.raises(ValueError, match="FILE_SIZE_MISMATCH"):
        calibration.run_experiment(config, tmp_path / "run")


def test_actual_cli_final_only_fresh_pid_reload_and_poisoned_heldout_do_not_change_weights(
    tmp_path,
):
    config = fixture_config(tmp_path / "assets")
    first, first_receipt = run_cli(
        config, tmp_path / "clean-config.json", tmp_path / "clean"
    )
    assert first["pid"] != first["training_pid"]
    assert first["new_pid_reload"] and first["status"] == "completed"
    for name, receipt in first_receipt["training"].items():
        assert receipt["optimizer_updates"] == 20
        assert receipt["frozen_before"]["base"] == first_receipt["base_sha256"]
    assert (
        first_receipt["training"]["learned_calibrated"]["initial_trainable_sha256"]
        == first_receipt["training"]["initial_calibrated"]["initial_trainable_sha256"]
    )
    assert (
        first_receipt["checkpoints"]["learned_calibrated"]["shared_sha256"]
        == first_receipt["source_shared_sha256"]["learned"]
    )
    assert (
        first_receipt["checkpoints"]["initial_calibrated"]["shared_sha256"]
        == first_receipt["source_shared_sha256"]["initial"]
    )
    for panel in ("old_validation_test", "fresh288"):
        for floor in ("fresh_learned", "fresh_initial"):
            assert first["results"][floor][panel] == first["results"]["raw"][panel]
    events = [
        json.loads(line)
        for line in (tmp_path / "clean/events.jsonl").read_text().splitlines()
    ]
    checkpoint_indices = [
        i for i, row in enumerate(events) if row["event"] == "checkpoint_saved"
    ]
    eval_indices = [
        i for i, row in enumerate(events) if row["event"] == "final_evaluation"
    ]
    assert max(checkpoint_indices) < min(eval_indices)
    poisoned = copy.deepcopy(config)
    for name in ("fixture", "fresh_fixture"):
        source = Path(config["inputs"][name]["path"])
        value = json.loads(source.read_text())
        heldout = (
            value["rows"]
            if name == "fresh_fixture"
            else [
                r
                for split, rows in value["splits"].items()
                if not split.endswith("_train")
                for r in rows
            ]
        )
        for row in heldout:
            row["gold_idx"] = (row["gold_idx"] + 1) % len(row["choices"])
            row["choices"] = [" 7 7 7", " 6 6 6", " 5 5 5", " 4 4 4"]
        path = tmp_path / f"poisoned-{name}.json"
        write_json(path, value)
        poisoned["inputs"][name] = {"path": str(path), **calibration.file_pin(path)}
    second, second_receipt = run_cli(
        poisoned, tmp_path / "poison-config.json", tmp_path / "poisoned"
    )
    for condition in calibration.TRAINED:
        assert (
            first_receipt["checkpoints"][condition]["tensor_sha256"]
            == second_receipt["checkpoints"][condition]["tensor_sha256"]
        )
        assert (
            first_receipt["training"][condition]
            == second_receipt["training"][condition]
        )
    assert first["results"]["raw"] != second["results"]["raw"]
    with pytest.raises(FileExistsError):
        calibration.run_experiment(config, tmp_path / "clean")


def test_actual_pinned_303_source_states_fresh_alignment_and_all_72_target_shapes(
    tmp_path,
):
    config = json.loads(CONFIG.read_text())
    geometry_config = json.loads(
        (ROOT / "configs/representability/source_rank8_303.json").read_text()
    )
    local_inputs = Path(geometry_config["native_initial"]["path"]).parent
    for name, directory in (
        ("source_initial", "source_initial"),
        ("source_learned", "source_learned"),
        ("target_portal", "qwen4"),
    ):
        config["inputs"][name]["path"] = str(local_inputs / directory)
    initial = calibration.load_native_bundle(config["inputs"]["source_initial"])
    learned = calibration.load_native_bundle(config["inputs"]["source_learned"])
    target = calibration.load_native_bundle(config["inputs"]["target_portal"])
    assert (
        tensor_hash(initial.state_dict())
        == config["source_provenance"]["initial_tensor_sha256"]
    )
    assert (
        tensor_hash(learned.state_dict())
        == config["source_provenance"]["learned_tensor_sha256"]
    )
    fresh = calibration.fresh_alignment(target)
    hashes, source = {}, {}
    for name, shared in (("initial", initial), ("learned", learned)):
        candidate = calibration.make_target(shared, target, fresh)
        assert shared_hash(candidate) == shared_hash(shared)
        hashes[name] = tensor_hash(candidate.alignment.state_dict())
        source[name] = shared_hash(shared)
        assert len(list(candidate.config.resolved_targets())) == 72
        with torch.no_grad():
            for task in SEQUENCE_TASKS:
                generated = calibration.factors_for(candidate, task)
                assert len(generated) == 72
                for spec in candidate.config.projection_targets:
                    a, b = generated[spec.key]
                    assert a.shape == (8, spec.in_features)
                    assert b.shape == (spec.out_features, 8)
                    assert torch.count_nonzero(b) == 0
        del candidate
    assert hashes["initial"] == hashes["learned"]
    assert source["initial"] != source["learned"]
    write_json(
        tmp_path / "actual_source_qualification.json",
        {
            "status": "source_cpu_qualified_not_target_trained",
            "source_shared_sha256": source,
            "paired_fresh_alignment_sha256": hashes,
            "target_hooks": config["target_hooks"],
            "source_hooks": len(initial.config.projection_targets),
            "target_hooks_count": 72,
            "base_4b_loaded": False,
            "target_optimizer_updates": 0,
        },
    )
