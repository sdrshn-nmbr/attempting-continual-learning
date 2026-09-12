import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import evaluate_repair as repair
import pytest
import torch
from data import write_json
from learner import (
    frozen_base_tensors,
    load_base,
    load_portal,
    save_native,
    tensor_hash,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from portallib import PortalModel
from test_sequence_targets import configuration, snapshot, tiny_sources, tiny_target

ROOT = repair.ROOT


def fresh_spec():
    path = ROOT / "data/sequence303_unused_inputs.json"
    return {"path": str(path), "sha256": repair.file_sha256(path)}


def child_evaluate(output):
    output = Path(output)
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer or backward in saved-checkpoint evaluation")

    torch.optim.AdamW = forbidden
    torch.Tensor.backward = forbidden
    config = json.loads((output / "config.json").read_text())
    repair.evaluate_prepared(
        config, output, repair.file_sha256(output / "prepared.json"), device="cpu"
    )


def run_child(output):
    with (output / "child.log").open("w") as stream:
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(Path(__file__).resolve()),
                "--child",
                str(output),
            ],
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
    result = json.loads((output / "result.json").read_text())
    assert result["parent_pid"] == os.getpid() != result["evaluator_pid"]
    assert result["optimizer_updates"] == 0
    assert all(row["base_and_portal_unchanged"] for row in result["checks"].values())
    return result


@pytest.mark.parametrize("name", ["qwen4", "mistral7"])
def test_full_native_target_graft_reload_and_behavior(name, tmp_path, monkeypatch):
    primary = configuration(name)
    base, states, receipts = tiny_sources(tmp_path / "source", primary)
    primary["target"] = tiny_target(
        tmp_path / "assets", primary, base.tokenizer, states["initial"]
    )
    original = load_portal(primary["target"]["portal"]).requires_grad_(False).eval()
    carriers = {}
    for label, increment in (("repair", 0.01), ("mismatch", 0.02)):
        carrier = copy.deepcopy(original)
        carrier.core.load_state_dict(states["native_replay"].core.state_dict())
        with torch.no_grad():
            for parameter in carrier.alignment.parameters():
                parameter.add_(increment)
        directory = tmp_path / "carriers" / label
        save_native(carrier, directory)
        carriers[label] = {
            "path": str(directory),
            "files": repair.checkpoint_files(directory),
        }
    audit_path = ROOT / "outputs/repair-alignment-cpu/independent-reload-audit.json"
    config = {
        "kind": "cpu_fixture_repair_target",
        "role": "target",
        "target": name,
        "seed": 90301,
        "fresh_inputs": fresh_spec(),
        "geometry_audit": {
            "path": str(audit_path),
            "sha256": repair.file_sha256(audit_path),
        },
        "carriers": carriers,
        "conditions": list(repair.TARGET_CONDITIONS),
        "evaluation": {"batch_size": 64, "max_prompt": 64},
        "protocol": {"test_fixture": True},
    }
    output = tmp_path / "evaluation"
    output.mkdir()
    write_json(output / "config.json", config)
    monkeypatch.setattr(repair, "validated_sources", lambda _: (receipts, states, []))
    prepared = repair.prepare_study(config, output, primary)
    for label in ("unchanged", "repair", "mismatch"):
        assert (
            prepared["cases"][label]["identity"]["shared_sha256"]
            == repair.portal_identity(states["native_replay"])["shared_sha256"]
        )
    with pytest.raises(ValueError, match="FRESH_PROCESS_REQUIRED"):
        repair.evaluate_prepared(
            config, output, repair.file_sha256(output / "prepared.json"), "cpu"
        )
    result = run_child(output)
    assert result["prediction_count"] == 5 * 864
    assert result["paired_rows"] == 864 and result["groups"] == 288
    assert set(result["panels"]) == set(repair.TARGET_CONDITIONS)
    assert all(
        row["forward_dtype"] == "torch.float32" for row in result["checks"].values()
    )
    mutated = PortalModel.from_pretrained(
        carriers["repair"]["path"],
        local_files_only=True,
        device="cpu",
        dtype=torch.float32,
    )
    with torch.no_grad():
        mutated.task_latents[0, 0].add_(1)
    unchanged = PortalModel.from_pretrained(
        carriers["mismatch"]["path"],
        local_files_only=True,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="CARRIER_CORE_VECTOR_OR_TARGET_MISMATCH"):
        repair.prepare_target_cases(
            states["initial"],
            states["native_replay"],
            original,
            {"repair": mutated, "mismatch": unchanged},
            tmp_path / "poison",
        )


def test_source_native_and_lora_use_actual_saved_base(tmp_path, monkeypatch):
    primary = configuration()
    base, states, receipts = tiny_sources(tmp_path / "source", primary)
    source_config = json.loads(
        (ROOT / "configs/sequence_native_replay.json").read_text()
    )
    source_spec = source_config["models"]["qwen8"]["base"]
    directory = tmp_path / "base" / source_spec["revision"]
    base.model.to(dtype=torch.bfloat16).save_pretrained(directory)
    base.tokenizer.save_pretrained(directory)
    source_spec = snapshot(directory, source_spec)
    base = load_base(source_spec, "cpu")
    base_sha = tensor_hash(frozen_base_tensors(base.model))
    model = get_peft_model(
        base.model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.add_(0.05)
    checkpoint = tmp_path / "lora"
    model.save_pretrained(checkpoint, save_embedding_layers=False)
    spec = {
        "path": str(checkpoint),
        "files": {
            name: repair.file_sha256(checkpoint / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
        "tensor_sha256": tensor_hash(
            get_peft_model_state_dict(model, save_embedding_layers=False)
        ),
    }
    config = {
        "kind": "cpu_fixture_repair_source",
        "role": "source",
        "target": "qwen8",
        "seed": 90301,
        "source_base": source_spec,
        "source_base_tensor_sha256": base_sha,
        "fresh_inputs": fresh_spec(),
        "lora": spec,
        "conditions": list(repair.SOURCE_CONDITIONS),
        "evaluation": {"batch_size": 64, "max_prompt": 64},
        "protocol": {"test_fixture": True},
    }
    output = tmp_path / "evaluation"
    output.mkdir()
    write_json(output / "config.json", config)
    monkeypatch.setattr(repair, "validated_sources", lambda _: (receipts, states, []))
    repair.prepare_study(config, output, primary)
    result = run_child(output)
    assert result["prediction_count"] == 4 * 864
    assert (
        result["checks"]["lora_replay"]["lora_tensor_sha256"] == spec["tensor_sha256"]
    )
    assert {row["base_sha256"] for row in result["checks"].values()} == {base_sha}


def test_canonical_worker_arguments_reject_unfrozen_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "evaluate_repair.py"),
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and "REPAIR_EVAL_UNDECLARED_CONFIG" in result.stderr


def test_actual_production_configs_have_complete_runtime_manifests():
    for path in (ROOT / "configs/repair_eval").glob("*.json"):
        config = json.loads(path.read_text())
        repair.validate_study_config(config)
        assert repair.file_sha256(path) == repair.CONFIG_SHA256S[config["target"]]
        if config["role"] == "source":
            del config["source_base"]["files"]
            with pytest.raises(ValueError, match="SOURCE_FILE_MANIFEST_REQUIRED"):
                repair.validate_study_config(config)


if __name__ == "__main__":
    child_evaluate(sys.argv[2])
