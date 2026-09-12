import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from test_contract import tiny_model
from test_sequence_targets import snapshot

import evaluate_rank as rank
from data import write_json
from fresh_inputs import load_fresh_rows
from learner import evaluate, frozen_base_tensors, load_base, tensor_hash

ROOT = Path(rank.__file__).parent


def fixture_config(directory):
    base, _ = tiny_model()
    source = json.loads((ROOT / "configs/repair_eval/source_qwen8.json").read_text())
    spec = source["source_base"]
    base_path = directory / "base" / spec["revision"]
    base.model.save_pretrained(base_path)
    base.tokenizer.save_pretrained(base_path)
    spec = snapshot(base_path, spec)
    base = load_base(spec, "cpu")
    adapters = {}
    for index, name in enumerate(rank.CONDITIONS):
        size = 8 if name.startswith("rank8") else 10
        model = get_peft_model(
            copy.deepcopy(base.model),
            LoraConfig(
                r=size,
                lora_alpha=2 * size,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        if name.endswith("final"):
            with torch.no_grad():
                for key, value in model.named_parameters():
                    if "lora_B" in key:
                        value.normal_(std=0.1 * (index + 1))
        path = directory / "adapters" / name
        model.save_pretrained(path, safe_serialization=True)
        adapters[name] = {
            "path": str(path),
            "files": {
                filename: rank.file_sha256(path / filename)
                for filename in ("adapter_config.json", "adapter_model.safetensors")
            },
            "tensor_sha256": tensor_hash(
                get_peft_model_state_dict(model, save_embedding_layers=False)
            ),
        }
    proof = directory / "projection.json"
    write_json(proof, {"test_projection_fixture": True})
    return {
        "conditions": list(rank.CONDITIONS),
        "criteria": rank.CRITERIA,
        "evaluation": {
            "batch_size": 8,
            "max_prompt": 768,
            "dtype": "float32",
            "autocast": False,
        },
        "source_base": spec,
        "source_base_tensor_sha256": tensor_hash(frozen_base_tensors(base.model)),
        "adapters": adapters,
        "fresh_inputs": {
            "path": str(ROOT / "data/sequence303_unused_inputs.json"),
            "sha256": rank.file_sha256(ROOT / "data/sequence303_unused_inputs.json"),
        },
        "projection_proof": {"path": str(proof), "sha256": rank.file_sha256(proof)},
        "seed": 90310,
        "claim_boundary": "CPU test fixture; no experimental behavior claim.",
    }


def child(output):
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer or backward is allowed")

    torch.optim.AdamW = forbidden
    torch.Tensor.backward = forbidden
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    rank.evaluate_prepared(
        config, output, rank.file_sha256(output / "prepared.json"), "cpu"
    )


def test_four_actual_peft_adapters_fresh_reload_and_single_adapter_equivalence(
    tmp_path,
):
    config = fixture_config(tmp_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    write_json(output / "config.json", config)
    rank.prepare(config, output)
    with pytest.raises(ValueError, match="FRESH_PROCESS_REQUIRED"):
        rank.evaluate_prepared(
            config, output, rank.file_sha256(output / "prepared.json"), "cpu"
        )
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
    assert result["optimizer_updates"] == 0 and result["prediction_count"] == 3456
    assert result["comparisons"]["initial_projection_control"]["passed"]
    rows, _ = load_fresh_rows(config["fresh_inputs"])
    for name in rank.CONDITIONS:
        base = load_base(config["source_base"], "cpu")
        model = (
            PeftModel.from_pretrained(
                base.model, config["adapters"][name]["path"], is_trainable=False
            )
            .requires_grad_(False)
            .eval()
        )
        base = replace(base, model=model)
        with torch.inference_mode():
            single = evaluate(base, rows[:8], 768, 8)
        observed = {row["id"]: row for row in result["panels"][name]["predictions"]}
        for row in single["predictions"]:
            assert row["scores"] == pytest.approx(
                observed[row["id"]]["scores"], abs=1e-6
            )
    assert (
        result["panels"]["full_final"]["predictions"][0]["scores"]
        != result["panels"]["rank8_final"]["predictions"][0]["scores"]
    )


def test_poisoned_adapter_rejected_before_model_load(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    rank.prepare(config, output)
    receipt = json.loads((output / "prepared.json").read_text())
    receipt["parent_pid"] = -1
    write_json(output / "prepared.json", receipt)
    path = Path(config["adapters"]["rank8_final"]["path"]) / "adapter_model.safetensors"
    path.write_bytes(path.read_bytes() + b"poison")
    monkeypatch.setattr(
        rank, "load_base", lambda *_: pytest.fail("base loaded before input validation")
    )
    with pytest.raises(ValueError, match="ADAPTER_FILE_CHANGED"):
        rank.evaluate_prepared(
            config, output, rank.file_sha256(output / "prepared.json"), "cpu"
        )


def test_canonical_worker_cli_and_frozen_schema(tmp_path):
    config = fixture_config(tmp_path)
    rank.validate(config)
    bad = copy.deepcopy(config)
    bad["source_base"].pop("files")
    with pytest.raises(ValueError, match="PINNED_BASE_FILES_REQUIRED"):
        rank.validate(bad)
    path = tmp_path / "bad.json"
    write_json(path, bad)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate_rank.py"),
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0 and "PINNED_BASE_FILES_REQUIRED" in run.stderr
    assert "unrecognized arguments" not in run.stderr


if __name__ == "__main__":
    child(sys.argv[2])
