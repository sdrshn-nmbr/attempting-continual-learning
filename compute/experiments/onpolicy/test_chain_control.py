import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import chain_control as chain
import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from tasks import Example, execute, grade, input_pool, prompt
from test_onpolicy import tiny_config, tiny_model, tiny_tokenizer


def test_roots_orders_and_privilege():
    examples = chain.build_examples()
    manifest = chain.input_manifest(examples)
    assert len(examples) == 128
    assert len({x.inputs for x in examples}) == 32
    assert not {x.inputs for x in examples} & set(input_pool(37)[:160])
    assert not {x.inputs for x in examples} & set(input_pool(37)[192:224])
    assert manifest["order_sensitive_examples"] > 100
    for example in examples:
        text = prompt(example, {}, "cue_only", privileged=False)
        assert "Examples:" not in text and "Authoritative" not in text
        assert "modulo" not in text and "Reverse the order" not in text
        assert text.endswith("Output:")
    with pytest.raises(ValueError, match="OVERLAP"):
        chain.input_manifest(
            [
                Example(
                    "permutation", "chain_diagnostic", input_pool(37)[4], ("dax", "wug")
                )
            ]
        )


def test_actual_prediction_is_chained_and_invalid_stops():
    root = (0, 1, 2, 3)
    example = Example("permutation", "diagnostic", root, ("dax", "wug"))
    seen = []

    def query(row):
        seen.append((row.program, row.inputs))
        value = (
            list(root) if row.program == ("dax",) else execute(row.inputs, row.program)
        )
        text = json.dumps(value)
        return {"key": row.key, "text": text, **grade(text, row)}

    measured = chain.measure_example(example, query)
    assert seen[2] == (("wug",), root)
    assert seen[4] == (("wug",), (3, 2, 1, 0))
    assert not measured["model_chain"]["correct"]
    assert measured["oracle_intermediate"]["correct"]
    assert not measured["oracle_intermediate"]["all_steps_correct"]
    for malformed in (
        "[true,1,2,3]",
        "[0,1,2]",
        "[0,1,2,4]",
        "text [0,1,2,3]",
        "[0,1,2,3] trailing",
    ):
        assert chain.parse_output(malformed) is None
    calls = []

    def invalid(row):
        calls.append(row)
        return {"key": row.key, "text": "invalid", "correct": False}

    result = chain.measure_example(example, invalid)
    assert len(calls) == 4
    assert result["model_chain"]["invalid"]
    assert len(result["model_chain"]["calls"]) == 1


def run_native_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    config = replace(
        tiny_config(directory), max_new_tokens=2, gradient_checkpointing=False
    )
    model, tokenizer = tiny_model(config), tiny_tokenizer()
    state = {
        name: value.detach().cpu().clone()
        for name, value in get_peft_model_state_dict(model).items()
    }
    checkpoints = {}
    for condition in chain.CONDITIONS:
        if condition == "after_symbol_map":
            for name, value in state.items():
                if "lora_B" in name:
                    value.add_(0.05)
        path = directory / f"{condition}.safetensors"
        save_file(state, path)
        checkpoints[condition] = {"path": str(path), "sha256": chain.file_hash(path)}
    recipe = {
        "checkpoints": checkpoints,
        "model_config": {"model_path": config.model_path},
        "base_config_sha256": chain.file_hash(Path(config.model_path) / "config.json"),
        "protocol": {"test": True},
    }

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "An evaluation must not create an optimizer or backpropagate"
        )

    torch.optim.AdamW = forbidden
    torch.Tensor.backward = forbidden
    output = directory / "output"
    output.mkdir()
    result = chain.evaluate_loaded(
        recipe,
        config,
        model,
        tokenizer,
        {"pid": os.getpid()},
        output,
        chain.build_examples()[:1],
    )
    assert result["optimizer_updates"] == 0
    assert all(
        panel["identity"]["base_and_adapter_unchanged"]
        for panel in result["panels"].values()
    )
    assert all(not value.requires_grad for value in model.parameters())
    assert all(
        0 < panel["budget"]["unique_calls"] <= 5 for panel in result["panels"].values()
    )
    return result


def test_native_reload_and_generation_in_separate_processes(tmp_path):
    results = []
    for index in range(2):
        directory = tmp_path / str(index)
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                "-c",
                "import sys; from test_chain_control import run_native_fixture; run_native_fixture(sys.argv[1])",
                str(directory),
            ],
            check=True,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        results.append(json.loads((directory / "output/result.json").read_text()))
    assert results[0]["runtime"]["pid"] != results[1]["runtime"]["pid"]
    for condition in chain.CONDITIONS:
        assert (
            results[0]["panels"][condition]["records"]
            == results[1]["panels"][condition]["records"]
        )
        assert (
            results[0]["panels"][condition]["identity"]["tensor_sha256"]
            == results[1]["panels"][condition]["identity"]["tensor_sha256"]
        )


def test_bad_checkpoint_rejected_before_model_access(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    state = tmp_path / "adapter.safetensors"
    state.write_bytes(b"invalid")
    recipe = {
        "model_config": {"model_path": str(tmp_path)},
        "base_config_sha256": chain.file_hash(config_file),
        "checkpoints": {"initial": {"path": str(state), "sha256": "0" * 64}},
    }
    with pytest.raises(ValueError, match="INPUT_HASH_MISMATCH"):
        chain.verify_files(recipe)


def test_worker_cli_contract_checks_config_before_loading(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(Path(chain.__file__)),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "CHAIN_FROZEN_CONFIG_MISMATCH" in result.stderr
    event = json.loads((output / "events.jsonl").read_text())
    assert event["event"] == "chain_failed"
