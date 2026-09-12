from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import calibrate_teacher as teacher
import pytest
import torch
from data import SEQUENCE_TASKS, write_json
from learner import frozen_base_tensors, tensor_hash
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from test_contract import tiny_model


def calibration_fixture(directory):
    base, _ = tiny_model()
    snapshot = directory / ("0" * 40)
    base.model.save_pretrained(snapshot)
    base.tokenizer.save_pretrained(snapshot)
    config = {
        "kind": "cpu_teacher_calibration_fixture",
        "seed": 90302,
        "base": {
            "repo_id": base.model_id,
            "revision": base.revision,
            "local_path": str(snapshot),
            "cache_dir": str(directory),
        },
        "base_tensor_sha256": tensor_hash(frozen_base_tensors(base.model)),
        "checkpoints": {},
        "inputs": [],
        "generation": {"batch_size": 3, "max_new_tokens": 4},
        "gate": {
            "minimum_accuracy_each_task_and_split": 0.9,
            "minimum_validation_gain_over_initial": 0.5,
        },
        "proof_boundary": "CPU fixture only",
    }
    model = get_peft_model(
        base.model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    for name in ("initial", "teacher"):
        if name == "teacher":
            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if "lora_B" in key:
                        tensor.add_(0.03)
        path = directory / name
        model.save_pretrained(path, save_embedding_layers=False)
        config["checkpoints"][name] = {
            "path": str(path),
            "files": {
                key: teacher.file_hash(path / key)
                for key in ("adapter_config.json", "adapter_model.safetensors")
            },
            "tensor_sha256": tensor_hash(get_peft_model_state_dict(model)),
        }
    for task in SEQUENCE_TASKS:
        for split, group in (("train", "0 1 2"), ("validation", "3 4 5")):
            path = directory / "data" / f"{task}_{split}.json"
            write_json(
                path,
                [
                    {
                        "id": f"{task}/{split}",
                        "task": task,
                        "prompt": f"Apply the {task} code.\nInput: {group}\nOutput:",
                        "choices": [" 7 6 5", " 4 3 2", " 1 0 7", " 6 5 4"],
                        "gold_idx": 0,
                        "group": group,
                    }
                ],
            )
            config["inputs"].append(
                {
                    "task": task,
                    "split": split,
                    "path": str(path),
                    "sha256": teacher.file_hash(path),
                    "total_rows": 1,
                    "take_first": 1,
                }
            )
    return config


def test_saved_real_model_generation_frozen_and_reproduced_in_fresh_process(
    tmp_path, monkeypatch
):
    config = calibration_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration must not train")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    result = teacher.calibrate(config, tmp_path / "parent", "cpu")
    assert result["optimizer_updates"] == result["trainable_parameters"] == 0
    assert result["base_before_sha256"] == result["base_after_sha256"]
    assert result["adapter_before_sha256"] == result["adapter_after_sha256"]
    assert (
        result["adapter_before_sha256"]["initial"]
        != result["adapter_before_sha256"]["teacher"]
    )
    config_path = tmp_path / "cpu-config.json"
    write_json(config_path, config)
    child = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-c",
            "import json,sys;from pathlib import Path;from calibrate_teacher import calibrate;calibrate(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]),'cpu')",
            str(config_path),
            str(tmp_path / "child"),
        ],
        cwd=Path(teacher.__file__).parent,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    child_result = json.loads((tmp_path / "child/result.json").read_text())
    assert child_result["pid"] != result["pid"]
    assert child_result["panels"] == result["panels"]
    for name in ("initial", "teacher"):
        records = json.loads((tmp_path / "parent" / f"{name}.json").read_text())
        assert records == json.loads((tmp_path / "child" / f"{name}.json").read_text())
        assert len(records) == 6
        for row in records:
            assert "7 6 5" not in row["prompt"]
            assert len(row["generated_token_ids"]) <= 4


def test_grading_rejects_extraction_cues_and_incomplete_answers():
    for text in (
        "The answer is 1 2 3",
        "1 2",
        "1 2 3 4",
        "7 2 3",
        "\n1 2 3",
        "1,2,3",
        "Output: 1 2 3",
    ):
        assert not teacher.grade_line(text, " 1 2 3")["correct"]
    assert teacher.grade_line(" 1 2 3\nInput:", " 1 2 3")["correct"]


def test_rejects_heldout_file_and_detects_checkpoint_tampering(tmp_path):
    config = calibration_fixture(tmp_path)
    changed = copy.deepcopy(config["inputs"])
    changed[0]["split"] = "test"
    changed[0]["path"] = str(tmp_path / "must-not-read-test.json")
    with pytest.raises(ValueError, match="SPLIT_CONTRACT"):
        teacher.read_calibration_rows(changed)
    path = Path(config["checkpoints"]["teacher"]["path"]) / "adapter_model.safetensors"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="CHECKPOINT_HASH_MISMATCH"):
        teacher.checkpoint_files(config)
