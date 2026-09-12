import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_final_retention import archived_run, command
from test_preserve_tasks import tiny_source
from test_target_calibration import tiny_models

import final_retention as final
import follow_through as follow
import learned_target
import prepare_learned_transfer as export
from calibrate_target import file_pin, pin_bundle, save_adapter
from data import SEQUENCE_TASKS, digest, write_json
from learner import save_native, shared_hash, tensor_hash
from preserve_tasks import OLD_TASKS

ROOT = Path(__file__).resolve().parents[1]
__all__ = ["archived_run"]


def successful_records(saved, step=512):
    perfect = {
        "metrics": {task: {"accuracy": 1.0, "examples": 32} for task in SEQUENCE_TASKS}
    }
    before = {
        "metrics": {
            task: {"accuracy": count / 32, "examples": 32}
            for task, count in zip(OLD_TASKS, (32, 23, 24, 20), strict=True)
        }
    }
    after = {
        "metrics": {
            task: {"accuracy": count / 32, "examples": 32}
            for task, count in zip(OLD_TASKS, (31, 23, 22, 24), strict=True)
        }
    }
    raw = {"metrics": {task: {"accuracy": 0.5, "examples": 32} for task in OLD_TASKS}}
    checkpoint = {
        "saved": saved,
        "train": perfect,
        "validation": perfect,
        "qualification": {"all_tasks_qualified": True},
    }
    training = {
        "arms": {
            "heads_new_latents": {
                "status": "completed_budget",
                "optimizer_updates": step,
                "finite_completed_updates": step,
                "checkpoints": {"0": {}, str(step): checkpoint},
            }
        }
    }
    result = {
        "arms": {
            "heads_new_latents": {
                "retention_steps": {"before": 0, "after": step},
                "checkpoints": {
                    "0": {"released_tasks": before},
                    str(step): {
                        "released_tasks": after,
                        "adapter": saved,
                        "validation": perfect,
                        "reload_predictions_exact": True,
                    },
                },
                "retention": follow.retention_comparison(before, after, raw),
            }
        },
        "raw": {"released_tasks": raw},
    }
    request = {
        "arm": "heads_new_latents",
        "step": step,
        "minimum_train_accuracy": 1.0,
        "minimum_validation_accuracy": 1.0,
        "maximum_lost_rows_per_old_task": 2,
        "expected_checkpoint": saved,
    }
    return request, result, training


def test_exact_acquisition_and_two_row_retention_gate():
    request, result, training = successful_records({"sha256": "source"})
    actual = export.qualify(request, result, training)
    assert actual["acquisition_qualified"] and actual["retention_qualified"]
    changed = copy.deepcopy(result)
    changed["arms"]["heads_new_latents"]["checkpoints"]["512"]["released_tasks"][
        "metrics"
    ]["hellaswag"]["accuracy"] = 21 / 32
    arm = changed["arms"]["heads_new_latents"]
    arm["retention"] = follow.retention_comparison(
        arm["checkpoints"]["0"]["released_tasks"],
        arm["checkpoints"]["512"]["released_tasks"],
        changed["raw"]["released_tasks"],
    )
    assert not export.qualify(request, changed, training)["retention_qualified"]
    training["arms"]["heads_new_latents"]["checkpoints"]["512"]["train"]["metrics"][
        SEQUENCE_TASKS[0]
    ]["accuracy"] = 0.99
    assert not export.qualify(request, result, training)["acquisition_qualified"]


@pytest.mark.parametrize("failure", ["endpoint", "adapter", "validation"])
def test_gate_rejects_wrong_checkpoint_or_validation(failure):
    request, result, training = successful_records({"sha256": "source"})
    arm = result["arms"]["heads_new_latents"]
    if failure == "endpoint":
        arm["retention_steps"]["after"] = 64
    elif failure == "adapter":
        arm["checkpoints"]["512"]["adapter"] = {"sha256": "other"}
    else:
        arm["checkpoints"]["512"]["validation"] = {}
    with pytest.raises(ValueError, match="FINAL_CHECKPOINT_BINDING_MISMATCH"):
        export.qualify(request, result, training)


def test_completed_corrective_and_original_receipts_are_verified(
    archived_run, tmp_path
):
    config, _ = archived_run
    path = tmp_path / "config.json"
    write_json(path, config)
    corrected = tmp_path / "corrective"
    command("final_retention.py", path, corrected)
    source_hash = final.archived_source_digest(ROOT)
    task = {
        "id": corrected.name,
        "code_dir": str(ROOT),
        "source_sha256": source_hash,
        "config": config,
    }
    execution = {
        "task_id": corrected.name,
        "task": task,
        "task_sha256": final.supervisor_digest(task),
        "config_sha256": final.supervisor_digest(config),
        "source_sha256": source_hash,
        "status": "completed",
        "exit_code": 0,
        "timed_out": False,
    }
    (corrected / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    request = {
        "corrected_retention": {
            "path": str(corrected),
            "task_id": corrected.name,
            "source_sha256": source_hash,
            "config_sha256": digest(config),
            "files": {
                name: file_pin(corrected / name)
                for name in ("execution.json", "config.json", "result.json")
            },
        },
        "source_run": config["source_run"],
        "expected_source_training_receipt": file_pin(
            Path(config["source_run"]["path"]) / "training_receipt.json"
        ),
    }
    actual = export.verify_completed_correction(request)
    assert actual[0] == corrected
    changed = copy.deepcopy(request)
    changed["corrected_retention"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CORRECTION_RECEIPT_MISMATCH"):
        export.verify_completed_correction(changed)


def source_fixture(tmp_path, monkeypatch):
    source_config, base, initial = tiny_source(tmp_path)
    learned = copy.deepcopy(initial)
    with torch.no_grad():
        for i, task in enumerate(SEQUENCE_TASKS):
            learned.task_latents[learned.config.tasks.index(task)].add_(0.1 * (i + 1))
        learned.core.A[next(iter(learned.core.A))].weight.add_(0.001)
    saved = save_adapter(learned, tmp_path / "step-0512")
    request, result, training = successful_records(saved)
    source_run, correction_run = tmp_path / "original-run", tmp_path / "corrected-run"
    source_run.mkdir()
    correction_run.mkdir()
    for filename in ("execution.json", "config.json", "result.json"):
        write_json(correction_run / filename, {})
    template = json.loads(
        (ROOT / "configs/follow_through/constructed_target64.json").read_text()
    )
    snapshot = tmp_path / "base" / base.revision
    snapshot.mkdir(parents=True)
    base.model.save_pretrained(snapshot)
    base.tokenizer.save_pretrained(snapshot)
    template["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "local_path": str(snapshot),
        "cache_dir": str(tmp_path / "cache"),
    }
    template["inputs"]["base"] = pin_bundle(snapshot)
    template["inputs"]["source_initial"] = source_config["inputs"]["source_initial"]
    _, _, _, target = tiny_models()
    save_native(target, tmp_path / "target")
    template["inputs"]["target_portal"] = {
        **pin_bundle(tmp_path / "target"),
        "tensor_sha256": tensor_hash(target.state_dict()),
    }
    template["runtime"] = source_config["runtime"]
    template["role"] = "cpu_contract_test"
    template["expected_hooks"] = len(target.config.projection_targets)
    template["protocol"]["checkpoints"] = [0, 4, 12]
    template["protocol"]["gradient_gate_step"] = 4
    template["protocol"]["training_ids"] = {
        task: ids[:4] for task, ids in template["protocol"]["training_ids"].items()
    }
    template["protocol_sha256"] = digest(template["protocol"])
    write_json(tmp_path / "template.json", template)
    request.update(
        {
            "kind": "qualify_example_learned_portal_source",
            "qualification_scope": "acquisition_and_retention",
            "source_run": {"path": str(source_run)},
            "corrected_retention": {"path": str(correction_run)},
            "calibration_template": {
                "path": str(tmp_path / "template.json"),
                **file_pin(tmp_path / "template.json"),
            },
            "code_dependencies": {
                name: file_pin(ROOT / name)
                for name in (
                    "follow_through.py",
                    "learned_target.py",
                    "prepare_learned_transfer.py",
                )
            },
        }
    )
    monkeypatch.setattr(
        export,
        "verify_completed_correction",
        lambda request: (correction_run, {}, result, {}, source_config, training, {}),
    )
    return request, result, training, learned


def test_source_export_and_target_new_pid_keep_actual_learned_vectors(
    tmp_path, monkeypatch
):
    request, _, _, learned = source_fixture(tmp_path, monkeypatch)
    exported = export.prepare(request, tmp_path / "exported")
    config_path = Path(exported["ready_config"]["path"])
    config = json.loads(config_path.read_text())
    assert (
        config["inputs"]["source_learned"]["files"]
        == pin_bundle(tmp_path / "step-0512")["files"]
    )
    assert "source_constructed" not in config["inputs"]
    arm = config["protocol"]["arms"][0]
    target = follow.build_adapter(config, arm)
    assert torch.equal(target.task_latents, learned.task_latents)
    assert shared_hash(target) == shared_hash(learned)
    assert arm["routing"] == "task_latents" and not arm["train_latents"]
    for task in SEQUENCE_TASKS:
        generated = follow.factors(target, task, arm)
        direct = target.generate(task)
        for key in generated:
            for a, b in zip(generated[key], direct[key], strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    command("learned_target.py", config_path, tmp_path / "preflight", "--prepare-only")
    command("learned_target.py", config_path, tmp_path / "training")
    result = json.loads((tmp_path / "training/result.json").read_text())
    assert result["training_pid"] != result["evaluation_pid"]
    assert set(result["arms"]) == {"learned", "untouched", "lora"}
    assert all(
        value["retention_steps"] == {"before": 0, "after": 12}
        for value in result["arms"].values()
    )
    receipt = json.loads((tmp_path / "training/training_receipt.json").read_text())
    restored = follow.restore_adapter(
        receipt["arms"]["learned"]["checkpoints"]["12"]["saved"], "cpu"
    )
    assert torch.equal(restored.task_latents, learned.task_latents)
    assert shared_hash(restored) == shared_hash(learned)
    rejected = copy.deepcopy(config)
    rejected["protocol"]["arms"][0]["routing"] = "fixed_rte"
    rejected["protocol_sha256"] = digest(rejected["protocol"])
    with pytest.raises(ValueError, match="LEARNED_VECTORS_MUST_BE_FROZEN"):
        learned_target.validate(rejected)


def test_export_refuses_failed_retention_before_creating_artifacts(
    tmp_path, monkeypatch
):
    request, _, _, _ = source_fixture(tmp_path, monkeypatch)
    request["maximum_lost_rows_per_old_task"] = 1
    with pytest.raises(ValueError, match="RETENTION_UNQUALIFIED"):
        export.prepare(request, tmp_path / "exported")
    assert not (tmp_path / "exported").exists()


def test_ready_request_uses_qualified_final_and_only_sixty_four_row_template():
    request = json.loads((ROOT / "configs/learned_transfer/source.json").read_text())
    assert request["arm"] == "heads_new_latents" and request["step"] == 512
    assert (
        request["minimum_train_accuracy"] == request["minimum_validation_accuracy"] == 1
    )
    assert request["maximum_lost_rows_per_old_task"] == 2
    template = follow.read_json_input(
        {"inputs": {"template": request["calibration_template"]}}, "template"
    )
    assert {len(ids) for ids in template["protocol"]["training_ids"].values()} == {64}
    assert template["protocol"]["checkpoints"] == [0, 20, 80, 160, 320]
