import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from test_compressed_replay import (
    model,
    native_fixture,
    publish_manifest,
    runtime_fixture,
    sources_fixture,
)
from test_followup import execution_receipt
from test_learning import EncodedFixture, config

import compressed_replay
import delete_rehearsal
import followup
from data import build_corpus, digest, scheduled_batch
from learning import capture, frozen_hash, load_checkpoint, new_optimizer, tree_record, write_json

ROOT = Path(__file__).resolve().parents[1]


class ObjectiveEncoded(EncodedFixture):
    def __init__(self, settings, short_id):
        super().__init__(settings)
        self.short_id = short_id

    def batch(self, rows, training):
        inputs, labels, keep = super().batch(rows, training)
        for index, row in enumerate(rows):
            inputs["input_ids"][index, 2] = 1 + int(row.id[:2], 16) % 8
            labels[index, -2] = -100 if row.id == self.short_id else 2 + int(row.id[2:4], 16) % 7
        return inputs, labels, keep


def test_gradient_is_exact_current_component_of_four_row_answer_loss():
    settings = config()
    _, corpus = build_corpus(settings)
    current = corpus["workflow"][settings["target_family"]]["train"][:2]
    old = corpus["primitive"]["text"]["train"][:2]
    encoded = ObjectiveEncoded(settings, current[0].id)
    candidate = model()
    with torch.no_grad():
        candidate.projection.B.normal_(0, 0.1)
    reference = copy.deepcopy(candidate)
    base = frozen_hash(candidate)
    loss, report = delete_rehearsal.current_objective(candidate, encoded, current)
    inputs, labels, keep = encoded.batch(current + old, training=True)
    logits = reference(**inputs, use_cache=False, logits_to_keep=keep + 1).logits[:, :-1]
    labels = labels[:, -keep:]
    tokens = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view_as(labels)
    mask = labels != -100
    rows = (tokens * mask).sum(1) / mask.sum(1)
    expected = rows[:2].sum() / 4
    actual_gradients = torch.autograd.grad(loss, [value for value in candidate.parameters() if value.requires_grad])
    expected_gradients = torch.autograd.grad(
        expected, [value for value in reference.parameters() if value.requires_grad], retain_graph=True
    )
    full_gradients = torch.autograd.grad(
        rows.mean(), [value for value in reference.parameters() if value.requires_grad]
    )
    torch.testing.assert_close(loss, expected)
    for actual, wanted in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-6)
    assert any(not torch.allclose(actual, full) for actual, full in zip(actual_gradients, full_gradients, strict=True))
    assert report["target_tokens"] == 3 and report["loss_denominator"] == 4
    assert report["current_mean_loss"] == pytest.approx(2 * float(loss.detach()))
    assert frozen_hash(candidate) == base


def fixture(tmp_path):
    settings, source = sources_fixture(tmp_path)
    protocol = json.loads((ROOT / "delete_rehearsal_protocol.json").read_text())
    code = tmp_path / "reference-code"
    for relative in protocol["dependency_files"]:
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    root = tmp_path / "reference"
    root.mkdir()
    task = {
        "id": "reference-hybrid",
        "entrypoint": "compressed_replay.py",
        "code_dir": str(code),
        "source_sha256": protocol["dependency_source_sha256"],
        "config": settings,
    }
    receipt = execution_receipt(task, attempt="d" * 32)
    receipt.update(started_at="2026-09-12T05:02:00+00:00", finished_at="2026-09-12T05:03:00+00:00")
    (root / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (root / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_json(root / "config.json", settings)
    compression = Path(settings["compression_run_dir"])
    candidate = model()
    initial_identity = digest(tree_record(capture(candidate)))
    optimizer_identity = digest(tree_record(new_optimizer(candidate, source.config).state_dict()))
    result = {
        "status": "compressed_workflow_training_completed",
        "source_result_sha256": source.result_sha256,
        "compression_result_sha256": followup.sha256_file(compression / "result.json"),
        "methods": {
            "agent_dice": {
                "status": "trained",
                "arms": {
                    arm: {"initial_adapter_sha256": initial_identity, "initial_optimizer_sha256": optimizer_identity}
                    for arm in ("compressed_continue", "compressed_replay")
                },
                "ancestral_accounting": {"source_expert_training_updates": 512},
            }
        },
    }
    for name in followup.evaluation_sets(source, "validation"):
        target = root / "agent_dice/initial-dev" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((compression / "agent_dice/compressed" / f"{name}.json").read_bytes())
    path = root / "agent_dice/compressed_replay/updates.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "step": step + 1,
            "current_ids": [
                row.id
                for row in scheduled_batch(
                    source.corpus["workflow"][source.config["target_family"]]["train"],
                    step,
                    4,
                    [source.config["optimization_seed"], "workflow", source.config["target_family"]],
                )[:2]
            ],
            "replay_ids": [row.id for row in source.corpus["primitive"]["text"]["train"][:2]],
        }
        for step in range(128)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    publish_manifest(root, result)
    settings = {
        **settings,
        "experiment": "skill-transfer-delete-rehearsal",
        "protocol": "delete_rehearsal_protocol.json",
        "protocol_sha256": followup.sha256_file(ROOT / "delete_rehearsal_protocol.json"),
        "reference_run_dir": str(root),
        "reference_task_id": task["id"],
        "reference_config_sha256": digest(task["config"]),
    }
    return settings, source


def transport(output, settings, monkeypatch):
    output.mkdir()
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") or part in {"tests", "outputs", "runs", "__pycache__"}
            for part in path.relative_to(ROOT).parts
        )
        and (
            path.suffix in {".py", ".json", ".txt", ".yaml", ".yml", ".sha256"}
            or path.name.startswith(("LICENSE", "NOTICE"))
        )
    )
    checksum = hashlib.sha256()
    for path in files:
        checksum.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    task = {
        "id": f"delete-{settings['stage']}",
        "entrypoint": "delete_rehearsal.py",
        "code_dir": str(ROOT),
        "source_sha256": checksum.hexdigest(),
        "config": settings,
    }
    receipt = execution_receipt(task, "running", ("e" if settings["stage"] == "train" else "f") * 32)
    receipt["started_at"] = "2026-09-12T05:04:00+00:00" if settings["stage"] == "train" else "2026-09-12T05:06:00+00:00"
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    monkeypatch.setattr(sys, "argv", [str(ROOT / "delete_rehearsal.py")])


@pytest.mark.parametrize("initial_mismatch", [False, True])
def test_control_updates_matched_rows_and_reloads_or_stops_before_updates(tmp_path, monkeypatch, initial_mismatch):
    settings, source = fixture(tmp_path)
    output = tmp_path / "train"
    transport(output, settings, monkeypatch)
    monkeypatch.setattr(delete_rehearsal, "fresh_runtime", runtime_fixture)
    monkeypatch.setattr(compressed_replay, "fresh_runtime", runtime_fixture)

    def native(model_, encoded, rows):
        records = native_fixture(model_, encoded, rows)
        if initial_mismatch and rows[0].kind == "workflow":
            records[0]["raw_generation_ids"].append(0)
        return records

    monkeypatch.setattr(followup, "generate", native)
    if initial_mismatch:

        def forbidden(*args, **kwargs):
            raise AssertionError("initial workflow DEV mismatch must block optimizer construction")

        monkeypatch.setattr(delete_rehearsal, "new_optimizer", forbidden)
    result = delete_rehearsal.experiment(settings, output)
    assert result["training_updates"] == (0 if initial_mismatch else 128)
    assert not result["test_evaluated"]
    assert not (output / "replay-memory.json").exists()
    assert not any(".test." in path.name for path in output.rglob("*"))
    if initial_mismatch:
        assert result["methods"]["agent_dice"]["reason"] == "fresh_dev_mismatch_or_unqualified"
        return
    arm = result["methods"]["agent_dice"]["arms"][delete_rehearsal.ARM]
    assert arm["exposures"]["current_exposures"] == 256 and arm["exposures"]["old_exposures"] == 0
    assert arm["loss_denominator"] == 4 and arm["actual_batch_size"] == 2 and arm["memory_examples"] == 0
    updates = [
        json.loads(line) for line in (output / "agent_dice/delete_rehearsal/updates.jsonl").read_text().splitlines()
    ]
    reference = [
        json.loads(line)
        for line in (Path(settings["reference_run_dir"]) / "agent_dice/compressed_replay/updates.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(
        row["current_ids"] == old["current_ids"] and not row["replay_ids"] and row["example_ids"] == row["current_ids"]
        for row, old in zip(updates, reference, strict=True)
    )
    assert all(row["loss"] == pytest.approx(0.5 * row["current_mean_loss"]) for row in updates)
    restored = model()
    optimizer = load_checkpoint(restored, output / arm["checkpoint"])
    assert {int(row["step"]) for row in optimizer["state"].values()} == {128}
    assert torch.count_nonzero(restored.projection.B) and restored.projection.offset is None
    assert frozen_hash(restored) == source.result["frozen_base"]["before"]
    assert tree_record(capture(restored)) != tree_record(capture(model()))
    receipt = json.loads((output / "execution.json").read_text())
    receipt.update(status="completed", finished_at="2026-09-12T05:05:00+00:00", exit_code=0, timed_out=False)
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    audit_settings = {
        **settings,
        "stage": "audit",
        "training_run_dir": str(output),
        "training_task_id": "delete-train",
        "training_config_sha256": digest(settings),
    }
    audit_output = tmp_path / "audit"
    transport(audit_output, audit_settings, monkeypatch)
    audited = delete_rehearsal.experiment(audit_settings, audit_output)
    assert audited["all_dev_reloads_exact"] and audited["evaluated_arms"] == 1 and audited["training_updates"] == 0
    assert len(audited["methods"]["agent_dice"]["arms"][delete_rehearsal.ARM]["test"]) == 6
    assert audited["original_test_previously_observed"] and audited["posthoc_mechanism_control"]
    assert audited["process_proof"]["task_id"] != result["process_proof"]["task_id"]
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    expected = restored(**inputs).logits.detach().tolist()
    script = """import json
import sys
import torch
from test_learning import TinyLanguageModel
from learning import load_checkpoint
torch.manual_seed(73)
model = TinyLanguageModel()
load_checkpoint(model, sys.argv[1])
inputs = {'input_ids': torch.tensor([[1,2,3,4,5]]), 'attention_mask': torch.ones(1,5,dtype=torch.long)}
print(json.dumps(model(**inputs).logits.detach().tolist()))
"""
    child = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-c",
            script,
            str(output / arm["checkpoint"]),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "tests") + os.pathsep + str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(child.stdout) == expected
