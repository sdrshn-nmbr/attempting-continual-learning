import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_followup import execution_receipt, response, source_fixture
from test_learning import EncodedFixture, TinyLanguageModel

import compressed_replay
import followup
from data import audit_corpus, build_corpus, digest, scheduled_batch
from learning import capture, frozen_hash, load_checkpoint, save_checkpoint, score, tree_record, write_json
from sandbox import FAMILIES

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / "compressed_replay_protocol.json").read_text())


def model():
    torch.manual_seed(73)
    return TinyLanguageModel()


def native_fixture(model_, encoded, rows):
    return [response(row, encoded.conventions, True) for row in rows]


class HybridEncoded(EncodedFixture):
    def batch(self, rows, training):
        inputs, labels, keep = super().batch(rows, training)
        for index, row in enumerate(rows):
            if row.kind == "primitive":
                labels[index, -2] = 6
        return inputs, labels, keep


def runtime_fixture(source, output):
    encoded = HybridEncoded(source.config)
    encoded.conventions = source.spec["conventions"]
    return model(), encoded


def publish_manifest(root, result):
    result["artifact_manifest"] = {
        str(path.relative_to(root)): followup.sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[0] not in {"config.json", "task.json", "execution.json", "result.json"}
    }
    (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")


def sources_fixture(tmp_path, qualifies=(False, True)):
    settings, old_protocol, _ = source_fixture(tmp_path)
    root = Path(settings["source_run_dir"])
    config = json.loads((root / "config.json").read_text())
    config["primitive_train_examples"] = 64
    spec, corpus = build_corpus(config)
    settings["source_config_sha256"] = digest(config)
    (root / "config.json").write_text(json.dumps(config))
    task = json.loads((root / "task.json").read_text())
    task["config"] = config
    (root / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (root / "execution.json").write_text(json.dumps(execution_receipt(task), indent=2) + "\n")
    (root / "dataset-audit.json").write_text(json.dumps(audit_corpus(spec, corpus)))
    result = json.loads((root / "result.json").read_text())
    result["config_sha256"] = digest(config)
    result["frozen_base"]["before"] = frozen_hash(model())
    result["qualification"]["rounds"] = [{"budget": 128}]
    for family in FAMILIES:
        path = root / f"qualification/{family}/128/updates.jsonl"
        records = [
            {
                "step": step + 1,
                "current_ids": [
                    row.id
                    for row in scheduled_batch(corpus["primitive"][family]["train"], step, 4, [17, "primitive", family])
                ],
                "replay_ids": [],
            }
            for step in range(128)
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    for record in result["transfer"].values():
        record.update(
            test={"fixture": "dense-reference"},
            novel_composition_test={"fixture": "dense-reference"},
            primitive_after_test={"fixture": "dense-reference"},
        )
    publish_manifest(root, result)
    source = followup.SourceRun(settings, old_protocol)
    compressed = tmp_path / "compression"
    compressed.mkdir()
    code = tmp_path / "compression-code"
    for relative in protocol()["dependency_files"]:
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    comp_settings = {**settings, "experiment": "skill-transfer-compress"}
    task = {
        "id": "compressed-fixture",
        "entrypoint": "compress_fusion.py",
        "code_dir": str(code),
        "source_sha256": protocol()["dependency_source_sha256"],
        "config": comp_settings,
    }
    (compressed / "config.json").write_text(json.dumps(comp_settings))
    (compressed / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    receipt = execution_receipt(task, attempt="b" * 32)
    receipt.update(started_at="2026-09-12T05:00:00+00:00", finished_at="2026-09-12T05:01:00+00:00")
    (compressed / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    comp_result = {
        "status": "compression_followup_completed",
        "checkpoint_scope": "pre_workflow_fusion",
        "source_result_sha256": source.result_sha256,
        "source_execution_sha256": source.execution_sha256,
        "methods": {},
    }
    for method, passes in zip(("arithmetic", "agent_dice"), qualifies, strict=True):
        save_checkpoint(model(), None, compressed / method / "checkpoint", {"rank": 8, "training_updates": 0})
        write_json(compressed / method / "svd.json", {"rank": 8, "training_updates": 0, "fixture": True})
        for split in ("validation", "test"):
            for name, rows in followup.evaluation_sets(source, split).items():
                records = [response(row, spec["conventions"], passes or row.family != "calendar") for row in rows]
                write_json(
                    compressed / method / "compressed" / f"{name}.json", {"metrics": score(records), "records": records}
                )
        comp_result["methods"][method] = {
            "svd_performed": True,
            "checkpoint": f"{method}/checkpoint",
            "memory": {
                "source_expert_training_updates": 512,
                "source_expert_training_example_exposures": 2048,
                "original": {"dense_offset_bytes": 16},
                "retained_source_expert_archive_bytes_fp32": 64,
            },
        }
    publish_manifest(compressed, comp_result)
    settings.pop("followup_protocol")
    settings.pop("followup_protocol_sha256")
    settings.update(
        experiment="skill-transfer-compressed-replay",
        stage="train",
        protocol="compressed_replay_protocol.json",
        protocol_sha256=followup.sha256_file(ROOT / "compressed_replay_protocol.json"),
        compression_run_dir=str(compressed),
        compression_task_id=task["id"],
        compression_config_sha256=digest(comp_settings),
    )
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
        "id": f"hybrid-{settings['stage']}",
        "entrypoint": "compressed_replay.py",
        "code_dir": str(ROOT),
        "source_sha256": checksum.hexdigest(),
        "config": settings,
    }
    receipt = execution_receipt(task, "running", ("e" if settings["stage"] == "train" else "f") * 32)
    receipt["started_at"] = "2026-09-12T05:02:00+00:00" if settings["stage"] == "train" else "2026-09-12T05:04:00+00:00"
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    monkeypatch.setattr(sys, "argv", [str(ROOT / "compressed_replay.py")])


def test_memory_has_exactly_sixty_four_exposed_train_rows_and_no_growth(tmp_path):
    _, source = sources_fixture(tmp_path)
    memory, record = compressed_replay.replay_memory(source, protocol())
    assert len(memory) == len({row.id for row in memory}) == 64
    assert all(
        value["retained"] == 16 and set(value["per_operation"].values()) == {4}
        for value in record["admission"].values()
    )
    assert memory == compressed_replay.replay_memory(source, protocol())[0]
    before = digest([row.record() for row in memory])
    counts = {arm: [0, 0] for arm in protocol()["arms"]}
    for step in range(128):
        new, old = compressed_replay.workflow_batch(source, memory, "compressed_replay", step, protocol())
        full, empty = compressed_replay.workflow_batch(source, memory, "compressed_continue", step, protocol())
        assert new == full[:2] and not empty
        assert len(old) == 2 and old[0].id != old[1].id
        assert all(row.split == "train" and row.kind == "primitive" for row in old)
        counts["compressed_continue"][0] += len(full)
        counts["compressed_replay"][0] += len(new)
        counts["compressed_replay"][1] += len(old)
    assert counts == {"compressed_continue": [512, 0], "compressed_replay": [256, 256]}
    assert digest([row.record() for row in memory]) == before


def test_same_operation_threshold_accepts_five_of_eight_but_rejects_four():
    config = json.loads((ROOT / "configs/stream_4409.json").read_text())
    baseline = {
        family: {"n": 32, "accuracy": 0, "per_pattern": {operation: 0 for operation in operations}}
        for family, operations in compressed_replay.SCHEMAS.items()
    }
    compressed = {
        family: {"n": 32, "accuracy": 1, "per_pattern": {operation: 1 for operation in operations}}
        for family, operations in compressed_replay.SCHEMAS.items()
    }
    compressed["calendar"].update(accuracy=29 / 32)
    compressed["calendar"]["per_pattern"]["add"] = 5 / 8
    assert compressed_replay.fusion_gate(baseline, compressed, config)["passed"]
    compressed["calendar"].update(accuracy=28 / 32)
    compressed["calendar"]["per_pattern"]["add"] = 4 / 8
    assert not compressed_replay.fusion_gate(baseline, compressed, config)["passed"]


def test_gate_failure_never_loads_model_trains_or_reads_test(tmp_path, monkeypatch):
    settings, _ = sources_fixture(tmp_path, qualifies=(False, False))
    output = tmp_path / "train"
    transport(output, settings, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("failed DEV gate must block runtime and training")

    monkeypatch.setattr(compressed_replay, "fresh_runtime", forbidden)
    monkeypatch.setattr(compressed_replay, "new_optimizer", forbidden)
    result = compressed_replay.experiment(settings, output)
    assert result["training_updates"] == 0 and not result["test_evaluated"]
    assert all(row["status"] == "branch_gate_failed" for row in result["methods"].values())
    consumed = json.loads((output / "consumed-compression.json").read_text())
    assert all("test" not in name and "checkpoint" not in name for name in consumed)


def test_fresh_dev_mismatch_blocks_updates_and_preserves_other_gate(tmp_path, monkeypatch):
    settings, _ = sources_fixture(tmp_path)
    output = tmp_path / "train"
    transport(output, settings, monkeypatch)
    monkeypatch.setattr(compressed_replay, "fresh_runtime", runtime_fixture)

    def changed(model_, encoded, rows):
        records = native_fixture(model_, encoded, rows)
        records[0]["raw_generation_ids"].append(0)
        return records

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched reload must not construct an optimizer")

    monkeypatch.setattr(followup, "generate", changed)
    monkeypatch.setattr(compressed_replay, "new_optimizer", forbidden)
    result = compressed_replay.experiment(settings, output)
    assert result["training_updates"] == 0
    assert result["methods"]["agent_dice"]["reason"] == "fresh_dev_mismatch_or_unqualified"


@pytest.mark.parametrize("audit_mismatch", [False, True])
def test_real_cpu_training_fixed_forks_and_separate_audit_contract(tmp_path, monkeypatch, audit_mismatch):
    settings, source = sources_fixture(tmp_path)
    output = tmp_path / "train"
    transport(output, settings, monkeypatch)
    monkeypatch.setattr(compressed_replay, "fresh_runtime", runtime_fixture)
    monkeypatch.setattr(followup, "generate", native_fixture)
    trained = compressed_replay.experiment(settings, output)
    assert trained["training_updates"] == 256 and not trained["test_evaluated"]
    assert trained["rehearsal_memory"]["actual_examples"] == 64 and trained["rehearsal_memory"]["file_bytes"] > 0
    arms = trained["methods"]["agent_dice"]["arms"]
    assert len({row["initial_adapter_sha256"] for row in arms.values()}) == 1
    assert len({row["initial_optimizer_sha256"] for row in arms.values()}) == 1
    final_states = []
    for name, record in arms.items():
        assert record["resident"]["dense_offset_bytes"] == 0 and record["resident"]["optimizer_bytes"] > 0
        assert record["exposures"]["updates"] == 128
        expected_new = 512 if name == "compressed_continue" else 256
        assert record["exposures"]["current_exposures"] == expected_new
        assert record["exposures"]["old_exposures"] == 512 - expected_new
        restored = model()
        load_checkpoint(restored, output / record["checkpoint"])
        assert (
            torch.count_nonzero(restored.projection.B)
            and frozen_hash(restored) == source.result["frozen_base"]["before"]
        )
        assert tree_record(capture(restored)) != tree_record(capture(model()))
        final_states.append(digest(tree_record(capture(restored))))
    assert len(set(final_states)) == 2
    assert not any(".test." in path.name for path in output.rglob("*"))
    receipt = json.loads((output / "execution.json").read_text())
    receipt.update(status="completed", finished_at="2026-09-12T05:03:00+00:00", exit_code=0, timed_out=False)
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    audit_settings = {
        **settings,
        "stage": "audit",
        "training_run_dir": str(output),
        "training_task_id": "hybrid-train",
        "training_config_sha256": digest(settings),
    }
    audit_output = tmp_path / "audit"
    transport(audit_output, audit_settings, monkeypatch)
    if audit_mismatch:
        measured = 0

        def changed(model_, encoded, rows):
            nonlocal measured
            records = native_fixture(model_, encoded, rows)
            if measured == 0:
                records[0]["raw_generation_ids"].append(0)
            measured += 1
            return records

        monkeypatch.setattr(followup, "generate", changed)
    audited = compressed_replay.experiment(audit_settings, audit_output)
    assert audited["training_updates"] == 0 and audited["all_dev_reloads_exact"] != audit_mismatch
    for name, record in audited["methods"]["agent_dice"]["arms"].items():
        if audit_mismatch and name == "compressed_continue":
            assert not record["test_evaluated"]
            assert not (audit_output / "agent_dice/compressed_continue/workflow.test.json").exists()
        else:
            assert len(record["test"]) == 6 and record["status"] == "fresh_process_audited"
    assert audited["process_proof"]["task_id"] != trained["process_proof"]["task_id"]
    checkpoint = output / arms["compressed_replay"]["checkpoint"]
    restored = model()
    load_checkpoint(restored, checkpoint)
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
        ["uv", "run", "--no-project", "--python", sys.executable, "python", "-c", script, str(checkpoint)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "tests") + os.pathsep + str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(child.stdout) == expected


def test_rank8_guard_rejects_hidden_dense_offset():
    candidate = model()
    candidate.projection.offset = torch.zeros(11, 8)
    with pytest.raises(RuntimeError, match="RANK8_WITHOUT_OFFSET"):
        compressed_replay.rank8_memory(candidate)


def test_no_eligible_arms_does_not_claim_persistent_reloads(tmp_path, monkeypatch):
    settings, _ = sources_fixture(tmp_path, qualifies=(False, False))
    output = tmp_path / "train"
    transport(output, settings, monkeypatch)
    result = compressed_replay.experiment(settings, output)
    receipt = json.loads((output / "execution.json").read_text())
    receipt.update(status="completed", finished_at="2026-09-12T05:03:00+00:00", exit_code=0, timed_out=False)
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    settings = {
        **settings,
        "stage": "audit",
        "training_run_dir": str(output),
        "training_task_id": "hybrid-train",
        "training_config_sha256": digest(settings),
    }
    audit_output = tmp_path / "audit"
    transport(audit_output, settings, monkeypatch)
    audited = compressed_replay.experiment(settings, audit_output)
    assert result["training_updates"] == 0
    assert audited["status"] == "no_eligible_compressed_arms"
    assert audited["evaluated_arms"] == 0 and audited["all_dev_reloads_exact"] is None


@pytest.mark.parametrize("mutation", ["exit", "config", "source_code"])
def test_compression_parent_receipt_and_archive_are_verified(tmp_path, mutation):
    settings, source = sources_fixture(tmp_path)
    root = Path(settings["compression_run_dir"])
    if mutation == "exit":
        receipt = json.loads((root / "execution.json").read_text())
        receipt["exit_code"] = 1
        (root / "execution.json").write_text(json.dumps(receipt))
    elif mutation == "config":
        settings["compression_config_sha256"] = "wrong"
    else:
        task = json.loads((root / "task.json").read_text())
        (Path(task["code_dir"]) / "learning.py").write_text("changed")
    with pytest.raises(RuntimeError):
        compressed_replay.ArtifactRun(
            root,
            settings["compression_task_id"],
            settings["compression_config_sha256"],
            protocol()["dependency_files"],
            source,
            protocol()["dependency_source_sha256"],
        )
