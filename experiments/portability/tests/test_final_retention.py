import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_preserve_tasks import tiny_source

import final_retention as final
import follow_through as follow
from calibrate_target import file_pin, pin_bundle
from data import digest, write_json
from preserve_tasks import OLD_TASKS

ROOT = Path(__file__).resolve().parents[1]


def command(script, config, output, *extra):
    executed = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / script),
            "--config",
            str(config),
            "--output-dir",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert executed.returncode == 0, executed.stdout[-4000:] + executed.stderr[-4000:]


@pytest.fixture(scope="module")
def archived_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retention_archive")
    original, base, _ = tiny_source(tmp)
    original["protocol"]["arms"] = original["protocol"]["arms"][:1]
    original["protocol"]["checkpoints"] = [0, 4, 12]
    snapshot = tmp / "base" / base.revision
    snapshot.mkdir(parents=True)
    base.model.save_pretrained(snapshot)
    base.tokenizer.save_pretrained(snapshot)
    original["inputs"]["base"] = pin_bundle(snapshot)
    original["base"] = {
        "repo_id": base.model_id,
        "revision": base.revision,
        "local_path": str(snapshot),
        "cache_dir": str(tmp / "cache"),
    }
    probes = {"validation": [], "indices": []}
    for task, row in zip(OLD_TASKS, follow.validation_rows(original)[:4], strict=True):
        probes["validation"].append(
            {
                "task": task,
                "prompt": row.prompt,
                "choices": list(row.choices),
                "gold_idx": row.gold_idx,
            }
        )
        probes["indices"].append(
            {
                "task": task,
                "prompt_sha256": hashlib.sha256(row.prompt.encode()).hexdigest(),
            }
        )
    write_json(tmp / "probes.json", probes)
    original["inputs"]["retention_probes"] = {
        "path": str(tmp / "probes.json"),
        **file_pin(tmp / "probes.json"),
    }
    original["protocol"]["retention_indices_sha256"] = digest(probes["indices"])
    original["protocol_sha256"] = digest(original["protocol"])
    write_json(tmp / "original.json", original)
    source = tmp / "completed-source"
    command("follow_through.py", tmp / "original.json", source)
    prior = json.loads((source / "result.json").read_text())
    last = prior["arms"]["control"]["checkpoints"]["12"]
    last.pop("released_tasks")
    write_json(source / "result.json", prior)
    source_hash = final.archived_source_digest(ROOT)
    task = {
        "id": source.name,
        "source_sha256": source_hash,
        "code_dir": str(ROOT),
        "entrypoint": "follow_through.py",
        "config": original,
    }
    execution = {
        "task_id": source.name,
        "status": "completed",
        "exit_code": 0,
        "timed_out": False,
        "task": task,
        "source_sha256": source_hash,
        "task_sha256": final.supervisor_digest(task),
        "config_sha256": final.supervisor_digest(original),
    }
    for name, value in (("task.json", task), ("execution.json", execution)):
        (source / name).write_text(json.dumps(value, indent=2) + "\n")
    config = {
        "kind": "portal_final_retention",
        "source_run": {
            "path": str(source),
            "task_id": source.name,
            "source_sha256": source_hash,
            "config_sha256": digest(original),
        },
        "selection": final.SELECTION,
        "expected_final_step": 12,
        "probe_counts": dict.fromkeys(OLD_TASKS, 1),
    }
    return config, source


def test_completed_archive_final_retention_new_pid_and_immutable_source(
    archived_run, tmp_path
):
    config, source = archived_run
    before = {str(p): file_pin(p) for p in source.rglob("*") if p.is_file()}
    path = tmp_path / "config.json"
    write_json(path, config)
    command("final_retention.py", path, tmp_path / "preflight", "--prepare-only")
    assert not (tmp_path / "preflight/result.json").exists()
    command("final_retention.py", path, tmp_path / "evaluation")
    result = json.loads((tmp_path / "evaluation/result.json").read_text())
    assert result["evaluation_pid"] != result["preparation_pid"]
    assert result["evaluation_pid"] != result["source_training_pid"]
    assert result["evaluation_pid"] != result["source_evaluation_pid"]
    arm = result["arms"]["control"]
    assert arm["retention_steps"] == {"before": 0, "after": 12}
    assert set(arm["checkpoints"]) == {"0", "12"}
    assert all(c["reload_predictions_exact"] for c in arm["checkpoints"].values())
    assert all(
        set(c)
        == {"released_tasks", "validation", "adapter", "reload_predictions_exact"}
        for c in arm["checkpoints"].values()
    )
    for task, score in arm["retention"].items():
        assert (
            score["after"]
            == arm["checkpoints"]["12"]["released_tasks"]["metrics"][task]["accuracy"]
        )
    assert before == {str(p): file_pin(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="OUTPUT_ALREADY_USED"):
        final.prepare(config, tmp_path / "preflight")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("expected_final_step", 4, "INCOMPLETE_BUDGET"),
        ("selection", "best_validation", "UNKNOWN_CONTRACT"),
        ("probe_counts", {"boolq": 32}, "PROBE_COUNTS_CHANGED"),
    ],
)
def test_source_prerequisites_fail_before_base_load(
    archived_run, monkeypatch, field, value, error
):
    config, _ = archived_run
    changed = copy.deepcopy(config)
    changed[field] = value
    monkeypatch.setattr(
        final, "load_base", lambda *a, **k: pytest.fail("unexpected model load")
    )
    with pytest.raises(ValueError, match=error):
        final.inspect_source(changed)


def test_rejects_source_overlap_and_wrong_provenance(archived_run):
    config, source = archived_run
    for output in (source, source / "repair", source.parent):
        with pytest.raises(ValueError, match="SEPARATE_FROM_SOURCE"):
            final.prepare(config, output)
    for key in ("source_sha256", "config_sha256"):
        changed = copy.deepcopy(config)
        changed["source_run"][key] = "0" * 64
        with pytest.raises(ValueError, match="SOURCE_EXECUTION_RECEIPT_MISMATCH"):
            final.inspect_source(changed)


def test_rejects_receipt_changes_and_same_pid_before_evaluation(archived_run, tmp_path):
    config, _ = archived_run
    output = tmp_path / "prepared"
    final.prepare(config, output)
    with pytest.raises(ValueError, match="NEW_EVALUATION_PID"):
        final.evaluate_saved(config, output, os.getpid())
    value = json.loads((output / "source_verification.json").read_text())
    value["endpoints"]["control"] = ["0", "4"]
    write_json(output / "source_verification.json", value)
    preparation = json.loads((output / "preparation.json").read_text())
    preparation["pid"] = -1
    write_json(output / "preparation.json", preparation)
    with pytest.raises(ValueError, match="FILE_.*MISMATCH"):
        final.evaluate_saved(config, output, -1)


@pytest.mark.parametrize("failure", ["base", "validation"])
def test_reloaded_base_and_validation_gate_precede_retention(
    archived_run, tmp_path, monkeypatch, failure
):
    config, _ = archived_run
    output = tmp_path / "prepared"
    final.prepare(config, output)
    preparation = json.loads((output / "preparation.json").read_text())
    preparation["pid"] = -1
    write_json(output / "preparation.json", preparation)
    load = final.load_base
    measure = follow.measure
    retained_adapters = []

    def load_changed(*args, **kwargs):
        base = load(*args, **kwargs)
        parameter = next(base.model.parameters())
        parameter.detach().view(-1)[0] += 1
        return base

    def changed_validation(base, adapter, rows, arm, original):
        if adapter is not None and rows[0].task in OLD_TASKS:
            retained_adapters.append(arm["name"])
        value = measure(base, adapter, rows, arm, original)
        if adapter is not None and rows[0].task not in OLD_TASKS:
            value["metrics"][rows[0].task]["accuracy"] += 1
        return value

    if failure == "base":
        monkeypatch.setattr(final, "load_base", load_changed)
        monkeypatch.setattr(
            follow,
            "measure",
            lambda *a, **k: pytest.fail("evaluation before base verification"),
        )
        error = "RELOADED_BASE_CHANGED"
    else:
        monkeypatch.setattr(follow, "measure", changed_validation)
        error = "RELOAD_PREDICTIONS_DIFFER"
    with pytest.raises(ValueError, match=error):
        final.evaluate_saved(config, output, -1)
    assert not retained_adapters
    assert not (output / "result.json").exists()


def test_ready_configs_pin_archived_task_identity():
    manifests = ROOT.parents[1] / "outputs/portfolio/manifests"
    for path in (ROOT / "configs/final_retention").glob("*.json"):
        config = json.loads(path.read_text())
        task = json.loads(
            (manifests / (config["source_run"]["task_id"] + ".json")).read_text()
        )
        assert config["source_run"]["config_sha256"] == digest(task["config"])
        assert config["source_run"]["source_sha256"] == task["source_sha256"]
        assert (
            config["expected_final_step"]
            == task["config"]["protocol"]["checkpoints"][-1]
        )
        assert sum(config["probe_counts"].values()) == 128
