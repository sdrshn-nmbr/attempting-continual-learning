import copy
import json
import os
import subprocess
import sys

import coverage_learner as lane
import pytest
import torch
from choice_consolidation import adapter_state, base_tensors, make_learner, tensor_hash
from choice_contract import ROOT, TASKS, digest, file_hash, load_corpus, write_json
from coverage_teacher import coverage_gate
from generation_teacher import generation_gate
from test_choice_consolidation import sample_row
from test_generation_consolidation import cached_record
from test_generation_teacher import GenerativeTokenizer, panels_for
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def design():
    return json.loads((ROOT / "configs/coverage_learner_protocol.json").read_text())


def test_exact_recipe_and_frozen_dependencies(design):
    coverage, previous, _, choice = lane.validate_design(design)
    assert design["training"] == previous["training"]
    assert design["training"]["updates_per_arm"] == 384
    assert design["training"]["initialization_seed"] == 91230417
    assert design["training"]["order_seed"] == 91230431
    assert choice["learner"]["rank"] == 8
    assert "coverage_learner.py" not in coverage["previous_source_sha256"]
    for handoff_name in ("coverage_teacher_handoff.json", "device4_curriculum_handoff.json"):
        frozen = json.loads((ROOT / handoff_name).read_text())["staging_files_sha256"]
        assert all(file_hash(ROOT / name) == checksum for name, checksum in frozen.items())
    for stage in ("run", "train", "evaluate"):
        config = json.loads((ROOT / f"configs/coverage_learner_{stage}.json").read_text())
        assert config["stage"] == stage and config["protocol_sha256"] == digest(design)
    changed = copy.deepcopy(design)
    changed["training"]["updates_per_arm"] = 32
    with pytest.raises(ValueError, match="MATCHED_RECIPE_CHANGED"):
        lane.validate_design(changed)
    changed = copy.deepcopy(design)
    changed["coverage_source_sha256"]["coverage_teacher.py"] = "stale"
    with pytest.raises(ValueError, match="QUALIFICATION_SOURCE_CHANGED"):
        lane.validate_design(changed)


def test_real_original_run_initialization_and_schedule_are_bound(design):
    coverage, previous, _, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    folder = ROOT.parent.parent / "outputs/portfolio/followthrough-20260912/runs/consolidation-generated303-20260912"
    design["reference_experiment"]["training_dir"] = str(folder)
    receipt = lane.reference_training(design, coverage, previous, choice, corpus, audit)
    assert receipt["teacher_targets_equal_oracle_count"] == 376
    assert receipt["initial"]["tensor_sha256"] == design["reference_experiment"]["initial_tensor_sha256"]
    design["reference_experiment"]["initial_tensor_sha256"] = "another-initialization"
    with pytest.raises(ValueError, match="ORIGINAL_INITIALIZATION_OR_SCHEDULE_MISMATCH"):
        lane.reference_training(design, coverage, previous, choice, corpus, audit)


def test_failed_coverage_qualification_stops_before_any_learner_loading(design, tmp_path, monkeypatch):
    coverage, previous, teacher, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    calls = []

    def rejected(*args):
        raise ValueError("NO_FULLY_COVERED_TEACHER")

    monkeypatch.setattr(lane, "verify_qualified", rejected)
    monkeypatch.setattr(lane, "load_base", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="NO_FULLY_COVERED_TEACHER"):
        lane.train(design, coverage, previous, teacher, choice, corpus, audit, tmp_path)
    assert not calls and not list(tmp_path.iterdir())


def make_fixture(design, tmp_path):
    coverage, previous, teacher, choice = lane.validate_design(design)
    corpus = {
        split: [{
            **sample_row(), "id": f"cpu-{split}-{task}-{digit}", "task": task,
            "group": f"{digit} {digit} {digit}", "prompt": "Output:",
            "choices": [f" {digit} {digit} {digit}", " 0 1 2", " 3 4 5", " 6 7 0"], "gold_idx": 0,
        } for task in TASKS for digit in (range(8) if split == "train" else [1])]
        for split in ("train", "validation", "test", "unused288")
    }
    audit = {"cpu_fixture_only": True, "rows_sha256": {split: digest(rows) for split, rows in corpus.items()}}
    recipe = design["training"]
    recipe.update(updates_per_arm=2, example_exposures_per_arm=24, epochs=1, examples_per_update=12, checkpoint_updates=[1, 2], probe_rows_per_task=1)
    previous["training"] = copy.deepcopy(recipe)
    choice["training"].update(updates_per_arm=2, example_exposures_per_arm=24, epochs=1, examples_per_update=12)
    choice["learner"]["expected_trainable_parameters"] = 1792
    torch.manual_seed(17)
    model_config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               head_dim=8, max_position_embeddings=256, use_cache=False)
    original = Qwen3ForCausalLM(model_config).requires_grad_(False).eval()
    weights = copy.deepcopy(original.state_dict())
    choice["source_base_tensor_sha256"] = tensor_hash(base_tensors(original))
    initial = make_learner(original, lane.learner_protocol(choice, design))
    initial_hash = tensor_hash(adapter_state(initial))
    reference = tmp_path / "reference"
    schedule = lane.training_schedule(lane.learner_protocol(choice, design), corpus["train"])
    write_json(reference / "schedule.json", [[corpus["train"][i]["id"] for i in batch] for batch in schedule])
    write_json(reference / "training.json", {
        "status": "trained_evaluation_pending", "dataset": audit, "design_sha256": digest(previous),
        "source_sha256": coverage["previous_source_sha256"], "initial": {"tensor_sha256": initial_hash},
        "files_sha256": {"schedule.json": file_hash(reference / "schedule.json")},
        "base_tensor_sha256": choice["source_base_tensor_sha256"], "trainable_parameters": 1792,
        "resident_learner_rank": 8, "teacher_model_loaded": False, "heldout_predictions_observed": False,
        "arms": {method: {"updates": 2, "example_exposures": 24} for method in lane.METHODS},
    })
    design["reference_experiment"] = {
        "training_dir": str(reference), "training_sha256": file_hash(reference / "training.json"),
        "schedule_sha256": file_hash(reference / "schedule.json"), "initial_tensor_sha256": initial_hash,
    }
    folder = tmp_path / "teacher"
    design["qualified_teacher_dir"] = str(folder)
    panels = panels_for(corpus)
    for split in ("train", "validation"):
        panels[split]["generated"] = [cached_record(row) for row in corpus[split]]
    write_json(folder / "qualification_panels.json", panels)
    train_gate = coverage_gate(corpus["train"], panels["train"]["generated"], teacher, 2, [0, 1, 2])
    assert train_gate["passed"] and train_gate["passed_cells"] == 72
    qualified = {
        "status": "qualified", "design_sha256": digest(coverage), "source_sha256": design["coverage_source_sha256"],
        "dataset": audit, "pid": 222, "screen_pid": 111, "teacher_optimizer_updates": 0, "learner_updates": 0,
        "test_predictions": 0, "selected_checkpoint": 128, "checkpoint": {"tensor_sha256": "cpu-teacher-not-learner"},
        "teacher_tensor_sha256": "cpu-teacher-not-learner", "panels_sha256": file_hash(folder / "qualification_panels.json"),
        "eos_token_id": 2, "special_token_ids": [0, 1, 2], "screen_sha256": "cpu-selection-proof",
        "gate": generation_gate(teacher, corpus, panels, 2, [0, 1, 2]), "train_coverage": train_gate,
        "validation_coverage_diagnostic_only": coverage_gate(corpus["validation"], panels["validation"]["generated"], teacher, 2, [0, 1, 2]),
    }
    write_json(folder / "qualification.json", qualified)
    return {
        "design": design, "coverage": coverage, "previous": previous, "teacher": teacher, "choice": choice,
        "corpus": corpus, "audit": audit, "qualified": qualified, "records": panels["train"]["generated"],
        "model_config": model_config.to_dict(), "weights": weights,
    }


def test_cpu_real_persistent_updates_and_teacher_absent_fresh_pid_evaluation(design, tmp_path, monkeypatch):
    fixture = make_fixture(design, tmp_path)

    def load_tiny(protocol, output):
        model = Qwen3ForCausalLM(Qwen3Config(**fixture["model_config"])).requires_grad_(False).eval()
        model.load_state_dict(fixture["weights"])
        return model, GenerativeTokenizer()

    monkeypatch.setattr(lane, "load_base", load_tiny)
    monkeypatch.setattr(lane, "verify_qualified", lambda *args: (fixture["qualified"], fixture["records"]))
    output = tmp_path / "run"
    output.mkdir()
    arguments = [fixture[name] for name in ("design", "coverage", "previous", "teacher", "choice", "corpus", "audit")]
    training = lane.train(*arguments, output)
    assert training["teacher_targets_all_equal_oracle"] and training["teacher_targets_equal_oracle_count"] == 24
    assert training["initial_matches_original_experiment"] and not training["teacher_model_loaded"]
    assert all(arm["updates"] == 2 and arm["example_exposures"] == 24 and arm["loss_token_exposures"] == 168 for arm in training["arms"].values())
    final_hashes = [arm["checkpoints"]["2"]["adapter"]["tensor_sha256"] for arm in training["arms"].values()]
    assert final_hashes[0] == final_hashes[1] != training["initial"]["tensor_sha256"]
    with pytest.raises(ValueError, match="NOT_FRESH_PROCESS"):
        lane.verify_training(output, *arguments)
    torch.save(fixture.pop("weights"), tmp_path / "base.pt")
    write_json(tmp_path / "fixture.json", fixture)
    (tmp_path / "teacher").rename(tmp_path / "teacher-unavailable")
    (tmp_path / "reference").rename(tmp_path / "reference-unavailable")
    script = tmp_path / "evaluate_tiny.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import torch\nfrom transformers import Qwen3Config, Qwen3ForCausalLM\n"
        "import coverage_learner as lane\nfrom test_generation_teacher import GenerativeTokenizer\n"
        "root=Path(sys.argv[1])\nfixture=json.loads((root/'fixture.json').read_text())\n"
        "weights=torch.load(root/'base.pt',weights_only=True)\n"
        "def load_tiny(protocol,output):\n"
        "    model=Qwen3ForCausalLM(Qwen3Config(**fixture['model_config'])).requires_grad_(False).eval()\n"
        "    model.load_state_dict(weights)\n    return model,GenerativeTokenizer()\n"
        "def forbidden(*args,**kwargs):\n"
        "    raise AssertionError('Teacher file or qualification access forbidden during evaluation')\n"
        "lane.load_base=load_tiny\nlane.verify_qualified=forbidden\n"
        "(root/'evaluation').mkdir()\n"
        "args=[fixture[key] for key in ('coverage','previous','teacher','choice','corpus','audit')]\n"
        "lane.evaluate_learners(fixture['design'],{'training_dir':str(root/'run')},*args,root/'evaluation')\n"
        "training=json.loads((root/'run/training.json').read_text())\n"
        "training['arms']['teacher_output_sft']['selected_checkpoint']=1\n"
        "(root/'run/training.json').write_text(json.dumps(training))\n"
        "try:\n    lane.verify_training(root/'run',fixture['design'],*args)\n"
        "except ValueError as error:\n    assert 'FINAL_SELECTION_CHANGED' in str(error)\n"
        "else:\n    raise AssertionError('Earlier checkpoint selection was accepted')\n"
    )
    child = subprocess.run(
        ["uv", "run", "--no-project", "--python", sys.executable, "python", str(script), str(tmp_path)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "evaluation/result.json").read_text())
    assert result["pid"] != training["pid"] and result["training_pid"] == training["pid"]
    assert result["learner_reload_generation_parity"] and result["final_arm_tensors_equal"]
    assert result["initial_and_schedule_match_original"]
    assert not result["teacher_model_loaded"] and not result["teacher_artifact_files_read_during_evaluation"]
    assert set(result["comparisons"]) == {"test", "unused288"}
    assert not (tmp_path / "teacher").exists() and not (tmp_path / "reference").exists()


def test_cached_certificate_cannot_rescue_wrong_train_outputs(design, tmp_path, monkeypatch):
    fixture = make_fixture(design, tmp_path)
    folder = tmp_path / "teacher"
    panels = json.loads((folder / "qualification_panels.json").read_text())
    tokenizer = GenerativeTokenizer()
    panels["train"]["generated"][0] = cached_record(fixture["corpus"]["train"][0], tokenizer(" 7 7 7", add_special_tokens=False).input_ids + [2])
    (folder / "qualification_panels.json").write_text(json.dumps(panels))
    fixture["qualified"]["panels_sha256"] = file_hash(folder / "qualification_panels.json")
    (folder / "qualification.json").write_text(json.dumps(fixture["qualified"]))
    monkeypatch.setattr(lane, "verify_qualified", lambda *args: (fixture["qualified"], panels["train"]["generated"]))
    calls = []
    monkeypatch.setattr(lane, "load_base", lambda *args: calls.append(args))
    output = tmp_path / "run"
    output.mkdir()
    arguments = [fixture[name] for name in ("design", "coverage", "previous", "teacher", "choice", "corpus", "audit")]
    with pytest.raises(ValueError, match="LOCAL_TEACHER_GATE_CHANGED"):
        lane.train(*arguments, output)
    assert not calls and not (output / "teacher_train_cache.json").exists()


def test_default_driver_requires_completed_fresh_uv_evaluator(tmp_path, monkeypatch):
    write_json(tmp_path / "training.json", {"pid": os.getpid()})
    calls = []

    def child(command, cwd, check):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-project", "--python"]
        config = json.loads((tmp_path / "evaluation_config.json").read_text())
        assert config["stage"] == "evaluate" and config["training_dir"] == str(tmp_path)
        write_json(tmp_path / "evaluation/result.json", {
            "status": "completed", "pid": os.getpid() + 1, "training_pid": os.getpid(), "comparisons": {},
            "selected_teacher_checkpoint": 128, "teacher_targets_all_equal_oracle": True, "claim_boundary": "CPU fixture",
        })

    monkeypatch.setattr(lane.subprocess, "run", child)
    lane.evaluate_in_new_process(tmp_path, {"protocol": "configs/coverage_learner_protocol.json", "protocol_sha256": "fixture"})
    assert len(calls) == 1 and json.loads((tmp_path / "result.json").read_text())["status"] == "completed"


def test_cpu_validate_only_without_teacher_or_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["coverage_learner.py", "--config", str(ROOT / "configs/coverage_learner_run.json"), "--output-dir", str(tmp_path), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validation.json").read_text())["gpu_qualified"]
