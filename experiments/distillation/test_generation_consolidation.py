import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional
from transformers import Qwen3Config, Qwen3ForCausalLM

import generation_consolidation as lane
from choice_consolidation import base_tensors, tensor_hash
from choice_contract import ROOT, TASKS, digest, file_hash, load_corpus, write_json
from generation_teacher import generation_gate
from generation_teacher import source_hashes as teacher_source_hashes
from test_choice_consolidation import perfect_panels, sample_row, save_qualification
from test_generation_teacher import GenerativeTokenizer, generated_record, panels_for


@pytest.fixture
def design():
    return json.loads((ROOT / "configs/generation_consolidation_protocol.json").read_text())


def cached_record(row, response=None):
    tokenizer = GenerativeTokenizer()
    if response is None:
        response = tokenizer(row["choices"][row["gold_idx"]], add_special_tokens=False).input_ids + [2]
    body_tokens = response[:-1] if response[-1] == 2 else response
    body = tokenizer.decode(body_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    record = generated_record(row, body, eos=response[-1] == 2)
    record["generation"].update(
        prompt_token_ids=tokenizer(row["prompt"], add_special_tokens=True).input_ids,
        token_ids=response,
        raw_text=tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False),
    )
    return record


def teacher_proof(role="whole_answer_trained"):
    return {
        "teacher_role": role, "qualification_dir": "/teacher-artifacts-not-available-in-evaluator",
        "qualification_sha256": "unit-test-qualified-receipt", "eos_token_id": 2,
        "special_token_ids": [0, 1, 2], "teacher_tensor_sha256": "distinct-from-learner",
    }


def save_generated_receipt(root, design, teacher_design, choice_protocol, corpus, audit, status, eos):
    panels = panels_for(corpus, eos=eos)
    write_json(root / "qualification_panels.json", panels)
    gate = generation_gate(teacher_design, corpus, panels, 2, [0, 1, 2])
    receipt = {
        "status": status, "design_sha256": digest(teacher_design),
        "source_sha256": teacher_source_hashes(), "dataset": audit,
        "learner_updates": 0, "panels_sha256": file_hash(root / "qualification_panels.json"),
        "eos_token_id": 2, "special_token_ids": [0, 1, 2], "gate": gate,
        "training_pid": None, "pid": 7,
        "teacher_tensor_sha256": choice_protocol["teacher"]["tensor_sha256"],
        "checkpoint": choice_protocol["teacher"],
    }
    write_json(root / "qualification.json", receipt)
    return receipt


def test_archived_dependencies_and_explicit_teacher_selection(design):
    teacher_design, choice = lane.validate_design(design)
    assert teacher_design["generation"]["max_new_tokens"] == 16
    assert choice["learner"]["rank"] == 8
    cfg = json.loads((ROOT / "configs/generation_consolidation_run.json").read_text())
    assert cfg["stage"] == "run"
    assert cfg["teacher_role"] == "whole_answer_trained"
    assert cfg["protocol_sha256"] == digest(design)
    changed = copy.deepcopy(design)
    changed["qualified_teacher_source_dependencies"]["generation_teacher.py"] = "changed"
    with pytest.raises(ValueError, match="ARCHIVED_QUALIFICATION_CODE_CHANGED"):
        lane.validate_design(changed)


def test_teacher_targets_are_verbatim_including_wrong_state_and_missing_eos():
    tokenizer = GenerativeTokenizer()
    row = {**sample_row(), "prompt": "Output:"}
    wrong = tokenizer(" 7 7 7", add_special_tokens=False).input_ids
    records = [cached_record(row, wrong)]
    cache = lane.cache_teacher_responses([row], records, teacher_proof())
    teacher = lane.target_sequences("teacher_output_sft", tokenizer, [row], cache)
    oracle = lane.target_sequences("oracle_sft", tokenizer, [row], cache)
    assert teacher == [wrong]
    assert teacher[0][-1] != tokenizer.eos_token_id
    assert oracle != teacher and oracle[0][-1] == tokenizer.eos_token_id
    assert "gold" not in cache["rows"][0]
    assert "correct" not in cache["rows"][0]
    assert cache["rows"][0]["original_generation_sha256"] == digest(records[0]["generation"])
    assert lane.validate_cache_tokens(tokenizer, [row], cache, 256)


def test_cache_rejects_reordered_rows_and_changed_prompt_tokenization():
    tokenizer = GenerativeTokenizer()
    rows = [{**sample_row(), "id": str(i), "prompt": "Output:"} for i in range(2)]
    records = [cached_record(row) for row in rows]
    with pytest.raises(ValueError, match="CACHE_ROW_BINDING"):
        lane.cache_teacher_responses(rows, records[::-1], teacher_proof())
    cache = lane.cache_teacher_responses(rows, records, teacher_proof())
    cache["rows"][0]["prompt_token_ids"] = [1, 3]
    with pytest.raises(ValueError, match="CACHED_TOKEN_BOUNDARY"):
        lane.validate_cache_tokens(tokenizer, rows, cache, 256)


def test_cache_rejects_changed_decode_without_rewriting_it():
    tokenizer = GenerativeTokenizer()
    row = {**sample_row(), "prompt": "Output:"}
    cache = lane.cache_teacher_responses([row], [cached_record(row)], teacher_proof())
    cache["rows"][0]["body_text"] = "1 2 3"
    with pytest.raises(ValueError, match="CACHED_DECODE_CHANGED"):
        lane.validate_cache_tokens(tokenizer, [row], cache, 256)


@pytest.mark.parametrize("status,eos,expected", [
    ("rejected", False, "TEACHER_NOT_QUALIFIED"),
    ("qualified", False, "TEACHER_GATE_FAILED"),
])
def test_failed_teacher_precedes_learner_load(design, tmp_path, monkeypatch, status, eos, expected):
    teacher_design, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    design["choice_qualification_dir"] = str(tmp_path / "choice")
    design["teacher_sources"]["diagnosed_source"]["qualification_dir"] = str(tmp_path / "generated")
    save_qualification(tmp_path / "choice", choice, corpus, audit, perfect_panels(corpus))
    save_generated_receipt(tmp_path / "generated", design, teacher_design, choice, corpus, audit, status, eos)
    calls = []
    monkeypatch.setattr(lane, "load_base", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match=expected):
        lane.train(design, {"teacher_role": "diagnosed_source"}, teacher_design, choice, corpus, audit, tmp_path / "learner")
    assert not calls
    assert not (tmp_path / "learner").exists()


def test_source_receipt_is_bound_to_exact_original_checkpoint(design, tmp_path):
    teacher_design, choice = lane.validate_design(design)
    choice = copy.deepcopy(choice)
    corpus, audit = load_corpus(choice)
    folder = tmp_path / "teacher"
    folder.mkdir()
    files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (folder / name).write_bytes(b"unit-test-file-identity")
        files[name] = file_hash(folder / name)
    choice["teacher"] = {"path": str(folder), "files": files, "tensor_sha256": "test-source-identity"}
    design["choice_qualification_dir"] = str(tmp_path / "choice")
    design["teacher_sources"]["diagnosed_source"]["qualification_dir"] = str(tmp_path / "generated")
    save_qualification(tmp_path / "choice", choice, corpus, audit, perfect_panels(corpus))
    saved = save_generated_receipt(tmp_path / "generated", design, teacher_design, choice, corpus, audit, "qualified", True)
    proof, records = lane.verify_teacher_receipt(design, "diagnosed_source", teacher_design, choice, corpus, audit)
    assert len(records) == 384
    assert proof["teacher_tensor_sha256"] == "test-source-identity"
    saved["training_pid"] = 9
    (tmp_path / "generated/qualification.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="WRONG_DIAGNOSED_SOURCE"):
        lane.verify_teacher_receipt(design, "diagnosed_source", teacher_design, choice, corpus, audit)


def test_response_loss_is_full_vocabulary_and_masks_prompt():
    torch.manual_seed(19)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )).eval()
    prompt, response = [1, 8, 7, 6], [5, 4, 2]
    ids = torch.tensor([prompt + response])
    expected = functional.cross_entropy(
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, len(prompt) - 1:-1],
        torch.tensor(response),
    )
    loss = lane.response_nll(model, prompt, response)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_trained_teacher_selection_and_exposure_provenance(design, tmp_path):
    teacher_design, choice = lane.validate_design(design)
    corpus, audit = load_corpus(choice)
    folder = tmp_path / "trained_teacher"
    folder.mkdir()
    diagnosis = tmp_path / "diagnosis"
    saved = save_generated_receipt(diagnosis, design, teacher_design, choice, corpus, audit, "rejected", False)
    assert not saved["gate"]["passed"]
    probes = lane.training_probes(corpus, teacher_design["training"]["train_probes_per_task"])
    checkpoints = {}
    for step in (32, 128, 384):
        checkpoint = folder / f"checkpoint{step}"
        adapter = checkpoint / "teacher"
        adapter.mkdir(parents=True)
        files = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (adapter / name).write_bytes(f"unit-test-checkpoint-{step}-{name}".encode())
            files[name] = file_hash(adapter / name)
        (checkpoint / "optimizer.pt").write_bytes(b"unit-test-optimizer")
        records = [cached_record(row) for row in probes]
        write_json(checkpoint / "train_probes.json", records)
        checkpoints[str(step)] = {
            "path": str(adapter), "tensor_sha256": f"checkpoint-{step}", "files": files,
            "optimizer_sha256": file_hash(checkpoint / "optimizer.pt"),
            "train_probes_sha256": file_hash(checkpoint / "train_probes.json"),
            "train_probe_summary": lane.summarize(records),
        }
    schedule = lane.training_schedule(choice, corpus["train"])
    ledger = [{"step": i + 1, "ids": [corpus["train"][j]["id"] for j in batch]} for i, batch in enumerate(schedule)]
    training = {
        "status": "trained_qualification_pending", "pid": 101,
        "design_sha256": digest(teacher_design), "source_sha256": teacher_source_hashes(), "dataset": audit,
        "source_teacher_sha256": choice["teacher"]["tensor_sha256"],
        "updates": 384, "example_exposures": 1536,
        "validation_predictions_observed_during_training": False, "learner_updates": 0,
        "trainable_parameters": choice["learner"]["expected_trainable_parameters"],
        "ledger": ledger, "diagnosis_sha256": file_hash(diagnosis / "qualification.json"),
        "checkpoints": checkpoints, "selected_checkpoint": 32,
        "selection": teacher_design["training"]["selection"],
        "eos_token_id": 2, "special_token_ids": [0, 1, 2],
    }
    write_json(folder / "training.json", training)
    qualified = {
        "training_pid": 101, "pid": 202, "checkpoint": checkpoints["32"],
        "teacher_tensor_sha256": "checkpoint-32", "eos_token_id": 2, "special_token_ids": [0, 1, 2],
    }
    spec = {"training_dir": str(folder), "diagnosis_dir": str(diagnosis)}
    proof = lane.verify_trained_teacher_history(spec, qualified, teacher_design, choice, corpus, audit)
    assert proof["selected_checkpoint"] == 32
    qualified["checkpoint"] = checkpoints["384"]
    qualified["teacher_tensor_sha256"] = "checkpoint-384"
    with pytest.raises(ValueError, match="TEACHER_SELECTION_CHANGED"):
        lane.verify_trained_teacher_history(spec, qualified, teacher_design, choice, corpus, audit)
    qualified["checkpoint"] = checkpoints["32"]
    qualified["teacher_tensor_sha256"] = "checkpoint-32"
    training["ledger"][0]["ids"].reverse()
    (folder / "training.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="TEACHER_TRAINING_PROVENANCE"):
        lane.verify_trained_teacher_history(spec, qualified, teacher_design, choice, corpus, audit)


def test_cpu_persistent_learning_and_real_fresh_process_evaluation(design, tmp_path, monkeypatch):
    teacher_design, choice = lane.validate_design(design)
    full_corpus, _ = load_corpus(choice)
    corpus = {
        split: [next(row for row in full_corpus[split] if row["task"] == task) for task in TASKS]
        for split in ("train", "test", "unused288")
    }
    audit = {"rows_sha256": {name: digest(rows) for name, rows in corpus.items()}, "cpu_test_only": True}
    recipe = design["training"]
    recipe.update(updates_per_arm=2, example_exposures_per_arm=6, epochs=2, examples_per_update=3, checkpoint_updates=[1, 2], probe_rows_per_task=1)
    choice["training"].update(updates_per_arm=2, example_exposures_per_arm=6, epochs=2, examples_per_update=3)
    choice["learner"]["expected_trainable_parameters"] = 1792
    torch.manual_seed(17)
    model_config = Qwen3Config(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, use_cache=False,
    )
    original = Qwen3ForCausalLM(model_config).requires_grad_(False).eval()
    weights = copy.deepcopy(original.state_dict())
    choice["source_base_tensor_sha256"] = tensor_hash(base_tensors(original))
    tokenizer = GenerativeTokenizer()

    def load_tiny(protocol, output):
        base = Qwen3ForCausalLM(model_config).requires_grad_(False).eval()
        base.load_state_dict(weights)
        return base, tokenizer

    records = [cached_record(row) for row in corpus["train"]]
    records[0] = cached_record(corpus["train"][0], tokenizer(" 7 7 7", add_special_tokens=False).input_ids)
    monkeypatch.setattr(lane, "load_base", load_tiny)
    monkeypatch.setattr(lane, "verify_teacher_receipt", lambda *args: (teacher_proof(), records))
    output = tmp_path / "run"
    output.mkdir()
    training = lane.train(design, {"teacher_role": "whole_answer_trained"}, teacher_design, choice, corpus, audit, output)
    assert training["pid"] == os.getpid()
    assert training["teacher_targets_equal_oracle_count"] == 2
    assert not training["teacher_targets_all_equal_oracle"]
    assert training["arms"]["teacher_output_sft"]["loss_token_exposures"] == 40
    assert training["arms"]["oracle_sft"]["loss_token_exposures"] == 42
    assert all(arm["updates"] == 2 and arm["example_exposures"] == 6 for arm in training["arms"].values())
    hashes = [arm["checkpoints"]["2"]["adapter"]["tensor_sha256"] for arm in training["arms"].values()]
    assert hashes[0] != hashes[1]
    assert all(value != training["initial"]["tensor_sha256"] for value in hashes)
    with pytest.raises(ValueError, match="NOT_FRESH_PROCESS"):
        lane.verify_training(output, design, teacher_design, choice, corpus, audit)
    torch.save(weights, tmp_path / "base.pt")
    write_json(tmp_path / "fixture.json", {
        "design": design, "teacher_design": teacher_design, "choice": choice,
        "corpus": corpus, "audit": audit, "model_config": model_config.to_dict(),
    })
    script = tmp_path / "evaluate_tiny.py"
    script.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        "import torch\nfrom transformers import Qwen3Config, Qwen3ForCausalLM\n"
        "import generation_consolidation as lane\n"
        "from test_generation_teacher import GenerativeTokenizer\n"
        "root=Path(sys.argv[1])\nfixture=json.loads((root/'fixture.json').read_text())\n"
        "weights=torch.load(root/'base.pt',weights_only=True)\n"
        "def load_tiny(protocol,output):\n"
        "    model=Qwen3ForCausalLM(Qwen3Config(**fixture['model_config'])).requires_grad_(False).eval()\n"
        "    model.load_state_dict(weights)\n"
        "    return model,GenerativeTokenizer()\n"
        "def forbidden(*args,**kwargs):\n"
        "    raise AssertionError('Teacher prerequisite/file access forbidden during evaluation')\n"
        "lane.load_base=load_tiny\nlane.verify_teacher_receipt=forbidden\n"
        "(root/'evaluation').mkdir()\n"
        "lane.evaluate_learners(fixture['design'],{'training_dir':str(root/'run')},fixture['teacher_design'],fixture['choice'],fixture['corpus'],fixture['audit'],root/'evaluation')\n"
    )
    child = subprocess.run(
        ["uv", "run", "--no-project", "--python", sys.executable, "python", str(script), str(tmp_path)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads((tmp_path / "evaluation/result.json").read_text())
    assert result["pid"] != training["pid"]
    assert result["training_pid"] == training["pid"]
    assert result["learner_reload_generation_parity"]
    assert not result["teacher_model_loaded"]
    assert not result["teacher_artifact_files_read_during_evaluation"]
    assert set(result["comparisons"]) == {"test", "unused288"}


def test_run_driver_launches_evaluation_through_uv_in_new_process(tmp_path, monkeypatch):
    write_json(tmp_path / "training.json", {"pid": os.getpid()})
    calls = []

    def child_run(command, check, cwd):
        calls.append(command)
        assert command[:4] == ["uv", "run", "--no-project", "--python"]
        config = json.loads((tmp_path / "evaluation_config.json").read_text())
        assert config["stage"] == "evaluate"
        assert config["training_dir"] == str(tmp_path)
        write_json(tmp_path / "evaluation/result.json", {
            "status": "completed", "pid": os.getpid() + 1, "training_pid": os.getpid(),
            "comparisons": {}, "teacher_targets_all_equal_oracle": False, "claim_boundary": "test",
        })

    monkeypatch.setattr(lane.subprocess, "run", child_run)
    lane.evaluate_in_new_process(tmp_path, {"protocol": "configs/generation_consolidation_protocol.json", "protocol_sha256": "test"})
    assert len(calls) == 1
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "completed"


def test_run_config_cpu_validation_preserves_archived_qualifications(tmp_path, monkeypatch):
    before = teacher_source_hashes()
    config = json.loads((ROOT / "configs/generation_consolidation_run.json").read_text())
    write_json(tmp_path / "config.json", config)
    monkeypatch.setattr(sys, "argv", ["generation_consolidation.py", "--config", str(tmp_path / "config.json"), "--output-dir", str(tmp_path), "--validate-only"])
    lane.main()
    assert not json.loads((tmp_path / "validation.json").read_text())["gpu_qualified"]
    assert teacher_source_hashes() == before
