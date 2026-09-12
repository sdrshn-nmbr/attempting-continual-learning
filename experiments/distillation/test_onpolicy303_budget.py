import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

import onpolicy303_budget as lane
from choice_consolidation import (
    adapter_state,
    base_tensors,
    make_learner,
    save_adapter,
    tensor_hash,
)
from choice_contract import ROOT, TASKS, digest, load_corpus, write_json
from generation_consolidation import learner_protocol
from onpolicy303 import load_saved_learner, new_optimizer, row_loss
from onpolicy303_budget_contract import (
    FINAL,
    METHODS,
    START,
    EvaluationReadBan,
    continuation_schedule,
    mapping_from_adapter,
    model_mapping,
    optimizer_options,
    optimizer_tensor_hash,
    require_teacher_absent,
    restore_optimizer,
    schedule_receipt,
    token_accounting,
    validate_design,
    validate_optimizer,
    verify_child_identity,
    verify_reload,
    verify_runtime_versions,
    verify_source_files,
)
from onpolicy303_contract import retention_prediction

LOCAL_RUNS = ROOT.parents[1] / "outputs/portfolio/followthrough-20260912/runs"


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    all_special_ids = (0, 1, 2)

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=([1] if add_special_tokens else []) + [ord(char) + 3 for char in text])

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join({0: "<pad>", 1: "<bos>", 2: "<eos>"}.get(i, chr(max(0, i - 3))) for i in ids)


@pytest.fixture(scope="module")
def sealed():
    design = json.loads((ROOT / "configs/onpolicy303-budget-protocol.json").read_text())
    original, context = validate_design(design)
    corpus, audit = load_corpus(context[-1])
    return design, original, context, corpus, audit


def tiny_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(81)
    config = Qwen3Config(vocab_size=128, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
                        eos_token_id=2, pad_token_id=0, bos_token_id=1, use_cache=False)
    base = Qwen3ForCausalLM(config).float().requires_grad_(False).eval()
    expert = get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                            lora_dropout=0, task_type="CAUSAL_LM"), adapter_name="teacher")
    with torch.no_grad():
        for name, parameter in expert.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.07)
    expert.requires_grad_(False).eval()
    protocol = {"training": {"initialization_seed": 92},
                "learner": {"rank": 8, "alpha": 16, "targets": ["q_proj", "v_proj"], "expected_trainable_parameters": 224}}
    recipe = {"learning_rate": 0.0003, "weight_decay": 0.0, "max_grad_norm": 1.0}
    original = {"training": recipe, "sampling": {"max_new_tokens": 16}}
    tokenizer = TinyTokenizer()
    rows = [{"id": f"tiny-{i}", "task": TASKS[i % 3], "group": f"{i % 8} 0 1", "prompt": "O:",
             "choices": [f" {i % 8} 0 1", " 1 2 3", " 3 2 1", " 7 6 5"], "gold_idx": 0} for i in range(384)]
    prompts = [tokenizer(row["prompt"], True).input_ids for row in rows]
    targets = [tokenizer(row["choices"][0], False).input_ids + [2] for row in rows]
    schedule = [[(4 * step + i) % 384 for i in range(4)] for step in range(FINAL)]
    return base, expert, protocol, original, tokenizer, rows, prompts, targets, schedule


@pytest.fixture(scope="module", params=METHODS)
def resumed_pair(request, tmp_path_factory):
    base, expert, protocol, original, tokenizer, rows, prompts, targets, schedule = tiny_fixture()
    method = request.param
    if method == "cached_teacher_sft":
        expert = None
    teacher_hash = None if expert is None else tensor_hash({**dict(expert.named_parameters()), **dict(expert.named_buffers())})
    student = make_learner(copy.deepcopy(base), protocol)
    optimizer, parameters = new_optimizer(student, original["training"])
    folder = tmp_path_factory.mktemp(method)
    before_hash = tensor_hash(base_tensors(student))
    mapping = model_mapping(student)
    saved_adapter = None
    for step, indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        for index in indices:
            loss, _ = row_loss(student, expert, prompts[index], targets[index], targets[index], tokenizer, method, 16)
            (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == START:
            saved_adapter = {key: value.detach().cpu().clone() for key, value in adapter_state(student).items()}
            torch.save(optimizer.state_dict(), folder / "optimizer384.pt")
    optimizer.zero_grad(set_to_none=True)
    saved_optimizer = torch.load(folder / "optimizer384.pt", map_location="cpu", weights_only=True)
    restored = make_learner(copy.deepcopy(base), protocol)
    set_peft_model_state_dict(restored, saved_adapter, adapter_name="learner")
    loaded, loaded_parameters, proof = restore_optimizer(restored, original["training"], saved_optimizer, mapping,
                                                        optimizer_options(saved_optimizer), optimizer_tensor_hash(saved_optimizer))
    ledger = lane.continue_updates(restored, expert, tokenizer, {"train": rows}, prompts, targets, schedule,
                                  method, original, loaded, loaded_parameters, folder)
    torch.save(loaded.state_dict(), folder / "optimizer405.pt")
    assert tensor_hash(base_tensors(student)) == tensor_hash(base_tensors(restored)) == before_hash
    if expert is not None:
        assert teacher_hash == tensor_hash({**dict(expert.named_parameters()), **dict(expert.named_buffers())})
        assert all(p.grad is None and not p.requires_grad for p in expert.parameters())
    return {"method": method, "student": student, "restored": restored, "optimizer": optimizer,
            "loaded": loaded, "saved": saved_optimizer, "mapping": mapping, "proof": proof,
            "base": base, "protocol": protocol, "original": original, "ledger": ledger, "folder": folder}


def test_uninterrupted405_equals384_plus21_with_saved_adam(resumed_pair):
    pair = resumed_pair
    assert tensor_hash(adapter_state(pair["student"])) == tensor_hash(adapter_state(pair["restored"]))
    assert optimizer_tensor_hash(pair["optimizer"].state_dict()) == optimizer_tensor_hash(pair["loaded"].state_dict())
    final = torch.load(pair["folder"] / "optimizer405.pt", map_location="cpu", weights_only=True)
    proof = validate_optimizer(final, pair["mapping"], FINAL, optimizer_options(pair["saved"]))
    assert proof["clock"] == 405 and pair["proof"]["clock"] == 384
    assert [entry["step"] for entry in pair["ledger"]] == list(range(385, 406))
    assert sum(entry["loss_tokens"] for entry in pair["ledger"]) == 588


@pytest.mark.parametrize("corruption", ["clock", "nan", "shape", "missing", "mapping", "options", "tensor_hash"])
def test_corrupt_optimizer_or_order_fails_before_update(resumed_pair, corruption):
    pair = resumed_pair
    saved, mapping = copy.deepcopy(pair["saved"]), copy.deepcopy(pair["mapping"])
    expected_hash, options = optimizer_tensor_hash(saved), optimizer_options(saved)
    if corruption == "clock":
        saved["state"][0]["step"].fill_(383)
    elif corruption == "nan":
        saved["state"][0]["exp_avg"][0, 0] = float("nan")
    elif corruption == "shape":
        saved["state"][0]["exp_avg"] = saved["state"][0]["exp_avg"][:1]
    elif corruption == "missing":
        del saved["state"][0]["exp_avg_sq"]
    elif corruption == "mapping":
        mapping[0], mapping[2] = mapping[2], mapping[0]
        mapping[0]["optimizer_id"], mapping[2]["optimizer_id"] = 0, 2
    elif corruption == "options":
        saved["param_groups"][0]["lr"] *= 2
    elif corruption == "tensor_hash":
        saved["state"][0]["exp_avg"][0, 0] += 0.0001
    model = make_learner(copy.deepcopy(pair["base"]), pair["protocol"])
    unchanged = tensor_hash(adapter_state(model))
    with pytest.raises(ValueError, match="ONPOLICY303_BUDGET"):
        restore_optimizer(model, pair["original"]["training"], saved, mapping, options, expected_hash)
    assert tensor_hash(adapter_state(model)) == unchanged
    assert all(p.grad is None for p in model.parameters())


def test_source_checkpoint384_exact_files_moments_schedule_tokens(sealed):
    design, original, context, corpus, _ = sealed
    protocol = learner_protocol(context[-1], original)
    for method, descriptor in design["sources"].items():
        root = LOCAL_RUNS / descriptor["task_id"]
        receipt = verify_source_files(root, descriptor)
        state = torch.load(root / method / "checkpoint384/optimizer.pt", map_location="cpu", weights_only=True)
        adapter = load_file(str(root / method / "checkpoint384/learner/adapter_model.safetensors"))
        assert mapping_from_adapter(adapter) == descriptor["parameter_mapping"]
        assert tensor_hash(adapter) == descriptor["checkpoint"]["adapter"]["tensor_sha256"]
        proof = validate_optimizer(state, descriptor["parameter_mapping"], START, descriptor["optimizer_options"])
        assert proof["tensor_sha256"] == descriptor["optimizer_tensor_sha256"]
        assert proof["parameter_states"] == 144
        old_ids = json.loads((root / "schedule.json").read_text())
        schedule = continuation_schedule(protocol, corpus["train"], old_ids)
        assert schedule_receipt(protocol, corpus["train"], old_ids) == design["schedule"]
        cache = json.loads((root / "teacher_train_cache.json").read_text())
        tokens = token_accounting(cache, schedule, receipt["arm"]["loss_token_exposures"])
        assert tokens["total_loss_tokens"] == 11340 and tokens["excess_over_ownprefix"] == 26
        assert tokens["total_row_exposures"] == 1620 and not tokens["exact_token_update_or_flop_match"]


def test_corrupt_schedule_and_changed_cache_are_rejected(sealed):
    design, original, context, corpus, _ = sealed
    descriptor = design["sources"]["cached_teacher_sft"]
    root = LOCAL_RUNS / descriptor["task_id"]
    ids = json.loads((root / "schedule.json").read_text())
    protocol = learner_protocol(context[-1], original)
    schedule = continuation_schedule(protocol, corpus["train"], ids)
    bad_ids = copy.deepcopy(ids)
    bad_ids[17].reverse()
    with pytest.raises(ValueError, match="original384 batch IDs"):
        continuation_schedule(protocol, corpus["train"], bad_ids)
    cache = json.loads((root / "teacher_train_cache.json").read_text())
    cache["rows"][0]["response_token_ids"].pop()
    with pytest.raises(ValueError, match="seven-token teacher cache"):
        token_accounting(cache, schedule, 10752)


def test_kl_teacher_failure_precedes_student_load(sealed, monkeypatch, tmp_path):
    design, original, context, corpus, audit = sealed
    descriptor = design["sources"]["cached_teacher_kl"]
    root = LOCAL_RUNS / descriptor["task_id"]
    receipt = json.loads((root / "training.json").read_text())
    cache = json.loads((root / "teacher_train_cache.json").read_text())
    experiment = copy.deepcopy(design)
    experiment["sources"]["cached_teacher_kl"]["run_dir"] = str(root)
    monkeypatch.setattr(lane, "verify_source_files", lambda *args: receipt)
    monkeypatch.setattr(lane, "verify_training", lambda *args: (receipt, cache))
    monkeypatch.setattr(lane, "execution_identity", lambda *args: receipt["execution"])
    calls = []

    def rejected(*args):
        calls.append("teacher_qualification")
        raise ValueError("teacher prerequisite failed")

    def forbidden_load(*args):
        pytest.fail("student loaded after teacher failure")

    monkeypatch.setattr(lane, "qualify_actual_teacher", rejected)
    monkeypatch.setattr(lane, "restored_student", forbidden_load)
    with pytest.raises(ValueError, match="teacher prerequisite failed"):
        lane.training(experiment, original, context, corpus, audit, {"method": "cached_teacher_kl"}, {}, tmp_path)
    assert calls == ["teacher_qualification"]
    assert not (tmp_path / "ledger.json").exists()


@pytest.mark.parametrize("corrupt", ["native", "generic", "scorer"])
def test_reload_mismatch_rejects_before_updates(corrupt):
    rows = [{"id": f"r{i}", "task": "boolq", "prompt": "p", "choices": [" yes", " no"], "gold_idx": 0} for i in range(128)]
    generic = [retention_prediction(row, [1.0, 0.0]) for row in rows]
    native = [{"id": i, "generation": {"correct": True}} for i in range(24)]
    expected_native, expected_generic = copy.deepcopy(native), copy.deepcopy(generic)
    if corrupt == "native":
        native[0]["generation"]["correct"] = False
    elif corrupt == "generic":
        generic[0]["scores"][0] += 0.01
    else:
        expected_generic[0]["correct"] = False
    with pytest.raises(ValueError, match="reload parity|scorer binding"):
        verify_reload(native, expected_native, generic, expected_generic, rows)


def test_torch_module_and_distribution_versions_are_distinct_exact_fields(sealed):
    expected = sealed[0]["runtime_versions"]
    assert expected["torch_module_version"] == "2.10.0+rocm7.2.4.git3d3aa833"
    assert expected["packages"]["torch"] == "2.10.0+rocm7.2.4.lw.git3d3aa833"
    assert verify_runtime_versions(expected, copy.deepcopy(expected)) == expected
    actual = copy.deepcopy(expected)
    actual["packages"]["torch"] = actual["torch_module_version"]
    with pytest.raises(ValueError, match="runtime versions changed"):
        verify_runtime_versions(expected, actual)


def test_native384_parity_failure_does_not_start_continuation(sealed, monkeypatch, tmp_path):
    design, original, context, corpus, audit = sealed
    descriptor = design["sources"]["cached_teacher_sft"]
    root = LOCAL_RUNS / descriptor["task_id"]
    receipt = json.loads((root / "training.json").read_text())
    cache = json.loads((root / "teacher_train_cache.json").read_text())
    experiment = copy.deepcopy(design)
    experiment["sources"]["cached_teacher_sft"]["run_dir"] = str(root)
    monkeypatch.setattr(lane, "verify_source_files", lambda *args: receipt)
    monkeypatch.setattr(lane, "verify_training", lambda *args: (receipt, cache))
    monkeypatch.setattr(lane, "execution_identity", lambda *args: receipt["execution"])
    monkeypatch.setattr(lane, "restored_student", lambda *args: (SimpleNamespace(config=SimpleNamespace(vocab_size=128)), None))
    monkeypatch.setattr(lane, "validate_cache_tokens", lambda *args: [])
    monkeypatch.setattr(lane, "target_sequences", lambda *args: [])
    monkeypatch.setattr(lane, "restore_optimizer", lambda *args: (None, [], {"clock": 384}))
    native = json.loads((root / "cached_teacher_sft/checkpoint384/train_probes.json").read_text())
    native[0]["generation"]["correct"] = False
    monkeypatch.setattr(lane, "measure", lambda *args, **kwargs: native)
    monkeypatch.setattr(lane, "retention_evaluate", lambda *args: json.loads((root / "final_retention.json").read_text()))

    def forbidden_updates(*args):
        pytest.fail("updates ran after native384 parity failed")

    monkeypatch.setattr(lane, "continue_updates", forbidden_updates)
    with pytest.raises(ValueError, match="native TRAIN24 reload parity"):
        lane.training(experiment, original, context, corpus, audit, {"method": "cached_teacher_sft"}, {}, tmp_path)
    assert not (tmp_path / "prerequisites.json").exists()
    assert not (tmp_path / "ledger.json").exists()


def cpu_child(config_path, output):
    config = json.loads(config_path.read_text())
    assert not torch.cuda.is_initialized()
    ban = EvaluationReadBan([config["teacher_path"]], [config["base_path"], config["checkpoint"]["path"]]).install()
    rejected = []
    for path in (config["teacher_path"], config["symlink"]):
        try:
            Path(path).read_bytes()
        except PermissionError:
            rejected.append(path)
    try:
        torch.load(config["teacher_path"], map_location="cpu", weights_only=True)
    except PermissionError:
        rejected.append("torch.load")
    assert len(rejected) == 3
    for path in (Path(config["base_path"]) / "model.safetensors", Path(config["checkpoint"]["path"]) / "adapter_model.safetensors"):
        ban.check(path)
    base = Qwen3ForCausalLM.from_pretrained(config["base_path"], local_files_only=True, dtype=torch.float32).requires_grad_(False).eval()
    model = load_saved_learner(base, config["checkpoint"])
    absent = require_teacher_absent(model)
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[1, 82, 61]]), use_cache=False).logits
    assert tensor_hash({"logits": logits}) == config["expected_logits_sha256"]
    write_json(output, {"status": "completed", "pid": os.getpid(), "parent_pid": os.getppid(),
                       "training_pid": config["training_pid"], "training_sha256": config["training_sha256"],
                       "child_token": config["child_token"], **absent, "train24_reload_parity": True,
                       "cpu_tiny_prediction_parity": True, "read_ban_rejections": rejected, "read_ban": ban.receipt()})


def test_fresh_uv_child_teacher_absent_and_read_ban(resumed_pair, tmp_path):
    pair = resumed_pair
    base_path = tmp_path / "base"
    pair["base"].save_pretrained(base_path, safe_serialization=True)
    checkpoint = save_adapter(pair["restored"], tmp_path / "checkpoint405")
    with torch.inference_mode():
        logits = pair["restored"](input_ids=torch.tensor([[1, 82, 61]]), use_cache=False).logits
    teacher = tmp_path / "teacher.pt"
    torch.save({"not_a_student": torch.tensor([9.0])}, teacher)
    link = tmp_path / "alias.pt"
    link.symlink_to(teacher)
    config = {"base_path": str(base_path), "checkpoint": checkpoint,
              "teacher_path": str(teacher), "symlink": str(link), "training_pid": os.getpid(),
              "training_sha256": "cpu-training-receipt", "child_token": "cpu-predeclared-child",
              "expected_logits_sha256": tensor_hash({"logits": logits})}
    write_json(tmp_path / "child.json", config)
    command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(Path(__file__).resolve()),
               "--cpu-child", str(tmp_path / "child.json"), "--output", str(tmp_path / "result.json")]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = child.communicate(timeout=60)
    assert child.returncode == 0, (stdout, stderr)
    result = json.loads((tmp_path / "result.json").read_text())
    verify_child_identity(result, os.getpid(), child.pid, config["training_sha256"], config["child_token"])
    assert result["teacher_model_loaded"] is False and len(result["read_ban_rejections"]) == 3


@pytest.mark.parametrize("key", ["pid", "parent_pid", "training_sha256", "child_token", "teacher_model_loaded"])
def test_child_receipt_cannot_impersonate_training_or_another_launch(key):
    result = {"status": "completed", "pid": 30, "parent_pid": 20, "training_pid": 10,
              "training_sha256": "sha", "child_token": "token", "teacher_model_loaded": False,
              "optimizer_updates": 0, "train24_reload_parity": True}
    result[key] = {"pid": 10, "parent_pid": 999, "training_sha256": "wrong", "child_token": "wrong", "teacher_model_loaded": True}[key]
    with pytest.raises(ValueError, match="ONPOLICY303_BUDGET"):
        verify_child_identity(result, 10, 20, "sha", "token")


def test_two_dispatchers_accept_existing_run_directory(sealed, tmp_path):
    design, _, _, _, _ = sealed
    for method in METHODS:
        config = ROOT / f"configs/onpolicy303-budget-{method.replace('_', '-')}.json"
        dispatch = json.loads(config.read_text())
        assert dispatch["protocol_sha256"] == digest(design)
        output = tmp_path / method
        output.mkdir()
        write_json(output / "execution.json", {"supervisor_already_created_this_directory": True})
        command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-B",
                   str(ROOT / "onpolicy303_budget.py"), "--config", str(config), "--output-dir", str(output), "--validate-only"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stderr
        assert json.loads((output / "study/validation.json").read_text())["updates"] == 0
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cpu-child", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_child(args.cpu_child, args.output)
