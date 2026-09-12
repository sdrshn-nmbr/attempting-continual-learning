import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM

import onpolicy303 as lane
from choice_consolidation import (
    adapter_state,
    base_tensors,
    make_learner,
    save_adapter,
    tensor_hash,
)
from choice_contract import (
    ROOT,
    SCORING,
    TASKS,
    digest,
    file_hash,
    load_corpus,
    write_json,
)
from generation_consolidation import cache_teacher_responses
from generation_consolidation import train_arm as frozen_sft_updates
from onpolicy303_contract import (
    METHODS,
    archive_identity,
    execution_identity,
    forward_kl,
    full_support_probabilities,
    prefix_diagnostics,
    require_standalone,
    response_batch,
    retention_evaluate,
    retention_rows,
    sample_response,
    supervisor_digest,
    validate_design,
)


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


@pytest.fixture
def design():
    return json.loads((ROOT / "configs/onpolicy303_protocol.json").read_text())


@pytest.fixture
def tiny(design):
    torch.set_num_threads(1)
    _, _, _, teacher_design, choice = validate_design(design)
    choice["learner"]["expected_trainable_parameters"] = 1792
    recipe = design["training"]
    recipe.update(updates_per_arm=2, example_exposures_per_arm=8, epochs=1,
                  checkpoint_updates=[1, 2], probe_rows_per_task=1)
    torch.manual_seed(73)
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                        eos_token_id=2, pad_token_id=0, bos_token_id=1, use_cache=False)
    base = Qwen3ForCausalLM(config).float().requires_grad_(False).eval()
    expert = get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                            lora_dropout=0, task_type="CAUSAL_LM"), adapter_name="teacher")
    with torch.no_grad():
        for name, parameter in expert.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.09)
    expert.requires_grad_(False).eval()
    tokenizer = TinyTokenizer()
    rows = [{"id": f"cpu-{task}-{digit}", "task": task, "group": f"{digit} {digit} {digit}",
             "prompt": "Output:", "choices": [f" {digit} {digit} {digit}", " 1 2 3", " 3 2 1", " 7 6 5"], "gold_idx": 0}
            for task in TASKS for digit in [0, 4]]
    prompts = [tokenizer(row["prompt"], True).input_ids for row in rows]
    targets = [tokenizer(row["choices"][0], False).input_ids + [2] for row in rows]
    return {"design": design, "choice": choice, "teacher_design": teacher_design, "base": base, "expert": expert,
            "tokenizer": tokenizer, "rows": rows, "prompts": prompts, "targets": targets,
            "schedule": [[0, 1, 2, 3], [4, 5, 0, 1]]}


def test_sealed_contract_and_real_coverage_control(design):
    old, _, _, _, choice = validate_design(design)
    assert design["training"] == {**old["training"], "methods": list(METHODS)}
    corpus, _ = load_corpus(choice)
    retention = retention_rows(design, corpus)
    assert len(retention) == 128
    assert {len(row["choices"]) for row in retention} == {2, 4, 5}
    assert choice["learner"]["rank"] == 8
    assert design["runtime_dependency"]["task_id"] == "followthrough-20260912-recovery-worker-runtime-qwen3-8b"
    for method in METHODS:
        for stage in ("train", "evaluate"):
            config = json.loads((ROOT / f"configs/onpolicy303_{method}_{stage}.json").read_text())
            assert config["protocol_sha256"] == digest(design)
            assert config["method"] == method and config["stage"] == stage
    root = ROOT.parent.parent / "outputs/portfolio/followthrough-20260912/runs/consolidation-coverage303-20260912"
    for name, expected in design["coverage_control"]["files_sha256"].items():
        assert file_hash(root / name) == expected
    qualified = root.parent / "consolidation-source303-coverage-20260912/qualification/qualification.json"
    assert file_hash(qualified) == design["teacher"]["qualification_sha256"]
    with pytest.raises(ValueError, match="LOSS_OR_SAMPLING"):
        altered = copy.deepcopy(design)
        altered["objective"]["kl_direction"] = "student_to_teacher"
        validate_design(altered)


def test_full_vocabulary_forward_kl_detach_and_response_masks():
    ids, attention, mask = response_batch([[1, 3, 4], [1]], [[5, 2], [6, 7, 2]], 0, 2, 16, "cpu")
    assert ids.tolist() == [[1, 3, 4, 5, 2], [1, 6, 7, 2, 0]]
    assert attention.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert mask.tolist() == [[False, False, True, True], [True, True, True, False]]
    torch.manual_seed(2)
    student = torch.randn(2, 4, 19, requires_grad=True)
    teacher = (torch.randn(2, 4, 19) * 2).requires_grad_()
    loss, per_token = forward_kl(student, teacher, mask)
    p, q = teacher.detach().softmax(-1), student.detach().softmax(-1)
    manual_tokens = (p * (p.log() - q.log())).sum(-1)
    expected = (manual_tokens[0, 2:].mean() + manual_tokens[1, :3].mean()) / 2
    reverse = ((q * (q.log() - p.log())).sum(-1) * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert not torch.allclose(loss, reverse)
    assert not per_token.requires_grad
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[~mask]) == 0
    expected_grad = (q - p) * mask.unsqueeze(-1) / mask.sum(-1).reshape(-1, 1, 1) / 2
    assert torch.allclose(student.grad, expected_grad, atol=1e-7)
    assert torch.count_nonzero(student.grad[mask][:, 8:]) > 0
    with pytest.raises(ValueError, match="POST_EOS"):
        response_batch([[1]], [[2, 9]], 0, 2, 16, "cpu")
    with pytest.raises(ValueError, match="MASK_OR_PRECISION"):
        forward_kl(student, teacher, torch.zeros_like(mask))


class ScriptedSampler(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.logits = logits
        self.calls = []

    def forward(self, input_ids, attention_mask, past_key_values, use_cache):
        self.calls.append((input_ids.tolist(), attention_mask.tolist(), past_key_values, use_cache))
        return SimpleNamespace(logits=self.logits.reshape(1, 1, -1), past_key_values=len(self.calls))


def test_sampler_has_positive_full_support_native_eos_and_no_forced_stop():
    probabilities = full_support_probabilities(torch.tensor([0.0, -120.0, -100.0, -2.0]))
    assert probabilities.dtype == torch.float64 and bool((probabilities > 0).all())
    assert torch.allclose(probabilities, torch.tensor([0.0, -120.0, -100.0, -2.0]).double().softmax(-1))
    model = ScriptedSampler(torch.tensor([0.0, 10.0, -120.0, 0.0]))
    response, info = sample_response(model, [1, 8], 2, 3, torch.Generator().manual_seed(4))
    assert response == [1, 1, 1]
    assert info["cap_without_eos"] and not info["forced_eos"]
    assert info["positive_support_count"] == 4
    assert model.calls[0][:2] == ([[1, 8]], [[1, 1]])
    assert model.calls[1][:2] == ([[1]], [[1, 1, 1]])
    eos_model = ScriptedSampler(torch.tensor([-120.0, -120.0, 10.0, -120.0]))
    response, info = sample_response(eos_model, [1], 2, 16, torch.Generator().manual_seed(2))
    assert response == [2] and info["ended_with_native_eos"] and len(eos_model.calls) == 1
    with pytest.raises(ValueError, match="SUPPORT_LOSS"):
        full_support_probabilities(torch.tensor([0.0, -10000.0]))
    with pytest.raises(ValueError, match="NONFINITE"):
        full_support_probabilities(torch.tensor([float("nan"), 0.0]))


def test_prefix_diagnostics_do_not_label_divergent_teacher_as_correct():
    student, teacher = torch.zeros(3, 10), torch.zeros(3, 10)
    teacher[0, 4], teacher[1, 5], teacher[2, 2] = 2, 2, 2
    result = prefix_diagnostics(student, teacher, [4, 7, 2], [4, 5, 2], 2, torch.ones(3))
    assert result["prefix_matches_qualified_answer"] == [True, True, False]
    assert result["teacher_next_token_matches_on_qualified_prefix"] == [True, True, None]
    assert result["qualified_next_token_ids"] == [4, 5, None]
    assert result["off_answer_prefix_competence_certified"] is False


@pytest.mark.parametrize("method", METHODS)
def test_real_updates_fixed_expert_reset_full_log_and_reload(tiny, tmp_path, method):
    student = make_learner(copy.deepcopy(tiny["base"]), tiny["choice"])
    initial = {k: v.detach().clone() for k, v in adapter_state(student).items()}
    initial_hash = tensor_hash(initial)
    base_hash = tensor_hash(base_tensors(student))
    teacher_hash = tensor_hash(tiny["expert"].state_dict())
    expert = None if method == "cached_teacher_sft" else tiny["expert"]
    result = lane.train_updates(student, expert, tiny["tokenizer"], tiny["rows"], tiny["prompts"], tiny["targets"],
                                tiny["schedule"], method, tiny["design"], tiny["teacher_design"], tmp_path)
    assert result["updates"] == 2 and result["example_exposures"] == 8
    assert result["optimizer_clocks"] == {"minimum": 2, "maximum": 2, "parameters_with_state": 8}
    assert tensor_hash(base_tensors(student)) == base_hash
    assert tensor_hash(tiny["expert"].state_dict()) == teacher_hash
    assert all(p.grad is None for p in tiny["expert"].parameters())
    assert tensor_hash(adapter_state(student)) != initial_hash
    final = result["checkpoints"]["2"]["adapter"]
    reloaded = lane.load_saved_learner(copy.deepcopy(tiny["base"]), final)
    assert set(reloaded.peft_config) == {"learner"} and not any(p.requires_grad for p in reloaded.parameters())
    ids = torch.tensor([tiny["prompts"][0]])
    with torch.no_grad():
        assert torch.equal(student(ids).logits, reloaded(ids).logits)
    trajectories = [json.loads(line) for line in (tmp_path / method / "trajectories.jsonl").read_text().splitlines()]
    assert len(trajectories) == 8
    assert result["loss_token_exposures"] == sum(r["diagnostics"]["loss_tokens"] for r in trajectories)
    assert all(r["diagnostics"]["vocabulary_size"] == 256 for r in trajectories)
    assert all(len(r["diagnostics"]["student_argmax_token_ids"]) == r["diagnostics"]["loss_tokens"] for r in trajectories)
    if method != "cached_teacher_sft":
        assert all(len(r["diagnostics"]["teacher_argmax_token_ids"]) == r["diagnostics"]["loss_tokens"] for r in trajectories)
    native = [{"id": row["id"], "row_sha256": digest(row),
               "generation": lane.generated_diagnostic(tiny["tokenizer"], row, prompt, target, 16)}
              for row, prompt, target in zip(tiny["rows"], tiny["prompts"], tiny["targets"], strict=True)]
    cache = cache_teacher_responses(tiny["rows"], native, {"eos_token_id": 2, "special_token_ids": [0, 1, 2]})
    write_json(tmp_path / "schedule.json", [[tiny["rows"][i]["id"] for i in batch] for batch in tiny["schedule"]])
    lane.verify_trajectory_ledger(tmp_path, {"method": method, "arm": result}, tiny["design"], tiny["rows"], cache)
    set_peft_model_state_dict(student, initial, adapter_name="learner")
    assert tensor_hash(adapter_state(student)) == initial_hash
    independent = make_learner(copy.deepcopy(tiny["base"]), tiny["choice"])
    assert tensor_hash(adapter_state(independent)) == initial_hash
    optimizer, _ = lane.new_optimizer(independent, tiny["design"]["training"])
    assert not optimizer.state
    state = torch.load(tmp_path / method / "checkpoint2/optimizer.pt", weights_only=True)
    assert all(int(value["step"]) == 2 for value in state["state"].values())


def test_cached_sft_is_bitwise_frozen_coverage_update_control(tiny, tmp_path):
    first = make_learner(copy.deepcopy(tiny["base"]), tiny["choice"])
    second = make_learner(copy.deepcopy(tiny["base"]), tiny["choice"])
    lane.train_updates(first, None, tiny["tokenizer"], tiny["rows"], tiny["prompts"], tiny["targets"], tiny["schedule"],
                       "cached_teacher_sft", tiny["design"], tiny["teacher_design"], tmp_path)
    frozen_sft_updates(second, tiny["tokenizer"], tiny["rows"], tiny["prompts"], tiny["targets"], tiny["schedule"],
                       "frozen_reference", tiny["design"], tiny["teacher_design"], tmp_path)
    assert tensor_hash(adapter_state(first)) == tensor_hash(adapter_state(second))
    for step in [1, 2]:
        a = torch.load(tmp_path / f"cached_teacher_sft/checkpoint{step}/optimizer.pt", weights_only=True)
        b = torch.load(tmp_path / f"frozen_reference/checkpoint{step}/optimizer.pt", weights_only=True)
        assert a["param_groups"] == b["param_groups"]
        for key, state in a["state"].items():
            for name, tensor in state.items():
                assert torch.equal(tensor, b["state"][key][name])


def test_teacher_prerequisite_failure_never_loads_learner(design, tmp_path, monkeypatch):
    context = validate_design(design)
    corpus, audit = load_corpus(context[-1])
    calls = []

    def reject(*args):
        raise ValueError("ACTUAL_NATIVE_GENERATION_FAILED")

    monkeypatch.setattr(lane, "qualify_actual_teacher", reject)
    monkeypatch.setattr(lane, "load_base", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="ACTUAL_NATIVE_GENERATION_FAILED"):
        lane.train(design, context, corpus, audit, {"method": "onpolicy_kl"}, {}, tmp_path)
    assert calls == [] and list(tmp_path.iterdir()) == []


def make_execution(tmp_path, config, task_id="cpu-train", completed=True):
    root = tmp_path / task_id
    root.mkdir()
    contents = {"onpolicy303.py": b"print('standalone fixture')\n", "dependency.json": b"{}\n"}
    source_hash = hashlib.sha256(b"".join(name.encode() + b"\0" + contents[name] for name in sorted(contents))).hexdigest()
    code = tmp_path / "code" / source_hash
    code.mkdir(parents=True, exist_ok=True)
    with tarfile.open(code.with_suffix(".tar"), "w") as archive:
        for name, payload in contents.items():
            (code / name).write_bytes(payload)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    task = {"id": task_id, "code_dir": str(code), "source_sha256": source_hash, "config": config, "entrypoint": "onpolicy303.py"}
    receipt = {"task_id": task_id, "attempt_id": task_id + "-attempt", "task": task,
               "task_sha256": supervisor_digest(task), "config_sha256": supervisor_digest(config),
               "status": "completed" if completed else "running", "exit_code": 0, "timed_out": False, "pid": 123}
    write_json(root / "task.json", task)
    (root / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_json(root / "config.json", config)
    return root, receipt, code


def test_execution_receipts_actual_archive_and_distinct_task_evidence(tmp_path):
    root, receipt, code = make_execution(tmp_path, {"stage": "train", "method": "onpolicy_kl"})
    proof = execution_identity(root, "cpu-train", True)
    current = {**proof, "task_id": "cpu-eval", "attempt_id": "cpu-eval-attempt", "config": {"stage": "evaluate", "method": "onpolicy_kl"}}
    result = require_standalone(current, {"execution": proof, "method": "onpolicy_kl", "pid": os.getpid()})
    assert result["distinct_task_ids"] and result["numeric_pid_inequality_is_not_the_process_proof"]
    with pytest.raises(ValueError, match="SEPARATE_EVALUATION_TASK"):
        require_standalone(proof, {"execution": proof, "method": "onpolicy_kl", "pid": -1})
    receipt["exit_code"] = 1
    (root / "execution.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="EXECUTION_BINDING"):
        execution_identity(root, "cpu-train", True)
    (code / "dependency.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="ARCHIVE_BYTES"):
        archive_identity(code, proof["source"]["source_sha256"])


def test_retention_preserves_original_candidate_cardinality_and_frozen_scoring(tiny):
    rows = [{"task": "generic", "prompt": "Output:", "choices": [" 0", " 1", " 2", " 3", " 4"][:n], "gold_idx": 0} for n in [2, 4, 5]]
    model = make_learner(copy.deepcopy(tiny["base"]), tiny["choice"])
    before = tensor_hash(adapter_state(model))
    records = retention_evaluate(model, tiny["tokenizer"], rows)
    assert [len(r["scores"]) for r in records] == [2, 4, 5]
    assert all(r["prediction"] == max(range(len(r["scores"])), key=r["scores"].__getitem__) for r in records)
    assert tensor_hash(adapter_state(model)) == before
    assert SCORING["candidate_batch_size"] == 8


def test_teacher_absent_fresh_process_checkpoint_reload(tiny, tmp_path):
    student = make_learner(copy.deepcopy(tiny["base"]), tiny["choice"])
    expert = tiny["expert"]
    loss, _ = lane.row_loss(student, expert, tiny["prompts"][0], tiny["targets"][0], tiny["targets"][0], tiny["tokenizer"], "cached_teacher_kl", 16)
    optimizer, _ = lane.new_optimizer(student, tiny["design"]["training"])
    loss.backward()
    optimizer.step()
    checkpoint = save_adapter(student, tmp_path / "checkpoint")
    tiny["base"].save_pretrained(tmp_path / "base")
    ids = torch.tensor([tiny["prompts"][0]])
    with torch.no_grad():
        expected = student(ids).logits
    torch.save(expected, tmp_path / "expected.pt")
    write_json(tmp_path / "spec.json", {"checkpoint": checkpoint, "prompt": tiny["prompts"][0], "training_pid": os.getpid()})
    script = """import json
import os
import sys
from pathlib import Path
import torch
from transformers import Qwen3ForCausalLM
from onpolicy303 import load_saved_learner
def audit(event, args):
    if event == 'open' and {'teacher', 'expert'}.intersection(Path(str(args[0])).parts):
        raise RuntimeError('Teacher artifact access forbidden')
root = Path(sys.argv[1])
spec = json.loads((root / 'spec.json').read_text())
sys.addaudithook(audit)
torch.set_num_threads(1)
base = Qwen3ForCausalLM.from_pretrained(root / 'base', dtype=torch.float32, local_files_only=True).requires_grad_(False).eval()
model = load_saved_learner(base, spec['checkpoint'])
assert set(model.peft_config) == {'learner'}
assert not any(p.requires_grad for p in model.parameters())
with torch.no_grad():
    actual = model(torch.tensor([spec['prompt']])).logits
assert torch.equal(actual, torch.load(root / 'expected.pt', weights_only=True))
assert os.getpid() != spec['training_pid']
(root / 'child.json').write_text(json.dumps({'pid': os.getpid(), 'parent_pid': os.getppid(), 'teacher_loaded': False}))
"""
    child = tmp_path / "reload_child.py"
    child.write_text(script)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "python", str(child), str(tmp_path)],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    proof = json.loads((tmp_path / "child.json").read_text())
    assert proof["pid"] != os.getpid() and proof["teacher_loaded"] is False
