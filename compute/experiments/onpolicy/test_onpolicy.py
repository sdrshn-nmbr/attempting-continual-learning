import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from peft import set_peft_model_state_dict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

import run
import tasks
from config import ARMS, Config, read_config
from model import ModelIO, StopFlag, StopRequested, attach_adapter, policy_mode
from objectives import (
    causal_example,
    completion_mask,
    dense_reverse_kl,
    oracle_nll,
    response_logits,
)
from run import Experiment, adapter_state, comparisons, select_calibration_variant
from tasks import (
    RULES,
    TASKS,
    Example,
    build_calibration_data,
    build_data,
    data_manifest,
    execute,
    grade,
    prompt,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(tmp_path):
    model_path = tmp_path / "native_fixture"
    model_path.mkdir(exist_ok=True)
    (model_path / "config.json").write_text("{}\n")
    return Config(
        model_id="tests/tiny-qwen",
        revision="0" * 40,
        model_path=str(model_path),
        seed=37,
        experiment_kind="cpu_test",
        steps_per_task=1,
        tokens_per_update=8,
        eval_examples=2,
        gate_examples=2,
        max_new_tokens=4,
        lora_rank=2,
        lora_alpha=4,
        device="cpu",
        dtype="float32",
        require_rocm=False,
    ).validate()


def tiny_model(config, family="qwen3_5"):
    torch.manual_seed(config.seed)
    if family == "qwen3_5":
        text_config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
            pad_token_id=0,
            eos_token_id=2,
        )
        vision_config = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            num_position_embeddings=16,
        )
        base = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=61,
                video_token_id=62,
                vision_start_token_id=63,
            )
        )
    else:
        base = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                pad_token_id=0,
                eos_token_id=2,
            )
        )
    base.generation_config.eos_token_id = 2
    return attach_adapter(base, config, family)


def calibration_config(tmp_path):
    return replace(
        tiny_config(tmp_path),
        mode="teacher_calibration",
        arms=(),
        steps_per_task=0,
        tokens_per_update=0,
        eval_examples=0,
        teacher_min_accuracy=1.0,
        gradient_checkpointing=False,
    ).validate()


def train_answer_config(tmp_path):
    return replace(
        calibration_config(tmp_path), mode="train_answer_calibration", gate_examples=0
    ).validate()


def oracle_control_config(tmp_path):
    return replace(
        tiny_config(tmp_path),
        mode="oracle_sft_control",
        arms=("oracle_sft",),
        steps_per_task=32,
        tokens_per_update=40,
        eval_examples=32,
        gate_examples=0,
        teacher_min_accuracy=1.0,
        lora_rank=8,
        lora_alpha=16,
        gradient_checkpointing=False,
    ).validate()


def tiny_tokenizer():
    vocabulary = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3}
    for text in (
        "0",
        "1",
        "2",
        "3",
        "[",
        "]",
        ",",
        "dax",
        "wug",
        "fep",
        "kiv",
        "Input",
        "Output",
        "Operations",
        ":",
        "Authoritative",
        "definitions",
        "Examples",
        "Assistant",
    ):
        vocabulary[text] = len(vocabulary)
    for i in range(len(vocabulary), 64):
        vocabulary[f"token{i}"] = i
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    tokenizer.chat_template = (
        "{{ messages[0]['content'] }}{% if add_generation_prompt %}\nAssistant:{% endif %}"
    )
    return tokenizer


def make_experiment(tmp_path, monkeypatch, family="qwen3_5"):
    config = tiny_config(tmp_path)
    monkeypatch.setattr(
        run,
        "load_model",
        lambda cfg: (tiny_model(cfg, family), tiny_tokenizer(), {"test_fixture": True}),
    )
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    experiment = Experiment(config, output, StopFlag())
    experiment.prepare()
    return experiment


def test_executable_oracle_and_order():
    assert execute([0, 1, 2, 3], ("dax",)) == [3, 2, 1, 0]
    assert execute([0, 1, 2, 3], ("wug",)) == [1, 2, 3, 0]
    assert execute([0, 1, 2, 3], ("dax", "wug")) == [2, 1, 0, 3]
    assert execute([0, 1, 2, 3], ("wug", "dax")) == [0, 3, 2, 1]
    assert execute([0, 1, 2, 3], ("fep", "kiv")) == [0, 3, 2, 1]
    assert execute([0, 1, 2, 3], ("kiv", "fep")) == [2, 1, 0, 3]


def test_splits_deterministic_disjoint_and_no_privilege_leak():
    data = build_data(37)
    assert data_manifest(data, 37, 16) == data_manifest(build_data(37), 37, 16)
    assert data_manifest(data, 37, 16)["sha256"] != data_manifest(build_data(38), 38, 16)["sha256"]
    for task in TASKS:
        used = set()
        for split in ("demo", "train", "gate", "heldout"):
            inputs = {x.inputs for x in data[task][split]}
            assert not inputs & used
            used |= inputs
        for example in data[task]["heldout"]:
            student = prompt(example, data, "cue_only")
            teacher = prompt(example, data, "cue_only", privileged=True)
            assert student.endswith("Output:") and teacher.endswith("Output:")
            assert "Examples:" not in student
            assert all(rule not in student for rule in RULES.values())
            assert "Authoritative operation definitions:" in teacher
        assert all(len(x.program) == 1 for x in data[task]["train"])
        assert all(len(x.program) == 2 for x in data[task]["heldout_compositions"])


def test_invalid_refusal_and_executable_accuracy_are_separate():
    example = Example("permutation", "heldout", (0, 1, 2, 3), ("dax",))
    assert grade("[3, 2, 1, 0]", example) == {"correct": True, "invalid": False, "refusal": False}
    assert grade("[0,1,2,3]", example) == {"correct": False, "invalid": False, "refusal": False}
    assert grade("Sorry, I cannot do that.", example) == {
        "correct": False,
        "invalid": True,
        "refusal": True,
    }
    for invalid in ("[true,2,1,0]", "[3,2,1]", "answer: [3,2,1,0]", "[3.0,2,1,0]"):
        assert grade(invalid, example)["invalid"]


def test_mask_includes_first_eos_and_excludes_padding_even_when_eos_is_pad():
    assert completion_mask(torch.tensor([4, 5, 2, 9, 0]), (2,), 0).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert completion_mask(torch.tensor([4, 2, 2, 2]), (2,), 2).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert completion_mask(torch.tensor([4, 0, 5]), (2,), 0).tolist() == [True, False, False]
    assert completion_mask(torch.tensor([4, 0, 5]), (2,), None).tolist() == [True, True, True]


def test_causal_alignment_with_different_privilege_prefix_lengths_and_gradients():
    torch.manual_seed(3)
    student = torch.randn(12, 7, requires_grad=True)
    teacher = torch.randn(19, 7, requires_grad=True)
    completion = torch.tensor([4, 5, 6, 2])
    short = causal_example(torch.tensor([3, 3, 3]), completion)
    long = causal_example(torch.tensor([3] * 10), completion)
    assert short.prediction_positions.tolist() == [2, 3, 4, 5]
    assert long.prediction_positions.tolist() == [9, 10, 11, 12]
    loss = dense_reverse_kl(
        student[short.prediction_positions],
        teacher[long.prediction_positions],
        torch.ones(4, dtype=torch.bool),
    )
    student_logp = student[2:6].log_softmax(-1)
    teacher_logp = teacher[9:13].detach().log_softmax(-1)
    expected = torch.nn.functional.kl_div(
        teacher_logp, student_logp, log_target=True, reduction="sum"
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:2].count_nonzero() == 0 and student.grad[6:].count_nonzero() == 0
    assert student.grad[2:6].abs().sum() > 0


def test_masked_nan_positions_do_not_change_loss_or_receive_gradient():
    student = torch.tensor([[0.3, -0.4], [float("nan"), float("nan")]], requires_grad=True)
    teacher = torch.tensor([[-0.1, 0.2], [float("nan"), float("nan")]], requires_grad=True)
    mask = torch.tensor([True, False])
    loss = dense_reverse_kl(student, teacher, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad[1].count_nonzero() == 0 and teacher.grad is None
    ce = oracle_nll(student, torch.tensor([1, -100]), mask)
    assert torch.isfinite(ce)
    with pytest.raises(ValueError):
        dense_reverse_kl(student, teacher, torch.tensor([False, False]))


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5"])
def test_native_generation_stops_at_chat_eos_when_model_eos_is_padding(
    tmp_path, monkeypatch, family
):
    config = tiny_config(tmp_path)
    model = tiny_model(config, family)
    tokenizer = tiny_tokenizer()
    model.generation_config.eos_token_id = tokenizer.pad_token_id
    io = ModelIO(model, tokenizer, config, build_data(config.seed), StopFlag())
    example = Example("permutation", "gate", (2, 3, 1, 2), ("dax",))
    gold = io.oracle_tokens(example)
    prefix = torch.tensor([4, 5, 6])
    planned = gold.tolist() + [7, tokenizer.pad_token_id]

    class PlannedTokens(LogitsProcessor):
        def __call__(self, input_ids, scores):
            next_token = planned[input_ids.shape[1] - len(prefix)]
            forced = torch.full_like(scores, -torch.inf)
            forced[:, next_token] = 0
            return forced

    native_generate = model.generate

    def controlled_logits(**kwargs):
        return native_generate(logits_processor=LogitsProcessorList([PlannedTokens()]), **kwargs)

    monkeypatch.setattr(model, "generate", controlled_logits)
    response, generated_tokens = io.generate(prefix, True, False, config.seed, len(planned))
    torch.testing.assert_close(response, gold, rtol=0, atol=0)
    assert generated_tokens == len(gold)
    assert io.model_eos_ids == (tokenizer.pad_token_id,)
    assert io.eos_ids == (tokenizer.eos_token_id, tokenizer.pad_token_id)
    assert grade(io.decode(response), example)["correct"]
    assert grade(example.answer + tokenizer.eos_token, example)["invalid"]
    assert grade("Answer: " + example.answer, example)["invalid"]


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5"])
def test_native_lora_gradients_frozen_teacher_alignment_and_seeded_generation(tmp_path, family):
    config = tiny_config(tmp_path)
    model = tiny_model(config, family)
    data = build_data(config.seed)
    io = ModelIO(model, tiny_tokenizer(), config, data, StopFlag())
    student_prefix = torch.tensor([4, 5, 6, 7])
    teacher_prefix = torch.tensor([4, 5, 6, 7, 8, 9, 10])
    completion = torch.tensor([8, 9, 2])
    with torch.no_grad(), policy_mode(model, privileged=True):
        teacher_before = response_logits(model, teacher_prefix, completion).clone()
        packed = causal_example(teacher_prefix, completion)
        full = model(
            input_ids=packed.input_ids, attention_mask=packed.attention_mask, use_cache=False
        ).logits[0]
        torch.testing.assert_close(teacher_before, full[packed.prediction_positions])
        changed_last_token = completion.clone()
        changed_last_token[-1] = 11
        torch.testing.assert_close(
            teacher_before,
            response_logits(model, teacher_prefix, changed_last_token),
            rtol=0,
            atol=0,
        )
    first, _ = io.generate(student_prefix, False, True, 1234, 4)
    second, _ = io.generate(student_prefix, False, True, 1234, 4)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    model.train()
    before = response_logits(model, student_prefix, completion)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss = dense_reverse_kl(before, teacher_before, torch.ones(3, dtype=torch.bool))
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    with torch.no_grad(), policy_mode(model, privileged=True):
        torch.testing.assert_close(
            response_logits(model, teacher_prefix, completion), teacher_before, rtol=0, atol=0
        )
    with torch.no_grad(), policy_mode(model):
        assert not torch.equal(before.detach(), response_logits(model, student_prefix, completion))


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5"])
def test_all_arms_use_identical_update_query_and_exact_token_budget(tmp_path, monkeypatch, family):
    experiment = make_experiment(tmp_path, monkeypatch, family)
    records = []
    for arm in ARMS:
        set_peft_model_state_dict(experiment.model, experiment.initial_adapter)
        experiment.optimizer = experiment.optimizer_for_model()
        record = experiment.update(arm, TASKS[0], 0)
        assert record["supervised_tokens"] == experiment.config.tokens_per_update
        assert record["grad_norm"] > 0
        records.append(record)
    assert len({x["example_id"] for x in records}) == 1
    assert records[-1]["generated_tokens"] == 0


def test_checkpoint_resume_reproduces_next_native_update(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch, "qwen3")
    arm = "on_policy_dense"
    experiment.current_arm = arm
    experiment.optimizer = experiment.optimizer_for_model()
    record = experiment.update(arm, TASKS[0], 0)
    experiment.next_step = 1
    experiment.metrics["arms"][arm] = {"updates": [record]}
    experiment.save_checkpoint()
    continued = experiment.update(arm, TASKS[1], 0)
    expected = adapter_state(experiment.model)
    resumed = Experiment(experiment.config, experiment.output, StopFlag())
    assert resumed.prepare()
    repeated = resumed.update(arm, TASKS[1], 0)
    actual = adapter_state(resumed.model)
    assert resumed.next_step == 1
    assert repeated == continued
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_teacher_gate_fails_closed_with_actual_random_native_generations(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch, "qwen3")
    before = adapter_state(experiment.model)
    assert not experiment.teacher_gate()
    assert experiment.metrics["status"] == "teacher_gate_failed"
    assert not experiment.metrics["arms"]
    assert (experiment.output / "checkpoint" / "state.pt").exists()
    gate = experiment.metrics["reference"]["teacher_gate"]
    assert all(row["n"] == 2 for row in gate["measurements"].values())
    for key, value in adapter_state(experiment.model).items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_signal_flag_stops_generation_and_is_checkpointable(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch)
    experiment.stop.request(signal.SIGTERM, None)
    with pytest.raises(StopRequested):
        experiment.io.generate(torch.tensor([4, 5]), False, True, 1, 4)
    experiment.metrics["status"] = "terminated"
    experiment.save_checkpoint()
    state = torch.load(experiment.output / "checkpoint" / "state.pt", weights_only=True)
    assert state["metrics"]["status"] == "terminated"
    assert state["next_step"] == 0 and state["optimizer"] is None


def test_comparisons_capture_forgetting_and_forward_transfer():
    baseline = {
        "permutation/inputs/cue_only": {"accuracy": 0.1, "mean_gold_answer_logprob": -10.0},
        "symbol_map/inputs/cue_only": {"accuracy": 0.0, "mean_gold_answer_logprob": -20.0},
    }
    after_a = {
        "permutation/inputs/cue_only": {"accuracy": 0.8, "mean_gold_answer_logprob": -2.0},
        "symbol_map/inputs/cue_only": {"accuracy": 0.3, "mean_gold_answer_logprob": -10.0},
    }
    after_b = {
        "permutation/inputs/cue_only": {"accuracy": 0.5, "mean_gold_answer_logprob": -3.0},
        "symbol_map/inputs/cue_only": {"accuracy": 0.7, "mean_gold_answer_logprob": -2.0},
    }
    rows = comparisons(
        {"before": baseline, "after_permutation": after_a, "after_symbol_map": after_b}
    )
    assert rows["permutation/inputs/cue_only"]["forgetting_accuracy_drop"] == pytest.approx(0.3)
    assert rows["symbol_map/inputs/cue_only"]["forward_transfer_accuracy_delta"] == 0.3


def test_ready_configs_and_strict_validation(tmp_path):
    for path in (Path(__file__).parent / "configs").glob("*.json"):
        config = read_config(path)
        assert config.model_path.endswith(config.revision)
    config = tiny_config(tmp_path)
    for invalid in (
        replace(config, tokens_per_update=0),
        replace(config, arms=("unknown",)),
        replace(config, teacher_min_accuracy=0),
    ):
        with pytest.raises(ValueError):
            invalid.validate()
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({**config.to_dict(), "unknown_option": True}))
    with pytest.raises(TypeError):
        read_config(unknown)


def test_rejected_resume_preserves_existing_metrics_and_checkpoint(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch)
    files = [experiment.output / "metrics.json", experiment.output / "checkpoint" / "state.pt"]
    original = {path: path.read_bytes() for path in files}
    conflict = tmp_path / "conflicting.json"
    conflict.write_text(json.dumps(replace(experiment.config, seed=99).to_dict()))
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--config", str(conflict), "--output-dir", str(experiment.output)]
    )
    term_handler, int_handler = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        assert run.main() == 1
    finally:
        signal.signal(signal.SIGTERM, term_handler)
        signal.signal(signal.SIGINT, int_handler)
    for path in files:
        assert path.read_bytes() == original[path]


def test_real_cli_sigterm_checkpoint_and_resume_to_generation_gate(tmp_path):
    config = tiny_config(tmp_path)
    base = tiny_model(config, "qwen3").unload()
    base.save_pretrained(config.model_path)
    tiny_tokenizer().save_pretrained(config.model_path)
    config_file = tmp_path / "cpu_config.json"
    config_file.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "cli_output"
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(Path(run.__file__)),
        "--config",
        str(config_file),
        "--output-dir",
        str(output),
    ]
    with (tmp_path / "subprocess.log").open("w+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while (
                not (output / "progress.json").exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert process.poll() is None, "CLI exited before the signal checkpoint"
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 143
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        log.seek(0)
        transcript = log.read()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "terminated", transcript
    assert (output / "checkpoint" / "state.pt").exists()
    resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 2, resumed.stdout + resumed.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "teacher_gate_failed"
    assert not metrics["evidence"]["gpu_training_observed"]
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "checkpoint_resumed" for row in events)
    assert any(
        row["event"] == "evaluation_example" and "gold_answer_logprob" in row for row in events
    )


@pytest.mark.parametrize("unknown_artifact", [False, True])
@pytest.mark.parametrize("mode", ["sequential", "train_answer_calibration", "oracle_sft_control", "oracle_sft_replay"])
def test_real_cli_runtime_output_contract_preserves_parent_files(tmp_path, unknown_artifact, mode):
    if mode in ("oracle_sft_control", "oracle_sft_replay"):
        config, family = oracle_control_config(tmp_path), "qwen3_5"
        if mode == "oracle_sft_replay":
            config = replace(config, replay_examples_per_update=2).validate()
        expected_status = "completed"
    elif mode == "train_answer_calibration":
        config, family = train_answer_config(tmp_path), "qwen3_5"
        expected_status = "train_answer_calibration_failed"
    else:
        config, family = tiny_config(tmp_path), "qwen3"
        expected_status = "teacher_gate_failed"
    tiny_model(config, family).unload().save_pretrained(config.model_path)
    tiny_tokenizer().save_pretrained(config.model_path)
    output = tmp_path / "runtime_output"
    output.mkdir()
    parent_files = {
        "task.json": b'{"task_id":"onpolicy-gate"}\n',
        "execution.json": b'{"state":"running","owner":"parent"}\n',
        "run.log": b"parent runtime initialized\n",
        "packages.txt": b"torch==2.10.0\ntransformers==5.16.1\n",
        "config.json": json.dumps(config.to_dict()).encode() + b"\n",
    }
    if unknown_artifact:
        parent_files["unrelated.data"] = b"another experiment's data\n"
    for name, contents in parent_files.items():
        (output / name).write_bytes(contents)
    history = output / "attempts" / ("a" * 32)
    history.mkdir(parents=True)
    receipt = parent_files["execution.json"]
    archived = history / f"{hashlib.sha256(receipt).hexdigest()}.json"
    archived.write_bytes(receipt)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(Path(run.__file__)),
            "--config",
            str(output / "config.json"),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    if unknown_artifact:
        expected_exit = 1
    elif mode in ("oracle_sft_control", "oracle_sft_replay"):
        expected_exit = 0
    else:
        expected_exit = 2
    assert result.returncode == expected_exit, result.stdout + result.stderr
    for name, contents in parent_files.items():
        assert (output / name).read_bytes() == contents
    assert archived.read_bytes() == receipt
    if unknown_artifact:
        assert "OUTPUT_CONFLICT" in result.stderr
        assert not (output / "metrics.json").exists()
    else:
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["status"] == expected_status
        assert metrics["config_sha256"] == config.sha256
        assert (output / "progress.json").exists()
        assert (output / "events.jsonl").exists()
        assert (output / "checkpoint" / "state.pt").exists()


def test_phase_machine_runs_sequential_arms_and_records_retention(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch, "qwen3")
    baseline = experiment.evaluate("before")
    experiment.metrics["baseline"] = baseline
    initial = adapter_state(experiment.model)
    for arm in ARMS:
        experiment.run_arm(arm)
        result = experiment.metrics["arms"][arm]
        assert result["completed"]
        assert list(result["evaluations"]) == ["before", "after_permutation", "after_symbol_map"]
        assert result["budget"]["supervised_tokens"] == 16
        assert result["budget"]["updates"] == 2
        assert "retention_accuracy_delta" in result["comparisons"]["permutation/inputs/cue_only"]
        assert any(
            not torch.equal(initial[key], value)
            for key, value in adapter_state(experiment.model).items()
        )


def test_calibration_dev_is_disjoint_from_existing_primary_splits():
    calibration = build_calibration_data(37, 32)
    dev = {example.inputs for task in TASKS for example in calibration[task]["gate"]}
    root = Path(__file__).parent / "configs"
    for name in ("preflight_qwen35_4b.json", "pilot_qwen35_4b.json"):
        config = read_config(root / name)
        primary = build_data(
            config.seed, config.train_examples, config.eval_examples, config.gate_examples
        )
        heldout = {example.inputs for task in TASKS for example in primary[task]["heldout"]}
        assert not dev & heldout
    assert all("heldout" not in split for splits in calibration.values() for split in splits)
    assert all(len(calibration[task]["gate"]) == 32 for task in TASKS)
    manifest = data_manifest(calibration, 37, 0)
    assert all(not steps for steps in manifest["training_schedule"].values())


def test_numbered_instructions_preserve_order_without_query_answer_access(monkeypatch):
    data = build_calibration_data(37, 16)
    example = data["permutation"]["gate_compositions"][0]
    demo_inputs = {x.inputs for task in TASKS for x in data[task]["demo"]}
    oracle = tasks.execute

    def demonstrations_only(inputs, program):
        assert tuple(inputs) in demo_inputs, "Prompt builder accessed a query oracle answer"
        return oracle(inputs, program)

    monkeypatch.setattr(tasks, "execute", demonstrations_only)
    current = prompt(example, data, "examples", True, "current")
    numbered = prompt(example, data, "examples", True, "numbered")
    assert current.endswith("Output:") and numbered.endswith("Output:")
    assert f"1. Apply {example.program[0]} to the Input list." in numbered
    assert f"2. Apply {example.program[1]} to the result of step 1." in numbered
    assert "Apply every numbered operation exactly once" in numbered
    for operation in example.program:
        assert RULES[operation] in current and RULES[operation] in numbered
    with pytest.raises(ValueError):
        prompt(example, data, "examples", True, "third_attempt")


def test_current_instruction_variant_retains_frozen_prompt_bytes():
    data = build_data(37)
    expected = {
        "permutation": "2284ffaa5f4db79630d2443fe45eb071f61d0b17d518ccc4cbfb5815f9568b1f",
        "symbol_map": "af7894d9dd5350925fa62bfc828d166290ae3d8afaac8733c2fa6a71474facca",
    }
    for task, digest in expected.items():
        example = data[task]["gate_compositions"][0]
        assert (
            hashlib.sha256(prompt(example, data, "examples", True).encode()).hexdigest() == digest
        )


def test_calibration_config_prohibits_updates_primary_eval_and_threshold_drop(tmp_path):
    config = calibration_config(tmp_path)
    for invalid in (
        replace(config, arms=("on_policy_dense",)),
        replace(config, steps_per_task=1),
        replace(config, tokens_per_update=8),
        replace(config, eval_examples=2),
        replace(config, teacher_min_accuracy=0.99),
        replace(config, train_examples=12),
        replace(config, gate_examples=33),
    ):
        with pytest.raises(ValueError):
            invalid.validate()


@pytest.mark.parametrize(
    "current,numbered,selected",
    [(1.0, 1.0, "current"), (0.5, 1.0, "numbered"), (1.0, 0.5, "current"), (0.5, 0.5, None)],
)
def test_calibration_selection_uses_fixed_all_suite_gate(current, numbered, selected):
    measurements = {
        "current": {"inputs": {"accuracy": 1.0}, "compositions": {"accuracy": current}},
        "numbered": {"inputs": {"accuracy": 1.0}, "compositions": {"accuracy": numbered}},
    }
    assert select_calibration_variant(measurements, 1.0) == selected


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5"])
def test_teacher_calibration_real_native_calls_are_dev_only_and_never_train(
    tmp_path, monkeypatch, family
):
    config = calibration_config(tmp_path)
    model = tiny_model(config, family)
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    monkeypatch.setattr(
        run, "load_model", lambda cfg: (model, tiny_tokenizer(), {"test_fixture": True})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher calibration touched primary evaluation or training")

    monkeypatch.setattr(run, "build_data", forbidden)
    for method in ("evaluate", "teacher_gate", "optimizer_for_model", "run_arm", "update"):
        monkeypatch.setattr(Experiment, method, forbidden)
    output = tmp_path / "calibration"
    output.mkdir()
    experiment = Experiment(config, output, StopFlag())
    assert experiment.run() == 2
    assert experiment.metrics["status"] == "teacher_calibration_failed"
    calibration = experiment.metrics["teacher_calibration"]
    assert calibration["selected_variant"] is None
    assert len(calibration["records"]) == 20
    for index in range(0, 20, 2):
        current, numbered = calibration["records"][index : index + 2]
        assert current["instruction_variant"] == "current"
        assert numbered["instruction_variant"] == "numbered"
        assert current["example_id"] == numbered["example_id"]
        assert current["expected"] == numbered["expected"]
        assert current["prompt_sha256"] != numbered["prompt_sha256"]
        assert current["privileged"] and numbered["privileged"]
        assert all(row["split"] in ("gate", "gate_compositions") for row in (current, numbered))
        assert all(row["generated_token_ids"] for row in (current, numbered))
    budget = experiment.metrics["budget_check"]
    assert (
        budget["optimizer_updates"]
        == budget["supervised_tokens"]
        == budget["primary_heldout_calls"]
        == 0
    )
    assert budget["expected_generation_calls"] == budget["generation_calls"] == 20
    assert experiment.optimizer is None and not experiment.metrics["arms"]
    assert "baseline" not in experiment.metrics and not experiment.metrics["reference"]
    assert all(
        "heldout" not in split
        for splits in experiment.manifest["examples"].values()
        for split in splits
    )
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, initial_parameters[name], rtol=0, atol=0)
        assert value.grad is None
    checkpoint = torch.load(output / "checkpoint" / "state.pt", weights_only=True)
    assert checkpoint["adapter"] is checkpoint["initial_adapter"] is checkpoint["optimizer"] is None
    before = (output / "events.jsonl").read_bytes()
    assert Experiment(config, output, StopFlag()).run() == 2
    assert (output / "events.jsonl").read_bytes() == before


def test_calibration_resume_does_not_repeat_completed_pair_member(tmp_path, monkeypatch):
    config = calibration_config(tmp_path)
    monkeypatch.setattr(
        run,
        "load_model",
        lambda cfg: (tiny_model(cfg, "qwen3"), tiny_tokenizer(), {"test_fixture": True}),
    )
    calls = []
    evaluate = Experiment.evaluate_example

    def interrupt_after_first(
        self, example, phase, name, context, privileged, instruction_variant="current"
    ):
        row = evaluate(self, example, phase, name, context, privileged, instruction_variant)
        calls.append((example.key, instruction_variant))
        if len(calls) == 1:
            self.stop.request(signal.SIGTERM, None)
        return row

    monkeypatch.setattr(Experiment, "evaluate_example", interrupt_after_first)
    output = tmp_path / "interrupted_calibration"
    output.mkdir()
    with pytest.raises(StopRequested):
        Experiment(config, output, StopFlag()).run()
    checkpoint = torch.load(output / "checkpoint" / "state.pt", weights_only=True)
    assert len(checkpoint["metrics"]["teacher_calibration"]["records"]) == 1
    resumed = Experiment(config, output, StopFlag())
    assert resumed.run() == 2
    assert len(calls) == len(set(calls)) == 20
    assert len(resumed.metrics["teacher_calibration"]["records"]) == 20


def test_train_answer_config_locks_train_only_budget_and_threshold(tmp_path):
    config = train_answer_config(tmp_path)
    for invalid in (
        replace(config, seed=38),
        replace(config, train_examples=15),
        replace(config, gate_examples=16),
        replace(config, eval_examples=1),
        replace(config, arms=("on_policy_dense",)),
        replace(config, steps_per_task=1),
        replace(config, tokens_per_update=1),
        replace(config, teacher_min_accuracy=0.99),
        replace(config, experiment_kind="pilot"),
    ):
        with pytest.raises(ValueError):
            invalid.validate()


def test_train_answer_data_is_exactly_five_train_suites_without_other_splits():
    data = tasks.build_train_answer_data(37)
    inputs = tasks.input_pool(37)[4:20]
    expected_programs = {
        "permutation/inputs": [("dax",), ("wug",)] * 8,
        "permutation/compositions": [("wug", "dax"), ("dax", "wug")] * 8,
        "symbol_map/inputs": [("fep",), ("kiv",)] * 8,
        "symbol_map/compositions": [("kiv", "fep"), ("fep", "kiv")] * 8,
        "cross_task/compositions": [("dax", "fep"), ("wug", "kiv"), ("fep", "dax", "kiv")] * 5
        + [("dax", "fep")],
    }
    measured = run.suites(data, "train")
    assert list(measured) == list(expected_programs)
    assert sum(map(len, measured.values())) == 80
    assert len({example.key for rows in measured.values() for example in rows}) == 80
    for name, examples in measured.items():
        assert [example.inputs for example in examples] == inputs
        assert [example.program for example in examples] == expected_programs[name]
    manifest = data_manifest(data, 37, 0)
    assert all(
        set(splits) <= {"train", "train_compositions"} for splits in manifest["examples"].values()
    )
    assert all(not steps for steps in manifest["training_schedule"].values())
    assert data_manifest(tasks.build_train_answer_data(37), 37, 0) == manifest
    with pytest.raises(ValueError):
        tasks.build_train_answer_data(38)


def test_train_answer_student_prompt_never_reads_oracle(tmp_path, monkeypatch):
    config = train_answer_config(tmp_path)
    data = tasks.build_train_answer_data(37)
    io = ModelIO(tiny_model(config), tiny_tokenizer(), config, data, StopFlag())

    def forbidden(example):
        raise AssertionError("Student prompt accessed an oracle answer")

    monkeypatch.setattr(Example, "answer", property(forbidden))
    for examples in run.suites(data, "train").values():
        for example in examples:
            expected = tasks.FORMAT + "\n\n" + tasks.query(example)
            assert prompt(example, data, "train_answer", False) == expected
            assert io.prefix(example, "train_answer", False).numel() > 0


def test_train_answer_teacher_prompt_is_fixed_and_reads_only_same_train_answer(monkeypatch):
    data = tasks.build_train_answer_data(37)
    answer = Example.answer.fget
    calls = []

    def observed(example):
        calls.append(example.key)
        return answer(example)

    monkeypatch.setattr(Example, "answer", property(observed))
    for examples in run.suites(data, "train").values():
        for example in examples:
            calls.clear()
            teacher = prompt(example, data, "train_answer", True)
            expected = (
                tasks.FORMAT
                + "\n\n"
                + tasks.query(example)
                + "\n\nThis is an example for a response to the question:\n"
                + answer(example)
                + "\n\nNow answer with a response of your own. Return only the final JSON array:"
            )
            assert teacher == expected
            assert calls == [example.key]
            assert all(rule not in teacher for rule in RULES.values())
            with pytest.raises(ValueError):
                prompt(example, data, "train_answer", True, "numbered")
    example = data["permutation"]["train"][0]
    for foreign in (
        replace(example, split="heldout"),
        replace(example, inputs=tasks.input_pool(37)[64]),
    ):
        data[foreign.task].setdefault(foreign.split, []).append(foreign)
        calls.clear()
        for privileged in (False, True):
            with pytest.raises(ValueError):
                prompt(foreign, data, "train_answer", privileged)
        assert calls == []


def test_train_answer_gate_requires_every_suite_perfect_and_complete():
    names = run.suites(tasks.build_train_answer_data(37), "train")
    measurements = {name: {"n": 16, "accuracy": 1.0, "truncated_rate": 0.0} for name in names}
    assert run.qualifies_train_answer(measurements)
    for name in names:
        for change in ({"accuracy": 15 / 16}, {"n": 15}, {"truncated_rate": 1 / 16}):
            changed = {**measurements, name: {**measurements[name], **change}}
            assert not run.qualifies_train_answer(changed)
    assert not run.qualifies_train_answer({name: measurements[name] for name in list(names)[:-1]})


def test_train_answer_native_calls_are_exactly_80_teacher_only_and_never_train(
    tmp_path, monkeypatch
):
    config = train_answer_config(tmp_path)
    model = tiny_model(config)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    monkeypatch.setattr(
        run, "load_model", lambda cfg: (model, tiny_tokenizer(), {"test_fixture": True})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Train-answer calibration entered a dev, heldout, scoring or training path"
        )

    monkeypatch.setattr(run, "build_data", forbidden)
    monkeypatch.setattr(run, "build_calibration_data", forbidden)
    monkeypatch.setattr(ModelIO, "gold_logprob", forbidden)
    for method in (
        "evaluate",
        "teacher_gate",
        "teacher_calibration",
        "optimizer_for_model",
        "run_arm",
        "update",
        "save_adapter",
    ):
        monkeypatch.setattr(Experiment, method, forbidden)
    calls = []
    generate = ModelIO.generate

    def observed(self, prefix, privileged, sample, seed, max_tokens):
        assert privileged and not sample and max_tokens == config.max_new_tokens
        calls.append(prefix.tolist())
        return generate(self, prefix, privileged, sample, seed, max_tokens)

    monkeypatch.setattr(ModelIO, "generate", observed)
    output = tmp_path / "train_answer"
    output.mkdir()
    experiment = Experiment(config, output, StopFlag())
    assert experiment.run() == 2
    assert experiment.metrics["status"] == "train_answer_calibration_failed"
    result = experiment.metrics["train_answer_calibration"]
    assert not result["qualified"]
    assert len(calls) == len(result["records"]) == 80
    expected = [
        (name, example)
        for name, examples in run.suites(experiment.data, "train").items()
        for example in examples
    ]
    for row, (name, example) in zip(result["records"], expected, strict=True):
        assert (row["suite"], row["example_id"]) == (name, example.key)
        assert row["phase"] == "train_answer_calibration"
        assert row["context"] == "train_answer" and row["privileged"]
        assert row["instruction_variant"] == "current"
        assert row["split"] in ("train", "train_compositions")
        assert row["expected"] == example.answer and row["generated_token_ids"]
        assert not any(key.startswith("gold_") for key in row)
        for key, privileged in (("prompt_sha256", True), ("student_prompt_sha256", False)):
            assert (
                row[key]
                == hashlib.sha256(
                    prompt(example, experiment.data, "train_answer", privileged).encode()
                ).hexdigest()
            )
    budget = experiment.metrics["budget_check"]
    assert budget["expected_generation_calls"] == budget["generation_calls"] == 80
    assert (
        budget["optimizer_updates"]
        == budget["supervised_tokens"]
        == budget["primary_heldout_calls"]
        == 0
    )
    assert experiment.optimizer is None and not experiment.metrics["arms"]
    assert not experiment.metrics["reference"] and "baseline" not in experiment.metrics
    assert not (output / "adapters").exists()
    checkpoint = torch.load(output / "checkpoint" / "state.pt", weights_only=True)
    assert checkpoint["adapter"] is checkpoint["initial_adapter"] is checkpoint["optimizer"] is None
    assert checkpoint["arm"] is None and checkpoint["next_step"] == 0
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        assert value.grad is None
    before = (output / "events.jsonl").read_bytes()
    assert Experiment(config, output, StopFlag()).run() == 2
    assert len(calls) == 80 and (output / "events.jsonl").read_bytes() == before


def test_train_answer_resume_preserves_completed_native_generations(tmp_path, monkeypatch):
    config = train_answer_config(tmp_path)
    monkeypatch.setattr(
        run, "load_model", lambda cfg: (tiny_model(cfg), tiny_tokenizer(), {"test_fixture": True})
    )
    calls = []
    evaluate = Experiment.evaluate_example

    def interrupt_after_third(
        self, example, phase, name, context, privileged, instruction_variant="current"
    ):
        row = evaluate(self, example, phase, name, context, privileged, instruction_variant)
        calls.append(example.key)
        if len(calls) == 3:
            self.stop.request(signal.SIGTERM, None)
        return row

    monkeypatch.setattr(Experiment, "evaluate_example", interrupt_after_third)
    output = tmp_path / "interrupted_train_answer"
    output.mkdir()
    with pytest.raises(StopRequested):
        Experiment(config, output, StopFlag()).run()
    checkpoint = torch.load(output / "checkpoint" / "state.pt", weights_only=True)
    completed = checkpoint["metrics"]["train_answer_calibration"]["records"]
    assert len(completed) == 3
    resumed = Experiment(config, output, StopFlag())
    assert resumed.run() == 2
    records = resumed.metrics["train_answer_calibration"]["records"]
    assert len(calls) == len(set(calls)) == len(records) == 80
    assert records[:3] == completed
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    emitted = [row["example_id"] for row in events if row["event"] == "evaluation_example"]
    assert emitted == calls


def test_oracle_control_config_fixes_one_complete_answer_budget(tmp_path):
    config = oracle_control_config(tmp_path)
    for change in (
        {"seed": 38}, {"arms": ("on_policy_dense",)}, {"steps_per_task": 31},
        {"tokens_per_update": 32}, {"train_examples": 15}, {"eval_examples": 16},
        {"gate_examples": 2}, {"lora_rank": 4}, {"lora_alpha": 8},
        {"learning_rate": 0.001}, {"checkpoint_every": 2},
    ):
        with pytest.raises(ValueError):
            replace(config, **change).validate()


def test_oracle_replay_config_allows_only_the_fixed_two_answer_allocation(tmp_path):
    config = oracle_control_config(tmp_path)
    assert config.replay_examples_per_update == 0
    assert replace(config, replay_examples_per_update=2).validate().replay_examples_per_update == 2
    for invalid in (1, 3, 4, -1, True, 2.0):
        with pytest.raises(ValueError, match="replay_examples_per_update"):
            replace(config, replay_examples_per_update=invalid).validate()
    with pytest.raises(ValueError, match="oracle_sft_control"):
        replace(tiny_config(tmp_path), replay_examples_per_update=2).validate()


def test_oracle_replay_schedule_preserves_stage_a_and_halves_current_exposure():
    data = tasks.build_oracle_train_data(37)
    control = tasks.oracle_train_manifest(data, 37)
    replay = tasks.oracle_train_manifest(data, 37, replay_examples=2)
    assert control["examples"] == replay["examples"]
    assert control["training_schedule"]["permutation"] == replay["training_schedule"]["permutation"]
    original_current = [key for batch in control["training_schedule"]["symbol_map"] for key in batch]
    old_ids = [example.key for example in data["permutation"]["train"]]
    current_ids = [example.key for example in data["symbol_map"]["train"]]
    current, old = [], []
    for step, batch in enumerate(replay["training_schedule"]["symbol_map"]):
        assert len(batch) == len(set(batch)) == 4
        assert batch[:2] == original_current[2 * step:2 * step + 2]
        assert batch[2:] == [old_ids[(2 * step + offset) % 16] for offset in range(2)]
        assert batch == [example.key for example in tasks.oracle_batch(data, "symbol_map", step, replay_examples=2)]
        current.extend(batch[:2])
        old.extend(batch[2:])
    assert Counter(current) == {key: 4 for key in current_ids}
    assert Counter(old) == {key: 4 for key in old_ids}
    assert len(current) == len(old) == 64
    assert all("/train/" in key for key in current + old)
    assert (128 + len(current) + len(old)) * 10 == 2560


def test_oracle_control_data_has_train_only_primitives_and_fresh_eval():
    data = tasks.build_oracle_train_data(37)
    train_inputs = tasks.input_pool(37)[4:20]
    for task in TASKS:
        assert set(data[task]) == {"train"}
        assert [example.inputs for example in data[task]["train"]] == train_inputs
        assert all(len(example.program) == 1 for example in data[task]["train"])
    assert not data["cross_task"]
    manifest = tasks.oracle_train_manifest(data, 37)
    for task in TASKS:
        batches = manifest["training_schedule"][task]
        assert len(batches) == 32 and all(len(set(batch)) == 4 for batch in batches)
        assert batches == [
            [data[task]["train"][(4 * step + offset) % 16].key for offset in range(4)]
            for step in range(32)
        ]
    heldout = tasks.build_oracle_eval_data(37)
    expected = tasks.input_pool(37)[128:160]
    measured = run.suites(heldout, "heldout")
    assert len(measured) == 5
    assert all([example.inputs for example in rows] == expected for rows in measured.values())
    assert not set(expected) & set(train_inputs)
    assert all("train" not in splits for splits in heldout.values())


def test_oracle_control_gates_use_original_initial_and_conditional_retention():
    evaluations = {
        stage: {name: {"accuracy": accuracy} for name, accuracy in values.items()}
        for stage, values in {
            "initial": {"permutation/inputs": 8 / 32, "symbol_map/inputs": 8 / 32},
            "after_permutation": {"permutation/inputs": 24 / 32, "symbol_map/inputs": 23 / 32},
            "after_symbol_map": {"permutation/inputs": 21 / 32, "symbol_map/inputs": 24 / 32},
        }.items()
    }
    gates = run.oracle_control_gates(evaluations)
    assert gates["feasible"] and gates["retention"]["eligible"]
    assert gates["acquisition"]["symbol_map"]["gain_over_initial"] == 0.5
    evaluations["after_symbol_map"]["permutation/inputs"]["accuracy"] = 20 / 32
    assert not run.oracle_control_gates(evaluations)["feasible"]
    evaluations["after_symbol_map"]["symbol_map/inputs"]["accuracy"] = 23 / 32
    gates = run.oracle_control_gates(evaluations)
    assert not gates["retention"]["eligible"] and gates["retention"]["passed"] is None


@pytest.mark.parametrize("replay_examples", [0, 2])
def test_oracle_control_native_resume_freezes_base_and_defers_all_eval(tmp_path, monkeypatch, replay_examples):
    """Native CPU updates and checkpoint faults; no claim of pretrained GPU equivalence."""
    config = replace(oracle_control_config(tmp_path), replay_examples_per_update=replay_examples).validate()
    models = []

    def load(cfg):
        native = tiny_model(cfg)
        original = {name: value.detach().clone() for name, value in native.named_parameters() if not value.requires_grad}
        models.append((native, original))
        return native, tiny_tokenizer(), {"test_fixture": True}

    monkeypatch.setattr(run, "load_model", load)

    def forbidden(*args, **kwargs):
        raise AssertionError("Oracle control entered a teacher or legacy training/scoring path")

    for method in ("teacher_gate", "teacher_calibration", "train_answer_calibration", "run_arm", "update"):
        monkeypatch.setattr(Experiment, method, forbidden)
    monkeypatch.setattr(run, "build_data", forbidden)
    monkeypatch.setattr(run, "build_calibration_data", forbidden)
    monkeypatch.setattr(run, "build_train_answer_data", forbidden)
    monkeypatch.setattr(ModelIO, "gold_logprob", forbidden)
    active = []
    eval_constructions = []
    build_eval = run.build_oracle_eval_data

    def post_training_only(seed):
        experiment = active[0]
        state = torch.load(experiment.output / "checkpoint/state.pt", weights_only=True)
        assert experiment.next_step == state["next_step"] == 64
        assert experiment.optimizer is state["optimizer"] is None
        assert state["metrics"]["oracle_sft_control"]["phase"] == "post_training"
        eval_constructions.append(experiment.output)
        return build_eval(seed)

    monkeypatch.setattr(run, "build_oracle_eval_data", post_training_only)
    prefix = ModelIO.prefix

    def cue_only(self, example, context="examples", privileged=False, instruction_variant="current"):
        assert context == "cue_only" and not privileged and instruction_variant == "current"
        return prefix(self, example, context, privileged, instruction_variant)

    monkeypatch.setattr(ModelIO, "prefix", cue_only)
    generate = ModelIO.generate
    generations = []

    def post_training_generation(self, prefix, privileged, sample, seed, max_tokens):
        experiment = active[0]
        assert experiment.next_step == 64 and experiment.optimizer is None
        assert experiment.metrics["oracle_sft_control"]["phase"] == "post_training"
        assert not privileged and not sample
        generations.append(experiment.output)
        return generate(self, prefix, privileged, sample, seed, max_tokens)

    monkeypatch.setattr(ModelIO, "generate", post_training_generation)
    update = Experiment.oracle_sft_update
    stopped = set()

    def interrupt(self, task, step):
        row = update(self, task, step)
        stop_at = {64} if self.output.name == "reference" else {33, 64}
        marker = (self.output, self.next_step + 1)
        if marker[1] in stop_at and marker not in stopped:
            stopped.add(marker)
            self.stop.request(signal.SIGTERM, None)
        return row

    monkeypatch.setattr(Experiment, "oracle_sft_update", interrupt)
    checkpoints = []
    for name in ("reference", "interrupted"):
        output = tmp_path / name
        output.mkdir()
        experiment = Experiment(config, output, StopFlag())
        active[:] = [experiment]
        with pytest.raises(StopRequested):
            experiment.run()
        if name == "interrupted":
            state = torch.load(output / "checkpoint/state.pt", weights_only=True)
            assert state["next_step"] == 33
            assert len(state["metrics"]["oracle_sft_control"]["updates"]) == 33
            assert state["optimizer"] is not None
            checkpoint_path = output / "checkpoint/state.pt"
            checkpoint_bytes = checkpoint_path.read_bytes()
            before_loads = len(models)
            try:
                torch.save({**state, "optimizer": None}, checkpoint_path)
                with pytest.raises(run.OutputConflict, match="persistent optimizer"):
                    Experiment(config, output, StopFlag()).prepare()
                assert len(models) == before_loads
                damaged = run.cpu_tree(state)
                damaged["metrics"]["oracle_sft_control"]["updates"][-1]["example_ids"][-1] = "permutation/heldout/dax/0001"
                torch.save(damaged, checkpoint_path)
                with pytest.raises(run.OutputConflict, match="train-only schedule"):
                    Experiment(config, output, StopFlag()).prepare()
                assert len(models) == before_loads
            finally:
                checkpoint_path.write_bytes(checkpoint_bytes)
            experiment = Experiment(config, output, StopFlag())
            active[:] = [experiment]
            with pytest.raises(StopRequested):
                experiment.run()
        state = torch.load(output / "checkpoint/state.pt", weights_only=True)
        assert state["next_step"] == 64 and state["optimizer"] is None
        assert state["metrics"]["oracle_sft_control"]["phase"] == "post_training"
        assert not (output / "eval_manifest.json").exists()
        checkpoints.append(state)
    assert not eval_constructions and not generations
    for key, tensor in checkpoints[0]["adapter"].items():
        torch.testing.assert_close(tensor, checkpoints[1]["adapter"][key], rtol=0, atol=0)
    assert checkpoints[0]["metrics"]["oracle_sft_control"]["updates"] == checkpoints[1]["metrics"]["oracle_sft_control"]["updates"]
    assert any(not torch.equal(tensor, checkpoints[1]["initial_adapter"][key]) for key, tensor in checkpoints[1]["adapter"].items())
    output = tmp_path / "interrupted"
    resumed = Experiment(config, output, StopFlag())
    active[:] = [resumed]
    assert resumed.run() == 0
    result = resumed.metrics["oracle_sft_control"]
    assert result["phase"] == "completed" and len(generations) == 480
    assert len(result["updates"]) == 64
    assert all(row["supervised_tokens"] == 40 and row["complete_answers"] == 4 for row in result["updates"])
    assert result["updates"][0]["grad_norm"] > 0
    assert set(result["saved_adapters"]) == {"initial", "after_permutation", "after_symbol_map"}
    assert all(len(value["records"]) == 160 for value in result["evaluations"].values())
    assert resumed.metrics["budget_check"]["supervised_tokens"] == 2560
    allocation = result["training_allocation"]
    assert allocation["permutation"]["current_complete_answers"] == 128
    assert allocation["permutation"]["replay_complete_answers"] == 0
    assert allocation["symbol_map"]["current_complete_answers"] == 32 * (4 - replay_examples)
    assert allocation["symbol_map"]["replay_complete_answers"] == 32 * replay_examples
    assert allocation["symbol_map"]["current_exposure_fraction_of_no_replay"] == (4 - replay_examples) / 4
    assert sum(row["current_supervised_tokens"] + row["replay_supervised_tokens"] for row in allocation.values()) == 2560
    with pytest.raises(RuntimeError):
        update(resumed, "permutation", 0)
    with pytest.raises(RuntimeError):
        resumed.optimizer_for_model()
    for native, original in models:
        for name, value in native.named_parameters():
            if name in original:
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
                assert value.grad is None
    previous_events = (output / "events.jsonl").read_bytes()
    assert Experiment(config, output, StopFlag()).run() == 0
    assert (output / "events.jsonl").read_bytes() == previous_events
