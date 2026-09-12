import json
import random
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from config import DataConfig, LoraSettings, PilotConfig, RunConfig, RuntimeConfig
from data import EVAL_SPLITS, make_dataset, require_gradient_examples, step_examples
from evaluation import score_output, summarize_method
from gradients import assign_gradient, choose_gradient
from modeling import (
    EncodedExample,
    collate,
    encode_examples,
    load_model,
    training_gradient,
)
from run import Artifacts, StopFlag, run_experiment
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)


def small_data():
    return DataConfig(
        num_tasks=3,
        train_examples=24,
        reference_examples=8,
        calibration_examples=2,
        id_examples=2,
        longer_examples=1,
        composed_examples=1,
    )


def build_fixture(path, family):
    path.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "[USER]": 3, "[ASSISTANT]": 4}
    words = list("0123456789") + [
        "Device",
        ":",
        "unit",
        "-",
        "Apply",
        "the",
        "device",
        "once",
        ".",
        "twice",
        "feed",
        "its",
        "first",
        "output",
        "back",
        "into",
        "same",
        "Input",
        "Output",
        "only",
        "resulting",
        "digits",
        "separated",
        "by",
        "spaces",
    ]
    for word in words:
        vocab[word] = len(vocab)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for message in messages %}[USER] {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}[ASSISTANT] {% endif %}"
    tokenizer.save_pretrained(path)
    text = {
        "vocab_size": len(vocab),
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "pad_token_id": 0,
        "bos_token_id": 3,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }
    torch.manual_seed(16)
    if family == "qwen3_5":
        text.update(
            layer_types=["linear_attention", "full_attention"],
            linear_conv_kernel_dim=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 2, 4],
            },
        )
        config = Qwen3_5Config(
            text_config=text,
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_heads": 2,
                "out_hidden_size": 32,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 1,
            },
            image_token_id=len(vocab) + 1,
            video_token_id=len(vocab) + 2,
            vision_start_token_id=len(vocab) + 3,
            vision_end_token_id=len(vocab) + 4,
        )
        model = Qwen3_5ForConditionalGeneration(config)
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**text))
    model.save_pretrained(path)
    return RunConfig(
        model_id=f"cpu-fixture/{family}",
        revision="local",
        model_path=str(path),
        model_source="local_cpu_fixture",
        purpose="integration_gate",
        data=small_data(),
        pilot=PilotConfig(
            steps_per_task=2,
            current_batch_size=2,
            reference_batch_size=2,
            eval_batch_size=2,
            max_sequence_length=192,
            max_new_tokens=24,
            checkpoint_every_steps=2,
        ),
        lora=LoraSettings(rank=2, alpha=4, last_n_layers=2),
        runtime=RuntimeConfig(
            device="cpu", dtype="float32", attention="eager", cpu_threads=1
        ),
    )


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5"])
def tiny_config(tmp_path_factory, request):
    return build_fixture(
        tmp_path_factory.mktemp(f"tiny_{request.param}"), request.param
    )


def test_generation_is_deterministic_and_globally_disjoint():
    data = make_dataset(DataConfig(), 1729)
    assert data.fingerprint() == make_dataset(DataConfig(), 1729).fingerprint()
    assert data.fingerprint() != make_dataset(DataConfig(), 1730).fingerprint()
    assert len({example.values for example in data.examples}) == len(data.examples)
    assert len({example.prompt for example in data.examples}) == len(data.examples)
    for example in data.examples:
        rule = data.rules[example.task]
        result = list(example.values)
        for _ in range(example.applications):
            mapped = [rule.mapping[value] for value in result]
            result = list(reversed(mapped)) if rule.reverse else mapped
        assert example.answer == " ".join(map(str, result))
        assert str(rule.mapping) not in example.prompt
        assert "reverse" not in example.prompt.lower()
        if example.split == "longer":
            assert len(example.values) > data.config.max_length
        elif example.split == "composed":
            assert example.applications == 2
    single = {example.values for example in data.examples if example.applications == 1}
    for example in data.examples:
        if example.split == "composed":
            assert data.rules[example.task].execute(example.values) not in single


def test_gradient_batches_reject_all_test_and_calibration_rows():
    data = make_dataset(small_data(), 123)
    for split in (*EVAL_SPLITS, "calibration"):
        with pytest.raises(ValueError, match="forbidden"):
            require_gradient_examples(data.select(0, split), 0, "current")
        with pytest.raises(ValueError, match="forbidden"):
            require_gradient_examples(data.select(0, split), 1, "reference")
    with pytest.raises(ValueError, match="forbidden"):
        require_gradient_examples(data.select(1, "reference"), 1, "reference")
    with pytest.raises(ValueError, match="forbidden"):
        require_gradient_examples(data.select(1, "train"), 0, "current")


def test_reference_schedules_are_method_and_rng_independent():
    data = make_dataset(small_data(), 17)
    all_schedules = []
    for _ in ("sequential", "replay", "agem"):
        random.random()
        schedule = []
        for task in range(3):
            for step in range(25):
                current, reference = step_examples(data, task, step, 2, 2)
                require_gradient_examples(current, task, "current")
                assert len({example.example_id for example in current}) == 2
                if task:
                    require_gradient_examples(reference, task, "reference")
                else:
                    assert not reference
                schedule.append(
                    (
                        [example.example_id for example in current],
                        [example.example_id for example in reference],
                    )
                )
        all_schedules.append(schedule)
    assert all_schedules[0] == all_schedules[1] == all_schedules[2]


def test_projection_matches_analytic_halfspace_and_actual_sgd_step():
    current = torch.tensor([-2.0, 3.0])
    reference = torch.tensor([1.0, 0.0])
    projected, info = choose_gradient("agem", current, reference, 0.5, 100)
    torch.testing.assert_close(projected, torch.tensor([0.0, 3.0]))
    assert info["conflict"] and info["projected"]
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    old_loss = parameter[0].item()
    assign_gradient([parameter], projected)
    optimizer.step()
    assert parameter[0].item() <= old_loss
    assert torch.dot(reference, projected).item() >= 0
    for seed in range(25):
        generator = torch.Generator().manual_seed(seed)
        current = torch.randn(113, generator=generator)
        reference = torch.randn(113, generator=generator)
        projected, _ = choose_gradient("agem", current, reference, 0.5, 1)
        assert torch.dot(projected, reference).item() >= -1e-5
        assert projected.norm().item() <= 1.00001


def test_sequential_replay_and_zero_reference_semantics():
    current = torch.tensor([1.0, 2.0])
    reference = torch.tensor([3.0, -2.0])
    sequential, _ = choose_gradient("sequential", current, reference, 0.25, 100)
    replay, _ = choose_gradient("replay", current, reference, 0.25, 100)
    torch.testing.assert_close(sequential, current)
    torch.testing.assert_close(replay, 0.75 * current + 0.25 * reference)
    for method in ("sequential", "replay", "agem"):
        value, info = choose_gradient(method, current, None, 0.5, 100)
        torch.testing.assert_close(value, current)
        assert info["cosine"] is None
    value, _ = choose_gradient("agem", current, torch.zeros_like(current), 0.5, 100)
    torch.testing.assert_close(value, current)
    with pytest.raises(FloatingPointError):
        choose_gradient("agem", torch.tensor([float("nan")]), reference[:1], 0.5, 1)


def test_completion_masks_and_left_padding_do_not_supervise_prompts():
    examples = make_dataset(small_data(), 17).examples[:2]
    rows = [
        EncodedExample(examples[0], (4, 5, 6), (7, 2)),
        EncodedExample(examples[1], (4,), (8, 2)),
    ]
    train = collate(rows, 0, torch.device("cpu"))
    assert train["labels"].tolist() == [
        [-100, -100, -100, 7, 2],
        [-100, 8, 2, -100, -100],
    ]
    generation = collate(rows, 0, torch.device("cpu"), generation=True)
    assert generation["input_ids"].tolist() == [[4, 5, 6], [0, 0, 4]]
    assert generation["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert "labels" not in generation


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5"])
def test_real_model_loader_and_adapter_update_freeze_base(tmp_path, family):
    config = build_fixture(tmp_path / family, family)
    model, tokenizer, info = load_model(config)
    assert info["trainable_numel"] > 0
    assert all(
        "visual" not in name and "lm_head" not in name
        for name in info["lora_target_modules"]
    )
    data = make_dataset(config.data, config.seed)
    encoded = encode_examples(
        data.examples,
        tokenizer,
        config.pilot.max_sequence_length,
        config.pilot.max_new_tokens,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    old = torch.cat([parameter.detach().flatten() for parameter in parameters]).clone()
    model.train()
    current, _ = step_examples(data, 0, 0, 2, 2)
    gradient, loss, counts = training_gradient(
        model,
        [encoded[example.example_id] for example in current],
        tokenizer,
        parameters,
        torch.device("cpu"),
        0,
        "current",
    )
    assert gradient.norm().item() > 0 and loss > 0 and counts["supervised_tokens"] > 0
    assign_gradient(parameters, gradient)
    torch.optim.SGD(parameters, lr=0.1).step()
    after = torch.cat([parameter.detach().flatten() for parameter in parameters])
    assert not torch.equal(old, after)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
            assert parameter.grad is None
    forbidden = data.select(0, "id")
    with pytest.raises(ValueError, match="forbidden"):
        training_gradient(
            model,
            [encoded[example.example_id] for example in forbidden],
            tokenizer,
            parameters,
            torch.device("cpu"),
            0,
            "current",
        )


def test_scoring_rejects_verbose_answers():
    assert score_output(" 1 2 3 \n", "1 2 3")["exact"]
    assert not score_output("The answer is 1 2 3", "1 2 3")["exact"]
    assert not score_output("1 2 3 4", "1 2 3")["exact"]
    assert score_output("", "1 2 3")["symbol_accuracy"] == 0


def test_metrics_separate_acquisition_from_backward_transfer():
    boundaries = []
    values = [
        [0.1, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.5, 0.7, 0.0],
        [0.5, 0.7, 0.0],
        [0.6, 0.4, 0.9],
    ]
    for index, scores in enumerate(values):
        tasks = {
            str(task): {
                split: {"exact_match": score, "answer_nll": 1 - score}
                for split in EVAL_SPLITS
            }
            for task, score in enumerate(scores)
        }
        boundaries.append(
            {
                "task": index // 2,
                "phase": "before" if index % 2 == 0 else "after",
                "evaluation": {"tasks": tasks},
            }
        )
    result = summarize_method(boundaries, 3)
    assert result["per_task"][0]["splits"]["id"]["new_task_gain"] == pytest.approx(0.7)
    assert result["per_task"][1]["splits"]["id"][
        "mean_backward_transfer"
    ] == pytest.approx(-0.3)
    assert result["final"]["id"]["mean_backward_transfer"] == pytest.approx(-0.25)
    assert result["final"]["id"]["mean_forgetting_from_best"] == pytest.approx(0.25)


def test_full_runner_and_resume_are_identical(tiny_config, tmp_path, monkeypatch):
    full = run_experiment(tiny_config, tmp_path / "full")
    assert full["status"] == "completed" and full["budget_match"]["passed"]
    assert full["optimizer_updates"] == 18
    assert full["headroom"]["passed"]
    first_task_hashes = []
    for method in full["methods"].values():
        assert len(method["boundaries"]) == 6
        assert method["budgets"]["optimizer_updates"] == 6
        assert method["budgets"]["forward_backward_passes"] == 10
        assert all(
            not entry["evaluation"]["gradients_enabled"]
            for entry in method["boundaries"]
        )
        first_task_hashes.append(method["boundaries"][1]["adapter_sha256"])
    assert len(set(first_task_hashes)) == 1
    stop = StopFlag()
    original_emit = Artifacts.emit

    def stop_after_update(artifacts, event, **fields):
        original_emit(artifacts, event, **fields)
        if event == "step_finished" and artifacts.position["optimizer_updates"] == 3:
            stop.handle(signal.SIGTERM, None)

    monkeypatch.setattr(Artifacts, "emit", stop_after_update)
    interrupted = run_experiment(tiny_config, tmp_path / "resumed", stop)
    assert interrupted["status"] == "interrupted"
    monkeypatch.setattr(Artifacts, "emit", original_emit)
    events_before = (tmp_path / "resumed/events.jsonl").read_bytes()
    resumed = run_experiment(tiny_config, tmp_path / "resumed")
    assert resumed["status"] == "completed" and resumed["optimizer_updates"] == 18
    assert (tmp_path / "resumed/events.jsonl").read_bytes().startswith(events_before)
    for name, method in full["methods"].items():
        other = resumed["methods"][name]
        assert other["final_adapter_sha256"] == method["final_adapter_sha256"]
        assert other["budgets"] == method["budgets"]
        assert other["schedule_sha256"] == method["schedule_sha256"]
        assert other["summary"] == method["summary"]
        assert (
            tmp_path / "resumed" / other["adapter_path"] / "adapter_model.safetensors"
        ).is_file()
    again = run_experiment(tiny_config, tmp_path / "resumed")
    assert again["optimizer_updates"] == 18
    assert len(again["methods"]["agem"]["steps"]) == 6
    changed = replace(tiny_config, seed=tiny_config.seed + 1)
    with pytest.raises(ValueError, match="different config"):
        run_experiment(changed, tmp_path / "resumed")


def test_configs_are_pinned_and_valid():
    for path in (Path(__file__).parents[1] / "configs").glob("*.json"):
        config = RunConfig.from_dict(json.loads(path.read_text()))
        assert len(config.revision) == 40
        assert config.runtime.device == "cuda:0"
        assert config.methods == ("sequential", "replay", "agem")
