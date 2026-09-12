import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)

from generated_contract import (
    digest,
    file_digest,
    grade_generation,
    learner_prompt,
    make_corpus,
    qualification_gate,
    qualification_rows,
    source_hashes,
    teacher_prompt,
    validate_protocol,
    verify_qualification,
)
from qualified import QualifiedRunner, main, prepare, training_comparison
from run import adapter_parameters, tensor_digest, write_json
from tasks import FAMILIES, digits, execute, make_task

ROOT = Path(__file__).parent


@pytest.fixture
def protocol():
    return json.loads((ROOT / "configs/generated_protocol.json").read_text())


def pairs_for(corpus, protocol, teacher_hits=24, learner_hits=0):
    pairs = []
    for index, task in enumerate(qualification_rows(corpus, protocol)):
        pair = {"uid": task.uid}
        for condition, count in (("learner", learner_hits), ("teacher", teacher_hits)):
            body = digits(task.answer) if index % 24 < count else "incorrect"
            pair[condition] = {
                "prompt": learner_prompt(task)
                if condition == "learner"
                else teacher_prompt(task, corpus["demonstration"]),
                "body_text": body,
                "token_ids": [ord(c) for c in body] + [255],
                "correct": True,
            }
        pairs.append(pair)
    return pairs


def test_sealed_configs(protocol):
    validate_protocol(protocol)
    for name in ("qwen35_4b_qualification.json", "qwen35_4b_qualified_training.json"):
        dispatch = json.loads((ROOT / "configs" / name).read_text())
        assert dispatch["protocol_sha256"] == digest(protocol)
    changed = copy.deepcopy(protocol)
    changed["qualification"]["minimum_gain"] = 0.01
    with pytest.raises(ValueError, match="FIXED_GATE_CHANGED"):
        validate_protocol(changed)


def test_splits_are_disjoint_and_seed_independent(protocol):
    corpus = make_corpus(protocol)
    by_split = {split: {t.initial for t in rows} for split, rows in corpus.items()}
    for left, left_values in by_split.items():
        for right, right_values in by_split.items():
            if left != right:
                assert not left_values & right_values
    changed = copy.deepcopy(protocol)
    changed["split_seeds"]["train"] += 100
    changed["sizes"]["train"] = 48
    other = make_corpus(changed)
    for split in (
        "demonstration",
        "validation",
        "test",
        "composition",
        "fallback_validation",
    ):
        assert corpus[split] == other[split]
    assert corpus["train"] != other["train"]
    assert {len(t.program) for t in corpus["composition"]} == {3, 4}
    assert max(len(t.program) for t in corpus["train"]) == 2
    qualified = make_corpus(protocol, ("demonstration", "train", "validation"))
    assert set(qualified) == {"demonstration", "train", "validation"}
    assert len(qualification_rows(qualified, protocol)) == 144


def reference(family, initial, program):
    current = tuple(initial)
    for command in program:
        if family == "A":
            permutation = {
                "dax": (1, 2, 3, 0),
                "wug": (3, 1, 2, 0),
                "zup": (3, 2, 1, 0),
            }[command]
            current = tuple(current[i] for i in permutation)
        elif family == "B":
            multiplier, shift = {"dax": (1, 2), "wug": (3, 0), "zup": (-1, 9)}[command]
            current = tuple((multiplier * x + shift) % 10 for x in current)
        elif command == "zup":
            current = current[:3] + ((current[0] + current[3]) % 10,)
        elif command == "dax":
            current = (
                current[::-1] if current[0] % 2 == 0 else current[1:] + current[:1]
            )
        else:
            index = 0 if current[-1] % 2 == 0 else 2
            mutable = list(current)
            mutable[index : index + 2] = reversed(mutable[index : index + 2])
            current = tuple(mutable)
    return current


def test_independent_oracle_and_no_answer_dependent_context(protocol):
    corpus = make_corpus(protocol)
    for tasks in corpus.values():
        for task in tasks:
            assert task.answer == reference(task.family, task.initial, task.program)
    for family in FAMILIES:
        for value in range(10000):
            initial = tuple(map(int, f"{value:04d}"))
            for command in ("dax", "wug", "zup"):
                assert execute(family, initial, (command,)) == reference(
                    family, initial, (command,)
                )
    task = corpus["train"][0]
    prompt = teacher_prompt(task, corpus["demonstration"])
    poisoned = replace(task, answer=(99, 99, 99, 99))
    assert prompt == teacher_prompt(poisoned, corpus["demonstration"])
    assert learner_prompt(task) == learner_prompt(poisoned)
    assert "99" not in prompt
    assert "operation instructions" in prompt


@pytest.mark.parametrize(
    "text,ids",
    [
        ("1 2 3 4", [49, 32, 50, 32, 51, 32, 52]),
        ("1 2 3 4 5", [49, 32, 50, 32, 51, 32, 52, 32, 53, 255]),
        ("1 2 3 4\n", [49, 32, 50, 32, 51, 32, 52, 10, 255]),
        (" 1 2 3 4", [32, 49, 32, 50, 32, 51, 32, 52, 255]),
        ("1 2 3 4", [49, 254, 50, 32, 51, 32, 52, 255]),
        ("1 2 3 4", [49, 255, 50, 32, 51, 32, 52, 255]),
        ("The answer is 1 2 3 4", [49, 50, 51, 52, 255]),
    ],
)
def test_whole_generation_rejects_rescue(text, ids):
    task = make_task("A", "train", (4, 1, 2, 3), ("dax",))
    assert not grade_generation(task, ids, text, 255, [254, 255], 16)["correct"]


def test_eos_is_required_and_accepted_at_cap():
    task = make_task("A", "train", (4, 1, 2, 3), ("dax",))
    tokens = [49, 32, 50, 32, 51, 32, 52, 255]
    result = grade_generation(task, tokens, "1 2 3 4", 255, [255], 8)
    assert result["correct"] and result["terminated"] and not result["cap_without_eos"]
    assert grade_generation(task, tokens[:-1], "1 2 3 4", 255, [255], 7)[
        "cap_without_eos"
    ]


def test_paired_gate_controls_and_boundary(protocol):
    corpus = make_corpus(protocol, ("demonstration", "train", "validation"))
    good = pairs_for(corpus, protocol, teacher_hits=21, learner_hits=15)
    result = qualification_gate(good, corpus, protocol, 255, [255])
    assert result["passed"]
    assert all(
        item["wins"] == 6
        and item["losses"] == 0
        and item["paired_one_sided_p"] == 1 / 64
        for item in result["panels"].values()
    )
    for teacher_hits, learner_hits in ((20, 0), (21, 16), (24, 24), (0, 0)):
        records = pairs_for(corpus, protocol, teacher_hits, learner_hits)
        assert not qualification_gate(records, corpus, protocol, 255, [255])["passed"]
    with pytest.raises(ValueError, match="INCOMPLETE_OR_REORDERED"):
        qualification_gate(good[:-1], corpus, protocol, 255, [255])
    for index, pair in enumerate(good):
        task = qualification_rows(corpus, protocol)[index]
        learner_hit = index % 24 < 12 or index % 24 >= 21
        pair["learner"]["body_text"] = (
            digits(task.answer) if learner_hit else "incorrect"
        )
    assert not qualification_gate(good, corpus, protocol, 255, [255])["passed"]


def test_dispatcher_metadata_and_immutable_output(tmp_path, protocol):
    dispatch = {"stage": "qualification"}
    (tmp_path / "config.json").write_text(json.dumps(dispatch))
    (tmp_path / "packages.txt").write_text("parent torch unchanged")
    seal = prepare(tmp_path, dispatch, protocol)
    assert seal["protocol_sha256"] == digest(protocol)
    assert not seal["prediction_outcomes_observed"]
    assert (tmp_path / "packages.txt").read_text() == "parent torch unchanged"
    with pytest.raises(FileExistsError, match="ALREADY_USED"):
        prepare(tmp_path, dispatch, protocol)


def test_canonical_cli_with_dispatcher_config_outside_lane(tmp_path, monkeypatch):
    dispatch = json.loads((ROOT / "configs/qwen35_4b_qualification.json").read_text())
    write_json(tmp_path / "config.json", dispatch)
    (tmp_path / "run.log").write_text("parent log\n")
    (tmp_path / "attempts").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualified.py",
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(tmp_path),
            "--validate-only",
        ],
    )
    main()
    result = json.loads((tmp_path / "validation.json").read_text())
    assert not result["gpu_qualified"]
    assert not (tmp_path / "qualification.json").exists()
    assert (tmp_path / "run.log").read_text() == "parent log\n"


def test_cli_training_requires_receipt_before_model_load(tmp_path, monkeypatch):
    dispatch = json.loads(
        (ROOT / "configs/qwen35_4b_qualified_training.json").read_text()
    )
    dispatch["qualification_dir"] = str(tmp_path / "absent")
    config = tmp_path / "dispatch.json"
    write_json(config, dispatch)
    output = tmp_path / "attempt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["qualified.py", "--config", str(config), "--output-dir", str(output)],
    )
    with pytest.raises(FileNotFoundError, match="qualification.json"):
        main()
    assert not (output / "runtime.json").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"


def test_receipt_recomputes_gate_and_requires_hardware_proof(tmp_path, protocol):
    corpus = make_corpus(protocol, ("demonstration", "train", "validation"))
    records = pairs_for(corpus, protocol)
    write_json(tmp_path / "pairs.json", records)
    write_json(
        tmp_path / "seal.json",
        {
            "protocol_sha256": digest(protocol),
            "protocol": protocol,
            "source_sha256": source_hashes(),
            "prediction_outcomes_observed": False,
        },
    )
    write_json(tmp_path / "runtime.json", {"fixture": True})
    receipt = {
        "status": "qualified",
        "candidate": "instruction_demo",
        "qualification_optimizer_updates": 0,
        "learner_optimizer_updates": 0,
        "protocol_sha256": digest(protocol),
        "source_sha256": source_hashes(),
        "optimizer_updates": 0,
        "device": "cpu",
        "hip": None,
        "eos_token_id": 255,
        "special_token_ids": [255],
        "adapter_before_sha256": "same",
        "adapter_after_sha256": "same",
        "files_sha256": {
            name: file_digest(tmp_path / name)
            for name in ("pairs.json", "seal.json", "runtime.json")
        },
        "gate": qualification_gate(records, corpus, protocol, 255, [255]),
    }
    write_json(tmp_path / "qualification.json", receipt)
    with pytest.raises(ValueError, match="GPU_RECEIPT_REQUIRED"):
        verify_qualification(tmp_path, protocol)
    receipt.update(device="cuda:0", hip="unit-test-only")
    write_json(tmp_path / "qualification.json", receipt)
    assert verify_qualification(tmp_path, protocol)["gate"]["passed"]
    records[0]["teacher"]["body_text"] += " 7"
    write_json(tmp_path / "pairs.json", records)
    with pytest.raises(ValueError, match="ARTIFACT_MISMATCH"):
        verify_qualification(tmp_path, protocol)
    records = pairs_for(corpus, protocol, teacher_hits=0)
    write_json(tmp_path / "pairs.json", records)
    receipt["files_sha256"]["pairs.json"] = file_digest(tmp_path / "pairs.json")
    write_json(tmp_path / "qualification.json", receipt)
    with pytest.raises(ValueError, match="RECOMPUTED_GATE_FAILED"):
        verify_qualification(tmp_path, protocol)


def tiny_runner(protocol, output):
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"])
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {
        value: index for index, value in enumerate(["<eos>", "<unk>", *alphabet])
    }
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{message['content']}}"
        "{% endfor %}{{'\\nAssistant:'}}"
    )
    model_config = Qwen3_5Config(
        text_config={
            "vocab_size": len(vocabulary),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "layer_types": ["linear_attention", "full_attention"],
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 64,
        },
    )
    lora = LoraConfig(
        r=2, lora_alpha=4, target_modules=protocol["lora_targets"], lora_dropout=0.0
    )
    model = get_peft_model(
        Qwen3_5ForConditionalGeneration(model_config).eval(),
        lora,
        adapter_name="student",
    )
    model.add_adapter("teacher", copy.deepcopy(lora))
    model.set_adapter("student")
    runner = QualifiedRunner(protocol, output)
    runner.device = torch.device("cpu")
    runner.model, runner.tokenizer = model, tokenizer
    runner.eos, runner.special_ids = tokenizer.eos_token_id, tokenizer.all_special_ids
    runner.snapshot = {"fixture": "tiny-random-qwen-not-cached-4b"}
    runner.initial_adapter = {
        name: value.detach().clone()
        for name, value in adapter_parameters(model, "student").items()
    }
    runner.student_parameters = list(adapter_parameters(model, "student").values())
    runner.frozen_parameters = [
        p for name, p in model.named_parameters() if ".lora_" not in name
    ]
    runner.frozen_versions = [p._version for p in runner.frozen_parameters]
    model.eval()
    return runner


def test_actual_qwen_cpu_generation_training_freeze_and_checkpoint(tmp_path, protocol):
    small = copy.deepcopy(protocol)
    small["sizes"].update(train=2, validation=2, test=2, composition=2)
    small.update(
        updates_per_family=1,
        examples_per_update=2,
        learning_rate=0.01,
        max_context_tokens=2048,
        max_new_tokens=8,
    )
    small["training"]["bootstrap_samples"] = 100
    runner = tiny_runner(small, tmp_path)
    task = runner.corpus["train"][0]
    target = runner.target_tokens(task)
    assert target[-1].item() == runner.eos
    assert runner.tokenizer.decode(
        target[:-1], clean_up_tokenization_spaces=False
    ) == digits(task.answer)
    runner.model.get_base_model().generation_config.forced_eos_token_id = runner.eos
    _, generated, _, _ = runner.generate(learner_prompt(task), sample=True)
    assert generated.numel() > 0
    assert runner.model.get_base_model().generation_config.forced_eos_token_id is None
    assert runner.model.get_base_model().generation_config.top_k == 0
    frozen_hash = tensor_digest(
        {str(i): p for i, p in enumerate(runner.frozen_parameters)}
    )
    receipt = {
        "candidate": "instruction_demo",
        "status": "qualified",
        "optimizer_updates": 0,
        "receipt_sha256": "cpu-component-fixture-only",
        "pairs": [
            {
                "learner": runner.generated_record(task, learner_prompt(task)),
                "teacher": runner.generated_record(
                    task, teacher_prompt(task, runner.corpus["demonstration"])
                ),
            }
        ],
        "snapshot_sha256": runner.snapshot,
        "eos_token_id": runner.eos,
        "special_token_ids": runner.special_ids,
        "adapter_before_sha256": tensor_digest(
            adapter_parameters(runner.model, "student")
        ),
    }
    result = runner.train(receipt)
    assert result["status"] == "completed"
    assert {b["updates"] for b in result["budgets"].values()} == {3}
    assert result["budgets"]["sft"]["loss_tokens"] == 48
    assert frozen_hash == tensor_digest(
        {str(i): p for i, p in enumerate(runner.frozen_parameters)}
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "training.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 12
    orders = {
        method: [row["uid"] for row in rows if row["method"] == method]
        for method in ("privileged_forward_kl", "sft")
    }
    assert orders["privileged_forward_kl"] == orders["sft"]
    assert all(
        row["response_ids"][-1] == runner.eos for row in rows if row["method"] == "sft"
    )
    for method in small["training"]["methods"]:
        receipts = [
            json.loads(
                (
                    tmp_path / "checkpoints" / method / family / "receipt.json"
                ).read_text()
            )
            for family in FAMILIES
        ]
        assert [row["step_after"] for row in receipts] == [1, 2, 3]
        assert len({row["teacher_sha256"] for row in receipts}) == 1
        assert all(
            row["adapter_reload_equal"] and row["logits_reload_equal"]
            for row in receipts
        )
        assert (
            receipts[0]["student_after_sha256"] == receipts[1]["student_before_sha256"]
        )
    events = [
        json.loads(line)
        for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()
    ]
    assert all("operation instructions" not in row["prompt"] for row in events)
    assert all(
        row["stage"] == "final_only" for row in events if row["split"] == "composition"
    )


def test_actual_qualification_entrypoint_cpu_has_zero_updates(tmp_path, protocol):
    prepare(tmp_path, {"stage": "cpu_fixture"}, protocol)
    write_json(tmp_path / "runtime.json", {"device": "cpu", "fixture": "tiny-qwen"})
    runner = tiny_runner(protocol, tmp_path)
    result = runner.qualify()
    assert result["optimizer_updates"] == 0
    assert result["device"] == "cpu"
    assert result["status"] == "rejected"
    assert not hasattr(runner, "optimizer")
    pairs = json.loads((tmp_path / "pairs.json").read_text())
    assert len(pairs) == 144
    assert {pair["split"] for pair in pairs} == {"train", "validation"}
    assert not (tmp_path / "training.jsonl").exists()
    with pytest.raises(ValueError, match="GPU_RECEIPT_REQUIRED"):
        verify_qualification(tmp_path, protocol)


def test_teacher_fallback_cpu_preserves_learner_and_requalifies_new_partition(
    tmp_path, protocol
):
    prepare(tmp_path, {"stage": "cpu_teacher_fixture"}, protocol)
    write_json(tmp_path / "runtime.json", {"device": "cpu", "fixture": "tiny-qwen"})
    runner = tiny_runner(protocol, tmp_path)
    failure = {
        "candidate": "instruction_demo",
        "status": "rejected",
        "pairs": [],
        "receipt_sha256": "cpu-fixture-initial-rejection-not-gpu-evidence",
        "snapshot_sha256": runner.snapshot,
        "eos_token_id": runner.eos,
        "special_token_ids": runner.special_ids,
        "adapter_before_sha256": tensor_digest(
            adapter_parameters(runner.model, "student")
        ),
    }
    with pytest.raises(ValueError, match="ONE_FALLBACK"):
        runner.train_teacher({**failure, "candidate": "trained_teacher"})
    result = runner.train_teacher(failure)
    assert result["candidate"] == "trained_teacher"
    assert result["optimizer_updates"] == 24
    assert result["learner_optimizer_updates"] == 0
    assert result["qualification_optimizer_updates"] == 0
    assert (
        result["adapter_before_sha256"]
        == result["adapter_after_sha256"]
        == failure["adapter_before_sha256"]
    )
    records = json.loads((tmp_path / "pairs.json").read_text())
    assert len(records) == 144
    assert {row["split"] for row in records} == {"train", "fallback_validation"}
    training = [
        json.loads(line)
        for line in (tmp_path / "teacher_training_rows.jsonl").read_text().splitlines()
    ]
    assert len(training) == 96
    assert {row["split"] for row in training} == {"train"}
    assert all(row["response_ids"][-1] == runner.eos for row in training)
    assert all("operation instructions" in row["prompt"] for row in training)
    training_receipt = json.loads((tmp_path / "teacher_training.json").read_text())
    assert (
        training_receipt["learner_before_sha256"]
        == training_receipt["learner_after_sha256"]
    )
    assert result["teacher_sha256"] == training_receipt["teacher_sha256"]
    assert set(training_receipt["training_ids"]) <= {
        task.uid for task in runner.corpus["train"]
    }
    assert not set(training_receipt["training_ids"]) & {
        task.uid for task in runner.corpus["fallback_validation"]
    }
    with pytest.raises(ValueError, match="KD_REQUIRES_POSITIVE"):
        runner.train({**result, "status": "rejected"})
    small = copy.deepcopy(protocol)
    small["sizes"].update(train=2, validation=2, test=2, composition=2)
    small.update(updates_per_family=1, examples_per_update=2)
    small["training"]["bootstrap_samples"] = 100
    kd_output = tmp_path / "cpu-loader-component"
    kd_output.mkdir()
    learner = tiny_runner(small, kd_output)
    component_receipt = {
        **result,
        "status": "qualified",
        "qualification_dir": str(tmp_path),
        "pairs": [],
        "receipt_sha256": "cpu-component-fixture-only",
    }
    comparison = learner.train(component_receipt)
    assert comparison["teacher_training_updates"] == 24
    assert comparison["teacher_candidate"] == "trained_teacher"
    for family in FAMILIES:
        checkpoint = json.loads(
            (
                kd_output
                / "checkpoints"
                / "privileged_forward_kl"
                / family
                / "receipt.json"
            ).read_text()
        )
        assert checkpoint["teacher_sha256"] == training_receipt["teacher_sha256"]


def test_training_comparison_rejects_unmatched_control(protocol):
    with pytest.raises(ValueError, match="CONTROL_UPDATE_MISMATCH"):
        training_comparison(
            protocol,
            {},
            {},
            {},
            {"privileged_forward_kl": {"updates": 72}, "sft": {"updates": 71}},
        )
