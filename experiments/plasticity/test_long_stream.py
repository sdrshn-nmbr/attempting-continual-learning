import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
import torch
from long_stream import (
    EncodedStream,
    base_hash,
    generate_records,
    headroom_gate,
    prepare_output,
    score_records,
    strict_grade,
    summarize_matrix,
    train_update,
    training_invariants,
    validate_config,
    validate_native_eos,
)
from run import adapter_hash, checkpoint_load, new_model
from stream_tasks import (
    TASK_IDS,
    TASKS,
    Reservoir,
    audit_corpus,
    build_corpus,
    corpus_digest,
    render,
    stage_batches,
)
from tasks import LABELS, digest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session", autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def study_config():
    return json.loads((ROOT / "configs" / "long_stream_seed17.json").read_text())


def tiny_snapshot(root, model_seed):
    revision = hashlib.sha1(
        f"independent-tiny-qwen35-cpu-{model_seed}".encode()
    ).hexdigest()
    snapshot = root / revision
    config = study_config()
    example = build_corpus(config)[TASK_IDS[0]]["train"][0]
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "<assistant>", *LABELS, *TASK_IDS]
    words.extend(str(number) for number in range(16))
    words.extend(render(example).split())
    vocabulary = dict.fromkeys(words)
    vocabulary = {word: index for index, word in enumerate(vocabulary)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        chat_template="{{ bos_token }} {{ messages[0]['content'] }} <assistant>",
    )
    text = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 3, 3],
            "partial_rotary_factor": 1.0,
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(
                text_config=text,
                vision_config=vision,
                image_token_id=124,
                video_token_id=125,
                vision_start_token_id=126,
                vision_end_token_id=127,
            )
        ).to(torch.bfloat16)
    model.save_pretrained(snapshot, safe_serialization=True)
    tokenizer.save_pretrained(snapshot)
    return snapshot


@pytest.fixture(scope="session")
def cpu_assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent-qwen-models")
    return {
        stages: tiny_snapshot(root / f"stages-{stages}", seed)
        for stages, seed in ((2, 107), (8, 211))
    }


def cpu_config(snapshot, stages):
    config = study_config()
    config.update(
        {
            "mode": "cpu_qualification",
            "device": "cpu",
            "attention": "eager",
            "model_id": f"local/random-tiny-qwen35-{stages}-stage-control",
            "model_revision": snapshot.name,
            "model_path": str(snapshot),
            "task_order": list(TASK_IDS[:stages]),
            "train_examples": 64,
            "validation_examples": 8,
            "test_examples": 8,
            "updates_per_stage": 4,
            "eval_batch_size": 8,
            "learning_rate": 0.01,
        }
    )
    return config


def test_fixed_protocol_and_balanced_shared_groups():
    base = study_config()
    configs = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "configs").glob("long_stream*.json"))
    ]
    assert sorted(config["seed"] for config in configs) == [17, 29, 43]
    for config in configs:
        validate_config(config)
        assert {key: value for key, value in config.items() if key != "seed"} == {
            key: value for key, value in base.items() if key != "seed"
        }
    corpus = build_corpus(base)
    audit = audit_corpus(corpus, base)
    assert audit["sha256"] == corpus_digest(build_corpus(base))
    assert audit["shared_input_groups"] == {"train": 512, "validation": 64, "test": 128}
    features = {
        split: {row.values for task in corpus.values() for row in task[split]}
        for split in ("train", "validation", "test")
    }
    assert not features["train"] & (features["validation"] | features["test"])
    assert not features["validation"] & features["test"]
    for splits in corpus.values():
        for rows in splits.values():
            assert Counter(row.label for row in rows) == Counter(
                {label: len(rows) // 4 for label in LABELS}
            )
    policies = {
        tuple(
            task.answer(values)
            for values in (
                (0, 0, 0),
                (0, 0, 9),
                (0, 9, 0),
                (0, 9, 9),
                (9, 0, 0),
                (9, 0, 9),
                (9, 9, 0),
                (9, 9, 9),
            )
        )
        for task in TASKS
    }
    assert len(policies) == 8


def test_heldout_poison_is_rejected_across_tasks():
    config = study_config()
    corpus = build_corpus(config)
    poisoned = copy.deepcopy(corpus)
    heldout = corpus[TASK_IDS[1]]["test"][0]
    poisoned[TASK_IDS[1]]["train"] += (replace(heldout, split="train"),)
    with pytest.raises(ValueError, match="HELDOUT_LEAK|SHARED_GROUPS"):
        audit_corpus(poisoned, config)
    poisoned = copy.deepcopy(corpus)
    row = poisoned[TASK_IDS[0]]["validation"][0]
    poisoned[TASK_IDS[0]]["validation"] = (
        replace(row, label=LABELS[(LABELS.index(row.label) + 1) % 4]),
    ) + poisoned[TASK_IDS[0]]["validation"][1:]
    with pytest.raises(ValueError, match="TASK_MAPPING"):
        audit_corpus(poisoned, config)
    with pytest.raises(ValueError, match="BUFFER_HELDOUT"):
        Reservoir(64, 17).offer([heldout])
    with pytest.raises(ValueError, match="CURRENT_SOURCE"):
        list(stage_batches([heldout], (), "continue", config, heldout.task_id))


def test_new_task_mapping_does_not_relabel_history():
    config = study_config()
    corpus = build_corpus(config)
    historical_hash = digest([asdict(row) for row in corpus[TASK_IDS[0]]["train"]])
    changed = TASKS[:-1] + (replace(TASKS[-1], mapping=("B", "C", "A", "D")),)
    new_corpus = build_corpus(config, tasks=changed)
    assert {key: value for key, value in corpus.items() if key != TASK_IDS[-1]} == {
        key: value for key, value in new_corpus.items() if key != TASK_IDS[-1]
    }
    assert corpus[TASK_IDS[-1]]["train"] != new_corpus[TASK_IDS[-1]]["train"]
    assert historical_hash == digest(
        [asdict(row) for row in corpus[TASK_IDS[0]]["train"]]
    )
    with pytest.raises(FrozenInstanceError):
        corpus[TASK_IDS[0]]["train"][0].label = "D"
    with pytest.raises(ValueError, match="TASK_MAPPING"):
        audit_corpus(new_corpus, config)


def test_reservoir_capacity_determinism_exposure_counts_and_storage(cpu_assets):
    config = study_config()
    corpus = build_corpus(config)
    tokenizer = AutoTokenizer.from_pretrained(cpu_assets[8], local_files_only=True)
    encoded = EncodedStream(tokenizer, {**config, "device": "cpu"})
    buffer = Reservoir(64, config["seed"])
    duplicate = Reservoir(64, config["seed"])
    replay_budget = Counter()
    continue_budget = Counter()
    exposed = set()
    for index, identity in enumerate(config["task_order"]):
        old_rows = tuple(buffer.rows)
        replay = list(
            stage_batches(
                corpus[identity]["train"], old_rows, "replay", config, identity
            )
        )
        plain = list(
            stage_batches(corpus[identity]["train"], (), "continue", config, identity)
        )
        replay_new = [
            row
            for rows, roles in replay
            for row, role in zip(rows, roles, strict=True)
            if role == "new"
        ]
        plain_new = [row for rows, _ in plain for row in rows]
        assert replay_new == plain_new[: len(replay_new)]
        assert len(replay) == len(plain) == config["updates_per_stage"]
        for rows, roles in replay:
            assert len(rows) == config["batch_size"]
            assert roles.count("replay") == (4 if index else 0)
            for row, role in zip(rows, roles, strict=True):
                if role == "replay":
                    assert (
                        row in old_rows and row.task_id in config["task_order"][:index]
                    )
            replay_budget.update(roles)
        continue_budget["new"] += len(plain_new)
        unique = list({row.id: row for row in replay_new}.values())
        exposed.update(row.id for row in unique)
        buffer.offer(unique)
        duplicate.offer(unique)
        assert buffer.payload() == duplicate.payload()
        assert len(buffer.rows) == 64
        assert buffer.seen == (index + 1) * config["train_examples"]
        assert {row.id for row in buffer.rows} <= exposed
        storage = buffer.storage(encoded, config["task_order"][:index])
        assert storage["examples"] == 64
        assert storage["prompt_label_eos_tokens"] <= 64 * config["max_length"]
        payload = json.dumps(
            buffer.payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        assert storage["serialized_bytes_including_rng"] == len(payload)
        assert storage["sha256"] == hashlib.sha256(payload).hexdigest()
        assert set(storage["old_ids"]) == {
            row.id for row in buffer.rows if row.task_id in config["task_order"][:index]
        }
    assert continue_budget == {"new": 8192}
    assert replay_budget == {"new": 4608, "replay": 3584}
    assert sum(continue_budget.values()) == sum(replay_budget.values())
    assert len({row.task_id for row in buffer.rows}) > 1
    other = Reservoir(64, 29)
    other.offer(row for splits in corpus.values() for row in splits["train"])
    assert [row.id for row in buffer.rows] != [row.id for row in other.rows]
    with pytest.raises(ValueError, match="REPLAY_CAPACITY"):
        list(
            stage_batches(
                corpus[TASK_IDS[-1]]["train"],
                corpus[TASK_IDS[0]]["train"][:65],
                "replay",
                config,
                TASK_IDS[-1],
            )
        )


def test_whole_output_and_native_eos_strict_grading(cpu_assets):
    config = cpu_config(cpu_assets[2], 2)
    tokenizer = AutoTokenizer.from_pretrained(cpu_assets[2], local_files_only=True)
    encoded = EncodedStream(tokenizer, config)
    a, b = encoded.label_ids["A"], encoded.label_ids["B"]
    eos, pad = encoded.eos_id, encoded.pad_id
    good = strict_grade([a, eos, pad, pad], "A", tokenizer, eos, pad)
    assert good["correct"] and good["generated_ids"] == [a, eos]
    poison = (
        [a],
        [a, a],
        [a, b, eos],
        [a, eos, b],
        [b, eos],
        [eos],
        [a, tokenizer.bos_token_id, eos],
        [a, pad, eos],
    )
    for tokens in poison:
        assert not strict_grade(tokens, "A", tokenizer, eos, pad)["correct"]
    first_letter_only = strict_grade([a, b, eos], "A", tokenizer, eos, pad)
    metrics = score_records([first_letter_only])
    assert metrics["accuracy"] == 0
    assert not headroom_gate(0, metrics["accuracy"], config)["passed"]
    assert not any("logit" in key or "candidate" in key for key in metrics)
    generation = encoded.generation_config
    assert generation.max_new_tokens == 4 and generation.do_sample is False
    assert generation.num_beams == 1 and generation.forced_eos_token_id is None
    assert generation.forced_bos_token_id is None
    assert (
        generation.suppress_tokens is None and generation.begin_suppress_tokens is None
    )


def test_native_chat_eos_can_differ_from_model_end_of_text(cpu_assets):
    config = cpu_config(cpu_assets[2], 2)
    tokenizer = AutoTokenizer.from_pretrained(cpu_assets[2], local_files_only=True)
    encoded = EncodedStream(tokenizer, config)
    model, _ = new_model(config)
    assert (
        validate_native_eos(model, encoded)["validated_by"]
        == "model_generation_default"
    )
    model.generation_config.eos_token_id = tokenizer.pad_token_id
    with pytest.raises(ValueError, match="does not close the assistant template"):
        validate_native_eos(model, encoded)
    tokenizer.chat_template = "{{ messages[-1]['content'] }} {{ eos_token }}\n"
    assert (
        validate_native_eos(model, encoded)["validated_by"]
        == "tokenizer_assistant_turn_terminator"
    )
    assert encoded.generation_config.forced_eos_token_id is None


def test_headroom_gates_do_not_require_impossible_initial_plus_gain():
    config = study_config()
    high_initial = headroom_gate(0.8, 0.95, config)
    assert high_initial["required_accuracy"] == pytest.approx(0.9)
    assert high_initial["passed"]
    saturated = headroom_gate(1.0, 1.0, config)
    assert saturated["required_accuracy"] == 1.0
    assert saturated["reason"] == "insufficient_baseline_headroom"
    assert saturated["headroom_gain_fraction"] is None and not saturated["passed"]
    assert not headroom_gate(0, 0.5, config)["passed"]
    assert headroom_gate(0, 0.7, config)["passed"]


def test_metric_matrix_has_correct_forgetting_bwt_and_learning():
    def point(value):
        return {"test": {"accuracy": value}}

    initial = {"A": point(0.2), "B": point(0.4), "C": point(0.3)}
    boundaries = [
        {"A": point(0.8)},
        {"A": point(0.9), "B": point(0.7)},
        {"A": point(0.5), "B": point(0.6), "C": point(0.9)},
    ]
    previous = [{"A": point(0.2)}, {"B": point(0.5)}, {"C": point(0.4)}]
    result = summarize_matrix(initial, boundaries, previous, ["A", "B", "C"], "test")
    assert result["final_average_accuracy"] == pytest.approx(2 / 3)
    assert result["max_forgetting"] == pytest.approx(0.4)
    assert result["backward_transfer"] == pytest.approx(-0.2)
    assert result["new_task_learning"] == pytest.approx(1.3 / 3)


def test_actual_letter_eos_training_freeze_and_poison_independence(cpu_assets):
    config = cpu_config(cpu_assets[2], 2)
    corpus = build_corpus(config)
    changed = copy.deepcopy(corpus)
    changed[TASK_IDS[0]]["test"] = tuple(
        replace(row, label="A") for row in changed[TASK_IDS[0]]["test"]
    )
    changed[TASK_IDS[1]]["train"] = tuple(
        replace(row, label="B") for row in changed[TASK_IDS[1]]["train"]
    )
    tokenizer = AutoTokenizer.from_pretrained(cpu_assets[2], local_files_only=True)
    encoded = EncodedStream(tokenizer, config)
    models = [new_model(config), new_model(config)]
    original = [base_hash(model) for model, _ in models]
    before = adapter_hash(models[0][0])
    updates = []
    for dataset, (model, optimizer) in zip((corpus, changed), models, strict=True):
        rows = next(
            stage_batches(
                dataset[TASK_IDS[0]]["train"], (), "continue", config, TASK_IDS[0]
            )
        )[0]
        inputs, targets = encoded.batch(rows, training=True)
        assert targets[:, 1].tolist() == [encoded.eos_id] * len(rows)
        assert inputs["input_ids"][:, -2:].tolist() == targets.tolist()
        update = train_update(model, optimizer, encoded, rows, config)
        assert (
            update["supervised_letter_tokens"]
            == update["supervised_eos_tokens"]
            == len(rows)
        )
        assert (
            update["letter_loss"] > 0
            and update["eos_loss"] > 0
            and update["gradient_norm"] > 0
        )
        training_invariants(model, optimizer, 1)
        updates.append(update)
    assert adapter_hash(models[0][0]) == adapter_hash(models[1][0]) != before
    assert updates[0] == updates[1]
    assert [base_hash(model) for model, _ in models] == original
    records = generate_records(models[0][0], encoded, corpus[TASK_IDS[0]]["test"], 4)
    assert len(records) == 8
    assert all(len(record["raw_generation_ids"]) <= 4 for record in records)
    models[0][0].generation_config.forced_eos_token_id = encoded.eos_id
    models[0][0].generation_config.forced_bos_token_id = encoded.label_ids["A"]
    models[0][0].generation_config.suppress_tokens = list(encoded.label_ids.values())
    models[0][0].generation_config.min_new_tokens = 4
    models[0][0].generation_config.do_sample = True
    assert (
        generate_records(models[0][0], encoded, corpus[TASK_IDS[0]]["test"], 4)
        == records
    )


def test_dispatcher_preservation_and_immutable_output(tmp_path):
    config = study_config()
    output = tmp_path / "dispatcher"
    output.mkdir()
    parent = {
        "config.json": json.dumps(config).encode(),
        "execution.json": b'{"status":"running","owner":"parent"}\n',
        "packages.txt": b"parent-owned-packages\n",
        "task.json": json.dumps(
            {"config": config, "entrypoint": "long_stream.py"}
        ).encode(),
        "run.log": b"parent-owned-live-log\n",
    }
    for name, content in parent.items():
        (output / name).write_bytes(content)
    (output / "attempts" / "parent-attempt").mkdir(parents=True)
    (output / "attempts" / "parent-attempt" / "receipt.json").write_text("{}")
    prepare_output(config, output)
    assert {name: (output / name).read_bytes() for name in parent} == parent
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["config_sha256"] == digest(config)
    with pytest.raises(RuntimeError, match="OUTPUT_ALREADY_USED"):
        prepare_output(config, output)
    conflicting = tmp_path / "conflicting"
    conflicting.mkdir()
    (conflicting / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="DISPATCH_CONFIG_MISMATCH"):
        prepare_output(config, conflicting)


@pytest.mark.parametrize("stages", [2, 8])
def test_actual_qwen_cpu_schedule_control_and_reloaded_behavior(
    cpu_assets, tmp_path, stages
):
    config = cpu_config(cpu_assets[stages], stages)
    validate_config(config)
    output = tmp_path / f"actual-{stages}-stage"
    output.mkdir()
    parent = {
        "config.json": (json.dumps(config, indent=2) + "\n").encode(),
        "packages.txt": b"actual-installed-cpu-test-runtime\n",
        "execution.json": b'{"status":"running","owner":"parent","device":"cpu"}\n',
        "task.json": json.dumps(
            {"config": config, "entrypoint": "long_stream.py"}
        ).encode(),
        "run.log": b"parent-owned-live-log\n",
    }
    for name, content in parent.items():
        (output / name).write_bytes(content)
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(ROOT / "long_stream.py"),
        "--config",
        str(output / "config.json"),
        "--output-dir",
        str(output),
    ]
    with (output / "run.log").open("a") as log:
        process = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=240,
            check=False,
        )
    assert process.returncode == 0, (output / "run.log").read_text()[-18000:]
    results = json.loads((output / "results.json").read_text())
    assert results["status"] == "completed" and results["history_immutable"]
    assert not results["claim_eligible"] and results["mode"] == "cpu_qualification"
    assert {
        name: (output / name).read_bytes() for name in parent if name != "run.log"
    } == {name: content for name, content in parent.items() if name != "run.log"}
    assert (output / "run.log").read_bytes().startswith(parent["run.log"])
    assert json.loads((output / "protocol.json").read_text())["config"] == config
    assert json.loads((output / "runtime.json").read_text())["device"] == "cpu"
    assert (output / "protocol.json").stat().st_mtime_ns <= (
        output / "evaluation" / "initial" / "metrics.json"
    ).stat().st_mtime_ns
    expected_examples = stages * config["updates_per_stage"] * config["batch_size"]
    expected_replay = (
        (stages - 1) * config["updates_per_stage"] * config["replay_per_batch"]
    )
    exposure = results["exposure_comparison"]
    assert exposure["examples_per_arm"] == expected_examples
    assert exposure["replay_old_examples"] == expected_replay
    assert exposure["new_exposure_difference_replay_minus_continue"] == -expected_replay
    for arm in ("continue", "replay"):
        result = results["arms"][arm]
        assert result["persistent_adapter_rank"] == 8
        assert result["base_all_tensors_unchanged"]
        assert result["counts"]["examples"] == expected_examples
        assert result["counts"]["updates"] == stages * config["updates_per_stage"]
        assert result["counts"]["supervised_eos_tokens"] == expected_examples
        assert result["counts"]["supervised_letter_tokens"] == expected_examples
        assert result["reload"]["whole_generation_records_exact"]
        assert result["reload"]["example_count"] == stages * 16
        matrix = json.loads((output / "matrices" / f"{arm}.json").read_text())
        assert len(matrix["boundaries"]) == stages
        for index, boundary in enumerate(matrix["boundaries"]):
            assert list(boundary) == sorted(config["task_order"][: index + 1])
            receipt = result["stages"][index]
            assert (
                receipt["invariants_after"]["optimizer_step"]
                == (index + 1) * config["updates_per_stage"]
            )
            assert receipt["initial_adapter_sha256"] != receipt["final_adapter_sha256"]
            if index:
                assert (
                    receipt["initial_adapter_sha256"]
                    == result["stages"][index - 1]["final_adapter_sha256"]
                )
                if arm == "replay":
                    assert receipt["buffer_after"]["examples"] <= 64
        loaded, optimizer, _ = checkpoint_load(output / "checkpoints" / arm, config)
        training_invariants(loaded, optimizer, stages * config["updates_per_stage"])
        assert base_hash(loaded) == result["base_tensor_sha256"]
    if stages == 8:
        assert results["arms"]["replay"]["resident_replay_buffer"]["examples"] == 64
    reference = results["fresh_final_reference"]
    assert reference["task_id"] == config["task_order"][-1]
    assert reference["training"]["counts"]["updates"] == config["updates_per_stage"]
    assert reference["reload"]["whole_generation_records_exact"]
    assert results["acquisition_gate"]["required_accuracy"] <= 1
    assert reference["gate"]["required_accuracy"] <= 1
    result_hash = hashlib.sha256((output / "results.json").read_bytes()).hexdigest()
    rerun = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rerun.returncode != 0 and "OUTPUT_ALREADY_USED" in rerun.stderr
    assert (
        hashlib.sha256((output / "results.json").read_bytes()).hexdigest()
        == result_hash
    )
    receipt = {
        "status": "passed",
        "stages": stages,
        "real_qwen_training": True,
        "independent_base_snapshot": str(cpu_assets[stages]),
        "cpu_only": True,
        "no_model_downloads": True,
        "results_sha256": result_hash,
        "results": str(output / "results.json"),
        "claim_eligible": False,
        "tested": [
            "label-plus-EOS gradients",
            "persistent rank eight",
            "full base freeze",
            "optimizer continuity",
            "bounded replay",
            "all-learned-task boundary matrix",
            "whole-output native-EOS grading",
            "checkpoint reload generation identity",
            "dispatcher preservation",
            "immutable rerun rejection",
        ],
    }
    (tmp_path / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
