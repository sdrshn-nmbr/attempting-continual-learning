import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from data import SEQUENCE_TASKS, prepare_data, read_data, sequence_splits

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sequence_a", "sequence_b", "sequence_c")
SIZES = {"train": 128, "validation": 32, "test": 64}
OLD_MAPPINGS = {(3, 1, 0, 4, 5, 2, 7, 6), (3, 7, 0, 6, 5, 4, 2, 1)}


def compact_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("fixture_seed,training_seed", [(303, 30301), (419, 41901)])
def test_frozen_sequence_fixture_contract(fixture_seed, training_seed):
    fixture = json.loads((ROOT / "data" / f"sequence{fixture_seed}.json").read_text())
    provenance = fixture["provenance"]
    assert provenance["fixture_seed"] == fixture_seed
    assert provenance["training_seed"] == training_seed
    assert provenance["tasks"] == list(TASKS)
    assert provenance["split_sizes_per_task"] == SIZES
    assert provenance["generator_sha256"] == compact_hash(fixture["splits"])
    assert provenance["input_sha256"] == compact_hash(fixture["input_splits"])
    maps = {task: tuple(provenance["rules"][task]) for task in TASKS}
    assert len(set(maps.values())) == 3
    assert not set(maps.values()) & OLD_MAPPINGS
    assert all(
        set(mapping) == set(range(8)) and len(mapping) == 8 for mapping in maps.values()
    )
    inputs = {
        split: {tuple(row) for row in rows}
        for split, rows in fixture["input_splits"].items()
    }
    assert {split: len(rows) for split, rows in inputs.items()} == SIZES
    assert len(set.union(*inputs.values())) == 224
    assert not inputs["test"] & (inputs["train"] | inputs["validation"])
    assert not inputs["train"] & inputs["validation"]
    ids, normalized_prompts, task_input_pairs = {}, {}, {}
    for split, size in SIZES.items():
        ids[split], normalized_prompts[split], task_input_pairs[split] = (
            set(),
            set(),
            set(),
        )
        for task in TASKS:
            rows = fixture["splits"][f"{task}_{split}"]
            assert len(rows) == size
            assert Counter(row["gold_idx"] for row in rows) == {
                index: size // 4 for index in range(4)
            }
            observed_inputs = []
            for row in rows:
                assert set(row) == {
                    "id",
                    "task",
                    "prompt",
                    "choices",
                    "gold_idx",
                    "group",
                }
                assert row["task"] == task
                digits = tuple(map(int, row["group"].split()))
                assert len(digits) == 3 and all(0 <= digit < 8 for digit in digits)
                assert (
                    row["prompt"]
                    == f"Apply the {task} code.\nInput: {row['group']}\nOutput:"
                )
                assert row["id"] == compact_hash({"task": task, "input": digits})
                choices = row["choices"]
                assert len(choices) == len(set(choices)) == 4
                assert all(
                    re.fullmatch(r" [0-7] [0-7] [0-7]", choice) for choice in choices
                )
                assert choices[row["gold_idx"]] == " " + " ".join(
                    str(maps[task][digit]) for digit in digits
                )
                observed_inputs.append(list(digits))
                ids[split].add(row["id"])
                normalized_prompts[split].add(
                    " ".join(row["prompt"].split()).casefold()
                )
                task_input_pairs[split].add((task, digits))
            assert observed_inputs == fixture["input_splits"][split]
        assert (
            len(ids[split])
            == len(normalized_prompts[split])
            == len(task_input_pairs[split])
            == size * 3
        )
    for keys in (ids, normalized_prompts, task_input_pairs):
        assert not keys["test"] & (keys["train"] | keys["validation"])
        assert not keys["train"] & keys["validation"]
    for task in TASKS:
        assert provenance["task_sha256"][task] == compact_hash(
            {split: fixture["splits"][f"{task}_{split}"] for split in SIZES}
        )


def test_confirmation_has_independent_mappings_and_fixture():
    fixtures = [
        json.loads((ROOT / "data" / f"sequence{seed}.json").read_text())
        for seed in (303, 419)
    ]
    maps = [
        tuple(fixture["provenance"]["rules"][task])
        for fixture in fixtures
        for task in TASKS
    ]
    assert len(set(maps)) == 6
    assert (
        fixtures[0]["provenance"]["generator_sha256"]
        != fixtures[1]["provenance"]["generator_sha256"]
    )
    assert (
        fixtures[0]["provenance"]["input_sha256"]
        != fixtures[1]["provenance"]["input_sha256"]
    )


@pytest.mark.parametrize("fixture_seed,training_seed", [(303, 30301), (419, 41901)])
def test_frozen_generator_reproduces_fixture_bytes(
    fixture_seed, training_seed, tmp_path
):
    generator = ROOT / "outputs" / "sequence-data-cpu" / "generator.py"
    frozen = ROOT / "data" / f"sequence{fixture_seed}.json"
    source_hash = hashlib.sha256(generator.read_bytes()).hexdigest()
    assert (
        json.loads(frozen.read_text())["provenance"]["source_generator_sha256"]
        == source_hash
    )
    output = tmp_path / "regenerated.json"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "python",
            str(generator),
            "--fixture-seed",
            str(fixture_seed),
            "--training-seed",
            str(training_seed),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.read_bytes() == frozen.read_bytes()


@pytest.fixture(params=[303, 419])
def config(request):
    directory = ROOT / "configs"
    if request.param == 419:
        directory /= "confirmation419"
    return json.loads((directory / "sequence_native_no_replay.json").read_text())


def write_modified_fixture(config, tmp_path, mutate):
    spec = dict(config["sequence"])
    value = json.loads((ROOT / spec["path"]).read_text())
    mutate(value)
    provenance = value["provenance"]
    provenance["generator_sha256"] = compact_hash(value["splits"])
    provenance["input_sha256"] = compact_hash(value["input_splits"])
    provenance["task_sha256"] = {
        task: compact_hash(
            {split: value["splits"][f"{task}_{split}"] for split in SIZES}
        )
        for task in TASKS
    }
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(value, sort_keys=True))
    spec.update(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        generator_sha256=provenance["generator_sha256"],
    )
    return spec


def test_prepare_and_explicit_reader_roundtrip(config, tmp_path):
    manifest = prepare_data(config, tmp_path)
    expected, metadata = sequence_splits(config["sequence"])
    assert SEQUENCE_TASKS == TASKS
    assert manifest["fixture"] == metadata
    assert manifest == json.loads((tmp_path / "data_manifest.json").read_text())
    assert manifest["counts"] == {
        f"{task}_{split}": size for task in TASKS for split, size in SIZES.items()
    }
    assert manifest["fixture"]["shared_unique_inputs"] == 224
    assert all(
        count == 0
        for exclusions in manifest["fixture"]["overlap_audit"].values()
        for count in exclusions.values()
    )
    assert "rules" not in manifest["fixture"]
    assert "input_splits" not in manifest["fixture"]
    selected = tuple(f"{task}_test" for task in TASKS)
    rows = read_data(tmp_path, selected)
    assert rows == {name: expected[name] for name in selected}
    for values in rows.values():
        for row in values:
            choice = row.choice()
            assert (choice.task, choice.prompt, choice.choices, choice.gold_idx) == (
                row.task,
                row.prompt,
                row.choices,
                row.gold_idx,
            )


def test_training_reader_needs_only_requested_splits(config, tmp_path):
    spec = write_modified_fixture(config, tmp_path, lambda value: None)
    staged_config = {**config, "sequence": spec}
    prepared = tmp_path / "prepared"
    prepare_data(staged_config, prepared)
    Path(spec["path"]).unlink()
    requested = ("sequence_a_train", "sequence_a_validation")
    for path in (prepared / "data").glob("*.json"):
        if path.stem not in requested:
            path.unlink()
    rows = read_data(prepared, requested)
    assert tuple(rows) == requested
    assert len(rows[requested[0]]) == 128
    assert len(rows[requested[1]]) == 32


@pytest.mark.parametrize(
    "requested,reason",
    [
        ((), "EXPLICIT_SEQUENCE_SPLITS_REQUIRED"),
        ("sequence_a_train", "EXPLICIT_SEQUENCE_SPLITS_REQUIRED"),
        (("sequence_a_train", "sequence_a_train"), "DUPLICATE_SEQUENCE_SPLIT_REQUEST"),
        (("../sequence_a_train",), "UNKNOWN_SEQUENCE_SPLIT_REQUEST"),
        (("amber_train",), "UNKNOWN_SEQUENCE_SPLIT_REQUEST"),
    ],
)
def test_reader_rejects_ambiguous_access_before_opening_files(
    tmp_path, requested, reason
):
    with pytest.raises(ValueError, match=reason):
        read_data(tmp_path, requested)


def test_reader_requires_split_argument(tmp_path):
    with pytest.raises(TypeError, match="splits"):
        read_data(tmp_path)


def test_reader_rejects_tampered_selected_file(config, tmp_path):
    prepare_data(config, tmp_path)
    path = tmp_path / "data" / "sequence_a_train.json"
    rows = json.loads(path.read_text())
    rows[0]["gold_idx"] = (rows[0]["gold_idx"] + 1) % 4
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="DATA_HASH_MISMATCH: sequence_a_train"):
        read_data(tmp_path, ("sequence_a_train",))


def test_fixture_requires_exact_bytes_and_source_provenance(config, tmp_path):
    spec = write_modified_fixture(config, tmp_path, lambda value: None)
    path = Path(spec["path"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="FROZEN_FIXTURE_HASH_MISMATCH"):
        sequence_splits(spec)
    spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec["source_generator_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SEQUENCE_GENERATOR_PROVENANCE_MISMATCH"):
        sequence_splits(spec)


def corrupt_fixture(value, corruption):
    rows = value["splits"]["sequence_a_train"]
    row = rows[0]
    if corruption == "cross_split_input":
        value["input_splits"]["test"][0] = value["input_splits"]["train"][0]
    elif corruption == "duplicate_input":
        value["input_splits"]["train"][1] = value["input_splits"]["train"][0]
    elif corruption == "task_input_order":
        rows[0], rows[1] = rows[1], rows[0]
    elif corruption == "wrong_oracle":
        row["gold_idx"] = (row["gold_idx"] + 1) % 4
    elif corruption == "duplicate_choices":
        row["choices"][1] = row["choices"][0]
    elif corruption == "unequal_choices":
        row["choices"][0] += " 0"
    elif corruption == "prompt_mapping_leak":
        row["prompt"] += "\nMapping: 0 -> 1"
    elif corruption == "incorrect_id":
        row["id"] = "0" * 64
    elif corruption == "hidden_schema_field":
        row["mapping"] = value["provenance"]["rules"]["sequence_a"]
    elif corruption == "unbalanced_choices":
        original = row["gold_idx"]
        other = (original + 1) % 4
        choices = row["choices"]
        choices[original], choices[other] = choices[other], choices[original]
        row["gold_idx"] = other
    elif corruption == "reused_mapping":
        value["provenance"]["rules"]["sequence_b"] = value["provenance"]["rules"][
            "sequence_a"
        ]
    elif corruption == "old_mapping":
        value["provenance"]["rules"]["sequence_a"] = [3, 1, 0, 4, 5, 2, 7, 6]
    else:
        raise AssertionError(f"Unknown corruption: {corruption}")


@pytest.mark.parametrize(
    "corruption,reason",
    [
        ("cross_split_input", "GLOBAL_SEQUENCE_INPUT_SPLIT_OVERLAP"),
        ("duplicate_input", "DUPLICATE_SEQUENCE_INPUT"),
        ("task_input_order", "GLOBAL_SEQUENCE_INPUT_ALIGNMENT_MISMATCH"),
        ("wrong_oracle", "SEQUENCE_ORACLE_MISMATCH"),
        ("duplicate_choices", "INVALID_SEQUENCE_CHOICES"),
        ("unequal_choices", "INVALID_SEQUENCE_CHOICES"),
        ("prompt_mapping_leak", "INVALID_SEQUENCE_PROMPT"),
        ("incorrect_id", "SEQUENCE_ID_INPUT_MISMATCH"),
        ("hidden_schema_field", "SEQUENCE_ROW_SCHEMA_MISMATCH"),
        ("unbalanced_choices", "UNBALANCED_SEQUENCE_GOLD_POSITIONS"),
        ("reused_mapping", "SEQUENCE_MAPPING_FRESHNESS_FAILURE"),
        ("old_mapping", "SEQUENCE_MAPPING_FRESHNESS_FAILURE"),
    ],
)
def test_semantic_corruption_is_rejected_even_with_updated_hashes(
    config, tmp_path, corruption, reason
):
    spec = write_modified_fixture(
        config, tmp_path, lambda value: corrupt_fixture(value, corruption)
    )
    with pytest.raises(ValueError, match=reason):
        sequence_splits(spec)


def test_training_seed_must_match_frozen_fixture(config, tmp_path):
    config["seed"] += 1
    with pytest.raises(ValueError, match="SEQUENCE_TRAINING_SEED_MISMATCH"):
        prepare_data(config, tmp_path)
    assert not (tmp_path / "data_manifest.json").exists()


@pytest.mark.parametrize("fixture_seed", [303, 419])
def test_four_arm_configs_hold_other_variables_fixed(fixture_seed):
    directory = ROOT / "configs"
    if fixture_seed == 419:
        directory /= "confirmation419"
    paths = sorted(directory.glob("sequence_*.json"))
    assert len(paths) == 4
    normalized, arms = [], set()
    for path in paths:
        config = json.loads(path.read_text())
        assert config["arm"] == path.stem
        method = config.pop("method")
        training = config["training"]
        replay = training.pop("replay_weight")
        assert training.pop("train_core") is (method == "native")
        arms.add((method, replay))
        config.pop("arm")
        assert config["seed"] == fixture_seed * 100 + 1
        assert config["sequence"]["fixture_seed"] == fixture_seed
        assert config["source"] == "qwen4"
        assert config["train_model"] == "qwen8"
        assert config["transport_model"] == "qwen17"
        assert (training["steps"], training["batch_size"], training["eval_every"]) == (
            96,
            4,
            24,
        )
        assert (training["replay_every"], training["replay_batch_size"]) == (4, 4)
        assert training["capacity_relative_tolerance"] == 0.005
        for model in config["models"].values():
            for artifact in model.values():
                assert re.fullmatch("[0-9a-f]{40}", artifact["revision"])
                assert Path(artifact["local_path"]).name == artifact["revision"]
        normalized.append(config)
    assert arms == {
        (method, replay) for method in ("native", "lora") for replay in (0.0, 0.25)
    }
    assert all(config == normalized[0] for config in normalized[1:])


def test_confirmation_changes_only_fixture_and_seed():
    for path in sorted((ROOT / "configs").glob("sequence_*.json")):
        primary = json.loads(path.read_text())
        confirmation = json.loads(
            (ROOT / "configs" / "confirmation419" / path.name).read_text()
        )
        assert primary.pop("sequence")["role"] == "primary"
        assert confirmation.pop("sequence")["role"] == "reserved_confirmation"
        assert primary.pop("seed") == 30301
        assert confirmation.pop("seed") == 41901
        assert primary == confirmation
