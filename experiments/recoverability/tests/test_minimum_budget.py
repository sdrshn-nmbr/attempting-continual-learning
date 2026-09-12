import pytest
import test_prospective as source_tests
from test_prospective import records, rows

import minimum_budget as budget_module
from minimum_budget import (
    checked_learning_sources,
    checked_measurement,
    checked_records,
    distribution_gate,
    learning_gate,
    load_manifest,
    measure,
    minimum_outcome,
    permutation_diagnostics,
    permutation_plan,
    pilot_identity,
    planned_permutations,
    qualify,
    run_budget_arms,
    schedule_for,
)
from prospective import finish, make_seal
from prospective_model import parameter_hashes
from protocol import ROOT, digest, file_hash, read_json, write_json

design = source_tests.design
tiny_model = source_tests.tiny_model


@pytest.fixture
def pilot():
    return read_json(ROOT / "minimum-budget-protocol.json")


def observations(stream, initial_old, before_old, after_old, initial_new=0, after_new=10):
    old = stream["units"][stream["old"][0]]["gate"]
    new = stream["units"][stream["new"][0]]["gate"]

    def scored(source, count):
        result = records(source, count)
        for item in result:
            logits = [-2.0] * 16
            logits[item["output"]["prediction"] - 32] = 2.0
            item["output"]["code_logits"] = logits
        return result

    return {
        "initial": {"old_gate": scored(old, initial_old), "new_gate": scored(new, initial_new)},
        "acquired": {"old_gate": scored(old, before_old), "new_gate": scored(new, initial_new)},
        "forgotten": {"old_gate": scored(old, after_old), "new_gate": scored(new, after_new)},
    }


@pytest.fixture
def small_stream():
    return {
        "id": "train-a",
        "split": "train",
        "seed": 101,
        "old": ["old"],
        "new": ["new"],
        "units": {
            intent: {
                role: rows(intent, code, role, 8 if role == "repair" else 10)
                for role in ("learn", "repair", "gate", "probe", "test")
            }
            for code, intent in enumerate(("old", "new"))
        },
    }


def test_recomputes_real_gain_for_old_and_new_and_forgetting(design, small_stream):
    _, spec = design
    spec["gates"].update(
        minimum_acquired_per_stream=1, minimum_new_acquired_per_stream=1, minimum_forgotten_per_stream=1
    )
    codes = list(range(32, 48))
    gate = learning_gate(observations(small_stream, 5, 8, 5), small_stream, spec, codes)
    assert gate == {"acquired": ["old"], "new_acquired": ["new"], "eligible": ["old"], "passed": True}
    assert not learning_gate(observations(small_stream, 8, 10, 0), small_stream, spec, codes)["passed"]
    assert not learning_gate(observations(small_stream, 0, 10, 6), small_stream, spec, codes)["passed"]
    assert not learning_gate(observations(small_stream, 0, 10, 0, initial_new=10), small_stream, spec, codes)["passed"]


def test_eligibility_rejects_forged_native_prediction(small_stream):
    source = observations(small_stream, 0, 10, 0)["acquired"]["old_gate"]
    source[0]["output"]["code_logits"] = list(range(16))
    with pytest.raises(ValueError, match="NATIVE_CODE_CONTRACT"):
        checked_records(source, small_stream["units"]["old"]["gate"], list(range(32, 48)))


def prediction_unit(budgets, targets, guards, after=0, guard_after=20):
    source, newer = rows("old", 0, "test", 20), rows("new", 1, "test", 20)
    return {
        "id": "train-a/old",
        "stream": "train-a",
        "intent": "old",
        "split": "train",
        "baseline": {
            "before": {"test": records(source, 20), "guard": records(newer, 0)},
            "after": {"test": records(source, after), "guard": records(newer, guard_after)},
        },
        "actions": {
            str(budget): {"test": records(source, score), "guard": records(newer, guard)}
            for budget, score, guard in zip(budgets, targets, guards, strict=True)
        },
    }


def test_minimum_rule_retains_guard_failure_nonmonotonicity_and_never(design, pilot):
    _, spec = design
    unit = prediction_unit(pilot["budgets"], [15, 16, 20, 20], [20, 18, 17, 20])
    result = minimum_outcome(unit, spec, pilot["budgets"])
    assert result["minimum_budget"] == "4"
    assert result["qualified_budgets"] == [4, 16]
    assert result["later_qualification_loss"]
    assert result["actions"]["8"]["accuracy"] == 1
    assert not result["actions"]["8"]["recovered"]
    unit = prediction_unit(pilot["budgets"], [20] * 4, [17] * 4)
    assert minimum_outcome(unit, spec, pilot["budgets"])["minimum_budget"] == "never"
    unit = prediction_unit(pilot["budgets"], [16] * 4, [20] * 4, after=11)
    assert minimum_outcome(unit, spec, pilot["budgets"])["minimum_budget"] == "never"


def gate_units(labels):
    return [
        {
            "id": f"train-{'a' if i % 2 else 'b'}/intent{i}",
            "stream": f"train-{'a' if i % 2 else 'b'}",
            "split": "train",
            "minimum_budget": label,
            "later_qualification_loss": False,
        }
        for i, label in enumerate(labels)
    ]


@pytest.mark.parametrize(
    "labels,qualified",
    [
        (["2"] * 12, False),
        (["2"] * 11 + ["4"], False),
        (["2"] * 10 + ["4"] * 2, True),
        (["never"] * 10 + ["8"] * 2, True),
        (["2"] * 2 + ["4"] * 2, False),
    ],
)
def test_gate_requires_two_supported_minimum_classes(labels, qualified, pilot):
    result = distribution_gate(gate_units(labels), pilot)
    assert result["qualified"] is qualified
    assert result["test_authorized"] is False
    if not qualified:
        assert result["status"] == "no_identifiable_prediction_task"


def test_gate_keeps_source_clusters_and_rejects_duplicate_or_heldout(pilot):
    units = gate_units(["2"] * 6 + ["4"] * 6)
    result = distribution_gate(units, pilot)
    assert result["source_clusters"] == 2
    assert result["counts_by_source"] == {"train-a": {"2": 3, "4": 3}, "train-b": {"2": 3, "4": 3}}
    units[1]["id"] = units[0]["id"]
    with pytest.raises(ValueError, match="DUPLICATE"):
        distribution_gate(units, pilot)
    units[0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        distribution_gate(units, pilot)


def test_multiple_permutations_keep_clusters_and_inert_draws_without_redraw(pilot):
    sources = ["train-a"] * 7 + ["train-b"] * 7
    ids = [f"{source}/{i}" for i, source in enumerate(sources)]
    plan = permutation_plan(ids, sources, pilot["label_permutations"]["seeds"], pilot["label_permutations"]["schemes"])
    assert len(plan["draws"]) == 40
    assert plan == permutation_plan(
        ids, sources, pilot["label_permutations"]["seeds"], pilot["label_permutations"]["schemes"]
    )
    for draw in plan["draws"]:
        assert sorted(draw["indices"]) == list(range(14))
        if draw["scheme"] == "within_source":
            assert [sources[i] for i in draw["indices"]] == sources
    outcomes = [
        {
            "id": key,
            "minimum_budget": "2",
            "actions": {str(b): {"recovered": True, "utility": i / 20} for b in pilot["budgets"]},
        }
        for i, key in enumerate(ids)
    ]
    control = permutation_diagnostics(plan, outcomes, pilot["budgets"])
    assert control["redraws"] == control["predictor_fits"] == 0
    assert all(draw["changed_labels"]["minimum_budget"] == 0 for draw in control["draws"])
    assert all(draw["changed_labels"]["2/recovered"] == 0 for draw in control["draws"])
    assert all(draw["changed_labels"]["2/utility"] > 0 for draw in control["draws"])
    assert all("minimum_budget" in draw["inert_targets"] for draw in control["draws"])
    with pytest.raises(ValueError, match="LABEL_ORDER"):
        permutation_diagnostics(plan, outcomes[::-1], pilot["budgets"])


def test_manifest_accepts_dispatch_interface_but_no_test_or_extra_stage(tmp_path):
    for name in ("train-a", "train-b", "gate"):
        assert load_manifest(ROOT / f"configs/minimum-budget-{name}.json")[0]["stage"] in {"measure", "qualify"}
    manifest = read_json(ROOT / "configs/minimum-budget-train-a.json")
    manifest["stream"] = "test-a"
    write_json(tmp_path / "test.json", manifest)
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        load_manifest(tmp_path / "test.json")
    manifest["stage"] = "analyze"
    write_json(tmp_path / "stage.json", manifest)
    with pytest.raises(ValueError, match="FIELDS_OR_STAGE"):
        load_manifest(tmp_path / "stage.json")


def test_dispatch_existing_run_directory_seals_failed_prerequisite_without_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model work occurred after failed eligibility")

    monkeypatch.setattr(budget_module, "checked_learning_sources", lambda *args: {"qualified": False})
    monkeypatch.setattr(budget_module, "verify_model", forbidden)
    monkeypatch.setattr(budget_module.ProspectiveModel, "fresh", forbidden)
    run = tmp_path / "existing-run"
    run.mkdir()
    budget_module.main(["--config", str(ROOT / "configs/minimum-budget-train-a.json"), "--output-dir", str(run)])
    receipt = read_json(run / "study/receipt.json")["payload"]
    assert receipt["status"] == "prerequisite_failed"
    assert receipt["repair_updates"] == receipt["lens_updates"] == receipt["predictor_fits"] == 0
    with pytest.raises(FileExistsError):
        budget_module.main(["--config", str(ROOT / "configs/minimum-budget-train-a.json"), "--output-dir", str(run)])


def test_actual_adamw_budgets_have_identical_prefixes_and_reproduce_anchor(tiny_model, small_stream, tmp_path, pilot):
    observer = tiny_model
    spec = observer.spec
    checkpoint = tmp_path / "forgotten"
    observer.checkpoint(checkpoint)
    frozen = parameter_hashes(observer.model, frozen_only=True)
    baseline = {
        "test": observer.observe(small_stream["units"]["old"]["test"]),
        "guard": observer.observe(rows(small_stream["new"][0], 1, "test", 10)),
    }
    repair_rows, batches = schedule_for(small_stream, "old", spec, 16)
    optimizer = observer.optimizer()
    observer.update(repair_rows, batches, optimizer)
    observer.checkpoint(tmp_path / "anchor")
    anchor = {
        "test": observer.observe(small_stream["units"]["old"]["test"]),
        "guard": observer.observe(small_stream["units"]["new"]["test"]),
        "updates": 16,
        "adapter_sha256": file_hash(tmp_path / "anchor/adapter_model.safetensors"),
        "schedule_sha256": digest({"rows": repair_rows, "batches": batches}),
    }
    actions = run_budget_arms(
        observer, small_stream, "old", spec, pilot["budgets"], checkpoint, baseline, tmp_path, anchor
    )
    assert [actions[str(b)]["updates"] for b in pilot["budgets"]] == [2, 4, 8, 16]
    assert sum(action["updates"] for action in actions.values()) == 30
    for b in pilot["budgets"]:
        assert actions[str(b)]["prefixes"][str(b)] == actions["16"]["prefixes"][str(b)]
        assert actions[str(b)]["old_exposures"] == actions[str(b)]["new_exposures"] == 4 * b
        assert actions[str(b)]["start_adapter_sha256"] == file_hash(checkpoint / "adapter_model.safetensors")
    assert frozen == parameter_hashes(observer.model, frozen_only=True)
    assert observer.observe(small_stream["units"]["old"]["test"]) == baseline["test"]


def test_baselines_must_match_before_any_budget_update(tiny_model, small_stream, tmp_path, monkeypatch, pilot):
    tiny_model.checkpoint(tmp_path / "acquired")
    tiny_model.checkpoint(tmp_path / "forgotten")
    unit = prediction_unit(pilot["budgets"], [20] * 4, [20] * 4)
    unit.update(probes={}, features={})

    def forbidden(*args, **kwargs):
        pytest.fail("Repair was attempted before baseline parity")

    monkeypatch.setattr(budget_module, "run_budget_arms", forbidden)
    with pytest.raises(ValueError, match="PREUPDATE_BASELINE_MISMATCH"):
        measure(tiny_model, small_stream, tiny_model.spec, pilot, tmp_path, [unit], tmp_path / "output")


def test_learning_source_hashes_are_required_before_measurement(design, pilot, tmp_path):
    config, spec = design
    data = read_json(ROOT / config["dataset"])
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "receipt.json", {"wrong": True})
    with pytest.raises(ValueError, match="LEARNING_RECEIPT_HASH"):
        checked_learning_sources({name: str(source) for name in pilot["train_streams"]}, config, spec, data, pilot)
    with pytest.raises(ValueError, match="MEASUREMENT_RECEIPT_HASH"):
        checked_measurement(source, data["streams"]["train-a"], {}, config, spec, data, pilot)


def test_gate_rejects_changed_runner_or_protocol_before_reading_outcomes(design, pilot, tmp_path):
    config, spec = design
    data = read_json(ROOT / config["dataset"])
    source = tmp_path / "source"
    source.mkdir()
    payload = make_seal(config, spec, data, "minimum-budget-measure")["payload"]
    payload["minimum_budget_identity"] = "changed"
    write_json(source / "seal.json", {"payload": payload, "sha256": digest(payload)})
    finish(source, {"stage": "minimum-budget-measure", "status": "completed", "stream": "train-a", "split": "train"})
    assert pilot_identity(config, spec, data, pilot) != pilot_identity(config, spec, data, pilot | {"budgets": [1, 2]})
    with pytest.raises(ValueError, match="GATE_SOURCE_BINDING"):
        qualify({name: source for name in pilot["train_streams"]}, {}, config, spec, data, pilot)


def test_actual_measurement_and_gate_receipts_recompute_without_new_training(
    tiny_model, small_stream, design, pilot, tmp_path, monkeypatch
):
    observer = tiny_model
    config, _ = design
    spec = observer.spec
    pilot = pilot | {"train_streams": ["train-a"]}
    learning, output = tmp_path / "learning", tmp_path / "study"
    output.mkdir()
    observer.checkpoint(learning / "acquired")
    observer.checkpoint(learning / "forgotten")
    panel = {
        "test": observer.observe(small_stream["units"]["old"]["test"]),
        "guard": observer.observe(small_stream["units"]["new"]["test"]),
    }
    source = {
        "id": "train-a/old",
        "stream": "train-a",
        "split": "train",
        "intent": "old",
        "baseline": {"before": panel, "after": panel},
        "probes": {},
        "features": {name: [0.0] for name in ("confidence", "output", "frozen", "tuned")},
        "actions": {"none": {"adapter_sha256": file_hash(learning / "forgotten/adapter_model.safetensors")}},
    }
    repair_rows, batches = schedule_for(small_stream, "old", spec, 16)
    observer.update(repair_rows, batches, observer.optimizer())
    observer.checkpoint(tmp_path / "anchor")
    source["actions"]["replay_balanced"] = {
        "test": observer.observe(small_stream["units"]["old"]["test"]),
        "guard": observer.observe(small_stream["units"]["new"]["test"]),
        "updates": 16,
        "adapter_sha256": file_hash(tmp_path / "anchor/adapter_model.safetensors"),
        "schedule_sha256": digest({"rows": repair_rows, "batches": batches}),
    }
    prerequisites = {"streams": {"train-a": {"eligible": ["old"], "source": str(learning)}}}
    data = {"codes": observer.codes, "streams": {"train-a": small_stream}}
    sealed = make_seal(config, spec, data, "minimum-budget-measure")["payload"]
    sealed.update(
        minimum_budget_identity=pilot_identity(config, spec, data, pilot),
        dispatch_manifest={"measurement_source": "old-train-measurement"},
    )
    write_json(output / "seal.json", {"payload": sealed, "sha256": digest(sealed)})
    write_json(output / "prerequisites.json", prerequisites)
    write_json(output / "permutation-plan.json", planned_permutations(prerequisites, pilot))
    write_json(
        output / "measurement-source.json",
        {"path": "old-train-measurement", "receipt_sha256": pilot["source_receipts"]["train-a"]["measure"]},
    )
    real_update = observer.update
    updates = []

    def witness(rows, schedule, optimizer):
        assert read_json(output / "features-frozen.json")[0]["actions"] == {}
        assert read_json(output / "source-outcomes-frozen.json") == [source]
        updates.append(len(schedule))
        return real_update(rows, schedule, optimizer)

    monkeypatch.setattr(observer, "update", witness)
    measured = measure(observer, small_stream, spec, pilot, learning, [source], output)
    finish(output, measured)
    assert sum(updates) == measured["repair_updates"] == 30
    result = qualify({"train-a": output}, prerequisites, config, spec, data, pilot)
    assert sum(updates) == 30
    assert result["status"] == "no_identifiable_prediction_task"
    assert not result["test_authorized"]
    assert result["run_level"]["train-a"]["intents"] == 1
    assert result["run_level"]["train-a"]["mean_readouts"] == source["features"]
