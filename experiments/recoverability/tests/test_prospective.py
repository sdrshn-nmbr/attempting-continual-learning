import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig

import prospective as prospective_module
import prospective_dispatch
from prospective import (
    checked_pilot_qualification,
    checked_qualification,
    finish,
    learning_identity,
    learning_implementation,
    make_seal,
    perform_actions,
    qualify_learning,
    run_learning,
    verify_result,
)
from prospective_analysis import (
    eligible,
    evaluate_predictors,
    feature_vector,
    fit_predictors,
    learned,
    repair_feasibility,
    ridge_apply,
    ridge_fit,
)
from prospective_data import balanced_batches, load_design, repair_batches, validate_data
from prospective_model import ProspectiveModel, code_features, code_loss, parameter_hashes
from prospective_reference import learn_reference
from protocol import ROOT, file_hash, normalized_hash, read_json, write_json
from tuned_lens import TranslationBank, evaluate_translations, fit_translations, forward_kl, train_lens


@pytest.fixture
def design():
    return load_design(ROOT / "configs/prospective.json")


@pytest.fixture
def tiny_model(design):
    _, spec = design
    spec = copy.deepcopy(spec)
    spec["runtime"].update(device="cpu", attention="eager", microbatch_size=1, eval_batch_size=1)
    spec["layers"] = [1, 2, 3]
    spec["optimizer"]["learning_rate"] = 0.02
    torch.set_num_threads(1)
    torch.manual_seed(43)
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        patch_size=2,
    )
    config = Qwen3_5Config(text_config=text, vision_config=vision)
    config._attn_implementation = "eager"
    base = Qwen3_5ForConditionalGeneration(config)
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r"model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
        ),
    )
    observer = ProspectiveModel(model, spec, list(range(32, 48)))
    observer.pad_token_id = 0
    return observer


def rows(intent, code, role, count=10):
    return [
        {
            "intent": intent,
            "code": code,
            "target": code + 32,
            "input_ids": [1, 4, 5 + i],
            "text": f"{intent}/{role}/{i}",
            "text_sha256": normalized_hash(f"{intent}/{role}/{i}"),
            "source_split": role,
            "source_index": i,
        }
        for i in range(count)
    ]


def records(source, correct):
    return [
        {
            "intent": row["intent"],
            "target": row["target"],
            "text_sha256": row["text_sha256"],
            "output": {"prediction": row["target"] if i < correct else (row["target"] + 1) % 16 + 32},
        }
        for i, row in enumerate(source)
    ]


def test_training_scoring_and_features_use_identical_sixteen_codes():
    logits = torch.arange(64, dtype=torch.float32).requires_grad_()
    codes = list(range(32, 48))
    value = code_loss(logits[None], [47], codes)
    expected = torch.nn.functional.cross_entropy(logits[32:48][None], torch.tensor([15]))
    assert value == expected
    value.backward()
    assert logits.grad[:32].count_nonzero() == logits.grad[48:].count_nonzero() == 0
    assert code_features(logits, 47, codes)["prediction"] == 47
    assert int(logits.argmax()) == 63


def test_actual_qwen_training_reduces_code_loss_and_preserves_backbone(tiny_model, tmp_path):
    observer = tiny_model
    source = rows("skill", 2, "train", count=2)
    before = parameter_hashes(observer.model, frozen_only=True)
    initial = float(code_loss(observer.logits(source), [34, 34], observer.codes).detach())
    optimizer = observer.optimizer()
    observer.update(source, [[0, 1]] * 12, optimizer)
    final = float(code_loss(observer.logits(source), [34, 34], observer.codes).detach())
    assert final < initial - 0.1
    assert parameter_hashes(observer.model, frozen_only=True) == before
    expected = observer.observe(source)
    observer.checkpoint(tmp_path / "learned", optimizer)
    observer.update(source, [[0, 1]], optimizer, sham_intent="skill")
    observer.reset(tmp_path / "learned")
    assert observer.observe(source) == expected


def test_actual_hybrid_qwen_post_block_capture_and_identity_lens(tiny_model):
    observer = tiny_model
    bank = TranslationBank(32, observer.spec["layers"])
    observed = observer.observe(rows("skill", 2, "probe", 2), lens=bank)
    assert all(row["terminal_max_error"] <= 1e-5 for row in observed)
    assert all(row["frozen"] == row["tuned"] for row in observed)
    assert all(not layer._forward_hooks for layer in observer.text.layers)


def test_actual_lens_fit_cache_and_saved_reload_preserve_entire_model(tiny_model, tmp_path):
    observer = tiny_model
    spec = copy.deepcopy(observer.spec)
    spec["lens"].update(updates=4, token_batch_size=4, positions_per_prompt=2)
    data = {"lens_fit": rows("background", 0, "train", 2), "lens_check": rows("background-check", 0, "train", 2)}
    data["lens_check"][0]["input_ids"] = [1, 8, 9]
    data["lens_check"][1]["input_ids"] = [1, 10, 11]
    before = parameter_hashes(observer.model)
    _, result = train_lens(observer, data, spec, tmp_path, 37)
    assert result["reload_exact"] and result["model_parameters_unchanged"]
    assert parameter_hashes(observer.model) == before
    assert result["fit"]["tokens"] == result["check"]["tokens"] == 4
    assert len(result["history"]) == 4


def test_actual_repair_controls_and_matched_reference_share_support_and_schedule(tiny_model, design, tmp_path):
    observer = tiny_model
    config, _ = design
    config["repair_updates"] = 2
    spec = observer.spec
    stream = fake_stream()
    for unit in stream["units"].values():
        for role in unit:
            unit[role] = unit[role][:2]
    intent = stream["old"][0]
    observer.checkpoint(tmp_path / "acquired")
    support = stream["units"][intent]["repair"]
    optimizer = observer.optimizer()
    observer.update(support, [[0, 1]] * 3, optimizer)
    observer.checkpoint(tmp_path / "late")
    baseline = {}
    guard_rows = [row for name in stream["new"] for row in stream["units"][name]["test"]]
    for phase, name in (("before", "acquired"), ("after", "late")):
        observer.reset(tmp_path / name)
        baseline[phase] = {
            "test": observer.observe(stream["units"][intent]["test"]),
            "guard": observer.observe(guard_rows),
        }
    first = perform_actions(
        observer,
        config,
        spec,
        stream,
        intent,
        tmp_path / "acquired",
        tmp_path / "late",
        baseline,
        tmp_path / "actions-main",
    )
    second = perform_actions(
        observer,
        config,
        spec,
        stream,
        intent,
        tmp_path / "acquired",
        tmp_path / "late",
        baseline,
        tmp_path / "actions-reference",
    )
    assert first == second
    assert first["none"]["test"] == baseline["after"]["test"]
    assert first["restore"]["test"] == baseline["before"]["test"]
    assert first["replay_target"]["total_exposures"] == first["replay_balanced"]["total_exposures"] == 16
    assert first["replay_target"]["old_exposures"] == 16
    assert first["replay_balanced"]["old_exposures"] == first["replay_balanced"]["new_exposures"] == 8
    assert first["replay_balanced"]["schedule_sha256"] == first["sham"]["schedule_sha256"]


def test_kl_direction_teacher_detach_and_full_vocabulary():
    teacher = torch.tensor([[2.0, 0.0, -2.0]], requires_grad=True)
    student = torch.tensor([[-1.0, 2.0, 1.0]], requires_grad=True)
    loss = forward_kl(teacher, student)
    expected = torch.nn.functional.kl_div(student.log_softmax(-1), teacher.softmax(-1), reduction="batchmean")
    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss, forward_kl(student, teacher))
    loss.backward()
    assert teacher.grad is None and torch.count_nonzero(student.grad) == 3


def test_tuned_translation_learns_known_affine_mismatch_on_heldout_activations():
    torch.manual_seed(13)
    head = torch.nn.Linear(8, 24, bias=False).requires_grad_(False)
    model = torch.nn.Sequential(head)
    observer = SimpleNamespace(
        model=model,
        head=head,
        text=SimpleNamespace(norm=torch.nn.Identity()),
        device=torch.device("cpu"),
        depth=2,
        spec={"layers": [1]},
    )
    fit, check = torch.randn(512, 8), torch.randn(128, 8)
    transform = torch.eye(8).roll(1, 0)
    fit_cache = {"1": fit, "2": fit @ transform + 0.5}
    check_cache = {"1": check, "2": check @ transform + 0.5}
    bank = TranslationBank(8, [1])
    before = evaluate_translations(bank, observer, check_cache, 32)
    frozen = head.weight.clone()
    fit_translations(bank, observer, fit_cache, {"learning_rate": 0.25, "updates": 64, "token_batch_size": 32}, 47)
    after = evaluate_translations(bank, observer, check_cache, 32)
    assert after["1"]["tuned"] < before["1"]["tuned"] * 0.3
    assert torch.equal(frozen, head.weight)
    assert head.weight.grad is None


def test_cohort_separation_and_deterministic_balanced_schedules(design):
    _, spec = design
    data = read_json(ROOT / "inputs/prospective-cohort.json")
    validate_data(data, spec)
    stream = data["streams"]["train-a"]
    source = [row for intent in stream["old"] for row in stream["units"][intent]["learn"]]
    first = balanced_batches(source, 128, 8, 19)
    assert first == balanced_batches(source, 128, 8, 19)
    assert first != balanced_batches(source, 128, 8, 21)
    assert all(len({source[i]["code"] for i in batch}) == 8 for batch in first)
    assert sum(len(batch) for batch in first) == 1024


@pytest.mark.parametrize("mutation", ["intent", "code", "text", "source", "tokens", "lens", "split"])
def test_prospective_leakage_rejected(design, mutation):
    _, spec = design
    data = read_json(ROOT / "inputs/prospective-cohort.json")
    a = next(iter(data["streams"]["train-a"]["units"].values()))
    b = next(iter(data["streams"]["test-a"]["units"].values()))
    if mutation == "intent":
        data["streams"]["test-a"]["old"][0] = data["streams"]["train-a"]["old"][0]
    elif mutation == "code":
        b["learn"][0]["target"] = 999
    elif mutation == "text":
        b["learn"][0]["text"] = a["learn"][0]["text"]
        b["learn"][0]["text_sha256"] = a["learn"][0]["text_sha256"]
    elif mutation == "tokens":
        b["learn"][0]["input_ids"] = a["learn"][0]["input_ids"]
    elif mutation == "source":
        b["learn"][0]["source_index"] = a["learn"][0]["source_index"]
    elif mutation == "lens":
        data["lens_fit"][0]["intent"] = data["streams"]["test-a"]["old"][0]
    else:
        data["streams"]["test-a"]["split"] = "train"
    with pytest.raises(ValueError, match="PROSPECTIVE_"):
        validate_data(data, spec)


def test_matched_repairs_count_old_and_new_exposures(design):
    _, spec = design
    spec["repair"]["updates"] = 16
    target, guard = rows("old", 0, "train", 8), rows("new", 8, "train", 64)
    mixed, plan = repair_batches(target, guard, spec, 13)
    assert all(sum(mixed[i]["intent"] == "old" for i in batch) == 4 for batch in plan)
    assert len(plan) == 16 and sum(len(batch) for batch in plan) == 128
    assert repair_batches(target, guard, spec, 13) == (mixed, plan)
    assert sum(len(batch) for batch in balanced_batches(target, 16, 8, 19)) == 128


class LearningEngine:
    def __init__(self, config, acquired_score=1.0, new_score=1.0, after_score=0.0, initial_score=0.0):
        self.config = config
        self.model = torch.nn.Linear(2, 2).requires_grad_(False)
        self.acquired_score, self.new_score, self.after_score = acquired_score, new_score, after_score
        self.initial_score = initial_score
        self.clock = 0
        self.updates = []
        self.observed_roles = []

    def optimizer(self):
        return SimpleNamespace()

    def optimizer_steps(self, optimizer):
        return [self.clock]

    def checkpoint(self, directory, optimizer):
        directory.mkdir(parents=True)
        write_json(directory / "state.json", {"clock": self.clock})
        return {"state.json": file_hash(directory / "state.json")}

    def reset(self, directory):
        self.clock = read_json(directory / "state.json")["clock"]

    def observe(self, source):
        self.observed_roles.extend(row["source_split"] for row in source)
        groups, counters = {}, {}
        for row in source:
            intent = row["intent"]
            groups[intent] = groups.get(intent, 0) + 1
        result = []
        for row in source:
            intent = row["intent"]
            if self.clock == 0:
                score = self.initial_score if intent.startswith("old") else 0.0
            elif intent.startswith("old"):
                score = self.acquired_score if self.clock <= self.config["acquisition_updates"] else self.after_score
            else:
                score = self.new_score if self.clock > self.config["acquisition_updates"] else 0.0
            index = counters.get(intent, 0)
            counters[intent] = index + 1
            result.extend(records([row], int(index < round(groups[intent] * score))))
        return result

    def update(self, source, batches, optimizer):
        self.updates.extend(batches)
        self.clock += len(batches)
        return [{"step": index + 1, "rows": batch} for index, batch in enumerate(batches)]


def fake_stream(name="train-a", split="train"):
    old, new = [f"old-{name}-{i}" for i in range(8)], [f"new-{name}-{i}" for i in range(8)]
    return {
        "id": name,
        "split": split,
        "seed": 13,
        "old": old,
        "new": new,
        "units": {
            intent: {role: rows(intent, i, role) for role in ("learn", "gate", "probe", "repair", "test")}
            for i, intent in enumerate(old + new)
        },
    }


def test_acquisition_failure_stops_before_forgetting_test_lens_or_repair(design, tmp_path):
    config, spec = design
    engine = LearningEngine(config, acquired_score=0.7)
    result = run_learning(engine, config, spec, fake_stream(), tmp_path)
    assert result["status"] == "acquisition_failed"
    assert result["forgetting_updates"] == result["repair_updates"] == result["lens_updates"] == 0
    assert len(engine.updates) == 128
    assert "test" not in engine.observed_roles
    assert not (tmp_path / "forgotten").exists()


def test_preexisting_correct_code_bias_is_not_counted_as_new_acquisition(design, tmp_path):
    config, spec = design
    engine = LearningEngine(config, initial_score=1.0)
    result = run_learning(engine, config, spec, fake_stream(), tmp_path)
    assert result["status"] == "acquisition_failed" and result["acquired"] == []
    assert result["forgetting_updates"] == result["repair_updates"] == 0


@pytest.mark.parametrize("new_score,after_score", [(0.7, 0.0), (1.0, 0.6)])
def test_invalid_forgetting_stops_before_interventions(design, tmp_path, new_score, after_score):
    config, spec = design
    engine = LearningEngine(config, new_score=new_score, after_score=after_score)
    result = run_learning(engine, config, spec, fake_stream(), tmp_path)
    assert result["status"] == "forgetting_failed"
    assert result["repair_updates"] == result["test_rows_evaluated"] == 0
    assert result["optimizer_steps"] == {"acquired": [128], "final": [256]}


def test_global_gate_recomputes_all_streams_and_rejects_underpowered_cohort(design, tmp_path):
    config, spec = design
    data = {"streams": {item["id"]: fake_stream(item["id"], item["split"]) for item in spec["streams"]}}
    sources = []
    for stream in data["streams"].values():
        path = tmp_path / stream["id"]
        path.mkdir()
        write_json(path / "seal.json", make_seal(config, spec, data, "learn"))
        result = run_learning(LearningEngine(config), config, spec, stream, path)
        finish(path, result)
        sources.append(path)
    result = qualify_learning(sources, config, spec, data)
    assert result["qualified"] and result["counts"] == {"train": 16, "test": 16}
    with pytest.raises(ValueError, match="ALL_PREDECLARED_STREAMS"):
        qualify_learning(sources[:-1], config, spec, data)
    spec["gates"]["minimum_forgotten"]["train"] = 17
    for path in sources:
        with pytest.raises(ValueError, match="LEARNING_SOURCE_BINDING"):
            qualify_learning([path], config, spec, data)


def test_failed_eligibility_receipt_cannot_open_action_phase(design, tmp_path):
    config, spec = design
    data = {"streams": {}}
    write_json(tmp_path / "seal.json", make_seal(config, spec, data, "qualify"))
    finish(tmp_path, {"stage": "qualify", "qualified": False})
    with pytest.raises(ValueError, match="GLOBAL_ELIGIBILITY_REQUIRED"):
        checked_qualification(tmp_path, config, spec, data)


def test_failed_small_repair_pilot_cannot_open_full_lens_or_action_stage(design, tmp_path):
    config, spec = design
    write_json(tmp_path / "seal.json", make_seal(config, spec, {}, "pilot-qualify"))
    finish(tmp_path, {"stage": "pilot-qualify", "qualified": False})
    with pytest.raises(ValueError, match="SMALL_REPAIR_PILOT_REQUIRED"):
        checked_pilot_qualification(tmp_path, config, spec, {}, tmp_path / "unused-cohort")


def test_receipt_rejects_changed_or_unrecorded_artifacts(design, tmp_path):
    config, spec = design
    write_json(tmp_path / "seal.json", make_seal(config, spec, {}, "learn"))
    write_json(tmp_path / "observations.json", {"score": 1})
    finish(tmp_path, {"stage": "learn"})
    verify_result(tmp_path)
    (tmp_path / "observations.json").write_text("{}")
    with pytest.raises(ValueError, match="ARTIFACT_HASH"):
        verify_result(tmp_path)


def sample_units():
    units = []
    for i in range(4):
        source, guard = rows(f"old-{i}", i, "test"), rows("new", 9, "test")
        units.append(
            {
                "id": f"train-a/old-{i}",
                "intent": f"old-{i}",
                "stream": "train-a",
                "split": "train",
                "baseline": {
                    "before": {"test": records(source, 10), "guard": records(guard, 0)},
                    "after": {"test": records(source, 0), "guard": records(guard, 10)},
                },
                "actions": {
                    action: {"test": records(source, 0), "guard": records(guard, 10)}
                    for action in ("none", "replay_target", "replay_balanced")
                },
                "features": {key: [float(i), float(i**2)] for key in ("confidence", "output", "frozen", "tuned")},
            }
        )
    return units


def test_all_zero_repairs_block_heldout_but_variable_continuous_gains_pass(design):
    _, spec = design
    units = sample_units()
    assert not repair_feasibility(units, spec)["qualified"]
    for i, unit in enumerate(units):
        source = rows(f"old-{i}", i, "test")
        unit["actions"]["replay_target"]["test"] = records(source, i + 2)
        unit["actions"]["replay_balanced"]["test"] = records(source, i + 1)
    check = repair_feasibility(units, spec)
    assert check["qualified"]
    assert check["single_class_binary_actions"] == ["replay_target", "replay_balanced"]
    units[0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        repair_feasibility(units, spec)


def test_predictor_is_fitted_once_on_training_skills_and_scaler_is_sealed(design):
    _, spec = design
    units = sample_units()
    first = fit_predictors(units, spec)
    assert first == fit_predictors(units, spec)
    units[0]["split"] = "test"
    with pytest.raises(ValueError, match="PREDICTOR_TRAIN_ONLY"):
        fit_predictors(units, spec)
    model = ridge_fit([[0], [1], [2]], [0, 0.5, 1], 10)
    np.testing.assert_array_equal(ridge_apply(model, [[0.5]]), ridge_apply(model, [[0.5], [1e10]])[:1])


def test_test_outcomes_cannot_change_fitted_predictions_or_action_choices(design):
    _, spec = design
    training = sample_units()
    fitted = fit_predictors(training, spec)
    heldout = copy.deepcopy(training)
    for i, unit in enumerate(heldout):
        unit.update(id=f"test-a/heldout-{i}", stream="test-a", split="test")
    first = evaluate_predictors(fitted, heldout, spec)
    for unit in heldout:
        for action in unit["actions"].values():
            for row in action["test"]:
                row["output"]["prediction"] = row["target"]
    second = evaluate_predictors(fitted, heldout, spec)
    for action in ("replay_target", "replay_balanced"):
        assert (
            first["recovery_prediction"][action]["predictions"] == second["recovery_prediction"][action]["predictions"]
        )
    for name in first["policies"]:
        assert first["policies"][name]["choices"] == second["policies"][name]["choices"]
    assert "test-a" in second["run_level"]


def test_probe_features_ignore_outcomes_and_matched_lenses_have_equal_dimension():
    metric = {"top_probability": 0.5, "entropy": 1.0, "top_margin": 0.1, "target_logp": -1.0, "target_margin": 0.2}
    source = [{"output": metric, "frozen": {"8": metric}, "tuned": {"8": metric}}]
    vector = feature_vector(source, source, [8])
    source[0]["test_accuracy"] = 999
    assert feature_vector(source, source, [8]) == vector
    assert len(vector["frozen"]) == len(vector["tuned"]) == 14


def test_exact_acquisition_and_drop_thresholds_and_followup_binding(design):
    config, spec = design
    assert eligible({"a": 0.5, "b": 0.4}, {"a": 0.8, "b": 0.7}, {"a": 0.5, "b": 0.0}, spec) == ["a"]
    assert learned({"a": 1.0, "b": 0.6, "c": 0.5, "d": 0.7}, {"a": 1.0, "b": 0.8, "c": 0.8, "d": 1.0}, spec) == [
        "c",
        "d",
    ]
    stronger = dict(config, repair_updates=64)
    assert learning_identity(config, spec, {}) == learning_identity(stronger, spec, {})
    assert learning_identity(config, spec, {}) != learning_identity(dict(config, acquisition_updates=256), spec, {})


def test_changed_learning_mechanism_is_rejected_before_source_observations(design, tmp_path, monkeypatch):
    config, spec = design
    write_json(tmp_path / "seal.json", make_seal(config, spec, {}, "learn"))
    finish(tmp_path, {"stage": "learn"})
    original = learning_implementation()
    monkeypatch.setattr(
        prospective_module, "learning_implementation", lambda: original | {"prospective_model.py": "0" * 64}
    )
    with pytest.raises(ValueError, match="LEARNING_SOURCE_BINDING"):
        qualify_learning([tmp_path], config, spec, {})


def test_dispatcher_changes_do_not_change_learning_identity(design, monkeypatch):
    config, spec = design
    expected = learning_identity(config, spec, {})
    original = prospective_module.file_hash

    def changed_dispatcher(path):
        return "0" * 64 if str(path).endswith("prospective_dispatch.py") else original(path)

    monkeypatch.setattr(prospective_module, "file_hash", changed_dispatcher)
    assert learning_identity(config, spec, {}) == expected
    assert make_seal(config, spec, {}, "learn")["payload"]["implementation"]["prospective_dispatch.py"] == "0" * 64


def test_compute_dispatch_contract_existing_directory_and_sealed_child(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"study_config": "configs/prospective.json", "stage": "learn", "stream": "train-a"})
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    received = []
    monkeypatch.setattr(prospective_dispatch, "run_prospective", received.append)
    monkeypatch.setattr(
        sys, "argv", ["prospective_dispatch.py", "--config", str(manifest), "--output-dir", str(run_dir)]
    )
    prospective_dispatch.main()
    assert received == [
        [
            "--config",
            str(ROOT / "configs/prospective.json"),
            "--stage",
            "learn",
            "--output-dir",
            str(run_dir / "study"),
            "--stream",
            "train-a",
        ]
    ]
    assert run_dir.is_dir() and not (run_dir / "study").exists()


def test_reference_omits_old_intents_and_uses_exact_saved_new_task_schedule(design, tmp_path):
    config, spec = design
    stream = fake_stream()
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    engine = LearningEngine(config)
    run_learning(engine, config, spec, stream, main_dir)
    write_json(main_dir / "receipt.json", {"cpu_fixture": True})
    (main_dir / "initial" / "adapter_model.safetensors").write_bytes(b"fixture")
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    result = learn_reference(LearningEngine(config), config, spec, stream, main_dir, ref_dir)
    assert result["old_training_exposures"] == result["test_rows_evaluated"] == 0
    trace = read_json(ref_dir / "updates.json")
    assert {row["intent"] for row in trace["rows"]} == set(stream["new"])
    assert [step["rows"] for step in trace["updates"]] == [
        step["rows"] for step in read_json(main_dir / "forgetting-updates.json")
    ]
