from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch
from portallib import PortalModel, PortalProjectionTarget
from safetensors.torch import load_file
from test_contract import ROOT, tiny_model

import run
from anchoring import AnchoredSequenceLearner, anchor_schedule
from data import SEQUENCE_TASKS, prepare_data, read_data
from learner import (
    SequenceLearner,
    extend_portal,
    parameter_plan,
    state_hash,
    tensor_hash,
)
from repair_alignment import PUBLISHED_TASKS


def protocol():
    frozen = json.loads(
        (ROOT / "configs/repair/sequence303_native_replay.json").read_text()
    )
    return {
        "train_tasks": frozen["train_tasks"],
        "holdout_tasks": frozen["holdout_tasks"],
        "normalization_epsilon": 1e-12,
    }


def tiny_assets():
    base, original = tiny_model()
    published = PortalModel(
        replace(original.config, tasks=PUBLISHED_TASKS),
        torch.randn(14, original.config.d_z),
    )
    published.core.load_state_dict(original.core.state_dict())
    published.alignment.load_state_dict(original.alignment.state_dict())
    targets = {}
    for name, layers, width in (("qwen4", 2, 24), ("mistral7", 3, 40)):
        projections = tuple(
            PortalProjectionTarget(
                layer, module, f"self_attn.{module}_proj", width, output, module, module
            )
            for layer in range(layers)
            for module, output in (("q", width), ("v", width // 2))
        )
        target_config = replace(
            published.config,
            base_model_name_or_path=name,
            n_layers=layers,
            projection_targets=projections,
        )
        target = PortalModel(target_config, published.task_latents)
        target.core.load_state_dict(published.core.state_dict())
        targets[name] = target.requires_grad_(False)
    return base, extend_portal(published, SEQUENCE_TASKS), targets


def source_config():
    return json.loads((ROOT / "configs/sequence_native_replay.json").read_text())


def make_learner(base, portal, targets, weight):
    return AnchoredSequenceLearner(
        base,
        portal,
        SEQUENCE_TASKS,
        0,
        run.learner_recipe(source_config(), parameter_plan(portal)),
        targets=targets,
        protocol=protocol(),
        anchor_weight=weight,
    )


def training_rows(path):
    prepare_data(source_config(), path)
    return read_data(path, tuple(f"{task}_train" for task in SEQUENCE_TASKS))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_three_surfaces_share_one_core_and_initial_identity_is_exact(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    base, portal, targets = tiny_assets()
    base.model.to(device)
    portal.to(device)
    targets = {name: target.to(device) for name, target in targets.items()}
    before = torch.get_rng_state().clone()
    learner = make_learner(base, portal, targets, 1)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(model.core is portal.core for model in learner.anchors.surfaces.values())
    assert all(
        parameter.device == next(portal.core.parameters()).device
        for model in learner.anchors.surfaces.values()
        for parameter in model.parameters()
    )
    assert all(parameter.requires_grad for parameter in portal.core.parameters())
    assert all(
        not parameter.requires_grad
        for model in learner.anchors.surfaces.values()
        for parameter in model.alignment.parameters()
    )
    for task in protocol()["train_tasks"]:
        loss, _ = learner.anchors.loss(task, None, 0)
        assert loss.item() == 0
        gradients = torch.autograd.grad(loss, tuple(portal.core.parameters()))
        assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
    learner.close()


def test_anchor_schedule_is_matched_and_balances_both_memory_pools():
    schedule = anchor_schedule(protocol()["train_tasks"], 96, 90302)
    assert len(schedule) == 288 and schedule == anchor_schedule(
        protocol()["train_tasks"], 96, 90302
    )
    for stage in range(3):
        rows = schedule[stage * 96 : (stage + 1) * 96]
        for start in range(0, 90, 10):
            assert {row["published"] for row in rows[start : start + 10]} == set(
                protocol()["train_tasks"]
            )
        assert {row["previous"] for row in rows} == (
            {None} if stage == 0 else set(SEQUENCE_TASKS[:stage])
        )
    assert sum(row["previous"] == "sequence_a" for row in schedule[192:]) == 48


def test_lambda_zero_matches_original_every_update_rng_and_optimizer_with_dropout(
    tmp_path,
):
    base, portal, targets = tiny_assets()
    source = source_config()
    rows = training_rows(tmp_path)
    schedule = anchor_schedule(protocol()["train_tasks"], 96, 90302)
    outcomes = {}
    calls = {}
    for mode in ("original", "zero"):
        candidate_base = copy.deepcopy(base)
        candidate_portal = copy.deepcopy(portal)
        if mode == "zero":
            candidate = make_learner(
                candidate_base, candidate_portal, copy.deepcopy(targets), 0
            )
        else:
            candidate = SequenceLearner(
                candidate_base,
                candidate_portal,
                SEQUENCE_TASKS,
                0,
                run.learner_recipe(source, parameter_plan(candidate_portal)),
            )
        count = [0]
        handle = candidate_base.model.register_forward_pre_hook(
            lambda *_, counter=count: counter.__setitem__(0, counter[0] + 1)
        )
        trace = []
        torch.manual_seed(198)
        for stage, task in enumerate(SEQUENCE_TASKS):
            if stage:
                candidate.advance_stage()
            frozen = run.frozen_snapshot(candidate)
            for current, reference, identity in run.stage_schedule(
                rows, stage, source["seed"], source["training"]
            ):
                if mode == "zero":
                    candidate.select_anchor(schedule[stage * 96 + identity["step"] - 1])
                measured = candidate.step(current, reference, 0.25)
                assert all(
                    candidate.vectors[old].grad is None
                    for old in SEQUENCE_TASKS[:stage]
                )
                trace.append(
                    (
                        tensor_hash(candidate.named_parameters),
                        state_hash(candidate.optimizer.state_dict()),
                        tensor_hash({"rng": torch.get_rng_state()}),
                        measured["current_loss"],
                        measured["reference_loss"],
                        count[0],
                    )
                )
            assert run.frozen_snapshot(candidate) == frozen
            assert run.check_optimizer_steps(candidate, 96)
            if mode == "zero":
                candidate.anchors.verify_frozen()
                if stage < 2:
                    candidate.anchors.capture_acquired(
                        task, candidate.vectors[task], stage
                    )
        outcomes[mode], calls[mode] = trace, count[0]
        handle.remove()
        candidate.close()
    assert outcomes["zero"] == outcomes["original"]
    assert calls["zero"] == calls["original"] == 360


def dense_anchor_loss(anchors, tasks):
    pools = []
    for task in tasks:
        reference = anchors.references[task]
        surfaces = []
        for name, model in anchors.surfaces.items():
            generated = model(reference.vector)
            scale = model.config.alpha / model.config.rank
            modules = []
            for key, (a, b) in generated.items():
                at, bt = reference.factors[name][key]
                observed = scale * b.double() @ a.double()
                expected = scale * bt.double() @ at.double()
                modules.append(
                    (observed - expected).square().sum()
                    / (expected.square().sum() + 1e-12)
                )
            surfaces.append(torch.stack(modules).mean())
        pools.append(torch.stack(surfaces).mean())
    return torch.stack(pools).mean()


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_full_shared_core_anchor_gradients_match_independent_dense_products(stage):
    base, portal, targets = tiny_assets()
    learner = make_learner(base, portal, targets, 1)
    for prior in range(stage):
        learner.anchors.capture_acquired(
            SEQUENCE_TASKS[prior], learner.vectors[SEQUENCE_TASKS[prior]], prior
        )
        learner.advance_stage()
    with torch.no_grad():
        for parameter in portal.core.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.0005)
    previous = SEQUENCE_TASKS[stage - 1] if stage else None
    implicit, receipt = learner.anchors.loss("rte", previous, stage)
    dense = dense_anchor_loss(
        learner.anchors, ["rte"] if previous is None else ["rte", previous]
    )
    torch.testing.assert_close(implicit, dense, atol=1e-10, rtol=1e-10)
    assert receipt["loss"] > 0
    current_vector_gradient = torch.autograd.grad(
        implicit,
        (learner.vectors[SEQUENCE_TASKS[stage]],),
        allow_unused=True,
        retain_graph=True,
    )
    assert current_vector_gradient == (None,)
    parameters = tuple(portal.core.parameters())
    for actual, expected in zip(
        torch.autograd.grad(implicit, parameters),
        torch.autograd.grad(dense, parameters),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    learner.anchors.verify_frozen()
    learner.close()


def test_acquisition_references_are_immutable_and_cannot_be_refreshed(tmp_path):
    base, portal, targets = tiny_assets()
    learner = make_learner(base, portal, targets, 1)
    original = learner.anchors.verify_frozen()
    learner.anchors.capture_acquired("sequence_a", learner.vectors["sequence_a"], 0)
    recorded = learner.anchors.save_reference("sequence_a", tmp_path)
    assert tensor_hash(load_file(recorded["path"])) == recorded["tensor_sha256"]
    assert learner.anchors.reference_hashes.items() >= original["references"].items()
    with pytest.raises(ValueError, match="BOUNDARY_ORDER"):
        learner.anchors.capture_acquired("sequence_a", learner.vectors["sequence_a"], 0)
    with pytest.raises(ValueError, match="TRAIN_TASK_REQUIRED"):
        learner.anchors.loss(protocol()["holdout_tasks"][0], None, 0)
    with pytest.raises(ValueError, match="PREVIOUS_ACQUIRED"):
        learner.anchors.loss("rte", "sequence_c", 1)
    with torch.no_grad():
        learner.anchors.references["sequence_a"].vector.add_(0.01)
    with pytest.raises(ValueError, match="REFERENCE_CHANGED"):
        learner.anchors.verify_frozen()
    learner.close()


def test_lambda_one_changes_learning_but_not_frozen_state_or_llm_call_count(tmp_path):
    base, portal, targets = tiny_assets()
    rows = training_rows(tmp_path)
    results = {}
    for weight in (0, 1):
        learner = make_learner(
            copy.deepcopy(base), copy.deepcopy(portal), copy.deepcopy(targets), weight
        )
        frozen = run.frozen_snapshot(learner)
        trace = []
        torch.manual_seed(523)
        for step in range(1, 9):
            learner.select_anchor(
                {
                    "stage": 0,
                    "step": step,
                    "published": protocol()["train_tasks"][(step - 1) % 10],
                    "previous": None,
                }
            )
            trace.append(
                learner.step(rows["sequence_a_train"][step : step + 4], [], 0.25)
            )
        assert run.frozen_snapshot(learner) == frozen
        learner.anchors.verify_frozen()
        assert all(
            parameter.grad is None
            for model in learner.anchors.surfaces.values()
            for parameter in model.alignment.parameters()
        )
        results[weight] = {
            "state": tensor_hash(learner.named_parameters),
            "rng": tensor_hash({"rng": torch.get_rng_state()}),
            "traces": trace,
        }
        learner.close()
    assert results[0]["state"] != results[1]["state"]
    assert results[0]["rng"] == results[1]["rng"]
    assert results[1]["traces"][-1]["anchor"]["loss"] > 0
