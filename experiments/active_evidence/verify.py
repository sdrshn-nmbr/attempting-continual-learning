import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from consolidation import training_plan
from environment import (
    FAMILY,
    candidate_pool,
    collect_device,
    digest,
    evaluate_rows,
    make_rules,
    partition,
    posterior,
    prompt,
    stage_devices,
)
from peft import LoraConfig, PeftModel, get_peft_model
from run import parameter_hash, summarize, write_json
from torch.nn import functional
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration


def check_consolidation(config):
    seed = config["seed"]
    settings = config["environment"]
    hidden = make_rules(seed)
    acquisition, heldout = partition(seed)
    checks = []
    for arm in ("random", "hypothesis_elimination_oracle"):
        ledgers = []
        calls = []
        for stage in range(2):
            ledger = []
            for device in stage_devices(stage):
                flips = (
                    settings["noise_flip_steps"] if stage == 1 and device == 3 else ()
                )

                def observe(
                    value,
                    step,
                    rule=hidden[stage][device],
                    noise=flips,
                    observations=calls,
                ):
                    observations.append((value, step))
                    return rule(value) ^ int(step in noise)

                evidence, _ = collect_device(
                    seed, stage, device, arm, settings, observe
                )
                ledger.extend(evidence)
            ledgers.append(ledger)
        assert len(calls) == settings["queries_per_device"] * 8
        audits = {}
        plans = {}
        for source in ("observed_query", "posterior_predictive"):
            variant = copy.deepcopy(config)
            variant["consolidation"]["label_source"] = source
            variant["consolidation"]["old_examples_per_batch"] = 2
            stage_audits = []
            stage_plans = []
            for stage in range(2):
                old = [row for row in ledgers[0] if row["device"] == 0] if stage else []
                with patch(
                    "environment.make_rules",
                    side_effect=AssertionError("HIDDEN_RULE_ACCESS"),
                ):
                    rows, audit = training_plan(variant, stage, ledgers[stage], old)
                    repeated = training_plan(variant, stage, ledgers[stage], old)
                assert (rows, audit) == repeated
                assert (
                    len(rows)
                    == config["training"]["updates_per_stage"]
                    * config["training"]["batch_size"]
                )
                assert all(
                    row["input"] in acquisition and row["input"] not in heldout
                    for row in rows
                )
                observed = {
                    (row["device"], row["input"], row["observed"])
                    for row in ledgers[stage] + old
                }
                if source == "observed_query":
                    assert all(
                        (row["device"], row["input"], row["label"]) in observed
                        for row in rows
                    )
                else:
                    assert audit["inferred_unqueried_examples"] > 0
                batch_size = variant["training"]["batch_size"]
                for start in range(0, len(rows), batch_size):
                    batch = rows[start : start + batch_size]
                    assert sum(row["old_replay"] for row in batch) == (
                        2 if stage else 0
                    )
                    assert all(
                        row["device"] == 0 and row["label_source"] == "observed_query"
                        for row in batch
                        if row["old_replay"]
                    )
                poisoned = copy.deepcopy(ledgers[stage])
                poisoned[0]["evaluator_only_clean_label"] = 1 - poisoned[0]["observed"]
                try:
                    training_plan(variant, stage, poisoned, old)
                except ValueError as error:
                    assert "QUERIED_LEDGER_ONLY" in str(error)
                else:
                    raise AssertionError(
                        "Evaluator labels were accepted by the learner"
                    )
                if source == "posterior_predictive":
                    inverted = [
                        {**row, "observed": 1 - row["observed"]}
                        for row in ledgers[stage]
                    ]
                    inverted_rows, _ = training_plan(variant, stage, inverted, old)
                    assert [(row["device"], row["input"]) for row in inverted_rows] == [
                        (row["device"], row["input"]) for row in rows
                    ]
                    changed = sum(
                        a["label"] != b["label"]
                        for a, b in zip(rows, inverted_rows, strict=True)
                        if not a["old_replay"]
                    )
                    assert changed > audit["current_examples"] * 0.8
                stage_audits.append(audit)
                stage_plans.append(rows)
            audits[source] = stage_audits
            plans[source] = stage_plans
        for stage in range(2):
            a, b = (audits[source][stage] for source in audits)
            for key in (
                "examples",
                "current_examples",
                "old_examples",
                "examples_by_device",
                "current_query_ledger_sha256",
                "old_query_ledger_sha256",
            ):
                assert a[key] == b[key], (stage, key)
            assert [
                row for row in plans["observed_query"][stage] if row["old_replay"]
            ] == [
                row for row in plans["posterior_predictive"][stage] if row["old_replay"]
            ]
        checks.append(
            {"arm": arm, "environment_calls": len(calls), "conditions": audits}
        )
    return {
        "checks": "passed",
        "seed": seed,
        "conditions": checks,
        "hidden_label_poison_rejected": True,
        "hidden_rule_api_unavailable_during_consolidation": True,
        "queried_observations_control_posterior_labels": True,
        "identical_ledgers_across_consolidation_conditions": True,
        "matched_device_and_old_replay_exposure": True,
        "claim_limit": "CPU data-flow and budget proof; no model learning result.",
    }


def promotion_fixture(arm, stable_change, config):
    settings = config["training"]
    training = {
        "updates": settings["updates_per_stage"],
        "input_tokens": 130 * settings["batch_size"] * settings["updates_per_stage"],
        "padded_input_tokens": 130
        * settings["batch_size"]
        * settings["updates_per_stage"],
        "supervised_tokens": settings["batch_size"] * settings["updates_per_stage"],
        "examples": settings["batch_size"] * settings["updates_per_stage"],
        "old_replay_examples": 0,
        "current_examples": settings["batch_size"] * settings["updates_per_stage"],
    }
    return {
        "arm": arm,
        "environment_queries": 8 * config["environment"]["queries_per_device"],
        "adapter": {"initial_adapter_sha256": "same-synthetic-initialization"},
        "stable_retention_change": stable_change,
        "new_skill_learning_gain": 0.25,
        "stages": [
            {
                "training": training,
                "metrics": {"single": {"accuracy": 0.8}, "stable": {"accuracy": 1.0}},
                "before_metrics": {"single": {"accuracy": 0.5}},
                "training_fit": {"single": {"accuracy": 1.0}},
            },
            {
                "training": training,
                "metrics": {
                    "single": {"accuracy": 0.6 if arm == "random" else 0.7},
                    "stable": {"accuracy": 1.0 + stable_change},
                    "composition": {"accuracy": 0.7},
                },
            },
        ],
    }


def check_promotion(config):
    full_config = copy.deepcopy(config)
    full_config["screen"]["qualification_only"] = False
    threshold = full_config["screen"]["maximum_stable_forgetting"]
    assert threshold == 0.05
    cases = [
        ("AE-001-total-forgetting", -1.0, -1.0, False, "discard_screen_hypothesis"),
        (
            "within-absolute-limit",
            -0.04,
            -0.04,
            False,
            "keep_for_independent_replication",
        ),
        ("at-absolute-limit", -0.05, -0.05, False, "keep_for_independent_replication"),
        (
            "exceeds-absolute-limit",
            -0.0501,
            -0.0501,
            False,
            "discard_screen_hypothesis",
        ),
        (
            "relative-limit-still-enforced",
            -0.04,
            -0.10,
            False,
            "discard_screen_hypothesis",
        ),
        ("mini-remains-qualification-only", -1.0, -1.0, True, "qualification_only"),
    ]
    checks = []
    for name, candidate_change, random_change, qualification_only, expected in cases:
        case_config = copy.deepcopy(full_config)
        case_config["screen"]["qualification_only"] = qualification_only
        results = [
            promotion_fixture("random", random_change, case_config),
            promotion_fixture(
                "hypothesis_elimination_oracle", candidate_change, case_config
            ),
        ]
        if name == "relative-limit-still-enforced":
            results[0]["stable_retention_change"] = 0.02
            results[0]["stages"][0]["metrics"]["stable"]["accuracy"] = 0.9
            results[0]["stages"][1]["metrics"]["stable"]["accuracy"] = 0.92
        candidate = summarize(results, case_config)["decisions"][1]
        assert candidate["decision"] == expected, (name, candidate)
        assert candidate["stable_retention_change"] == candidate_change
        assert candidate["maximum_stable_forgetting"] == threshold
        checks.append(
            {"case": name, "expected_decision": expected, "actual": candidate}
        )
    return {
        "checks": "passed",
        "review_finding": "AE-001",
        "maximum_stable_forgetting": threshold,
        "cases": checks,
        "proof_limit": "Synthetic decision records only; no model training, GPU execution, or mini rerun.",
    }


def check_environment(config):
    settings = config["environment"]
    measurements = []
    for seed in (config["seed"], 29, 43, 71):
        acquisition, heldout = partition(seed)
        assert not set(acquisition) & set(heldout)
        assert set(acquisition) | set(heldout) == set(range(1024))
        first, second = make_rules(seed)
        assert all(
            first[0](x) == second[0](x) and first[3](x) == second[3](x)
            for x in range(1024)
        )
        assert sum(first[1](x) != second[1](x) for x in range(1024)) == 512
        assert all((first[2](x) != second[2](x)) == bool(x & 16) for x in range(1024))
        for stage in range(2):
            rules = (first, second)[stage]
            rows = evaluate_rows(seed, stage, settings)
            assert all(row["input"] in heldout for row in rows)
            for row in rows:
                expected = 0
                for device in row["devices"]:
                    expected ^= rules[device](row["input"])
                assert row["label"] == expected
            for device in stage_devices(stage):
                pool = candidate_pool(seed, stage, device, settings)
                assert set(pool).issubset(acquisition)
                assert any(
                    all(rule(x) == rules[device](x) for x in range(32))
                    for rule in FAMILY
                )
                shared = []
                for arm in ("random", "hypothesis_elimination_oracle"):
                    calls = []

                    flip_steps = (
                        settings["noise_flip_steps"]
                        if stage == 1 and device == 3
                        else ()
                    )

                    def observe(
                        value,
                        step,
                        observations=calls,
                        rule=rules[device],
                        flips=flip_steps,
                    ):
                        observations.append((value, step))
                        return rule(value) ^ int(step in flips)

                    evidence, events = collect_device(
                        seed, stage, device, arm, settings, observe
                    )
                    assert len(calls) == settings["queries_per_device"]
                    assert {row["input"] for row in evidence}.issubset(acquisition)
                    assert len(evidence) == len(events)
                    assert (
                        len({row["input"] for row in evidence})
                        == len(evidence) - settings["repeat_queries"]
                    )
                    assert all(
                        "label" not in row and "rule" not in row for row in evidence
                    )
                    repeated, repeated_events = collect_device(
                        seed, stage, device, arm, settings, observe
                    )
                    assert repeated == evidence and repeated_events == events
                    shared.append(evidence[: settings["common_queries"]])
                    weights, state = posterior(evidence, settings["assumed_noise"])
                    assert math.isclose(sum(weights), 1.0, abs_tol=1e-9)
                    relevant = [
                        row
                        for row in rows
                        if row["kind"] == "single" and row["device"] == device
                    ]
                    accuracy = sum(
                        int(
                            sum(
                                weight * rule(row["input"])
                                for weight, rule in zip(weights, FAMILY, strict=True)
                            )
                            > 0.5
                        )
                        == row["label"]
                        for row in relevant
                    ) / len(relevant)
                    measurements.append(
                        {
                            "seed": seed,
                            "stage": stage,
                            "device": device,
                            "arm": arm,
                            "analytic_accuracy": accuracy,
                            "posterior_entropy": state["entropy_bits"],
                        }
                    )
                assert shared[0] == shared[1]
    averages = {
        arm: sum(row["analytic_accuracy"] for row in measurements if row["arm"] == arm)
        / sum(row["arm"] == arm for row in measurements)
        for arm in ("random", "hypothesis_elimination_oracle")
    }
    return {
        "checks": "passed",
        "seeds": [config["seed"], 29, 43, 71],
        "analytic_acquisition_upper_bound": averages,
        "measurements": measurements,
        "claim_limit": "Analytic known-family inference only; no pretrained-model learning or generalization has been measured.",
    }


def check_tokenizer(config, output):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        cache_dir=output / "tokenizer-cache",
    )
    acquisition, _ = partition(config["seed"])
    prompts = [
        prompt({"device": device, "input": value})
        for device in range(5)
        for value in acquisition
    ]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in prompts
    ]
    sizes = {
        len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]
    }
    answer_ids = [
        tokenizer.encode(str(label), add_special_tokens=False) for label in range(2)
    ]
    assert len(sizes) == 1, f"Unequal training token lengths: {sizes}"
    assert all(len(tokens) == 1 for tokens in answer_ids)
    return {
        "checks": "passed",
        "prompts_checked": len(prompts),
        "training_input_tokens": next(iter(sizes)),
        "answer_ids": answer_ids,
        "prompt_sha256": digest(prompts),
    }


def check_sdk(output):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "num_position_embeddings": 16,
        },
    )
    base = Qwen3_5ForConditionalGeneration(config)
    original_base = copy.deepcopy(base.state_dict())
    targets = [
        name
        for name, module in base.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ".language_model." in name
        and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj", "in_proj_qkv", "out_proj")
    ]
    assert {name.rsplit(".", 1)[-1] for name in targets} == {
        "q_proj",
        "v_proj",
        "in_proj_qkv",
        "out_proj",
    }
    model = get_peft_model(
        base, LoraConfig(r=2, lora_alpha=4, target_modules=targets, lora_dropout=0.0)
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    inputs = torch.randint(0, 64, (2, 8))
    mask = torch.ones_like(inputs)
    labels = torch.tensor([5, 6])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.02,
    )
    initial_hash = parameter_hash(model, trainable_only=True)
    losses = []
    for _ in range(4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(
            model(
                input_ids=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1
            )
            .logits[:, -1]
            .float(),
            labels,
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
            error_if_nonfinite=True,
        )
        assert torch.isfinite(loss) and float(norm) > 0
        assert all(
            parameter.grad is None
            for parameter in model.parameters()
            if not parameter.requires_grad
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        before = model(
            input_ids=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits
    trained_hash = parameter_hash(model, trainable_only=True)
    assert trained_hash != initial_hash and losses[-1] < losses[0]
    checkpoint = output / "tiny-checkpoint"
    model.save_pretrained(
        checkpoint, safe_serialization=True, save_embedding_layers=False
    )
    fresh = Qwen3_5ForConditionalGeneration(config)
    fresh.load_state_dict(original_base)
    reloaded = PeftModel.from_pretrained(fresh, checkpoint, is_trainable=True).eval()
    assert parameter_hash(reloaded, trainable_only=True) == trained_hash
    with torch.inference_mode():
        after = reloaded(
            input_ids=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    unloaded = reloaded.unload()
    assert set(unloaded.state_dict()) == set(original_base)
    assert all(
        torch.equal(unloaded.state_dict()[name], value)
        for name, value in original_base.items()
    )
    return {
        "checks": "passed",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft")
        },
        "tiny_random_model_cpu_only": True,
        "losses": losses,
        "adapter_changed": True,
        "reload_tensor_equality": True,
        "reload_logit_equality": True,
        "frozen_base_equality": True,
        "target_modules": targets,
        "claim_limit": "This validates the SDK path on a tiny random CPU model. It does not qualify ROCm or Qwen3.5-4B learning.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    result = {
        "promotion": check_promotion(config),
        "consolidation": check_consolidation(config),
        "environment": check_environment(config),
    }
    write_json(args.output_dir / "verification.json", result)
    result["sdk"] = check_sdk(args.output_dir)
    write_json(args.output_dir / "verification.json", result)
    result["tokenizer"] = check_tokenizer(config, args.output_dir)
    write_json(args.output_dir / "verification.json", result)
    print(
        json.dumps(
            {
                name: {
                    key: value for key, value in proof.items() if key != "measurements"
                }
                for name, proof in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
