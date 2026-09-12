import copy
import hashlib
import json
import signal
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from compare import compare_support, support_gates
from config import Config, digest
from data import (
    PreparedData,
    balanced_schedule,
    collate,
    replay_buffer_record,
    replay_buffers,
    retain_replay,
    select_rows,
    tokenize_tasks,
    training_schedules,
    validate_replay_refs,
)
from engine import Experiment, Reporter, StopRequest, file_sha256, replay_screen
from learning import (
    acquisition_summary,
    activation_statistics,
    adapter_layers,
    apply_boundary,
    attach_adapter,
    class_logits,
    evaluate,
    optimizer_for,
    score_logits,
    selective_reset,
    snapshot,
    update_diagnostics,
)
from run import (
    execute,
    model_provenance,
    prepare_output,
    runtime_provenance,
    source_provenance,
)
from verify import execute as verify_saved_adapters
from verify import reload_and_score


def small_config(**overrides):
    values = {
        "model_id": "local/random-tiny-qwen3",
        "revision": "0" * 40,
        "model_path": "/unused",
        "seed": 13,
        "device": "cpu",
        "dtype": "float32",
        "tasks": 2,
        "classes_per_task": 2,
        "train_per_class": 4,
        "validation_per_class": 2,
        "test_per_class": 2,
        "updates_per_task": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "eval_batch_size": 2,
        "eval_every": 1,
        "checkpoint_every": 2,
        "lora_rank": 4,
        "lora_alpha": 8,
        "diagnostic_examples": 4,
        "diagnostic_features": 8,
    }
    return Config(**(values | overrides))


def tiny_base(cfg, hybrid=False):
    torch.manual_seed(cfg.seed)
    if hybrid:
        text = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=256,
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
        configuration = Qwen3_5Config(text_config=text, vision_config=vision)
        configuration._attn_implementation = "sdpa"
        base = Qwen3_5ForConditionalGeneration(configuration)
    else:
        configuration = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=256,
            pad_token_id=0,
        )
        configuration._attn_implementation = "sdpa"
        base = Qwen3ForCausalLM(configuration)
    return base


def tiny_model(cfg, hybrid=False):
    return attach_adapter(tiny_base(cfg, hybrid), cfg)


def fake_splits():
    names = [f"intent_{i}" for i in range(16)] + ["oos"]
    splits = {
        split: [
            {"text": f"{split} class {intent} example {sample}", "intent": intent}
            for intent in range(17)
            for sample in range(40)
        ]
        for split in ("train", "validation", "test")
    }
    return splits, names


def prepared_data(cfg):
    splits, names = fake_splits()
    tasks, selection = select_rows(splits, names, cfg)
    for task in tasks:
        for rows in task.rows.values():
            for row in rows:
                row["input_ids"] = [
                    10 + row["intent"],
                    20 + row["source_index"] % 4,
                    30,
                ]
        task.schedule = balanced_schedule(task.rows["train"], cfg, task.index)
    buffers = replay_buffers(tasks, cfg)
    return PreparedData(
        tasks,
        list(range(2, 2 + cfg.tasks * cfg.classes_per_task)),
        0,
        selection,
        training_schedules(tasks, cfg, buffers),
        buffers,
    )


class InterruptReporter(Reporter):
    def __init__(self, output_dir, cfg, stop, after_updates):
        super().__init__(output_dir, cfg)
        self.stop = stop
        self.after_updates = after_updates
        self.updates = 0

    def event(self, kind, logical_id=None, **values):
        super().event(kind, logical_id, **values)
        if kind == "update":
            self.updates += 1
            if self.updates == self.after_updates:
                self.stop.handle(signal.SIGTERM, None)


class PlasticityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selection_is_disjoint_novel_and_independent_of_methods(self):
        cfg = small_config()
        splits, names = fake_splits()
        splits["validation"].extend(copy.deepcopy(splits["train"]))
        splits["test"].extend(copy.deepcopy(splits["validation"]))
        tasks, selection = select_rows(splits, names, cfg)
        other, other_selection = select_rows(
            splits, names, replace(cfg, methods=["fresh_base"])
        )
        self.assertEqual(selection, other_selection)
        self.assertEqual(tasks, other)
        self.assertTrue(set(tasks[0].intents).isdisjoint(tasks[1].intents))
        all_hashes = [
            row["text_sha256"]
            for task in tasks
            for rows in task.rows.values()
            for row in rows
        ]
        self.assertEqual(len(all_hashes), len(set(all_hashes)))
        train_texts = {row["text"] for row in splits["train"]}
        self.assertTrue(
            all(
                row["text"] not in train_texts
                for task in tasks
                for split in ("validation", "test")
                for row in task.rows[split]
            )
        )
        self.assertNotEqual(
            select_rows(splits, names, replace(cfg, seed=14))[1], selection
        )

    def test_balanced_budget_and_no_heldout_training(self):
        cfg = small_config(
            updates_per_task=9, batch_size=2, gradient_accumulation_steps=2
        )
        prepared = prepared_data(cfg)
        for task in prepared.tasks:
            schedule = balanced_schedule(task.rows["train"], cfg, task.index)
            self.assertEqual(schedule, task.schedule)
            for batch in schedule:
                labels = [task.rows["train"][index]["code"] for index in batch]
                self.assertEqual(len(labels), cfg.effective_batch_size)
                self.assertTrue(all(labels.count(code) == 2 for code in task.codes))
                self.assertTrue(
                    all(
                        task.rows["train"][index]["source_split"] == "train"
                        for index in batch
                    )
                )

    def test_replay_is_deterministic_balanced_and_excludes_future_and_heldout(self):
        cfg = small_config(
            tasks=4,
            classes_per_task=4,
            batch_size=4,
            updates_per_task=32,
            train_per_class=32,
        )
        prepared = prepared_data(cfg)
        self.assertEqual(prepared.schedules, prepared_data(cfg).schedules)
        self.assertEqual(
            prepared.schedules["persistent"][0],
            prepared.schedules["balanced_replay"][0],
        )
        self.assertEqual(
            prepared.schedules["balanced_replay"],
            prepared_data(replace(cfg, methods=["balanced_replay"])).schedules[
                "balanced_replay"
            ],
        )
        heldout = {
            row["text_sha256"]
            for task in prepared.tasks
            for split in ("validation", "test")
            for row in task.rows[split]
        }
        totals = Counter()
        for task_index, schedule in enumerate(prepared.schedules["balanced_replay"]):
            prior_codes = [
                code for task in prepared.tasks[:task_index] for code in task.codes
            ]
            old_counts = Counter({code: 0 for code in prior_codes})
            current_refs = []
            for batch in schedule:
                self.assertEqual(len(batch), 8)
                current_codes = []
                for source, index in batch:
                    row = prepared.tasks[source].rows["train"][index]
                    self.assertLessEqual(source, task_index)
                    self.assertEqual(row["source_split"], "train")
                    self.assertNotIn(row["text_sha256"], heldout)
                    if source == task_index:
                        current_codes.append(row["code"])
                        current_refs.append((source, index))
                        totals["current"] += 1
                    else:
                        old_counts[row["code"]] += 1
                        totals["replay"] += 1
                expected = 2 if task_index == 0 else 1
                self.assertEqual(
                    Counter(current_codes),
                    Counter(
                        {code: expected for code in prepared.tasks[task_index].codes}
                    ),
                )
                if old_counts:
                    self.assertLessEqual(
                        max(old_counts.values()) - min(old_counts.values()), 1
                    )
            self.assertEqual(len(set(current_refs)), 128)
        self.assertEqual(totals, {"current": 640, "replay": 384})
        altered = copy.deepcopy(prepared.tasks)
        for task in altered:
            task.rows["validation"] = []
            task.rows["test"] = []
        self.assertEqual(
            training_schedules(altered, cfg, replay_buffers(altered, cfg)),
            prepared.schedules,
        )

    def test_scores_distinguish_interference_from_discrimination(self):
        logits = torch.tensor([[8.0, 1.0, 9.0, 0.0], [1.0, 8.0, 9.0, 0.0]])
        score = score_logits(logits, torch.tensor([0, 1]), [0, 1])
        self.assertEqual(score["global_correct"], 0)
        self.assertEqual(score["task_local_correct"], 2)
        curve = [
            {
                "updates": u,
                "training_input_tokens": u * 8,
                "accuracy": a,
                "task_local_accuracy": 0.8,
                "loss": 1 - a,
                "task_local_loss": 0.2,
            }
            for u, a in ((0, 0.0), (2, 0.5), (4, 1.0))
        ]
        summary = acquisition_summary(curve, 0.7)
        self.assertEqual(summary["accuracy_auc_per_update"], 0.5)
        self.assertEqual(summary["accuracy_threshold"]["first_observed_update"], 4)
        self.assertEqual(
            summary["task_local_accuracy_threshold"]["first_observed_update"], 0
        )
        self.assertTrue(
            acquisition_summary(curve[:2], 0.7)["accuracy_threshold"]["right_censored"]
        )

    def test_fixed_support_preserves_original_schedule_and_current_slots(self):
        full_cfg = small_config(
            methods=["balanced_replay"],
            tasks=4,
            classes_per_task=4,
            batch_size=4,
            updates_per_task=32,
            train_per_class=32,
        )
        bounded_cfg = replace(full_cfg, replay_capacity=32)
        full, bounded = prepared_data(full_cfg), prepared_data(bounded_cfg)
        self.assertEqual(
            digest(full.schedules["balanced_replay"]),
            "d20dffaa19ea71180cecbeb8a58a01af6d3c8e06131f19eb83166086afb7925c",
        )
        self.assertEqual(
            [task.rows for task in full.tasks], [task.rows for task in bounded.tasks]
        )
        self.assertNotEqual(full.schedules, bounded.schedules)
        self.assertEqual(bounded.schedules, prepared_data(bounded_cfg).schedules)
        self.assertEqual(
            [len(members) for members in full.replay_buffers], [128, 256, 384, 512]
        )
        self.assertEqual(
            [len(members) for members in bounded.replay_buffers], [32, 32, 32, 32]
        )
        totals = Counter()
        for task_index, (left, right) in enumerate(
            zip(
                full.schedules["balanced_replay"],
                bounded.schedules["balanced_replay"],
                strict=True,
            )
        ):
            members = bounded.replay_buffers[task_index - 1] if task_index else []
            self.assertEqual(len(left), 32)
            for full_batch, bounded_batch in zip(left, right, strict=True):
                validate_replay_refs(bounded_batch, members, task_index)
                for full_ref, bounded_ref in zip(
                    full_batch, bounded_batch, strict=True
                ):
                    source, index = full_ref
                    other_source, other_index = bounded_ref
                    self.assertEqual(source == task_index, other_source == task_index)
                    if source == task_index:
                        self.assertEqual(full_ref, bounded_ref)
                        totals["current"] += 1
                    else:
                        self.assertEqual(
                            full.tasks[source].rows["train"][index]["code"],
                            bounded.tasks[other_source].rows["train"][other_index][
                                "code"
                            ],
                        )
                        totals["replay"] += 1
        self.assertEqual(totals, {"current": 640, "replay": 384})
        evicted = next(
            ref
            for ref in full.replay_buffers[1]
            if ref not in bounded.replay_buffers[1]
        )
        with self.assertRaisesRegex(
            RuntimeError, "REPLAY_REFERENCE_OUTSIDE_CURRENT_BUFFER"
        ):
            validate_replay_refs([evicted], bounded.replay_buffers[1], 2)
        with self.assertRaisesRegex(
            RuntimeError, "REPLAY_REFERENCE_OUTSIDE_CURRENT_BUFFER"
        ):
            validate_replay_refs([(3, 0)], bounded.replay_buffers[1], 2)
        for capacity in (0, 15, 32.0, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                replace(full_cfg, replay_capacity=capacity)
        with self.assertRaises(ValueError):
            replace(bounded_cfg, methods=["persistent", "balanced_replay"])

    def test_fixed_support_shrinking_quotas_never_reacquire_and_ignore_heldout(self):
        for seed in (0, 13, 23, 47):
            cfg = small_config(
                seed=seed,
                methods=["balanced_replay"],
                tasks=4,
                classes_per_task=4,
                batch_size=4,
                train_per_class=32,
                replay_capacity=32,
            )
            data = prepared_data(cfg)
            previous, evicted = set(), set()
            previous_counts = Counter()
            for task_index, members in enumerate(data.replay_buffers):
                incoming = {
                    (task_index, index)
                    for index in range(len(data.tasks[task_index].rows["train"]))
                }
                current = set(members)
                self.assertEqual(len(current), 32)
                self.assertTrue(current <= previous | incoming)
                self.assertFalse(current & evicted)
                counts = Counter(
                    data.tasks[source].rows["train"][index]["code"]
                    for source, index in current
                )
                self.assertEqual(len(counts), (task_index + 1) * 4)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                for code, count in previous_counts.items():
                    self.assertLessEqual(counts[code], count)
                receipt = replay_buffer_record(data.tasks, members)
                self.assertEqual(receipt["members_sha256"], digest(receipt["members"]))
                evicted |= (previous | incoming) - current
                previous, previous_counts = current, counts
            altered = copy.deepcopy(data.tasks)
            for task in altered:
                for split in ("validation", "test"):
                    for row in task.rows[split]:
                        row.update(
                            code=-1,
                            text="heldout values must not affect retention",
                            input_ids=[63],
                        )
            self.assertEqual(replay_buffers(altered, cfg), data.replay_buffers)
            self.assertEqual(
                training_schedules(altered, cfg, replay_buffers(altered, cfg)),
                data.schedules,
            )
            with self.assertRaisesRegex(
                RuntimeError, "REPLAY_RETENTION_INVALID_PREVIOUS_MEMBERS"
            ):
                retain_replay([(3, 0)], data.tasks[1], data.tasks, cfg)
            altered[1].rows["train"][0]["source_split"] = "test"
            with self.assertRaisesRegex(RuntimeError, "REPLAY_RETENTION_NONTRAIN_ROW"):
                retain_replay(data.replay_buffers[0], altered[1], altered, cfg)

    def test_support_gate_uses_exact_integer_thresholds(self):
        def counts(acquisition, old):
            return {
                "acquisition": [
                    {"correct": value, "examples": 80} for value in acquisition
                ],
                "final_old": {"correct": old, "examples": 240},
            }

        full = counts([80, 76, 72, 80], 228)
        bounded = counts([76, 72, 72, 80], 216)
        self.assertTrue(all(support_gates(full, bounded).values()))
        self.assertFalse(
            support_gates(full, counts([75, 72, 72, 80], 216))[
                "bounded_acquisition_at_most_5_points_below_full"
            ]
        )
        self.assertFalse(
            support_gates(counts([80, 76, 71, 80], 228), bounded)[
                "both_acquisition_at_least_90_percent_every_task"
            ]
        )
        failed = support_gates(full, counts([76, 72, 72, 80], 215))
        self.assertFalse(failed["bounded_final_old_at_least_90_percent"])
        self.assertFalse(failed["bounded_final_old_at_most_5_points_below_full"])

    def test_tiny_paired_support_preserves_initialization_and_interrupted_resume(self):
        full_cfg = small_config(
            methods=["balanced_replay"],
            tasks=4,
            classes_per_task=4,
            train_per_class=8,
            batch_size=4,
            updates_per_task=3,
            checkpoint_every=1,
        )
        bounded_cfg = replace(full_cfg, replay_capacity=32)
        full_data, bounded_data = prepared_data(full_cfg), prepared_data(bounded_cfg)
        full_model, bounded_model = (
            tiny_model(full_cfg, hybrid=True),
            tiny_model(bounded_cfg, hybrid=True),
        )
        for (left_name, left), (right_name, right) in zip(
            full_model.named_parameters(), bounded_model.named_parameters(), strict=True
        ):
            self.assertEqual(left_name, right_name)
            torch.testing.assert_close(left, right, rtol=0, atol=0)

        def provenance(data):
            return {
                "model": {"fixture": "tiny-qwen35", "seed": full_cfg.seed},
                "runtime": runtime_provenance(full_cfg),
                "source_sha256": source_provenance(),
                "data": {
                    "selected_rows": [task.rows for task in data.tasks],
                    "code_token_ids": data.code_token_ids,
                    "evaluation_sha256": digest(
                        [
                            {
                                split: task.rows[split]
                                for split in ("validation", "test")
                            }
                            for task in data.tasks
                        ]
                    ),
                    "replay_buffers_after_task": [
                        replay_buffer_record(data.tasks, members)
                        for members in data.replay_buffers
                    ],
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = Experiment(
                full_model,
                full_data,
                full_cfg,
                Reporter(root / "full", full_cfg),
                provenance(full_data),
            )
            bounded = Experiment(
                bounded_model,
                bounded_data,
                bounded_cfg,
                Reporter(root / "bounded", bounded_cfg),
                provenance(bounded_data),
            )
            self.assertEqual(full.run(), 0)
            self.assertEqual(bounded.run(), 0)
            full_metrics = json.loads((root / "full/metrics.json").read_text())
            bounded_metrics = json.loads((root / "bounded/metrics.json").read_text())
            comparison = compare_support(full_metrics, bounded_metrics)
            self.assertTrue(comparison["capacity_is_only_config_difference"])
            self.assertTrue(
                comparison["current_refs_replay_class_slots_and_exposures_match"]
            )
            self.assertEqual(full.state["total_updates"], 12)
            self.assertEqual(bounded.state["total_updates"], 12)
            for index in (0, 1):
                left = full.state["methods"]["balanced_replay"]["tasks"][index]
                right = bounded.state["methods"]["balanced_replay"]["tasks"][index]
                self.assertEqual(left["test_after_task"], right["test_after_task"])
                self.assertEqual(
                    [step["loss"] for step in left["training_curve"]],
                    [step["loss"] for step in right["training_curve"]],
                )
            self.assertTrue(
                any(
                    not torch.equal(value, snapshot(bounded.model)[name])
                    for name, value in snapshot(full.model).items()
                )
            )
            for cfg, experiment in ((full_cfg, full), (bounded_cfg, bounded)):
                total = Counter()
                for stage in experiment.state["methods"]["balanced_replay"]["tasks"]:
                    total.update(
                        {
                            key: stage[key]
                            for key in ("current_examples", "replay_examples")
                        }
                    )
                    before = stage["replay_buffer_before"]
                    for step in stage["training_curve"]:
                        validate_replay_refs(
                            step["training_refs"], before["members"], stage["task"]
                        )
                    self.assertEqual(
                        stage["replay_buffer_after"],
                        replay_buffer_record(
                            experiment.data.tasks,
                            experiment.data.replay_buffers[stage["task"]],
                        ),
                    )
                self.assertEqual(total, {"current_examples": 60, "replay_examples": 36})
            stop = StopRequest()
            interrupted = Experiment(
                tiny_model(bounded_cfg, hybrid=True),
                bounded_data,
                bounded_cfg,
                InterruptReporter(root / "resumed", bounded_cfg, stop, after_updates=8),
                provenance(bounded_data),
                stop,
            )
            self.assertEqual(interrupted.run(), 143)
            self.assertEqual(interrupted.state["task_index"], 2)
            self.assertEqual(interrupted.state["step"], 2)
            self.assertEqual(
                interrupted.state["replay_buffer"], bounded_data.replay_buffers[1]
            )
            resumed = Experiment(
                tiny_model(bounded_cfg, hybrid=True),
                prepared_data(bounded_cfg),
                bounded_cfg,
                Reporter(root / "resumed", bounded_cfg),
                provenance(bounded_data),
            )
            self.assertEqual(resumed.run(), 0)
            for name, value in snapshot(bounded.model).items():
                torch.testing.assert_close(
                    value, snapshot(resumed.model)[name], rtol=0, atol=0
                )
            for left, right in zip(
                bounded.optimizer.state.values(),
                resumed.optimizer.state.values(),
                strict=True,
            ):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertEqual(
                bounded.state["replay_buffer"], resumed.state["replay_buffer"]
            )
            for left, right in zip(
                bounded.state["methods"]["balanced_replay"]["tasks"],
                resumed.state["methods"]["balanced_replay"]["tasks"],
                strict=True,
            ):
                self.assertEqual(left["test_after_task"], right["test_after_task"])
                self.assertEqual(
                    left["replay_buffer_after"], right["replay_buffer_after"]
                )
                self.assertEqual(
                    [step["training_refs"] for step in left["training_curve"]],
                    [step["training_refs"] for step in right["training_curve"]],
                )
            altered = copy.deepcopy(bounded_data)
            for task in altered.tasks:
                for split in ("validation", "test"):
                    for row in task.rows[split]:
                        row["code"] = next(
                            code for code in task.codes if code != row["code"]
                        )
            changed_heldout = Experiment(
                tiny_model(bounded_cfg, hybrid=True),
                altered,
                bounded_cfg,
                Reporter(root / "heldout-changed", bounded_cfg),
                provenance(altered),
            )
            self.assertEqual(changed_heldout.run(), 0)
            for name, value in snapshot(bounded.model).items():
                torch.testing.assert_close(
                    value, snapshot(changed_heldout.model)[name], rtol=0, atol=0
                )
            self.assertEqual(
                bounded.state["replay_buffer"], changed_heldout.state["replay_buffer"]
            )

    def test_rank_detects_collapsed_features(self):
        collapsed = torch.ones(8, 4)
        stats = activation_statistics(collapsed, torch.full_like(collapsed, -6))
        self.assertEqual(stats["centered_effective_rank"], 0)
        self.assertEqual(stats["low_variance_feature_fraction_var_lt_1e_6"], 1)
        self.assertEqual(stats["silu_negative_saturation_proxy_gate_lt_minus_5"], 1)
        spread = activation_statistics(torch.eye(8), torch.zeros(8, 8))
        self.assertAlmostEqual(spread["centered_effective_rank"], 7, places=6)

    def test_screen_gate_uses_global_retention_and_requires_acquisition(self):
        cfg = small_config()
        acquired = {
            "accuracy": 0.95,
            "correct": 95,
            "examples": 100,
            "task_local_accuracy": 1.0,
        }
        forgotten = {
            "accuracy": 0.20,
            "correct": 20,
            "examples": 100,
            "task_local_accuracy": 1.0,
        }
        retained = {
            "accuracy": 0.30,
            "correct": 30,
            "examples": 100,
            "task_local_accuracy": 1.0,
        }
        methods = {
            "persistent": {
                "tasks": [
                    {"test_after_task": [acquired]},
                    {"test_after_task": [forgotten, acquired]},
                ]
            },
            "balanced_replay": {
                "tasks": [
                    {"test_after_task": [acquired]},
                    {"test_after_task": [retained, acquired]},
                ]
            },
        }
        self.assertEqual(
            replay_screen(methods, cfg)["decision"], "keep_for_confirmation"
        )
        methods["balanced_replay"]["tasks"][-1]["test_after_task"][0] = forgotten
        self.assertEqual(
            replay_screen(methods, cfg)["decision"], "redesign_retention_mechanism"
        )
        methods["balanced_replay"]["tasks"][-1]["test_after_task"] = [
            retained,
            {**acquired, "accuracy": 0.85, "correct": 85},
        ]
        self.assertEqual(
            replay_screen(methods, cfg)["decision"],
            "redesign_acquisition_replay_tradeoff",
        )

    def test_boundaries_control_weights_and_optimizer_independently(self):
        cfg = small_config()
        data = prepared_data(cfg)
        model = tiny_model(cfg)
        initial = snapshot(model)
        optimizer = optimizer_for(model, cfg)
        inputs, labels = collate(data.tasks[0].rows["train"][:2], 0, "cpu")
        initial_logits = (
            class_logits(model, inputs, data.code_token_ids).detach().clone()
        )
        loss = torch.nn.functional.cross_entropy(
            class_logits(model, inputs, data.code_token_ids), labels
        )
        loss.backward()
        optimizer.step()
        learned = snapshot(model)
        self.assertTrue(optimizer.state)
        for method in ("persistent", "balanced_replay"):
            carried, changes = apply_boundary(model, optimizer, initial, method, 1, cfg)
            self.assertIs(carried, optimizer)
            self.assertEqual(changes, [])
            for name, value in snapshot(model).items():
                torch.testing.assert_close(value, learned[name], rtol=0, atol=0)
        reset, _ = apply_boundary(model, optimizer, initial, "optimizer_reset", 1, cfg)
        self.assertFalse(reset.state)
        for name, value in snapshot(model).items():
            torch.testing.assert_close(value, learned[name], rtol=0, atol=0)
        fresh, _ = apply_boundary(model, optimizer, initial, "fresh_base", 1, cfg)
        self.assertFalse(fresh.state)
        torch.testing.assert_close(
            class_logits(model, inputs, data.code_token_ids),
            initial_logits,
            rtol=0,
            atol=0,
        )

    def test_component_renewal_and_effective_update_norm(self):
        cfg = small_config()
        model = tiny_model(cfg)
        with torch.no_grad():
            for _, layer in adapter_layers(model):
                layer.lora_A["default"].weight.fill_(0.2)
                b = layer.lora_B["default"].weight
                b.copy_(torch.arange(1, b.shape[1] + 1).float().expand_as(b))
        before = snapshot(model)
        changes = selective_reset(model, cfg, 1)
        expected_squared = 0.0
        for name, layer in adapter_layers(model):
            a = layer.lora_A["default"].weight
            b = layer.lora_B["default"].weight
            old_a = before[f"{name}.lora_A.default.weight"]
            old_b = before[f"{name}.lora_B.default.weight"]
            torch.testing.assert_close(a[1:], old_a[1:], rtol=0, atol=0)
            torch.testing.assert_close(b[:, 1:], old_b[:, 1:], rtol=0, atol=0)
            self.assertEqual(float(b[:, 0].detach().abs().sum()), 0)
            self.assertFalse(torch.equal(a[0], old_a[0]))
            expected_squared += float(
                ((b @ a - old_b @ old_a) * layer.scaling["default"])
                .detach()
                .square()
                .sum()
            )
        self.assertTrue(all(item["components"] == [0] for item in changes))
        measured = update_diagnostics(model, before)[
            "effective_adapter_update_frobenius"
        ]
        self.assertAlmostEqual(measured, expected_squared**0.5, places=4)

    def test_last_token_left_padding_invariance_both_qwen_families(self):
        cfg = small_config()
        rows = [
            {"input_ids": [11, 30], "code": 0},
            {"input_ids": [12, 21, 30], "code": 1},
        ]
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                model = tiny_model(cfg, hybrid).eval()
                inputs, _ = collate(rows, 0, "cpu")
                batch_logits = class_logits(model, inputs, [2, 3, 4, 5])
                single, _ = collate(rows[:1], 0, "cpu")
                one_logits = class_logits(model, single, [2, 3, 4, 5])
                torch.testing.assert_close(
                    batch_logits[:1], one_logits, rtol=1e-4, atol=2e-5
                )
                batch_logits.sum().backward()
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                )

    def test_two_arms_match_first_task_and_saved_adapters_reload_on_tiny_qwen35(self):
        cfg = small_config(updates_per_task=2)
        model = tiny_model(cfg, hybrid=True)
        frozen_before = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if not p.requires_grad
        }
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment(
                model,
                prepared_data(cfg),
                cfg,
                Reporter(directory, cfg),
                {"test_fixture": "tiny-qwen3"},
            )
            self.assertEqual(experiment.run(), 0)
            results = experiment.state["methods"]
            reference = results["persistent"]["tasks"][0]
            for method in cfg.methods:
                record = results[method]["tasks"][0]
                self.assertEqual(
                    record["test_after_task"], reference["test_after_task"]
                )
                self.assertEqual(
                    [v["loss"] for v in record["training_curve"]],
                    [v["loss"] for v in reference["training_curve"]],
                )
            self.assertTrue(
                experiment.comparison()["matched_update_and_example_budgets_verified"]
            )
            for method in cfg.methods:
                endpoint = results[method]["tasks"][-1]
                reloaded = PeftModel.from_pretrained(
                    tiny_base(cfg, hybrid=True),
                    endpoint["adapter_path"],
                    is_trainable=True,
                )
                for task in experiment.data.tasks:
                    self.assertEqual(
                        evaluate(reloaded, task, "test", experiment.data, cfg),
                        endpoint["test_after_task"][task.index],
                    )
                for name, sha256 in endpoint["adapter_files_sha256"].items():
                    self.assertEqual(
                        file_sha256(Path(endpoint["adapter_path"]) / name), sha256
                    )
            self.assertEqual(experiment.state["total_updates"], 8)
            self.assertEqual(results["persistent"]["tasks"][1]["current_examples"], 8)
            self.assertEqual(
                results["balanced_replay"]["tasks"][1]["current_examples"], 4
            )
            self.assertEqual(
                results["balanced_replay"]["tasks"][1]["replay_examples"], 4
            )
            self.assertEqual(
                json.loads((Path(directory) / "metrics.json").read_text())["status"],
                "completed",
            )
            for name, parameter in model.named_parameters():
                if name in frozen_before:
                    torch.testing.assert_close(
                        parameter, frozen_before[name], rtol=0, atol=0
                    )

    def test_resume_matches_uninterrupted_weights_optimizer_and_scores(self):
        cfg = small_config(updates_per_task=3)
        provenance = {"fixture": "tiny-qwen3", "digest": digest(cfg.as_dict())}
        with (
            tempfile.TemporaryDirectory() as uninterrupted,
            tempfile.TemporaryDirectory() as resumed,
        ):
            full = Experiment(
                tiny_model(cfg),
                prepared_data(cfg),
                cfg,
                Reporter(uninterrupted, cfg),
                provenance,
            )
            self.assertEqual(full.run(), 0)
            stop = StopRequest()
            partial = Experiment(
                tiny_model(cfg),
                prepared_data(cfg),
                cfg,
                InterruptReporter(resumed, cfg, stop, after_updates=10),
                provenance,
                stop,
            )
            self.assertEqual(partial.run(), 143)
            self.assertEqual(partial.state["step"], 1)
            continued = Experiment(
                tiny_model(cfg),
                prepared_data(cfg),
                cfg,
                Reporter(resumed, cfg),
                provenance,
            )
            self.assertEqual(continued.run(), 0)
            for name, value in snapshot(full.model).items():
                torch.testing.assert_close(
                    value, snapshot(continued.model)[name], rtol=0, atol=0
                )
            for a, b in zip(
                full.optimizer.state.values(),
                continued.optimizer.state.values(),
                strict=True,
            ):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            for a, b in zip(
                full.state["methods"]["balanced_replay"]["tasks"],
                continued.state["methods"]["balanced_replay"]["tasks"],
                strict=True,
            ):
                self.assertEqual(a["test_after_task"], b["test_after_task"])
                self.assertEqual(a["acquisition"], b["acquisition"])
                self.assertEqual(
                    [p["training_refs"] for p in a["training_curve"]],
                    [p["training_refs"] for p in b["training_curve"]],
                )
            self.assertEqual(continued.state["total_updates"], 12)
            events = [
                json.loads(line)
                for line in (Path(resumed) / "events.jsonl").read_text().splitlines()
            ]
            resume_receipt = next(
                event for event in events if event["event"] == "resumed"
            )
            interrupted_checkpoint = next(
                event
                for event in events
                if event["event"] == "checkpoint" and event["status"] == "interrupted"
            )
            self.assertEqual(
                resume_receipt["checkpoint_sha256"],
                interrupted_checkpoint["checkpoint_sha256"],
            )
            self.assertEqual(resume_receipt["cursor"]["total_updates"], 10)
            prior_hashes = {
                str(path): file_sha256(path)
                for path in Path(resumed).rglob("*")
                if path.is_file()
            }
            with self.assertRaisesRegex(RuntimeError, "RUN_ID_ALREADY_EXISTS"):
                execute(cfg, Path(resumed))
            with self.assertRaisesRegex(RuntimeError, "COMPLETED_RUN_IMMUTABLE"):
                prepare_output(cfg, Path(resumed), resume=True)
            self.assertEqual(
                prior_hashes,
                {
                    str(path): file_sha256(path)
                    for path in Path(resumed).rglob("*")
                    if path.is_file()
                },
            )
            with self.assertRaisesRegex(RuntimeError, "RESUME_SIGNATURE_MISMATCH"):
                changed = replace(cfg, learning_rate=0.0001)
                Experiment(
                    tiny_model(changed),
                    prepared_data(changed),
                    changed,
                    Reporter(resumed, changed),
                    provenance,
                )

    def test_eval_only_verifier_reloads_tiny_qwen35_and_rejects_modified_artifacts(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, source, run = (root / name for name in ("base", "source", "run"))
            cfg = small_config(model_path=str(base_path), updates_per_task=2)
            base = tiny_base(cfg, hybrid=True)
            base.save_pretrained(base_path)
            vocabulary = {"[PAD]": 0, "[UNK]": 1, "A": 2, "B": 3, "C": 4, "D": 5}
            vocabulary.update({str(i): 6 + i for i in range(40)})
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
            )
            tokenizer.save_pretrained(base_path)
            splits, names = fake_splits()
            tasks, selection = select_rows(splits, names, cfg)
            prepared = tokenize_tasks(tasks, tokenizer, cfg, selection)
            provenance = {
                "model": model_provenance(cfg),
                "data": prepared.provenance,
                "runtime": runtime_provenance(cfg),
                "source_sha256": source_provenance(),
            }
            source.mkdir()
            bundle = hashlib.sha256()
            for name in sorted(provenance["source_sha256"]):
                content = Path(__file__).with_name(name).read_bytes()
                (source / name).write_bytes(content)
                bundle.update(name.encode() + b"\0" + content)
            experiment = Experiment(
                attach_adapter(base, cfg), prepared, cfg, Reporter(run, cfg), provenance
            )
            self.assertEqual(experiment.run(), 0)
            execution = {
                "task_id": run.name,
                "status": "completed",
                "exit_code": 0,
                "runtime_sha256": "cpu-test-runtime",
                "source_sha256": bundle.hexdigest(),
                "task": {
                    "id": run.name,
                    "config": cfg.as_dict(),
                    "code_dir": str(source),
                    "source_sha256": bundle.hexdigest(),
                },
            }
            for name, value in (
                ("config.json", cfg.as_dict()),
                ("provenance.json", provenance),
                ("execution.json", execution),
            ):
                (run / name).write_text(json.dumps(value))
            config = {
                "runtime_sha256": "cpu-test-runtime",
                "source_sha256": bundle.hexdigest(),
                "runs": [
                    {
                        "run_dir": str(run),
                        **{
                            name + "_sha256": file_sha256(run / (name + suffix))
                            for name, suffix in (
                                ("metrics", ".json"),
                                ("execution", ".json"),
                                ("events", ".jsonl"),
                            )
                        },
                    }
                ],
            }

            def input_hashes():
                return {
                    str(path): file_sha256(path)
                    for directory in (base_path, source, run)
                    for path in directory.rglob("*")
                    if path.is_file()
                }

            before = input_hashes()
            output = root / "verification"
            output.mkdir()
            bootstrap = {
                "execution.json": json.dumps(
                    {
                        "task_id": output.name,
                        "status": "running",
                        "runtime_sha256": "cpu-test-runtime",
                    }
                ),
                "config.json": json.dumps(config),
                "task.json": json.dumps({"id": output.name, "config": config}),
                "packages.txt": "cpu-test-fixture\n",
                "run.log": "supervisor bootstrap\n",
            }
            for name, content in bootstrap.items():
                (output / name).write_text(content)
            receipt = verify_saved_adapters(config, output)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["total_updates"], 0)
            for method in cfg.methods:
                observed = receipt["runs"][0]["methods"][method]
                endpoint = experiment.state["methods"][method]["tasks"][-1]
                self.assertTrue(observed["identical_predictions_and_counts"])
                self.assertEqual(observed["test"], endpoint["test_after_task"])
            self.assertEqual(before, input_hashes())
            self.assertEqual(
                bootstrap, {name: (output / name).read_text() for name in bootstrap}
            )
            with self.assertRaisesRegex(RuntimeError, "VERIFY_OUTPUT_NOT_FRESH"):
                verify_saved_adapters(config, output)
            with self.assertRaisesRegex(
                RuntimeError, "VERIFY_SOURCE_RUN_OUTPUT_OVERLAP"
            ):
                verify_saved_adapters(config, run / "verification")
            self.assertFalse((run / "verification").exists())

            endpoint = experiment.state["methods"]["balanced_replay"]["tasks"][-1]
            model = tiny_model(cfg, hybrid=True)
            initial_adapter = snapshot(model)
            frozen_base = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if not p.requires_grad
            }
            reload_and_score(model, endpoint, prepared, cfg)
            self.assertTrue(
                any(
                    not torch.equal(initial_adapter[name], parameter)
                    for name, parameter in model.named_parameters()
                    if name in initial_adapter
                )
            )
            self.assertFalse(
                any(p.requires_grad or p.grad is not None for p in model.parameters())
            )
            for name, p in model.named_parameters():
                if name in frozen_base:
                    torch.testing.assert_close(p, frozen_base[name], rtol=0, atol=0)
            incorrect = copy.deepcopy(endpoint)
            incorrect["test_after_task"][0]["predictions"][0] = (
                incorrect["test_after_task"][0]["predictions"][0] + 1
            ) % 4
            with self.assertRaisesRegex(RuntimeError, "VERIFY_PREDICTION_MISMATCH"):
                reload_and_score(model, incorrect, prepared, cfg)
            for index, artifact in enumerate(
                (
                    run / "checkpoint.pt",
                    Path(endpoint["adapter_path"]) / "adapter_model.safetensors",
                )
            ):
                original = artifact.read_bytes()
                artifact.write_bytes(b"invalid serialized weights")
                failed_output = root / f"corrupt-{index}"
                with (
                    self.assertLogs("verify", level="ERROR"),
                    self.assertRaisesRegex(RuntimeError, "VERIFY_HASH_MISMATCH"),
                ):
                    verify_saved_adapters(config, failed_output)
                self.assertEqual(
                    json.loads((failed_output / "receipt.json").read_text())["status"],
                    "failed",
                )
                artifact.write_bytes(original)

    def test_heldout_labels_never_change_training_or_intervention(self):
        cfg = small_config(methods=["balanced_replay"], updates_per_task=2)
        clean = prepared_data(cfg)
        altered = copy.deepcopy(clean)
        for task in altered.tasks:
            for split in ("validation", "test"):
                for row in task.rows[split]:
                    row["code"] = next(
                        code for code in task.codes if code != row["code"]
                    )
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            left = Experiment(
                tiny_model(cfg), clean, cfg, Reporter(one, cfg), {"fixture": "clean"}
            )
            right = Experiment(
                tiny_model(cfg),
                altered,
                cfg,
                Reporter(two, cfg),
                {"fixture": "altered"},
            )
            self.assertEqual(left.run(), 0)
            self.assertEqual(right.run(), 0)
            for name, value in snapshot(left.model).items():
                torch.testing.assert_close(
                    value, snapshot(right.model)[name], rtol=0, atol=0
                )
            self.assertEqual(
                left.state["methods"]["balanced_replay"]["tasks"][1][
                    "boundary_changes"
                ],
                right.state["methods"]["balanced_replay"]["tasks"][1][
                    "boundary_changes"
                ],
            )

    def test_supervisor_bootstrap_preserves_metadata_and_rejects_existing_research(
        self,
    ):
        cfg = small_config(updates_per_task=1)
        for history in (False, True):
            with (
                self.subTest(history=history),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory)
                execution = {"status": "running", "task_id": output.name}
                parent_files = {
                    "execution.json": json.dumps(execution).encode(),
                    "packages.txt": b"torch==2.10.0\n",
                    "config.json": json.dumps(cfg.as_dict()).encode(),
                    "run.log" if history else "output.log": b"parent initialized\n",
                }
                if history:
                    parent_files["task.json"] = json.dumps(
                        {"id": output.name, "config": cfg.as_dict()}
                    ).encode()
                    content = json.dumps(execution).encode()
                    parent_files[
                        f"attempts/{'a' * 32}/{hashlib.sha256(content).hexdigest()}.json"
                    ] = content
                for name, content in parent_files.items():
                    path = output / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                for unknown in ("metrics.json", "checkpoint.pt", "unrelated.data"):
                    foreign = output / unknown
                    foreign.write_bytes(b"existing research output")
                    with self.assertRaisesRegex(RuntimeError, "RUN_ID_ALREADY_EXISTS"):
                        prepare_output(cfg, output, resume=False)
                    self.assertEqual(foreign.read_bytes(), b"existing research output")
                    foreign.unlink()
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR_CONFIG_MISMATCH"):
                    prepare_output(replace(cfg, seed=99), output, resume=False)
                (output / "execution.json").write_text(
                    json.dumps({**execution, "status": "completed"})
                )
                with self.assertRaisesRegex(
                    RuntimeError, "SUPERVISOR_EXECUTION_MISMATCH"
                ):
                    prepare_output(cfg, output, resume=False)
                (output / "execution.json").write_bytes(parent_files["execution.json"])
                self.assertIsNone(prepare_output(cfg, output, resume=False))
                experiment = Experiment(
                    tiny_model(cfg),
                    prepared_data(cfg),
                    cfg,
                    Reporter(output, cfg),
                    {"fixture": "supervisor-bootstrap"},
                )
                self.assertEqual(experiment.run(), 0)
                for name, content in parent_files.items():
                    self.assertEqual((output / name).read_bytes(), content)
                with self.assertRaisesRegex(RuntimeError, "RUN_ID_ALREADY_EXISTS"):
                    prepare_output(cfg, output, resume=False)


if __name__ == "__main__":
    unittest.main()
