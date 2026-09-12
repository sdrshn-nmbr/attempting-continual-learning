import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from objectives import distillation_loss
from run import (
    adapter_parameters,
    prepare_output,
    tensor_digest,
    validate_config,
    validate_snapshot,
)
from tasks import (
    FAMILIES,
    OPS,
    audit_dataset,
    demo_context,
    digits,
    execute,
    feedback_context,
    make_dataset,
    make_task,
    paired_gate,
    parse_answer,
    score,
)


class DataTests(unittest.TestCase):
    def test_model_revision_and_snapshot_guards(self):
        config = json.loads(
            (Path(__file__).parent / "configs/qwen35_4b_screen.json").read_text()
        )
        validate_config(config)
        with self.assertRaisesRegex(ValueError, "INVALID_MODEL_REVISION"):
            validate_config(
                {**config, "model_revision": config["model_revision"] + "9"}
            )
        with self.assertRaisesRegex(ValueError, "SNAPSHOT_REVISION_MISMATCH"):
            validate_config({**config, "model_path": "/missing/wrong-revision"})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "SNAPSHOT_MISSING"):
                validate_snapshot(
                    {
                        **config,
                        "model_path": str(Path(temporary) / config["model_revision"]),
                    }
                )

    def test_dispatcher_metadata_preserved_and_duplicate_run_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"seed": 1703}
            (output / "config.json").write_text(json.dumps(config))
            (output / "packages.txt").write_text("torch=vendor-rocm\n")
            (output / "stdout.log").write_text("dispatcher initialized\n")
            prepare_output(output, config)
            self.assertEqual(
                (output / "packages.txt").read_text(), "torch=vendor-rocm\n"
            )
            self.assertEqual(
                (output / "stdout.log").read_text(), "dispatcher initialized\n"
            )
            with self.assertRaises(FileExistsError):
                prepare_output(output, config)

    def test_disjoint_splits_and_oracle(self):
        sizes = {
            "train": 96,
            "gate": 24,
            "validation": 32,
            "test": 64,
            "composition": 64,
        }
        corpus = make_dataset(71239, sizes)
        self.assertEqual(corpus, make_dataset(71239, sizes))
        self.assertEqual(audit_dataset(corpus)["tasks"], 858)
        for split, tasks in corpus.items():
            for task in tasks:
                self.assertTrue(score(task, digits(task.answer))["correct"])
                if split != "demonstration":
                    context, ids = demo_context(task, corpus["demonstration"])
                    self.assertNotIn(task.uid, ids)
                    self.assertTrue(ids)
                    self.assertNotIn(digits(task.answer), context)
                if split == "composition":
                    self.assertGreater(len(task.program), 2)
        corpus["test"].append(corpus["train"][0])
        with self.assertRaisesRegex(ValueError, "SPLIT_LEAK"):
            audit_dataset(corpus)

    def test_reference_transitions_and_composition(self):
        initial = (2, 3, 4, 5)
        expected = {
            "A": ((3, 4, 5, 2), (5, 3, 4, 2), (5, 4, 3, 2)),
            "B": ((4, 5, 6, 7), (6, 9, 2, 5), (7, 6, 5, 4)),
            "C": ((5, 4, 3, 2), (2, 3, 5, 4), (2, 3, 4, 7)),
        }
        for family in FAMILIES:
            for op, target in zip(OPS, expected[family]):
                self.assertEqual(execute(family, initial, (op,)), target)
            composed = execute(family, execute(family, initial, ("dax",)), ("wug",))
            self.assertEqual(execute(family, initial, ("dax", "wug")), composed)
        self.assertEqual(execute("C", (1, 3, 5, 8), ("dax", "wug")), (3, 5, 1, 8))

    def test_strict_verifier_and_feedback_boundary(self):
        task = make_task("B", "train", (2, 4, 6, 8), ("wug",))
        for text in ("The answer is 6 2 8 4", "[6,2,8,4]", "6 2 8 4\n0", "6  2 8 4"):
            self.assertIsNone(parse_answer(text))
        feedback = feedback_context(task, "0 0 0 0")
        self.assertNotIn(digits(task.answer), feedback)
        self.assertIn("1, 2, 3, 4", feedback)
        self.assertIn("modulo 10", feedback)
        self.assertTrue(score(task, " 6 2 8 4\n")["correct"])

    def test_teacher_gate_rejects_no_advantage(self):
        good = paired_gate([False] * 24, [True] * 20 + [False] * 4, 0.5, 0.15, 0.1)
        bad = paired_gate([True] * 24, [True] * 24, 0.5, 0.15, 0.1)
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])


class ObjectiveTests(unittest.TestCase):
    def test_forward_kl_value_and_teacher_stop_gradient(self):
        student = torch.tensor([[0.2, -0.5, 1.1]], requires_grad=True)
        teacher = torch.tensor([[-0.3, 0.8, 0.0]], requires_grad=True)
        loss, diagnostic = distillation_loss(
            student, teacher, torch.tensor([2]), "sdft_forward"
        )
        p = student.softmax(-1)
        q = teacher.softmax(-1)
        expected = (q.detach() * (q.detach().log() - p.log())).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(student.grad, p.detach() - q.detach())
        self.assertIsNone(teacher.grad)
        self.assertGreater(diagnostic["forward_kl"], 0)

    def test_sampled_reverse_expected_gradient_matches_exact_kl(self):
        student = torch.tensor([[0.2, -0.5, 1.1]], requires_grad=True)
        teacher = torch.tensor([[-0.3, 0.8, 0.0]], requires_grad=True)
        p = student.softmax(-1)
        q = teacher.detach().softmax(-1)
        exact = (p * (p.log() - q.log())).sum()
        exact_gradient = torch.autograd.grad(exact, student)[0]
        expected_gradient = torch.zeros_like(student)
        for token in range(3):
            logits = student.detach().clone().requires_grad_(True)
            old = logits.detach().log_softmax(-1)[:, token]
            loss, metrics = distillation_loss(
                logits, teacher, torch.tensor([token]), "sdpo_sampled_reverse", old
            )
            gradient = torch.autograd.grad(loss, logits)[0]
            expected_gradient += p.detach()[0, token] * gradient
            self.assertEqual(metrics["staleness_updates"], 0)
        torch.testing.assert_close(
            expected_gradient, exact_gradient, atol=1e-6, rtol=1e-6
        )
        self.assertIsNone(teacher.grad)

    def test_kl_zero_for_identical_distributions(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]], requires_grad=True)
        ids = torch.tensor([1, 2])
        old = logits.detach().log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
        for method in ("sdft_forward", "sdpo_sampled_reverse"):
            loss, metrics = distillation_loss(logits, logits.detach(), ids, method, old)
            self.assertAlmostEqual(float(loss.detach()), 0)
            self.assertAlmostEqual(metrics["reverse_kl"], 0)


class TinyQwenTests(unittest.TestCase):
    def test_real_adapter_gradient_freeze_ema_and_checkpoint(self):
        torch.manual_seed(12)
        config = Qwen3_5Config(
            text_config={
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "layer_types": ["linear_attention", "full_attention"],
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 32,
                "num_position_embeddings": 64,
            },
        )
        base = Qwen3_5ForConditionalGeneration(config).eval()
        lora = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj", "in_proj_qkv", "out_proj"],
            lora_dropout=0.0,
        )
        model = get_peft_model(base, lora, adapter_name="student")
        model.add_adapter("teacher", copy.deepcopy(lora))
        model.set_adapter("student")
        parameters = adapter_parameters(model, "student")
        before = tensor_digest(parameters)
        frozen = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if ".lora_" not in name
        }
        teacher_parameters = adapter_parameters(model, "teacher")
        with torch.no_grad():
            for name, parameter in teacher_parameters.items():
                parameter.copy_(parameters[name])
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=0.02)
        logits = model(
            input_ids=inputs[:, :-1], use_cache=False, logits_to_keep=3
        ).logits[0]
        full = model(input_ids=inputs[:, :-1], use_cache=False).logits[0]
        torch.testing.assert_close(logits, full[-3:])
        loss = torch.nn.functional.cross_entropy(logits.float(), inputs[0, -3:])
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.norm() > 0
                for parameter in parameters.values()
            )
        )
        optimizer.step()
        self.assertNotEqual(before, tensor_digest(parameters))
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
                self.assertIsNone(parameter.grad)
        with torch.no_grad():
            for name, teacher in teacher_parameters.items():
                old = teacher.clone()
                teacher.lerp_(parameters[name], 0.05)
                torch.testing.assert_close(
                    teacher, old * 0.95 + parameters[name] * 0.05
                )
        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(
                temporary,
                selected_adapters=["student", "teacher"],
                save_embedding_layers=False,
            )
            expected = model(input_ids=inputs, use_cache=False).logits.detach()
            model.load_adapter(
                Path(temporary) / "student", adapter_name="reload", is_trainable=False
            )
            model.set_adapter("reload", inference_mode=True)
            actual = model(input_ids=inputs, use_cache=False).logits.detach()
            self.assertEqual(
                tensor_digest(parameters),
                tensor_digest(adapter_parameters(model, "reload")),
            )
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            model.set_adapter("student")
            model.delete_adapter("reload")


if __name__ == "__main__":
    unittest.main()
