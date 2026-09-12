import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional
from transformers import GenerationConfig

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
from objectives import distillation_loss
from run import (
    Runner,
    adapter_parameters,
    append_json,
    confidence_interval,
    tensor_digest,
    validate_snapshot,
    write_json,
)
from tasks import FAMILIES, audit_dataset, digits, to_records

LOGGER = logging.getLogger("distillation.qualified")


def prepare(output, dispatch, protocol):
    output.mkdir(parents=True, exist_ok=True)
    allowed = {
        "config.json",
        "packages.txt",
        "stdout.log",
        "stderr.log",
        "run.log",
        "task.json",
        "execution.json",
        "attempts",
    }
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError(f"QUALIFY_OUTPUT_ALREADY_USED: {output}")
    if (output / "config.json").exists() and json.loads(
        (output / "config.json").read_text()
    ) != dispatch:
        raise ValueError("QUALIFY_DISPATCH_CONFIG_MISMATCH")
    corpus = make_corpus(protocol)
    seal = {
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "source_sha256": source_hashes(),
        "dataset_sha256": {
            split: digest(rows) for split, rows in to_records(corpus).items()
        },
        "split_audit": audit_dataset(corpus),
        "sealed_at_unix": time.time(),
        "prediction_outcomes_observed": False,
    }
    with (output / "seal.json").open("x") as stream:
        json.dump(seal, stream, indent=2, allow_nan=False)
    write_json(output / "config.json", dispatch)
    return seal


class QualifiedRunner(Runner):
    def __init__(self, protocol, output):
        self.config = protocol
        self.output = output
        self.corpus = make_corpus(protocol, ("demonstration", "train", "validation"))
        self.device = torch.device("cuda:0")
        self.step = 0
        self.total_rollout_tokens = self.total_loss_tokens = (
            self.total_teacher_tokens
        ) = 0

    def load(self):
        path = validate_snapshot(self.config)
        self.snapshot = {
            p.name: file_digest(p) for p in sorted(path.iterdir()) if p.is_file()
        }
        super().load()
        self.special_ids = list(self.tokenizer.all_special_ids)

    def prompt_ids(self, prompt):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=False,
        ).to(self.device)
        if (
            ids.shape[-1] + self.config["max_new_tokens"]
            > self.config["max_context_tokens"]
        ):
            raise ValueError("QUALIFY_CONTEXT_LIMIT")
        return ids

    def generate(self, prompt, sample):
        inputs = self.prompt_ids(prompt)
        generation = GenerationConfig(
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=sample,
            num_beams=1,
            num_return_sequences=1,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=self.eos,
            pad_token_id=self.eos,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=sample,
        )
        self.model.generation_config = generation
        self.model.get_base_model().generation_config = generation
        with torch.inference_mode():
            result = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation,
            )
        response = result.sequences[0, inputs.shape[-1] :].clone()
        if not response.numel():
            raise ValueError("QUALIFY_EMPTY_GENERATION")
        tokens = response.tolist()
        body = tokens[:-1] if tokens[-1] == self.eos else tokens
        text = self.tokenizer.decode(
            body, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        old = None
        if sample:
            if len(result.scores) != len(tokens):
                raise ValueError("QUALIFY_ROLLOUT_SCORE_COUNT")
            old = torch.stack(
                [
                    functional.log_softmax(logits[0].float(), -1)[token]
                    for logits, token in zip(result.scores, response, strict=True)
                ]
            ).detach()
            if not torch.isfinite(old).all():
                raise ValueError("QUALIFY_NONFINITE_ROLLOUT")
        return inputs, response, text, old

    def generated_record(self, task, prompt):
        inputs, response, text, _ = self.generate(prompt, sample=False)
        tokens = response.tolist()
        return {
            "prompt": prompt,
            "prompt_token_ids": inputs[0].tolist(),
            "token_ids": tokens,
            "body_text": text,
            "raw_text": self.tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            **grade_generation(
                task,
                tokens,
                text,
                self.eos,
                self.special_ids,
                self.config["max_new_tokens"],
            ),
        }

    def assert_frozen(self):
        if any(
            p.grad is not None or p._version != version
            for p, version in zip(
                self.frozen_parameters, self.frozen_versions, strict=True
            )
        ):
            raise ValueError("QUALIFY_FROZEN_BASE_MUTATED_OR_HAS_GRADIENT")

    def qualify(self, candidate="instruction_demo", teacher_training=None):
        validation_split = (
            "validation" if candidate == "instruction_demo" else "fallback_validation"
        )
        if candidate == "trained_teacher":
            self.corpus = make_corpus(
                self.config, ("demonstration", "train", validation_split)
            )
        teacher_adapter = "student" if candidate == "instruction_demo" else "teacher"
        self.model.set_adapter("student", inference_mode=True)
        before = tensor_digest(adapter_parameters(self.model, "student"))
        teacher_before = tensor_digest(adapter_parameters(self.model, teacher_adapter))
        records = []
        for task in qualification_rows(self.corpus, self.config, validation_split):
            row = {"uid": task.uid, "family": task.family, "split": task.split}
            self.model.set_adapter("student", inference_mode=True)
            row["learner"] = self.generated_record(task, learner_prompt(task))
            self.model.set_adapter(teacher_adapter, inference_mode=True)
            row["teacher"] = self.generated_record(
                task, teacher_prompt(task, self.corpus["demonstration"])
            )
            records.append(row)
            append_json(self.output / "pairs.partial.jsonl", row)
            self.event(
                "qualification_pair",
                uid=task.uid,
                split=task.split,
                family=task.family,
                learner_correct=row["learner"]["correct"],
                teacher_correct=row["teacher"]["correct"],
            )
        self.assert_frozen()
        seal = json.loads((self.output / "seal.json").read_text())
        if seal["source_sha256"] != source_hashes():
            raise ValueError("QUALIFY_SOURCE_CHANGED_DURING_RUN")
        after = tensor_digest(adapter_parameters(self.model, "student"))
        if before != after or teacher_before != tensor_digest(
            adapter_parameters(self.model, teacher_adapter)
        ):
            raise ValueError("QUALIFY_ADAPTER_MUTATED")
        write_json(self.output / "pairs.json", records)
        gate = qualification_gate(
            records,
            self.corpus,
            self.config,
            self.eos,
            self.special_ids,
            validation_split,
        )
        receipt = {
            "status": "qualified" if gate["passed"] else "rejected",
            "protocol_sha256": digest(self.config),
            "source_sha256": source_hashes(),
            "gate": gate,
            "candidate": candidate,
            "optimizer_updates": 0
            if teacher_training is None
            else teacher_training["updates"],
            "qualification_optimizer_updates": 0,
            "learner_optimizer_updates": 0,
            "device": str(self.device),
            "hip": torch.version.hip,
            "eos_token_id": self.eos,
            "special_token_ids": self.special_ids,
            "snapshot_sha256": self.snapshot,
            "adapter_before_sha256": before,
            "adapter_after_sha256": after,
            "files_sha256": {
                name: file_digest(self.output / name)
                for name in ("pairs.json", "seal.json", "runtime.json")
            },
            "failure_action": (
                "For instruction_demo rejection, only the predefined teacher_training "
                "stage is eligible. For trained_teacher rejection, stop this protocol."
            ),
        }
        if teacher_training is not None:
            receipt.update(
                teacher_sha256=teacher_before,
                teacher_files_sha256=teacher_training["checkpoint_files_sha256"],
                teacher_training_sha256=file_digest(
                    self.output / "teacher_training.json"
                ),
                initial_failure_sha256=teacher_training["initial_failure_sha256"],
            )
        write_json(self.output / "qualification.json", receipt)
        self.event("qualification_finished", status=receipt["status"], gate=gate)
        return receipt

    def train_teacher(self, failed_receipt):
        if (
            failed_receipt["status"] != "rejected"
            or failed_receipt["candidate"] != "instruction_demo"
        ):
            raise ValueError("QUALIFY_ONE_FALLBACK_AFTER_INITIAL_REJECTION_ONLY")
        self.check_initialization(failed_receipt)
        before = tensor_digest(adapter_parameters(self.model, "student"))
        recipe = self.config["teacher_training"]
        torch.manual_seed(recipe["seed"])
        teacher_parameters = adapter_parameters(self.model, "teacher")
        with torch.no_grad():
            for name, parameter in teacher_parameters.items():
                parameter.copy_(self.initial_adapter[name])
        self.model.set_adapter("teacher")
        optimizer = torch.optim.AdamW(
            list(teacher_parameters.values()),
            lr=recipe["learning_rate"],
            weight_decay=0.0,
        )
        tasks = [
            task
            for family in FAMILIES
            for task in [t for t in self.corpus["train"] if t.family == family][
                : recipe["train_per_family"]
            ]
        ]
        random.Random(recipe["seed"]).shuffle(tasks)
        batch_size = recipe["examples_per_update"]
        if len(tasks) != recipe["updates"] * batch_size:
            raise ValueError("QUALIFY_TEACHER_TRAINING_DATA_BUDGET")
        updates = 0
        for offset in range(0, len(tasks), batch_size):
            optimizer.zero_grad(set_to_none=True)
            for task in tasks[offset : offset + batch_size]:
                prompt = teacher_prompt(task, self.corpus["demonstration"])
                ids, response = self.prompt_ids(prompt), self.target_tokens(task)
                logits = self.logits(ids, response)
                loss = functional.cross_entropy(logits.float(), response)
                if not torch.isfinite(loss):
                    raise ValueError("QUALIFY_TEACHER_NONFINITE_LOSS")
                (loss / batch_size).backward()
                append_json(
                    self.output / "teacher_training_rows.jsonl",
                    {
                        "uid": task.uid,
                        "split": task.split,
                        "step": updates,
                        "prompt": prompt,
                        "response_ids": response.tolist(),
                        "loss": float(loss.detach()),
                    },
                )
                del logits, loss
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(teacher_parameters.values()), self.config["max_grad_norm"]
                )
            )
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("QUALIFY_TEACHER_INVALID_GRADIENT")
            optimizer.step()
            updates += 1
            self.assert_frozen()
            if any(p.grad is not None for p in self.student_parameters):
                raise ValueError("QUALIFY_FALLBACK_LEARNER_GRADIENT")
            self.event("teacher_optimizer_update", step=updates, grad_norm=norm)
        optimizer.zero_grad(set_to_none=True)
        if tensor_digest(adapter_parameters(self.model, "student")) != before:
            raise ValueError("QUALIFY_FALLBACK_LEARNER_MUTATED")
        teacher_hash = tensor_digest(teacher_parameters)
        if teacher_hash == before:
            raise ValueError("QUALIFY_TEACHER_DID_NOT_LEARN")
        folder = self.output / "teacher-checkpoint"
        self.model.save_pretrained(
            folder,
            selected_adapters=["teacher"],
            save_embedding_layers=False,
            safe_serialization=True,
        )
        self.model.load_adapter(
            folder / "teacher", adapter_name="teacher_reload", is_trainable=False
        )
        self.model.set_adapter("teacher_reload", inference_mode=True)
        if (
            tensor_digest(adapter_parameters(self.model, "teacher_reload"))
            != teacher_hash
        ):
            raise ValueError("QUALIFY_TEACHER_RELOAD_MISMATCH")
        self.model.set_adapter("teacher", inference_mode=True)
        self.model.delete_adapter("teacher_reload")
        receipt = {
            "status": "teacher_trained_frozen",
            "updates": updates,
            "learner_updates": 0,
            "learner_before_sha256": before,
            "learner_after_sha256": tensor_digest(
                adapter_parameters(self.model, "student")
            ),
            "teacher_sha256": teacher_hash,
            "training_ids": [task.uid for task in tasks],
            "initial_failure_sha256": failed_receipt["receipt_sha256"],
            "checkpoint_files_sha256": {
                path.name: file_digest(path)
                for path in (folder / "teacher").iterdir()
                if path.is_file()
            },
            "validation_used_in_training": False,
        }
        write_json(self.output / "teacher_training.json", receipt)
        return self.qualify("trained_teacher", receipt)

    def target_tokens(self, task):
        tokens = self.tokenizer.encode(
            digits(task.answer), add_special_tokens=False
        ) + [self.eos]
        if len(tokens) > self.config["max_new_tokens"]:
            raise ValueError("QUALIFY_SFT_TARGET_EXCEEDS_CAP")
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def train_family(self, method, family):
        if method not in self.config["training"]["methods"]:
            raise ValueError("QUALIFY_UNKNOWN_TRAINING_METHOD")
        tasks = [task for task in self.corpus["train"] if task.family == family]
        random.Random(self.config["seed"] + FAMILIES.index(family)).shuffle(tasks)
        batch_size = self.config["examples_per_update"]
        teacher_hash = tensor_digest(adapter_parameters(self.model, "teacher"))
        before = tensor_digest(adapter_parameters(self.model, "student"))
        start = self.step
        for offset in range(0, len(tasks), batch_size):
            self.optimizer.zero_grad(set_to_none=True)
            for task in tasks[offset : offset + batch_size]:
                self.model.set_adapter("student")
                prompt = learner_prompt(task)
                teacher_ids = None
                if method == "sft":
                    ids, response = self.prompt_ids(prompt), self.target_tokens(task)
                    logits = self.logits(ids, response)
                    loss = functional.cross_entropy(logits.float(), response)
                    diagnostics = {
                        "objective": "oracle_response_cross_entropy_including_eos"
                    }
                else:
                    ids, response, _, old = self.generate(prompt, sample=True)
                    self.total_rollout_tokens += response.numel()
                    teacher_ids = self.prompt_ids(
                        teacher_prompt(task, self.corpus["demonstration"])
                    )
                    self.model.set_adapter("teacher", inference_mode=True)
                    with torch.no_grad():
                        teacher_logits = self.logits(teacher_ids, response).detach()
                    self.model.set_adapter("student")
                    logits = self.logits(ids, response)
                    loss, diagnostics = distillation_loss(
                        logits, teacher_logits, response, "sdft_forward", old
                    )
                    del teacher_logits
                    if diagnostics["rollout_to_forward_logprob_max_error"] > 0.25:
                        raise ValueError("QUALIFY_ROLLOUT_FORWARD_MISMATCH")
                    self.total_teacher_tokens += teacher_ids.numel() + response.numel()
                if not torch.isfinite(loss):
                    raise ValueError("QUALIFY_NONFINITE_LOSS")
                (loss / batch_size).backward()
                self.total_loss_tokens += response.numel()
                append_json(
                    self.output / "training.jsonl",
                    {
                        "method": method,
                        "family": family,
                        "step": self.step,
                        "uid": task.uid,
                        "response_ids": response.tolist(),
                        "loss": float(loss.detach()),
                        "response_mask": [1] * response.numel(),
                        "student_response_start": ids.numel(),
                        "teacher_response_start": teacher_ids.numel()
                        if teacher_ids is not None
                        else None,
                        **diagnostics,
                    },
                )
                del logits, loss
            grads = [p.grad for p in self.student_parameters if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError("QUALIFY_INVALID_GRADIENTS")
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.student_parameters, self.config["max_grad_norm"]
                )
            )
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("QUALIFY_ZERO_OR_NONFINITE_GRADIENT")
            self.optimizer.step()
            self.step += 1
            self.assert_frozen()
            if any(
                p.grad is not None
                for p in adapter_parameters(self.model, "teacher").values()
            ):
                raise ValueError("QUALIFY_TEACHER_GRADIENT")
            self.event(
                "optimizer_update",
                method=method,
                family=family,
                step=self.step,
                grad_norm=norm,
            )
        if tensor_digest(adapter_parameters(self.model, "teacher")) != teacher_hash:
            raise ValueError("QUALIFY_FROZEN_TEACHER_MUTATED")
        after = tensor_digest(adapter_parameters(self.model, "student"))
        if after == before or self.step - start != self.config["updates_per_family"]:
            raise ValueError("QUALIFY_UPDATE_OR_WEIGHT_MISMATCH")
        return {
            "step_before": start,
            "step_after": self.step,
            "student_before_sha256": before,
            "student_after_sha256": after,
            "teacher_sha256": teacher_hash,
        }

    def save_reload(self, method, family, continuity):
        folder = self.output / "checkpoints" / method / family
        probe = self.corpus["validation"][0]
        ids, response = (
            self.prompt_ids(learner_prompt(probe)),
            self.target_tokens(probe),
        )
        self.model.set_adapter("student")
        with torch.no_grad():
            expected = self.logits(ids, response).detach().cpu()
        self.model.save_pretrained(
            folder,
            selected_adapters=["student"],
            save_embedding_layers=False,
            safe_serialization=True,
        )
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "torch_rng": torch.get_rng_state(),
            },
            folder / "training_state.pt",
        )
        self.model.load_adapter(
            folder / "student", adapter_name="reload", is_trainable=False
        )
        self.model.set_adapter("reload", inference_mode=True)
        with torch.no_grad():
            actual = self.logits(ids, response).detach().cpu()
        loaded = adapter_parameters(self.model, "reload")
        if tensor_digest(loaded) != continuity[
            "student_after_sha256"
        ] or not torch.equal(expected, actual):
            raise ValueError("QUALIFY_CHECKPOINT_RELOAD_MISMATCH")
        self.model.set_adapter("student")
        with torch.no_grad():
            for name, parameter in adapter_parameters(self.model, "student").items():
                parameter.copy_(loaded[name])
        self.model.delete_adapter("reload")
        state = torch.load(
            folder / "training_state.pt", map_location="cpu", weights_only=True
        )
        self.optimizer.load_state_dict(state["optimizer"])
        if state["step"] != self.step or any(
            float(s["step"]) != self.step for s in self.optimizer.state.values()
        ):
            raise ValueError("QUALIFY_OPTIMIZER_CONTINUITY_MISMATCH")
        write_json(
            folder / "receipt.json",
            {
                **continuity,
                "probe_uid": probe.uid,
                "adapter_reload_equal": True,
                "logits_reload_equal": True,
                "optimizer_reset_at_boundary": False,
                "fresh_context": True,
                "proof_boundary": (
                    "Separate adapter deserialization on the same frozen base; "
                    "no process restart proof."
                ),
            },
        )

    def evaluate_rows(self, label, stage, corpus, adapter="student"):
        self.model.set_adapter(adapter, inference_mode=True)
        results = {}
        for split, tasks in corpus.items():
            for task in tasks:
                row = self.generated_record(task, learner_prompt(task))
                append_json(
                    self.output / "evaluation.jsonl",
                    {
                        "method": label,
                        "stage": stage,
                        "split": split,
                        "family": task.family,
                        "uid": task.uid,
                        **row,
                    },
                )
                results.setdefault(f"{split}/{task.family}", {})[task.uid] = int(
                    row["correct"]
                )
        return results

    def check_initialization(self, receipt):
        if (
            receipt["snapshot_sha256"] != self.snapshot
            or receipt["eos_token_id"] != self.eos
            or receipt["special_token_ids"] != self.special_ids
        ):
            raise ValueError("QUALIFY_MODEL_OR_TOKENIZER_CHANGED")
        if (
            tensor_digest(adapter_parameters(self.model, "student"))
            != receipt["adapter_before_sha256"]
        ):
            raise ValueError("QUALIFY_INITIAL_ADAPTER_CHANGED")
        for pair in receipt["pairs"]:
            for condition in ("learner", "teacher"):
                row = pair[condition]
                tokens = row["token_ids"]
                body = tokens[:-1] if tokens and tokens[-1] == self.eos else tokens
                text = self.tokenizer.decode(
                    body,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if (
                    text != row["body_text"]
                    or self.prompt_ids(row["prompt"])[0].tolist()
                    != row["prompt_token_ids"]
                ):
                    raise ValueError("QUALIFY_TOKENIZER_TRACE_MISMATCH")

    def train(self, receipt):
        if receipt["status"] != "qualified":
            raise ValueError("QUALIFY_KD_REQUIRES_POSITIVE_QUALIFICATION")
        self.check_initialization(receipt)
        qualified_teacher = self.initial_adapter
        if receipt["candidate"] == "trained_teacher":
            path = Path(receipt["qualification_dir"]) / "teacher-checkpoint" / "teacher"
            self.model.load_adapter(
                path, adapter_name="qualified_teacher", is_trainable=False
            )
            qualified_teacher = {
                name: value.detach().clone()
                for name, value in adapter_parameters(
                    self.model, "qualified_teacher"
                ).items()
            }
            if tensor_digest(qualified_teacher) != receipt["teacher_sha256"]:
                raise ValueError("QUALIFY_TRAINED_TEACHER_TENSOR_MISMATCH")
            self.model.set_adapter("student")
            self.model.delete_adapter("qualified_teacher")
        stages, budgets = {}, {}
        for method in self.config["training"]["methods"]:
            self.reset_method()
            with torch.no_grad():
                for name, parameter in adapter_parameters(
                    self.model, "teacher"
                ).items():
                    parameter.copy_(qualified_teacher[name])
            stages[method] = {}
            for family in FAMILIES:
                continuity = self.train_family(method, family)
                self.save_reload(method, family, continuity)
                stages[method][family] = self.evaluate_rows(
                    method, family, {"validation": self.corpus["validation"]}
                )
            budgets[method] = {
                "updates": self.step,
                "loss_tokens": self.total_loss_tokens,
                "rollout_tokens": self.total_rollout_tokens,
                "teacher_tokens": self.total_teacher_tokens,
            }
        heldout = make_corpus(self.config, ("test", "composition"))
        self.reset_method()
        baseline = self.evaluate_rows("base", "final_only", heldout)
        finals = {}
        for method in self.config["training"]["methods"]:
            self.model.load_adapter(
                self.output / "checkpoints" / method / "C" / "student",
                adapter_name="evaluation",
                is_trainable=False,
            )
            finals[method] = self.evaluate_rows(
                method, "final_only", heldout, adapter="evaluation"
            )
            self.model.set_adapter("student")
            self.model.delete_adapter("evaluation")
        result = training_comparison(self.config, baseline, finals, stages, budgets)
        result.update(
            teacher_candidate=receipt["candidate"],
            teacher_training_updates=receipt["optimizer_updates"],
            qualification_receipt_sha256=receipt["receipt_sha256"],
        )
        write_json(self.output / "training_result.json", result)
        return result


def training_comparison(protocol, baseline, finals, stages, budgets):
    expected_updates = len(FAMILIES) * protocol["updates_per_family"]
    if any(item["updates"] != expected_updates for item in budgets.values()):
        raise ValueError("QUALIFY_CONTROL_UPDATE_MISMATCH")
    deltas = {}
    for split in ("test", "composition"):
        keys = [f"{split}/{family}" for family in FAMILIES]
        for reference, scores in (("base", baseline), ("sft", finals["sft"])):
            values = [
                finals["privileged_forward_kl"][key][uid] - hit
                for key in keys
                for uid, hit in scores[key].items()
            ]
            deltas[f"{split}_gain_over_{reference}"] = {
                "mean": sum(values) / len(values),
                "paired_task_bootstrap_95ci": confidence_interval(
                    values, protocol["seed"], protocol["training"]["bootstrap_samples"]
                ),
            }
    forgetting = {}
    for method, method_stages in stages.items():
        forgetting[method] = {}
        for index, family in enumerate(FAMILIES):
            key = f"validation/{family}"
            observed = [
                sum(method_stages[stage][key].values()) / len(method_stages[stage][key])
                for stage in FAMILIES[index:]
            ]
            forgetting[method][family] = max(observed) - observed[-1]
    decision = all(
        item["mean"] >= protocol["training"]["minimum_gain"]
        and item["paired_task_bootstrap_95ci"][0] > 0
        for item in deltas.values()
    )
    decision = (
        decision
        and max(forgetting["privileged_forward_kl"].values())
        <= protocol["training"]["maximum_forgetting"]
    )
    return {
        "status": "completed",
        "decision": "keep_for_independent_replication"
        if decision
        else "reject_this_recipe",
        "deltas": deltas,
        "forgetting": forgetting,
        "budgets": budgets,
        "baseline": baseline,
        "finals": finals,
        "validation_stages": stages,
        "protocol_sha256": digest(protocol),
        "limitations": [
            "One model seed and three synthetic families; no general CL claim.",
            "Equal updates and task order; supervision and token counts differ.",
            (
                "Frozen privileged teacher with the existing forward-KL kernel; "
                "not an SDFT or SDPO reproduction."
            ),
            (
                "Composition is generated internally in one response; "
                "no external command execution by the learner."
            ),
            "Checkpoint reload in one process; no independent process-restart proof.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [distillation-qualified] %(message)s"
    )
    dispatch = json.loads(args.config.read_text())
    protocol = json.loads((Path(__file__).parent / dispatch["protocol"]).read_text())
    validate_protocol(protocol)
    if digest(protocol) != dispatch["protocol_sha256"]:
        raise ValueError("QUALIFY_SEALED_PROTOCOL_CHANGED")
    if dispatch["stage"] not in ("qualification", "teacher_training", "training"):
        raise ValueError("QUALIFY_UNKNOWN_STAGE")
    seal = prepare(args.output_dir, dispatch, protocol)
    if args.validate_only:
        write_json(
            args.output_dir / "validation.json",
            {"status": "cpu_contract_validated_only", "gpu_qualified": False, **seal},
        )
        return
    runner = QualifiedRunner(protocol, args.output_dir)
    try:
        receipt = (
            verify_qualification(
                Path(dispatch["qualification_dir"]),
                protocol,
                required_status="rejected"
                if dispatch["stage"] == "teacher_training"
                else "qualified",
            )
            if dispatch["stage"] != "qualification"
            else None
        )
        if (
            dispatch["stage"] == "teacher_training"
            and receipt["candidate"] != "instruction_demo"
        ):
            raise ValueError("QUALIFY_ONE_FALLBACK_AFTER_INITIAL_REJECTION_ONLY")
        runner.load()
        if receipt is None:
            runner.qualify()
        elif dispatch["stage"] == "teacher_training":
            runner.train_teacher(receipt)
        else:
            runner.train(receipt)
    except Exception as error:
        runner.event("failed", exception=type(error).__name__, detail=str(error))
        write_json(
            args.output_dir / "failure.json",
            {
                "status": "failed",
                "exception": type(error).__name__,
                "detail": str(error),
            },
        )
        LOGGER.exception("QUALIFY_EXECUTION_FAILED")
        raise


if __name__ == "__main__":
    main()
