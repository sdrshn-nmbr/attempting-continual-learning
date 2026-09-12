import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Qwen3_5ForConditionalGeneration,
)

from objectives import distillation_loss
from tasks import (
    FAMILIES,
    audit_dataset,
    demo_context,
    digits,
    feedback_context,
    make_dataset,
    paired_gate,
    score,
    student_prompt,
    to_records,
)

LOGGER = logging.getLogger("distillation")
METHODS = {"sft", "sdft_forward", "sdpo_sampled_reverse"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(
            value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def adapter_parameters(model, adapter):
    return {
        name.replace(f".{adapter}.", ".ADAPTER."): value
        for name, value in model.named_parameters()
        if f".{adapter}." in name
    }


def confidence_interval(values, seed, samples):
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples)
    )
    return [
        estimates[int(0.025 * samples)],
        estimates[min(samples - 1, int(0.975 * samples))],
    ]


def validate_config(config):
    if re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]) is None:
        raise ValueError(
            "DISTILL_INVALID_MODEL_REVISION: expected 40 lowercase hexadecimal characters"
        )
    if Path(config["model_path"]).name != config["model_revision"]:
        raise ValueError("DISTILL_SNAPSHOT_REVISION_MISMATCH")
    if not set(config["methods"]) <= METHODS or len(set(config["methods"])) != len(
        config["methods"]
    ):
        raise ValueError("DISTILL_INVALID_METHODS")
    if config["family_order"] != list(FAMILIES):
        raise ValueError("DISTILL_SCREEN_REQUIRES_ABC_ORDER")
    if (
        config["updates_per_family"] * config["examples_per_update"]
        != config["sizes"]["train"]
    ):
        raise ValueError(
            "DISTILL_BUDGET_MISMATCH: screen requires exactly one pass per family"
        )
    if not 0 < config["teacher_ema_rate"] <= 1:
        raise ValueError("DISTILL_INVALID_EMA_RATE")
    if set(config["sizes"]) != {"train", "gate", "validation", "test", "composition"}:
        raise ValueError("DISTILL_INVALID_SPLITS")


def validate_snapshot(config):
    model_path = Path(config["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"DISTILL_SNAPSHOT_MISSING: {model_path}")
    config_path = model_path / "config.json"
    if (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        != config["model_config_sha256"]
    ):
        raise ValueError("DISTILL_MODEL_CONFIG_CHECKSUM_MISMATCH")
    files = [model_path / name for name in config["weight_files"]]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("DISTILL_WEIGHT_SHARD_MISSING")
    if sum(path.stat().st_size for path in files) != config["weight_bytes"]:
        raise ValueError("DISTILL_WEIGHT_SIZE_MISMATCH")
    return model_path


def prepare_output(output, config):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("DISTILL_DISPATCHER_CONFIG_MISMATCH")
    owned_paths = (
        "dataset.json",
        "split_audit.json",
        "results.json",
        "events.jsonl",
        "metrics",
        "checkpoints",
        "autoresearch",
    )
    if any((output / name).exists() for name in owned_paths):
        raise FileExistsError(f"DISTILL_PREVIOUS_ARTIFACTS_EXIST: {output}")
    with (output / "distillation.started.json").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "started_at": time.time(),
                    "config_sha256": hashlib.sha256(
                        json.dumps(config, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            + "\n"
        )


class Runner:
    def __init__(self, config, output):
        self.config = config
        self.output = output
        self.corpus = make_dataset(config["data_seed"], config["sizes"])
        self.step = 0
        self.model = None
        self.tokenizer = None
        self.total_rollout_tokens = 0
        self.total_loss_tokens = 0
        self.total_teacher_tokens = 0
        self.ledger = output / "autoresearch" / "distillation-results.tsv"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text(
            "method\tdecision\tfinal_test_gain\tfinal_composition_gain\tmax_forgetting\treason\n"
        )
        write_json(output / "config.json", config)
        write_json(output / "dataset.json", to_records(self.corpus))
        write_json(output / "split_audit.json", audit_dataset(self.corpus))

    def event(self, name, **fields):
        record = {"event": name, "wall_time": time.time(), **fields}
        append_json(self.output / "events.jsonl", record)
        LOGGER.info("%s", json.dumps(record))

    def load(self):
        if (
            not torch.cuda.is_available()
            or not torch.version.hip
            or torch.cuda.device_count() != 1
        ):
            raise RuntimeError("DISTILL_GPU_CONTRACT: require one visible ROCm GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(self.config["seed"])
        model_path = validate_snapshot(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
        if any(parameter.device.type != "cuda" for parameter in base.parameters()):
            raise RuntimeError("DISTILL_MODEL_OFFLOAD_DISALLOWED")
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise RuntimeError("DISTILL_MISSING_EOS_TOKEN")
        lora = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=0.0,
            target_modules=self.config["lora_targets"],
            bias="none",
        )
        self.model = get_peft_model(base, lora, adapter_name="student")
        self.model.add_adapter("teacher", copy.deepcopy(lora))
        self.model.set_adapter("student")
        self.initial_adapter = {
            name: parameter.detach().clone()
            for name, parameter in adapter_parameters(self.model, "student").items()
        }
        self.student_parameters = list(
            adapter_parameters(self.model, "student").values()
        )
        self.frozen_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if ".lora_" not in name
        ]
        self.frozen_versions = [
            parameter._version for parameter in self.frozen_parameters
        ]
        self.model.eval()
        write_json(
            self.output / "runtime.json",
            {
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "peft", "safetensors")
                },
                "rocm": torch.version.hip,
                "gpu": str(torch.cuda.get_device_properties(0)),
                "visible_devices": {
                    key: os.environ.get(key)
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                    )
                },
                "base_dtype": str(base.dtype),
                "adapter_dtypes": sorted(
                    {str(value.dtype) for value in self.student_parameters}
                ),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.student_parameters
                ),
                "adapter_initial_sha256": tensor_digest(self.initial_adapter),
                "targeted_parameter_names": list(self.initial_adapter),
                "sampling": {
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "support": "full_vocabulary",
                    "staleness": 0,
                },
                "evaluation": "greedy, fresh single user message, no retrieval, no verifier or demonstration context",
                "source_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in Path(__file__).parent.glob("*.py")
                },
            },
        )
        self.event(
            "runtime_qualified",
            trainable_parameters=sum(
                parameter.numel() for parameter in self.student_parameters
            ),
        )

    def prompt_ids(self, prompt):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
        ).to("cuda:0")
        if (
            ids.shape[-1] + self.config["max_new_tokens"]
            > self.config["max_context_tokens"]
        ):
            raise RuntimeError("DISTILL_CONTEXT_LIMIT: refusing silent truncation")
        return ids

    def generate(self, prompt, sample):
        inputs = self.prompt_ids(prompt)
        generation = GenerationConfig(
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=sample,
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
        with torch.inference_mode():
            result = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation,
            )
        response = result.sequences[0, inputs.shape[-1] :].clone()
        text = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        if not response.numel():
            raise RuntimeError("DISTILL_EMPTY_ROLLOUT")
        rollout_logprobs = None
        if sample:
            rollout_logprobs = torch.stack(
                [
                    functional.log_softmax(logits[0].float(), dim=-1)[token]
                    for logits, token in zip(result.scores, response)
                ]
            ).detach()
            if (
                len(result.scores) != response.numel()
                or not torch.isfinite(rollout_logprobs).all()
            ):
                raise RuntimeError("DISTILL_INVALID_ROLLOUT_LOGPROBS")
        return inputs, response, text, rollout_logprobs

    def logits(self, prompt_ids, response):
        tokens = torch.cat((prompt_ids, response[None, :]), dim=-1)[:, :-1]
        if tokens.shape[-1] >= self.config["max_context_tokens"]:
            raise RuntimeError("DISTILL_CONTEXT_LIMIT: training sequence too long")
        result = self.model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
            logits_to_keep=response.numel(),
        ).logits[0]
        if result.shape[0] != response.numel() or not torch.isfinite(result).all():
            raise RuntimeError("DISTILL_INVALID_RESPONSE_LOGITS")
        return result

    def evaluate(self, method, stage, splits):
        self.model.set_adapter("student")
        self.model.eval()
        results = {}
        for split in splits:
            for family in FAMILIES:
                rows = []
                for task in self.corpus[split]:
                    if task.family != family:
                        continue
                    _, _, text, _ = self.generate(student_prompt(task), sample=False)
                    row = {
                        "method": method,
                        "stage": stage,
                        "split": split,
                        "family": family,
                        "uid": task.uid,
                        "response": text,
                        **score(task, text),
                    }
                    append_json(self.output / "evaluations.jsonl", row)
                    rows.append(row)
                key = f"{split}/{family}"
                results[key] = {
                    "n": len(rows),
                    "accuracy": sum(row["correct"] for row in rows) / len(rows),
                    "format_valid": sum(row["format_valid"] for row in rows)
                    / len(rows),
                    "digit_accuracy": sum(row["digit_accuracy"] for row in rows)
                    / len(rows),
                    "by_id": {row["uid"]: int(row["correct"]) for row in rows},
                }
                self.event(
                    "evaluation",
                    method=method,
                    stage=stage,
                    split=split,
                    family=family,
                    accuracy=results[key]["accuracy"],
                )
        write_json(self.output / "metrics" / f"{method}-{stage}.json", results)
        return results

    def gate(self):
        outcomes = {}
        self.model.set_adapter("student")
        for family in FAMILIES:
            students, demonstrations, feedbacks = [], [], []
            for task in self.corpus["gate"]:
                if task.family != family:
                    continue
                prompt = student_prompt(task)
                _, _, attempt, _ = self.generate(prompt, sample=False)
                demo, demo_ids = demo_context(task, self.corpus["demonstration"])
                feedback = feedback_context(task, attempt)
                _, _, demo_answer, _ = self.generate(
                    prompt + "\n\n" + demo, sample=False
                )
                _, _, feedback_answer, _ = self.generate(
                    prompt + "\n\n" + feedback, sample=False
                )
                students.append(score(task, attempt)["correct"])
                demonstrations.append(score(task, demo_answer)["correct"])
                feedbacks.append(score(task, feedback_answer)["correct"])
                append_json(
                    self.output / "teacher_gate_traces.jsonl",
                    {
                        "uid": task.uid,
                        "family": family,
                        "prompt": prompt,
                        "student_response": attempt,
                        "demo_context": demo,
                        "demonstration_ids": demo_ids,
                        "demo_response": demo_answer,
                        "feedback_context": feedback,
                        "feedback_response": feedback_answer,
                        "student_correct": students[-1],
                        "demo_correct": demonstrations[-1],
                        "feedback_correct": feedbacks[-1],
                        "current_target_answer_in_demo": digits(task.answer) in demo,
                        "current_target_answer_in_feedback": digits(task.answer)
                        in feedback,
                    },
                )
            outcomes[family] = {
                method: paired_gate(
                    students,
                    answers,
                    self.config["gate_min_accuracy"],
                    self.config["gate_min_gain"],
                    self.config["gate_max_p"],
                )
                for method, answers in (
                    ("sdft_forward", demonstrations),
                    ("sdpo_sampled_reverse", feedbacks),
                )
            }
            self.event(
                "teacher_advantage_gate", family=family, results=outcomes[family]
            )
        write_json(self.output / "teacher_gate.json", outcomes)
        return outcomes

    def reset_method(self):
        for adapter in ("student", "teacher"):
            parameters = adapter_parameters(self.model, adapter)
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self.initial_adapter[name])
        self.model.set_adapter("student")
        self.model.eval()
        if tensor_digest(adapter_parameters(self.model, "student")) != tensor_digest(
            self.initial_adapter
        ):
            raise RuntimeError("DISTILL_INITIALIZATION_MISMATCH")
        torch.manual_seed(self.config["seed"])
        self.step = 0
        self.total_rollout_tokens = 0
        self.total_loss_tokens = 0
        self.total_teacher_tokens = 0
        self.optimizer = torch.optim.AdamW(
            self.student_parameters,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def update_teacher(self):
        students = adapter_parameters(self.model, "student")
        teachers = adapter_parameters(self.model, "teacher")
        with torch.no_grad():
            for name, teacher in teachers.items():
                teacher.lerp_(students[name], self.config["teacher_ema_rate"])

    def learn_family(self, method, family):
        tasks = [task for task in self.corpus["train"] if task.family == family]
        random.Random(self.config["seed"] + FAMILIES.index(family)).shuffle(tasks)
        batch_size = self.config["examples_per_update"]
        before = tensor_digest(adapter_parameters(self.model, "student"))
        optimizer_step_before = self.step
        for offset in range(0, len(tasks), batch_size):
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            started = time.perf_counter()
            for task in tasks[offset : offset + batch_size]:
                self.model.set_adapter("student")
                prompt = student_prompt(task)
                prompt_ids, response, attempt, old_logprobs = self.generate(
                    prompt, sample=True
                )
                rollout_version = self.step
                self.total_rollout_tokens += response.numel()
                teacher_prompt = None
                teacher_ids = None
                demo_ids = []
                if method == "sft":
                    target = self.tokenizer.encode(
                        digits(task.answer), add_special_tokens=False
                    ) + [self.eos]
                    train_response = torch.tensor(target, device="cuda:0")
                    student_logits = self.logits(prompt_ids, train_response)
                    loss = functional.cross_entropy(
                        student_logits.float(), train_response
                    )
                    diagnostics = {"objective": "hard_label_cross_entropy"}
                else:
                    if method == "sdft_forward":
                        context, demo_ids = demo_context(
                            task, self.corpus["demonstration"]
                        )
                    else:
                        context = feedback_context(task, attempt)
                    teacher_prompt = prompt + "\n\n" + context
                    teacher_ids = self.prompt_ids(teacher_prompt)
                    self.total_teacher_tokens += teacher_ids.numel() + response.numel()
                    self.model.set_adapter("teacher")
                    with torch.no_grad():
                        teacher_logits = self.logits(teacher_ids, response).detach()
                    self.model.set_adapter("student")
                    student_logits = self.logits(prompt_ids, response)
                    train_response = response
                    loss, diagnostics = distillation_loss(
                        student_logits, teacher_logits, response, method, old_logprobs
                    )
                    if diagnostics["rollout_to_forward_logprob_max_error"] > 0.25:
                        raise RuntimeError(
                            f"DISTILL_ROLLOUT_DISTRIBUTION_MISMATCH: {diagnostics}"
                        )
                    del teacher_logits
                if rollout_version != self.step:
                    raise RuntimeError("DISTILL_STALE_ROLLOUT")
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"DISTILL_NONFINITE_LOSS: {method} {family} {self.step}"
                    )
                (loss / batch_size).backward()
                losses.append(float(loss.detach()))
                self.total_loss_tokens += train_response.numel()
                append_json(
                    self.output / "trajectories.jsonl",
                    {
                        "method": method,
                        "family": family,
                        "step": self.step,
                        "uid": task.uid,
                        "student_prompt": prompt,
                        "student_prompt_ids": prompt_ids[0].tolist(),
                        "response": attempt,
                        "response_ids": response.tolist(),
                        "rollout_logprobs": old_logprobs.cpu().tolist(),
                        "rollout_policy_version": rollout_version,
                        "train_response_ids": train_response.tolist(),
                        "response_mask": [1] * train_response.numel(),
                        "student_response_start": prompt_ids.numel(),
                        "teacher_prompt": teacher_prompt,
                        "teacher_prompt_ids": teacher_ids[0].tolist()
                        if teacher_ids is not None
                        else None,
                        "teacher_response_start": teacher_ids.numel()
                        if teacher_ids is not None
                        else None,
                        "demonstration_ids": demo_ids,
                        "loss": losses[-1],
                        **score(task, attempt),
                        **diagnostics,
                    },
                )
                del student_logits, loss
            gradients = [
                parameter.grad
                for parameter in self.student_parameters
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise RuntimeError("DISTILL_NONFINITE_OR_MISSING_GRADIENT")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.student_parameters, self.config["max_grad_norm"]
                )
            )
            if grad_norm <= 0 or not math.isfinite(grad_norm):
                raise RuntimeError("DISTILL_ZERO_OR_NONFINITE_GRADIENT_NORM")
            self.optimizer.step()
            self.step += 1
            self.update_teacher()
            if any(parameter.grad is not None for parameter in self.frozen_parameters):
                raise RuntimeError("DISTILL_FROZEN_BASE_HAS_GRADIENT")
            if any(
                parameter._version != version
                for parameter, version in zip(
                    self.frozen_parameters, self.frozen_versions
                )
            ):
                raise RuntimeError("DISTILL_FROZEN_BASE_MUTATED")
            self.event(
                "optimizer_update",
                method=method,
                family=family,
                step=self.step,
                mean_loss=sum(losses) / len(losses),
                grad_norm=grad_norm,
                seconds=time.perf_counter() - started,
                rollout_tokens=self.total_rollout_tokens,
                loss_tokens=self.total_loss_tokens,
                teacher_tokens=self.total_teacher_tokens,
            )
        after = tensor_digest(adapter_parameters(self.model, "student"))
        if before == after:
            raise RuntimeError("DISTILL_PARAMETERS_DID_NOT_CHANGE")
        return {
            "adapter_before_sha256": before,
            "adapter_after_sha256": after,
            "optimizer_step_before": optimizer_step_before,
            "optimizer_step_after": self.step,
            "optimizer_reset_at_family_boundary": False,
        }

    def checkpoint(self, method, family, continuity):
        folder = self.output / "checkpoints" / method / family
        self.model.set_adapter("student")
        probe = self.corpus["validation"][0]
        prompt_ids = self.prompt_ids(student_prompt(probe))
        response = torch.tensor(
            self.tokenizer.encode(digits(probe.answer), add_special_tokens=False)
            + [self.eos],
            device="cuda:0",
        )
        with torch.no_grad():
            before_logits = self.logits(prompt_ids, response).float().cpu()
        before_hash = tensor_digest(adapter_parameters(self.model, "student"))
        self.model.save_pretrained(
            folder,
            selected_adapters=["student", "teacher"],
            safe_serialization=True,
            save_embedding_layers=False,
        )
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            folder / "training_state.pt",
        )
        self.model.load_adapter(
            folder / "student", adapter_name="reload", is_trainable=False
        )
        self.model.set_adapter("reload", inference_mode=True)
        with torch.no_grad():
            after_logits = self.logits(prompt_ids, response).float().cpu()
        after_hash = tensor_digest(adapter_parameters(self.model, "reload"))
        equal = torch.equal(before_logits, after_logits)
        if before_hash != after_hash or not equal:
            raise RuntimeError(
                f"DISTILL_CHECKPOINT_RELOAD_MISMATCH: weights={before_hash == after_hash} logits_maxdiff={(before_logits - after_logits).abs().max()}"
            )
        self.model.set_adapter("student")
        with torch.no_grad():
            for name, parameter in adapter_parameters(self.model, "student").items():
                parameter.copy_(adapter_parameters(self.model, "reload")[name])
        self.model.delete_adapter("reload")
        state = torch.load(
            folder / "training_state.pt", map_location="cpu", weights_only=True
        )
        previous_steps = [float(item["step"]) for item in self.optimizer.state.values()]
        self.optimizer.load_state_dict(state["optimizer"])
        restored_steps = [float(item["step"]) for item in self.optimizer.state.values()]
        if previous_steps != restored_steps or state["step"] != self.step:
            raise RuntimeError("DISTILL_OPTIMIZER_RELOAD_MISMATCH")
        receipt = {
            **continuity,
            "probe_uid": probe.uid,
            "weight_reload_equal": True,
            "logits_reload_bitwise_equal": True,
            "optimizer_steps_reload_equal": True,
            "adapter_sha256": before_hash,
            "fresh_context": True,
            "next_stage_student_weights_restored_from_checkpoint": True,
            "files_sha256": {
                str(path.relative_to(folder)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in folder.rglob("*")
                if path.is_file()
            },
            "proof_boundary": "Adapter deserialized into a separate adapter on the same mutation-checked frozen base; this check does not restart the process.",
        }
        write_json(folder / "receipt.json", receipt)
        self.event("checkpoint_qualified", method=method, family=family, **receipt)

    def method_result(self, method, baseline, stages):
        final = stages["C"]
        test_deltas, composition_deltas = [], []
        for family in FAMILIES:
            for split, deltas in (
                ("test", test_deltas),
                ("composition", composition_deltas),
            ):
                key = f"{split}/{family}"
                deltas.extend(
                    final[key]["by_id"][uid] - correct
                    for uid, correct in baseline[key]["by_id"].items()
                )
        forgetting = {}
        gains_by_stage = {}
        for index, family in enumerate(FAMILIES):
            key = f"validation/{family}"
            observed = [stages[stage][key]["accuracy"] for stage in FAMILIES[index:]]
            forgetting[family] = max(observed) - final[key]["accuracy"]
            prior = baseline if index == 0 else stages[FAMILIES[index - 1]]
            gains_by_stage[family] = (
                stages[family][key]["accuracy"] - prior[key]["accuracy"]
            )
        gain = sum(test_deltas) / len(test_deltas)
        composition_gain = sum(composition_deltas) / len(composition_deltas)
        interval = confidence_interval(
            test_deltas, self.config["seed"], self.config["bootstrap_samples"]
        )
        keep = (
            gain >= self.config["keep_min_final_gain"]
            and interval[0] > 0
            and max(forgetting.values()) <= self.config["keep_max_forgetting"]
            and all(value > 0 for value in gains_by_stage.values())
        )
        result = {
            "method": method,
            "decision": "keep_for_replication"
            if keep
            else "discard_current_configuration",
            "final_test_gain": gain,
            "paired_task_bootstrap_95ci": interval,
            "final_composition_gain": composition_gain,
            "maximum_old_skill_forgetting": max(forgetting.values()),
            "forgetting_by_family": forgetting,
            "skill_gain_at_each_stage": gains_by_stage,
            "updates": self.step,
            "rollout_tokens": self.total_rollout_tokens,
            "supervised_tokens": self.total_loss_tokens,
            "teacher_forward_tokens": self.total_teacher_tokens,
            "seed": self.config["seed"],
            "hyperparameter_selection": "none; frozen first-screen config",
            "limitations": [
                "Synthetic command semantics and short sequence composition; no broad capability evidence.",
                "One seed; task bootstrap does not quantify seed or task-family variability.",
                "Updates and sampled question counts are matched; conditioning information and loss-token counts differ.",
                "SDFT uses other-input worked examples; SDPO uses rich verifier contracts and a sampled score-function gradient.",
                "Retention covers acquired synthetic skills; no general-language benchmark is included.",
            ],
        }
        write_json(self.output / "metrics" / f"{method}-result.json", result)
        with self.ledger.open("a") as stream:
            stream.write(
                f"{method}\t{result['decision']}\t{gain}\t{composition_gain}\t{max(forgetting.values())}\tprespecified_screen_criteria\n"
            )
        self.event("method_completed", **result)
        return result

    def run(self):
        self.load()
        gates = self.gate()
        baseline = self.evaluate(
            "base", "initial", ("validation", "test", "composition")
        )
        results = []
        for method in self.config["methods"]:
            if method != "sft" and not all(
                gates[family][method]["passed"] for family in FAMILIES
            ):
                result = {
                    "method": method,
                    "decision": "discard_before_training",
                    "reason": "teacher_advantage_gate_failed",
                    "updates": 0,
                }
                results.append(result)
                with self.ledger.open("a") as stream:
                    stream.write(
                        f"{method}\tdiscard_before_training\tNA\tNA\tNA\tteacher_advantage_gate_failed\n"
                    )
                self.event("method_gated", **result)
                continue
            self.reset_method()
            stages = {}
            for family in FAMILIES:
                continuity = self.learn_family(method, family)
                self.checkpoint(method, family, continuity)
                splits = (
                    ("validation", "test", "composition")
                    if family == "C"
                    else ("validation",)
                )
                stages[family] = self.evaluate(method, family, splits)
            results.append(self.method_result(method, baseline, stages))
        write_json(
            self.output / "results.json",
            {"status": "completed", "results": results, "teacher_gate": gates},
        )
        self.event("screen_completed", results=results)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [distillation] %(message)s"
    )
    config = json.loads(args.config.read_text())
    validate_config(config)
    prepare_output(args.output_dir, config)
    runner = Runner(config, args.output_dir)
    if args.validate_only:
        runner.event("data_validated", **audit_dataset(runner.corpus))
        return
    try:
        runner.run()
    except Exception as error:
        runner.event("screen_failed", exception=type(error).__name__, detail=str(error))
        write_json(
            args.output_dir / "results.json",
            {
                "status": "failed",
                "exception": type(error).__name__,
                "detail": str(error),
            },
        )
        LOGGER.exception("DISTILL_SCREEN_FAILED")
        raise


if __name__ == "__main__":
    main()
