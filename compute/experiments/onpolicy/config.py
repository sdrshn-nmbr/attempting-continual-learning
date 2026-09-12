import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ARMS = ("off_policy_dense", "on_policy_dense", "oracle_sft")
RUNTIME = {
    "torch": "2.10.0",
    "transformers": "5.16.1",
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "huggingface-hub": "1.30.0",
    "numpy": "2.5.3",
    "safetensors": "0.8.0",
    "tokenizers": "0.23.2",
}
PROTOCOL = {
    "scope": "bounded algorithmic list transformations; not general continual learning",
    "study": "https://labs.baseten.co/articles/dense-on-policy-or-both",
    "study_limits": "The study leaves mechanism, principle acquisition, larger-scale and broader transfer unresolved.",
    "objective": "Full-vocabulary reverse KL(student || frozen privileged teacher), with differentiable student probabilities; identical objective in both dense arms.",
    "difference_from_study": "Direct differentiable conditional KL on sampled prefixes, not the study's sampled token-advantage RL estimator; also different tasks, models and sequential setting.",
    "budget": "Identical ordered task/query schedule, optimizer updates and valid continuation-token count per update. Repeat that query after EOS until its quota is met. Include EOS once; never train after EOS. Last rollout may be truncated.",
    "unmatched": "Prompt tokens, rollout counts, generated tokens and total FLOPs are logged, not matched. Oracle SFT is an extra-supervision diagnostic control.",
    "policy": "Unconstrained full-vocabulary sampling at temperature 1, no top-k/top-p filtering, fresh rollouts before each optimizer update; rollout tokens detached.",
    "privilege": "Teacher gets operation definitions, never the current oracle answer. Student training gets only disjoint demonstrations and task cues. All evaluation inputs are disjoint from train/demo/gate inputs.",
    "primary": "Greedy executable accuracy and gold-answer log probability with task cues only, before training and after each sequential task. Rules and demonstrations are absent.",
    "secondary": "Matched evaluations with fixed disjoint demonstrations, plus unseen within-task and cross-task compositions; report invalid outputs, lexical refusal and length truncation.",
    "refusal_measurement": "Refusal is a fixed lexical detector, not an exhaustive semantic classifier. All malformed/non-JSON output is also counted as invalid.",
    "teacher_gate": "Independent calibration inputs cover each primitive task, within-task composition and cross-task composition. Every suite must meet the generation-accuracy threshold before any update.",
    "interpretation": "Single-seed tiny pilots are feasibility measurements, not robust comparative evidence. No claim that interpretability improves learning is tested by this lane; saved task adapters permit subsequent causal analysis.",
}
CALIBRATION_PROTOCOL = {
    "scope": "Frozen-teacher instruction qualification on synthetic train/dev inputs only; no learning claim.",
    "variants": ["current", "numbered"],
    "comparison": "One current/numbered pair per dev query, in that fixed order, with identical model weights, rules, demonstrations, program semantics and greedy decoding budget.",
    "numbered_change": "Number the operations left to right and state that every step consumes the preceding result. Provide no query answers or intermediate results.",
    "data_boundary": "Use demonstrations from input-pool indices 0:4 and dev inputs starting at index 64. No primary heldout split is constructed, loaded, scored or written. The first 64 indices contain all existing preflight/pilot primary evaluation inputs at seed 37.",
    "selection": "Select current if every dev suite meets the unchanged threshold; otherwise select numbered if every suite meets it; otherwise select nothing and leave qualification failed.",
    "budget": "Two fixed instruction variants, one greedy generation each per dev query, no optimizer construction or updates, no automatic training or additional prompt search.",
    "resume": "Checkpoint every completed dev generation and resume at the next uncompleted pair member. Do not resample completed records.",
}
TRAIN_ANSWER_CALIBRATION_PROTOCOL = {
    "name": "train_answer_calibration",
    "scope": "Train-only answer-conditioned selfteacher generation preflight; no SDFT learning or transfer claim.",
    "paper": "https://arxiv.org/html/2601.19897v1#S3",
    "teacher": "The initial student base model, with adapters disabled, conditioned on the current train query and its oracle response. No external teacher or weight updates.",
    "student": "Query and output-format instructions only. The student prompt never reads the oracle response; no student generation or scoring calls.",
    "prompt": "One fixed question/demonstration/respond template, adapted from the paper to JSON-only output with thinking disabled. No rules, disjoint demonstrations, prompt variants or search.",
    "data_boundary": "Exactly input_pool(37)[4:20], reused across two primitive and three composition suites. No demo, dev, gate or heldout split is constructed, loaded, scored or written.",
    "budget": "Exactly 80 greedy teacher generations at 32 new tokens per research call; zero optimizer construction, updates, loss tokens, extra logprob scoring or automatic training.",
    "qualification": "Every one of the five 16-example suites must have accuracy 1.0 and zero truncated responses. Success qualifies this answer-conditioned generation path only; copying is possible.",
    "ema": "At zero updates, the initialized EMA teacher equals the initial student. Implementing and testing EMA during learning is separate future work.",
    "resume": "Checkpoint each completed generation and preserve checkpointed records without resampling.",
    "limitations": "Synthetic final-answer demonstrations, JSON-only output and a 32-token cap differ from the paper's reasoning tasks. This does not measure proximity to the student policy, compositional learning, retention or SDFT effectiveness.",
}
ORACLE_CONTROL_PROTOCOL = {
    "name": "oracle_sft_control",
    "scope": "One teacher-independent oracle-SFT acquisition/retention feasibility control; no comparative or SDFT claim.",
    "training": "One persistent LoRA rank 8, alpha 16, AdamW learning rate 0.0002. Train permutation then symbol_map for 32 updates each, preserving adapter and optimizer across tasks.",
    "objective": "Hard-target cross entropy on complete compact oracle answers plus native tokenizer EOS. Each update averages 40 tokens from four distinct train queries; no partial tails, teacher, generated training responses or extra scoring.",
    "information": "Student prefix is FORMAT plus its query, without demonstrations, rules or teacher context. Oracle answers enter only the supervised continuation targets.",
    "train_data": "Seed 37, input_pool(37)[4:20], primitive programs only. Cycle the same 16 inputs deterministically. Stage A always uses four current examples; stage B uses four current examples without replay, or two current followed by two old examples with replay.",
    "replay": "The one bounded replay control sets replay_examples_per_update=2 for stage B only. Current and old streams each cycle their original 16 train examples four times. Stage B current-task exposure is intentionally halved from 128 to 64 complete answers, allocating the other 64 to old-task replay within the unchanged 2560-token total. This tests memory allocation, not a comparison with equal current-task exposure.",
    "evaluation_boundary": "Do not construct or score heldout data during training. After all 64 updates are committed, permanently enter post_training with no optimizer. Reload original initial, after_permutation and after_symbol_map adapters from disk and evaluate each.",
    "heldout": "Fresh fixed input_pool(37)[128:160]: 32 inputs across two primitive and three composition suites. Cue-only greedy generation; compositions never enter training.",
    "acquisition": "Each primitive must reach accuracy at least 0.75 and gain at least 0.25 over its ORIGINAL initial adapter score. Measure permutation after its stage and symbol_map after its stage.",
    "retention": "Report permutation accuracy drop from after_permutation to final. Assess the <=0.10 retention criterion only if both primitive acquisition criteria pass; otherwise label retention unqualified.",
    "composition": "Report composition accuracy changes descriptively and separately from feasibility gates.",
    "budget": "Exactly 64 optimizer updates, 256 complete answers and 2560 supervised tokens. Exactly 480 post-training greedy evaluation calls across three saved adapters. No automatic tuning, teacher fitting, extra model grid or automatic next experiment.",
}


@dataclass(frozen=True)
class Config:
    model_id: str
    revision: str
    model_path: str
    seed: int
    experiment_kind: str = "pilot"
    mode: str = "sequential"
    arms: tuple[str, ...] = ARMS
    steps_per_task: int = 16
    tokens_per_update: int = 32
    replay_examples_per_update: int = 0
    train_examples: int = 16
    eval_examples: int = 8
    gate_examples: int = 4
    teacher_min_accuracy: float = 0.75
    max_new_tokens: int = 32
    max_prompt_tokens: int = 1024
    learning_rate: float = 0.0002
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    gradient_checkpointing: bool = True
    checkpoint_every: int = 1
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    require_rocm: bool = True

    def validate(self):
        if not self.model_id or not Path(self.model_path).is_absolute():
            raise ValueError("model_id and an absolute cached model_path are required")
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("revision must be an immutable 40-character commit SHA")
        if self.experiment_kind not in ("preflight", "pilot", "cpu_test"):
            raise ValueError("experiment_kind must be preflight, pilot or cpu_test")
        if self.mode not in (
            "sequential", "teacher_calibration", "train_answer_calibration", "oracle_sft_control"
        ):
            raise ValueError("Unknown experiment mode")
        if type(self.replay_examples_per_update) is not int or self.replay_examples_per_update not in (0, 2):
            raise ValueError("replay_examples_per_update must be 0 or the fixed allocation 2")
        if self.replay_examples_per_update and self.mode != "oracle_sft_control":
            raise ValueError("Replay is supported only by oracle_sft_control")
        calibration = self.mode in ("teacher_calibration", "train_answer_calibration")
        if len(set(self.arms)) != len(self.arms) or set(self.arms) - set(ARMS):
            raise ValueError(f"arms must be distinct values from {ARMS}")
        for field in (
            "train_examples",
            "max_new_tokens",
            "max_prompt_tokens",
            "lora_rank",
            "lora_alpha",
            "checkpoint_every",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("steps_per_task", "tokens_per_update", "eval_examples", "gate_examples"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if self.train_examples < 2 or (
            self.mode not in ("train_answer_calibration", "oracle_sft_control")
            and self.gate_examples < 2
        ):
            raise ValueError("Every suite must include both primitive operations")
        if self.mode == "oracle_sft_control":
            fixed = {
                "seed": 37, "arms": ("oracle_sft",), "steps_per_task": 32,
                "tokens_per_update": 40, "train_examples": 16, "eval_examples": 32,
                "gate_examples": 0, "lora_rank": 8, "lora_alpha": 16,
                "learning_rate": 0.0002, "checkpoint_every": 1,
            }
            for name, expected in fixed.items():
                if getattr(self, name) != expected:
                    raise ValueError(f"oracle_sft_control fixes {name} at {expected}")
            if self.experiment_kind not in ("pilot", "cpu_test"):
                raise ValueError("oracle_sft_control is a pilot or CPU test")
            if self.experiment_kind != "cpu_test" and (
                self.model_id != "Qwen/Qwen3.5-4B"
                or self.revision != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
                or self.max_new_tokens != 32
            ):
                raise ValueError("oracle_sft_control fixes the 4B revision and 32-token eval cap")
        elif calibration:
            if self.arms or self.steps_per_task or self.tokens_per_update or self.eval_examples:
                raise ValueError(
                    "Calibration requires empty arms and zero update/token/eval budgets"
                )
            if self.mode == "train_answer_calibration":
                if (
                    self.seed != 37
                    or self.train_examples != 16
                    or self.gate_examples != 0
                    or self.experiment_kind == "pilot"
                ):
                    raise ValueError(
                        "train_answer_calibration requires seed 37, 16 train inputs, zero dev inputs and preflight/cpu_test scope"
                    )
                if self.experiment_kind != "cpu_test" and self.max_new_tokens != 32:
                    raise ValueError("train_answer_calibration fixes the research token cap at 32")
            else:
                minimum = 2 if self.experiment_kind == "cpu_test" else 16
                if self.train_examples != 16 or not minimum <= self.gate_examples <= 32:
                    raise ValueError(
                        f"teacher_calibration requires 16 train inputs and {minimum} to 32 dev inputs"
                    )
            if self.teacher_min_accuracy != 1.0:
                raise ValueError("Calibration preserves the 1.0 per-suite qualification threshold")
        elif (
            not self.arms
            or not self.steps_per_task
            or not self.tokens_per_update
            or self.eval_examples < 2
        ):
            raise ValueError(
                "Sequential experiments require arms, positive budgets and at least two eval inputs"
            )
        if not 0 < self.teacher_min_accuracy <= 1 or not math.isfinite(self.teacher_min_accuracy):
            raise ValueError("teacher_min_accuracy must be in (0, 1]")
        for field in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if (
            not calibration
            and self.experiment_kind == "pilot"
            and not 16 <= self.steps_per_task <= 32
        ):
            raise ValueError("The bounded pilot requires 16 to 32 updates per task")
        if self.dtype not in ("float32", "bfloat16") or self.device not in ("cpu", "cuda:0"):
            raise ValueError("Supported execution: float32/bfloat16 on cpu or the lane's cuda:0")
        if self.device == "cpu" and self.require_rocm:
            raise ValueError("CPU execution cannot require ROCm")
        if self.experiment_kind != "cpu_test" and not self.require_rocm:
            raise ValueError("Research configs require the ROCm lane; CPU is reserved for tests")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def read_config(path):
    payload = json.loads(Path(path).read_text())
    if "arms" in payload:
        payload["arms"] = tuple(payload["arms"])
    return Config(**payload).validate()
