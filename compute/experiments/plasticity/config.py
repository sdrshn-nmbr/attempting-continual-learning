import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

METHODS = (
    "persistent",
    "balanced_replay",
    "optimizer_reset",
    "selective_reset",
    "fresh_base",
)
DATASET_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"


@dataclass(frozen=True)
class Config:
    model_id: str
    revision: str
    model_path: str
    seed: int = 17
    pilot: bool = True
    methods: list[str] = field(
        default_factory=lambda: ["persistent", "balanced_replay"]
    )
    tasks: int = 4
    classes_per_task: int = 4
    train_per_class: int = 32
    validation_per_class: int = 10
    test_per_class: int = 20
    updates_per_task: int = 32
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    eval_batch_size: int = 8
    eval_every: int = 8
    checkpoint_every: int = 8
    max_length: int = 128
    learning_rate: float = 0.0003
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    reset_fraction: float = 0.25
    acquisition_threshold: float = 0.7
    replay_fraction: float = 0.5
    replay_capacity: int | None = None
    acquisition_floor: float = 0.9
    max_acquisition_drop: float = 0.05
    min_retention_gain: float = 0.1
    diagnostic_examples: int = 32
    diagnostic_features: int = 256
    dataset_cache_dir: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if not self.model_id or not self.model_path:
            raise ValueError("model_id and local model_path are required")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be an immutable 40-character model commit")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        if set(self.methods) - set(METHODS):
            raise ValueError(f"Unknown method; supported methods: {METHODS}")
        for name in (
            "tasks",
            "classes_per_task",
            "train_per_class",
            "validation_per_class",
            "test_per_class",
            "updates_per_task",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "eval_every",
            "checkpoint_every",
            "max_length",
            "lora_rank",
            "lora_alpha",
            "diagnostic_examples",
            "diagnostic_features",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.tasks <= 4 or self.classes_per_task < 2:
            raise ValueError(
                "This pilot requires 2-4 tasks with at least two classes each"
            )
        if self.tasks * self.classes_per_task > 26:
            raise ValueError(
                "At most 26 classes fit the verified single-letter code set"
            )
        if self.effective_batch_size % self.classes_per_task:
            raise ValueError(
                "Effective batch size must be divisible by classes_per_task"
            )
        if "balanced_replay" in self.methods:
            replay_size = self.effective_batch_size * self.replay_fraction
            if not 0 < self.replay_fraction < 1 or not replay_size.is_integer():
                raise ValueError(
                    "replay_fraction must allocate an integer, nonempty strict subset of the effective batch"
                )
            if (self.effective_batch_size - int(replay_size)) % self.classes_per_task:
                raise ValueError(
                    "Current examples per replay update must be divisible by classes_per_task"
                )
        if self.replay_capacity is not None and (
            type(self.replay_capacity) is not int
            or self.replay_capacity < self.tasks * self.classes_per_task
            or self.methods != ["balanced_replay"]
        ):
            raise ValueError(
                "A bounded replay capacity requires balanced_replay alone and at least one retained example per selected class"
            )
        if (
            not 0 < self.reset_fraction < 1
            or int(self.lora_rank * self.reset_fraction) < 1
        ):
            raise ValueError(
                "reset_fraction must renew at least one, but not all, rank components"
            )
        if not 0 < self.acquisition_threshold <= 1:
            raise ValueError("acquisition_threshold must be in (0, 1]")
        if (
            not 0 < self.acquisition_floor <= 1
            or not 0 <= self.max_acquisition_drop < 1
            or not 0 < self.min_retention_gain <= 1
        ):
            raise ValueError("Invalid acquisition/retention screening thresholds")
        if (
            not 0 < self.learning_rate < 1
            or self.weight_decay < 0
            or self.max_grad_norm <= 0
        ):
            raise ValueError("Invalid optimizer hyperparameters")
        if self.device not in ("cuda:0", "cpu") or self.dtype not in (
            "bfloat16",
            "float32",
        ):
            raise ValueError(
                "Supported devices are cuda:0/cpu and dtypes bfloat16/float32"
            )
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU verification requires float32")

    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps

    def as_dict(self):
        return asdict(self)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def derived_seed(seed, *parts):
    return int(digest([seed, *parts])[:15], 16)
