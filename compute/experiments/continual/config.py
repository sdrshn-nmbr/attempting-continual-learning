import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DataConfig:
    num_tasks: int = 3
    alphabet_size: int = 8
    train_examples: int = 128
    reference_examples: int = 32
    calibration_examples: int = 8
    id_examples: int = 16
    longer_examples: int = 8
    composed_examples: int = 8
    min_length: int = 3
    max_length: int = 5
    longer_min_length: int = 7
    longer_max_length: int = 9
    include_reversal: bool = True


@dataclass(frozen=True)
class PilotConfig:
    steps_per_task: int = 32
    current_batch_size: int = 2
    reference_batch_size: int = 2
    eval_batch_size: int = 4
    max_sequence_length: int = 256
    max_new_tokens: int = 48
    checkpoint_every_steps: int = 4
    base_accuracy_ceiling: float = 0.5


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 8
    alpha: int = 16
    last_n_layers: int = 8
    modules: tuple[str, ...] = (
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 0.1
    max_grad_norm: float = 1.0
    replay_weight: float = 0.5


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    gradient_checkpointing: bool = False
    deterministic_algorithms: bool = False
    cpu_threads: int = 4


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    revision: str
    model_path: str
    seed: int = 1729
    purpose: str = "pilot"
    model_source: str = "huggingface"
    methods: tuple[str, ...] = ("sequential", "replay", "agem")
    data: DataConfig = field(default_factory=DataConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        for name, kind in (
            ("data", DataConfig),
            ("pilot", PilotConfig),
            ("lora", LoraSettings),
            ("optimization", OptimizationConfig),
            ("runtime", RuntimeConfig),
        ):
            nested = dict(values.get(name, {}))
            if name == "lora" and "modules" in nested:
                nested["modules"] = tuple(nested["modules"])
            values[name] = kind(**nested)
        if "methods" in values:
            values["methods"] = tuple(values["methods"])
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        scalars = (
            self.pilot.base_accuracy_ceiling,
            self.optimization.learning_rate,
            self.optimization.max_grad_norm,
            self.optimization.replay_weight,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError(
                "[config] optimization and screening scalars must be finite"
            )
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("[config] methods must be nonempty and unique")
        if set(self.methods) - {"sequential", "replay", "agem"}:
            raise ValueError("[config] unknown method")
        if self.purpose not in {"integration_gate", "pilot", "scaled"}:
            raise ValueError("[config] unknown purpose")
        if self.model_source not in {"huggingface", "local_cpu_fixture"}:
            raise ValueError("[config] unknown model_source")
        if self.model_source == "huggingface" and not re.fullmatch(
            r"[0-9a-f]{40}", self.revision
        ):
            raise ValueError(
                "[config] Hugging Face revision must be an immutable 40-character commit"
            )
        if self.model_source == "local_cpu_fixture" and self.runtime.device != "cpu":
            raise ValueError(
                "[config] local_cpu_fixture is restricted to CPU validation"
            )
        if self.runtime.device not in {"cuda:0", "cpu"}:
            raise ValueError("[config] GPU isolation must expose cuda:0")
        if self.runtime.dtype not in {"bfloat16", "float32"}:
            raise ValueError("[config] supported dtypes are bfloat16 and float32")
        if self.runtime.device == "cpu" and self.runtime.dtype != "float32":
            raise ValueError("[config] CPU validation requires float32")
        if self.runtime.cpu_threads < 1:
            raise ValueError("[config] cpu_threads must be positive")
        if self.runtime.attention not in {"sdpa", "eager"}:
            raise ValueError("[config] use an installed PyTorch attention backend")
        data = self.data
        if not 3 <= data.alphabet_size <= 10 or data.num_tasks < 2:
            raise ValueError("[config] require >=2 tasks and 3..10 digit symbols")
        if (
            not 1
            <= data.min_length
            <= data.max_length
            < data.longer_min_length
            <= data.longer_max_length
        ):
            raise ValueError(
                "[config] longer transfer lengths must exceed the training range"
            )
        for name in (
            "train_examples",
            "reference_examples",
            "calibration_examples",
            "id_examples",
            "longer_examples",
            "composed_examples",
        ):
            if getattr(data, name) < 1:
                raise ValueError(f"[config] {name} must be positive")
        for name in (
            "steps_per_task",
            "current_batch_size",
            "reference_batch_size",
            "eval_batch_size",
            "max_sequence_length",
            "max_new_tokens",
            "checkpoint_every_steps",
        ):
            if getattr(self.pilot, name) < 1:
                raise ValueError(f"[config] {name} must be positive")
        if self.pilot.current_batch_size > data.train_examples:
            raise ValueError("[config] current batch exceeds the training pool")
        if self.pilot.reference_batch_size > data.reference_examples:
            raise ValueError(
                "[config] reference batch exceeds the first past-task pool"
            )
        if not 0 < self.pilot.base_accuracy_ceiling <= 1:
            raise ValueError("[config] base_accuracy_ceiling must be in (0, 1]")
        if min(self.lora.rank, self.lora.alpha, self.lora.last_n_layers) < 1:
            raise ValueError("[config] LoRA sizes must be positive")
        allowed = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if not self.lora.modules or set(self.lora.modules) - allowed:
            raise ValueError(
                "[config] LoRA targets must be text attention/MLP projections"
            )
        if self.optimization.learning_rate <= 0 or self.optimization.max_grad_norm <= 0:
            raise ValueError(
                "[config] learning rate and clipping norm must be positive"
            )
        if not 0 < self.optimization.replay_weight < 1:
            raise ValueError("[config] replay_weight must be in (0, 1)")

    def as_dict(self):
        return asdict(self)

    def fingerprint(self):
        return digest(self.as_dict())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
