import inspect
from contextlib import contextmanager
from importlib.metadata import version

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

from config import RUNTIME
from objectives import completion_mask, response_logits
from tasks import prompt

LORA_TARGETS = {
    "qwen3_5": r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    "qwen3": r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
}


class StopRequested(Exception):
    pass


class StopFlag(StoppingCriteria):
    def __init__(self):
        self.signum = None

    def request(self, signum, frame):
        self.signum = signum

    def check(self):
        if self.signum is not None:
            raise StopRequested(f"Received signal {self.signum}")

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.signum is not None, device=input_ids.device)


@contextmanager
def policy_mode(model, privileged=False):
    was_training = model.training
    model.eval()
    try:
        if privileged:
            with model.disable_adapter():
                yield
        else:
            yield
    finally:
        model.train(was_training)


def attach_adapter(base, config, family):
    if family not in LORA_TARGETS:
        raise ValueError(f"Unsupported cached model family: {family}")
    model = get_peft_model(
        base,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS[family],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Optimizer isolation failed: only LoRA parameters may be trainable")
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
        raise RuntimeError("Nonzero dropout would confound teacher and rollout comparisons")
    return model


def load_model(config):
    versions = {name: version(name) for name in RUNTIME}
    mismatches = {
        name: actual for name, actual in versions.items() if actual.split("+")[0] != RUNTIME[name]
    }
    if mismatches:
        raise RuntimeError(f"Runtime dependency mismatch: {mismatches}; expected {RUNTIME}")
    if config.require_rocm:
        if (
            not torch.cuda.is_available()
            or not torch.version.hip
            or not torch.version.hip.startswith("7.2")
        ):
            raise RuntimeError("The experiment requires the existing ROCm 7.2 GPU runtime")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        if (
            properties.gcnArchName.split(":")[0] != "gfx950"
            or properties.total_memory < 280 * 1024**3
        ):
            raise RuntimeError("The selected lane is not the expected MI355X with >=280 GiB")
    torch.manual_seed(config.seed)
    native_config = AutoConfig.from_pretrained(config.model_path, local_files_only=True)
    family = native_config.model_type
    if family not in LORA_TARGETS:
        raise ValueError(f"Unsupported cached model type: {family}")
    loader = Qwen3_5ForConditionalGeneration if family == "qwen3_5" else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    base = loader.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=getattr(torch, config.dtype),
        device_map=config.device,
        attn_implementation="sdpa",
    )
    model = attach_adapter(base, config, family)
    if any(p.device != torch.device(config.device) for p in model.parameters()):
        raise RuntimeError("The model is not entirely resident on the selected lane device")
    runtime = {
        "versions": versions,
        "rocm": torch.version.hip,
        "device": config.device,
        "device_name": torch.cuda.get_device_name(0) if config.device.startswith("cuda") else "CPU",
        "model_type": family,
        "attention": "sdpa",
        "dtype": config.dtype,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "base_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "sdk": {
            "forward": str(inspect.signature(base.forward)),
            "disable_adapter": str(inspect.signature(model.disable_adapter)),
        },
        "determinism": "Seeded sampling and zero dropout; bitwise ROCm determinism is not asserted.",
    }
    return model, tokenizer, runtime


class ModelIO:
    def __init__(self, model, tokenizer, config, data, stop):
        self.model, self.tokenizer, self.config, self.data, self.stop = (
            model,
            tokenizer,
            config,
            data,
            stop,
        )
        self.device = torch.device(config.device)
        eos = model.generation_config.eos_token_id
        self.eos_ids = tuple(eos if isinstance(eos, list) else [eos])
        if (
            not self.eos_ids
            or any(x is None for x in self.eos_ids)
            or tokenizer.eos_token_id is None
        ):
            raise ValueError("The native tokenizer and model must define an EOS token")
        self.model_eos_ids = self.eos_ids
        self.eos_ids = tuple(dict.fromkeys((tokenizer.eos_token_id, *self.model_eos_ids)))
        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def prefix(self, example, context="examples", privileged=False, instruction_variant="current"):
        encoded = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt(example, self.data, context, privileged, instruction_variant),
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"][0]
        if len(encoded) > self.config.max_prompt_tokens:
            raise ValueError(f"Prompt exceeds max_prompt_tokens: {example.key}, {len(encoded)}")
        return encoded.to(self.device)

    def oracle_tokens(self, example):
        answer = self.tokenizer.encode(example.answer, add_special_tokens=False)
        if not answer or any(x in self.tokenizer.all_special_ids for x in answer):
            raise ValueError("Oracle answer tokenization is empty or contains special tokens")
        return torch.tensor(answer + [self.tokenizer.eos_token_id], device=self.device)

    def generate(self, prefix, privileged, sample, seed, max_tokens):
        self.stop.check()
        generation = GenerationConfig(
            do_sample=sample,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_tokens,
            eos_token_id=list(self.eos_ids),
            pad_token_id=self.pad_id,
            use_cache=True,
            repetition_penalty=1.0,
        )
        devices = [0] if self.device.type == "cuda" else []
        with (
            torch.random.fork_rng(devices=devices),
            torch.no_grad(),
            policy_mode(self.model, privileged),
        ):
            torch.manual_seed(seed)
            generated = self.model.generate(
                input_ids=prefix.unsqueeze(0),
                attention_mask=torch.ones_like(prefix).unsqueeze(0),
                generation_config=generation,
                stopping_criteria=StoppingCriteriaList([self.stop]),
            )[0, len(prefix) :].detach()
        self.stop.check()
        response = generated[completion_mask(generated, self.eos_ids, None)]
        if not len(response):
            raise RuntimeError("Generation produced no continuation tokens")
        return response, len(generated)

    def decode(self, completion):
        body = completion[:-1] if int(completion[-1]) in self.eos_ids else completion
        return self.tokenizer.decode(body, skip_special_tokens=False).strip()

    def gold_logprob(self, prefix, example, privileged):
        gold = self.oracle_tokens(example)
        with torch.no_grad(), policy_mode(self.model, privileged):
            logits = response_logits(self.model, prefix, gold).float().log_softmax(-1)
            selected = logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(selected).all():
            raise FloatingPointError(f"Non-finite gold log probability: {example.key}")
        return {
            "gold_answer_logprob": float(selected[:-1].sum()),
            "gold_complete_logprob": float(selected.sum()),
            "gold_answer_tokens": len(gold) - 1,
        }
