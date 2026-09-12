import argparse
import inspect
import json
import logging
import time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [training-qualification] %(message)s"
    )
    if torch.cuda.device_count() != 1 or not torch.version.hip:
        raise RuntimeError("Expected exactly one isolated ROCm GPU")
    torch.manual_seed(config["seed"])
    path = config["model_path"]
    source_config = AutoConfig.from_pretrained(path, local_files_only=True)
    model_class = (
        Qwen3_5ForConditionalGeneration
        if source_config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    logging.info(
        "Loader %s; PEFT API %s",
        model_class.__name__,
        inspect.signature(get_peft_model),
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )
    model.train()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for parameter in parameters)
    if not 0 < trainable < 100_000_000:
        raise RuntimeError(f"Unexpected trainable parameter count: {trainable}")
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    prompts = [
        [
            {"role": "user", "content": "Return only the result of 17 + 25."},
            {"role": "assistant", "content": "42"},
        ],
        [
            {"role": "user", "content": "Reverse this sequence: red blue green."},
            {"role": "assistant", "content": "green blue red"},
        ],
    ]
    texts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, enable_thinking=False)
        for prompt in prompts
    ]
    batch = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to("cuda:0")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    before = [parameter.detach().float().cpu().clone() for parameter in parameters]
    steps = []
    for step in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError(f"Invalid gradient norm: {gradient_norm}")
        optimizer.step()
        torch.cuda.synchronize()
        measured = {
            "step": step,
            "loss": loss.item(),
            "gradient_norm": gradient_norm.item(),
            "seconds": time.perf_counter() - start,
        }
        logging.info("%s", json.dumps(measured))
        steps.append(measured)
    delta_squared = sum(
        (parameter.detach().float().cpu() - initial).square().sum().item()
        for parameter, initial in zip(parameters, before, strict=True)
    )
    if delta_squared <= 0:
        raise RuntimeError("Optimizer produced no parameter change")
    result = {
        "status": "passed",
        "kind": "runtime_qualification",
        "model_id": config["model_id"],
        "revision": config["revision"],
        "model_path": path,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "trainable_parameters": trainable,
        "steps": steps,
        "parameter_delta_l2": delta_squared**0.5,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "gpu_count": torch.cuda.device_count(),
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    logging.info("Training qualification passed")


if __name__ == "__main__":
    main()
