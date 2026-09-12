import argparse
import gc
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

logger = logging.getLogger("train-probe")


def load_model(path):
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, dest="output")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model_path = Path(config["model_path"])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [train-probe] %(message)s"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    assert torch.version.hip and torch.cuda.device_count() == 1
    properties = torch.cuda.get_device_properties(0)
    assert properties.gcnArchName.split(":")[0] == "gfx950"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = load_model(model_path)
    targets = [
        name
        for name, module in base.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith((".q_proj", ".v_proj"))
        and ".visual." not in name
    ]
    assert targets
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    before = [parameter.detach().cpu().clone() for parameter in trainable]
    frozen = next(
        parameter for parameter in model.parameters() if not parameter.requires_grad
    )
    frozen_before = frozen.detach().flatten()[:1024].cpu().clone()
    messages = [
        {
            "role": "user",
            "content": "In this training fixture, the token associated with the invented city Velnara is amber. What is the token? Answer with only the token.",
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer("amber" + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    inputs = torch.tensor([prompt_ids + answer_ids], device="cuda")
    labels = inputs.clone()
    labels[:, : len(prompt_ids)] = -100
    optimizer = torch.optim.AdamW(trainable, lr=3e-4)
    losses = []
    start = time.perf_counter()
    model.train()
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm) and norm.item() > 0
        optimizer.step()
        losses.append(float(loss.detach()))
        logger.info("step=%s loss=%.6f gradient_norm=%.6f", step, losses[-1], norm)
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - start
    delta = (
        sum(
            float((parameter.detach().cpu() - prior).square().sum())
            for parameter, prior in zip(trainable, before)
        )
        ** 0.5
    )
    assert delta > 0
    torch.testing.assert_close(
        frozen.detach().flatten()[:1024].cpu(), frozen_before, rtol=0, atol=0
    )
    model.eval()
    with torch.inference_mode():
        reference = (
            model(input_ids=inputs, use_cache=False, logits_to_keep=1)
            .logits.float()
            .cpu()
        )
    checkpoint = args.output / "adapter"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    del model, base, trainable, optimizer, frozen, loss
    gc.collect()
    torch.cuda.empty_cache()
    restored = PeftModel.from_pretrained(load_model(model_path), checkpoint).eval()
    with torch.inference_mode():
        after = (
            restored(input_ids=inputs, use_cache=False, logits_to_keep=1)
            .logits.float()
            .cpu()
        )
    torch.testing.assert_close(after, reference, rtol=0, atol=0)
    receipt = {
        "kind": "training_runtime_qualification",
        "research_result": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "gpu": str(properties),
        "target_modules": targets,
        "losses": losses,
        "parameter_delta_l2": delta,
        "frozen_parameter_sample_unchanged": True,
        "checkpoint_reload_max_abs_logit_error": float((after - reference).abs().max()),
        "optimizer_updates": len(losses),
        "update_seconds": update_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "adapter_sha256": hashlib.sha256(
            (checkpoint / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    logger.info("Training and exact checkpoint reload passed: %s", json.dumps(receipt))


if __name__ == "__main__":
    main()
