import argparse
import gc
import hashlib
import inspect
import json
import logging
import os
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

logger = logging.getLogger(__name__)


def check_gpus(expected_gpus):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU runtime is unavailable")
    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} GPUs, found {count}")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    properties = torch.cuda.get_device_properties(rank)
    logger.info("GPU %s inventory: %s", rank, properties)
    if properties.gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError(
            f"Unexpected GPU architecture on rank {rank}: {properties.gcnArchName}"
        )
    if properties.total_memory < 280 * 1024**3:
        raise RuntimeError(
            f"Unexpected GPU memory on rank {rank}: {properties.total_memory}"
        )
    torch.manual_seed(42)
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    product = a @ b
    torch.testing.assert_close(
        product.float().cpu(), a.float().cpu() @ b.float().cpu(), rtol=0.01, atol=0.25
    )
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=180), device_id=torch.device("cuda", rank)
    )
    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)
    expected_sum = expected_gpus * (expected_gpus + 1) / 2
    if value.item() != expected_sum:
        raise RuntimeError(
            f"RCCL all-reduce returned {value.item()}, expected {expected_sum}"
        )
    logger.info(
        "GPU %s passed bf16 matmul and RCCL all-reduce: %s", rank, properties.name
    )
    dist.barrier()
    dist.destroy_process_group()
    del a, b, product, value
    torch.cuda.empty_cache()
    return rank


def infer(model_entry):
    model_path = Path(model_entry["path"])
    logger.info("Loading %s from local files", model_entry["id"])
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError(f"Model is not entirely GPU resident: {model_entry['id']}")
    load_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    samples = []
    prompts = [
        "What is 17 + 25? Answer with only the number.",
        "Explain in one sentence what it means for a model to learn by updating its weights.",
    ]
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda:0")
            logits = model(**inputs, logits_to_keep=1).logits
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite logits for {model_entry['id']}")
            del logits
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            generated = output[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not text or (index == 0 and text != "42"):
                raise RuntimeError(
                    f"Inference check failed for {model_entry['id']}: {text!r}"
                )
            sample = {
                "prompt": prompt,
                "output": text,
                "new_tokens": generated.numel(),
                "seconds": elapsed,
            }
            samples.append(sample)
            logger.info(
                "Inference passed for %s: %s", model_entry["id"], json.dumps(sample)
            )
    result = {
        **model_entry,
        "load_seconds": load_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "dtype": str(model.dtype),
        "device": str(model.device),
        "attention": model.config._attn_implementation,
        "samples": samples,
    }
    del model, tokenizer, inputs, output, generated
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [compute-smoke] %(message)s"
    )
    rank = check_gpus(args.expected_gpus)
    if rank:
        return
    manifest = Path("/code/models.json")
    downloads = json.loads(Path("/mnt/shared/cl-smoke/downloads.json").read_text())
    if (
        downloads["manifest_sha256"]
        != hashlib.sha256(manifest.read_bytes()).hexdigest()
    ):
        raise RuntimeError("Cached models do not match the requested manifest")
    logger.info(
        "Model SDK: %s",
        inspect.signature(Qwen3_5ForConditionalGeneration.from_pretrained),
    )
    result = {
        "completed_at": None,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "rocm": torch.version.hip,
        "gpus": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "architecture": torch.cuda.get_device_properties(i).gcnArchName,
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(args.expected_gpus)
        ],
        "bf16_matmul": "passed",
        "rccl_all_reduce": "passed",
        "models": [infer(entry) for entry in downloads["models"]],
    }
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps({"event": "compute_smoke_passed", "receipt": str(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
