import argparse
import gc
import importlib.metadata
import json
import logging
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from peft.tuners.lora.layer import Linear
from safetensors.torch import load_file, save_file
from torch.nn import functional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Qwen3ForCausalLM,
)

from native303_contract import (
    CONTRACT,
    ROOT,
    TOLERANCES,
    ReadBoundary,
    check_protocol,
    compare_generic,
    compare_native,
    compare_prefixes,
    digest,
    file_pin,
    grade,
    manifest_files,
    now,
    numeric_difference,
    read_json,
    require,
    source_base_pins,
    source_inputs,
    tensor_digest,
    validate_generic,
    validate_native,
    verify_execution,
    verify_export,
    verify_source_archive,
    write_json,
)

LOG = logging.getLogger("native303")


def event(name, **fields):
    LOG.info(
        "%s",
        json.dumps(
            {"event": name, "at": now(), "pid": os.getpid(), **fields}, allow_nan=False
        ),
    )


def base_values(model):
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    values = {}
    for kind, iterator in (
        ("parameter", base.named_parameters()),
        ("buffer", base.named_buffers()),
    ):
        for name, tensor in iterator:
            if "lora_" in name:
                continue
            key = f"{kind}:{name.replace('.base_layer.', '.')}"
            require(
                key not in values and not tensor.requires_grad and tensor.grad is None,
                "FROZEN_UNIQUE_BASE",
                key,
            )
            values[key] = tensor
    return values


def model_guard(model, device):
    values = list(model.parameters()) + list(model.buffers())
    require(
        values and not torch.is_autocast_enabled(torch.device(device).type),
        "NO_AUTOCAST",
    )
    require(
        all(str(value.device) == device for value in values), "SINGLE_DEVICE_RESIDENCY"
    )
    require(
        all(not value.requires_grad and value.grad is None for value in values),
        "ZERO_UPDATE_FROZEN_MODEL",
    )
    require(
        all(
            not value.is_floating_point() or value.dtype == torch.float32
            for value in values
        ),
        "FP32_MODEL",
    )
    require(all(bool(torch.isfinite(value).all()) for value in values), "FINITE_MODEL")
    require(
        all(not hasattr(module, "_hf_hook") for module in model.modules()),
        "NO_OFFLOAD_HOOKS",
    )
    return {
        "device": device,
        "dtype": "float32",
        "finite": True,
        "frozen": True,
        "tensor_count": len(values),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def ordinary_inventory(model):
    require(
        type(model) is Qwen3ForCausalLM and model.config.model_type == "qwen3",
        "ORDINARY_MODEL_CLASS",
    )
    require(
        not hasattr(model, "peft_config")
        and not getattr(model, "_hf_peft_config_loaded", False),
        "NO_RESIDENT_ADAPTERS",
    )
    classes = {}
    for name, module in model.named_modules():
        package = type(module).__module__
        require(
            package.startswith(
                ("torch.nn.", "transformers.models.qwen3.", "transformers.activations")
            ),
            "NO_CUSTOM_MODULES",
            [name, package],
        )
        require(
            not module._forward_hooks and not module._forward_pre_hooks,
            "NO_CUSTOM_FORWARD_HOOKS",
            name,
        )
        classes[name] = package + "." + type(module).__qualname__
    require(
        all(
            "lora_" not in key and ".base_layer." not in key
            for key in model.state_dict()
        ),
        "NO_ADAPTER_WEIGHT_KEYS",
    )
    return classes


def safe_merge_student(student, before, classes):
    require(
        isinstance(student, PeftModel) and set(student.peft_config) == {"learner"},
        "ONE_STUDENT_ADAPTER",
    )
    spec = student.peft_config["learner"]
    require(
        spec.r == 8
        and spec.lora_alpha == 16
        and set(spec.target_modules) == {"q_proj", "v_proj"}
        and spec.bias == "none"
        and spec.lora_dropout == 0,
        "RANK8_QV_CONTRACT",
    )
    require(
        not any(
            (
                spec.use_dora,
                spec.use_rslora,
                spec.fan_in_fan_out,
                spec.modules_to_save,
                spec.rank_pattern,
                spec.alpha_pattern,
                spec.layer_replication,
                spec.lora_bias,
                spec.target_parameters,
                spec.trainable_token_indices,
                spec.alora_invocation_tokens,
            )
        ),
        "NO_ADAPTER_VARIANTS",
    )
    base = student.get_base_model()
    targets = {}
    for name, module in base.named_modules():
        if isinstance(module, Linear):
            require(
                name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
                and not module.merged
                and module.active_adapters == ["learner"],
                "MERGE_TARGET",
                name,
            )
            require(
                set(module.lora_A) == set(module.lora_B) == {"learner"}
                and module.scaling["learner"] == 2,
                "ONE_FACTOR_PAIR",
                name,
            )
            require(
                type(module.base_layer) is torch.nn.Linear and not module.lora_variant,
                "PLAIN_LINEAR_MERGE",
                name,
            )
            targets[f"parameter:{name}.weight"] = (
                module.base_layer.weight.detach(),
                module.lora_A["learner"].weight.detach(),
                module.lora_B["learner"].weight.detach(),
                module.scaling["learner"],
            )
    require(len(targets) == 2 * base.config.num_hidden_layers, "ALL_QV_LAYERS_TARGETED")
    require(set(targets) <= set(before["tensors"]), "TARGET_BASE_NAMES")
    event("safe_merge_start", target_matrices=len(targets))
    with torch.no_grad():
        merged = student.merge_and_unload(safe_merge=True, adapter_names=["learner"])
    merged.requires_grad_(False).eval()
    require(ordinary_inventory(merged) == classes, "ORIGINAL_MODULE_TREE_RESTORED")
    after_values = base_values(merged)
    after = tensor_digest(after_values)
    require(set(after["tensors"]) == set(before["tensors"]), "MERGED_BASE_TENSOR_NAMES")
    untouched = [name for name in before["tensors"] if name not in targets]
    changed_untargeted = [
        name for name in untouched if before["tensors"][name] != after["tensors"][name]
    ]
    numerical = {}
    with torch.no_grad():
        for name, (original, a, b, scale) in targets.items():
            expected = original + (b @ a) * scale
            numerical[name] = numeric_difference(
                after_values[name], expected, TOLERANCES["merged_weights"]
            )
    audit = {
        "passed": not changed_untargeted
        and all(row["passed"] for row in numerical.values()),
        "target_matrices": len(targets),
        "untargeted_tensors": len(untouched),
        "changed_untargeted": changed_untargeted,
        "target_weight_comparisons": numerical,
        "before": before,
        "after": after,
        "ordinary_module_classes": classes,
        "equation": "FP32 W_export = W_original + 2*(B@A); PEFT safe_merge=True; no optimizer or backward.",
    }
    require(audit["passed"], "MERGED_WEIGHT_INVARIANTS", audit["changed_untargeted"])
    return merged, audit


def generation_config(tokenizer):
    require(tokenizer.eos_token_id is not None, "NATIVE_EOS_REQUIRED")
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=16,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        repetition_penalty=1.0,
    )


def native_panel(model, tokenizer, rows, label):
    records = []
    generation = generation_config(tokenizer)
    model.generation_config = generation
    if isinstance(model, PeftModel):
        model.get_base_model().generation_config = generation
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, row in enumerate(rows):
            prompt = tokenizer(row["prompt"], add_special_tokens=True).input_ids
            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            result = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=generation,
            )
            tokens = result[0, len(prompt) :].tolist()
            content = (
                tokens[:-1]
                if tokens and tokens[-1] == tokenizer.eos_token_id
                else tokens
            )
            body = tokenizer.decode(
                content, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            output = {
                "token_ids": tokens,
                "prompt_token_ids": prompt,
                "body_text": body,
                "raw_text": tokenizer.decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                **grade(
                    row,
                    tokens,
                    body,
                    tokenizer.eos_token_id,
                    tokenizer.all_special_ids,
                    16,
                ),
            }
            records.append(
                {
                    "id": row["id"],
                    "task": row["task"],
                    "group": row["group"],
                    "row_sha256": digest(row),
                    "generation": output,
                }
            )
            if (index + 1) % 32 == 0:
                event("native_progress", panel=label, rows=index + 1, total=len(rows))
    return records


def generic_panel(model, tokenizer, rows):
    encoded = []
    for row in rows:
        stripped = row["prompt"].rstrip()
        boundary = row["prompt"][len(stripped) :]
        prompt = tokenizer(stripped, add_special_tokens=True).input_ids
        require(0 < len(prompt) <= 768, "GENERIC_PROMPT_NO_TRUNCATION")
        for choice in row["choices"]:
            text = boundary + choice
            answer = tokenizer(text, add_special_tokens=False).input_ids
            require(bool(answer) and bool(text), "GENERIC_CONTINUATION")
            encoded.append((prompt + answer, len(answer), len(text)))
    scores = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for offset in range(0, len(encoded), 8):
            batch = encoded[offset : offset + 8]
            ids = torch.full(
                (len(batch), max(len(row[0]) for row in batch)),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            mask = torch.zeros_like(ids)
            for i, (sequence, _, _) in enumerate(batch):
                ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
                mask[i, : len(sequence)] = 1
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[
                :, :-1
            ]
            require(logits.dtype == torch.float32, "FP32_GENERIC_LOGITS")
            logps = functional.log_softmax(logits, dim=-1)
            totals = []
            for i, (sequence, length, _) in enumerate(batch):
                start = len(sequence) - length
                selected = (
                    logps[i, start - 1 : len(sequence) - 1]
                    .gather(-1, ids[i, start : len(sequence)].unsqueeze(-1))
                    .squeeze(-1)
                )
                totals.append(selected.sum())
            scores.extend(
                total / item[2]
                for total, item in zip(torch.stack(totals).tolist(), batch, strict=True)
            )
            del logits, logps
    records, start = [], 0
    for row in rows:
        subset = scores[start : start + len(row["choices"])]
        winner = max(range(len(subset)), key=subset.__getitem__)
        records.append(
            {
                "row_sha256": digest(row),
                "task": row["task"],
                "scores": subset,
                "choice_count": len(subset),
                "gold": row["gold_idx"],
                "prediction": winner,
                "correct": winner == row["gold_idx"],
            }
        )
        start += len(subset)
    validate_generic(rows, records)
    return records


def prefix_logits(model, tokenizer, rows, references):
    witnesses = {}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, (row, record) in enumerate(zip(rows, references, strict=True)):
            prompt = tokenizer(row["prompt"], add_special_tokens=True).input_ids
            require(
                prompt == record["generation"]["prompt_token_ids"],
                "PREFIX_PROMPT_TOKENS",
            )
            tokens = record["generation"]["token_ids"]
            cached, logits = None, []
            for step in range(len(tokens)):
                current = prompt if step == 0 else [tokens[step - 1]]
                ids = torch.tensor([current], dtype=torch.long, device=device)
                mask = torch.ones(
                    (1, len(prompt) + step), dtype=torch.long, device=device
                )
                output = model(
                    input_ids=ids,
                    attention_mask=mask,
                    past_key_values=cached,
                    use_cache=True,
                )
                cached = output.past_key_values
                require(output.logits.dtype == torch.float32, "FP32_PREFIX_LOGITS")
                logits.append(output.logits[0, -1].detach().cpu())
            witnesses[f"row_{index:04d}"] = torch.stack(logits)
    return witnesses


def probe_panel(model, tokenizer, inputs, label):
    native = native_panel(model, tokenizer, inputs["rows"]["train"], label)
    validate_native(inputs["rows"]["train"], native, inputs["tokens"])
    return {
        "train": native,
        "generic": generic_panel(model, tokenizer, inputs["rows"]["generic"]),
    }


def probe_comparison(panel, reference, tolerance):
    native = compare_native(panel["train"], reference["train"])
    generic = compare_generic(panel["generic"], reference["generic"], tolerance)
    return {
        "passed": native["passed"] and generic["passed"],
        "train": native,
        "generic": generic,
    }


def load_standard(path, device):
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    ordinary_inventory(model)
    model_guard(model, device)
    return model, tokenizer


def save_standard(model, tokenizer, path, shard_size):
    ordinary_inventory(model)
    model.config._name_or_path = ""
    model.generation_config = generation_config(tokenizer)
    model.save_pretrained(
        path, max_shard_size=shard_size, safe_serialization=True, save_peft_format=False
    )
    tokenizer.save_pretrained(path)
    require(
        (Path(path) / "model.safetensors.index.json").is_file(),
        "STANDARD_SHARDED_SAVE_REQUIRED",
    )


class Publication:
    def __init__(self, root):
        self.root = Path(root)
        self.final = self.root / "export"
        self.stage = None
        self.lock = self.root / ".publication.lock"
        self.acquired = False

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        require(not self.final.exists(), "EXPORT_ALREADY_EXISTS", str(self.final))
        with self.lock.open("x") as stream:
            self.acquired = True
            stream.write(
                json.dumps(
                    {"pid": os.getpid(), "hostname": socket.gethostname(), "at": now()}
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        self.stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))
        return self

    def promote(self, metadata):
        require(not self.final.exists(), "EXPORT_ALREADY_EXISTS", str(self.final))
        for path in self.stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        manifest = {
            **metadata,
            "contract": CONTRACT,
            "optimizer_updates": 0,
            "files": manifest_files(self.stage),
        }
        write_json(self.stage / "native303_manifest.json", manifest)
        pin = file_pin(self.stage / "native303_manifest.json")
        verified, audit = verify_export(self.stage, pin)
        require(verified == manifest, "MANIFEST_ROUNDTRIP")
        self.stage.rename(self.final)
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {"directory": str(self.final), "manifest": pin, **audit}

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired:
            self.lock.unlink()


def runtime():
    return {
        "at": now(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("peft", "transformers", "safetensors")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }


def current_execution(output, dispatch):
    execution = verify_execution(
        output, dispatch["task_id"], completed=False, config=dispatch
    )
    task = execution["task"]
    require(
        Path(task["code_dir"]).resolve() == ROOT
        and task["entrypoint"] == "native303_export.py"
        and task["gpus"] == 1
        and len(execution["gpus"]) == 1,
        "STANDALONE_DISPATCH",
    )
    require(
        task["source_sha256"] == ROOT.name == execution["source_sha256"],
        "STAGED_SOURCE_DIRECTORY",
    )
    archive_path = ROOT.with_suffix(".tar")
    archive = verify_source_archive(
        {
            "path": str(archive_path),
            "archive": file_pin(archive_path),
            "source_sha256": ROOT.name,
        }
    )
    for name in archive:
        require(
            archive[name] == file_pin(ROOT / name)["sha256"],
            "STAGED_IMPLEMENTATION_BYTES",
            name,
        )
    return {
        "execution": execution,
        "execution_pin": file_pin(Path(output) / "execution.json"),
        "source_archive": file_pin(archive_path),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }


def export_stage(design, dispatch, output, identity):
    inputs = source_inputs(design)
    base_pins = source_base_pins(design["source_base"])
    write_json(
        output / "input_manifest.json",
        {
            "source": inputs["source_pins"],
            "base": base_pins,
            "protocol": digest(design),
        },
    )
    require(
        os.getpid()
        not in (inputs["source_training_pid"], inputs["source_evaluation_pid"]),
        "FRESH_EXPORT_PID",
    )
    base, tokenizer = load_standard(design["source_base"]["local_path"], "cuda:0")
    require(
        tokenizer.eos_token_id == inputs["tokens"]["eos_token_id"]
        and tokenizer.all_special_ids == inputs["tokens"]["special_token_ids"],
        "EXACT_TOKEN_CONTRACT",
    )
    classes = ordinary_inventory(base)
    before = tensor_digest(base_values(base))
    require(
        before["tensor_sha256"] == design["base_tensor_sha256"],
        "ORIGINAL_BASE_TENSOR_HASH",
    )
    student = PeftModel.from_pretrained(
        base,
        design["adapter"]["path"],
        adapter_name="learner",
        is_trainable=False,
        local_files_only=True,
    )
    student.requires_grad_(False).eval()
    adapter = tensor_digest(
        get_peft_model_state_dict(
            student, adapter_name="learner", save_embedding_layers=False
        )
    )
    require(
        adapter["tensor_sha256"] == design["adapter"]["tensor_sha256"]
        and adapter["elements"] == 3833856
        and len(adapter["tensors"]) == 144,
        "EXACT_LOADED_STUDENT",
    )
    require(
        tensor_digest(base_values(student))["tensor_sha256"] == before["tensor_sha256"],
        "ADAPTER_LOAD_BASE_IMMUTABLE",
    )
    model_guard(student, "cuda:0")
    initial = probe_panel(student, tokenizer, inputs, "premerge_train24")
    premerge = probe_comparison(
        initial, inputs["reference"], TOLERANCES["source_scores"]
    )
    write_json(output / "premerge.json", {"panel": initial, "comparison": premerge})
    require(premerge["passed"], "SOURCE_PREREQUISITE_FAILED_BEFORE_MERGE")
    source_prefix = prefix_logits(
        student, tokenizer, inputs["rows"]["train"], inputs["reference"]["train"]
    )
    merged, weight_audit = safe_merge_student(student, before, classes)
    del student, base
    gc.collect()
    merged_panel = probe_panel(merged, tokenizer, inputs, "merged_train24")
    merged_prefix = prefix_logits(
        merged, tokenizer, inputs["rows"]["train"], inputs["reference"]["train"]
    )
    comparisons = {
        "source": premerge,
        "merged": probe_comparison(
            merged_panel, inputs["reference"], TOLERANCES["merged_scores"]
        ),
        "prefix_logits": compare_prefixes(
            merged_prefix,
            source_prefix,
            inputs["reference"]["train"],
            TOLERANCES["prefix_logits"],
        ),
    }
    write_json(
        output / "merge_parity.json",
        {"comparisons": comparisons, "panel": merged_panel},
    )
    write_json(output / "weight_audit.json", weight_audit)
    require(
        all(value["passed"] for value in comparisons.values()), "MERGED_PARITY_FAILED"
    )
    model_guard(merged, "cuda:0")
    with Publication(design["artifact_root"]) as transaction:
        event("standard_sharded_save", directory=str(transaction.stage))
        evidence = transaction.stage / "evidence"
        write_json(evidence / "inputs.json", inputs)
        write_json(evidence / "merged_probes.json", merged_panel)
        write_json(evidence / "merge_parity.json", comparisons)
        write_json(evidence / "weight_audit.json", weight_audit)
        save_file(source_prefix, evidence / "source_prefix_logits.safetensors")
        save_file(merged_prefix, evidence / "merged_prefix_logits.safetensors")
        save_standard(
            merged, tokenizer, transaction.stage / "model", design["max_shard_size"]
        )
        artifact = transaction.promote(
            {
                "created_at": now(),
                "identity": identity,
                "protocol_sha256": digest(design),
                "source_checkpoint": 384,
                "base_file_pins": base_pins,
                "ordinary_weight_digest": weight_audit["after"],
                "ordinary_module_classes": classes,
                "claim_boundary": design["claim_boundary"],
            }
        )
    receipt = {
        "status": "exported_pending_separate_job_parity",
        "contract": CONTRACT,
        "optimizer_updates": 0,
        "identity": identity,
        "pid": os.getpid(),
        "protocol_sha256": digest(design),
        "artifact": artifact,
        "source_adapter": design["adapter"],
        "source_base_tensor_sha256": before["tensor_sha256"],
        "probe_qualification": {
            key: value["passed"] for key, value in comparisons.items()
        },
        "teacher_model_loaded": False,
        "claim_boundary": design["claim_boundary"],
        "completed_at": now(),
    }
    write_json(output / "export_receipt.json", receipt)
    return receipt


def export_dependency(design, dispatch):
    root = Path(dispatch["export_run"])
    completed = verify_execution(root, design["export_task_id"])
    receipt = read_json(root / "export_receipt.json")
    require(
        receipt["status"] == "exported_pending_separate_job_parity"
        and receipt["optimizer_updates"] == 0
        and receipt["protocol_sha256"] == digest(design),
        "EXPORT_RECEIPT",
    )
    original = receipt["identity"]["execution"]
    for key in (
        "task_id",
        "attempt_id",
        "task",
        "task_sha256",
        "config_sha256",
        "source_sha256",
        "started_at",
        "pid",
    ):
        require(
            original[key] == completed[key], "EXPORT_COMPLETED_ATTEMPT_BINDING", key
        )
    require(
        receipt["artifact"]["directory"]
        == str(Path(design["artifact_root"]) / "export"),
        "EXPORT_DESTINATION",
    )
    return receipt, {
        "execution": completed,
        "receipt_pin": file_pin(root / "export_receipt.json"),
        "execution_pin": file_pin(root / "execution.json"),
    }


def reload_verified(root, pin, device):
    manifest, audit = verify_export(root, pin)
    event(
        "all_export_files_verified_before_model_load",
        shards=len(audit["shards"]),
        bytes=audit["tensor_payload_bytes"],
    )
    model, tokenizer = load_standard(Path(root) / "model", device)
    require(
        ordinary_inventory(model) == manifest["ordinary_module_classes"],
        "RELOADED_ORDINARY_MODULE_TREE",
    )
    require(
        tensor_digest(base_values(model)) == manifest["ordinary_weight_digest"],
        "RELOADED_EXACT_MODEL_WEIGHTS",
    )
    return model, tokenizer, manifest, audit


def evaluate_stage(design, dispatch, output, identity):
    receipt, dependency = export_dependency(design, dispatch)
    require(
        receipt["pid"] != os.getpid()
        and dependency["execution"]["attempt_id"] != identity["execution"]["attempt_id"]
        and dependency["execution"]["task_id"] != dispatch["task_id"],
        "SEPARATE_RELOAD_JOB_AND_PID",
    )
    root = Path(receipt["artifact"]["directory"])
    forbidden = [
        design["source_base"]["local_path"],
        *[source["directory"] for source in design["source_runs"].values()],
        *design["teacher_forbidden_roots"],
    ]
    boundary = ReadBoundary(forbidden, [root, ROOT, output])
    with boundary:
        model, tokenizer, manifest, audit = reload_verified(
            root, receipt["artifact"]["manifest"], "cuda:0"
        )
        require(
            manifest["identity"] == receipt["identity"]
            and manifest["protocol_sha256"] == digest(design),
            "MANIFEST_RECEIPT_BINDING",
        )
        inputs = read_json(root / "evidence/inputs.json")
        require(
            os.getpid()
            not in (inputs["source_training_pid"], inputs["source_evaluation_pid"]),
            "FRESH_NATIVE_RELOAD_PID",
        )
        require(
            tokenizer.eos_token_id == inputs["tokens"]["eos_token_id"]
            and tokenizer.all_special_ids == inputs["tokens"]["special_token_ids"],
            "RELOADED_TOKEN_CONTRACT",
        )
        panel = probe_panel(model, tokenizer, inputs, "reloaded_train24")
        fresh_prefix = prefix_logits(
            model, tokenizer, inputs["rows"]["train"], inputs["reference"]["train"]
        )
        save_file(fresh_prefix, output / "reloaded_prefix_logits.safetensors")
        comparisons = {
            "original_student": probe_comparison(
                panel, inputs["reference"], TOLERANCES["merged_scores"]
            ),
            "exported_student": probe_comparison(
                panel,
                read_json(root / "evidence/merged_probes.json"),
                TOLERANCES["source_scores"],
            ),
        }
        for name in ("source", "merged"):
            prior = load_file(
                root / f"evidence/{name}_prefix_logits.safetensors", device="cpu"
            )
            comparisons[name + "_prefix_logits"] = compare_prefixes(
                fresh_prefix,
                prior,
                inputs["reference"]["train"],
                TOLERANCES["prefix_logits"],
            )
        write_json(
            output / "reload_probe_parity.json",
            {"panel": panel, "comparisons": comparisons},
        )
        require(
            all(value["passed"] for value in comparisons.values()),
            "FRESH_RELOAD_PROBE_PARITY_FAILED",
        )
        panels, outcomes = {}, {}
        for split in ("test", "unused288"):
            panels[split] = native_panel(model, tokenizer, inputs["rows"][split], split)
            validate_native(inputs["rows"][split], panels[split], inputs["tokens"])
            outcomes[split] = compare_native(panels[split], inputs["reference"][split])
            write_json(
                output / f"{split}.json",
                {"records": panels[split], "comparison": outcomes[split]},
            )
        final_weights = tensor_digest(base_values(model))
        require(
            final_weights == manifest["ordinary_weight_digest"],
            "EVALUATION_ZERO_WEIGHT_UPDATES",
        )
    passed = all(item["passed"] for item in outcomes.values()) and not boundary.denied
    result = {
        "status": "completed_parity_verified"
        if passed
        else "completed_with_prediction_differences",
        "contract": CONTRACT,
        "optimizer_updates": 0,
        "identity": identity,
        "pid": os.getpid(),
        "export_dependency": dependency,
        "artifact": audit,
        "protocol_sha256": digest(design),
        "probe_comparisons": comparisons,
        "native_panels": outcomes,
        "passed": passed,
        "read_boundary": boundary.proof(),
        "model_loader": "AutoModelForCausalLM.from_pretrained(ordinary_export/model), local_files_only=True, trust_remote_code=False, FP32",
        "teacher_model_loaded": False,
        "adapter_model_loaded": False,
        "teacher_or_adapter_artifact_read": bool(boundary.denied),
        "reloaded_weight_tensor_sha256": final_weights["tensor_sha256"],
        "prefix_witness": file_pin(output / "reloaded_prefix_logits.safetensors"),
        "claim_boundary": design["claim_boundary"],
        "completed_at": now(),
    }
    write_json(output / "result.json", result)
    require(passed, "FINAL_NATIVE_PREDICTION_DIFFERENCES")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    dispatch = read_json(args.config)
    design = read_json(ROOT / dispatch["protocol"])
    check_protocol(design)
    require(digest(design) == dispatch["protocol_sha256"], "DISPATCH_PROTOCOL_HASH")
    require(
        dispatch["stage"] in ("export", "evaluate")
        and dispatch["task_id"] == design[dispatch["stage"] + "_task_id"],
        "STAGE_TASK_ID",
    )
    if args.prepare_only:
        if dispatch["stage"] == "export":
            inputs = source_inputs(design)
            proof = {
                "source_pins": inputs["source_pins"],
                "base": source_base_pins(design["source_base"]),
            }
            require(
                not (Path(design["artifact_root"]) / "export").exists(),
                "EXPORT_ALREADY_EXISTS",
            )
        else:
            receipt, dependency = export_dependency(design, dispatch)
            _, audit = verify_export(
                receipt["artifact"]["directory"], receipt["artifact"]["manifest"]
            )
            proof = {"dependency": dependency, "artifact": audit}
        print(
            json.dumps(
                {
                    "status": "prepared_without_model_load_or_writes",
                    "optimizer_updates": 0,
                    "protocol_sha256": digest(design),
                    "proof": proof,
                },
                allow_nan=False,
            )
        )
        return
    require(args.output_dir is not None, "OUTPUT_DIRECTORY_REQUIRED")
    output = args.output_dir.resolve()
    require(
        output.is_dir()
        and not any(
            (output / name).exists()
            for name in (
                "started.json",
                "export_receipt.json",
                "result.json",
                "failure.json",
            )
        ),
        "NEW_RUN_OUTPUT_REQUIRED",
    )
    identity = current_execution(output, dispatch)
    write_json(
        output / "started.json",
        {
            "at": now(),
            "identity": identity,
            "dispatch": dispatch,
            "optimizer_updates": 0,
        },
    )
    try:
        observed = runtime()
        write_json(output / "runtime.json", observed)
        require(
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and torch.version.hip is not None,
            "ONE_ROCM_GPU_REQUIRED",
        )
        require(observed["packages"] == design["packages"], "PINNED_PACKAGES")
        require(observed["torch"] == design["worker_torch"], "PINNED_WORKER_TORCH")
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")
        if dispatch["stage"] == "export":
            result = export_stage(design, dispatch, output, identity)
        else:
            result = evaluate_stage(design, dispatch, output, identity)
        event("stage_finished", status=result["status"])
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_no_success_claim",
                "at": now(),
                "optimizer_updates": 0,
                "pid": os.getpid(),
                "exception": type(error).__name__,
                "message": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
