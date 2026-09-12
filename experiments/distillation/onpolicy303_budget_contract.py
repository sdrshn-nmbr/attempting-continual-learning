import copy
import importlib.metadata
import json
import math
import os
import re
import sys
from pathlib import Path

import torch

from choice_consolidation import adapter_state, tensor_hash
from choice_contract import ROOT, digest, file_hash, training_schedule
from onpolicy303 import new_optimizer, optimizer_clocks
from onpolicy303_contract import retention_prediction
from onpolicy303_contract import validate_design as validate_original_design

CONTRACT = "onpolicy303_cached_controls_384_plus_21_post_observation"
METHODS = ("cached_teacher_sft", "cached_teacher_kl")
START = 384
ADDITIONAL = 21
FINAL = 405
ORIGINAL_TOKENS = 10752
FINAL_TOKENS = 11340
OWNPREFIX_TOKENS = 11314
SOURCE_ARCHIVE = "85c375eef798c920ef3c89d3fbb08b4b55bcb6481fe281b0403bcc277cbc3a9b"
PARAMETER = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.weight")


def require(condition, detail):
    if not condition:
        raise ValueError(f"ONPOLICY303_BUDGET {detail}")


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def source_hashes(design):
    return {name: file_hash(ROOT / name) for name in design["scientific_files_sha256"]}


def runtime_versions():
    return {"torch_module_version": str(torch.__version__),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")}}


def verify_runtime_versions(expected, actual):
    require(actual == expected, f"runtime versions changed: expected={expected!r} actual={actual!r}")
    return actual


def validate_design(design):
    require(design["contract"] == CONTRACT, "contract")
    require(source_hashes(design) == design["scientific_files_sha256"], "scientific source changed")
    original_spec = design["original_design"]
    path = ROOT / original_spec["path"]
    require(file_hash(path) == original_spec["file_sha256"], "original design file")
    original = json.loads(path.read_text())
    require(digest(original) == original_spec["payload_sha256"], "original design payload")
    context = validate_original_design(original)
    require(design["methods"] == list(METHODS), "only cached controls allowed")
    require(design["training"] == original["training"], "original optimizer and loss recipe")
    require(design["evaluation"] == original["evaluation"], "unchanged evaluation")
    require(design["start_update"] == START and design["additional_updates"] == ADDITIONAL and design["final_update"] == FINAL, "fixed continuation budget")
    require(design["tokens"] == {"cached_at384": ORIGINAL_TOKENS, "additional": 588, "cached_at405": FINAL_TOKENS, "ownprefix_at384": OWNPREFIX_TOKENS, "excess_over_ownprefix": 26}, "fixed token accounting")
    require(design["native_export"] == {"method": "onpolicy_kl", "checkpoint": START, "selection_changed_by_control": False}, "native export remains ownprefix384")
    require(set(design["sources"]) == set(METHODS), "source set")
    require(design["teacher"] == original["teacher"] and design["teacher_checkpoint"]["tensor_sha256"] == original["teacher"]["tensor_sha256"], "unchanged coverage expert")
    require(design["schedule"]["mode"] == "next_epoch_prefix", "sealed next-epoch continuation")
    return original, context


def continuation_schedule(protocol, rows, original_ids):
    require(len(rows) == 384, "same original 384 TRAIN rows")
    recipe = protocol["training"]
    require(recipe["epochs"] == 4 and recipe["updates_per_arm"] == START and recipe["examples_per_update"] == 4, "original schedule recipe")
    original = training_schedule(protocol, rows)
    require([[rows[i]["id"] for i in batch] for batch in original] == original_ids, "original384 batch IDs changed")
    expanded = copy.deepcopy(protocol)
    epochs = math.ceil(FINAL * recipe["examples_per_update"] / len(rows))
    expanded["training"].update(epochs=epochs, updates_per_arm=epochs * len(rows) // 4, example_exposures_per_arm=epochs * len(rows))
    full = training_schedule(expanded, rows)[:FINAL]
    require(full[:START] == original, "extended384 prefix changed")
    require(len(full) == FINAL and all(len(batch) == 4 for batch in full), "405 four-row batches")
    return full


def schedule_receipt(protocol, rows, original_ids):
    indices = continuation_schedule(protocol, rows, original_ids)
    ids = [[rows[i]["id"] for i in batch] for batch in indices]
    return {
        "mode": "next_epoch_prefix",
        "recipe": "Frozen training_schedule generates five full epochs (480 batches); retain its first405. The first384 equal the original schedule exactly. Execute only the final21 batches from epoch seed order_seed+4.",
        "order_seed": protocol["training"]["order_seed"],
        "continuation_epoch_seed": protocol["training"]["order_seed"] + 4,
        "original_prefix_equal": True,
        "original_batch_ids_sha256": digest(ids[:START]),
        "full405_batch_ids_sha256": digest(ids),
        "continuation_indices": indices[START:],
        "continuation_batch_ids": ids[START:],
        "continuation_batch_ids_sha256": digest(ids[START:]),
        "repeated_original_prefix": False,
    }


def token_accounting(cache, schedule, original_total):
    rows = cache["rows"]
    require(cache["split"] == "train" and len(rows) == 384, "cached TRAIN population")
    require(len(schedule) == FINAL and all(len(batch) == 4 for batch in schedule), "accounting schedule")
    require(all(len(row["response_token_ids"]) == 7 for row in rows), "unchanged seven-token teacher cache")
    counts = [sum(len(rows[i]["response_token_ids"]) for i in batch) for batch in schedule]
    require(sum(counts[:START]) == original_total == ORIGINAL_TOKENS, "original10752 loss tokens")
    require(sum(counts[START:]) == 588 and sum(counts) == FINAL_TOKENS, "continuation11340 loss tokens")
    return {"original_updates": START, "additional_updates": ADDITIONAL, "total_updates": FINAL, "additional_row_exposures": 84, "total_row_exposures": 1620, "unique_training_rows": 384, "original_loss_tokens": original_total, "additional_loss_tokens": sum(counts[START:]), "total_loss_tokens": sum(counts), "ownprefix384_loss_tokens": OWNPREFIX_TOKENS, "excess_over_ownprefix": sum(counts) - OWNPREFIX_TOKENS, "loss_tokens_by_continuation_update": counts[START:], "exact_token_update_or_flop_match": False}


def mapping_from_adapter(tensors):
    parsed = []
    for key, tensor in tensors.items():
        match = PARAMETER.fullmatch(key)
        require(match is not None and tensor.dtype == torch.float32 and tensor.ndim == 2, f"adapter parameter shape/name {key}")
        layer, projection, factor = match.groups()
        require(tensor.shape[0 if factor == "A" else 1] == 8, "rank8 adapter only")
        parsed.append(((int(layer), ("q_proj", "v_proj").index(projection), ("A", "B").index(factor)), key, tensor))
    parsed.sort(key=lambda entry: entry[0])
    layers = {entry[0][0] for entry in parsed}
    require(layers == set(range(len(layers))) and len(parsed) == 4 * len(layers), "complete q/v A/B layer ordering")
    return [{"optimizer_id": index, "parameter_name": key.removesuffix(".weight") + ".learner.weight", "adapter_key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel()} for index, (_, key, tensor) in enumerate(parsed)]


def model_mapping(model):
    actual = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(".learner.weight" in name and parameter.dtype == torch.float32, f"unexpected trainable {name}")
            actual.append({"optimizer_id": len(actual), "parameter_name": name, "adapter_key": name.replace(".learner.weight", ".weight"), "shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": parameter.numel()})
    require(actual == mapping_from_adapter(adapter_state(model)), "factory parameter order differs from sealed q/v ordering")
    return actual


def optimizer_options(state):
    require(len(state["param_groups"]) == 1, "one original AdamW parameter group")
    return normalized({key: value for key, value in state["param_groups"][0].items() if key != "params"})


def optimizer_tensor_hash(state):
    return tensor_hash({f"{index}/{key}": value for index, slot in state["state"].items() for key, value in slot.items()})


def validate_optimizer(state, mapping, clock, options):
    require(set(state) == {"state", "param_groups"}, "optimizer state schema")
    require(optimizer_options(state) == normalized(options), "original AdamW options changed")
    ids = state["param_groups"][0]["params"]
    require(ids == list(range(len(mapping))) and set(state["state"]) == set(ids), "optimizer mapping IDs")
    require([m["optimizer_id"] for m in mapping] == ids and len({m["parameter_name"] for m in mapping}) == len(ids), "optimizer mapping names")
    for index, expected in enumerate(mapping):
        slot = state["state"][index]
        require(set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "complete AdamW moments")
        step = slot["step"]
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and step.dtype == torch.float32 and bool(torch.isfinite(step).all()) and float(step) == clock, "AdamW clock")
        for name in ("exp_avg", "exp_avg_sq"):
            value = slot[name]
            require(isinstance(value, torch.Tensor) and list(value.shape) == expected["shape"] and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"AdamW moment {index}/{name}")
        require(bool((slot["exp_avg_sq"] >= 0).all()), "negative AdamW second moment")
    return {"clock": clock, "parameter_states": len(ids), "parameter_order_sha256": digest(mapping), "tensor_sha256": optimizer_tensor_hash(state), "options": optimizer_options(state), "all_moments_finite": True}


def restore_optimizer(model, recipe, saved, mapping, options, expected_hash):
    actual = model_mapping(model)
    require(actual == mapping, "saved optimizer parameter order does not match learner")
    before = validate_optimizer(saved, mapping, START, options)
    require(before["tensor_sha256"] == expected_hash, "saved AdamW tensor hash")
    optimizer, parameters = new_optimizer(model, recipe)
    require(optimizer_options(optimizer.state_dict()) == normalized(options), "runtime AdamW defaults differ from source")
    optimizer.load_state_dict(copy.deepcopy(saved))
    after = validate_optimizer(optimizer.state_dict(), mapping, START, options)
    require(after == before, "restored AdamW moments differ")
    require(optimizer_clocks(optimizer, parameters) == {"minimum": START, "maximum": START, "parameters_with_state": len(mapping)}, "restored optimizer clock mismatch")
    for parameter in parameters:
        require(all(optimizer.state[parameter][key].device == parameter.device for key in ("exp_avg", "exp_avg_sq")), "restored moment placement")
    return optimizer, parameters, before


def verify_source_files(root, descriptor):
    root = Path(root)
    for relative, checksum in descriptor["files_sha256"].items():
        require(file_hash(root / relative) == checksum, f"original artifact changed {relative}")
    receipt = json.loads((root / "training.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    method = descriptor["method"]
    require(method in METHODS and receipt["method"] == method, "source method")
    require(execution["status"] == "completed" and execution["exit_code"] == 0 and execution["timed_out"] is False and execution["source_sha256"] == SOURCE_ARCHIVE and execution["task_id"] == descriptor["task_id"], "completed original source")
    require(receipt["arm"]["selected_checkpoint"] == START and receipt["arm"]["updates"] == START and receipt["arm"]["loss_token_exposures"] == ORIGINAL_TOKENS, "original checkpoint384")
    require(receipt["teacher_targets_all_equal_oracle"] and receipt["teacher_optimizer_updates"] == 0, "original qualified cache")
    require(receipt["arm"]["checkpoints"]["384"] == descriptor["checkpoint"], "checkpoint384 binding")
    expected_path = Path(descriptor["run_dir"]) / method / "checkpoint384/learner"
    require(Path(descriptor["checkpoint"]["adapter"]["path"]) == expected_path, "original adapter path")
    return receipt


class EvaluationReadBan:
    def __init__(self, forbidden_roots, allowed_model_roots):
        self.forbidden = tuple(Path(p).resolve() for p in forbidden_roots)
        self.allowed = tuple(Path(p).resolve() for p in allowed_model_roots)
        self.denied = []
        self.weight_reads = []

    def check(self, path):
        if isinstance(path, int):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        forbidden = any(resolved == p or p in resolved.parents for p in self.forbidden)
        weight = bool({Path(os.fsdecode(path)).suffix, resolved.suffix} & {".safetensors", ".bin", ".pt", ".pth"})
        allowed_weight = any(resolved == p or p in resolved.parents for p in self.allowed)
        if forbidden or (weight and not allowed_weight):
            self.denied.append(str(resolved))
            raise PermissionError(f"ONPOLICY303_BUDGET_EVALUATOR_READ_BAN {resolved}")
        if weight:
            self.weight_reads.append(str(resolved))

    def audit(self, event, args):
        if event == "open":
            self.check(args[0])

    def install(self):
        sys.addaudithook(self.audit)
        return self

    def receipt(self):
        return {"scope": "Python audited file opens plus explicit approval of every model-loader path; not an OS syscall sandbox.", "forbidden_roots": [str(p) for p in self.forbidden], "allowed_model_roots": [str(p) for p in self.allowed], "denied_reads": list(self.denied), "observed_weight_paths": sorted(set(self.weight_reads)), "installed_before_model_load": True}


def require_teacher_absent(model):
    require(set(model.peft_config) == {"learner"}, "evaluator only learner adapter")
    config = model.peft_config["learner"]
    require(config.r == 8 and config.lora_alpha == 16 and set(config.target_modules) == {"q_proj", "v_proj"}, "evaluator rank8 q/v")
    require(not any(parameter.requires_grad for parameter in model.parameters()), "evaluator is frozen")
    return {"teacher_model_loaded": False, "active_adapters": ["learner"], "learner_rank": 8, "optimizer_updates": 0}


def verify_child_identity(result, training_pid, child_pid, training_sha256, token):
    require(result["status"] == "completed" and result["pid"] not in {child_pid, training_pid} and child_pid != training_pid, "fresh evaluator PID")
    require(result["training_pid"] == training_pid and result["parent_pid"] == child_pid and result["training_sha256"] == training_sha256 and result["child_token"] == token, "evaluator process binding through launched uv PID")
    require(result["teacher_model_loaded"] is False and result["optimizer_updates"] == 0 and result["train24_reload_parity"] is True, "teacher-absent evaluation receipt")


def verify_reload(native, expected_native, generic, expected_generic, retention):
    require(len(native) == len(expected_native) == 24, "native TRAIN24 prerequisite count")
    require(native == expected_native, "native TRAIN24 reload parity before updates/evaluation")
    require(len(generic) == len(expected_generic) == len(retention) == 128, "generic128 prerequisite count")
    for row, record in zip(retention, expected_generic, strict=True):
        require(record == retention_prediction(row, record["scores"]), "generic128 saved scorer binding")
    require(generic == expected_generic, "generic128 reload parity before updates/evaluation")
    return {"train24_exact_native_parity": True, "generic128_exact_score_parity": True}
