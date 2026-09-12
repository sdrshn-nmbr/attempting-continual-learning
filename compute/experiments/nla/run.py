import argparse
import hashlib
import inspect
import logging
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import Qwen2ForCausalLM

from analysis import calibration_diagnostics, fidelity_gate, forgetting_prediction, readout_drift, stage_behavior
from native import (
    NativeAR,
    NativeAV,
    Sidecar,
    chat_query,
    completion_ids,
    extract_activation,
    file_digest,
    load_source,
    load_tokenizer,
    norm_matched_random,
    reconstruction_metrics,
    score_behavior,
    verify_model_files,
)
from protocol import (
    digest,
    ensure_paired_states,
    load_config,
    make_calibration,
    make_records,
    matched_donors,
    training_query,
)
from storage import LOGGER, StopRequested, Store, atomic_json, atomic_torch, gpu_memory, read_torch


def runtime_provenance():
    dependencies = {
        name: version(name)
        for name in ("torch", "transformers", "accelerate", "huggingface-hub", "peft", "safetensors", "PyYAML", "numpy")
    }
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines():
        name, expected = line.split("==")
        if dependencies[name] != expected:
            raise RuntimeError(f"NLA_DEPENDENCY_VERSION: {name}=={dependencies[name]}; expected {expected}")
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("NLA_TORCH_VERSION: preserve the parent-provided ROCm torch 2.10.0 image")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("NLA_GPU_UNAVAILABLE: ROCm GPU required; native inference has not run")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("NLA_GPU_ISOLATION: parent must expose exactly one GPU using ROCR_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("NLA_BF16_UNSUPPORTED: pilot requires native bf16")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dependencies,
        "torch_build": torch.__version__,
        "rocm": torch.version.hip,
        "gpu_name": properties.name,
        "gpu_total_bytes": properties.total_memory,
        "device": "cuda:0",
        "source_forward_signature": str(inspect.signature(Qwen2ForCausalLM.forward)),
        "generation_signature": str(inspect.signature(Qwen2ForCausalLM.generate)),
        "peft_factory_signature": str(inspect.signature(get_peft_model)),
        "attention": "sdpa",
        "dtype": "bfloat16",
        "cuda_device_count": 1,
    }


def source_hash():
    root = Path(__file__).parent
    names = [path.name for path in root.glob("*.py")] + ["requirements.txt", "upstream_contract.json"]
    files = {name: file_digest(root / name) for name in sorted(names)}
    return digest(files), files


def source_corpus(name, records, model, tokenizer, config, store, layer, calibration=False):
    started = time.monotonic()
    states = {}
    model.eval()
    for i, record in enumerate(records):
        store.check_stop()
        identifier = record["id"] if calibration else record.id
        path = store.root / "states" / name / f"{identifier}.pt"
        if path.exists():
            state = read_torch(path)
        else:
            content = record["content"] if calibration else record.query
            query = chat_query(tokenizer, content, config["max_sequence_length"])
            started_row = time.monotonic()
            activation = extract_activation(model, query, layer)
            behavior = (
                None if calibration else score_behavior(model, tokenizer, query, record, config["max_sequence_length"])
            )
            state = {
                "id": identifier,
                "stage": name,
                "query": query,
                "activation": activation,
                "source_layer": layer,
                "source_norm": float(activation.norm()),
                "behavior": behavior,
                "source_layer_convention": "HF hidden_states[20], output of zero-based block 19, before final model norm",
                "source_has_output_or_gold_tokens": False,
                "seconds": time.monotonic() - started_row,
            }
            atomic_torch(path, state)
            atomic_json(path.with_suffix(".json"), {key: value for key, value in state.items() if key != "activation"})
            store.event(
                "source_measured",
                stage=name,
                id=identifier,
                source_token_index=query["source_token_index"],
                norm=state["source_norm"],
                seconds=state["seconds"],
            )
        states[identifier] = state
        store.progress("source_corpus", stage=name, completed=i + 1, total=len(records), gpu=gpu_memory())
    store.event(
        "source_corpus_complete",
        stage=name,
        rows=len(states),
        session_seconds=time.monotonic() - started,
        gpu=gpu_memory(),
    )
    return states


def readout_corpus(name, records, states, av, ar, config, store, calibration=False):
    donors = matched_donors(records, config["seed"] + 211)
    identifiers = [r["id"] if calibration else r.id for r in records]
    result = {}
    for i, identifier in enumerate(identifiers):
        recipient = states[identifier]["activation"]
        donor_id = identifiers[donors[i]]
        inputs = {"true": recipient, "empty": torch.zeros_like(recipient), "shuffled": states[donor_id]["activation"]}
        random_control = None
        if calibration:
            inputs["random"], random_control = norm_matched_random(recipient, config["seed"], identifier)
        row = {"id": identifier, "stage": name, "shuffled_donor_id": donor_id, "reconstructions": {}}
        for condition, vector in inputs.items():
            store.check_stop()
            path = store.root / "readouts" / name / identifier / f"{condition}.pt"
            if path.exists():
                measured = read_torch(path)
            else:
                started = time.monotonic()
                description = av.describe(vector, config["max_new_tokens"])
                reconstruction = ar.reconstruct(description["explanation"]) if description["format_valid"] else None
                metrics = (
                    reconstruction_metrics(reconstruction, recipient, ar.sidecar.mse_scale)
                    if reconstruction is not None
                    else None
                )
                measured = {
                    **description,
                    "metrics": metrics,
                    "reconstruction": reconstruction,
                    "injected_raw_norm": float(vector.norm()),
                    "condition": condition,
                    "score_reference": "recipient's nonzero query activation, for every condition",
                    "seconds": time.monotonic() - started,
                }
                if condition == "random":
                    measured["random_control"] = random_control
                atomic_torch(path, measured)
                atomic_json(
                    path.with_suffix(".json"),
                    {key: value for key, value in measured.items() if key != "reconstruction"},
                )
                store.event(
                    "readout_measured",
                    stage=name,
                    id=identifier,
                    condition=condition,
                    format_valid=description["format_valid"],
                    exact_format_valid=description["exact_format_valid"],
                    stop_reason=description["stop_reason"],
                    eos_before_closing_tag=description["eos_before_closing_tag"],
                    metrics=metrics,
                    seconds=measured["seconds"],
                )
            row[condition] = {key: value for key, value in measured.items() if key != "reconstruction"}
            if measured["reconstruction"] is not None:
                row["reconstructions"][condition] = measured["reconstruction"]
        result[identifier] = row
        store.progress("readout_corpus", stage=name, completed=i + 1, total=len(records), gpu=gpu_memory())
    return result


def train_batch(records, tokenizer, config, update, step):
    generator = torch.Generator().manual_seed(config["seed"] + 401 + (step * update["batch_size"]) // len(records))
    order = torch.randperm(len(records), generator=generator).tolist()
    selected = [records[order[(step * update["batch_size"] + i) % len(records)]] for i in range(update["batch_size"])]
    rows = []
    for record in selected:
        prompt = training_query(record, step)
        if prompt == record.query:
            raise ValueError("NLA_TRAIN_TEST_LEAK: evaluation query entered training")
        query = chat_query(tokenizer, prompt, config["max_sequence_length"])
        complete, suffix = completion_ids(
            tokenizer, query, record.answer + tokenizer.eos_token, config["max_sequence_length"]
        )
        labels = [-100] * len(query["input_ids"]) + suffix
        rows.append((complete, labels))
    width = max(len(ids) for ids, _ in rows)
    return {
        "input_ids": torch.tensor([ids + [tokenizer.pad_token_id] * (width - len(ids)) for ids, _ in rows]),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows]),
        "labels": torch.tensor([labels + [-100] * (width - len(labels)) for _, labels in rows]),
    }


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch, use_cache=False).loss
    if not torch.isfinite(loss):
        raise RuntimeError("NLA_TRAIN_NONFINITE: optimizer loss")
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


def adapter_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()
    }


def adapter_digest(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def update_weights(model, tokenizer, records, update, config, store):
    selected = [r for r in records if r.learning_stage == update["name"]]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=update["learning_rate"], weight_decay=0.0)
    path = store.root / "checkpoints" / f"{update['name']}.pt"
    trace = {
        "stage": update["name"],
        "completed_steps": 0,
        "losses": [],
        "training_seconds": 0.0,
        "input_tokens": 0,
        "target_tokens": 0,
        "examples": 0,
        "initial_adapter_sha256": adapter_digest(adapter_state(model)),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "update_config": update,
    }
    if path.exists():
        checkpoint = read_torch(path)
        if checkpoint["config_sha256"] != store.config_hash:
            raise ValueError("NLA_CHECKPOINT_IDENTITY: training checkpoint belongs to another config")
        result = set_peft_model_state_dict(model, checkpoint["adapter"], ignore_mismatched_sizes=False)
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("NLA_CHECKPOINT_ADAPTER: LoRA state keys did not load exactly")
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        trace = checkpoint["trace"]
        store.event("training_resumed", stage=update["name"], completed_steps=trace["completed_steps"])

    def save_checkpoint():
        state = adapter_state(model)
        trace["adapter_sha256"] = adapter_digest(state)
        atomic_torch(
            path,
            {
                "adapter": state,
                "optimizer": optimizer.state_dict(),
                "trace": trace,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "config_sha256": store.config_hash,
            },
        )

    if not path.exists():
        save_checkpoint()
    model.train()
    for step in range(trace["completed_steps"], update["steps"]):
        if store.stop_signal is not None:
            save_checkpoint()
            store.check_stop()
        started = time.monotonic()
        batch = {
            key: value.to(model.device) for key, value in train_batch(selected, tokenizer, config, update, step).items()
        }
        loss, grad_norm = train_step(model, optimizer, batch)
        torch.cuda.synchronize()
        trace["completed_steps"] = step + 1
        trace["losses"].append(loss)
        trace["training_seconds"] += time.monotonic() - started
        trace["input_tokens"] += int(batch["attention_mask"].sum())
        trace["target_tokens"] += int((batch["labels"] != -100).sum())
        trace["examples"] += len(batch["input_ids"])
        store.event(
            "optimizer_step",
            stage=update["name"],
            step=step + 1,
            loss=trace["losses"][-1],
            grad_norm=float(grad_norm),
            target_tokens=trace["target_tokens"],
        )
        store.progress("training", stage=update["name"], completed=step + 1, total=update["steps"], gpu=gpu_memory())
        if (
            (step + 1) % config["checkpoint_every_steps"] == 0
            or step + 1 == update["steps"]
            or store.stop_signal is not None
        ):
            save_checkpoint()
        store.check_stop()
    model.eval()
    if trace["adapter_sha256"] == trace["initial_adapter_sha256"]:
        raise RuntimeError("NLA_NO_WEIGHT_CHANGE: optimizer completed without changing the adapter")
    store.event("weight_update_complete", **trace)
    return trace


def run(config, store, code_files):
    store.metrics["runtime"] = runtime_provenance()
    store.metrics["code_files_sha256"] = code_files
    store.save("running")
    store.progress("verify_checkpoints")
    last_hash_report = time.monotonic()

    def hash_tick(count):
        nonlocal last_hash_report
        store.check_stop()
        if time.monotonic() - last_hash_report >= 10:
            store.progress("verify_checkpoints", current_file_bytes_read=count)
            store.event("checkpoint_hash_progress", current_file_bytes_read=count)
            last_hash_report = time.monotonic()

    store.metrics["provenance"] = verify_model_files(config, hash_tick)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    records = make_records(config) if config["mode"] == "continual" else []
    calibration = make_calibration(config)
    data = {
        "generator": "deterministic hand-authored native topics/capital-city anchors plus seeded fictional Neral entity-color bindings",
        "seed": config["seed"],
        "records": [r.payload() for r in records],
        "calibration": calibration,
        "training_examples": [
            {
                "id": r.id,
                "stage": r.learning_stage,
                "queries": [training_query(r, i) for i in range(2)],
                "answer": r.answer,
            }
            for r in records
            if r.learning_stage is not None
        ],
        "split_semantics": "development/heldout are entity-disjoint forgetting-predictor splits. Acquisition tests use unseen question wording for trained entities. Native facts never enter the optimizer. Stage b entities are unexposed controls until update b.",
        "calibration_scope": "Input-only prose and counterfactual entity-binding prefixes calibrate the released readout. These contextual calibration examples are excluded from durable-learning metrics and optimizer updates.",
    }
    store.metrics["data_sha256"] = digest(data)
    store.metrics["data_provenance"] = {
        key: value for key, value in data.items() if key not in {"records", "calibration", "training_examples"}
    }
    atomic_json(store.root / "data.json", data)
    store.save()
    tokenizer = load_tokenizer(config["model_path"])
    sidecar = Sidecar.load(config["av_model_path"], "av")
    store.progress("load_source")
    source = load_source(config["model_path"])
    source.requires_grad_(False)
    calibration_states = source_corpus(
        "native_calibration", calibration, source, tokenizer, config, store, sidecar.layer, calibration=True
    )
    store.progress("load_released_av_ar")
    av, ar = NativeAV(config["av_model_path"]), NativeAR(config["ar_model_path"])
    store.metrics["native_trainable_parameter_counts"] = {
        name: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for name, model in (("source", source), ("av", av.model), ("ar", ar.model), ("ar_head", ar.head))
    }
    if any(store.metrics["native_trainable_parameter_counts"].values()):
        raise RuntimeError("NLA_NATIVE_TRAINABLE_PARAMETERS: source, AV, and AR must be frozen during calibration")
    store.event(
        "native_models_loaded",
        layer=av.sidecar.layer,
        injection_norm=av.sidecar.injection_scale,
        ar_mse_scale=ar.sidecar.mse_scale,
        gpu=gpu_memory(),
    )
    calibration_readouts = readout_corpus(
        "native_calibration", calibration, calibration_states, av, ar, config, store, calibration=True
    )
    gate = fidelity_gate(calibration, calibration_states, calibration_readouts, config)
    store.metrics["measurements"]["native_fidelity"] = gate
    store.metrics["measurements"]["native_diagnostics"] = calibration_diagnostics(
        calibration, calibration_states, calibration_readouts
    )
    store.save()
    store.event("native_fidelity_gate", **gate)
    if not gate["passed"]:
        store.finish("native_fidelity_gate_failed", weight_updates_executed=0, final_gpu_memory=gpu_memory())
        return 2
    if config["mode"] == "calibrate":
        store.finish("native_calibration_complete", weight_updates_executed=0, final_gpu_memory=gpu_memory())
        return 0
    torch.manual_seed(config["seed"] + 103)
    lora = LoraConfig(task_type="CAUSAL_LM", bias="none", lora_dropout=0.0, **config["lora"])
    source = get_peft_model(source, lora, autocast_adapter_dtype=True)
    with source.disable_adapter():
        base = source_corpus("base", records, source, tokenizer, config, store, av.sidecar.layer)
    prior_readouts = readout_corpus("base", records, base, av, ar, config, store)
    before = base
    store.metrics["measurements"]["updates"] = {}
    for update in config["updates"]:
        name = update["name"]
        trace = update_weights(source, tokenizer, records, update, config, store)
        after = source_corpus(f"after_{name}", records, source, tokenizer, config, store, av.sidecar.layer)
        with source.disable_adapter():
            frozen = source_corpus(f"frozen_after_{name}", records, source, tokenizer, config, store, av.sidecar.layer)
        ensure_paired_states(records, before, after)
        ensure_paired_states(records, base, frozen)
        behavior = stage_behavior(records, base, before, after, frozen, name)
        frozen_check = behavior["frozen_repeat"]
        if frozen_check["max_gold_logprob_error"] > 1e-3 or frozen_check["max_activation_absolute_error"] > 1e-3:
            raise RuntimeError(f"NLA_FROZEN_CONTROL_DRIFT: {frozen_check}")
        current_readouts = readout_corpus(f"after_{name}", records, after, av, ar, config, store)
        prediction = forgetting_prediction(records, before, after, prior_readouts, name, config)
        store.metrics["measurements"]["updates"][name] = {
            "training": trace,
            "behavior": behavior,
            "readout_drift": readout_drift(records, before, after, prior_readouts, current_readouts),
            "heldout_forgetting_prediction": prediction,
        }
        store.save()
        before, prior_readouts = after, current_readouts
    store.finish("pilot_complete", weight_updates_executed=len(config["updates"]), final_gpu_memory=gpu_memory())
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Native Qwen2.5 kitft NLA calibration for weight-based continual learning"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [nla] %(message)s")
    try:
        config = load_config(args.config)
    except (ValueError, KeyError, OSError, TypeError) as error:
        LOGGER.exception("NLA_CONFIG_FAILURE")
        if not (args.output_dir / "metrics.json").exists():
            atomic_json(
                args.output_dir / "metrics.json", {"status": "config_failed", "error": str(error), "measurements": {}}
            )
        return 1
    code_hash, files = source_hash()
    store = Store(args.output_dir, config, code_hash)
    try:
        if config.get("resume") and store.metrics["status"] in {"pilot_complete", "native_calibration_complete"}:
            store.event("already_complete", status=store.metrics["status"])
            return 0
        return run(config, store, files)
    except StopRequested as error:
        store.finish("interrupted", error=str(error), resumable=True)
        return 128 + (store.stop_signal or 15)
    except Exception as error:
        LOGGER.exception("NLA_RUN_FAILURE")
        store.finish("failed", error_type=type(error).__name__, error=str(error), resumable=True)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
