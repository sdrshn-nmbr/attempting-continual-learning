from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import peft
import torch
from peft import LoraConfig, PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn import functional

import compression_spectrum as spectrum

LOGGER = logging.getLogger(__name__)
RANK = 8
ALPHA = 16
SCALE = ALPHA / RANK
BOUNDARY = (
    "Static posthoc, independent rank-8 SVD approximations of the initial and "
    "final LoRA adapters. This controls representation rank and does not "
    "implement a shared generator, learned compression, or demonstrate task "
    "correctness. CPU probes isolate adapter injection using sparse zero base "
    "projections; full-model behavior is not evaluated. The parent's 288-group "
    "864-row dataset was already observed for other conditions, so the planned "
    "comparison is exploratory mechanistic evidence, not a new mapping or "
    "training replication. Failure does not imply compression is impossible."
)


def tensor_hash(tensors: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        hasher.update(name.encode())
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        hasher.update(memoryview(raw))
    return hasher.hexdigest()


def best_rank8(term: spectrum.FactorTerm) -> tuple[spectrum.FactorTerm, torch.Tensor]:
    if term.a.device.type != "cpu" or term.b.device.type != "cpu":
        raise ValueError("RANK_CONTROL_CPU_ONLY")
    if (
        term.a.ndim != 2
        or term.b.ndim != 2
        or term.a.shape[0] != term.b.shape[1]
        or not term.a.is_floating_point()
        or not term.b.is_floating_point()
        or not math.isfinite(term.scale)
    ):
        raise ValueError("RANK_CONTROL_INVALID_FACTORS")
    a = term.a.detach().to(torch.float64)
    b = term.b.detach().to(torch.float64) * term.scale
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("RANK_CONTROL_NONFINITE_FACTORS")
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    u, values, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    retained = min(RANK, len(values))
    root = torch.sqrt(values[:retained] / SCALE)
    fitted_b = torch.zeros((b.shape[0], RANK), dtype=torch.float64)
    fitted_a = torch.zeros((RANK, a.shape[1]), dtype=torch.float64)
    fitted_b[:, :retained] = (qb @ u[:, :retained]) * root
    fitted_a[:retained] = root[:, None] * (vh[:retained] @ qa.T)
    return spectrum.FactorTerm(fitted_a, fitted_b, SCALE), values


class ZeroLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.sparse_coo_tensor(
                size=(out_features, in_features),
                dtype=torch.float32,
                check_invariants=True,
            ),
            requires_grad=False,
        )
        self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.new_zeros((*inputs.shape[:-1], self.out_features))


def reload_projection_carrier(
    directory: Path, factors: dict[str, spectrum.FactorTerm]
) -> PeftModel:
    config = LoraConfig.from_pretrained(directory)
    base = torch.nn.Module()
    base.config = {"model_type": "qwen3", "tie_word_embeddings": False}
    for path, term in factors.items():
        parent = base
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent.get_submodule(part)
        parent.add_module(parts[-1], ZeroLinear(term.a.shape[1], term.b.shape[0]))
    model = PeftModel(base, config)
    loaded = model.load_adapter(
        directory, adapter_name="default", is_trainable=False, torch_device="cpu"
    )
    if loaded.unexpected_keys or any("lora_" in key for key in loaded.missing_keys):
        raise ValueError(f"RANK_CONTROL_PEFT_RELOAD_KEYS: {loaded}")
    return model.requires_grad_(False).eval()


def save_adapter(
    source_config: LoraConfig,
    factors: dict[str, spectrum.FactorTerm],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    config = replace(
        source_config,
        r=RANK,
        lora_alpha=ALPHA,
        rank_pattern={path: RANK for path in factors},
        alpha_pattern={path: ALPHA for path in factors},
        inference_mode=True,
        lora_dropout=0.0,
    )
    config.target_modules = sorted(config.target_modules)
    config.save_pretrained(directory)
    weights = {}
    for path, term in factors.items():
        weights[f"base_model.model.{path}.lora_A.weight"] = term.a.to(
            torch.float32
        ).contiguous()
        weights[f"base_model.model.{path}.lora_B.weight"] = term.b.to(
            torch.float32
        ).contiguous()
    save_file(
        weights, directory / "adapter_model.safetensors", metadata={"format": "pt"}
    )


def difference_energy(left: spectrum.FactorTerm, right: spectrum.FactorTerm) -> float:
    values = spectrum.factor_singular_values(
        left, spectrum.FactorTerm(right.a, right.b, -right.scale)
    )
    return float(values.square().sum())


def summarize(rows: list[dict], field: str) -> dict:
    energy = math.fsum(row["original_frobenius_squared"] for row in rows)
    residual = math.fsum(row[field] for row in rows)
    return spectrum.residual_statistics(energy, residual)


def output_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = actual.double() - expected.double()
    maximum = float(error.abs().max())
    denominator = float(torch.linalg.vector_norm(expected.double()))
    relative = (
        float(torch.linalg.vector_norm(error)) / denominator if denominator else maximum
    )
    return maximum, relative


def project_checkpoint(
    source: Path,
    destination: Path,
    config: dict,
    label: str,
    sealed: dict[str, dict],
) -> dict:
    raw_config, originals = spectrum.checkpoint_factors(source)
    source_config = LoraConfig.from_pretrained(source)
    if len(originals) != config["expected_projection_count"]:
        raise ValueError("RANK_CONTROL_PROJECTION_COUNT_MISMATCH")
    if (
        source_config.base_model_name_or_path
        != config["base_identity"]["base_model_name_or_path"]
    ):
        raise ValueError("RANK_CONTROL_BASE_IDENTITY_MISMATCH")
    if source_config.revision != config["base_identity"]["revision"]:
        raise ValueError("RANK_CONTROL_BASE_REVISION_MISMATCH")
    if set(source_config.target_modules) != {"q_proj", "v_proj"}:
        raise ValueError("RANK_CONTROL_REQUIRES_Q_AND_V_PROJECTIONS")
    gates = config["cpu_gates"]
    fitted, rows = {}, []
    for path in sorted(originals):
        term = originals[path]
        approximation, values = best_rank8(term)
        fitted[path] = approximation
        energy = float(values.square().sum())
        tail = float(values[RANK:].square().sum())
        actual = difference_energy(term, approximation)
        if not math.isclose(
            actual,
            tail,
            rel_tol=gates["residual_energy_rtol"],
            abs_tol=gates["residual_energy_atol"],
        ):
            raise ValueError(
                f"RANK_CONTROL_NOT_BEST_FROBENIUS_APPROXIMATION: {label}/{path}"
            )
        if label == "final":
            reference = sealed[path]["final_adapter"]
            for observed, expected in (
                (energy, reference["frobenius_squared"]),
                (actual, reference["residual_frobenius_squared"]),
            ):
                if not math.isclose(
                    observed,
                    expected,
                    rel_tol=gates["residual_energy_rtol"],
                    abs_tol=gates["residual_energy_atol"],
                ):
                    raise ValueError(f"RANK_CONTROL_SEALED_SPECTRUM_MISMATCH: {path}")
        ideal_relative = math.sqrt(actual / energy) if energy else 0.0
        if (
            label == "initial"
            and ideal_relative > gates["initial_fp64_relative_error_max"]
        ):
            raise ValueError(f"RANK_CONTROL_INITIAL_NOT_RANK8: {path}")
        rows.append(
            {
                "path": path,
                "projection": path.rsplit(".", 1)[-1],
                "shape": [term.b.shape[0], term.a.shape[1]],
                "source_factor_rank": term.a.shape[0],
                "source_scale": term.scale,
                "saved_rank": RANK,
                "saved_alpha": ALPHA,
                "saved_scale": SCALE,
                "original_frobenius_squared": energy,
                "optimal_tail_frobenius_squared": tail,
                "fp64_residual_frobenius_squared": actual,
                "fp64_relative_frobenius_error": ideal_relative,
                "sealed_spectrum_matched": True if label == "final" else None,
            }
        )
    save_adapter(source_config, fitted, destination)
    saved_config = LoraConfig.from_pretrained(destination)
    for field in ("base_model_name_or_path", "revision", "task_type"):
        if getattr(saved_config, field) != getattr(source_config, field):
            raise ValueError(f"RANK_CONTROL_SAVED_IDENTITY_CHANGED: {field}")
    model = reload_projection_carrier(destination, fitted)
    reloaded = get_peft_model_state_dict(model, save_embedding_layers=False)
    saved = load_file(destination / "adapter_model.safetensors", device="cpu")
    if set(saved) != set(reloaded) or any(
        not torch.equal(saved[key], reloaded[key]) for key in saved
    ):
        raise ValueError("RANK_CONTROL_RELOADED_TENSORS_CHANGED")
    rng = torch.Generator(device="cpu").manual_seed(config["probe_seed"])
    for row in rows:
        path = row["path"]
        prefix = f"base_model.model.{path}"
        loaded = spectrum.FactorTerm(
            reloaded[f"{prefix}.lora_A.weight"],
            reloaded[f"{prefix}.lora_B.weight"],
            SCALE,
        )
        layer = model.get_submodule(prefix)
        if layer.scaling["default"] != SCALE or layer.r["default"] != RANK:
            raise ValueError(f"RANK_CONTROL_PEFT_SCALING_OR_RANK_MISMATCH: {path}")
        residual = difference_energy(originals[path], loaded)
        energy = row["original_frobenius_squared"]
        relative = math.sqrt(residual / energy) if energy else 0.0
        if (
            abs(relative - row["fp64_relative_frobenius_error"])
            > gates["serialization_relative_error_change_max"]
        ):
            raise ValueError(
                f"RANK_CONTROL_SERIALIZATION_RESIDUAL_DRIFT: {label}/{path}"
            )
        probe = torch.randn(config["probe_rows"], loaded.a.shape[1], generator=rng)
        actual_output = layer(probe)
        fp32_reference = (
            functional.linear(
                functional.linear(probe, fitted[path].a.float()), fitted[path].b.float()
            )
            * SCALE
        )
        fp64_reference = (
            functional.linear(
                functional.linear(probe.double(), fitted[path].a), fitted[path].b
            )
            * SCALE
        )
        max_abs, relative_l2 = output_error(actual_output, fp32_reference)
        rounding_abs, rounding_relative = output_error(actual_output, fp64_reference)
        if (
            max_abs > gates["projection_probe_max_abs"]
            or relative_l2 > gates["projection_probe_relative_l2_max"]
        ):
            raise ValueError(
                f"RANK_CONTROL_PEFT_INJECTION_MISMATCH: {label}/{path}: abs={max_abs}, rel={relative_l2}"
            )
        row.update(
            {
                "serialized_residual_frobenius_squared": residual,
                "serialized_relative_frobenius_error": relative,
                "peft_probe_max_abs_error": max_abs,
                "peft_probe_relative_l2_error": relative_l2,
                "fp32_vs_fp64_fit_probe_max_abs": rounding_abs,
                "fp32_vs_fp64_fit_probe_relative_l2": rounding_relative,
            }
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("RANK_CONTROL_UNEXPECTED_TRAINABLE_PARAMETER")
    artifact = {
        "path": str(destination.resolve()),
        "files": {
            name: {
                "sha256": spectrum.file_hash(destination / name),
                "bytes": (destination / name).stat().st_size,
            }
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
        "tensor_sha256": tensor_hash(reloaded),
        "tensor_hash_format": "learner.tensor_hash(get_peft_model_state_dict(model, save_embedding_layers=False))",
        "tensor_count": len(reloaded),
        "parameter_count": sum(tensor.numel() for tensor in reloaded.values()),
        "base_model_name_or_path": raw_config["base_model_name_or_path"],
        "revision": raw_config["revision"],
        "rank": RANK,
        "alpha": ALPHA,
        "scaling": SCALE,
        "dtype": "torch.float32",
    }
    summary = {
        name: {
            field: summarize(selected, field)
            for field in (
                "fp64_residual_frobenius_squared",
                "serialized_residual_frobenius_squared",
            )
        }
        for name, selected in {
            "all": rows,
            **{
                projection: [row for row in rows if row["projection"] == projection]
                for projection in ("q_proj", "v_proj")
            },
        }.items()
    }
    LOGGER.info(
        "RANK_CONTROL_CHECKPOINT_COMPLETE label=%s projections=%s tensor_sha256=%s",
        label,
        len(rows),
        artifact["tensor_sha256"],
    )
    return {
        "artifact": artifact,
        "summary": summary,
        "projections": rows,
        "peft_reload_exact": True,
        "all_parameters_frozen": True,
    }


def run(config_path: Path, output: Path) -> dict:
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["rank"] != RANK or config["alpha"] != ALPHA:
        raise ValueError("RANK_CONTROL_FIXED_RANK8_ALPHA16_REQUIRED")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    pins = {config_path: hashlib.sha256(config_bytes).hexdigest()}
    for item in config["frozen_references"].values():
        path = (config_path.parent / item["path"]).resolve()
        if spectrum.file_hash(path) != item["sha256"]:
            raise ValueError(f"RANK_CONTROL_REFERENCE_HASH_MISMATCH: {path}")
        pins[path] = item["sha256"]
    directories = {}
    for label, checkpoint in config["checkpoints"].items():
        directory = (config_path.parent / checkpoint["local_path"]).resolve()
        directories[label] = directory
        for name, expected in checkpoint["sha256"].items():
            path = directory / name
            if spectrum.file_hash(path) != expected:
                raise ValueError(f"RANK_CONTROL_INPUT_HASH_MISMATCH: {path}")
            pins[path] = expected
    source = Path(__file__).resolve()
    for path in (source, source.parent / "tests/test_rank_control.py"):
        pins[path] = spectrum.file_hash(path)
    closure = {
        str(path): digest
        for path, digest in pins.items()
        if path.suffix == ".py" or path == config_path
    }
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "pins": {str(path): digest for path, digest in pins.items()},
        "source_closure": closure,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "peft": peft.__version__,
            "device": "cpu",
            "fit_dtype": "float64",
            "saved_dtype": "float32",
        },
        "proof_boundary": BOUNDARY,
    }
    frozen_path = output / "frozen-inputs.json"
    spectrum.write_json(frozen_path, frozen)
    frozen_sha = spectrum.file_hash(frozen_path)
    sealed_path = (
        config_path.parent / config["frozen_references"]["spectrum"]["path"]
    ).resolve()
    sealed = {
        row["path"]: row for row in json.loads(sealed_path.read_text())["projections"]
    }
    LOGGER.info("RANK_CONTROL_FROZEN sha256=%s", frozen_sha)
    with torch.no_grad():
        checkpoints = {
            label: project_checkpoint(
                directories[label], output / label / "adapter", config, label, sealed
            )
            for label in ("initial", "final")
        }
    for path, expected in pins.items():
        if spectrum.file_hash(path) != expected:
            raise ValueError(f"RANK_CONTROL_INPUT_CHANGED_DURING_RUN: {path}")
    if spectrum.file_hash(frozen_path) != frozen_sha:
        raise ValueError("RANK_CONTROL_FROZEN_RECEIPT_CHANGED")
    proof = {
        "passed": True,
        "checkpoints": checkpoints,
        "all_inputs_unchanged": True,
        "optimizer_updates": 0,
        "task_dataset_reads": 0,
        "proof_boundary": BOUNDARY,
    }
    spectrum.write_json(output / "proof.json", proof)
    receipt = {
        "passed": True,
        "artifacts": {
            f"rank8_{label}": result["artifact"]
            for label, result in checkpoints.items()
        },
        "originals": config["checkpoints"],
        "source_closure": closure,
        "frozen_inputs_sha256": frozen_sha,
        "proof": {
            "path": str((output / "proof.json").resolve()),
            "sha256": spectrum.file_hash(output / "proof.json"),
        },
        "planned_parent_evaluation": config["planned_parent_evaluation"],
        "proof_boundary": BOUNDARY,
    }
    spectrum.write_json(output / "receipt.json", receipt)
    LOGGER.info(
        "RANK_CONTROL_READY %s", json.dumps(receipt["artifacts"], sort_keys=True)
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run(args.config, args.output_dir)
    except Exception:
        LOGGER.exception("RANK_CONTROL_FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
