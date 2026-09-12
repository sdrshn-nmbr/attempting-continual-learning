from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load_file

LOGGER = logging.getLogger(__name__)
PROOF_BOUNDARY = (
    "These are independent per-projection Frobenius approximations of saved "
    "weight updates, not measurements of behavior or task complexity. Native "
    "rank 8 constrains the final generated adapter; the difference of two "
    "rank-8 adapters can have rank 16. A rank-8 approximation of the learned "
    "delta is therefore a separate diagnostic. Shared-core and alignment "
    "constraints may further restrict native adapters. No learning, "
    "consolidation, or transfer impossibility follows from these residuals."
)


@dataclass(frozen=True)
class FactorTerm:
    a: torch.Tensor
    b: torch.Tensor
    scale: float


def factor_singular_values(*terms: FactorTerm) -> torch.Tensor:
    if not terms:
        raise ValueError("SPECTRUM_EMPTY_TERMS")
    left, right = [], []
    shape = None
    for term in terms:
        a, b = term.a, term.b
        if a.device.type != "cpu" or b.device.type != "cpu":
            raise ValueError("SPECTRUM_CPU_ONLY")
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != b.shape[1]
            or min(*a.shape, *b.shape) <= 0
            or not a.is_floating_point()
            or not b.is_floating_point()
            or not math.isfinite(term.scale)
        ):
            raise ValueError("SPECTRUM_INVALID_FACTORS")
        current_shape = (b.shape[0], a.shape[1])
        if shape is not None and shape != current_shape:
            raise ValueError("SPECTRUM_TERM_SHAPE_MISMATCH")
        shape = current_shape
        left.append(b.detach().to(torch.float64) * term.scale)
        right.append(a.detach().to(torch.float64))
    b = torch.cat(left, dim=1)
    a = torch.cat(right, dim=0)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("SPECTRUM_NONFINITE_FACTORS")
    rb = torch.linalg.qr(b, mode="r").R
    ra = torch.linalg.qr(a.T, mode="r").R
    return torch.linalg.svdvals(rb @ ra.T)


def residual_statistics(energy: float, residual_energy: float) -> dict:
    fraction = residual_energy / energy if energy else 0.0
    return {
        "frobenius_squared": energy,
        "residual_frobenius_squared": residual_energy,
        "relative_frobenius_error": math.sqrt(fraction),
        "residual_energy_fraction": fraction,
        "zero_energy": energy == 0.0,
    }


def spectrum_statistics(
    values: torch.Tensor, shape: tuple[int, int], rank: int
) -> dict:
    if rank < 0:
        raise ValueError("SPECTRUM_NEGATIVE_APPROXIMATION_RANK")
    energy = float(values.square().sum())
    residual_energy = float(values[rank:].square().sum())
    tolerance = max(shape) * torch.finfo(torch.float64).eps * float(values[0])
    return {
        "singular_values": values.tolist(),
        "numerical_rank": int((values > tolerance).sum()),
        "numerical_rank_tolerance": tolerance,
        "approximation_rank": rank,
        **residual_statistics(energy, residual_energy),
    }


def checkpoint_factors(directory: Path) -> tuple[dict, dict[str, FactorTerm]]:
    config = json.loads((directory / "adapter_config.json").read_text())
    if (
        config["peft_type"] != "LORA"
        or config["bias"] != "none"
        or any(
            config.get(flag)
            for flag in (
                "use_dora",
                "use_rslora",
                "lora_bias",
                "fan_in_fan_out",
                "modules_to_save",
            )
        )
    ):
        raise ValueError(f"SPECTRUM_REQUIRES_PLAIN_LORA: {directory}")
    weights = load_file(directory / "adapter_model.safetensors", device="cpu")
    ranks, alphas = config["rank_pattern"], config["alpha_pattern"]
    if set(ranks) != set(alphas):
        raise ValueError(f"SPECTRUM_PATTERN_MISMATCH: {directory}")
    expected = {
        f"base_model.model.{path}.lora_{factor}.weight"
        for path in ranks
        for factor in ("A", "B")
    }
    if set(weights) != expected:
        raise ValueError(f"SPECTRUM_CHECKPOINT_TARGET_MISMATCH: {directory}")
    result = {}
    for path, rank in ranks.items():
        prefix = f"base_model.model.{path}"
        a, b = (weights[f"{prefix}.lora_{factor}.weight"] for factor in ("A", "B"))
        alpha = float(alphas[path])
        if (
            type(rank) is not int
            or rank <= 0
            or a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != rank
            or b.shape[1] != rank
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError(f"SPECTRUM_CHECKPOINT_RANK_OR_ALPHA_MISMATCH: {path}")
        result[path] = FactorTerm(a, b, alpha / rank)
    return config, result


def aggregate(projections: list[dict], component: str) -> dict:
    spectra = [projection[component] for projection in projections]
    return {
        "projection_count": len(spectra),
        "numerical_rank_min": min(row["numerical_rank"] for row in spectra),
        "numerical_rank_max": max(row["numerical_rank"] for row in spectra),
        "weighting": "Sum squared Frobenius energies before taking the ratio.",
        **residual_statistics(
            math.fsum(row["frobenius_squared"] for row in spectra),
            math.fsum(row["residual_frobenius_squared"] for row in spectra),
        ),
    }


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(config_path: Path, output_directory: Path) -> dict:
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    native_rank = config["native_rank"]
    if type(native_rank) is not int or native_rank <= 0:
        raise ValueError("SPECTRUM_INVALID_NATIVE_RANK")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    directories = {}
    pins = {config_path: hashlib.sha256(config_bytes).hexdigest()}
    inputs = {}
    for label in ("initial", "final"):
        checkpoint = config["checkpoints"][label]
        directory = (config_path.parent / checkpoint["local_path"]).resolve()
        directories[label] = directory
        inputs[label] = {"origin": checkpoint["origin"], "files": {}}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = directory / name
            actual = file_hash(path)
            if actual != checkpoint["sha256"][name]:
                raise ValueError(f"SPECTRUM_INPUT_HASH_MISMATCH: {path}: {actual}")
            pins[path] = actual
            inputs[label]["files"][name] = {
                "path": str(path),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
    source = Path(__file__).resolve()
    for path in (source, source.parent / "tests/test_compression_spectrum.py"):
        pins[path] = file_hash(path)
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "inputs": inputs,
        "pins": {str(path): digest for path, digest in pins.items()},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float64",
            "cpu_threads": torch.get_num_threads(),
        },
        "algorithm": "Thin QR of B and A transpose, then SVD of R_B @ R_A.T.",
        "dense_projection_matrices_materialized": False,
        "proof_boundary": PROOF_BOUNDARY,
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    frozen_path = output_directory / "frozen-inputs.json"
    write_json(frozen_path, frozen)
    frozen_sha = file_hash(frozen_path)
    LOGGER.info("SPECTRUM_FROZEN sha256=%s", frozen_sha)
    initial_config, initial = checkpoint_factors(directories["initial"])
    final_config, final = checkpoint_factors(directories["final"])
    if set(initial) != set(final) or len(final) != config["expected_projection_count"]:
        raise ValueError("SPECTRUM_CHECKPOINT_PROJECTION_MISMATCH")
    projections = []
    with torch.no_grad():
        for index, path in enumerate(sorted(final), start=1):
            start, end = initial[path], final[path]
            shape = (end.b.shape[0], end.a.shape[1])
            delta_terms = (end, FactorTerm(start.a, start.b, -start.scale))
            final_values = factor_singular_values(end)
            delta_values = factor_singular_values(*delta_terms)
            projections.append(
                {
                    "path": path,
                    "projection": path.rsplit(".", 1)[-1],
                    "shape": list(shape),
                    "initial": {
                        "factor_rank": start.a.shape[0],
                        "alpha": initial_config["alpha_pattern"][path],
                        "scale": start.scale,
                        "dtype": str(start.a.dtype),
                    },
                    "final": {
                        "factor_rank": end.a.shape[0],
                        "alpha": final_config["alpha_pattern"][path],
                        "scale": end.scale,
                        "dtype": str(end.a.dtype),
                    },
                    "final_adapter": spectrum_statistics(
                        final_values, shape, native_rank
                    ),
                    "learned_delta": spectrum_statistics(
                        delta_values, shape, native_rank
                    ),
                }
            )
            if index == len(final) or index % 12 == 0:
                LOGGER.info(
                    "SPECTRUM_PROJECTIONS_COMPLETE count=%s/%s", index, len(final)
                )
    for path, expected in pins.items():
        if file_hash(path) != expected:
            raise ValueError(f"SPECTRUM_INPUT_CHANGED_DURING_RUN: {path}")
    if file_hash(frozen_path) != frozen_sha:
        raise ValueError("SPECTRUM_FROZEN_RECEIPT_CHANGED")
    groups = {"all": projections}
    for projection in sorted({row["projection"] for row in projections}):
        groups[projection] = [
            row for row in projections if row["projection"] == projection
        ]
    report = {
        "name": config["name"],
        "native_rank": native_rank,
        "frozen_inputs_sha256": frozen_sha,
        "inputs_unchanged_after_measurement": True,
        "aggregate": {
            name: {
                component: aggregate(rows, component)
                for component in ("final_adapter", "learned_delta")
            }
            for name, rows in groups.items()
        },
        "projections": projections,
        "proof_boundary": PROOF_BOUNDARY,
    }
    write_json(output_directory / "spectrum.json", report)
    LOGGER.info(
        "SPECTRUM_COMPLETE %s", json.dumps(report["aggregate"]["all"], sort_keys=True)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run(args.config, args.output_dir)
    except Exception:
        LOGGER.exception("SPECTRUM_FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
