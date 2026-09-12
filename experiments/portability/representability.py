from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import compression_spectrum as spectrum
import portallib
import rank_control
import torch
from peft import LoraConfig
from portallib import PortalModel
from portallib.evaluation import PortalInjector
from safetensors.torch import load_file, save_file

LOGGER = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
DERIVATION = {
    "family": "D_hat = scale * O * B_c * A_c * I; each canonical product has rank <= r.",
    "bases": "Reduced QR then SVD gives O = U_o S_o V_o^T and I^T = U_i S_i V_i^T. All alignment singular directions must clear the FP64 rank tolerance; otherwise this study stops without an impossibility claim.",
    "minimum": "Let C = U_o^T D U_i. The exact independent-canonical-factor minimum squared error is ||(Id-U_o U_o^T)D||_F^2 + ||U_o^T D(Id-U_i U_i^T)||_F^2 + sum_{j>r} sigma_j(C)^2.",
    "reason": "The three residuals are orthogonal. Truncating the SVD of C attains the minimum, and the nonsingular alignment singular coordinates lift its factors back to canonical space.",
    "low_rank_computation": "D = scale * B A. Reduced QR of its thin factors and SVD of the small R_B R_A^T computes energies without forming a base-model weight matrix. Outside-span energies are computed directly, not by subtracting nearly equal total and retained energies.",
    "learned_change": "Also measure D_final-D_initial using B_change=[scale_final*B_final,-scale_initial*B_initial], A_change=[A_final;A_initial]. Both are the historical rank8 carriers. Use rank 16 for this span-only diagnostic: a difference of two rank8 adapters may have rank 16. Report its own energy normalization and input/output span loss separately from full-adapter loss. It does not change the fit gate, impose rank8 on the change, or prescribe a second fit.",
    "generator": "The minimum relaxes sharing across layers. Conditional on the sealed gate, a single minimum-change affine-head solve uses fixed hidden features H: W_new^T = W_old^T + [H,1]^+ (Y-[H,1]W_old^T). Four heads share one solution across every layer, with no optimizer or sweep.",
    "arithmetic": "Exact deterministic QR/SVD algorithm in FP64, with explicit numerical tolerances and independent dense controls; not a formal interval-arithmetic proof.",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"REPRESENTABILITY_{message}")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def pin(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }


def verify_pins(pins: list[dict]) -> None:
    for spec in pins:
        path = Path(spec["path"])
        require(path.is_file(), f"MISSING_INPUT: {path}")
        require(
            path.stat().st_size == spec["bytes"] and file_hash(path) == spec["sha256"],
            f"PIN_MISMATCH: {path}",
        )


def runtime() -> dict:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {
            name: version(name)
            for name in ("torch", "portallib", "peft", "safetensors")
        },
        "device": "cpu",
        "math_dtype": "float64",
        "injection_dtype": "float32",
    }


def config_pins(config: dict) -> list[dict]:
    return [
        spec
        for name in ("native_initial", "rank8_initial", "rank8_final")
        for spec in config[name]["files"].values()
    ] + list(config["historical"].values())


def seal(config_path: Path, output: Path) -> str:
    config = json.loads(config_path.read_text())
    require(config["kind"] == "portal_fixed_alignment_representability", "CONFIG_KIND")
    require(
        config["numerics"]["device"] == "cpu"
        and config["numerics"]["math_dtype"] == "float64",
        "CPU_FP64_REQUIRED",
    )
    require(config["rank"] == 8 and config["alpha"] == 16, "RANK8_ALPHA16_REQUIRED")
    require(
        config["fit"]["configurations"] == 1 and config["fit"]["optimizer_steps"] == 0,
        "ONE_ANALYTIC_FIT_REQUIRED",
    )
    require(
        config["fit"]["solver"] == "single_affine_head_minimum_change_svd", "FIT_SOLVER"
    )
    require(
        config["numerics"]["require_full_alignment_rank"] is True,
        "FULL_RANK_CERTIFICATE_REQUIRED",
    )
    for name, value in config["gates"].items():
        require(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
            f"INVALID_GATE: {name}",
        )
    package = Path(portallib.__file__).parent
    sources = [
        Path(__file__),
        Path(spectrum.__file__),
        Path(rank_control.__file__),
        Path(__file__).parent / "tests/test_representability.py",
        config_path,
    ]
    sources.extend(
        package / name
        for name in ("_architecture.py", "config.py", "model.py", "evaluation.py")
    )
    pins = config_pins(config) + [pin(path) for path in sources]
    verify_pins(pins)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "pins": pins,
        "runtime": runtime(),
        "derivation": DERIVATION,
        "sdk_signatures": {
            name: str(inspect.signature(obj))
            for name, obj in {
                "PortalModel": PortalModel,
                "PortalModel.from_pretrained": PortalModel.from_pretrained,
                "PortalModel.forward": PortalModel.forward,
                "PortalInjector": PortalInjector,
                "PortalInjector.activate": PortalInjector.activate,
                "safetensors.load_file": load_file,
                "safetensors.save_file": save_file,
            }.items()
        },
        "access_order": [
            "protocol sealed",
            "verify pins",
            "initial exact control",
            "all72 projection and gauge checks",
            "persist projection and fit gate",
            "conditional single shared-head fit",
            "serialization and injection receipts",
        ],
        "no_behavioral_selection": True,
        "control_tests": str(Path(__file__).parent / "tests/test_representability.py"),
    }
    write_json(output / "protocol.json", protocol)
    return file_hash(output / "protocol.json")


def verify_protocol(path: Path, expected_hash: str) -> dict:
    require(file_hash(path) == expected_hash, "PROTOCOL_HASH_MISMATCH")
    protocol = json.loads(path.read_text())
    verify_pins(protocol["pins"])
    require(runtime() == protocol["runtime"], "RUNTIME_CHANGED")
    return protocol


def fp64(value: torch.Tensor) -> torch.Tensor:
    require(
        value.device.type == "cpu" and value.ndim == 2 and value.is_floating_point(),
        "CPU_FLOAT_MATRIX_REQUIRED",
    )
    require(bool(torch.isfinite(value).all()), "NONFINITE_MATRIX")
    return value.detach().to(torch.float64)


@dataclass(frozen=True)
class Span:
    basis: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    evidence: dict


def alignment_span(matrix: torch.Tensor) -> Span:
    matrix = fp64(matrix)
    require(min(matrix.shape) > 0, "EMPTY_ALIGNMENT")
    q, r = torch.linalg.qr(matrix, mode="reduced")
    u, values, vh = torch.linalg.svd(r, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * float(values[0])
    numerical_rank = int((values > tolerance).sum())
    require(
        numerical_rank == min(matrix.shape),
        "ALIGNMENT_RANK_UNRESOLVED_NO_IMPOSSIBILITY_CLAIM",
    )
    basis = q @ u
    reconstruction = basis @ (values[:, None] * vh)
    relative = float(
        torch.linalg.vector_norm(matrix - reconstruction)
        / torch.linalg.vector_norm(matrix)
    )
    orthogonality = float(
        (basis.T @ basis - torch.eye(numerical_rank, dtype=torch.float64)).abs().max()
    )
    require(
        relative <= 1e-10 and orthogonality <= 1e-10, "ALIGNMENT_DECOMPOSITION_FAILED"
    )
    return Span(
        basis,
        values,
        vh,
        {
            "shape": list(matrix.shape),
            "rank": numerical_rank,
            "singular_values": values.tolist(),
            "rank_tolerance": tolerance,
            "condition_number": float(values[0] / values[-1]),
            "relative_reconstruction_error": relative,
            "maximum_orthogonality_error": orthogonality,
        },
    )


def energy(term: spectrum.FactorTerm) -> float:
    return float(spectrum.factor_singular_values(term).square().sum())


def distance(left: spectrum.FactorTerm, right: spectrum.FactorTerm) -> float:
    return float(
        spectrum.factor_singular_values(
            left, spectrum.FactorTerm(right.a, right.b, -right.scale)
        )
        .square()
        .sum()
    )


def difference_term(
    final: spectrum.FactorTerm, initial: spectrum.FactorTerm
) -> spectrum.FactorTerm:
    require(
        (final.b.shape[0], final.a.shape[1])
        == (initial.b.shape[0], initial.a.shape[1]),
        "CHANGE_SHAPE_MISMATCH",
    )
    return spectrum.FactorTerm(
        torch.cat((fp64(final.a), fp64(initial.a)), dim=0),
        torch.cat(
            (fp64(final.b) * final.scale, -fp64(initial.b) * initial.scale), dim=1
        ),
        1.0,
    )


@dataclass(frozen=True)
class Projection:
    canonical_a: torch.Tensor
    canonical_b: torch.Tensor
    effective: spectrum.FactorTerm
    metrics: dict


def project(
    term: spectrum.FactorTerm, inputs: Span, outputs: Span, rank: int, scale: float
) -> Projection:
    a, b = fp64(term.a), fp64(term.b)
    require(
        a.shape[0] == b.shape[1]
        and a.shape[1] == inputs.basis.shape[0]
        and b.shape[0] == outputs.basis.shape[0],
        "PROJECTION_SHAPE_MISMATCH",
    )
    require(
        rank > 0 and math.isfinite(scale) and scale > 0 and math.isfinite(term.scale),
        "INVALID_RANK_OR_SCALE",
    )
    ui, uo = inputs.basis, outputs.basis
    reduced_a, reduced_b = a @ ui, uo.T @ b
    output_residual = energy(spectrum.FactorTerm(a, b - uo @ reduced_b, term.scale))
    input_residual = energy(
        spectrum.FactorTerm(a - reduced_a @ ui.T, reduced_b, term.scale)
    )
    qb, rb = torch.linalg.qr(reduced_b * term.scale, mode="reduced")
    qa, ra = torch.linalg.qr(reduced_a.T, mode="reduced")
    u, values, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    retained = min(rank, len(values))
    root = torch.sqrt(values[:retained] / scale)
    coordinate_a = torch.zeros((rank, ui.shape[1]), dtype=torch.float64)
    coordinate_b = torch.zeros((uo.shape[1], rank), dtype=torch.float64)
    coordinate_a[:retained] = root[:, None] * (vh[:retained] @ qa.T)
    coordinate_b[:, :retained] = (qb @ u[:, :retained]) * root
    canonical_a = (coordinate_a / inputs.singular_values) @ inputs.vh
    canonical_b = outputs.vh.T @ (coordinate_b / outputs.singular_values[:, None])
    effective = spectrum.FactorTerm(coordinate_a @ ui.T, uo @ coordinate_b, scale)
    total = energy(term)
    tail = float(values[rank:].square().sum())
    residual = math.fsum((output_residual, input_residual, tail))
    return Projection(
        canonical_a,
        canonical_b,
        effective,
        {
            **spectrum.residual_statistics(total, residual),
            "output_span_residual_squared": output_residual,
            "additional_input_span_residual_squared": input_residual,
            "rank_tail_squared": tail,
            "projected_singular_values": values.tolist(),
            "energy_partition_relative_error": abs(
                total
                - math.fsum(
                    (output_residual, input_residual, float(values.square().sum()))
                )
            )
            / total
            if total
            else 0.0,
            "attained_residual_squared": distance(term, effective),
        },
    )


def aggregate(rows: list[dict]) -> dict:
    require(bool(rows), "EMPTY_MATRIX_SET")
    return {
        **spectrum.residual_statistics(
            math.fsum(row["frobenius_squared"] for row in rows),
            math.fsum(row["residual_frobenius_squared"] for row in rows),
        ),
        "matrix_count": len(rows),
        "maximum_matrix_relative_frobenius_error": max(
            row["relative_frobenius_error"] for row in rows
        ),
        "weighting": "Sum squared energies across all matrices, then take the relative Frobenius norm.",
    }


def span_aggregate(rows: list[dict]) -> dict:
    result = aggregate(rows)
    total = result["frobenius_squared"]
    for field in (
        "output_span_residual_squared",
        "additional_input_span_residual_squared",
        "rank_tail_squared",
    ):
        result[field] = math.fsum(row[field] for row in rows)
        result[field + "_fraction"] = result[field] / total if total else 0.0
    return result


def checked_projection(term, inputs, outputs, rank, scale, gauge, gates, label):
    projection = project(term, inputs, outputs, rank, scale)
    transformed = spectrum.FactorTerm(
        torch.linalg.solve(gauge, term.a.double()),
        term.b.double() @ gauge,
        term.scale,
    )
    changed = project(transformed, inputs, outputs, rank, scale)
    metrics = projection.metrics
    denominator = metrics["frobenius_squared"] or 1.0
    gauge_delta = (
        abs(
            changed.metrics["residual_frobenius_squared"]
            - metrics["residual_frobenius_squared"]
        )
        / denominator
    )
    require(
        gauge_delta <= gates["gauge_relative_energy_difference_max"],
        f"GAUGE_CONTROL: {label}",
    )
    require(
        metrics["energy_partition_relative_error"]
        <= gates["numerical_relative_energy_tolerance"],
        f"ENERGY_PARTITION: {label}",
    )
    require(
        abs(
            metrics["attained_residual_squared"] - metrics["residual_frobenius_squared"]
        )
        / denominator
        <= gates["numerical_relative_energy_tolerance"],
        f"PROJECTION_MINIMUM_NOT_ATTAINED: {label}",
    )
    return projection, gauge_delta


def fit_gate(summary: dict, gates: dict) -> bool:
    return (
        summary["relative_frobenius_error"]
        <= gates["fit_aggregate_relative_frobenius_max"]
        and summary["maximum_matrix_relative_frobenius_error"]
        <= gates["fit_every_matrix_relative_frobenius_max"]
    )


def load_native(directory: Path, dtype: torch.dtype = torch.float64) -> PortalModel:
    require(
        all(
            (directory / name).is_file()
            for name in ("config.json", "model.safetensors")
        ),
        f"MISSING_NATIVE_CHECKPOINT: {directory}",
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = PortalModel.from_pretrained(
            directory, local_files_only=True, device="cpu", dtype=dtype
        )
    finally:
        torch.set_default_dtype(previous)
    return model.requires_grad_(False).eval()


def save_native(model: PortalModel, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", model.config.to_dict())
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        directory / "model.safetensors",
        metadata={"format": "portallib", "format_version": "1"},
    )
    before = rank_control.tensor_hash(model.state_dict())
    restored = load_native(directory, next(model.parameters()).dtype)
    require(
        before == rank_control.tensor_hash(restored.state_dict()),
        "NATIVE_RELOAD_NOT_EXACT",
    )
    return {
        "path": str(directory),
        "state_sha256": before,
        "reload_exact": True,
        "dtype": str(next(model.parameters()).dtype),
    }


def generated_terms(model: PortalModel, task: str) -> dict[str, spectrum.FactorTerm]:
    generated = model.generate(task)
    return {
        path: spectrum.FactorTerm(*generated[target.key], model.config.scaling)
        for target, path in model.config.resolved_targets()
    }


def zero_carrier(config) -> torch.nn.Module:
    base = torch.nn.Module()
    for target, path in config.resolved_targets():
        parent = base
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent.get_submodule(part)
        parent.add_module(
            parts[-1], rank_control.ZeroLinear(target.in_features, target.out_features)
        )
    return base


@torch.no_grad()
def serialization_probe(
    config, source_config: LoraConfig, terms: dict, directory: Path, protocol: dict
) -> dict:
    gates, numerics = protocol["gates"], protocol["numerics"]
    require(
        set(terms) == {path for _, path in config.resolved_targets()},
        "INJECTION_MATRIX_COVERAGE",
    )
    rank_control.save_adapter(source_config, terms, directory)
    _, restored = spectrum.checkpoint_factors(directory)
    require(set(restored) == set(terms), "SERIALIZED_MATRIX_COVERAGE")
    peft_model = rank_control.reload_projection_carrier(directory, restored)
    native = zero_carrier(config)
    generated = {
        target.key: (restored[path].a, restored[path].b)
        for target, path in config.resolved_targets()
    }
    rng = torch.Generator(device="cpu").manual_seed(numerics["seed"])
    rows = []
    with PortalInjector(native, config) as injector:
        with injector.activate(generated):
            for target, path in config.resolved_targets():
                original, term = terms[path], restored[path]
                require(
                    torch.equal(term.a, original.a.float())
                    and torch.equal(term.b, original.b.float())
                    and term.scale == original.scale,
                    f"SERIALIZED_FACTORS_CHANGED: {path}",
                )
                original_energy = energy(original)
                relative_serialization = (
                    math.sqrt(distance(original, term) / original_energy)
                    if original_energy
                    else 0.0
                )
                require(
                    relative_serialization
                    <= gates["serialization_relative_frobenius_max"],
                    f"SERIALIZATION_ERROR: {path}",
                )
                x = torch.randn(
                    (numerics["probe_rows"], target.in_features),
                    generator=rng,
                    dtype=torch.float32,
                )
                reference = (x @ term.a.T) @ term.b.T * term.scale
                injected = native.get_submodule(path)(x)
                peft_output = peft_model.base_model.model.get_submodule(path)(x)
                require(
                    injected.dtype == peft_output.dtype == torch.float32,
                    f"INJECTION_DTYPE: {path}",
                )
                native_error = rank_control.output_error(injected, reference)
                peft_error = rank_control.output_error(peft_output, reference)
                require(
                    max(native_error[0], peft_error[0])
                    <= gates["fp32_injection_max_abs"]
                    and max(native_error[1], peft_error[1])
                    <= gates["fp32_injection_relative_l2_max"],
                    f"INJECTION_ERROR: {path}",
                )
                fp64_reference = (
                    (x.double() @ original.a.double().T)
                    @ original.b.double().T
                    * original.scale
                )
                rows.append(
                    {
                        "path": path,
                        "serialization_relative_frobenius": relative_serialization,
                        "native_max_abs": native_error[0],
                        "native_relative_l2": native_error[1],
                        "peft_max_abs": peft_error[0],
                        "peft_relative_l2": peft_error[1],
                        "fp32_vs_fp64_forward": rank_control.output_error(
                            reference, fp64_reference
                        ),
                    }
                )
        require(
            all(
                not bool(
                    native.get_submodule(path)(
                        torch.ones(1, target.in_features, dtype=torch.float32)
                    ).any()
                )
                for target, path in config.resolved_targets()
            ),
            "INJECTOR_CONTEXT_LEAK",
        )
    result = {
        "matrix_count": len(rows),
        "passed": True,
        "rows": rows,
        "scope": "Actual serialized FP32 factors reloaded through PEFT and PortalInjector on exact-path sparse zero base projections. No base LLM or behavioral proof.",
    }
    write_json(directory.parent / (directory.name + "-injection.json"), result)
    return result


@torch.no_grad()
def fit_shared_heads(
    model: PortalModel,
    canonical: dict[str, tuple[torch.Tensor, torch.Tensor]],
    config: dict,
) -> dict:
    allowed = set(config["fit"]["trainable"])
    heads = {
        f"core.{factor}.{name}.{parameter}"
        for factor in ("A", "B")
        for name in model.config.modules
        for parameter in ("weight", "bias")
    }
    require(allowed == heads, "FIT_PARAMETER_SCOPE")
    frozen_before = rank_control.tensor_hash(
        {key: value for key, value in model.state_dict().items() if key not in allowed}
    )
    z = model.task_latents[model.config.tasks.index(config["fit"]["task"])]
    hidden = model.core.hidden(z, model.alignment.layer_embeddings.weight)
    design = torch.cat(
        (hidden, torch.ones((len(hidden), 1), dtype=hidden.dtype)), dim=1
    )
    u, values, vh = torch.linalg.svd(design, full_matrices=False)
    cutoff = max(design.shape) * torch.finfo(torch.float64).eps * float(values[0])
    full_row_rank = len(values) == len(hidden) and bool((values > cutoff).all())
    condition = float(values[0] / values[-1]) if values[-1] > 0 else None
    evidence = {
        "design_shape": list(design.shape),
        "singular_values": values.tolist(),
        "full_row_rank": full_row_rank,
        "condition_number": condition,
        "linear_solves": 0,
        "optimizer_steps": 0,
    }
    if (
        not full_row_rank
        or condition > config["gates"]["shared_head_fit_condition_max"]
    ):
        return {
            **evidence,
            "status": "inconclusive_hidden_design",
            "impossibility_claim": False,
        }
    inverse = (vh.T / values) @ u.T
    residuals = []
    for name in model.config.modules:
        targets = sorted(
            (
                (target, path)
                for target, path in model.config.resolved_targets()
                if target.module_name == name
            ),
            key=lambda pair: pair[0].layer_index,
        )
        require(
            [target.layer_index for target, _ in targets]
            == list(range(model.config.n_layers)),
            "FIT_LAYER_COVERAGE",
        )
        for index, factor in enumerate(("A", "B")):
            head = getattr(model.core, factor)[name]
            expected = torch.stack(
                [canonical[path][index].reshape(-1) for _, path in targets]
            )
            weights = torch.cat((head.weight.T, head.bias.unsqueeze(0)), dim=0)
            fitted = weights + inverse @ (expected - design @ weights)
            head.weight.copy_(fitted[:-1].T)
            head.bias.copy_(fitted[-1])
            denominator = float(torch.linalg.vector_norm(expected))
            error = float(torch.linalg.vector_norm(head(hidden) - expected))
            residuals.append(error / denominator if denominator else error)
            evidence["linear_solves"] += 1
    require(
        evidence["linear_solves"] <= config["fit"]["maximum_linear_solves"],
        "FIT_BOUND_EXCEEDED",
    )
    frozen_after = rank_control.tensor_hash(
        {key: value for key, value in model.state_dict().items() if key not in allowed}
    )
    require(frozen_before == frozen_after, "FIT_MUTATED_FROZEN_STATE")
    passed = max(residuals) <= config["gates"]["shared_head_fit_relative_factor_max"]
    return {
        **evidence,
        "status": "constructed" if passed else "inconclusive_numerical_residual",
        "maximum_relative_factor_error": max(residuals),
        "frozen_state_sha256": frozen_before,
        "frozen_state_exact": True,
        "impossibility_claim": False,
    }


def verify_source_identity(config: dict, model: PortalModel) -> dict:
    historical = {
        key: json.loads(Path(spec["path"]).read_text())
        for key, spec in config["historical"].items()
    }
    audit, result, behavior = (
        historical[key]
        for key in ("behavior_audit", "behavior_result", "behavior_config")
    )
    require(
        audit["passed"]
        and audit["rank8_final_correct"] == 850
        and audit["result_sha256"] == config["historical"]["behavior_result"]["sha256"],
        "HISTORICAL_AUDIT_BINDING",
    )
    require(
        result["config_sha256"] == canonical_json_hash(behavior),
        "HISTORICAL_CONFIG_BINDING",
    )
    require(
        model.config.base_model_name_or_path == config["source_base"]["repo_id"]
        and model.config.base_model_revision == config["source_base"]["revision"],
        "NATIVE_BASE_IDENTITY",
    )
    require(
        model.config.rank == config["rank"] and model.config.alpha == config["alpha"],
        "NATIVE_RANK_OR_SCALING",
    )
    for label in ("rank8_initial", "rank8_final"):
        for filename, spec in config[label]["files"].items():
            require(
                behavior["adapters"][label]["files"][filename] == spec["sha256"],
                f"HISTORICAL_FILE_BINDING: {label}/{filename}",
            )
        tensors = load_file(
            Path(config[label]["path"]) / "adapter_model.safetensors", device="cpu"
        )
        require(
            rank_control.tensor_hash(tensors) == result["adapter_tensor_hashes"][label],
            f"HISTORICAL_TENSOR_BINDING: {label}",
        )
    return {
        "historical_correct": 850,
        "historical_rows": result["rows"],
        "historical_behavior_binding_verified": True,
        "historical_config_hash_kind": "canonical compact sorted JSON; raw file bytes are separately pinned",
        "new_behavior_evaluation": False,
    }


@torch.no_grad()
def study(config: dict, output: Path) -> dict:
    gates = config["gates"]
    model = load_native(Path(config["native_initial"]["path"]))
    identity = verify_source_identity(config, model)
    targets = list(model.config.resolved_targets())
    require(
        len(targets) == config["expected_matrix_count"]
        and len({path for _, path in targets}) == len(targets),
        "MATRIX_COUNT",
    )
    initial_config, initial = spectrum.checkpoint_factors(
        Path(config["rank8_initial"]["path"])
    )
    final_config, final = spectrum.checkpoint_factors(
        Path(config["rank8_final"]["path"])
    )
    require(initial_config == final_config, "SOURCE_ADAPTER_CONFIGS_DIFFER")
    require(
        final_config["base_model_name_or_path"].endswith(
            config["source_base"]["repo_id"] + "/" + config["source_base"]["revision"]
        ),
        "SOURCE_BASE_IDENTITY",
    )
    require(
        set(final) == set(initial) == {path for _, path in targets},
        "SOURCE_MATRIX_COVERAGE",
    )
    require(
        all(
            term.a.shape[0] == config["rank"] and term.scale == model.config.scaling
            for term in final.values()
        ),
        "SOURCE_RANK_OR_SCALING",
    )
    source_config = LoraConfig.from_pretrained(config["rank8_final"]["path"])
    inputs, outputs = {}, {}
    for group, matrix in model.alignment.input.items():
        LOGGER.info("REPRESENTABILITY_ALIGNMENT_INPUT %s", group)
        inputs[group] = alignment_span(matrix.T)
    for group, matrix in model.alignment.output.items():
        LOGGER.info("REPRESENTABILITY_ALIGNMENT_OUTPUT %s", group)
        outputs[group] = alignment_span(matrix)
    write_json(
        output / "alignments.json",
        {
            "input": {key: span.evidence for key, span in inputs.items()},
            "output": {key: span.evidence for key, span in outputs.items()},
            "alignment_state_sha256": rank_control.tensor_hash(
                model.alignment.state_dict()
            ),
        },
    )
    positive_path = output / "initial"
    positive_path.mkdir()
    native_receipt = save_native(model, positive_path / "native")
    restored_native = load_native(positive_path / "native")
    exact = generated_terms(restored_native, config["initial_task"])
    initial_rows = []
    for target, path in targets:
        projection = project(
            exact[path],
            inputs[target.input_group],
            outputs[target.output_group],
            config["rank"],
            model.config.scaling,
        )
        previous_error = math.sqrt(
            distance(exact[path], initial[path]) / energy(exact[path])
        )
        require(
            projection.metrics["relative_frobenius_error"]
            <= gates["initial_exact_relative_frobenius_max"],
            f"INITIAL_EXACT_CONTROL: {path}",
        )
        require(
            previous_error <= gates["historical_initial_relative_frobenius_max"],
            f"HISTORICAL_INITIAL_CONTROL: {path}",
        )
        initial_rows.append(
            {
                "path": path,
                **projection.metrics,
                "historical_rank8_initial_relative_frobenius": previous_error,
            }
        )
    initial_probe = serialization_probe(
        model.config, source_config, exact, positive_path / "adapter", config
    )
    write_json(
        positive_path / "control.json",
        {
            "passed": True,
            "native": native_receipt,
            "summary": span_aggregate(initial_rows),
            "rows": initial_rows,
            "injection_passed": initial_probe["passed"],
            "scope": "Exact FP64 initial native generator is a separate positive control; historical FP32 rank8 initial is compared with a separate rounding tolerance.",
        },
    )
    del restored_native
    LOGGER.info(
        "REPRESENTABILITY_INITIAL_CONTROL_PASSED %d matrices", len(initial_rows)
    )
    rng = torch.Generator(device="cpu").manual_seed(config["numerics"]["seed"])
    orthogonal = torch.linalg.qr(
        torch.randn(
            (config["rank"], config["rank"]), generator=rng, dtype=torch.float64
        )
    ).Q
    gauge = orthogonal @ torch.diag(
        torch.linspace(0.5, 2.0, config["rank"], dtype=torch.float64)
    )
    rows, change_rows, canonical, fitted = [], [], {}, {}
    change_gauge = torch.block_diag(gauge, gauge)
    for target, path in targets:
        term = final[path]
        projection, gauge_delta = checked_projection(
            term,
            inputs[target.input_group],
            outputs[target.output_group],
            config["rank"],
            model.config.scaling,
            gauge,
            gates,
            path,
        )
        metrics = projection.metrics
        require(metrics["frobenius_squared"] > 0, f"ZERO_SOURCE_MATRIX: {path}")
        canonical[path] = (projection.canonical_a, projection.canonical_b)
        fitted[path] = projection.effective
        rows.append(
            {
                "path": path,
                "input_group": target.input_group,
                "output_group": target.output_group,
                **metrics,
                "gauge_relative_energy_difference": gauge_delta,
            }
        )
        change = difference_term(term, initial[path])
        change_projection, change_gauge_delta = checked_projection(
            change,
            inputs[target.input_group],
            outputs[target.output_group],
            2 * config["rank"],
            1.0,
            change_gauge,
            gates,
            "learned_change/" + path,
        )
        change_rows.append(
            {
                "path": path,
                "input_group": target.input_group,
                "output_group": target.output_group,
                **change_projection.metrics,
                "gauge_relative_energy_difference": change_gauge_delta,
                "historical_initial_frobenius_squared": energy(initial[path]),
                "full_final_frobenius_squared": metrics["frobenius_squared"],
                "learned_over_full_energy_fraction": change_projection.metrics[
                    "frobenius_squared"
                ]
                / metrics["frobenius_squared"],
            }
        )
    summary = span_aggregate(rows)
    change_summary = span_aggregate(change_rows)
    historical_initial_energy = math.fsum(
        row["historical_initial_frobenius_squared"] for row in change_rows
    )
    change_summary.update(
        {
            "historical_initial_frobenius_squared": historical_initial_energy,
            "full_final_frobenius_squared": summary["frobenius_squared"],
            "initial_over_full_energy_fraction": historical_initial_energy
            / summary["frobenius_squared"],
            "learned_over_full_energy_fraction": change_summary["frobenius_squared"]
            / summary["frobenius_squared"],
        }
    )
    write_json(
        output / "learned-change.json",
        {
            "summary": change_summary,
            "by_module": {
                name: span_aggregate(
                    [row for row in change_rows if row["input_group"] == name]
                )
                for name in model.config.modules
            },
            "rows": change_rows,
            "diagnostic_only": True,
            "rank_cap": 2 * config["rank"],
            "scope": DERIVATION["learned_change"],
        },
    )
    write_json(
        output / "projection.json",
        {
            "summary": summary,
            "by_module": {
                name: span_aggregate(
                    [row for row in rows if row["input_group"] == name]
                )
                for name in model.config.modules
            },
            "rows": rows,
            "derivation": DERIVATION,
        },
    )
    eligible = fit_gate(summary, gates)
    write_json(
        output / "fit-gate.json",
        {
            "eligible": eligible,
            "summary": summary,
            "aggregate_limit": gates["fit_aggregate_relative_frobenius_max"],
            "every_matrix_limit": gates["fit_every_matrix_relative_frobenius_max"],
            "basis": "Exact independent-canonical-factor lower bound, assessed before any shared-head fit.",
            "failure_scope": "Excludes the sealed weight-approximation tolerance under these fixed alignments, not behavior or other alignments.",
        },
    )
    LOGGER.info(
        "REPRESENTABILITY_BOUND aggregate=%g worst=%g fit_eligible=%s",
        summary["relative_frobenius_error"],
        summary["maximum_matrix_relative_frobenius_error"],
        eligible,
    )
    witness_path = output / "canonical.safetensors"
    save_file(
        {
            f"{path}.{factor}": pair[index].contiguous()
            for path, pair in canonical.items()
            for index, factor in enumerate(("A", "B"))
        },
        witness_path,
        metadata={"format": "independent_canonical_factors", "dtype": "float64"},
    )
    witness = load_file(witness_path, device="cpu")
    decoded, witness_errors = {}, []
    for target, path in targets:
        require(
            torch.equal(witness[f"{path}.A"], canonical[path][0])
            and torch.equal(witness[f"{path}.B"], canonical[path][1]),
            f"CANONICAL_RELOAD: {path}",
        )
        term = spectrum.FactorTerm(
            witness[f"{path}.A"] @ model.alignment.input[target.input_group],
            model.alignment.output[target.output_group] @ witness[f"{path}.B"],
            model.config.scaling,
        )
        error = (
            math.sqrt(distance(term, fitted[path]) / energy(fitted[path]))
            if energy(fitted[path])
            else math.sqrt(energy(term))
        )
        require(
            error <= gates["projection_witness_relative_frobenius_max"],
            f"CANONICAL_WITNESS: {path}",
        )
        decoded[path] = term
        witness_errors.append({"path": path, "relative_frobenius": error})
    write_json(
        output / "canonical-reload.json",
        {
            "exact_reload": True,
            "rows": witness_errors,
            "maximum_relative_frobenius": max(
                row["relative_frobenius"] for row in witness_errors
            ),
            "scope": "Independent canonical factors, not a shared-generator checkpoint.",
        },
    )
    projected_probe = serialization_probe(
        model.config, source_config, decoded, output / "projected-adapter", config
    )
    fit = {
        "status": "not_run_lower_bound_exceeds_sealed_tolerance",
        "linear_solves": 0,
        "optimizer_steps": 0,
        "native_candidate": None,
    }
    if eligible:
        fit = fit_shared_heads(model, canonical, config)
        fit["native_candidate"] = save_native(model, output / "shared-native")
        candidate = load_native(output / "shared-native")
        generated = generated_terms(candidate, config["fit"]["task"])
        candidate_rows = [
            {
                "path": path,
                **spectrum.residual_statistics(
                    energy(final[path]), distance(final[path], generated[path])
                ),
            }
            for _, path in targets
        ]
        fit["weight_error"] = aggregate(candidate_rows)
        fit["reloaded_weight_tolerance_passed"] = fit_gate(fit["weight_error"], gates)
        fit["injection_passed"] = serialization_probe(
            candidate.config,
            source_config,
            generated,
            output / "shared-adapter",
            config,
        )["passed"]
    write_json(output / "shared-fit.json", fit)
    return {
        "status": "completed",
        "identity": identity,
        "projection": summary,
        "learned_change": change_summary,
        "fit": fit,
        "all72_initial_control_passed": len(initial_rows) == 72,
        "projection_injection_passed": projected_probe["passed"],
        "claim_boundary": config["claim_boundary"],
        "next_interface": {
            "native_checkpoint": fit["native_candidate"],
            "independent_projection_adapter": str(output / "projected-adapter"),
            "initial_positive_control": str(positive_path / "adapter"),
            "remote_launch_required": False,
            "gpu_behavior_required_before_behavioral_claims": True,
            "interpretation": "The projected adapter is an independent-canonical-factor oracle. Only a qualified shared-native checkpoint would witness the tested shared-generator construction. No behavioral qualification is provided.",
        },
    }


def run(protocol_path: Path, expected_hash: str) -> dict:
    protocol = verify_protocol(protocol_path, expected_hash)
    output = protocol_path.parent
    start = time.monotonic()
    write_json(
        output / "started.json",
        {
            "protocol_sha256": expected_hash,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime(),
        },
    )
    config = protocol["configuration"]
    torch.set_num_threads(config["numerics"]["cpu_threads"])
    torch.manual_seed(config["numerics"]["seed"])
    torch.use_deterministic_algorithms(True)
    try:
        result = study(config, output)
        verify_protocol(protocol_path, expected_hash)
        result.update(
            {
                "protocol_sha256": expected_hash,
                "elapsed_seconds": time.monotonic() - start,
                "input_pins_unchanged": True,
            }
        )
        write_json(output / "result.json", result)
        write_json(
            output / "artifacts.json",
            [pin(path) for path in sorted(output.rglob("*")) if path.is_file()],
        )
    except Exception as exc:
        LOGGER.exception("REPRESENTABILITY_STUDY_FAILED")
        write_json(
            output / "failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - start,
                "protocol_sha256": expected_hash,
                "impossibility_claim": False,
            },
        )
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("seal")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--sha256", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output = args.output if args.command == "seal" else args.protocol.parent
    require(
        output.resolve().is_relative_to(REPO / "outputs/representability"),
        "OUTPUT_OUTSIDE_OWNED_LOCAL_DIRECTORY",
    )
    if args.command == "seal":
        print(
            json.dumps(
                {
                    "protocol": str((args.output / "protocol.json").resolve()),
                    "sha256": seal(args.config, args.output.resolve()),
                }
            )
        )
    else:
        print(
            json.dumps(
                run(args.protocol.resolve(), args.sha256),
                sort_keys=True,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
