import json
from pathlib import Path

import torch
import torch.nn.functional as F
from portallib import PortalConfig, PortalModel
from safetensors.torch import load_file, save_file

from calibrate_target import assert_fp32, pin_bundle, verify_bundle
from data import SEQUENCE_TASKS, write_json
from learner import tensor_hash
from preserve_tasks import OLD_TASKS
from repair_alignment import product_squared_error

CONTRACT = {
    "method": "frozen_feature_head_nullspace",
    "old_tasks": list(OLD_TASKS),
    "parameterization": "augmented_affine_head_W_equals_W0_plus_U_at_N_transpose",
    "bias_in_constraint": True,
    "rank_threshold": "max_shape_times_float64_epsilon_times_largest_singular_value",
    "svd_device": "cpu",
    "basis_dtype": "float64",
    "projection_dtype": "float64",
    "head_and_learner_dtype": "float32",
    "null_residual_relative_limit": 1e-12,
    "basis_orthogonality_max_abs_limit": 1e-12,
    "minimum_null_dimension": "number_of_layers",
    "minimum_ABC_projected_rank": "number_of_layers",
    "minimum_ABC_projected_relative_norm": 1e-6,
    "factor_relative_frobenius_limit": 1e-6,
    "factor_max_abs_limit": 1e-5,
    "matrix_relative_frobenius_limit": 2e-6,
    "control": "all_SVD_directions_with_null_directions_first",
    "protected_arm_prerequisite": "control_final_train_and_validation_qualified",
    "freeze": [
        "alignment",
        "layer_embeddings",
        "hidden_core",
        "original_heads",
        "original_task_vectors",
    ],
    "train": ["affine_head_update_coordinates", "three_ABC_vectors"],
    "old_behavioral_training_examples": 0,
}


def hidden_features(source, vector):
    ids = torch.arange(source.config.n_layers, device=vector.device)
    hidden = source.core.hidden(vector, source.alignment.layer_embeddings(ids))
    return F.pad(hidden, (0, 1), value=1).double()


def matrix_rank(matrix):
    singular = torch.linalg.svdvals(matrix)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * singular[0]
    return int((singular > tolerance).sum()), float(tolerance), singular


def feature_space(source):
    source.requires_grad_(False).eval()
    with torch.no_grad():
        design = torch.cat(
            [
                hidden_features(
                    source, source.task_latents[source.config.tasks.index(task)]
                )
                for task in OLD_TASKS
            ]
        ).cpu()
        if not torch.isfinite(design).all():
            raise ValueError("HEAD_NULLSPACE_NONFINITE_OLD_FEATURES")
        _, singular, vh = torch.linalg.svd(design, full_matrices=True)
        tolerance = max(design.shape) * torch.finfo(torch.float64).eps * singular[0]
        rank = int((singular > tolerance).sum())
        null = vh[rank:].T.contiguous()
        if null.shape[1] < source.config.n_layers:
            raise ValueError("HEAD_NULLSPACE_INSUFFICIENT_NULL_DIMENSION")
        full = torch.cat((null, vh[:rank].T), dim=1).contiguous()
        residual = float(torch.linalg.norm(design @ null) / torch.linalg.norm(design))
        orthogonality = float(
            (full.T @ full - torch.eye(full.shape[1], dtype=full.dtype)).abs().max()
        )
        if (
            residual > CONTRACT["null_residual_relative_limit"]
            or orthogonality > CONTRACT["basis_orthogonality_max_abs_limit"]
        ):
            raise ValueError("HEAD_NULLSPACE_BASIS_NUMERICAL_GATE_FAILED")
        projected = {}
        for task in SEQUENCE_TASKS:
            features = hidden_features(
                source, source.task_latents[source.config.tasks.index(task)]
            ).cpu()
            projected_features = features @ null
            projected_rank, _, spectrum = matrix_rank(projected_features)
            ratio = float(
                torch.linalg.norm(projected_features) / torch.linalg.norm(features)
            )
            projected[task] = {
                "rank": projected_rank,
                "relative_norm": ratio,
                "singular_min": float(spectrum[-1]),
            }
            if (
                projected_rank < source.config.n_layers
                or ratio < CONTRACT["minimum_ABC_projected_relative_norm"]
            ):
                raise ValueError(
                    f"HEAD_NULLSPACE_INSUFFICIENT_NEW_TASK_FEATURES: {task}"
                )
        output_rows = sum(
            head.out_features
            for heads in (source.core.A, source.core.B)
            for head in heads.values()
        )
    proof = {
        "source_tensor_sha256": tensor_hash(source.state_dict()),
        "old_design_shape": list(design.shape),
        "old_design_tensor_sha256": tensor_hash({"features": design}),
        "rank": rank,
        "rank_tolerance": float(tolerance),
        "null_dimension": null.shape[1],
        "singular_max": float(singular[0]),
        "singular_min_nonzero": float(singular[rank - 1]),
        "null_relative_residual": residual,
        "full_basis_orthogonality_max_abs": orthogonality,
        "ABC_projected_features": projected,
        "trainable_parameters": {
            name: output_rows * width + len(SEQUENCE_TASKS) * source.config.d_z
            for name, width in (
                ("head_only_control", full.shape[1]),
                ("head_nullspace", null.shape[1]),
            )
        },
        "basis_bytes_per_learner": {
            "head_only_control": full.numel() * full.element_size(),
            "head_nullspace": null.numel() * null.element_size(),
        },
        "old_features_are_fixed_only_while_core_alignment_and_old_vectors_are_frozen": True,
    }
    return {"old_design": design, "full_basis": full, "null_basis": null}, proof


class HeadSpacePortal(PortalModel):
    def __init__(self, source, basis):
        source.requires_grad_(False)
        super().__init__(
            source.config,
            source.task_latents,
            core=source.core,
            alignment=source.alignment,
        )
        self.requires_grad_(False)
        self.register_buffer("head_basis", basis.detach().clone())
        self.coordinates = torch.nn.ParameterDict(
            {
                f"{factor}_{name}": torch.nn.Parameter(
                    torch.zeros(
                        head.out_features,
                        basis.shape[1],
                        device=source.task_latents.device,
                        dtype=torch.float32,
                    )
                )
                for factor, heads in (("A", self.core.A), ("B", self.core.B))
                for name, head in heads.items()
            }
        )

    def forward(self, vector):
        if vector.shape != (self.config.d_z,):
            raise ValueError("HEAD_NULLSPACE_WRONG_VECTOR_SHAPE")
        features = hidden_features(self, vector)
        hidden = features[:, :-1].float()
        projected = (features @ self.head_basis).float()
        generated = {}
        for target in self.config.projection_targets:
            index, module = target.layer_index, target.module_name
            a = self.core.A[module](hidden[index]) + F.linear(
                projected[index], self.coordinates[f"A_{module}"]
            )
            b = self.core.B[module](hidden[index]) + F.linear(
                projected[index], self.coordinates[f"B_{module}"]
            )
            generated[target.key] = (
                a.view(self.config.rank, self.config.d_core)
                @ self.alignment.input[target.input_group],
                self.alignment.output[target.output_group]
                @ b.view(self.config.d_core, self.config.rank),
            )
        return generated


def trainable_plan(adapter, arm):
    adapter.requires_grad_(False)
    adapter.coordinates.requires_grad_(True)
    vectors = torch.nn.ParameterDict(
        {
            task: torch.nn.Parameter(
                adapter.task_latents[adapter.config.tasks.index(task)].detach().clone()
            )
            for task in SEQUENCE_TASKS
        }
    )
    named = {
        f"core.{key.split('_', 1)[0]}.{key.split('_', 1)[1]}.coordinates": parameter
        for key, parameter in adapter.coordinates.items()
    }
    groups = [
        {"params": list(named.values()), "lr": arm["learning_rate"]},
        {"params": list(vectors.parameters()), "lr": arm["latent_learning_rate"]},
    ]
    named.update({f"latent.{task}": vector for task, vector in vectors.items()})
    assert_fp32(named)
    return named, groups, vectors


def conservation(adapter, reference):
    results = {}
    with torch.no_grad():
        for task in OLD_TASKS:
            generated = adapter(reference.vectors[task])
            factors = []
            matrices = []
            for key, original in reference.factors[task].items():
                actual = generated[key]
                for before, after in zip(original, actual, strict=True):
                    residual = after.double() - before.double()
                    factors.append(
                        {
                            "max_abs": float(residual.abs().max()),
                            "relative": float(
                                torch.linalg.norm(residual)
                                / torch.linalg.norm(before.double()).clamp_min(1e-12)
                            ),
                            "bitwise_equal": torch.equal(
                                before.contiguous().view(torch.uint8),
                                after.contiguous().view(torch.uint8),
                            ),
                        }
                    )
                error = product_squared_error(*actual, *original) * reference.scale**2
                if not torch.isfinite(error) or float(error) < -1e-10:
                    raise FloatingPointError("HEAD_NULLSPACE_INVALID_MATRIX_ERROR")
                matrices.append(
                    float((error.clamp_min(0) / reference.energies[task][key]).sqrt())
                )
            results[task] = {
                "factor_max_abs": max(row["max_abs"] for row in factors),
                "factor_relative_frobenius_max": max(
                    row["relative"] for row in factors
                ),
                "all_factors_bitwise_equal": all(
                    row["bitwise_equal"] for row in factors
                ),
                "matrix_relative_frobenius_max": max(matrices),
                "projections": len(matrices),
            }
    values = [
        value
        for row in results.values()
        for key, value in row.items()
        if key != "all_factors_bitwise_equal"
    ]
    if not torch.isfinite(torch.tensor(values)).all():
        raise FloatingPointError("HEAD_NULLSPACE_NONFINITE_CONSERVATION_AUDIT")
    passed = all(
        row["factor_max_abs"] <= CONTRACT["factor_max_abs_limit"]
        and row["factor_relative_frobenius_max"]
        <= CONTRACT["factor_relative_frobenius_limit"]
        and row["matrix_relative_frobenius_max"]
        <= CONTRACT["matrix_relative_frobenius_limit"]
        for row in results.values()
    )
    return {"passed": passed, "tasks": results}


def save_adapter(adapter, output):
    output.mkdir(parents=True)
    write_json(output / "config.json", adapter.config.to_dict())
    write_json(
        output / "method.json", {"kind": CONTRACT["method"], "contract": CONTRACT}
    )
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in adapter.state_dict().items()
        },
        output / "model.safetensors",
        metadata={"format": CONTRACT["method"]},
    )
    return {
        "kind": CONTRACT["method"],
        "artifact": pin_bundle(output),
        "tensor_sha256": tensor_hash(adapter.state_dict()),
    }


def restore_adapter(saved, device):
    if saved["kind"] != CONTRACT["method"]:
        raise ValueError("HEAD_NULLSPACE_WRONG_ARTIFACT_KIND")
    verify_bundle(saved["artifact"])
    root = Path(saved["artifact"]["path"])
    if json.loads((root / "method.json").read_text())["contract"] != CONTRACT:
        raise ValueError("HEAD_NULLSPACE_SAVED_CONTRACT_CHANGED")
    config = PortalConfig.from_dict(json.loads((root / "config.json").read_text()))
    state = load_file(root / "model.safetensors")
    source = PortalModel(config, state["task_latents"])
    adapter = HeadSpacePortal(source, state["head_basis"])
    adapter.load_state_dict(state, strict=True)
    if tensor_hash(adapter.state_dict()) != saved["tensor_sha256"]:
        raise ValueError("HEAD_NULLSPACE_RELOAD_TENSOR_MISMATCH")
    return adapter.to(device).requires_grad_(False).eval()
