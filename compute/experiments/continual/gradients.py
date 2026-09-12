import math

import torch


def choose_gradient(method, current, reference, replay_weight, max_grad_norm):
    if not torch.isfinite(current).all() or (
        reference is not None and not torch.isfinite(reference).all()
    ):
        raise FloatingPointError("[gradient] non-finite input gradient")
    current_norm = torch.linalg.vector_norm(current).item()
    record = {
        "current_norm": current_norm,
        "reference_norm": None,
        "dot_before": None,
        "cosine": None,
        "conflict": None,
        "projected": False,
        "dot_after": None,
    }
    update = current.clone()
    if method not in {"sequential", "replay", "agem"}:
        raise ValueError(f"[gradient] unknown method {method}")
    if reference is not None:
        dot = torch.dot(current.double(), reference.double()).item()
        ref_sq = torch.dot(reference.double(), reference.double()).item()
        ref_norm = math.sqrt(ref_sq)
        record.update(
            reference_norm=ref_norm,
            dot_before=dot,
            conflict=dot < 0,
            cosine=dot / (current_norm * ref_norm)
            if current_norm * ref_norm > 0
            else None,
        )
        if method == "replay":
            update.mul_(1 - replay_weight).add_(reference, alpha=replay_weight)
        elif method == "agem" and dot < 0 and ref_sq > 0:
            update.add_(reference, alpha=-dot / ref_sq)
            record["projected"] = True
    unclipped = torch.linalg.vector_norm(update).item()
    scale = min(1.0, max_grad_norm / max(unclipped, 1e-30))
    update.mul_(scale)
    if reference is not None:
        after = torch.dot(update.double(), reference.double()).item()
        record["dot_after"] = after
        tolerance = 1e-6 * max(
            torch.linalg.vector_norm(update).item() * record["reference_norm"], 1e-12
        )
        if method == "agem" and after < -tolerance:
            raise FloatingPointError(
                f"[projection] infeasible SGD direction: dot={after}, tolerance={tolerance}"
            )
    if not torch.isfinite(update).all():
        raise FloatingPointError("[gradient] non-finite selected update")
    record.update(
        unclipped_update_norm=unclipped,
        clip_scale=scale,
        update_norm=torch.linalg.vector_norm(update).item(),
    )
    return update, record


def assign_gradient(parameters, flat):
    if sum(parameter.numel() for parameter in parameters) != flat.numel():
        raise ValueError("[gradient] adapter vector length mismatch")
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].view_as(parameter).clone()
        offset += count
