import torch


def consensus_fusion(task_vectors):
    if not task_vectors or any(value.shape != task_vectors[0].shape for value in task_vectors):
        raise ValueError("DICE_SHAPES: nonempty, equal-shaped task vectors are required")
    if any(value.dtype not in (torch.float32, torch.float64) for value in task_vectors):
        raise ValueError("DICE_PRECISION: compute effective updates in FP32 or FP64")
    vectors = torch.stack(task_vectors)
    if not torch.isfinite(vectors).all():
        raise ValueError("DICE_NONFINITE")
    positive = vectors >= 0
    votes = positive.sum(dim=0)
    count = len(task_vectors)
    keep = torch.where(votes * 2 > count, positive, torch.where(votes * 2 < count, ~positive, True))
    weights = torch.softmax(vectors.abs().masked_fill(~keep, -torch.inf), dim=0)
    fused = (weights * vectors).sum(dim=0)
    return fused, {
        "tasks": count,
        "coordinates": fused.numel(),
        "tie_coordinates": int((votes * 2 == count).sum()),
        "pruned_contributions": int((~keep).sum()),
        "max_weight_sum_error": float((weights.sum(dim=0) - 1).abs().max()),
    }


def effective_delta(state, name, scaling):
    value = scaling * (state[name]["B"] @ state[name]["A"])
    return value if state[name]["offset"] is None else state[name]["offset"] + value


def fuse_experts(experts, scaling, method="agent_dice"):
    if not experts:
        raise ValueError("DICE_NO_EXPERTS")
    if method not in {"agent_dice", "arithmetic"}:
        raise ValueError("UNKNOWN_FUSION")
    names = set(experts[0])
    if any(set(expert) != names for expert in experts):
        raise ValueError("DICE_EXPERT_KEYS")
    fused, audits = {}, {}
    for name in sorted(names):
        if any(expert[name]["offset"] is not None for expert in experts):
            raise ValueError("DICE_REFERENCE: experts must start from the same unmodified base")
        updates = [effective_delta(expert, name, scaling) for expert in experts]
        if method == "agent_dice":
            value, audits[name] = consensus_fusion(updates)
        else:
            value = torch.stack(updates).mean(dim=0)
            audits[name] = {"tasks": len(experts), "coordinates": value.numel(), "equal_weight": 1 / len(experts)}
        fused[name] = {
            "A": experts[0][name]["A"].clone(),
            "B": torch.zeros_like(experts[0][name]["B"]),
            "offset": value,
        }
    return fused, audits
