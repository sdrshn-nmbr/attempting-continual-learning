from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CausalExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: torch.Tensor
    targets: torch.Tensor


def completion_mask(tokens, eos_ids, pad_id):
    if tokens.ndim != 1:
        raise ValueError("A completion must be a one-dimensional token sequence")
    keep = torch.ones_like(tokens, dtype=torch.bool)
    ended = False
    for index, token in enumerate(tokens.tolist()):
        if ended or (token == pad_id and token not in eos_ids):
            keep[index] = False
            ended = True
        elif token in eos_ids:
            ended = True
    return keep


def causal_example(prefix, completion):
    if prefix.ndim != 1 or completion.ndim != 1 or not len(prefix) or not len(completion):
        raise ValueError("Causal scoring requires nonempty one-dimensional prefix and completion")
    if prefix.device != completion.device:
        raise ValueError("Prefix and completion must be on the same device")
    inputs = torch.cat((prefix, completion)).unsqueeze(0)
    positions = torch.arange(
        len(prefix) - 1, len(prefix) + len(completion) - 1, device=prefix.device
    )
    return CausalExample(inputs, torch.ones_like(inputs), positions, completion)


def response_logits(model, prefix, completion):
    batch = causal_example(prefix, completion)
    logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        logits_to_keep=batch.prediction_positions,
        use_cache=False,
    ).logits[0]
    if logits.shape[0] != len(completion):
        raise RuntimeError("Model returned incorrectly aligned continuation logits")
    return logits


def dense_reverse_kl(student_logits, teacher_logits, mask):
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != mask.shape:
        raise ValueError(
            "KL requires aligned student/teacher positions and a matching response mask"
        )
    if not bool(mask.any()):
        raise ValueError("KL received no supervised completion tokens")
    student = student_logits[mask].float().log_softmax(dim=-1)
    teacher = teacher_logits[mask].detach().float().log_softmax(dim=-1)
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise FloatingPointError("Non-finite token log probabilities in dense KL")
    return (student.exp() * (student - teacher)).sum()


def oracle_nll(logits, targets, mask):
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape or not bool(mask.any()):
        raise ValueError("Oracle NLL requires aligned, nonempty response targets")
    return F.cross_entropy(logits[mask].float(), targets[mask], reduction="sum")
