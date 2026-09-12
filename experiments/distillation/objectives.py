from torch.nn import functional


def distillation_loss(
    student_logits, teacher_logits, response_ids, method, rollout_logprobs=None
):
    student = functional.log_softmax(student_logits.float(), dim=-1)
    teacher = functional.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_taken = student.gather(-1, response_ids[:, None]).squeeze(-1)
    teacher_taken = teacher.gather(-1, response_ids[:, None]).squeeze(-1)
    forward = (teacher.exp() * (teacher - student)).sum(-1)
    reverse = (student.exp() * (student - teacher)).sum(-1)
    sampled_ratio = student_taken.detach() - teacher_taken
    if method == "sdft_forward":
        loss = forward.mean()
    elif method == "sdpo_sampled_reverse":
        if rollout_logprobs is None:
            raise ValueError("DISTILL_MISSING_ROLLOUT_LOGPROBS")
        importance = (student_taken.detach() - rollout_logprobs).exp()
        loss = (importance * sampled_ratio * student_taken).mean()
    else:
        raise ValueError(f"unknown distillation method: {method}")
    diagnostics = {
        "forward_kl": float(forward.detach().mean()),
        "reverse_kl": float(reverse.detach().mean()),
        "sampled_log_ratio": sampled_ratio.detach().cpu().tolist(),
        "student_logprobs": student_taken.detach().cpu().tolist(),
        "teacher_logprobs": teacher_taken.detach().cpu().tolist(),
        "student_entropy": float(
            -(student.detach().exp() * student.detach()).sum(-1).mean()
        ),
        "teacher_entropy": float(-(teacher.exp() * teacher).sum(-1).mean()),
        "support": "full_vocabulary",
        "vocabulary_size": student.shape[-1],
    }
    if rollout_logprobs is not None:
        ratios = (student_taken.detach() - rollout_logprobs).exp()
        diagnostics.update(
            {
                "rollout_to_forward_ratio_min": float(ratios.min()),
                "rollout_to_forward_ratio_max": float(ratios.max()),
                "rollout_to_forward_logprob_max_error": float(
                    (student_taken.detach() - rollout_logprobs).abs().max()
                ),
                "ratio_clipped_fraction": 0.0,
                "advantage_clipped_fraction": 0.0,
                "staleness_updates": 0,
            }
        )
    return loss, diagnostics
