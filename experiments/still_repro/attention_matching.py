"""Attention Matching baseline (Zweiger et al., 2026) as configured in the STILL paper's comparison:
repeat-prefill reference queries, top-k key selection by RMS attention score, nonnegative least-squares
bias (beta) fitting, and least-squares value reconstruction, per layer and KV head."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask

import layout
from still import prefill

NAME = "still_bias_sdpa"
REPEAT = "\n</document>\n\nRepeat the document above exactly.<|im_end|>\n<|im_start|>assistant\n"


class State:
    capture = None
    capture_positions = None
    bias = None


STATE = State()


def causal_bias(q_length, k_length, device, dtype):
    rows = torch.arange(q_length, device=device).unsqueeze(1) + (k_length - q_length)
    allowed = torch.arange(k_length, device=device).unsqueeze(0) <= rows
    return torch.zeros(q_length, k_length, device=device, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)


def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    layer = module.layer_idx
    if STATE.capture is not None:
        STATE.capture[layer] = query[:, :, STATE.capture_positions].detach().float()
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    q_length, k_length = query.shape[2], key.shape[2]
    mask = attention_mask[..., :k_length] if attention_mask is not None else \
        causal_bias(q_length, k_length, query.device, query.dtype)[None, None]
    if STATE.bias is not None:
        beta = STATE.bias[layer].repeat_interleave(groups, dim=1).to(query.dtype)
        extra = torch.zeros(beta.shape[0], beta.shape[1], 1, k_length, device=query.device, dtype=query.dtype)
        extra[..., :beta.shape[-1]] = beta.unsqueeze(2)
        mask = mask + extra
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
    return output.transpose(1, 2).contiguous(), None


AttentionInterface.register(NAME, attention)
AttentionMaskInterface.register(NAME, eager_mask)


def fit_head(queries, keys, values, budget, scale, rows_for_nnls):
    logits = queries @ keys.T * scale
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    probabilities = torch.exp(logits - lse)
    target = probabilities @ values
    scores = probabilities.square().mean(0).sqrt()
    selected = scores.topk(budget).indices.sort().values
    mass = torch.exp(logits[:, selected] - lse)
    rows = torch.linspace(0, queries.shape[0] - 1, min(rows_for_nnls, queries.shape[0]), device=queries.device).long()
    weights, _ = nnls(mass[rows].double().cpu().numpy(), np.ones(len(rows)), maxiter=20 * budget)
    weights = torch.tensor(weights, device=queries.device, dtype=torch.float32).clamp_min(1e-8)
    blend = mass * weights
    blend = blend / blend.sum(1, keepdim=True)
    gram = blend.T @ blend
    ridge = 1e-4 * gram.diagonal().mean() * torch.eye(budget, device=queries.device)
    compact_values = torch.linalg.solve(gram + ridge, blend.T @ target)
    return selected, compact_values, torch.log(weights)


@torch.no_grad()
def compact(model, tokenizer, document_ids, budget, reference_queries=1024, rows_for_nnls=1024):
    prefix = layout.prefix_ids(tokenizer, document_ids)
    repeat = layout.encode(tokenizer, REPEAT)
    sequence = prefix + repeat + list(document_ids[:len(prefix)])
    start = len(prefix) + len(repeat)
    STATE.capture = {}
    STATE.capture_positions = torch.linspace(start, len(sequence) - 1, reference_queries,
                                             device=model.device).long()
    try:
        pairs = prefill(model, torch.tensor([sequence], device=model.device))
        captured = STATE.capture
    finally:
        STATE.capture = None
        STATE.capture_positions = None
    heads, kv_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    groups = heads // kv_heads
    scale = model.config.head_dim ** -0.5
    compact_pairs, betas = [], []
    for layer, (keys, values) in enumerate(pairs):
        keys, values = keys[0, :, :len(prefix)].float(), values[0, :, :len(prefix)].float()
        queries = captured[layer][0]
        layer_keys, layer_values, layer_beta = [], [], []
        for head in range(kv_heads):
            group = queries[head * groups:(head + 1) * groups].reshape(-1, queries.shape[-1])
            selected, compact_values, beta = fit_head(group, keys[head], values[head], budget, scale, rows_for_nnls)
            layer_keys.append(keys[head, selected])
            layer_values.append(compact_values)
            layer_beta.append(beta)
        compact_pairs.append((torch.stack(layer_keys)[None].to(model.dtype),
                              torch.stack(layer_values)[None].to(model.dtype)))
        betas.append(torch.stack(layer_beta)[None])
    return compact_pairs, betas
