"""Evaluate KV caches compacted by the official Attention Matching code (compact_official.py). The official
output per layer is (C1 keys, beta, C2 values); beta is an additive attention bias on the compacted columns, applied
here through a registered attention function. Per-head budgets may differ, so heads are padded with beta=-inf."""
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask

import cache_format

NAME = "still_bias_sdpa"


class State:
    bias = None


STATE = State()


def causal_bias(q_length, k_length, device, dtype):
    rows = torch.arange(q_length, device=device).unsqueeze(1) + (k_length - q_length)
    allowed = torch.arange(k_length, device=device).unsqueeze(0) <= rows
    return torch.zeros(q_length, k_length, device=device, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)


def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    layer = module.layer_idx
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


def cache_path(directory, domain, row):
    return Path(directory) / f"{domain}-{row}.pt"


def load(directory, domain, row, device):
    path = cache_path(directory, domain, row)
    if not path.exists():
        raise FileNotFoundError(f"AM_CACHE_MISSING {path}")
    saved = torch.load(path, map_location=device)
    return cache_format.unpack(saved)


def batch(compacted, dtype):
    """Stack per-document (keys [L,H,t,D], beta [L,H,t], values) into per-layer batched pairs and biases, padding
    shorter documents with zero keys/values and beta=-inf so padded columns receive no attention."""
    length = max(keys.shape[2] for keys, _, _ in compacted)
    pairs, betas = [], []
    for layer in range(compacted[0][0].shape[0]):
        layer_keys, layer_values, layer_beta = [], [], []
        for keys, beta, values in compacted:
            pad = length - keys.shape[2]
            layer_keys.append(F.pad(keys[layer], (0, 0, 0, pad)))
            layer_values.append(F.pad(values[layer], (0, 0, 0, pad)))
            layer_beta.append(F.pad(beta[layer].float(), (0, pad), value=float("-inf")))
        pairs.append((torch.stack(layer_keys).to(dtype), torch.stack(layer_values).to(dtype)))
        betas.append(torch.stack(layer_beta))
    return pairs, betas
