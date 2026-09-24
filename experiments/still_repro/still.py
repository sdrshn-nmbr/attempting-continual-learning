import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

COMPACTOR_ROPE_BASE = 10.0


def rotate_half(values):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotary(values, cos, sin, inverse=False):
    while cos.ndim < values.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    if inverse:
        return values * cos - rotate_half(values) * sin
    return values * cos + rotate_half(values) * sin


def phase(positions, dimension, dtype):
    frequencies = COMPACTOR_ROPE_BASE ** (
        -torch.arange(0, dimension, 2, device=positions.device, dtype=torch.float32) / dimension)
    angles = positions.float().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rms(values):
    return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)


class CompactorBlock(nn.Module):
    def __init__(self, dimension, first):
        super().__init__()
        self.dimension = dimension
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)
        self.self_query = nn.Linear(dimension, dimension)
        self.self_key = nn.Linear(dimension, dimension)
        self.self_value = nn.Linear(dimension, dimension, bias=False)
        self.self_output = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            if first:
                self.query.weight.copy_(torch.eye(dimension))
                self.key.weight.zero_()
                self.query.bias.fill_(1 / math.sqrt(dimension))
                self.key.bias.fill_(1 / math.sqrt(dimension))
                self.value.weight.copy_(torch.eye(dimension))
                self.output.weight.copy_(torch.eye(dimension))
            else:
                self.output.weight.zero_()
            self.self_output.weight.zero_()

    def forward(self, latents, inputs, input_positions, latent_positions):
        queries = F.normalize(self.query(rms(latents)), dim=-1)
        keys = F.normalize(self.key(inputs), dim=-1)
        qcos, qsin = phase(latent_positions, self.dimension, queries.dtype)
        kcos, ksin = phase(input_positions, self.dimension, keys.dtype)
        queries = rotary(queries, qcos, qsin)
        keys = rotary(keys, kcos, ksin)
        weights = torch.softmax(self.dimension * queries @ keys.transpose(-1, -2), dim=-1)
        latents = latents + self.output(weights @ self.value(inputs))
        normalized = rms(latents)
        queries = F.normalize(self.self_query(normalized), dim=-1)
        keys = F.normalize(self.self_key(normalized), dim=-1)
        weights = torch.softmax(self.dimension * queries @ keys.transpose(-1, -2), dim=-1)
        return latents + self.self_output(weights @ self.self_value(normalized))


class LayerCompactor(nn.Module):
    def __init__(self, heads, head_dim, slots):
        super().__init__()
        dimension = 2 * head_dim
        self.latents = nn.Parameter(torch.zeros(heads, slots, dimension))
        self.blocks = nn.ModuleList([CompactorBlock(dimension, index == 0) for index in range(2)])
        self.key_output = nn.Linear(dimension, head_dim, bias=False)
        self.value_output = nn.Linear(dimension, head_dim, bias=False)
        with torch.no_grad():
            self.key_output.weight.zero_()
            self.value_output.weight.zero_()
            self.key_output.weight[:, :head_dim].copy_(torch.eye(head_dim))
            self.value_output.weight[:, head_dim:].copy_(torch.eye(head_dim))

    def forward(self, keys, values, input_positions, latent_positions):
        inputs = torch.cat((keys, values), dim=-1)
        latents = self.latents.unsqueeze(0).expand(keys.shape[0], -1, -1, -1)
        for block in self.blocks:
            latents = block(latents, inputs, input_positions, latent_positions)
        return self.key_output(latents), self.value_output(latents)


class StillCompactor(nn.Module):
    """Per-layer Perceiver compactor: two blocks with repeated cross-attention, one head each, no FFN,
    latent width 2*head_dim, identity-style initialization, RoPE-fix, compactor RoPE base 10."""

    def __init__(self, model_config, slots):
        super().__init__()
        self.slots = slots
        self.layers = nn.ModuleList([
            LayerCompactor(model_config.num_key_value_heads, model_config.head_dim, slots)
            for _ in range(model_config.num_hidden_layers)])

    def forward(self, model, pairs):
        if len(pairs) != len(self.layers):
            raise ValueError(f"STILL_LAYER_COUNT pairs={len(pairs)} layers={len(self.layers)}")
        length = pairs[0][0].shape[-2]
        device = pairs[0][0].device
        positions = torch.arange(length, device=device).unsqueeze(0)
        latent_positions = torch.linspace(0.0, float(length - 1), self.slots, device=device).unsqueeze(0)
        dummy = torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=pairs[0][0].dtype)
        cos, sin = model.model.rotary_emb(dummy, positions)
        outcos, outsin = model.model.rotary_emb(dummy, latent_positions)
        output = []
        for layer, (keys, values) in zip(self.layers, pairs, strict=True):
            plain = rotary(keys.float(), cos.float(), sin.float(), inverse=True)
            compact_keys, compact_values = layer(plain, values.float(), positions, latent_positions)
            compact_keys = rotary(compact_keys, outcos.float(), outsin.float())
            output.append((compact_keys.to(keys.dtype), compact_values.to(values.dtype)))
        return output


def cache_pairs(cache):
    return [(layer.keys, layer.values) for layer in cache.layers]


def prefill(model, prefix_ids):
    with torch.no_grad():
        result = model(input_ids=prefix_ids, use_cache=True)
    return cache_pairs(result.past_key_values)


def build_cache(pairs):
    cache = DynamicCache()
    for index, (keys, values) in enumerate(pairs):
        cache.update(keys, values, index)
    return cache


def streaming_pairs(pairs, budget, sinks=4):
    length = pairs[0][0].shape[-2]
    head = min(sinks, budget)
    keep = torch.cat((torch.arange(head), torch.arange(length - (budget - head), length))).to(pairs[0][0].device)
    return [(keys[:, :, keep], values[:, :, keep]) for keys, values in pairs]


def continue_from(model, pairs, input_ids, logical_start, attention_mask=None, position_ids=None):
    """Run input_ids after a possibly compacted prefix cache. The physical cache can be shorter than
    logical_start; RoPE positions of new tokens continue from the uncompacted prefix length."""
    if position_ids is None:
        length = input_ids.shape[1]
        position_ids = torch.arange(logical_start, logical_start + length,
                                    device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
    return model(input_ids=input_ids, past_key_values=build_cache(pairs), position_ids=position_ids,
                 attention_mask=attention_mask, use_cache=True)


def support_kl(teacher_logits, student_logits, gold, top_k=200):
    """Forward KL(teacher || student) restricted to the teacher's top-k tokens plus the gold token. Both
    distributions are renormalized over that support, so the loss is zero exactly when they agree on it."""
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    support = torch.zeros_like(teacher_logp, dtype=torch.bool)
    support.scatter_(-1, teacher_logp.topk(top_k, dim=-1).indices, True)
    support.scatter_(-1, gold.unsqueeze(-1), True)
    teacher_logq = torch.log_softmax(teacher_logp.masked_fill(~support, float("-inf")), dim=-1)
    student_logq = torch.log_softmax(student_logits.float().masked_fill(~support, float("-inf")), dim=-1)
    difference = (teacher_logq - student_logq).masked_fill(~support, 0.0)
    return (teacher_logq.exp() * difference).sum(-1)
