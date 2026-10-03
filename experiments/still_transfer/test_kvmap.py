import sys
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "still_repro"))
import kvmap
from refit import Mapper, step_loss
from still import StillCompactor, continue_from, prefill

PREFIX = 24


def tiny(theta, seed=0, layers=2):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         rope_theta=theta)
    return Qwen3ForCausalLM(config).float().eval()


def identity_mapper(layers, heads, dim):
    part = {"selections": [[layer] for layer in range(layers)],
            "weights": [torch.eye(dim).expand(heads, dim, dim).clone() for _ in range(layers)],
            "biases": [torch.zeros(heads, dim) for _ in range(layers)]}
    return {"keys": part, "values": part}


def transfer(source, receiver, mapper, prefix, positions):
    content = kvmap.to_content(source, prefill(source, prefix), positions)
    return kvmap.from_content(receiver, kvmap.apply(mapper, content), positions, torch.float32)


def test_rotary_base_swap_reproduces_receiver_exactly():
    source, receiver = tiny(5_000_000.0, layers=1), tiny(1_000_000.0, layers=1)
    receiver.load_state_dict(source.state_dict())
    ids = torch.randint(0, 97, (2, PREFIX + 8))
    prefix, continuation = ids[:, :PREFIX], ids[:, PREFIX:]
    positions = torch.arange(PREFIX).unsqueeze(0)
    with torch.no_grad():
        own = continue_from(receiver, prefill(receiver, prefix), continuation, PREFIX).logits
        mapped = continue_from(receiver, transfer(source, receiver, identity_mapper(1, 2, 16), prefix, positions),
                               continuation, PREFIX).logits
        unconverted = continue_from(receiver, prefill(source, prefix), continuation, PREFIX).logits
    assert torch.allclose(mapped, own, atol=1e-4)
    assert not torch.allclose(unconverted, own, atol=1e-2)


def test_fractional_positions_round_trip():
    model = tiny(5_000_000.0)
    pairs = prefill(model, torch.randint(0, 97, (1, PREFIX)))
    positions = torch.linspace(0.0, 1000.0, PREFIX).unsqueeze(0)
    rotated = kvmap.from_content(model, kvmap.to_content(model, pairs, torch.arange(PREFIX).unsqueeze(0)),
                                 positions, torch.float32)
    back = kvmap.to_content(model, rotated, positions)
    original = kvmap.to_content(model, pairs, torch.arange(PREFIX).unsqueeze(0))
    for (k1, v1), (k2, v2) in zip(back, original, strict=True):
        assert torch.allclose(k1, k2, atol=1e-5) and torch.equal(v1, v2)


def test_fit_on_self_selects_matching_layer_and_recovers_cache():
    model = tiny(1_000_000.0)
    positions = torch.arange(PREFIX).unsqueeze(0)
    keys, values = kvmap.Moments(2, 32, 32, "cpu"), kvmap.Moments(2, 32, 32, "cpu")
    for seed in range(8):
        torch.manual_seed(100 + seed)
        content = kvmap.to_content(model, prefill(model, torch.randint(0, 97, (4, PREFIX))), positions)
        keys.add(kvmap.stack_heads(content, 0), kvmap.stack_heads(content, 0))
        values.add(kvmap.stack_heads(content, 1), kvmap.stack_heads(content, 1))
    mapper = {"keys": kvmap.fit(kvmap.statistics(keys, 16, 1e-6), 1),
              "values": kvmap.fit(kvmap.statistics(values, 16, 1e-6), 1)}
    assert mapper["keys"]["selections"] == [[0], [1]] and mapper["values"]["selections"] == [[0], [1]]
    prefix = torch.randint(0, 97, (2, PREFIX))
    mapped = transfer(model, model, mapper, prefix, positions)
    for (mk, mv), (ok, ov) in zip(mapped, prefill(model, prefix), strict=True):
        assert torch.allclose(mk, ok, atol=1e-3) and torch.allclose(mv, ov, atol=1e-3)


def test_selection_follows_predictive_layer():
    stats_moments = kvmap.Moments(1, 32, 16, "cpu")
    torch.manual_seed(0)
    mixing = torch.randn(16, 16, dtype=torch.float64)
    for _ in range(4):
        x = torch.randn(1, 512, 32, dtype=torch.float64)
        y = x[:, :, 16:] @ mixing + 0.01 * torch.randn(1, 512, 16, dtype=torch.float64)
        stats_moments.add(x, y)
    stats = kvmap.statistics(stats_moments, 16, 1e-3)
    assert stats["scores"][0, 1] > 0.99 and stats["scores"][0, 0] < 0.1
    assert kvmap.fit(stats, 1)["selections"] == [[1]]


def test_refit_lowers_loss_on_a_fixed_batch():
    source, receiver = tiny(5_000_000.0, seed=0), tiny(1_000_000.0, seed=1)
    compactor = StillCompactor(source.config, 6).eval().requires_grad_(False)
    for model in (source, receiver):
        model.requires_grad_(False)
    mapper = Mapper(identity_mapper(2, 2, 16))
    optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-2)
    torch.manual_seed(3)
    prefixes = torch.randint(0, 97, (2, PREFIX))
    ids = torch.randint(0, 97, (2, 6))
    rows = torch.arange(2).repeat_interleave(5)
    columns = torch.arange(5).repeat(2)
    batch = (prefixes, ids, (rows, columns), ids[:, 1:].reshape(-1))
    losses = []
    for _ in range(30):
        loss = step_loss(source, compactor, receiver, mapper, batch, PREFIX, 6, top_k=20)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
