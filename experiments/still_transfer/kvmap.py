"""Closed-form cross-model KV mapping after Heo et al. (arXiv 2608.03893): for every target layer and KV head,
ridge regression from the same head of the top-k most predictive source layers. Keys are mapped with RoPE
removed, so each model applies its own rotary base at the original positions."""
import torch

from still import rotary


def rope(model, positions):
    dummy = torch.zeros(1, 1, model.config.hidden_size, device=positions.device, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, positions)
    return cos.float(), sin.float()


def to_content(model, pairs, positions):
    cos, sin = rope(model, positions)
    return [(rotary(keys.float(), cos, sin, inverse=True), values.float()) for keys, values in pairs]


def from_content(model, pairs, positions, dtype):
    cos, sin = rope(model, positions)
    return [(rotary(keys, cos, sin).to(dtype), values.to(dtype)) for keys, values in pairs]


def stack_heads(content, part):
    tensors = torch.stack([pair[part] for pair in content])
    layers, batch, heads, tokens, dim = tensors.shape
    return tensors.permute(2, 1, 3, 0, 4).reshape(heads, batch * tokens, layers * dim).double()


class Moments:
    def __init__(self, heads, source_width, target_width, device):
        zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=device)
        self.count = zeros(())
        self.sum_x = zeros(heads, source_width)
        self.sum_y = zeros(heads, target_width)
        self.sum_xx = zeros(heads, source_width, source_width)
        self.sum_xy = zeros(heads, source_width, target_width)
        self.sum_yy = zeros(heads, target_width)

    def add(self, x, y):
        self.count += x.shape[1]
        self.sum_x += x.sum(1)
        self.sum_y += y.sum(1)
        self.sum_xx += x.transpose(1, 2) @ x
        self.sum_xy += x.transpose(1, 2) @ y
        self.sum_yy += y.square().sum(1)

    def tensors(self):
        return [self.count, self.sum_x, self.sum_y, self.sum_xx, self.sum_xy, self.sum_yy]


def statistics(moments, dim, lam):
    n = moments.count
    mean_x, mean_y = moments.sum_x / n, moments.sum_y / n
    cxx = moments.sum_xx / n - mean_x[:, :, None] * mean_x[:, None, :]
    cxy = moments.sum_xy / n - mean_x[:, :, None] * mean_y[:, None, :]
    var_y = moments.sum_yy / n - mean_y.square()
    heads, source_width, target_width = cxy.shape
    sources, targets = source_width // dim, target_width // dim
    total = var_y.view(heads, targets, dim).sum(-1)
    eye = torch.eye(dim, dtype=cxx.dtype, device=cxx.device)
    scores = torch.zeros(targets, sources, dtype=torch.float64)
    for source in range(sources):
        block = slice(source * dim, (source + 1) * dim)
        weights = torch.linalg.solve(cxx[:, block, block] + lam * eye, cxy[:, block, :])
        explained = 2 * (cxy[:, block, :] * weights).sum(1) - (weights * (cxx[:, block, block] @ weights)).sum(1)
        scores[:, source] = (explained.view(heads, targets, dim).sum(-1) / total).mean(0).cpu()
    return {"mean_x": mean_x, "mean_y": mean_y, "cxx": cxx, "cxy": cxy, "scores": scores, "dim": dim, "lam": lam}


def fit(stats, k):
    dim, lam, cxx, cxy = stats["dim"], stats["lam"], stats["cxx"], stats["cxy"]
    selections, weights, biases = [], [], []
    for target, row in enumerate(stats["scores"]):
        chosen = sorted(row.topk(k).indices.tolist())
        index = torch.cat([torch.arange(s * dim, (s + 1) * dim) for s in chosen]).to(cxx.device)
        columns = slice(target * dim, (target + 1) * dim)
        system = cxx[:, index][:, :, index] + lam * torch.eye(len(index), dtype=cxx.dtype, device=cxx.device)
        weight = torch.linalg.solve(system, cxy[:, index, columns])
        bias = stats["mean_y"][:, columns] - (stats["mean_x"][:, index].unsqueeze(1) @ weight).squeeze(1)
        selections.append(chosen)
        weights.append(weight.float())
        biases.append(bias.float())
    return {"selections": selections, "weights": weights, "biases": biases}


def project(part, target, tensors):
    inputs = torch.cat([tensors[s] for s in part["selections"][target]], dim=-1)
    weight = part["weights"][target].to(inputs.device)
    bias = part["biases"][target].to(inputs.device)
    return torch.einsum("bhtn,hnd->bhtd", inputs, weight) + bias[None, :, None, :]


def apply(mapper, content):
    keys = [pair[0] for pair in content]
    values = [pair[1] for pair in content]
    return [(project(mapper["keys"], target, keys), project(mapper["values"], target, values))
            for target in range(len(mapper["keys"]["selections"]))]
