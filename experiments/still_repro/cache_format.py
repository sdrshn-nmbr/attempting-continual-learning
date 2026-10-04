"""On-disk format for Attention Matching caches. Only real entries (finite beta) are stored, packed per head, with a
[layers, heads] table of lengths, so uneven per-head budgets cost no padding on disk. unpack restores [L,H,t,D] keys
and values and [L,H,t] beta, padding shorter heads with zeros and beta=-inf so the padding receives no attention."""
import torch


def pack(layers):
    """layers: per-layer (keys [H,t,D], beta [H,t], values [H,t,D]); padded entries are those with beta=-inf."""
    keys, beta, values, lengths = [], [], [], []
    for layer_keys, layer_beta, layer_values in layers:
        row = []
        for head in range(layer_keys.shape[0]):
            real = torch.isfinite(layer_beta[head])
            keys.append(layer_keys[head][real])
            values.append(layer_values[head][real])
            beta.append(layer_beta[head][real].float())
            row.append(int(real.sum()))
        lengths.append(row)
    return {"keys": torch.cat(keys), "beta": torch.cat(beta), "values": torch.cat(values),
            "lengths": torch.tensor(lengths, dtype=torch.long)}


def unpack(saved):
    lengths = saved["lengths"]
    layers, heads = lengths.shape
    longest = int(lengths.max())
    sizes = lengths.flatten().tolist()
    keys = saved["keys"].new_zeros(layers, heads, longest, saved["keys"].shape[-1])
    values = saved["values"].new_zeros(layers, heads, longest, saved["values"].shape[-1])
    beta = saved["beta"].new_full((layers, heads, longest), float("-inf"))
    chunks = zip(torch.split(saved["keys"], sizes), torch.split(saved["beta"], sizes),
                 torch.split(saved["values"], sizes), strict=True)
    for index, (head_keys, head_beta, head_values) in enumerate(chunks):
        layer, head = divmod(index, heads)
        keys[layer, head, :len(head_keys)] = head_keys
        beta[layer, head, :len(head_beta)] = head_beta
        values[layer, head, :len(head_values)] = head_values
    return keys, beta, values
