import logging
import random

import torch
from safetensors.torch import load_file, save_file

from prospective_model import parameter_hashes
from protocol import file_hash


def forward_kl(teacher_logits, lens_logits):
    teacher_logp = teacher_logits.detach().float().log_softmax(-1)
    lens_logp = lens_logits.float().log_softmax(-1)
    loss = (teacher_logp.exp() * (teacher_logp - lens_logp)).sum(-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("PROSPECTIVE_NONFINITE_LENS_KL")
    return loss


class TranslationBank(torch.nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.translators = torch.nn.ModuleDict({str(layer): torch.nn.Linear(width, width) for layer in layers})
        for translator in self.translators.values():
            torch.nn.init.zeros_(translator.weight)
            torch.nn.init.zeros_(translator.bias)

    def decode(self, hidden, layer, norm, head):
        return head(norm(hidden + self.translators[str(layer)](hidden)))


def cache_activations(observer, rows, positions_per_prompt):
    observer.model.eval()
    cache = {str(layer): [] for layer in (*observer.spec["layers"], observer.depth)}
    positions = []
    errors = []
    for index, row in enumerate(rows):
        chosen = (
            torch.linspace(0, len(row["input_ids"]) - 1, min(len(row["input_ids"]), positions_per_prompt))
            .round()
            .long()
            .tolist()
        )
        _, hidden, error = observer.capture(row, chosen)
        for layer, value in hidden.items():
            cache[str(layer)].append(value.cpu())
        positions.extend({"text_sha256": row["text_sha256"], "position": position} for position in chosen)
        errors.append(error)
        if index % 64 == 0:
            logging.info("PROSPECTIVE_LENS_CACHE prompt=%d/%d", index + 1, len(rows))
    return {layer: torch.cat(values) for layer, values in cache.items()}, {
        "positions": positions,
        "terminal_max_error": max(errors),
        "prompts": len(rows),
        "tokens": len(positions),
    }


@torch.no_grad()
def evaluate_translations(bank, observer, cache, batch_size):
    sums = {str(layer): {"frozen": 0.0, "tuned": 0.0} for layer in observer.spec["layers"]}
    count = len(cache[str(observer.depth)])
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        terminal = cache[str(observer.depth)][start:end].to(observer.device)
        teacher = observer.head(observer.text.norm(terminal))
        for layer in observer.spec["layers"]:
            hidden = cache[str(layer)][start:end].to(observer.device)
            for name, logits in (
                ("frozen", observer.head(observer.text.norm(hidden))),
                ("tuned", bank.decode(hidden, layer, observer.text.norm, observer.head)),
            ):
                sums[str(layer)][name] += float(forward_kl(teacher, logits)) * (end - start)
    return {layer: {name: value / count for name, value in values.items()} for layer, values in sums.items()}


def fit_translations(bank, observer, cache, spec, seed):
    parameters = list(bank.parameters())
    if any(parameter.requires_grad for parameter in observer.text.norm.parameters()):
        raise ValueError("PROSPECTIVE_LENS_NORM_NOT_FROZEN")
    if any(parameter.requires_grad for parameter in observer.head.parameters()):
        raise ValueError("PROSPECTIVE_LENS_HEAD_NOT_FROZEN")
    optimizer = torch.optim.SGD(
        parameters, lr=spec["learning_rate"], momentum=0.9, nesterov=True, weight_decay=0.001, foreach=False
    )
    count = len(cache[str(observer.depth)])
    rng = random.Random(seed)
    queue, history = [], []
    observer.model.zero_grad(set_to_none=True)
    bank.train()
    for step in range(spec["updates"]):
        chosen = []
        for _ in range(spec["token_batch_size"]):
            if not queue:
                queue = rng.sample(range(count), count)
            chosen.append(queue.pop())
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher = observer.head(observer.text.norm(cache[str(observer.depth)][chosen].to(observer.device)))
        losses = {}
        for layer in observer.spec["layers"]:
            hidden = cache[str(layer)][chosen].to(observer.device)
            logits = bank.decode(hidden, layer, observer.text.norm, observer.head)
            loss = forward_kl(teacher, logits)
            loss.backward()
            losses[str(layer)] = float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        learning_rate = spec["learning_rate"] * (1 - step / spec["updates"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "positions": chosen,
                "kl": losses,
                "gradient_norm": float(norm),
                "learning_rate": learning_rate,
            }
        )
        if step == 0 or (step + 1) % 16 == 0:
            logging.info("PROSPECTIVE_LENS_UPDATE step=%d/%d kl=%s", step + 1, spec["updates"], losses)
    if any(parameter.grad is not None for parameter in observer.model.parameters()):
        raise ValueError("PROSPECTIVE_LENS_GRADIENT_LEAKAGE")
    bank.eval()
    return history


def train_lens(observer, data, spec, output, seed):
    before = parameter_hashes(observer.model)
    fit_cache, fit_metadata = cache_activations(observer, data["lens_fit"], spec["lens"]["positions_per_prompt"])
    check_cache, check_metadata = cache_activations(observer, data["lens_check"], spec["lens"]["positions_per_prompt"])
    bank = TranslationBank(observer.head.in_features, spec["layers"]).to(observer.device)
    initial = evaluate_translations(bank, observer, check_cache, spec["lens"]["token_batch_size"])
    if any(abs(value["frozen"] - value["tuned"]) > 1e-7 for value in initial.values()):
        raise ValueError("PROSPECTIVE_LENS_IDENTITY_CONTROL")
    history = fit_translations(bank, observer, fit_cache, spec["lens"], seed)
    final = evaluate_translations(bank, observer, check_cache, spec["lens"]["token_batch_size"])
    after = parameter_hashes(observer.model)
    if before != after:
        raise ValueError("PROSPECTIVE_LENS_CHANGED_MODEL")
    path = output / "tuned-lens.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in bank.state_dict().items()}, str(path))
    restored = TranslationBank(observer.head.in_features, spec["layers"]).to(observer.device)
    restored.load_state_dict(load_file(str(path)), strict=True)
    if any(not torch.equal(value, restored.state_dict()[key]) for key, value in bank.state_dict().items()):
        raise ValueError("PROSPECTIVE_LENS_SERIALIZATION")
    replay = evaluate_translations(restored, observer, check_cache, spec["lens"]["token_batch_size"])
    if replay != final:
        raise ValueError("PROSPECTIVE_LENS_RELOAD_BEHAVIOR")
    improvement = 1 - sum(value["tuned"] for value in final.values()) / max(
        sum(value["frozen"] for value in final.values()), 1e-12
    )
    result = {
        "fit": fit_metadata,
        "check": check_metadata,
        "initial_heldout_kl": initial,
        "final_heldout_kl": final,
        "relative_kl_improvement": improvement,
        "translator_qualified": improvement >= spec["lens"]["min_relative_kl_improvement"],
        "history": history,
        "model_parameters_unchanged": True,
        "model_sha256": before,
        "lens_sha256": file_hash(path),
        "reload_exact": True,
        "trainable_parameters": sum(parameter.numel() for parameter in bank.parameters()),
    }
    return restored, result
