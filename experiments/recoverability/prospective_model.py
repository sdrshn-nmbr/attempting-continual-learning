import hashlib
import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration

from model import Observer
from protocol import file_hash


def tensor_hash(tensor):
    result = hashlib.sha256()
    flat = tensor.detach().contiguous().view(-1)
    for chunk in flat.split(4 * 1024 * 1024):
        result.update(chunk.cpu().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def parameter_hashes(model, frozen_only=False):
    return {
        name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "sha256": tensor_hash(parameter)}
        for name, parameter in model.named_parameters()
        if not frozen_only or not parameter.requires_grad
    }


def code_features(logits, target, codes):
    selected = logits.detach().float()[codes]
    if not torch.isfinite(selected).all():
        raise ValueError("PROSPECTIVE_NONFINITE_CODES")
    logp = selected.log_softmax(-1)
    probabilities = logp.exp()
    top = probabilities.topk(2).values
    index = codes.index(target)
    alternatives = [i for i in range(len(codes)) if i != index]
    return {
        "prediction": codes[int(selected.argmax())],
        "top_probability": float(top[0]),
        "entropy": float(-(probabilities * logp).sum()),
        "top_margin": float(top[0] - top[1]),
        "target_logp": float(logp[index]),
        "target_margin": float(selected[index] - selected[alternatives].max()),
        "code_logits": selected.tolist(),
    }


def code_loss(logits, targets, codes):
    selected = logits.float()[:, codes]
    labels = torch.tensor([codes.index(target) for target in targets], device=logits.device)
    return torch.nn.functional.cross_entropy(selected, labels)


class ProspectiveModel(Observer):
    @classmethod
    def fresh(cls, config, spec, data, seed):
        torch.set_num_threads(spec["runtime"]["threads"])
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if spec["runtime"]["device"].startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("PROSPECTIVE_GPU_REQUIRED")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model_path"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            device_map=spec["runtime"]["device"],
            attn_implementation=spec["runtime"]["attention"],
        )
        model = get_peft_model(
            base,
            LoraConfig(
                r=spec["adapter"]["rank"],
                lora_alpha=spec["adapter"]["alpha"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=r"model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
            ),
        )
        observer = cls(model, spec, data["codes"])
        observer.pad_token_id = data["pad_token_id"]
        return observer

    def inputs(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row["input_ids"]) :] = torch.tensor(row["input_ids"], device=self.device)
            mask[i, -len(row["input_ids"]) :] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def logits(self, rows):
        return self.model(**self.inputs(rows), use_cache=False, logits_to_keep=1).logits[:, -1].float()

    def optimizer(self, parameters=None):
        return torch.optim.AdamW(
            self.trainable if parameters is None else parameters,
            lr=self.spec["optimizer"]["learning_rate"],
            weight_decay=0,
            foreach=False,
            fused=False,
        )

    def update(self, rows, batches, optimizer, parameters=None, sham_intent=None):
        parameters = self.trainable if parameters is None else parameters
        self.model.eval()
        history = []
        microbatch = self.spec["runtime"]["microbatch_size"]
        for step, indices in enumerate(batches, 1):
            self.model.zero_grad(set_to_none=True)
            loss_value = 0.0
            for start in range(0, len(indices), microbatch):
                selected = [rows[index] for index in indices[start : start + microbatch]]
                targets = [
                    self.codes[(row["code"] + 1) % 16] if row["intent"] == sham_intent else row["target"]
                    for row in selected
                ]
                loss = code_loss(self.logits(selected), targets, self.codes) * len(selected) / len(indices)
                if not torch.isfinite(loss):
                    raise ValueError(f"PROSPECTIVE_NONFINITE_LOSS step={step}")
                loss.backward()
                loss_value += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.spec["optimizer"]["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            history.append({"step": step, "loss": loss_value, "gradient_norm": float(norm), "rows": indices})
            if step == 1 or step % 16 == 0 or step == len(batches):
                logging.info("PROSPECTIVE_UPDATE step=%d/%d loss=%.6f", step, len(batches), loss_value)
        self.model.zero_grad(set_to_none=True)
        return history

    @torch.no_grad()
    def capture(self, row, positions):
        captured = {}
        handles = []
        for layer in (*self.spec["layers"], self.depth):

            def hook(module, args, output, layer=layer):
                captured[layer] = output[0, positions].detach().clone()

            handles.append(self.text.layers[layer - 1].register_forward_hook(hook))
        try:
            logits = self.logits([row])[0]
            if len(row["input_ids"]) - 1 in positions:
                index = positions.index(len(row["input_ids"]) - 1)
                terminal = self.head(self.text.norm(captured[self.depth][index]))
                if not torch.allclose(terminal, logits, atol=1e-5, rtol=1e-5):
                    raise ValueError("PROSPECTIVE_TERMINAL_PARITY")
                error = float((terminal - logits).abs().max())
            else:
                error = None
            return logits.detach(), captured, error
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def observe(self, rows, lens=None):
        self.model.eval()
        result = []
        if lens is None:
            batch_size = self.spec["runtime"]["eval_batch_size"]
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                for row, logits in zip(batch, self.logits(batch), strict=True):
                    result.append(
                        {
                            "text_sha256": row["text_sha256"],
                            "intent": row["intent"],
                            "target": row["target"],
                            "output": code_features(logits, row["target"], self.codes),
                        }
                    )
            return result
        for row in rows:
            output, hidden, error = self.capture(row, [len(row["input_ids"]) - 1])
            result.append(
                {
                    "text_sha256": row["text_sha256"],
                    "intent": row["intent"],
                    "target": row["target"],
                    "output": code_features(output, row["target"], self.codes),
                    "terminal_max_error": error,
                    "frozen": {
                        str(layer): code_features(
                            self.head(self.text.norm(hidden[layer]))[0], row["target"], self.codes
                        )
                        for layer in self.spec["layers"]
                    },
                    "tuned": {
                        str(layer): code_features(
                            lens.decode(hidden[layer], layer, self.text.norm, self.head)[0], row["target"], self.codes
                        )
                        for layer in self.spec["layers"]
                    },
                }
            )
        return result

    def checkpoint(self, directory, optimizer=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.peft_config["default"].save_pretrained(directory)
        self.save_adapter(directory / "adapter_model.safetensors")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        return {path.name: file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()}

    def optimizer_steps(self, optimizer):
        return sorted(set(int(value["step"]) for value in optimizer.state.values()))
