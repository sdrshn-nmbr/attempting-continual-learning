import inspect
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5ForConditionalGeneration

from protocol import file_hash, read_json


def runtime_info():
    return {
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "safetensors", "numpy")},
        "forward_signature": str(inspect.signature(Qwen3_5ForConditionalGeneration.forward)),
        "adapter_signature": str(inspect.signature(set_peft_model_state_dict)),
    }


def distribution_features(logits, target, codes):
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError("RECOVERY_NONFINITE_LOGITS")
    logp = logits.log_softmax(-1)
    p = logp.exp()
    top = p.topk(2).values
    alternatives = [code for code in codes if code != target]
    return {
        "prediction": int(logits.argmax()),
        "code_prediction": codes[int(logits[codes].argmax())],
        "top_probability": float(top[0]),
        "entropy": float(-(p * logp).sum()),
        "top_margin": float(top[0] - top[1]),
        "target_logp": float(logp[target]),
        "target_code_margin": float(logits[target] - logits[alternatives].max()),
    }


class Observer:
    def __init__(self, model, spec, codes):
        self.model, self.spec, self.codes = model, spec, codes
        base = model.get_base_model()
        if base.config.model_type != "qwen3_5":
            raise ValueError("RECOVERY_BACKBONE_CLASS")
        self.text = base.model.language_model
        self.head = base.lm_head
        self.depth = len(self.text.layers)
        if any(layer < 1 or layer >= self.depth for layer in spec["layers"]):
            raise ValueError("RECOVERY_LENS_LAYER_RANGE")
        self.trainable = [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name]
        if not self.trainable or any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise ValueError("RECOVERY_OPTIMIZER_ISOLATION")
        self.device = torch.device(spec["runtime"]["device"])
        if any(p.device != self.device or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("RECOVERY_DEVICE_FLOAT32_REQUIRED")
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
            raise ValueError("RECOVERY_DROPOUT_NOT_ZERO")
        self.adapter_config = model.peft_config["default"].to_dict()

    @classmethod
    def load(cls, config, spec, codes, checkpoint):
        torch.set_num_threads(spec["runtime"]["threads"])
        torch.manual_seed(spec["runtime"]["seed"])
        torch.use_deterministic_algorithms(True)
        if spec["runtime"]["device"].startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("RECOVERY_PARENT_GPU_REQUIRED")
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
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True, local_files_only=True)
        return cls(model, spec, codes)

    def reset(self, checkpoint):
        cfg = read_json(Path(checkpoint) / "adapter_config.json")
        for key in (
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "modules_to_save",
            "use_dora",
            "use_rslora",
        ):
            expected, actual = cfg.get(key), self.adapter_config.get(key)
            if key == "target_modules" and isinstance(expected, list):
                expected, actual = set(expected), set(actual)
            if expected != actual:
                raise ValueError(f"RECOVERY_ADAPTER_CONFIG {key}")
        state = load_file(str(Path(checkpoint) / "adapter_model.safetensors"), device="cpu")
        result = set_peft_model_state_dict(self.model, state)
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("RECOVERY_ADAPTER_LOAD_KEYS")
        actual = get_peft_model_state_dict(self.model, save_embedding_layers=False)
        if state.keys() != actual.keys() or any(
            not torch.equal(state[key].float(), actual[key].float().cpu()) for key in state
        ):
            raise ValueError("RECOVERY_RESET_PARITY")
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    @torch.inference_mode()
    def observe(self, rows, lens=False):
        self.model.eval()
        result = []
        for row in rows:
            hidden = {}
            handles = []
            if lens:
                for layer in (*self.spec["layers"], self.depth):

                    def capture(module, args, output, layer=layer, hidden=hidden):
                        hidden[layer] = output[:, -1, :].detach().clone()

                    handles.append(self.text.layers[layer - 1].register_forward_hook(capture))
            try:
                inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device=self.device)
                output = self.model(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1
                ).logits[0, -1]
                record = {
                    "text_sha256": row["text_sha256"],
                    "output": distribution_features(output, row["target"], self.codes),
                }
                if lens:
                    terminal = self.head(self.text.norm(hidden[self.depth]))[0]
                    if not torch.allclose(terminal, output, atol=1e-5, rtol=1e-5):
                        raise ValueError("RECOVERY_TERMINAL_LENS_PARITY")
                    record["terminal_max_error"] = float((terminal - output).abs().max())
                    record["lens"] = {
                        str(layer): distribution_features(
                            self.head(self.text.norm(hidden[layer]))[0],
                            row["target"],
                            self.codes,
                        )
                        for layer in self.spec["layers"]
                    }
                result.append(record)
            finally:
                for handle in handles:
                    handle.remove()
        return result

    def repair(self, rows, action):
        steps = self.spec["actions"][action]
        opt = self.spec["optimizer"]
        optimizer = torch.optim.SGD(self.trainable, lr=opt["learning_rate"])
        self.model.eval()
        losses = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            batch = [rows[(step * opt["batch_size"] + i) % len(rows)] for i in range(opt["batch_size"])]
            total = 0.0
            for row in batch:
                target = row["target"]
                if action == "sham":
                    target = self.codes[(self.codes.index(target) + 1) % len(self.codes)]
                inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device=self.device)
                logits = (
                    self.model(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1
                    )
                    .logits[:, -1]
                    .float()
                )
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target], device=self.device)) / len(
                    batch
                )
                if not torch.isfinite(loss):
                    raise ValueError("RECOVERY_NONFINITE_REPAIR")
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(self.trainable, opt["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            losses.append({"step": step + 1, "loss": total, "gradient_norm": float(norm)})
        self.model.zero_grad(set_to_none=True)
        return losses

    def save_adapter(self, path):
        state = {
            key: tensor.detach().cpu().contiguous().clone()
            for key, tensor in get_peft_model_state_dict(self.model, save_embedding_layers=False).items()
        }
        save_file(state, str(path))
        return file_hash(path)
