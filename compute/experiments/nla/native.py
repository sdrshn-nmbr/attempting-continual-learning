import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig, Qwen2ForCausalLM, Qwen2Model

from protocol import AR_ID, AV_ID, BASE_ID, candidate_metrics, normalized_text

EXPLANATION = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def file_digest(path, tick=None):
    hasher = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            hasher.update(block)
            total += len(block)
            if tick is not None:
                tick(total)
    return hasher.hexdigest()


def verify_model_files(config, tick=None):
    contract = json.loads(Path(__file__).with_name("upstream_contract.json").read_text())
    entries = (
        (BASE_ID, "model_path", "revision"),
        (AV_ID, "av_model_path", "av_revision"),
        (AR_ID, "ar_model_path", "ar_revision"),
    )
    observed = {}
    for model_id, path_key, revision_key in entries:
        expected = contract["models"][model_id]
        if config[revision_key] != expected["revision"]:
            raise ValueError(
                f"NLA_UNSUPPORTED_REVISION: {model_id}; refresh the inspected contract before using another release"
            )
        root = Path(config[path_key])
        files = {}
        for filename, facts in {**expected["files"], **expected["weights"]}.items():
            path = root / filename
            if not path.is_file() or path.stat().st_size != facts["bytes"]:
                raise ValueError(f"NLA_CHECKPOINT_FILE: missing or size mismatch: {path}")
            is_weight = filename in expected["weights"]
            verify = config["verify_weight_hashes"] or not is_weight
            sha = file_digest(path, tick) if verify else None
            if verify and sha != facts["sha256"]:
                raise ValueError(f"NLA_CHECKPOINT_HASH: content mismatch: {path}")
            files[filename] = {"bytes": facts["bytes"], "expected_sha256": facts["sha256"], "observed_sha256": sha}
        observed[model_id] = {
            "revision": config[revision_key],
            "path": str(root.resolve()),
            "license": expected["license"],
            "verification": "all_file_sha256"
            if config["verify_weight_hashes"]
            else "metadata_sha256_and_weight_sizes_only",
            "files": files,
        }
    return {"models": observed, "upstream_commit": contract["upstream_commit"], "backend": contract["backend"]}


@dataclass(frozen=True)
class Sidecar:
    role: str
    width: int
    layer: int
    injection_scale: float | None
    mse_scale: float
    injection_char: str
    injection_token_id: int
    left_id: int
    right_id: int
    av_template: str
    ar_template: str
    suffix_ids: tuple[int, ...]

    @classmethod
    def load(cls, root, role):
        data = yaml.safe_load((Path(root) / "nla_meta.yaml").read_text())
        if data["kind"] != "nla_model" or data["schema_version"] != 2 or data["role"] != role:
            raise ValueError("NLA_SIDECAR_SCHEMA: expected current released av/ar sidecar")
        if data["d_model"] != 3584 or data["extraction_layer_index"] != 20:
            raise ValueError("NLA_SIDECAR_NATIVE: only Qwen2.5-7B hidden_states[20] is supported")
        extraction = data["extraction"]
        if not math.isclose(extraction["mse_scale"], math.sqrt(data["d_model"]), rel_tol=1e-6):
            raise ValueError("NLA_SIDECAR_NORM: direction MSE requires mse_scale=sqrt(d_model)")
        scale = extraction["injection_scale"]
        if role == "av" and (scale is None or scale <= 0):
            raise ValueError("NLA_SIDECAR_INJECTION: AV injection_scale is required")
        tokens = data["tokens"]
        return cls(
            role,
            data["d_model"],
            data["extraction_layer_index"],
            scale,
            extraction["mse_scale"],
            tokens["injection_char"],
            tokens["injection_token_id"],
            tokens["injection_left_neighbor_id"],
            tokens["injection_right_neighbor_id"],
            data["prompt_templates"]["av"],
            data["prompt_templates"]["ar"],
            tuple(tokens["critic_suffix_ids"] or []),
        )


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("NLA_TOKENIZER_OFFSETS: fast tokenizer required for audited source positions")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def check_architecture(model, layers):
    cfg = model.config
    if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers) != ("qwen2", 3584, layers):
        raise ValueError(f"NLA_ARCHITECTURE: expected native qwen2 width=3584 layers={layers}")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("NLA_PLACEMENT: entire model must reside on the isolated GPU")


def load_source(path):
    model, loading = Qwen2ForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"NLA_SOURCE_LOAD: {loading}")
    check_architecture(model, 28)
    return model.eval()


def chat_query(tokenizer, content, max_length):
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    canonical = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if ids != canonical or rendered.count(content) != 1:
        raise ValueError("NLA_QUERY_ALIGNMENT: template tokenization or unique content span mismatch")
    if len(ids) > max_length:
        raise ValueError(f"NLA_QUERY_LENGTH: {len(ids)} exceeds {max_length}; no silent truncation")
    start = rendered.index(content)
    end = start + len(content)
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start and b <= end]
    if not positions:
        raise ValueError("NLA_QUERY_ALIGNMENT: no content token found")
    position = positions[-1]
    if position < 10:
        raise ValueError("NLA_QUERY_POSITION: early-sequence activation is outside this calibration protocol")
    return {
        "input_ids": ids,
        "rendered": rendered,
        "source_token_index": position,
        "source_token_id": ids[position],
        "source_token_text": tokenizer.decode([ids[position]]),
        "source_char_span": list(encoded["offset_mapping"][position]),
        "source_policy": "last_user_content_token_before_chat_suffix; never an answer/output token",
        "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }


def completion_ids(tokenizer, query, answer, max_length):
    complete = tokenizer(query["rendered"] + answer, add_special_tokens=False)["input_ids"]
    prefix = query["input_ids"]
    if complete[: len(prefix)] != prefix or len(complete) <= len(prefix):
        raise ValueError("NLA_COMPLETION_ALIGNMENT: target tokenization changed the query prefix")
    if len(complete) > max_length:
        raise ValueError("NLA_COMPLETION_LENGTH: no silent answer truncation")
    return complete, complete[len(prefix) :]


@torch.inference_mode()
def extract_activation(model, query, layer):
    ids = torch.tensor([query["input_ids"]], dtype=torch.long, device=model.device)
    output = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False, logits_to_keep=1
    )
    if layer >= len(output.hidden_states) - 1:
        raise ValueError("NLA_SOURCE_LAYER: final normalized hidden state is not a supported residual readout")
    activation = output.hidden_states[layer][0, query["source_token_index"]].float().cpu()
    if not torch.isfinite(activation).all() or activation.norm() <= 0:
        raise RuntimeError("NLA_SOURCE_VECTOR: non-finite or zero hidden state")
    return activation


@torch.inference_mode()
def score_behavior(model, tokenizer, query, record, max_length):
    completions = [completion_ids(tokenizer, query, answer, max_length) for answer in record.candidates]
    max_answer = max(len(suffix) for _, suffix in completions)
    prefix_length = len(query["input_ids"])
    sequences = [complete + [tokenizer.pad_token_id] * (max_answer - len(suffix)) for complete, suffix in completions]
    masks = [[1] * len(complete) + [0] * (max_answer - len(suffix)) for complete, suffix in completions]
    ids = torch.tensor(sequences, dtype=torch.long, device=model.device)
    mask = torch.tensor(masks, dtype=torch.long, device=model.device)
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=max_answer + 1, use_cache=False).logits.float()
    if logits.shape[1] != max_answer + 1:
        raise ValueError("NLA_LOGIT_ALIGNMENT: logits_to_keep no longer selects the answer-prediction suffix")
    scores = []
    for i, (_, suffix) in enumerate(completions):
        selected = logits[i, : len(suffix)]
        targets = ids[i, prefix_length : prefix_length + len(suffix)]
        scores.append(float(-F.cross_entropy(selected, targets, reduction="sum")))
    result = candidate_metrics(scores, [len(s) for _, s in completions], record.candidates, record.answer)
    prompt = torch.tensor([query["input_ids"]], dtype=torch.long, device=model.device)
    output = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=16,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        ),
    )
    generated = tokenizer.decode(output[0, prefix_length:], skip_special_tokens=True).strip()
    result.update(
        {"greedy_text": generated, "greedy_exact": int(normalized_text(generated) == normalized_text(record.answer))}
    )
    return result


def injection_position(ids, sidecar):
    matches = [i for i, value in enumerate(ids) if value == sidecar.injection_token_id]
    if len(matches) != 1:
        raise ValueError(f"NLA_INJECTION_COUNT: expected one marker, found {len(matches)}")
    position = matches[0]
    if (
        position == 0
        or position == len(ids) - 1
        or ids[position - 1] != sidecar.left_id
        or ids[position + 1] != sidecar.right_id
    ):
        raise ValueError("NLA_INJECTION_NEIGHBORS: tokenization differs from released sidecar")
    return position


def inject_activation(embeddings, activation, position, scale):
    if activation.ndim != 1 or activation.shape[0] != embeddings.shape[-1] or embeddings.shape[0] != 1:
        raise ValueError("NLA_INJECTION_SHAPE: one native-width vector and one prompt required")
    if not torch.isfinite(activation).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("NLA_INJECTION_NONFINITE: invalid vector or scale")
    vector = activation.float()
    vector = vector * (scale / vector.norm().clamp_min(1e-12))
    result = embeddings.clone()
    result[0, position] = vector.to(result.device, result.dtype)
    return result


def norm_matched_random(activation, seed, identifier):
    target = activation.detach().to(device="cpu", dtype=torch.float32)
    target_norm = target.norm()
    if target.ndim != 1 or not torch.isfinite(target).all() or not torch.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("NLA_RANDOM_CONTROL_VECTOR: finite nonzero source vector required")
    seed_bytes = hashlib.sha256(f"nla-random-control:{seed}:{identifier}".encode()).digest()
    generator_seed = int.from_bytes(seed_bytes[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    vector = torch.randn(target.shape, generator=generator, device="cpu", dtype=torch.float32)
    vector *= target_norm / vector.norm()
    return vector, {
        "algorithm": "cpu_float32_gaussian_direction_scaled_to_recipient_raw_l2",
        "generator_seed": generator_seed,
        "vector_sha256": hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
        "recipient_raw_norm": float(target_norm),
    }


def generation_diagnostics(tokenizer, token_ids, max_new_tokens):
    if len(token_ids) > max_new_tokens:
        raise ValueError("NLA_AV_OUTPUT_ALIGNMENT: embeds-only generation unexpectedly returned prompt IDs")
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    matches = EXPLANATION.findall(raw)
    valid = len(matches) == 1 and bool(matches[0].strip())
    eos_positions = [i for i, token in enumerate(token_ids) if token == tokenizer.eos_token_id]
    ended_on_eos = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
    hit_token_cap = len(token_ids) == max_new_tokens
    body = tokenizer.decode(token_ids[:-1] if ended_on_eos else token_ids, skip_special_tokens=False).strip()
    before_eos = tokenizer.decode(token_ids[: eos_positions[0]], skip_special_tokens=False) if eos_positions else None
    return {
        "raw": raw,
        "explanation": matches[0].strip() if valid else None,
        "format_valid": valid,
        "exact_format_valid": bool(
            valid and raw.count("<explanation>") == raw.count("</explanation>") == 1 and EXPLANATION.fullmatch(body)
        ),
        "opening_tag_count": raw.count("<explanation>"),
        "closing_tag_count": raw.count("</explanation>"),
        "complete_pair_count": len(matches),
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "eos_token_id": tokenizer.eos_token_id,
        "eos_token_positions": eos_positions,
        "final_token_id": token_ids[-1] if token_ids else None,
        "ended_on_eos": ended_on_eos,
        "eos_before_closing_tag": before_eos is not None and "</explanation>" not in before_eos,
        "hit_token_cap": hit_token_cap,
        "stop_reason": "eos" if ended_on_eos else "length" if hit_token_cap else "other",
    }


class NativeAV:
    def __init__(self, path):
        self.sidecar = Sidecar.load(path, "av")
        self.tokenizer = load_tokenizer(path)
        if self.tokenizer.encode(self.sidecar.injection_char, add_special_tokens=False) != [
            self.sidecar.injection_token_id
        ]:
            raise ValueError("NLA_INJECTION_TOKEN: released injection character no longer maps to its token")
        content = self.sidecar.av_template.format(injection_char=self.sidecar.injection_char)
        self.ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, return_dict=False, add_generation_prompt=True
        )
        self.position = injection_position(self.ids, self.sidecar)
        self.model = load_source(path)
        self.model.requires_grad_(False)
        ids = torch.tensor([self.ids], dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            self.embeddings = self.model.get_input_embeddings()(ids).detach()

    @torch.inference_mode()
    def describe(self, activation, max_new_tokens):
        embeds = inject_activation(self.embeddings, activation, self.position, self.sidecar.injection_scale)
        generated = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            ),
        )
        return {
            **generation_diagnostics(self.tokenizer, generated[0].tolist(), max_new_tokens),
            "injection_slot_norm": float(embeds[0, self.position].float().norm()),
            "injection_scale": self.sidecar.injection_scale,
        }


class NativeAR:
    def __init__(self, path):
        self.sidecar = Sidecar.load(path, "ar")
        self.tokenizer = load_tokenizer(path)
        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and probe[0] != bos:
            raise ValueError("NLA_AR_BOS: tokenizer no longer follows released AR BOS convention")
        self.model, loading = Qwen2Model.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if (
            set(loading.get("missing_keys", [])) != {"norm.weight"}
            or loading.get("unexpected_keys")
            or loading.get("mismatched_keys")
        ):
            raise ValueError(f"NLA_AR_LOAD: unexpected released backbone layout: {loading}")
        self.model.norm = torch.nn.Identity()
        check_architecture(self.model, 21)
        self.model.eval().requires_grad_(False)
        self.head = torch.nn.Linear(self.sidecar.width, self.sidecar.width, bias=False, dtype=torch.bfloat16)
        self.head.load_state_dict(load_file(str(Path(path) / "value_head.safetensors")), strict=True)
        self.head.to("cuda:0").eval().requires_grad_(False)

    @torch.inference_mode()
    def reconstruct(self, explanation):
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("NLA_AR_TEXT: a valid AV explanation is required")
        prompt = self.sidecar.ar_template.format(explanation=explanation)
        ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        suffix = self.sidecar.suffix_ids
        if not suffix or tuple(ids[-len(suffix) :]) != suffix:
            raise ValueError("NLA_AR_SUFFIX: final-token alignment differs from released AR suffix")
        tokens = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        hidden = self.model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
        ).last_hidden_state[0, -1]
        result = self.head(hidden).float().cpu()
        if not torch.isfinite(result).all() or result.norm() <= 0:
            raise ValueError("NLA_AR_VECTOR: invalid reconstruction")
        return result


def cosine(a, b):
    if a.norm() <= 0 or b.norm() <= 0:
        raise ValueError("NLA_COSINE_ZERO: a zero vector has no direction")
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)))


def reconstruction_metrics(reconstruction, activation, scale):
    cos = cosine(reconstruction, activation)
    predicted = F.normalize(reconstruction.float(), dim=0) * scale
    target = F.normalize(activation.float(), dim=0) * scale
    mse = float((predicted - target).square().mean())
    if not math.isclose(mse, 2 * (1 - cos), rel_tol=1e-4, abs_tol=1e-5):
        raise ValueError("NLA_MSE_NORMALIZATION: direction-MSE identity failed")
    return {"cosine": cos, "direction_mse": mse, "reconstruction_norm": float(reconstruction.norm())}
