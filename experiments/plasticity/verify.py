import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from run import (
    EncodedCorpus,
    activate,
    adapter_hash,
    checkpoint_load,
    frozen_adapter_hash,
    frozen_probe_hash,
    lora_config,
    new_optimizer,
    next_logits,
    save_and_verify,
    tree_hash,
    write_json,
)
from tasks import LABELS, audit_corpus, build_corpus, digest, oracle, stage_batches
from transformers import (
    AutoTokenizer,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)


def tiny_base(config):
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 3, 3],
            "partial_rotary_factor": 1.0,
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(107)
        return Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(
                text_config=text,
                vision_config=vision,
                image_token_id=60,
                video_token_id=61,
                vision_start_token_id=62,
                vision_end_token_id=63,
            )
        )


class TinyBatch:
    def batch(self, rows, observed=False):
        ids = torch.tensor([[7, 8 + row["x"], 25 + row["y"], 9] for row in rows])
        labels = torch.tensor(
            [
                2 + LABELS.index(row["observed_label"] if observed else row["label"])
                for row in rows
            ]
        )
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels


def update(model, optimizer, encoded, rows):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, labels = encoded.batch(rows, observed=True)
    loss = F.cross_entropy(next_logits(model, inputs), labels)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
        error_if_nonfinite=True,
    )
    assert torch.isfinite(loss) and float(norm) > 0
    optimizer.step()
    return float(loss.detach())


def verify_data(config):
    corpus = build_corpus(config)
    audit = audit_corpus(corpus)
    assert audit["sha256"] == audit_corpus(build_corpus(config))["sha256"]
    for x, y, expected in [(1, 1, "A"), (1, 12, "B"), (12, 1, "C"), (12, 12, "D")]:
        assert oracle("MERA", 0, x, y, 3, True, True) == expected
        assert oracle("MERA", 1, x, y, 3, True, False) == expected
        assert (
            oracle("MERA", 1, x, y, 3, True, True)
            == LABELS[(LABELS.index(expected) + 1) % 4]
        )
        assert (
            oracle("MERA", 2, x, y, 3, False, False)
            == LABELS[(LABELS.index(expected) + 2) % 4]
        )
    poisoned = copy.deepcopy(corpus)
    poisoned["A0"]["train"].append(copy.deepcopy(corpus["A0"]["test_composition"][0]))
    caught = False
    try:
        audit_corpus(poisoned)
    except AssertionError:
        caught = True
    assert caught, "data-leakage negative control was not rejected"
    for stage in ("B", "C"):
        streams = {
            arm: list(stage_batches(corpus, stage, arm, config))
            for arm in config["arms"]
        }
        assert (
            streams["continue"]
            == streams["optimizer_reset"]
            == streams["capacity_refresh"]
        )
        for plain, replay in zip(streams["continue"], streams["replay"], strict=True):
            assert len(plain[0]) == len(replay[0]) == config["batch_size"]
            assert replay[1].count("replay") == config["replay_per_batch"]
            assert replay[1].count("revision") == config["revision_per_batch"]
            plain_current = [
                row["id"] for row, role in zip(*plain, strict=True) if role == "new"
            ]
            replay_current = [
                row["id"] for row, role in zip(*replay, strict=True) if role == "new"
            ]
            assert replay_current == plain_current[: -config["replay_per_batch"]]
    return corpus, audit


def verify_adapters(config, corpus, output):
    encoded = TinyBatch()
    rows = corpus["A0"]["train"][:8]
    torch.manual_seed(config["seed"])
    base = tiny_base(config)
    model = get_peft_model(base, lora_config(base, config), adapter_name="acquired")
    activate(model, ["acquired"], "acquired")
    optimizer = new_optimizer(model, config)
    before = adapter_hash(model)
    frozen = frozen_probe_hash(model)
    losses = [update(model, optimizer, encoded, rows) for _ in range(3)]
    assert adapter_hash(model) != before
    assert frozen_probe_hash(model) == frozen
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    with patch("run.load_base", tiny_base):
        loaded, loaded_optimizer, proof = save_and_verify(
            model,
            optimizer,
            ["acquired"],
            "acquired",
            output / "A",
            config,
            encoded,
            rows[:4],
        )
        assert proof["logits_exact"]
        assert digest(tree_hash(loaded_optimizer.state_dict())) == digest(
            tree_hash(optimizer.state_dict())
        )
        reset = new_optimizer(loaded, config)
        assert not reset.state and loaded_optimizer.state
        continue_model, continue_optimizer, _ = checkpoint_load(output / "A", config)
        update(continue_model, continue_optimizer, encoded, rows)
        assert all(
            float(state["step"]) == 4 for state in continue_optimizer.state.values()
        )
        update(loaded, reset, encoded, rows)
        assert all(float(state["step"]) == 1 for state in reset.state.values())
        adapters = ["acquired"]
        proofs = [proof]
        for stage in ("B", "C"):
            loaded.eval()
            inputs, _ = encoded.batch(rows)
            with torch.inference_mode():
                preceding = next_logits(loaded, inputs)
            current = f"stage_{stage}"
            loaded.add_adapter(
                current,
                LoraConfig(
                    r=config["rank"],
                    lora_alpha=config["lora_alpha"],
                    target_modules=list(loaded.peft_config["acquired"].target_modules),
                    lora_dropout=0,
                    bias="none",
                ),
            )
            adapters.append(current)
            activate(loaded, adapters, current)
            with torch.inference_mode():
                assert torch.equal(next_logits(loaded, inputs), preceding), (
                    "adding zero-initialized capacity changed outputs"
                )
            assert (
                sum(
                    parameter.numel()
                    for parameter in loaded.parameters()
                    if parameter.requires_grad
                )
                == trainable_count
            )
            historical = frozen_adapter_hash(loaded)
            optimizer = new_optimizer(loaded, config)
            update(loaded, optimizer, encoded, rows)
            assert frozen_adapter_hash(loaded) == historical
            loaded, optimizer, proof = save_and_verify(
                loaded,
                optimizer,
                list(adapters),
                current,
                output / stage,
                config,
                encoded,
                rows[:4],
            )
            proofs.append(proof)
    return {
        "real_gradient_updates": True,
        "losses": losses,
        "reload": proofs,
        "optimizer_continue_and_reset_distinguished": True,
        "historical_adapter_weights_frozen": True,
        "capacity_addition_is_initial_noop": True,
        "trainable_parameter_count_matched": trainable_count,
        "model": "random 103744-parameter Qwen3.5 wrapper on CPU; not 4B learning evidence",
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((Path(__file__).parent / "configs" / "screen.json").read_text())
    corpus, audit = verify_data(config)
    result = {
        "data": audit,
        "adapters": verify_adapters(config, corpus, args.output_dir / "checkpoints"),
    }
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        encoded = EncodedCorpus(tokenizer, corpus, config)
        result["tokenizer"] = {
            "label_ids": encoded.label_ids,
            "all_examples_encoded_without_truncation": len(encoded.cache),
            "length_max": max(int(mask.sum()) for _, mask in encoded.cache.values()),
        }
    result["status"] = "passed"
    write_json(args.output_dir / "verification.json", result)
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": str(args.output_dir / "verification.json"),
                "gpu_training_run": False,
            }
        )
    )


if __name__ == "__main__":
    main()
