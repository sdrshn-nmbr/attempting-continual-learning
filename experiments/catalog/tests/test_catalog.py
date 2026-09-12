import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from factory import local_asset
from portallib import (
    ChoiceDataset,
    ChoiceExample,
    PortalBase,
    PortalConfig,
    PortalModel,
)
from run import (
    EvidenceEvaluator,
    paired_statistics,
    peft_reload_check,
    roundtrip_portal,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


class CatalogIntegration(unittest.TestCase):
    def test_same_size_configuration_and_weight_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "snapshot"
            path.mkdir()
            config = b'{"alpha":16}\n'
            weights = b"weights"
            (path / "config.json").write_bytes(config)
            (path / "model.safetensors").write_bytes(weights)
            asset = {
                "repo": "test/model",
                "revision": "0" * 40,
                "relative_path": "snapshot",
                "files": [
                    {
                        "path": "config.json",
                        "bytes": len(config),
                        "sha256": None,
                        "git_blob_sha1": hashlib.sha1(
                            f"blob {len(config)}\0".encode() + config
                        ).hexdigest(),
                    },
                    {
                        "path": "model.safetensors",
                        "bytes": len(weights),
                        "sha256": hashlib.sha256(weights).hexdigest(),
                        "git_blob_sha1": None,
                    },
                ],
            }
            manifest = {"assets": {"base": asset}}
            self.assertEqual(local_asset(manifest, root, "base"), path)
            (path / "config.json").write_bytes(b'{"alpha":32}\n')
            with self.assertRaisesRegex(
                ValueError, "SNAPSHOT_CONTENT_MISMATCH.*config.json"
            ):
                local_asset(manifest, root, "base")
            (path / "config.json").write_bytes(config)
            (path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(
                ValueError, "SNAPSHOT_CONTENT_MISMATCH.*model.safetensors"
            ):
                local_asset(manifest, root, "base")

    def test_real_qwen_native_export_reload_and_paired_identity(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)
        vocabulary = {
            token: index
            for index, token in enumerate(
                [
                    "[UNK]",
                    "[PAD]",
                    "[BOS]",
                    "[EOS]",
                    "red",
                    "blue",
                    "is",
                    "color",
                    "animal",
                    "yes",
                    "no",
                    "dog",
                ]
            )
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(vocabulary),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                use_cache=False,
            )
        )
        model.eval()
        model.requires_grad_(False)
        portal_config = PortalConfig.from_model(
            model,
            tasks=["fixture"],
            base_model_name_or_path="catalog/fixture",
            rank=2,
            alpha=4,
            d_z=4,
            d_layer=4,
            hidden=8,
            d_core=8,
        )
        portal = PortalModel(portal_config, torch.randn(1, 4))
        with torch.no_grad():
            for parameter in portal.parameters():
                parameter.normal_(mean=0, std=0.1)
        portal.requires_grad_(False)
        base = PortalBase("catalog/fixture", model, tokenizer)
        train = [ChoiceExample("fixture", "dog is animal ", ("yes", "no"), 0)]
        rows = [
            ChoiceExample("fixture", "red is color ", ("yes", "no"), 0),
            ChoiceExample("fixture", "blue is animal ", ("yes", "no"), 1),
        ]
        dataset = ChoiceDataset(train, rows)
        root = Path(__file__).resolve().parents[1] / "outputs" / "fixtures"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            output = Path(directory)
            evaluator = EvidenceEvaluator(
                output=output / "predictions.jsonl",
                condition="base",
                max_prompt=32,
                batch_size=2,
            )
            evaluator.evaluate(base, dataset)
            original = evaluator.records.copy()
            evaluator.records.clear()
            evaluator.condition = "portal"
            evaluator.evaluate(base, dataset, portal=portal)
            adapted = evaluator.records.copy()
            self.assertTrue(
                any(
                    abs(a - b) > 1e-7
                    for before, after in zip(original, adapted, strict=True)
                    for a, b in zip(before["scores"], after["scores"], strict=True)
                )
            )
            stats = paired_statistics(original, list(reversed(adapted)), 11)
            self.assertEqual(stats["paired_examples"], 2)
            with self.assertRaises(ValueError):
                paired_statistics(original, adapted + [adapted[0]], 11)
            native = roundtrip_portal(portal, output, ["fixture"])
            self.assertTrue(native["generated_factors_exact_equal"])
            parity = peft_reload_check(
                base,
                portal,
                dataset,
                "fixture",
                output,
                {"max_prompt": 32, "batch_size": 2, "peft_parity_atol": 1e-6},
            )
            self.assertTrue(parity["reload_scores_exact_equal"])


if __name__ == "__main__":
    unittest.main()
