import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from native import load_tokenizer

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "pilot.json").read_text())


@pytest.fixture
def diagnostic_config():
    return json.loads((ROOT / "native_diagnostic.json").read_text())


@pytest.fixture
def tokenizer():
    words = ["<unk>", "<pad>", "<eos>", "start", "user", "end", "assistant", "red", "blue", "green", "yellow"]
    words += [f"word{i}" for i in range(40)]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )
    tokenizer.chat_template = "start user {{ messages[0]['content'] }} end assistant "
    return tokenizer


@pytest.fixture
def tiny_model(tokenizer):
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    return model.eval()


@pytest.fixture
def native_paths(config):
    paths = {}
    for role, key in (("base", "model_path"), ("av", "av_model_path"), ("ar", "ar_model_path")):
        local = ROOT / ".fixtures" / role
        path = local if local.is_dir() else Path(config[key])
        if not (path / "tokenizer.json").is_file():
            pytest.skip(
                "Native tokenizer integration requires pinned cached tokenizer files; pure alignment tests still run offline"
            )
        paths[role] = path
    return paths


@pytest.fixture
def native_tokenizers(native_paths):
    return {role: load_tokenizer(path) for role, path in native_paths.items()}
