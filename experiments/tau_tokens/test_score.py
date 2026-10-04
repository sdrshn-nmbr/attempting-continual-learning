"""CPU test for score.py:
  uv run --no-project --with torch --with transformers --with pytest --with numpy pytest experiments/tau_tokens/test_score.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "still_repro"))
import attention_matching
import score
from still import prefill
from test_still import tiny_model


@pytest.mark.parametrize("attention", ["default", attention_matching.NAME])
def test_chunked_value_scores_match_one_uninterrupted_forward_over_prefix_and_rest(attention):
    model = tiny_model()
    if attention != "default":
        model.set_attn_implementation(attention)
        assert model.config._attn_implementation == attention_matching.NAME
    torch.manual_seed(2)
    prefix, rest = torch.randint(0, 97, (40,)).tolist(), torch.randint(0, 97, (30,)).tolist()
    values = [{"path": "a", "label": "copy", "token_start": 5, "token_end": 8},
              {"path": "b", "label": "documents_only", "token_start": 20, "token_end": 23}]
    scores = score.value_scores(model, prefill(model, torch.tensor([prefix])), len(prefix), rest, values, chunk=7)
    with torch.no_grad():
        whole = torch.log_softmax(model(input_ids=torch.tensor([prefix + rest])).logits[0].float(), dim=-1)
    for value, result in zip(values, scores):
        expected = [whole[len(prefix) + t - 1, rest[t]].item() for t in range(value["token_start"], value["token_end"])]
        assert torch.allclose(torch.tensor(result["token_logprobs"]), torch.tensor(expected), atol=1e-4)
        greedy = all(whole[len(prefix) + t - 1].argmax().item() == rest[t]
                     for t in range(value["token_start"], value["token_end"]))
        assert result["greedy_exact"] == greedy
        assert abs(result["mean_logprob"] - sum(expected) / len(expected)) < 1e-4
