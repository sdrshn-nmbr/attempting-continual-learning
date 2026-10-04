"""CPU tests for conditions.py:
  uv run --no-project --with torch --with transformers --with pytest --with numpy pytest experiments/tau_tokens/test_conditions.py
"""
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "still_repro"))
import conditions
import cache_format
from still import BudgetedStillCompactor, StillCompactor, continue_from
from test_still import tiny_model

GOLD = ("{{component:policy_header}}\n\n## Documents you need to solve the problem\n\nThe following documents contain "
        "the information you need to answer the customer's question and determine what actions to take.\n\n"
        "{{component:additional_instructions}}\n\n<required_documents>\n{{required_documents}}\n</required_documents>\n")
PLAIN = "{{component:policy_header}}\n\n{{component:additional_instructions}}\n"
DOCUMENTS = [{"title": "Opening Accounts", "content": "Use open_bank_account_4821."},
             {"title": "Fees", "content": "Monthly fee: $12."}]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(conditions.TOKENIZER)


def block():
    return conditions.documents_block(*conditions.block_template(GOLD, PLAIN), DOCUMENTS)


def test_documents_block_is_tau2_wording_around_the_documents():
    assert block() == ("## Documents you need to solve the problem\n\nThe following documents contain the information "
                       "you need to answer the customer's question and determine what actions to take.\n\n"
                       "<required_documents>\n## Opening Accounts\n\nUse open_bank_account_4821.\n\n---\n\n"
                       "## Fees\n\nMonthly fee: $12.\n</required_documents>\n\n")
    with pytest.raises(SystemExit, match="GOLD_TEMPLATE_IS_NOT_PLAIN_PLUS_BLOCK"):
        conditions.block_template(GOLD, PLAIN + "extra")


def test_conditions_share_every_token_after_the_prefix_and_spans_hit_the_value(tokenizer):
    value = "open_bank_account_4821"
    text = (conditions.HEADER + "<instructions>\nHelp.\n</instructions><|im_end|>\n<|im_start|>assistant\n<tool_call>\n"
            '{"name": "unlock_discoverable_agent_tool", "arguments": {"agent_tool_name": "' + value + '"}}\n</tool_call>')
    start = text.index(value)
    record = {"decision_id": "t:0", "task_id": "t", "action_name": "unlock_discoverable_agent_tool", "text": text,
              "values": [{"path": "agent_tool_name", "value": value, "label": "documents_only", "nesting": 0,
                          "start": start, "end": start + len(value)}]}
    header_ids, prefix_ids = conditions.task_prefix(tokenizer, block())
    inputs = conditions.decision_inputs(tokenizer, record, header_ids, prefix_ids, block())
    assert header_ids + inputs["rest_ids"] == tokenizer(text, add_special_tokens=False).input_ids
    assert prefix_ids[:len(header_ids)] == header_ids
    span = inputs["values"][0]
    assert span["exact"]
    assert tokenizer.decode(inputs["rest_ids"][span["token_start"]:span["token_end"]]) == value


def packed_am(layers, heads, dim, lengths):
    return {**cache_format.pack([(torch.randn(heads, max(lengths), dim),
                                  torch.tensor([[0.0] * n + [float("-inf")] * (max(lengths) - n) for n in lengths]),
                                  torch.randn(heads, max(lengths), dim)) for _ in range(layers)]), "header": 3}


def test_prefix_states_have_the_condition_lengths_and_continue_at_the_full_prefix_length():
    model = tiny_model()
    torch.manual_seed(1)
    prefix = {"task_id": "t", "header_ids": [1, 2, 3], "prefix_ids": [1, 2, 3] + torch.randint(4, 97, (297,)).tolist()}
    layers, heads, dim = model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.head_dim
    am = packed_am(layers, heads, dim, [100, 228])
    compactor = StillCompactor(model.config, slots=conditions.SLOTS)
    expected = {"none": (3, 3), "full": (300, 300), "streaming": (164, 300), "still": (164, 300), "am": (228, 300)}
    rest = torch.randint(4, 97, (1, 7))
    for condition, (physical, logical) in expected.items():
        pairs, start, bias = conditions.prefix_state(model, condition, prefix, compactor=compactor, am=am)
        assert (pairs[0][0].shape[-2], start) == (physical, logical), condition
        assert (bias is not None) == (condition == "am")
        assert continue_from(model, pairs, rest, start).logits.shape == (1, 7, 97)
    _, _, bias = conditions.prefix_state(model, "am", prefix, am=am)
    assert torch.isinf(bias[0][0, 0, 100:]).all() and torch.isfinite(bias[0][0, 1]).all()
    with pytest.raises(SystemExit, match="AM_CACHE_SHAPE"):
        conditions.prefix_state(model, "am", prefix, am=dict(am, header=4))
    with pytest.raises(SystemExit, match="AM_CACHE_SHAPE"):
        conditions.prefix_state(model, "am", prefix, am=packed_am(layers, heads, dim, [200, 228]))
    budgeted = BudgetedStillCompactor(model.config, [[100, 228], [164, 164]])
    pairs, start, bias = conditions.prefix_state(model, "still", prefix, compactor=budgeted)
    assert (pairs[0][0].shape[-2], start, len(bias)) == (228, 300, layers)
    assert torch.isinf(bias[0][0, 0, 100:]).all() and continue_from(model, pairs, rest, start).logits.shape == (1, 7, 97)
    with pytest.raises(SystemExit, match="STILL_COMPACTOR_SLOTS"):
        conditions.prefix_state(model, "still", prefix, compactor=BudgetedStillCompactor(model.config, [[200, 228]] * 2))
    with pytest.raises(SystemExit, match="STILL_COMPACTOR_SLOTS"):
        conditions.prefix_state(model, "still", prefix, compactor=StillCompactor(model.config, slots=40))
