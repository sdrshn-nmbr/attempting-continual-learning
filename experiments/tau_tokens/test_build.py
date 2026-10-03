"""CPU tests for build.py on one real leaderboard run: task_099, Qwen 3.8 Max, trial 1, with message raw_data removed.
Run inside the tau2 v1.0.1 checkout:
  cd <tau2-bench> && uv run --with transformers --with pytest pytest <repo>/experiments/tau_tokens
"""
import json
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.domains.banking_knowledge.environment import get_tasks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "task_099_qwen3-8-max_trial1.json"


@pytest.fixture(scope="module")
def case():
    fixture = json.loads(FIXTURE.read_text())
    task = next(task for task in get_tasks() if task.id == "task_099")
    simulation = SimulationRun.model_validate(fixture["simulation"])
    context = build.task_context(task)
    prepared = build.prepare(task, simulation, context)
    tokenizer = AutoTokenizer.from_pretrained(build.TOKENIZER)
    records, unscored = build.decisions_for(task, fixture["source"], simulation, prepared, context, tokenizer)
    return {"context": context, "prepared": prepared, "records": {r["action_id"]: r for r in records},
            "unscored": unscored}


def test_encode_reproduces_json_dumps_and_spans_leaves_inside_nested_json():
    value = {"agent_tool_name": "file_dispute_4829",
             "arguments": json.dumps({"note": 'said "no" \u2014 twice', "amount": 14.5, "keep": True, "ids": ["a1", None]})}
    text, spans = build.encode(value)
    assert text == json.dumps(value)
    assert [(path, nesting) for path, _, _, _, nesting in spans] == [
        ("agent_tool_name", 0), ("arguments.note", 1), ("arguments.amount", 1), ("arguments.keep", 1),
        ("arguments.ids[0]", 1), ("arguments.ids[1]", 1)]
    for _, start, end, leaf, nesting in spans:
        assert text[start:end] == build.encoded_leaf(leaf, nesting)
    assert text[spans[1][1]:spans[1][2]] == 'said \\\\\\"no\\\\\\" \\\\u2014 twice'


def test_strip_removes_every_search_call_and_its_result(case):
    history = case["prepared"]["history"]
    assert dict(case["prepared"]["removed"]) == {"KB_search_bm25": 2, "KB_search_dense": 2, "shell": 13}
    calls = [call for message in history if isinstance(message, AssistantMessage) for call in message.tool_calls or []]
    results = [message for message in history if isinstance(message, ToolMessage)]
    assert {call.name for call in calls} <= case["context"]["allowed"]
    assert sorted(message.id for message in results) == sorted(call.id for call in calls)


def test_alignment_matches_the_hand_checked_positions(case):
    matched = [(action.action_id, message, call) for action, message, call in case["prepared"]["matched"]]
    assert matched == [("099_0", 8, 0), ("099_1", 11, 0), ("099_2", 13, 0), ("099_3", 8, 1)]
    assert case["prepared"]["missing"] == []
    assert case["unscored"] == []
    same_turn = case["records"]["099_3"]["text"].rsplit("<|im_start|>assistant\n", 1)[1]
    assert same_turn.startswith('<tool_call>\n{"name": "log_verification"')
    assert '</tool_call>\n<tool_call>\n{"name": "get_referrals_by_user"' in same_turn


def test_documents_only_values_never_appear_before_their_span(case):
    unlock = case["records"]["099_1"]["values"]
    assert [(v["path"], v["value"], v["label"]) for v in unlock] == [
        ("agent_tool_name", "get_all_user_accounts_by_user_id_3847", "documents_only")]
    for record in case["records"].values():
        for value in record["values"]:
            if value["label"] == "documents_only":
                assert build.leaf_text(value["value"]) not in record["text"][:value["start"]]
                assert build.leaf_text(value["value"]) in case["context"]["documents"]
        assert "KB_search" not in record["text"] and '"name": "shell"' not in record["text"]
        assert "doc_bank_accounts_bank_accounts__general__009" not in record["text"]


def test_decision_text_ends_with_the_gold_call_in_written_form(case):
    call = case["records"]["099_2"]
    assert call["text"].endswith('"arguments": "{\\"user_id\\": \\"ky83m9p2t6\\"}"}}\n</tool_call>')
    assert [(v["path"], v["nesting"], v["label"]) for v in call["values"]] == [
        ("agent_tool_name", 0, "copy"), ("arguments.user_id", 1, "copy")]
    for record in case["records"].values():
        build.verify(record, case["prepared"]["removed"])
