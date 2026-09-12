import json
from dataclasses import replace

import pytest

from data import make_example, stream_spec
from sandbox import ExecutionError, FormatError, apply_operation, execute, grade_text, parse_calls


def test_oracle_execution_does_not_mutate_input():
    spec = stream_spec(1103)
    for family in spec["order"]:
        for index in range(8):
            row = make_example(spec, family, "workflow", "test", index)
            before = json.dumps(row.state, sort_keys=True)
            record = grade_text(json.dumps(row.calls), row, spec["conventions"])
            assert record["correct"]
            assert json.dumps(row.state, sort_keys=True) == before


@pytest.mark.parametrize(
    "text",
    [
        "```json\n[]\n```",
        "[] trailing",
        "[]",
        "{}",
        '[{"tool":"a","tool":"b","args":{}}]',
        '[{"tool":"a","args":{"n":1,"n":2}}]',
        '[{"tool":"a","args":{"n":NaN}}]',
        '[{"tool":"a","args":{},"explanation":"hi"}]',
        '[{"tool":"a","args":{}}]' * 2,
    ],
)
def test_strict_whole_json(text):
    with pytest.raises(FormatError):
        parse_calls(text)


def test_whitespace_key_order_and_semantic_equivalence_are_valid():
    spec = stream_spec(1103)
    row = make_example(spec, "files", "primitive", "test", 0)
    call = row.calls[0]
    output = json.dumps([{"args": call["args"], "tool": call["tool"]}], indent=2)
    assert grade_text(output, row, spec["conventions"])["correct"]
    source, destination = call["args"]["source"], call["args"]["destination"]
    alternative = [
        {"tool": spec["conventions"]["files"]["write"], "args": {"path": destination, "text": row.state[source]}}
    ]
    grade = grade_text(json.dumps(alternative), row, spec["conventions"])
    assert grade["correct"] and not grade["canonical_calls_match"]


def test_format_and_execution_errors_are_separate():
    spec = stream_spec(1103)
    row = make_example(spec, "records", "primitive", "test", 2)
    wrong = [{"tool": row.calls[0]["tool"], "args": {"count": True}}]
    grade = grade_text(json.dumps(wrong), row, spec["conventions"])
    assert grade["format_valid"] and not grade["executable"] and grade["error"] == "ARGUMENT_TYPES"
    wrong[0]["args"] = {"count": 1}
    grade = grade_text(json.dumps(wrong), row, spec["conventions"])
    assert grade["executable"] and not grade["correct"]
    assert not grade_text(json.dumps(row.calls), row, spec["conventions"], native_eos=False)["format_valid"]
    assert not grade_text(json.dumps(row.calls), row, spec["conventions"], padding_only=False)["format_valid"]


def test_operations_against_hand_computed_states():
    state = [
        {"id": "a", "team": "red", "score": 3},
        {"id": "b", "team": "blue", "score": 5},
        {"id": "c", "team": "red", "score": 1},
    ]
    state = apply_operation(state, "records", "filter", {"field": "team", "value": "red"})
    state = apply_operation(state, "records", "sort", {"field": "score", "descending": False})
    state = apply_operation(state, "records", "project", {"fields": ["id"]})
    assert state == [{"id": "c"}, {"id": "a"}]
    state = apply_operation(["ALERT x", "Note y"], "text", "lower", {})
    state = apply_operation(state, "text", "keep", {"contains": "alert"})
    state = apply_operation(state, "text", "join", {"separator": ";"})
    assert state == "alert x"
    state = apply_operation({"a": "old"}, "files", "copy", {"source": "a", "destination": "b"})
    state = apply_operation(state, "files", "write", {"path": "b", "text": "new"})
    state = apply_operation(state, "files", "move", {"source": "b", "destination": "c"})
    assert state == {"a": "old", "c": "new"}
    state = apply_operation({}, "calendar", "add", {"id": "a", "start": 30, "title": "old"})
    state = apply_operation(state, "calendar", "shift", {"id": "a", "minutes": 15})
    state = apply_operation(state, "calendar", "rename", {"id": "a", "title": "new"})
    assert state == {"a": {"start": 45, "title": "new"}}


def test_wrong_domain_and_invalid_state_transitions():
    spec = stream_spec(1103)
    row = make_example(spec, "files", "primitive", "test", 0)
    wrong = [{"tool": spec["conventions"]["calendar"]["cancel"], "args": {"id": "x"}}]
    with pytest.raises(ExecutionError, match="UNKNOWN_TOOL"):
        execute(wrong, row.state, row.family, spec["conventions"])
    with pytest.raises(ExecutionError, match="DESTINATION_EXISTS"):
        apply_operation({"a": "1", "b": "2"}, "files", "copy", {"source": "a", "destination": "b"})
    with pytest.raises(ExecutionError, match="TIME_BOUNDS"):
        apply_operation({"a": {"start": 5}}, "calendar", "shift", {"id": "a", "minutes": -10})
    altered = replace(row, expected={})
    assert not grade_text(json.dumps(row.calls), altered, spec["conventions"])["correct"]
