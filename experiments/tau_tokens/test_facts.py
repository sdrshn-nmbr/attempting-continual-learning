"""CPU tests for facts.py, run inside a tau2 v1.0.1 checkout:
  cd <tau2-bench> && uv run --with torch --with transformers --with httpx --with pytest pytest <this>
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facts

BLOCK = ("| Fee | Amount |\n| --- | --- |\n| Monthly maintenance | $10.00 |\n| Returned deposit | $10.00 |\n"
         "| Foreign ATM withdrawal | 2.5% of amount (max $6.00) |\n| Overdraft transfer | 0 |\n"
         "- The primary account holder must be at least 17.\n- Debit card daily purchase limit: $1,500\n"
         "- Disputes filed within 60 days of the 2024 statement.\n- Call option 1 or use open_bank_account_4821.\n")


def test_candidates_are_the_values_that_occur_once():
    assert facts.candidates(BLOCK) == ["2.5%", "$6.00", "17", "$1,500", "60"]


def test_edit_changes_only_the_questioned_values_and_keeps_their_form():
    answers = ["$6.00", "17", "$1,500", "2.5%"]
    edited, edits = facts.edit_block(BLOCK, answers, "task_x")
    for answer in answers:
        prefix, amount, decimals, comma, suffix = facts.parse(answer)
        new = facts.parse(edits[answer])[1]
        assert facts.render(prefix, new, decimals, comma, suffix) == edits[answer]
        assert new != amount and new > 0
    restored = edited
    for answer in answers:
        restored = restored.replace(edits[answer], answer, 1)
    assert restored == BLOCK
    assert edited.count("$10.00") == 2 and "$1,500" not in edited
    assert facts.edit_block(BLOCK, answers, "task_x") == (edited, edits)


def test_edit_skips_a_value_with_no_edit_outside_the_avoided_text():
    avoid = "".join(str(n) for n in range(1, 40))
    edited, edits = facts.edit_block(BLOCK, ["17", "$6.00"], "task_x", avoid=avoid)
    assert "17" not in edits and "$6.00" in edits
    assert edited == BLOCK.replace("$6.00", edits["$6.00"])


def test_edit_refuses_a_value_that_occurs_twice():
    with pytest.raises(SystemExit, match="NOT_A_SINGLE_OCCURRENCE"):
        facts.edit_block(BLOCK, ["$10.00"], "task_x")
