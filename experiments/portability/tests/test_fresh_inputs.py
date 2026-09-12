import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import fresh_inputs
import pytest
from data import SEQUENCE_TASKS, sequence_splits, write_json


def test_full_unused_domain_and_choice_oracle(tmp_path, monkeypatch):
    root = Path(fresh_inputs.__file__).parent
    spec = json.loads((root / "configs/sequence_native_replay.json").read_text())[
        "sequence"
    ]
    value = fresh_inputs.build_fresh_fixture(spec)
    splits, _ = sequence_splits(spec)
    prior = {row.group for rows in splits.values() for row in rows}
    new = {row["group"] for row in value["rows"]}
    domain = {" ".join(map(str, row)) for row in itertools.product(range(8), repeat=3)}
    assert len(value["rows"]) == 864
    assert len(prior) == 224 and len(new) == 288
    assert not prior & new and prior | new == domain
    mappings = json.loads((root / spec["path"]).read_text())["provenance"]["rules"]
    for task in SEQUENCE_TASKS:
        rows = [row for row in value["rows"] if row["task"] == task]
        assert Counter(row["gold_idx"] for row in rows) == dict.fromkeys(range(4), 72)
        for row in rows:
            assert len(set(row["choices"])) == 4
            expected = " " + " ".join(
                str(mappings[task][int(x)]) for x in row["group"].split()
            )
            assert row["choices"][row["gold_idx"]] == expected
            assert (
                row["prompt"]
                == f"Apply the {task} code.\nInput: {row['group']}\nOutput:"
            )
    path = tmp_path / "fresh.json"
    write_json(path, value)
    loaded, _ = fresh_inputs.load_fresh_rows(
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    assert len(loaded) == 864
    value["rows"][0]["gold_idx"] = (value["rows"][0]["gold_idx"] + 1) % 4
    write_json(path, value)
    with pytest.raises(ValueError, match="ROUNDTRIP_MISMATCH"):
        fresh_inputs.load_fresh_rows(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
