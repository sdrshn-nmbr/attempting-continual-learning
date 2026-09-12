import copy
import json
from pathlib import Path

import pytest

from data import Reservoir, audit_corpus, build_corpus, scheduled_batch, stream_spec


def configuration():
    return json.loads((Path(__file__).parents[1] / "configs/stream_1103.json").read_text())


def test_independent_streams_and_determinism():
    seeds = [1103, 2207, 3301, 4409]
    specs = [stream_spec(seed) for seed in seeds]
    assert len({json.dumps(spec["conventions"], sort_keys=True) for spec in specs}) == 4
    assert len({tuple(spec["order"]) for spec in specs}) >= 3
    assert specs[0] == stream_spec(1103)
    digests = []
    for seed in seeds:
        config = {**configuration(), "data_seed": seed}
        spec, corpus = build_corpus(config)
        audit = audit_corpus(spec, corpus)
        assert audit["rows"] == audit["unique_scenes"] == audit["unique_prompts"]
        digests.append(audit["corpus_sha256"])
        for family in spec["order"]:
            assert all(len(row.calls) == 1 for row in corpus["primitive"][family]["train"])
            assert {row.pattern for row in corpus["workflow"][family]["train"]} == {"0", "1"}
            assert {row.pattern for row in corpus["workflow"][family]["novel_test"]} == {"2", "3"}
    assert len(set(digests)) == 4


def test_leakage_audit_rejects_reused_scene():
    spec, corpus = build_corpus(configuration())
    corpus["primitive"]["records"]["test"][0] = corpus["primitive"]["records"]["train"][0]
    with pytest.raises(RuntimeError, match="SPLIT_LEAKAGE"):
        audit_corpus(spec, corpus)


def test_replay_is_fixed_total_unique_exposed_train_only():
    spec, corpus = build_corpus(configuration())
    buffer = Reservoir(7, 17)
    exposed = set()
    for family in spec["order"]:
        rows = corpus["primitive"][family]["train"][:16]
        exposed.update(row.id for row in rows)
        buffer.observe(rows + rows)
        assert len(buffer.rows) == 7
        assert len({row.id for row in buffer.rows}) == 7
        assert {row.id for row in buffer.rows} <= exposed
    assert buffer.record()["seen_unique"] == 64
    with pytest.raises(ValueError, match="REPLAY_LEAKAGE"):
        buffer.observe(corpus["primitive"]["text"]["test"])


def test_matched_current_schedule_and_replay_replacement():
    _, corpus = build_corpus(configuration())
    seed = [17, "primitive", "files"]
    buffer = Reservoir(64, 17)
    buffer.observe(corpus["primitive"]["records"]["train"][:64])
    new_ids, old_ids = [], []
    for step in range(10):
        full = scheduled_batch(corpus["primitive"]["files"]["train"], step, 4, seed)
        assert full == scheduled_batch(corpus["primitive"]["files"]["train"], step, 4, seed)
        current = full[:2]
        old = buffer.sample(2, seed, step, "files")
        assert len(current + old) == len(full)
        assert all(row.family == "records" for row in old)
        new_ids += [row.id for row in current]
        old_ids += [row.id for row in old]
    assert len(new_ids) == len(old_ids) == 20
    with pytest.raises(ValueError, match="TRAIN_SPLIT_REQUIRED"):
        scheduled_batch(corpus["primitive"]["files"]["test"], 0, 4, 17)
    dirty = copy.deepcopy(buffer)
    dirty.observe(corpus["primitive"]["files"]["train"][:64])
    with pytest.raises(ValueError, match="REPLAY_CURRENT_TASK"):
        dirty.sample(2, 17, 0, "files")
