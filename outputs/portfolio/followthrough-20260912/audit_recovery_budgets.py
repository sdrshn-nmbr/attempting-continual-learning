import argparse
import ast
import hashlib
import json
import math
import random
import re
import tarfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from safetensors.numpy import load_file

PREFIX = "followthrough-20260912-recovery-"
PILOT_CODE = "674800836fba82539ed5e8eff984b367888afa0ec592b3a7e02b6f1192464711"
BUDGETS = (2, 4, 8, 16)
STREAMS = ("train-a", "train-b")
ROLES = {
    "learn": ("train", 32),
    "repair": ("train", 8),
    "gate": ("validation", 10),
    "probe": ("validation", 10),
    "test": ("test", 20),
}
FEATURES = ("confidence", "output", "frozen", "tuned")
HELPER_SOURCE_SHA256 = (
    "45dcf7f5b147b2cc61eb1a6438b396325671b3991f4025dc7a3ff0876fccc3f5"
)
HELPERS_ADAPTED = (
    "require",
    "sha",
    "digest",
    "file_hash",
    "binding",
    "score",
    "rows_for",
    "data_audit",
    "check_updates",
    "close",
    "feature_audit",
)


def require(condition, detail):
    if not condition:
        raise ValueError(f"RECOVERY_INDEPENDENT_AUDIT: {detail}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binding(row):
    return row["text_sha256"], row["intent"], row["target"]


def score(records, expected, codes, counters):
    require(
        Counter(map(binding, records)) == Counter(map(binding, expected)),
        "observation row/target binding",
    )
    correct = Counter()
    counts = Counter()
    for row in records:
        output, target = row["output"], row["target"]
        logits = output["code_logits"]
        require(
            len(logits) == 16 and all(math.isfinite(x) for x in logits),
            "finite 16-code logits",
        )
        prediction = codes[max(range(16), key=logits.__getitem__)]
        require(
            prediction == output["prediction"],
            "recorded prediction differs from raw argmax",
        )
        index, maximum = codes.index(target), max(logits)
        logz = math.log(sum(math.exp(value - maximum) for value in logits))
        logp = [value - maximum - logz for value in logits]
        probability = [math.exp(value) for value in logp]
        ranked = sorted(probability, reverse=True)
        metrics = {
            "top_probability": ranked[0],
            "top_margin": ranked[0] - ranked[1],
            "entropy": -sum(p * lp for p, lp in zip(probability, logp, strict=True)),
            "target_logp": logp[index],
            "target_margin": logits[index]
            - max(value for i, value in enumerate(logits) if i != index),
        }
        for name, value in metrics.items():
            error = abs(value - output[name])
            counters["max_feature_error"] = max(counters["max_feature_error"], error)
            require(
                math.isclose(value, output[name], abs_tol=1e-5, rel_tol=1e-5),
                f"raw feature {name}",
            )
        counters["records_scored"] += 1
        counts[row["intent"]] += 1
        correct[row["intent"]] += prediction == target
    return {intent: correct[intent] / count for intent, count in sorted(counts.items())}


def rows_for(stream, intents, role):
    return [row for intent in intents for row in stream["units"][intent][role]]


def data_audit(data, spec):
    seen = {key: set() for key in ("text", "tokens", "source", "intents")}
    row_count = 0
    groups = []
    require(len(data["codes"]) == len(set(data["codes"])) == 16, "code vocabulary")
    declared = {stream["id"]: stream for stream in spec["streams"]}
    require(set(data["streams"]) == set(declared), "all four predeclared streams")
    for name, stream in data["streams"].items():
        require(
            all(stream[key] == declared[name][key] for key in ("id", "split", "seed")),
            "seed/split identity",
        )
        intents = stream["old"] + stream["new"]
        require(
            len(stream["old"]) == len(stream["new"]) == 8 and len(set(intents)) == 16,
            "old/new intent counts",
        )
        require(
            not seen["intents"].intersection(intents), "predictor source intent overlap"
        )
        seen["intents"].update(intents)
        mapping = []
        for intent in intents:
            unit = stream["units"][intent]
            require(set(unit) == set(ROLES), "dataset roles")
            code = unit["learn"][0]["code"]
            mapping.append(code)
            for role, (split, count) in ROLES.items():
                require(len(unit[role]) == count, "role sample count")
                groups.append((unit[role], split, intent, code))
        require(sorted(mapping) == list(range(16)), "one code per intent")
    backgrounds = {row["intent"] for row in data["lens_fit"]}
    require(
        len(backgrounds) == 16 and not backgrounds.intersection(seen["intents"]),
        "lens background leakage",
    )
    for role, count in (("lens_fit", 32), ("lens_check", 8)):
        require(
            Counter(row["intent"] for row in data[role])
            == {intent: count for intent in backgrounds},
            "lens split counts",
        )
        groups.extend(([row], "train", row["intent"], None) for row in data[role])
    for rows, split, intent, code in groups:
        for row in rows:
            text = " ".join(
                unicodedata.normalize("NFKC", row["text"]).casefold().split()
            )
            require(
                row["text_sha256"] == sha(text.encode()), "normalized text identity"
            )
            require(
                row["intent"] == intent
                and row["source_split"] == split
                and row["code"] == code
                and row["target"] == (None if code is None else data["codes"][code]),
                "dataset row binding",
            )
            require(
                0 < len(row["input_ids"]) <= spec["runtime"]["max_length"],
                "input length",
            )
            keys = {
                "text": row["text_sha256"],
                "tokens": tuple(row["input_ids"]),
                "source": (row["source_split"], row["source_index"]),
            }
            for key, value in keys.items():
                require(value not in seen[key], f"global {key} leakage")
                seen[key].add(value)
            row_count += 1
    return {
        "rows": row_count,
        "stream_intents": len(seen["intents"]),
        "lens_intents": len(backgrounds),
        "global_text_token_source_overlaps": 0,
        "predictor_train_test_intent_overlap": 0,
        "lens_training_cohort_overlap": 0,
        "streams": declared,
    }


def check_updates(trace, rows, steps, batch_size, label):
    require(
        len(trace) == steps
        and [item["step"] for item in trace] == list(range(1, steps + 1)),
        f"{label} steps",
    )
    for item in trace:
        require(
            len(item["rows"]) == batch_size
            and all(0 <= i < len(rows) for i in item["rows"]),
            f"{label} indices",
        )
        require(
            math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"]),
            f"{label} finite updates",
        )
    return [index for item in trace for index in item["rows"]]


def close(actual, expected, label, tolerance=1e-9):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label} keys")
        for key in expected:
            close(actual[key], expected[key], f"{label}/{key}", tolerance)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label} length")
        for i, value in enumerate(expected):
            close(actual[i], value, f"{label}/{i}", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f"{label}: {actual} != {expected}",
        )
    else:
        require(actual == expected, label)


def feature_audit(unit, stream, data, spec, counters):
    features = {name: [] for name in FEATURES if name != "permuted"}
    metrics = (
        "top_probability",
        "entropy",
        "top_margin",
        "target_logp",
        "target_margin",
    )
    for phase in ("before", "after"):
        records = unit["probes"][phase]
        expected = stream["units"][unit["intent"]]["probe"]
        score(records, expected, data["codes"], counters)
        base = [mean(row["output"][key] for row in records) for key in metrics]
        features["confidence"].extend(base[:3])
        features["output"].extend(base)
        for lens in ("frozen", "tuned"):
            features[lens].extend(base)
            for layer in map(str, spec["layers"]):
                score(
                    [{**row, "output": row[lens][layer]} for row in records],
                    expected,
                    data["codes"],
                    counters,
                )
                features[lens].extend(
                    mean(row[lens][layer][key] for row in records)
                    for key in metrics[-2:]
                )
    close(
        unit["features"],
        features,
        "independent pre-action feature averages",
        tolerance=1e-12,
    )
    return features


def read_json(path):
    return json.loads(path.read_text())


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe path {name}")
    return path


def read_archive(root, identity, archives):
    if identity in archives:
        return archives[identity]
    path = root / "code" / f"{identity}.tar"
    closure, members = hashlib.sha256(), {}
    with tarfile.open(path) as archive:
        for item in sorted(archive.getmembers(), key=lambda item: Path(item.name)):
            safe_relative(item.name)
            require(
                item.isfile() and item.name not in members,
                f"archive member {item.name}",
            )
            raw = archive.extractfile(item).read()
            closure.update(item.name.encode() + b"\0" + raw)
            members[item.name] = raw
    require(closure.hexdigest() == identity, f"archive closure {identity}")
    result = {
        "members": members,
        "path": str(path),
        "sha256": file_hash(path),
        "closure": identity,
    }
    archives[identity] = result
    return result


def verify_run(root, suffix, archives):
    directory = root / "runs" / (PREFIX + suffix)
    print(f"AUDIT_RECEIPT {directory.name}", flush=True)
    execution = read_json(directory / "execution.json")
    require(
        execution["status"] == "completed"
        and execution["exit_code"] == 0
        and not execution["timed_out"],
        f"completed execution {suffix}",
    )
    require(execution["task_id"] == directory.name, f"execution task ID {suffix}")
    task, config = (
        read_json(directory / "task.json"),
        read_json(directory / "config.json"),
    )
    require(
        execution["task"] == task and task["config"] == config,
        f"task/config binding {suffix}",
    )
    for name in ("task", "config"):
        require(
            file_hash(directory / f"{name}.json") == execution[f"{name}_sha256"],
            f"{name} hash {suffix}",
        )
    require(
        execution["source_sha256"] == task["source_sha256"],
        f"source task binding {suffix}",
    )
    archive = read_archive(root, execution["source_sha256"], archives)
    collection = read_json(directory / "collection.json")
    require(
        collection["execution_status"] == "completed", f"collection status {suffix}"
    )
    require(
        collection["remote"] == f"/mnt/shared/cl-portfolio/runs/{directory.name}",
        "collection remote",
    )
    actual = {
        str(p.relative_to(directory)): p
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "collection.json"
    }
    require(
        set(actual) == set(collection["files"]),
        f"collection complete file set {suffix}",
    )
    hashes = {}
    for name, path in sorted(actual.items()):
        safe_relative(name)
        require(not path.is_symlink(), f"symlink {path}")
        hashes[name] = file_hash(path)
        require(
            collection["files"][name]
            == {"bytes": path.stat().st_size, "sha256": hashes[name]},
            f"collection content {suffix}/{name}",
        )
    result, sealed = (
        read_json(directory / "study/receipt.json"),
        read_json(directory / "study/seal.json"),
    )
    for label, value in (("receipt", result), ("seal", sealed)):
        require(digest(value["payload"]) == value["sha256"], f"{label} digest {suffix}")
    manifest = {
        name.removeprefix("study/"): value
        for name, value in hashes.items()
        if name.startswith("study/") and name != "study/receipt.json"
    }
    require(manifest == result["payload"]["files"], f"receipt full manifest {suffix}")
    for name, expected in sealed["payload"]["implementation"].items():
        require(
            sha(archive["members"][name]) == expected,
            f"sealed implementation {suffix}/{name}",
        )
    current = execution
    statuses, process_ids, visited = [], [], set()
    while True:
        raw = json.dumps(current, indent=2).encode() + b"\n"
        candidate = f"attempts/{current['attempt_id']}/{sha(raw)}.json"
        require(
            candidate in hashes and (directory / candidate).read_bytes() == raw,
            f"archived receipt {suffix}",
        )
        require(candidate not in visited, f"receipt cycle {suffix}")
        visited.add(candidate)
        statuses.append(current["status"])
        process_ids.append(current.get("pid"))
        previous = current.get("previous_receipt")
        if previous is None:
            break
        safe_relative(previous)
        require(
            previous in hashes and Path(previous).stem == hashes[previous],
            f"receipt predecessor hash {suffix}",
        )
        parent = read_json(directory / previous)
        require(
            parent["task_id"] == execution["task_id"]
            and parent["task_sha256"] == execution["task_sha256"],
            f"receipt predecessor identity {suffix}",
        )
        current = parent
    require(
        statuses == ["completed", "running", "running"]
        and process_ids == [execution["pid"], execution["pid"], None],
        f"attempt chronology {suffix}/{statuses}",
    )
    return {
        "path": directory,
        "hashes": hashes,
        "execution": execution,
        "receipt": result["payload"],
        "seal": sealed["payload"],
        "archive": archive,
        "provenance": {
            "run": str(directory),
            "execution_sha256": hashes["execution.json"],
            "receipt_sha256": hashes["study/receipt.json"],
            "seal_sha256": hashes["study/seal.json"],
            "source_closure": execution["source_sha256"],
            "files_verified": len(hashes),
            "bytes_verified": sum(p.stat().st_size for p in actual.values()),
            "attempt_states_newest_first": statuses,
        },
    }


def doc(run, name):
    return read_json(run["path"] / "study" / name)


def ratio(records):
    return Fraction(
        sum(row["output"]["prediction"] == row["target"] for row in records),
        len(records),
    )


def repair_counts(old, baseline, guard, guard_baseline):
    old_accuracy, old_baseline = Fraction(*old), Fraction(*baseline)
    new_accuracy, new_baseline = Fraction(*guard), Fraction(*guard_baseline)
    gain = old_accuracy - old_baseline
    loss = max(Fraction(0), new_baseline - new_accuracy)
    checks = {
        "old_accuracy": old_accuracy >= Fraction(4, 5),
        "old_gain": gain >= Fraction(3, 10),
        "new_guard": loss <= Fraction(1, 10),
    }
    float_gain = float(old_accuracy) - float(old_baseline)
    float_loss = max(0.0, float(new_baseline) - float(new_accuracy))
    return {
        "old_correct": old[0],
        "old_total": old[1],
        "old_gain_correct": old[0] - baseline[0],
        "guard_correct": guard[0],
        "guard_total": guard[1],
        "guard_lost_correct": max(0, guard_baseline[0] - guard[0]),
        "gain": float(gain),
        "guard_drop": float(loss),
        "utility_exact": str(gain - loss),
        "checks": checks,
        "recovered": all(checks.values()),
        "published": {
            "accuracy": float(old_accuracy),
            "gain": float_gain,
            "guard_accuracy": float(new_accuracy),
            "utility": float_gain - float_loss,
            "recovered": all(checks.values()),
        },
    }


def select_minimum(actions):
    passing = [budget for budget in BUDGETS if actions[str(budget)]["recovered"]]
    later = [
        budget
        for budget in BUDGETS
        if passing and budget > min(passing) and not actions[str(budget)]["recovered"]
    ]
    return {
        "minimum_budget": str(min(passing)) if passing else "never",
        "qualified_budgets": passing,
        "later_qualification_loss": bool(later),
    }, later


def count_gate(units):
    require(
        len({u["id"] for u in units}) == len(units)
        and all(u["split"] == "train" for u in units),
        "unique TRAIN units",
    )
    classes = [str(b) for b in BUDGETS] + ["never"]
    require(all(u["minimum_budget"] in classes for u in units), "budget class domain")
    counts = Counter(u["minimum_budget"] for u in units)
    by_source = {
        s: dict(Counter(u["minimum_budget"] for u in units if u["stream"] == s))
        for s in sorted({u["stream"] for u in units})
    }
    supported = [label for label in classes if counts[label] >= 2]
    return {
        "counts": dict(counts),
        "counts_by_source": by_source,
        "supported_classes": supported,
        "train_skills": len(units),
        "source_clusters": len(by_source),
        "qualified": len(supported) >= 2
        and len(units) >= 12
        and set(by_source) == set(STREAMS),
        "later_qualification_loss_ids": [
            u["id"] for u in units if u["later_qualification_loss"]
        ],
    }


def raw_counts(records, expected, codes, counters):
    score(records, expected, codes, counters)
    return sum(row["output"]["prediction"] == row["target"] for row in records), len(
        records
    )


def acquisition_audit(run, stream, spec, codes, counters):
    observations = doc(run, "observations.json")
    by_phase = {}
    for phase, observations_at_phase in observations.items():
        by_phase[phase] = {}
        for group in ("old", "new"):
            records = observations_at_phase[f"{group}_gate"]
            score(records, rows_for(stream, stream[group], "gate"), codes, counters)
            by_phase[phase][group] = {
                intent: ratio([r for r in records if r["intent"] == intent])
                for intent in stream[group]
            }
    old_rows, new_rows = {}, {}
    acquired, forgotten, new_acquired = [], [], []
    for intent in stream["old"]:
        initial = by_phase["initial"]["old"][intent]
        before = by_phase["acquired"]["old"][intent]
        after = by_phase["forgotten"]["old"][intent]
        learned = before >= Fraction(4, 5) and before - initial >= Fraction(3, 10)
        eligible = (
            learned and before - after >= Fraction(3, 10) and after <= Fraction(1, 2)
        )
        if learned:
            acquired.append(intent)
        if eligible:
            forgotten.append(intent)
        old_rows[intent] = {
            "initial_correct": int(initial * 10),
            "acquired_correct": int(before * 10),
            "forgotten_correct": int(after * 10),
            "denominator": 10,
            "acquired": learned,
            "eligible": eligible,
        }
    for intent in stream["new"]:
        before, after = (
            by_phase["acquired"]["new"][intent],
            by_phase["forgotten"]["new"][intent],
        )
        learned = after >= Fraction(4, 5) and after - before >= Fraction(3, 10)
        if learned:
            new_acquired.append(intent)
        new_rows[intent] = {
            "phase_start_correct": int(before * 10),
            "final_correct": int(after * 10),
            "denominator": 10,
            "acquired": learned,
        }
    gate = {
        "acquired": sorted(acquired),
        "new_acquired": sorted(new_acquired),
        "eligible": sorted(forgotten),
        "passed": len(acquired) >= 6 and len(new_acquired) >= 6 and len(forgotten) >= 4,
    }
    for key in ("acquired", "new_acquired", "eligible"):
        require(
            gate[key] == run["receipt"][key],
            f"raw acquisition gate {stream['id']}/{key}",
        )
    require(
        gate["passed"] == (run["receipt"]["status"] == "qualified"),
        "acquisition status",
    )
    require(
        run["receipt"]["repair_updates"] == run["receipt"]["test_rows_evaluated"] == 0,
        "eligibility before repairs and TEST utterances",
    )
    states = {}
    for phase, steps in (("initial", 0), ("acquired", 128), ("forgotten", 256)):
        path = run["path"] / "study" / phase / "optimizer.pt"
        optimizer = torch.load(path, map_location="cpu", weights_only=True)
        require(len(optimizer["param_groups"]) == 1, "one AdamW group")
        group = optimizer["param_groups"][0]
        require(
            group["lr"] == spec["optimizer"]["learning_rate"]
            and group["weight_decay"] == 0
            and group["foreach"] is False
            and group["fused"] is False,
            "saved optimizer settings",
        )
        require(
            len(group["params"]) == len(set(group["params"])) == 192,
            "optimizer parameter IDs",
        )
        require(
            set(optimizer["state"]) == (set(group["params"]) if steps else set()),
            "Adam state coverage",
        )
        for state in optimizer["state"].values():
            require(
                set(state) == {"step", "exp_avg", "exp_avg_sq"},
                "Adam state tensor keys",
            )
            require(state["step"].item() == steps, "saved original Adam clock")
            require(
                state["exp_avg"].shape == state["exp_avg_sq"].shape,
                "Adam moment shapes",
            )
            require(
                all(torch.isfinite(v).all().item() for v in state.values()),
                "finite saved Adam tensors",
            )
        states[phase] = {
            "sha256": run["hashes"][f"study/{phase}/optimizer.pt"],
            "states": len(optimizer["state"]),
            "steps": [steps] if steps else [],
            "moment_tensors_loaded_on_cpu": True,
        }
    return gate, {
        "old": old_rows,
        "new": new_rows,
        "original_learning_optimizers": states,
    }


def repair_schedule(stream, intent):
    rows = stream["units"][intent]["repair"] + rows_for(stream, stream["new"], "repair")
    require(
        len(rows) == 72 and len(stream["units"][intent]["repair"]) == 8,
        "8 old + 64 new repair rows",
    )
    rng = random.Random(int(digest([stream["seed"], intent, "repair"])[:15], 16))
    old_queue, new_queue, batches = [], [], []
    for _ in range(16):
        batch = []
        for _ in range(4):
            if not old_queue:
                old_queue = rng.sample(range(8), 8)
            if not new_queue:
                new_queue = rng.sample(range(8, 72), 64)
            batch.extend((old_queue.pop(), new_queue.pop()))
        rng.shuffle(batch)
        batches.append(batch)
    return rows, batches


def audit_adapter(path, expected_config, tensor_cache, expected_schema=None):
    configuration = read_json(path / "adapter_config.json")
    require(configuration == expected_config, f"constant adapter config {path}")
    model_file = path / "adapter_model.safetensors"
    identity = file_hash(model_file)
    if identity not in tensor_cache:
        tensors = load_file(str(model_file))
        schema = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in tensors.items()
        }
        require(len(tensors) == 192, f"192 rank-8 adapter factors {path}")
        for key, value in tensors.items():
            require(
                re.fullmatch(
                    r"base_model\.model\.model\.language_model\.layers\.\d+\.mlp\."
                    r"(?:gate_proj|up_proj|down_proj)\.lora_[AB]\.weight",
                    key,
                ),
                "adapter-only tensor",
            )
            require(
                value.ndim == 2
                and value.dtype == np.float32
                and np.isfinite(value).all(),
                "finite FP32 factors",
            )
            require(
                value.shape[0 if ".lora_A." in key else 1] == 8, "fixed factor rank"
            )
        tensor_cache[identity] = {
            "schema": schema,
            "parameters": sum(v.size for v in tensors.values()),
            "tensor_bytes": sum(v.nbytes for v in tensors.values()),
            "file_bytes": model_file.stat().st_size,
        }
    result = tensor_cache[identity]
    if expected_schema is not None:
        require(result["schema"] == expected_schema, "fixed adapter capacity")
    return identity, result


def prefix_check(proof, adapter_hash, trace, stop, common):
    require(
        proof
        == {
            "adapter_sha256": adapter_hash,
            "updates_sha256": digest(trace[:stop]),
            "optimizer_steps": [stop],
        },
        f"actual adapter/trace/optimizer prefix at {stop}",
    )
    if stop in common:
        require(proof == common[stop], f"shared prefix at {stop}")
    common[stop] = proof


def unit_audit(
    run, source, learning, unit, frozen, anchor, stream, data, spec, counters, tensors
):
    intent, identity = unit["intent"], unit["id"]
    require(
        identity == f"{stream['id']}/{intent}" and unit["split"] == "train",
        "TRAIN unit binding",
    )
    require(not frozen["actions"], "frozen features contain no repair outcomes")
    for key in ("id", "intent", "stream", "split", "baseline", "probes", "features"):
        require(
            unit[key] == frozen[key] == anchor[key],
            f"feature/baseline unchanged {identity}/{key}",
        )
    require(
        unit == doc(run, f"units/{intent}.json"), f"per-skill complete unit {identity}"
    )
    feature_audit(unit, stream, data, spec, counters)
    old_rows, guard_rows = (
        stream["units"][intent]["test"],
        rows_for(stream, stream["new"], "test"),
    )
    baseline_counts = {}
    for phase, checkpoint, control in (
        ("before", "acquired", "restore"),
        ("after", "forgotten", "none"),
    ):
        baseline_counts[phase] = {
            label: raw_counts(
                unit["baseline"][phase][label], rows, data["codes"], counters
            )
            for label, rows in (("test", old_rows), ("guard", guard_rows))
        }
        action = anchor["actions"][control]
        require(
            all(action[k] == unit["baseline"][phase][k] for k in ("test", "guard")),
            "original reset control",
        )
        require(
            action["updates"] == 0
            and action["adapter_sha256"]
            == learning["hashes"][f"study/{checkpoint}/adapter_model.safetensors"],
            "reset checkpoint identity",
        )
    before, after = baseline_counts["before"]["test"], baseline_counts["after"]["test"]
    guard_before = baseline_counts["after"]["guard"]
    rows, schedule = repair_schedule(stream, intent)
    full_trace = doc(run, f"actions/{intent}/16-updates.json")["updates"]
    source_trace = doc(source, f"actions/{intent}/replay_balanced-updates.json")
    require(
        source_trace == {"rows": rows, "updates": full_trace},
        f"original 16 update trace {identity}",
    )
    original_config = doc(learning, "forgotten/adapter_config.json")
    require(
        original_config["r"] == 8
        and original_config["lora_alpha"] == 16
        and original_config["lora_dropout"] == 0,
        "rank8 alpha16 no dropout",
    )
    require(
        set(unit["actions"]) == {str(b) for b in BUDGETS},
        "complete numeric budget grid",
    )
    common, audits, published_actions = {}, {}, {}
    for budget in BUDGETS:
        action = unit["actions"][str(budget)]
        history = doc(run, f"actions/{intent}/{budget}-updates.json")
        require(
            history["rows"] == rows and history["updates"] == full_trace[:budget],
            "shared actual update trace",
        )
        check_updates(history["updates"], rows, budget, 8, f"{identity}/{budget}")
        require(
            [u["rows"] for u in history["updates"]] == schedule[:budget],
            "predeclared sampled row prefix",
        )
        require(
            all(sum(i < 8 for i in u["rows"]) == 4 for u in history["updates"]),
            "4 old + 4 new each update",
        )
        require(
            action["updates"] == budget
            and action["total_exposures"] == 8 * budget
            and action["old_exposures"] == action["new_exposures"] == 4 * budget,
            "exposure accounting",
        )
        require(
            action["fresh_optimizer"] is True and action["reload_exact"] is True,
            "runtime reload/freshness assertions",
        )
        require(
            action["start_adapter_sha256"]
            == learning["hashes"]["study/forgotten/adapter_model.safetensors"],
            "same forgotten start",
        )
        require(
            action["schedule_sha256"]
            == digest({"rows": rows, "batches": schedule[:budget]}),
            "schedule hash",
        )
        require(
            action == doc(run, f"actions/{intent}/{budget}-outcome.json"),
            "per-arm observations equal unit",
        )
        require(
            set(action["prefixes"]) == {str(s) for s in BUDGETS if s <= budget},
            "complete prefix proof",
        )
        for stop in BUDGETS:
            if stop > budget:
                continue
            path = (
                run["path"] / "study/actions" / intent / str(budget) / f"prefix-{stop}"
            )
            adapter_hash, info = audit_adapter(
                path, original_config, tensors, counters.get("adapter_schema")
            )
            counters["adapter_schema"] = info["schema"]
            prefix_check(
                action["prefixes"][str(stop)], adapter_hash, full_trace, stop, common
            )
            counters["prefix_checkpoints"] += 1
        require(
            action["adapter_sha256"] == common[budget]["adapter_sha256"],
            "final prefix hash",
        )
        old = raw_counts(action["test"], old_rows, data["codes"], counters)
        guard = raw_counts(action["guard"], guard_rows, data["codes"], counters)
        audit = repair_counts(old, after, guard, guard_before)
        published_actions[str(budget)] = audit.pop("published")
        audits[str(budget)] = audit
        counters["repair_updates"] += budget
    original = anchor["actions"]["replay_balanced"]
    for key in ("test", "guard", "adapter_sha256", "schedule_sha256", "updates"):
        require(
            unit["actions"]["16"][key] == original[key],
            f"original16 anchor {identity}/{key}",
        )
    require(
        original["adapter_sha256"]
        == source["hashes"][
            f"study/actions/{intent}/replay_balanced/adapter_model.safetensors"
        ],
        "original16 saved adapter bytes",
    )
    minimum, later = select_minimum(audits)
    published = {k: unit[k] for k in ("id", "intent", "stream", "split")}
    published.update(
        before=float(Fraction(*before)),
        after=float(Fraction(*after)),
        actions=published_actions,
        **minimum,
    )
    return published, {
        "id": identity,
        "acquired_old_correct": before[0],
        "forgotten_old_correct": after[0],
        "old_denominator": after[1],
        "forgotten_guard_correct": guard_before[0],
        "guard_denominator": guard_before[1],
        "minimum_budget": minimum["minimum_budget"],
        "qualified_budgets": minimum["qualified_budgets"],
        "later_loss_budgets": later,
        "budgets": audits,
        "common_prefixes": {str(k): v for k, v in common.items()},
        "original16_anchor_exact": True,
    }


def permutation_audit(units, audits, pilot, runs):
    ids, sources = [u["id"] for u in units], [u["stream"] for u in units]
    labels = {"minimum_budget": [u["minimum_budget"] for u in units]}
    labels.update(
        {
            f"{b}/{target}": [u["actions"][str(b)][target] for u in units]
            for b in BUDGETS
            for target in ("recovered", "utility")
        }
    )
    rational = {
        f"{b}/utility": [
            Fraction(u["budgets"][str(b)]["utility_exact"]) for u in audits
        ]
        for b in BUDGETS
    }
    draws, diagnostics, rounding_differences = [], [], []
    for scheme in pilot["label_permutations"]["schemes"]:
        for seed in pilot["label_permutations"]["seeds"]:
            rng = np.random.default_rng(seed)
            permutation = list(range(len(ids)))
            groups = (
                [list(range(len(ids)))]
                if scheme == "pooled_train"
                else [
                    [i for i, source in enumerate(sources) if source == name]
                    for name in sorted(set(sources))
                ]
            )
            for group in groups:
                for destination, origin in zip(
                    group, rng.permutation(group).tolist(), strict=True
                ):
                    permutation[destination] = origin
            require(
                sorted(permutation) == list(range(len(ids))), "permutation bijection"
            )
            if scheme == "within_source":
                require(
                    all(sources[i] == sources[j] for i, j in enumerate(permutation)),
                    "source-stratified permutation",
                )
            draw = {
                "scheme": scheme,
                "seed": seed,
                "indices": permutation,
                "fixed_indices": [i for i, j in enumerate(permutation) if i == j],
            }
            draws.append(draw)
            changed = {
                name: sum(values[i] != values[j] for i, j in enumerate(permutation))
                for name, values in labels.items()
            }
            diagnostics.append(
                draw
                | {
                    "changed_labels": changed,
                    "inert_targets": [
                        key for key, count in changed.items() if count == 0
                    ],
                }
            )
            for name, values in rational.items():
                count = sum(values[i] != values[j] for i, j in enumerate(permutation))
                if count != changed[name]:
                    rounding_differences.append(
                        {
                            "scheme": scheme,
                            "seed": seed,
                            "target": name,
                            "published_float_changed": changed[name],
                            "rational_changed": count,
                        }
                    )
    plan = {"unit_ids": ids, "source_streams": sources, "draws": draws, "redraws": 0}
    require(
        len(draws) == 40 and len({(d["scheme"], d["seed"]) for d in draws}) == 40,
        "all40 declared draws",
    )
    for run in runs:
        require(
            doc(run, "permutation-plan.json") == plan,
            "same predeclared permutations in all three runs",
        )
    recorded = doc(runs[-1], "label-permutations.json")
    require(
        recorded["labels"] == labels and recorded["draws"] == diagnostics,
        "all40 actual changed-label counts",
    )
    require(
        recorded["redraws"] == recorded["predictor_fits"] == 0,
        "no redraw or null predictor fitting",
    )
    summary = {}
    for scheme in pilot["label_permutations"]["schemes"]:
        selected = [d for d in diagnostics if d["scheme"] == scheme]
        summary[scheme] = {
            name: {
                "changed_min": min(d["changed_labels"][name] for d in selected),
                "changed_max": max(d["changed_labels"][name] for d in selected),
                "inert_draws": sum(d["changed_labels"][name] == 0 for d in selected),
            }
            for name in labels
        }
    return {
        "draw_count": 40,
        "source_clusters": 2,
        "redraws": 0,
        "predictor_fits": 0,
        "plan_sha256": runs[-1]["hashes"]["study/permutation-plan.json"],
        "target_summary": summary,
        "draws": diagnostics,
        "utility_rounding_changed_count_differences": rounding_differences,
        "utility_count_rule": "Recompute prescribed float arithmetic; also check exact rational utility ties.",
    }


def code_order_audit(members):
    tree = ast.parse(members["minimum_budget.py"])
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(node, name):
        return sorted(
            [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == name
            ],
            key=lambda n: n.lineno,
        )

    main, measure, arms = (functions[n] for n in ("main", "measure", "run_budget_arms"))
    prerequisites = calls(main, "checked_learning_sources")[0]
    create = calls(main, "ProspectiveModel.fresh")[0]
    plan_write = next(
        n
        for n in calls(main, "write_json")
        if "permutation-plan.json" in ast.unparse(n)
    )
    checked_source = calls(main, "checked_measurement")[0]
    require(
        prerequisites.lineno
        < plan_write.lineno
        < checked_source.lineno
        < create.lineno,
        "prerequisites and permutation plan before model loading",
    )
    failed_gate = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not prerequisites['qualified']"
    )
    require(
        any(isinstance(n, ast.Return) for n in failed_gate.body)
        and failed_gate.lineno < create.lineno,
        "prerequisite failure exits before loading/update",
    )
    feature_write = next(
        n
        for n in calls(measure, "write_json")
        if "features-frozen.json" in ast.unparse(n)
    )
    repair_call = calls(measure, "run_budget_arms")[0]
    require(
        feature_write.lineno < repair_call.lineno,
        "features stored before any budget arm",
    )
    frozen_guard = next(
        n
        for n in ast.walk(measure)
        if isinstance(n, ast.If) and "feature_hash != file_hash" in ast.unparse(n.test)
    )
    require(
        frozen_guard.lineno > repair_call.lineno
        and "parameter_hashes" in ast.unparse(frozen_guard.test),
        "post-repair frozen feature/backbone hash check",
    )
    outer = next(
        n
        for n in arms.body
        if isinstance(n, ast.For) and ast.unparse(n.target) == "budget"
    )
    inner = next(
        n
        for n in outer.body
        if isinstance(n, ast.For) and ast.unparse(n.target) == "stop"
    )
    fresh = calls(outer, "observer.optimizer")
    require(
        len(fresh) == 1
        and fresh[0].lineno < inner.lineno
        and not calls(inner, "observer.optimizer"),
        "fresh per arm and continuous across prefix boundaries",
    )
    update = calls(inner, "observer.update")[0]
    require(
        ast.unparse(update.args[-1]) == "optimizer",
        "same optimizer object passed to every segment",
    )
    clock_guard = next(
        n
        for n in inner.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "observer.optimizer_steps(optimizer) != [stop]"
    )
    checkpoint = calls(inner, "observer.checkpoint")[0]
    require(
        update.lineno < clock_guard.lineno < checkpoint.lineno
        and len(checkpoint.args) == 1,
        "actual optimizer clock checked before adapter-only save",
    )
    require(
        "optimizer.state.values()" in members["prospective_model.py"].decode(),
        "clock reads actual Adam state",
    )
    return {
        "source": "minimum_budget.py",
        "source_sha256": sha(members["minimum_budget.py"]),
        "prerequisites_line": prerequisites.lineno,
        "plan_written_line": plan_write.lineno,
        "feature_file_written_line": feature_write.lineno,
        "first_repair_call_line": repair_call.lineno,
        "frozen_hash_guard_line": frozen_guard.lineno,
        "fresh_optimizer_line": fresh[0].lineno,
        "actual_optimizer_clock_guard_line": clock_guard.lineno,
        "adapter_only_save_line": checkpoint.lineno,
        "basis": "Control flow in exact archived source plus earlier completed feature receipts and matching file bytes; not collected file mtimes.",
    }


def stream_summary(units, raw):
    return {
        "intents": len(units),
        "before_accuracy": mean(u["before"] for u in units),
        "after_accuracy": mean(u["after"] for u in units),
        "actions": {
            str(b): {
                key: mean(u["actions"][str(b)][key] for u in units)
                for key in (
                    "accuracy",
                    "gain",
                    "guard_accuracy",
                    "recovered",
                    "utility",
                )
            }
            for b in BUDGETS
        },
        "mean_readouts": {
            key: [
                mean(u["features"][key][i] for u in raw)
                for i in range(len(raw[0]["features"][key]))
            ]
            for key in FEATURES
        },
    }


def check_subset(actual, expected, name):
    for key, value in expected.items():
        close(actual[key], value, f"{name}/{key}", tolerance=1e-12)


def self_test():
    tests = []
    boundary = repair_counts((16, 20), (10, 20), (144, 160), (160, 160))
    require(boundary["recovered"], "all rational boundaries pass exactly")
    tests.append("accuracy .8 / gain .3 / guard drop .1 equality")
    for old, baseline, guard, failed in (
        ((15, 20), (9, 20), (144, 160), "old_accuracy"),
        ((16, 20), (11, 20), (144, 160), "old_gain"),
        ((16, 20), (10, 20), (143, 160), "new_guard"),
    ):
        result = repair_counts(old, baseline, guard, (160, 160))
        require(
            not result["recovered"] and not result["checks"][failed],
            f"one-row threshold failure {failed}",
        )
        tests.append(f"one row beyond {failed}")
    improvement = repair_counts((20, 20), (10, 20), (160, 160), (144, 160))
    require(
        improvement["guard_drop"] == 0 and improvement["published"]["utility"] == 0.5,
        "guard improvement clipped",
    )
    tests.append("guard improvements give no extra utility")
    actions = {str(b): {"recovered": b in (2, 16)} for b in BUDGETS}
    result, later = select_minimum(actions)
    require(
        result["minimum_budget"] == "2" and later == [4, 8],
        "nonmonotonic later loss retained",
    )
    tests.append("nonmonotonic budgets and numeric minimum")
    result, later = select_minimum({str(b): {"recovered": False} for b in BUDGETS})
    require(
        result["minimum_budget"] == "never" and not later, "never is not later loss"
    )
    tests.append("never class")
    units = [
        {
            "id": str(i),
            "stream": STREAMS[i % 2],
            "split": "train",
            "minimum_budget": label,
            "later_qualification_loss": False,
        }
        for i, label in enumerate(["2"] * 12 + ["8", "16"])
    ]
    require(not count_gate(units)["qualified"], "singleton classes do not qualify")
    units[-2]["minimum_budget"] = "16"
    require(count_gate(units)["qualified"], "two supported classes qualify")
    tests.append("class support including singleton rejection")
    tests.append("two classes with two skills each")
    trace = [{"step": 1}, {"step": 2}]
    proof = {
        "adapter_sha256": "a",
        "updates_sha256": digest(trace),
        "optimizer_steps": [2],
    }
    for key, value in (
        ("adapter_sha256", "b"),
        ("updates_sha256", "changed"),
        ("optimizer_steps", [1]),
    ):
        rejected = False
        try:
            prefix_check(proof | {key: value}, "a", trace, 2, {})
        except ValueError:
            rejected = True
        require(rejected, f"prefix mutation rejected {key}")
        tests.append(f"reject modified prefix {key}")
    return {"passed": len(tests), "cases": tests}


def audit(root):
    archives, counters, tensors = (
        {},
        {
            "records_scored": 0,
            "max_feature_error": 0.0,
            "prefix_checkpoints": 0,
            "repair_updates": 0,
        },
        {},
    )
    pilot_archive = read_archive(root, PILOT_CODE, archives)
    members = pilot_archive["members"]
    pilot = json.loads(members["minimum-budget-protocol.json"])
    require(
        sha(members["minimum-budget-protocol.json"])
        == members["minimum-budget-protocol.sha256"].decode().strip(),
        "sealed budget protocol hash",
    )
    require(
        pilot["budgets"] == list(BUDGETS) and pilot["train_streams"] == list(STREAMS),
        "declared budget/stream grid",
    )
    require(
        pilot["recovered"]
        == {"accuracy_min": 0.8, "gain_min": 0.3, "guard_drop_max": 0.1},
        "fixed repair thresholds",
    )
    require(
        [
            pilot["gate"][k]
            for k in (
                "minimum_budget_classes",
                "minimum_skills_per_class",
                "minimum_train_skills",
            )
        ]
        == [2, 2, 12],
        "declared gate requirements",
    )
    config, spec, data = (
        json.loads(members[p])
        for p in (
            "configs/prospective.json",
            "prospective-protocol.json",
            "inputs/prospective-cohort.json",
        )
    )
    require(
        sha(members["configs/prospective.json"]) == pilot["source_config_sha256"]
        and sha(members["prospective-protocol.json"]) == pilot["source_protocol_sha256"]
        and digest(data) == pilot["source_dataset_sha256"],
        "source configuration/protocol/data pins",
    )
    require(
        spec["gates"] == pilot["learning_gates"]
        and spec["recovered"] == pilot["recovered"],
        "unchanged scoring",
    )
    require(
        [
            spec["gates"][k]
            for k in (
                "acquired_min",
                "acquisition_gain_min",
                "after_max",
                "drop_min",
                "minimum_acquired_per_stream",
                "minimum_new_acquired_per_stream",
                "minimum_forgotten_per_stream",
            )
        ]
        == [0.8, 0.3, 0.5, 0.3, 6, 6, 4],
        "acquisition thresholds",
    )
    mechanism = pilot["learning_implementation"] | {
        "tuned_lens.py": pilot["tuned_lens_implementation_sha256"],
        "minimum_budget.py": sha(members["minimum_budget.py"]),
    }
    require(
        all(sha(members[name]) == expected for name, expected in mechanism.items()),
        "original mechanism pins",
    )
    identity = digest(
        {
            "config": config,
            "protocol": spec,
            "dataset": digest(data),
            "pilot": pilot,
            "implementation": mechanism,
        }
    )
    dataset_report = data_audit(data, spec)
    order_report = code_order_audit(members)
    runs, prerequisite, eligibility = {}, {}, {}
    learning_identity = digest(
        {
            "settings": {
                k: config[k]
                for k in (
                    "model_id",
                    "revision",
                    "acquisition_updates",
                    "forgetting_updates",
                )
            },
            "protocol": spec,
            "dataset": digest(data),
            "implementation": pilot["learning_implementation"],
        }
    )
    measurement_identity = digest({"learning": learning_identity, "repair_updates": 16})
    for stream_name in STREAMS:
        for stage in ("learn", "measure"):
            key = f"{stage}-{stream_name}"
            runs[key] = verify_run(root, key, archives)
            run = runs[key]
            require(
                run["hashes"]["study/receipt.json"]
                == pilot["source_receipts"][stream_name][stage],
                "pinned original receipt",
            )
            require(
                run["seal"]["learning_identity"] == learning_identity
                and run["seal"]["measurement_identity"] == measurement_identity,
                "original study identity",
            )
            require(
                run["seal"]["learning_implementation"]
                == pilot["learning_implementation"],
                "original learning code",
            )
        learning = runs[f"learn-{stream_name}"]
        gate, detail = acquisition_audit(
            learning, data["streams"][stream_name], spec, data["codes"], counters
        )
        require(
            gate["passed"], "acquisition and forgetting eligibility before repair audit"
        )
        prerequisite[stream_name] = gate | {
            "source": f"/mnt/shared/cl-portfolio/runs/{learning['path'].name}/study",
            "receipt_sha256": learning["hashes"]["study/receipt.json"],
        }
        eligibility[stream_name] = detail
    require(
        sum(len(g["eligible"]) for g in prerequisite.values()) >= 12,
        "TRAIN prerequisite sample count",
    )
    prerequisites = {
        "qualified": True,
        "streams": prerequisite,
        "eligible_train_skills": sum(len(g["eligible"]) for g in prerequisite.values()),
        "source_clusters": 2,
        "new_repair_updates": 0,
        "heldout_outcomes_read": 0,
    }
    outcomes, units_report, timing, stream_reports = [], [], {}, {}
    for stream_name in STREAMS:
        suffix = f"budget-pilot-{stream_name}"
        run = runs[suffix] = verify_run(root, suffix, archives)
        learning, source = runs[f"learn-{stream_name}"], runs[f"measure-{stream_name}"]
        require(
            run["seal"]["minimum_budget_identity"] == identity
            and run["execution"]["source_sha256"] == PILOT_CODE,
            "pilot mechanism identity",
        )
        require(
            run["seal"]["dispatch_manifest"] == run["execution"]["task"]["config"],
            "exact pilot config",
        )
        require(
            doc(run, "prerequisites.json") == prerequisites,
            "independent pilot prerequisites",
        )
        require(
            doc(run, "measurement-source.json")
            == {
                "path": f"/mnt/shared/cl-portfolio/runs/{source['path'].name}/study",
                "receipt_sha256": source["hashes"]["study/receipt.json"],
            },
            "original measurement provenance",
        )
        units, frozen, anchors = (
            doc(run, "units.json"),
            doc(run, "features-frozen.json"),
            doc(run, "source-outcomes-frozen.json"),
        )
        require(
            anchors == doc(source, "units.json"),
            "all original5 action records retained exactly",
        )
        require(
            frozen == doc(source, "features-frozen.json"),
            "original pre-repair feature records reused unchanged",
        )
        feature_hash = run["hashes"]["study/features-frozen.json"]
        require(
            feature_hash
            == source["hashes"]["study/features-frozen.json"]
            == run["receipt"]["features_sha256"]
            == source["receipt"]["features_sha256"],
            "byte-exact original frozen features",
        )
        require(
            doc(run, "frozen-backbone.json")
            == doc(learning, "frozen-backbone.json")
            == doc(source, "frozen-backbone.json"),
            "original and pilot frozen parameter hash manifests",
        )
        require(
            len(doc(run, "frozen-backbone.json")) == 723,
            "all723 frozen backbone tensors",
        )
        require(
            [u["intent"] for u in units] == prerequisite[stream_name]["eligible"],
            "all and only eligible TRAIN skills",
        )
        started = datetime.fromisoformat(run["execution"]["started_at"])
        finished = datetime.fromisoformat(source["execution"]["finished_at"])
        require(
            finished < started,
            "source frozen feature receipt completed before pilot started",
        )
        timing[stream_name] = {
            "original_measurement_finished": source["execution"]["finished_at"],
            "pilot_started": run["execution"]["started_at"],
            "feature_file_sha256": feature_hash,
            "same_feature_bytes_predate_pilot": True,
        }
        recomputed = []
        for unit, prior, anchor in zip(units, frozen, anchors, strict=True):
            computed, detail = unit_audit(
                run,
                source,
                learning,
                unit,
                prior,
                anchor,
                data["streams"][stream_name],
                data,
                spec,
                counters,
                tensors,
            )
            recomputed.append(computed)
            units_report.append(detail)
        close(
            recomputed,
            doc(run, "outcomes.json"),
            f"independent outcomes {stream_name}",
            tolerance=1e-12,
        )
        summary = stream_summary(recomputed, units)
        check_subset(
            run["receipt"]["stream_summary"],
            summary,
            "independent pilot stream summary",
        )
        stream_reports[stream_name] = summary
        check_subset(
            run["receipt"],
            {
                "status": "completed",
                "stream": stream_name,
                "split": "train",
                "repair_updates": len(units) * sum(BUDGETS),
                "lens_updates": 0,
                "predictor_fits": 0,
                "test_authorized": False,
                "actual_prefixes_equal": True,
                "anchor_sixteen_reproduced": True,
                "unit_ids": [u["id"] for u in recomputed],
            },
            "pilot receipt",
        )
        require(
            not list((run["path"] / "study").rglob("*.pt")),
            "pilot saves no optimizer tensors",
        )
        outcomes.extend(recomputed)
    gate = runs["budget-pilot-gate-slot"] = verify_run(
        root, "budget-pilot-gate-slot", archives
    )
    require(
        gate["seal"]["minimum_budget_identity"] == identity
        and gate["execution"]["source_sha256"] == PILOT_CODE,
        "gate exact study identity",
    )
    require(doc(gate, "prerequisites.json") == prerequisites, "gate prerequisites")
    computed_gate = count_gate(outcomes)
    require(computed_gate["qualified"], "two sufficiently supported budget classes")
    saved_gate = doc(gate, "budget-qualification.json")
    expected_gate = computed_gate | {
        "sources": {
            s: runs[f"budget-pilot-{s}"]["hashes"]["study/receipt.json"]
            for s in STREAMS
        },
        "status": "informative_train_budget_labels",
        "stage": "minimum-budget-qualify",
        "units": outcomes,
        "lens_updates": 0,
        "repair_updates": 0,
        "predictor_fits": 0,
        "test_authorized": False,
    }
    for result in (saved_gate, gate["receipt"]):
        check_subset(result, expected_gate, "independent final gate")
        for stream_name in STREAMS:
            check_subset(
                result["run_level"][stream_name],
                stream_reports[stream_name],
                "gate stream arithmetic",
            )
    permutation_report = permutation_audit(
        outcomes,
        units_report,
        pilot,
        [runs[f"budget-pilot-{s}"] for s in STREAMS] + [gate],
    )
    require(
        counters["prefix_checkpoints"] == 140
        and counters["repair_updates"] == 420
        and len(tensors) == 56,
        "total actual prefix/arm/update counts",
    )
    tensor_info = next(iter(tensors.values()))
    counters.pop("adapter_schema")
    for run in runs.values():
        require(
            file_hash(run["path"] / "execution.json") == run["hashes"]["execution.json"]
            and file_hash(run["path"] / "study/receipt.json")
            == run["hashes"]["study/receipt.json"],
            "inputs unchanged during audit",
        )
    return {
        "status": "passed",
        "gate": computed_gate,
        "budget_outcomes": units_report,
        "acquisition_eligibility": eligibility,
        "dataset": dataset_report,
        "counts": counters
        | {
            "repair_arms": len(outcomes) * len(BUDGETS),
            "total_exposures": counters["repair_updates"] * 8,
            "old_exposures": counters["repair_updates"] * 4,
            "new_exposures": counters["repair_updates"] * 4,
        },
        "prefix_evidence": {
            "all_actual_prefix_weights_and_trace_hashes_equal": True,
            "all14_original16_weight_schedule_trace_and_raw_observation_anchors_exact": True,
            "unique_adapter_contents_loaded_and_finite": len(tensors),
            "parameters_per_adapter": tensor_info["parameters"],
            "tensors_per_adapter": len(tensor_info["schema"]),
            "tensor_bytes_per_adapter": tensor_info["tensor_bytes"],
            "file_bytes_per_adapter": tensor_info["file_bytes"],
            "frozen_backbone_manifest_tensors": 723,
            "pilot_optimizer_state_files": 0,
            "original_learning_optimizer_state_files_loaded": 6,
            "optimizer_proof_boundary": "Pilot fresh/continuous Adam clocks are saved JSON assertions enforced by archived code and exact trace/weight reproduction. Pilot Adam moments were not saved and cannot be independently reloaded. Original acquisition optimizer states were saved and inspected on CPU.",
            "reload_proof_boundary": "Runtime code requires exact observations after adapter reload and baseline reset. This audit verifies those receipts and saved bytes; it does not rerun model inference.",
        },
        "features_frozen_before_pilot": timing,
        "source_order": order_report,
        "permutations": permutation_report,
        "provenance": {
            "runs": {k: v["provenance"] for k, v in runs.items()},
            "archives": {
                k: {n: v[n] for n in ("path", "sha256", "closure")}
                for k, v in archives.items()
            },
            "minimum_budget_identity": identity,
            "source_protocol_sha256": pilot["source_protocol_sha256"],
            "minimum_budget_protocol_sha256": sha(
                members["minimum-budget-protocol.json"]
            ),
            "source_dataset_canonical_sha256": digest(data),
            "implementation": mechanism,
            "helper_source": str(root / "recovery-audit.py"),
            "helper_source_sha256": HELPER_SOURCE_SHA256,
            "helpers_adapted_at_authoring": HELPERS_ADAPTED,
        },
        "scope": {
            "scoring": "Argmax among the same16 native answer codes throughout learning, forgetting and repair; not full-vocabulary generation.",
            "threshold_arithmetic": "Integer counts and exact rational comparisons, independent of experiment gate functions.",
            "unit_boundary": "14 intent-to-code bindings in2 independently trained TRAIN streams; no semantic-erasure or general skill-recovery claim.",
            "prospective_validity": "Supports preparing a separately sealed predictor on new disjoint cohorts. No predictor was fit here; fresh TEST outcomes, predictor accuracy, policy gain and recovery-versus-relearning speed remain untested by this pilot.",
            "outcome_rows": "20 old and160 new-guard CLINC TEST utterances per action, all within TRAIN predictor cohorts. No predictor TEST outcomes opened in this audit.",
            "test_authorized": False,
            "gpu_inference_or_dispatch": False,
            "backbone_boundary": "723 saved frozen tensor hashes agree and archived code enforces the before/after check; base model weights were not reloaded locally.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests, indent=2))
        return
    require(
        not args.output.exists(), "audit output must be new; immutable prior reports"
    )
    torch.set_num_threads(2)
    result = audit(Path(__file__).resolve().parent)
    result.update(
        self_tests=tests,
        audited_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=file_hash(Path(__file__)),
        runtime={"numpy": np.__version__, "torch": torch.__version__, "device": "cpu"},
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "counts": result["counts"],
                "report": str(args.output.resolve()),
                "sha256": file_hash(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
