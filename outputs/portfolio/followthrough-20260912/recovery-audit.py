import argparse
import ast
import hashlib
import json
import math
import subprocess
import tarfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

PREFIX = "followthrough-20260912-recovery-"
REMOTE_ROOT = "/mnt/shared/cl-portfolio/runs"
CORE = (
    "prospective.py", "prospective_data.py", "prospective_model.py",
    "prospective_analysis.py", "model.py", "protocol.py",
)
ROLES = {"learn": ("train", 32), "repair": ("train", 8),
         "gate": ("validation", 10), "probe": ("validation", 10), "test": ("test", 20)}
ACTIONS = ("none", "replay_target", "replay_balanced", "restore", "sham")
TRAINABLE = ("replay_target", "replay_balanced")
FEATURES = ("confidence", "output", "frozen", "tuned", "permuted")


def require(condition, detail):
    if not condition:
        raise ValueError(f"RECOVERY_INDEPENDENT_AUDIT: {detail}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def archive_manifest(path, expected):
    closure, members = hashlib.sha256(), {}
    with tarfile.open(path) as archive:
        for item in sorted(archive.getmembers(), key=lambda item: Path(item.name)):
            require(item.isfile() and not item.name.startswith("/") and ".." not in Path(item.name).parts,
                    f"unsafe archive member {item.name}")
            raw = archive.extractfile(item).read()
            closure.update(item.name.encode() + b"\0" + raw)
            members[item.name] = sha(raw)
    require(closure.hexdigest() == expected, f"source closure {path}")
    return {"closure": expected, "archive_sha256": file_hash(path), "members": members}


def snapshot(directory):
    if not (directory / "execution.json").exists():
        return {"available": False}
    execution = json.loads((directory / "execution.json").read_text())
    if execution["status"] != "completed":
        return {"available": False, "execution": execution}
    hashes, texts = {}, {}
    paths = [directory / "execution.json", directory / "run.log", *sorted((directory / "study").rglob("*"))]
    for path in paths:
        if path.is_file():
            name = str(path.relative_to(directory))
            if path.suffix == ".json" or path.name == "run.log":
                raw = path.read_bytes()
                hashes[name], texts[name] = sha(raw), raw.decode()
            else:
                hashes[name] = file_hash(path)
    require(hashes["execution.json"] == file_hash(directory / "execution.json"), "execution changed during read")
    require(hashes["study/receipt.json"] == file_hash(directory / "study/receipt.json"), "receipt changed during read")
    return {"available": True, "hashes": hashes, "texts": texts}


def doc(bundle, name):
    return json.loads(bundle["texts"]["study/" + name])


def receipt(bundle):
    require(bundle["available"], "required completed run unavailable")
    for name, raw in bundle["texts"].items():
        require(sha(raw.encode()) == bundle["hashes"][name], f"snapshot transport {name}")
    result, seal = doc(bundle, "receipt.json"), doc(bundle, "seal.json")
    for value in (result, seal):
        require(digest(value["payload"]) == value["sha256"], "receipt/seal digest")
    actual = {key.removeprefix("study/"): value for key, value in bundle["hashes"].items()
              if key.startswith("study/") and key != "study/receipt.json"}
    require(actual == result["payload"]["files"], "complete receipt file manifest")
    return result["payload"], seal["payload"]


def live_snapshots(control, context, suffixes):
    own_source = Path(__file__).read_text()
    remote_names = {"require", "sha", "file_hash", "archive_manifest", "snapshot"}
    declarations = "import hashlib\nimport json\nimport tarfile\nfrom pathlib import Path\n"
    declarations += f"PREFIX = {PREFIX!r}\n"
    declarations += "\n\n".join(ast.get_source_segment(own_source, node) for node in ast.parse(own_source).body
                                if isinstance(node, ast.FunctionDef) and node.name in remote_names) + "\n"
    read_only = declarations + '''
root = Path("/mnt/shared/cl-portfolio/runs")
bundles = {suffix: snapshot(root / (PREFIX + suffix)) for suffix in SUFFIXES}
archives = {}
for bundle in bundles.values():
    if bundle["available"]:
        code = json.loads(bundle["texts"]["execution.json"])["source_sha256"]
        if code not in archives:
            archives[code] = archive_manifest(root.parent / "code" / (code + ".tar"), code)
print(json.dumps({"runs": bundles, "archives": archives}))
'''
    read_only = read_only.replace("for suffix in SUFFIXES", f"for suffix in {suffixes!r}")
    result = subprocess.run(
        ["kubectl", "--context", context, "exec", control, "--", "uv", "run", "--no-project",
         "python", "-c", read_only], check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def binding(row):
    return row["text_sha256"], row["intent"], row["target"]


def score(records, expected, codes, counters):
    require(Counter(map(binding, records)) == Counter(map(binding, expected)), "observation row/target binding")
    correct = Counter()
    counts = Counter()
    for row in records:
        output, target = row["output"], row["target"]
        logits = output["code_logits"]
        require(len(logits) == 16 and all(math.isfinite(x) for x in logits), "finite 16-code logits")
        prediction = codes[max(range(16), key=logits.__getitem__)]
        require(prediction == output["prediction"], "recorded prediction differs from raw argmax")
        index, maximum = codes.index(target), max(logits)
        logz = math.log(sum(math.exp(value - maximum) for value in logits))
        logp = [value - maximum - logz for value in logits]
        probability = [math.exp(value) for value in logp]
        ranked = sorted(probability, reverse=True)
        metrics = {
            "top_probability": ranked[0], "top_margin": ranked[0] - ranked[1],
            "entropy": -sum(p * lp for p, lp in zip(probability, logp, strict=True)),
            "target_logp": logp[index],
            "target_margin": logits[index] - max(value for i, value in enumerate(logits) if i != index),
        }
        for name, value in metrics.items():
            error = abs(value - output[name])
            counters["max_feature_error"] = max(counters["max_feature_error"], error)
            require(math.isclose(value, output[name], abs_tol=1e-5, rel_tol=1e-5), f"raw feature {name}")
        counters["records_scored"] += 1
        counts[row["intent"]] += 1
        correct[row["intent"]] += prediction == target
    return {intent: correct[intent] / count for intent, count in sorted(counts.items())}


def learned(initial, final, gates):
    return sorted(intent for intent in final if final[intent] >= gates["acquired_min"]
                  and final[intent] - initial[intent] >= gates["acquisition_gain_min"] - 1e-12)


def outcome(old, guard, baseline, guard_baseline, rule):
    gain, loss = old - baseline, max(0.0, guard_baseline - guard)
    return {"accuracy": old, "guard_accuracy": guard, "gain": gain, "guard_loss": loss,
            "utility": gain - loss,
            "recovered": old >= rule["accuracy_min"] and gain >= rule["gain_min"] - 1e-12
            and loss <= rule["guard_drop_max"] + 1e-12}


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
        require(all(stream[key] == declared[name][key] for key in ("id", "split", "seed")), "seed/split identity")
        intents = stream["old"] + stream["new"]
        require(len(stream["old"]) == len(stream["new"]) == 8 and len(set(intents)) == 16, "old/new intent counts")
        require(not seen["intents"].intersection(intents), "predictor source intent overlap")
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
    require(len(backgrounds) == 16 and not backgrounds.intersection(seen["intents"]), "lens background leakage")
    for role, count in (("lens_fit", 32), ("lens_check", 8)):
        require(Counter(row["intent"] for row in data[role]) == {intent: count for intent in backgrounds}, "lens split counts")
        groups.extend(([row], "train", row["intent"], None) for row in data[role])
    for rows, split, intent, code in groups:
        for row in rows:
            text = " ".join(unicodedata.normalize("NFKC", row["text"]).casefold().split())
            require(row["text_sha256"] == sha(text.encode()), "normalized text identity")
            require(row["intent"] == intent and row["source_split"] == split and row["code"] == code
                    and row["target"] == (None if code is None else data["codes"][code]), "dataset row binding")
            require(0 < len(row["input_ids"]) <= spec["runtime"]["max_length"], "input length")
            keys = {"text": row["text_sha256"], "tokens": tuple(row["input_ids"]),
                    "source": (row["source_split"], row["source_index"])}
            for key, value in keys.items():
                require(value not in seen[key], f"global {key} leakage")
                seen[key].add(value)
            row_count += 1
    return {"rows": row_count, "stream_intents": len(seen["intents"]), "lens_intents": len(backgrounds),
            "global_text_token_source_overlaps": 0, "predictor_train_test_intent_overlap": 0,
            "lens_training_cohort_overlap": 0, "streams": declared}


def check_updates(trace, rows, steps, batch_size, label):
    require(len(trace) == steps and [item["step"] for item in trace] == list(range(1, steps + 1)), f"{label} steps")
    for item in trace:
        require(len(item["rows"]) == batch_size and all(0 <= i < len(rows) for i in item["rows"]), f"{label} indices")
        require(math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"]), f"{label} finite updates")
    return [index for item in trace for index in item["rows"]]


def source_order_audit(root):
    tree = ast.parse((root / "prospective.py").read_text())
    learning = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_learning")
    guard = next(node for node in learning.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "qualification['acquisition_passed']")
    gated_updates = [node.lineno for node in ast.walk(guard) if isinstance(node, ast.Call)
                     and ast.unparse(node.func) == "observer.update"]
    require(len(gated_updates) == 2, "both forgetting halves inside acquisition guard")
    pilot = ast.parse((root / "prospective_pilot.py").read_text())
    entry = next(node for node in pilot.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = {name: min(node.lineno for node in ast.walk(entry) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == name)
             for name in ("checked_qualification", "ProspectiveModel.fresh", "measure_pilot")}
    require(calls["checked_qualification"] < calls["ProspectiveModel.fresh"] < calls["measure_pilot"], "pilot gate call order")
    return {"forgetting_guard_line": guard.lineno, "forgetting_update_lines": gated_updates,
            "pilot_entrypoint_lines": calls, "basis": "Reviewed control flow in hash-bound source, plus runtime chronology below."}


def self_test():
    gates = {"acquired_min": 0.8, "acquisition_gain_min": 0.3}
    require(learned({"biased": 1.0, "new": 0.5}, {"biased": 1.0, "new": 0.8}, gates) == ["new"], "gain control")
    rule = {"accuracy_min": 0.8, "gain_min": 0.3, "guard_drop_max": 0.1}
    require(math.isclose(outcome(1, 0, .45, .975, rule)["utility"], -.425), "guard-loss control")
    require(outcome(1, 1, .45, .975, rule)["utility"] == .55, "guard improvement is not extra utility")
    require(not outcome(1, 0, 0, 1, rule)["recovered"], "target-only false-positive control")
    raw = [{"id": f"{stream}/{i}", "stream": stream, "split": "test",
            "features": {feature: [x] for feature in FEATURES if feature != "permuted"}}
           for i, (stream, x) in enumerate((("test-a", -1.0), ("test-b", 0.0), ("test-b", 1.0)))]
    actual = [{"id": raw[i]["id"], "actions": {"none": {"utility": 0.0, "recovered": False},
                           "replay_target": {"utility": target, "recovered": i == 2},
                           "replay_balanced": {"utility": balanced, "recovered": i != 1}}}
              for i, (target, balanced) in enumerate(((-.5, .75), (.2, .4), (.8, .1)))]
    fitted = {"train_units": ["train-a/example"], "models": {}}
    for action in TRAINABLE:
        fitted["models"][action] = {}
        for label in ("recovered", "utility"):
            intercept = .5 if label == "recovered" else .25 if action == "replay_balanced" else 0.0
            weight = .5 if label == "recovered" else 0.0 if action == "replay_balanced" else 1.0
            fitted["models"][action][label] = {
                "prevalence": .25, "estimable_binary": True,
                "models": {feature: {"center": [0.0], "scale": [1.0], "intercept": intercept, "weights": [weight]}
                           for feature in FEATURES},
            }
    checked = heldout_audit(raw, actual, fitted, None)
    close(checked["policies"]["tuned"]["choices"], ["replay_balanced", "replay_balanced", "replay_target"], "known policy control")
    close(checked["policies"]["tuned"]["mean_utility"], .65, "known pooled utility control")
    close(checked["policies"]["tuned"]["source_balanced_utility"], .675, "unequal cluster weights control")
    close(checked["policies"]["tuned"]["mean_oracle_regret"], 0.0, "known oracle control")
    close(checked["recovery_prediction"]["replay_target"]["brier"]["tuned"], 1 / 12, "known Brier control")
    regression = checked["utility_prediction_audit_diagnostic"]["replay_balanced"]["tuned"]
    close(regression["mse"], .295 / 3, "continuous forecast pooled MSE control")
    close(regression["mae"], .8 / 3, "continuous forecast pooled MAE control")
    close(regression["mse_by_source"], {"test-a": .25, "test-b": .0225}, "continuous source MSE control")
    close(regression["mae_by_source"], {"test-a": .5, "test-b": .15}, "continuous source MAE control")
    close(regression["equal_source_mse"], .13625, "continuous equal-source MSE control")
    close(regression["equal_source_mae"], .325, "continuous equal-source MAE control")
    changed = [{"id": row["id"], "actions": {key: {"utility": -2.0, "recovered": not values["recovered"]}
                            for key, values in row["actions"].items()}} for row in actual]
    other = heldout_audit(raw, changed, fitted, None)
    require(other["policies"]["tuned"]["choices"] == checked["policies"]["tuned"]["choices"]
            and other["recovery_prediction"]["replay_target"]["predictions"] == checked["recovery_prediction"]["replay_target"]["predictions"], "heldout labels cannot alter predictions/actions")
    old = {"after": .4, "guard_after": .95, "actions": {"balanced": outcome(.9, .9, .4, .95, rule)}}
    fresh = {"after": .1, "guard_after": 1.0, "actions": {"balanced": outcome(.8, 1, .1, 1, rule)}}
    excess = paired_metrics(old, fresh, "balanced")
    close(excess["recovery_minus_fresh_accuracy"], .1, "reference accuracy-excess control")
    close(excess["recovery_minus_fresh_gain"], -.2, "reference baseline-adjusted gain control")
    close(excess["recovery_minus_fresh_utility"], -.25, "reference guard-adjusted utility control")
    close(excess["recovery_minus_fresh_guard_accuracy"], -.1, "reference paired guard excess control")
    close(excess["recovery_minus_fresh_guard_baseline"], -.05, "reference paired guard baseline control")
    close(excess["fresh_minus_recovery_guard_drop"], -.05, "reference guard drop advantage control")
    close(excess["recovery_minus_fresh_gain"] + excess["fresh_minus_recovery_guard_drop"],
          excess["recovery_minus_fresh_utility"], "utility excess decomposes into old gain and guard retention")
    return {"passed": 23}


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
        require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), f"{label}: {actual} != {expected}")
    else:
        require(actual == expected, label)


def audit_action_unit(bundle, unit, stream, data, spec, counters, checkpoint_hashes, reference=False):
    intent = unit["intent"]
    require(unit["id"] == f"{stream['id']}/{intent}" and unit["stream"] == stream["id"]
            and unit["split"] == stream["split"], "action unit source grouping")
    scored = {}
    for name, panel in {**unit["baseline"], **unit["actions"]}.items():
        target = mean(score(panel["test"], stream["units"][intent]["test"], data["codes"], counters).values())
        guard = mean(score(panel["guard"], rows_for(stream, stream["new"], "test"), data["codes"], counters).values())
        scored[name] = (target, guard)
    actions = tuple(action for action in ACTIONS if action != "restore" or not reference)
    require(set(unit["actions"]) == set(actions), "complete action set")
    metrics, histories = {}, {}
    for action in actions:
        raw = unit["actions"][action]
        history = doc(bundle, f"actions/{intent}/{action}-updates.json")
        histories[action] = history
        steps = 16 if action in (*TRAINABLE, "sham") else 0
        target_rows = stream["units"][intent]["repair"]
        support = target_rows if action == "replay_target" else target_rows + rows_for(stream, stream["new"], "repair")
        require(history["rows"] == (support if steps else []), "disjoint TRAIN repair support")
        used = check_updates(history["updates"], history["rows"], steps, 8, f"{unit['id']}/{action}")
        old_exposures = sum(history["rows"][i]["intent"] == intent for i in used)
        require((raw["updates"], raw["total_exposures"], raw["old_exposures"], raw["new_exposures"])
                == (steps, len(used), old_exposures, len(used) - old_exposures), "exposure accounting")
        require(raw["schedule_sha256"] == digest({"rows": history["rows"], "batches": [item["rows"] for item in history["updates"]]}), "schedule digest")
        require(raw["reload_exact"] and raw["adapter_sha256"] == bundle["hashes"][f"study/actions/{intent}/{action}/adapter_model.safetensors"], "action checkpoint")
        if action in ("none", "restore"):
            baseline = unit["baseline"]["after" if action == "none" else "before"]
            require(all(raw[key] == baseline[key] for key in ("test", "guard")), "exact control observations")
            require(raw["adapter_sha256"] == checkpoint_hashes[action], "exact control weights")
        metrics[action] = outcome(*scored[action], *scored["after"], spec["recovered"])
    require(histories["sham"]["rows"] == histories["replay_balanced"]["rows"] and
            [row["rows"] for row in histories["sham"]["updates"]] == [row["rows"] for row in histories["replay_balanced"]["updates"]], "sham matched schedule")
    require(unit["actions"]["replay_balanced"]["old_exposures"] == 64
            and unit["actions"]["replay_target"]["old_exposures"] == 128, "old exposure boundary")
    return {"id": unit["id"], "stream": stream["id"], "before": scored.get("before", (None,))[0],
            "after": scored["after"][0], "guard_after": scored["after"][1], "target_test_rows": 20,
            "guard_test_rows": 160, "actions": metrics}


def action_means(units):
    return {action: {key: mean(unit["actions"][action][key] for unit in units)
                     for key in units[0]["actions"][action]} for action in units[0]["actions"]}


def check_feasibility(units, observed, spec):
    rule = spec["repair_feasibility"]
    responsive = [unit["id"] for unit in units if max(unit["actions"][action]["gain"] for action in TRAINABLE) >= rule["response_gain_min"] - 1e-12]
    best = [max(unit["actions"][action]["utility"] for action in TRAINABLE) for unit in units]
    paired = [unit["actions"]["replay_target"]["utility"] - unit["actions"]["replay_balanced"]["utility"] for unit in units]
    variation = max(best) - min(best) >= rule["utility_range_min"] - 1e-12 or max(paired) - min(paired) >= rule["paired_action_range_min"] - 1e-12
    recomputed = {"qualified": len(responsive) >= rule["minimum_responsive_train_skills"] and variation,
                  "responsive_train_units": responsive, "utility_range": max(best) - min(best),
                  "paired_action_range": max(paired) - min(paired),
                  "single_class_binary_actions": [action for action in TRAINABLE if len({unit["actions"][action]["recovered"] for unit in units}) == 1]}
    close({key: observed[key] for key in recomputed}, recomputed, "full TRAIN feasibility")
    return recomputed


def audit_lens(bundle, data, spec):
    trained, late = doc(bundle, "lens-training.json"), doc(bundle, "lens-late-check.json")
    for key, rows in (("fit", data["lens_fit"]), ("check", data["lens_check"])):
        metadata = trained[key]
        positions = []
        for row in rows:
            length = len(row["input_ids"])
            count = min(length, spec["lens"]["positions_per_prompt"])
            positions.extend({"text_sha256": row["text_sha256"], "position": round(i * (length - 1) / (count - 1))}
                             for i in range(count))
        require(metadata["positions"] == positions and metadata["prompts"] == len(rows)
                and metadata["tokens"] == len(positions), f"lens {key} activation identities")
        require(math.isfinite(metadata["terminal_max_error"]), "terminal parity diagnostic")
    require(late["metadata"]["positions"] == trained["check"]["positions"], "late lens same check rows")
    require(len(trained["history"]) == spec["lens"]["updates"], "lens fixed update budget")
    for i, step in enumerate(trained["history"]):
        require(step["step"] == i + 1 and len(step["positions"]) == spec["lens"]["token_batch_size"]
                and all(0 <= value < trained["fit"]["tokens"] for value in step["positions"]), "lens updates index FIT only")
        close(step["learning_rate"], spec["lens"]["learning_rate"] * (1 - i / spec["lens"]["updates"]), "lens learning rate")
        require(all(math.isfinite(value) for value in step["kl"].values()), "lens finite KL history")
    for layer in map(str, spec["layers"]):
        initial = trained["initial_heldout_kl"][layer]
        close(initial["frozen"], initial["tuned"], "identity lens control")
        close(initial["frozen"], trained["final_heldout_kl"][layer]["frozen"], "frozen KL unchanged")
    improvement = 1 - sum(value["tuned"] for value in trained["final_heldout_kl"].values()) / sum(value["frozen"] for value in trained["final_heldout_kl"].values())
    close(improvement, trained["relative_kl_improvement"], "lens qualification arithmetic")
    require(trained["translator_qualified"] == (improvement >= spec["lens"]["min_relative_kl_improvement"]), "lens qualification")
    require(trained["model_parameters_unchanged"] and trained["reload_exact"]
            and trained["lens_sha256"] == bundle["hashes"]["study/tuned-lens.safetensors"], "lens serialization receipt")
    for name, value in doc(bundle, "frozen-backbone.json").items():
        require(trained["model_sha256"][name] == value, "lens frozen model identity")
    return {"translator_qualified": trained["translator_qualified"], "relative_kl_improvement": improvement,
            "fit_prompts": trained["fit"]["prompts"], "check_prompts": trained["check"]["prompts"],
            "fit_positions": trained["fit"]["tokens"], "check_positions": trained["check"]["tokens"],
            "training_updates": len(trained["history"]), "final_heldout_kl": trained["final_heldout_kl"],
            "late_heldout_kl": late["heldout_kl"], "raw_full_vocabulary_KL_independently_recomputed": False}


def feature_audit(unit, stream, data, spec, counters):
    features = {name: [] for name in FEATURES if name != "permuted"}
    metrics = ("top_probability", "entropy", "top_margin", "target_logp", "target_margin")
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
                score([{**row, "output": row[lens][layer]} for row in records], expected, data["codes"], counters)
                features[lens].extend(mean(row[lens][layer][key] for row in records) for key in metrics[-2:])
    close(unit["features"], features, "independent pre-action feature averages", tolerance=1e-12)
    return features


def refit_predictors(raw, outcomes, fitted, spec):
    require([unit["id"] for unit in raw] == fitted["train_units"] and all(unit["split"] == "train" for unit in raw), "predictor TRAIN-only order")
    require(digest(raw) == fitted["training_data_sha256"], "predictor complete training-data hash")
    permutation = np.random.default_rng(spec["data_seed"]).permutation(len(raw)).tolist()
    require(fitted["permutation"] == permutation and fitted["alpha"] == spec["predictor"]["alpha"], "fixed permutation and ridge penalty")
    maximum, permutation_audit = 0.0, {}
    for action in TRAINABLE:
        permutation_audit[action] = {}
        for label in ("recovered", "utility"):
            y = np.array([unit["actions"][action][label] for unit in outcomes], dtype=float)
            changed = np.flatnonzero(y != y[permutation]).tolist()
            permutation_audit[action][label] = {
                "labels": len(y), "changed_labels": len(changed), "unchanged_labels": len(y) - len(changed),
                "original_labels": y.tolist(), "permuted_labels": y[permutation].tolist(),
                "changed_unit_ids": [raw[i]["id"] for i in changed], "inert_for_this_scalar": not changed,
            }
            observed = fitted["models"][action][label]
            close(observed["prevalence"], float(y.mean()), "TRAIN target mean")
            require(observed["estimable_binary"] == (len(set(y)) > 1), "TRAIN label variation")
            for feature in FEATURES:
                matrix = np.array([unit["features"]["tuned" if feature == "permuted" else feature] for unit in raw])
                center = matrix.mean(axis=0)
                scale = np.sqrt(np.mean((matrix - center) ** 2, axis=0))
                scale[scale < 1e-12] = 1
                z = (matrix - center) / scale
                target = y[permutation] if feature == "permuted" else y
                design = np.vstack((z, np.sqrt(fitted["alpha"]) * np.eye(z.shape[1])))
                residual = np.concatenate((target - target.mean(), np.zeros(z.shape[1])))
                weights = np.linalg.lstsq(design, residual, rcond=None)[0]
                model = {"center": center.tolist(), "scale": scale.tolist(), "intercept": float(target.mean()), "weights": weights.tolist()}
                close(observed["models"][feature], model, "independent augmented least-squares refit")
                error = np.abs(weights - np.array(observed["models"][feature]["weights"]))
                maximum = max(maximum, float(error.max()))
    return {"models_refitted": 20, "max_weight_difference": maximum, "training_units": fitted["train_units"],
            "training_source_clusters": len({unit["stream"] for unit in raw}), "permutation": permutation,
            "solver": "Augmented least squares via SVD, independently of experiment normal-equation solve.",
            "heldout_labels_used": 0, "feature_dimensions": {key: len(raw[0]["features"][key]) for key in raw[0]["features"]},
            "permutation_audit": permutation_audit,
            "fixed_permutation_indices": [i for i, value in enumerate(permutation) if i == value],
            "permutation_redraws": 0}


def prediction(model, features):
    return model["intercept"] + sum(weight * (value - center) / scale for value, center, scale, weight
                                    in zip(features, model["center"], model["scale"], model["weights"], strict=True))


def heldout_audit(raw, outcomes, fitted, published):
    require(all(unit["split"] == "test" for unit in raw)
            and not set(fitted["train_units"]).intersection(unit["id"] for unit in raw), "heldout predictor separation")
    groups = {stream: [i for i, row in enumerate(raw) if row["stream"] == stream] for stream in sorted({row["stream"] for row in raw})}
    predictions, utilities = {}, {feature: {"none": [0.0] * len(raw)} for feature in FEATURES}
    for action in TRAINABLE:
        y = [float(row["actions"][action]["recovered"]) for row in outcomes]
        trained = fitted["models"][action]
        probs = {"prevalence": [trained["recovered"]["prevalence"]] * len(raw)}
        for feature in FEATURES:
            x = [row["features"]["tuned" if feature == "permuted" else feature] for row in raw]
            probs[feature] = [min(1.0, max(0.0, prediction(trained["recovered"]["models"][feature], row))) for row in x]
            utilities[feature][action] = [prediction(trained["utility"]["models"][feature], row) for row in x]
        errors = {feature: [(p - label) ** 2 for p, label in zip(values, y, strict=True)] for feature, values in probs.items()}
        result = {"labels": y, "predictions": probs, "brier": {name: mean(values) for name, values in errors.items()},
                  "brier_per_skill": errors, "train_binary_variation": trained["recovered"]["estimable_binary"],
                  "test_binary_variation": len(set(y)) > 1, "test_units": [row["id"] for row in raw]}
        if published:
            close(published["recovery_prediction"][action], result, "heldout Brier and probabilities")
        result["brier_by_source"] = {name: {stream: mean(values[i] for i in indices) for stream, indices in groups.items()} for name, values in errors.items()}
        result["source_balanced_brier"] = {name: mean(values.values()) for name, values in result["brier_by_source"].items()}
        predictions[action] = result
    choices = {feature: [max(("none", *TRAINABLE), key=lambda action, i=i: values[action][i]) for i in range(len(raw))]
               for feature, values in utilities.items()}
    choices.update({"always_" + action: [action] * len(raw) for action in ("none", *TRAINABLE)})
    policies = {}
    oracle = [max(row["actions"][action]["utility"] for action in ("none", *TRAINABLE)) for row in outcomes]
    for name, selected in choices.items():
        actual = [row["actions"][action]["utility"] for row, action in zip(outcomes, selected, strict=True)]
        policy = {"choices": selected, "utility_per_skill": actual, "mean_utility": mean(actual),
                  "mean_oracle_regret": mean(best - value for best, value in zip(oracle, actual, strict=True)),
                  "by_stream": {stream: mean(actual[i] for i in indices) for stream, indices in groups.items()}}
        if published:
            close(published["policies"][name], policy, "heldout policy choices, utility and regret")
        policy["source_balanced_utility"] = mean(policy["by_stream"].values())
        policies[name] = policy
    always = policies["always_replay_balanced"]
    for policy in policies.values():
        policy["utility_minus_always_balanced"] = policy["mean_utility"] - always["mean_utility"]
        policy["paired_difference_by_source"] = {stream: value - always["by_stream"][stream] for stream, value in policy["by_stream"].items()}
    if published:
        close(published["counts"], {key: len(value) for key, value in groups.items()}, "heldout cluster counts")
        require([row["id"] for row in published["units"]] == [row["id"] for row in raw], "published heldout unit order")
        for row, observed in zip(outcomes, published["units"], strict=True):
            close({key: observed[key] for key in ("before", "after")}, {key: row[key] for key in ("before", "after")}, "published heldout baselines")
            for action, values in observed["actions"].items():
                close(values, {key: row["actions"][action][key] for key in values}, "published heldout outcomes")
        for stream, indices in groups.items():
            summary = {
                "intents": len(indices),
                "mean_readouts": {feature: [mean(raw[i]["features"][feature][j] for i in indices)
                                             for j in range(len(raw[0]["features"][feature]))]
                                  for feature in FEATURES if feature != "permuted"},
                "mean_gain": {action: mean(outcomes[i]["actions"][action]["gain"] for i in indices) for action in TRAINABLE},
            }
            close(published["run_level"][stream], summary, "published run-level readouts and gain")
    regression = {}
    for action in TRAINABLE:
        targets = [row["actions"][action]["utility"] for row in outcomes]
        forecasts = {feature: utilities[feature][action] for feature in FEATURES}
        forecasts["train_mean"] = [fitted["models"][action]["utility"]["prevalence"]] * len(raw)
        regression[action] = {}
        for feature, values in forecasts.items():
            errors = [(value - target) ** 2 for value, target in zip(values, targets, strict=True)]
            absolute = [abs(value - target) for value, target in zip(values, targets, strict=True)]
            source_mse = {stream: mean(errors[i] for i in indices) for stream, indices in groups.items()}
            source_mae = {stream: mean(absolute[i] for i in indices) for stream, indices in groups.items()}
            regression[action][feature] = {"predictions": values, "mse": mean(errors), "mse_by_source": source_mse,
                                            "equal_source_mse": mean(source_mse.values()), "mae": mean(absolute),
                                            "mae_by_source": source_mae, "equal_source_mae": mean(source_mae.values())}
    winners = {row["id"]: [action for action in ("none", *TRAINABLE)
                            if math.isclose(row["actions"][action]["utility"], max(row["actions"][candidate]["utility"]
                                                                                    for candidate in ("none", *TRAINABLE)), abs_tol=1e-12)]
               for row in outcomes}
    return {"published_analysis_available": published is not None, "source_counts": {key: len(value) for key, value in groups.items()},
            "recovery_prediction": predictions, "policies": policies, "source_clusters": len(groups),
            "utility_prediction_audit_diagnostic": regression,
            "utility_prediction_diagnostic_scope": "Post-hoc audit of sealed continuous forecasts; no refitting, selection or new hyperparameters.",
            "skill_oracle_actions": winners,
            "all_predictor_choices_equal_always_balanced": all(policies[name]["choices"] == always["choices"] for name in FEATURES)}


def extended_audit(bundles, payloads, executions, data, spec, counters):
    present = set(payloads)
    result = {"pending": {PREFIX + key: bundle.get("execution", {}).get("status", "not_collected_or_not_completed")
                          for key, bundle in bundles.items() if not bundle["available"]}, "measurements": {}, "references": {}}
    raw_by_stream, outcomes_by_stream = {}, {}
    cohort = doc(bundles["cohort"], "qualification.json")

    def prerequisite(suffix, parent, field):
        require(parent in present and payloads[suffix][field] == bundles[parent]["hashes"]["study/receipt.json"], f"{suffix} prerequisite {parent}")
        require(executions[parent]["finished_at"] < executions[suffix]["started_at"], f"{suffix} starts after {parent}")

    for stream_name, stream in data["streams"].items():
        suffix = "measure-" + stream_name
        if suffix not in present:
            continue
        bundle, payload = bundles[suffix], payloads[suffix]
        prerequisite(suffix, "cohort", "qualification_sha256")
        prerequisite(suffix, "pilot-gate", "pilot_qualification_sha256")
        if stream["split"] == "test":
            prerequisite(suffix, "repair-gate", "repair_qualification_sha256")
        raw = doc(bundle, "units.json")
        frozen = doc(bundle, "features-frozen.json")
        require(payload["features_sha256"] == bundle["hashes"]["study/features-frozen.json"], "frozen feature artifact")
        require([row["intent"] for row in raw] == cohort["streams"][stream_name]["eligible"], "all eligible measurement units")
        checkpoint_hashes = {"none": bundles["learn-" + stream_name]["hashes"]["study/forgotten/adapter_model.safetensors"],
                             "restore": bundles["learn-" + stream_name]["hashes"]["study/acquired/adapter_model.safetensors"]}
        reduced, reused = [], 0
        for unit, before_actions in zip(raw, frozen, strict=True):
            require(before_actions == {**{key: value for key, value in unit.items() if key != "reused_pilot_receipt_sha256"}, "actions": {}}, "frozen features/probes/baselines exact")
            feature_audit(unit, stream, data, spec, counters)
            if "reused_pilot_receipt_sha256" in unit:
                require(stream["split"] == "train", "no heldout pilot reuse")
                pilot = bundles["pilot-" + stream_name]
                prior = next(row for row in doc(pilot, "units.json") if row["id"] == unit["id"])
                require(unit["reused_pilot_receipt_sha256"] == pilot["hashes"]["study/receipt.json"]
                        and unit["baseline"] == prior["baseline"] and unit["actions"] == prior["actions"], "pilot reuse exact")
                reused += 1
            reduced.append(audit_action_unit(bundle, unit, stream, data, spec, counters, checkpoint_hashes))
        require(payload["reused_pilot_units"] == reused and payload["repair_updates"] == (len(raw) - reused) * 48
                and payload["represented_repair_updates"] == len(raw) * 48, "new versus reused update accounting")
        lens = audit_lens(bundle, data, spec)
        means = action_means(reduced)
        for action, reported in payload["stream_summary"]["actions"].items():
            close(reported, {key: means[action][key] for key in reported}, "measurement action means")
        readouts = {key: np.mean([row["features"][key] for row in raw], axis=0).tolist() for key in raw[0]["features"]}
        close(payload["stream_summary"]["mean_readouts"], readouts, "run-level readouts")
        result["measurements"][stream_name] = {"split": stream["split"], "intents": len(raw), "lens": lens,
                                               "actions": means, "units": reduced, "new_repair_updates": payload["repair_updates"],
                                               "pilot_units_reused": reused, "mean_readouts": readouts}
        raw_by_stream[stream_name], outcomes_by_stream[stream_name] = raw, reduced
    fitted = None
    if "repair-gate" in present:
        prerequisite("repair-gate", "cohort", "qualification_sha256")
        streams = [Path(path).parent.name.removeprefix(PREFIX + "measure-") for path in payloads["repair-gate"]["sources"]]
        require(set(streams) == {name for name, stream in data["streams"].items() if stream["split"] == "train"}, "both TRAIN predictor sources")
        require(all(executions["measure-" + stream]["finished_at"] < executions["repair-gate"]["started_at"] for stream in streams), "TRAIN measurements before fitting")
        raw = [row for stream in streams for row in raw_by_stream[stream]]
        reduced = [row for stream in streams for row in outcomes_by_stream[stream]]
        fitted = doc(bundles["repair-gate"], "predictors.json")
        gate = check_feasibility(reduced, payloads["repair-gate"]["feasibility"], spec)
        result["full_train_gate"] = {**gate, "predictor_refit": refit_predictors(raw, reduced, fitted, spec)}
    test_streams = [name for name, stream in data["streams"].items() if stream["split"] == "test"]
    if fitted is not None and all(name in raw_by_stream for name in test_streams):
        analysis = doc(bundles["analyze"], "analysis.json") if "analyze" in present else None
        result["heldout"] = heldout_audit([row for name in test_streams for row in raw_by_stream[name]],
                                          [row for name in test_streams for row in outcomes_by_stream[name]], fitted, analysis)
    result["references"] = reference_audit(bundles, payloads, executions, data, spec, counters, raw_by_stream, outcomes_by_stream)
    result["coverage"] = {
        "completed_stages_audited": len(present), "pending_stages": len(result["pending"]),
        "published_heldout_analysis_audited": "heldout" in result and result["heldout"]["published_analysis_available"],
        "matched_reference_test_split_complete": result["references"]["aggregate"].get("test", {}).get("complete_split", False),
    }
    return result


def paired_metrics(old, fresh, action):
    a, b = old["actions"][action], fresh["actions"][action]
    return {
        "recovery_accuracy": a["accuracy"], "fresh_acquisition_accuracy": b["accuracy"],
        "recovery_minus_fresh_accuracy": a["accuracy"] - b["accuracy"],
        "recovery_minus_fresh_gain": a["gain"] - b["gain"],
        "recovery_guard": a["guard_accuracy"], "fresh_guard": b["guard_accuracy"],
        "recovery_minus_fresh_guard_accuracy": a["guard_accuracy"] - b["guard_accuracy"],
        "recovery_minus_fresh_guard_baseline": old["guard_after"] - fresh["guard_after"],
        "recovery_guard_drop": a["guard_loss"], "fresh_guard_drop": b["guard_loss"],
        "fresh_minus_recovery_guard_drop": b["guard_loss"] - a["guard_loss"],
        "recovery_utility": a["utility"], "fresh_utility": b["utility"],
        "recovery_minus_fresh_utility": a["utility"] - b["utility"],
        "recovery_baseline": old["after"], "fresh_baseline": fresh["after"],
        "recovery_guard_baseline": old["guard_after"], "fresh_guard_baseline": fresh["guard_after"],
        "both_at_accuracy_ceiling": a["accuracy"] == b["accuracy"] == 1.0,
    }


def reference_audit(bundles, payloads, executions, data, spec, counters, main_raw, main_outcomes):
    references = {}
    for stream_name, stream in data["streams"].items():
        learning, measurement, comparison = [f"reference-{stage}-{stream_name}" for stage in ("learn", "measure", "compare")]
        if learning not in payloads:
            continue
        bundle, payload = bundles[learning], payloads[learning]
        source = bundles["learn-" + stream_name]
        require(payload["learning_receipt_sha256"] == source["hashes"]["study/receipt.json"]
                and payload["initial_adapter_sha256"] == source["hashes"]["study/initial/adapter_model.safetensors"], "retain-only initial source")
        observations = doc(bundle, "observations.json")
        require(observations["initial"] == doc(source, "observations.json")["initial"], "retain-only same initial behavior")
        scores = {}
        for phase, panels in observations.items():
            scores[phase] = {key: score(records, rows_for(stream, stream["old" if key == "old_gate" else "new"], "gate"), data["codes"], counters)
                             for key, records in panels.items()}
        acquired = learned(scores["initial"]["new_gate"], scores["retained"]["new_gate"], spec["gates"])
        require(acquired == payload["acquired_new"] and payload["qualified"]
                and len(acquired) >= spec["gates"]["minimum_new_acquired_per_stream"], "retain-only gain gate")
        updates = doc(bundle, "updates.json")
        require(updates["rows"] == rows_for(stream, stream["new"], "learn"), "retain-only never acquires old intents")
        check_updates(updates["updates"], updates["rows"], 128, 8, "retain-only learning")
        schedule = [step["rows"] for step in updates["updates"]]
        require(schedule == [step["rows"] for step in doc(source, "forgetting-updates.json")]
                and payload["schedule_sha256"] == digest({"rows": updates["rows"], "batches": schedule}), "retain-only same new-task schedule")
        require(payload["old_training_exposures"] == payload["test_rows_evaluated"] == 0, "retain-only exposure exclusion")
        reference = {"split": stream["split"], "learning_qualified": True, "new_intents_acquired": len(acquired),
                     "old_training_exposures": 0, "new_training_updates": 128, "matched_initial_adapter_and_new_schedule": True,
                     "measurement_available": measurement in payloads, "published_comparison_available": comparison in payloads}
        references[stream_name] = reference
        if measurement not in payloads:
            continue
        observed, measured = bundles[measurement], payloads[measurement]
        require(measured["reference_receipt_sha256"] == bundle["hashes"]["study/receipt.json"], "reference measurement source receipt")
        for parent, field in (("cohort", "qualification_sha256"), ("pilot-gate", "pilot_qualification_sha256")):
            require(measured[field] == bundles[parent]["hashes"]["study/receipt.json"]
                    and executions[parent]["finished_at"] < executions[measurement]["started_at"], "reference measurement main gate")
        if stream["split"] == "test":
            require("repair-gate" in payloads and payloads["repair-gate"]["qualified"]
                    and executions["repair-gate"]["finished_at"] < executions[measurement]["started_at"], "reference TEST after full TRAIN gate")
        raw = doc(observed, "units.json")
        require([row["intent"] for row in raw] == doc(bundles["cohort"], "qualification.json")["streams"][stream_name]["eligible"], "same reference eligible units")
        frozen = doc(observed, "baselines-frozen.json")
        reduced = []
        for row in raw:
            require(row["baseline"] == frozen[row["intent"]], "reference frozen baselines")
            reduced.append(audit_action_unit(observed, row, stream, data, spec, counters,
                                             {"none": bundle["hashes"]["study/retained/adapter_model.safetensors"]}, reference=True))
        reference["actions"] = action_means(reduced)
        if stream_name not in main_raw:
            continue
        require([row["id"] for row in raw] == [row["id"] for row in main_raw[stream_name]], "paired reference unit order")
        pairs = []
        for old_raw, fresh_raw, old, fresh in zip(main_raw[stream_name], raw, main_outcomes[stream_name], reduced, strict=True):
            actions = {}
            for action in fresh["actions"]:
                x, y = old_raw["actions"][action], fresh_raw["actions"][action]
                require(all(x[key] == y[key] for key in ("updates", "total_exposures", "old_exposures", "new_exposures", "schedule_sha256")), "reference matched repair support/schedule/budget")
                actions[action] = paired_metrics(old, fresh, action)
            pairs.append({"id": old["id"], "stream": stream_name, "actions": actions})
        means = action_means(pairs)
        if comparison in payloads:
            published = doc(bundles[comparison], "comparison.json")
            require([row["id"] for row in published["units"]] == [row["id"] for row in pairs], "published paired unit set")
            for expected, row in zip(published["units"], pairs, strict=True):
                for action, values in expected["actions"].items():
                    close(values, {key: row["actions"][action][key] for key in values}, "published reference excess")
            for action, values in published["run_level"].items():
                close(values, {key: means[action][key] for key in values}, "published run-level reference excess")
        reference.update({"paired_intents": len(pairs), "paired_units": pairs, "run_level": means})
    aggregate = {}
    for split in ("train", "test"):
        selected = {name: value for name, value in references.items() if value["split"] == split and "run_level" in value}
        if not selected:
            continue
        all_pairs = [row for value in selected.values() for row in value["paired_units"]]
        aggregate[split] = {"source_clusters": len(selected), "intents": len(all_pairs), "complete_split": len(selected) == 2,
                            "pooled_intent_means": action_means(all_pairs),
                            "equal_source_means": {action: {key: mean(value["run_level"][action][key] for value in selected.values())
                                                            for key in next(iter(selected.values()))["run_level"][action]}
                                                   for action in next(iter(selected.values()))["run_level"]}}
    return {
        "streams": references, "aggregate": aggregate,
        "paired_guard_scope": "Per-intent guard differences are paired on identical newer examples and repair support/schedule. Also report pre-repair guard differences and clipped guard-drop advantage; positive fresh_minus_recovery_guard_drop means less new-skill damage after the historical path.",
        "history_confound": {
            "historical_before_repair": "128 old-intent updates (1024 exposures), then 128 newer-intent updates with continuous AdamW state.",
            "retain_only_before_repair": "128 identical newer-intent batches from the same initial adapter with initially fresh AdamW; zero old-intent training exposures.",
            "matched_at_repair": "Same pretrained base, adapter capacity, code contract, support rows, schedules, update count and fresh AdamW reset for both histories.",
            "not_isolated": "Lifetime updates/exposures, starting weights for newer training, and optimizer history during newer training differ. A guard advantage can reflect path-dependent robustness from the extra training history; it does not by itself isolate stored old bindings as the cause.",
            "ceiling_limit": "At the 16-update old-accuracy ceiling, endpoint differences cannot establish recovery speed. Smaller matched recovery and retain-only budget curves are needed.",
        },
    }


def audit(args):
    experiment = args.repo / "experiments/recoverability"
    data_path = experiment / "inputs/prospective-cohort.json"
    data = json.loads(data_path.read_text())
    spec = json.loads((experiment / "prospective-protocol.json").read_text())
    core = {name: file_hash(experiment / name) for name in CORE}
    counters = {"records_scored": 0, "max_feature_error": 0.0}
    result = {"status": "pass", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "CLINC intent-to-code binding acquisition, forgetting and TRAIN-only repair feasibility; 16-code classification.",
              "independence": "Standalone scorer and NumPy least-squares refit; imports no experiment scoring, eligibility, utility or gate functions.",
              "self_test": self_test(), "data": data_audit(data, spec), "source_order": source_order_audit(experiment),
              "dataset_file_sha256": file_hash(data_path), "learning_source_sha256": core,
              "sources": {}, "learning": {}}
    bundles, archives = {}, {}
    for suffix in [*("learn-" + stream for stream in data["streams"]), "cohort"]:
        bundles[suffix] = snapshot(args.run_root / (PREFIX + suffix))
    later = ("pilot-train-a", "pilot-train-b", "pilot-gate", "measure-train-a", "measure-train-b", "repair-gate",
             "measure-test-a", "measure-test-b", "analyze",
             *(f"reference-{stage}-{stream}" for stream in data["streams"] for stage in ("learn", "measure", "compare")))
    missing = []
    for suffix in later:
        local = snapshot(args.run_root / (PREFIX + suffix))
        if local["available"]:
            bundles[suffix] = local
        else:
            missing.append(suffix)
    if missing and args.control_pod:
        live = live_snapshots(args.control_pod, args.context, missing)
        bundles.update(live["runs"])
        archives.update(live["archives"])
    for suffix in missing:
        bundles.setdefault(suffix, {"available": False})
    require(all(bundles[name]["available"] for name in later[:2]), "both completed pilots required")
    payloads, seals, executions = {}, {}, {}
    for suffix, bundle in bundles.items():
        if not bundle["available"]:
            continue
        payload, seal = receipt(bundle)
        execution = json.loads(bundle["texts"]["execution.json"])
        require(execution["task_id"] == PREFIX + suffix and execution["exit_code"] == 0, "run execution identity")
        code = execution["source_sha256"]
        if code not in archives:
            archives[code] = archive_manifest(args.run_root.parent / "code" / (code + ".tar"), code)
        members = archives[code]["members"]
        for name, expected in seal["implementation"].items():
            require(members[name] == expected, f"archived implementation {suffix}/{name}")
        require(members["inputs/prospective-cohort.json"] == file_hash(data_path), "archived dataset bytes")
        require(members["prospective-protocol.json"] == file_hash(experiment / "prospective-protocol.json"), "archived protocol bytes")
        require(seal["learning_implementation"] == core, "six frozen learning files")
        require(seal["protocol"] == spec and seal["dataset_sha256"] == digest(data), "shared protocol/data")
        for name in (*CORE, "prospective_pilot.py", "tuned_lens.py", "prospective_reference.py"):
            require(file_hash(experiment / name) == seal["implementation"][name], f"local/source review binding {name}")
        config = seal["config"]
        identity = digest({"settings": {key: config[key] for key in
                            ("model_id", "revision", "acquisition_updates", "forgetting_updates")},
                           "protocol": spec, "dataset": digest(data), "implementation": core})
        require(seal["learning_identity"] == identity, "learning identity recomputation")
        require(seal["measurement_identity"] == digest({"learning": identity, "repair_updates": config["repair_updates"]}),
                "measurement identity recomputation")
        payloads[suffix], seals[suffix], executions[suffix] = payload, seal, execution
        result["sources"][PREFIX + suffix] = {
            "location": "local" if (args.run_root / (PREFIX + suffix)).exists() else "live read-only helper",
            "source_sha256": code, "receipt_sha256": bundle["hashes"]["study/receipt.json"],
            "seal_sha256": bundle["hashes"]["study/seal.json"], "artifacts_verified": len(payload["files"]),
            "started_at": execution["started_at"], "finished_at": execution["finished_at"],
        }
    result["source_archives"] = {key: {k: v for k, v in value.items() if k != "members"} for key, value in archives.items()}
    gates, counts = spec["gates"], Counter()
    for stream_name, stream in data["streams"].items():
        suffix = "learn-" + stream_name
        bundle, payload, config = bundles[suffix], payloads[suffix], seals[suffix]["config"]
        observed, phase_scores = doc(bundle, "observations.json"), {}
        for phase, panels in observed.items():
            require(set(panels) <= {"old_gate", "new_gate", "train"}, "no learning-stage TEST/probe records")
            phase_scores[phase] = {}
            for panel, records in panels.items():
                intents = stream["old"] if panel == "old_gate" or (panel == "train" and phase == "acquired") else stream["new"]
                role = "learn" if panel == "train" else "gate"
                phase_scores[phase][panel] = score(records, rows_for(stream, intents, role), data["codes"], counters)
        initial, acquired, forgotten = [phase_scores[phase]["old_gate"] for phase in ("initial", "acquired", "forgotten")]
        new_start, new_final = phase_scores["acquired"]["new_gate"], phase_scores["forgotten"]["new_gate"]
        acquired_intents = learned(initial, acquired, gates)
        new_intents = learned(new_start, new_final, gates)
        eligible = [intent for intent in acquired_intents if forgotten[intent] <= gates["after_max"]
                    and acquired[intent] - forgotten[intent] >= gates["drop_min"] - 1e-12]
        require(payload["acquired"] == acquired_intents and payload["new_acquired"] == new_intents
                and payload["eligible"] == eligible, "independent learning eligibility")
        passed = len(acquired_intents) >= gates["minimum_acquired_per_stream"] and len(new_intents) >= gates["minimum_new_acquired_per_stream"]
        passed = passed and len(eligible) >= gates["minimum_forgotten_per_stream"]
        require(passed and payload["status"] == "qualified", "all learning prerequisites")
        require(all(payload[name] == 0 for name in ("repair_updates", "lens_updates", "test_rows_evaluated")), "no premature repair/lens/TEST")
        acquisition_gate = doc(bundle, "acquisition-gate.json")
        require(acquisition_gate["acquired"] == acquired_intents and acquisition_gate["acquisition_passed"], "acquisition receipt")
        require(acquisition_gate["old_validation"] == acquired and acquisition_gate["initial_old_validation"] == initial
                and acquisition_gate["old_validation_gain"] == {name: acquired[name] - initial[name] for name in acquired}, "acquisition gain arithmetic")
        for phase, group in (("acquisition", "old"), ("forgetting", "new")):
            training = rows_for(stream, stream[group], "learn")
            trace = doc(bundle, phase + "-updates.json")
            check_updates(trace, training, config[phase + "_updates"], spec["optimizer"]["batch_size"], suffix + phase)
            require(all(len({training[i]["intent"] for i in item["rows"]}) == 8 for item in trace), "balanced learning intent coverage")
        for phase, files in doc(bundle, "checkpoints.json").items():
            require(all(bundle["hashes"][f"study/{phase}/{name}"] == expected for name, expected in files.items()), "checkpoint identity")
        lines = bundle["texts"]["run.log"].splitlines()
        marker = [i for i, line in enumerate(lines) if "PROSPECTIVE_ACQUISITION_PASSED" in line]
        halves = [i for i, line in enumerate(lines) if "PROSPECTIVE_UPDATE step=1/64 " in line]
        require(len(marker) == 1 and len(halves) == 2 and marker[0] < min(halves), "acquisition logged before forgetting")
        counts[stream["split"]] += len(eligible)
        result["learning"][stream_name] = {
            "old_acquired": len(acquired_intents), "new_acquired": len(new_intents), "forgotten_eligible": len(eligible),
            "old": {name: {"initial": initial[name], "acquired": acquired[name], "gain": acquired[name] - initial[name],
                           "forgotten": forgotten[name], "drop": acquired[name] - forgotten[name], "eligible": name in eligible}
                    for name in initial},
            "new": {name: {"phase_start": new_start[name], "final": new_final[name], "gain": new_final[name] - new_start[name],
                           "acquired": name in new_intents} for name in new_start},
            "acquisition_gate_log": lines[marker[0]], "first_forgetting_update_log": lines[halves[0]],
            "repair_updates": 0, "lens_updates": 0, "test_rows_evaluated": 0,
        }
    cohort = doc(bundles["cohort"], "qualification.json")
    require(cohort["counts"] == dict(counts) and cohort["clusters"] == {"train": 2, "test": 2} and cohort["qualified"], "global cohort gate")
    require(all(counts[split] >= gates["minimum_forgotten"][split] for split in counts), "global eligible counts")
    for name, item in cohort["streams"].items():
        require(item["eligible"] == payloads["learn-" + name]["eligible"] and item["passed"], "cohort independent eligible set")
        require(item["receipt_sha256"] == bundles["learn-" + name]["hashes"]["study/receipt.json"], "cohort source receipt")
        require(executions["learn-" + name]["finished_at"] < executions["cohort"]["started_at"], "learn before cohort")
    result["cohort"] = {"qualified": True, "counts": dict(counts), "clusters": cohort["clusters"], "independently_recomputed": True}
    units, stream_means = [], {}
    for suffix in later[:2]:
        bundle, payload, config = bundles[suffix], payloads[suffix], seals[suffix]["config"]
        stream_name, stream_units = payload["stream"], doc(bundle, "units.json")
        stream = data["streams"][stream_name]
        require(stream["split"] == payload["split"] == "train" and payload["lens_updates"] == 0, "TRAIN-only pilot")
        require(payload["qualification_sha256"] == bundles["cohort"]["hashes"]["study/receipt.json"], "pilot prerequisite receipt")
        require(executions["cohort"]["finished_at"] < executions[suffix]["started_at"], "cohort completed before pilot")
        require([unit["intent"] for unit in stream_units] == cohort["streams"][stream_name]["eligible"][:2], "predeclared pilot subset")
        require(config["repair_updates"] == 16 and payload["repair_updates"] == 96, "fixed pilot update budget")
        reduced = []
        frozen = doc(bundle, "baselines-frozen.json")
        for index, unit in enumerate(stream_units):
            require(unit["baseline"] == frozen[index]["baseline"] and not frozen[index]["actions"], "pre-action frozen baseline")
            checkpoint_hashes = {
                "none": bundles["learn-" + stream_name]["hashes"]["study/forgotten/adapter_model.safetensors"],
                "restore": bundles["learn-" + stream_name]["hashes"]["study/acquired/adapter_model.safetensors"],
            }
            reduced.append(audit_action_unit(bundle, unit, stream, data, spec, counters, checkpoint_hashes))
        summary = payload["stream_summary"]
        aggregate = {action: {key: mean(unit["actions"][action][key] for unit in reduced)
                              for key in ("accuracy", "guard_accuracy", "gain", "utility", "recovered")} for action in ACTIONS}
        require(summary["before_accuracy"] == mean(unit["before"] for unit in reduced)
                and summary["after_accuracy"] == mean(unit["after"] for unit in reduced), "stream baseline means")
        for action in ACTIONS:
            require(all(math.isclose(value, summary["actions"][action][key], abs_tol=1e-12) for key, value in aggregate[action].items()), "reported guard/utility/recovery arithmetic")
        stream_means[stream_name] = aggregate
        units.extend(reduced)
    rule = spec["repair_feasibility"]
    responders = [unit["id"] for unit in units if max(unit["actions"][action]["gain"] for action in TRAINABLE) >= rule["response_gain_min"] - 1e-12]
    best = [max(unit["actions"][action]["utility"] for action in TRAINABLE) for unit in units]
    paired = [unit["actions"]["replay_target"]["utility"] - unit["actions"]["replay_balanced"]["utility"] for unit in units]
    utility_range, paired_range = max(best) - min(best), max(paired) - min(paired)
    qualified = len(responders) >= rule["minimum_responsive_train_skills"] and (utility_range >= rule["utility_range_min"] - 1e-12 or paired_range >= rule["paired_action_range_min"] - 1e-12)
    single_class = [action for action in TRAINABLE if len({unit["actions"][action]["recovered"] for unit in units}) == 1]
    gate = {"available": bundles["pilot-gate"]["available"]}
    if gate["available"]:
        official = doc(bundles["pilot-gate"], "pilot-qualification.json")
        checked = official["feasibility"]
        require(official["qualified"] == qualified and checked["responsive_train_units"] == responders
                and math.isclose(checked["utility_range"], utility_range, abs_tol=1e-12)
                and math.isclose(checked["paired_action_range"], paired_range, abs_tol=1e-12)
                and checked["single_class_binary_actions"] == single_class, "independent pilot gate agreement")
        gate.update({"qualified": official["qualified"], "agrees_with_independent_recomputation": True})
    result["pilot"] = {"units": units, "cluster_means": stream_means, "source_clusters": 2,
                       "repair_updates": 192, "lens_updates": 0, "responsive_units": responders,
                       "utility_range": utility_range, "paired_action_range": paired_range,
                       "feasibility_qualified": qualified, "single_class_binary_actions": single_class,
                       "best_deployable_action": {unit["id"]: max(("none", *TRAINABLE), key=lambda action: unit["actions"][action]["utility"]) for unit in units},
                       "official_gate_snapshot": gate}
    result.update(extended_audit(bundles, payloads, executions, data, spec, counters))
    result["checks"] = {**counters, "prediction_mismatches": 0, "eligibility_mismatches": 0,
                        "reported_utility_mismatches": 0, "frozen_source_unchanged": True,
                        "remote_reads_only": True, "gpu_dispatches": 0}
    result["limitations"] = [
        "Code-restricted classification of learned bindings; no claim about free generation, global vocabulary or semantic concepts being erased.",
        "TRAIN/test predictor cohorts denote disjoint intents and source runs. TRAIN action outcomes use those intents' heldout CLINC TEST utterances as training labels; heldout predictor-intent actions were not inspected.",
        "Acquisition/forgetting selection uses ten validation utterances per intent. Passing thresholds does not imply every unseen utterance was learned or forgotten; retain every gate-eligible unit regardless of its TEST baseline.",
        "Shared guards and source weights require clustering. There are only two independently trained source runs per prediction split, not one independent model per intent or utterance.",
        "Utility variation alone does not establish adaptive action-selection value or lens predictive value; compare against always-balanced and output-only baselines on heldout source runs.",
        "Constant binary guard-aware recovery labels permit continuous utility evaluation but cannot support binary discrimination claims for that arm/split.",
        "Matched updates and total exposures are not matched old-support exposures: target-only 128, balanced 64 old plus 64 new. The comparison includes both dilution and rehearsal of new bindings.",
        "Restore reinstates an archived adapter and sacrifices newer skills; it is a checkpoint accessibility control and old-behavior ceiling, not repair evidence within the forgotten weights or a deployable policy.",
        "Sham uses balanced support with target codes rotated, preserving newer labels. It is wrong-binding supervision, not a zero-update placebo; repeated wrong labels can change the model.",
        "Actual gradient targets and reload equality are source/receipt-backed, not independently rerun on the model here. Readout arithmetic and all checkpoint file hashes were independently verified.",
        "Positive recovery excess over retain-only acquisition is budget-specific evidence, not proof of semantic erasure or a particular storage mechanism. A single 16-update endpoint cannot establish recovery speed, and target accuracy ceilings can conceal differences.",
        "Lens KL reductions are recomputed from stored aggregate KL values; raw activation caches and full-vocabulary teacher/lens logits were not persisted, so independent forward-KL reevaluation is unavailable in this CPU audit.",
        "One permutation shuffles TRAIN labels across intents globally, not within source clusters. It is a comparison model, not an exchangeability-respecting significance test.",
        "A scalar whose TRAIN labels are unchanged by the saved permutation has an inert shuffled-label control. The original permutation is preserved, not redrawn after observing this result; inspect per-action binary and utility change counts separately.",
        "Pooled intent means weight the two TEST sources by their eligible intent counts. The audit additionally reports equal-source means and paired differences by source, without confidence or significance claims.",
        "An independent passing pilot recomputation does not replace downstream entrypoint gates: full TRAIN measurement and repair qualification still precede heldout actions.",
    ]
    require(core == {name: file_hash(experiment / name) for name in CORE}, "frozen source changed while auditing")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--control-pod")
    parser.add_argument("--context", default="us-mi355x-nambiar-k8s")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    result = audit(args)
    result["audit_script_sha256"] = file_hash(Path(__file__))
    source_archive = args.output.parent / "recovery-audit.sources" / (result["audit_script_sha256"] + ".py")
    source_archive.parent.mkdir(exist_ok=True)
    if source_archive.exists():
        require(file_hash(source_archive) == result["audit_script_sha256"], "audit source archive identity")
    else:
        with source_archive.open("xb") as handle:
            handle.write(Path(__file__).read_bytes())
    result["audit_script_snapshot"] = str(source_archive.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                      "eligible": result["cohort"]["counts"], "pilot_qualified": result["pilot"]["feasibility_qualified"],
                      "official_gate": result["pilot"]["official_gate_snapshot"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
