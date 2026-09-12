import argparse
import hashlib
import json
import math
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from audit_consolidation import (
    check_rows,
    corpus_from_fixtures,
    file_hash,
    require,
    summarize,
)
from safetensors.torch import load_file
from tokenizers import Tokenizer

SOURCE = "85c375eef798c920ef3c89d3fbb08b4b55bcb6481fe281b0403bcc277cbc3a9b"
PROTOCOL = "31ec528d68e04062d7be72c3b8d0c8d1c5b915b7670c3be4be5af0a69f13a2be"
REMOTE = r'''import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
root = Path('/mnt/shared/cl-portfolio')
source = '85c375eef798c920ef3c89d3fbb08b4b55bcb6481fe281b0403bcc277cbc3a9b'
code = root / 'code' / source
archive = code.with_suffix('.tar')
digest = hashlib.sha256()
files = {}
with tarfile.open(archive) as stream:
    for member in sorted(stream.getmembers(), key=lambda row: Path(row.name)):
        path = Path(member.name)
        if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name in files:
            raise ValueError('AUDIT_ARCHIVE_MEMBER')
        payload = stream.extractfile(member).read()
        if (code / path).read_bytes() != payload:
            raise ValueError('AUDIT_SOURCE_BYTES_CHANGED: ' + member.name)
        digest.update(member.name.encode() + b'\0' + payload)
        files[member.name] = hashlib.sha256(payload).hexdigest()
if digest.hexdigest() != source or len(files) != 130:
    raise ValueError('AUDIT_ARCHIVE_IDENTITY')
result = {'observed_at': datetime.now(timezone.utc).isoformat(),
          'source': {'sha256': source, 'files_sha256': files, 'file_count': len(files),
                     'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, 'runs': {}}
base = json.loads((code / 'configs/choice_consolidation_protocol.json').read_text())['source_base']
payload = (Path(base['local_path']) / 'config.json').read_bytes()
result['model_config'] = {'value': json.loads(payload), 'bytes': len(payload),
                          'sha256': hashlib.sha256(payload).hexdigest(),
                          'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()}
for method in ['cached-teacher-kl', 'cached-teacher-sft', 'onpolicy-kl']:
    for stage in ['train', 'evaluate']:
        identity = 'followthrough-20260912-onpolicy303-' + method + '-' + stage
        folder = root / 'runs' / identity
        item = {'exists': folder.exists(), 'files': {}}
        for name in ['execution.json', 'task.json', 'config.json', 'seal.json', 'failure.json', 'runtime.json',
                     'runtime_dependency.json', 'teacher_runtime/runtime.json', 'teacher_fresh_qualification.json',
                     'teacher_fresh_train.json', 'teacher_fresh_validation.json', 'teacher_rejected.json',
                     'initial_train_probes.json', 'final_train_probes.json', 'untouched_train_probes.json', 'initial_retention.json',
                     'final_retention.json', 'training.json', 'result.json', 'panels.json']:
            path = folder / name
            if path.exists():
                payload = path.read_bytes()
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError as error:
                    item['files'][name] = {'incomplete_read': str(error), 'bytes': len(payload)}
                    continue
                item['files'][name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                                      'mtime_ns': path.stat().st_mtime_ns, 'value': value}
        path = folder / 'events.jsonl'
        if path.exists():
            payload = path.read_bytes()
            stop = payload.rfind(b'\n') + 1
            rows = [json.loads(line) for line in payload[:stop].splitlines()]
            item['events'] = {'complete_rows': rows, 'sha256_of_complete_prefix': hashlib.sha256(payload[:stop]).hexdigest(),
                              'ignored_partial_bytes': len(payload) - stop}
        if stage == 'train':
            path = folder / method.replace('-', '_') / 'trajectories.jsonl'
            if path.exists():
                payload = path.read_bytes()
                stop = payload.rfind(b'\n') + 1
                item['trajectory_prefix'] = {'complete_rows': [json.loads(line) for line in payload[:stop].splitlines()],
                                             'sha256_of_complete_prefix': hashlib.sha256(payload[:stop]).hexdigest(),
                                             'ignored_partial_bytes': len(payload) - stop}
            training = item['files'].get('training.json', {}).get('value')
            if training:
                path = folder / method.replace('-', '_') / 'checkpoint384/train_probes.json'
                payload = path.read_bytes()
                item['files']['final_train_probes.json'] = {'sha256': hashlib.sha256(payload).hexdigest(),
                    'bytes': len(payload), 'mtime_ns': path.stat().st_mtime_ns, 'value': json.loads(payload),
                    'original_relative_path': str(path.relative_to(folder))}
                item['stored_checkpoints'] = {}
                specifications = {'initial': training['initial'], 'final': training['arm']['checkpoints']['384']['adapter']}
                for label, spec in specifications.items():
                    location = Path(spec['path'])
                    payload = (location / 'adapter_model.safetensors').read_bytes()
                    length = int.from_bytes(payload[:8], 'little')
                    header = json.loads(payload[8:8 + length])
                    data = memoryview(payload)[8 + length:]
                    tensor_hash = hashlib.sha256()
                    elements = 0
                    shapes = {}
                    for name, tensor in sorted(header.items()):
                        if name == '__metadata__':
                            continue
                        if tensor['dtype'] != 'F32':
                            raise ValueError('AUDIT_ADAPTER_PRECISION')
                        shape = tuple(tensor['shape'])
                        start, end = tensor['data_offsets']
                        tensor_hash.update(name.encode())
                        tensor_hash.update(('(' + str(shape) + ', torch.float32)').encode())
                        tensor_hash.update(data[start:end])
                        elements += (end - start) // 4
                        shapes[name] = list(shape)
                    config_path = location / 'adapter_config.json'
                    item['stored_checkpoints'][label] = {'tensor_sha256': tensor_hash.hexdigest(), 'parameter_count': elements,
                        'tensor_payload_bytes': len(data), 'shapes': shapes,
                        'files_sha256': {'adapter_model.safetensors': hashlib.sha256(payload).hexdigest(),
                                         'adapter_config.json': hashlib.sha256(config_path.read_bytes()).hexdigest()},
                        'config': json.loads(config_path.read_text())}
        result['runs'][identity] = item
print(json.dumps(result, allow_nan=False))
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def context(repo, wave):
    lane = repo / "experiments/distillation"
    handoff = read(lane / "onpolicy303_handoff.json")
    design = read(lane / "configs/onpolicy303_protocol.json")
    require(digest(design) == PROTOCOL, "protocol identity")
    for name, expected in handoff["scientific_closure_sha256"].items():
        require(file_hash(lane / name) == expected, f"local frozen source {name}")
    choice, corpus = corpus_from_fixtures(repo)
    tokenizer_path = wave / "tokenizer/tokenizer.json"
    spec = next(row for row in choice["source_base"]["files"] if row["path"] == "tokenizer.json")
    require(file_hash(tokenizer_path) == spec["sha256"], "independent tokenizer identity")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source = wave / "runs/consolidation-source303-coverage-20260912/qualification"
    require(file_hash(source / "qualification.json") == design["teacher"]["qualification_sha256"], "teacher certificate")
    qualification = read(source / "qualification.json")
    require(tokenizer.token_to_id("<|im_end|>") == qualification["eos_token_id"] == 151645, "native EOS")
    canonical = read(source / "qualification_panels.json")
    return {"design": design, "handoff": handoff, "choice": choice, "corpus": corpus, "tokenizer": tokenizer,
            "teacher": qualification, "canonical": canonical, "cap": 16,
            "schedule": read(wave / "runs/consolidation-coverage303-20260912/schedule.json")}


def verify_generations(ctx, rows, records):
    flags = check_rows(rows, records, ctx["tokenizer"], ctx["teacher"]["eos_token_id"], ctx["cap"])
    special = set(ctx["teacher"]["special_token_ids"])
    for row, record in zip(rows, records, strict=True):
        generated = record["generation"]
        tokens, body = generated["token_ids"], generated["body_text"]
        terminated = bool(tokens) and tokens[-1] == ctx["teacher"]["eos_token_id"]
        contents = tokens[:-1] if terminated else tokens
        require(not generated["format_valid"] or not special.intersection(contents), "special token counted as valid")
        extra = {"cap_without_eos": len(tokens) >= 16 and not terminated,
                 "extra_body_characters": len(body) > 6,
                 "correct_prefix_with_extra_text": body.startswith(row["choices"][row["gold_idx"]]) and body != row["choices"][row["gold_idx"]],
                 "exact_body_without_valid_stop": body == row["choices"][row["gold_idx"]] and not terminated,
                 "digit_position_correct": [len(body) > i and body[i] == row["choices"][row["gold_idx"]][i] for i in [1, 3, 5]]}
        require(all(generated[key] == value for key, value in extra.items()), "native error flags")
    return {"correct": sum(flags), "count": len(flags), "families": summarize(flags, rows)}


def trajectory_audit(ctx, records, method):
    rows = ctx["corpus"]["train"]
    canonical = ctx["canonical"]["train"]["generated"]
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    counts = Counter()
    token_counts = Counter()
    prefix = Counter()
    row_losses = {}
    for number, record in enumerate(records):
        step, micro = divmod(number, 4)
        require(record["step"] == step + 1 and record["microstep"] == micro, "trajectory order")
        require(record["id"] == ctx["schedule"][step][micro], "scheduled trajectory")
        index = by_id[record["id"]]
        row = rows[index]
        require(record["row_index"] == index and record["row_sha256"] == digest(row), "trajectory row identity")
        require(record["method"] == method and record["student_optimizer_clock_before_sampling"] == step, "pre-update trajectory state")
        generated = record["generation"]
        native = {"id": row["id"], "task": row["task"], "group": row["group"], "row_sha256": record["row_sha256"], "generation": generated}
        verify_generations(ctx, [row], [native])
        response, prompt = generated["token_ids"], generated["prompt_token_ids"]
        target = canonical[index]["generation"]["token_ids"]
        require(digest(target) == record["canonical_teacher_response_sha256"], "teacher target hash")
        require(ctx["teacher"]["eos_token_id"] not in response[:-1] and 0 < len(response) <= 16, "native sampled stopping")
        diagnostics = record["diagnostics"]
        require(diagnostics["prediction_mask"] == [False] * (len(prompt) - 1) + [True] * len(response), "prompt/EOS/padding loss mask")
        require(diagnostics["loss_tokens"] == len(response) and diagnostics["loss_token_ids"] == response, "loss targets")
        require(diagnostics["vocabulary_size"] == ctx["vocabulary_size"], "full vocabulary size")
        require(all(0 <= token < diagnostics["vocabulary_size"] for token in response), "vocabulary token range")
        require(math.isfinite(diagnostics["row_loss"]), "finite row loss")
        counts[row["task"]] += 1
        token_counts[row["task"]] += len(response)
        prefix["trajectory_correct"] += generated["correct"]
        prefix["trajectory_format_valid"] += generated["format_valid"]
        prefix["trajectory_native_eos"] += generated["terminated"]
        prefix["trajectory_cap_without_eos"] += generated["cap_without_eos"]
        if method != "cached_teacher_sft":
            fields = ["student_argmax_token_ids", "teacher_argmax_token_ids", "student_sampled_token_logp", "teacher_sampled_token_logp", "teacher_entropy", "teacher_eos_probability", "forward_kl_per_token", "prefix_matches_qualified_answer", "qualified_next_token_ids", "teacher_next_token_matches_on_qualified_prefix"]
            require(all(len(diagnostics[key]) == len(response) for key in fields), "per-token diagnostics count")
            matched = [response[:i] == target[:i] and i < len(target) for i in range(len(response))]
            require(diagnostics["prefix_matches_qualified_answer"] == matched, "prefix agreement identity")
            require(diagnostics["qualified_next_token_ids"] == [target[i] if ok else None for i, ok in enumerate(matched)], "prefix labels")
            expected_correct = [diagnostics["teacher_argmax_token_ids"][i] == target[i] if ok else None for i, ok in enumerate(matched)]
            require(diagnostics["teacher_next_token_matches_on_qualified_prefix"] == expected_correct, "teacher-prefix correctness boundary")
            require(not diagnostics["off_answer_prefix_competence_certified"], "no invented off-prefix competence")
            kl = diagnostics["forward_kl_per_token"]
            require(all(math.isfinite(value) and value >= -1e-5 for value in kl), "finite nonnegative full-vocabulary KL")
            require(math.isclose(sum(kl) / len(kl), diagnostics["row_loss"], abs_tol=2e-5, rel_tol=2e-5), "token-mean KL reduction")
            prefix["qualified_prefix_tokens"] += sum(matched)
            prefix["divergent_prefix_tokens"] += len(matched) - sum(matched)
            prefix["teacher_agrees_on_qualified_prefix"] += sum(value is True for value in expected_correct)
        if method == "onpolicy_kl":
            sampling = record["sampling"]
            require(sampling["vocabulary_size"] == sampling["positive_support_count"] == diagnostics["vocabulary_size"], "sampling entire vocabulary")
            require(sampling["temperature"] == 1 and not sampling["forced_eos"], "native sampling policy")
            require(sampling["ended_with_native_eos"] == generated["terminated"] and sampling["cap_without_eos"] == generated["cap_without_eos"], "sampled stop records")
            for key in ("minimum_vocabulary_probabilities", "sampled_token_probabilities"):
                require(len(sampling[key]) == len(response) and all(math.isfinite(value) and 0 < value <= 1 for value in sampling[key]), "full-support finite sampling probabilities")
            gaps = [abs(math.log(probability) - logp) for probability, logp in zip(sampling["sampled_token_probabilities"], diagnostics["student_sampled_token_logp"], strict=True)]
            prefix["maximum_sampling_vs_recomputed_logp_gap"] = max(prefix["maximum_sampling_vs_recomputed_logp_gap"], max(gaps))
        else:
            require(response == target and record["sampling"] is None, "unmodified cached teacher trajectory")
        row_losses.setdefault(step + 1, []).append(diagnostics["row_loss"])
    return {"rows": len(records), "complete_updates_represented": len(records) // 4, "partial_update_rows": len(records) % 4,
            "exposures_per_family": dict(counts), "loss_tokens_per_family": dict(token_counts), "prefix_diagnostics": dict(prefix),
            "cached_trajectory_grades_are_not_student_generated_performance": method != "onpolicy_kl",
            "row_mean_losses_by_update": {str(step): sum(values) / len(values) for step, values in row_losses.items() if len(values) == 4}}


def verify_execution(item, task_id):
    execution, task, config = [item["files"][name]["value"] for name in ("execution.json", "task.json", "config.json")]
    require(execution["task"] == task and execution["task_id"] == task_id == task["id"], "execution/task identity")
    require(task["source_sha256"] == SOURCE and task["config"] == config and config["protocol_sha256"] == PROTOCOL, "actual source/config")
    for name, value in (("task", execution["task"]), ("config", execution["task"]["config"])):
        actual = hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()
        require(actual == execution[name + "_sha256"], "supervisor " + name + " hash")
    require("followthrough-20260912-recovery-worker-runtime-qwen3-8b" in task["depends_on"], "exact runtime dependency")
    return execution, config


def stored_checkpoint_audit(item, ctx):
    training = item["files"]["training.json"]["value"]
    checked = {}
    for label, actual in item["stored_checkpoints"].items():
        spec = training["initial"] if label == "initial" else training["arm"]["checkpoints"]["384"]["adapter"]
        require(actual["files_sha256"] == spec["files"], "actual remote checkpoint file hashes")
        require(actual["tensor_sha256"] == spec["tensor_sha256"], "actual remote checkpoint tensor hash")
        require(actual["parameter_count"] == 3833856 and actual["tensor_payload_bytes"] == 15335424, "resident adapter storage")
        require(actual["config"]["r"] == 8 and actual["config"]["lora_alpha"] == 16 and set(actual["config"]["target_modules"]) == {"q_proj", "v_proj"}, "resident rank/alpha/targets")
        for name, shape in actual["shapes"].items():
            require(len(shape) == 2 and (("lora_A" in name and shape[0] == 8) or ("lora_B" in name and shape[1] == 8)), "only rank8 A/B factors")
        checked[label] = {key: actual[key] for key in ("tensor_sha256", "parameter_count", "tensor_payload_bytes", "files_sha256")}
    require(checked["initial"]["tensor_sha256"] == ctx["design"]["coverage_control"]["initial_tensor_sha256"], "fresh identical learner initialization")
    require(checked["initial"]["tensor_sha256"] != checked["final"]["tensor_sha256"], "actual persistent learner change")
    expected = checked["final"]["tensor_sha256"] == ctx["design"]["coverage_control"]["final_tensor_sha256"]
    require(expected == training["coverage_sft_final_tensors_equal"], "actual coverage control tensor comparison")
    return {"status": "passed", "checkpoints": checked, "matches_coverage_sft_final_tensors": expected}


def retention_audit(ctx, records):
    artifact = Path(ctx["handoff"]["ownership"]["canonical_repository"]) / "experiments/distillation/configs/onpolicy303_retention.json"
    rows = read(artifact)["validation"]
    require(len(records) == len(rows) == 128, "generic retention count")
    flags = []
    for row, record in zip(rows, records, strict=True):
        scores = record["scores"]
        require(record["row_sha256"] == digest(row) and record["task"] == row["task"], "generic retention row binding")
        require(len(scores) == len(row["choices"]) == record["choice_count"] and all(math.isfinite(value) for value in scores), "original generic choice cardinality")
        expected = max(range(len(scores)), key=scores.__getitem__)
        require(record["prediction"] == expected and record["gold"] == row["gold_idx"] and record["correct"] == (expected == row["gold_idx"]), "generic retention argmax")
        flags.append(expected == row["gold_idx"])
    return {"correct": sum(flags), "count": 128, "families": summarize(flags, rows), "scoring": "original multiple-choice likelihood; separate from native generation"}


def prefix_panel_audit(folder, records, arm):
    path = folder / "prefix_diagnostics.json"
    require(file_hash(path) == arm["prefix_diagnostics_sha256"], "prefix panel hash")
    panels, lengths = {}, {}
    minimum_probability = None
    for record in records:
        begin, end = next((begin, end) for begin, end in ((1, 32), (33, 128), (129, 384)) if begin <= record["step"] <= end)
        key = f"{begin}-{end}/{record['task']}"
        panel = panels.setdefault(key, {
            "trajectory_rows": 0, "loss_tokens": 0, "student_sampled_rows": 0,
            "trajectory_whole_answer_correct": 0, "trajectory_format_valid": 0, "trajectory_native_eos": 0,
            "qualified_prefix_positions": 0, "teacher_agrees_on_qualified_prefix_positions": 0,
            "divergent_prefix_positions": 0, "divergent_teacher_entropy_sum": 0.0,
            "divergent_teacher_eos_probability_sum": 0.0, "divergent_teacher_sampled_token_logp_sum": 0.0,
            "full_vocabulary_forward_kl_sum": 0.0, "kl_positions": 0,
        })
        generation, diagnostics = record["generation"], record["diagnostics"]
        panel["trajectory_rows"] += 1
        panel["loss_tokens"] += len(generation["token_ids"])
        panel["student_sampled_rows"] += record["sampling"] is not None
        for destination, field in (("trajectory_whole_answer_correct", "correct"), ("trajectory_format_valid", "format_valid"), ("trajectory_native_eos", "terminated")):
            panel[destination] += generation[field]
        lengths.setdefault(key, Counter())[str(len(generation["token_ids"]))] += 1
        if record["sampling"]:
            observed = min(record["sampling"]["minimum_vocabulary_probabilities"])
            minimum_probability = observed if minimum_probability is None else min(minimum_probability, observed)
        if "forward_kl_per_token" in diagnostics:
            for position, matched in enumerate(diagnostics["prefix_matches_qualified_answer"]):
                panel["kl_positions"] += 1
                panel["full_vocabulary_forward_kl_sum"] += diagnostics["forward_kl_per_token"][position]
                if matched:
                    panel["qualified_prefix_positions"] += 1
                    panel["teacher_agrees_on_qualified_prefix_positions"] += diagnostics["teacher_next_token_matches_on_qualified_prefix"][position]
                else:
                    panel["divergent_prefix_positions"] += 1
                    panel["divergent_teacher_entropy_sum"] += diagnostics["teacher_entropy"][position]
                    panel["divergent_teacher_eos_probability_sum"] += diagnostics["teacher_eos_probability"][position]
                    panel["divergent_teacher_sampled_token_logp_sum"] += diagnostics["teacher_sampled_token_logp"][position]
    require(read(path) == {"panels": panels, "off_answer_prefix_competence_certified": False,
                           "cached_trajectory_grades_are_teacher_targets_not_student_generated_performance": True,
                           "selection_from_diagnostics": False}, "recomputed complete prefix panels")
    return {"panels": panels, "response_length_histograms": {key: dict(histogram) for key, histogram in lengths.items()},
            "minimum_observed_vocabulary_probability": minimum_probability,
            "off_answer_prefix_competence_certified": False}


def live_report(ctx, snapshot):
    spec = next(row for row in ctx["choice"]["source_base"]["files"] if row["path"] == "config.json")
    require(snapshot["model_config"]["bytes"] == spec["bytes"], "model config bytes")
    for field in ("sha256", "git_blob_sha1"):
        if spec[field] is not None:
            require(snapshot["model_config"][field] == spec[field], "actual model config " + field)
    ctx["vocabulary_size"] = snapshot["model_config"]["value"]["vocab_size"]
    for name, expected in ctx["handoff"]["scientific_closure_sha256"].items():
        require(snapshot["source"]["files_sha256"][name] == expected, "actual frozen closure " + name)
    result = {"status": "live_audit", "observed_at": snapshot["observed_at"], "source_sha256": SOURCE,
              "protocol_digest": PROTOCOL, "source_files_checked": 130, "scientific_closure_files_checked": 25,
              "runs": {}, "limitations": "Read-only artifact audit; does not recompute GPU distributions or inference. No recipe changes."}
    for task_id, item in snapshot["runs"].items():
        if "execution.json" not in item["files"]:
            result["runs"][task_id] = {"status": "not_started"}
            continue
        execution, config = verify_execution(item, task_id)
        outcome = {"status": execution["status"], "attempt_id": execution["attempt_id"], "gpus": execution["gpus"],
                   "execution_sha256": item["files"]["execution.json"]["sha256"]}
        events = item.get("events", {}).get("complete_rows", [])
        updates = [row for row in events if row["event"] == "onpolicy303_update"]
        outcome["observed_updates"] = len(updates)
        outcome["last_event"] = events[-1] if events else None
        if "failure.json" in item["files"]:
            outcome["failure"] = item["files"]["failure.json"]
        if config["stage"] == "train":
            outcome["teacher_generation_events"] = {}
            for split in ("train", "validation"):
                teacher_events = [row for row in events if row.get("condition") == "teacher_fresh/" + split]
                outcome["teacher_generation_events"][split] = {"count": len(teacher_events), "correct": sum(row["correct"] for row in teacher_events)}
                name = "teacher_fresh_" + split + ".json"
                if name in item["files"] and "value" in item["files"][name]:
                    records = item["files"][name]["value"]
                    outcome["teacher_generation_events"][split]["rescored"] = verify_generations(ctx, ctx["corpus"][split], records)
                    require(all(a["generation"] == b["generation"] for a, b in zip(records, ctx["canonical"][split]["generated"], strict=True)), "teacher saved/fresh token identity")
            certificate = item["files"].get("teacher_fresh_qualification.json")
            outcome["teacher_certificate_present"] = certificate is not None and "value" in certificate
            if outcome["teacher_certificate_present"]:
                for split, count in (("train", 384), ("validation", 96)):
                    gate = certificate["value"]["gates"][split]
                    require(gate == {"correct": count, "count": count, "coverage_passed": True, "qualified_native_tokens_equal": True}, "teacher native gate")
                require(certificate["value"]["learner_updates"] == certificate["value"]["teacher_optimizer_updates"] == 0 and not certificate["value"]["learner_loaded"], "teacher before learner")
            if updates:
                require(outcome["teacher_certificate_present"], "learner updates without native teacher certificate")
                first_index = events.index(updates[0])
                before = events[:first_index]
                for split, count in (("train", 384), ("validation", 96)):
                    previous = [row for row in before if row.get("condition") == "teacher_fresh/" + split]
                    require(len(previous) == count and all(row["correct"] for row in previous), "teacher event order before first update")
                require(certificate["mtime_ns"] / 1e9 <= updates[0]["unix_time"], "teacher certificate timestamp before first update")
                require([row["step"] for row in updates] == list(range(1, len(updates) + 1)), "actual update order")
                outcome["native_teacher_verified_before_first_update"] = True
                outcome["first_update_unix_time"] = updates[0]["unix_time"]
            if "trajectory_prefix" in item:
                audit = trajectory_audit(ctx, item["trajectory_prefix"]["complete_rows"], config["method"])
                outcome["trajectories"] = {key: value for key, value in audit.items() if key != "row_mean_losses_by_update"}
                comparable_updates = [update for update in updates if str(update["step"]) in audit["row_mean_losses_by_update"]]
                outcome["non_atomic_live_snapshot"] = {"event_updates": len(updates),
                                                       "complete_trajectory_updates": audit["complete_updates_represented"],
                                                       "common_updates_checked": len(comparable_updates),
                                                       "event_updates_awaiting_visible_trajectories": len(updates) - len(comparable_updates)}
                for update in updates:
                    clocks = update["optimizer_clocks"]
                    require(clocks["minimum"] == clocks["maximum"] == update["step"], "actual optimizer clocks")
                for update in comparable_updates:
                    require(math.isclose(update["loss"], audit["row_mean_losses_by_update"][str(update["step"])], rel_tol=1e-6, abs_tol=1e-6), "row-mean update loss")
            if "stored_checkpoints" in item:
                outcome["actual_checkpoint_audit"] = stored_checkpoint_audit(item, ctx)
                training = item["files"]["training.json"]["value"]
                require(training["arm"]["updates"] == 384 and training["arm"]["example_exposures"] == 1536, "completed update/exposure budget")
                require(len(updates) == 384 and audit["rows"] == 1536, "complete events and trajectory accounting")
                require(sum(audit["loss_tokens_per_family"].values()) == training["arm"]["loss_token_exposures"], "completed token exposure accounting")
                require(training["teacher_optimizer_updates"] == training["arm"]["teacher_optimizer_updates"] == 0, "fixed expert optimization")
                require(training["immutable_weights"]["teacher_adapter_sha256_before_and_after"] == ctx["teacher"]["teacher_tensor_sha256"], "recorded fixed teacher identity")
                for label in ("initial", "final"):
                    outcome[label + "_generic_retention"] = retention_audit(ctx, item["files"][label + "_retention.json"]["value"])
        elif "result.json" in item["files"] and "value" in item["files"]["result.json"]:
            receipt = item["files"]["result.json"]["value"]
            require(receipt["status"] == "completed" and not receipt["teacher_model_loaded"] and receipt["optimizer_updates"] == 0, "teacher absent final receipt")
            panels = item["files"]["panels.json"]
            require(panels["sha256"] == receipt["panels_sha256"], "final panel hash")
            outcome["native"] = {split: verify_generations(ctx, ctx["corpus"][split], panels["value"][split]) for split in ("test", "unused288")}
            outcome["new_process_proof"] = receipt["process_proof"]
            require(receipt["process_proof"]["distinct_task_ids"] and receipt["process_proof"]["distinct_attempt_ids"] and receipt["process_proof"]["standalone_cli"], "separate evaluation task evidence")
            training_id = Path(config["training_dir"]).name
            training_item = snapshot["runs"][training_id]
            training = training_item["files"]["training.json"]
            source_execution = training_item["files"]["execution.json"]["value"]
            require(source_execution["status"] == "completed" and source_execution["exit_code"] == 0 and not source_execution["timed_out"], "completed source training execution")
            require(receipt["training_sha256"] == training["sha256"], "evaluation training hash")
            require(task_id != training_id and execution["attempt_id"] != source_execution["attempt_id"], "actual distinct evaluation execution")
            require(receipt["process_proof"]["training_pid"] == training["value"]["pid"] and receipt["process_proof"]["evaluation_pid"] == receipt["pid"], "actual child process binding")
            outcome["generic_retention"] = {}
            for label in ("initial", "final"):
                require(item["files"][label + "_train_probes.json"]["value"] == training_item["files"][label + "_train_probes.json"]["value"], "saved native reload parity")
                require(panels["value"][label + "_retention"] == training_item["files"][label + "_retention.json"]["value"], "exact saved generic score vectors")
                outcome["generic_retention"][label] = retention_audit(ctx, panels["value"][label + "_retention"])
        result["runs"][task_id] = outcome
    return result


def checkpoint_audit(root, receipt):
    initial = receipt["initial"]
    final = receipt["arm"]["checkpoints"]["384"]["adapter"]
    hashes = []
    for label, spec in (("initial", initial), ("final", final)):
        relative = Path(spec["path"]).relative_to(Path("/mnt/shared/cl-portfolio/runs") / root.name)
        folder = root / relative
        if not folder.exists():
            return {"status": "awaiting_parent_checkpoint_collection"}
        for name, checksum in spec["files"].items():
            require(file_hash(folder / name) == checksum, "checkpoint file " + name)
        config = read(folder / "adapter_config.json")
        require(config["r"] == 8 and config["lora_alpha"] == 16 and set(config["target_modules"]) == {"q_proj", "v_proj"}, "fixed rank8 q/v adapter")
        tensors = load_file(str(folder / "adapter_model.safetensors"), device="cpu")
        require(all("lora_A" in name or "lora_B" in name for name in tensors), "adapter only no dense offsets")
        require(all(tensor.ndim == 2 and 8 in tensor.shape for tensor in tensors.values()), "all effective factors rank8")
        require(sum(tensor.numel() for tensor in tensors.values()) == 3833856, "fixed trainable capacity")
        checksum = hashlib.sha256()
        for name, tensor in sorted(tensors.items()):
            tensor = tensor.detach().cpu().contiguous()
            checksum.update(name.encode())
            checksum.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        require(checksum.hexdigest() == spec["tensor_sha256"], "actual stored adapter tensor hash")
        hashes.append({"label": label, "file_tensor_sha256": spec["files"]["adapter_model.safetensors"],
                       "resident_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
                       "rank": 8, "parameter_count": 3833856})
    require(initial["tensor_sha256"] != final["tensor_sha256"], "changed learner")
    return {"status": "passed", "checkpoints": hashes}


def collected_audit(ctx, root, wave):
    collection = read(root / "collection.json")
    require(collection["execution_status"] == "completed", "completed collection")
    for name, spec in collection["files"].items():
        path = root / name
        require(path.stat().st_size == spec["bytes"] and file_hash(path) == spec["sha256"], "collection file " + name)
    archive = wave / "code" / (SOURCE + ".tar")
    checksum = hashlib.sha256()
    archived = {}
    with tarfile.open(archive) as stream:
        for member in sorted(stream.getmembers(), key=lambda entry: Path(entry.name)):
            require(member.isfile(), "archived source member")
            payload = stream.extractfile(member).read()
            checksum.update(member.name.encode() + b"\0" + payload)
            archived[member.name] = hashlib.sha256(payload).hexdigest()
    require(checksum.hexdigest() == SOURCE and len(archived) == 130, "collected archive source identity")
    for name, expected in ctx["handoff"]["scientific_closure_sha256"].items():
        require(archived[name] == expected, "archived scientific source " + name)
    evidence = max((wave / "onpolicy303-independent-audit").glob("raw-*.json"))
    model_config = read(evidence)["model_config"]
    spec = next(row for row in ctx["choice"]["source_base"]["files"] if row["path"] == "config.json")
    require(model_config["bytes"] == spec["bytes"], "pinned model configuration bytes")
    for field in ("sha256", "git_blob_sha1"):
        if spec[field] is not None:
            require(model_config[field] == spec[field], "pinned model config " + field)
    ctx["vocabulary_size"] = model_config["value"]["vocab_size"]
    item = {"files": {name: {"value": read(root / name)} for name in ("execution.json", "task.json", "config.json")}}
    execution, config = verify_execution(item, root.name)
    require(execution["status"] == "completed" and execution["exit_code"] == 0 and not execution["timed_out"], "successful supervisor completion")
    training = read(root / "training.json")
    require(training["design_sha256"] == PROTOCOL and training["method"] == config["method"], "training protocol/method")
    require(training["initial"]["tensor_sha256"] == ctx["design"]["coverage_control"]["initial_tensor_sha256"], "exact fresh initialization")
    require(not training["learner_initialized_from_teacher"] and not training["heldout_predictions_observed_during_training"], "training boundaries")
    for name, expected in training["files_sha256"].items():
        require(file_hash(root / name) == expected, "training snapshot " + name)
    require(file_hash(root / "teacher_qualification.json") == ctx["design"]["teacher"]["qualification_sha256"], "copied teacher certificate")
    require(read(root / "schedule.json") == ctx["schedule"], "exact coverage row schedule")
    teacher_panels = {}
    for split in ("train", "validation"):
        fresh = read(root / ("teacher_fresh_" + split + ".json"))
        teacher_panels[split] = verify_generations(ctx, ctx["corpus"][split], fresh)
        require(teacher_panels[split]["correct"] == len(ctx["corpus"][split]), "fully qualified native teacher")
        require(all(a["generation"] == b["generation"] for a, b in zip(fresh, ctx["canonical"][split]["generated"], strict=True)), "exact fresh teacher generations")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["event"] == "onpolicy303_update"]
    require([event["step"] for event in updates] == list(range(1, 385)), "384 sequential optimizer updates")
    prior = events[:events.index(updates[0])]
    for split, count in (("train", 384), ("validation", 96)):
        rows = [event for event in prior if event.get("condition") == "teacher_fresh/" + split]
        require(len(rows) == count and all(row["correct"] for row in rows), "teacher qualification event order")
    arm = training["arm"]
    require(arm["updates"] == 384 and arm["example_exposures"] == 1536 and arm["selected_checkpoint"] == 384, "fixed budget")
    folder = root / config["method"]
    require(file_hash(folder / "ledger.json") == arm["ledger_sha256"] and file_hash(folder / "trajectories.jsonl") == arm["trajectories_sha256"], "actual ledger and trajectory hashes")
    trajectories = [json.loads(line) for line in (folder / "trajectories.jsonl").read_text().splitlines()]
    audit = trajectory_audit(ctx, trajectories, config["method"])
    prefix_panels = prefix_panel_audit(folder, trajectories, arm)
    require(audit["rows"] == 1536 and sum(audit["loss_tokens_per_family"].values()) == arm["loss_token_exposures"], "actual exposure and token counts")
    ledger = read(folder / "ledger.json")
    require(len(ledger) == 384, "ledger count")
    for i, (record, event) in enumerate(zip(ledger, updates, strict=True)):
        block = trajectories[i * 4:(i + 1) * 4]
        require(record["step"] == i + 1 and record["ids"] == ctx["schedule"][i], "ledger row order")
        require(record["trajectory_sha256"] == [digest(row) for row in block], "ledger raw trajectory binding")
        require(record["loss_tokens"] == sum(row["diagnostics"]["loss_tokens"] for row in block), "ledger token masks")
        require(all(event[name] == value for name, value in record.items()), "event/ledger identity")
        require(math.isclose(record["loss"], audit["row_mean_losses_by_update"][str(i + 1)], rel_tol=1e-6, abs_tol=1e-6), "four-row mean loss")
        require(record["optimizer_clocks"]["minimum"] == record["optimizer_clocks"]["maximum"] == i + 1, "actual per-update clocks")
    checkpoints = checkpoint_audit(root, training)
    require(checkpoints["status"] == "passed", "complete checkpoint collection")
    probes = [row for task in ("sequence_a", "sequence_b", "sequence_c") for row in [r for r in ctx["corpus"]["train"] if r["task"] == task][:8]]
    checkpoint_probes, optimizer_checks = {}, {}
    for step in (32, 128, 384):
        spec = arm["checkpoints"][str(step)]
        checkpoint_root = folder / f"checkpoint{step}"
        require(file_hash(checkpoint_root / "optimizer.pt") == spec["optimizer_sha256"], "saved optimizer file")
        require(file_hash(checkpoint_root / "train_probes.json") == spec["train_probes_sha256"], "saved native probes")
        state = torch.load(checkpoint_root / "optimizer.pt", map_location="cpu", weights_only=True)
        require(len(state["state"]) == 144, "optimizer state count")
        require(all(int(value["step"]) == step for value in state["state"].values()), "saved optimizer actual clocks")
        require(all(torch.isfinite(value["exp_avg"]).all() and torch.isfinite(value["exp_avg_sq"]).all() for value in state["state"].values()), "finite optimizer moments")
        group = state["param_groups"][0]
        require(group["lr"] == 0.0003 and group["weight_decay"] == 0 and group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8, "matched AdamW")
        optimizer_checks[str(step)] = {"states": len(state["state"]), "actual_clock": step, "finite_moments": True}
        checkpoint_probes[str(step)] = verify_generations(ctx, probes, read(checkpoint_root / "train_probes.json"))
    control = wave / "runs/consolidation-coverage303-20260912"
    require(file_hash(root / "teacher_train_cache.json") == file_hash(control / "teacher_train_cache.json"), "exact coverage teacher target artifact")
    final_equal = arm["checkpoints"]["384"]["adapter"]["tensor_sha256"] == ctx["design"]["coverage_control"]["final_tensor_sha256"]
    require(final_equal == training["coverage_sft_final_tensors_equal"], "control tensor comparison")
    optimizer_equal = None
    if config["method"] == "cached_teacher_sft":
        previous = torch.load(control / "teacher_output_sft/checkpoint384/optimizer.pt", map_location="cpu", weights_only=True)
        optimizer_equal = state["param_groups"] == previous["param_groups"] and all(torch.equal(value[name], previous["state"][index][name]) for index, value in state["state"].items() for name in value)
        require(final_equal and optimizer_equal, "SFT bitwise coverage control adapter and optimizer")
    require(training["teacher_optimizer_updates"] == arm["teacher_optimizer_updates"] == 0, "no teacher updates")
    if config["method"] != "cached_teacher_sft":
        require(arm["teacher_tensor_sha256_before"] == arm["teacher_tensor_sha256_after"] == ctx["teacher"]["teacher_tensor_sha256"], "recorded fixed teacher tensor hashes")
    audit.pop("row_mean_losses_by_update")
    return {"status": "passed_collected_training_audit", "task_id": root.name, "method": config["method"],
            "source_sha256": SOURCE, "source_archive_sha256": file_hash(archive), "source_files_checked": 130,
            "scientific_closure_files_checked": 25, "collection_files_checked": len(collection["files"]),
            "collection_sha256": file_hash(root / "collection.json"), "execution_sha256": file_hash(root / "execution.json"),
            "training_sha256": file_hash(root / "training.json"), "native_teacher_before_first_update": teacher_panels,
            "trajectories": audit, "checkpoint_capacity_and_bytes": checkpoints, "optimizer_checkpoints": optimizer_checks,
            "complete_prefix_sampling_diagnostics": prefix_panels,
            "native_train_probes": checkpoint_probes, "coverage_sft_final_tensors_equal": final_equal,
            "coverage_sft_final_optimizer_tensors_equal": optimizer_equal,
            "generic_retention": {label: retention_audit(ctx, read(root / (label + "_retention.json"))) for label in ("initial", "final")},
            "fixed_teacher_and_base_immutability": {"recorded_training_assertions": training["immutable_weights"], "independent_process_memory_remeasurement": False},
            "fresh_process_native_heldout_evaluation_audited": False,
            "limitations": "Completed training artifact and independent tokenizer rescore audit. No repeated GPU inference or independent access to live teacher weights. Native final heldout persistence awaits separate queued evaluation. Generic retention remains multiple-choice."}


def evaluation_audit(ctx, root, wave):
    collection = read(root / "collection.json")
    require(collection["execution_status"] == "completed", "completed evaluation collection")
    for name, spec in collection["files"].items():
        require((root / name).stat().st_size == spec["bytes"] and file_hash(root / name) == spec["sha256"], "evaluation collection file " + name)
    item = {"files": {name: {"value": read(root / name)} for name in ("execution.json", "task.json", "config.json")}}
    execution, config = verify_execution(item, root.name)
    require(execution["status"] == "completed" and execution["exit_code"] == 0 and execution["timed_out"] is False, "successful evaluation execution")
    receipt = read(root / "result.json")
    training_root = wave / "runs" / Path(config["training_dir"]).name
    training = read(training_root / "training.json")
    previous_execution = read(training_root / "execution.json")
    require(previous_execution["status"] == "completed" and previous_execution["exit_code"] == 0 and previous_execution["timed_out"] is False, "completed training before evaluation")
    require(receipt["training_sha256"] == file_hash(training_root / "training.json"), "exact training receipt")
    require(receipt["status"] == "completed" and receipt["method"] == config["method"] and receipt["design_sha256"] == PROTOCOL, "evaluation receipt identity")
    require(receipt["source_sha256"] == training["source_sha256"] == ctx["design"]["scientific_files_sha256"], "same scientific source hashes")
    require(receipt["source_execution"]["execution_sha256"] == file_hash(training_root / "execution.json"), "source completed receipt hash")
    require(root.name != training_root.name and execution["attempt_id"] != previous_execution["attempt_id"], "distinct execution identities")
    process = receipt["process_proof"]
    require(process["distinct_task_ids"] and process["distinct_attempt_ids"] and process["standalone_cli"], "standalone execution proof")
    require(process["training_pid"] == training["pid"] and process["evaluation_pid"] == receipt["pid"], "actual process IDs bound to receipts")
    require(receipt["new_process_persistence_evaluation_completed"] and receipt["initial_and_final_exact_reload_parity"], "reported completed reload validation")
    require(not receipt["teacher_model_loaded"] and not receipt["teacher_artifact_files_read_during_evaluation"] and receipt["optimizer_updates"] == 0, "teacher-absent zero-update evaluator")
    require(receipt["model_guard"]["dtype"] == "float32" and receipt["model_guard"]["devices"] == ["cuda:0"] and not receipt["model_guard"]["offload_hooks"], "qualified evaluator numerical path")
    require(file_hash(root / "panels.json") == receipt["panels_sha256"], "heldout panel hash")
    panels = read(root / "panels.json")
    native = {split: verify_generations(ctx, ctx["corpus"][split], panels[split]) for split in ("test", "unused288")}
    for split, audited in native.items():
        for task, scores in audited["families"].items():
            reported = receipt["native"][split][task]
            require(scores["count"] == reported["count"] and scores["correct"] / scores["count"] == reported["correct"], "reported native family score")
    probes = [row for task in ("sequence_a", "sequence_b", "sequence_c") for row in [r for r in ctx["corpus"]["train"] if r["task"] == task][:8]]
    reloads, generic = {}, {}
    for label in ("initial", "final"):
        source = training_root / ("initial_train_probes.json" if label == "initial" else config["method"] + "/checkpoint384/train_probes.json")
        generated = read(root / (label + "_train_probes.json"))
        require(generated == read(source), "exact saved native TRAIN reload")
        reloads[label] = verify_generations(ctx, probes, generated)
        generic[label] = retention_audit(ctx, panels[label + "_retention"])
        require(panels[label + "_retention"] == read(training_root / (label + "_retention.json")) == read(root / (label + "_retention.json")), "exact saved generic score-vector reload")
        for task, values in generic[label]["families"].items():
            require(values == receipt["generic_retention"][label][task], "reported generic family score")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    require(all(row["pid"] == receipt["pid"] for row in events), "evaluation event process identity")
    require(not any(row["event"] == "onpolicy303_update" for row in events), "no evaluation updates")
    index = next(i for i, row in enumerate(events) if row.get("condition") == "final/test")
    for label in ("initial", "final"):
        require(sum(row.get("condition") == "reload/" + label + "/train_only" for row in events[:index]) == 24, "reload before heldout event order")
    old_panels = read(wave / "runs/consolidation-coverage303-20260912/evaluation/panels.json")
    if config["method"] == "cached_teacher_sft":
        require(all(panels[split] == old_panels[split]["teacher_output_sft"] for split in ("test", "unused288")), "exact coverage SFT generated records")
    return {"status": "passed_collected_fresh_process_evaluation", "task_id": root.name, "method": config["method"],
            "collection_files_checked": len(collection["files"]), "collection_sha256": file_hash(root / "collection.json"),
            "execution_sha256": file_hash(root / "execution.json"), "result_sha256": file_hash(root / "result.json"),
            "panels_sha256": file_hash(root / "panels.json"), "source_sha256": SOURCE,
            "native": native, "native_reload_panels": reloads, "generic_retention": generic,
            "process_proof": process, "teacher_absent": True, "optimizer_updates": 0,
            "teacher_absence_evidence": "Completed distinct task/attempt, frozen evaluator code, learner-only checkpoint load assertions and exact saved behavior. Independent audit does not rerun model inference or inspect arbitrary process memory."}


def paired_native(rows, before, after):
    result = {"gained": [], "lost": [], "still_wrong": [], "unchanged_correct": 0}
    for row, left, right in zip(rows, before, after, strict=True):
        require(row["id"] == left["id"] == right["id"], "paired native row IDs")
        a, b = left["generation"]["correct"], right["generation"]["correct"]
        if a and b:
            result["unchanged_correct"] += 1
            continue
        kind = "gained" if b else "lost" if a else "still_wrong"
        result[kind].append({"id": row["id"], "task": row["task"], "input": row["group"],
                             "gold": row["choices"][row["gold_idx"]],
                             "before": {key: left["generation"][key] for key in ("body_text", "token_ids", "correct", "format_valid", "terminated", "digit_position_correct")},
                             "after": {key: right["generation"][key] for key in ("body_text", "token_ids", "correct", "format_valid", "terminated", "digit_position_correct")}})
    result["counts"] = {key: len(result[key]) for key in ("gained", "lost", "still_wrong")}
    result["counts"]["net_gain"] = len(result["gained"]) - len(result["lost"])
    result["per_family"] = {task: {kind: sum(row["task"] == task for row in result[kind]) for kind in ("gained", "lost", "still_wrong")} for task in ("sequence_a", "sequence_b", "sequence_c")}
    return result


def paired_generic(rows, before, after):
    gained, lost = [], []
    for index, (row, left, right) in enumerate(zip(rows, before, after, strict=True)):
        require(digest(row) == left["row_sha256"] == right["row_sha256"], "paired generic row hashes")
        if left["correct"] == right["correct"]:
            continue
        record = {"row_index": index, "task": row["task"], "row_sha256": digest(row),
                  "prompt_excerpt": row["prompt"][:180], "choices": row["choices"], "gold_idx": row["gold_idx"],
                  "before_prediction": left["prediction"], "after_prediction": right["prediction"],
                  "before_scores": left["scores"], "after_scores": right["scores"]}
        (gained if right["correct"] else lost).append(record)
    return {"gained": gained, "lost": lost, "counts": {"gained": len(gained), "lost": len(lost), "net_gain": len(gained) - len(lost)},
            "per_family": {task: {"gained": sum(row["task"] == task for row in gained), "lost": sum(row["task"] == task for row in lost)} for task in sorted({row["task"] for row in rows})}}


def final_audit(ctx, wave):
    methods = ("cached_teacher_sft", "cached_teacher_kl", "onpolicy_kl")
    training, evaluation, panels = {}, {}, {}
    for method in methods:
        name = "followthrough-20260912-onpolicy303-" + method.replace("_", "-")
        training[method] = collected_audit(ctx, wave / "runs" / (name + "-train"), wave)
        evaluation[method] = evaluation_audit(ctx, wave / "runs" / (name + "-evaluate"), wave)
        panels[method] = read(wave / "runs" / (name + "-evaluate") / "panels.json")
    for method in methods[1:]:
        require(panels[method]["initial_retention"] == panels[methods[0]]["initial_retention"], "identical initial generic scores across independent jobs")
    comparisons = {}
    for left, right in (("cached_teacher_sft", "cached_teacher_kl"), ("cached_teacher_sft", "onpolicy_kl"), ("cached_teacher_kl", "onpolicy_kl")):
        comparisons[right + "_versus_" + left] = {split: paired_native(ctx["corpus"][split], panels[left][split], panels[right][split]) for split in ("test", "unused288")}
    rows = read(Path(ctx["handoff"]["ownership"]["canonical_repository"]) / "experiments/distillation/configs/onpolicy303_retention.json")["validation"]
    old_generic = {method: paired_generic(rows, panels[method]["initial_retention"], panels[method]["final_retention"]) for method in methods}
    return {"status": "all_six_completed_runs_independently_audited", "protocol_digest": PROTOCOL, "source_sha256": SOURCE,
            "training": training, "evaluation": evaluation, "paired_native_outcomes": comparisons, "paired_generic_before_after": old_generic,
            "scope": "One fixed-expert full-vocabulary forward-KL hybrid, one Sequence303 mapping and learner seed, reused historical heldouts. Frozen rank8 q/v learner, no native-base weight updates. Actual on-policy KL updates and teacher-absent persistence demonstrated; not full SDFT/SDPO, general continual learning, composition or a superiority guarantee across tasks/seeds.",
            "exposure_boundary": {"updates_per_arm": 384, "row_exposures_per_arm": 1536, "cached_loss_tokens_per_arm": 10752,
                                  "onpolicy_loss_tokens": 11314, "additional_sampled_loss_tokens": 562,
                                  "all_arms_equal_token_or_compute_budget": False,
                                  "interpretation": "Cached KL versus cached SFT holds target trajectories and token exposures fixed. On-policy versus cached KL changes prefix distribution and sampled response lengths; this comparison does not isolate prefix sampling at equal token/compute exposure."},
            "recipe_changes": [], "gpu_or_provider_mutations_by_auditor": []}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--wave", type=Path, required=True)
    parser.add_argument("--control-pod", default="cl-portfolio-control-20260912")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--collected-task", action="append")
    parser.add_argument("--final-audit", action="store_true")
    args = parser.parse_args()
    ctx = context(args.repo, args.wave)
    output = args.wave / "onpolicy303-independent-audit"
    output.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.final_audit:
        report = final_audit(ctx, args.wave)
        report["audit_source_sha256"] = file_hash(Path(__file__))
        report["independent_scorer_source_sha256"] = file_hash(Path(__file__).with_name("audit_consolidation.py"))
        path = output / f"final-audit-{stamp}.json"
        write(path, report)
        print(json.dumps({"audit": str(path), "status": report["status"],
                          "native": {method: value["native"] for method, value in report["evaluation"].items()},
                          "paired_native": {name: {split: rows["counts"] for split, rows in comparison.items()} for name, comparison in report["paired_native_outcomes"].items()},
                          "paired_generic": {method: value["counts"] for method, value in report["paired_generic_before_after"].items()}}, indent=2))
        return
    if args.collected_task:
        for task_id in args.collected_task:
            report = collected_audit(ctx, args.wave / "runs" / task_id, args.wave)
            report["audit_source_sha256"] = file_hash(Path(__file__))
            report["independent_scorer_source_sha256"] = file_hash(Path(__file__).with_name("audit_consolidation.py"))
            path = output / f"collected-{task_id}-{stamp}.json"
            write(path, report)
            print(json.dumps({"audit": str(path), "status": report["status"], "teacher": report["native_teacher_before_first_update"],
                              "trajectory_rows": report["trajectories"]["rows"], "optimizer_clocks": report["optimizer_checkpoints"],
                              "coverage_sft_final_tensors_equal": report["coverage_sft_final_tensors_equal"],
                              "coverage_sft_final_optimizer_tensors_equal": report["coverage_sft_final_optimizer_tensors_equal"],
                              "native_train_probes": report["native_train_probes"], "generic_retention": report["generic_retention"]}, indent=2))
        return
    if args.snapshot:
        snapshot = read(args.snapshot)
        snapshot_path = args.snapshot
    else:
        process = subprocess.run(["kubectl", "--context", "us-mi355x-nambiar-k8s", "-n", "default", "exec", args.control_pod,
                                  "--", "uv", "run", "--no-project", "python", "-B", "-c", REMOTE], capture_output=True, text=True, check=True)
        snapshot = json.loads(process.stdout)
        snapshot_path = output / f"raw-{stamp}.json"
        write(snapshot_path, snapshot)
    report = live_report(ctx, snapshot)
    report["raw_snapshot"] = str(snapshot_path)
    report["raw_snapshot_sha256"] = file_hash(snapshot_path)
    report["audit_source_sha256"] = file_hash(Path(__file__))
    report["independent_scorer_source_sha256"] = file_hash(Path(__file__).with_name("audit_consolidation.py"))
    for task_id, result in report["runs"].items():
        local = args.wave / "runs" / task_id
        if task_id.endswith("-train") and (local / "training.json").exists():
            result["collected_checkpoint_audit"] = checkpoint_audit(local, read(local / "training.json"))
    path = output / f"audit-{stamp}.json"
    write(path, report)
    print(json.dumps({"audit": str(path), "snapshot": str(snapshot_path), "observed_at": report["observed_at"],
                      "runs": {task_id: {key: value for key, value in outcome.items() if key not in {"last_event", "trajectories", "teacher_generation_events"}}
                               for task_id, outcome in report["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
