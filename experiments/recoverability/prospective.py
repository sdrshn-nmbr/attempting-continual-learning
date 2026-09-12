import argparse
import logging
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from model import runtime_info
from prospective_analysis import (
    accuracy,
    eligible,
    evaluate_predictors,
    feature_vector,
    fit_predictors,
    learned,
    per_intent,
    repair_feasibility,
    stream_summary,
)
from prospective_data import balanced_batches, load_design, repair_batches, rows_for, seed_for, validate_data
from prospective_model import ProspectiveModel, parameter_hashes
from protocol import ROOT, digest, file_hash, read_json, write_json
from tuned_lens import cache_activations, evaluate_translations, train_lens

LEARNING_SOURCE_FILES = (
    "prospective.py",
    "prospective_data.py",
    "prospective_model.py",
    "prospective_analysis.py",
    "model.py",
    "protocol.py",
)


def learning_implementation():
    return {name: file_hash(ROOT / name) for name in LEARNING_SOURCE_FILES}


def learning_identity(config, spec, data):
    return digest(
        {
            "settings": {
                key: config[key] for key in ("model_id", "revision", "acquisition_updates", "forgetting_updates")
            },
            "protocol": spec,
            "dataset": digest(data),
            "implementation": learning_implementation(),
        }
    )


def measurement_identity(config, spec, data):
    return digest({"learning": learning_identity(config, spec, data), "repair_updates": config["repair_updates"]})


def make_seal(config, spec, data, stage):
    payload = {
        "config": config,
        "protocol": spec,
        "dataset_sha256": digest(data),
        "stage": stage,
        "learning_identity": learning_identity(config, spec, data),
        "learning_implementation": learning_implementation(),
        "measurement_identity": measurement_identity(config, spec, data),
        "runtime": runtime_info(),
        "implementation": {path.name: file_hash(path) for path in sorted(ROOT.glob("*.py"))},
    }
    return {"payload": payload, "sha256": digest(payload)}


def finish(output, payload):
    files = {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"payload": payload | {"files": files}, "sha256": digest(payload | {"files": files})}
    write_json(output / "receipt.json", result)
    return result["payload"]


def verify_result(output):
    output = Path(output).resolve()
    result = read_json(output / "receipt.json")
    payload = result["payload"]
    if digest(payload) != result["sha256"]:
        raise ValueError("PROSPECTIVE_RECEIPT_DIGEST")
    actual = {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file() and path.name != "receipt.json"
    }
    if actual != set(payload["files"]):
        raise ValueError("PROSPECTIVE_RECEIPT_FILE_SET")
    for name, expected in payload["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"PROSPECTIVE_ARTIFACT_HASH {output / name}")
    sealed = read_json(output / "seal.json")
    if digest(sealed["payload"]) != sealed["sha256"]:
        raise ValueError("PROSPECTIVE_SEAL_DIGEST")
    return payload, sealed["payload"]


def verify_model(config, data):
    model_path = Path(config["model_path"])
    for item in data["base_manifest"]:
        path = model_path / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"PROSPECTIVE_MODEL_IDENTITY {path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(16)] != [
        [code] for code in data["codes"]
    ]:
        raise ValueError("PROSPECTIVE_RUNTIME_CODE_TOKENIZATION")
    rows = data["lens_fit"] + data["lens_check"]
    for stream in data["streams"].values():
        rows += [row for unit in stream["units"].values() for values in unit.values() for row in values]
    for row in rows:
        if tokenizer.encode(data["prompt"].format(text=row["text"]), add_special_tokens=True) != row["input_ids"]:
            raise ValueError(f"PROSPECTIVE_RUNTIME_PROMPT_TOKENIZATION {row['text_sha256']}")
    if tokenizer.pad_token_id != data["pad_token_id"]:
        raise ValueError("PROSPECTIVE_RUNTIME_PADDING_ID")


def run_learning(observer, config, spec, stream, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    optimizer = observer.optimizer()
    records, checkpoints = {}, {}
    old_gate = rows_for(stream, stream["old"], "gate")
    new_gate = rows_for(stream, stream["new"], "gate")
    old_train = rows_for(stream, stream["old"], "learn")
    checkpoints["initial"] = observer.checkpoint(output / "initial", optimizer)
    records["initial"] = {"old_gate": observer.observe(old_gate), "new_gate": observer.observe(new_gate)}
    batches = balanced_batches(
        old_train,
        config["acquisition_updates"],
        spec["optimizer"]["batch_size"],
        seed_for(stream["seed"], "acquisition"),
    )
    trace = observer.update(old_train, batches, optimizer)
    write_json(output / "acquisition-updates.json", trace)
    records["acquired"] = {
        "old_gate": observer.observe(old_gate),
        "new_gate": observer.observe(new_gate),
        "train": observer.observe(old_train),
    }
    checkpoints["acquired"] = observer.checkpoint(output / "acquired", optimizer)
    observer.reset(output / "acquired")
    if observer.observe(old_gate) != records["acquired"]["old_gate"]:
        raise ValueError("PROSPECTIVE_ACQUISITION_RELOAD")
    acquired_scores = per_intent(records["acquired"]["old_gate"])
    initial_scores = per_intent(records["initial"]["old_gate"])
    acquired = learned(initial_scores, acquired_scores, spec)
    before_steps = observer.optimizer_steps(optimizer)
    if before_steps != [config["acquisition_updates"]]:
        raise ValueError("PROSPECTIVE_ACQUISITION_OPTIMIZER_CLOCK")
    qualification = {
        "acquired": acquired,
        "old_validation": acquired_scores,
        "initial_old_validation": initial_scores,
        "old_validation_gain": {intent: acquired_scores[intent] - initial_scores[intent] for intent in acquired_scores},
        "old_train": per_intent(records["acquired"]["train"]),
        "acquisition_passed": len(acquired) >= spec["gates"]["minimum_acquired_per_stream"],
    }
    write_json(output / "acquisition-gate.json", qualification)
    status = "acquisition_failed"
    forgotten, new_acquired = [], []
    after_steps = before_steps
    if qualification["acquisition_passed"]:
        logging.info("PROSPECTIVE_ACQUISITION_PASSED stream=%s intents=%d", stream["id"], len(acquired))
        new_train = rows_for(stream, stream["new"], "learn")
        batches = balanced_batches(
            new_train,
            config["forgetting_updates"],
            spec["optimizer"]["batch_size"],
            seed_for(stream["seed"], "forgetting"),
        )
        midpoint = len(batches) // 2
        early = observer.update(new_train, batches[:midpoint], optimizer)
        records["forgetting_midpoint"] = {
            "old_gate": observer.observe(old_gate),
            "new_gate": observer.observe(new_gate),
        }
        late = observer.update(new_train, batches[midpoint:], optimizer)
        for record in late:
            record["step"] += midpoint
        write_json(output / "forgetting-updates.json", early + late)
        records["forgotten"] = {
            "old_gate": observer.observe(old_gate),
            "new_gate": observer.observe(new_gate),
            "train": observer.observe(new_train),
        }
        checkpoints["forgotten"] = observer.checkpoint(output / "forgotten", optimizer)
        observer.reset(output / "forgotten")
        if observer.observe(old_gate) != records["forgotten"]["old_gate"]:
            raise ValueError("PROSPECTIVE_FORGETTING_RELOAD")
        after_steps = observer.optimizer_steps(optimizer)
        if after_steps != [config["acquisition_updates"] + config["forgetting_updates"]]:
            raise ValueError("PROSPECTIVE_CONTINUOUS_OPTIMIZER_CLOCK")
        new_acquired = learned(
            per_intent(records["acquired"]["new_gate"]), per_intent(records["forgotten"]["new_gate"]), spec
        )
        forgotten = eligible(initial_scores, acquired_scores, per_intent(records["forgotten"]["old_gate"]), spec)
        status = (
            "qualified"
            if (
                len(new_acquired) >= spec["gates"]["minimum_new_acquired_per_stream"]
                and len(forgotten) >= spec["gates"]["minimum_forgotten_per_stream"]
            )
            else "forgetting_failed"
        )
    else:
        logging.warning(
            "PROSPECTIVE_ACQUISITION_FAILED stream=%s acquired=%s; no forgetting or repairs", stream["id"], acquired
        )
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("PROSPECTIVE_FROZEN_BACKBONE_CHANGED")
    write_json(output / "observations.json", records)
    write_json(output / "checkpoints.json", checkpoints)
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "stage": "learn",
        "status": status,
        "stream": stream["id"],
        "split": stream["split"],
        "seed": stream["seed"],
        "acquired": acquired,
        "new_acquired": new_acquired,
        "eligible": forgotten,
        "acquisition_updates": config["acquisition_updates"],
        "forgetting_updates": config["forgetting_updates"] if qualification["acquisition_passed"] else 0,
        "repair_updates": 0,
        "test_rows_evaluated": 0,
        "lens_updates": 0,
        "optimizer_steps": {"acquired": before_steps, "final": after_steps},
        "frozen_backbone_unchanged": True,
        "next_config_if_failed": "configs/prospective-acquisition256.json"
        if not qualification["acquisition_passed"]
        else "configs/prospective-forgetting256.json",
        "decision": "await_global_cohort_gate" if status == "qualified" else "stop_before_repairs",
    }


def qualify_learning(sources, config, spec, data):
    by_stream = {}
    for path in sources:
        result, sealed = verify_result(path)
        if (
            result["stage"] != "learn"
            or sealed["learning_identity"] != learning_identity(config, spec, data)
            or sealed["learning_implementation"] != learning_implementation()
            or {name: sealed["implementation"][name] for name in LEARNING_SOURCE_FILES} != learning_implementation()
        ):
            raise ValueError("PROSPECTIVE_LEARNING_SOURCE_BINDING")
        stream = data["streams"][result["stream"]]
        if stream["id"] in by_stream:
            raise ValueError("PROSPECTIVE_DUPLICATE_LEARNING_STREAM")
        observations = read_json(Path(path) / "observations.json")
        for phase in observations.values():
            for key, intents in (("old_gate", stream["old"]), ("new_gate", stream["new"])):
                assert_bound_records(phase[key], rows_for(stream, intents, "gate"))
        before = per_intent(observations["acquired"]["old_gate"])
        initial = per_intent(observations["initial"]["old_gate"])
        acquired = learned(initial, before, spec)
        forgotten, new_acquired = [], []
        if "forgotten" in observations:
            forgotten = eligible(initial, before, per_intent(observations["forgotten"]["old_gate"]), spec)
            new_acquired = learned(
                per_intent(observations["acquired"]["new_gate"]),
                per_intent(observations["forgotten"]["new_gate"]),
                spec,
            )
        if (
            acquired != result["acquired"]
            or forgotten != result["eligible"]
            or new_acquired != result["new_acquired"]
            or result["repair_updates"] != 0
            or result["test_rows_evaluated"] != 0
        ):
            raise ValueError("PROSPECTIVE_LEARNING_RECOMPUTATION")
        passed = (
            len(acquired) >= spec["gates"]["minimum_acquired_per_stream"]
            and len(new_acquired) >= spec["gates"]["minimum_new_acquired_per_stream"]
            and len(forgotten) >= spec["gates"]["minimum_forgotten_per_stream"]
        )
        by_stream[stream["id"]] = {
            "source": str(Path(path).resolve()),
            "receipt_sha256": file_hash(Path(path) / "receipt.json"),
            "passed": passed,
            "eligible": forgotten,
            "split": stream["split"],
        }
    if set(by_stream) != set(data["streams"]):
        raise ValueError("PROSPECTIVE_ALL_PREDECLARED_STREAMS_REQUIRED")
    counts = {
        split: sum(len(value["eligible"]) for value in by_stream.values() if value["split"] == split)
        for split in ("train", "test")
    }
    clusters = {
        split: sum(value["passed"] and value["split"] == split for value in by_stream.values()) for split in counts
    }
    passed = (
        all(value["passed"] for value in by_stream.values())
        and all(counts[split] >= spec["gates"]["minimum_forgotten"][split] for split in counts)
        and min(clusters.values()) >= spec["gates"]["minimum_streams_per_split"]
    )
    return {
        "stage": "qualify",
        "qualified": passed,
        "streams": by_stream,
        "counts": counts,
        "clusters": clusters,
        "decision": "train_actions_allowed" if passed else "stop_before_any_lens_or_repair",
    }


def checked_qualification(path, config, spec, data):
    result, sealed = verify_result(path)
    if (
        result["stage"] != "qualify"
        or not result["qualified"]
        or sealed["measurement_identity"] != measurement_identity(config, spec, data)
    ):
        raise ValueError("PROSPECTIVE_GLOBAL_ELIGIBILITY_REQUIRED")
    recomputed = qualify_learning([item["source"] for item in result["streams"].values()], config, spec, data)
    if recomputed != {key: value for key, value in result.items() if key != "files"}:
        raise ValueError("PROSPECTIVE_QUALIFICATION_RECOMPUTATION")
    return result


def assert_bound_records(records, rows):
    if len(records) != len(rows) or any(
        (record["text_sha256"], record["intent"], record["target"])
        != (row["text_sha256"], row["intent"], row["target"])
        for record, row in zip(records, rows, strict=True)
    ):
        raise ValueError("PROSPECTIVE_OBSERVATION_ROW_BINDING")


def collect_units(sources, config, spec, data, split, qualification):
    units, seen_streams = [], set()
    for path in sources:
        result, sealed = verify_result(path)
        if (
            result["stage"] != "measure"
            or result["split"] != split
            or sealed["measurement_identity"] != measurement_identity(config, spec, data)
            or result["qualification_sha256"] != file_hash(Path(qualification) / "receipt.json")
        ):
            raise ValueError("PROSPECTIVE_MEASUREMENT_SOURCE_BINDING")
        if result["stream"] in seen_streams:
            raise ValueError("PROSPECTIVE_DUPLICATE_MEASURED_STREAM")
        seen_streams.add(result["stream"])
        measured = read_json(Path(path) / "units.json")
        if [unit["id"] for unit in measured] != result["unit_ids"]:
            raise ValueError("PROSPECTIVE_MEASURED_UNIT_BINDING")
        stream = data["streams"][result["stream"]]
        expected = read_json(Path(qualification) / "qualification.json")["streams"][stream["id"]]["eligible"]
        if [unit["intent"] for unit in measured] != expected:
            raise ValueError("PROSPECTIVE_ELIGIBLE_UNIT_SET")
        for unit in measured:
            if unit["split"] != split or unit["stream"] != stream["id"]:
                raise ValueError("PROSPECTIVE_PREDICTION_SPLIT")
            for phase in ("before", "after"):
                assert_bound_records(unit["baseline"][phase]["test"], stream["units"][unit["intent"]]["test"])
                assert_bound_records(unit["baseline"][phase]["guard"], rows_for(stream, stream["new"], "test"))
                assert_bound_records(unit["probes"][phase], stream["units"][unit["intent"]]["probe"])
            if unit["features"] != feature_vector(unit["probes"]["before"], unit["probes"]["after"], spec["layers"]):
                raise ValueError("PROSPECTIVE_FEATURE_RECOMPUTATION")
            for action in spec["repair"]["actions"]:
                assert_bound_records(unit["actions"][action]["test"], stream["units"][unit["intent"]]["test"])
                assert_bound_records(unit["actions"][action]["guard"], rows_for(stream, stream["new"], "test"))
                if action in {"none", "restore"}:
                    baseline = unit["baseline"]["before" if action == "restore" else "after"]
                    if any(unit["actions"][action][key] != baseline[key] for key in ("test", "guard")):
                        raise ValueError("PROSPECTIVE_ACTION_CONTROL_RECOMPUTATION")
        units.extend(measured)
    if seen_streams != {name for name, stream in data["streams"].items() if stream["split"] == split}:
        raise ValueError("PROSPECTIVE_ALL_SPLIT_STREAMS_REQUIRED")
    return units


def checked_repair_qualification(path, config, spec, data, qualification):
    result, sealed = verify_result(path)
    if (
        result["stage"] != "qualify-repairs"
        or not result["qualified"]
        or sealed["measurement_identity"] != measurement_identity(config, spec, data)
        or result["qualification_sha256"] != file_hash(Path(qualification) / "receipt.json")
    ):
        raise ValueError("PROSPECTIVE_TRAIN_REPAIR_SIGNAL_REQUIRED")
    units = collect_units(result["sources"], config, spec, data, "train", qualification)
    if repair_feasibility(units, spec) != result["feasibility"]:
        raise ValueError("PROSPECTIVE_REPAIR_SIGNAL_RECOMPUTATION")
    if fit_predictors(units, spec) != read_json(Path(path) / "predictors.json"):
        raise ValueError("PROSPECTIVE_PREDICTOR_RECOMPUTATION")
    return result


def checked_pilot_qualification(path, config, spec, data, qualification):
    result, sealed = verify_result(path)
    if (
        result["stage"] != "pilot-qualify"
        or not result["qualified"]
        or sealed["measurement_identity"] != measurement_identity(config, spec, data)
        or result["qualification_sha256"] != file_hash(Path(qualification) / "receipt.json")
    ):
        raise ValueError("PROSPECTIVE_SMALL_REPAIR_PILOT_REQUIRED")
    units, streams = [], set()
    cohort = read_json(Path(qualification) / "qualification.json")
    for source in result["sources"]:
        receipt, source_seal = verify_result(source)
        if (
            receipt["stage"] != "pilot-measure"
            or receipt["split"] != "train"
            or source_seal["measurement_identity"] != sealed["measurement_identity"]
            or receipt["qualification_sha256"] != result["qualification_sha256"]
        ):
            raise ValueError("PROSPECTIVE_PILOT_SOURCE_BINDING")
        stream = data["streams"][receipt["stream"]]
        if stream["split"] != "train" or stream["id"] in streams:
            raise ValueError("PROSPECTIVE_PILOT_STREAM_SPLIT")
        streams.add(stream["id"])
        measured = read_json(Path(source) / "units.json")
        if [unit["intent"] for unit in measured] != cohort["streams"][stream["id"]]["eligible"][:2]:
            raise ValueError("PROSPECTIVE_PILOT_SUBSET_CHANGED")
        for unit in measured:
            if unit["split"] != "train" or unit["stream"] != stream["id"]:
                raise ValueError("PROSPECTIVE_PILOT_UNIT_BINDING")
            for observation in (*unit["baseline"].values(), *unit["actions"].values()):
                assert_bound_records(observation["test"], stream["units"][unit["intent"]]["test"])
                assert_bound_records(observation["guard"], rows_for(stream, stream["new"], "test"))
        units.extend(measured)
    if streams != {name for name, stream in data["streams"].items() if stream["split"] == "train"}:
        raise ValueError("PROSPECTIVE_PILOT_BOTH_TRAIN_STREAMS_REQUIRED")
    if repair_feasibility(units, spec) != result["feasibility"]:
        raise ValueError("PROSPECTIVE_PILOT_SIGNAL_RECOMPUTATION")
    return result


def perform_actions(
    observer, config, spec, stream, intent, before_checkpoint, after_checkpoint, baseline, output, actions=None
):
    guards = rows_for(stream, stream["new"], "test")
    new_repair = rows_for(stream, stream["new"], "repair")
    repair_rows = stream["units"][intent]["repair"]
    repair_spec = {**spec, "repair": {**spec["repair"], "updates": config["repair_updates"]}}
    mixed_rows, mixed_batches = repair_batches(
        repair_rows, new_repair, repair_spec, seed_for(stream["seed"], intent, "repair")
    )
    target_batches = balanced_batches(
        repair_rows,
        config["repair_updates"],
        spec["repair"]["batch_size"],
        seed_for(stream["seed"], intent, "repair_target"),
    )
    outcomes = {}
    for action in spec["repair"]["actions"] if actions is None else actions:
        observer.reset(before_checkpoint if action == "restore" else after_checkpoint)
        trace = []
        if action in {"replay_target", "replay_balanced", "sham"}:
            rows, batches = (repair_rows, target_batches) if action == "replay_target" else (mixed_rows, mixed_batches)
            optimizer = observer.optimizer()
            trace = observer.update(rows, batches, optimizer, sham_intent=intent if action == "sham" else None)
            if observer.optimizer_steps(optimizer) != [config["repair_updates"]]:
                raise ValueError("PROSPECTIVE_REPAIR_OPTIMIZER_CLOCK")
        else:
            rows, batches = [], []
        observed = {"test": observer.observe(stream["units"][intent]["test"]), "guard": observer.observe(guards)}
        expected = baseline["before" if action == "restore" else "after"]
        if action in {"none", "restore"} and observed != expected:
            raise ValueError(f"PROSPECTIVE_{action.upper()}_IDENTITY")
        checkpoint = output / "actions" / intent / action
        observer.checkpoint(checkpoint)
        observer.reset(checkpoint)
        reloaded = {"test": observer.observe(stream["units"][intent]["test"]), "guard": observer.observe(guards)}
        if reloaded != observed:
            raise ValueError("PROSPECTIVE_REPAIR_RELOAD_BEHAVIOR")
        history = {"rows": rows, "updates": trace}
        write_json(output / "actions" / intent / f"{action}-updates.json", history)
        outcomes[action] = {
            **observed,
            "updates": len(trace),
            "total_exposures": sum(len(batch) for batch in batches),
            "old_exposures": sum(rows[index]["intent"] == intent for batch in batches for index in batch),
            "new_exposures": sum(rows[index]["intent"] != intent for batch in batches for index in batch),
            "schedule_sha256": digest({"rows": rows, "batches": batches}),
            "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
            "reload_exact": True,
        }
        logging.info(
            "PROSPECTIVE_REPAIR_COMPLETE stream=%s intent=%s action=%s accuracy=%.3f guard=%.3f",
            stream["id"],
            intent,
            action,
            accuracy(observed["test"]),
            accuracy(observed["guard"]),
        )
    return outcomes


def run_measurement(observer, config, spec, data, stream, learning, eligible_intents, output, pilot_sources=()):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    observer.reset(learning / "acquired")
    lens, lens_result = train_lens(observer, data, spec, output, seed_for(stream["seed"], "lens"))
    write_json(output / "lens-training.json", lens_result)
    guards = rows_for(stream, stream["new"], "test")
    units = [
        {
            "id": f"{stream['id']}/{intent}",
            "stream": stream["id"],
            "split": stream["split"],
            "intent": intent,
            "baseline": {},
            "probes": {},
            "actions": {},
        }
        for intent in eligible_intents
    ]
    for phase, checkpoint in (("before", "acquired"), ("after", "forgotten")):
        observer.reset(learning / checkpoint)
        guard = observer.observe(guards)
        for unit in units:
            unit["baseline"][phase] = {
                "test": observer.observe(stream["units"][unit["intent"]]["test"]),
                "guard": guard,
            }
            unit["probes"][phase] = observer.observe(stream["units"][unit["intent"]]["probe"], lens=lens)
    late_cache, late_metadata = cache_activations(observer, data["lens_check"], spec["lens"]["positions_per_prompt"])
    late_kl = evaluate_translations(lens, observer, late_cache, spec["lens"]["token_batch_size"])
    write_json(output / "lens-late-check.json", {"metadata": late_metadata, "heldout_kl": late_kl})
    del late_cache
    for unit in units:
        unit["features"] = feature_vector(unit["probes"]["before"], unit["probes"]["after"], spec["layers"])
    write_json(output / "features-frozen.json", units)
    feature_hash = file_hash(output / "features-frozen.json")
    reusable = {}
    for source in pilot_sources:
        payload, _ = verify_result(source)
        if payload["stream"] == stream["id"]:
            for unit in read_json(Path(source) / "units.json"):
                reusable[unit["intent"]] = (Path(source), unit)
    reused = 0
    for unit in units:
        intent = unit["intent"]
        if intent in reusable:
            source, prior = reusable[intent]
            if prior["baseline"] != unit["baseline"]:
                raise ValueError("PROSPECTIVE_PILOT_REUSE_BASELINE_MISMATCH")
            shutil.copytree(source / "actions" / intent, output / "actions" / intent)
            unit["actions"] = prior["actions"]
            unit["reused_pilot_receipt_sha256"] = file_hash(source / "receipt.json")
            reused += 1
        else:
            unit["actions"] = perform_actions(
                observer,
                config,
                spec,
                stream,
                intent,
                learning / "acquired",
                learning / "forgotten",
                unit["baseline"],
                output,
            )
        write_json(output / "units" / f"{intent}.json", unit)
    if feature_hash != file_hash(output / "features-frozen.json"):
        raise ValueError("PROSPECTIVE_FEATURES_MUTATED")
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("PROSPECTIVE_REPAIR_CHANGED_BACKBONE")
    write_json(output / "units.json", units)
    write_json(output / "frozen-backbone.json", frozen)
    return {
        "stage": "measure",
        "stream": stream["id"],
        "split": stream["split"],
        "status": "completed",
        "unit_ids": [unit["id"] for unit in units],
        "features_sha256": feature_hash,
        "stream_summary": stream_summary(units, spec),
        "translator_qualified": lens_result["translator_qualified"],
        "repair_updates": (len(units) - reused) * 3 * config["repair_updates"],
        "reused_pilot_units": reused,
        "represented_repair_updates": len(units) * 3 * config["repair_updates"],
        "frozen_backbone_unchanged": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("learn", "qualify", "measure", "qualify-repairs", "analyze"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stream")
    parser.add_argument("--sources", type=Path, nargs="+", default=[])
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--repair-qualification", type=Path)
    parser.add_argument("--pilot-qualification", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [prospective-recovery] %(message)s")
    config, spec = load_design(args.config)
    data = read_json(ROOT / config["dataset"])
    validate_data(data, spec)
    if args.stage in {"learn", "measure"} and args.stream not in data["streams"]:
        raise ValueError("PROSPECTIVE_STREAM_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_json(args.output_dir / "seal.json", make_seal(config, spec, data, args.stage))
        if args.stage == "learn":
            verify_model(config, data)
            stream = data["streams"][args.stream]
            observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
            result = run_learning(observer, config, spec, stream, args.output_dir)
        elif args.stage == "qualify":
            result = qualify_learning(args.sources, config, spec, data)
            write_json(args.output_dir / "qualification.json", result)
        else:
            if args.qualification is None:
                raise ValueError("PROSPECTIVE_QUALIFICATION_PATH_REQUIRED")
            qualified = checked_qualification(args.qualification, config, spec, data)
            if args.stage == "measure":
                if args.pilot_qualification is None:
                    raise ValueError("PROSPECTIVE_SMALL_REPAIR_PILOT_REQUIRED")
                pilot = checked_pilot_qualification(args.pilot_qualification, config, spec, data, args.qualification)
                stream = data["streams"][args.stream]
                if stream["split"] == "test":
                    if args.repair_qualification is None:
                        raise ValueError("PROSPECTIVE_TRAIN_REPAIR_SIGNAL_REQUIRED")
                    checked_repair_qualification(args.repair_qualification, config, spec, data, args.qualification)
                verify_model(config, data)
                observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
                source = qualified["streams"][args.stream]
                result = run_measurement(
                    observer,
                    config,
                    spec,
                    data,
                    stream,
                    Path(source["source"]),
                    source["eligible"],
                    args.output_dir,
                    pilot["sources"],
                )
                result["repair_qualification_sha256"] = (
                    file_hash(args.repair_qualification / "receipt.json") if args.repair_qualification else None
                )
                result["pilot_qualification_sha256"] = file_hash(args.pilot_qualification / "receipt.json")
            elif args.stage == "qualify-repairs":
                units = collect_units(args.sources, config, spec, data, "train", args.qualification)
                feasibility = repair_feasibility(units, spec)
                if feasibility["qualified"]:
                    write_json(args.output_dir / "predictors.json", fit_predictors(units, spec))
                result = {
                    "stage": "qualify-repairs",
                    "qualified": feasibility["qualified"],
                    "feasibility": feasibility,
                    "sources": [str(path.resolve()) for path in args.sources],
                }
                write_json(args.output_dir / "repair-qualification.json", result)
            else:
                if args.repair_qualification is None:
                    raise ValueError("PROSPECTIVE_TRAIN_REPAIR_SIGNAL_REQUIRED")
                checked_repair_qualification(args.repair_qualification, config, spec, data, args.qualification)
                units = collect_units(args.sources, config, spec, data, "test", args.qualification)
                for path in args.sources:
                    received = read_json(path / "receipt.json")["payload"]["repair_qualification_sha256"]
                    if received != file_hash(args.repair_qualification / "receipt.json"):
                        raise ValueError("PROSPECTIVE_HELDOUT_PREDICTOR_BINDING")
                result = {
                    "stage": "analyze",
                    **evaluate_predictors(read_json(args.repair_qualification / "predictors.json"), units, spec),
                }
                write_json(args.output_dir / "analysis.json", result)
            result["qualification_sha256"] = file_hash(args.qualification / "receipt.json")
        completed = finish(args.output_dir, result)
        logging.info(
            "PROSPECTIVE_STAGE_FINISHED stage=%s status=%s",
            args.stage,
            completed.get("status", completed.get("qualified")),
        )
    except Exception as error:
        logging.exception("PROSPECTIVE_STAGE_FAILED stage=%s", args.stage)
        write_json(args.output_dir / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
