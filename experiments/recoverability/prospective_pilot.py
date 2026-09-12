import argparse
import logging
from pathlib import Path

from prospective import (
    assert_bound_records,
    checked_qualification,
    finish,
    make_seal,
    measurement_identity,
    perform_actions,
    verify_model,
    verify_result,
)
from prospective_analysis import repair_feasibility, stream_summary
from prospective_data import load_design, rows_for, validate_data
from prospective_model import ProspectiveModel, parameter_hashes
from protocol import ROOT, file_hash, read_json, write_json

PILOT_PROTOCOL = ROOT / "repair-pilot-protocol.json"


def measure_pilot(observer, config, spec, stream, learning, intents, output):
    frozen = parameter_hashes(observer.model, frozen_only=True)
    units = [
        {
            "id": f"{stream['id']}/{intent}",
            "stream": stream["id"],
            "split": stream["split"],
            "intent": intent,
            "baseline": {},
            "actions": {},
        }
        for intent in intents
    ]
    for phase, checkpoint in (("before", "acquired"), ("after", "forgotten")):
        observer.reset(learning / checkpoint)
        guard = observer.observe(rows_for(stream, stream["new"], "test"))
        for unit in units:
            unit["baseline"][phase] = {
                "test": observer.observe(stream["units"][unit["intent"]]["test"]),
                "guard": guard,
            }
    write_json(output / "baselines-frozen.json", units)
    for unit in units:
        unit["actions"] = perform_actions(
            observer,
            config,
            spec,
            stream,
            unit["intent"],
            learning / "acquired",
            learning / "forgotten",
            unit["baseline"],
            output,
        )
        write_json(output / "units" / f"{unit['intent']}.json", unit)
    if frozen != parameter_hashes(observer.model, frozen_only=True):
        raise ValueError("PROSPECTIVE_PILOT_BACKBONE_CHANGED")
    write_json(output / "units.json", units)
    return {
        "stage": "pilot-measure",
        "stream": stream["id"],
        "split": stream["split"],
        "unit_ids": [unit["id"] for unit in units],
        "lens_updates": 0,
        "stream_summary": stream_summary(units, spec),
        "repair_updates": len(units) * 3 * config["repair_updates"],
    }


def qualify_pilot(sources, config, spec, data, qualification, cohort, pilot_spec):
    units, seen_streams = [], set()
    for source in sources:
        receipt, sealed = verify_result(source)
        if (
            receipt["stage"] != "pilot-measure"
            or receipt["split"] != "train"
            or sealed["measurement_identity"] != measurement_identity(config, spec, data)
            or receipt["qualification_sha256"] != file_hash(qualification / "receipt.json")
        ):
            raise ValueError("PROSPECTIVE_PILOT_TRAIN_ONLY_BINDING")
        stream = data["streams"][receipt["stream"]]
        if stream["id"] in seen_streams:
            raise ValueError("PROSPECTIVE_PILOT_DUPLICATE_STREAM")
        seen_streams.add(stream["id"])
        measured = read_json(source / "units.json")
        expected = cohort["streams"][stream["id"]]["eligible"][: pilot_spec["intents_per_train_stream"]]
        if [unit["intent"] for unit in measured] != expected:
            raise ValueError("PROSPECTIVE_PILOT_PREDECLARED_SUBSET")
        for unit in measured:
            if unit["split"] != "train" or unit["stream"] != stream["id"]:
                raise ValueError("PROSPECTIVE_PILOT_UNIT_SPLIT")
            for observation in (*unit["baseline"].values(), *unit["actions"].values()):
                assert_bound_records(observation["test"], stream["units"][unit["intent"]]["test"])
                assert_bound_records(observation["guard"], rows_for(stream, stream["new"], "test"))
        units.extend(measured)
    if seen_streams != {name for name, stream in data["streams"].items() if stream["split"] == "train"}:
        raise ValueError("PROSPECTIVE_PILOT_BOTH_TRAIN_STREAMS_REQUIRED")
    feasibility = repair_feasibility(units, spec)
    return {
        "stage": "pilot-qualify",
        "qualified": feasibility["qualified"],
        "feasibility": feasibility,
        "sources": [str(path.resolve()) for path in sources],
        "qualification_sha256": file_hash(qualification / "receipt.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("measure", "qualify"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stream")
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", default=[])
    args = parser.parse_args(argv)
    config, spec = load_design(args.config)
    data = read_json(ROOT / config["dataset"])
    validate_data(data, spec)
    if file_hash(PILOT_PROTOCOL) != PILOT_PROTOCOL.with_suffix(".sha256").read_text().strip():
        raise ValueError("PROSPECTIVE_PILOT_PROTOCOL_CHANGED")
    pilot_spec = read_json(PILOT_PROTOCOL)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [prospective-repair-pilot] %(message)s")
    try:
        write_json(args.output_dir / "seal.json", make_seal(config, spec, data, "pilot-" + args.stage))
        write_json(args.output_dir / "pilot-protocol.json", pilot_spec)
        cohort = checked_qualification(args.qualification, config, spec, data)
        if args.stage == "measure":
            if args.stream not in data["streams"] or data["streams"][args.stream]["split"] != "train":
                raise ValueError("PROSPECTIVE_PILOT_TRAIN_ONLY")
            stream = data["streams"][args.stream]
            source = cohort["streams"][args.stream]
            verify_model(config, data)
            observer = ProspectiveModel.fresh(config, spec, data, stream["seed"])
            result = measure_pilot(
                observer,
                config,
                spec,
                stream,
                Path(source["source"]),
                source["eligible"][: pilot_spec["intents_per_train_stream"]],
                args.output_dir,
            )
            result["qualification_sha256"] = file_hash(args.qualification / "receipt.json")
        else:
            result = qualify_pilot(args.sources, config, spec, data, args.qualification, cohort, pilot_spec)
            write_json(args.output_dir / "pilot-qualification.json", result)
        finish(args.output_dir, result)
    except Exception as error:
        logging.exception("PROSPECTIVE_REPAIR_PILOT_FAILED")
        write_json(args.output_dir / "failure.json", {"exception": type(error).__name__, "detail": str(error)})
        raise


if __name__ == "__main__":
    main()
