import argparse
import logging
from pathlib import Path

from analysis import analyze, features
from model import Observer, runtime_info
from protocol import ACTIONS, PROTOCOL, ROOT, digest, file_hash, prepare, read_json, seal, verify_seal, write_json

LOG = logging.getLogger("recoverability")


def execute(sealed, output):
    payload = verify_seal(sealed, runtime_info())
    spec, config = payload["protocol"], payload["config"]
    data = read_json(config["dataset"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "seal.json", sealed)
    first = next(iter(data["runs"].values()))["after"]["path"]
    observer = Observer.load(config, spec, data["codes"], first)
    observations = {"units": {}, "guards": {}}
    for run, checkpoints in data["runs"].items():
        observations["guards"][run] = {}
        for phase in ("before", "after"):
            observer.reset(checkpoints[phase]["path"])
            observations["guards"][run][phase] = observer.observe(data["guards"][run])
            for unit in (u for u in data["units"] if u["run"] == run):
                LOG.info("RECOVERY_OBSERVE unit=%s phase=%s", unit["id"], phase)
                observations["units"].setdefault(unit["id"], {})[phase] = {
                    "probe": observer.observe(unit["probe"], lens=True),
                    "label": observer.observe(unit["label"]),
                }
    feature_table = {
        unit_id: features({phase: raw[phase]["probe"] for phase in ("before", "after")})
        for unit_id, raw in observations["units"].items()
    }
    write_json(output / "observations.json", observations)
    write_json(output / "features.json", feature_table)
    write_json(
        output / "features_frozen.json",
        {
            "seal_sha256": sealed["sha256"],
            "observations_sha256": file_hash(output / "observations.json"),
            "features_sha256": file_hash(output / "features.json"),
            "recovery_actions_started": False,
        },
    )
    outcomes = {}
    state_files = {}
    (output / "adapters").mkdir()
    for index, unit in enumerate(data["units"]):
        outcomes[unit["id"]] = {}
        checkpoints = data["runs"][unit["run"]]
        for action in ACTIONS:
            LOG.info("RECOVERY_ACTION unit=%s action=%s", unit["id"], action)
            observer.reset(checkpoints["after"]["path"])
            losses = []
            if action == "restore":
                observer.reset(checkpoints["before"]["path"])
            elif action != "none":
                losses = observer.repair(unit["repair"], action)
            record = {
                "label": observer.observe(unit["label"]),
                "guard": observer.observe(data["guards"][unit["run"]]),
                "updates": losses,
            }
            if action in {"none", "restore"}:
                phase = "after" if action == "none" else "before"
                expected = observations["units"][unit["id"]][phase]["label"]
                expected_guard = observations["guards"][unit["run"]][phase]
                if record["label"] != expected or record["guard"] != expected_guard:
                    raise ValueError(f"RECOVERY_CONTROL_IDENTITY {unit['id']} {action}")
            else:
                relative = f"adapters/{index}-{action}.safetensors"
                state_files[relative] = observer.save_adapter(output / relative)
            outcomes[unit["id"]][action] = record
        write_json(output / f"unit-{index}.json", {"id": unit["id"], "outcomes": outcomes[unit["id"]]})
    write_json(output / "outcomes.json", outcomes)
    verify_seal(sealed, runtime_info())
    artifacts = {
        name: file_hash(output / name)
        for name in (
            "seal.json",
            "observations.json",
            "features.json",
            "features_frozen.json",
            "outcomes.json",
        )
    }
    write_json(
        output / "receipt.json",
        {
            "status": "complete",
            "seal_sha256": sealed["sha256"],
            "artifacts": artifacts | state_files,
            "units": len(data["units"]),
            "device": spec["runtime"]["device"],
            "classification": "measured_coded_intent_recovery",
        },
    )


def analyze_output(output):
    receipt = read_json(output / "receipt.json")
    if receipt["status"] != "complete":
        raise ValueError("RECOVERY_INCOMPLETE_RUN")
    for name, expected in receipt["artifacts"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"RECOVERY_OUTPUT_CHANGED {name}")
    sealed = read_json(output / "seal.json")
    if digest(sealed["payload"]) != sealed["sha256"] or receipt["seal_sha256"] != sealed["sha256"]:
        raise ValueError("RECOVERY_ANALYSIS_SEAL")
    payload = verify_seal(sealed, runtime_info())
    data = read_json(payload["config"]["dataset"])
    observations = read_json(output / "observations.json")
    table = read_json(output / "features.json")
    frozen = read_json(output / "features_frozen.json")
    if frozen["features_sha256"] != file_hash(output / "features.json") or frozen["recovery_actions_started"]:
        raise ValueError("RECOVERY_FEATURE_FREEZE")
    recomputed = {
        unit_id: features({phase: raw[phase]["probe"] for phase in ("before", "after")})
        for unit_id, raw in observations["units"].items()
    }
    if recomputed != table:
        raise ValueError("RECOVERY_FEATURE_PROVENANCE")
    result = analyze(data, observations, read_json(output / "outcomes.json"), table, payload["protocol"])
    result["seal_sha256"] = sealed["sha256"]
    write_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Frozen logit-lens recoverability; parent-owned GPU launch")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    lock = sub.add_parser("seal")
    lock.add_argument("--config", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--seal", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("analyze")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("RECOVERY_OUTPUT_SCOPE: outputs must be under experiments/recoverability")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "prepare":
        if file_hash(PROTOCOL) != (ROOT / "protocol.sha256").read_text().strip():
            raise ValueError("RECOVERY_PROTOCOL_CHANGED")
        data = prepare(args.source_root)
        sources = {}
        for path, expected in data["sources"].items():
            source = Path(path)
            relative = Path("sources") / source.parent.name / source.name
            destination = args.output.parent / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(source.read_bytes())
            sources[str(relative)] = expected
        data["sources"] = sources
        write_json(args.output, data)
        LOG.info("RECOVERY_COHORT_PREPARED units=%d sha256=%s", len(data["units"]), file_hash(args.output))
    elif args.command == "seal":
        write_json(args.output, seal(args.config, runtime_info()))
        LOG.info("RECOVERY_SEALED %s", args.output)
    elif args.command == "run":
        execute(read_json(args.seal), args.output)
    else:
        result = analyze_output(args.output)
        LOG.info("RECOVERY_ANALYZED status=%s feasibility=%s", result["status"], result["predictor_feasibility"])


if __name__ == "__main__":
    main()
