from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from data import SEQUENCE_TASKS, write_json
from run import (
    child_command,
    emit,
    initial_paths,
    prepare,
    reload_probes,
    require_gpu,
    restore_learner,
    validate_config,
    verify_prepared,
)


def reload_only(config: dict, output: Path) -> None:
    require_gpu(output, config["seed"])
    directory = initial_paths(output, config["method"])
    learner = restore_learner(config, output, directory, 0)
    try:
        reload = reload_probes(learner, output, directory, SEQUENCE_TASKS)
    finally:
        learner.close()
    initialization = json.loads((output / "initialization.json").read_text())
    result = {
        "kind": "initialization_and_reload_qualification_only",
        "arm": config["arm"],
        "qualified": initialization["parity"]["passed"] and reload["passed"],
        "initialization_parity": initialization["parity"],
        "parameter_match": initialization["parameter_match"],
        "reload": reload,
        "training_updates": 0,
        "scientific_result": None,
        "heldout_access": False,
        "remaining_evidence": "No sequence training, retention, or learned-state transport has been measured by this qualification.",
    }
    write_json(output / "qualification.json", result)
    emit(output, "initialization_reload_qualified", qualified=result["qualified"])
    if not result["qualified"]:
        raise ValueError("INITIALIZATION_RELOAD_QUALIFICATION_FAILED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "reload"), default="run")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config)
    try:
        if args.phase == "reload":
            verify_prepared(config, output)
            reload_only(config, output)
            return
        prepare(config, output)
        subprocess.run(child_command(output, "initialize"), check=True)
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(Path(__file__).resolve()),
                "--config",
                str(output / "config.json"),
                "--output-dir",
                str(output),
                "--phase",
                "reload",
            ],
            check=True,
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "phase": args.phase,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        emit(output, "qualification_failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
