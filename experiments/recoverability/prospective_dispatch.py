import argparse
from pathlib import Path

from prospective import main as run_prospective
from prospective_pilot import main as run_pilot
from prospective_reference import main as run_reference
from protocol import ROOT, read_json


def dispatch_arguments(manifest, output):
    if any(type(manifest[key]) is not bool for key in ("reference", "pilot") if key in manifest):
        raise ValueError("PROSPECTIVE_DISPATCH_LANE_TYPE")
    if set(manifest) - {
        "study_config",
        "stage",
        "stream",
        "sources",
        "qualification",
        "repair_qualification",
        "reference",
        "pilot",
        "pilot_qualification",
        "learning_source",
        "reference_source",
        "main_source",
    }:
        raise ValueError("PROSPECTIVE_DISPATCH_UNKNOWN_FIELD")
    config_path = ROOT / manifest["study_config"]
    arguments = ["--config", str(config_path), "--stage", manifest["stage"], "--output-dir", str(output / "study")]
    for field in (
        "stream",
        "qualification",
        "repair_qualification",
        "pilot_qualification",
        "learning_source",
        "reference_source",
        "main_source",
    ):
        if field in manifest:
            arguments.extend(("--" + field.replace("_", "-"), manifest[field]))
    if manifest.get("sources"):
        arguments.extend(("--sources", *manifest["sources"]))
    return arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.config)
    if manifest.get("reference") and manifest.get("pilot"):
        raise ValueError("PROSPECTIVE_DISPATCH_AMBIGUOUS_LANE")
    runner = (
        run_pilot
        if manifest.get("pilot", False)
        else run_reference
        if manifest.get("reference", False)
        else run_prospective
    )
    runner(dispatch_arguments(manifest, args.output_dir))


if __name__ == "__main__":
    main()
