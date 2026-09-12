import argparse
import json
import logging
from pathlib import Path

from followup import comparison, finish, fresh_runtime, measure, prepare_followup, restore_source
from learning import event, write_json
from sandbox import FAMILIES


def audit_conditions(source, protocol):
    available = source.result["transfer"]
    allowed = protocol["audit"]["conditions"]
    if not {"fresh", "relevant", "irrelevant", "arithmetic", "agent_dice"} <= set(available) <= set(allowed):
        raise RuntimeError("AUDIT_CONDITION_INVENTORY")
    return [condition for condition in allowed if condition in available]


def test_surfaces(source, condition):
    directory = f"transfer/{condition}"
    target = source.corpus["workflow"][source.config["target_family"]]
    surfaces = {"workflow.test": (f"{directory}/test.json", target["test"])}
    surfaces["workflow.novel-test"] = (f"{directory}/novel-test.json", target["novel_test"])
    for family in FAMILIES:
        surfaces[f"primitive-{family}.test"] = (
            f"{directory}/primitive-after/{family}.test.json",
            source.corpus["primitive"][family]["test"],
        )
    return surfaces


def audit(settings, output):
    source, protocol = prepare_followup(settings, output, "audit")
    conditions = audit_conditions(source, protocol)
    model, encoded = fresh_runtime(source, output)
    results = {}
    for condition in conditions:
        checkpoint = f"transfer/{condition}/checkpoint"
        metadata = source.read_json(f"{checkpoint}/metadata.json")
        if metadata["updates"] != source.config["workflow_checkpoints"][-1]:
            raise RuntimeError(f"AUDIT_NOT_FINAL_CHECKPOINT: {condition}")
        restore_source(model, source, checkpoint)
        comparisons = {}
        for name, (reference, rows) in test_surfaces(source, condition).items():
            expected = source.evaluation(reference, rows)
            actual = measure(model, encoded, rows, output / condition / f"{name}.json")
            comparisons[name] = comparison(expected, actual)
        results[condition] = {
            "checkpoint": checkpoint,
            "all_records_exact": all(item["all_records_exact"] for item in comparisons.values()),
            "comparisons": comparisons,
        }
        write_json(output / condition / "audit.json", results[condition])
        event(
            output / "events.jsonl",
            "persistent_load_audit",
            condition=condition,
            exact=results[condition]["all_records_exact"],
        )
    exact = all(item["all_records_exact"] for item in results.values())
    return finish(
        source,
        output,
        {
            "status": "persistent_loads_verified" if exact else "persistent_load_mismatch",
            "all_records_exact": exact,
            "conditions": results,
            "unavailable_stopped_conditions": [
                name for name in protocol["audit"]["conditions"] if name not in conditions
            ],
            "scope": "final workflow-trained conditions; original learned primitives and workflow/novel TEST records",
        },
        model,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = audit(json.loads(args.config.read_text()), args.output_dir)
    if not result["all_records_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
