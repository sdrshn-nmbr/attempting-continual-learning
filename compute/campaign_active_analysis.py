import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ARMS = ("random", "hypothesis_elimination_oracle")


def read(path):
    return json.loads(path.read_text())


def quantile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def contrast(predictions, coefficients, seed, replicates=2000):
    reference = predictions[next(iter(coefficients))]
    ids = [row["id"] for row in reference]
    if len(set(ids)) != len(ids):
        raise ValueError("DUPLICATE_EVALUATION_ID")
    by_device = defaultdict(lambda: defaultdict(list))
    values = []
    for index, row in enumerate(reference):
        effect = 0
        for condition, coefficient in coefficients.items():
            other = predictions[condition][index]
            if (row["id"], row["label"], row["device"], row["input"]) != (
                other["id"],
                other["label"],
                other["device"],
                other["input"],
            ):
                raise ValueError("UNPAIRED_EVALUATION_ROWS")
            effect += coefficient * (other["prediction"] == other["label"])
        values.append(effect)
        by_device[row["device"]][row["input"] & 31].append(effect)
    device_groups = [list(groups.values()) for _, groups in sorted(by_device.items())]
    if len({sum(map(len, groups)) for groups in device_groups}) != 1:
        raise ValueError("UNEQUAL_DEVICE_EVALUATION_COUNTS")
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        device_means = []
        for groups in device_groups:
            drawn = [rng.choice(groups) for _ in groups]
            device_means.append(sum(map(sum, drawn)) / sum(map(len, drawn)))
        samples.append(sum(device_means) / len(device_means))
    samples.sort()
    return {
        "effect": sum(values) / len(values),
        "interval_95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "paired_examples": len(values),
        "causal_groups_by_device": {
            str(device): len(groups) for device, groups in by_device.items()
        },
        "bootstrap_seed": seed,
        "replicates": replicates,
        "scope": "Device-stratified paired resampling of causal-input groups; unadjusted for multiple contrasts and not uncertainty over training seeds.",
    }


def analyze(observed, posterior):
    roots = {"observed": observed, "posterior": posterior}
    configs = {name: read(root / "config.json") for name, root in roots.items()}
    normalized = {
        name: {key: value for key, value in config.items() if key != "consolidation"}
        for name, config in configs.items()
    }
    if normalized["observed"] != normalized["posterior"]:
        raise ValueError("FACTORIAL_CONFIG_MISMATCH")
    predictions, records, sources, budgets, initializations = {}, {}, [], [], []
    evidence = []
    for mode, root in roots.items():
        result = read(root / "result.json")
        execution = read(root / "execution.json")
        if result["status"] != "complete" or execution["status"] != "completed":
            raise ValueError(f"RUN_NOT_COMPLETE: {root}")
        sources.append(read(root / "task.json")["source_sha256"])
        budgets.append(result["matched_budgets"])
        evidence.append(
            {
                "path": str((root / "result.json").resolve()),
                "sha256": hashlib.sha256(
                    (root / "result.json").read_bytes()
                ).hexdigest(),
            }
        )
        for arm in result["arms"]:
            name = f"{mode}/{arm['arm']}"
            records[name] = arm
            initializations.append(arm["adapter"]["initial_adapter_sha256"])
            for stage in arm["stages"]:
                if not all(
                    stage["reload"][key]
                    for key in ("reload_tensor_equality", "reload_logit_equality")
                ):
                    raise ValueError(f"RELOAD_NOT_QUALIFIED: {name}")
                if (
                    not stage["training"]["parameters_changed"]
                    or not stage["training"]["all_losses_finite"]
                ):
                    raise ValueError(f"TRAINING_NOT_QUALIFIED: {name}")
            predictions[name] = [
                row
                for row in read(root / arm["arm"] / "eval-stage1.json")["predictions"]
                if row["kind"] == "single"
            ]
    if (
        len(set(sources)) != 1
        or len(set(initializations)) != 1
        or budgets[0] != budgets[1]
    ):
        raise ValueError("FACTORIAL_SOURCE_INITIALIZATION_OR_BUDGET_MISMATCH")
    for arm in ARMS:
        for stage in range(2):
            ledgers = [
                read(root / arm / f"observations-stage{stage}.json")
                for root in roots.values()
            ]
            if ledgers[0] != ledgers[1]:
                raise ValueError("ACQUISITION_DEPENDS_ON_CONSOLIDATION")
            rows = [
                read(root / arm / f"training-rows-stage{stage}.json")
                for root in roots.values()
            ]
            if [row for row in rows[0] if row["old_replay"]] != [
                row for row in rows[1] if row["old_replay"]
            ]:
                raise ValueError("FACTORIAL_OLD_REPLAY_MISMATCH")
    names = {mode: {arm: f"{mode}/{arm}" for arm in ARMS} for mode in roots}
    pairs = {
        "active_effect_observed": (
            names["observed"][ARMS[1]],
            names["observed"][ARMS[0]],
        ),
        "active_effect_posterior": (
            names["posterior"][ARMS[1]],
            names["posterior"][ARMS[0]],
        ),
        "consolidation_effect_random": (
            names["posterior"][ARMS[0]],
            names["observed"][ARMS[0]],
        ),
        "consolidation_effect_oracle": (
            names["posterior"][ARMS[1]],
            names["observed"][ARMS[1]],
        ),
    }
    seed = configs["observed"]["seed"]
    contrasts = {
        name: contrast(predictions, {target: 1, reference: -1}, seed)
        for name, (target, reference) in pairs.items()
    }
    contrasts["interaction"] = contrast(
        predictions,
        {
            names["posterior"][ARMS[1]]: 1,
            names["posterior"][ARMS[0]]: -1,
            names["observed"][ARMS[1]]: -1,
            names["observed"][ARMS[0]]: 1,
        },
        seed,
    )
    summary = {}
    for name, record in records.items():
        stages = record["stages"]
        summary[name] = {
            "stage0_accuracy": stages[0]["metrics"]["single"]["accuracy"],
            "stage0_learning_gain": stages[0]["metrics"]["single"]["accuracy"]
            - stages[0]["before_metrics"]["single"]["accuracy"],
            "stage1_accuracy": stages[1]["metrics"]["single"]["accuracy"],
            "stage0_queried_ledger_fit": stages[0]["training_fit"]["single"][
                "accuracy"
            ],
            "new_skill_gain": record["new_skill_learning_gain"],
            "stable_retention_change": record["stable_retention_change"],
            "stage1_composition_accuracy": stages[1]["metrics"]["composition"][
                "accuracy"
            ],
        }
    return {
        "status": "reviewed_factorial_gpu_results",
        "seed": seed,
        "conditions": summary,
        "contrasts": contrasts,
        "source_sha256": sources[0],
        "equal_budgets": budgets[0],
        "initialization_equal": True,
        "same_queries_across_consolidation": True,
        "same_old_replay_across_consolidation": True,
        "reloads_and_updates_qualified": True,
        "evidence": evidence,
        "claim_limit": "Known-family exploratory screen. Positive directions require independent-seed confirmation; contrast intervals are descriptive and unadjusted.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.observed, args.posterior)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"conditions": result["conditions"], "contrasts": result["contrasts"]}
        )
    )


if __name__ == "__main__":
    main()
