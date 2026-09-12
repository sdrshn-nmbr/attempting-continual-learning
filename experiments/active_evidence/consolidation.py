from collections import Counter, defaultdict

from environment import FAMILY, PROFILES, digest, keyed_rng, partition, posterior


def observed_rows(evidence):
    rows = []
    for observation in evidence:
        if set(observation) != {"device", "input", "observed", "query_index"}:
            raise ValueError("CONSOLIDATION_REQUIRES_QUERIED_LEDGER_ONLY")
        if observation["observed"] not in (0, 1):
            raise ValueError("NONBINARY_OBSERVATION")
        rows.append(
            training_row(
                observation["device"],
                observation["input"],
                observation["observed"],
                "observed_query",
            )
        )
    return rows


def training_row(device, value, label, source):
    return {
        "device": device,
        "input": value,
        "label": label,
        "profile": PROFILES[device],
        "context": (value >> 4) & 1,
        "kind": "single",
        "devices": [device],
        "label_source": source,
    }


def resample(rows, count, rng):
    if not rows and count:
        raise ValueError("EMPTY_REPLAY_LEDGER")
    result = []
    while len(result) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:count]


def training_plan(config, stage, evidence, old_evidence):
    settings = config["training"]
    consolidation = config["consolidation"]
    source = consolidation["label_source"]
    if source not in ("observed_query", "posterior_predictive"):
        raise ValueError(f"UNKNOWN_CONSOLIDATION_SOURCE: {source}")
    observed = observed_rows(evidence)
    old_observed = observed_rows(old_evidence)
    acquisition, heldout = partition(config["seed"])
    acquisition_set = set(acquisition)
    heldout_set = set(heldout)
    if any(row["input"] not in acquisition_set for row in observed + old_observed):
        raise ValueError("CONSOLIDATION_HELDOUT_LEAKAGE")
    devices = (0, 1, 2, 3) if stage == 0 else (1, 2, 3, 4)
    if {row["device"] for row in observed} != set(devices):
        raise ValueError("CURRENT_LEDGER_DEVICE_MISMATCH")
    if any(row["device"] != 0 for row in old_observed):
        raise ValueError("OLD_REPLAY_MUST_CONTAIN_ONLY_STABLE_DEVICE")
    old_slots = consolidation["old_examples_per_batch"] if stage else 0
    current_slots = settings["batch_size"] - old_slots
    current_count = current_slots * settings["updates_per_stage"]
    if not 0 <= old_slots < settings["batch_size"] or current_count % len(devices):
        raise ValueError("UNBALANCED_CONSOLIDATION_BUDGET")
    per_device = current_count // len(devices)
    grouped = defaultdict(list)
    for row in evidence:
        grouped[row["device"]].append(row)
    current, posterior_states = [], {}
    for device in devices:
        rng = keyed_rng(config["seed"], stage, device, "consolidation")
        device_evidence = grouped[device]
        if source == "observed_query":
            current.extend(resample(observed_rows(device_evidence), per_device, rng))
        else:
            weights, state = posterior(
                device_evidence, config["environment"]["assumed_noise"]
            )
            posterior_states[str(device)] = state
            selected = resample(list(acquisition), per_device, rng)
            for value in selected:
                probability = sum(
                    weight * rule(value)
                    for weight, rule in zip(weights, FAMILY, strict=True)
                )
                row = training_row(device, value, int(probability > 0.5), source)
                row["posterior_probability_one"] = probability
                current.append(row)
    keyed_rng(config["seed"], stage, "current-order").shuffle(current)
    old = resample(
        old_observed,
        old_slots * settings["updates_per_stage"],
        keyed_rng(config["seed"], stage, "old-order"),
    )
    rows = []
    for update in range(settings["updates_per_stage"]):
        batch = [
            {**row, "old_replay": False}
            for row in current[update * current_slots : (update + 1) * current_slots]
        ] + [
            {**row, "old_replay": True}
            for row in old[update * old_slots : (update + 1) * old_slots]
        ]
        keyed_rng(config["seed"], stage, update, "batch-order").shuffle(batch)
        rows.extend(batch)
    if len(rows) != settings["updates_per_stage"] * settings["batch_size"]:
        raise AssertionError("CONSOLIDATION_EXPOSURE_MISMATCH")
    if any(row["input"] in heldout_set for row in rows):
        raise AssertionError("GENERATED_TRAINING_HELDOUT_LEAKAGE")
    queried = {(row["device"], row["input"]) for row in evidence + old_evidence}
    audit = {
        "label_source": source,
        "current_query_ledger_sha256": digest(evidence),
        "old_query_ledger_sha256": digest(old_evidence),
        "training_rows_sha256": digest(rows),
        "examples": len(rows),
        "current_examples": len(current),
        "old_examples": len(old),
        "examples_by_device": dict(Counter(str(row["device"]) for row in rows)),
        "examples_by_label": dict(Counter(str(row["label"]) for row in rows)),
        "unique_device_inputs": len({(row["device"], row["input"]) for row in rows}),
        "inferred_unqueried_examples": sum(
            row["label_source"] == "posterior_predictive"
            and (row["device"], row["input"]) not in queried
            for row in rows
        ),
        "posterior_states": posterior_states,
        "posterior_input": "queried input and observed bit only",
        "posterior_family_size": len(FAMILY),
        "posterior_decision": "probability_one > 0.5; ties select zero",
        "posterior_training_inputs": "uniform shuffled acquisition split without replacement until exhausted",
        "old_label_source": "observed_query",
        "hidden_rule_access": False,
        "heldout_access": False,
    }
    return rows, audit
