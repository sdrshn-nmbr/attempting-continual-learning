import hashlib
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass


GRAMMAR = (
    "Each device maps ten binary inputs b0 through b9 to one bit. "
    "Its rule is a bit, or AND, OR, or XOR of two bits, among b0 through b3; "
    "the result may be negated and may additionally be toggled when b4 is 1. "
    "Bits b5 through b9 are irrelevant. Device rules are independent. "
    "Return exactly one digit: 0 or 1."
)
PROFILES = ("stable", "drift", "exception", "noise", "new_skill")
ARMS = ("random", "model_uncertainty", "hypothesis_elimination_oracle")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def keyed_rng(*parts):
    return random.Random(int(digest(parts)[:16], 16))


def bits(value):
    return tuple((value >> index) & 1 for index in range(10))


def bit_text(value):
    return " ".join(str(bit) for bit in bits(value))


@dataclass(frozen=True)
class Rule:
    operation: str
    first: int
    second: int = 0
    negate: int = 0
    exception: int = 0

    def __call__(self, value):
        left, right = (value >> self.first) & 1, (value >> self.second) & 1
        if self.operation == "copy":
            output = left
        elif self.operation == "and":
            output = left & right
        elif self.operation == "or":
            output = left | right
        elif self.operation == "xor":
            output = left ^ right
        else:
            raise ValueError(f"Unknown rule operation: {self.operation}")
        return output ^ self.negate ^ (self.exception & ((value >> 4) & 1))

    def text(self):
        expression = f"b{self.first}"
        if self.operation != "copy":
            expression = f"(b{self.first} {self.operation.upper()} b{self.second})"
        if self.negate:
            expression = f"NOT {expression}"
        if self.exception:
            expression = f"({expression}) XOR b4"
        return expression


def hypothesis_family():
    candidates = []
    for negate, exception in itertools.product(range(2), repeat=2):
        candidates.extend(
            Rule("copy", i, negate=negate, exception=exception) for i in range(4)
        )
        for operation, pair in itertools.product(
            ("and", "or", "xor"), itertools.combinations(range(4), 2)
        ):
            candidates.append(Rule(operation, *pair, negate, exception))
    signatures = [tuple(rule(x) for x in range(32)) for rule in candidates]
    if len(set(signatures)) != len(signatures):
        raise AssertionError("Duplicate hypothesis truth tables")
    return tuple(candidates)


FAMILY = hypothesis_family()


def stage_devices(stage):
    return tuple(range(4)) if stage == 0 else (1, 2, 3, 4)


def all_stage_devices(stage):
    return tuple(range(4 + stage))


def make_rules(seed):
    rng = keyed_rng(seed, "hidden_rules")
    order = list(range(4))
    rng.shuffle(order)
    polarity = [rng.randrange(2) for _ in range(5)]
    first = {i: Rule("copy", order[i], negate=polarity[i]) for i in range(4)}
    second = dict(first)
    second[1] = Rule("copy", order[2], negate=polarity[1])
    second[2] = Rule("copy", order[2], negate=polarity[2], exception=1)
    second[4] = Rule("xor", *sorted((order[0], order[2])), negate=polarity[4])
    return (first, second)


def partition(seed):
    acquisition, heldout = [], []
    for signal in range(32):
        values = [signal | (nuisance << 5) for nuisance in range(32)]
        keyed_rng(seed, "split", signal).shuffle(values)
        acquisition.extend(values[:24])
        heldout.extend(values[24:])
    return tuple(sorted(acquisition)), tuple(sorted(heldout))


def is_observational(value):
    vector = bits(value)
    return len(set(vector[:4])) == 1 and vector[4] == 0


def candidate_pool(seed, stage, device, config):
    acquisition, _ = partition(seed)
    observational = [x for x in acquisition if is_observational(x)]
    diagnostic = [x for x in acquisition if not is_observational(x)]
    rng = keyed_rng(seed, stage, device, "pool")
    rng.shuffle(observational)
    rng.shuffle(diagnostic)
    count = config["observational_count"]
    if count > len(observational):
        raise ValueError("Observational pool exceeds the acquisition split")
    pool = observational[:count] + diagnostic[: config["pool_size"] - count]
    if len(pool) != config["pool_size"] or len(set(pool)) != len(pool):
        raise AssertionError("Invalid acquisition pool")
    return pool


def posterior(evidence, assumed_noise):
    if not 0 < assumed_noise < 0.5:
        raise ValueError(
            "Posterior noise probability must be between zero and one half"
        )
    mismatches = [
        sum(rule(row["input"]) != row["observed"] for row in evidence)
        for rule in FAMILY
    ]
    log_weights = [
        count * math.log(assumed_noise)
        + (len(evidence) - count) * math.log1p(-assumed_noise)
        for count in mismatches
    ]
    peak = max(log_weights)
    weights = [math.exp(weight - peak) for weight in log_weights]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    entropy = -sum(weight * math.log2(weight) for weight in weights if weight)
    best = max(range(len(weights)), key=weights.__getitem__)
    return weights, {
        "entropy_bits": entropy,
        "effective_candidates": 2**entropy,
        "hard_consistent_candidates": mismatches.count(0),
        "map_candidate": best,
        "map_rule": FAMILY[best].text(),
        "map_probability": weights[best],
    }


def binary_entropy(probability):
    if probability <= 0 or probability >= 1:
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(
        1 - probability
    )


def selector_inputs(seed, stage, device, step, pool, chosen, config):
    available = [value for value in pool if value not in chosen]
    keyed_rng(seed, stage, device, step, "panel").shuffle(available)
    return available[: config["candidate_panel"]]


def choose_query(arm, candidates, evidence, assumed_noise, uncertainty=None):
    if not candidates:
        raise ValueError("Acquisition candidate panel is empty")
    if arm == "random":
        return candidates[0], []
    if arm == "hypothesis_elimination_oracle":
        weights, _ = posterior(evidence, assumed_noise)
        scores = [
            binary_entropy(
                sum(
                    weight * rule(x)
                    for rule, weight in zip(FAMILY, weights, strict=True)
                )
            )
            for x in candidates
        ]
    elif arm == "model_uncertainty":
        if uncertainty is None:
            raise ValueError("Model uncertainty requires a frozen-model scorer")
        scores = uncertainty(candidates, evidence)
        if len(scores) != len(candidates) or any(
            not math.isfinite(score) for score in scores
        ):
            raise ValueError("Model uncertainty scores are malformed or non-finite")
    else:
        raise ValueError(f"Unknown acquisition arm: {arm}")
    selected = max(range(len(candidates)), key=lambda index: scores[index])
    return candidates[selected], [
        {"input": value, "score": score}
        for value, score in zip(candidates, scores, strict=True)
    ]


def collect_device(
    seed, stage, device, arm, config, query_environment, uncertainty=None
):
    pool = candidate_pool(seed, stage, device, config)
    evidence, events = [], []
    total = config["queries_per_device"]
    for step in range(total):
        before_weights, before = posterior(evidence, config["assumed_noise"])
        if step < config["common_queries"]:
            selected, scores, method = pool[step], [], "common_observation"
        elif step >= total - config["repeat_queries"]:
            index = step - (total - config["repeat_queries"])
            selected, scores, method = evidence[index]["input"], [], "common_repeat"
        else:
            candidates = selector_inputs(
                seed,
                stage,
                device,
                step,
                pool,
                {row["input"] for row in evidence},
                config,
            )
            selected, scores = choose_query(
                arm, candidates, evidence, config["assumed_noise"], uncertainty
            )
            method = arm
        observed = query_environment(selected, step)
        if observed not in (0, 1):
            raise ValueError("Environment returned a non-binary observation")
        evidence.append(
            {
                "device": device,
                "input": selected,
                "observed": observed,
                "query_index": step,
            }
        )
        after_weights, after = posterior(evidence, config["assumed_noise"])
        events.append(
            {
                "stage": stage,
                "device": device,
                "profile": PROFILES[device],
                "arm": arm,
                "query_index": step,
                "method": method,
                "input": selected,
                "bits": bits(selected),
                "observed": observed,
                "observational": is_observational(selected),
                "posterior_before": before,
                "posterior_after": after,
                "candidates_dropped_below_1pct": [
                    i
                    for i, (a, b) in enumerate(
                        zip(before_weights, after_weights, strict=True)
                    )
                    if a >= 0.01 > b
                ],
                "candidate_scores": scores,
            }
        )
    return evidence, events


def evaluate_rows(seed, stage, config):
    rules = make_rules(seed)[stage]
    _, heldout = partition(seed)
    rows = []
    for device in all_stage_devices(stage):
        strata = {
            (label, context): [
                x
                for x in heldout
                if rules[device](x) == label and bits(x)[4] == context
            ]
            for label, context in itertools.product(range(2), repeat=2)
        }
        amount = config["eval_per_device"] // 4
        if amount * 4 != config["eval_per_device"]:
            raise ValueError("Evaluation size must be divisible by four")
        for (label, context), values in strata.items():
            keyed_rng(seed, device, "evaluation", label, context).shuffle(values)
            if len(values) < amount:
                raise ValueError("Insufficient held-out stratum")
            for value in values[:amount]:
                rows.append(
                    {
                        "id": f"s{stage}-d{device}-x{value}",
                        "stage": stage,
                        "device": device,
                        "profile": PROFILES[device],
                        "input": value,
                        "label": label,
                        "context": context,
                        "kind": "single",
                        "devices": [device],
                    }
                )
    for pair in itertools.combinations(all_stage_devices(stage), 2):
        for label in range(2):
            values = [
                x for x in heldout if rules[pair[0]](x) ^ rules[pair[1]](x) == label
            ]
            keyed_rng(seed, pair, "composition", label).shuffle(values)
            amount = config["compositions_per_pair"] // 2
            for value in values[:amount]:
                rows.append(
                    {
                        "id": f"s{stage}-p{pair[0]}{pair[1]}-x{value}",
                        "stage": stage,
                        "device": None,
                        "profile": "composition",
                        "input": value,
                        "label": label,
                        "context": bits(value)[4],
                        "kind": "composition",
                        "devices": list(pair),
                    }
                )
    return rows


def prompt(row, evidence=(), specification=None):
    devices = row.get("devices", [row["device"]])
    text = GRAMMAR + "\n"
    if specification is not None:
        text += (
            "\n".join(
                f"Device {device} rule: {specification[device].text()}."
                for device in devices
            )
            + "\n"
        )
    selected = [example for example in evidence if example["device"] in devices]
    if selected:
        text += "Calibration observations (some readings may be noisy):\n"
        text += (
            "\n".join(
                f"Device {example['device']}: {bit_text(example['input'])} -> {example['observed']}"
                for example in selected
            )
            + "\n"
        )
    if len(devices) == 1:
        text += f"Device {devices[0]}. Input b0..b9: {bit_text(row['input'])}\nOutput:"
    else:
        text += f"For the same input, evaluate devices {devices[0]} and {devices[1]}, then XOR their two outputs. Input b0..b9: {bit_text(row['input'])}\nOutput:"
    return text


def manifest(seed, config):
    acquisition, heldout = partition(seed)
    return {
        "seed": seed,
        "grammar": GRAMMAR,
        "acquisition_inputs": acquisition,
        "heldout_inputs": heldout,
        "split_sha256": digest([acquisition, heldout]),
        "hypotheses": [asdict(rule) for rule in FAMILY],
        "hidden_rules_evaluator_only": [
            {device: asdict(rule) for device, rule in stage.items()}
            for stage in make_rules(seed)
        ],
        "evaluation": [evaluate_rows(seed, stage, config) for stage in range(2)],
        "profiles": PROFILES,
        "scope": "Closed-family procedural rule acquisition and parameter consolidation",
        "noise": "Only stage 1 device 3 has observation flips; identical query-index flip masks across arms. Held-out targets are clean.",
        "split_counts": {"acquisition": len(acquisition), "heldout": len(heldout)},
        "profile_counts": dict(Counter(PROFILES)),
    }
