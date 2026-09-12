import random
import string
from collections import Counter
from dataclasses import asdict, dataclass

from config import DataConfig, digest

GENERATOR_VERSION = "opaque-digit-transducers-1"
EVAL_SPLITS = ("id", "longer", "composed")
TRAIN_SPLITS = ("train", "reference")


@dataclass(frozen=True)
class Rule:
    task: int
    cue: str
    mapping: tuple[int, ...]
    reverse: bool

    def execute(self, values, applications=1):
        result = tuple(values)
        for _ in range(applications):
            result = tuple(self.mapping[value] for value in result)
            if self.reverse:
                result = result[::-1]
        return result


@dataclass(frozen=True)
class Example:
    example_id: str
    task: int
    split: str
    values: tuple[int, ...]
    applications: int
    prompt: str
    answer: str


@dataclass
class Dataset:
    rules: list[Rule]
    examples: list[Example]
    seed: int
    config: DataConfig

    def select(self, task, split):
        return [
            example
            for example in self.examples
            if example.task == task and example.split == split
        ]

    def manifest(self):
        return {
            "generator": GENERATOR_VERSION,
            "source": "newly sampled executable rules; no external corpus",
            "seed": self.seed,
            "config": asdict(self.config),
            "rules": [asdict(rule) for rule in self.rules],
            "examples": [asdict(example) for example in self.examples],
            "split_policy": {
                "raw_inputs": "globally unique across tasks and all splits",
                "reference": "disjoint training-only pool; previous tasks only",
                "calibration": "base headroom screening only; never used for gradients",
                "evaluation": "unseen inputs; longer lengths; two applications of the same device",
                "composed_intermediates": "absent from every single-application input",
                "task_cue": "sampled independently of rule parameters; never includes the rule",
            },
        }

    def fingerprint(self):
        return digest(self.manifest())


def make_dataset(config: DataConfig, seed: int):
    rule_rng = random.Random(f"rules:{seed}")
    cue_rng = random.Random(f"cues:{seed}")
    rules = []
    signatures = set()
    cues = set()
    for task in range(config.num_tasks):
        for _ in range(10000):
            cycle = rule_rng.sample(range(config.alphabet_size), config.alphabet_size)
            mapping = [0] * config.alphabet_size
            for left, right in zip(cycle, cycle[1:] + cycle[:1]):
                mapping[left] = right
            reverse = (
                bool(rule_rng.getrandbits(1)) if config.include_reversal else False
            )
            signature = (tuple(mapping), reverse)
            if signature not in signatures:
                break
        else:
            raise ValueError("[data] unable to sample enough distinct rules")
        signatures.add(signature)
        cue = "unit-" + "".join(cue_rng.choices(string.ascii_lowercase, k=8))
        if cue in cues:
            raise ValueError("[data] duplicate independently sampled cue")
        cues.add(cue)
        rules.append(Rule(task, cue, tuple(mapping), reverse))
    examples = []
    reserved = set()
    specs = (
        ("train", config.train_examples, config.min_length, config.max_length, 1),
        (
            "reference",
            config.reference_examples,
            config.min_length,
            config.max_length,
            1,
        ),
        (
            "calibration",
            config.calibration_examples,
            config.min_length,
            config.max_length,
            1,
        ),
        ("id", config.id_examples, config.min_length, config.max_length, 1),
        (
            "longer",
            config.longer_examples,
            config.longer_min_length,
            config.longer_max_length,
            1,
        ),
        ("composed", config.composed_examples, config.min_length, config.max_length, 2),
    )
    for split, count, lower, upper, applications in specs:
        for rule in rules:
            rng = random.Random(f"inputs:{seed}:{rule.task}:{split}")
            for index in range(count):
                length = lower + index % (upper - lower + 1)
                for _ in range(100000):
                    values = tuple(
                        rng.randrange(config.alphabet_size) for _ in range(length)
                    )
                    intermediate = rule.execute(values)
                    if values not in reserved and (
                        applications == 1 or intermediate not in reserved
                    ):
                        break
                else:
                    raise ValueError(
                        f"[data] exhausted unique length-{length} inputs for {split}"
                    )
                reserved.add(values)
                if applications == 2:
                    reserved.add(intermediate)
                instruction = (
                    "Apply the device once."
                    if applications == 1
                    else (
                        "Apply the device twice: feed its first output back into the same device."
                    )
                )
                prompt = (
                    f"Device: {rule.cue}\n{instruction}\n"
                    f"Input: {' '.join(map(str, values))}\n"
                    "Output only the resulting digits separated by spaces."
                )
                answer = " ".join(map(str, rule.execute(values, applications)))
                example_id = digest([seed, rule.task, split, values, applications])[:24]
                examples.append(
                    Example(
                        example_id,
                        rule.task,
                        split,
                        values,
                        applications,
                        prompt,
                        answer,
                    )
                )
    dataset = Dataset(rules, examples, seed, config)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset):
    inputs = [example.values for example in dataset.examples]
    ids = [example.example_id for example in dataset.examples]
    prompts = [example.prompt for example in dataset.examples]
    if (
        len(set(inputs)) != len(inputs)
        or len(set(ids)) != len(ids)
        or len(set(prompts)) != len(prompts)
    ):
        raise ValueError("[data] duplicate input, example ID or prompt across splits")
    single_inputs = {
        example.values for example in dataset.examples if example.applications == 1
    }
    for rule in dataset.rules:
        training = dataset.select(rule.task, "train")
        coverage = Counter(value for example in training for value in example.values)
        if set(coverage) != set(range(dataset.config.alphabet_size)):
            raise ValueError(
                f"[data] task {rule.task} training pool does not cover every symbol"
            )
    for example in dataset.examples:
        rule = dataset.rules[example.task]
        expected = " ".join(
            map(str, rule.execute(example.values, example.applications))
        )
        if example.answer != expected:
            raise ValueError(f"[data] incorrect oracle label: {example.example_id}")
        if example.applications == 2 and rule.execute(example.values) in single_inputs:
            raise ValueError(
                "[data] composed intermediate overlaps a single-application input"
            )


def step_examples(dataset, task, step, current_batch_size, reference_batch_size):
    current_pool = dataset.select(task, "train")
    offset = step * current_batch_size
    current = []
    for position in range(offset, offset + current_batch_size):
        epoch, index = divmod(position, len(current_pool))
        order = random.Random(f"current:{dataset.seed}:{task}:{epoch}").sample(
            current_pool, len(current_pool)
        )
        current.append(order[index])
    if task == 0:
        return current, []
    reference_pool = [
        example
        for example in dataset.examples
        if example.split == "reference" and example.task < task
    ]
    reference = random.Random(f"reference:{dataset.seed}:{task}:{step}").sample(
        reference_pool, reference_batch_size
    )
    return current, reference


def require_gradient_examples(examples, current_task, role):
    expected_split = "train" if role == "current" else "reference"
    if role not in {"current", "reference"} or not examples:
        raise ValueError("[gradient-data] unknown role or empty gradient batch")
    for example in examples:
        task_allowed = (
            example.task == current_task
            if role == "current"
            else example.task < current_task
        )
        if (
            example.split != expected_split
            or not task_allowed
            or example.applications != 1
        ):
            raise ValueError(
                f"[gradient-data] forbidden {role} example {example.example_id}: {example.split}"
            )
