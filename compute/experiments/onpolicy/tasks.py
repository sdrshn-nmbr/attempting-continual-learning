import hashlib
import itertools
import json
import random
import re
from dataclasses import asdict, dataclass

TASKS = ("permutation", "symbol_map")
OPERATIONS = {"permutation": ("dax", "wug"), "symbol_map": ("fep", "kiv")}
RULES = {
    "dax": "Reverse the order of the list.",
    "wug": "Rotate the list one position to the left: move its first item to the end.",
    "fep": "Replace each item x by (x + 1) modulo 4; preserve item order.",
    "kiv": "Replace each item using 0->1, 1->0, 2->3, 3->2; preserve item order.",
}
FORMAT = (
    "Execute the named operations on the input list, in the given order. "
    "Return only the final list as a JSON array of integers from 0 through 3. "
    "Do not explain or repeat the input."
)
NUMBERED_FORMAT = (
    "Execute the numbered list program in order: step 1, then step 2, and so on. "
    "Start with the Input list. Each step takes the result of the preceding step, "
    "not the original Input. Apply every numbered operation exactly once. "
    "Return only the final list after the last step as a JSON array of integers from 0 through 3. "
    "Do not print intermediate lists, explain, or repeat the input."
)
CALIBRATION_VARIANTS = ("current", "numbered")
CALIBRATION_DEV_START = 64
TRAIN_ANSWER_TEACHER_TEMPLATE = (
    "{question}\n\nThis is an example for a response to the question:\n{answer}"
    "\n\nNow answer with a response of your own. Return only the final JSON array:"
)
REFUSAL = re.compile(
    r"\b(cannot|can't|unable|sorry|refuse|don't know|do not know|not defined|unknown operation)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Example:
    task: str
    split: str
    inputs: tuple[int, ...]
    program: tuple[str, ...]

    @property
    def key(self):
        return f"{self.task}/{self.split}/{'-'.join(self.program)}/{''.join(map(str, self.inputs))}"

    @property
    def answer(self):
        return json.dumps(execute(self.inputs, self.program), separators=(",", ":"))


def execute(inputs, program):
    value = list(inputs)
    for operation in program:
        if operation == "dax":
            value = value[::-1]
        elif operation == "wug":
            value = value[1:] + value[:1]
        elif operation == "fep":
            value = [(x + 1) % 4 for x in value]
        elif operation == "kiv":
            value = [x ^ 1 for x in value]
        else:
            raise ValueError(f"Unknown operation: {operation}")
    return value


def stable_seed(seed, *parts):
    encoded = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**31)


def input_pool(seed):
    pool = [x for x in itertools.product(range(4), repeat=4) if len(set(x)) > 1]
    random.Random(stable_seed(seed, "inputs")).shuffle(pool)
    return pool


def build_data(seed, train_examples=16, eval_examples=8, gate_examples=4):
    count = 4 + train_examples + eval_examples + gate_examples
    pool = input_pool(seed)
    if count > len(pool):
        raise ValueError("Requested input partitions exceed the finite task domain")
    partitions = {}
    offset = 0
    for split, size in (
        ("demo", 4),
        ("train", train_examples),
        ("gate", gate_examples),
        ("heldout", eval_examples),
    ):
        partitions[split] = pool[offset : offset + size]
        offset += size
    return task_data(partitions)


def build_calibration_data(seed, gate_examples):
    pool = input_pool(seed)
    return task_data(
        {
            "demo": pool[:4],
            "train": pool[4:20],
            "gate": pool[CALIBRATION_DEV_START : CALIBRATION_DEV_START + gate_examples],
        }
    )


def build_train_answer_data(seed):
    if seed != 37:
        raise ValueError("Train-answer calibration fixes input_pool seed at 37")
    return task_data({"train": input_pool(seed)[4:20]}, composition_splits=("train",))


def build_oracle_train_data(seed):
    if seed != 37:
        raise ValueError("Oracle control fixes input_pool seed at 37")
    return task_data({"train": input_pool(seed)[4:20]}, composition_splits=())


def build_oracle_eval_data(seed):
    if seed != 37:
        raise ValueError("Oracle control fixes input_pool seed at 37")
    return task_data({"heldout": input_pool(seed)[128:160]})


def oracle_batch(data, task, step, replay_examples=0):
    old_count = replay_examples if task == TASKS[1] else 0
    current_count = 4 - old_count
    current = [
        data[task]["train"][(current_count * step + offset) % 16]
        for offset in range(current_count)
    ]
    old = [
        data[TASKS[0]]["train"][(old_count * step + offset) % 16]
        for offset in range(old_count)
    ]
    return current + old


def oracle_train_manifest(data, seed, replay_examples=0):
    manifest = data_manifest(data, seed, 0)
    del manifest["sha256"]
    manifest["training_schedule"] = {
        task: [
            [example.key for example in oracle_batch(data, task, step, replay_examples)]
            for step in range(32)
        ]
        for task in TASKS
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def task_data(partitions, composition_splits=("gate", "heldout")):
    data = {}
    for task in TASKS:
        operations = OPERATIONS[task]
        data[task] = {}
        for split, inputs in partitions.items():
            data[task][split] = [
                Example(task, split, value, (operations[i % 2],)) for i, value in enumerate(inputs)
            ]
        for split in composition_splits:
            if split not in partitions:
                continue
            data[task][f"{split}_compositions"] = [
                Example(
                    task, f"{split}_compositions", value, operations if i % 2 else operations[::-1]
                )
                for i, value in enumerate(partitions[split])
            ]
    cross_programs = (("dax", "fep"), ("wug", "kiv"), ("fep", "dax", "kiv"))
    data["cross_task"] = {
        split: [
            Example("cross_task", split, value, cross_programs[i % len(cross_programs)])
            for i, value in enumerate(partitions[split])
        ]
        for split in composition_splits
        if split in partitions
    }
    assert_no_leaks(data)
    return data


def assert_no_leaks(data):
    groups = {
        name: set() for name in ("demo", "train", "gate", "heldout") if name in data[TASKS[0]]
    }
    for task in TASKS:
        for split in groups:
            groups[split].update(x.inputs for x in data[task][split])
        trained = {x.program for x in data[task].get("train", [])}
        for split in ("gate_compositions", "heldout_compositions"):
            if split in data[task] and trained & {x.program for x in data[task][split]}:
                raise ValueError("Composition target leaked into training programs")
    for left, right in itertools.combinations(groups, 2):
        if groups[left] & groups[right]:
            raise ValueError(f"Input leakage between {left} and {right}")


def query(example, instruction_variant="current"):
    if instruction_variant == "numbered":
        steps = []
        for index, name in enumerate(example.program, start=1):
            source = "the Input list" if index == 1 else f"the result of step {index - 1}"
            steps.append(f"{index}. Apply {name} to {source}.")
        return (
            f"Input: {json.dumps(example.inputs)}\n"
            "Program (execute every step in this order):\n" + "\n".join(steps) + "\nOutput:"
        )
    if instruction_variant != "current":
        raise ValueError(f"Unknown instruction variant: {instruction_variant}")
    return f"Operations: {', '.join(example.program)}\nInput: {json.dumps(example.inputs)}\nOutput:"


def prompt(example, data, context, privileged=False, instruction_variant="current"):
    if context == "train_answer":
        if (
            instruction_variant != "current"
            or example.split not in ("train", "train_compositions")
            or example.inputs not in input_pool(37)[4:20]
            or example not in data.get(example.task, {}).get(example.split, [])
        ):
            raise ValueError("Train-answer prompts require a member of the fixed train schedule")
        question = FORMAT + "\n\n" + query(example)
        if not privileged:
            return question
        return TRAIN_ANSWER_TEACHER_TEMPLATE.format(question=question, answer=example.answer)
    if context not in ("cue_only", "examples"):
        raise ValueError(f"Unknown context: {context}")
    if instruction_variant not in CALIBRATION_VARIANTS:
        raise ValueError(f"Unknown instruction variant: {instruction_variant}")
    sections = [NUMBERED_FORMAT if instruction_variant == "numbered" else FORMAT]
    tasks = TASKS if example.task == "cross_task" else (example.task,)
    if context == "examples":
        demonstrations = [x for task in tasks for x in data[task]["demo"]]
        if any(x.inputs == example.inputs for x in demonstrations):
            raise ValueError("Query input appeared in demonstrations")
        sections.append(
            "Examples:\n"
            + "\n\n".join(f"{query(x, instruction_variant)} {x.answer}" for x in demonstrations)
        )
    if privileged:
        rules = [f"{name}: {RULES[name]}" for task in tasks for name in OPERATIONS[task]]
        sections.append("Authoritative operation definitions:\n" + "\n".join(rules))
    sections.append(query(example, instruction_variant))
    return "\n\n".join(sections)


def grade(text, example):
    refusal = bool(REFUSAL.search(text))
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        parsed = None
    valid = (
        isinstance(parsed, list)
        and len(parsed) == len(example.inputs)
        and all(type(x) is int and 0 <= x <= 3 for x in parsed)
    )
    return {
        "correct": bool(valid and parsed == execute(example.inputs, example.program)),
        "invalid": not valid,
        "refusal": refusal,
    }


def data_manifest(data, seed, steps_per_task):
    examples = {
        task: {split: [asdict(x) for x in rows] for split, rows in splits.items()}
        for task, splits in data.items()
    }
    schedule = {
        task: [data[task]["train"][i % len(data[task]["train"])].key for i in range(steps_per_task)]
        for task in TASKS
    }
    manifest = {"seed": seed, "examples": examples, "training_schedule": schedule}
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest
