import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass

FAMILIES = ("A", "B", "C")
OPS = ("dax", "wug", "zup")
RULES = {
    "A": {
        "dax": "rotate the four digits left by one position",
        "wug": "swap the first and last digits; keep the middle digits",
        "zup": "reverse the order of all four digits",
    },
    "B": {
        "dax": "add 2 to each digit, reducing each result modulo 10",
        "wug": "multiply each digit by 3, reducing each result modulo 10",
        "zup": "replace each digit d with 9 minus d",
    },
    "C": {
        "dax": "if the first digit is even, reverse all four digits; otherwise rotate left by one",
        "wug": "if the last digit is even, swap the first two digits; otherwise swap the last two digits",
        "zup": "add the first digit to the last digit modulo 10; keep the first three digits",
    },
}


@dataclass(frozen=True)
class Task:
    uid: str
    family: str
    split: str
    initial: tuple[int, ...]
    program: tuple[str, ...]
    answer: tuple[int, ...]


def execute(family, initial, program):
    state = list(initial)
    for op in program:
        a, b, c, d = state
        if family == "A":
            state = {"dax": [b, c, d, a], "wug": [d, b, c, a], "zup": [d, c, b, a]}[op]
        elif family == "B":
            state = {
                "dax": [(x + 2) % 10 for x in state],
                "wug": [(x * 3) % 10 for x in state],
                "zup": [9 - x for x in state],
            }[op]
        elif family == "C":
            state = {
                "dax": [d, c, b, a] if a % 2 == 0 else [b, c, d, a],
                "wug": [b, a, c, d] if d % 2 == 0 else [a, b, d, c],
                "zup": [a, b, c, (a + d) % 10],
            }[op]
        else:
            raise ValueError(f"unknown family: {family}")
    return tuple(state)


def digits(values):
    return " ".join(map(str, values))


def make_task(family, split, initial, program):
    content = json.dumps([family, list(initial), list(program)], separators=(",", ":"))
    uid = hashlib.sha256(content.encode()).hexdigest()
    return Task(
        uid,
        family,
        split,
        tuple(initial),
        tuple(program),
        execute(family, initial, program),
    )


def make_dataset(seed, sizes):
    rng = random.Random(seed)
    corpus = {split: [] for split in ("demonstration", *sizes)}
    for family in FAMILIES:
        inputs = list(range(10000))
        rng.shuffle(inputs)
        cursor = 0
        for op in OPS:
            for _ in range(2):
                initial = tuple(int(x) for x in f"{inputs[cursor]:04d}")
                cursor += 1
                corpus["demonstration"].append(
                    make_task(family, "demonstration", initial, (op,))
                )
        for split, count in sizes.items():
            for _ in range(count):
                initial = tuple(int(x) for x in f"{inputs[cursor]:04d}")
                cursor += 1
                depth = rng.choice((3, 4) if split == "composition" else (1, 2))
                program = tuple(rng.choice(OPS) for _ in range(depth))
                corpus[split].append(make_task(family, split, initial, program))
    audit_dataset(corpus)
    return corpus


def audit_dataset(corpus):
    seen_ids = set()
    seen_inputs = set()
    for split, tasks in corpus.items():
        for task in tasks:
            if task.uid in seen_ids or (task.family, task.initial) in seen_inputs:
                raise ValueError(f"DISTILL_SPLIT_LEAK: {task.uid}")
            if (
                task.split != split
                or execute(task.family, task.initial, task.program) != task.answer
            ):
                raise ValueError(f"DISTILL_ORACLE_MISMATCH: {task.uid}")
            seen_ids.add(task.uid)
            seen_inputs.add((task.family, task.initial))
    return {"tasks": len(seen_ids), "exact_task_overlap": 0, "family_input_overlap": 0}


def student_prompt(task):
    return (
        f"You operate device {task.family}. Its command conventions remain fixed across requests.\n"
        f"Initial state: {digits(task.initial)}\n"
        f"Commands (apply from left to right): {' '.join(task.program)}\n"
        "Return the final state as exactly four digits separated by single spaces, with no explanation."
    )


def demo_context(task, demonstrations):
    eligible = [
        item
        for item in demonstrations
        if item.family == task.family
        and item.initial != task.initial
        and item.answer != task.answer
        and item.initial != task.answer
    ]
    by_op = {op: [item for item in eligible if item.program == (op,)] for op in OPS}
    lines = ["Verified worked examples from other inputs on this device:"]
    used = []
    for op in dict.fromkeys(task.program):
        if not by_op[op]:
            raise ValueError(f"DISTILL_DEMO_COVERAGE: {task.uid} {op}")
        for example in by_op[op]:
            lines.append(
                f"Input {digits(example.initial)}; command {op}; final {digits(example.answer)}. "
                f"The demonstrated operation is: {RULES[task.family][op]}."
            )
            used.append(example.uid)
    lines.append(
        "Apply the demonstrated conventions to the original request; output only its final state."
    )
    return "\n".join(lines), used


def parse_answer(text):
    clean = text.strip()
    if re.fullmatch(r"[0-9](?: [0-9]){3}", clean) is None:
        return None
    return tuple(int(value) for value in clean.split())


def score(task, text):
    parsed = parse_answer(text)
    return {
        "correct": parsed == task.answer,
        "format_valid": parsed is not None,
        "digit_accuracy": sum(a == b for a, b in zip(parsed, task.answer)) / 4
        if parsed
        else 0.0,
    }


def feedback_context(task, attempt):
    result = score(task, attempt)
    if result["correct"]:
        feedback = "The verifier accepted your previous final state."
    elif not result["format_valid"]:
        feedback = "The verifier rejected the response format: expected exactly four space-separated digits."
    else:
        incorrect = [
            str(i + 1)
            for i, (a, b) in enumerate(zip(parse_answer(attempt), task.answer))
            if a != b
        ]
        feedback = f"The verifier rejected final-state positions {', '.join(incorrect)}; expected digits are withheld."
    rules = "\n".join(
        f"Device trace contract for {op}: {RULES[task.family][op]}."
        for op in dict.fromkeys(task.program)
    )
    return (
        f"Your previous attempt was: {attempt}\n{feedback}\n{rules}\n"
        "Use this verifier feedback to solve the original request. Return only its final state."
    )


def to_records(corpus):
    return {split: [asdict(task) for task in tasks] for split, tasks in corpus.items()}


def paired_gate(student, teacher, minimum_accuracy, minimum_gain, maximum_p):
    n = len(student)
    wins = sum(t and not s for s, t in zip(student, teacher))
    losses = sum(s and not t for s, t in zip(student, teacher))
    discordant = wins + losses
    p = (
        sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
        / 2**discordant
        if discordant
        else 1.0
    )
    accuracy = sum(teacher) / n
    gain = (sum(teacher) - sum(student)) / n
    return {
        "n": n,
        "student_accuracy": sum(student) / n,
        "teacher_accuracy": accuracy,
        "gain": gain,
        "wins": wins,
        "losses": losses,
        "paired_one_sided_p": p,
        "passed": accuracy >= minimum_accuracy
        and gain >= minimum_gain
        and p <= maximum_p,
    }
