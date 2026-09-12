import hashlib
import itertools
import json
import random
import re
from pathlib import Path

from tasks import (
    FAMILIES,
    OPS,
    RULES,
    audit_dataset,
    digits,
    make_task,
    paired_gate,
    student_prompt,
)

CONTRACT = "device_rules_whole_answer_native_eos_20260911"
SPLITS = (
    "demonstration",
    "train",
    "validation",
    "test",
    "composition",
    "fallback_validation",
)
SOURCE_FILES = (
    "generated_contract.py",
    "qualified.py",
    "run.py",
    "tasks.py",
    "objectives.py",
    "requirements.txt",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    return {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES}


def validate_protocol(protocol):
    if protocol["contract"] != CONTRACT:
        raise ValueError("QUALIFY_UNKNOWN_CONTRACT")
    if (
        re.fullmatch(r"[0-9a-f]{40}", protocol["model_revision"]) is None
        or Path(protocol["model_path"]).name != protocol["model_revision"]
    ):
        raise ValueError("QUALIFY_PINNED_SNAPSHOT_REQUIRED")
    if protocol["family_order"] != list(FAMILIES):
        raise ValueError("QUALIFY_FAMILY_ORDER")
    if set(protocol["split_seeds"]) != set(SPLITS) or len(
        set(protocol["split_seeds"].values())
    ) != len(SPLITS):
        raise ValueError("QUALIFY_INDEPENDENT_SPLIT_SEEDS_REQUIRED")
    if set(protocol["sizes"]) != set(SPLITS) or protocol["sizes"]["demonstration"] != 6:
        raise ValueError("QUALIFY_SPLIT_SIZES")
    if any(
        not isinstance(n, int) or not 1 <= n <= 512 for n in protocol["sizes"].values()
    ):
        raise ValueError("QUALIFY_INVALID_COUNT")
    gate = protocol["qualification"]
    if gate != {
        "train_per_family": 24,
        "validation_per_family": 24,
        "minimum_accuracy": 0.875,
        "minimum_gain": 0.25,
        "maximum_p": 0.05,
    }:
        raise ValueError("QUALIFY_FIXED_GATE_CHANGED")
    if min(protocol["sizes"]["train"], protocol["sizes"]["validation"]) < 24:
        raise ValueError("QUALIFY_GATE_TOO_SMALL")
    if protocol["training"]["methods"] != ["privileged_forward_kl", "sft"]:
        raise ValueError("QUALIFY_MATCHED_SFT_CONTROL_REQUIRED")
    if (
        protocol["updates_per_family"] * protocol["examples_per_update"]
        != protocol["sizes"]["train"]
    ):
        raise ValueError("QUALIFY_EQUAL_UPDATE_BUDGET")
    if not 8 <= protocol["max_new_tokens"] <= 64:
        raise ValueError("QUALIFY_GENERATION_CAP")
    fallback = protocol["teacher_training"]
    if (
        fallback["updates"] != 24
        or fallback["examples_per_update"] != 4
        or fallback["train_per_family"] != 32
        or fallback["learning_rate"] != 0.0002
        or protocol["sizes"]["fallback_validation"] != 24
    ):
        raise ValueError("QUALIFY_FIXED_TEACHER_TRAINING_BUDGET")


def make_corpus(protocol, splits=SPLITS):
    pools = {split: [] for split in SPLITS}
    for value in range(10000):
        key = f"{protocol['partition_seed']}:{value:04d}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(SPLITS)
        pools[SPLITS[bucket]].append(value)
    corpus = {split: [] for split in splits}
    for split in splits:
        for family in FAMILIES:
            rng = random.Random(f"{protocol['split_seeds'][split]}:{family}")
            inputs = rng.sample(pools[split], protocol["sizes"][split])
            depths = (3, 4) if split == "composition" else (1, 2)
            programs = [
                p for depth in depths for p in itertools.product(OPS, repeat=depth)
            ]
            rng.shuffle(programs)
            for index, value in enumerate(inputs):
                program = (
                    (OPS[index // 2],)
                    if split == "demonstration"
                    else programs[index % len(programs)]
                )
                initial = tuple(map(int, f"{value:04d}"))
                corpus[split].append(make_task(family, split, initial, program))
    audit_dataset(corpus)
    return corpus


def learner_prompt(task):
    return (
        student_prompt(task) + "\nEnd your answer immediately after the fourth digit. "
        "Do not add a newline or any other text."
    )


def teacher_prompt(task, demonstrations):
    rules = "\n".join(f"{op}: {RULES[task.family][op]}." for op in OPS)
    examples = [item for item in demonstrations if item.family == task.family]
    if len(examples) != 6 or any(item.initial == task.initial for item in examples):
        raise ValueError("QUALIFY_DEMONSTRATION_BOUNDARY")
    worked = "\n".join(
        f"Input {digits(item.initial)}; command {item.program[0]}; "
        f"final {digits(item.answer)}."
        for item in examples
    )
    return (
        f"Device {task.family} operation instructions:\n{rules}\n"
        "Apply each command to the current state, in the listed order. "
        "Each result becomes the next command's input.\n"
        f"Verified examples on separate inputs:\n{worked}\n\n" + learner_prompt(task)
    )


def grade_generation(task, token_ids, body_text, eos, special_ids, cap):
    terminated = bool(token_ids) and token_ids[-1] == eos
    body = token_ids[:-1] if terminated else token_ids
    clean_tokens = not any(token in special_ids for token in body)
    format_valid = re.fullmatch(r"[0-9](?: [0-9]){3}", body_text) is not None
    return {
        "correct": len(token_ids) <= cap
        and terminated
        and clean_tokens
        and format_valid
        and body_text == digits(task.answer),
        "format_valid": format_valid and clean_tokens,
        "terminated": terminated,
        "cap_without_eos": len(token_ids) >= cap and not terminated,
    }


def qualification_rows(corpus, protocol, validation_split="validation"):
    return [
        task
        for split in ("train", validation_split)
        for family in FAMILIES
        for task in [t for t in corpus[split] if t.family == family][
            : protocol["qualification"][
                "train_per_family" if split == "train" else "validation_per_family"
            ]
        ]
    ]


def qualification_gate(
    records, corpus, protocol, eos, special_ids, validation_split="validation"
):
    tasks = qualification_rows(corpus, protocol, validation_split)
    if [row["uid"] for row in records] != [task.uid for task in tasks]:
        raise ValueError("QUALIFY_INCOMPLETE_OR_REORDERED_PAIRS")
    panels = {}
    for task, record in zip(tasks, records, strict=True):
        expected_prompts = {
            "learner": learner_prompt(task),
            "teacher": teacher_prompt(task, corpus["demonstration"]),
        }
        panel = panels.setdefault(
            f"{task.family}/{task.split}", {"learner": [], "teacher": []}
        )
        for condition, prompt in expected_prompts.items():
            row = record[condition]
            if (
                row["prompt"] != prompt
                or len(row["token_ids"]) > protocol["max_new_tokens"]
            ):
                raise ValueError("QUALIFY_TRACE_CONTRACT_MISMATCH")
            grade = grade_generation(
                task,
                row["token_ids"],
                row["body_text"],
                eos,
                special_ids,
                protocol["max_new_tokens"],
            )
            panel[condition].append(grade["correct"])
    gate = protocol["qualification"]
    results = {
        key: paired_gate(
            panel["learner"],
            panel["teacher"],
            gate["minimum_accuracy"],
            gate["minimum_gain"],
            gate["maximum_p"],
        )
        for key, panel in panels.items()
    }
    return {
        "passed": all(result["passed"] for result in results.values()),
        "panels": results,
    }


def verify_qualification(folder, protocol, required_status="qualified"):
    receipt = json.loads((folder / "qualification.json").read_text())
    if (
        receipt["protocol_sha256"] != digest(protocol)
        or receipt["source_sha256"] != source_hashes()
    ):
        raise ValueError("QUALIFY_RECEIPT_PROTOCOL_OR_CODE_MISMATCH")
    if (
        receipt["status"] != required_status
        or receipt["qualification_optimizer_updates"] != 0
        or receipt["learner_optimizer_updates"] != 0
        or receipt["device"] != "cuda:0"
        or not receipt["hip"]
    ):
        raise ValueError("QUALIFY_POSITIVE_GPU_RECEIPT_REQUIRED")
    candidate = receipt["candidate"]
    if candidate not in ("instruction_demo", "trained_teacher"):
        raise ValueError("QUALIFY_UNKNOWN_TEACHER_CANDIDATE")
    expected_updates = (
        0
        if candidate == "instruction_demo"
        else protocol["teacher_training"]["updates"]
    )
    if receipt["optimizer_updates"] != expected_updates:
        raise ValueError("QUALIFY_TEACHER_UPDATE_BUDGET_MISMATCH")
    for name, expected in receipt["files_sha256"].items():
        if (
            name not in ("pairs.json", "seal.json", "runtime.json")
            or file_digest(folder / name) != expected
        ):
            raise ValueError("QUALIFY_RECEIPT_ARTIFACT_MISMATCH")
    if set(receipt["files_sha256"]) != {"pairs.json", "seal.json", "runtime.json"}:
        raise ValueError("QUALIFY_RECEIPT_ARTIFACT_MISSING")
    seal = json.loads((folder / "seal.json").read_text())
    if (
        seal["protocol_sha256"] != digest(protocol)
        or seal["protocol"] != protocol
        or seal["source_sha256"] != source_hashes()
        or seal["prediction_outcomes_observed"]
    ):
        raise ValueError("QUALIFY_PREOUTCOME_SEAL_MISMATCH")
    records = json.loads((folder / "pairs.json").read_text())
    validation_split = (
        "validation" if candidate == "instruction_demo" else "fallback_validation"
    )
    corpus = make_corpus(protocol, ("demonstration", "train", validation_split))
    gate = qualification_gate(
        records,
        corpus,
        protocol,
        receipt["eos_token_id"],
        receipt["special_token_ids"],
        validation_split,
    )
    if (
        gate["passed"] != (required_status == "qualified")
        or gate != receipt["gate"]
        or receipt["adapter_before_sha256"] != receipt["adapter_after_sha256"]
    ):
        raise ValueError("QUALIFY_RECOMPUTED_GATE_FAILED")
    if candidate == "trained_teacher":
        checkpoint = folder / "teacher-checkpoint" / "teacher"
        actual = {
            path.name: file_digest(path)
            for path in checkpoint.iterdir()
            if path.is_file()
        }
        if actual != receipt["teacher_files_sha256"]:
            raise ValueError("QUALIFY_TRAINED_TEACHER_CHECKPOINT_CHANGED")
        if (
            file_digest(folder / "teacher_training.json")
            != receipt["teacher_training_sha256"]
        ):
            raise ValueError("QUALIFY_TEACHER_TRAINING_RECEIPT_CHANGED")
    return {
        **receipt,
        "pairs": records,
        "qualification_dir": str(folder),
        "receipt_sha256": file_digest(folder / "qualification.json"),
    }
