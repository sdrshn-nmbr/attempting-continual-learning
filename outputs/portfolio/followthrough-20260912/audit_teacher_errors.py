import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from generated_contract import make_corpus, qualification_rows, source_hashes

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent


def main():
    protocol = json.loads((ROOT / "experiments/distillation/configs/generated_protocol.json").read_text())
    corpus = make_corpus(protocol)
    runs = {
        "instruction_demo": ("distillation-generated-qualification-20260911", "validation"),
        "trained_teacher": ("distillation-generated-teacher-training-20260911", "fallback_validation"),
    }
    result = {"kind": "post_hoc_teacher_error_decomposition", "boundary": "Diagnostics of saved qualification generations, not new outcomes or a changed qualification rule. Per-position and prefix measures are descriptive only.", "candidates": {}}
    for candidate, (identity, split) in runs.items():
        root = ROOT / "outputs/portfolio/resume-20260911/runs" / identity
        receipt = json.loads((root / "qualification.json").read_text())
        if receipt["source_sha256"] != source_hashes():
            raise ValueError("TEACHER_DIAGNOSTIC_SOURCE_MISMATCH")
        raw = (root / "pairs.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["files_sha256"]["pairs.json"]:
            raise ValueError("TEACHER_DIAGNOSTIC_ARTIFACT_MISMATCH")
        records = json.loads(raw)
        tasks = qualification_rows(corpus, protocol, split)
        groups = defaultdict(list)
        for task, pair in zip(tasks, records, strict=True):
            if task.uid != pair["uid"]:
                raise ValueError("TEACHER_DIAGNOSTIC_ROW_BINDING")
            row = pair["teacher"]
            expected = " ".join(map(str, task.answer))
            terminated = bool(row["token_ids"]) and row["token_ids"][-1] == receipt["eos_token_id"]
            formatted = bool(re.fullmatch(r"[0-9](?: [0-9]){3}", row["body_text"]))
            correct = formatted and terminated and row["body_text"] == expected
            if correct != row["correct"]:
                raise ValueError("TEACHER_DIAGNOSTIC_GRADING_MISMATCH")
            prefix = re.match(r"([0-9]) ([0-9]) ([0-9]) ([0-9])", row["body_text"])
            predicted = tuple(map(int, prefix.groups())) if prefix else ()
            record = {
                "correct": correct, "format_valid": formatted, "terminated": terminated,
                "wrong_state_with_valid_format_and_eos": formatted and terminated and not correct,
                "correct_four_digit_prefix_but_invalid_completion": row["body_text"].startswith(expected) and not correct,
                "per_position_correct": [bool(predicted) and predicted[i] == task.answer[i] for i in range(4)],
                "returned_input_unchanged": row["body_text"] == " ".join(map(str, task.initial)),
            }
            groups[f"{task.split}/{task.family}"].append(record)
            groups[f"{task.split}/depth{len(task.program)}"].append(record)
        summary = {}
        for key, rows in groups.items():
            summary[key] = {"n": len(rows), **{metric: sum(row[metric] for row in rows) for metric in rows[0] if metric != "per_position_correct"}, "per_position_correct": [sum(row["per_position_correct"][i] for row in rows) for i in range(4)]}
        result["candidates"][candidate] = {"source": str(root), "pairs_sha256": hashlib.sha256(raw).hexdigest(), "panels": summary}
    (OUTPUT / "teacher-error-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["panels"] for key, value in result["candidates"].items()}))


if __name__ == "__main__":
    main()
