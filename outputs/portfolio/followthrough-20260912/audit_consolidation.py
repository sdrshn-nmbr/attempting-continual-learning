import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(f"CONSOLIDATION_INDEPENDENT_AUDIT: {detail}")


def read(path):
    return json.loads(path.read_text())


def corpus_from_fixtures(repo):
    lane = repo / "experiments/distillation"
    protocol = read(lane / "configs/choice_consolidation_protocol.json")
    fixtures = {}
    for name, spec in protocol["fixtures"].items():
        path = lane / spec["path"]
        require(file_hash(path) == spec["sha256"], f"fixture hash {name}")
        fixtures[name] = read(path)
    original = fixtures["original"]
    corpus = {split: [r for task in ("sequence_a", "sequence_b", "sequence_c") for r in original["splits"][f"{task}_{split}"]] for split in ("train", "validation", "test")}
    corpus["unused288"] = fixtures["unused"]["rows"]
    sizes = {"train": 128, "validation": 32, "test": 64, "unused288": 288}
    previous = set()
    for split, rows in corpus.items():
        require(Counter(r["task"] for r in rows) == {task: sizes[split] for task in ("sequence_a", "sequence_b", "sequence_c")}, f"split sizes {split}")
        groups = {r["group"] for r in rows}
        require(not groups & previous, f"cross-split input overlap {split}")
        previous |= groups
        for row in rows:
            inputs = tuple(map(int, row["group"].split()))
            expected = " " + " ".join(str(original["provenance"]["rules"][row["task"]][i]) for i in inputs)
            require(row["choices"][row["gold_idx"]] == expected, f"oracle {row['id']}")
            require(row["id"] == digest({"task": row["task"], "input": inputs}), f"row id {row['id']}")
    return protocol, corpus


def summarize(flags, rows):
    return {task: {"correct": sum(v for v, r in zip(flags, rows, strict=True) if r["task"] == task), "count": sum(r["task"] == task for r in rows)} for task in sorted({r["task"] for r in rows})}


def check_rows(rows, records, tokenizer=None, eos=None, cap=None):
    require(len(rows) == len(records), "prediction count")
    flags = []
    for row, record in zip(rows, records, strict=True):
        require(all(record[k] == row[k] for k in ("id", "task", "group")), f"prediction identity {row['id']}")
        require(record["row_sha256"] == digest(row), f"prediction row hash {row['id']}")
        if tokenizer is None:
            scores = record["scores"]
            require(len(scores) == 4 and all(math.isfinite(s) for s in scores), "invalid choice scores")
            chosen = max(range(4), key=scores.__getitem__)
            correct = chosen == row["gold_idx"]
            require(record["prediction"] == chosen and record["gold"] == row["gold_idx"] and record["correct"] == correct, f"choice grading {row['id']}")
        else:
            generation = record["generation"]
            tokens = generation["token_ids"]
            terminated = bool(tokens) and tokens[-1] == eos
            body_tokens = tokens[:-1] if terminated else tokens
            body = tokenizer.decode(body_tokens, skip_special_tokens=False)
            require(body == generation["body_text"], f"decoded body {row['id']}")
            require(tokenizer.decode(tokens, skip_special_tokens=False) == generation["raw_text"], f"decoded raw text {row['id']}")
            require(tokenizer.encode(row["prompt"]).ids == generation["prompt_token_ids"], f"prompt token binding {row['id']}")
            valid = re.fullmatch(r" [0-7] [0-7] [0-7]", body) is not None
            correct = len(tokens) <= cap and terminated and valid and body == row["choices"][row["gold_idx"]]
            require(generation["terminated"] == terminated and generation["format_valid"] == valid and generation["correct"] == correct, f"whole generation grading {row['id']}")
        flags.append(correct)
    return flags


def completed_run(root):
    execution = read(root / "execution.json")
    require(execution["status"] == "completed" and execution["exit_code"] == 0, f"incomplete execution {root.name}")
    return execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--wave", type=Path, required=True)
    args = parser.parse_args()
    protocol, corpus = corpus_from_fixtures(args.repo)
    tokenizer_path = args.wave / "tokenizer/tokenizer.json"
    spec = next(s for s in protocol["source_base"]["files"] if s["path"] == "tokenizer.json")
    require(file_hash(tokenizer_path) == spec["sha256"], "tokenizer file hash")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<|im_end|>")
    require(eos == 151645, "native EOS identity")
    teacher_design = read(args.repo / "experiments/distillation/configs/generation_teacher_protocol.json")
    cap = teacher_design["generation"]["max_new_tokens"]
    result = {"status": "passed", "tokenizer_sha256": file_hash(tokenizer_path), "cross_split_input_overlap": 0, "claim_boundary": "Independent rescoring and artifact checks, not a rerun of model inference. Existing Sequence303 task mapping and historical evaluation panels; no independent task replication.", "runs": {}}
    definitions = [
        ("consolidation-choice303-evaluate-20260912", "panels.json", "choice"),
        ("consolidation-source303-generation-evaluate-20260912", "qualification_panels.json", "teacher"),
        ("consolidation-generated303-20260912", "evaluation/panels.json", "generated"),
        ("consolidation-coverage303-20260912", "evaluation/panels.json", "generated"),
    ]
    for name, panel_file, kind in definitions:
        root = args.wave / "runs" / name
        if not (root / panel_file).exists():
            result["runs"][name] = {"status": "not_collected_yet"}
            continue
        execution = completed_run(root)
        summary = read(root / ("qualification.json" if kind == "teacher" else "evaluation/result.json" if kind == "generated" else "result.json"))
        require(file_hash(root / panel_file) == summary["panels_sha256"], f"panel hash {name}")
        require(summary["pid"] != summary["training_pid"], f"distinct evaluation process {name}")
        if kind != "teacher":
            require(not summary["teacher_model_loaded"], f"teacher absent {name}")
            training_root = args.wave / "runs" / ("consolidation-choice303-train-20260912" if kind == "choice" else name)
            require(file_hash(training_root / "training.json") == summary["training_sha256"], f"training receipt hash {name}")
        audited = {}
        for split, conditions in read(root / panel_file).items():
            audited[split] = {}
            for condition, records in conditions.items():
                generation = kind == "generated" or (kind == "teacher" and condition == "generated")
                flags = check_rows(corpus[split], records, tokenizer if generation else None, eos, cap)
                audited[split][condition] = {"correct": sum(flags), "count": len(flags), "tasks": summarize(flags, corpus[split])}
        result["runs"][name] = {"status": "passed", "source_sha256": execution["source_sha256"], "execution_sha256": file_hash(root / "execution.json"), "panels_sha256": file_hash(root / panel_file), "new_process": True, "scores": audited}
        if name == "consolidation-coverage303-20260912":
            training = read(root / "training.json")
            previous = root.parent / "consolidation-generated303-20260912"
            previous_training = read(previous / "training.json")
            require(training["initial"]["tensor_sha256"] == previous_training["initial"]["tensor_sha256"], "matched initial learner")
            require(file_hash(root / "schedule.json") == file_hash(previous / "schedule.json"), "matched learner schedule")
            source = read(root / "teacher_qualification_panels.json")["train"]["generated"]
            require(all(check_rows(corpus["train"], source, tokenizer, eos, cap)), "qualified teacher complete TRAIN accuracy")
            require(training["teacher_targets_all_equal_oracle"] and training["teacher_targets_equal_oracle_count"] == len(corpus["train"]), "coverage target identity")
            require(summary["teacher_targets_all_equal_oracle"] and summary["selected_teacher_checkpoint"] == 384, "selected coverage teacher")
            require(not summary["teacher_artifact_files_read_during_evaluation"] and summary["resident_active_adapters"] == 1, "independent persistent learner")
            hashes = []
            for arm in training["arms"].values():
                require(arm["selected_checkpoint"] == 384, "fixed final learner")
                hashes.append(arm["checkpoints"]["384"]["adapter"]["tensor_sha256"])
            require(summary["final_arm_tensors_equal"] == (len(set(hashes)) == 1), "reported final tensor equality")
            result["runs"][name]["target_identity"] = {
                "all_384_teacher_targets_equal_oracle": True,
                "initial_and_schedule_match_previous_noisy_teacher_run": True,
                "final_arm_tensors_equal": len(set(hashes)) == 1,
                "interpretation": "Correct teacher outputs induce the same SFT objective as the oracle. This is a target-quality comparison with the earlier noisy teacher and a replication control, not a superior distillation objective.",
            }
    root = args.wave / 'runs/consolidation-source303-coverage-20260912'
    if (root / 'qualification/qualification.json').exists():
        execution = completed_run(root)
        screen = read(root / 'coverage.json')
        qualification = read(root / 'qualification/qualification.json')
        require(qualification['screen_sha256'] == file_hash(root/'coverage.json'), 'coverage screen binding')
        require(qualification['pid'] != screen['pid'] == qualification['screen_pid'], 'coverage process boundary')
        require(qualification['panels_sha256'] == file_hash(root/'qualification/qualification_panels.json'), 'coverage qualification binding')
        require(screen['teacher_optimizer_updates'] == screen['learner_updates'] == screen['test_predictions'] == screen['new_validation_predictions'] == 0, 'coverage selection split or updates')
        counts, passing = {}, []
        for step in (32,128,384):
            path = root/f'train{step}.json'
            candidate = screen['candidates'][str(step)]
            require(candidate['records_sha256'] == file_hash(path), 'coverage candidate binding')
            records = read(path)
            flags = check_rows(corpus['train'],records,tokenizer,eos,cap)
            cells = {}
            for row,record in zip(corpus['train'],records,strict=True):
                generation = record['generation']
                valid = generation['format_valid'] and generation['terminated']
                actual = generation['body_text'].split() if valid else []
                expected = row['choices'][row['gold_idx']].split()
                for position,digit in enumerate(row['group'].split()):
                    key = f'{row["task"]}/position{position+1}/input{digit}'
                    cell = cells.setdefault(key,{'n':0,'correct':0,'wrong_ids':[]})
                    correct = valid and actual[position] == expected[position]
                    cell['n'] += 1
                    cell['correct'] += int(correct)
                    if not correct:
                        cell['wrong_ids'].append(row['id'])
                    require(generation['digit_position_correct'][position] == correct, 'literal digit diagnostic')
            require(len(cells) == 72, 'coverage cell completeness')
            for key,cell in cells.items():
                saved = candidate['coverage']['cells'][key]
                require(all(saved[k] == v for k,v in cell.items()), 'coverage cell rescore')
            passed = all(v['n'] == v['correct'] for v in cells.values())
            require(candidate['coverage']['passed'] == passed and passed == all(flags), 'coverage gate')
            require(candidate['coverage']['whole_answers_correct'] == sum(flags), 'coverage whole answer count')
            counts[str(step)] = {'correct':sum(flags),'n':len(flags),'passed':passed}
            if passed:
                passing.append(step)
        require(screen['selected_checkpoint'] == min(passing) == qualification['selected_checkpoint'], 'earliest complete TRAIN selection')
        panels = read(root/'qualification/qualification_panels.json')
        require(panels['train']['generated'] == read(root/f'train{screen["selected_checkpoint"]}.json'), 'selected targets unchanged')
        qualification_counts = {}
        for split,conditions in panels.items():
            qualification_counts[split] = {}
            for kind,records in conditions.items():
                flags = check_rows(corpus[split],records,tokenizer if kind == 'generated' else None,eos,cap)
                groups = summarize(flags,corpus[split])
                require(all(v['correct']/v['count'] >= .9 for v in groups.values()), 'selected teacher qualification')
                qualification_counts[split][kind] = {'correct':sum(flags),'n':len(flags),'tasks':groups}
        original = args.wave/'runs/consolidation-source303-generation-evaluate-20260912/qualification_panels.json'
        require(read(root/'train32.json') == read(original)['train']['generated'], 'original32 records remain unchanged')
        result['runs'][root.name] = {'status':'passed','source_sha256':execution['source_sha256'],
            'execution_sha256':file_hash(root/'execution.json'),'candidate_TRAIN_counts':counts,
            'selected_checkpoint':screen['selected_checkpoint'],'qualification':qualification_counts,
            'claim_boundary':'Perfect coverage of these384TRAIN answers and96reusedvalidation answers; no TEST selection and no newly executed teacher optimization. Perfect generated targets equal the oracle SFT labels.'}
    destination = args.wave / "consolidation-audit.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
