import hashlib
import json
from collections import defaultdict

from common import asset_path, sha256_file, write_json
from portallib import ChoiceDataset, ChoiceExample
from pyarrow import parquet


def prompt_hash(row):
    text = row.prompt if isinstance(row, ChoiceExample) else row["prompt"]
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def prepare_dataset(manifest, root, config, output):
    asset = manifest["assets"][manifest["dataset"]]
    path = asset_path(root, asset)
    splits = {}
    for split in ("train", "validation"):
        files = [
            file for file in asset["files"] if file["path"].startswith(f"data/{split}-")
        ]
        rows = []
        for file in files:
            local = path / file["path"]
            if sha256_file(local) != file["sha256"]:
                raise ValueError(f"dataset checksum mismatch: {local}")
            rows.extend(parquet.read_table(local).to_pylist())
        splits[split] = rows
    validation_by_task = defaultdict(list)
    for global_index, row in enumerate(splits["validation"]):
        validation_by_task[row["task"]].append((global_index, row))
    prefix = config["excluded_validation_prefix"]
    training_hashes = {prompt_hash(row) for row in splits["train"]}
    selection_hashes = {
        prompt_hash(row)
        for rows in validation_by_task.values()
        for _, row in rows[:prefix]
    }
    excluded = training_hashes | selection_hashes
    selected_rows, selected_ids, counts = [], [], {}
    for task in config["tasks"]:
        candidates, seen = [], set()
        excluded_overlap = 0
        for task_index, (global_index, row) in enumerate(
            validation_by_task[task][prefix:], start=prefix
        ):
            digest = prompt_hash(row)
            if digest in excluded or digest in seen:
                excluded_overlap += 1
                continue
            seen.add(digest)
            key = hashlib.sha256(f"{config['seed']}:{digest}".encode()).hexdigest()
            candidates.append((key, global_index, task_index, row, digest))
        candidates.sort(key=lambda item: item[0])
        wanted = config["examples_per_task"]
        if len(candidates) < wanted:
            raise ValueError(
                f"insufficient disjoint holdout for {task}: {len(candidates)} < {wanted}"
            )
        for _, global_index, task_index, row, digest in candidates[:wanted]:
            selected_rows.append(row)
            selected_ids.append(
                {
                    "task": task,
                    "validation_index": global_index,
                    "within_task_index": task_index,
                    "prompt_sha256": digest,
                }
            )
        counts[task] = {
            "validation_total": len(validation_by_task[task]),
            "eligible_after_exclusions": len(candidates),
            "overlap_or_duplicate_excluded": excluded_overlap,
            "selected": wanted,
        }
    eval_hashes = {prompt_hash(row) for row in selected_rows}
    if (
        eval_hashes & training_hashes
        or eval_hashes & selection_hashes
        or len(eval_hashes) != len(selected_rows)
    ):
        raise ValueError("holdout disjointness invariant violated")
    selection = {
        "dataset_repo": asset["repo"],
        "dataset_revision": asset["revision"],
        "seed": config["seed"],
        "excluded_validation_prefix": prefix,
        "counts": counts,
        "train_prompt_overlap": 0,
        "checkpoint_selection_prefix_overlap": 0,
        "duplicate_eval_prompts": 0,
        "indices": selected_ids,
        "validation": selected_rows,
        "scope": "disjoint from released training rows and documented checkpoint selection prefix; unknown pretraining and unpublished selection exposure are not ruled out",
    }
    selection["selection_sha256"] = hashlib.sha256(
        json.dumps(selected_ids, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "selection.json", selection)
    dataset = ChoiceDataset(
        [
            ChoiceExample.from_dict(row)
            for row in splits["train"]
            if row["task"] in config["tasks"]
        ],
        [ChoiceExample.from_dict(row) for row in selected_rows],
    )
    return dataset, selection
