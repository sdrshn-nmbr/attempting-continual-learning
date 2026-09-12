import argparse
import copy
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from calibrate_target import file_pin
from data import SEQUENCE_TASKS, Example, digest, write_json
from follow_through import validate
from learner import tensor_hash
from representability import load_native

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sealed_length_holdout(fixture):
    seed = 2026091203
    inputs = list(itertools.product(range(8), repeat=4))
    random.Random(seed).shuffle(inputs)
    inputs = inputs[:128]
    rows = []
    for task in SEQUENCE_TASKS:
        rng = random.Random(int(digest([seed, task]), 16))
        positions = list(range(4)) * 32
        rng.shuffle(positions)
        for values, position in zip(inputs, positions, strict=True):
            group = " ".join(map(str, values))
            gold = " " + " ".join(
                str(fixture["provenance"]["rules"][task][v]) for v in values
            )
            distractors = set()
            while len(distractors) < 3:
                candidate = " " + " ".join(str(rng.randrange(8)) for _ in values)
                if candidate != gold:
                    distractors.add(candidate)
            choices = sorted(distractors)
            rng.shuffle(choices)
            choices.insert(position, gold)
            record = asdict(
                Example(
                    digest(["length4", task, values]),
                    task,
                    f"Apply the {task} code.\nInput: {group}\nOutput:",
                    tuple(choices),
                    position,
                    group,
                )
            )
            record["choices"] = list(record["choices"])
            rows.append(record)
    return {
        "kind": "sequence303_length4_holdout",
        "seed": seed,
        "rows": rows,
        "rows_sha256": digest(rows),
        "provenance": {
            "fresh_length_not_new_mapping": True,
            "training_length": 3,
            "evaluation_length": 4,
            "selected_before_follow_through_runs": True,
            "no_training_or_selection_access": True,
            "source_fixture_sha256": file_pin(ROOT / "data/sequence303.json")["sha256"],
            "claim_boundary": "Four-choice length generalization on known mappings. No free-generation or new-task-transfer claim.",
        },
    }


def arm(
    name,
    source,
    train,
    routing="fixed_rte",
    latents=False,
    rate=0.001,
    initialization="released",
):
    return {
        "name": name,
        "source": source,
        "train": train,
        "routing": routing,
        "train_latents": latents,
        "learning_rate": rate,
        "latent_learning_rate": 2 * rate,
        "initialization": initialization,
    }


def build_configs():
    previous = json.loads(
        (ROOT / "configs/target_calibration/qwen4_sequence303.json").read_text()
    )
    construction = json.loads(
        (ROOT / "configs/representability/constructed_behavior303.json").read_text()
    )
    fixture = json.loads((ROOT / "data/sequence303.json").read_text())
    selected_path = (
        REPO / "outputs/portfolio/runs/catalog-qwen3-8b-screen/selection.json"
    )
    saved_selection = json.loads(selected_path.read_text())
    retention_path = ROOT / "data/released_task_probes.json"
    if (
        retention_path.exists()
        and retention_path.read_bytes() != selected_path.read_bytes()
    ):
        raise ValueError("FOLLOW_THROUGH_RETENTION_SNAPSHOT_ALREADY_DIFFERS")
    retention_path.write_bytes(selected_path.read_bytes())
    holdout_path = ROOT / "data/sequence303_length4_holdout.json"
    holdout = sealed_length_holdout(fixture)
    if holdout_path.exists() and json.loads(holdout_path.read_text()) != holdout:
        raise ValueError("FOLLOW_THROUGH_HOLDOUT_ALREADY_DIFFERS")
    write_json(holdout_path, holdout)
    common = {
        "fixture": copy.deepcopy(previous["inputs"]["fixture"]),
        "old_holdout": copy.deepcopy(previous["inputs"]["fresh_fixture"]),
        "length_holdout": {
            "path": "data/sequence303_length4_holdout.json",
            **file_pin(holdout_path),
        },
        "retention_probes": {
            "path": "data/released_task_probes.json",
            **file_pin(retention_path),
        },
        "source_initial": {
            **copy.deepcopy(previous["inputs"]["source_initial"]),
            "tensor_sha256": previous["source_provenance"]["initial_tensor_sha256"],
        },
    }
    native_path = REPO / "outputs/representability/learned-alignment303-20260911/native"
    constructed = {
        "path": construction["native"]["path"],
        "tensor_sha256": construction["native"]["tensor_sha256"],
        "files": {
            name: file_pin(native_path / name)
            for name in ("config.json", "model.safetensors")
        },
    }
    if any(
        constructed["files"][name]["sha256"] != pin
        for name, pin in construction["native"]["files"].items()
    ):
        raise ValueError("FOLLOW_THROUGH_CONSTRUCTION_LOCAL_REMOTE_PIN_MISMATCH")
    target_path = (
        Path(
            json.loads(
                (ROOT / "configs/representability/source_rank8_303.json").read_text()
            )["native_initial"]["path"]
        ).parent
        / "qwen4"
    )
    target = copy.deepcopy(previous["inputs"]["target_portal"])
    target["tensor_sha256"] = tensor_hash(
        load_native(target_path, torch.float32).state_dict()
    )
    train128 = {
        task: [row["id"] for row in fixture["splits"][f"{task}_train"]]
        for task in SEQUENCE_TASKS
    }
    train16 = copy.deepcopy(previous["calibration_ids"])
    train64 = {
        task: train16[task]
        + [key for key in train128[task] if key not in train16[task]][:48]
        for task in SEQUENCE_TASKS
    }
    boundary = [
        "Exploratory follow-through on the existing Sequence-303 mapping. The historical 864 exhaustive-triple rows and old validation/test rows have already been evaluated.",
        "The sealed length4 panel is new length generalization, not a new mapping, free-generation result, or an unfamiliar semantic skill.",
        "All metrics use unchanged character-normalized choice likelihood and gold-answer token NLL; no prefix extraction or output-contract change.",
        "Checkpoints and budgets are predefined; validation records diagnose acquisition only. All final panels are first opened in a fresh PID after all arms have finished training. No best-checkpoint selection.",
        "Released-task retention uses the exact prior catalog selection: 32 each of BoolQ, HellaSwag, WinoGrande, CommonsenseQA. It supports no claim about unmeasured tasks.",
        "Native old-task evaluation routes its original frozen task vectors. Persistent LoRA applies one adapter to all tasks. Retention compares each arm with its own starting performance, not unmatched absolute cross-arm baselines.",
        "Rank8 q/v, alpha16, training examples and updates are matched; trainable parameter counts are reported and are not equal.",
    ]

    def config(mode, ids, checkpoints, arms):
        protocol = {
            "mode": mode,
            "scoring": "character_normalized_four_choice_gold_answer_nll",
            "training_ids": ids,
            "checkpoints": checkpoints,
            "arms": arms,
            "model_seed": 912303,
            "data_seed": 83303,
            "batch_per_task": 4,
            "max_prompt": 768,
            "eval_batch": 4,
            "gradient_clip": 1.0,
            "gradient_gate_step": 8,
            "acquisition_train_floor": 0.9,
            "acquisition_validation_floor": 0.85,
            "retention_indices_sha256": digest(saved_selection["indices"]),
            "continuation": {
                "finite_improving_but_not_qualified": "complete_all_predefined_checkpoints",
                "zero_or_nonfinite_gradient": "stop_that_arm_before_more_updates_and_record_diagnostic_then_continue_other_arms",
                "last_checkpoint_underfits_train": "inspect_loss_gradient_and_scale_before_a_separately_sealed_recipe",
                "train_pass_validation_fail": "do_not_extend_same_fit_blindly_test_more_unique_train_rows",
                "successful_acquisition": "retention_and_transfer_are_separate_required_tests",
            },
        }
        result = {
            "kind": "portal_follow_through",
            "role": "mechanistic_exploratory",
            "protocol": protocol,
            "protocol_sha256": digest(protocol),
            "inputs": copy.deepcopy(common),
            "base": copy.deepcopy(previous["base"]),
            "runtime": copy.deepcopy(previous["runtime"]),
            "expected_hooks": 72,
            "claim_boundary": list(boundary),
            "method_source": {
                "url": "https://labs.ramp.com/research/portal-portable-task-adaptation/",
                "source": "https://github.com/ramp-public/portallib",
                "verified": "2026-09-12",
                "source_objective": "Balanced gold-answer token NLL with EMA normalization; train task vectors and decoder on a frozen base.",
                "transfer": "Freeze core and all task vectors; fresh target input/output maps and layer embeddings. Longer curves supplement the published five-epoch point.",
            },
            "retention_provenance": {
                "original_path": str(selected_path),
                "original_file": file_pin(selected_path),
                "dataset": saved_selection["dataset_repo"],
                "revision": saved_selection["dataset_revision"],
                "selection_sha256": saved_selection["selection_sha256"],
            },
        }
        if mode == "calibration":
            result["inputs"].update(
                source_constructed=constructed,
                target_portal=target,
                base=copy.deepcopy(previous["inputs"]["target_base"]),
            )
            result["claim_boundary"].append(
                "Constructed and untouched sources BOTH use the same fixed original RTE vector for ABC. The successful reconstruction was obtained with privileged source adapter weights; this tests transfer of that representation, not learning it from examples."
            )
        else:
            result["base"] = {
                k: v for k, v in construction["source_base"].items() if k != "files"
            }
            result["inputs"]["base"] = {
                "path": result["base"]["local_path"],
                "files": {
                    p["path"]: {k: v for k, v in p.items() if k != "path"}
                    for p in construction["source_base"]["files"]
                },
            }
            if mode == "retention":
                result["inputs"]["source_constructed"] = constructed
                result["claim_boundary"].append(
                    "Zero optimizer updates. Compare old-task behavior of the exact known-weight construction against untouched released task adapters; this is an insertion/retention audit only."
                )
            else:
                result["claim_boundary"].append(
                    "The examples-only trainer never loads the successful adapter or constructed generator. Its only ABC supervision is existing TRAIN answers; release weights initialize generic components."
                )
        validate(result)
        return result

    configs = {
        "constructed_target16": config(
            "calibration",
            train16,
            [0, 20, 80, 160, 320],
            [
                arm("constructed", "source_constructed", "alignment"),
                arm("untouched", "source_initial", "alignment"),
                arm("lora", "source_initial", "lora", "persistent"),
            ],
        ),
        "constructed_target64": config(
            "calibration",
            train64,
            [0, 20, 80, 160, 320],
            [
                arm("constructed", "source_constructed", "alignment"),
                arm("untouched", "source_initial", "alignment"),
                arm("lora", "source_initial", "lora", "persistent"),
            ],
        ),
        "examples_released": config(
            "examples",
            train128,
            [0, 16, 64, 128, 256, 512],
            [
                arm("heads_fixed_rte", "source_initial", "heads", rate=0.0001),
                arm(
                    "heads_new_latents",
                    "source_initial",
                    "heads",
                    "task_latents",
                    True,
                    rate=0.0001,
                ),
                arm(
                    "full_new_latents",
                    "source_initial",
                    "full",
                    "task_latents",
                    True,
                    rate=0.0001,
                ),
                arm(
                    "lora_rte_start",
                    "source_initial",
                    "lora",
                    "persistent",
                    rate=0.0001,
                ),
            ],
        ),
        "examples_zero": config(
            "examples",
            train128,
            [0, 16, 64, 128, 256, 512],
            [
                arm(
                    "portal_zero",
                    "source_initial",
                    "full",
                    "task_latents",
                    True,
                    initialization="zero",
                ),
                arm(
                    "lora_zero",
                    "source_initial",
                    "lora",
                    "persistent",
                    initialization="zero",
                ),
            ],
        ),
        "constructed_retention": config(
            "retention",
            train16,
            [0],
            [
                arm("untouched", "source_initial", "none"),
                arm("constructed", "source_constructed", "none"),
            ],
        ),
    }
    for name, value in configs.items():
        path = ROOT / "configs/follow_through" / f"{name}.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"FOLLOW_THROUGH_CONFIG_ALREADY_DIFFERS: {name}")
        write_json(path, value)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    configs = build_configs()
    print(
        json.dumps(
            {name: value["protocol_sha256"] for name, value in configs.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
