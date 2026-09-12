import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer

from config import Config, digest
from data import PROMPT, PreparedData, Task, normalized_text
from engine import atomic_json, file_sha256, utc_now
from learning import evaluate, load_model
from run import LANE_ROOT, SUPERVISOR_FILES, model_provenance, runtime_provenance

logger = logging.getLogger(__name__)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_file(path, expected):
    require(path.is_file() and not path.is_symlink(), f"VERIFY_FILE_MISSING {path}")
    require(file_sha256(path) == expected, f"VERIFY_HASH_MISMATCH {path}")


def read_run(pin, config):
    root = Path(pin["run_dir"]).resolve(strict=True)
    for name in ("metrics", "execution", "events"):
        suffix = ".jsonl" if name == "events" else ".json"
        check_file(root / (name + suffix), pin[name + "_sha256"])
    metrics = json.loads((root / "metrics.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    cfg = Config(**metrics["config"])
    provenance = metrics["provenance"]
    require(
        execution["task_id"] == execution["task"]["id"] == root.name
        and execution["status"] == metrics["status"] == "completed"
        and execution["exit_code"] == 0
        and metrics["completed"],
        f"VERIFY_RUN_NOT_COMPLETED {root}",
    )
    require(
        Config(**execution["task"]["config"]) == cfg
        and Config(**json.loads((root / "config.json").read_text())) == cfg
        and json.loads((root / "provenance.json").read_text()) == provenance
        and metrics["signature"]
        == digest({"config": cfg.as_dict(), "provenance": provenance}),
        f"VERIFY_RUN_IDENTITY_MISMATCH {root}",
    )
    require(
        execution["runtime_sha256"] == config["runtime_sha256"]
        and execution["source_sha256"]
        == execution["task"]["source_sha256"]
        == config["source_sha256"],
        f"VERIFY_RUNTIME_OR_SOURCE_PIN_MISMATCH {root}",
    )
    source = Path(execution["task"]["code_dir"])
    bundle = hashlib.sha256()
    for path in sorted(source.iterdir()):
        if path.name == "__pycache__" and path.is_dir():
            continue
        require(
            path.is_file() and not path.is_symlink(),
            f"VERIFY_SOURCE_FILE_INVALID {path}",
        )
        bundle.update(path.name.encode() + b"\0" + path.read_bytes())
    require(
        bundle.hexdigest() == config["source_sha256"], "VERIFY_SOURCE_BUNDLE_MISMATCH"
    )
    require(
        {"config.py", "data.py", "learning.py", "run.py"}
        <= provenance["source_sha256"].keys(),
        "VERIFY_SOURCE_MANIFEST_INCOMPLETE",
    )
    for name, expected in provenance["source_sha256"].items():
        require(Path(name).name == name, "VERIFY_SOURCE_PATH_INVALID")
        check_file(source / name, expected)
        check_file(LANE_ROOT / name, expected)
    expected_updates = len(cfg.methods) * cfg.tasks * cfg.updates_per_task
    require(
        metrics["total_updates"] == expected_updates, "VERIFY_UPDATE_COUNT_MISMATCH"
    )
    require(
        events[-1]["event"] == "completed"
        and events[-1]["signature"] == metrics["signature"]
        and events[-1]["total_updates"] == expected_updates,
        "VERIFY_TERMINAL_EVENT_MISMATCH",
    )
    checkpoints = [event for event in events if event["event"] == "checkpoint"]
    final = checkpoints[-1]
    require(
        final["status"] == "completed"
        and final["cursor"]["total_updates"] == expected_updates,
        "VERIFY_CHECKPOINT_NOT_TERMINAL",
    )
    check_file(root / "checkpoint.pt", final["checkpoint_sha256"])
    require(set(metrics["methods"]) == set(cfg.methods), "VERIFY_METHODS_MISMATCH")
    for method in cfg.methods:
        tasks = metrics["methods"][method]["tasks"]
        require(
            len(tasks) == cfg.tasks and all(task["completed"] for task in tasks),
            "VERIFY_TASKS_INCOMPLETE",
        )
        endpoint = tasks[-1]
        adapter = root / "adapters" / method / f"task-{cfg.tasks - 1}"
        require(
            Path(endpoint["adapter_path"]) == adapter, "VERIFY_ADAPTER_PATH_MISMATCH"
        )
        require(
            {path.name for path in adapter.iterdir()}
            == set(endpoint["adapter_files_sha256"])
            and {"adapter_config.json", "adapter_model.safetensors"}
            <= endpoint["adapter_files_sha256"].keys(),
            "VERIFY_ADAPTER_MANIFEST_MISMATCH",
        )
        for name, expected in endpoint["adapter_files_sha256"].items():
            check_file(adapter / name, expected)
    return root, cfg, metrics, final["checkpoint_sha256"]


def saved_test_data(provenance, tokenizer, cfg):
    data = provenance["data"]
    require(
        data["selection_sha256"]
        == digest(
            {key: value for key, value in data.items() if key != "selection_sha256"}
        )
        and data["evaluation_sha256"]
        == digest(
            [
                {split: rows[split] for split in ("validation", "test")}
                for rows in data["selected_rows"]
            ]
        ),
        "VERIFY_DATA_IDENTITY_MISMATCH",
    )
    require(data["prompt_template"] == PROMPT, "VERIFY_PROMPT_MISMATCH")
    letters = [chr(ord("A") + code) for code in range(cfg.tasks * cfg.classes_per_task)]
    require(
        [tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        == [[token] for token in data["code_token_ids"]]
        and tokenizer.pad_token_id is not None,
        "VERIFY_TOKENIZER_CODE_MISMATCH",
    )
    require(len(data["selected_rows"]) == cfg.tasks, "VERIFY_DATA_TASK_COUNT_MISMATCH")
    tasks = []
    for index, selected in enumerate(data["selected_rows"]):
        codes = [
            ord(data["intent_to_code"][name]) - ord("A")
            for name in data["task_intent_names"][index]
        ]
        intents = data["task_intent_ids"][index]
        code_for_intent = dict(zip(intents, codes, strict=True))
        rows = selected["test"]
        require(
            len(codes) == cfg.classes_per_task
            and Counter(row["code"] for row in rows)
            == {code: cfg.test_per_class for code in codes},
            "VERIFY_TEST_CLASS_COUNTS_MISMATCH",
        )
        for row in rows:
            require(
                row["source_split"] == "test"
                and row["code"] == code_for_intent[row["intent"]]
                and hashlib.sha256(normalized_text(row["text"]).encode()).hexdigest()
                == row["text_sha256"]
                and tokenizer.encode(
                    PROMPT.format(text=row["text"]), add_special_tokens=False
                )
                == row["input_ids"],
                f"VERIFY_SAVED_TEST_ROW_MISMATCH task={index} row={row['source_index']}",
            )
        tasks.append(Task(index, intents, codes, {"test": rows}, []))
    return PreparedData(
        tasks, data["code_token_ids"], tokenizer.pad_token_id, data, {}, []
    )


def reload_and_score(model, endpoint, prepared, cfg):
    adapter = Path(endpoint["adapter_path"])
    for name, expected in endpoint["adapter_files_sha256"].items():
        check_file(adapter / name, expected)
    saved = load_file(adapter / "adapter_model.safetensors", device="cpu")
    expected = get_peft_model_state_dict(model, save_embedding_layers=False)
    require(saved.keys() == expected.keys(), "VERIFY_ADAPTER_KEYS_MISMATCH")
    require(
        all(
            saved[key].shape == expected[key].shape
            and saved[key].dtype == expected[key].dtype
            for key in saved
        ),
        "VERIFY_ADAPTER_SHAPE_OR_DTYPE_MISMATCH",
    )
    result = set_peft_model_state_dict(model, saved)
    require(not result.unexpected_keys, "VERIFY_ADAPTER_UNEXPECTED_KEYS")
    restored = get_peft_model_state_dict(model, save_embedding_layers=False)
    require(
        all(
            torch.equal(value.cpu(), restored[key].cpu())
            for key, value in saved.items()
        ),
        "VERIFY_ADAPTER_RELOAD_MISMATCH",
    )
    model.requires_grad_(False)
    model.eval()
    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "VERIFY_MODEL_NOT_FROZEN",
    )
    require(
        len(endpoint["test_after_task"]) == len(prepared.tasks),
        "VERIFY_ENDPOINT_TASK_COUNT_MISMATCH",
    )
    scores = []
    for task, previous in zip(prepared.tasks, endpoint["test_after_task"], strict=True):
        score = evaluate(model, task, "test", prepared, cfg)
        require(
            all(
                score[key] == previous[key]
                for key in (
                    "task",
                    "split",
                    "examples",
                    "predictions",
                    "correct",
                    "accuracy",
                    "input_tokens",
                )
            ),
            f"VERIFY_PREDICTION_MISMATCH task={task.index} adapter={adapter}",
        )
        scores.append(score)
    return scores


def prepare_output(config, output):
    for pin in config["runs"]:
        require(
            not output.is_relative_to(Path(pin["run_dir"]).resolve()),
            "VERIFY_SOURCE_RUN_OUTPUT_OVERLAP",
        )
    if output.exists():
        entries = {path.name: path for path in output.iterdir()}
        require(
            {"execution.json", "config.json", "packages.txt"} <= entries.keys()
            and bool({"output.log", "run.log"} & entries.keys())
            and not entries.keys() - SUPERVISOR_FILES
            and all(
                not path.is_symlink()
                and (path.is_dir() if name == "attempts" else path.is_file())
                for name, path in entries.items()
            ),
            "VERIFY_OUTPUT_NOT_FRESH",
        )
        execution = json.loads(entries["execution.json"].read_text())
        require(
            execution["status"] == "running"
            and execution["task_id"] == output.name
            and execution["runtime_sha256"] == config["runtime_sha256"]
            and json.loads(entries["config.json"].read_text()) == config,
            "VERIFY_SUPERVISOR_MISMATCH",
        )
        if "task.json" in entries:
            task = json.loads(entries["task.json"].read_text())
            require(
                task["id"] == output.name and task["config"] == config,
                "VERIFY_SUPERVISOR_TASK_MISMATCH",
            )
    else:
        output.mkdir(parents=True)
    (output / ".verify.lock").touch(exist_ok=False)


def execute(config, output):
    output = output.resolve()
    require(bool(config["runs"]), "VERIFY_NO_SOURCE_RUNS")
    prepare_output(config, output)
    receipt = {
        "status": "running",
        "started_at": utc_now(),
        "total_updates": 0,
        "config": config,
        "runs": [],
    }
    try:
        runs = [read_run(pin, config) for pin in config["runs"]]
        for pin, (root, cfg, metrics, checkpoint_sha256) in zip(
            config["runs"], runs, strict=True
        ):
            logger.info("VERIFY_SOURCE_ACCEPTED run=%s seed=%s", root.name, cfg.seed)
            provenance = metrics["provenance"]
            require(
                runtime_provenance(cfg) == provenance["runtime"],
                "VERIFY_NATIVE_RUNTIME_MISMATCH",
            )
            require(
                model_provenance(cfg) == provenance["model"],
                "VERIFY_FROZEN_BASE_MISMATCH",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_path, local_files_only=True, trust_remote_code=False
            )
            prepared = saved_test_data(provenance, tokenizer, cfg)
            model = load_model(cfg)
            require(
                all(
                    parameter.device == torch.device(cfg.device)
                    for parameter in model.parameters()
                ),
                "VERIFY_MODEL_DEVICE_MISMATCH",
            )
            methods = {}
            for method in cfg.methods:
                endpoint = metrics["methods"][method]["tasks"][-1]
                scores = reload_and_score(model, endpoint, prepared, cfg)
                methods[method] = {
                    "test": scores,
                    "adapter_files_sha256": endpoint["adapter_files_sha256"],
                    "identical_predictions_and_counts": True,
                }
                logger.info(
                    "VERIFY_ADAPTER_MATCH run=%s method=%s examples=%s",
                    root.name,
                    method,
                    sum(score["examples"] for score in scores),
                )
            del model
            read_run(pin, config)
            receipt["runs"].append(
                {
                    "run_dir": str(root),
                    "seed": cfg.seed,
                    "signature": metrics["signature"],
                    "evaluation_sha256": provenance["data"]["evaluation_sha256"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "methods": methods,
                }
            )
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        logger.exception("VERIFY_FAILED")
        raise
    finally:
        receipt["finished_at"] = utc_now()
        atomic_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(json.loads(args.config.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
