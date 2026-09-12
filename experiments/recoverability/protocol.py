import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
DEPLOYABLE = ("none", "replay_short", "replay_long")
ACTIONS = (*DEPLOYABLE, "restore", "sham")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def normalized_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def validate_dataset(data, spec):
    if len(data["units"]) > spec["budget"]["max_intents"] or len(data["codes"]) != len(set(data["codes"])):
        raise ValueError("RECOVERY_COHORT_BUDGET_OR_CODES")
    seen = set()
    tokens_seen = set()
    identities = defaultdict(set)
    runs = defaultdict(set)
    units = set()
    for unit in data["units"]:
        split = unit["split"]
        if split not in {"train", "test"} or unit["id"] in units:
            raise ValueError("RECOVERY_UNIT_ID_OR_SPLIT")
        if unit["skill"] in identities[split]:
            raise ValueError("RECOVERY_REPEATED_SKILL")
        units.add(unit["id"])
        identities[split].add(unit["skill"])
        runs[split].add(unit["run"])
        for role, required in spec["rows"].items():
            rows = unit[role]
            if len(rows) != required:
                raise ValueError(f"RECOVERY_ROW_COUNT {unit['id']} {role} {len(rows)} != {required}")
            for row in rows:
                validate_row(row, data["codes"], spec)
                if row["text_sha256"] in seen:
                    raise ValueError("RECOVERY_TEXT_LEAKAGE")
                if tuple(row["input_ids"]) in tokens_seen:
                    raise ValueError("RECOVERY_TOKENIZED_LEAKAGE")
                seen.add(row["text_sha256"])
                tokens_seen.add(tuple(row["input_ids"]))
    if identities["train"] & identities["test"] or runs["train"] & runs["test"]:
        raise ValueError("RECOVERY_GROUP_LEAKAGE")
    if not identities["train"] or not identities["test"]:
        raise ValueError("RECOVERY_EMPTY_PREDICTOR_SPLIT")
    for run, rows in data["guards"].items():
        if run not in data["runs"] or not rows or len(rows) > spec["budget"]["max_guard_rows_per_run"]:
            raise ValueError("RECOVERY_GUARD_MISSING")
        for row in rows:
            validate_row(row, data["codes"], spec)
            if row["text_sha256"] in seen:
                raise ValueError("RECOVERY_GUARD_LEAKAGE")
            if tuple(row["input_ids"]) in tokens_seen:
                raise ValueError("RECOVERY_GUARD_TOKENIZED_LEAKAGE")
            seen.add(row["text_sha256"])
            tokens_seen.add(tuple(row["input_ids"]))
    if set(data["runs"]) != set(data["guards"]) or set(data["runs"]) != runs["train"] | runs["test"]:
        raise ValueError("RECOVERY_RUN_BINDING")


def validate_row(row, codes, spec):
    ids = row["input_ids"]
    if not ids or len(ids) > spec["runtime"]["max_length"] or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("RECOVERY_INPUT_IDS")
    if row["target"] not in codes or row["text_sha256"] != normalized_hash(row["text"]):
        raise ValueError("RECOVERY_ROW_BINDING")


def prepare(source_root):
    spec = read_json(PROTOCOL)
    names = ("plasticity-clinc-replay-screen-seed23-6a27d9c4", "plasticity-clinc-replay-confirmation-seed47-c31f860b")
    data = {"units": [], "runs": {}, "guards": {}, "sources": {}, "excluded_test_intents": []}
    train_identities = set()
    for split, name in zip(("train", "test"), names, strict=True):
        source = Path(source_root) / name
        full_provenance = read_json(source / "provenance.json")
        provenance = full_provenance["data"]
        base_manifest = full_provenance["model"]["files"]
        data.setdefault("base_manifest", base_manifest)
        if data["base_manifest"] != base_manifest:
            raise ValueError("RECOVERY_BASE_MANIFEST_MISMATCH")
        cfg = read_json(source / "config.json")
        if (cfg["model_id"], cfg["revision"]) != (spec["backbone"], spec["revision"]):
            raise ValueError("RECOVERY_SOURCE_BACKBONE")
        data["sources"][str(source / "provenance.json")] = file_hash(source / "provenance.json")
        collection = read_json(source / "collection.json")["binary_file_hashes"]
        data["sources"][str(source / "collection.json")] = file_hash(source / "collection.json")
        data["sources"][str(source / "config.json")] = file_hash(source / "config.json")
        data.setdefault("codes", provenance["code_token_ids"])
        if data["codes"] != provenance["code_token_ids"]:
            raise ValueError("RECOVERY_CODE_VOCABULARY")
        data["runs"][name] = {}
        for phase, task in (("before", 2), ("after", 3)):
            prefix = f"adapters/persistent/task-{task}"
            record = collection[f"{prefix}/adapter_model.safetensors"]
            data["runs"][name][phase] = {
                "path": str(Path(record["remote_path"]).parent),
                "files": {
                    filename: collection[f"{prefix}/{filename}"]["sha256"]
                    for filename in ("adapter_model.safetensors", "adapter_config.json")
                },
            }
        intent_names = dict(
            zip(
                [i for group in provenance["task_intent_ids"] for i in group],
                [i for group in provenance["task_intent_names"] for i in group],
                strict=True,
            )
        )
        if split == "train":
            train_identities = set(intent_names.values())
        for task in range(3):
            selected = provenance["selected_rows"][task]
            for intent in provenance["task_intent_ids"][task]:
                skill = intent_names[intent]
                if split == "test" and skill in train_identities:
                    data["excluded_test_intents"].append(skill)
                    continue
                unit = {"id": f"{name}/{skill}", "skill": skill, "run": name, "split": split}
                for role, source_split in (("repair", "train"), ("probe", "validation"), ("label", "test")):
                    candidates = sorted(
                        (r for r in selected[source_split] if r["intent"] == intent), key=lambda r: r["source_index"]
                    )
                    unit[role] = [source_row(r, data["codes"]) for r in candidates[: spec["rows"][role]]]
                data["units"].append(unit)
        data["guards"][name] = [
            source_row(r, data["codes"])
            for r in provenance["selected_rows"][3]["test"]
            if split == "train" or intent_names[r["intent"]] not in train_identities
        ]
    validate_dataset(data, spec)
    return data


def source_row(row, codes):
    return {key: row[key] for key in ("text", "text_sha256", "input_ids", "source_split", "source_index")} | {
        "target": codes[row["code"]]
    }


def artifact_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"RECOVERY_ASSET_MISSING {directory}")
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"RECOVERY_WEIGHTS_MISSING {directory}")
    return {
        str(path): file_hash(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
    }


def implementation_hashes():
    return {str(path): file_hash(path) for path in sorted(ROOT.glob("*.py"))}


def seal(config_path, runtime):
    config = read_json(config_path)
    config["dataset"] = str((ROOT / config["dataset"]).resolve())
    spec = read_json(PROTOCOL)
    if file_hash(PROTOCOL) != (ROOT / "protocol.sha256").read_text().strip():
        raise ValueError("RECOVERY_PROTOCOL_CHANGED")
    data = read_json(config["dataset"])
    validate_dataset(data, spec)
    if config["model_id"] != spec["backbone"] or config["revision"] != spec["revision"]:
        raise ValueError("RECOVERY_MODEL_SCOPE")
    assets = artifact_files(config["model_path"])
    for entry in data["base_manifest"]:
        path = str(Path(config["model_path"]) / entry["path"])
        if assets.get(path) != entry["sha256"] or Path(path).stat().st_size != entry["bytes"]:
            raise ValueError(f"RECOVERY_BASE_IDENTITY {path}")
    for run in data["runs"].values():
        for checkpoint in run.values():
            for filename, expected in checkpoint["files"].items():
                path = str(Path(checkpoint["path"]) / filename)
                if not Path(path).is_file() or file_hash(path) != expected:
                    raise ValueError(f"RECOVERY_CHECKPOINT_HASH {path}")
                assets[path] = expected
    for path, expected in data["sources"].items():
        path = str((Path(config["dataset"]).parent / path).resolve())
        if file_hash(path) != expected:
            raise ValueError(f"RECOVERY_SOURCE_CHANGED {path}")
        assets[path] = expected
    payload = {
        "config": config,
        "protocol": spec,
        "protocol_sha256": file_hash(PROTOCOL),
        "dataset_sha256": file_hash(config["dataset"]),
        "assets": assets,
        "implementation": implementation_hashes(),
        "runtime": runtime,
    }
    return {"payload": payload, "sha256": digest(payload)}


def verify_seal(sealed, runtime):
    payload = sealed["payload"]
    if digest(payload) != sealed["sha256"] or runtime != payload["runtime"]:
        raise ValueError("RECOVERY_SEAL_OR_RUNTIME_CHANGED")
    if payload["protocol"] != read_json(PROTOCOL):
        raise ValueError("RECOVERY_PROTOCOL_PAYLOAD_CHANGED")
    if implementation_hashes() != payload["implementation"] or file_hash(PROTOCOL) != payload["protocol_sha256"]:
        raise ValueError("RECOVERY_IMPLEMENTATION_CHANGED")
    if file_hash(payload["config"]["dataset"]) != payload["dataset_sha256"]:
        raise ValueError("RECOVERY_DATASET_CHANGED")
    for path, expected in payload["assets"].items():
        if file_hash(path) != expected:
            raise ValueError(f"RECOVERY_ASSET_CHANGED {path}")
    return payload
