import json
from pathlib import Path

import device4_recovery as recovery
from generated_contract import (
    digest,
    file_digest,
    make_corpus,
    qualification_gate,
    qualification_rows,
)

from tasks import FAMILIES

ROOT = recovery.ROOT
SOURCE_FILES = ("device4_recovered_source.py", *recovery.SOURCE_FILES)


def validate_source(spec):
    if spec["kind"] != "recovered_curriculum" or spec["source_sha256"] != recovery.source_hashes():
        raise ValueError("RECOVERED_SOURCE_CONTRACT_OR_CODE_CHANGED")
    design = recovery.checked_json(spec["design"])
    upstream, original, snapshot = recovery.validate_design(design)
    if Path(spec["training_dir"]).resolve() == Path(snapshot["source_run"]).resolve():
        raise ValueError("RECOVERED_SOURCE_CANNOT_IMPERSONATE_INTERRUPTED_RUN")
    if Path(spec["qualification_dir"]).resolve() != (Path(spec["training_dir"]) / "evaluation").resolve():
        raise ValueError("RECOVERED_SOURCE_REQUIRES_OWN_FRESH_QUALIFICATION")
    return design, upstream, original, snapshot


def verify_source(spec):
    design, upstream, original, snapshot = validate_source(spec)
    old = recovery.original_lane
    corpus = make_corpus(original, old.SPLITS)
    schedules = old.make_schedules(corpus, upstream)
    history = old.verify_history(upstream, original, corpus, schedules)
    folder, root = Path(spec["training_dir"]), Path(spec["qualification_dir"])
    training = recovery.verify_recovered_training(folder, design, upstream, original, snapshot, corpus, schedules)
    qualified = json.loads((root / "qualification.json").read_text())
    if (
        qualified["status"] != "completed" or qualified["source_sha256"] != recovery.source_hashes()
        or qualified["design_sha256"] != digest(design) or qualified["original_design_sha256"] != digest(upstream)
        or qualified["training_sha256"] != file_digest(folder / "training.json")
        or qualified["training_pid"] != training["pid"] or qualified["pid"] == training["pid"]
        or qualified["interrupted_execution_sha256"] != snapshot["execution_sha256"]
        or qualified["new_optimizer_updates"] != design["resume"]["new_updates"]
        or any(qualified[key] != 0 for key in ("learner_updates", "test_predictions", "composition_predictions"))
        or set(qualified["arms"]) != set(old.ARMS)
        or file_digest(root / "panels.json") != qualified["panels_sha256"]
    ):
        raise ValueError("RECOVERED_SOURCE_QUALIFICATION_RECEIPT_CHANGED")
    panels = json.loads((root / "panels.json").read_text())
    tasks = qualification_rows(corpus, original, "fallback_validation")
    eligible, checkpoints, diagnostics, raw_reference = [], {}, {}, None
    for arm in old.ARMS:
        result, panel = qualified["arms"][arm], panels[arm]
        old.verify_records(tasks, panel["diagnostics"], corpus, original, training["eos_token_id"], training["special_token_ids"])
        gate = qualification_gate(panel["pairs"], corpus, original, training["eos_token_id"], training["special_token_ids"], "fallback_validation")
        if [row["teacher"] for row in panel["pairs"]] != [row["generation"] for row in panel["diagnostics"]]:
            raise ValueError("RECOVERED_SOURCE_PAIR_DIAGNOSTIC_MISMATCH")
        raw = [row["learner"] for row in panel["pairs"]]
        if raw_reference is not None and raw_reference != raw:
            raise ValueError("RECOVERED_SOURCE_UNMATCHED_RAW_REFERENCE")
        raw_reference = raw
        summaries = {split: old.summarize([row for row in panel["diagnostics"] if row["split"] == split]) for split in ("train", "fallback_validation")}
        depth_ok = all(summaries[split][f"{family}/depth{depth}"]["accuracy"] >= upstream["evaluation"]["minimum_depth_accuracy_for_consolidation"] for split in summaries for family in FAMILIES for depth in (1, 2))
        report = training["arms"][arm]
        saved_checkpoint = report["checkpoints"][str(report["selected_checkpoint"])]
        checkpoint = saved_checkpoint["adapter"]
        if (
            result["gate"] != gate or result["summary"] != summaries or result["depth_accuracy_check_passed"] != depth_ok
            or result["eligible_for_depth3_4_prerequisite"] != (gate["passed"] and depth_ok)
            or result["status"] != ("qualified" if gate["passed"] else "rejected")
            or result["checkpoint"] != checkpoint or result["checkpoint_update"] != upstream["training"]["updates_per_arm"]
            or not result["reload_generation_parity"] or result["learner_updates"] != 0
            or result["origin"] != report["origin"]
        ):
            raise ValueError("RECOVERED_SOURCE_ELIGIBILITY_RECOMPUTATION_FAILED")
        saved_path = Path(saved_checkpoint["train_diagnostics_path"])
        saved = json.loads(saved_path.read_text())
        lookup = {row["uid"]: row["generation"] for row in saved}
        if any(row["generation"] != lookup[row["uid"]] for row in panel["diagnostics"] if row["split"] == "train"):
            raise ValueError("RECOVERED_SOURCE_RELOAD_PARITY_CHANGED")
        checkpoints[arm] = checkpoint
        diagnostics[arm] = {"path": str(saved_path), "sha256": saved_checkpoint["train_diagnostics_sha256"]}
        if gate["passed"] and depth_ok:
            eligible.append(arm)
    return {
        "source_contract": recovery.CONTRACT, "source_spec": spec,
        "qualification_sha256": file_digest(root / "qualification.json"), "panels_sha256": qualified["panels_sha256"],
        "training_sha256": file_digest(folder / "training.json"), "eligible": eligible, "checkpoints": checkpoints,
        "initial_tensor_sha256": training["initial"]["tensor_sha256"], "snapshot_sha256": training["snapshot_sha256"],
        "eos_token_id": training["eos_token_id"], "special_token_ids": training["special_token_ids"],
        "base_tensor_sha256": training["base_after_sha256"], "train_diagnostics": diagnostics,
        "interrupted_execution_sha256": snapshot["execution_sha256"],
    }, corpus, history
