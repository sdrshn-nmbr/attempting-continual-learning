import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import onpolicy303_budget_evaluate as lane
from choice_contract import ROOT, digest, file_hash, load_corpus, write_json
from onpolicy303_budget_contract import EvaluationReadBan

LOCAL = ROOT.parents[1] / "outputs/portfolio/followthrough-20260912/runs"


@pytest.fixture(scope="module")
def sealed():
    design = lane.read_json(ROOT / "configs/onpolicy303-budget-evaluate-protocol.json")
    budget, original, context = lane.validate_design(design)
    corpus, audit = load_corpus(context[-1])
    return design, budget, original, context, corpus, audit


@pytest.fixture(scope="module", params=lane.METHODS)
def verified(request, sealed):
    design, budget, original, context, corpus, audit = sealed
    spec = design["sources"][request.param]
    source = LOCAL / spec["task_id"]
    receipt, gate = lane.verify_materials(source, spec, budget, original, context, corpus, audit)
    return spec, source, receipt, gate


@pytest.fixture
def prepared(verified, tmp_path, sealed):
    spec, source, receipt, gate = verified
    output = tmp_path / "study"
    output.mkdir()
    projected, copied = lane.prepare_copy(source, output, spec, receipt, gate)
    lane.verify_copy(output, spec, sealed[1], sealed[-1])
    return spec, source, receipt, output, projected, copied


def test_real_failed405_sources_pass_complete_cpu_gate(verified):
    spec, source, receipt, gate = verified
    assert gate["ledger_updates"] == 21 and gate["loss_tokens"] == 588 and gate["total_tokens"] == 11340
    assert gate["optimizer_initial_clock"] == 384 and gate["optimizer_final"]["clock"] == 405
    assert gate["optimizer_final"]["parameter_states"] == 144
    assert gate["terminal"]["source_sha256"] == lane.FAILED_SOURCE
    assert gate["new_optimizer_updates"] == 0 and not gate["cuda_initialized"]
    assert file_hash(source / "study/training.json") == spec["files_sha256"]["study/training.json"]
    assert receipt["selected_checkpoint"] == 405


def test_projection_has_only_relocated_path_and_origin_hash(prepared):
    spec, source, original, output, projected, copied = prepared
    assert (output / "original_training.json").read_bytes() == (source / "study/training.json").read_bytes()
    restored = copy.deepcopy(projected)
    del restored["evaluation_only_origin_sha256"]
    restored["checkpoint405"]["path"] = original["checkpoint405"]["path"]
    assert restored == original
    assert projected["pid"] == original["pid"] and projected["updates_this_run"] == 21
    assert projected["evaluation_only_origin_sha256"] == copied["origin_training_sha256"]
    assert copied["new_optimizer_updates"] == 0 and not (output / "checkpoint405/optimizer.pt").exists()
    lane.verify_pins(source, spec)


@pytest.mark.parametrize("corruption", ["adapter", "origin", "receipt_pid", "receipt_tokens", "copy_identity"])
def test_corrupt_copy_or_unapproved_receipt_diff_rejected(prepared, sealed, corruption):
    spec, _, _, output, _, _ = prepared
    if corruption == "adapter":
        path = output / "checkpoint405/learner/adapter_model.safetensors"
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 1]))
    elif corruption == "origin":
        path = output / "original_training.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif corruption in {"receipt_pid", "receipt_tokens"}:
        path = output / "training.json"
        receipt = lane.read_json(path)
        if corruption == "receipt_pid":
            receipt["pid"] += 1
        else:
            receipt["tokens"]["additional_loss_tokens"] -= 1
        path.write_text(json.dumps(receipt))
    else:
        path = output / "copy.json"
        copied = lane.read_json(path)
        copied["origin_training_sha256"] = "0" * 64
        path.write_text(json.dumps(copied))
    with pytest.raises(ValueError, match="EVAL_ONLY"):
        lane.verify_copy(output, spec, sealed[1], sealed[-1])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("corruption", ["running", "timeout", "wrong_failure", "wrong_archive", "same_pid"])
def test_source_must_be_exact_ended_attempt(verified, corruption):
    spec, source, receipt, _ = verified
    execution, task, config, failure = [lane.read_json(source / name) for name in
                                        ("execution.json", "task.json", "config.json", "study/failure.json")]
    receipt = copy.deepcopy(receipt)
    if corruption == "running":
        execution["status"] = "running"
    elif corruption == "timeout":
        execution["timed_out"] = True
    elif corruption == "wrong_failure":
        failure["detail"] = "some unrelated incomplete training failure"
    elif corruption == "wrong_archive":
        execution["source_sha256"] = "0" * 64
    else:
        receipt["pid"] = failure["pid"] = os.getpid()
    with pytest.raises(ValueError, match="EVAL_ONLY"):
        lane.terminal_identity(execution, task, config, receipt, failure, spec)


@pytest.mark.parametrize("corruption", ["ledger_clock", "response_mask", "adam405"])
def test_semantic_gate_rejects_corrupt_training_material_even_with_new_file_hash(verified, sealed, tmp_path, corruption):
    spec, source, _, _ = verified
    changed = copy.deepcopy(spec)
    destination = tmp_path / "source"
    for name in spec["files_sha256"]:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
    if corruption == "ledger_clock":
        relative = "study/ledger.json"
        ledger = lane.read_json(destination / relative)
        ledger[0]["optimizer_clocks"]["minimum"] = 1
        (destination / relative).write_text(json.dumps(ledger))
    elif corruption == "response_mask":
        relative = "study/trajectories.jsonl"
        records = [json.loads(line) for line in (destination / relative).read_text().splitlines()]
        records[0]["diagnostics"]["prediction_mask"][-1] = False
        (destination / relative).write_text("\n".join(json.dumps(record) for record in records) + "\n")
        ledger = lane.read_json(destination / "study/ledger.json")
        ledger[0]["trajectory_sha256"][0] = digest(records[0])
        (destination / "study/ledger.json").write_text(json.dumps(ledger))
        changed["files_sha256"]["study/ledger.json"] = file_hash(destination / "study/ledger.json")
    else:
        relative = "study/checkpoint405/optimizer.pt"
        state = torch.load(destination / relative, map_location="cpu", weights_only=True)
        state["state"][0]["step"].fill_(404)
        torch.save(state, destination / relative)
    changed["files_sha256"][relative] = file_hash(destination / relative)
    with pytest.raises(ValueError, match="EVAL_ONLY|AdamW clock"):
        lane.verify_materials(destination, changed, *sealed[1:])
    assert not torch.cuda.is_initialized()


def test_copy_is_exclusive_and_never_overwrites_origin(prepared):
    spec, source, _, output, _, _ = prepared
    origin = source / "study/training.json"
    checksum = file_hash(origin)
    with pytest.raises(FileExistsError):
        lane.copy_bound(origin, output / "original_training.json", checksum)
    assert checksum == spec["files_sha256"]["study/training.json"] == file_hash(origin)


def cpu_child(config_path, output):
    config = lane.read_json(config_path)
    design = lane.read_json(ROOT / "configs/onpolicy303-budget-evaluate-protocol.json")
    budget, _, context = lane.validate_design(design)
    _, audit = load_corpus(context[-1])
    spec = design["sources"][config["method"]]
    copied = Path(config["copied_study"])
    source = Path(config["local_source"])
    ban = EvaluationReadBan([source], [copied / "checkpoint405/learner"]).install()
    projected, proof = lane.verify_copy(copied, spec, budget, audit)
    rejected = False
    try:
        (source / "study/training.json").read_bytes()
    except PermissionError:
        rejected = True
    assert rejected and ban.denied and not torch.cuda.is_initialized()
    write_json(output, {"pid": os.getpid(), "parent_pid": os.getppid(), "cpu_copy_only": True,
                       "original_training_pid": projected["pid"], "copy_origin_sha256": proof["origin_training_sha256"],
                       "source_read_banned": rejected, "cuda_initialized": False, "new_optimizer_updates": 0})


def test_fresh_uv_process_can_only_read_copied_inputs(prepared, tmp_path):
    _, source, original, output, _, copied = prepared
    config = {"method": original["method"], "copied_study": str(output), "local_source": str(source)}
    write_json(tmp_path / "child.json", config)
    command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(Path(__file__).resolve()),
               "--cpu-child", str(tmp_path / "child.json"), "--output", str(tmp_path / "child_result.json")]
    launcher = lane.invoke_fresh(command, tmp_path / "child.log")
    result = lane.read_json(tmp_path / "child_result.json")
    assert result["pid"] not in {os.getpid(), original["pid"], launcher}
    assert result["parent_pid"] == launcher and result["source_read_banned"]
    assert result["copy_origin_sha256"] == copied["origin_training_sha256"]
    assert result["new_optimizer_updates"] == 0 and not result["cuda_initialized"]


def test_configs_use_supervisor_cli_and_refuse_existing_study(sealed, tmp_path):
    for arm in ("sft", "kl"):
        config = ROOT / f"configs/onpolicy303-budget-{arm}-evaluate-only.json"
        dispatch = lane.read_json(config)
        assert dispatch["protocol_sha256"] == digest(sealed[0])
        assert dispatch["task_id"] == f"followthrough-20260912-onpolicy303-budget-{arm}-evaluate-only"
        output = tmp_path / arm
        output.mkdir()
        write_json(output / "execution.json", {"supervisor_existing_run": True})
        command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-B", str(ROOT / "onpolicy303_budget_evaluate.py"),
                   "--config", str(config), "--output-dir", str(output), "--validate-only"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stderr
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cpu-child", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_child(args.cpu_child, args.output)
