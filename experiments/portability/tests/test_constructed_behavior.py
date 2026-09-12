from pathlib import Path

import evaluate_constructed as study
import representability as geometry
import torch
from data import write_json
from learner import frozen_base_tensors, tensor_hash
from safetensors.torch import load_file
from test_contract import tiny_model
from test_sequence_targets import snapshot


def test_native_rte_carrier_matches_export_for_all_abc_tasks(tmp_path):
    base, portal = tiny_model()
    with torch.no_grad():
        for value in portal.alignment.output.values():
            value.normal_(0, 0.1)
    weights = portal.generate("rte")
    assert any(torch.count_nonzero(b) for _, b in weights.values())
    base_path = tmp_path / base.revision
    base.model.save_pretrained(base_path)
    base.tokenizer.save_pretrained(base_path)
    spec = snapshot(
        base_path,
        {"repo_id": base.model_id, "revision": base.revision, "cache_dir": None},
    )
    native = tmp_path / "native"
    geometry.save_native(portal, native)
    adapter = portal.export_peft("rte", tmp_path / "adapter")
    adapter_spec = {
        "path": str(adapter),
        "files": {
            name: study.file_sha256(adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
        "tensor_sha256": tensor_hash(load_file(adapter / "adapter_model.safetensors")),
    }
    proof = tmp_path / "construction.json"
    write_json(proof, {"role": "tiny CPU model control"})
    root = Path(study.__file__).parent
    config = {
        "kind": "constructed_native_generator_behavior",
        "adapters": {name: adapter_spec for name in study.STATIC},
        "source_base": spec,
        "source_base_tensor_sha256": tensor_hash(frozen_base_tensors(base.model)),
        "fixed_task": "rte",
        "native": {
            "path": str(native),
            "files": {
                name: study.file_sha256(native / name)
                for name in ("config.json", "model.safetensors")
            },
            "tensor_sha256": tensor_hash(portal.state_dict()),
        },
        "fresh_inputs": {
            "path": "data/sequence303_unused_inputs.json",
            "sha256": study.file_sha256(root / "data/sequence303_unused_inputs.json"),
        },
        "evaluation": {
            "batch_size": 8,
            "max_prompt": 768,
            "dtype": "float32",
            "autocast": False,
        },
        "construction_result": {"path": str(proof), "sha256": study.file_sha256(proof)},
        "seed": 19,
        "claim_boundary": "Tiny CPU architecture and routing control; no capability result.",
    }
    output = tmp_path / "evaluation"
    output.mkdir()
    result = study.run(config, output, "cpu")
    assert result["status"] == "completed" and result["optimizer_steps"] == 0
    assert result["rows"] == 864 and result["native_task_for_all_ABC"] == "rte"
    assert result["native_export_parity_passed"]
    assert result["native_source_parity"]["choice_agreement"] == 1
    assert result["native_source_parity"]["maximum_choice_score_gap"] < 1e-5
