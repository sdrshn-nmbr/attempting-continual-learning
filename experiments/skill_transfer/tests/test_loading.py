import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig

import learning
from learning import (
    LowRankLinear,
    capture,
    check_model_precision,
    check_snapshot_unchanged,
    install_adapters,
    load_model,
    require_rocm_runtime,
    restore,
    verify_snapshot,
)


def snapshot_fixture(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    files = []
    for name, content, required in (
        ("config.json", b'{"test":true}', True),
        ("model.safetensors", b"model-bytes", True),
        ("tokenizer.json", b'{"tokens":[]}', True),
        ("merges.txt", b"optional-merges", False),
    ):
        if required:
            (snapshot / name).write_bytes(content)
        files.append(
            {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "required": required}
        )
    manifest = {
        "model_id": "test",
        "revision": "revision",
        "files": files,
        "tokenizer_files": ["config.json", "tokenizer.json"],
        "ignored_nonloading_documents": ["README.md"],
        "optional_metadata_directory": ".cache",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = {
        "model_id": "test",
        "model_revision": "revision",
        "model_path": str(snapshot),
        "snapshot_manifest": str(path),
        "snapshot_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return config, snapshot


def test_snapshot_verifies_actual_content_and_loading_identity(tmp_path):
    config, root = snapshot_fixture(tmp_path)
    (root / "README.md").write_text("not loaded")
    (root / ".cache").mkdir()
    proof = verify_snapshot(config)
    assert len(proof["files"]) == 3
    check_snapshot_unchanged(proof)
    (root / "model.safetensors").write_bytes(b"changeddata")
    with pytest.raises(RuntimeError, match="SNAPSHOT_CHANGED_DURING_LOAD"):
        check_snapshot_unchanged(proof)
    with pytest.raises(RuntimeError, match="SNAPSHOT_HASH"):
        verify_snapshot(config)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("unknown", "SNAPSHOT_UNPINNED_ENTRY"),
        ("missing", "SNAPSHOT_SIZE"),
        ("size", "SNAPSHOT_SIZE"),
        ("optional", "SNAPSHOT_HASH"),
        ("manifest", "SNAPSHOT_MANIFEST_CHANGED"),
        ("revision", "SNAPSHOT_REVISION_MISMATCH"),
    ],
)
def test_snapshot_rejects_unpinned_missing_and_changed_inputs(tmp_path, mutation, message):
    config, root = snapshot_fixture(tmp_path)
    if mutation == "unknown":
        (root / "generation_config.json").write_text("{}")
    elif mutation == "missing":
        (root / "tokenizer.json").unlink()
    elif mutation == "size":
        (root / "model.safetensors").write_bytes(b"x")
    elif mutation == "optional":
        (root / "merges.txt").write_bytes(b"incorrectmerge!")
    elif mutation == "manifest":
        config["snapshot_manifest_sha256"] = "0" * 64
    else:
        config["model_revision"] = "other"
    with pytest.raises(RuntimeError, match=message):
        verify_snapshot(config)


def test_tokenizer_is_hash_checked_before_loading_weights(tmp_path):
    config, root = snapshot_fixture(tmp_path)
    (root / "model.safetensors").write_bytes(b"bad-weights")
    proof = verify_snapshot(config, tokenizer_only=True)
    assert set(proof["files"]) == {"config.json", "tokenizer.json"}
    with pytest.raises(RuntimeError, match="SNAPSHOT_HASH"):
        verify_snapshot(config)


def test_gpu_isolation_and_no_autocast_are_fail_fast(monkeypatch):
    with pytest.raises(RuntimeError, match="RUNTIME_DEVICE"):
        require_rocm_runtime({"device": "cpu"})
    monkeypatch.setattr(torch.version, "hip", "7.2.test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="REQUIRE_ONE_VISIBLE_ROCM_GPU"):
        require_rocm_runtime({"device": "cuda:0"})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="AUTOCAST_FORBIDDEN"):
        require_rocm_runtime({"device": "cuda:0"})
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="REQUIRE_ONE_VISIBLE_ROCM_GPU"):
        require_rocm_runtime({"device": "cuda:0"})


def guarded_model():
    model = torch.nn.Module()
    model.projection = LowRankLinear(torch.nn.Linear(3, 4, bias=False).to(torch.bfloat16), 2, 4)
    model.register_buffer("native_fp32", torch.ones(2))
    model.hf_device_map = {"": "cpu"}
    return model


def test_model_guard_accepts_bf16_base_and_fp32_adapter_and_native_buffer():
    record = check_model_precision(guarded_model(), torch.device("cpu"))
    assert record["base_load_dtype"] == "bfloat16"
    assert record["adapter_and_offset_dtype"] == "float32"
    assert record["all_floating_tensors_finite"]
    assert record["no_accelerate_hooks"]


@pytest.mark.parametrize("metadata", ["absent", None, {}, {"": "cpu"}, {"": torch.device("cpu")}])
def test_model_guard_accepts_optional_map_and_records_actual_placement(metadata, caplog):
    model = guarded_model()
    if metadata == "absent":
        del model.hf_device_map
    else:
        model.hf_device_map = metadata
    with caplog.at_level("INFO", logger="skill_transfer"):
        record = check_model_precision(model, torch.device("cpu"))
    assert record["tensor_devices"] == {"parameter:cpu": 3, "buffer:cpu": 1}
    assert record["hf_device_map_present"] == (metadata != "absent")
    assert record["hf_device_map_repr"] == repr(None if metadata == "absent" else metadata)
    assert record["accelerate_hooks"] == {}
    assert "MODEL_PLACEMENT_AUDIT" in caplog.text
    assert "hf_device_map_type" in caplog.text and "parameter:cpu" in caplog.text


def test_uniform_qwen_pretrained_load_has_no_device_map_and_passes_guard(tmp_path):
    text = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
    )
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict())
    Qwen3_5ForConditionalGeneration(config).save_pretrained(tmp_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        tmp_path, dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True, trust_remote_code=False
    )
    assert not hasattr(model, "hf_device_map")
    install_adapters(model, {"target_suffixes": ["q_proj", "v_proj"], "rank": 8, "lora_alpha": 16})
    record = check_model_precision(model, torch.device("cpu"))
    assert record["all_tensors_on_assigned_device"] and record["all_floating_tensors_finite"]
    assert record["no_accelerate_hooks"] and not record["hf_device_map_present"]


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("metadata", ["absent", None, {"": "cpu"}])
def test_actual_misplacement_fails_independently_of_map(kind, metadata):
    model = guarded_model()
    if metadata == "absent":
        del model.hf_device_map
    else:
        model.hf_device_map = metadata
    if kind == "parameter":
        model.register_parameter("misplaced", torch.nn.Parameter(torch.empty(1, device="meta")))
    else:
        model.register_buffer("misplaced", torch.empty(1, device="meta"))
    with pytest.raises(RuntimeError, match=f"MODEL_PLACEMENT: {kind}:misplaced") as error:
        check_model_precision(model, torch.device("cpu"))
    assert f"{kind}:meta" in str(error.value)
    assert '"hf_device_map_type"' in str(error.value)
    assert '"hf_device_map_repr"' in str(error.value)


def test_gpu_metadata_cannot_hide_cpu_tensors():
    model = guarded_model()
    model.hf_device_map = {"": "cuda:0"}
    with pytest.raises(RuntimeError, match="MODEL_PLACEMENT") as error:
        check_model_precision(model, torch.device("cuda:0"))
    assert '"expected_device": "cuda:0"' in str(error.value)
    assert "parameter:cpu" in str(error.value)
    assert "cuda:0" in str(error.value)


@pytest.mark.parametrize("metadata", ["cpu", ["cpu"], {"": []}, {"": False}, {"": "cuda:1"}, {"": "disk"}])
def test_invalid_optional_map_reports_type_and_actual_devices(metadata):
    model = guarded_model()
    model.hf_device_map = metadata
    with pytest.raises(RuntimeError, match="MODEL_DEVICE_MAP_TYPE|MODEL_OFFLOAD") as error:
        check_model_precision(model, torch.device("cpu"))
    assert repr(metadata) in str(error.value)
    assert f"builtins.{type(metadata).__name__}" in str(error.value)
    assert "parameter:cpu" in str(error.value) and "buffer:cpu" in str(error.value)


@pytest.mark.parametrize("location", ["root", "nested"])
def test_accelerate_hooks_rejected_even_when_all_tensors_are_resident(location):
    model = guarded_model()
    del model.hf_device_map
    target = model if location == "root" else model.projection.base
    add_hook_to_module(target, AlignDevicesHook(execution_device="cpu", offload=False))
    with pytest.raises(RuntimeError, match="MODEL_DISPATCH_HOOK") as error:
        check_model_precision(model, torch.device("cpu"))
    assert "accelerate.hooks.AlignDevicesHook" in str(error.value)
    assert "parameter:cpu" in str(error.value)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("offload", "MODEL_OFFLOAD"),
        ("meta", "MODEL_PLACEMENT"),
        ("nan", "MODEL_NONFINITE"),
        ("adapter_bf16", "MODEL_PRECISION"),
        ("base_fp32", "MODEL_BASE_PRECISION"),
        ("buffer_fp64", "MODEL_PRECISION"),
    ],
)
def test_model_guard_rejects_precision_offload_and_nonfinite(mutation, message):
    model = guarded_model()
    if mutation == "offload":
        model.hf_device_map = {"": "disk"}
    elif mutation == "meta":
        model.register_buffer("misplaced", torch.empty(1, device="meta"))
    elif mutation == "nan":
        model.native_fp32[0] = float("nan")
    elif mutation == "adapter_bf16":
        model.projection.A.data = model.projection.A.data.to(torch.bfloat16)
    elif mutation == "base_fp32":
        model.projection.base.float()
    else:
        model.native_fp32 = model.native_fp32.double()
    with pytest.raises(RuntimeError, match=message):
        check_model_precision(model, torch.device("cpu"))


def test_loader_uses_qualified_dtype_and_verifies_before_and_after(monkeypatch):
    order = []
    config = {"device": "cuda:0", "model_path": "/cached", "optimization_seed": 17, "gradient_checkpointing": False}

    def runtime(config):
        order.append("runtime")
        return {"base_load_dtype": "bfloat16"}

    def snapshot(config):
        order.append("snapshot")
        return {"verified": True}

    def model_loader(path, **kwargs):
        order.append("model")
        assert kwargs["dtype"] == torch.bfloat16
        assert kwargs["device_map"] == {"": "cuda:0"}
        assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        return SimpleNamespace(config=SimpleNamespace(use_cache=True))

    monkeypatch.setattr(learning, "require_rocm_runtime", runtime)
    monkeypatch.setattr(learning, "verify_snapshot", snapshot)
    monkeypatch.setattr(learning.Qwen3_5ForConditionalGeneration, "from_pretrained", model_loader)
    monkeypatch.setattr(learning, "install_adapters", lambda *args: order.append("adapters"))
    monkeypatch.setattr(learning, "check_model_precision", lambda *args: order.append("model_guard"))
    monkeypatch.setattr(learning, "check_snapshot_unchanged", lambda *args: order.append("snapshot_unchanged"))
    model = load_model(config)
    assert order == ["runtime", "snapshot", "model", "adapters", "model_guard", "snapshot_unchanged"]
    assert model.skill_transfer_load_proof["runtime"]["base_load_dtype"] == "bfloat16"


def test_restore_rejects_nonfinite_before_mutating_model():
    model = guarded_model()
    state = capture(model)
    before = model.projection.B.detach().clone()
    state["projection"]["B"][0, 0] = float("inf")
    with pytest.raises(RuntimeError, match="ADAPTER_NOT_FINITE_FP32"):
        restore(model, state)
    assert torch.equal(model.projection.B, before)
