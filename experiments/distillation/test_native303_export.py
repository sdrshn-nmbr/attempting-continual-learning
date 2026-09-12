import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

import native303_contract as contract
import native303_export as lane

CHILD = """
import json
import os
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from native303_contract import ReadBoundary, read_json, write_json
from native303_export import reload_verified, native_panel, generic_panel, prefix_logits

torch.set_num_threads(1)
root = Path(sys.argv[1])
request = read_json(root / 'request.json')
with ReadBoundary(request['forbidden']) as boundary:
    model, tokenizer, manifest, audit = reload_verified(root / 'publication/export', request['pin'], 'cpu')
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[4, 5, 6]]), use_cache=False).logits
    records = native_panel(model, tokenizer, request['rows'], 'tiny_fresh_native')
    generic = generic_panel(model, tokenizer, request['generic'])
    prefixes = prefix_logits(model, tokenizer, request['rows'], request['reference'])
    save_file({'logits': logits, **prefixes}, root / 'child_logits.safetensors')
write_json(root / 'child.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'records': records, 'generic': generic, 'boundary': boundary.proof(), 'parameters': sum(p.numel() for p in model.parameters()), 'loader': type(model).__name__})
"""


def tiny_student():
    torch.set_num_threads(1)
    torch.manual_seed(912303)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    base = Qwen3ForCausalLM(config).float().requires_grad_(False).eval()
    before, classes = (
        contract.tensor_digest(lane.base_values(base)),
        lane.ordinary_inventory(base),
    )
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3,
                **{str(i): i + 4 for i in range(28)},
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    student = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        adapter_name="learner",
    )
    with torch.no_grad():
        for name, value in student.named_parameters():
            if "lora_B" in name:
                value.normal_(0, 0.025)
    student.requires_grad_(False).eval()
    return student, tokenizer, before, classes


def tiny_export(root):
    student, tokenizer, before, classes = tiny_student()
    rows = [
        {
            "id": "tiny",
            "task": "sequence_a",
            "group": "0 1 2",
            "prompt": "0 1 2",
            "choices": [" 0 1 2"],
            "gold_idx": 0,
        }
    ]
    generic = [
        {
            "task": "tiny_choices",
            "prompt": "0 1 2 ",
            "choices": ["3 4", "5 6", "7"],
            "gold_idx": 1,
        }
    ]
    reference = lane.native_panel(student, tokenizer, rows, "tiny_source")
    scores = lane.generic_panel(student, tokenizer, generic)
    prefixes = lane.prefix_logits(student, tokenizer, rows, reference)
    with torch.inference_mode():
        old_logits = student(
            input_ids=torch.tensor([[4, 5, 6]]), use_cache=False
        ).logits
    merged, audit = lane.safe_merge_student(student, before, classes)
    with lane.Publication(root / "publication") as transaction:
        lane.save_standard(merged, tokenizer, transaction.stage / "model", "12KB")
        exported = transaction.promote(
            {
                "ordinary_weight_digest": audit["after"],
                "ordinary_module_classes": classes,
                "identity": {"pid": os.getpid()},
            }
        )
    contract.write_json(
        root / "request.json",
        {
            "pin": exported["manifest"],
            "forbidden": [
                str(root / "original_base"),
                str(root / "teacher"),
                str(root / "original_adapter"),
            ],
            "rows": rows,
            "reference": reference,
            "generic": generic,
        },
    )
    return exported, audit, old_logits, reference, scores, prefixes


def child(root):
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-c",
            CHILD,
            str(root),
        ],
        cwd=contract.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_actual_nonzero_rank8_qwen_merge_shards_and_separate_process(tmp_path):
    exported, audit, logits, reference, scores, prefixes = tiny_export(tmp_path)
    assert len(exported["shards"]) > 1
    assert audit["target_matrices"] == 4 and not audit["changed_untargeted"]
    assert all(
        audit["before"]["tensors"][name] != audit["after"]["tensors"][name]
        for name in audit["target_weight_comparisons"]
    )
    result = child(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    data = contract.read_json(tmp_path / "child.json")
    assert data["pid"] != os.getpid() and data["loader"] == "Qwen3ForCausalLM"
    assert data["records"] == reference and not data["boundary"]["denied_reads"]
    assert contract.compare_generic(
        data["generic"], scores, contract.TOLERANCES["merged_scores"]
    )["passed"]
    observed = load_file(tmp_path / "child_logits.safetensors")
    assert contract.numeric_difference(
        observed.pop("logits"), logits, contract.TOLERANCES["prefix_logits"]
    )["passed"]
    assert contract.compare_prefixes(
        observed, prefixes, reference, contract.TOLERANCES["prefix_logits"]
    )["passed"]
    assert all(
        not Path(name).name.startswith(("adapter_", "teacher_"))
        for name in data["boundary"]["python_audit_read_paths"]
    )


def test_corrupt_shard_fails_in_actual_fresh_process_before_loader(tmp_path):
    exported, *_ = tiny_export(tmp_path)
    shard = Path(exported["directory"]) / "model" / next(iter(exported["shards"]))
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    result = child(tmp_path)
    assert result.returncode != 0 and "NATIVE303_EXPORT_FILE_HASH" in result.stderr
    assert (
        not (tmp_path / "child.json").exists()
        and not (tmp_path / "child_logits.safetensors").exists()
    )
    with patch.object(
        lane, "load_standard", side_effect=AssertionError("must not load")
    ) as loader:
        with pytest.raises(ValueError, match="EXPORT_FILE_HASH"):
            lane.reload_verified(exported["directory"], exported["manifest"], "cpu")
        loader.assert_not_called()


def test_corrupt_manifest_and_nonstandard_file_rejected(tmp_path):
    exported, *_ = tiny_export(tmp_path)
    with pytest.raises(ValueError, match="FILE_PIN_MISMATCH"):
        contract.verify_export(exported["directory"], {"sha256": "0" * 64})
    path = Path(exported["directory"]) / "model/adapter_config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="EXPORT_FILE_SET"):
        contract.verify_export(exported["directory"], exported["manifest"])


def test_read_boundary_blocks_exact_teacher_adapter_and_symlink(tmp_path):
    teacher = tmp_path / "old_teacher"
    teacher.mkdir()
    path = teacher / "model.safetensors"
    path.write_bytes(b"teacher")
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    with contract.ReadBoundary([teacher]) as boundary:
        for candidate in (path, alias, adapter):
            with pytest.raises(PermissionError, match="FORBIDDEN_ARTIFACT_READ"):
                candidate.read_bytes()
    assert len(boundary.denied) == 3


def test_actual_transformers_loader_cannot_read_forbidden_source(tmp_path):
    student, tokenizer, before, classes = tiny_student()
    merged, _ = lane.safe_merge_student(student, before, classes)
    source = tmp_path / "forbidden_base"
    lane.save_standard(merged, tokenizer, source, "12KB")
    with (
        contract.ReadBoundary([source]) as boundary,
        pytest.raises((PermissionError, OSError), match="FORBIDDEN_ARTIFACT_READ"),
    ):
        lane.load_standard(source, "cpu")
    assert boundary.denied


def test_nan_merge_cannot_publish(tmp_path):
    student, _tokenizer, before, classes = tiny_student()
    with torch.no_grad():
        next(p for name, p in student.named_parameters() if "lora_B" in name)[0, 0] = (
            float("nan")
        )
    with lane.Publication(tmp_path / "publication") as transaction:
        with pytest.raises(ValueError, match="NaNs detected"):
            lane.safe_merge_student(student, before, classes)
        assert not transaction.final.exists()
    assert not (tmp_path / "publication/export").exists()


def test_publication_is_exclusive_and_never_overwrites(tmp_path):
    with lane.Publication(tmp_path) as first:
        with pytest.raises(FileExistsError), lane.Publication(tmp_path):
            pytest.fail("lock allowed second owner")
        assert first.stage.is_dir()
    (tmp_path / "export").mkdir()
    sentinel = tmp_path / "export/sentinel"
    sentinel.write_text("immutable")
    with (
        pytest.raises(ValueError, match="EXPORT_ALREADY_EXISTS"),
        lane.Publication(tmp_path),
    ):
        pytest.fail("existing export allowed")
    assert sentinel.read_text() == "immutable"


def test_resident_adapter_and_custom_module_rejected():
    student, _, _, _ = tiny_student()
    with pytest.raises(ValueError, match="ORDINARY_MODEL_CLASS"):
        lane.ordinary_inventory(student)
    base = Qwen3ForCausalLM(student.config)
    base.extra = CustomDenseOffset()
    with pytest.raises(ValueError, match="NO_CUSTOM_MODULES"):
        lane.ordinary_inventory(base)


class CustomDenseOffset(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))


def test_reloaded_tensor_digest_rejects_changed_but_rehashed_weights(tmp_path):
    exported, *_ = tiny_export(tmp_path)
    root = Path(exported["directory"])
    shard = root / "model" / next(iter(exported["shards"]))
    values = load_file(shard)
    first = next(iter(values.values()))
    first.reshape(-1)[0] += 0.1
    save_file(values, shard, metadata={"format": "pt"})
    manifest = contract.read_json(root / "native303_manifest.json")
    manifest["files"] = contract.manifest_files(root)
    (root / "native303_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="RELOADED_EXACT_MODEL_WEIGHTS"):
        lane.reload_verified(
            root, contract.file_pin(root / "native303_manifest.json"), "cpu"
        )


def test_numeric_bounds_and_all_prediction_differences():
    tolerance = {"atol": 1e-4, "rtol": 0.0}
    assert contract.numeric_difference(
        torch.tensor([0.0]), torch.tensor([0.00009]), tolerance
    )["passed"]
    assert not contract.numeric_difference(
        torch.tensor([0.0]), torch.tensor([0.00011]), tolerance
    )["passed"]
    old = [
        {
            "id": str(i),
            "generation": {"correct": True, "terminated": True, "token_ids": [1]},
        }
        for i in range(3)
    ]
    new = copy.deepcopy(old)
    for row in new:
        row["generation"]["token_ids"] = [2]
    result = contract.compare_native(new, old)
    assert not result["passed"] and len(result["differences"]) == 3


def test_complete_collected_source_pins_and_independent_panel_arithmetic():
    design = contract.read_json(contract.ROOT / "configs/native303_protocol.json")
    wave = contract.ROOT.parents[1] / "outputs/portfolio/followthrough-20260912"
    local = copy.deepcopy(design)
    local["source_archive"]["path"] = str(
        wave / "code" / Path(local["source_archive"]["path"]).name
    )
    for source in local["source_runs"].values():
        source["directory"] = str(wave / "runs" / source["task_id"])
    inputs = contract.source_inputs(local)
    assert {key: len(rows) for key, rows in inputs["rows"].items()} == {
        "train": 24,
        "generic": 128,
        "test": 192,
        "unused288": 864,
    }
    assert sum(r["generation"]["correct"] for r in inputs["reference"]["test"]) == 192
    assert (
        sum(r["generation"]["correct"] for r in inputs["reference"]["unused288"]) == 864
    )
    assert (
        inputs["source_adapter_tensor_sha256"]
        == "ce1441892f01f7a6fabb005892b5ca845c1c92e340f45f411ccef659ed4008f1"
    )


def test_header_aware_git_blob_and_immutable_json(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"hello\n")
    assert (
        contract.file_pin(path, True)["git_blob_sha1"]
        == hashlib.sha1(b"blob 6\0hello\n").hexdigest()
    )
    contract.write_json(tmp_path / "once.json", {"a": 1})
    with pytest.raises(FileExistsError):
        contract.write_json(tmp_path / "once.json", {"a": 2})
