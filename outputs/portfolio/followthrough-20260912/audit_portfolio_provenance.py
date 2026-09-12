import argparse
import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

STATES = {"running", "completed", "failed", "interrupted", "blocked"}
TERMINAL = STATES - {"running"}
HASH = re.compile(r"[0-9a-f]{64}\Z")
BLOB_HASH = re.compile(r"[0-9a-f]{40}\Z")
ATTEMPT = re.compile(r"(?:[0-9a-f]{32}|prior-[0-9a-f]{64})\Z")
REMOTE = "/mnt/shared/cl-portfolio/"
REMOTE_ROOTS = ("/mnt/shared/cl-portfolio", "/mnt/shared/cl-smoke")
REMOTE_CONTEXT = "us-mi355x-nambiar-k8s"
REMOTE_POD = "cl-portfolio-control-20260912"
REMOTE_WORKER = "cl-portfolio-recovery-mfxws-scwl6"
IMPLEMENTATION_COPIES = {
    f"/tmp/cl-environments/4a75f5f779a2c224/lib/python3.12/site-packages/portallib/{name}": f"/Users/sudarshan/Documents/Codex/2026-09-05/i-want-you-to-run-extensive/experiments/portability/.venv/lib/python3.12/site-packages/portallib/{name}"
    for name in ("evaluation.py", "model.py")
}
CHUNK_BYTES = 8 * 1024 * 1024
CONTROL_FILES = {
    "collection.json",
    "execution.json",
    "task.json",
    "config.json",
    "packages.txt",
    "run.log",
}
DOCUMENT_NAMES = {
    "receipt.json",
    "export_receipt.json",
    "seal.json",
    "result.json",
    "metrics.json",
    "qualification.json",
    "training.json",
    "training_receipt.json",
    "training_receipt.pin.json",
    "input_manifest.json",
    "source-code.json",
    "consumed-source.json",
    "dispatcher.json",
    "source-verification.json",
    "failure.json",
}
EVALUATION_COPY_FILES = (
    "parameter_mapping.json",
    "checkpoint405/train_probes.json",
    "checkpoint405/retention.json",
    "checkpoint405/learner/adapter_config.json",
    "checkpoint405/learner/adapter_model.safetensors",
)
CHILD_CONFIG_FILES = {
    ("device4_chain.py", "evaluation/seal.json"): "evaluate_config.json",
    ("device4_sufficiency.py", "evaluation/seal.json"): "evaluate_config.json",
}
BLOCKED_DISPATCH_FILES = {
    "followthrough-20260912-native303-evaluate": "native303-blocked-dispatch-binding.json",
}
SKIP_RECORDS = {
    "trace",
    "raw",
    "predictions",
    "records",
    "examples",
    "panels",
    "train",
    "validation",
    "test",
    "guard",
    "probes",
    "features",
    "scale",
    "steps",
    "history",
}


class AuditError(ValueError):
    pass


def require(condition, detail):
    if not condition:
        raise AuditError(detail)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def supervisor_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key}")
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_keys)


def relative(name):
    require(
        isinstance(name, str) and name and "\\" not in name,
        f"invalid relative path {name!r}",
    )
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and str(path) == name
        and name != ".",
        f"unsafe or noncanonical relative path {name!r}",
    )
    return path


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, f"timestamp has no timezone {value}")
    return result


def issue(issues, severity, code, scope, detail):
    item = {"severity": severity, "code": code, "scope": scope, "detail": str(detail)}
    if item not in issues:
        issues.append(item)


def allowed_remote_path(name, scope="shared"):
    if scope == "worker_implementation":
        return name in IMPLEMENTATION_COPIES
    if scope != "shared":
        return False
    if not isinstance(name, str) or "\\" in name or "\0" in name:
        return False
    path = PurePosixPath(name)
    return (
        path.is_absolute()
        and str(path) == name
        and ".." not in path.parts
        and any(root in path.parents for root in map(PurePosixPath, REMOTE_ROOTS))
    )


def remote_plan(pins, blob_pins=(), scope="shared"):
    return {
        "scope": scope,
        "paths": sorted(
            {
                p["path"]
                for p in list(pins) + list(blob_pins)
                if allowed_remote_path(p["path"], scope)
            }
        ),
        "git_blob_paths": sorted(
            {p["path"] for p in blob_pins if allowed_remote_path(p["path"], scope)}
        ),
    }


def stat_signature(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stream_hashes(handle, blob_size=None):
    digest, size = hashlib.sha256(), 0
    blob = (
        hashlib.sha1(b"blob " + str(blob_size).encode() + b"\0")
        if blob_size is not None
        else None
    )
    while block := handle.read(CHUNK_BYTES):
        digest.update(block)
        if blob is not None:
            blob.update(block)
        size += len(block)
    result = {"bytes": size, "sha256": digest.hexdigest()}
    if blob is not None:
        result.update(git_blob_sha1=blob.hexdigest(), git_blob_header_bytes=blob_size)
    return result


def remote_file(name, cache, counts, scope="shared", blob=False):
    result = {"path": name, "bytes": None, "sha256": None}
    if not allowed_remote_path(name, scope):
        return result | {"status": "outside_authorized_roots"}
    try:
        path = Path(name)
        resolved = path.resolve(strict=True)
        if not allowed_remote_path(str(resolved), scope):
            return result | {"status": "resolved_target_outside_authorized_roots"}
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return result | {"status": "not_regular_file"}
            signature = stat_signature(before)
            if stat_signature(path.stat()) != signature:
                return result | {"status": "changed_before_read"}
            if signature in cache:
                counts["physical_file_cache_hits"] += 1
                content = cache[signature]
            else:
                content = stream_hashes(handle, signature[2] if blob else None)
                counts["physical_files_hashed"] += 1
                counts["physical_bytes_hashed"] += content["bytes"]
                cache[signature] = content
            require(
                not blob or "git_blob_sha1" in content,
                "blob paths must precede SHA256-only aliases",
            )
            if (
                stat_signature(os.fstat(handle.fileno())) != signature
                or path.resolve(strict=True) != resolved
                or stat_signature(path.stat()) != signature
                or content["bytes"] != signature[2]
            ):
                return result | {"status": "changed_during_read"}
        return result | content | {"status": "hashed", "signature": signature}
    except FileNotFoundError:
        return result | {"status": "absent_path"}
    except PermissionError:
        return result | {"status": "permission_denied"}
    except (OSError, RuntimeError) as error:
        return result | {"status": "read_error", "errno": getattr(error, "errno", None)}


def remote_hash_main():
    request = decode(sys.stdin.buffer.read(2 * 1024 * 1024))
    require(
        set(request) == {"scope", "paths", "git_blob_paths"}, "remote request schema"
    )
    scope = request["scope"]
    require(scope in {"shared", "worker_implementation"}, "remote scope")
    names = request["paths"]
    require(isinstance(names, list) and len(names) <= 4096, "bounded remote request")
    require(all(isinstance(p, str) for p in names), "remote path strings")
    require(names == sorted(set(names)), "remote paths must be deduplicated")
    require(
        all(allowed_remote_path(p, scope) for p in names), "unauthorized remote path"
    )
    blobs = request["git_blob_paths"]
    require(
        isinstance(blobs, list) and all(isinstance(p, str) for p in blobs),
        "blob path strings",
    )
    require(
        blobs == sorted(set(blobs)) and set(blobs) <= set(names),
        "blob paths must be a unique request subset",
    )
    counts, cache = Counter(), {}
    started = datetime.now(timezone.utc).isoformat()
    by_path = {
        name: remote_file(name, cache, counts, scope, name in blobs)
        for name in sorted(names, key=lambda name: name not in blobs)
    }
    results = [by_path[name] for name in names]
    for result in results:
        if result["status"] != "hashed":
            continue
        try:
            path = Path(result["path"])
            stable = allowed_remote_path(str(path.resolve(strict=True)), scope) and (
                stat_signature(path.stat()) == tuple(result["signature"])
            )
        except (OSError, RuntimeError):
            stable = False
        if not stable:
            result["status"] = "changed_after_read"
        del result["signature"]
    print(
        json.dumps(
            {
                "request_sha256": canonical(request),
                "snapshot_started_at": started,
                "snapshot_finished_at": datetime.now(timezone.utc).isoformat(),
                "files": results,
                "hashing": dict(counts) | {"chunk_bytes": CHUNK_BYTES},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def compare_file_pin(expected, actual, hash_key="sha256"):
    require(hash_key in {"sha256", "git_blob_sha1"}, "supported file digest")
    if actual is None:
        return {"status": "not_observed", "verified": False}
    if actual.get("status", "hashed") != "hashed":
        return {"status": actual["status"], "verified": False}
    digest = expected.get(hash_key)
    size = expected.get("bytes")
    pattern = HASH if hash_key == "sha256" else BLOB_HASH
    if not isinstance(digest, str) or not pattern.fullmatch(digest):
        return {"status": "unknown_expected_hash", "verified": False}
    if size is not None and (type(size) is not int or size < 0):
        return {"status": "invalid_expected_size", "verified": False}
    hash_matches = digest == actual.get(hash_key)
    size_matches = size == actual.get("bytes") if size is not None else None
    return {
        "status": "hash_mismatch"
        if not hash_matches
        else "size_mismatch"
        if size_matches is False
        else f"verified_{hash_key}_expected_size_unknown"
        if size is None
        else f"verified_{hash_key}_and_size",
        "verified": hash_matches and size_matches is not False,
        f"{hash_key}_matches": hash_matches,
        "expected_size_known": size is not None,
        "bytes_match": size_matches,
    }


def execute_remote_plan(request, enabled, script, issues, issue_scope):
    worker = request["scope"] == "worker_implementation"
    pod = REMOTE_WORKER if worker else REMOTE_POD
    report = {
        "enabled": enabled,
        "context": REMOTE_CONTEXT,
        "pod": pod,
        "request": request,
        "request_sha256": canonical(request),
        "program_sha256": sha(script.read_bytes()),
        "read_only_exec_calls": 0,
        "files": [],
    }
    if not enabled or not request["paths"]:
        return report
    command = [
        "kubectl",
        "--context",
        REMOTE_CONTEXT,
        "exec",
        "-i",
        pod,
        "--container",
        "portfolio" if worker else "control",
        "--",
        "/tools/uv" if worker else "uv",
        "run",
        "--no-project",
        "--offline",
        "--no-sync",
        "--no-python-downloads",
    ]
    if not worker:
        command.extend(["--python", "/usr/local/bin/python"])
    command.extend(
        [
            "python",
            "-B",
            "-c",
            remote_bootstrap(script.read_bytes()),
            "--remote-hash-stdin",
        ]
    )
    report["read_only_exec_calls"] = 1
    print(
        f"PROVENANCE_REMOTE_HASH {pod} {len(request['paths'])} unique paths, {len(request['git_blob_paths'])} blob hashes",
        flush=True,
    )
    try:
        result = subprocess.run(
            command,
            input=json.dumps(request).encode(),
            capture_output=True,
            timeout=1800,
            check=False,
        )
        report.update(
            exec_exit_code=result.returncode,
            exec_stderr_sha256=sha(result.stderr),
            exec_stderr_bytes=len(result.stderr),
        )
        require(result.returncode == 0, "read-only remote hash command failed")
        response = decode(result.stdout)
        require(
            response["request_sha256"] == canonical(request), "remote request binding"
        )
        require(
            [p["path"] for p in response["files"]] == request["paths"],
            "remote response exact path set and ordering",
        )
        for item in response["files"]:
            if item["status"] == "hashed":
                require(HASH.fullmatch(item["sha256"]), "remote file digest")
                require(
                    type(item["bytes"]) is int and item["bytes"] >= 0,
                    "remote file bytes",
                )
                if item["path"] in request["git_blob_paths"]:
                    require(
                        BLOB_HASH.fullmatch(item["git_blob_sha1"]),
                        "remote Git blob digest",
                    )
                    require(
                        item["git_blob_header_bytes"] == item["bytes"],
                        "Git blob header must use measured file size",
                    )
        report.update(response)
        report["response_sha256"] = sha(result.stdout)
    except (
        AuditError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ) as error:
        report["command_status"] = "remote_verification_unavailable"
        issue(
            issues,
            "gap",
            "remote_verification_unavailable",
            issue_scope,
            f"{pod}: {type(error).__name__}",
        )
    return report


def local_implementation_source(name, files):
    path = Path(IMPLEMENTATION_COPIES[name])
    result = {"path": str(path), "bytes": None, "sha256": None}
    try:
        if not path.exists():
            return result | {"status": "absent_path"}
        if path.is_symlink() or not path.is_file():
            return result | {"status": "not_regular_file"}
        return result | files.info(path) | {"status": "hashed"}
    except PermissionError:
        return result | {"status": "permission_denied"}
    except (OSError, AuditError):
        return result | {"status": "source_copy_read_error"}


def implementation_source_check(pin, worker, local):
    candidates = [
        {"verification_location": "same_path_on_active_worker", "actual": worker}
        | compare_file_pin(pin, worker),
        {"verification_location": "local_identical_source_copy", "actual": local}
        | compare_file_pin(pin, local),
    ]
    matching = next((c for c in candidates if c["verified"]), None)
    selected = matching or {
        "verification_location": None,
        "actual": None,
        "status": "no_matching_source_copy",
        "verified": False,
    }
    return {
        "expected": pin,
        "candidate_observations": candidates,
        "historical_process_reobserved": False,
    } | selected


def remote_bootstrap(source):
    encoded = base64.b64encode(zlib.compress(source)).decode("ascii")
    return f"import base64,zlib; exec(compile(zlib.decompress(base64.b64decode({encoded!r})), '<portfolio-auditor>', 'exec'))"


def verify_external_pins(bindings, enabled, script, worker_snapshot=None):
    pins = sorted(bindings.unavailable.values(), key=lambda p: (p["path"], p["sha256"]))
    plans = {
        "shared": remote_plan(pins, bindings.git_blob_inputs),
        "worker_implementation": remote_plan(pins, scope="worker_implementation"),
    }
    snapshots = {
        name: worker_snapshot
        if name == "worker_implementation" and worker_snapshot is not None
        else execute_remote_plan(
            request, enabled, script, bindings.issues, str(bindings.root)
        )
        for name, request in plans.items()
    }
    if worker_snapshot is not None:
        require(
            worker_snapshot["request"] == plans["worker_implementation"],
            "early worker hash request differs from referenced implementation pins",
        )
    observed = {
        item["path"]: item
        for snapshot in snapshots.values()
        for item in snapshot["files"]
    }
    local_sources = {
        name: local_implementation_source(name, bindings.files)
        for name in sorted({p["path"] for p in pins} & IMPLEMENTATION_COPIES.keys())
    }
    checks, implementation_checks = [], []
    for pin in pins:
        actual = observed.get(pin["path"])
        if pin["path"] in IMPLEMENTATION_COPIES:
            check = implementation_source_check(pin, actual, local_sources[pin["path"]])
            implementation_checks.append(check)
            if not check["verified"]:
                issue(
                    bindings.issues,
                    "gap",
                    "implementation_source_pin_unverified",
                    pin["path"],
                    check["status"],
                )
        else:
            check = {
                "expected": pin,
                "actual": actual,
                "verification_location": "remote_shared_filesystem",
            } | compare_file_pin(pin, actual)
            if not allowed_remote_path(pin["path"]):
                check.update(
                    status="outside_authorized_roots",
                    verified=False,
                    verification_location=None,
                )
            elif not enabled:
                check.update(status="remote_verification_not_requested", verified=False)
            if check["status"] in {
                "hash_mismatch",
                "size_mismatch",
                "invalid_expected_size",
            }:
                issue(
                    bindings.issues,
                    "error",
                    "remote_file_pin_mismatch",
                    pin["path"],
                    check["status"],
                )
            elif enabled and allowed_remote_path(pin["path"]) and not check["verified"]:
                issue(
                    bindings.issues,
                    "gap",
                    "remote_file_pin_unverified",
                    pin["path"],
                    check["status"],
                )
        checks.append(check)
    unique_blobs = {}
    for pin in bindings.git_blob_inputs:
        key = (pin["path"], pin["git_blob_sha1"], pin["bytes"])
        record = unique_blobs.setdefault(
            key,
            {
                "path": pin["path"],
                "git_blob_sha1": pin["git_blob_sha1"],
                "bytes": pin["bytes"],
                "references": [],
            },
        )
        reference = f"{pin['run']}/{pin['document']}"
        if reference not in record["references"]:
            record["references"].append(reference)
    git_checks = []
    for pin in sorted(
        unique_blobs.values(), key=lambda p: (p["path"], p["git_blob_sha1"], p["bytes"])
    ):
        actual = observed.get(pin["path"])
        check = {
            "expected": pin,
            "actual": actual,
            "verification_location": "remote_header_aware_blob_hash",
        } | compare_file_pin(pin, actual, "git_blob_sha1")
        if check["status"] in {
            "hash_mismatch",
            "size_mismatch",
            "invalid_expected_size",
        }:
            issue(
                bindings.issues,
                "error",
                "git_blob_file_pin_mismatch",
                pin["path"],
                check["status"],
            )
        elif enabled and not check["verified"]:
            issue(
                bindings.issues,
                "gap",
                "git_blob_file_pin_unverified",
                pin["path"],
                check["status"],
            )
        git_checks.append(check)
    return {
        "enabled": enabled,
        "remote_snapshots": snapshots,
        "authorized_shared_roots": list(REMOTE_ROOTS),
        "authorized_worker_files": sorted(IMPLEMENTATION_COPIES),
        "authorized_local_source_copies": IMPLEMENTATION_COPIES,
        "program_sha256": sha(script.read_bytes()),
        "read_only_exec_calls": sum(
            s["read_only_exec_calls"] for s in snapshots.values()
        ),
        "provider_mutations": 0,
        "files": list(observed.values()),
        "local_implementation_sources": local_sources,
        "implementation_source_checks": implementation_checks,
        "sha256_pin_checks": checks,
        "sha256_status_counts": dict(Counter(c["status"] for c in checks)),
        "sha256_verification_locations": dict(
            Counter(c["verification_location"] for c in checks if c["verified"])
        ),
        "sha256_pins_verified": sum(c["verified"] for c in checks),
        "sha256_pins_unverified": sum(not c["verified"] for c in checks),
        "git_blob_checks": git_checks,
        "git_blob_status_counts": dict(Counter(c["status"] for c in git_checks)),
        "git_blob_pin_reference_count": len(bindings.git_blob_inputs),
        "git_blob_unique_pin_count": len(git_checks),
        "git_blob_pins_verified": sum(c["verified"] for c in git_checks),
        "git_blob_hash_method": "SHA1(b'blob ' + str(measured_file_bytes).encode('ascii') + b'\x00' + file_bytes). SHA256 and header-aware SHA1 share one bounded streaming read. No Git repository, install, fetch or download is needed.",
        "unique_authorized_remote_paths": sum(len(p["paths"]) for p in plans.values()),
        "scope": "Current remote bytes at each recorded snapshot and exact local source copies. A matching source copy verifies the recorded bytes, not observation of the historical process or its former temporary environment. Active-worker observations retain their own absence/mismatch status even when a local copy matches. Expected sizes absent from pins remain unknown metadata.",
    }


def reconcile_requested_pins(bindings, baseline, external):
    observed = {item["path"]: item for item in external["files"]}
    resolved_pins = {
        (item["expected"]["path"], item["expected"]["sha256"]): item
        for item in external["sha256_pin_checks"]
    }
    result = []
    for pin in baseline:
        actual = None
        for reference in pin["references"]:
            run = bindings.states.get(reference.split("/", 1)[0])
            if run is not None:
                actual, _ = bindings.resolve(pin["path"], run)
                if actual is not None:
                    break
        location = "local_collection_or_archive"
        if actual is None:
            resolved = resolved_pins.get((pin["path"], pin["sha256"]))
            actual = resolved["actual"] if resolved else observed.get(pin["path"])
            location = resolved["verification_location"] if resolved else "remote"
        check = {
            "expected": pin,
            "actual": actual,
            "verification_location": location,
        } | compare_file_pin(pin, actual)
        if pin["path"] in IMPLEMENTATION_COPIES:
            check["historical_process_reobserved"] = False
        if actual is None and not (
            allowed_remote_path(pin["path"]) or pin["path"] in IMPLEMENTATION_COPIES
        ):
            check.update(status="outside_authorized_roots", verification_location=None)
        result.append(check)
    return {
        "pins": result,
        "status_counts": dict(Counter(c["status"] for c in result)),
        "location_counts": dict(
            Counter(c["verification_location"] for c in result if c["verified"])
        ),
    }


class Files:
    def __init__(self):
        self.cache = {}
        self.stats = Counter()

    @staticmethod
    def signature(path):
        require(
            not path.is_symlink() and path.is_file(),
            f"missing/nonregular/symlink file {path}",
        )
        value = path.stat()
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def get(self, path):
        path = Path(path)
        signature = self.signature(path)
        if path in self.cache:
            require(
                self.cache[path]["signature"] == signature,
                f"input changed during audit {path}",
            )
            self.stats["hash_cache_hits"] += 1
            return self.cache[path]
        keep = (
            path.suffix in {".json", ".tar", ".py", ".txt"}
            and signature[2] <= 64 * 1024 * 1024
        )
        digest, parts = hashlib.sha256(), []
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
                if keep:
                    parts.append(block)
        require(
            self.signature(path) == signature, f"input changed while hashing {path}"
        )
        result = {
            "sha256": digest.hexdigest(),
            "bytes": signature[2],
            "signature": signature,
            "raw": b"".join(parts) if keep else None,
        }
        self.cache[path] = result
        self.stats["physical_files_hashed"] += 1
        self.stats["physical_bytes_hashed"] += signature[2]
        if signature[2] >= 16 * 1024 * 1024:
            self.stats["large_files_hashed"] += 1
        return result

    def info(self, path):
        entry = self.get(path)
        return {key: entry[key] for key in ("bytes", "sha256")}

    def json(self, path):
        entry = self.get(path)
        require(entry["raw"] is not None, f"JSON exceeds audit read limit {path}")
        return decode(entry["raw"])

    def stable(self):
        for path, entry in self.cache.items():
            require(
                self.signature(path) == entry["signature"],
                f"input changed before report {path}",
            )


def archive_contents(raw, expected):
    members, closure = {}, hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: Path(item.name)):
            relative(member.name)
            require(
                member.isfile() and member.name not in members,
                f"unsafe/duplicate archive member {member.name}",
            )
            contents = archive.extractfile(member).read()
            require(len(contents) == member.size, f"short archive member {member.name}")
            closure.update(member.name.encode() + b"\0" + contents)
            members[member.name] = contents
    require(
        members and closure.hexdigest() == expected,
        f"source closure mismatch {expected}",
    )
    return members


def verify_archive(path, files, issues):
    scope, source = str(path), path.stem
    report = {
        "path": scope,
        "source_sha256": source,
        "closure_verified": False,
        "collection_receipt_verified": False,
    }
    try:
        require(HASH.fullmatch(source), "archive filename must be source digest")
        entry = files.get(path)
        require(entry["raw"] is not None, "archive exceeds64MiB local audit read limit")
        members = archive_contents(entry["raw"], source)
        report.update(
            closure_verified=True, files=len(members), archive=files.info(path)
        )
        collection_path = path.with_suffix(".json")
        if collection_path.is_file():
            collection = files.json(collection_path)
            require(
                collection["source_sha256"] == source
                and collection["archive"] == files.info(path),
                "archive collection pin",
            )
            require(
                collection["files"] == len(members), "archive collection member count"
            )
            require(
                collection["remote"] == f"{REMOTE}code/{source}.tar",
                "archive remote identity",
            )
            report.update(
                collection_receipt_verified=True,
                collection_receipt=files.info(collection_path),
                initially_collected_for=collection["tasks"],
            )
        else:
            issue(
                issues,
                "gap",
                "source_collection_receipt_missing",
                scope,
                collection_path,
            )
        return report, members
    except (
        AuditError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        tarfile.TarError,
    ) as error:
        issue(issues, "error", "source_archive_integrity", scope, error)
        return report, {}


def receipt_name(value, raw):
    identity = value.get("attempt_id", "prior-" + sha(raw))
    require(
        isinstance(identity, str) and ATTEMPT.fullmatch(identity), "invalid attempt ID"
    )
    return f"attempts/{identity}/{sha(raw)}.json"


def task_identity(value, task_id):
    require(
        value.get("task_id") == task_id and value.get("status") in STATES,
        "execution identity/status",
    )
    task = value.get("task")
    if task is None:
        require(value["status"] == "blocked", "launched receipt lacks task")
        return
    require(task["id"] == task_id, "embedded task ID")
    require(sha(supervisor_bytes(task)) == value["task_sha256"], "embedded task digest")
    require(
        sha(supervisor_bytes(task.get("config"))) == value["config_sha256"],
        "embedded config digest",
    )
    require(
        task["source_sha256"] == value["source_sha256"]
        and HASH.fullmatch(value["source_sha256"]),
        "embedded source digest",
    )
    require(
        task["code_dir"] == f"{REMOTE}code/{value['source_sha256']}",
        "code directory identity",
    )
    if value["status"] != "blocked":
        require(
            type(task.get("gpus")) is int and 1 <= task["gpus"] <= 8,
            "launched GPU count",
        )
        slots = value.get("gpus")
        require(
            isinstance(slots, list)
            and len(slots) == task["gpus"]
            and len(set(slots)) == len(slots)
            and all(type(i) is int and 0 <= i < 8 for i in slots),
            "assigned GPU slots",
        )
    if value.get("started_at") and value.get("finished_at"):
        require(
            timestamp(value["finished_at"]) >= timestamp(value["started_at"]),
            "negative execution duration",
        )
    if value["status"] == "completed":
        require(
            value.get("exit_code") == 0
            and value.get("timed_out") is False
            and value.get("finished_at"),
            "completed process evidence",
        )
    if value["status"] == "failed":
        require(
            value.get("exit_code") not in (None, 0) or value.get("timed_out") is True,
            "failed process evidence",
        )


def blocked_dispatch_binding(directory, execution_raw, submitted_raw, evidence):
    task_id = directory.name
    execution = decode(execution_raw)
    require(
        execution.get("task_id") == task_id
        and execution.get("status") == "blocked"
        and not any(
            key in execution
            for key in (
                "task",
                "task_sha256",
                "config_sha256",
                "source_sha256",
                "pid",
                "started_at",
                "finished_at",
                "exit_code",
                "runtime_sha256",
                "gpus",
                "pod",
                "supervisor",
                "attempt_id",
            )
        ),
        "separate dispatch evidence requires an unlaunched blocked receipt",
    )
    require(
        evidence["task_id"] == task_id
        and evidence["execution_record"] == str(directory / "execution.json")
        and evidence["execution_sha256"] == sha(execution_raw),
        "blocked sidecar task/execution binding",
    )
    manifest_path = directory.parents[2] / "manifests" / f"{task_id}.json"
    require(
        evidence["local_submitted_manifest_path"] == str(manifest_path),
        "blocked submitted manifest path",
    )
    require(
        type(evidence["dispatch_bytes"]) is int
        and evidence["dispatch_bytes"] == len(submitted_raw)
        and HASH.fullmatch(evidence["dispatch_sha256"])
        and evidence["dispatch_sha256"] == sha(submitted_raw),
        "blocked submitted manifest exact bytes/hash",
    )
    require(
        evidence["remote_dispatch_path"] == f"{REMOTE}queue/{task_id}.json"
        and evidence["remote_bytes_equal_local_submitted_manifest"] is True,
        "blocked remote queue observation/pin",
    )
    require(
        timestamp(evidence["observed_at"]) >= timestamp(execution["observed_at"]),
        "blocked dispatch observation predates receipt",
    )
    task = decode(submitted_raw)
    require(
        task == evidence["task"]
        and task["id"] == task_id
        and task["config"]["task_id"] == task_id,
        "blocked sidecar versus submitted task/config",
    )
    require(
        HASH.fullmatch(task["source_sha256"])
        and task["code_dir"] == f"{REMOTE}code/{task['source_sha256']}",
        "blocked submitted source identity",
    )
    relative(task["entrypoint"])
    dependencies, observed = task["depends_on"], execution["dependencies"]
    require(
        isinstance(dependencies, list)
        and dependencies
        and all(isinstance(name, str) for name in dependencies)
        and len(set(dependencies)) == len(dependencies)
        and task_id not in dependencies
        and isinstance(observed, dict)
        and set(dependencies) <= set(observed)
        and all(value is None or value in STATES for value in observed.values())
        and any(value in {"failed", "blocked"} for value in observed.values()),
        "blocked submitted dependencies versus receipt observations",
    )
    return {
        "saved_submission_binding_verified": True,
        "embedded_execution_task_identity": False,
        "model_execution_proven": False,
        "runtime_identity_proven": False,
        "task": task,
        "execution_sha256": sha(execution_raw),
        "submitted_manifest": {
            "path": str(manifest_path),
            "sha256": sha(submitted_raw),
            "bytes": len(submitted_raw),
        },
        "remote_queue_observation": {
            "path": evidence["remote_dispatch_path"],
            "sha256": evidence["dispatch_sha256"],
            "bytes": evidence["dispatch_bytes"],
            "observed_at": evidence["observed_at"],
            "saved_observation_reports_exact_local_bytes": True,
            "independently_reobserved_here": False,
        },
        "declared_dependency_observations": {
            name: observed[name] for name in dependencies
        },
        "implicit_dependency_observations": {
            name: status
            for name, status in observed.items()
            if name not in dependencies
        },
        "boundary": "Separately saved submitted-task identity and queue-byte observation bound to the immutable blocked receipt. No task/source identity was embedded in that receipt; no launch, runtime or model execution is inferred. Optional remote hashing rechecks queue bytes separately.",
    }


def run_source(run):
    return run.get("execution", {}).get("source_sha256") or run.get(
        "submitted_task", {}
    ).get("source_sha256")


def verify_chain(task_id, execution_raw, snapshots):
    current = decode(execution_raw)
    task_identity(current, task_id)
    decoded = {}
    for name, raw in snapshots.items():
        relative(name)
        value = decode(raw)
        task_identity(value, task_id)
        require(receipt_name(value, raw) == name, f"snapshot identity/digest {name}")
        decoded[name] = value
    current_name = receipt_name(current, execution_raw)
    require(
        snapshots.get(current_name) == execution_raw,
        "current execution absent from immutable history",
    )
    for name, value in decoded.items():
        previous = value.get("previous_receipt")
        if previous is None:
            continue
        require(
            isinstance(previous, str) and previous in decoded,
            f"missing predecessor {name}/{previous}",
        )
        parent = decoded[previous]
        same_attempt = Path(name).parent == Path(previous).parent
        if same_attempt:
            for key in (
                "task_sha256",
                "config_sha256",
                "source_sha256",
                "started_at",
                "supervisor",
                "runtime_sha256",
                "gpus",
                "pod",
            ):
                require(
                    value.get(key) == parent.get(key),
                    f"same-attempt identity changed {name}/{key}",
                )
            require(
                parent["status"] == "running"
                or value["status"] in {"blocked", "interrupted"},
                "invalid terminal-state transition",
            )
            if parent.get("pid") is not None:
                require(value.get("pid") == parent["pid"], "same-attempt PID changed")
        elif value["status"] == "running":
            require(
                value.get("resumed_from") == previous
                and parent["status"] == "interrupted",
                "unbound resumed attempt",
            )
        if value.get("resumed_from"):
            require(
                value["resumed_from"] in decoded
                and decoded[value["resumed_from"]]["status"] == "interrupted",
                "resume link",
            )
    for start in decoded:
        seen, name = set(), start
        while name is not None:
            require(name not in seen, "receipt predecessor cycle")
            seen.add(name)
            name = decoded[name].get("previous_receipt")
    chain, name = [], current_name
    while name is not None:
        chain.append(name)
        name = decoded[name].get("previous_receipt")
    require(
        set(chain) == set(decoded),
        "history contains snapshots outside current predecessor chain",
    )
    attempts = defaultdict(list)
    for name in reversed(chain):
        value = decoded[name]
        attempts[Path(name).parent.name].append(
            {
                "receipt": name,
                "status": value["status"],
                "pid": value.get("pid"),
                "started_at": value.get("started_at"),
                "finished_at": value.get("finished_at"),
                "exit_code": value.get("exit_code"),
                "blocked_reason": value.get("blocked_reason"),
            }
        )
    return {
        "verified": True,
        "snapshot_count": len(decoded),
        "current_receipt": current_name,
        "attempts": dict(attempts),
        "all_snapshots_reachable": True,
    }


def collect_inventory(root):
    selected, excluded = {}, []
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and directory.name.startswith("cl-collect-"):
            excluded.append(
                {"path": str(directory), "reason": "in_progress_collection_tempdir"}
            )
    for directory in sorted((root / "runs").iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.startswith("cl-collect-"):
            excluded.append(
                {"path": str(directory), "reason": "in_progress_collection_tempdir"}
            )
        elif not (directory / "collection.json").is_file():
            excluded.append(
                {"path": str(directory), "reason": "no_completed_collection_manifest"}
            )
        else:
            selected[directory.name] = directory
    return selected, excluded


def check_manifest(expected, actual):
    require(all(str(relative(name)) == name for name in expected), "collection paths")
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    mismatches = [
        name for name in set(actual) & set(expected) if actual[name] != expected[name]
    ]
    require(
        not missing and not extra and not mismatches,
        f"collection manifest missing={sorted(missing)} extra={sorted(extra)} mismatched={sorted(mismatches)}",
    )


def verify_run(directory, files, issues):
    scope = directory.name
    report = {
        "run": str(directory),
        "snapshot_started_at": datetime.now(timezone.utc).isoformat(),
        "collection_integrity": False,
        "execution_chain_integrity": False,
    }
    state = {"path": directory, "report": report, "documents": {}, "actual": {}}
    try:
        collection = files.json(directory / "collection.json")
        require(
            collection["remote"] == f"{REMOTE}runs/{scope}",
            "collection remote/task identity",
        )
        require(
            collection["execution_status"] in TERMINAL, "collection is not terminal"
        )
        actual = {}
        for path in sorted(directory.rglob("*")):
            require(not path.is_symlink(), f"symlink in collection {path}")
            if path.is_file() and path != directory / "collection.json":
                actual[str(path.relative_to(directory))] = files.info(path)
        state["actual"] = actual
        expected = collection["files"]
        check_manifest(expected, actual)
        report.update(
            collection_integrity=True,
            collection_sha256=files.info(directory / "collection.json")["sha256"],
            files_verified=len(actual),
            bytes_verified=sum(x["bytes"] for x in actual.values()),
        )
        execution_entry = files.get(directory / "execution.json")
        execution = decode(execution_entry["raw"])
        state["execution"] = execution
        report.update(
            execution_status=execution["status"],
            execution_sha256=execution_entry["sha256"],
            exit_code=execution.get("exit_code"),
            timed_out=execution.get("timed_out"),
            source_sha256=execution.get("source_sha256"),
            runtime_sha256=execution.get("runtime_sha256"),
        )
        require(
            execution["status"] == collection["execution_status"],
            "execution/collection status mismatch",
        )
        snapshots = {
            name: files.get(directory / name)["raw"]
            for name in actual
            if name.startswith("attempts/")
        }
        report["history"] = verify_chain(scope, execution_entry["raw"], snapshots)
        report["execution_chain_integrity"] = True
        task = execution.get("task")
        if task is not None:
            state["task"] = task
            report.update(
                lane=task.get("lane"), entrypoint=task.get("entrypoint", "run.py")
            )
            for name in ("task", "config"):
                path = directory / f"{name}.json"
                if not path.is_file() and execution["status"] == "blocked":
                    issue(issues, "gap", "unlaunched_dispatch_file_absent", scope, name)
                    continue
                require(
                    files.info(path)["sha256"] == execution[f"{name}_sha256"],
                    f"{name} file digest",
                )
                require(
                    files.json(path) == (task if name == "task" else task["config"]),
                    f"{name} file identity",
                )
            report["task_config_identity"] = True
        elif (
            scope in BLOCKED_DISPATCH_FILES
            and (directory.parent.parent / BLOCKED_DISPATCH_FILES[scope]).is_file()
        ):
            sidecar_path = directory.parent.parent / BLOCKED_DISPATCH_FILES[scope]
            manifest_path = directory.parents[2] / "manifests" / f"{scope}.json"
            evidence = files.json(sidecar_path)
            binding = blocked_dispatch_binding(
                directory,
                execution_entry["raw"],
                files.get(manifest_path)["raw"],
                evidence,
            )
            require(
                all(
                    item["status"] == "blocked"
                    and item["pid"] is None
                    and item["started_at"] is None
                    for attempt in report["history"]["attempts"].values()
                    for item in attempt
                ),
                "blocked submitted-only binding has launched predecessor",
            )
            require(
                not (set(actual) - {"execution.json"} - set(snapshots)),
                "unlaunched blocked collection contains execution artifacts",
            )
            binding["sidecar"] = {"path": str(sidecar_path), **files.info(sidecar_path)}
            state["submitted_task"] = state["task"] = binding["task"]
            report.update(
                submitted_dispatch=binding,
                submitted_source_sha256=binding["task"]["source_sha256"],
                task_config_identity=False,
                submitted_task_config_identity=True,
                unlaunched=True,
                lane=binding["task"]["lane"],
                entrypoint=binding["task"]["entrypoint"],
            )
        else:
            issue(
                issues,
                "gap",
                "blocked_receipt_without_dispatch_identity",
                scope,
                "No task/config/source recorded",
            )
        for name in actual:
            path = Path(name)
            if (
                (
                    path.name in DOCUMENT_NAMES
                    or (
                        state.get("task", {}).get("entrypoint")
                        == "onpolicy303_budget_evaluate.py"
                        and name in {"study/copy.json", "study/original_training.json"}
                    )
                )
                and len(path.parts) <= 3
                and not name.startswith("attempts/")
            ):
                state["documents"][name] = files.json(directory / name)
    except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
        issue(issues, "error", "run_provenance_integrity", scope, error)
    return state


class Bindings:
    def __init__(self, root, states, archives, files, issues):
        self.root, self.states, self.archives = root, states, archives
        self.files, self.issues = files, issues
        self.counts = Counter()
        self.unavailable = {}
        self.git_blob_inputs = []
        self.unavailable_protocols = []
        self.external_run_files = {}

    def resolve(self, path, run):
        if path.startswith(("data/", "configs/", "inputs/")):
            source = run_source(run)
            path = f"{REMOTE}code/{source}/{path}"
        if path.startswith(REMOTE + "code/"):
            parts = path.removeprefix(REMOTE + "code/").split("/", 1)
            if len(parts) == 2 and parts[0] in self.archives:
                member = str(relative(parts[1]))
                raw = self.archives[parts[0]]["members"].get(member)
                if raw is not None:
                    return {"bytes": len(raw), "sha256": sha(raw)}, None
            return None, None
        if path.startswith(REMOTE + "runs/"):
            parts = path.removeprefix(REMOTE + "runs/").split("/", 1)
            if len(parts) != 2:
                return None, None
            task_id, name = parts
            relative(task_id)
            relative(name)
            if task_id in self.states:
                target = self.states[task_id]["path"] / name
            else:
                candidates = sorted(self.root.parent.glob(f"*/runs/{task_id}"))
                direct = self.root.parent / "runs" / task_id
                if direct.is_dir():
                    candidates.append(direct)
                candidates = [
                    p for p in candidates if (p / "collection.json").is_file()
                ]
                if len(candidates) != 1:
                    return None, None
                target = candidates[0] / name
            if target.is_file():
                actual = self.files.info(target)
                if task_id not in self.states:
                    self.external_run_files[str(target)] = actual
                return actual, target
            return None, target
        if not Path(path).is_absolute():
            target = run["path"] / str(relative(path))
            return (self.files.info(target) if target.is_file() else None), target
        return None, None

    def pin(self, path, expected, run, origin):
        require(isinstance(path, str), "hash pointer path")
        expected = {"sha256": expected} if isinstance(expected, str) else expected
        require(
            isinstance(expected, dict) and HASH.fullmatch(expected.get("sha256", "")),
            f"invalid pin {origin}",
        )
        actual, target = self.resolve(path, run)
        if actual is None:
            if target is not None and target.is_relative_to(run["path"]):
                issue(
                    self.issues,
                    "error",
                    "internal_manifest_target_missing",
                    run["path"].name,
                    f"{origin}: {path}",
                )
            else:
                key = (path, expected["sha256"], expected.get("bytes"))
                record = self.unavailable.setdefault(
                    key,
                    {
                        "path": path,
                        "sha256": expected["sha256"],
                        "bytes": expected.get("bytes"),
                        "references": [],
                    },
                )
                reference = f"{run['path'].name}/{origin}"
                if reference not in record["references"]:
                    record["references"].append(reference)
            return
        require(
            actual["sha256"] == expected["sha256"]
            and (expected.get("bytes") is None or actual["bytes"] == expected["bytes"]),
            f"file pin mismatch {origin}: {path}",
        )
        self.counts["explicit_file_pins_verified"] += 1

    def manifest(self, manifest, base, run, origin):
        require(isinstance(manifest, dict), f"manifest mapping {origin}")
        for name, expected in manifest.items():
            name = str(relative(name))
            self.pin(
                f"{base.rstrip('/')}/{name}" if base else name, expected, run, origin
            )
        self.counts["manifests_verified"] += 1

    def code_pin(self, name, value, run, origin):
        expected = value if isinstance(value, dict) else {"sha256": value}
        require(
            HASH.fullmatch(expected.get("sha256", "")),
            f"invalid source file pin {origin}/{name}",
        )
        if name.startswith("/"):
            self.pin(name, expected, run, origin)
            return
        source = run_source(run)
        self.pin(f"{REMOTE}code/{source}/{name}", expected, run, origin)
        self.counts["declared_code_file_pins_checked"] += 1

    def protocol(self, config, payload, run, origin):
        name = str(relative(config["protocol"]))
        source = run_source(run)
        raw = self.archives.get(source, {}).get("members", {}).get(name)
        if raw is None:
            record = {
                "path": f"{REMOTE}code/{source}/{name}",
                "canonical_json_sha256": config["protocol_sha256"],
            }
            if record not in self.unavailable_protocols:
                self.unavailable_protocols.append(record)
            return
        document = decode(raw)
        semantic = run.get("task", {}).get("lane") == "distillation"
        actual = canonical(document) if semantic else sha(raw)
        require(
            actual == config["protocol_sha256"], f"dispatch protocol identity {origin}"
        )
        for key in ("protocol", "design"):
            if isinstance(payload.get(key), dict):
                require(
                    payload[key] == document, f"sealed protocol body {origin}/{key}"
                )
        run["report"].setdefault("protocol_bindings", {})[name] = {
            "file_sha256": sha(raw),
            "canonical_json_sha256": canonical(document),
            "dispatch_hash_kind": "canonical_json" if semantic else "file_bytes",
            "source_sha256": source,
        }
        self.counts["dispatch_protocol_bindings_verified"] += 1

    def walk(self, value, run, origin):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    self.walk(item, run, f"{origin}/{i}")
            return
        if not isinstance(value, dict):
            return
        local_manifest = isinstance(value.get("local_path"), str) and isinstance(
            value.get("files"), list
        )
        if local_manifest:
            for entry in value["files"]:
                target = (
                    value["local_path"].rstrip("/") + "/" + str(relative(entry["path"]))
                )
                if isinstance(entry.get("sha256"), str):
                    self.pin(target, entry, run, origin + "/files")
                else:
                    require(
                        re.fullmatch(r"[0-9a-f]{40}", entry.get("git_blob_sha1", "")),
                        "source Git-blob pin",
                    )
                    record = {
                        "path": target,
                        "git_blob_sha1": entry["git_blob_sha1"],
                        "bytes": entry["bytes"],
                        "run": run["path"].name,
                        "document": origin,
                        "bytes_verified_locally": False,
                    }
                    if record not in self.git_blob_inputs:
                        self.git_blob_inputs.append(record)
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            self.pin(value["path"], value, run, origin)
        if isinstance(value.get("path"), str) and isinstance(value.get("files"), dict):
            self.manifest(value["files"], value["path"], run, origin)
        for key, item in value.items():
            if key in SKIP_RECORDS or (key == "files" and local_manifest):
                continue
            if key in {
                "implementation",
                "learning_implementation",
                "code_dependencies",
                "source_sha256",
            } and isinstance(item, dict):
                for name, info in item.items():
                    if name.endswith((".py", ".json", ".txt")):
                        self.code_pin(name, info, run, origin + "/" + key)
            else:
                self.walk(item, run, f"{origin}/{key}")


def child_dispatch_config(dispatch, candidates, entrypoint, seal_name):
    name = CHILD_CONFIG_FILES.get(
        (entrypoint, seal_name), Path(seal_name).parts[0] + "_config.json"
    )
    require(
        name in candidates,
        f"archived writer child config missing: {entrypoint}: {name}",
    )
    require(candidates[name] == dispatch, f"sealed child dispatch mismatch: {name}")
    return name


def budget_evaluation_copy(
    copy_record,
    original_raw,
    projected_raw,
    spec,
    destination,
    copied_files,
    source_files,
    local_names,
):
    require(
        copy_record["kind"]
        == "zero_update_evaluator_input_copy_not_a_new_training_receipt"
        and copy_record["new_optimizer_updates"] == 0
        and copy_record["original_optimizer_copied"] is False,
        "budget evaluation copy type/zero-update boundary",
    )
    require(copy_record["source_run"] == spec["run_dir"], "budget copy source run")
    for name, checksum in spec["files_sha256"].items():
        relative(name)
        require(
            HASH.fullmatch(checksum)
            and name in source_files
            and source_files[name]["sha256"] == checksum,
            f"budget copy original source file missing/changed: {name}",
        )
    origin_sha = spec["files_sha256"]["study/training.json"]
    require(
        sha(original_raw) == origin_sha == copy_record["origin_training_sha256"],
        "budget byte-exact original training receipt",
    )
    original, projected = decode(original_raw), decode(projected_raw)
    require(
        "evaluation_only_origin_sha256" not in original
        and original["checkpoint405"]["path"]
        == spec["run_dir"] + "/study/checkpoint405/learner"
        and original["checkpoint405"] == spec["checkpoint405"],
        "budget original checkpoint405 source",
    )
    expected_projection = original | {
        "checkpoint405": original["checkpoint405"]
        | {"path": destination + "/checkpoint405/learner"},
        "evaluation_only_origin_sha256": origin_sha,
    }
    require(
        projected == expected_projection
        and sha(projected_raw) == copy_record["projected_training_sha256"]
        and copy_record["exact_allowed_json_changes"]
        == ["/checkpoint405/path", "/evaluation_only_origin_sha256"],
        "budget projected receipt changed beyond path/origin or hash differs",
    )
    expected_copies = {"original_training.json": origin_sha} | {
        name: spec["files_sha256"]["study/" + name] for name in EVALUATION_COPY_FILES
    }
    require(
        copy_record["copies_sha256"] == expected_copies,
        "budget exact declared copy set",
    )
    require(
        set(copied_files) == set(expected_copies)
        and all(
            copied_files[name]["sha256"] == checksum
            for name, checksum in expected_copies.items()
        ),
        "budget copied artifact missing/changed",
    )
    resolved = {}
    for name, checksum in original["files_sha256"].items():
        relative(name)
        original_name = "study/" + name
        require(
            HASH.fullmatch(checksum)
            and original_name in source_files
            and source_files[original_name]["sha256"] == checksum,
            f"budget inherited original artifact missing/changed: {name}",
        )
        copied = name in EVALUATION_COPY_FILES
        if copied:
            require(
                copied_files[name]["sha256"] == checksum,
                f"budget inherited copied artifact changed: {name}",
            )
        else:
            require(
                name not in local_names,
                f"budget unpermitted original training artifact copied: {name}",
            )
        resolved[name] = {
            "path": (destination if copied else spec["run_dir"] + "/study")
            + "/"
            + name,
            "sha256": checksum,
            "binding": "byte_exact_evaluator_copy"
            if copied
            else "original_training_study",
        }
    return {
        "verified": True,
        "kind": copy_record["kind"],
        "source_run": spec["run_dir"],
        "origin_training_sha256": origin_sha,
        "projected_training_sha256": sha(projected_raw),
        "exact_allowed_json_changes": copy_record["exact_allowed_json_changes"],
        "copied_files": copied_files,
        "inherited_training_manifest": resolved,
        "original_source_files_verified": len(spec["files_sha256"]),
        "new_optimizer_updates": 0,
        "boundary": "The original receipt and copied subset retain their byte identities. Only checkpoint405.path and evaluation_only_origin_sha256 change in the projected receipt. Its full training file map remains an original-study manifest; uncopied entries are required and verified at the original source, not waived or inferred present in the evaluator copy.",
    }


def verify_budget_evaluation_copy(run, bindings):
    documents = run["documents"]
    require(
        "study/copy.json" in documents, "completed budget evaluator lacks copy binding"
    )
    config = run["task"]["config"]
    require(config["stage"] == "evaluate_only", "budget evaluation-only dispatch stage")
    archived = bindings.archives[run_source(run)]["members"]
    design = decode(archived[str(relative(config["protocol"]))])
    require(
        design["contract"] == "onpolicy303_budget405_zero_update_evaluation_recovery"
        and canonical(design) == config["protocol_sha256"]
        and design["files_sha256"]["onpolicy303_budget_evaluate.py"]
        == sha(archived["onpolicy303_budget_evaluate.py"]),
        "budget copy archived helper/protocol binding",
    )
    spec = design["sources"][config["method"]]
    require(
        spec["run_dir"] == f"{REMOTE}runs/{spec['task_id']}",
        "budget original run path identity",
    )
    source = bindings.states.get(spec["task_id"])
    require(source is not None, "budget copy original run is not collected")
    require(
        source["report"]["collection_integrity"]
        and source["report"]["execution_chain_integrity"]
        and source["execution"]["status"] == "failed",
        "budget copy original failed run integrity",
    )
    copied = documents["study/copy.json"]
    source_gate = bindings.files.json(run["path"] / "study/source_gate.json")
    require(copied["source_gate"] == source_gate, "budget copy source gate identity")
    terminal = source_gate["terminal"]
    require(
        terminal["task_id"] == spec["task_id"]
        and terminal["attempt_id"] == source["execution"]["attempt_id"]
        and terminal["source_sha256"] == source["execution"]["source_sha256"]
        and terminal["finished_at"] == source["execution"]["finished_at"],
        "budget copy original terminal execution binding",
    )
    original = bindings.files.get(run["path"] / "study/original_training.json")["raw"]
    projected = bindings.files.get(run["path"] / "study/training.json")["raw"]
    require(
        original is not None and projected is not None, "budget copy receipt read limit"
    )
    local_names = {
        name.removeprefix("study/")
        for name in run["actual"]
        if name.startswith("study/")
    }
    copied_files = {
        name: run["actual"]["study/" + name]
        for name in ("original_training.json", *EVALUATION_COPY_FILES)
        if "study/" + name in run["actual"]
    }
    proof = budget_evaluation_copy(
        copied,
        original,
        projected,
        spec,
        f"{REMOTE}runs/{run['path'].name}/study",
        copied_files,
        source["actual"],
        local_names,
    )
    proof["copy_receipt"] = run["actual"]["study/copy.json"]
    proof["original_execution_sha256"] = source["report"]["execution_sha256"]
    run["report"]["budget_evaluation_copy"] = proof
    bindings.counts["typed_budget_evaluation_copies_verified"] += 1
    bindings.counts["inherited_original_training_file_pins_verified"] += len(
        proof["inherited_training_manifest"]
    )


def native_artifact_pins(artifact):
    root = artifact["directory"]
    require(
        root
        == REMOTE + "artifacts/followthrough-20260912-native-consolidated303/export",
        "native external artifact root",
    )
    files = artifact["files"]
    expected_shards = {
        name.removeprefix("model/"): pin
        for name, pin in files.items()
        if name.startswith("model/") and name.endswith(".safetensors")
    }
    require(
        artifact["shards"] == expected_shards and len(expected_shards) == 9,
        "native standard shard manifest subset",
    )
    require(
        artifact["tensor_payload_bytes"] == 4 * artifact["weight_elements"],
        "native FP32 payload accounting",
    )
    pins = {root + "/native303_manifest.json": artifact["manifest"]}
    for name, pin in files.items():
        relative(name)
        require(
            "adapter" not in Path(name).name and not name.endswith(".py"),
            "native export unexpected adapter/code file",
        )
        pins[root + "/" + name] = pin
    return pins


def verify_native_artifact(run, payload, bindings, origin):
    require(
        payload["contract"] == "onpolicy303_fp32_native_weight_export_20260912"
        and payload["optimizer_updates"] == 0,
        "native export/persistence contract",
    )
    artifact = payload["artifact"]
    if origin == "result.json":
        dependency = payload["export_dependency"]
        export_id = dependency["execution"]["task_id"]
        export_run = bindings.states.get(export_id)
        require(
            export_run is not None and export_run["execution"]["status"] == "completed",
            "native export dependency not collected/completed",
        )
        require(
            dependency["execution"] == export_run["execution"],
            "native export dependency execution identity",
        )
        require(
            dependency["execution_pin"] == export_run["actual"]["execution.json"]
            and dependency["receipt_pin"]
            == export_run["actual"]["export_receipt.json"],
            "native export dependency receipt pins",
        )
        exported = export_run["documents"]["export_receipt.json"]["artifact"]
        require(
            artifact
            == {key: value for key, value in exported.items() if key != "directory"},
            "native reload/export external manifest equality",
        )
        artifact = exported
    pins = native_artifact_pins(artifact)
    for path, pin in pins.items():
        bindings.pin(path, pin, run, origin + "/artifact")
    run["report"]["native_external_artifact"] = {
        "directory": artifact["directory"],
        "files_pinned_including_manifest": len(pins),
        "shards": len(artifact["shards"]),
        "tensor_payload_bytes": artifact["tensor_payload_bytes"],
        "scope": "Receipt-bound external model/evidence SHA256 and byte pins; current bytes verified by optional remote hashing. No re-scoring or new model execution.",
    }
    bindings.counts["native_external_artifact_bindings_verified"] += 1


def verify_documents(run, bindings, issues):
    directory = run["path"]
    scope = directory.name
    verified = []
    if run.get("task", {}).get("entrypoint") == "onpolicy303_budget_evaluate.py" and (
        "study/copy.json" in run["documents"]
        or run.get("execution", {}).get("status") == "completed"
    ):
        try:
            verify_budget_evaluation_copy(run, bindings)
        except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
            issue(issues, "error", "typed_budget_evaluation_copy_binding", scope, error)
    for name, value in run["documents"].items():
        try:
            require(
                isinstance(value, dict), f"provenance document is not object {name}"
            )
            payload = value
            if run.get("task", {}).get(
                "entrypoint"
            ) == "native303_export.py" and name in {
                "export_receipt.json",
                "result.json",
            }:
                verify_native_artifact(run, payload, bindings, name)
            if set(value) == {"payload", "sha256"}:
                require(
                    canonical(value["payload"]) == value["sha256"],
                    f"payload digest {name}",
                )
                payload = value["payload"]
                bindings.counts["self_digest_documents_verified"] += 1
                if "files" in payload:
                    base = Path(name).parent
                    prefix = str(base) + "/" if str(base) != "." else ""
                    expected = {
                        prefix + str(relative(k)): v
                        for k, v in payload["files"].items()
                    }
                    actual = {
                        key: info["sha256"]
                        for key, info in run["actual"].items()
                        if key.startswith(prefix) and key != name
                    }
                    require(
                        expected == actual,
                        f"complete study receipt file closure {name}",
                    )
                    bindings.counts["complete_study_manifests_verified"] += 1
            if "artifact_manifest" in payload:
                base = str(Path(name).parent)
                bindings.manifest(
                    payload["artifact_manifest"], "" if base == "." else base, run, name
                )
                bindings.counts["scientific_subset_manifests_verified"] += 1
            if name.endswith(".pin.json"):
                bindings.pin(name.replace(".pin.json", ".json"), payload, run, name)
            config = run.get("task", {}).get("config")
            if "dispatch_manifest" in payload:
                require(
                    payload["dispatch_manifest"] == config,
                    f"sealed dispatch manifest {name}",
                )
            if "dispatch" in payload:
                if name in {"evaluation/seal.json", "qualification/seal.json"}:
                    candidates = {
                        candidate: bindings.files.json(directory / candidate)
                        for candidate in run["actual"]
                        if len(Path(candidate).parts) == 1
                        and candidate.endswith("_config.json")
                    }
                    entrypoint = run["task"].get("entrypoint", "run.py")
                    child_config = child_dispatch_config(
                        payload["dispatch"], candidates, entrypoint, name
                    )
                    config = candidates[child_config]
                    source = run["execution"]["source_sha256"]
                    writer = (
                        bindings.archives.get(source, {})
                        .get("members", {})
                        .get(entrypoint)
                    )
                    if writer is None:
                        issue(
                            issues,
                            "gap",
                            "child_config_writer_source_unavailable",
                            scope,
                            f"{source}/{entrypoint}",
                        )
                    run["report"].setdefault("subprocess_config_bindings", {})[name] = {
                        "config": child_config,
                        "sha256": run["actual"][child_config]["sha256"],
                        "selection": "archived_entrypoint_filename_contract_and_exact_dispatch",
                        "source_archive_sha256": source,
                        "source_entrypoint": entrypoint,
                        "source_entrypoint_sha256": sha(writer)
                        if writer is not None
                        else None,
                    }
                require(payload["dispatch"] == config, f"sealed dispatch config {name}")
            if (
                "input_manifest" in payload
                and "input_manifest.json" in run["documents"]
            ):
                require(
                    payload["input_manifest"]
                    == run["documents"]["input_manifest.json"],
                    f"training input manifest identity {name}",
                )
            if "config_sha256" in payload and config is not None:
                encodings = {
                    "dispatcher_file": sha(supervisor_bytes(config)),
                    "canonical_json": canonical(config),
                    "sorted_pretty_json": sha(
                        (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
                    ),
                }
                matched = [
                    key
                    for key, digest in encodings.items()
                    if digest == payload["config_sha256"]
                ]
                require(
                    matched,
                    f"scientific config hash does not bind dispatcher config {name}",
                )
                run["report"].setdefault("scientific_config_hash_encodings", {})[
                    name
                ] = matched
            if (
                isinstance(config, dict)
                and isinstance(config.get("protocol"), str)
                and isinstance(config.get("protocol_sha256"), str)
            ):
                bindings.protocol(config, payload, run, name)
            if "arms" in payload and "receipt" in name:
                for arm, record in payload["arms"].items():
                    for step, checkpoint in record.get("checkpoints", {}).items():
                        if "optimizer" in checkpoint:
                            bindings.pin(
                                f"{arm}/checkpoints/optimizer-{int(step):04d}.pt",
                                checkpoint["optimizer"],
                                run,
                                name,
                            )
            if (
                name.endswith("source-code.json")
                and payload
                and all(HASH.fullmatch(str(v)) for v in payload.values())
            ):
                for key, expected in payload.items():
                    bindings.code_pin(key, expected, run, name)
            bindings.walk(payload, run, name)
            verified.append(name)
        except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
            issue(
                issues,
                "error",
                "scientific_manifest_or_binding",
                scope,
                f"{name}: {error}",
            )
    run["report"]["provenance_documents_verified"] = verified


def leaf_statuses(value, prefix=""):
    result = []
    if not isinstance(value, dict):
        return result
    if isinstance(value.get("status"), str):
        result.append({"path": prefix or ".", "status": value["status"]})
    for key in ("methods", "conditions", "arms", "qualification", "gate"):
        child = value.get(key)
        if isinstance(child, dict):
            for name, item in child.items():
                result.extend(leaf_statuses(item, f"{prefix}/{key}/{name}"))
    return result


def scientific_outcome(run, files):
    execution, documents = run.get("execution", {}), run["documents"]
    state = execution.get("status", "unknown")
    report = {
        "classification": "unassessed",
        "scientific_completion_claimed_by_this_audit": False,
        "documents": [],
        "nested_statuses": [],
    }
    for name, value in documents.items():
        if Path(name).name not in {
            "result.json",
            "receipt.json",
            "export_receipt.json",
            "metrics.json",
            "qualification.json",
            "training.json",
        }:
            continue
        payload = value.get("payload", value)
        report["documents"].append(
            {
                "path": name,
                "sha256": run["actual"][name]["sha256"],
                **{
                    key: payload[key]
                    for key in (
                        "status",
                        "stage",
                        "kind",
                        "qualified",
                        "teacher_qualified",
                        "decision",
                    )
                    if key in payload and isinstance(payload[key], (str, bool, int))
                },
            }
        )
        report["nested_statuses"].extend(
            {"document": name, **item} for item in leaf_statuses(payload)
        )
    status_values = [d.get("status", "") for d in report["documents"]]
    if state in {"failed", "interrupted", "blocked"}:
        report["classification"] = f"{state}_no_final_scientific_result"
        report["partial_artifacts_retained"] = any(
            name.endswith((".safetensors", ".pt", ".jsonl")) for name in run["actual"]
        )
        report["interruption_review"] = execution.get("interruption_review")
        log = run["path"] / "run.log"
        if log.is_file() and state == "failed":
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 16000))
                lines = handle.read().decode(errors="replace").splitlines()
            errors = [
                line for line in lines if re.match(r"\w*(?:Error|Exception):", line)
            ]
            report["recorded_failure"] = (
                errors[-1][:1000] if errors else "See immutable execution and run.log."
            )
    elif state == "completed":
        if run.get("task", {}).get("lane") in {"qualification", "runtime"}:
            report["classification"] = "runtime_qualification_only"
        elif (
            run.get("task", {}).get("entrypoint") == "native303_export.py"
            and documents.get("export_receipt.json", {}).get("contract")
            == "onpolicy303_fp32_native_weight_export_20260912"
            and documents["export_receipt.json"].get("optimizer_updates") == 0
            and "exported_pending_separate_job_parity" in status_values
        ):
            report["classification"] = (
                "completed_engineering_export_separate_parity_not_in_this_run"
            )
        elif not report["documents"]:
            report["classification"] = (
                "completed_process_no_recognized_scientific_result"
            )
        elif "rejected" in status_values and "completed" not in status_values:
            report["classification"] = "completed_negative_qualification"
        elif (
            any(s.endswith("pending") for s in status_values)
            and "completed" not in status_values
        ):
            report["classification"] = (
                "completed_training_stage_evaluation_not_in_this_run"
            )
        elif any("qualif" in s for s in status_values) or any(
            d.get("stage", "").startswith("qualify") for d in report["documents"]
        ):
            report["classification"] = (
                "completed_qualification_or_prerequisite_measurement"
            )
        else:
            report["classification"] = (
                "completed_measurement_or_analysis_not_a_success_claim"
            )
    return report


def verify_runtimes(root, states, files, issues):
    result = {}
    for path in sorted(root.glob("runtime-configmap*.json")):
        try:
            value = files.json(path)
            contents = value["data"]
            digest = sha(
                "".join(
                    contents[k]
                    for k in (
                        "supervisor.py",
                        "run_task.py",
                        "runtime-requirements.txt",
                    )
                ).encode()
            )
            require(
                value.get("immutable") is True,
                "runtime ConfigMap snapshot not immutable",
            )
            result[digest] = {
                "path": str(path),
                "sha256": files.info(path)["sha256"],
                "name": value["metadata"]["name"],
                "immutable": True,
                "local_snapshot_verified": True,
            }
        except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
            issue(issues, "error", "runtime_snapshot_integrity", str(path), error)
    for task_id, state in states.items():
        if state["report"].get("unlaunched"):
            state["report"].update(
                runtime_snapshot_verified=None,
                runtime_proof_status="not_applicable_unlaunched_blocked_submission",
            )
            continue
        digest = state.get("execution", {}).get("runtime_sha256")
        state["report"]["runtime_snapshot_verified"] = digest in result
        if digest not in result:
            issue(issues, "gap", "runtime_source_snapshot_missing", task_id, digest)
        review = state.get("execution", {}).get("interruption_review")
        if review:
            expected = review.get("pod_proof_sha256")
            candidates = [
                root / "worker-failure.json",
                root / "interruption-review-runtime/worker-failure.json",
            ]
            matches = [
                p
                for p in candidates
                if p.is_file() and files.info(p)["sha256"] == expected
            ]
            if not matches:
                issue(
                    issues, "gap", "interruption_pod_proof_missing", task_id, expected
                )
            else:
                proof = files.json(matches[0])
                require(
                    proof["metadata"]["name"] == state["execution"]["pod"],
                    "interruption pod identity",
                )
                state["report"]["interruption_pod_proof"] = {
                    "path": str(matches[0]),
                    "sha256": expected,
                    "child_exit_code_observed": review.get("child_exit_code_observed"),
                }
            if review.get("child_exit_code_observed") is False:
                require(
                    state["execution"].get("exit_code") is None,
                    "unobserved child exit code was invented",
                )
            require(
                review.get("scientific_completion_claimed") is False,
                "interrupted run claims scientific completion",
            )
    return result


def verify_dependencies(states, issues):
    edges, adjacency = [], {}
    for name, run in states.items():
        execution = run.get("execution", {})
        dependencies = run.get("task", {}).get("depends_on", [])
        require(
            isinstance(dependencies, list)
            and len(set(dependencies)) == len(dependencies),
            f"dependency list {name}",
        )
        adjacency[name] = dependencies
        for dependency in dependencies:
            edge = {
                "task": name,
                "dependency": dependency,
                "collected": dependency in states,
                "task_identity_basis": "separate_submitted_dispatch"
                if run.get("submitted_task")
                else "embedded_execution_task",
            }
            if dependency not in states:
                issue(
                    issues, "gap", "declared_dependency_not_collected", name, dependency
                )
            else:
                prior = states[dependency].get("execution", {})
                edge.update(
                    status=prior.get("status"),
                    execution_sha256=states[dependency]["report"].get(
                        "execution_sha256"
                    ),
                )
                if execution.get("started_at"):
                    qualified = (
                        prior.get("status") == "completed"
                        and prior.get("finished_at") is not None
                    )
                    before = qualified and timestamp(prior["finished_at"]) <= timestamp(
                        execution["started_at"]
                    )
                    edge["completed_before_task_started"] = before
                    if not before:
                        issue(
                            issues,
                            "error",
                            "dependency_not_completed_before_launch",
                            name,
                            dependency,
                        )
            edges.append(edge)

    def visit(name, active, done):
        require(name not in active, f"task dependency cycle at {name}")
        if name in done:
            return
        for child in adjacency.get(name, []):
            visit(child, active | {name}, done)
        done.add(name)

    done = set()
    for name in adjacency:
        visit(name, set(), done)
    return {
        "declared_edges": edges,
        "acyclic": True,
        "scope": "Explicit depends_on edges. Implicit supervisor runtime gates are not present in per-task manifests; runtime source snapshots are checked separately.",
    }


def self_test():
    passed = []

    def rejects(label, function):
        try:
            function()
        except AuditError:
            passed.append(label)
        else:
            raise AuditError(f"self-test did not reject {label}")

    task_id = "audit-fixture"
    task = {
        "id": task_id,
        "source_sha256": "a" * 64,
        "code_dir": REMOTE + "code/" + "a" * 64,
        "config": {"seed": 7},
        "gpus": 1,
    }
    initial = {
        "task_id": task_id,
        "task": task,
        "attempt_id": "1" * 32,
        "status": "running",
        "pid": None,
        "gpus": [0],
        "task_sha256": sha(supervisor_bytes(task)),
        "config_sha256": sha(supervisor_bytes(task["config"])),
        "source_sha256": task["source_sha256"],
        "started_at": "2026-09-12T00:00:00+00:00",
        "previous_receipt": None,
    }
    first_raw = supervisor_bytes(initial)
    first_name = receipt_name(initial, first_raw)
    active = initial | {"pid": 41, "previous_receipt": first_name}
    active_raw = supervisor_bytes(active)
    active_name = receipt_name(active, active_raw)
    for status, exit_code in (("completed", 0), ("failed", 1), ("interrupted", None)):
        terminal = active | {
            "status": status,
            "exit_code": exit_code,
            "timed_out": False,
            "finished_at": "2026-09-12T00:01:00+00:00",
            "previous_receipt": active_name,
        }
        terminal_raw = supervisor_bytes(terminal)
        terminal_name = receipt_name(terminal, terminal_raw)
        snapshots = {
            first_name: first_raw,
            active_name: active_raw,
            terminal_name: terminal_raw,
        }
        require(
            verify_chain(task_id, terminal_raw, snapshots)["snapshot_count"] == 3,
            "valid terminal chain",
        )
        passed.append(f"valid {status} chain with correct exit-code boundary")
    rejects(
        "missing predecessor",
        lambda: verify_chain(task_id, terminal_raw, {terminal_name: terminal_raw}),
    )
    bad_snapshots = dict(snapshots)
    bad_snapshots[first_name] = first_raw.replace(b'"seed": 7', b'"seed": 8')
    rejects(
        "snapshot/task tamper",
        lambda: verify_chain(task_id, terminal_raw, bad_snapshots),
    )
    rejects(
        "wrong task identity",
        lambda: verify_chain("other-task", terminal_raw, snapshots),
    )
    wrong = terminal | {"task_sha256": "b" * 64}
    wrong_raw = supervisor_bytes(wrong)
    rejects(
        "forged embedded task digest",
        lambda: verify_chain(
            task_id, wrong_raw, snapshots | {receipt_name(wrong, wrong_raw): wrong_raw}
        ),
    )
    changed = terminal | {"pid": 99}
    changed_raw = supervisor_bytes(changed)
    rejects(
        "same-attempt PID changed",
        lambda: verify_chain(
            task_id,
            changed_raw,
            {
                first_name: first_raw,
                active_name: active_raw,
                receipt_name(changed, changed_raw): changed_raw,
            },
        ),
    )
    bad_success = terminal | {"status": "completed", "exit_code": None}
    rejects(
        "unknown child exit cannot become completion",
        lambda: task_identity(bad_success, task_id),
    )
    expected = {"weights.bin": {"sha256": "a" * 64, "bytes": 42}}
    check_manifest(expected, expected)
    passed.append("complete artifact manifest")
    rejects("missing artifact", lambda: check_manifest(expected, {}))
    rejects("unexpected artifact", lambda: check_manifest({}, expected))
    rejects(
        "altered artifact digest",
        lambda: check_manifest(
            expected, {"weights.bin": {"sha256": "b" * 64, "bytes": 42}}
        ),
    )
    rejects("relative path escape", lambda: relative("../weights.bin"))
    rejects(
        "duplicate JSON key",
        lambda: decode(b'{"status":"failed","status":"completed"}'),
    )

    def tar_bytes(names):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name in names:
                entry = tarfile.TarInfo(name)
                entry.size = 3
                archive.addfile(entry, io.BytesIO(b"abc"))
        return buffer.getvalue()

    good_archive = tar_bytes(["x.py"])
    expected_closure = sha(b"x.py\0abc")
    require(
        archive_contents(good_archive, expected_closure) == {"x.py": b"abc"},
        "archive closure control",
    )
    passed.append("valid name-delimited source closure")
    rejects(
        "incorrect source closure", lambda: archive_contents(good_archive, "b" * 64)
    )
    rejects(
        "duplicate archive member",
        lambda: archive_contents(tar_bytes(["x.py", "x.py"]), expected_closure),
    )
    rejects(
        "archive traversal",
        lambda: archive_contents(tar_bytes(["../x.py"]), expected_closure),
    )
    checks = Bindings(
        Path("/unopened"),
        {},
        {"a" * 64: {"members": {"data/fixture.json": b"abc"}}},
        Files(),
        [],
    )
    fixture_run = {
        "path": Path("/unopened/runs/audit-fixture"),
        "execution": {"source_sha256": "a" * 64},
    }
    require(
        checks.resolve("data/fixture.json", fixture_run)[0]
        == {"sha256": sha(b"abc"), "bytes": 3},
        "source-relative fixture binding",
    )
    passed.append("source-relative fixture binds archived bytes")
    checks.walk(
        {
            "local_path": "/mnt/shared/models/example",
            "files": [{"path": "weights.bin", "bytes": 3, "sha256": sha(b"abc")}],
        },
        fixture_run,
        "seal/model",
    )
    require(
        not checks.issues
        and ("/mnt/shared/models/example/weights.bin", sha(b"abc"), 3)
        in checks.unavailable,
        "external model-root binding",
    )
    passed.append("relative model file resolves under model root and stays unverified")
    protocol_value = {"seed": 7}
    protocol_raw = b'{"seed": 7}\n'
    checks.archives["a" * 64]["members"]["configs/protocol.json"] = protocol_raw
    fixture_run["report"] = {}
    fixture_run["task"] = {"lane": "distillation"}
    config = {
        "protocol": "configs/protocol.json",
        "protocol_sha256": canonical(protocol_value),
    }
    checks.protocol(config, {"protocol": protocol_value}, fixture_run, "seal.json")
    require(
        sha(protocol_raw) != config["protocol_sha256"],
        "file and canonical protocol hashes distinct",
    )
    passed.append(
        "canonical protocol identity distinguished from serialized file bytes"
    )
    rejects(
        "sealed protocol body mutation",
        lambda: checks.protocol(
            config, {"protocol": {"seed": 8}}, fixture_run, "seal.json"
        ),
    )
    fixture_run["task"] = {"lane": "skill_transfer"}
    checks.protocol(
        config | {"protocol_sha256": sha(protocol_raw)},
        {},
        fixture_run,
        "dispatcher.json",
    )
    passed.append("skill protocol pins serialized file bytes")
    rejects(
        "wrong protocol digest convention",
        lambda: checks.protocol(config, {}, fixture_run, "dispatcher.json"),
    )
    valid_path = REMOTE + "models/example/weights.bin"
    require(allowed_remote_path(valid_path), "allowed portfolio path")
    require(allowed_remote_path("/mnt/shared/cl-smoke/models/x"), "allowed smoke path")
    passed.append("authorized root boundaries")
    for path in (
        "/tmp/package.py",
        "/mnt/shared/cl-portfolio-other/model",
        "/mnt/shared/cl-portfolio",
        REMOTE + "../secret",
        REMOTE + "x/./secret",
        REMOTE + "x//secret",
        REMOTE + "x\\secret",
        "configs/generated_protocol.json",
        REMOTE + "x\0secret",
    ):
        require(not allowed_remote_path(path), f"must reject remote path {path!r}")
    passed.append(
        "remote traversal alias prefix-lookalike and unauthorized paths rejected"
    )
    require(
        remote_plan(
            [
                {"path": valid_path, "sha256": "a" * 64},
                {"path": valid_path, "sha256": "b" * 64},
                {"path": "/tmp/package.py"},
            ]
        )
        == {"scope": "shared", "paths": [valid_path], "git_blob_paths": []},
        "deduplicate paths before reading despite distinct expected hashes",
    )
    passed.append("path deduplication independent of expected metadata")

    class ChunkControl(io.BytesIO):
        def read(self, size=-1):
            require(size == CHUNK_BYTES, "unbounded stream read")
            return super().read(min(size, 2))

    require(
        stream_hashes(ChunkControl(b"abcdefg"))
        == {"sha256": sha(b"abcdefg"), "bytes": 7},
        "chunked stream digest",
    )
    passed.append("bounded streaming matches exact content digest and size")
    expected = {"sha256": sha(b"abc"), "bytes": 3}
    actual = expected | {"status": "hashed"}
    require(compare_file_pin(expected, actual)["verified"], "matched file control")
    passed.append("matched SHA256 and byte count")
    require(
        compare_file_pin(expected | {"bytes": None}, actual)["status"]
        == "verified_sha256_expected_size_unknown",
        "unknown size is not mismatch",
    )
    passed.append("absent expected size preserved as unknown metadata")
    require(
        compare_file_pin(expected | {"bytes": 0}, actual)["status"] == "size_mismatch",
        "zero size is known",
    )
    passed.append("recorded zero size is checked")
    require(
        compare_file_pin(expected | {"sha256": "a" * 64}, actual)["status"]
        == "hash_mismatch",
        "hash mismatch control",
    )
    passed.append("hash mismatch independent of matching size")
    require(
        compare_file_pin({}, actual)["status"] == "unknown_expected_hash",
        "unknown hash control",
    )
    passed.append("unknown hash cannot establish verification")
    for status in (
        "absent_path",
        "permission_denied",
        "changed_during_read",
        "changed_after_read",
        "resolved_target_outside_authorized_roots",
    ):
        check = compare_file_pin(expected, actual | {"status": status})
        require(
            check["status"] == status and not check["verified"],
            "nonmeasurement cannot pass",
        )
    passed.append(
        "absent inaccessible changed and escaped files remain nonmeasurements"
    )
    require(
        remote_file("/tmp/package.py", {}, Counter())["status"]
        == "outside_authorized_roots",
        "unauthorized file is never opened",
    )
    passed.append("remote reader rejects unauthorized path before IO")
    hello_blob = stream_hashes(ChunkControl(b"hello\n"), 6)
    require(
        hello_blob["git_blob_sha1"] == "ce013625030ba8dba906f756967f9e9ca394464a",
        "known Git blob fixture",
    )
    require(
        hello_blob["sha256"] == sha(b"hello\n")
        and hello_blob["bytes"] == hello_blob["git_blob_header_bytes"] == 6,
        "joint digest and measured header fixture",
    )
    passed.append("canonical Git blob header and SHA256 share one streaming read")
    require(
        stream_hashes(ChunkControl(b""), 0)["git_blob_sha1"]
        == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        "empty Git blob fixture",
    )
    passed.append("zero-byte Git blob includes the canonical header")
    blob_expected = {"git_blob_sha1": hello_blob["git_blob_sha1"], "bytes": 6}
    require(
        compare_file_pin(blob_expected, hello_blob, "git_blob_sha1")["verified"],
        "known blob passes",
    )
    require(
        not compare_file_pin(
            blob_expected,
            hello_blob | {"git_blob_sha1": hashlib.sha1(b"hello\n").hexdigest()},
            "git_blob_sha1",
        )["verified"],
        "raw SHA1 cannot stand in for Git blob SHA1",
    )
    passed.append("header-aware blob hash required instead of raw file SHA1")
    require(
        compare_file_pin(blob_expected | {"bytes": 7}, hello_blob, "git_blob_sha1")[
            "status"
        ]
        == "size_mismatch",
        "blob expected size separately checked",
    )
    require(
        stream_hashes(ChunkControl(b"hello\n"), 7)["git_blob_sha1"]
        != hello_blob["git_blob_sha1"],
        "header byte count changes blob hash",
    )
    passed.append("incorrect blob header and expected byte count rejected")
    blob_plan = remote_plan(
        [{"path": valid_path}], [{"path": valid_path}, {"path": valid_path}]
    )
    require(
        blob_plan
        == {"scope": "shared", "paths": [valid_path], "git_blob_paths": [valid_path]},
        "SHA256 and repeated blob pins deduplicated together",
    )
    passed.append("repeated blob references and SHA256 pin use one path read")
    worker_path = next(iter(IMPLEMENTATION_COPIES))
    require(
        allowed_remote_path(worker_path, "worker_implementation")
        and not allowed_remote_path(worker_path),
        "worker path is isolated from shared-root scope",
    )
    require(
        not allowed_remote_path(valid_path, "worker_implementation"),
        "worker scope excludes NFS scan",
    )
    for name in (
        worker_path + ".other",
        worker_path.replace("evaluation.py", "other.py"),
        worker_path.replace("4a75f5f779a2c224", "other-environment"),
    ):
        require(
            not allowed_remote_path(name, "worker_implementation"),
            "worker exact-path allowlist",
        )
    passed.append("worker scope permits only the two exact authorized source paths")
    copy = actual | {"path": "/local/source.py"}
    absent_worker = {
        "path": worker_path,
        "status": "absent_path",
        "sha256": None,
        "bytes": None,
    }
    copy_check = implementation_source_check(expected, absent_worker, copy)
    require(
        copy_check["verified"]
        and copy_check["verification_location"] == "local_identical_source_copy",
        "local matching copy verifies source bytes",
    )
    require(
        not copy_check["historical_process_reobserved"]
        and copy_check["candidate_observations"][0]["status"] == "absent_path",
        "copy does not become historical process observation",
    )
    passed.append(
        "matching local source copy preserves absent worker and historical-process boundary"
    )
    different_worker = actual | {"sha256": "a" * 64, "path": worker_path}
    copy_check = implementation_source_check(expected, different_worker, copy)
    require(
        copy_check["verified"]
        and copy_check["candidate_observations"][0]["status"] == "hash_mismatch",
        "worker mismatch retained beside matching source copy",
    )
    passed.append(
        "different active-worker bytes remain a recorded mismatch beside local match"
    )
    require(
        not implementation_source_check(expected, absent_worker, absent_worker)[
            "verified"
        ],
        "no copy no source verification",
    )
    passed.append("absent source candidates cannot qualify source bytes")
    child = {
        "stage": "evaluate",
        "family": "B",
        "training_dir": REMOTE + "runs/chain-b",
    }
    require(
        child_dispatch_config(
            child,
            {"evaluate_config.json": child},
            "device4_chain.py",
            "evaluation/seal.json",
        )
        == "evaluate_config.json",
        "chain child config contract",
    )
    passed.append(
        "chain evaluation seal binds archived writer evaluate_config spelling and dispatch"
    )
    require(
        child_dispatch_config(
            child,
            {"evaluate_config.json": child},
            "device4_sufficiency.py",
            "evaluation/seal.json",
        )
        == "evaluate_config.json",
        "sufficiency child config contract",
    )
    passed.append(
        "sufficiency evaluation seal binds archived writer evaluate_config spelling and dispatch"
    )
    rejects(
        "sufficiency altered child dispatch despite matching filename",
        lambda: child_dispatch_config(
            child,
            {"evaluate_config.json": child | {"family": "C"}},
            "device4_sufficiency.py",
            "evaluation/seal.json",
        ),
    )
    rejects(
        "sufficiency wrong config spelling despite identical contents",
        lambda: child_dispatch_config(
            child,
            {"evaluation_config.json": child},
            "device4_sufficiency.py",
            "evaluation/seal.json",
        ),
    )
    export_record = {
        "contract": "onpolicy303_fp32_native_weight_export_20260912",
        "status": "exported_pending_separate_job_parity",
        "optimizer_updates": 0,
    }
    export_run = {
        "execution": {"status": "completed"},
        "task": {"lane": "distillation", "entrypoint": "native303_export.py"},
        "documents": {"export_receipt.json": export_record},
        "actual": {"export_receipt.json": {"sha256": canonical(export_record)}},
    }
    export_outcome = scientific_outcome(export_run, None)
    require(
        export_outcome["classification"]
        == "completed_engineering_export_separate_parity_not_in_this_run"
        and export_outcome["scientific_completion_claimed_by_this_audit"] is False,
        "native export remains a zero-update engineering stage",
    )
    passed.append("native export recognizes evidence without claiming parity or learning")
    require(
        scientific_outcome(
            export_run
            | {
                "documents": {
                    "export_receipt.json": export_record | {"optimizer_updates": 1}
                }
            },
            None,
        )["classification"]
        != "completed_engineering_export_separate_parity_not_in_this_run",
        "optimization cannot qualify as a zero-update export",
    )
    passed.append("nonzero updates cannot receive native zero-update export classification")
    require(
        child_dispatch_config(
            child,
            {
                "evaluation_config.json": child,
                "qualification_config.json": {"stage": "qualify"},
            },
            "device4_curriculum.py",
            "evaluation/seal.json",
        )
        == "evaluation_config.json",
        "distinct child dispatch config",
    )
    passed.append("other writer evaluation_config contract remains strict")
    rejects(
        "missing child dispatch config",
        lambda: child_dispatch_config(
            child, {}, "device4_chain.py", "evaluation/seal.json"
        ),
    )
    rejects(
        "altered child dispatch config",
        lambda: child_dispatch_config(
            child,
            {"evaluate_config.json": child | {"family": "C"}},
            "device4_chain.py",
            "evaluation/seal.json",
        ),
    )
    rejects(
        "chain wrong config spelling despite identical contents",
        lambda: child_dispatch_config(
            child,
            {"evaluation_config.json": child},
            "device4_chain.py",
            "evaluation/seal.json",
        ),
    )
    blocked_directory = Path("/audit/portfolio/wave/runs/blocked-fixture")
    blocked_task = {
        "id": blocked_directory.name,
        "lane": "distillation",
        "source_sha256": "a" * 64,
        "code_dir": REMOTE + "code/" + "a" * 64,
        "entrypoint": "native303_export.py",
        "gpus": 1,
        "config": {"task_id": blocked_directory.name},
        "depends_on": ["prior"],
    }
    blocked = {
        "task_id": blocked_directory.name,
        "status": "blocked",
        "observed_at": "2026-09-12T00:00:00+00:00",
        "dependencies": {"prior": "failed", "implicit-runtime": "completed"},
        "previous_receipt": None,
    }
    blocked_raw, submitted_raw = (
        supervisor_bytes(blocked),
        supervisor_bytes(blocked_task),
    )
    evidence = {
        "task_id": blocked_directory.name,
        "execution_record": str(blocked_directory / "execution.json"),
        "execution_sha256": sha(blocked_raw),
        "local_submitted_manifest_path": str(
            blocked_directory.parents[2]
            / "manifests"
            / f"{blocked_directory.name}.json"
        ),
        "dispatch_bytes": len(submitted_raw),
        "dispatch_sha256": sha(submitted_raw),
        "remote_dispatch_path": f"{REMOTE}queue/{blocked_directory.name}.json",
        "remote_bytes_equal_local_submitted_manifest": True,
        "observed_at": "2026-09-12T00:01:00+00:00",
        "task": blocked_task,
    }
    bound = blocked_dispatch_binding(
        blocked_directory, blocked_raw, submitted_raw, evidence
    )
    require(
        bound["saved_submission_binding_verified"]
        and not bound["embedded_execution_task_identity"]
        and not bound["model_execution_proven"]
        and not bound["runtime_identity_proven"]
        and bound["declared_dependency_observations"] == {"prior": "failed"}
        and bound["implicit_dependency_observations"]
        == {"implicit-runtime": "completed"}
        and "source_sha256" not in blocked
        and run_source({"execution": blocked, "submitted_task": bound["task"]})
        == "a" * 64,
        "separate submitted identity must not invent embedded source or runtime proof",
    )
    passed.append(
        "separate blocked submission binds bytes/source without inventing execution or runtime"
    )
    for label, changed in (
        ("wrong blocked sidecar task ID", {"task_id": "other"}),
        ("wrong blocked sidecar execution hash", {"execution_sha256": "b" * 64}),
        (
            "wrong blocked sidecar execution path",
            {"execution_record": "/elsewhere/execution.json"},
        ),
        (
            "wrong submitted manifest path",
            {"local_submitted_manifest_path": "/elsewhere/task.json"},
        ),
        ("wrong submitted byte count", {"dispatch_bytes": len(submitted_raw) + 1}),
        ("wrong saved remote queue hash pin", {"dispatch_sha256": "b" * 64}),
        (
            "wrong remote queue path",
            {"remote_dispatch_path": f"{REMOTE}queue/other.json"},
        ),
        (
            "remote queue equality not established",
            {"remote_bytes_equal_local_submitted_manifest": False},
        ),
        (
            "wrong sidecar dependency list",
            {"task": blocked_task | {"depends_on": ["other"]}},
        ),
        (
            "wrong sidecar source identity",
            {"task": blocked_task | {"source_sha256": "b" * 64}},
        ),
        (
            "sidecar observation before blocked receipt",
            {"observed_at": "2026-09-11T00:00:00+00:00"},
        ),
    ):
        rejects(
            label,
            lambda changed=changed: blocked_dispatch_binding(
                blocked_directory, blocked_raw, submitted_raw, evidence | changed
            ),
        )
    rejects(
        "semantically equal manifest with changed bytes",
        lambda: blocked_dispatch_binding(
            blocked_directory, blocked_raw, submitted_raw + b" ", evidence
        ),
    )
    for label, changed_task in (
        (
            "submitted dependency absent from receipt",
            blocked_task | {"depends_on": ["missing"]},
        ),
        (
            "submitted source differs from code directory",
            blocked_task | {"source_sha256": "b" * 64},
        ),
    ):
        changed_raw = supervisor_bytes(changed_task)
        changed_evidence = evidence | {
            "task": changed_task,
            "dispatch_sha256": sha(changed_raw),
            "dispatch_bytes": len(changed_raw),
        }
        rejects(
            label,
            lambda changed_raw=changed_raw, changed_evidence=changed_evidence: (
                blocked_dispatch_binding(
                    blocked_directory, blocked_raw, changed_raw, changed_evidence
                )
            ),
        )
    for label, changed_execution in (
        (
            "blocked receipt with child PID cannot become unlaunched",
            blocked | {"pid": 41},
        ),
        (
            "blocked receipt with embedded source cannot use submitted-only proof",
            blocked | {"source_sha256": "a" * 64},
        ),
        (
            "blocked receipt without failed or blocked dependency",
            blocked | {"dependencies": {"prior": "completed"}},
        ),
    ):
        changed_raw = supervisor_bytes(changed_execution)
        rejects(
            label,
            lambda changed_raw=changed_raw: blocked_dispatch_binding(
                blocked_directory,
                changed_raw,
                submitted_raw,
                evidence | {"execution_sha256": sha(changed_raw)},
            ),
        )
    original_root = REMOTE + "runs/original405"
    destination = REMOTE + "runs/evaluate405/study"
    content = {
        name: name.encode()
        for name in (
            *EVALUATION_COPY_FILES,
            "checkpoint405/optimizer.pt",
            "ledger.json",
        )
    }
    original = {
        "pid": 101,
        "updates_this_run": 21,
        "tokens": {"additional": 588},
        "checkpoint405": {
            "path": original_root + "/study/checkpoint405/learner",
            "files": {
                "adapter_config.json": sha(content[EVALUATION_COPY_FILES[3]]),
                "adapter_model.safetensors": sha(content[EVALUATION_COPY_FILES[4]]),
            },
        },
        "files_sha256": {
            name: sha(raw) for name, raw in content.items() if "/learner/" not in name
        },
    }
    original_raw = supervisor_bytes(original)
    projected = original | {
        "checkpoint405": original["checkpoint405"]
        | {"path": destination + "/checkpoint405/learner"},
        "evaluation_only_origin_sha256": sha(original_raw),
    }
    projected_raw = supervisor_bytes(projected)
    source_files = {
        "study/" + name: {"sha256": sha(raw), "bytes": len(raw)}
        for name, raw in content.items()
    }
    source_files["study/training.json"] = {
        "sha256": sha(original_raw),
        "bytes": len(original_raw),
    }
    spec = {
        "run_dir": original_root,
        "checkpoint405": original["checkpoint405"],
        "files_sha256": {name: info["sha256"] for name, info in source_files.items()},
    }
    copied_files = {
        name: source_files["study/" + name] for name in EVALUATION_COPY_FILES
    }
    copied_files["original_training.json"] = source_files["study/training.json"]
    copy_record = {
        "kind": "zero_update_evaluator_input_copy_not_a_new_training_receipt",
        "source_run": original_root,
        "origin_training_sha256": sha(original_raw),
        "projected_training_sha256": sha(projected_raw),
        "new_optimizer_updates": 0,
        "original_optimizer_copied": False,
        "exact_allowed_json_changes": [
            "/checkpoint405/path",
            "/evaluation_only_origin_sha256",
        ],
        "copies_sha256": {name: info["sha256"] for name, info in copied_files.items()},
    }
    local_names = set(copied_files) | {
        "training.json",
        "copy.json",
        "evaluation/result.json",
    }
    copy_arguments = (
        copy_record,
        original_raw,
        projected_raw,
        spec,
        destination,
        copied_files,
        source_files,
        local_names,
    )
    proof = budget_evaluation_copy(*copy_arguments)
    require(
        proof["verified"]
        and proof["new_optimizer_updates"] == 0
        and proof["inherited_training_manifest"]["checkpoint405/optimizer.pt"]["path"]
        == original_root + "/study/checkpoint405/optimizer.pt"
        and proof["inherited_training_manifest"]["checkpoint405/train_probes.json"][
            "path"
        ]
        == destination + "/checkpoint405/train_probes.json",
        "typed copy must resolve inherited references at their actual locations",
    )
    passed.append(
        "typed zero-update copy resolves copied probes locally and uncopied optimizer at original study"
    )
    for label, changes in (
        ("copy from wrong original run", {"source_run": REMOTE + "runs/other"}),
        ("copy wrong origin receipt hash", {"origin_training_sha256": "b" * 64}),
        ("copy wrong projected receipt hash", {"projected_training_sha256": "b" * 64}),
        ("copy cannot claim new training updates", {"new_optimizer_updates": 1}),
        ("copy wrong typed kind", {"kind": "training"}),
        ("copy cannot include original optimizer", {"original_optimizer_copied": True}),
        (
            "copy cannot authorize extra receipt changes",
            {
                "exact_allowed_json_changes": [
                    "/checkpoint405/path",
                    "/evaluation_only_origin_sha256",
                    "/pid",
                ]
            },
        ),
        (
            "copy declared file set incomplete",
            {
                "copies_sha256": {
                    name: checksum
                    for name, checksum in copy_record["copies_sha256"].items()
                    if name != "parameter_mapping.json"
                }
            },
        ),
    ):
        rejects(
            label,
            lambda changes=changes: budget_evaluation_copy(
                copy_record | changes, *copy_arguments[1:]
            ),
        )
    rejects(
        "copy original receipt semantic equality cannot replace exact bytes",
        lambda: budget_evaluation_copy(
            copy_record, original_raw + b" ", *copy_arguments[2:]
        ),
    )
    for label, changes in (
        ("copy altered original training PID", {"pid": 102}),
        ("copy altered original loss token counts", {"tokens": {"additional": 587}}),
        (
            "copy checkpoint relocated to wrong destination",
            {
                "checkpoint405": projected["checkpoint405"]
                | {"path": destination + "/wrong"}
            },
        ),
    ):
        changed_raw = supervisor_bytes(projected | changes)
        rejects(
            label,
            lambda changed_raw=changed_raw: budget_evaluation_copy(
                copy_record | {"projected_training_sha256": sha(changed_raw)},
                original_raw,
                changed_raw,
                *copy_arguments[3:],
            ),
        )
    missing_copy = {
        name: info
        for name, info in copied_files.items()
        if name != "checkpoint405/learner/adapter_model.safetensors"
    }
    rejects(
        "missing copied adapter cannot be skipped",
        lambda: budget_evaluation_copy(
            *copy_arguments[:5], missing_copy, source_files, local_names
        ),
    )
    bad_copy = copied_files | {
        "checkpoint405/learner/adapter_model.safetensors": {
            "sha256": "b" * 64,
            "bytes": 1,
        }
    }
    rejects(
        "changed copied adapter bytes",
        lambda: budget_evaluation_copy(
            *copy_arguments[:5], bad_copy, source_files, local_names
        ),
    )
    missing_original = {
        name: info
        for name, info in source_files.items()
        if name != "study/checkpoint405/optimizer.pt"
    }
    rejects(
        "uncopied original optimizer still required",
        lambda: budget_evaluation_copy(
            *copy_arguments[:6], missing_original, local_names
        ),
    )
    bad_original = source_files | {
        "study/ledger.json": {"sha256": "b" * 64, "bytes": 1}
    }
    rejects(
        "uncopied original ledger still hash-checked",
        lambda: budget_evaluation_copy(*copy_arguments[:6], bad_original, local_names),
    )
    rejects(
        "unreported original optimizer copied into evaluator",
        lambda: budget_evaluation_copy(
            *copy_arguments[:7], local_names | {"checkpoint405/optimizer.pt"}
        ),
    )
    artifact = {
        "directory": REMOTE
        + "artifacts/followthrough-20260912-native-consolidated303/export",
        "files": {
            f"model/model-{i:05d}-of-00009.safetensors": {
                "sha256": "a" * 64,
                "bytes": 4,
            }
            for i in range(1, 10)
        },
        "manifest": {"sha256": "b" * 64, "bytes": 20},
        "tensor_payload_bytes": 36,
        "weight_elements": 9,
    }
    artifact["shards"] = {
        name.removeprefix("model/"): pin for name, pin in artifact["files"].items()
    }
    require(
        len(native_artifact_pins(artifact)) == 10,
        "native model manifest plus every shard pinned",
    )
    passed.append("native external manifest and all nine shards registered")
    rejects(
        "native shard subset mismatch",
        lambda: native_artifact_pins(artifact | {"shards": {}}),
    )
    rejects(
        "native artifact root changed",
        lambda: native_artifact_pins(artifact | {"directory": REMOTE + "other"}),
    )
    rejects(
        "native artifact relative escape",
        lambda: native_artifact_pins(
            artifact
            | {"files": artifact["files"] | {"../escape.json": {"sha256": "c" * 64}}}
        ),
    )
    payload = b"print('bounded-transport-control')\n"
    observed = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-B",
            "-c",
            remote_bootstrap(payload),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    require(
        observed.stdout == b"bounded-transport-control\n",
        "compressed remote transport exact program",
    )
    passed.append("bounded compressed script transport executes exact program bytes")
    return {
        "passed": len(passed),
        "cases": passed,
        "fixture_scope": "In-memory audit controls; no experiment or provider execution.",
    }


def audit(root, verify_remote=False, baseline=None):
    started = datetime.now(timezone.utc).isoformat()
    selected, excluded = collect_inventory(root)
    require(selected, "no terminal collected runs")
    archive_paths = sorted((root / "code").glob("*.tar"))
    files, issues, archives, states = Files(), [], {}, {}
    worker_snapshot = execute_remote_plan(
        {
            "scope": "worker_implementation",
            "paths": sorted(IMPLEMENTATION_COPIES),
            "git_blob_paths": [],
        },
        verify_remote,
        Path(__file__).resolve(),
        issues,
        str(root),
    )
    for path in archive_paths:
        report, members = verify_archive(path, files, issues)
        archives[path.stem] = {"report": report, "members": members}
    for index, (name, path) in enumerate(selected.items(), 1):
        print(f"PROVENANCE_AUDIT {index}/{len(selected)} {name}", flush=True)
        state = states[name] = verify_run(path, files, issues)
        source = run_source(state)
        archive = archives.get(source)
        state["report"]["source_archive_identity_basis"] = (
            "separate_submitted_dispatch"
            if state.get("submitted_task")
            else "embedded_execution_source"
        )
        state["report"]["source_archive_closure_verified"] = bool(
            archive and archive["report"]["closure_verified"]
        )
        if archive is None:
            issue(
                issues,
                "gap",
                "source_archive_not_collected",
                name,
                f"{root / 'code'}/{source}.tar",
            )
        elif state.get("task"):
            task = state["task"]
            for required in (
                task.get("entrypoint", "run.py"),
                task.get("requirements", "requirements.txt"),
            ):
                if required not in archive["members"]:
                    issue(
                        issues,
                        "error",
                        "entrypoint_or_requirements_absent_from_archive",
                        name,
                        required,
                    )
        state["report"]["scientific_outcome"] = scientific_outcome(state, files)
    bindings = Bindings(root, states, archives, files, issues)
    for state in states.values():
        if state.get("submitted_task"):
            try:
                binding = state["report"]["submitted_dispatch"]
                origin = Path(binding["sidecar"]["path"]).name
                bindings.pin(
                    binding["remote_queue_observation"]["path"],
                    binding["remote_queue_observation"],
                    state,
                    origin,
                )
                bindings.protocol(
                    state["submitted_task"]["config"],
                    {},
                    state,
                    origin,
                )
                bindings.counts["separate_blocked_dispatch_bindings_verified"] += 1
            except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
                issue(
                    issues,
                    "error",
                    "blocked_submitted_dispatch_binding",
                    state["path"].name,
                    error,
                )
        verify_documents(state, bindings, issues)
    try:
        dependencies = verify_dependencies(states, issues)
    except (AuditError, ValueError, KeyError, TypeError) as error:
        issue(issues, "error", "dependency_graph_integrity", str(root), error)
        dependencies = {"acyclic": False}
    try:
        runtimes = verify_runtimes(root, states, files, issues)
    except (AuditError, ValueError, KeyError, TypeError) as error:
        issue(issues, "error", "runtime_or_interruption_proof", str(root), error)
        runtimes = {}
    external = verify_external_pins(
        bindings, verify_remote, Path(__file__).resolve(), worker_snapshot
    )
    for state in states.values():
        if state.get("submitted_task"):
            binding = state["report"]["submitted_dispatch"]
            binding["remote_queue_reverification"] = next(
                (
                    check
                    for check in external["sha256_pin_checks"]
                    if check["expected"]["path"]
                    == binding["remote_queue_observation"]["path"]
                ),
                None,
            )
    requested_pins = reconcile_requested_pins(bindings, baseline or [], external)
    for name, run in states.items():
        actual_names = set(run["actual"]) | {"collection.json"}
        current_names = {
            str(p.relative_to(run["path"]))
            for p in run["path"].rglob("*")
            if p.is_file()
        }
        stable = actual_names == current_names
        run["report"]["snapshot_file_set_unchanged_at_finish"] = stable
        if not stable:
            issue(
                issues,
                "error",
                "collection_file_set_changed_during_audit",
                name,
                sorted(actual_names ^ current_names),
            )
    finished_selected, finished_excluded = collect_inventory(root)
    added = sorted(set(finished_selected) - set(selected))
    removed = sorted(set(selected) - set(finished_selected))
    if removed:
        issue(issues, "error", "collections_removed_during_audit", str(root), removed)
    try:
        files.stable()
    except (AuditError, OSError) as error:
        issue(issues, "error", "input_changed_during_audit", str(root), error)
    errors = sum(i["severity"] == "error" for i in issues)
    gaps = sum(i["severity"] == "gap" for i in issues)
    for name, run in states.items():
        own = [item for item in issues if item["scope"] == name]
        run["report"]["audit_status"] = (
            "integrity_error"
            if any(i["severity"] == "error" for i in own)
            else "verified_with_gaps"
            if own
            else "verified"
        )
    return {
        "audit_completed": True,
        "status": "integrity_errors_found"
        if errors
        else "collected_integrity_verified_with_provenance_gaps"
        if gaps
        else "collected_integrity_verified",
        "snapshot_started_at": started,
        "snapshot_finished_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_scope": str(root),
        "collected_run_count": len(states),
        "execution_counts": dict(
            Counter(
                r["report"].get("execution_status", "unreadable")
                for r in states.values()
            )
        ),
        "scientific_stage_counts": dict(
            Counter(
                r["report"]["scientific_outcome"]["classification"]
                for r in states.values()
            )
        ),
        "integrity_error_count": errors,
        "provenance_gap_count": gaps,
        "issues": issues,
        "runs": {name: state["report"] for name, state in states.items()},
        "source_archives": {name: value["report"] for name, value in archives.items()},
        "runtime_snapshots": runtimes,
        "dependencies": dependencies,
        "explicit_binding_checks": dict(bindings.counts),
        "external_file_pins_not_locally_rehashed": sorted(
            bindings.unavailable.values(), key=lambda v: (v["path"], v["sha256"])
        ),
        "unavailable_external_file_pins": [
            c["expected"] for c in external["sha256_pin_checks"] if not c["verified"]
        ],
        "external_pin_verification": external,
        "external_pin_request_baseline": baseline or [],
        "requested_external_pin_resolution": requested_pins,
        "git_blob_input_pins": bindings.git_blob_inputs,
        "unverified_git_blob_pins": [
            c["expected"] for c in external["git_blob_checks"] if not c["verified"]
        ],
        "unavailable_canonical_protocol_pins": bindings.unavailable_protocols,
        "external_collected_run_files_checked": bindings.external_run_files,
        "hashing": dict(files.stats)
        | {
            "persistent_hash_cache_used": False,
            "each_physical_path_hashed_at_most_once_this_invocation": True,
        },
        "excluded_at_start": excluded,
        "excluded_at_finish": finished_excluded,
        "collections_added_after_snapshot_not_audited": added,
        "source_archives_added_after_snapshot_not_audited": sorted(
            {p.name for p in (root / "code").glob("*.tar")}
            - {p.name for p in archive_paths}
        ),
        "boundaries": [
            "Execution completion, scientific qualification and successful scientific findings are distinct. This audit does not rescore predictions or reproduce training.",
            "An unlaunched blocked task may have separately verified submitted manifest and saved queue-byte evidence. Its submitted source archive/protocol are checked without inserting task/source/runtime fields into the execution receipt; submitted identity is not proof of model execution. Optional current queue hashing is reported separately from the saved observation.",
            "Each complete local collection is hashed against its recorded file manifest; every execution predecessor snapshot and source archive closure is checked, including failed/interrupted attempts.",
            "Study receipt closures, declared artifact subsets, explicit path/hash pointers, adapter/optimizer checkpoint pins and available source-file hashes are checked. Tensor-content hashes without a file-byte contract are not recomputed.",
            "External SHA256 pins and canonical Git blob identities have separate verification results. Remote reads are opt-in and restricted to the two authorized NFS roots on the control pod and two exact portallib paths on the named active worker. Local source copies are checked only at the explicitly authorized paths. Only status/hash/byte-count metadata is returned; missing sizes remain unknown metadata.",
            "External prior-wave files are verified only when explicitly referenced; their entire runs and dependencies are outside this snapshot unless listed among the audited runs.",
            "Runtime ConfigMaps and interruption pod proofs are local saved snapshots. Optional provider operations perform read-only hashing through the named control and active worker pods. An identical implementation copy establishes matching source bytes, not observation of the dead process or its old temporary environment. No provider or remote file mutations are performed.",
            "The input set is fixed at invocation start. A rerun refreshes only this derived audit JSON and fully rehashes its current collected snapshot; experiment history remains untouched.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-remote", action="store_true")
    parser.add_argument(
        "--remote-hash-stdin", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.remote_hash_stdin:
        return remote_hash_main()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests, indent=2))
        return 0
    script = Path(__file__).resolve()
    output = script.with_suffix(".json")
    previous_raw = output.read_bytes() if output.is_file() else None
    previous = sha(previous_raw) if previous_raw else None
    prior = decode(previous_raw) if previous_raw else {}
    baseline = prior.get(
        "external_pin_request_baseline", prior.get("unavailable_external_file_pins", [])
    )
    result = audit(script.parent, args.verify_remote, baseline)
    result.update(
        auditor_sha256=sha(script.read_bytes()),
        self_tests=tests,
        previous_derived_report_sha256=previous,
        python_version=sys.version,
        readonly_inputs=True,
        provider_operations=result["external_pin_verification"]["read_only_exec_calls"],
        provider_mutations=0,
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    summary = {
        key: result[key]
        for key in (
            "status",
            "collected_run_count",
            "execution_counts",
            "scientific_stage_counts",
            "integrity_error_count",
            "provenance_gap_count",
            "hashing",
        )
    }
    summary.update(
        report=str(output),
        report_sha256=sha(output.read_bytes()),
        unavailable_external_file_pins=len(result["unavailable_external_file_pins"]),
        external_sha256_status_counts=result["external_pin_verification"][
            "sha256_status_counts"
        ],
        git_blob_status_counts=result["external_pin_verification"][
            "git_blob_status_counts"
        ],
        requested_external_pin_resolution=result["requested_external_pin_resolution"][
            "location_counts"
        ],
    )
    print(json.dumps(summary, indent=2), flush=True)
    return (
        2
        if result["integrity_error_count"]
        else 1
        if result["provenance_gap_count"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
