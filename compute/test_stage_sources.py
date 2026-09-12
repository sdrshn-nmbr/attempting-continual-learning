import hashlib
import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from stage_sources import install


def bundle(root, name, files):
    path = root / f".incoming-{name}.tar"
    digest = hashlib.sha256()
    with tarfile.open(path, "w") as archive:
        for relative in sorted(files, key=Path):
            payload = files[relative]
            entry = tarfile.TarInfo(relative)
            entry.size = len(payload)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(payload))
            digest.update(relative.encode() + b"\0" + payload)
    return path, digest.hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest()


def test_reuse_preserves_open_files_inodes_and_timestamps(tmp_path):
    files = {"data.py": b"VALUE = 7\n", "data/fixture.json": b"{}\n"}
    incoming, identity, checksum = bundle(tmp_path, "first", files)
    assert install(incoming, tmp_path, identity, checksum)["source_installation"] == "atomically_installed"
    target = tmp_path / identity
    code = target / "data.py"
    archive = tmp_path / f"{identity}.tar"
    before = [(p.stat().st_ino, p.stat().st_mtime_ns) for p in (code, archive)]
    cache = target / "__pycache__"
    cache.mkdir()
    (cache / "data.pyc").write_bytes(b"runtime cache")
    incoming, repeated, checksum = bundle(tmp_path, "second", files)
    with code.open("rb") as reader:
        receipt = install(incoming, tmp_path, repeated, checksum)
        assert reader.read() == files["data.py"]
    assert receipt["source_installation"] == "reused_without_writes"
    assert before == [(p.stat().st_ino, p.stat().st_mtime_ns) for p in (code, archive)]


def test_concurrent_publication_installs_one_complete_source(tmp_path):
    files = {"runner.py": b"VALUE = 7\n" * 10000, "configs/run.json": b"{}\n"}
    first = bundle(tmp_path, "one", files)
    second = bundle(tmp_path, "two", files)
    start = Barrier(2)

    def publish(spec):
        start.wait()
        incoming, identity, checksum = spec
        return install(incoming, tmp_path, identity, checksum)

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(publish, (first, second)))
    assert sorted(x["source_installation"] for x in receipts) == ["atomically_installed", "reused_without_writes"]
    for relative, expected in files.items():
        assert (tmp_path / first[1] / relative).read_bytes() == expected
    assert not list(tmp_path.glob(".source-stage-*"))
    assert not list(tmp_path.glob(".incoming-*"))


def test_tampered_existing_source_is_rejected_without_repair(tmp_path):
    files = {"runner.py": b"VALUE = 7\n"}
    incoming, identity, checksum = bundle(tmp_path, "one", files)
    install(incoming, tmp_path, identity, checksum)
    target = tmp_path / identity / "runner.py"
    target.write_bytes(b"partial import")
    incoming, identity, checksum = bundle(tmp_path, "two", files)
    with pytest.raises(ValueError, match="SOURCE_DIRECTORY_BYTES_MISMATCH"):
        install(incoming, tmp_path, identity, checksum)
    assert target.read_bytes() == b"partial import"


def test_bad_archive_hash_never_publishes(tmp_path):
    incoming, identity, _ = bundle(tmp_path, "bad", {"runner.py": b"VALUE = 7\n"})
    with pytest.raises(ValueError, match="ARCHIVE_HASH_MISMATCH"):
        install(incoming, tmp_path, identity, "0" * 64)
    assert not (tmp_path / identity).exists()
    assert not (tmp_path / f"{identity}.tar").exists()


def test_archive_escape_never_publishes(tmp_path):
    incoming, identity, checksum = bundle(tmp_path, "escape", {"../escaped.py": b"bad"})
    with pytest.raises(ValueError, match="ARCHIVE_MEMBER_INVALID"):
        install(incoming, tmp_path, identity, checksum)
    assert not (tmp_path.parent / "escaped.py").exists()
    assert not (tmp_path / identity).exists()


def test_canonical_archive_cannot_be_used_as_scratch(tmp_path):
    incoming, identity, checksum = bundle(tmp_path, "first", {"runner.py": b"VALUE = 7\n"})
    install(incoming, tmp_path, identity, checksum)
    canonical = tmp_path / f"{identity}.tar"
    with pytest.raises(ValueError, match="INCOMING_PATH_INVALID"):
        install(canonical, tmp_path, identity, checksum)
    assert canonical.exists()
