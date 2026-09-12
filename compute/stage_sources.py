import argparse
import errno
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def deployable_paths(directory):
    excluded = {"__pycache__", "tests", "outputs", "runs"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") or part in excluded
            for part in path.relative_to(directory).parts
        )
        and (
            path.suffix in {".py", ".json", ".txt", ".yaml", ".yml", ".sha256"}
            or path.name.startswith(("LICENSE", "NOTICE"))
        )
    )


def verify_directory(directory, contents):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"SOURCE_DIRECTORY_INVALID: {directory}")
    files = deployable_paths(directory)
    if {str(path.relative_to(directory)) for path in files} != set(contents):
        raise ValueError(f"SOURCE_DIRECTORY_MANIFEST_MISMATCH: {directory}")
    for path in files:
        if path.is_symlink() or path.read_bytes() != contents[str(path.relative_to(directory))]:
            raise ValueError(f"SOURCE_DIRECTORY_BYTES_MISMATCH: {path}")


def install(incoming, root, source_sha256, archive_sha256):
    root = root.resolve()
    if (
        incoming.parent.resolve() != root
        or not incoming.name.startswith(".incoming-")
        or incoming.is_symlink()
    ):
        raise ValueError("SOURCE_INCOMING_PATH_INVALID")
    contents = {}
    temporary = None
    try:
        if hashlib.sha256(incoming.read_bytes()).hexdigest() != archive_sha256:
            raise ValueError("SOURCE_INCOMING_ARCHIVE_HASH_MISMATCH")
        with tarfile.open(incoming) as archive:
            members = sorted(archive.getmembers(), key=lambda item: Path(item.name))
            for member in members:
                name = Path(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or str(name) != member.name
                    or member.name in contents
                ):
                    raise ValueError(f"SOURCE_ARCHIVE_MEMBER_INVALID: {member.name}")
                contents[member.name] = archive.extractfile(member).read()
        digest = hashlib.sha256()
        for name, payload in contents.items():
            digest.update(name.encode() + b"\0" + payload)
        if not contents or digest.hexdigest() != source_sha256:
            raise ValueError("SOURCE_ARCHIVE_CLOSURE_MISMATCH")
        target = root / source_sha256
        canonical_archive = root / f"{source_sha256}.tar"
        if canonical_archive.is_symlink():
            raise ValueError("SOURCE_CANONICAL_ARCHIVE_IS_SYMLINK")
        try:
            os.link(incoming, canonical_archive)
        except FileExistsError:
            if hashlib.sha256(canonical_archive.read_bytes()).hexdigest() != archive_sha256:
                raise ValueError("SOURCE_CANONICAL_ARCHIVE_CHANGED") from None
        reused = target.exists()
        if reused:
            verify_directory(target, contents)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".source-stage-", dir=root))
            for name, payload in contents.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                path.chmod(0o644)
            verify_directory(temporary, contents)
            try:
                os.rename(temporary, target)
                temporary = None
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                verify_directory(target, contents)
                reused = True
        return {
            "source_sha256": source_sha256,
            "source_files": len(contents),
            "source_installation": "reused_without_writes" if reused else "atomically_installed",
        }
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        incoming.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.incoming, args.root, args.source_sha256, args.archive_sha256)))


if __name__ == "__main__":
    main()
