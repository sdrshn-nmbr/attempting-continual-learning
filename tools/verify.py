import argparse
import hashlib
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("verify")


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, artifacts):
    snapshot = read_json(root / "research/snapshot.json")
    failures = []
    checked = 0
    records = {row["path"]: row for row in snapshot["files"]}
    for row in records.values():
        if row["storage"] != "git" and not artifacts:
            continue
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"MISSING: {row['path']}")
        elif path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            failures.append(f"CONTENT_CHANGED: {row['path']}")
        checked += 1
    evidence_root = root / "site/evidence"
    evidence = read_json(evidence_root / "manifest.json")
    for row in evidence:
        path = evidence_root / f"{row['id']}.json"
        original = str(
            Path(row["original_path"]).relative_to(snapshot["source_workspace"])
        )
        if original not in records or records[original]["sha256"] != row["sha256"]:
            failures.append(f"PROVENANCE_MISMATCH: {row['id']}")
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or file_hash(path) != row["sha256"]
        ):
            failures.append(f"EVIDENCE_CHANGED: {row['id']}")
        else:
            read_json(path)
        checked += 1
    if failures:
        return checked, failures
    sequence = read_json(evidence_root / "replay-sequences.json")
    expected = {
        "lora_replay": [191, 192],
        "lora_no_replay": [80, 89],
        "native_replay": [183, 157],
        "native_no_replay": [120, 130],
    }
    for arm, counts in expected.items():
        observed = [
            sum(
                sequence["arms"][arm]["fixtures"][seed]["source_counts_out_of_64"][
                    "after_c"
                ]
            )
            for seed in ["303", "419"]
        ]
        if observed != counts:
            failures.append(f"SEQUENCE_COUNTS_MISMATCH: {arm}: {observed}")
    rank = read_json(evidence_root / "rank.json")
    for arm, count in [("full_final", 862), ("rank8_final", 850)]:
        if (
            sum(rank["counts"][arm].values()) != count
            or rank[f"{arm}_correct"] != count
        ):
            failures.append(f"RANK_COUNTS_MISMATCH: {arm}")
    return checked, failures


def main():
    parser = argparse.ArgumentParser(
        description="Verify the research snapshot without loading models or accessing the network."
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Also require and hash every file restored from both release archives.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    checked, failures = verify(ROOT, args.artifacts)
    for failure in failures:
        LOG.error(failure)
    if failures:
        raise SystemExit(1)
    LOG.info(
        "VERIFIED: %s files; evidence provenance and headline arithmetic agree.",
        checked,
    )
    if not args.artifacts:
        LOG.info(
            "Release artifacts were not checked. Restore both archives and use --artifacts to verify them."
        )


if __name__ == "__main__":
    main()
