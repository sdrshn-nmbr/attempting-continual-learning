import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    state = json.loads((args.root / "orchestrator-state.json").read_text())
    required = state.get(
        "required_lanes", ["catalog", "portability", "active_evidence"]
    )
    gaps = []
    for lane in required:
        entry = state["lanes"][lane]
        if entry["status"] != "reviewed_gpu_results":
            gaps.append(f"{lane}: {entry['status']}")
        evidence = entry.get("evidence", [])
        if not evidence:
            gaps.append(f"{lane}: no GPU evidence recorded")
        for receipt in evidence:
            path = Path(receipt["path"])
            if not path.is_file():
                gaps.append(f"{lane}: missing {path}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                gaps.append(f"{lane}: evidence changed at {path}")
        if entry.get("promising", False) and not entry.get(
            "independent_replication_passed", False
        ):
            gaps.append(f"{lane}: promising finding awaits independent replication")
    expected_ports = 7
    qualified_ports = state["lanes"]["catalog"].get("gpu_qualified_ports", [])
    if len(set(qualified_ports)) != expected_ports:
        gaps.append(
            f"catalog: {len(set(qualified_ports))}/{expected_ports} ports qualified on GPUs"
        )
    if args.verify:
        print("INCOMPLETE" if gaps else "COMPLETE")
    else:
        print(
            json.dumps(
                {"complete": not gaps, "gaps": gaps, "lanes": state["lanes"]}, indent=2
            )
        )
    raise SystemExit(bool(gaps))


if __name__ == "__main__":
    main()
