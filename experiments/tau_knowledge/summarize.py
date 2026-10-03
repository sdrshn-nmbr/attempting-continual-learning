"""Score the τ-Knowledge pilot: pass^k, action recall and terminations per model and condition, plus the
gate (gold documents must beat no knowledge by at least GATE_TASKS tasks' worth of pass^1)."""
import argparse
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

TASK_COUNT = 97
GATE_TASKS = 15
CONDITIONS = ("no_knowledge", "golden_retrieval")


def log(message):
    print(f"[summarize] {message}", flush=True)


def success(reward):
    return reward is not None and abs(reward - 1.0) <= 1e-6


def load(path):
    data = json.loads(path.read_text())
    simulations = data["simulations"]
    if not simulations:
        raise SystemExit(f"NO_SIMULATIONS {path}")
    outcomes, recalls, terminations = defaultdict(list), [], Counter()
    for simulation in simulations:
        reward_info = simulation.get("reward_info") or {}
        outcomes[simulation["task_id"]].append(success(reward_info.get("reward")))
        checks = reward_info.get("action_checks") or []
        if checks:
            recalls.append(sum(c["action_match"] for c in checks) / len(checks))
        terminations[str(simulation.get("termination_reason"))] += 1
    return {"outcomes": dict(outcomes), "recalls": recalls, "terminations": dict(terminations),
            "agent": data["info"]["agent_info"], "user_llm": data["info"]["user_info"]["llm"],
            "user_args": data["info"]["user_info"]["llm_args"], "commit": data["info"].get("git_commit")}


def pass_hat(outcomes, k):
    usable = [o for o in outcomes.values() if len(o) >= k]
    return sum(comb(sum(o), k) / comb(len(o), k) for o in usable) / len(usable)


def task_rate(outcomes):
    return {task: sum(o) / len(o) for task, o in outcomes.items()}


def paired_gap(gold, floor, samples=4000, seed=0):
    tasks = sorted(set(gold) & set(floor))
    gold_rate, floor_rate = task_rate(gold), task_rate(floor)
    gaps = [gold_rate[t] - floor_rate[t] for t in tasks]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(gaps) for _ in gaps) / len(gaps) for _ in range(samples))
    return sum(gaps) / len(gaps), means[int(0.025 * samples)], means[int(0.975 * samples)], len(tasks)


def summarize_cell(cell):
    outcomes = cell["outcomes"]
    trials = min(len(o) for o in outcomes.values())
    return {"tasks": len(outcomes), "trials": trials,
            "pass_hat": {k: round(100 * pass_hat(outcomes, k), 2) for k in range(1, trials + 1)},
            "action_recall": round(100 * sum(cell["recalls"]) / len(cell["recalls"]), 2) if cell["recalls"] else None,
            "terminations": cell["terminations"], "agent": cell["agent"], "user_llm": cell["user_llm"],
            "user_args": cell["user_args"], "tau2_commit": cell["commit"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True,
                        help="directory holding <model>-<condition>/results.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cells = {}
    for path in sorted(args.results.glob("*/results.json")):
        model, _, condition = path.parent.name.rpartition("-")
        if condition not in CONDITIONS:
            raise SystemExit(f"UNKNOWN_CONDITION dir={path.parent.name}")
        cells[(model, condition)] = load(path)
        log(f"loaded {path.parent.name}: {len(cells[(model, condition)]['outcomes'])} tasks")

    report = {"gate_tasks": GATE_TASKS, "task_count": TASK_COUNT, "models": {}}
    for model in sorted({m for m, _ in cells}):
        entry = {c: summarize_cell(cells[(model, c)]) for c in CONDITIONS if (model, c) in cells}
        if all((model, c) in cells for c in CONDITIONS):
            gold, floor = cells[(model, "golden_retrieval")]["outcomes"], cells[(model, "no_knowledge")]["outcomes"]
            gap, low, high, shared = paired_gap(gold, floor)
            complete = shared == TASK_COUNT
            decision = ("proceed" if gap * TASK_COUNT >= GATE_TASKS else "stop") if complete else "incomplete"
            entry["gap"] = {"pass1_points": round(100 * gap, 2), "ci95": [round(100 * low, 2), round(100 * high, 2)],
                            "task_equivalents": round(gap * TASK_COUNT, 1), "shared_tasks": shared,
                            "decision": decision}
            if not complete:
                log(f"GATE_INCOMPLETE model={model} shared_tasks={shared} expected={TASK_COUNT}")
        report["models"][model] = entry

    args.out.write_text(json.dumps(report, indent=2))
    for model, entry in report["models"].items():
        for condition in CONDITIONS:
            if condition in entry:
                cell = entry[condition]
                log(f"{model:>28} {condition:<17} tasks={cell['tasks']} trials={cell['trials']} "
                    f"pass^k={cell['pass_hat']} action_recall={cell['action_recall']}")
        if "gap" in entry:
            gap = entry["gap"]
            log(f"{model:>28} gap={gap['pass1_points']} pts ci95={gap['ci95']} "
                f"= {gap['task_equivalents']} tasks -> {gap['decision']}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
