import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    source = args.wave / "runs/followthrough-20260912-recovery-predict-analyze/study/analysis.json"
    audit_path = args.wave / "recovery-prediction-audit.final.json"
    analysis = json.loads(source.read_text())
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "complete_verified" or audit["missing_stages"]:
        raise ValueError("RECOVERY_FIGURE_REQUIRES_COMPLETE_INDEPENDENT_AUDIT")
    methods = ["always/2", "always/4", "always/8", "always/16", "frozen", "tuned"]
    labels = ["Fixed 2", "Fixed 4", "Fixed 8", "Fixed 16", "Frozen\nreadout", "Tuned\nreadout"]
    colors = ["#246b88", "#b66a35"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(methods))
    data = {}
    for offset, (stream, count) in enumerate(analysis["source_counts"].items()):
        fractions = [analysis["policies"][m]["qualified_fraction"]["by_source"][stream] for m in methods]
        bars = axes[0].bar(x + (offset - 0.5) * 0.36, np.array(fractions) * 100, width=0.36,
                           label=f"Held-out model {offset + 1} ({count} skills)", color=colors[offset])
        for bar, fraction in zip(bars, fractions, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                         f"{round(fraction * count)}/{count}", ha="center", fontsize=9)
        data[stream] = {m: {"qualified": round(f * count), "n": count} for m, f in zip(methods, fractions, strict=True)}
    axes[0].set(title="Skills recovered while preserving new tasks", ylabel="Qualified repairs (%)", ylim=(0, 117))
    axes[0].set_xticks(x, labels)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.19), frameon=False, ncol=2, fontsize=10)
    steps = [analysis["policies"][m]["mean_updates"]["pooled"] for m in methods]
    bars = axes[1].bar(x, steps, width=0.62, color=["#9da7af"] * 4 + ["#576877", "#583d72"])
    for bar, value in zip(bars, steps, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:g}" if value.is_integer() else f"{value:.2f}", ha="center", fontsize=10)
    axes[1].set(title="Updates selected by each policy", ylabel="Mean updates per skill (13 skills)", ylim=(0, 18))
    axes[1].set_xticks(x, labels)
    axes[1].set_yticks([0, 4, 8, 12, 16])
    for ax in axes:
        ax.grid(axis="y", color="#e5e8eb", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Internal readouts did not improve repair allocation", x=0.065, ha="left", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.065, 0.90, "Forecasts were sealed before held-out repairs. Every fixed budget was specified in advance.", fontsize=11)
    fig.text(0.065, 0.075, "Two independent held-out models; 16-code intent classification. TRAIN-majority, output and confidence policies equal Fixed 2.", fontsize=9, color="#48545f")
    fig.text(0.065, 0.035, "All repair budgets were actually run. The right panel estimates policy allocations; it does not report saved study compute.", fontsize=9, color="#48545f")
    fig.subplots_adjust(left=0.065, right=0.98, top=0.81, bottom=0.29, wspace=0.26)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.wave / f"recovery-prediction.{extension}", dpi=180, facecolor="white")
    metadata = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "independent_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "methods": methods, "source_counts": analysis["source_counts"], "qualification": data,
        "pooled_mean_updates": dict(zip(methods, steps, strict=True)), "independent_model_clusters": 2,
        "uncertainty": "Descriptive paired source results; no intent-level confidence intervals.",
    }
    (args.wave / "recovery-prediction.figure.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figure": str(args.wave / "recovery-prediction.png"), "audit": audit["status"]}))


if __name__ == "__main__":
    main()
