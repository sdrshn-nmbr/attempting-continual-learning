import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import fmean

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

matplotlib.use("Agg")

ANALYSIS_SHA256 = "0f3477af996ab9e86f4b91ef903074917127069c3d5be3e20c9a93a9248fa86e"
AUDIT_SHA256 = "d4c70e2dc03634d1f595eda951ab889fbf051bc07b4bf02541e27393bdf5fe1f"
BUDGETS = (2, 4, 8, 16)
SOURCES = {"fresh-test-a": 7, "fresh-test-b": 6}
HISTORIES = {
    "original_history": {
        "label": "Original history\n128 original + 128 new updates",
        "color": "#246883", "marker": "o", "linestyle": "-",
    },
    "new_only": {
        "label": "New-only\n128 new updates; lifetime unmatched",
        "color": "#B7642E", "marker": "s", "linestyle": (0, (5, 2)),
    },
    "alternate_binding": {
        "label": "Alternate binding\n128 rotated + 128 new; lifetime matched",
        "color": "#7857A1", "marker": "^", "linestyle": (0, (2, 2)),
    },
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("HISTORY_FIGURE: " + message)


def source_data(wave):
    run = wave / "runs/followthrough-20260912-recovery-history-analyze"
    source = run / "study/analysis.json"
    collection = read(run / "collection.json")
    require(collection["execution_status"] == "completed", "analysis collection incomplete")
    for name, expected in collection["files"].items():
        path = run / name
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], "collected file " + name)
    require(sha(source) == ANALYSIS_SHA256, "analysis differs from designated immutable result")
    analysis = read(source)
    receipt = read(run / "study/receipt.json")
    execution = read(run / "execution.json")
    task = read(run / "task.json")
    require(execution["status"] == "completed" and execution["exit_code"] == 0 and not execution["timed_out"], "analysis execution incomplete")
    require(execution["task"] == task and execution["task_id"] == run.name, "analysis task identity")
    require(digest(receipt["payload"]) == receipt["sha256"], "analysis receipt digest")
    require(receipt["payload"]["analysis_sha256"] == ANALYSIS_SHA256, "receipt analysis binding")
    audit_path = wave / "audit_history_controls.json"
    require(sha(audit_path) == AUDIT_SHA256, "designated completed independent audit")
    audit = read(audit_path)
    require(audit["status"] == "complete_verified" and not audit["issues"], "independent audit completion")
    require(not audit["snapshot"]["pending"] and not audit["snapshot"]["changed_during_audit"] and not audit["execution_failed_or_interrupted"], "independent audit closure")
    require(audit["self_tests"]["passed"] and audit["self_tests"]["tests_run"] == 9 and not audit["self_tests"]["failures"] and not audit["self_tests"]["errors"], "auditor CPU controls")
    require(sha(wave / "audit_history_controls.py") == audit["auditor"]["sha256"], "frozen independent auditor source")
    require(audit["collected_analysis"]["status"] == "verified_collected_analysis" and audit["collected_analysis"]["per_unit_curves_and_paired_results_recomputed"], "independently recomputed analysis")
    require(audit["collected_analysis"]["receipt"]["sha256"] == sha(run / "study/receipt.json"), "audited analysis receipt identity")
    require(audit["snapshot"]["runs"][run.name]["collection"]["sha256"] == sha(run / "collection.json"), "audited analysis collection identity")
    require(audit["verification"]["raw_native_records_checked"] == 86208, "audited panel record count")
    require(analysis["aggregate"]["test"]["source_clusters"] == 2, "two held-out source clusters")
    require(analysis["aggregate"]["test"]["intents"] == 13, "thirteen paired held-out intents")
    protocol_path = wave.parents[2] / "experiments/recoverability/history-control-protocol.json"
    protocol = read(protocol_path)
    require(sha(protocol_path) == analysis["design_inputs"]["protocol_sha256"], "qualification protocol file binding")
    require(protocol["repair"]["budgets"] == list(BUDGETS), "predefined repair budgets")
    qualification = protocol["repair"]["qualification"]
    require(qualification == {"accuracy_min": 0.8, "gain_min": 0.3, "guard_drop_max": 0.1}, "qualification caption")
    plotted = {}
    for name, count in SOURCES.items():
        group = analysis["per_source"][name]
        require(group["split"] == "test" and group["common_intents"] == count, "source denominator " + name)
        common = analysis["common_eligibility"]["streams"][name]["common"]
        plotted[name] = {"paired_intents": count, "source_models": 1, "histories": {}}
        for history in HISTORIES:
            curves = group["curves"][history]
            require(len(curves) == count and [row["intent"] for row in curves] == common, "paired source intents")
            for row in curves:
                require(row["stream"] == name and row["split"] == "test", "TEST row identity")
                for budget in BUDGETS:
                    value = row["curve"][str(budget)]
                    base = row["curve"]["0"]
                    passed = (
                        value["accuracy"] >= qualification["accuracy_min"]
                        and value["accuracy"] - base["accuracy"] >= qualification["gain_min"] - 1e-12
                        and max(0.0, base["guard"] - value["guard"]) <= qualification["guard_drop_max"] + 1e-12
                    )
                    require(passed == value["qualified"], "qualification from per-intent analysis scalars")
            values = {}
            for budget in BUDGETS:
                entries = [row["curve"][str(budget)] for row in curves]
                reported = group["means"][history][str(budget)]
                independently_audited = audit["paired_analysis"]["per_source"][name]["means"][history][str(budget)]
                averages = {metric: fmean(row[metric] for row in entries) for metric in ("accuracy", "guard")}
                qualified = sum(row["qualified"] for row in entries)
                for metric, value in averages.items():
                    require(math.isfinite(value) and 0 <= value <= 1, "bounded accuracy")
                    require(math.isclose(value, reported[metric], abs_tol=1e-12), "source mean " + metric)
                    require(math.isclose(value, independently_audited[metric], abs_tol=1e-12), "independent audit mean " + metric)
                require(math.isclose(qualified / count, reported["qualified"], abs_tol=1e-12), "integer qualification count")
                require(math.isclose(qualified / count, independently_audited["qualified"], abs_tol=1e-12), "independent audit qualification count")
                values[str(budget)] = {
                    "original_code_accuracy": averages["accuracy"],
                    "new_skill_guard_accuracy": averages["guard"],
                    "joint_qualified": qualified, "paired_intents": count,
                }
            require(values["8"]["joint_qualified"] == count, "eight-update caption")
            plotted[name]["histories"][history] = {"intent_ids": [row["id"] for row in curves], "budgets": values}
    bindings = {
        "analysis": str(source), "analysis_sha256": ANALYSIS_SHA256,
        "collection_sha256": sha(run / "collection.json"),
        "collected_files_checked": len(collection["files"]),
        "study_receipt_file_sha256": sha(run / "study/receipt.json"),
        "execution_file_sha256": sha(run / "execution.json"),
        "experiment_code_closure_sha256_from_execution": execution["source_sha256"],
        "analysis_attempt_id": execution["attempt_id"],
        "protocol_file": str(protocol_path), "protocol_file_sha256": sha(protocol_path),
        "protocol_digest": digest(protocol), "analysis_finished_at": execution["finished_at"],
        "independent_audit_file": str(audit_path), "independent_audit_sha256": AUDIT_SHA256,
        "independent_auditor_file_sha256": audit["auditor"]["sha256"],
        "independent_audit_status": audit["status"],
        "independent_audit_finished_at": audit["finished_at"],
        "independent_audit_self_tests": audit["self_tests"],
        "independent_audit_native_records": audit["verification"]["raw_native_records_checked"],
        "independent_audit_record_count_scope": audit["verification"]["record_count_scope"],
    }
    return plotted, qualification, bindings


def draw(plotted):
    figure = Figure(figsize=(15.6, 8.8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    figure.subplots_adjust(left=0.14, right=0.985, bottom=0.26, top=0.755, hspace=0.42, wspace=0.28)
    for row, (source, count) in enumerate(SOURCES.items()):
        for column in range(3):
            axis = axes[row, column]
            axis.set_xticks(range(4), BUDGETS)
            axis.set_xlim(-0.48, 3.48)
            axis.set_xlabel("Repair updates", labelpad=7)
            axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.tick_params(length=0, pad=6, colors="#42515E")
            axis.spines["left"].set_color("#CFD6DC")
            axis.spines["bottom"].set_color("#CFD6DC")
        axes[row, 0].set_ylim(50, 105)
        axes[row, 0].set_yticks([50, 60, 70, 80, 90, 100])
        axes[row, 1].set_ylim(80, 104)
        axes[row, 1].set_yticks([80, 85, 90, 95, 100])
        axes[row, 2].set_ylim(0, 115)
        axes[row, 2].set_yticks([0, 25, 50, 75, 100])
        axes[row, 2].set_ylabel("Qualified (%)", labelpad=8)
        for index, (history, style) in enumerate(HISTORIES.items()):
            values = plotted[source]["histories"][history]["budgets"]
            positions = [x + (index - 1) * 0.065 for x in range(4)]
            for column, metric in enumerate(("original_code_accuracy", "new_skill_guard_accuracy")):
                scores = [100 * values[str(budget)][metric] for budget in BUDGETS]
                axes[row, column].plot(
                    positions, scores, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=2.0, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                )
            counts = [values[str(budget)]["joint_qualified"] for budget in BUDGETS]
            bars = axes[row, 2].bar(
                [x + (index - 1) * 0.25 for x in range(4)],
                [100 * n / count for n in counts], width=0.22,
                color=style["color"], zorder=3,
            )
            for bar, n in zip(bars, counts, strict=True):
                axes[row, 2].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                    f"{n}/{count}", ha="center", va="bottom", fontsize=9, color=style["color"],
                )
        center = (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2
        figure.text(0.022, center + 0.022, source.removeprefix("fresh-").upper(), fontsize=15, weight="bold", color="#263642")
        figure.text(0.022, center - 0.025, f"{count} paired intents\n1 source model", fontsize=10, color="#51616E", linespacing=1.4)
    titles = (
        "Original-code accuracy\nMean %, 50–100% view",
        "New-skill guard accuracy\nMean %, 80–100% view",
        "Joint qualification\nQualified / paired intents",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, loc="left", fontsize=12, pad=13, fontweight="bold", color="#263642")
    figure.text(0.022, 0.965, "Early accuracy gains do not guarantee a joint qualification benefit", fontsize=20, weight="bold", color="#20323F")
    figure.text(0.022, 0.925, "Held-out TEST sources shown separately · Consistent 16-code intent classification throughout", fontsize=11.5, color="#51616E")
    handles = [Line2D([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2) for style in HISTORIES.values()]
    figure.legend(handles, [style["label"] for style in HISTORIES.values()], loc="upper left", bbox_to_anchor=(0.135, 0.89), ncol=3, frameon=False, fontsize=10.5, handlelength=2.4, columnspacing=2.5)
    figure.text(0.022, 0.164, "At 8 updates, every history qualifies on all 13 intents: 7/7 in TEST-a and 6/6 in TEST-b.", fontsize=12, weight="bold", color="#263642")
    figure.text(0.022, 0.123, "Joint gate per intent: original-code accuracy ≥80%, gain ≥30 percentage points, and new-skill guard loss ≤10 points from its own baseline.", fontsize=10, color="#42515E")
    figure.text(0.022, 0.091, "Protocol: same repair inputs/order; fresh AdamW from each history’s post-stream checkpoint per budget; 4 original + 4 new examples per update.", fontsize=9.7, color="#42515E")
    figure.text(0.022, 0.059, "Two source-model clusters; the 13 intents are not independent replicates. Descriptive comparisons; no significance tests or confidence intervals.", fontsize=9.7, color="#42515E")
    figure.text(0.022, 0.025, "Independent raw audit complete · 86,208 records (including repeated panels) · analysis " + ANALYSIS_SHA256[:12] + "… · audit " + AUDIT_SHA256[:12] + "…", fontsize=9.5, color="#697783")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    for text in figure.findobj(match=Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            require(bounds.x0 >= -1 and bounds.y0 >= -1 and bounds.x1 <= width + 1 and bounds.y1 <= height + 1, "text outside canvas: " + text.get_text())
    return figure


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--wave", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    wave = args.wave.resolve()
    plotted, qualification, bindings = source_data(wave)
    style = {
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": ANALYSIS_SHA256,
    }
    outputs = {}
    created = datetime.fromisoformat(bindings["analysis_finished_at"])
    description = "Collected history-controls analysis " + ANALYSIS_SHA256 + "; completed independent raw audit " + AUDIT_SHA256 + "; 86208 scored records including repeated panels."
    with matplotlib.rc_context(style):
        figure = draw(plotted)
        for extension in ("png", "pdf", "svg"):
            path = wave / f"history-controls.{extension}"
            if extension == "pdf":
                metadata = {"Title": "History controls", "Subject": description, "CreationDate": created, "ModDate": created}
            elif extension == "svg":
                metadata = {"Title": "History controls", "Description": description, "Date": created.isoformat()}
            else:
                metadata = {"Title": "History controls", "Description": description}
            figure.savefig(path, dpi=180, facecolor="white", metadata=metadata)
            outputs[extension] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    metadata = {
        "status": "rendered_from_independently_audited_analysis",
        "source_bindings": bindings, "script_sha256": sha(Path(__file__)),
        "matplotlib_version": matplotlib.__version__, "backend": "Agg",
        "budgets": list(BUDGETS), "histories": HISTORIES, "plotted": plotted,
        "qualification": qualification, "source_clusters": 2, "paired_intents": 13,
        "scoring": "Consistent 16-code classification, not full-vocabulary generated-answer capability.",
        "aggregation": "Mean per-intent accuracy within each source model; qualification uses integer per-intent pass counts with source denominators 7 and 6.",
        "uncertainty": "No significance tests, confidence intervals, or assumption of 13 independent source models.",
        "axis_ranges_percent": {"original_code_accuracy": [50, 105], "new_skill_guard_accuracy": [80, 104], "joint_qualification": [0, 115]},
        "positioning": "Budgets are categorical. Line markers have small visual-only horizontal offsets (-0.065,0,0.065) so equal values remain visible; budgets are identical across histories.",
        "lifetime_adapter_updates_before_repair": {"original_history": 256, "new_only": 128, "alternate_binding": 256},
        "interpretation": "Original history has the highest original-code mean accuracy at budget2 in each source. Joint qualification is lower than either control in TEST-a and higher in TEST-b. All histories qualify all paired intents at budget8. A pooled old-accuracy advantage does not guarantee a source-wise joint benefit.",
        "raw_independent_final_audit_complete": True,
        "verification_boundary": "Plot checks collection hashes, completed analysis receipt, protocol binding, paired IDs, means and qualification against the hash-bound completed independent audit. The separate audit checks 16-code raw logits, targets, input UIDs and threshold arithmetic, including repeated panels; this figure task does not repeat model inference or rerun that audit.",
        "scientific_inputs_modified": [], "gpu_or_provider_actions": [],
        "text_canvas_bounds_checked": True, "outputs": outputs,
    }
    receipt = wave / "history-controls.figure.json"
    receipt.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": outputs, "figure_metadata": str(receipt), "source_sha256": ANALYSIS_SHA256}, indent=2))


if __name__ == "__main__":
    main()
