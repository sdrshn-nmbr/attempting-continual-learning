import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt


def interval(record, transport):
    key = "transport_gain_intervals" if transport else "b_gain_intervals"
    gain = record["transport_b_gain" if transport else "b_gain"]
    if record[key] is None:
        return None
    estimate = min(record[key].values(), key=lambda item: item["mean_group_delta"])
    if estimate["mean_group_delta"] != gain:
        raise ValueError("STRONGER_FLOOR_INTERVAL_MISMATCH")
    low, high = estimate["interval_95"]
    return 100 * gain, 100 * (gain - low), 100 * (high - gain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text()) for path in (args.screen, args.confirmation)
    ]
    if [report["arms"]["core"]["fixture_seed"] for report in reports] != [17, 29]:
        raise ValueError("EXPECTED_SCREEN_17_AND_CONFIRMATION_29")
    matplotlib.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.5), sharey=True)
    colors = {"core": "#167D9A", "latent": "#A66E13", "lora": "#735DA5"}
    labels = {
        "core": "Core + new vector",
        "latent": "New vector only",
        "lora": "Task LoRA",
    }
    positions = [5, 4, 3, 1, 0, -1]
    titles = ["Learn after the 4B → 8B upgrade", "Move the learned skill to 1.7B"]
    for column, axis in enumerate(axes):
        axis.set_title(titles[column], loc="left", fontsize=13, pad=24, weight="bold")
        axis.axvline(0, color="#52616D", linewidth=1)
        axis.axvline(5 if column else 10, color="#9AA5AE", linestyle="--", linewidth=1)
        axis.axhline(2, color="#E4E8EB", linewidth=1)
        axis.set_xlim(-35, 100)
        axis.set_ylim(-1.75, 6.15)
        axis.set_xticks([-25, 0, 25, 50, 75, 100])
        axis.set_xlabel(
            "Accuracy gain over the stronger starting baseline (points)", labelpad=12
        )
        axis.grid(axis="x", color="#E8ECEF", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#D4DCE1")
        axis.tick_params(axis="y", length=0)
        for index, report in enumerate(reports):
            for offset, arm in enumerate(labels):
                y = positions[index * 3 + offset]
                point = interval(report["arms"][arm], bool(column))
                if point is None:
                    axis.text(
                        1, y, "Not measured", color="#7B8791", va="center", fontsize=9
                    )
                    continue
                gain, lower, upper = point
                axis.errorbar(
                    gain,
                    y,
                    xerr=[[lower], [upper]],
                    fmt="o" if index == 0 else "s",
                    color=colors[arm],
                    capsize=4,
                    markersize=7,
                    elinewidth=2,
                )
                axis.text(
                    gain,
                    y + 0.24,
                    f"{gain:+.1f}",
                    ha="center",
                    color=colors[arm],
                    fontsize=10,
                )
        axis.text(
            -34, 5.72, "FIXTURE 17 · SCREEN", color="#56636E", fontsize=9, weight="bold"
        )
        axis.text(
            -34,
            1.72,
            "FIXTURE 29 · CONFIRMATION",
            color="#56636E",
            fontsize=9,
            weight="bold",
        )
    axes[0].set_yticks(positions, [labels[arm] for _ in reports for arm in labels])
    figure.suptitle(
        "Amber: learning after an upgrade and transferring that learning",
        x=0.02,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.02,
        0.085,
        "Points: gains over max(raw base, published initialization). Whiskers: paired 95% bootstrap intervals over 64 test triples.\n"
        "Dashed lines: prespecified improvement thresholds (10 points for learning; 5 for transfer). Intervals do not measure variation across training seeds.",
        fontsize=9,
        color="#52616D",
        linespacing=1.6,
    )
    figure.text(
        0.02,
        0.025,
        "All arms: 96 updates, 384 new-task examples. The core arm adds 96 old-task examples; trainable capacities differ.\n"
        "Core mean old-task forgetting: 3.13 points on fixture 17; 1.04 on fixture 29. Four-choice scoring, supplied task identity, one skill family.",
        fontsize=9,
        color="#52616D",
        linespacing=1.6,
    )
    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.23, top=0.82, wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        figure.savefig(args.output.with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps({"figure": str(args.output.with_suffix(".png").resolve())}))


if __name__ == "__main__":
    main()
