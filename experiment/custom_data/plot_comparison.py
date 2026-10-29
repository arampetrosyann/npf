#!/usr/bin/env python3
"""Line charts for comparison-all-DPA.csv.

NPF plots the mean of run_index and, when one numeric factor has more than
two levels, draws a line. With several builds and two factors it instead
puts every SIZE × NTHREADS pair on one categorical axis. These figures keep the line chart: number of threads on x, one line per
build, and one panel per packet size, with two panels per row. The bar
at each point is the standard deviation of the three runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "comparison-all-DPA.csv"
OUT_DIR = ROOT / "comparison-graphs"

BUILDS = ["DPA CRC", "DPA IPLOOKUP 1K", "DPA IPLOOKUP 10K", "DPA ETHERMIRROR"]
SIZES = [64, 128, 256, 512, 1024]
THREADS = [16, 32, 64, 128, 190, 254]
SIZE_LABELS = [str(size) for size in SIZES]
# Okabe–Ito, one color per build.
PALETTE = {
    "DPA CRC": "#0072B2",
    "DPA IPLOOKUP 1K": "#E69F00",
    "DPA IPLOOKUP 10K": "#009E73",
    "DPA ETHERMIRROR": "#D55E00",
}
MARKERS = {
    "DPA CRC": "o",
    "DPA IPLOOKUP 1K": "s",
    "DPA IPLOOKUP 10K": "D",
    "DPA ETHERMIRROR": "^",
}

# AVG-LAT is 0 on every row, so a chart of it would be a flat line.
# (column, filename stem, axis label, scale)
METRICS = [
    ("y_RX-RATE-PPS", "RX-RATE-PPS", "Receive rate (packets/s)", "log"),
    ("y_RX-RATE-MBPS", "RX-RATE-MBPS", "Bit rate (Mbps)", "log"),
    ("y_LOSE-RATE", "LOSE-RATE", "Loss rate", "linear"),
    ("y_GOODRATEPC", "GOODRATEPC", "Good rate (%)", "linear"),
    ("y_RX-GOODPUT-GBPS-PKTGEN", "RX-GOODPUT-GBPS-PKTGEN", "Receive goodput (Gb/s)", "linear"),
]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_metric(df: pd.DataFrame, column: str, filename: str, ylabel: str, scale: str) -> Path:
    plot_df = df.dropna(subset=[column]).copy()
    plot_df["Packet size"] = plot_df["SIZE"].map(lambda value: str(int(value)))
    thread_labels = [str(thread) for thread in THREADS]
    plot_df["Threads"] = pd.Categorical(
        plot_df["NTHREADS"].map(lambda value: str(int(value))),
        thread_labels,
        ordered=True,
    )
    present_sizes = [size for size in SIZE_LABELS if size in set(plot_df["Packet size"])]
    present_builds = [build for build in BUILDS if build in set(plot_df["build"])]
    g = sns.relplot(
        data=plot_df,
        x="Threads",
        y=column,
        hue="build",
        hue_order=present_builds,
        style="build",
        style_order=present_builds,
        col="Packet size",
        col_order=present_sizes,
        col_wrap=2,
        kind="line",
        palette=PALETTE,
        markers=MARKERS,
        dashes=False,
        errorbar="sd",
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 0.8},
        height=3.4,
        aspect=1.15,
        facet_kws={"sharex": True, "sharey": True},
        legend=False,
    )
    if scale == "log":
        g.set(yscale="log")
    else:
        g.set(ylim=(0, None))

    g.set_axis_labels("Number of threads", ylabel)
    g.set_titles("Packet size {col_name} (bytes)")
    for ax in g.axes.flat:
        ax.set_xticks(range(len(thread_labels)))
        ax.set_xticklabels(thread_labels)
        ax.set_xlim(-0.5, len(thread_labels) - 0.5)
        if scale == "log":
            ymin, ymax = ax.get_ylim()
            ticks = [
                tick
                for tick in ax.yaxis.get_major_locator().tick_values(ymin, ymax)
                if ymin <= tick <= ymax and abs(tick - 1.0) > 1e-8
            ]
            ax.set_yticks(ticks)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=PALETTE[build],
            marker=MARKERS[build],
            linestyle="-",
            label=build,
        )
        for build in present_builds
    ]
    g.figure.legend(
        handles=handles,
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    g.figure.subplots_adjust(top=0.90, hspace=0.28, wspace=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"comparison-all-DPA-{filename}.pdf"
    g.savefig(path, bbox_inches="tight")
    plt.close(g.figure)
    return path


def main() -> None:
    _style()
    df = pd.read_csv(DATA_PATH)

    df.info()

    df["build"] = pd.Categorical(df["build"], BUILDS, ordered=True)
    for column, filename, ylabel, scale in METRICS:
        path = plot_metric(df, column, filename, ylabel, scale)
        print(path)


if __name__ == "__main__":
    main()
