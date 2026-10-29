#!/usr/bin/env python3
"""Line charts for iplookup-cpu.csv.

One build and two factors. NPF draws one line chart per metric, with the
thread count on x and one line per packet size. These figures keep that
layout, space the thread counts evenly, and use readable axis labels.
The bar at each point is the standard deviation of the three runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "iplookup-cpu.csv"
OUT_DIR = ROOT / "iplookup-graphs"

SIZES = [64, 128, 256, 512, 1024]
THREADS = [1, 2, 4, 8, 16]
SIZE_LABELS = [str(size) for size in SIZES]
THREAD_LABELS = [str(thread) for thread in THREADS]

# Okabe–Ito, one color per packet size.
PALETTE = {
    "64": "#0072B2",
    "128": "#E69F00",
    "256": "#009E73",
    "512": "#D55E00",
    "1024": "#CC79A7",
}
MARKERS = {
    "64": "o",
    "128": "s",
    "256": "D",
    "512": "^",
    "1024": "v",
}

# (column, filename stem, display transform, axis label)
# Latency in the CSV is seconds. Rates in Mb/s are shown in Gb/s.
METRICS = [
    ("RX-RATE-PPS", "RX-RATE-PPS", lambda series: series / 1e6, "Receive rate (million packets/s)"),
    ("RX-RATE-MBPS", "RX-RATE-MBPS", lambda series: series / 1e3, "Bit rate (Gb/s)"),
    ("LOSE-RATE", "LOSE-RATE", None, "Loss rate"),
    ("AVG-LAT", "AVG-LAT", lambda series: series * 1e6, "Average latency (µs)"),
    ("RX-GOODPUT-GBPS-PKTGEN", "RX-GOODPUT-GBPS-PKTGEN", None, "Goodput (Gbps)"),
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


def plot_metric(df: pd.DataFrame, column: str, filename: str, transform, ylabel: str) -> Path:
    plot_df = df.dropna(subset=[column]).copy()
    plot_df["Packet size"] = pd.Categorical(
        plot_df["SIZE"].map(lambda value: str(int(value))),
        SIZE_LABELS,
        ordered=True,
    )
    plot_df["Threads"] = pd.Categorical(
        plot_df["NTHREADS"].map(lambda value: str(int(value))),
        THREAD_LABELS,
        ordered=True,
    )
    plot_df["value"] = plot_df[column] if transform is None else transform(plot_df[column])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.lineplot(
        data=plot_df,
        x="Threads",
        y="value",
        hue="Packet size",
        hue_order=SIZE_LABELS,
        style="Packet size",
        style_order=SIZE_LABELS,
        palette=PALETTE,
        markers=MARKERS,
        dashes=False,
        errorbar="sd",
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 0.8},
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Number of threads")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(THREAD_LABELS)))
    ax.set_xticklabels(THREAD_LABELS)
    ax.set_xlim(-0.5, len(THREAD_LABELS) - 0.5)
    grouped = plot_df.groupby(["Threads", "Packet size"], observed=True)["value"]
    deviation = grouped.std(ddof=1).fillna(0)
    high = max(float(plot_df["value"].max()), float((grouped.mean() + deviation).max()))
    pad = high * 0.08 if high else 1e-9
    top = high + pad
    ax.set_ylim(0, top)
    ticks = MaxNLocator(nbins=6, min_n_ticks=4).tick_values(0, top)
    ticks = [tick for tick in ticks if 0 <= tick < top]
    if not ticks or min(ticks) > 1e-8 * max(1.0, abs(top)):
        ticks = [0.0, *ticks]
    ax.set_yticks(ticks)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    sns.despine(ax=ax, top=True, right=True, left=True)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=PALETTE[size],
            marker=MARKERS[size],
            linestyle="-",
            label=size,
        )
        for size in SIZE_LABELS
    ]
    fig.legend(
        handles=handles,
        title="Packet size (bytes)",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
    )
    fig.subplots_adjust(top=0.82)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"iplookup-cpu-{filename}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    _style()
    df = pd.read_csv(DATA_PATH)
    for column, filename, transform, ylabel in METRICS:
        path = plot_metric(df, column, filename, transform, ylabel)
        print(path)


if __name__ == "__main__":
    main()
