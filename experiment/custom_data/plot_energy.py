#!/usr/bin/env python3
"""Charts for scalability-energy-crc-TIME.csv.

TIME is dropped and is not a factor. Repeated readings of one
configuration are collapsed to a single mean before anything is ranked
or drawn, so the timestamp is neither an axis nor part of the error bar.
A decision tree then scores the
experimental factors for each power sensor. Factors that explain at least
10% are drawn: an ordered sweep on x, the main comparison as lines, and a
third factor as subplots when it clears the same bar. The rest is the
standard deviation around each point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "scalability-energy-crc-TIME.csv"
OUT_DIR = ROOT / "energy-graphs"

FACTORS = ["build", "UNCORE_PSTATE", "NTHREADS", "CPUFREQ", "RATE"]
BUILDS = ["CPU", "ROI", "Zero-Copy"]
# A factor below this share of the tree is left inside the error bar.
SHOW_THRESHOLD = 0.10
AXIS_LABELS = {
    "build": "",
    "UNCORE_PSTATE": "Uncore P-state",
    "NTHREADS": "Number of threads",
    "CPUFREQ": "CPU frequency (MHz)",
    "RATE": "Rate",
}
SERIES_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "D", "^", "v"]

METRICS = [
    ("y_GPU-POWER", "GPU-POWER", "GPU power (W)"),
    ("y_IPMI-POWER", "IPMI-POWER", "IPMI power (W)"),
    ("y_RAPL-POWER", "RAPL-POWER", "RAPL power (W)"),
    ("y_PDU-POWER", "PDU-POWER", "PDU power (W)"),
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


def _ranking(frame: pd.DataFrame, column: str) -> list[tuple[str, float]]:
    encoded = pd.DataFrame({factor: frame[factor].astype("category").cat.codes for factor in FACTORS})
    model = DecisionTreeRegressor(random_state=0).fit(encoded, frame[column])
    order = sorted(zip(FACTORS, model.feature_importances_), key=lambda item: item[1], reverse=True)
    return [(name, float(score)) for name, score in order]


def _layout(frame: pd.DataFrame, ranking: list[tuple[str, float]]) -> tuple[str, str, str | None]:
    """Return (x variable, line variable, subplot variable or None)."""
    scores = dict(ranking)
    shown = [name for name, score in ranking if score >= SHOW_THRESHOLD][:3]
    if len(shown) < 2:
        shown = [name for name, _score in ranking[:2]]
    if "build" in shown:
        hue = "build"
        rest = [name for name in shown if name != "build"]
    else:
        hue = shown[0]
        rest = [name for name in shown if name != hue]
    rest.sort(key=lambda name: scores[name], reverse=True)
    if len(rest) == 1:
        return rest[0], hue, None
    first, second = rest[0], rest[1]
    if frame[second].nunique() > frame[first].nunique():
        return second, hue, first
    return first, hue, second


def _display(variable: str, value) -> str:
    if variable == "CPUFREQ":
        return str(int(value))
    if variable == "NTHREADS":
        count = int(value)
        return "1 thread" if count == 1 else f"{count} threads"
    if variable == "UNCORE_PSTATE":
        return str(int(value))
    if variable == "build":
        return str(value)
    return str(int(value))


def _order(variable: str, values) -> list[str]:
    if variable == "build":
        return [build for build in BUILDS if build in set(map(str, values))]
    return [_display(variable, value) for value in sorted(set(values))]


def _limits(frame: pd.DataFrame, column: str, keys: list[str]) -> tuple[float, float]:
    grouped = frame.groupby(keys, observed=True)[column]
    deviation = grouped.std(ddof=1).fillna(0)
    low = float((grouped.mean() - deviation).min())
    high = float((grouped.mean() + deviation).max())
    span = high - low
    pad = span * 0.08 if span else max(abs(high) * 0.08, 1e-9)
    bottom = low - pad
    data_min = float(frame[column].min())
    if data_min > 0 and bottom <= 0:
        bottom = data_min * 0.92
    elif bottom < 0:
        bottom = 0.0
    return bottom, high + pad


def _decorate(ax: plt.Axes, x_order: list[str], ylabel: str, x_label: str, bottom: float, top: float, ticks: list[float]) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order)
    ax.set_xlim(-0.5, len(x_order) - 0.5)
    ax.set_ylim(bottom, top)
    if ticks:
        ax.set_yticks(ticks)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    sns.despine(ax=ax, top=True, right=True)


def plot_metric(cells: pd.DataFrame, column: str, filename: str, ylabel: str) -> Path:
    frame = cells.dropna(subset=[column]).copy()
    ranking = _ranking(frame, column)
    x_var, hue_var, facet_var = _layout(frame, ranking)
    print(
        filename,
        "x =",
        x_var,
        "lines =",
        hue_var,
        "panels =",
        facet_var,
        "importance =",
        ", ".join(f"{name} {score:.2f}" for name, score in ranking),
    )

    x_order = _order(x_var, frame[x_var])
    hue_order = _order(hue_var, frame[hue_var])
    frame[AXIS_LABELS[x_var]] = pd.Categorical(
        frame[x_var].map(lambda value: _display(x_var, value)), x_order, ordered=True
    )
    frame[AXIS_LABELS[hue_var]] = pd.Categorical(
        frame[hue_var].map(lambda value: _display(hue_var, value)), hue_order, ordered=True
    )
    x_name = AXIS_LABELS[x_var]
    hue_name = AXIS_LABELS[hue_var]
    palette = dict(zip(hue_order, SERIES_COLORS))
    markers = dict(zip(hue_order, SERIES_MARKERS))
    group_keys = [x_name, hue_name]
    if facet_var is not None:
        panel_order = _order(facet_var, frame[facet_var])
        frame["Panel"] = pd.Categorical(
            frame[facet_var].map(lambda value: _display(facet_var, value)), panel_order, ordered=True
        )
        group_keys.append("Panel")

    bottom, top = _limits(frame, column, group_keys)
    ticks = MaxNLocator(nbins=5, min_n_ticks=4).tick_values(bottom, top)
    ticks = [tick for tick in ticks if bottom < tick < top and abs(tick) > 1e-8 * max(1.0, abs(top))]

    common = dict(
        data=frame,
        x=x_name,
        y=column,
        hue=hue_name,
        hue_order=hue_order,
        style=hue_name,
        style_order=hue_order,
        palette=palette,
        markers=markers,
        dashes=False,
        errorbar="sd",
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 0.8},
        legend=False,
    )
    if facet_var is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        sns.lineplot(**common, ax=ax)
        _decorate(ax, x_order, ylabel, x_name, bottom, top, ticks)
        figure = fig
    else:
        grid = sns.relplot(
            **common,
            col="Panel",
            col_order=_order(facet_var, frame[facet_var]),
            col_wrap=2,
            kind="line",
            height=3.3,
            aspect=1.15,
            facet_kws={"sharex": True, "sharey": True},
        )
        title = "Uncore P-state {col_name}" if facet_var == "UNCORE_PSTATE" else "{col_name}"
        grid.set_titles(title)
        for ax in grid.axes.flat:
            if not ax.axison:
                continue
            _decorate(ax, x_order, ylabel, x_name, bottom, top, ticks)
            ax.xaxis.grid(True, linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
        figure = grid.figure
        figure.subplots_adjust(top=0.86, hspace=0.38, wspace=0.14)

    handles = [
        plt.Line2D([0], [0], color=palette[level], marker=markers[level], linestyle="-", label=level)
        for level in hue_order
    ]
    figure.legend(
        handles=handles,
        title=hue_name,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
    )
    if facet_var is None:
        figure.subplots_adjust(top=0.82)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"scalability-energy-{filename}.pdf"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    _style()
    df = pd.read_csv(DATA_PATH).drop(columns=["TIME"])
    metric_columns = [column for column, _, _ in METRICS]
    # One row per experimental configuration. TIME is already gone, so
    # extra samples at the same settings do not add spread or extra weight.
    cells = df.groupby(FACTORS, as_index=False)[metric_columns].mean()
    for column, filename, ylabel in METRICS:
        path = plot_metric(cells, column, filename, ylabel)
        print(path)


if __name__ == "__main__":
    main()
