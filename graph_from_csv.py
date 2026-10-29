#!/usr/bin/env python3
"""Generate NPF graphs from result CSV files.

NPF's ``--csv`` / ``--single-output`` export stores one row per run.
``build`` is the series, ``test_index`` is one parameter combination, and
``run_index`` is the repeat of that combination. ``index`` is the row id.
Those columns are not plotted.

Result metrics are columns whose names start with ``y_``. When that prefix
is missing, a column that stays constant inside each ``test_index`` is a
factor and a column that changes across ``run_index`` is a metric.

Examples:
  python graph_from_csv.py experiment/custom_data/*.csv
  python graph_from_csv.py experiment/custom_data/iplookup-cpu.csv \\
      --var-columns SIZE NTHREADS \\
      --metric-columns RX-RATE-PPS RX-RATE-MBPS LOSE-RATE AVG-LAT RX-GOODPUT-GBPS-PKTGEN
  python graph_from_csv.py data.csv --output-dir graphs --config graph_max_variables=4
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import OrderedDict

import numpy as np
import pandas as pd

import npf.cmdline
from npf.cmdline import ExtendAction
from npf.globals import set_args
from npf.models.dataset import Run
from npf.output import generate_outputs
from npf.parsing import parse_variables
from npf.tests.build import Build
from npf.tests.sections import SectionConfig, SectionVariable
from npf.models.variables.SimpleVariable import SimpleVariable

# Columns written by to_pandas() that are not experiment factors.
_METADATA = {"index", "build", "test_index", "run_index"}


class _CsvRepo:
    """Stand-in repository so Grapher can name a series without a .repo file."""

    def __init__(self, name: str):
        self.name = name
        self.reponame = name

    def get_identifier(self) -> str:
        return "local"

    def pretty_name(self) -> str:
        return self.name


class _CsvTest:
    """Minimal test object. Grapher only needs config plus a readable script path."""

    def __init__(self, path: str, filename: str, config: SectionConfig):
        self.path = path
        self.filename = filename
        self.config = config
        self.tags = []


def _is_metadata(name: str) -> bool:
    return name in _METADATA or str(name).startswith("Unnamed")


def _as_variable(value):
    """Convert a CSV cell to a plain Python value NPF can hash and plot."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return number
    return value


def _resolve_columns(requested, available):
    resolved = []
    for name in requested:
        if name in available:
            resolved.append(name)
        elif f"y_{name}" in available:
            resolved.append(f"y_{name}")
        elif str(name).startswith("y_") and name[2:] in available:
            resolved.append(name[2:])
        else:
            known = ", ".join(available)
            raise SystemExit(f"Column '{name}' was not found. Columns: {known}")
    return resolved


def classify_columns(df: pd.DataFrame, var_columns=None, metric_columns=None):
    """Return (variable column names, [(metric name, source column), ...])."""
    columns = [c for c in df.columns if not _is_metadata(str(c))]
    y_columns = [c for c in columns if str(c).startswith("y_")]

    if var_columns or metric_columns:
        variables = _resolve_columns(var_columns, columns) if var_columns else None
        metrics_src = _resolve_columns(metric_columns, columns) if metric_columns else None
        if metrics_src is None:
            metrics_src = [c for c in columns if c not in variables]
        if variables is None:
            variables = [c for c in columns if c not in metrics_src]
    elif y_columns:
        metrics_src = y_columns
        variables = [c for c in columns if c not in y_columns]
    elif "test_index" in df.columns:
        # Factors are fixed for a combination; metrics change from run to run.
        group_keys = ["build", "test_index"] if "build" in df.columns else ["test_index"]
        variables = []
        metrics_src = []
        for column in columns:
            if int(df.groupby(group_keys)[column].nunique(dropna=False).max()) <= 1:
                variables.append(column)
            else:
                metrics_src.append(column)
    else:
        # No y_ prefix and no test_index: repeated factor columns vs results.
        nrows = max(len(df), 1)
        limit = max(12, int(0.3 * nrows))
        variables = []
        metrics_src = []
        for column in columns:
            nunique = int(df[column].nunique(dropna=False))
            if nunique < nrows and nunique <= limit:
                variables.append(column)
            else:
                metrics_src.append(column)

    if not metrics_src:
        raise SystemExit(
            "No result columns found. Pass --metric-columns, or use y_ names "
            "for metrics as in an NPF --csv export."
        )

    overlap = set(variables) & set(metrics_src)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise SystemExit(f"Columns cannot be both variables and metrics: {names}")

    metrics = []
    for source in metrics_src:
        name = source[2:] if str(source).startswith("y_") else source
        metrics.append((name, source))
    return variables, metrics


def _metric_values(group, source):
    """Metric samples for one combination, in run_index order."""
    if "run_index" in group.columns:
        group = group.sort_values("run_index", kind="mergesort")
    values = []
    for value in group[source].tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        values.append(float(value))
    return values


def _combinations(part, variables):
    """Yield one dataframe per parameter combination.

    ``test_index`` is that combination. ``run_index`` only orders the repeats
    inside it. Without ``test_index``, rows that share the same factor values
    are the repeats.
    """
    if "test_index" in part.columns:
        ordered = part.sort_values("test_index", kind="mergesort")
        for _, group in ordered.groupby("test_index", sort=True):
            yield group
        return
    if variables:
        for _, group in part.groupby(variables, sort=False, dropna=False):
            yield group
        return
    yield part


def csv_to_series(csv_path: str, var_columns=None, metric_columns=None, config_overrides=None):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} has no rows")

    if "TIME" in df.columns:
        df.drop(columns=["TIME"], inplace=True)

    variables, metrics = classify_columns(df, var_columns, metric_columns)
    print(f"{os.path.basename(csv_path)}: variables={variables or ['(none)']}")
    print(f"  metrics={[name for name, _ in metrics]}")

    frame = df.copy()
    for column in variables + [source for _, source in metrics]:
        frame[column] = frame[column].map(_as_variable)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    if "build" in df.columns:
        groups = [
            (str(build_name), part)
            for build_name, part in frame.groupby(df["build"].map(lambda value: str(value)), sort=False)
        ]
    else:
        groups = [(stem, frame)]

    stub = tempfile.NamedTemporaryFile(
        prefix="npf-csv-", suffix=".npf", mode="w", delete=False, encoding="utf-8"
    )
    try:
        stub.write(f"%info\nGraphs generated from {os.path.basename(csv_path)}\n")
        stub.close()

        config = SectionConfig()
        if config_overrides:
            config.override_all(parse_variables(config_overrides, [], config))

        variables_section = SectionVariable()
        for column in variables:
            sample = next((value for value in frame[column].tolist() if value is not None), 0)
            variables_section.vlist[column] = SimpleVariable(column, sample)

        test = _CsvTest(os.path.dirname(stub.name), os.path.basename(stub.name), config)
        test.variables = variables_section
        series = []
        for build_name, part in groups:
            dataset = OrderedDict()
            for group in _combinations(part, variables):
                run_variables = OrderedDict(
                    (column, _as_variable(group[column].iloc[0])) for column in variables
                )
                run = Run(run_variables)
                results = OrderedDict()
                for metric_name, source in metrics:
                    values = _metric_values(group, source)
                    if values:
                        results[metric_name] = values
                if not results:
                    continue
                if run in dataset:
                    for metric_name, values in results.items():
                        dataset[run].setdefault(metric_name, []).extend(values)
                else:
                    dataset[run] = results

            n_runs = [len(next(iter(results.values()))) for results in dataset.values() if results]
            if "run_index" in part.columns:
                observed = sorted(int(value) for value in part["run_index"].dropna().unique())
                run_note = f", run_index {observed[0]}–{observed[-1]}" if observed else ""
            else:
                run_note = ""
            if n_runs:
                run_note += f", {min(n_runs)}–{max(n_runs)} samples each"
            build = Build(_CsvRepo(str(build_name)), "local")
            build._pretty_name = str(build_name)
            series.append((test, build, dataset))
            print(f"  series '{build_name}': {len(dataset)} combinations{run_note}")

        return series, stub.name
    except Exception:
        os.unlink(stub.name)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NPF graphs from one or more result CSV files."
    )
    parser.add_argument("csvs", nargs="+", help="NPF result CSV files")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for PDF graphs (default: a graphs/ folder next to each CSV)",
    )
    parser.add_argument(
        "--var-columns",
        nargs="+",
        default=None,
        help="Columns to treat as experiment variables. Default: non-y_ columns, "
        "or low-cardinality columns when the file has no y_ prefix.",
    )
    parser.add_argument(
        "--metric-columns",
        nargs="+",
        default=None,
        help="Columns to plot as results. y_ prefixes are optional.",
    )
    parser.add_argument("--graph-title", type=str, default=None, help="Graph title")
    parser.add_argument(
        "--config",
        metavar="config=value",
        type=str,
        nargs="+",
        action=ExtendAction,
        default=[],
        help="NPF %%config overrides, for example graph_variables_as_series=NTHREADS",
    )
    npf.cmdline.add_graph_options(parser)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    set_args(args)

    for csv_path in args.csvs:
        if not os.path.isfile(csv_path):
            raise SystemExit(f"CSV not found: {csv_path}")

        out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(csv_path)), "graphs")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        # NPF appends the result name and .pdf itself.
        args.graph_filename = os.path.join(out_dir, stem)

        series, stub = csv_to_series(
            csv_path,
            var_columns=args.var_columns,
            metric_columns=args.metric_columns,
            config_overrides=args.config,
        )
        try:
            generate_outputs(args.graph_filename, series, [], options=args)
        finally:
            if os.path.exists(stub):
                os.unlink(stub)


if __name__ == "__main__":
    main()
