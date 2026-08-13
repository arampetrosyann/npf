"""
Lux → Vega-Lite → static chart rendering for NPF.

Fully automatic: plot type, marks, scales, and layout are chosen by Lux.
User graph_%config / graph_type options are intentionally ignored here.

Flow:
  GraphData.series → to_pandas → one-metric DataFrame (real variable columns)
                    →  LuxDataFrame (intent = result metric; Lux picks related vars)
                    →  Vega-Lite JSON (to_vegalite)
                    →  PNG/PDF via vl-convert
"""

from __future__ import annotations

import json
import math
import os
import re
import types
import traceback
import warnings
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from npf.output.transform.pandas import to_pandas

def _ensure_lux_pandas_compat():
    """Lux 0.5.x probes pandas.io.gbq, which was removed in recent pandas."""
    if not hasattr(pd.io, "gbq"):
        gbq = types.ModuleType("pandas.io.gbq")

        class _Dummy:
            pass

        gbq.DataFrame = _Dummy
        import sys

        sys.modules["pandas.io.gbq"] = gbq
        pd.io.gbq = gbq

def _import_lux(topk: int = 1):
    _ensure_lux_pandas_compat()
    import lux
    from lux.vis.Vis import Vis
    from lux import Clause

    lux.config.plotting_backend = "vegalite"
    lux.config.topk = max(1, int(topk))
    lux.config.number_of_bars = 20
    lux.config.plotting_scale = 2
    lux.config.sort = "descending"
    return lux, Vis, Clause

def _import_vl_convert():
    import vl_convert as vlc

    return vlc

def series_result_to_dataframe(series, result_type: str) -> pd.DataFrame:
    """
    Build a DataFrame for one result metric with NPF variables as columns.
    """
    y_col = f"y_{result_type}"
    df = to_pandas(series, float_format="{:,.2f}".format)

    if df is None or df.empty or y_col not in df.columns:
        return pd.DataFrame()

    # Drop other metrics and indices that the system should not treat as dimensions
    drop_cols = [c for c in df.columns if c.startswith("y_") and c != y_col]
    drop_cols += [c for c in ("test_index", "run_index") if c in df.columns]
    
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    df.rename(columns={y_col: result_type}, inplace=True)
    
    return df

def collect_result_types(series) -> List[str]:
    """Discover metric names present in series"""
    types: Set[str] = set()
    for _test, _build, all_results in series:
        for _run, run_results in all_results.items():
            types.update(run_results.keys())
    return sorted(types)

def get_n_charts(n_vars: int) -> int:
    """
    Number of charts to generate.

    Each chart stays readable with ~3 variables.
    """
    return max(1, math.floor(n_vars / 3))

def resolve_n_charts(grapher, n_vars: int) -> int:
    topk = grapher.config("graph_topk")
    if topk is None or topk == "":
        return get_n_charts(n_vars)
    try:
        return max(1, int(topk))
    except (TypeError, ValueError):
        print(f"WARNING: Invalid graph_topk={topk!r}, falling back to auto")
        return get_n_charts(n_vars)

def resolve_graph_intent(grapher, columns) -> Optional[List[str]]:
    """
    Read ``graph_intent`` from %config as an explicit Lux intent column list.
    """
    raw = grapher.configlist("graph_intent", [])
    if not raw:
        return None

    cols = set(columns)
    intent: List[str] = []
    for item in raw:
        name = str(item).strip()
        if not name:
            continue
        if name not in cols:
            print(f"WARNING: graph_intent column {name!r} not in data, ignoring")
            continue
        if name not in intent:
            intent.append(name)
    return intent or None

def _inline_vegalite_data(spec: dict, df: pd.DataFrame) -> dict:
    """Ensure Vega-Lite carries inline values (vl-convert friendly)."""
    if "datasets" in spec and isinstance(spec.get("data"), dict) and "name" in spec["data"]:
        name = spec["data"]["name"]
        if name in spec["datasets"]:
            spec["data"] = {"values": spec["datasets"][name]}
            del spec["datasets"]
    elif not (isinstance(spec.get("data"), dict) and "values" in spec["data"]):
        plain = pd.DataFrame(df)
        spec["data"] = {"values": json.loads(plain.to_json(orient="records"))}
    spec.pop("params", None)
    spec.pop("vislib", None)
    return spec

def _field_display_name(grapher, field: str, result_type: Optional[str]) -> str:
    """Map a DataFrame column to its ``var_names`` label when available."""
    if not field:
        return field
    if result_type and field == result_type:
        return grapher.var_name("result", result_type=result_type)
    return grapher.var_name(field)

def _apply_var_name_to_encoding_channel(enc: dict, grapher, result_type: Optional[str]) -> None:
    if not isinstance(enc, dict):
        return
    field = enc.get("field")
    if not isinstance(field, str) or not field:
        return
    label = _field_display_name(grapher, field, result_type)
    enc["title"] = label
    if isinstance(enc.get("axis"), dict):
        enc["axis"]["title"] = label
    if isinstance(enc.get("legend"), dict):
        enc["legend"]["title"] = label

def _apply_var_names_to_vegalite(spec: dict, grapher, result_type: Optional[str] = None) -> dict:
    """
    Rewrite Vega-Lite axis/legend titles using NPF ``var_names``.
    """
    if grapher is None or not isinstance(spec, dict):
        return spec

    encoding = spec.get("encoding")
    if isinstance(encoding, dict):
        for enc in encoding.values():
            if isinstance(enc, list):
                for item in enc:
                    _apply_var_name_to_encoding_channel(item, grapher, result_type)
            else:
                _apply_var_name_to_encoding_channel(enc, grapher, result_type)

    for key in ("layer", "hconcat", "vconcat", "concat"):
        nested = spec.get(key)
        if isinstance(nested, list):
            for sub in nested:
                _apply_var_names_to_vegalite(sub, grapher, result_type)

    if isinstance(spec.get("spec"), dict):
        _apply_var_names_to_vegalite(spec["spec"], grapher, result_type)

    return spec

def _vis_to_vegalite(
    vis,
    df: pd.DataFrame,
    title: Optional[str] = None,
    grapher=None,
    result_type: Optional[str] = None,
) -> dict:
    spec = vis.to_vegalite(prettyOutput=False)
    if isinstance(spec, str):
        spec = json.loads(spec)
    if not isinstance(spec, dict):
        raise RuntimeError(f"Unexpected Vega-Lite type: {type(spec)}")
    spec = _inline_vegalite_data(spec, df)
    spec = _apply_var_names_to_vegalite(spec, grapher, result_type=result_type)
    if title:
        spec["title"] = title
    return spec

def _clear_lux_recs(ldf) -> None:
    if hasattr(ldf, "_recommendation"):
        ldf._recommendation = {}
    if hasattr(ldf, "_rec_info"):
        ldf._rec_info = []


def _collect_vis_from_recs(ldf, intent, n: int) -> list:
    """Set Lux intent, then collect up to ``n`` visualizations from recommendations."""
    ldf.intent = intent
    _clear_lux_recs(ldf)

    try:
        recs = ldf.recommendation or {}
    except Exception as e:
        print(f"WARNING: Lux recommendation failed for intent {intent}: {e}")
        return []

    picked = []
    for action in ("Current Vis", "Enhance"):
        vis_list = recs.get(action)
        if not vis_list:
            continue
        for vis in vis_list:
            picked.append(vis)
            if len(picked) >= n:
                return picked
    return picked[:n]

def _pick_auto_vis_list(ldf, result_type: str, Clause, n: int, user_intent=None) -> list:
    """Collect up to ``n`` charts for the result metric."""
    n = max(1, int(n))

    if user_intent:
        return _collect_vis_from_recs(ldf, list(user_intent), n)

    var_cols = [c for c in ldf.columns if c != result_type and c != "build"]
    max_wildcards = min(2, len(var_cols))
    picked = []

    for n_wildcards in range(max_wildcards, -1, -1):
        intent = [Clause(attribute=result_type, channel='y')] + [Clause("?")] * n_wildcards
        picked = _collect_vis_from_recs(ldf, intent, n)
        if picked:
            return picked

    return picked[:n]

TIMESTAMP_RE = re.compile(
    r"^("
    r"\d{4}-\d{2}-\d{2}"  # 2024-01-15
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"  # optional time / tz
    r"|"
    r"\d{4}/\d{2}/\d{2}"  # 2024/01/15
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?" # optional time
    r"|"
    r"\d{1,2}/\d{1,2}/\d{2,4}"  # 1/15/2024 or 15/01/24
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?" # optional time
    r"|"
    r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"  # 14:30 or 14:30:00
    r")$"
)

def _values_look_temporal(pd_series: pd.Series) -> bool:
    if pd_series.empty:
        return False

    if pd_series.map(lambda v: isinstance(v, pd.Timestamp)).any():
        return True

    sample = pd_series.astype(str).str.strip().head(50)
    if sample.str.match(r"^-?\d+(\.\d+)?$").all():
        return False

    return bool(sample.map(lambda s: bool(TIMESTAMP_RE.match(s))).mean() > 0.5)

def _infer_column_lux_type(series: pd.Series) -> tuple[str, Optional[pd.Series]]:
    # if numeric and number of unique values is less than 3 and greater than 1, return nominal!
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) > 1 and series.nunique(dropna=True) <= 3:
            return "nominal", None
        return "quantitative", None
    if pd.api.types.is_bool_dtype(series):
        return "nominal", None
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal", None

    non_null = series.dropna()
    if non_null.empty:
        return "nominal", None

    # temporal values
    if _values_look_temporal(non_null):
        converted = pd.to_datetime(series, errors="coerce")
        ok = int(converted.notna().sum())
        if ok >= max(1, int(0.9 * len(non_null))): # check if the converted series is not null for at least 90% of the non-null values
            return "temporal", converted

    # numeric values
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        converted = pd.to_numeric(series, errors="coerce")
        finite = converted.dropna()
        if len(finite) and (finite % 1 == 0).all():
            try:
                converted = converted.astype("Int64")
            except (TypeError, ValueError):
                pass
        if converted.nunique(dropna=True) > 1 and converted.nunique(dropna=True) <= 3:
            return "nominal", converted
        return "quantitative", converted

    return "nominal", None

def process_data_types_inplace(lux_df) -> None:
    """
    Fix Lux inferred types in place.
    """
    overrides: Dict[str, str] = {}
    for col in list(lux_df.columns):
        lux_type, converted = _infer_column_lux_type(lux_df[col])
        if converted is not None:
            lux_df[col] = converted
        overrides[col] = lux_type

    if overrides:
        for col, lux_type in overrides.items():
            lux_df.set_data_type({col: lux_type})

def lux_auto_chart(
    df: pd.DataFrame,
    *,
    result_type: str,
    title: Optional[str] = None,
    grapher=None,
) -> List[dict]:
    """
    Run Lux on a multi-variable DataFrame focused on ``result_type``.

    Returns Vega-Lite spec(s).
    """
    if df is None or df.empty:
        return []
    if result_type not in df.columns:
        return []

    df = df.dropna(subset=[result_type])
    df.drop_duplicates(inplace=True)

    if "build" in df.columns and df["build"].nunique() == 1:
        df.drop(columns=["build"], inplace=True)

    if df.empty:
        return []

    n_vars = len([c for c in df.columns if c != result_type])
    n_charts = resolve_n_charts(grapher, n_vars)

    lux, _Vis, Clause = _import_lux(topk=n_charts)
    specs: List[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ldf = lux.LuxDataFrame(df.copy())

        process_data_types_inplace(ldf)

        user_intent = resolve_graph_intent(grapher, ldf.columns)

        vis_list = _pick_auto_vis_list(
            ldf, result_type, Clause, n_charts, user_intent=user_intent
        )
        for i, vis in enumerate(vis_list):
            chart_title = title

            specs.append(
                _vis_to_vegalite(
                    vis,
                    ldf,
                    title=chart_title,
                    grapher=grapher,
                    result_type=result_type,
                )
            )

    if not specs:
        print(f"WARNING: Lux produced no visualizations for {result_type}")
    return specs

def build_chart_for_result(
    grapher,
    *,
    result_type: str,
    series,
    title: Optional[str],
) -> List[dict]:
    """
    Produce Vega-Lite spec(s) for one result metric via Lux auto-visualization.
    """
    df = series_result_to_dataframe(series, result_type)

    if df.empty:
        return []

    return lux_auto_chart(df, result_type=result_type, title=title, grapher=grapher)

def render_vegalite(
    spec: dict,
    *,
    fmt: str = "pdf",
    scale: float = 2.0,
) -> bytes:
    """Render a Vega-Lite dict to PNG or PDF bytes via vl-convert."""
    vlc = _import_vl_convert()
    fmt = fmt.lower().lstrip(".")

    if fmt == "png":
        return vlc.vegalite_to_png(vl_spec=spec, scale=scale)
    if fmt == "svg":
        return vlc.vegalite_to_svg(vl_spec=spec).encode("utf-8")
    if fmt in ("pdf",):
        return vlc.vegalite_to_pdf(vl_spec=spec)
    return vlc.vegalite_to_pdf(vl_spec=spec)

def save_chart(
    spec: dict,
    path: str,
    *,
    dpi_scale: float = 2.0,
    also_save_vl: bool = True,
) -> None:
    """Write chart bytes (and optional .vl.json) to disk."""
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "pdf"
    data = render_vegalite(spec, fmt=ext, scale=dpi_scale)
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if also_save_vl:
        vl_path = os.path.splitext(path)[0] + ".vl.json"
        with open(vl_path, "w") as f:
            json.dump(spec, f, indent=2)

def plot_graphs_with_lux(grapher, graphs, filename, fileprefix, f_series=None) -> Dict[str, Any]:
    """
    Lux-based replacement for Grapher.plot_graphs.

    For each result metric, generates several charts based on how many
    independent variables are present. Chart data comes from ``f_series`` when provided (series before series_to_graph /
    variable-to-series extraction) so all Run variables remain columns for Lux.
    """
    import npf

    assert len(graphs) > 0
    graph = graphs[0]
    if len(graph.series) == 0:
        return {}

    # Still materialize XYEB once so --output CSV side-effects keep working
    _ = graph.dataset(kind=fileprefix)
    chart_series = f_series if f_series is not None else graph.series
    if len(chart_series) == 0:
        return {}
    one_test, one_build, _ = chart_series[0]

    if grapher.options.no_graph:
        return {}

    ret: Dict[str, Any] = {}
    dpi = getattr(grapher.options, "graph_dpi", 300) or 300
    scale = max(1.0, float(dpi) / 150.0)

    title = graph.subtitle if graph.subtitle else graph.title
    result_types = collect_result_types(chart_series)

    for result_type in sorted(result_types):
        try:
            specs = build_chart_for_result(
                grapher,
                result_type=result_type,
                series=chart_series,
                title=title,
            )
        except Exception as e:
            print(f"ERROR: Lux failed for {result_type}: {e}")
            traceback.print_exc()
            specs = []

        if not specs:
            continue

        for i, spec in enumerate(specs, start=1):
            out_key = (
                result_type
                if len(specs) == 1
                else f"{result_type}-{i}"
            )

            if grapher.return_fig:
                ret[out_key] = spec
            elif not filename:
                ret[out_key] = render_vegalite(spec, fmt="pdf", scale=scale)
            else:
                type_filename = npf.build_filename(
                    one_test,
                    one_build,
                    filename if filename is not True else None,
                    graph.statics(),
                    "pdf",
                    type_str=(fileprefix + "-" if fileprefix else "") + out_key,
                    show_serie=False,
                )
                try:
                    save_chart(spec, type_filename, dpi_scale=scale, also_save_vl=True)
                    print("Graph saved to %s" % type_filename)
                    ret[out_key] = None
                except Exception as e:
                    print("ERROR : Could not draw the graph!")
                    print(e)
                    traceback.print_exc()
                    ret[out_key] = None
    return ret
