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
import os
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

def _import_lux():
    _ensure_lux_pandas_compat()
    import lux
    from lux.vis.Vis import Vis
    from lux import Clause

    lux.config.plotting_backend = "vegalite"
    lux.config.topk = 1
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

def _vis_to_vegalite(vis, df: pd.DataFrame, title: Optional[str] = None) -> dict:
    spec = vis.to_vegalite(prettyOutput=False)
    if isinstance(spec, str):
        spec = json.loads(spec)
    if not isinstance(spec, dict):
        raise RuntimeError(f"Unexpected Vega-Lite type: {type(spec)}")
    spec = _inline_vegalite_data(spec, df)
    if title:
        spec["title"] = title
    return spec

def _pick_auto_vis(ldf, result_type: str, Clause):
    var_cols = [c for c in ldf.columns if c != result_type and c != "build"]
    max_wildcards = min(2, len(var_cols))

    for n_wildcards in range(max_wildcards, -1, -1):
        intent = [result_type] + [Clause("?")] * n_wildcards
        ldf.intent = intent
        # Clear cached recs so Lux recomputes for this intent
        if hasattr(ldf, "_recommendation"):
            ldf._recommendation = {}
        if hasattr(ldf, "_rec_info"):
            ldf._rec_info = []

        try:
            recs = ldf.recommendation or {}
        except Exception as e:
            print(f"WARNING: Lux recommendation failed for intent {intent}: {e}")
            continue

        vis = None

        for vislist in (recs or {}).values():
            if vislist and len(vislist) > 0:
                vis = vislist[0]
                break

        if vis is not None:
            return vis

    return None

def lux_auto_chart(
    df: pd.DataFrame,
    *,
    result_type: str,
    title: Optional[str] = None,
) -> Optional[dict]:
    """Run Lux automatically on a multi-variable DataFrame; return Vega-Lite dict."""
    if df is None or df.empty:
        return None
    if result_type not in df.columns:
        return None

    df = df.dropna(subset=[result_type])
    df.drop_duplicates(inplace=True)
    if df.empty:
        return None

    lux, _Vis, Clause = _import_lux()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ldf = lux.LuxDataFrame(df.copy())
        print(f"{ldf}")
        vis = _pick_auto_vis(ldf, result_type, Clause)
        print(f"ldf.columns: {ldf.columns}")
        print(f"ldf.dtypes: {ldf.dtypes}")
        print(f"ldf.intent: {ldf.intent}")
        print(f"vis: {vis}")
        if vis is None:
            print(f"WARNING: Lux produced no visualizations for {result_type}")
            return None
        return _vis_to_vegalite(vis, ldf, title=title)

def build_chart_for_result(
    grapher,
    *,
    result_type: str,
    series,
    title: Optional[str],
) -> Optional[dict]:
    """
    Produce a Vega-Lite spec for one result metric via Lux auto-visualization.
    """
    df = series_result_to_dataframe(series, result_type)

    if df.empty:
        return None

    return lux_auto_chart(df, result_type=result_type, title=title)

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

    One automatic chart per result metric. Chart data comes from ``f_series``
    when provided (series before series_to_graph / variable-to-series extraction)
    so all Run variables remain columns for Lux.
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
        spec = build_chart_for_result(
            grapher,
            result_type=result_type,
            series=chart_series,
            title=title,
        )
        if spec is None:
            continue

        out_key = result_type

        if grapher.return_fig:
            ret[out_key] = spec
        elif not filename:
            ret[out_key] = render_vegalite(spec, fmt="png", scale=scale)
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
