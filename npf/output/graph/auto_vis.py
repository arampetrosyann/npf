"""
LIDA → automatic chart generation for NPF.

Fully automatic: plot type, marks, scales, and layout are chosen by LIDA (LLM).
Configurable via %config: graph_openai_api_key, graph_generation_library.

Flow:
  series → to_pandas → one-metric DataFrame (real variable columns)
        → LIDA summarize + visualize (goal focused on the metric)
        → raster PNG/PDF (+ optional chart code sidecar)
"""

from __future__ import annotations

import base64
import os
import sys
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Set
from PIL import Image as PILImage

import pandas as pd

from npf.globals import npf_root_path
from npf.output.transform.pandas import to_pandas

def _ensure_local_lida():
    local = os.path.join(npf_root_path(), "lida")

    if os.path.isdir(os.path.join(local, "lida")):
        return local

    raise RuntimeError(
        f"Vendored LIDA fork not found at {local}. "
        "Clone/fork LIDA into the NPF repo as ./lida."
    )

def _load_openai_api_key(grapher):
    """
    Resolve the OpenAI API key from the grapher config or environment variable.
    """
    cfg = grapher.config("graph_openai_api_key", None)

    if cfg is not None and str(cfg).strip():
        return str(cfg).strip()

    value = os.environ.get("OPENAI_API_KEY")

    if value and value.strip():
        return value.strip()

    print("ERROR: OpenAI API key not found. Set the OPENAI_API_KEY environment variable or graph_openai_api_key in the .npf %config.")

    return None

def _import_lida():
    _ensure_local_lida()
    from lida import Manager, TextGenerationConfig, llm, goal_for_metric

    return Manager, TextGenerationConfig, llm, goal_for_metric

def _get_lida_manager(grapher):
    Manager, TextGenerationConfig, llm, _goal_for_metric = _import_lida()
    api_key = _load_openai_api_key(grapher)

    if api_key is None:
        return None, None, None

    text_gen = llm("openai", api_key=api_key)
    manager = Manager(text_gen=text_gen)
    config = TextGenerationConfig(
        n=1,
        temperature=0,
        top_p=1,
        seed=42
    )
    
    return manager, config, _goal_for_metric

def _generation_library(grapher) -> str:
    cfg = grapher.config("graph_generation_library", None)

    if cfg is None or not str(cfg).strip():
        return "matplotlib"
    
    return str(cfg).strip()

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

def _chart_raster_bytes(chart) -> Optional[bytes]:
    raster = getattr(chart, "raster", None)
    if not raster:
        return None
    if isinstance(raster, bytes):
        return raster
    if isinstance(raster, str):
        # LIDA typically returns a base64-encoded PNG string
        try:
            return base64.b64decode(raster)
        except Exception:
            return raster.encode("utf-8")
    return None

def _png_to_pdf_bytes(png_bytes: bytes) -> bytes:
    image = PILImage.open(BytesIO(png_bytes)).convert("RGB")
    out = BytesIO()
    image.save(out, format="PDF")
    return out.getvalue()

def lida_auto_chart(
    df: pd.DataFrame,
    *,
    result_type: str,
    title: Optional[str] = None,
    grapher=None,
) -> Optional[Dict[str, Any]]:
    """
    Run LIDA on a multi-variable DataFrame focused on ``result_type``.

    Returns a dict with keys: raster (bytes), code (str), library, status.
    """
    if df is None or df.empty:
        return None
    if result_type not in df.columns:
        return None

    df = df.dropna(subset=[result_type])
    df.drop_duplicates(inplace=True)
    if df.empty:
        return None

    library = _generation_library(grapher)

    manager, textgen_config, goal_for_metric = _get_lida_manager(grapher)

    if manager is None or textgen_config is None or goal_for_metric is None:
        return None

    summary = manager.summarize(
        df,
        summary_method="default",
        textgen_config=textgen_config,
    )

    # Goal text is modified in the LIDA fork (lida/components/goal.py)
    goal = goal_for_metric(result_type, title=title)
    charts = manager.visualize(
        summary=summary,
        goal=goal,
        textgen_config=textgen_config,
        library=library,
        return_error=True,
    )

    if not charts:
        print(f"WARNING: LIDA produced no visualizations for {result_type}")
        return None

    chart = charts[0]
    status = getattr(chart, "status", True)
    error = getattr(chart, "error", None)

    if status is False or error:
        print(f"WARNING: LIDA chart error for {result_type}: {error}")
        return None

    raster = _chart_raster_bytes(chart)
    if raster is None:
        print(f"WARNING: LIDA chart for {result_type} has no image data")
        return None

    return {
        "raster": raster,
        "code": getattr(chart, "code", None),
        "library": library,
        "status": True,
        "result_type": result_type,
        "title": title,
    }

def build_chart_for_result(
    grapher,
    *,
    result_type: str,
    series,
    title: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Produce one LIDA chart for a result metric.
    """
    df = series_result_to_dataframe(series, result_type)

    if df.empty:
        return None

    return lida_auto_chart(
        df,
        result_type=result_type,
        title=title,
        grapher=grapher,
    )

def save_chart(
    chart: Dict[str, Any],
    path: str,
    *,
    save_code: bool = True,
) -> None:
    """Write the chart raster as PDF, and optionally the generated LIDA code."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    if not path.lower().endswith(".pdf"):
        path = os.path.splitext(path)[0] + ".pdf"

    with open(path, "wb") as f:
        f.write(_png_to_pdf_bytes(chart["raster"]))

    if save_code and chart.get("code"):
        code_path = os.path.splitext(path)[0] + ".py"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(chart["code"])

def plot_graphs_with_lida(grapher, graphs, filename, fileprefix, f_series=None) -> Dict[str, Any]:
    """
    LIDA-based replacement for Grapher.plot_graphs.

    One automatic chart per result metric. Chart data comes from ``f_series``
    when provided (series before series_to_graph / variable-to-series extraction)
    so all Run variables remain columns for LIDA.
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
    
    title = graph.subtitle if graph.subtitle else graph.title
    result_types = collect_result_types(chart_series)

    for result_type in sorted(result_types):
        try:
            chart = build_chart_for_result(
                grapher,
                result_type=result_type,
                series=chart_series,
                title=title,
            )
        except Exception as e:
            print(f"ERROR: LIDA failed for {result_type}: {e}")
            traceback.print_exc()
            chart = None

        if chart is None:
            continue

        out_key = result_type

        if grapher.return_fig:
            ret[out_key] = {
                "code": chart.get("code"),
                "library": chart.get("library"),
                "result_type": result_type,
            }
        elif not filename:
            ret[out_key] = chart["raster"]
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
                save_chart(chart, type_filename, save_code=True)
                print("Graph saved to %s" % type_filename)
                ret[out_key] = None
            except Exception as e:
                print("ERROR : Could not draw the graph!")
                print(e)
                traceback.print_exc()
                ret[out_key] = None
    return ret
