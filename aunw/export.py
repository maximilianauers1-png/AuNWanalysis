"""Export of results: CSV, Excel workbook, figures and overlays as one ZIP bundle.

Every export contains the analysis settings and calibration so that a result
can always be traced back to how it was produced (reproducibility).
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from . import plots
from .config import AnalysisConfig
from .stats import edge_weights, length_width_frame, summary_table


def wires_table(results) -> pd.DataFrame:
    frames = [r.wires for r in results if len(r.wires)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_image_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r.wires
        rows.append({
            "image": r.name, "height_px": r.image_shape[0], "width_px": r.image_shape[1],
            "nm_per_px": r.nm_per_px, "analysis_scale": r.scale,
            "wires_total": len(d), "wires_complete": int(d["complete"].sum()) if len(d) else 0,
            "wires_touching_border": int(d["touches_border"].sum()) if len(d) else 0,
            "coverage": r.info.get("coverage"), "reliability": r.info.get("reliability"),
            "crossing_resolution": r.info.get("crossing_resolution"),
            "stereological_mean_length_px": r.info.get("stereological_mean_length_px"),
            "stereological_mean_length_nm": (r.info.get("stereological_mean_length_px", np.nan)
                                             * r.nm_per_px) if r.nm_per_px else np.nan,
            "junctions": r.info.get("n_junctions"),
            "median_length_px_complete": d.loc[d["complete"], "length_px"].median() if len(d) else np.nan,
            "median_width_px": d["width_px"].median() if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def frames_of(results) -> dict[str, tuple[int, int]]:
    return {r.name: (r.image_shape[1], r.image_shape[0]) for r in results}


def settings_dict(cfg: AnalysisConfig, model_name: str = "") -> dict:
    return {"exported": datetime.now().isoformat(timespec="seconds"), "model": model_name,
            "analysis": asdict(cfg)}


def to_excel(results, cfg: AnalysisConfig, unit: str = "nm", model_name: str = "") -> bytes:
    wires = wires_table(results)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        summary_table(wires, unit, True, frames_of(results)).to_excel(xl, sheet_name="Summary")
        wires.to_excel(xl, sheet_name="Wires", index=False)
        per_image_table(results).to_excel(xl, sheet_name="Per image", index=False)
        pd.DataFrame(list(_flatten(settings_dict(cfg, model_name)).items()),
                     columns=["setting", "value"]).to_excel(xl, sheet_name="Settings", index=False)
    return buf.getvalue()


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v if not isinstance(v, (list, tuple)) else str(v)
    return out


def standard_figures(results, unit: str = "nm", edge_correction: bool = True) -> dict:
    """The default set of figures for a list of ImageResults."""
    wires = wires_table(results)
    if wires.empty:
        return {}
    if unit != "px" and "length_nm" not in wires:
        unit = "px"
    suffix, label, f = length_width_frame(wires, unit)
    complete = wires[wires["complete"]]
    weights = edge_weights(complete, frames_of(results)) if edge_correction else None
    title = "Wire length (complete wires" + (", edge-corrected)" if edge_correction else ")")
    return {
        "length_histogram": plots.histogram_fit(complete[f"length{suffix}"] * f, "Length", label,
                                                weights=weights, title=title),
        "width_histogram": plots.histogram_fit(wires[f"width{suffix}"] * f, "Width", label,
                                               title="Wire width (all wires)"),
        "length_vs_width": plots.length_vs_width(complete, f"length{suffix}", f"width{suffix}", label, f),
        "length_per_image": plots.box_by_image(complete, f"length{suffix}", "Length", label, f),
        "width_per_image": plots.box_by_image(wires, f"width{suffix}", "Width", label, f),
        "width_cdf": plots.ecdf_by_image(wires, f"width{suffix}", "Width", label, f),
    }


def export_bundle(results, cfg: AnalysisConfig, unit: str = "nm", images: dict | None = None,
                  model_name: str = "", fig_formats=("png", "svg", "pdf")) -> bytes:
    """ZIP: tables (CSV + XLSX), figures (PNG/SVG/PDF), overlays, settings."""
    wires = wires_table(results)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wires.csv", wires.to_csv(index=False))
        z.writestr("summary.csv", summary_table(wires, unit, True, frames_of(results)).to_csv())
        z.writestr("per_image.csv", per_image_table(results).to_csv(index=False))
        z.writestr("results.xlsx", to_excel(results, cfg, unit, model_name))
        z.writestr("settings.json", json.dumps(settings_dict(cfg, model_name), indent=2))
        for name, fig in standard_figures(results, unit).items():
            for fmt in fig_formats:
                z.writestr(f"figures/{name}.{fmt}", plots.figure_bytes(fig, fmt))
        for r in results:
            if images and r.name in images:
                rgb = plots.overlay(images[r.name], r.wires, r.info["paths"])
                ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                z.writestr(f"overlays/{r.name}_overlay.png", png.tobytes())
    return buf.getvalue()
