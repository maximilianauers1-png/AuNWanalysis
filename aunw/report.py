"""Per-wire reports: which wire is where, how long and how wide.

For every analysed image this module produces
    * a wire table (one row per wire): ID, status, length, width, start/end/
      centre coordinates in pixels and micrometres,
    * the traced path of every wire as a list of points,
    * an annotated map: the micrograph with every wire drawn, numbered and
      (optionally) labelled with its length and width, on a coordinate grid,
    * Excel (table + embedded map + path points) and ZIP exports.

Coordinates: origin at the top-left corner of the analysed image (after the
info bar was cropped), x to the right, y downwards - the convention of ImageJ.
Start/end are the wire tips (centre-line ends + half a width), so that the
distance between them corresponds to the reported length of a straight wire.

Length of incomplete wires: if a wire ends in a crossing or leaves the image,
only its visible part can be measured. Its length is reported as a lower bound
("visible length", shown as "L >= ..." on the map).
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.figure import Figure

from .plots import AQUA, BLUE, ORANGE, figure_bytes

STATUS_COLOURS = {"complete": BLUE, "ends in crossing": ORANGE, "cut by border": AQUA}


# --------------------------------------------------------------------------- geometry
def _status(row) -> str:
    if row["complete"]:
        return "complete"
    return "cut by border" if row["touches_border"] else "ends in crossing"


def _endpoints_and_label(paths: list[np.ndarray], width: float,
                         shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start tip, end tip and a label anchor (on the wire, near its middle) in (x, y) pixels.

    The centre-line ends are the extreme points along the wire's principal axis
    (also for curved wires). The physical tips lie half a width further out
    (hemispherical caps), consistent with length = centre-line + width; they are
    clipped to the image.
    """
    pts = np.concatenate(paths)[:, ::-1].astype(float)          # (y, x) -> (x, y)
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    proj = (pts - centre) @ vt[0]
    h, w = shape
    lim = np.array([w - 1, h - 1], float)
    start = np.clip(pts[np.argmin(proj)] - vt[0] * width / 2, 0, lim)
    end = np.clip(pts[np.argmax(proj)] + vt[0] * width / 2, 0, lim)
    if (start[0], start[1]) > (end[0], end[1]):                   # deterministic: left end first
        start, end = end, start
    label = pts[np.argmin(np.abs(proj - np.median(proj)))]
    return start, end, label


def wire_table(result) -> pd.DataFrame:
    """User-facing table: one row per wire with its ID, measurements and coordinates."""
    df, paths = result.wires, result.info.get("paths", [])
    if df.empty:
        return pd.DataFrame()
    nm = result.nm_per_px
    rows = []
    for (_, w), p in zip(df.iterrows(), paths):
        start, end, label = _endpoints_and_label(p, w["width_px"], result.image_shape)
        status = _status(w)
        row = {"image": result.name, "wire_id": int(w["wire_id"]), "status": status,
               "length_type": "full" if status == "complete" else "visible (lower bound)",
               "length_px": w["length_px"], "width_px": w["width_px"]}
        if nm:
            row.update(length_nm=w["length_px"] * nm, width_nm=w["width_px"] * nm)
        row.update(aspect_ratio=w["aspect_ratio"], angle_deg=w["angle_deg"],
                   start_x_px=start[0], start_y_px=start[1], end_x_px=end[0], end_y_px=end[1],
                   centre_x_px=w["x_px"], centre_y_px=w["y_px"],
                   label_x_px=label[0], label_y_px=label[1])
        if nm:
            for k in ("start", "end", "centre"):
                row[f"{k}_x_um"] = row[f"{k}_x_px"] * nm / 1000
                row[f"{k}_y_um"] = row[f"{k}_y_px"] * nm / 1000
        rows.append(row)
    return pd.DataFrame(rows)


def path_points(result) -> pd.DataFrame:
    """Every traced centre-line point of every wire (for exact identification, e.g. in ImageJ)."""
    rows = []
    for wid, p in zip(result.wires["wire_id"], result.info.get("paths", [])):
        for seg_i, seg in enumerate(p, 1):
            for x, y in seg[:, ::-1]:
                rows.append((result.name, int(wid), seg_i, float(x), float(y)))
    out = pd.DataFrame(rows, columns=["image", "wire_id", "segment", "x_px", "y_px"])
    if result.nm_per_px:
        out["x_um"] = out["x_px"] * result.nm_per_px / 1000
        out["y_um"] = out["y_px"] * result.nm_per_px / 1000
    return out


# --------------------------------------------------------------------------- map
def _place_label(paths: list[np.ndarray], bw: float, bh: float,
                 placed: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """Greedy label placement: try points along the wire (middle first) until the
    label box does not overlap an already placed one. Returns (x, y) in pixels."""
    pts = np.concatenate(paths)[:, ::-1].astype(float)
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    order = np.argsort((pts - c) @ axis)
    candidates = [pts[order[int(q * (len(order) - 1))]] for q in (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9)]

    def free(x, y):
        return all(abs(x - px) * 2 > (bw + pw) or abs(y - py) * 2 > (bh + ph) for px, py, pw, ph in placed)

    x, y = next(((x, y) for x, y in candidates if free(x, y)), tuple(candidates[0]))
    placed.append((x, y, bw, bh))
    return x, y



def _fmt_len(v_nm: float) -> str:
    return f"{v_nm / 1000:.2f} µm" if v_nm >= 1000 else f"{v_nm:.0f} nm"


def wire_map(image: np.ndarray, result, show_values: bool = True,
             highlight: set[int] | None = None, title: str | None = None) -> Figure:
    """The micrograph with every wire drawn, numbered and labelled.

    show_values  True: "#ID  L .. / W .." next to every wire; False: IDs only
                 (more readable for very dense images, use with the table).
    highlight    wire IDs to emphasise (all others are dimmed).
    Axes are in micrometres if the image is calibrated, else in pixels.
    """
    h, w = image.shape
    table = wire_table(result)
    nm = result.nm_per_px
    k = nm / 1000 if nm else 1.0                                   # px -> axis unit
    unit = "µm" if nm else "px"

    dpi = 150
    scale_up = max(1.0, 1400 / max(w, h))                          # small images: larger figure
    fig = Figure(figsize=(w / dpi * scale_up + 1.2, h / dpi * scale_up + 1.0), dpi=dpi,
                 layout="constrained")
    ax = fig.subplots()
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, extent=(0, w * k, h * k, 0))
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.grid(True, color="white", alpha=0.25, linewidth=0.5)
    fs = float(np.clip(min(w, h) * scale_up / 110, 6, 11))
    lw = float(np.clip(min(w, h) * scale_up / 500, 1.2, 3.0))
    outline = [patheffects.withStroke(linewidth=2.2, foreground="black")]

    placed: list[tuple[float, float, float, float]] = []      # label boxes in image pixels
    px_per_pt = dpi / 72 / scale_up                            # 1 pt of text in image pixels
    for (_, row), paths in zip(table.iterrows(), result.info.get("paths", [])):
        wid = int(row["wire_id"])
        dim = highlight is not None and wid not in highlight
        colour = STATUS_COLOURS[row["status"]]
        for seg in paths:
            ax.plot(seg[:, 1] * k, seg[:, 0] * k, color=colour, lw=lw * (1.8 if highlight and not dim else 1),
                    alpha=0.25 if dim else 0.95, solid_capstyle="round")
        if dim:
            continue
        for end in ("start", "end"):
            ax.plot(row[f"{end}_x_px"] * k, row[f"{end}_y_px"] * k, "o", ms=lw * 1.6,
                    color=colour, mec="black", mew=0.5)
        text = f"{wid}"
        if show_values:
            if nm:
                ge = "" if row["status"] == "complete" else "≥"
                text += f"\nL{ge}{_fmt_len(row['length_nm'])}\nW {row['width_nm']:.0f} nm"
            else:
                ge = "" if row["status"] == "complete" else "≥"
                text += f"\nL{ge}{row['length_px']:.0f} px\nW {row['width_px']:.1f} px"
        lines = text.split("\n")
        bw = max(len(t) for t in lines) * fs * 0.62 * px_per_pt     # approximate label box
        bh = len(lines) * fs * 1.15 * px_per_pt
        lx, ly = _place_label(paths, bw, bh, placed)
        ax.text(lx * k, ly * k, text, color="white", fontsize=fs,
                ha="center", va="center", fontweight="bold", path_effects=outline, linespacing=1.0)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2.5, label=s) for s, c in STATUS_COLOURS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06 if nm else -0.08),
              ncol=3, frameon=False, fontsize=8)
    ax.set_title(title or f"{result.name} – {len(table)} wires"
                 + ("" if not show_values else "   (L ≥: visible length, wire not complete)"),
                 fontsize=10, loc="left")
    return fig


# --------------------------------------------------------------------------- exports
def map_png(image: np.ndarray, result, show_values: bool = True) -> bytes:
    return figure_bytes(wire_map(image, result, show_values), "png")


def image_excel(image: np.ndarray, result) -> bytes:
    """Excel workbook for one image: wire table, annotated map (embedded), path points."""
    from openpyxl.drawing.image import Image as XLImage
    table, points = wire_table(result), path_points(result)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        table.to_excel(xl, sheet_name="Wires", index=False)
        pd.DataFrame().to_excel(xl, sheet_name="Map (values)")
        pd.DataFrame().to_excel(xl, sheet_name="Map (IDs)")
        points.to_excel(xl, sheet_name="Path points", index=False)
        pd.DataFrame({"setting": ["image", "nm per pixel", "analysis scale", "coordinates",
                                  "length of incomplete wires"],
                      "value": [result.name, result.nm_per_px, result.scale,
                                "origin top-left, x right, y down (as in ImageJ)",
                                "visible length = lower bound"]}).to_excel(
            xl, sheet_name="Info", index=False)
        ws = xl.sheets["Wires"]
        ws.freeze_panes = "C2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(10, len(str(col[0].value)) + 2)
        for sheet, values in (("Map (values)", True), ("Map (IDs)", False)):
            png = io.BytesIO(figure_bytes(wire_map(image, result, values), "png"))
            img = XLImage(png)
            ratio = 1100 / img.width                                  # fit on screen
            img.width, img.height = img.width * ratio, img.height * ratio
            xl.sheets[sheet].add_image(img, "A1")
    return buf.getvalue()


def files_for_image(image: np.ndarray, result) -> dict[str, bytes]:
    """All per-wire result files of one image: {file name: content}."""
    stem = result.name.rsplit(".", 1)[0]
    files = {f"{stem}_wires.csv": wire_table(result).to_csv(index=False).encode("utf-8-sig"),
             f"{stem}_path_points.csv": path_points(result).to_csv(index=False).encode("utf-8-sig"),
             f"{stem}_wires.xlsx": image_excel(image, result)}
    for values, tag in ((True, "map_values"), (False, "map_ids")):
        fig = wire_map(image, result, values)
        files[f"{stem}_{tag}.png"] = figure_bytes(fig, "png")
        files[f"{stem}_{tag}.pdf"] = figure_bytes(fig, "pdf")
    return files


def all_files(results, images: dict[str, np.ndarray]) -> dict[str, bytes]:
    """Files for all images, one sub-folder per image, plus one combined table."""
    out, tables = {}, []
    for r in results:
        if r.name not in images:
            continue
        stem = r.name.rsplit(".", 1)[0]
        for fname, data in files_for_image(images[r.name], r).items():
            out[f"{stem}/{fname}"] = data
        tables.append(wire_table(r))
    if tables:
        out["all_wires.csv"] = pd.concat(tables, ignore_index=True).to_csv(index=False).encode("utf-8-sig")
    return out


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def image_zip(image: np.ndarray, result) -> bytes:
    """Everything for one image: tables (CSV + Excel) and maps (PNG + PDF)."""
    stem = result.name.rsplit(".", 1)[0]
    return _zip({f"{stem}/{k}": v for k, v in files_for_image(image, result).items()})


def all_images_zip(results, images: dict[str, np.ndarray]) -> bytes:
    """Per-wire reports for all images (one folder each) plus one combined table."""
    return _zip(all_files(results, images))


# --------------------------------------------------------------------------- saving to disk
def default_save_folder() -> Path:
    """Documents/AuNW Analysis (falls back to the home folder if there is no Documents)."""
    docs = Path.home() / "Documents"
    return (docs if docs.exists() else Path.home()) / "AuNW Analysis"


def save_files(files: dict[str, bytes], folder: Path) -> Path:
    """Write files into a new time-stamped sub-folder of `folder`; returns that sub-folder."""
    target = Path(folder) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for name, data in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target


def open_folder(folder: Path) -> None:
    """Open a folder in the file manager (Explorer, Finder, ...)."""
    folder = str(folder)
    if sys.platform == "win32":
        os.startfile(folder)                                   # noqa: S606 - local folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])
