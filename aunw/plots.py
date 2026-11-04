"""All figures (matplotlib, object-oriented API -> thread-safe in the GUI).

Visual conventions: one categorical palette in fixed order (blue, orange,
aqua), thin marks, light recessive grid, text in neutral ink, a legend
whenever two or more series are shown. Every figure can be exported as PNG
(300 dpi), SVG or PDF via `figure_bytes`.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats as st

from .config import ModelConfig
from .stats import freedman_diaconis_bins, weighted_quantile

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAVY = "#0d2c6b"


def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")


def _fig(w: float = 6.4, h: float = 4.0, n: int = 1, m: int = 1, **kw):
    fig = Figure(figsize=(w, h), dpi=110, facecolor="white", layout="constrained")
    return fig, fig.subplots(n, m, **kw)


def figure_bytes(fig: Figure, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300 if fmt == "png" else None, facecolor="white")
    return buf.getvalue()


# --------------------------------------------------------------------------- statistics
def histogram_fit(x, label: str, unit: str, weights=None, title: str = "") -> Figure:
    """Histogram (probability density) with normal and log-normal ML fits."""
    x = np.asarray(x, float)
    keep = np.isfinite(x) & (x > 0)
    x = x[keep]
    w = None if weights is None else np.asarray(weights, float)[keep]
    fig, ax = _fig()
    _style(ax, f"{label} [{unit}]", "Probability density", title)
    if len(x) < 2:
        ax.text(0.5, 0.5, "not enough wires", transform=ax.transAxes, ha="center", color=INK2)
        return fig
    ax.hist(x, bins=freedman_diaconis_bins(x), weights=w, density=True, color=BLUE, alpha=0.55,
            edgecolor="white", linewidth=1.0, label=f"measured (n = {len(x)})")
    grid = np.linspace(max(x.min() * 0.8, 1e-9), x.max() * 1.1, 400)
    mu = np.average(np.log(x), weights=w)
    sig = np.sqrt(np.average((np.log(x) - mu) ** 2, weights=w))
    mean = np.average(x, weights=w)
    sd = np.sqrt(np.average((x - mean) ** 2, weights=w))
    ax.plot(grid, st.lognorm.pdf(grid, sig, scale=np.exp(mu)), color=ORANGE, lw=2,
            label=f"log-normal fit (median {np.exp(mu):.3g}, σ = {sig:.2f})")
    ax.plot(grid, st.norm.pdf(grid, mean, sd), color=AQUA, lw=2, ls="--",
            label=f"normal fit ({mean:.3g} ± {sd:.3g})")
    med = weighted_quantile(x, 0.5, w)
    ax.axvline(med, color=INK2, lw=1, ls=":")
    ax.text(med, ax.get_ylim()[1] * 0.97, f" median {med:.3g}", color=INK2, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return fig


def length_vs_width(df: pd.DataFrame, lcol: str, wcol: str, unit: str, f: float = 1.0) -> Figure:
    fig, ax = _fig()
    _style(ax, f"Width [{unit}]", f"Length [{unit}]", "Length vs. width")
    if df.empty:
        return fig
    x, y = df[wcol] * f, df[lcol] * f
    if len(df) > 400:
        hb = ax.hexbin(x, y, gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="wires per bin")
    else:
        ax.scatter(x, y, s=18, color=BLUE, alpha=0.7, edgecolor="white", linewidth=0.5)
    if len(df) > 3:
        r = st.spearmanr(x, y)
        ax.text(0.02, 0.97, f"Spearman ρ = {r.statistic:.2f} (p = {r.pvalue:.2g}), n = {len(df)}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    return fig


def ecdf_by_image(df: pd.DataFrame, col: str, label: str, unit: str, f: float = 1.0) -> Figure:
    """Empirical CDFs per image (first three images coloured, others grey)."""
    fig, ax = _fig()
    _style(ax, f"{label} [{unit}]", "Cumulative fraction", f"{label}: cumulative distribution")
    colours = [BLUE, ORANGE, AQUA]
    for i, (name, g) in enumerate(df.groupby("image", sort=False)):
        x = np.sort(g[col].to_numpy() * f)
        if not len(x):
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        c = colours[i] if i < 3 else "#9a9893"
        ax.step(x, y, where="post", color=c, lw=2 if i < 3 else 1,
                label=name if i < 3 else ("other images" if i == 3 else None))
    ax.legend(frameon=False, fontsize=8)
    return fig


def box_by_image(df: pd.DataFrame, col: str, label: str, unit: str, f: float = 1.0) -> Figure:
    groups = [(n, g[col].to_numpy() * f) for n, g in df.groupby("image", sort=False)]
    fig, ax = _fig(max(4.0, 1.0 + 0.7 * len(groups)), 4.0)
    _style(ax, "", f"{label} [{unit}]", f"{label} per image")
    if not groups:
        return fig
    bp = ax.boxplot([g for _, g in groups], patch_artist=True, widths=0.5,
                    medianprops={"color": INK, "lw": 1.5}, flierprops={"markersize": 3})
    for patch in bp["boxes"]:
        patch.set(facecolor=BLUE, alpha=0.45, edgecolor=INK2)
    ax.set_xticks(range(1, len(groups) + 1), [n[:18] for n, _ in groups], rotation=30, ha="right")
    return fig


# --------------------------------------------------------------------------- images
def overlay(image: np.ndarray, wires: pd.DataFrame, paths: list, show_ids: bool = True,
            thickness: int = 2) -> np.ndarray:
    """RGB overlay: complete wires blue, ending in a crossing orange, cut by the border green."""
    rgb = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    rgb = (rgb * 0.75).astype(np.uint8)
    colours = {"complete": (42, 120, 214), "partial": (235, 104, 52), "border": (27, 175, 122)}
    thickness = max(thickness, int(round(min(image.shape) / 300)))   # visible on large images
    for (_, row), segs in zip(wires.iterrows(), paths):
        kind = "complete" if row["complete"] else ("border" if row["touches_border"] else "partial")
        for p in segs:
            pts = np.round(p[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(rgb, [pts], False, colours[kind], thickness, cv2.LINE_AA)
        if show_ids:
            cv2.putText(rgb, str(int(row["wire_id"])), (int(row["x_px"]), int(row["y_px"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def maps_panel(image: np.ndarray, maps: dict) -> Figure:
    """Input + the three network outputs side by side."""
    fig, axes = _fig(12, 3.4, 1, 4)
    ang = 0.5 * np.arctan2(maps["orientation"][..., 1], maps["orientation"][..., 0])
    conf = np.clip(np.hypot(*np.moveaxis(maps["orientation"], -1, 0)), 0, 1)
    hsv = np.stack([(ang % np.pi) / np.pi, np.ones_like(ang), conf * maps["mask"]], -1)
    from matplotlib.colors import hsv_to_rgb
    panels = [(image, "gray", "SEM input"), (maps["mask"], "magma", "Wire probability"),
              (maps["centerline"], "magma", "Centre-line probability"),
              (hsv_to_rgb(hsv), None, "Orientation (hue = angle)")]
    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.axis("off")
    return fig


# --------------------------------------------------------------------------- training
def training_curves(history: list[dict] | pd.DataFrame) -> Figure:
    """Small multiples: total loss, mask Dice, centre-line Dice, angle error, learning rate."""
    h = pd.DataFrame(history)
    fig, axes = _fig(12, 6.4, 2, 3)
    axes = axes.ravel()
    panels = [("loss", "Total loss", True), ("mask_dice_coef", "Mask Dice", False),
              ("centerline_dice_coef", "Centre-line Dice", False),
              ("orientation_angle_error_deg", "Angle error [deg]", True),
              ("mask_loss", "Mask loss", True)]
    for ax, (key, title, lower_better) in zip(axes, panels):
        _style(ax, "Epoch", "", title + (" (lower is better)" if lower_better else " (higher is better)"))
        if key in h:
            ax.plot(h["epoch"], h[key], color=BLUE, lw=2, marker="o", ms=3, label="training")
        if f"val_{key}" in h:
            ax.plot(h["epoch"], h[f"val_{key}"], color=ORANGE, lw=2, marker="o", ms=3, label="validation")
            if lower_better and len(h):
                b = h[f"val_{key}"].idxmin()
                ax.axvline(h["epoch"][b], color=INK2, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[5]
    _style(ax, "Epoch", "", "Learning rate")
    if "lr" in h and len(h):
        ax.plot(h["epoch"], h["lr"], color=BLUE, lw=2, marker="o", ms=3)
        ax.set_yscale("log")
        ax.set_ylim(h["lr"].min() / 3, h["lr"].max() * 3)   # log autoscale fails for constant LR
    return fig


def prediction_preview(preview: dict) -> Figure:
    """Rows = validation samples; columns = input, true/predicted mask & centre-line."""
    n = len(preview["image"])
    fig, axes = _fig(12, 2.6 * n, n, 5, squeeze=False)
    titles = ["Input", "Mask (truth)", "Mask (network)", "Centre-line (truth)", "Centre-line (network)"]
    keys = ["image", "mask_true", "mask_pred", "centerline_true", "centerline_pred"]
    for r in range(n):
        for c, (k, t) in enumerate(zip(keys, titles)):
            axes[r, c].imshow(preview[k][r], cmap="gray" if c == 0 else "magma", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(t, fontsize=10, color=INK, loc="left")
    return fig


# --------------------------------------------------------------------------- network
def unet_diagram(cfg: ModelConfig) -> Figure:
    """Schematic of the residual U-Net: box height ~ resolution, width ~ log(filters)."""
    fig, ax = _fig(12, 5.6)
    ax.set_xlim(0, 2 * cfg.depth + 4.2)
    ax.set_ylim(-cfg.depth - 1.8, 1.4)
    ax.axis("off")
    filters, size = cfg.filters, cfg.tile_size

    def block(x, y, f, res, colour, text):
        w = 0.35 + 0.12 * np.log2(f)
        hgt = 0.35 + 0.5 * res / size
        ax.add_patch(FancyBboxPatch((x - w / 2, y - hgt / 2), w, hgt, boxstyle="round,pad=0.02",
                                    facecolor=colour, edgecolor=NAVY, lw=1, alpha=0.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x, y - hgt / 2 - 0.12, f"{res}×{res}×{f}", ha="center", va="top", fontsize=7, color=INK2)

    def arrow(a, b, colour=INK2, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, color=colour,
                                     lw=1.2, linestyle=ls, shrinkA=14, shrinkB=14))

    ax.text(0.2, 1.15, "Input SEM tile", fontsize=9, color=INK)
    enc_x = [1 + i for i in range(cfg.depth)]
    dec_x = [2 * cfg.depth + 2 - i - 1 for i in range(cfg.depth)]
    for i in range(cfg.depth):
        res = size // 2**i
        block(enc_x[i], -i, filters[i], res, BLUE, f"Res{filters[i]}")
        block(dec_x[i], -i, filters[i], res, AQUA, f"Res{filters[i]}")
        arrow((enc_x[i], -i), (dec_x[i], -i), ORANGE, "-|>", "--")          # skip connection
        nxt = (enc_x[i + 1], -i - 1) if i + 1 < cfg.depth else (cfg.depth + 1, -cfg.depth)
        arrow((enc_x[i], -i), nxt)                                         # max-pool
        prev = (dec_x[i + 1], -i - 1) if i + 1 < cfg.depth else (cfg.depth + 1, -cfg.depth)
        arrow(prev, (dec_x[i], -i))                                        # up-sample
    block(cfg.depth + 1, -cfg.depth, filters[-1], size // 2**cfg.depth, NAVY,
          f"Res{filters[-1]}")
    ax.text(cfg.depth + 1, -cfg.depth - 0.75, "bottleneck + SpatialDropout", ha="center",
            fontsize=8, color=INK2)
    hx = dec_x[0] + 1.25
    for k, (name, act) in enumerate((("mask", "sigmoid"), ("centre-line", "sigmoid"),
                                     ("orientation (cos2θ, sin2θ)", "tanh"))):
        y = 0.6 - 0.6 * k
        arrow((dec_x[0], 0), (hx - 0.25, y))
        ax.text(hx - 0.2, y, f"1×1 conv → {name}\n[{act}]", va="center", fontsize=8, color=INK)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, label="max-pool ↓ / bilinear up-sample ↑"),
                       Line2D([], [], color=ORANGE, lw=1.2, ls="--", label="skip connection (concatenate)"),
                       Line2D([], [], marker="s", ls="", color=BLUE, label="encoder residual block"),
                       Line2D([], [], marker="s", ls="", color=AQUA, label="decoder residual block")],
              loc="lower left", frameon=False, fontsize=8)
    return fig


def feature_maps(model, image: np.ndarray, layer_name: str, n: int = 16) -> Figure:
    """Activations of one layer for an input tile: what does the network 'see'?"""
    import keras
    sub = keras.Model(model.input, model.get_layer(layer_name).output)
    x = image[None, ..., None].astype(np.float32)
    act = np.asarray(sub.predict_on_batch(x))[0]
    # show the most active channels
    order = np.argsort(act.reshape(-1, act.shape[-1]).std(0))[::-1][:n]
    cols = 8
    rows = int(np.ceil(len(order) / cols))
    fig, axes = _fig(12, 1.6 * rows + 0.4, rows, cols, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), order):
        ax.imshow(act[..., c], cmap="viridis")
        ax.set_title(f"ch {c}", fontsize=7, color=INK2)
    return fig


# --------------------------------------------------------------------------- evaluation
def parity_plot(true, measured, label: str, unit: str = "px") -> Figure:
    fig, ax = _fig(4.6, 4.4)
    _style(ax, f"True {label.lower()} [{unit}]", f"Measured {label.lower()} [{unit}]",
           f"{label}: measured vs. truth")
    true, measured = np.asarray(true), np.asarray(measured)
    if len(true):
        lim = [0, max(true.max(), measured.max()) * 1.05]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--", label="y = x")
        ax.scatter(true, measured, s=14, color=BLUE, alpha=0.6, edgecolor="white", lw=0.4,
                   label="matched wires")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        rel = (measured - true) / true
        ax.text(0.02, 0.97, f"median rel. error {np.median(rel) * 100:+.1f} %\n"
                f"|error| < 10 %: {np.mean(np.abs(rel) < 0.1) * 100:.0f} % of wires",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    return fig
