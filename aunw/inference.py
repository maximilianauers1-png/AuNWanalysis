"""Applying the trained network to SEM images of any size, and the full analysis chain.

Tiled prediction
    Large micrographs (e.g. 2048 x 1536) do not fit through the network at once
    on modest hardware, so the image is processed in overlapping tiles. Each
    tile prediction is multiplied by a window that is 1 in the centre and falls
    smoothly (sin^2) to ~0 in the overlap; the weighted sum is divided by the sum
    of weights. This removes seams, because tile borders (where convolutions see
    zero padding) get almost no weight.

Test-time augmentation (TTA)
    The image is also predicted flipped horizontally, vertically and both; the
    outputs are flipped back and averaged. For the orientation channels the flip
    is undone on the double-angle vector: (c, s) -> (c, -s) for a single flip.

Automatic scale
    The network was trained on wires 3-20 px wide. If a micrograph shows much
    wider (or thinner) wires, it is resized so that the median wire width is
    about `target_width_px`; all measurements are converted back afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import keras
import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import SEMImage
from .measure import analyze_maps

MULTIPLE = 16   # input size must be divisible by 2**depth (depth <= 4)


def _window(tile: int, overlap: int) -> np.ndarray:
    ramp = np.sin(0.5 * np.pi * (np.arange(overlap) + 0.5) / overlap) ** 2
    w1 = np.ones(tile, np.float32)
    w1[:overlap], w1[-overlap:] = ramp, ramp[::-1]
    return np.outer(w1, w1)[..., None]


def _predict_batch(model: keras.Model, batch: np.ndarray, tta: bool) -> np.ndarray:
    """Return stacked outputs (N, H, W, 4): mask, centerline, cos2t, sin2t."""
    def run(x):
        out = model.predict_on_batch(x)
        return np.concatenate([out["mask"], out["centerline"], out["orientation"]], -1)

    if not tta:
        return run(batch)
    acc = run(batch)
    for axes in ((2,), (1,), (1, 2)):                      # LR, UD, both
        y = np.flip(run(np.flip(batch, axes)), axes)
        if len(axes) == 1:
            y[..., 3] *= -1                                 # a single flip mirrors the angle
        acc += y
    return acc / 4


def predict_tiled(model: keras.Model, image: np.ndarray, tile: int = 256, overlap: int = 64,
                  batch_size: int = 8, tta: bool = True) -> dict[str, np.ndarray]:
    """Sliding-window prediction with smooth blending. image: float32 (H, W) in [0, 1]."""
    h, w = image.shape
    step = tile - overlap
    # pad by `overlap` on every side (so image borders get full weight), then up to
    # an integer number of tiles: n = ceil((H + 2o - T) / s) + 1
    n_y = max(1, int(np.ceil((h + 2 * overlap - tile) / step)) + 1)
    n_x = max(1, int(np.ceil((w + 2 * overlap - tile) / step)) + 1)
    ph, pw = (n_y - 1) * step + tile, (n_x - 1) * step + tile
    pad = ((overlap, ph - h - overlap), (overlap, pw - w - overlap))
    img = np.pad(image, pad, mode="reflect")[..., None].astype(np.float32)

    acc = np.zeros(img.shape[:2] + (4,), np.float32)
    wsum = np.zeros(img.shape[:2] + (1,), np.float32)
    win = _window(tile, overlap)
    coords = [(y, x) for y in range(0, ph - tile + 1, step) for x in range(0, pw - tile + 1, step)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        pred = _predict_batch(model, np.stack([img[y:y + tile, x:x + tile] for y, x in chunk]), tta)
        for (y, x), p in zip(chunk, pred):
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    out = (acc / np.maximum(wsum, 1e-6))[overlap:overlap + h, overlap:overlap + w]
    return {"mask": out[..., 0], "centerline": out[..., 1], "orientation": out[..., 2:4]}


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1) < 1e-3:
        return img
    h, w = img.shape
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (max(32, round(w * scale)), max(32, round(h * scale))),
                              interpolation=interp), 0, 1)


def estimate_scale(model: keras.Model, image: np.ndarray, cfg: AnalysisConfig,
                   max_iter: int = 3) -> float:
    """Find a resize factor that brings the median wire width to cfg.target_width_px.

    Works on a central crop for speed and iterates because a network that sees
    wires far outside its training range under-estimates how wrong it is.
    """
    h, w = image.shape
    crop = image[max(0, h // 2 - 384):h // 2 + 384, max(0, w // 2 - 384):w // 2 + 384]
    scale = 1.0
    for _ in range(max_iter):
        maps = predict_tiled(model, resize_image(crop, scale), tta=False)
        _, info = analyze_maps(maps, cfg)
        wpx = info["width_px_global"]
        if not np.isfinite(wpx):
            break
        ratio = cfg.target_width_px / wpx
        if 0.75 < ratio < 1.33:
            break
        scale = float(np.clip(scale * ratio, 0.1, 4.0))
    return scale


# --------------------------------------------------------------------------- full analysis
@dataclass
class ImageResult:
    name: str
    wires: pd.DataFrame          # one row per wire, px and (if calibrated) nm columns
    maps: dict[str, np.ndarray]  # network outputs at analysis scale
    scale: float                 # analysis scale relative to the original image
    nm_per_px: float | None
    image_shape: tuple[int, int]
    info: dict


def analyze_image(model: keras.Model, sem: SEMImage, cfg: AnalysisConfig,
                  scale: float | None = None) -> ImageResult:
    """SEM image -> table of individual wires with physical units."""
    if scale is None:
        scale = estimate_scale(model, sem.pixels, cfg) if cfg.auto_scale else 1.0
    img = resize_image(sem.pixels, scale)
    maps = predict_tiled(model, img, overlap=cfg.tile_overlap, tta=cfg.tta)
    wires, info = analyze_maps(maps, cfg)

    df = pd.DataFrame([{k: v for k, v in w.items() if k != "path"} for w in wires])
    if len(df):
        for col in ("length_px", "width_px", "x_px", "y_px", "extent_x_px", "extent_y_px"):
            df[col] = df[col] / scale                       # back to original pixels
        if sem.nm_per_px:
            for col in ("length", "width"):
                df[f"{col}_nm"] = df[f"{col}_px"] * sem.nm_per_px
        df.insert(0, "image", sem.name)
    paths = [[p / scale for p in w["path"]] for w in wires]
    info["paths"] = paths
    return ImageResult(sem.name, df, maps, scale, sem.nm_per_px, sem.shape, info)
