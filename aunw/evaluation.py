"""How good is the model? Evaluation on synthetic test images and on real data.

Two levels are evaluated separately, because they fail for different reasons:

1. Segmentation (pixel level): Dice = 2|A∩B| / (|A| + |B|) and
   IoU = |A∩B| / |A∪B| for the wire mask and the centre-line, mean axial
   angle error for the orientation head.
2. Measurement (object level), only possible with synthetic ground truth:
   detected wires are matched to true wires, and the relative errors
   (measured - true) / true of length and width are computed. Length is only
   evaluated for wires flagged *complete* (both tips visible, not touching the
   border), exactly as in the final statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .dataset import list_test_images
from .inference import resize_image, estimate_scale, predict_tiled
from .measure import analyze_maps, match_to_ground_truth


def dice_iou(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    t, p = truth.astype(bool), pred.astype(bool)
    inter = np.logical_and(t, p).sum()
    s = t.sum() + p.sum()
    union = np.logical_or(t, p).sum()
    return (2 * inter / s if s else 1.0), (inter / union if union else 1.0)


def evaluate_synthetic(model, data_dir: Path, cfg: AnalysisConfig | None = None,
                       n_images: int = 12, progress=None) -> dict:
    """Run the full pipeline on synthetic test images and compare with the truth."""
    cfg = cfg or AnalysisConfig()
    seg_rows, wire_rows, img_rows = [], [], []
    files = list_test_images(data_dir)[:n_images]
    for k, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        lab = np.load(str(f).replace(".png", "_labels.npz"))
        gt = pd.DataFrame(json.loads(Path(str(f).replace(".png", ".json")).read_text())).set_index("id")
        maps = predict_tiled(model, img, overlap=cfg.tile_overlap, tta=cfg.tta)

        # --- 1) pixel level
        dm, im = dice_iou(lab["mask"], maps["mask"] > cfg.mask_threshold)
        dc, _ = dice_iou(lab["centerline"], maps["centerline"] > cfg.centerline_threshold)
        valid = lab["angle"] < 180
        t_true = np.radians(lab["angle"][valid])
        t_pred = 0.5 * np.arctan2(maps["orientation"][..., 1], maps["orientation"][..., 0])[valid]
        d = np.abs(t_true - t_pred) % np.pi
        seg_rows.append({"image": f.name, "mask_dice": dm, "mask_iou": im, "centerline_dice": dc,
                         "angle_error_deg": float(np.degrees(np.minimum(d, np.pi - d)).mean())})

        # --- 2) object level: matched wires
        wires, info = analyze_maps(maps, cfg)
        for di, gid in match_to_ground_truth(wires, lab["instance"]).items():
            w, g = wires[di], gt.loc[gid]
            wire_rows.append({"image": f.name, "true_length": g.length_px, "true_width": g.width_px,
                              "length": w["length_px"], "width": w["width_px"],
                              "width_profile": w.get("width_profile_px", np.nan),
                              "complete": w["complete"], "true_inside": bool(g.inside)})

        # --- 3) sample level: what a user compares (statistics of one image)
        det = pd.DataFrame([{k: v for k, v in w.items() if k != "path"} for w in wires])
        inside = gt[gt.inside]
        comp = det[det.complete] if len(det) else det
        bias = lambda a, b: 100 * (a / b - 1)
        img_rows.append({
            "image": f.name, "coverage": float(lab["mask"].mean()), "true_wires": len(gt),
            "detected": len(det), "complete": len(comp),
            "crossing_resolution": info["crossing_resolution"],
            "median_length_bias_%": bias(comp.length_px.median(), inside.length_px.median())
                                    if len(comp) else np.nan,
            "stereological_mean_length_bias_%": bias(info["stereological_mean_length_px"],
                                                     gt.length_px.mean()),
            "median_width_bias_%": bias(det.width_px.median(), gt.width_px.median())
                                   if len(det) else np.nan})
        if progress:
            progress((k + 1) / len(files))

    seg, wires_df, per_image = pd.DataFrame(seg_rows), pd.DataFrame(wire_rows), pd.DataFrame(img_rows)
    reliable = per_image.coverage <= cfg.coverage_reliable
    summary = {"images": len(files), "images_reliable_coverage": int(reliable.sum()),
               "mask_dice": seg.mask_dice.mean(), "mask_iou": seg.mask_iou.mean(),
               "centerline_dice": seg.centerline_dice.mean(),
               "angle_error_deg": seg.angle_error_deg.mean(),
               "crossing_resolution": per_image.crossing_resolution.mean()}
    for tag, sel in (("all", per_image.image), ("reliable", per_image.image[reliable])):
        w = wires_df[wires_df.image.isin(sel)] if len(wires_df) else wires_df
        c = w[w.complete & w.true_inside] if len(w) else w
        rel_l = (c.length - c.true_length) / c.true_length
        rel_w = (w.width - w.true_width) / w.true_width
        p = per_image[per_image.image.isin(sel)]
        summary.update({
            f"{tag}: complete matched wires": len(c),
            f"{tag}: length median rel. error %": 100 * rel_l.median(),
            f"{tag}: length within 10 % fraction": float((rel_l.abs() < 0.1).mean()) if len(c) else np.nan,
            f"{tag}: width median rel. error %": 100 * rel_w.median(),
            f"{tag}: width within 10 % fraction": float((rel_w.abs() < 0.1).mean()) if len(w) else np.nan,
            f"{tag}: profile width within 10 % fraction":
                float(((w.width_profile - w.true_width).abs() / w.true_width < 0.1).mean()) if len(w) else np.nan,
            f"{tag}: per-image |median length bias| %": p["median_length_bias_%"].abs().median(),
            f"{tag}: per-image |stereological mean length bias| %":
                p["stereological_mean_length_bias_%"].abs().median(),
            f"{tag}: per-image |median width bias| %": p["median_width_bias_%"].abs().median(),
        })
    comp_all = wires_df[wires_df.complete & wires_df.true_inside] if len(wires_df) else wires_df
    return {"summary": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                        for k, v in summary.items()},
            "segmentation": seg, "wires": wires_df, "complete": comp_all, "per_image": per_image}


def evaluate_real(model, images: list[np.ndarray], masks: list[np.ndarray], names: list[str],
                  cfg: AnalysisConfig | None = None) -> pd.DataFrame:
    """Segmentation quality on hand-annotated real SEM images."""
    cfg = cfg or AnalysisConfig()
    rows = []
    for img, mask, name in zip(images, masks, names):
        scale = estimate_scale(model, img, cfg) if cfg.auto_scale else 1.0
        maps = predict_tiled(model, resize_image(img, scale), overlap=cfg.tile_overlap, tta=cfg.tta)
        prob = cv2.resize(maps["mask"], img.shape[::-1], interpolation=cv2.INTER_LINEAR)
        dm, im = dice_iou(mask > 0, prob > cfg.mask_threshold)
        rows.append({"image": name, "analysis_scale": scale, "mask_dice": dm, "mask_iou": im})
    return pd.DataFrame(rows)
