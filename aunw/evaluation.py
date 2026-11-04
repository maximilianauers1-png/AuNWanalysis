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
    seg_rows, wire_rows, dist_rows = [], [], []
    pooled = {"true_length": [], "length": [], "true_width": [], "width": []}   # population level
    files = list_test_images(data_dir)[:n_images]
    for k, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        lab = np.load(str(f).replace(".png", "_labels.npz"))
        gt = pd.DataFrame(json.loads(Path(str(f).replace(".png", ".json")).read_text())).set_index("id")
        maps = predict_tiled(model, img, overlap=cfg.tile_overlap, tta=cfg.tta)

        dm, im = dice_iou(lab["mask"], maps["mask"] > cfg.mask_threshold)
        dc, _ = dice_iou(lab["centerline"], maps["centerline"] > cfg.centerline_threshold)
        valid = lab["angle"] < 180
        t_true = np.radians(lab["angle"][valid])
        t_pred = 0.5 * np.arctan2(maps["orientation"][..., 1], maps["orientation"][..., 0])[valid]
        d = np.abs(t_true - t_pred) % np.pi
        seg_rows.append({"image": f.name, "mask_dice": dm, "mask_iou": im, "centerline_dice": dc,
                         "angle_error_deg": float(np.degrees(np.minimum(d, np.pi - d)).mean())})

        wires, info = analyze_maps(maps, cfg)
        match = match_to_ground_truth(wires, lab["instance"])
        for di, gid in match.items():
            w, g = wires[di], gt.loc[gid]
            wire_rows.append({"image": f.name, "true_length": g.length_px, "true_width": g.width_px,
                              "length": w["length_px"], "width": w["width_px"],
                              "complete": w["complete"], "true_inside": bool(g.inside)})
        inside = gt[gt.inside]
        det = pd.DataFrame([{k: v for k, v in w.items() if k != "path"} for w in wires])
        pooled["true_length"] += inside.length_px.tolist()
        pooled["true_width"] += gt.width_px.tolist()
        if len(det):
            pooled["length"] += det.loc[det.complete, "length_px"].tolist()
            pooled["width"] += det.width_px.tolist()
        dist_rows.append({"image": f.name, "true_wires_inside": len(inside),
                          "detected": len(det), "complete": int(det["complete"].sum()) if len(det) else 0,
                          "crossing_resolution": info["crossing_resolution"],
                          "true_mean_length": inside.length_px.mean(),
                          "measured_mean_length_complete":
                              det.loc[det.complete, "length_px"].mean() if len(det) else np.nan,
                          "true_mean_width": gt.width_px.mean(),
                          "measured_mean_width": det.width_px.mean() if len(det) else np.nan})
        if progress:
            progress((k + 1) / len(files))

    seg, wires_df, dist = pd.DataFrame(seg_rows), pd.DataFrame(wire_rows), pd.DataFrame(dist_rows)
    comp = wires_df[wires_df.complete & wires_df.true_inside] if len(wires_df) else wires_df
    rel_l = (comp.length - comp.true_length) / comp.true_length if len(comp) else pd.Series(dtype=float)
    rel_w = ((wires_df.width - wires_df.true_width) / wires_df.true_width
             if len(wires_df) else pd.Series(dtype=float))
    summary = {
        "images": len(files),
        "mask_dice": seg.mask_dice.mean(), "mask_iou": seg.mask_iou.mean(),
        "centerline_dice": seg.centerline_dice.mean(), "angle_error_deg": seg.angle_error_deg.mean(),
        "matched_wires": len(wires_df), "complete_matched_wires": len(comp),
        "length_median_rel_error_%": 100 * rel_l.median(),
        "length_within_10%_fraction": float((rel_l.abs() < 0.1).mean()) if len(rel_l) else np.nan,
        "width_median_rel_error_%": 100 * rel_w.median(),
        "width_within_10%_fraction": float((rel_w.abs() < 0.1).mean()) if len(rel_w) else np.nan,
        # population level: all complete measured wires vs all fully visible true wires
        **{f"population_{stat}_{q}_bias_%": 100 * (fn(pooled[q]) / fn(pooled[f"true_{q}"]) - 1)
           for q in ("length", "width") for stat, fn in (("mean", np.mean), ("median", np.median))
           if pooled[q]},
        "crossing_resolution": dist.crossing_resolution.mean(),
    }
    return {"summary": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                        for k, v in summary.items()},
            "segmentation": seg, "wires": wires_df, "complete": comp, "per_image": dist}


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
