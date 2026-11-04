"""Dataset creation (synthetic + real annotations) and the tf.data input pipeline.

Storage format
    train.npz / val.npz contain uint8 arrays of shape (N, H, W):
        image, mask, centerline, angle   (angle in degrees, 255 = ignore)
    test/ contains full-size images (PNG) + ground-truth tables (JSON) used to
    validate the *measurement* pipeline end-to-end, not just the segmentation.

Orientation target
    A wire axis is a direction *modulo 180 deg* (theta and theta + pi are the
    same wire). Regressing theta directly would produce a discontinuity at
    0/180 deg. We therefore regress the double-angle vector
        o = (cos 2 theta, sin 2 theta),
    which is continuous and unique. A third channel holds the per-pixel loss
    weight (1 on single-wire pixels, 0 on background and crossings).

Width target
    For every wire pixel the width (px) of the top-most wire there. In dense
    meshes the wire mask is one connected area, so widths cannot be measured
    from mask edges; a learned width map is not affected by touching wires.

Geometric augmentation must transform the labels consistently:
    horizontal or vertical flip:  theta -> -theta (mod pi)   =>  (c, s) -> (c, -s)
    rotation by 90 deg:           theta -> theta + pi/2      =>  (c, s) -> (-c, -s)
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf
from skimage.morphology import skeletonize

from .config import SyntheticConfig
from .synthetic import IGNORE_ANGLE, WIDTH_SCALE, render_sample, wire_table

KEYS = ("image", "mask", "centerline", "angle", "width")


# --------------------------------------------------------------------------- generation
def _render_tiles(args: tuple[int, int, SyntheticConfig]) -> dict[str, np.ndarray]:
    seed, n, cfg = args
    rng = np.random.default_rng(seed)
    samples = [render_sample(rng, cfg) for _ in range(n)]
    return {k: np.stack([s[k] for s in samples]) for k in KEYS}


def generate_split(n: int, cfg: SyntheticConfig, seed: int, workers: int = 4,
                   progress: Callable[[float], None] | None = None) -> dict[str, np.ndarray]:
    """Render `n` training tiles in parallel processes (deterministic for a given seed)."""
    chunk = 32
    jobs = [(seed * 100_003 + i, min(chunk, n - i * chunk), cfg)
            for i in range((n + chunk - 1) // chunk)]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(_render_tiles, jobs)):
            parts.append(part)
            if progress:
                progress((i + 1) / len(jobs))
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def generate_dataset(out_dir: Path, cfg: SyntheticConfig, n_train: int = 1500,
                     n_val: int = 250, n_test: int = 12, test_size: int = 768,
                     seed: int = 0, workers: int = 4,
                     progress: Callable[[str, float], None] | None = None) -> None:
    """Create train/val tiles and full-size test images with ground truth.

    Validation and test data come from *different random seeds* than training
    data, so they are statistically independent samples of the generator.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, n, s in (("train", n_train, seed), ("val", n_val, seed + 1)):
        cb = (lambda f, name=name: progress(name, f)) if progress else None
        np.savez_compressed(out_dir / f"{name}.npz", **generate_split(n, cfg, s, workers, cb))

    test_dir = out_dir / "test"
    test_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed + 2)
    for i in range(n_test):
        s = render_sample(rng, cfg, size=test_size)
        cv2.imwrite(str(test_dir / f"test_{i:03d}.png"), s["image"])
        np.savez_compressed(test_dir / f"test_{i:03d}_labels.npz",
                            **{k: s[k] for k in ("mask", "centerline", "angle", "width", "instance")})
        (test_dir / f"test_{i:03d}.json").write_text(
            json.dumps(wire_table(s["wires"], test_size), indent=1))
        if progress:
            progress("test", (i + 1) / n_test)
    (out_dir / "generator_config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=list))


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        out = {k: f[k] for k in KEYS if k in f.files}
    if "width" not in out:                      # datasets made before the width head existed
        out["width"] = np.zeros_like(out["mask"])
    return out


def list_test_images(data_dir: Path) -> list[Path]:
    return sorted((Path(data_dir) / "test").glob("test_*[0-9].png"))


# --------------------------------------------------------------------------- real data
def labels_from_mask(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Derive training targets from a hand-drawn binary wire mask.

    The centreline is approximated by the skeleton of the mask (dilated to the
    3-px width used for synthetic labels). Orientation is *not* derivable
    reliably at crossings, so it is set to IGNORE: real data then trains the
    mask and centreline heads only, while synthetic data keeps teaching the
    orientation head.
    """
    m = mask > 0
    cl = cv2.dilate(skeletonize(m).astype(np.uint8), np.ones((3, 3), np.uint8)) & m
    return {"mask": m.astype(np.uint8), "centerline": cl.astype(np.uint8),
            "angle": np.full(m.shape, IGNORE_ANGLE, np.uint8),
            "width": np.zeros(m.shape, np.uint8)}          # unknown -> excluded from the width loss


def tiles_from_annotations(images: list[np.ndarray], masks: list[np.ndarray],
                           tile: int = 256, stride: int = 192) -> dict[str, np.ndarray]:
    """Cut annotated real images (float [0,1] + binary mask) into training tiles."""
    out = {k: [] for k in KEYS}
    for img, mask in zip(images, masks):
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = labels_from_mask(mask)
        h, w = img8.shape
        ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
        xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
        for y in ys:
            for x in xs:
                crop = (slice(y, y + tile), slice(x, x + tile))
                if img8[crop].shape != (tile, tile):
                    continue
                out["image"].append(img8[crop])
                for k in ("mask", "centerline", "angle", "width"):
                    out[k].append(lab[k][crop])
    if not out["image"]:
        raise ValueError(f"Annotated images must be at least {tile}x{tile} px.")
    return {k: np.stack(v) for k, v in out.items()}


# --------------------------------------------------------------------------- tf.data
def _to_targets(image, mask, centerline, angle, width):
    """uint8 arrays -> (network input, dict of targets)."""
    x = tf.cast(image, tf.float32)[..., None] / 255.0
    valid = tf.cast(angle < 180, tf.float32)
    theta = tf.cast(angle, tf.float32) * (np.pi / 180.0)
    orient = tf.stack([tf.cos(2 * theta) * valid, tf.sin(2 * theta) * valid, valid], -1)
    w = tf.cast(width, tf.float32) / WIDTH_SCALE                 # px; 0 = unknown/background
    y = {"mask": tf.cast(mask, tf.float32)[..., None],
         "centerline": tf.cast(centerline, tf.float32)[..., None],
         "orientation": orient,
         "width": tf.stack([w, tf.cast(w > 0, tf.float32)], -1)}  # (value, loss weight)
    return x, y


def _augment(x, y):
    """Random flips/90-degree rotations (label-consistent) + intensity jitter."""
    stacked = tf.concat([x, y["mask"], y["centerline"], y["orientation"], y["width"]], -1)
    sign_s = tf.constant(1.0)
    if tf.random.uniform(()) < 0.5:
        stacked, sign_s = tf.image.flip_left_right(stacked), -sign_s
    if tf.random.uniform(()) < 0.5:
        stacked, sign_s = tf.image.flip_up_down(stacked), -sign_s
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    stacked = tf.image.rot90(stacked, k)
    sign_r = 1.0 - 2.0 * tf.cast(k % 2, tf.float32)     # -1 for odd k (90/270 deg)

    x, m, c, o, wd = (stacked[..., :1], stacked[..., 1:2], stacked[..., 2:3], stacked[..., 3:6],
                      stacked[..., 6:])                 # the width is invariant to flips/rotations
    o = tf.stack([o[..., 0] * sign_r, o[..., 1] * sign_r * sign_s, o[..., 2]], -1)

    # photometric jitter: contrast, brightness, gamma, additive noise
    x = (x - 0.5) * tf.random.uniform((), 0.75, 1.3) + 0.5 + tf.random.uniform((), -0.1, 0.1)
    x = tf.clip_by_value(x, 0.0, 1.0) ** tf.random.uniform((), 0.75, 1.35)
    x = x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform((), 0.0, 0.04))
    return tf.clip_by_value(x, 0.0, 1.0), {"mask": m, "centerline": c, "orientation": o, "width": wd}


def make_dataset(arrays: dict[str, np.ndarray], batch_size: int, training: bool,
                 augment: bool = True, seed: int = 0) -> tf.data.Dataset:
    """Build an efficient input pipeline: from_tensor_slices -> map -> shuffle -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices(tuple(arrays[k] for k in KEYS))
    ds = ds.map(_to_targets, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(arrays["image"]), 2048), seed=seed, reshuffle_each_iteration=True)
        if augment:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)


def mixed_dataset(real: dict, synthetic: dict, batch_size: int, real_fraction: float = 0.5,
                  seed: int = 0) -> tf.data.Dataset:
    """Fine-tuning stream: real and synthetic tiles sampled with a fixed ratio.

    Keeping synthetic data in the mix prevents *catastrophic forgetting* of what
    was learned from the (much larger) synthetic set.
    """
    def stream(a):
        ds = tf.data.Dataset.from_tensor_slices(tuple(a[k] for k in KEYS)).repeat()
        ds = ds.shuffle(min(len(a["image"]), 2048), seed=seed)
        return ds.map(_to_targets, num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.sample_from_datasets(
        [stream(real), stream(synthetic)], [real_fraction, 1 - real_fraction], seed=seed)
    return (ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
