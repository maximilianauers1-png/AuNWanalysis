"""Synthetic SEM images of chaotic gold-nanowire (AuNW) meshes with exact labels.

Why synthetic data?
    Pixel-accurate manual labels of hundreds of overlapping wires are practically
    impossible to obtain. A physically motivated renderer gives unlimited images
    with *exact* ground truth (mask, centreline, orientation, true length and
    width of every wire). Wide randomisation of every nuisance parameter
    ("domain randomisation") forces the network to learn the wire shape itself
    rather than the rendering style, which is what makes transfer to real SEM
    images possible. Real annotated images can be added later (fine-tuning).

Geometry of one wire
    A wire is a *capsule* swept along a circular arc: centreline length L_c,
    width w, round caps of radius w/2. Its true tip-to-tip length is
    L = L_c + w. The curvature is kappa = beta / L_c with total turning angle
    beta. For every pixel p we compute analytically
        d(p)     = Euclidean distance to the centreline arc,
        theta(p) = tangent angle of the closest centreline point,
    which gives sub-pixel exact masks and orientations without rasterisation bias.

Image formation (secondary-electron contrast)
    * SE yield of a tilted surface element ~ 1 / cos(phi). On a cylinder of
      radius R the local tilt satisfies cos(phi) = sqrt(1 - (r/R)^2) at
      lateral offset r, which produces the bright wire edges typical of SEM.
    * A weak dark halo models detector shadowing next to the wire.
    * Gaussian blur = finite probe size, Poisson noise = finite electron dose,
      row-wise offsets = scan-line noise, optional JPEG compression.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import SyntheticConfig
from .io import normalize

IGNORE_ANGLE = 255  # label value for "no reliable orientation" (background, crossings)


@dataclass
class Wire:
    """Ground truth of one synthetic wire (all units in pixels)."""

    p0: np.ndarray        # centreline start (x, y)
    theta0: float         # tangent angle at the start (rad, image coordinates)
    kappa: float          # curvature (1/px); 0 means straight
    lc: float             # centreline length
    width: float

    @property
    def length(self) -> float:
        """Tip-to-tip length including the two hemispherical caps."""
        return self.lc + self.width

    def point(self, s: np.ndarray | float) -> np.ndarray:
        """Centreline point(s) at arc length s."""
        s = np.asarray(s, dtype=np.float64)
        if abs(self.kappa) < 1e-9:
            d = np.array([np.cos(self.theta0), np.sin(self.theta0)])
            return self.p0 + s[..., None] * d
        th = self.theta0 + self.kappa * s
        return self.p0 + np.stack([np.sin(th) - np.sin(self.theta0),
                                   np.cos(self.theta0) - np.cos(th)], -1) / self.kappa

    @property
    def mean_angle_deg(self) -> float:
        return float(np.degrees(self.theta0 + 0.5 * self.kappa * self.lc) % 180.0)

    def bbox(self, pad: float) -> tuple[float, float, float, float]:
        pts = self.point(np.linspace(0, self.lc, 16))
        (x0, y0), (x1, y1) = pts.min(0) - pad, pts.max(0) + pad
        return x0, y0, x1, y1

    def distance_field(self, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact distance d to the centreline and tangent angle theta of the closest point."""
        if abs(self.kappa) < 1e-9:                       # straight segment
            u = np.array([np.cos(self.theta0), np.sin(self.theta0)])
            t = np.clip((xx - self.p0[0]) * u[0] + (yy - self.p0[1]) * u[1], 0, self.lc)
            d = np.hypot(xx - self.p0[0] - t * u[0], yy - self.p0[1] - t * u[1])
            return d, np.full_like(d, self.theta0)

        rho = 1.0 / abs(self.kappa)                      # radius of curvature
        sgn = np.sign(self.kappa)
        c = self.p0 + rho * sgn * np.array([-np.sin(self.theta0), np.cos(self.theta0)])
        vx, vy = xx - c[0], yy - c[1]
        theta_p = np.arctan2(vy, vx) + sgn * np.pi / 2   # tangent angle at the radial projection
        th_mid = self.theta0 + 0.5 * self.kappa * self.lc
        delta = np.angle(np.exp(1j * (theta_p - th_mid)))  # wrapped to (-pi, pi]
        s = 0.5 * self.lc + delta / self.kappa
        on_arc = (s >= 0) & (s <= self.lc)

        d_arc = np.abs(np.hypot(vx, vy) - rho)
        e0, e1 = self.point(0.0), self.point(self.lc)
        d0, d1 = np.hypot(xx - e0[0], yy - e0[1]), np.hypot(xx - e1[0], yy - e1[1])
        d = np.where(on_arc, d_arc, np.minimum(d0, d1))
        theta_end = np.where(d0 < d1, self.theta0, self.theta0 + self.kappa * self.lc)
        return d, np.where(on_arc, theta_p, theta_end)


# --------------------------------------------------------------------------- sampling
def _sample_wires(rng: np.random.Generator, cfg: SyntheticConfig, size: int) -> list[Wire]:
    area_scale = (size / cfg.tile_size) ** 2           # keep the density for larger canvases
    n = int(rng.integers(*cfg.n_wires) * area_scale)
    median = rng.uniform(*cfg.length_median)
    sigma = rng.uniform(*cfg.length_sigma)
    w_mean = rng.uniform(*cfg.width_mean)
    w_cv = rng.uniform(*cfg.width_cv)

    wires = []
    for _ in range(n):
        width = float(np.clip(rng.normal(w_mean, w_cv * w_mean), 2.0, 40.0))
        length = float(rng.lognormal(np.log(median), sigma))
        lc = max(length - width, 2.0)
        bend = np.radians(rng.uniform(-cfg.max_bend_deg, cfg.max_bend_deg))
        theta0 = rng.uniform(0, 2 * np.pi)
        centre = rng.uniform(-0.15 * size, 1.15 * size, 2)
        tmp = Wire(np.zeros(2), theta0, bend / lc, lc, width)
        p0 = centre - tmp.point(lc / 2)                 # shift so the arc midpoint is `centre`
        wires.append(Wire(p0, theta0, bend / lc, lc, width))
    return wires


def _substrate(rng: np.random.Generator, size: int, level: float, porous: bool) -> np.ndarray:
    img = np.full((size, size), level, np.float32)
    img += 0.06 * gaussian_filter(rng.standard_normal((size, size)), size / 10) * (size / 10)
    img += rng.uniform(0.0, 0.04) * gaussian_filter(rng.standard_normal((size, size)), 1.0)
    gy, gx = np.mgrid[0:size, 0:size] / size
    img += rng.uniform(-0.08, 0.08) * gx + rng.uniform(-0.08, 0.08) * gy   # charging gradient
    if porous:                                          # dark pores of a filter membrane
        n = int(rng.uniform(0.5, 3.0) * size * size / 1000)
        for x, y, r in zip(rng.uniform(0, size, n), rng.uniform(0, size, n), rng.uniform(1, 4, n)):
            cv2.circle(img, (int(x), int(y)), int(r), float(level * 0.4), -1, cv2.LINE_AA)
    return img


def _draw_particles(rng, img, size, level, edge_gain):
    """Au nanoparticles (by-product of the synthesis); they are *not* labelled as wires."""
    for _ in range(int(rng.integers(1, 12) * (size / 256) ** 2)):
        r = rng.uniform(2, 12)
        cx, cy = rng.uniform(0, size, 2)
        x0, x1 = int(max(cx - r - 2, 0)), int(min(cx + r + 3, size))
        y0, y1 = int(max(cy - r - 2, 0)), int(min(cy + r + 3, size))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - cx, yy - cy)
        soft = np.clip(r - d + 0.5, 0, 1)
        u = np.clip(d / r, 0, 0.95)
        prof = np.clip(level * (1 + edge_gain * (1 / np.sqrt(1 - u**2) - 1)), 0, 1)
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - soft) + prof * soft


# --------------------------------------------------------------------------- rendering
def render_sample(rng: np.random.Generator, cfg: SyntheticConfig,
                  size: int | None = None) -> dict:
    """Render one synthetic SEM image with labels.

    Returns a dict with
        image       uint8 (H, W)   normalised SEM-like image
        mask        uint8 (H, W)   1 = wire
        centerline  uint8 (H, W)   1 = within 1 px of a wire centreline
        angle       uint8 (H, W)   wire orientation in degrees [0, 180), 255 = ignore
        instance    int32 (H, W)   centreline pixels labelled with wire index + 1
        wires       list[Wire]     ground-truth geometry
    """
    size = size or cfg.tile_size
    bg = rng.uniform(0.05, 0.45)
    wire_level = rng.uniform(bg + 0.25, 0.95)
    edge_gain = rng.uniform(*cfg.edge_gain)
    halo = rng.uniform(0.0, 0.25), rng.uniform(1.0, 4.0)   # (strength, decay length)

    img = _substrate(rng, size, bg, rng.random() < cfg.pore_prob)
    if rng.random() < cfg.particle_prob:
        _draw_particles(rng, img, size, wire_level, edge_gain)

    count = np.zeros((size, size), np.uint8)         # number of wires covering each pixel
    centerline = np.zeros((size, size), bool)
    angle = np.zeros((size, size), np.float32)
    instance = np.zeros((size, size), np.int32)

    wires = _sample_wires(rng, cfg, size)
    for idx, wire in enumerate(wires):              # list order = z-order (later = on top)
        r = wire.width / 2
        pad = r + 3 * halo[1] + 2
        x0, y0, x1, y1 = wire.bbox(pad)
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(np.ceil(x1)), size), min(int(np.ceil(y1)), size)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        d, theta = wire.distance_field(xx, yy)
        win = (slice(y0, y1), slice(x0, x1))

        # 1) shadow halo outside the wire, 2) wire with SE edge-effect profile on top
        img[win] *= 1 - halo[0] * np.exp(-np.maximum(d - r, 0) / halo[1]) * (d > r)
        soft = np.clip(r - d + 0.5, 0, 1)            # anti-aliased coverage, exact width 2r
        u = np.clip(d / r, 0, 0.97)
        level = wire_level * rng.uniform(0.9, 1.1)
        profile = np.clip(level * (1 + edge_gain * (1 / np.sqrt(1 - u**2) - 1)), 0, 1)
        img[win] = img[win] * (1 - soft) + profile * soft

        inside = soft >= 0.5
        count[win] += inside
        angle[win] = np.where(inside, theta, angle[win])
        centerline[win] |= d <= 1.0
        instance[win] = np.where(d <= 1.5, idx + 1, instance[win])

    # ---- acquisition model: probe blur, shot noise, read noise, scan lines, JPEG
    img = gaussian_filter(img, rng.uniform(*cfg.blur_sigma))
    dose = rng.uniform(*cfg.photons)
    img = rng.poisson(np.clip(img, 0, None) * dose) / dose
    img += rng.normal(0, rng.uniform(0.005, 0.03), img.shape)
    img += rng.normal(0, rng.uniform(0.0, 0.02), (size, 1))      # scan-line offsets
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    img8 = (normalize(img) * 255).astype(np.uint8)
    if rng.random() < cfg.jpeg_prob:
        ok, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        img8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # orientation label only where exactly one wire is present (crossings are ambiguous)
    ang_deg = np.round(np.degrees(angle) % 180.0) % 180
    ang_lbl = np.where(count == 1, ang_deg, IGNORE_ANGLE).astype(np.uint8)
    return {"image": img8, "mask": (count > 0).astype(np.uint8),
            "centerline": (centerline & (count > 0)).astype(np.uint8),
            "angle": ang_lbl, "instance": instance, "wires": wires}


def wire_table(wires: list[Wire], size: int) -> list[dict]:
    """Ground-truth table; `inside` marks wires whose full length is visible."""
    rows = []
    for i, w in enumerate(wires):
        pts = w.point(np.linspace(0, w.lc, 32))
        r = w.width / 2
        inside = bool((pts.min() >= r) and (pts.max() <= size - 1 - r))
        rows.append({"id": i + 1, "length_px": w.length, "width_px": w.width,
                     "angle_deg": w.mean_angle_deg, "inside": inside})
    return rows
