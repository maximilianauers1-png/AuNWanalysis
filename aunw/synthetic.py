"""Synthetic SEM images of gold-nanowire (AuNW) meshes with exact labels.

Why synthetic data?
    Pixel-accurate manual labels of hundreds of overlapping wires are practically
    impossible to obtain. A physically motivated renderer gives unlimited images
    with *exact* ground truth (mask, centre-line, orientation, width map, true
    length and width of every wire). Wide randomisation of every nuisance
    parameter ("domain randomisation") forces the network to learn the wire
    itself rather than one rendering style.

Geometry of one wire
    A wire is a *capsule* swept along a circular arc: centre-line length L_c,
    width w, round caps of radius w/2, so the true tip-to-tip length is
    L = L_c + w. The curvature is kappa = beta / L_c (total turning angle beta).
    For every pixel p we compute analytically
        d(p)     distance to the centre-line,
        theta(p) tangent angle of the closest centre-line point,
        s(p)     arc-length coordinate of that point (for striations),
        n(p)     signed lateral offset (for facets),
    which gives sub-pixel exact labels without rasterisation bias.

Two imaging styles (chosen at random per image)
    sparse    wires lying on a flat substrate (filter membrane, wafer); strong
              secondary-electron edge contrast  I(r) ~ 1/sqrt(1 - (r/R)^2).
    dense3d   a thick, chaotic 3-D mesh as seen with an in-lens detector at low
              kV: several depth layers; deeper layers are darker (fewer
              electrons escape from below) and more defocused (limited depth of
              field); every wire shades the wire below it; wires are flat
              faceted ribbons with longitudinal facet lines, dark outlines and
              fine transverse striations; most wires cross the whole field.

Acquisition (both styles)
    Gaussian probe blur, Poisson shot noise, read noise, scan-line offsets,
    detector gamma, optional JPEG compression.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import SyntheticConfig
from .io import normalize

IGNORE_ANGLE = 255   # label value for "no reliable orientation" (background, crossings)
WIDTH_SCALE = 4      # width labels are stored as uint8 in units of 1/4 px


@dataclass
class Wire:
    """Ground truth of one synthetic wire (all units in pixels)."""

    p0: np.ndarray        # centre-line start (x, y)
    theta0: float         # tangent angle at the start (rad, image coordinates)
    kappa: float          # curvature (1/px); 0 means straight
    lc: float             # centre-line length
    width: float
    labelled: bool = True  # False for blurred background "texture" wires

    @property
    def length(self) -> float:
        """Tip-to-tip length including the two hemispherical caps."""
        return self.lc + self.width

    def point(self, s: np.ndarray | float) -> np.ndarray:
        """Centre-line point(s) at arc length s."""
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

    def fields(self, xx: np.ndarray, yy: np.ndarray):
        """Exact (d, theta, s, n) for every pixel: see module docstring."""
        if abs(self.kappa) < 1e-9:                       # straight segment
            u = np.array([np.cos(self.theta0), np.sin(self.theta0)])
            rx, ry = xx - self.p0[0], yy - self.p0[1]
            s = np.clip(rx * u[0] + ry * u[1], 0, self.lc)
            d = np.hypot(rx - s * u[0], ry - s * u[1])
            n = -rx * u[1] + ry * u[0]
            return d, np.full_like(d, self.theta0), s, n

        rho = 1.0 / abs(self.kappa)                      # radius of curvature
        sgn = np.sign(self.kappa)
        c = self.p0 + rho * sgn * np.array([-np.sin(self.theta0), np.cos(self.theta0)])
        vx, vy = xx - c[0], yy - c[1]
        r = np.hypot(vx, vy)
        theta_p = np.arctan2(vy, vx) + sgn * np.pi / 2   # tangent angle at the radial projection
        th_mid = self.theta0 + 0.5 * self.kappa * self.lc
        delta = np.angle(np.exp(1j * (theta_p - th_mid)))  # wrapped to (-pi, pi]
        s = 0.5 * self.lc + delta / self.kappa
        on_arc = (s >= 0) & (s <= self.lc)

        e0, e1 = self.point(0.0), self.point(self.lc)
        d0, d1 = np.hypot(xx - e0[0], yy - e0[1]), np.hypot(xx - e1[0], yy - e1[1])
        d = np.where(on_arc, np.abs(r - rho), np.minimum(d0, d1))
        theta_end = np.where(d0 < d1, self.theta0, self.theta0 + self.kappa * self.lc)
        theta = np.where(on_arc, theta_p, theta_end)
        return d, theta, np.clip(s, 0, self.lc), sgn * (r - rho)

    def distance_field(self, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d, theta, _, _ = self.fields(xx, yy)
        return d, theta


@dataclass
class Look:
    """Appearance parameters shared by the wires of one image (domain randomisation)."""

    edge_gain: float      # secondary-electron edge brightening
    facet_amp: float      # brightness difference between longitudinal facets
    n_facets: int
    outline: float        # dark outline at the wire edge
    stripe_amp: float     # transverse striations
    stripe_period: float  # in units of the wire width
    halo: tuple[float, float]  # (strength, decay length) of the shadow cast on what lies below


# --------------------------------------------------------------------------- sampling
def _random_wire(rng, size, median, sigma, w_mean, w_cv, max_bend, margin=0.15) -> Wire:
    width = float(np.clip(rng.normal(w_mean, w_cv * w_mean), 2.0, 40.0))
    length = float(rng.lognormal(np.log(median), sigma))
    lc = max(length - width, 2.0)
    bend = np.radians(rng.uniform(-max_bend, max_bend))
    theta0 = rng.uniform(0, 2 * np.pi)
    centre = rng.uniform(-margin * size, (1 + margin) * size, 2)
    tmp = Wire(np.zeros(2), theta0, bend / lc, lc, width)
    return Wire(centre - tmp.point(lc / 2), theta0, bend / lc, lc, width)


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


class _Canvas:
    """Image plus label maps that wires are drawn into (bottom to top)."""

    def __init__(self, img: np.ndarray):
        size = img.shape[0]
        self.img = img
        self.count = np.zeros((size, size), np.uint8)     # labelled wires covering a pixel
        self.centerline = np.zeros((size, size), bool)
        self.angle = np.zeros((size, size), np.float32)
        self.width = np.zeros((size, size), np.float32)   # width of the top-most labelled wire
        self.instance = np.zeros((size, size), np.int32)

    def draw(self, rng, wire: Wire, idx: int, level: float, look: Look) -> None:
        size = self.img.shape[0]
        r = wire.width / 2
        pad = r + 3 * look.halo[1] + 2
        x0, y0, x1, y1 = wire.bbox(pad)
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(np.ceil(x1)), size), min(int(np.ceil(y1)), size)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        d, theta, s, n = wire.fields(xx, yy)
        win = (slice(y0, y1), slice(x0, x1))

        # shadow cast on everything below, then the wire itself on top
        self.img[win] *= 1 - look.halo[0] * np.exp(-np.maximum(d - r, 0) / look.halo[1]) * (d > r)
        soft = np.clip(r - d + 0.5, 0, 1)                  # anti-aliased coverage, exact width 2r
        u = np.clip(d / r, 0, 0.97)
        prof = 1 + look.edge_gain * (1 / np.sqrt(1 - u**2) - 1)            # SE edge effect
        if look.facet_amp:                                 # longitudinal facets (pentagonal wire)
            facet = np.floor((n / wire.width + 0.5) * look.n_facets)
            prof *= 1 + look.facet_amp * np.cos(facet * 2.4 + rng.uniform(0, 6.3))
        if look.outline:                                   # dark outline at the edge
            prof *= 1 - look.outline * np.exp(-((r - d) / 0.9) ** 2)
        if look.stripe_amp:                                # transverse striations
            lam = max(2.0, look.stripe_period * wire.width)
            prof *= 1 + look.stripe_amp * np.sin(2 * np.pi * s / lam + rng.uniform(0, 6.3))
        level = level * rng.uniform(0.9, 1.1)
        self.img[win] = self.img[win] * (1 - soft) + np.clip(level * prof, 0, 1) * soft
        if not wire.labelled:
            return
        inside = soft >= 0.5
        self.count[win] += inside
        self.angle[win] = np.where(inside, theta, self.angle[win])
        self.width[win] = np.where(inside, wire.width, self.width[win])
        self.centerline[win] |= d <= 1.0
        self.instance[win] = np.where(d <= 1.5, idx + 1, self.instance[win])


def _acquire(rng, img: np.ndarray, cfg: SyntheticConfig, low_noise: bool = False) -> np.ndarray:
    """Probe blur, shot noise, read noise, scan lines, gamma, optional JPEG -> uint8."""
    size = img.shape[0]
    img = gaussian_filter(img, rng.uniform(*cfg.blur_sigma))
    # in-lens images of dense meshes are usually recorded with long dwell times (low noise)
    dose = np.exp(rng.uniform(np.log(30), np.log(1500))) if low_noise else rng.uniform(*cfg.photons)
    img = rng.poisson(np.clip(img, 0, None) * dose) / dose
    img += rng.normal(0, rng.uniform(0.005, 0.03), img.shape)
    img += rng.normal(0, rng.uniform(0.0, 0.02), (size, 1))      # scan-line offsets
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    img8 = (normalize(img) * 255).astype(np.uint8)
    if rng.random() < cfg.jpeg_prob:
        ok, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        img8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img8


# --------------------------------------------------------------------------- styles
def _sparse(rng, cfg: SyntheticConfig, size: int):
    """Wires on a flat substrate (style used for sparse meshes)."""
    bg = rng.uniform(0.05, 0.45)
    level = rng.uniform(bg + 0.25, 0.95)
    look = Look(edge_gain=rng.uniform(*cfg.edge_gain), facet_amp=rng.uniform(0, 0.08) * (rng.random() < 0.3),
                n_facets=int(rng.integers(2, 4)), outline=rng.uniform(0, 0.2) * (rng.random() < 0.3),
                stripe_amp=rng.uniform(0, 0.1) * (rng.random() < 0.3), stripe_period=rng.uniform(0.15, 0.4),
                halo=(rng.uniform(0.0, 0.25), rng.uniform(1.0, 4.0)))
    canvas = _Canvas(_substrate(rng, size, bg, rng.random() < cfg.pore_prob))
    if rng.random() < cfg.particle_prob:
        _draw_particles(rng, canvas.img, size, level, look.edge_gain)

    n = int(rng.integers(*cfg.n_wires) * (size / cfg.tile_size) ** 2)
    median, sigma = rng.uniform(*cfg.length_median), rng.uniform(*cfg.length_sigma)
    w_mean, w_cv = rng.uniform(*cfg.width_mean), rng.uniform(*cfg.width_cv)
    wires = [_random_wire(rng, size, median, sigma, w_mean, w_cv, cfg.max_bend_deg) for _ in range(n)]
    for i, w in enumerate(wires):
        canvas.draw(rng, w, i, level, look)
    return canvas, wires


def _dense3d(rng, cfg: SyntheticConfig, size: int):
    """Thick chaotic mesh imaged with an in-lens detector: depth layers, defocus, shading."""
    canvas = _Canvas(_substrate(rng, size, rng.uniform(0.02, 0.2), False))
    top = rng.uniform(0.55, 0.95)
    look = Look(edge_gain=rng.uniform(0.0, 0.35), facet_amp=rng.uniform(0.0, 0.12),
                n_facets=int(rng.integers(2, 4)), outline=rng.uniform(0.0, 0.4),
                stripe_amp=rng.uniform(0.0, 0.08) * (rng.random() < 0.7),
                stripe_period=rng.uniform(0.08, 0.3), halo=(rng.uniform(0.1, 0.45), rng.uniform(1.0, 5.0)))
    n_layers = int(rng.integers(*cfg.dense_layers))
    texture = rng.random() < 0.5                       # unlabelled, very blurred bottom layer
    median = rng.uniform(*cfg.dense_length_median) * size / cfg.tile_size
    w_mean, w_cv = rng.uniform(*cfg.dense_width_mean), rng.uniform(*cfg.width_cv)
    darkest = rng.uniform(0.25, 0.6)                   # brightness of the deepest layer / top layer
    gamma = rng.uniform(0.7, 1.6)

    wires, idx = [], 0
    for layer in range(n_layers):
        depth = layer / max(n_layers - 1, 1)           # 0 = deepest, 1 = top
        level = top * (darkest + (1 - darkest) * depth ** gamma)
        n = int(rng.integers(*cfg.dense_wires_per_layer) * (size / cfg.tile_size) ** 2)
        for _ in range(n):
            w = _random_wire(rng, size, median, 0.4, w_mean, w_cv, cfg.max_bend_deg * 0.5, 0.4)
            w.labelled = not (texture and layer == 0)
            canvas.draw(rng, w, idx, level, look)
            wires.append(w)
            idx += 1
        if layer < n_layers - 1:                       # everything below the next layer gets defocused
            canvas.img = gaussian_filter(canvas.img, rng.uniform(0.4, 1.6))
    return canvas, wires


# --------------------------------------------------------------------------- public
def render_sample(rng: np.random.Generator, cfg: SyntheticConfig, size: int | None = None,
                  style: str | None = None) -> dict:
    """Render one synthetic SEM image with labels.

    Returns a dict with
        image       uint8 (H, W)   normalised SEM-like image
        mask        uint8 (H, W)   1 = (labelled) wire
        centerline  uint8 (H, W)   1 = within 1 px of a wire centre-line
        angle       uint8 (H, W)   orientation in degrees [0, 180), 255 = ignore
        width       uint8 (H, W)   width of the top-most wire in 1/4 px (0 = background)
        instance    int32 (H, W)   centre-line pixels labelled with wire index + 1
        wires       list[Wire]     ground-truth geometry
    """
    size = size or cfg.tile_size
    style = style or ("dense3d" if rng.random() < cfg.dense_prob else "sparse")
    canvas, wires = (_dense3d if style == "dense3d" else _sparse)(rng, cfg, size)
    img8 = _acquire(rng, canvas.img, cfg, low_noise=style == "dense3d")

    single = canvas.count == 1                     # orientation is ambiguous at crossings
    ang = np.round(np.degrees(canvas.angle) % 180.0) % 180
    return {"image": img8, "mask": (canvas.count > 0).astype(np.uint8),
            "centerline": (canvas.centerline & (canvas.count > 0)).astype(np.uint8),
            "angle": np.where(single, ang, IGNORE_ANGLE).astype(np.uint8),
            "width": np.clip(np.round(canvas.width * WIDTH_SCALE), 0, 255).astype(np.uint8),
            "instance": canvas.instance, "wires": wires, "style": style}


def wire_table(wires: list[Wire], size: int) -> list[dict]:
    """Ground-truth table of the labelled wires; `inside` = full length visible."""
    rows = []
    for i, w in enumerate(wires):
        if not w.labelled:
            continue
        pts = w.point(np.linspace(0, w.lc, 32))
        r = w.width / 2
        inside = bool((pts.min() >= r) and (pts.max() <= size - 1 - r))
        rows.append({"id": i + 1, "length_px": w.length, "width_px": w.width,
                     "angle_deg": w.mean_angle_deg, "inside": inside})
    return rows
