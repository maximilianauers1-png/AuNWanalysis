"""From probability maps to individual nanowires: graph reconstruction and measurement.

Pipeline (deterministic, few parameters)

1.  Skeleton.   Threshold the centreline map (inside the wire mask) and thin it
                to a 1-px wide, 8-connected skeleton.
2.  Junctions.  Classify skeleton pixels with the *crossing number*
                    CN(p) = 1/2 * sum_{k=1..8} |P_{k+1} - P_k|      (P_9 = P_1),
                the number of 0->1 transitions around p's cyclic 8-neighbourhood
                (CN = 1 end, CN = 2 line, CN >= 3 junction). Compact junction
                clusters, where each pixel has CN = 2, are caught by a neighbour
                count >= 4. Junction pixels are grown into zones of radius ~ w/2.
3.  Branches.   Skeleton minus junction zones = branches (simple pixel chains),
                ordered by walking along them.
4.  Pairing.    In every junction, branch ends are paired if one is the straight
                continuation of the other. With outward unit vectors u_a, u_b and
                tip separation r:
                    deflection  delta  = arccos(-u_a . u_b)           < delta_max
                    lateral     l      = max(|u_a x r|, |u_b x r|)    < tol(|r|)
                    cost               = delta / delta_max + l / tol
                Pairs are accepted greedily by increasing cost. u comes from the
                network's orientation head where it is confident. Facing free tips
                separated by a short gap that lies inside the wire mask are joined
                the same way (the lower wire at a crossing is hidden).
5.  Wires.      Union-find over the accepted pairs: connected branches = one wire.
6.  Length.     Sum of branch lengths (polyline resampled every 4 px, which removes
                the pixel-staircase bias) + gaps bridged inside junctions + w/2 per
                end (the centreline stops at the centre of the hemispherical cap).
7.  Width.      Primary: median of the network's width map along the wire, away
                from junctions (works in dense meshes where wires touch). Cross-
                check: the mask probability p(t) is sampled on the local normal;
                the two sub-pixel 0.5 crossings give w_i = t_right - t_left, and
                width_profile = median(w_i) (exact for isolated wires).
8.  End types.  "tip" (free end whose mask ends too), "t_end" (ends on another
                wire), "unresolved" (unpaired in an ambiguous junction, or the
                centre-line stops although the mask continues) or "border". A wire is
                *complete* if both ends are free tips and it does not touch the
                border: only then is its length fully observed.
9.  Reliability. The areal coverage of the mask decides how many wire ends are
                visible. The stereological estimate <L> = 2 * total length / ends
                is an independent cross-check of the mean length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects, skeletonize

from .config import AnalysisConfig

# 8-neighbourhood in cyclic order N, NE, E, SE, S, SW, W, NW as (dy, dx)
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
EIGHT = np.ones((3, 3), bool)
BORDER_MARGIN = 3   # px; closer than this to the frame counts as "touching the border"


@dataclass
class Branch:
    path: np.ndarray                 # ordered (y, x) pixel coordinates
    length: float
    junction: list[int] = field(default_factory=lambda: [0, 0])   # junction id at start / end (0 = none)


@dataclass
class Graph:
    """Skeleton graph: branches connected through junction zones."""

    branches: list[Branch]
    jzone: np.ndarray                # bool map of junction zones
    jcentre: np.ndarray              # (n_j + 1, 2) centres, row index = junction id
    j_at_border: np.ndarray          # (n_j + 1,) junction zone touches the frame
    skeleton: np.ndarray
    width_px: float                  # global width estimate
    r_j: int                         # junction zone radius
    edt: np.ndarray                  # distance transform of the wire mask


# =========================================================================== small helpers
def crossing_number(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crossing number and neighbour count of every skeleton pixel."""
    p = np.pad(skel.astype(np.int8), 1)
    h, w = skel.shape
    ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]
    cn = sum(np.abs(ring[(k + 1) % 8] - ring[k]) for k in range(8)) // 2
    return cn * skel, sum(ring) * skel


def order_path(pix: np.ndarray) -> np.ndarray:
    """Order the pixels of a chain-like branch by walking from one end to the other."""
    n = len(pix)
    if n <= 2:
        return pix
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(pix)}
    nbrs = [[index[(y + dy, x + dx)] for dy, dx in RING if (y + dy, x + dx) in index]
            for y, x in map(tuple, pix.tolist())]
    ends = [i for i, v in enumerate(nbrs) if len(v) == 1]
    start = ends[0] if ends else 0
    seen = np.zeros(n, bool)
    seen[start] = True

    def walk(cur: int) -> list[int]:
        out = []
        while True:
            cand = [j for j in nbrs[cur] if not seen[j]]
            if not cand:
                return out
            # prefer 4-neighbours so that no pixel of a staircase corner is skipped
            cand.sort(key=lambda j: abs(pix[j][0] - pix[cur][0]) + abs(pix[j][1] - pix[cur][1]))
            cur = cand[0]
            seen[cur] = True
            out.append(cur)

    # Walk both ways: the start is not always a true end (an end pixel with two
    # touching neighbours has no degree-1 pixel at all).
    forward = walk(start)
    backward = walk(start)
    return pix[backward[::-1] + [start] + forward]


def polyline_length(path: np.ndarray, step: int = 4) -> float:
    """Length of a pixel chain measured on vertices every `step` pixels.

    Summing 1 and sqrt(2) pixel steps over-estimates straight lines at generic
    angles by up to ~8 %. Resampling every few pixels averages the staircase
    out (residual bias < 1 %).
    """
    if len(path) < 2:
        return 0.0
    idx = np.arange(0, len(path), step)
    if idx[-1] != len(path) - 1:
        idx = np.append(idx, len(path) - 1)
    return float(np.linalg.norm(np.diff(path[idx].astype(float), axis=0), axis=1).sum())


def end_direction(path: np.ndarray, at_end: bool, m: int,
                  orient: np.ndarray | None) -> np.ndarray:
    """Outward unit vector (dx, dy) of a branch end (pointing into the junction)."""
    seg = path[-m:] if at_end else path[:m][::-1]          # ordered towards the tip
    v = (seg[-1] - seg[0]).astype(float)[::-1]            # (y, x) -> (dx, dy)
    norm = np.linalg.norm(v)
    u = v / norm if norm > 0 else np.array([1.0, 0.0])
    if orient is not None:
        o = orient[seg[:, 0], seg[:, 1]]
        o = o[np.hypot(o[:, 0], o[:, 1]) > 0.5]            # confident pixels only
        if len(o) >= 3:
            c, s = o.mean(axis=0)                          # axial mean = mean double-angle vector
            th = 0.5 * np.arctan2(s, c)
            e = np.array([np.cos(th), np.sin(th)])
            u = e if e @ u >= 0 else -e                    # the head gives the axis, geometry the sign
    return u


def near_border(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pts = np.atleast_2d(pts)
    h, w = shape
    m = BORDER_MARGIN
    return (pts[:, 0] < m) | (pts[:, 0] >= h - m) | (pts[:, 1] < m) | (pts[:, 1] >= w - m)


def _find(parent: list[int], i: int) -> int:
    """Union-find root with path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def profile_widths(prob: np.ndarray, pts: np.ndarray, normals: np.ndarray,
                   half_range: float, step: float = 0.25) -> np.ndarray:
    """Sub-pixel width at each point from the 0.5 crossings of the mask probability.

    pts: (N, 2) (y, x); normals: (N, 2) unit (ny, nx). NaN where no closed
    profile was found (e.g. the profile runs into a neighbouring wire).
    """
    t = np.arange(-half_range, half_range + step / 2, step)
    yy = pts[:, :1] + t[None] * normals[:, :1]
    xx = pts[:, 1:] + t[None] * normals[:, 1:]
    p = ndi.map_coordinates(prob, [yy, xx], order=1, mode="constant", cval=0.0)
    c = len(t) // 2

    def edge(side: np.ndarray) -> np.ndarray:
        """Distance from the centre to the first sample with p <= 0.5 (interpolated)."""
        below = side <= 0.5
        k = np.argmax(below, axis=1)
        ok = below.any(axis=1) & (k > 0)
        k = np.where(ok, k, 1)
        rows = np.arange(len(side))
        p_in, p_out = side[rows, k - 1], side[rows, k]
        frac = (p_in - 0.5) / np.maximum(p_in - p_out, 1e-6)
        return np.where(ok, (k - 1 + frac) * step, np.nan)

    w = edge(p[:, c:]) + edge(p[:, c::-1])
    w[p[:, c] <= 0.5] = np.nan
    return w


def tangents(path: np.ndarray, k: int = 3) -> np.ndarray:
    """Unit tangents (dy, dx) by central differences over +-k pixels."""
    n = len(path)
    t = (path[np.clip(np.arange(n) + k, 0, n - 1)]
         - path[np.clip(np.arange(n) - k, 0, n - 1)]).astype(float)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


# =========================================================================== stages
def build_graph(prob: np.ndarray, centerline: np.ndarray, cfg: AnalysisConfig,
                width_map: np.ndarray | None = None) -> Graph | None:
    """Stages 1-3: skeleton, junction zones, ordered branches."""
    fg = prob > cfg.mask_threshold
    cl = remove_small_objects((centerline > cfg.centerline_threshold) & fg, max_size=8)
    skel = skeletonize(cl)
    if not skel.any():
        return None

    # Global width scale: from the network's width map if available. The fallback,
    # the distance transform of the mask (w ~ 2d - 1 on the axis), is only valid for
    # sparse meshes: in a dense mesh the mask is one connected area.
    edt = ndi.distance_transform_edt(fg)
    if width_map is not None:
        w_glob = float(max(2.0, np.median(width_map[skel])))
    else:
        w_glob = float(max(2.0, np.median(2 * edt[skel] - 1)))
    r_j = int(np.ceil(0.5 * w_glob)) + 1

    cn, nb = crossing_number(skel)
    is_junction = (cn >= 3) | (nb >= 4)
    jzone = ndi.binary_dilation(is_junction, structure=disk(r_j))
    jlab, n_j = ndi.label(jzone, structure=EIGHT)
    centres = ndi.center_of_mass(is_junction, jlab, range(1, n_j + 1))
    jcentre = np.vstack([np.zeros((1, 2)), np.array(centres).reshape(-1, 2)])

    m = BORDER_MARGIN
    j_at_border = np.zeros(n_j + 1, bool)
    j_at_border[np.unique(np.concatenate([jlab[:m].ravel(), jlab[-m:].ravel(),
                                          jlab[:, :m].ravel(), jlab[:, -m:].ravel()]))] = True
    j_at_border[0] = False

    # branches: connected pieces of the skeleton outside junction zones
    blab, n_b = ndi.label(skel & ~jzone, structure=EIGHT)
    flat = np.flatnonzero(blab)
    order = np.argsort(blab.flat[flat], kind="stable")
    groups = np.split(flat[order], np.cumsum(np.bincount(blab.flat[flat])[1:])[:-1]) if n_b else []
    jlab_near = ndi.grey_dilation(jlab, size=(3, 3))      # junction id seen from a branch tip

    branches = []
    for g in groups:
        path = order_path(np.column_stack(np.unravel_index(g, skel.shape)))
        b = Branch(path, polyline_length(path),
                   [int(jlab_near[tuple(path[0])]), int(jlab_near[tuple(path[-1])])])
        is_spur = min(b.junction) == 0 and max(b.junction) > 0 and b.length < 0.75 * w_glob
        if len(path) >= 2 and not is_spur:                 # tiny dead-end spurs are skeleton noise
            branches.append(b)
    return Graph(branches, jzone, jcentre, j_at_border, skel, w_glob, r_j, edt)


def _pair_cost(ea, eb, cos_max: float, base_tol: float, max_deg: float) -> float | None:
    """Cost of joining two branch ends into one straight wire (None = not allowed)."""
    (ba, _, ua, pa), (bb, _, ub, pb) = ea, eb
    if ba == bb:
        return None
    straightness = -ua @ ub                                 # cos(deflection)
    r = pb - pa
    lateral = max(abs(ua[0] * r[1] - ua[1] * r[0]), abs(ub[0] * r[1] - ub[1] * r[0]))
    tol = base_tol + 0.1 * np.linalg.norm(r)                # direction error grows with the gap
    if straightness <= cos_max or lateral >= tol:
        return None
    return np.degrees(np.arccos(min(straightness, 1.0))) / max_deg + lateral / tol


def _greedy_link(cands, parent, linked, gap_len) -> None:
    """Accept candidate pairs by increasing cost; each end once, no cycles."""
    for _, (ba, sa, _, pa), (bb, sb, _, pb) in sorted(cands, key=lambda c: c[0]):
        if (ba, sa) in linked or (bb, sb) in linked or _find(parent, ba) == _find(parent, bb):
            continue
        linked |= {(ba, sa), (bb, sb)}
        parent[_find(parent, ba)] = _find(parent, bb)
        gap_len[ba] += np.linalg.norm(pb - pa)


def _inside_mask(prob: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> bool:
    """True if the straight segment pa -> pb (x, y) runs entirely inside the wire mask."""
    n = int(np.ceil(np.linalg.norm(pb - pa))) + 2
    xs, ys = np.linspace(pa[0], pb[0], n), np.linspace(pa[1], pb[1], n)
    return bool(ndi.map_coordinates(prob, [ys, xs], order=1).min() > 0.5)


def pair_branches(g: Graph, prob: np.ndarray, orient: np.ndarray | None, cfg: AnalysisConfig):
    """Stages 4-5: link branch ends into wires; union-find over the links.

    (a) Junctions: ends meeting in a junction zone are paired if collinear.
    (b) Gap bridging: two *free* tips facing each other across a short gap are
        joined if the gap lies inside the wire mask. This happens where the
        lower of two crossing wires is hidden: its centre-line is interrupted
        but the wire mask is continuous.

    Returns (parent, linked, gap_len, unpaired_at):
        parent       union-find array over branch indices
        linked       set of (branch, side) ends that were paired
        gap_len      bridged distance booked per branch
        unpaired_at  number of unpaired ends per junction id
    """
    m = max(8, int(round(3.0 * g.width_px)))
    ends_at: dict[int, list] = {}
    free_tips = []
    for bi, b in enumerate(g.branches):
        for side in (0, 1):
            tip = b.path[-1] if side else b.path[0]
            u = end_direction(b.path, bool(side), min(m, len(b.path)), orient)
            end = (bi, side, u, tip[::-1].astype(float))
            j = b.junction[side]
            if j:
                ends_at.setdefault(j, []).append(end)
            elif not near_border(tip, prob.shape)[0]:
                free_tips.append(end)

    parent = list(range(len(g.branches)))
    linked: set[tuple[int, int]] = set()
    gap_len = np.zeros(len(g.branches))
    args = (np.cos(np.radians(cfg.max_pair_angle_deg)), max(2.0, 0.5 * g.width_px),
            cfg.max_pair_angle_deg)

    # (a) pairing inside junctions
    for ends in ends_at.values():
        cands = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                c = _pair_cost(ends[a], ends[b], *args)
                if c is not None:
                    cands.append((c, ends[a], ends[b]))
        _greedy_link(cands, parent, linked, gap_len)

    # (b) bridging gaps between facing free tips
    if len(free_tips) > 1:
        from scipy.spatial import cKDTree
        pts = np.array([e[3] for e in free_tips])
        max_gap = 2.5 * g.width_px + 3
        cands = []
        for a, b in cKDTree(pts).query_pairs(max_gap):
            ea, eb = free_tips[a], free_tips[b]
            r = eb[3] - ea[3]
            if ea[2] @ r <= 0:                              # tips must face each other
                continue
            c = _pair_cost(ea, eb, *args)
            if c is not None and _inside_mask(prob, ea[3], eb[3]):
                cands.append((c, ea, eb))
        _greedy_link(cands, parent, linked, gap_len)

    unpaired_at = np.zeros(len(g.jcentre), int)
    for j, ends in ends_at.items():
        unpaired_at[j] = sum((bi, side) not in linked for bi, side, _, _ in ends)
    return parent, linked, gap_len, unpaired_at


def head_width(paths: list[np.ndarray], width_map: np.ndarray, near_junction: np.ndarray) -> float:
    """Median of the network's width map along the wire, away from junctions.

    Away from crossings the top-most wire at a centre-line pixel is the wire itself,
    so the width map there is this wire's width - even if neighbours touch it.
    """
    allp = np.concatenate(paths)
    keep = ~near_junction[allp[:, 0], allp[:, 1]]
    sel = allp[keep] if keep.sum() >= 3 else allp
    return float(np.median(width_map[sel[:, 0], sel[:, 1]]))


def measure_width(paths: list[np.ndarray], prob: np.ndarray, near_junction: np.ndarray,
                  w_glob: float, edt: np.ndarray) -> tuple[float, int]:
    """Stage 7: median sub-pixel profile width (fallback: distance transform)."""
    cap = int(np.ceil(w_glob))
    samples = []
    for p in paths:
        sel = np.arange(len(p))
        if len(p) > 2 * cap + 2:
            sel = sel[cap:len(p) - cap]                    # stay away from the round tips
        sel = sel[::2]
        sel = sel[~near_junction[p[sel, 0], p[sel, 1]]]
        if len(sel):
            t = tangents(p)[sel]
            normals = np.column_stack([t[:, 1], -t[:, 0]])
            samples.append(profile_widths(prob, p[sel].astype(float), normals, max(6.0, 3 * w_glob)))
    s = np.concatenate(samples) if samples else np.array([])
    s = s[np.isfinite(s)]
    if len(s) >= 3:
        return float(np.median(s)), len(s)
    allp = np.concatenate(paths)
    return float(max(1.0, np.median(2 * edt[allp[:, 0], allp[:, 1]] - 1))), 0


def _mask_continues(path: np.ndarray, at_end: bool, prob: np.ndarray, width: float,
                    orient: np.ndarray | None, m: int) -> bool:
    """Does the wire mask continue beyond this free centre-line end?

    A real wire end also ends in the mask: just beyond the hemispherical cap
    (w/2 + 3 px along the wire direction) the mask must be background. If it is
    still wire, the centre-line merely faded out (e.g. under a crossing wire in a
    dense mesh), and the end is not a real tip.
    """
    u = end_direction(path, at_end, min(m, len(path)), orient)          # (dx, dy), outward
    tip = path[-1] if at_end else path[0]
    probe = tip[::-1] + u * (width / 2 + 3)                              # (x, y)
    val = ndi.map_coordinates(prob, [[probe[1]], [probe[0]]], order=1, mode="constant", cval=0.0)
    return bool(val[0] > 0.5)


# =========================================================================== main entry
def analyze_maps(maps: dict[str, np.ndarray], cfg: AnalysisConfig) -> tuple[list[dict], dict]:
    """Reconstruct and measure individual wires.

    maps: "mask" and "centerline" probabilities (H, W) in [0, 1], optional
          "orientation" (H, W, 2) double-angle vectors.
    Returns (wires, info). Each wire is a dict of pixel-unit measurements plus
    its `path` (list of (y, x) arrays) for overlays.
    """
    prob = maps["mask"].astype(np.float32)
    width_map = maps.get("width")
    g = build_graph(prob, maps["centerline"], cfg, width_map)
    if g is None:
        return [], {"crossing_resolution": np.nan, "width_px_global": np.nan, "coverage": 0.0,
                    "stereological_mean_length_px": np.nan, "reliability": "no wires",
                    "junctions": np.zeros((0, 2)), "n_junctions": 0}
    parent, linked, gap_len, unpaired_at = pair_branches(g, prob, maps.get("orientation"), cfg)

    wires_of: dict[int, list[int]] = {}
    for bi in range(len(g.branches)):
        wires_of.setdefault(_find(parent, bi), []).append(bi)

    near_junction = ndi.binary_dilation(g.jzone, structure=disk(max(1, int(np.ceil(g.width_px)))))
    orient = maps.get("orientation")
    m_dir = max(8, int(round(3.0 * g.width_px)))
    results = []
    for members in wires_of.values():
        paths = [g.branches[i].path for i in members]
        width_profile, n_samples = measure_width(paths, prob, near_junction, g.width_px, g.edt)
        width = (head_width(paths, width_map, near_junction) if width_map is not None
                 else width_profile)
        length = sum(g.branches[i].length for i in members) + gap_len[members].sum()

        ends = {"tip": 0, "t_end": 0, "unresolved": 0, "border": 0}
        for i in members:
            for side in (0, 1):
                if (i, side) in linked:
                    continue
                j = g.branches[i].junction[side]
                tip = g.branches[i].path[-1 if side else 0]
                if j:   # stops inside a junction: extend to its centre (bounded)
                    length += min(np.linalg.norm(g.jcentre[j] - tip), g.r_j + width)
                    kind = "t_end" if unpaired_at[j] == 1 else "unresolved"
                    ends["border" if g.j_at_border[j] else kind] += 1
                elif near_border(tip, prob.shape)[0]:
                    ends["border"] += 1
                elif _mask_continues(g.branches[i].path, bool(side), prob, width, orient, m_dir):
                    ends["unresolved"] += 1     # centre-line stops but the wire does not
                else:
                    ends["tip"] += 1
                length += width / 2                         # hemispherical cap

        if length < cfg.min_length_px or length / max(width, 1e-6) < cfg.min_aspect_ratio:
            continue
        allp = np.concatenate(paths)
        # principal axis -> angle in the mathematical sense (y axis up), in [0, 180)
        xy = np.column_stack([allp[:, 1], -allp[:, 0]]).astype(float)
        xy -= xy.mean(0)
        evecs = np.linalg.eigh(xy.T @ xy)[1]
        angle = float(np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1])) % 180.0)
        border = ends["border"] > 0 or bool(near_border(allp, prob.shape).any())
        extent = allp.max(0) - allp.min(0) + width          # bounding box (dy, dx) incl. caps
        results.append({
            "length_px": float(length), "width_px": width,
            "width_profile_px": width_profile,
            "aspect_ratio": float(length / width), "angle_deg": angle,
            "complete": ends["tip"] == 2 and not border, "touches_border": border,
            "end_tips": ends["tip"], "end_t": ends["t_end"], "end_unresolved": ends["unresolved"],
            "n_segments": len(members), "width_samples": n_samples,
            "extent_x_px": float(extent[1]), "extent_y_px": float(extent[0]),
            "x_px": float(allp[:, 1].mean()), "y_px": float(allp[:, 0].mean()),
            "path": paths,
        })

    results.sort(key=lambda r: (r["y_px"], r["x_px"]))
    for i, r in enumerate(results, 1):
        r["wire_id"] = i

    # Quality index: fraction of crossings resolved (a failed pairing leaves ~2 unresolved ends)
    n_links = len(linked) // 2
    n_unres = sum(r["end_unresolved"] for r in results)
    # Stereological mean length (needs no wire separation): a window contains on average
    # 2*lambda*A wire ends and lambda*A*<L> wire length, so <L> = 2 * total length / ends.
    n_ends = sum(r["end_tips"] + r["end_t"] for r in results)
    total = sum(r["length_px"] for r in results)
    coverage = float((prob > cfg.mask_threshold).mean())
    info = {"crossing_resolution": n_links / (n_links + 0.5 * n_unres) if n_links else 1.0,
            "coverage": coverage, "n_ends": n_ends,
            "stereological_mean_length_px": 2 * total / n_ends if n_ends else np.nan,
            "reliability": ("good" if coverage <= cfg.coverage_reliable else
                            "caution" if coverage <= cfg.coverage_limit else "length unreliable"),
            "width_px_global": g.width_px, "junctions": g.jcentre[1:],
            "n_junctions": len(g.jcentre) - 1, "skeleton": g.skeleton}
    return results, info


# =========================================================================== evaluation
def match_to_ground_truth(wires: list[dict], instance: np.ndarray,
                          min_overlap: float = 0.5) -> dict[int, int]:
    """Map detected wire index -> ground-truth wire id by majority vote along its path.

    A detection is matched if >= `min_overlap` of its skeleton pixels lie on one
    ground-truth centreline. If several detections claim the same wire (it was
    split), only the one with the largest overlap is kept.
    """
    best: dict[int, tuple[int, int]] = {}
    for di, wire in enumerate(wires):
        pts = np.concatenate(wire["path"])
        ids = instance[pts[:, 0], pts[:, 1]]
        ids = ids[ids > 0]
        if not len(ids):
            continue
        vals, counts = np.unique(ids, return_counts=True)
        k = int(np.argmax(counts))
        if counts[k] >= min_overlap * len(pts):
            gid = int(vals[k])
            if gid not in best or counts[k] > best[gid][1]:
                best[gid] = (di, int(counts[k]))
    return {di: gid for gid, (di, _) in best.items()}
