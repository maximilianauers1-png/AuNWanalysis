"""Statistics of nanowire populations.

Descriptive statistics
    mean, SD, SEM = SD / sqrt(n), 95 % CI of the mean from Student's t,
    median with a bootstrap 95 % CI, quartiles, coefficient of variation.

Distribution model
    Nanowire lengths (and often widths) are right-skewed and are commonly
    described by a log-normal distribution: ln X ~ N(mu, sigma^2). The MLE is
    mu = mean(ln x), sigma = SD(ln x); median = e^mu, mean = e^(mu + sigma^2 / 2).
    Normal and log-normal fits are compared by AIC = 2k - 2 ln L (k = 2 for both),
    lower is better.

Edge (frame) correction for lengths
    A wire of bounding box (dx, dy) lies completely inside a W x H image with
    probability ~ (W - dx)(H - dy) / (W H) (for uniformly placed wires). Long
    wires are therefore under-represented among the complete wires. Weighting
    each complete wire by the inverse of this probability (Miles-Lantuejoul
    correction) removes that bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def weighted_quantile(x: np.ndarray, q, w: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, float)
    if w is None:
        return np.quantile(x, q)
    order = np.argsort(x)
    x, w = x[order], np.asarray(w, float)[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cdf, x)


def miles_lantuejoul_weights(df: pd.DataFrame, width: float, height: float) -> np.ndarray:
    """Inverse inclusion probability of complete wires inside a W x H frame (pixels)."""
    p = ((width - df["extent_x_px"]).clip(lower=1) * (height - df["extent_y_px"]).clip(lower=1)
         / (width * height))
    return (1.0 / p).to_numpy()


def edge_weights(complete: pd.DataFrame, frames: dict[str, tuple[int, int]]) -> np.ndarray:
    """Miles-Lantuejoul weights aligned with the rows of `complete` (any image order)."""
    w = np.ones(len(complete))
    for name, (fw, fh) in frames.items():
        sel = (complete["image"] == name).to_numpy()
        if sel.any():
            w[sel] = miles_lantuejoul_weights(complete[sel], fw, fh)
    return w


def describe(x, weights=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Descriptive statistics of a 1-D sample (optionally weighted)."""
    x = np.asarray(x, float)
    keep = np.isfinite(x) & (x > 0)
    x = x[keep]
    w = None if weights is None else np.asarray(weights, float)[keep]
    n = len(x)
    if n == 0:
        return {"n": 0}
    if w is None:
        mean, sd = x.mean(), x.std(ddof=1) if n > 1 else 0.0
        n_eff = n
    else:
        mean = np.average(x, weights=w)
        sd = np.sqrt(np.average((x - mean) ** 2, weights=w) * n / max(n - 1, 1))
        n_eff = w.sum() ** 2 / (w**2).sum()                  # Kish effective sample size
    sem = sd / np.sqrt(n_eff)
    tcrit = st.t.ppf(0.975, max(n_eff - 1, 1))
    q1, med, q3 = weighted_quantile(x, [0.25, 0.5, 0.75], w)

    rng = np.random.default_rng(seed)                        # bootstrap CI of the median
    if n > 1:
        idx = rng.integers(0, n, (n_boot, n))
        meds = (np.median(x[idx], axis=1) if w is None else
                np.array([weighted_quantile(x[i], 0.5, w[i]) for i in idx[:400]]))
        med_lo, med_hi = np.percentile(meds, [2.5, 97.5])
    else:
        med_lo = med_hi = med

    lx = np.log(x)
    mu, sigma = (np.average(lx, weights=w), np.sqrt(np.average((lx - np.average(lx, weights=w)) ** 2,
                                                             weights=w))) if n > 1 else (lx[0], 0.0)
    out = {"n": n, "mean": mean, "sd": sd, "sem": sem,
           "ci95_low": mean - tcrit * sem, "ci95_high": mean + tcrit * sem,
           "median": med, "median_ci_low": med_lo, "median_ci_high": med_hi,
           "q1": q1, "q3": q3, "iqr": q3 - q1, "min": x.min(), "max": x.max(),
           "cv_percent": 100 * sd / mean if mean else np.nan,
           "lognorm_mu": mu, "lognorm_sigma": sigma, "geo_mean": np.exp(mu),
           "geo_sd": np.exp(sigma)}
    if n > 2 and sd > 0 and sigma > 0:
        ll_norm = st.norm.logpdf(x, mean, sd)
        ll_logn = st.lognorm.logpdf(x, sigma, scale=np.exp(mu))
        aic_n = 4 - 2 * (ll_norm.sum() if w is None else (w * ll_norm).sum() * n / w.sum())
        aic_l = 4 - 2 * (ll_logn.sum() if w is None else (w * ll_logn).sum() * n / w.sum())
        out.update(aic_normal=aic_n, aic_lognormal=aic_l,
                   best_model="log-normal" if aic_l < aic_n else "normal")
    return out


def length_width_frame(df: pd.DataFrame, unit: str) -> tuple[str, str, float]:
    """Column suffix and factor for the chosen unit ('nm', 'µm' or 'px')."""
    if unit == "px":
        return "_px", "px", 1.0
    if unit == "µm":
        return "_nm", "µm", 1e-3
    return "_nm", "nm", 1.0


def summary_table(wires: pd.DataFrame, unit: str = "nm", edge_correction: bool = True,
                  frames: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Rows: Length (complete wires), Width (all wires), Aspect ratio (complete wires).

    `frames` maps image name -> (width, height) in px for the edge correction.
    """
    if wires.empty:
        return pd.DataFrame()
    if unit != "px" and "length_nm" not in wires:
        unit = "px"
    suffix, label, f = length_width_frame(wires, unit)
    complete = wires[wires["complete"]]
    w = edge_weights(complete, frames) if edge_correction and frames and len(complete) else None
    rows = {
        f"Length [{label}]" + (" (edge-corrected)" if w is not None else ""):
            describe(complete[f"length{suffix}"] * f, w),
        f"Width [{label}]": describe(wires[f"width{suffix}"] * f),
        "Aspect ratio L/W": describe(complete["aspect_ratio"], w),
    }
    return pd.DataFrame(rows).T


def compare_samples(a, b, label_a: str = "automatic", label_b: str = "manual") -> dict:
    """Do two measurement sets describe the same population?

    Kolmogorov-Smirnov (shape), Mann-Whitney U (location, non-parametric),
    Welch's t (means, unequal variances), plus relative differences.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ks = st.ks_2samp(a, b)
    mw = st.mannwhitneyu(a, b)
    wt = st.ttest_ind(a, b, equal_var=False)
    return {f"n_{label_a}": len(a), f"n_{label_b}": len(b),
            f"mean_{label_a}": a.mean(), f"mean_{label_b}": b.mean(),
            "relative_difference_mean_%": 100 * (a.mean() - b.mean()) / b.mean(),
            f"median_{label_a}": np.median(a), f"median_{label_b}": np.median(b),
            "relative_difference_median_%": 100 * (np.median(a) - np.median(b)) / np.median(b),
            "KS_statistic": ks.statistic, "KS_p": ks.pvalue,
            "MannWhitney_p": mw.pvalue, "Welch_t_p": wt.pvalue}


def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 60) -> int:
    """Bin count from the Freedman-Diaconis rule h = 2 IQR n^(-1/3)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return max(1, len(x))
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return 10
    h = 2 * iqr * len(x) ** (-1 / 3)
    return int(np.clip(np.ceil((x.max() - x.min()) / h), 5, max_bins))
