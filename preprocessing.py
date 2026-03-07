"""
preprocessing.py — Baseline correction, smoothing and normalisation for
Raman spectra.

Public API
----------
fast_baseline(y)
als_baseline(y)
preprocess_spectrum(s, grid, use_als, norm)  → (proc, d2)
preprocess_map_pixels(pixels, grid, ...)     → (proc, d2)
"""

from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Baseline estimators
# ---------------------------------------------------------------------------

def fast_baseline(y: np.ndarray, degree: int = 6) -> np.ndarray:
    """Iteratively-reweighted polynomial baseline."""
    x = np.arange(len(y))
    w = np.ones(len(y))
    for _ in range(7):
        c   = np.polyfit(x, y, degree, w=w)
        bl  = np.polyval(c, x)
        res = y - bl
        thr = np.percentile(res, 15)
        rng = max(res.max() - thr, 1e-10)
        w   = np.where(res <= thr, 1.0,
                       np.clip(1.0 - (res - thr) / rng, 0.05, 1.0))
    return bl


def als_baseline(
    y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10
) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens)."""
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L))
        z = spsolve(W + H, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


# ---------------------------------------------------------------------------
# Single-spectrum pipeline
# ---------------------------------------------------------------------------

def preprocess_spectrum(
    s: np.ndarray,
    grid: np.ndarray,
    use_als: bool = False,
    norm: str = "snv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full single-spectrum preprocessing pipeline.

    Returns
    -------
    (processed_spectrum, second_derivative)
    """
    s  = s.copy()
    bl = als_baseline(s) if use_als else fast_baseline(s)
    s  = np.clip(s - bl, 0, None)
    s  = savgol_filter(s, window_length=11, polyorder=3)
    d2 = savgol_filter(s, window_length=11, polyorder=3, deriv=2)

    if norm == "snv":
        mu, sigma = s.mean(), s.std()
        if sigma > 1e-10:
            s = (s - mu) / sigma
    elif norm == "peak_phe":
        mask = (grid >= 988) & (grid <= 1018)
        ref  = s[mask].max() if mask.sum() > 0 else 0
        if ref > 1e-3:
            s = s / ref
        else:
            mu, sigma = s.mean(), s.std()
            if sigma > 1e-10:
                s = (s - mu) / sigma
    elif norm == "area":
        a = np.trapz(np.abs(s))
        if a > 1e-10:
            s /= a

    mu2, sigma2 = d2.mean(), d2.std()
    if sigma2 > 1e-10:
        d2 = (d2 - mu2) / sigma2

    return s, d2


# ---------------------------------------------------------------------------
# Batch (map) preprocessing
# ---------------------------------------------------------------------------

def preprocess_map_pixels(
    pixels: np.ndarray,
    grid: np.ndarray,
    use_als: bool = False,
    norm: str = "snv",
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a 2-D array of raw pixel spectra in parallel."""
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(preprocess_spectrum)(px, grid, use_als, norm)
        for px in pixels
    )
    proc = np.array([r[0] for r in results])
    d2   = np.array([r[1] for r in results])
    return proc, d2
