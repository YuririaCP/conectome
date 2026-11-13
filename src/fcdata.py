"""Shared helpers for the FC-only analyses (Domhof HCP data, Schaefer-100)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_d = np.load(ROOT / "data" / "raw" / "domhof_hcp" / "schaefer100.npz")
FC_all, SC_all = _d["fc"], _d["sc"]          # (200, 4, N, N), (200, N, N)
S_, K_, N = FC_all.shape[0], FC_all.shape[1], FC_all.shape[-1]
iu = np.triu_indices(N, 1)


def prep_fc(F):
    F = (F + F.T) / 2
    np.fill_diagonal(F, 1.0)
    return F


def gsr(F):
    """Global signal regression on the covariance of z-scored signals, then renormalize."""
    c = F.mean(axis=1)
    R = F - np.outer(c, c) / c.mean()
    s = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    return R / np.outer(s, s)


def laplacian(S):
    return np.diag(S.sum(axis=1)) - S


def icc_31(X):
    """ICC(3,1) consistency; X: subjects x sessions."""
    n, k = X.shape
    gm = X.mean()
    ssr = k * ((X.mean(1) - gm) ** 2).sum()
    ssc = n * ((X.mean(0) - gm) ** 2).sum()
    sse = ((X - gm) ** 2).sum() - ssr - ssc
    msr, mse = ssr / (n - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)
