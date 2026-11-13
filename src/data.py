"""Loaders for the HCP subset distributed with neurolib (94-region AAL2 parcellation).

Source: https://github.com/neurolib-dev/neurolib/tree/master/neurolib/data/datasets/hcp
Each subject: DTI_CM.mat (streamline counts), DTI_LEN.mat (fibre lengths, mm),
TC_rsfMRI_REST1_LR.mat (BOLD time series, regions x TRs).  HCP resting-state TR = 0.72 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
HCP_DIR = ROOT / "data" / "raw" / "neurolib_hcp"
TR_HCP = 0.72


def _load_single(path: Path) -> np.ndarray:
    m = loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one variable, found {keys}")
    return np.asarray(m[keys[0]], dtype=float)


def subjects() -> list[str]:
    return sorted(p.name for p in HCP_DIR.iterdir() if p.is_dir())


def load_subject(sid: str) -> dict:
    d = HCP_DIR / sid
    sc = _load_single(d / "DTI_CM.mat")
    ln = _load_single(d / "DTI_LEN.mat")
    ts = _load_single(d / "TC_rsfMRI_REST1_LR.mat")
    if ts.shape[0] != sc.shape[0]:
        ts = ts.T
    return {"sc": sc, "len": ln, "ts": ts}


def normalize_sc(sc: np.ndarray, max_value: float = 0.2) -> np.ndarray:
    """Symmetrize, zero the diagonal and scale so the largest weight equals max_value
    (Deco et al. 2017 convention)."""
    C = (sc + sc.T) / 2.0
    np.fill_diagonal(C, 0.0)
    return C / C.max() * max_value


def load_group(max_value: float = 0.2) -> dict:
    """Group-average SC (mean of per-subject normalized SCs) and per-subject BOLD."""
    sids = subjects()
    subs = [load_subject(s) for s in sids]
    C = np.mean([normalize_sc(s["sc"], 1.0) for s in subs], axis=0)
    C = C / C.max() * max_value
    return {"sids": sids, "C": C, "ts": [s["ts"] for s in subs],
            "len": np.mean([s["len"] for s in subs], axis=0), "TR": TR_HCP}
