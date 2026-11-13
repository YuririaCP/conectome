"""Empirical/simulated BOLD statistics: FC, FCD, metastability, peak frequencies."""
from __future__ import annotations

import numpy as np
from scipy import signal, stats


def filter_ts(ts: np.ndarray, TR: float, band=(0.008, 0.08), order=2) -> np.ndarray:
    b, a = signal.butter(order, list(band), btype="band", fs=1.0 / TR)
    x = signal.detrend(ts, axis=1)
    x = x - x.mean(axis=1, keepdims=True)
    return signal.filtfilt(b, a, x, axis=1)


def fc(ts: np.ndarray) -> np.ndarray:
    return np.corrcoef(ts)


def triu(M: np.ndarray, k: int = 1) -> np.ndarray:
    return M[np.triu_indices(M.shape[0], k)]


def fc_fit(fc_sim: np.ndarray, fc_emp: np.ndarray) -> dict:
    """Pearson r between upper triangles, and mean-squared difference."""
    s, e = triu(fc_sim), triu(fc_emp)
    return {"r": float(np.corrcoef(s, e)[0, 1]), "mse": float(np.mean((s - e) ** 2))}


def peak_frequencies(ts: np.ndarray, TR: float, band=(0.008, 0.08)) -> np.ndarray:
    """Peak of the Welch power spectrum of each filtered signal (Hz)."""
    x = filter_ts(ts, TR, band)
    f, P = signal.welch(x, fs=1.0 / TR, nperseg=min(x.shape[1], 256), axis=1)
    sel = (f >= band[0]) & (f <= band[1])
    return f[sel][np.argmax(P[:, sel], axis=1)]


def fcd(ts_filt: np.ndarray, TR: float, window: float = 60.0, step: float = 2.0) -> np.ndarray:
    """Upper triangle of the FCD matrix (correlations between sliding-window FCs)."""
    w = int(round(window / TR))
    s = max(1, int(round(step / TR)))
    starts = range(0, ts_filt.shape[1] - w + 1, s)
    iu = np.triu_indices(ts_filt.shape[0], 1)
    vecs = np.array([np.corrcoef(ts_filt[:, t:t + w])[iu] for t in starts])
    return triu(np.corrcoef(vecs))


def autocorr(ts_filt: np.ndarray, lags) -> np.ndarray:
    """Normalized autocorrelation of each row at integer lags (in samples): (n_lags, N)."""
    x = ts_filt - ts_filt.mean(axis=1, keepdims=True)
    v = (x * x).mean(axis=1)
    T = x.shape[1]
    return np.array([(x[:, k:] * x[:, :T - k]).mean(axis=1) / v for k in lags])


def ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ks_2samp(a, b).statistic)


def kuramoto_order(ts_filt: np.ndarray) -> np.ndarray:
    phase = np.angle(signal.hilbert(ts_filt, axis=1))
    return np.abs(np.exp(1j * phase).mean(axis=0))


def metastability(ts_filt: np.ndarray, trim: int = 10) -> float:
    R = kuramoto_order(ts_filt)[trim:-trim]
    return float(R.std())


def synchrony(ts_filt: np.ndarray, trim: int = 10) -> float:
    return float(kuramoto_order(ts_filt)[trim:-trim].mean())
