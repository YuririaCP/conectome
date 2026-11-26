"""Shared temporal estimator of the critical rate delta (observation-aware autocorrelation fit).

Used by recovery_delta.py (synthetic data) and hcp_delta.py (HCP).  The model autocorrelation of a
projected BOLD signal is that of a Lorentzian pair at +-omega with width delta, filtered by the
canonical hemodynamic response and by the forward-backward high-pass filter, plus a white-noise fraction.
"""
import numpy as np
from scipy import signal, stats
from scipy.optimize import least_squares

TR = 0.72
FS = 1 / TR
LAGS = np.arange(0, int(60 / TR) + 1)
NF = 8192


def hrf(tr=TR, length=32.0):
    t = np.arange(0, length, tr)
    h = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6
    return h / h.sum()


H = hrf()
B_HP, A_HP = signal.butter(2, 0.008, btype="high", fs=FS)
_freqs = np.fft.rfftfreq(NF, d=TR)
_Hf = np.abs(np.fft.rfft(H, NF)) ** 2
_HP = np.abs(signal.freqz(B_HP, A_HP, worN=_freqs, fs=FS)[1]) ** 4


def highpass(x):
    return signal.filtfilt(B_HP, A_HP, x, axis=-1)


def acf(s, lags=LAGS):
    s = s - s.mean()
    f = np.fft.rfft(s, 2 * len(s))
    a = np.fft.irfft(f * np.conj(f))[: lags[-1] + 1]
    return a[lags] / a[0]


def model_ac(delta, om, white, aware=True, lags=LAGS):
    if not aware:
        tau = lags * TR
        return np.exp(-delta * tau) * np.cos(om * tau)
    w = 2 * np.pi * _freqs
    lor = delta / (delta ** 2 + (w - om) ** 2) + delta / (delta ** 2 + (w + om) ** 2)
    spec = lor * _Hf * _HP
    spec = spec / spec.sum()
    ac = np.fft.irfft(spec, NF)[: lags[-1] + 1]
    ac = (1 - white) * ac / ac[0]
    ac[0] = 1.0
    return ac[lags]


def fit_delta(ac, aware=True):
    def res(p):
        return model_ac(np.exp(p[0]), p[1], p[2], aware) - ac
    best = None
    for d0 in [0.01, 0.05, 0.2]:
        try:
            s = least_squares(res, x0=[np.log(d0), 0.15, 0.2], bounds=([np.log(1e-3), 0.0, 0.0], [np.log(2.0), 1.0, 0.95]))
            if best is None or s.cost < best.cost:
                best = s
        except ValueError:
            pass
    return np.exp(best.x[0]) if best is not None else np.nan


def regress_out(y, R):
    """Remove the column space of R (T x k) from each row of y (n x T)."""
    R = np.column_stack([np.ones(R.shape[0]), R])
    beta, *_ = np.linalg.lstsq(R, y.T, rcond=None)
    return y - (R @ beta).T
