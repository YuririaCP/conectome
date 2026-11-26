"""Recovery of delta in the range seen in HCP data (0.1-0.5 s^-1), with a misspecified hemodynamics.

Ground truth: symmetric diffusive linearized Stuart-Landau on the Schaefer-100 group connectome, empirical
heterogeneous frequencies (as M1 in recovery_delta.py).  Observation: regional HRFs that differ from the
canonical one (peak time 6 s jittered by a subject-level shift ~ N(0, 0.5 s) and a regional jitter
~ N(0, 0.5 s)), global physiological component, white noise, high-pass 0.008 Hz.  Cleaning: regression
of a recorded physiological signal (correlation 0.9 with the truth).  Estimator: the observation-aware
fit of temporal_est.py, which assumes the canonical HRF, on the global signal.
Compared with the canonical-HRF case, and with 1 vs 4 sessions.
Output: results/recovery_fast.npz
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import metrics as M  # noqa: E402
import temporal_est as te  # noqa: E402
from fcdata import SC_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TR, N_VOL, REPS, G = te.TR, 4 * 1200, 10, 0.5
DELTAS = [0.03, 0.1, 0.2, 0.3, 0.5]

g = data.load_group()
om_aal = 2 * np.pi * np.mean([M.peak_frequencies(ts, TR) for ts in g["ts"]], axis=0)
C = SC_all.mean(0)
C = (C + C.T) / 2
np.fill_diagonal(C, 0)
C = C / C.sum(1).mean()
N = len(C)
B = G * -(np.diag(C.sum(1)) - C)
om = np.random.default_rng(0).permutation(np.resize(om_aal, N))


def jac(shift):
    A = B - shift * np.eye(N)
    return np.block([[A, -np.diag(om)], [np.diag(om), A]])


def shift_for(delta):
    return brentq(lambda s: -np.max(np.linalg.eigvals(jac(s)).real) - delta, -1.0, 5.0)


def simulate(J, n, seed):
    m = len(J)
    Phi = expm(J * TR)
    Sig = solve_continuous_lyapunov(J, -np.eye(m))
    Qd = Sig - Phi @ Sig @ Phi.T
    Lq = np.linalg.cholesky((Qd + Qd.T) / 2 + 1e-12 * np.eye(m))
    r = np.random.default_rng(seed)
    u = np.linalg.cholesky(Sig + 1e-12 * np.eye(m)) @ r.standard_normal(m)
    out = np.empty((m // 2, n))
    for t in range(n):
        u = Phi @ u + Lq @ r.standard_normal(m)
        out[:, t] = u[: m // 2]
    return out


def hrf_peak(peak, length=32.0):
    t = np.arange(0, length, TR)
    h = stats.gamma.pdf(t, peak) - stats.gamma.pdf(t, peak + 10) / 6     # shape a ~ peak time + 1
    return h / h.sum()


def physio(n, r):
    phi = np.exp(-TR / 20.0)
    slow = signal.lfilter([1.0], [1.0, -phi], r.standard_normal(n))
    bb, aa = signal.butter(2, [0.25, 0.35], btype="band", fs=te.FS)
    resp = signal.lfilter(bb, aa, r.standard_normal(n))
    p = slow / slow.std() + 0.7 * resp / resp.std()
    return p - p.mean()


def observe(x, seed, variable_hrf):
    r = np.random.default_rng(seed)
    n = x.shape[1]
    if variable_hrf:
        subj = r.normal(0, 0.5)
        peaks = 6 + subj + r.normal(0, 0.5, N)
        bold = np.stack([signal.lfilter(hrf_peak(pk), [1.0], xi) for pk, xi in zip(peaks, x)])
    else:
        bold = signal.lfilter(te.H, [1.0], x, axis=1)
    bold = bold - bold.mean(1, keepdims=True)
    p = physio(n, r)
    load = 1 + 0.3 * r.standard_normal(N)
    y = bold + np.sqrt(bold.mean(0).var()) * np.outer(load, p)
    y = y + np.sqrt(0.5 * bold.var(1, keepdims=True)) * r.standard_normal(bold.shape)
    y = te.highpass(y)
    p_rec = te.highpass(0.9 * p / p.std() + np.sqrt(0.19) * r.standard_normal(n))
    return te.regress_out(y, p_rec[:, None])


EST = np.full((len(DELTAS), 2, 2, REPS), np.nan)      # delta, hrf (canonical, variable), length (1, 4 sessions), rep
for i, d in enumerate(DELTAS):
    J = jac(shift_for(d))
    for rep in range(REPS):
        x = simulate(J, N_VOL, seed=1000 * i + rep)
        for h, var in enumerate([False, True]):
            for k, n in enumerate([1200, N_VOL]):
                y = observe(x[:, :n], seed=1000 * i + rep + 17 * h, variable_hrf=var)
                EST[i, h, k, rep] = te.fit_delta(te.acf(y.mean(0)))
    for h, lab in enumerate(["canonical HRF", "variable HRF"]):
        for k, n in enumerate([1, 4]):
            rel = EST[i, h, k] / d - 1
            q = np.percentile(rel, [25, 50, 75])
            print(f"delta={d:.2f} {lab:13s} {n} session(s): median rel. error {q[1]:+.2f} [IQR {q[0]:+.2f}, {q[2]:+.2f}]", flush=True)
np.savez(ROOT / "results" / "recovery_fast.npz", EST=EST, DELTAS=DELTAS)
