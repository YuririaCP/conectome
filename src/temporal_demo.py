"""What to measure instead: the physical distance delta from the autocorrelation of the global signal.

A. Model (linear diffusive Stuart-Landau on the Schaefer-100 group connectome, homogeneous a, common
   omega).  Along the family (a, omega, G, beta) -> c (a, omega, G, beta), FC is invariant
   (Proposition 1) while delta = c|a|.  From finite simulated sessions (4 x 1200 TR) we estimate delta
   by fitting exp(-delta tau) cos(omega tau) to the autocorrelation of (i) the global signal,
   (ii) each region (median), (iii) each region after GSR (median).
B. Data (7 HCP subjects): autocorrelation of the global signal vs regional signals, before and after
   GSR (high-pass 0.008 Hz only).  A consistency check: the slowest mode of the model is the global
   signal, and GSR should shorten regional time scales.
Output: results/temporal_demo.npz
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from fcdata import SC_all, laplacian  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TR = 0.72
rng = np.random.default_rng(5)

# ---------------------------------------------------------------- A. model
C = SC_all.mean(0)
C = (C + C.T) / 2
np.fill_diagonal(C, 0)
C = C / C.sum(1).mean()
N = len(C)
iu = np.triu_indices(N, 1)
L = laplacian(C)
a0, G0, w0, b0 = -0.02, 0.8, 2 * np.pi * 0.02, 0.02
SEEDS = range(5)
n_T = 4 * 1200
LAGS = np.arange(0, int(60 / TR) + 1)


def simulate_linear(a, w, G, beta, n, seed):
    """Exact discretization at step TR of the linear complex OU process dz = (A + i w) z dt + beta dW."""
    J = np.block([[a * np.eye(N) - G * L, -w * np.eye(N)], [w * np.eye(N), a * np.eye(N) - G * L]])
    Phi = expm(J * TR)
    Sig = solve_continuous_lyapunov(J, -beta ** 2 * np.eye(2 * N))
    Qd = Sig - Phi @ Sig @ Phi.T
    Lq = np.linalg.cholesky((Qd + Qd.T) / 2 + 1e-14 * np.eye(2 * N))
    r = np.random.default_rng(seed)
    u = np.linalg.cholesky(Sig) @ r.standard_normal(2 * N)
    out = np.empty((N, n))
    for t in range(n):
        u = Phi @ u + Lq @ r.standard_normal(2 * N)
        out[:, t] = u[:N]
    return out


def acf(x, lags):
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[: lags[-1] + 1]
    return ac[lags] / ac[0]


def fit_delta(ac, w_guess):
    tau = LAGS * TR
    f = lambda t, d, w: np.exp(-d * t) * np.cos(w * t)
    try:
        p, _ = curve_fit(f, tau, ac, p0=[0.05, w_guess], bounds=([1e-4, 0], [5, 2]), maxfev=20000)
        return p[0]
    except RuntimeError:
        return np.nan


def gsr_ts(x):
    gs = x.mean(0)
    return x - np.outer(x @ gs / (gs @ gs), gs)


cs = np.array([0.5, 1.0, 2.0, 4.0])
est = np.zeros((len(cs), len(SEEDS), 3))
fc_dist = np.zeros((len(cs), len(SEEDS)))
J0 = lambda c: np.block([[c * (a0 * np.eye(N) - G0 * L), -c * w0 * np.eye(N)], [c * w0 * np.eye(N), c * (a0 * np.eye(N) - G0 * L)]])
S_an = [solve_continuous_lyapunov(J0(c), -(c * b0) ** 2 * np.eye(2 * N))[:N, :N] for c in cs]
FC_an = [ (S / np.sqrt(np.outer(np.diag(S), np.diag(S))))[iu] for S in S_an]
an_diff = max(np.abs(f - FC_an[1]).max() for f in FC_an)
for i, c in enumerate(cs):
    for k, sd in enumerate(SEEDS):
        x = simulate_linear(c * a0, c * w0, c * G0, c * b0, n_T, seed=1000 * i + sd)
        fc_dist[i, k] = np.sqrt(np.mean((np.corrcoef(x)[iu] - FC_an[i]) ** 2))
        est[i, k, 0] = fit_delta(acf(x.mean(0), LAGS), c * w0)
        est[i, k, 1] = np.nanmedian([fit_delta(acf(xi, LAGS), c * w0) for xi in x])
        est[i, k, 2] = np.nanmedian([fit_delta(acf(xi, LAGS), c * w0) for xi in gsr_ts(x)])
    m, sdv = est[i].mean(0), est[i].std(0)
    print(f"c={c}: true delta={c * abs(a0):.3f}  GS {m[0]:.3f}+-{sdv[0]:.3f}  regional {m[1]:.3f}+-{sdv[1]:.3f}  "
          f"regional after GSR {m[2]:.3f}+-{sdv[2]:.3f}   rms(sampled FC - analytic FC) {fc_dist[i].mean():.3f}")
print(f"analytic FC identical across c: max |dFC| = {an_diff:.1e}; mean FC = {FC_an[1].mean():.3f}")
lam = np.sort(np.linalg.eigvalsh(L))
print(f"theta = {abs(a0) / G0:.3f}; next mode rate / slowest = {(abs(a0) / G0 + lam[1]) / (abs(a0) / G0):.2f}")
rows = np.column_stack([cs, cs * abs(a0), est.mean(1), est.std(1)])
r_fc = fc_dist

# ---------------------------------------------------------------- B. data
g = data.load_group()
b, a = signal.butter(2, 0.008, btype="high", fs=1 / TR)
LAG_D = np.arange(0, int(40 / TR) + 1)
ac_gs, ac_reg, ac_regg = [], [], []
for ts in g["ts"]:
    x = signal.filtfilt(b, a, signal.detrend(ts, axis=1), axis=1)
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    ac_gs.append(acf(x.mean(0), LAG_D))
    ac_reg.append(np.median([acf(xi, LAG_D) for xi in x], axis=0))
    ac_regg.append(np.median([acf(xi, LAG_D) for xi in gsr_ts(x)], axis=0))
ac_gs, ac_reg, ac_regg = map(np.array, (ac_gs, ac_reg, ac_regg))


def t_int(ac):                       # integrated autocorrelation time up to the first zero crossing
    k = np.argmax(ac <= 0) if np.any(ac <= 0) else len(ac)
    return TR * (0.5 + ac[1:k].sum())


tg, tr_, trg = (np.array([t_int(c_) for c_ in A_]) for A_ in (ac_gs, ac_reg, ac_regg))
print(f"data: integrated AC time (s)  GS {tg.round(1)}  regional {tr_.round(1)}  regional after GSR {trg.round(1)}")
print(f"   GS > regional in {np.sum(tg > tr_)}/7; GSR shortens regional in {np.sum(trg < tr_)}/7")
np.savez(ROOT / "results" / "temporal_demo.npz", rows=rows, r_fc=r_fc, LAGS=LAGS, LAG_D=LAG_D,
         ac_gs=ac_gs, ac_reg=ac_reg, ac_regg=ac_regg, tg=tg, tr=tr_, trg=trg, a0=a0, G0=G0, w0=w0)
