"""Global-mode confound in the full nonlinear Stuart-Landau model, below and above the bifurcation.

Setup: neurolib HCP group SC (AAL2, N=94) normalized to mean node strength 1, diffusive coupling
G = 0.2, heterogeneous empirical peak frequencies omega_j, beta = 0.02, homogeneous a swept across
the bifurcation (a<0 damped, a>0 self-sustained oscillations).  FC is the zero-lag correlation of
x = Re z sampled every TR (unfiltered; band-passed FC is reported as a robustness check).
For each pair of neighbouring a values we measure how much FC changes without and with GSR,
against the seed-to-seed sampling floor, and compare with the linear (Lyapunov) prediction for a<0.
Output: results/nonlinear_confound.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import hopf  # noqa: E402
import metrics as M  # noqa: E402
from fcdata import gsr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
g = data.load_group()
TR = g["TR"]
C = (g["C"] + g["C"].T) / 2
np.fill_diagonal(C, 0)
C = C / C.sum(1).mean()
N = len(C)
iu = np.triu_indices(N, 1)
omega = 2 * np.pi * np.mean([M.peak_frequencies(ts, TR) for ts in g["ts"]], axis=0)
G, BETA, T, SEEDS = 0.2, 0.02, 50000.0, (1, 2, 3, 4)
A = np.array([-0.2, -0.1, -0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05])

F_raw = np.zeros((len(A), len(SEEDS), len(iu[0])))
F_gsr = np.zeros_like(F_raw)
F_bp = np.zeros_like(F_raw)
F_bpg = np.zeros_like(F_raw)
F_lin = np.full((len(A), len(iu[0])), np.nan)
F_ling = np.full_like(F_lin, np.nan)
amp = np.zeros((len(A), len(SEEDS)))
for i, a in enumerate(A):
    for k, sd in enumerate(SEEDS):
        x = hopf.simulate(a, omega, C, G, BETA, T=T, TR=TR, transient=500.0, seed=100 * i + sd)
        F = np.corrcoef(x)
        F_raw[i, k], F_gsr[i, k] = F[iu], gsr(F)[iu]
        Fb = np.corrcoef(M.filter_ts(x, TR))
        F_bp[i, k], F_bpg[i, k] = Fb[iu], gsr(Fb)[iu]
        amp[i, k] = np.sqrt(np.mean(x ** 2))
    if a < 0:
        S = hopf.linear_covariance(a, omega, C, G, BETA)[:N, :N]
        Fl = hopf.cov_to_corr(S)
        F_lin[i], F_ling[i] = Fl[iu], gsr(Fl)[iu]
    print(f"a={a:+.3f}  mean FC={F_raw[i].mean():.3f}  rms x={amp[i].mean():.3f}", flush=True)


def rms(v):
    return np.sqrt(np.mean(v ** 2, axis=-1))


def table(Fr, Fg, label):
    m = Fr.mean(1)
    mg = Fg.mean(1)
    d_raw = rms(np.diff(m, axis=0))
    d_gsr = rms(np.diff(mg, axis=0))
    K = Fr.shape[1]                                          # sampling noise (rms) of a K-seed mean
    floor_raw = np.sqrt(np.mean(Fr.var(1, ddof=1), axis=-1) / K)
    floor_gsr = np.sqrt(np.mean(Fg.var(1, ddof=1), axis=-1) / K)
    # noise-corrected rms change: E|diff of means|^2 = signal^2 + n_j^2 + n_{j+1}^2
    nr = floor_raw[:-1] ** 2 + floor_raw[1:] ** 2
    ng = floor_gsr[:-1] ** 2 + floor_gsr[1:] ** 2
    s_raw = np.sqrt(np.clip(d_raw ** 2 - nr, 0, None))
    s_gsr = np.sqrt(np.clip(d_gsr ** 2 - ng, 0, None))
    print(f"\n{label}:  interval      dFC raw   dFC GSR  (noise-corrected)   ratio   noise rms raw/GSR")
    for j in range(len(A) - 1):
        print(f"   [{A[j]:+.3f},{A[j + 1]:+.3f}]  {d_raw[j]:.4f}   {d_gsr[j]:.4f}  ({s_raw[j]:.4f} / {s_gsr[j]:.4f})"
              f"   {s_gsr[j] / max(s_raw[j], 1e-12):.3f}   {np.sqrt(nr[j]):.4f}/{np.sqrt(ng[j]):.4f}")
    return d_raw, d_gsr, s_raw, s_gsr, np.sqrt(nr), np.sqrt(ng)


out = {}
out["unf"] = table(F_raw, F_gsr, "unfiltered")
out["bp"] = table(F_bp, F_bpg, "band-passed 0.008-0.08 Hz")
d_lin = rms(np.diff(F_lin, axis=0))
d_ling = rms(np.diff(F_ling, axis=0))
print("\nlinear prediction (a<0):", np.round(d_ling / d_lin, 3))
r_lin = [np.corrcoef(F_lin[i], F_raw[i].mean(0))[0, 1] for i in range(len(A)) if A[i] < 0]
print("r(linear FC, nonlinear FC) for a<0:", np.round(r_lin, 3))
np.savez(ROOT / "results" / "nonlinear_confound.npz", A=A, G=G, F_raw=F_raw, F_gsr=F_gsr, F_bp=F_bp, F_bpg=F_bpg,
         F_lin=F_lin, F_ling=F_ling, amp=amp,
         **{f"{k}_{n}": v for k, vals in out.items() for n, v in zip(["draw", "dgsr", "sraw", "sgsr", "nraw", "ngsr"], vals)},
         d_lin=d_lin, d_ling=d_ling)
