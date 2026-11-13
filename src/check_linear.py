"""Fase 0: compare nonlinear SDE simulations with the exact linear (Lyapunov) statistics.

Uses a random sparse symmetric network so it does not depend on the HCP data.
Output: figures/fase0_lineal_vs_simulacion.png and printed table.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hopf  # noqa: E402
from metrics import triu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
N = 20
C = rng.random((N, N)) * (rng.random((N, N)) < 0.3)
C = (C + C.T) / 2
np.fill_diagonal(C, 0)
C = C / C.max() * 0.2
omega = 2 * np.pi * rng.uniform(0.04, 0.07, N)
beta, G, TR = 0.02, 1.0, 0.72
T = 12000.0  # seconds of simulated data per run

rows = []
examples = {}
for a in [-0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005, -0.002, -0.001]:
    S_lin = hopf.linear_covariance(a, omega, C, G, beta)[:N, :N]
    x = hopf.simulate(a, omega, C, G, beta, T=T, TR=TR, dt=TR / 20, seed=1)
    S_sim = np.cov(x)
    var_err = np.mean(np.abs(np.diag(S_sim) - np.diag(S_lin)) / np.diag(S_lin))
    r_corr = np.corrcoef(triu(hopf.cov_to_corr(S_sim)), triu(hopf.cov_to_corr(S_lin)))[0, 1]
    # nonlinearity ratio: <|z|^2> / |a|
    ratio = 2 * np.mean(np.diag(S_lin)) / abs(a)
    rows.append((a, ratio, var_err, r_corr))
    examples[a] = (S_lin, S_sim)
    print(f"a={a:7.3f}  <|z|^2>/|a|={ratio:6.3f}  err.rel.var={var_err:6.3f}  r(FC_lin,FC_sim)={r_corr:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, mk in [(-0.5, "o"), (-0.02, "s")]:
    S_lin, S_sim = examples[a]
    ax[0].scatter(np.diag(S_lin), np.diag(S_sim), s=12, marker=mk, label=f"a={a}")
    ax[1].scatter(triu(hopf.cov_to_corr(S_lin)), triu(hopf.cov_to_corr(S_sim)), s=6, marker=mk, label=f"a={a}")
for k in (0, 1):
    lo, hi = ax[k].get_xlim()
    ax[k].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[k].legend()
ax[0].set(xscale="log", yscale="log", xlabel="varianza lineal (Lyapunov)", ylabel="varianza simulada", title="Varianzas")
ax[1].set(xlabel="FC lineal", ylabel="FC simulada", title="Correlaciones")
arr = np.array(rows)
ax[2].plot(arr[:, 1], arr[:, 2], "o-", label="error rel. varianza")
ax[2].plot(arr[:, 1], 1 - arr[:, 3], "s-", label="1 - r(FC)")
ax[2].set(xscale="log", yscale="log", xlabel=r"$\langle|z|^2\rangle / |a|$  (no linealidad)", title="Validez de la linealización")
ax[2].legend()
fig.tight_layout()
out = ROOT / "figures" / "fase0_lineal_vs_simulacion.png"
fig.savefig(out, dpi=130)
print("saved", out)
