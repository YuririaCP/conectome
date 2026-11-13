"""Fase 1: group-level fit of the Stuart-Landau model to HCP resting-state data.

1. Empirical targets: group FC (Fisher-averaged), pooled FCD, metastability.
2. Natural frequencies from the BOLD spectral peak of each region.
3. Fast sweep over (a, G) using the analytic band-passed FC of the linearization.
4. Check analytic filtered FC against filtered nonlinear simulations.
5. Nonlinear simulations along G at the best a: FC fit, FCD KS distance, metastability.

Outputs: figures/fase1_*.png and results/fase1_*.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import hopf  # noqa: E402
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)
FIG = ROOT / "figures"
BAND = (0.008, 0.08)
BETA = 0.02
KIND = "diffusive"

# ---------------------------------------------------------------- empirical
g = data.load_group()
C, TR = g["C"], g["TR"]
N = C.shape[0]
filt = [M.filter_ts(ts, TR, BAND) for ts in g["ts"]]
fcs = [M.fc(x) for x in filt]
FC_emp = np.tanh(np.mean([np.arctanh(np.clip(f, -0.999999, 0.999999)) for f in fcs], axis=0))
np.fill_diagonal(FC_emp, 1.0)
FCD_emp = np.concatenate([M.fcd(x, TR) for x in filt])
meta_emp = np.array([M.metastability(x) for x in filt])
freqs = np.mean([M.peak_frequencies(ts, TR, BAND) for ts in g["ts"]], axis=0)
omega = 2 * np.pi * freqs
n_TR = g["ts"][0].shape[1]
print(f"N={N} regions, {len(filt)} subjects, {n_TR} TRs, TR={TR}s")
print(f"empirical: mean FC={M.triu(FC_emp).mean():.3f}  metastability={meta_emp.mean():.3f}±{meta_emp.std():.3f}")
print(f"peak freqs: {freqs.min():.3f}-{freqs.max():.3f} Hz (mean {freqs.mean():.3f})")
print(f"SC: density={np.mean(M.triu(C) > 0):.2f}  lambda_max={np.linalg.eigvalsh(C).max():.3f}  mean degree={C.sum(1).mean():.3f}")

# ---------------------------------------------------------------- analytic sweep
a_grid = np.array([-0.2, -0.1, -0.05, -0.02, -0.01, -0.005])
G_grid = np.linspace(0, 6, 61)
R = np.full((len(a_grid), len(G_grid)), np.nan)
MSE = np.full_like(R, np.nan)
DELTA = np.full_like(R, np.nan)
t0 = time.time()
for i, a in enumerate(a_grid):
    for j, G in enumerate(G_grid):
        S = hopf.linear_covariance_filtered(a, omega, C, G, BETA, TR, KIND, band=BAND, fmax=0.3)
        fit = M.fc_fit(hopf.cov_to_corr(S), FC_emp)
        R[i, j], MSE[i, j] = fit["r"], fit["mse"]
        DELTA[i, j] = hopf.distance_to_criticality(a, omega, C, G, KIND)
print(f"analytic sweep: {R.size} points in {time.time() - t0:.1f}s")
ib, jb = np.unravel_index(np.nanargmax(R), R.shape)
a_best, G_best = a_grid[ib], G_grid[jb]
print(f"best analytic fit: a={a_best}, G={G_best:.2f}, r={R[ib, jb]:.3f}")
# the "mse-best" G at a_best (matches the overall level of correlation, not just the pattern)
jm = np.nanargmin(MSE[ib])
print(f"min-MSE G at a={a_best}: G={G_grid[jm]:.2f}, r={R[ib, jm]:.3f}, mse={MSE[ib, jm]:.4f}")

# ---------------------------------------------------------------- analytic vs simulated (filtered)
check = {}
for G in [0.5, G_best, G_grid[jm]]:
    S = hopf.linear_covariance_filtered(a_best, omega, C, G, BETA, TR, KIND, band=BAND, fmax=0.3)
    x = hopf.simulate(a_best, omega, C, G, BETA, T=20 * n_TR * TR, TR=TR, kind=KIND, seed=3)
    FC_sim = M.fc(M.filter_ts(x, TR, BAND))
    r = np.corrcoef(M.triu(FC_sim), M.triu(hopf.cov_to_corr(S)))[0, 1]
    check[float(G)] = (hopf.cov_to_corr(S), FC_sim)
    print(f"G={G:.2f}: r(FC analytic filtered, FC simulated filtered) = {r:.3f}")

# ---------------------------------------------------------------- nonlinear sweep
G_sim = np.linspace(0, 6, 25)
n_runs = len(filt)
sim_r, sim_ks, sim_meta, sim_meanfc = [], [], [], []
t0 = time.time()
for G in G_sim:
    fc_list, fcd_list, meta_list = [], [], []
    for s in range(n_runs):
        x = M.filter_ts(hopf.simulate(a_best, omega, C, G, BETA, T=n_TR * TR, TR=TR, kind=KIND, seed=100 + s), TR, BAND)
        fc_list.append(M.fc(x))
        fcd_list.append(M.fcd(x, TR))
        meta_list.append(M.metastability(x))
    FC_s = np.mean(fc_list, axis=0)
    sim_r.append(M.fc_fit(FC_s, FC_emp)["r"])
    sim_meanfc.append(M.triu(FC_s).mean())
    sim_ks.append(M.ks(np.concatenate(fcd_list), FCD_emp))
    sim_meta.append(np.mean(meta_list))
print(f"nonlinear sweep: {len(G_sim)} G values x {n_runs} runs in {time.time() - t0:.1f}s")
sim_r, sim_ks, sim_meta, sim_meanfc = map(np.array, (sim_r, sim_ks, sim_meta, sim_meanfc))
for G, r, k, m, mf in zip(G_sim, sim_r, sim_ks, sim_meta, sim_meanfc):
    print(f"  G={G:4.2f}  r(FC)={r:.3f}  meanFC={mf:.3f}  KS(FCD)={k:.3f}  metast={m:.3f}")

np.savez(ROOT / "results" / "fase1_group_fit.npz", a_grid=a_grid, G_grid=G_grid, R=R, MSE=MSE, DELTA=DELTA,
         G_sim=G_sim, sim_r=sim_r, sim_ks=sim_ks, sim_meta=sim_meta, sim_meanfc=sim_meanfc,
         FC_emp=FC_emp, C=C, freqs=freqs, meta_emp=meta_emp, a_best=a_best, G_best=G_best)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, a in enumerate(a_grid):
    ax[0].plot(G_grid, R[i], label=f"a={a}")
ax[0].set(xlabel="G", ylabel="r(FC modelo, FC empírica)", title="Ajuste analítico (lineal, filtrado)")
ax[0].legend(fontsize=8)
for i, a in enumerate(a_grid):
    ax[1].plot(G_grid, DELTA[i], label=f"a={a}")
ax[1].set(xlabel="G", ylabel=r"$\delta = -\max\,\mathrm{Re}\,\lambda(J)$", yscale="log",
          title="Distancia a la criticalidad")
for i, a in enumerate(a_grid):
    ax[2].plot(G_grid, MSE[i], label=f"a={a}")
ax[2].set(xlabel="G", ylabel="MSE(FC)", yscale="log", title="Error cuadrático medio")
fig.tight_layout()
fig.savefig(FIG / "fase1_barrido_analitico.png", dpi=130)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(G_sim, sim_r, "o-", label="r(FC) simulación")
ax[0].plot(G_grid, R[ib], "-", alpha=0.6, label="r(FC) analítico")
ax[0].plot(G_sim, sim_meanfc, "s-", label="FC media sim.")
ax[0].axhline(M.triu(FC_emp).mean(), color="k", ls=":", label="FC media emp.")
ax[0].set(xlabel="G", title=f"FC (a={a_best})")
ax[0].legend(fontsize=8)
ax[1].plot(G_sim, sim_ks, "o-")
ax[1].set(xlabel="G", ylabel="KS(FCD sim, FCD emp)", title="Dinámica de la FC")
ax[2].plot(G_sim, sim_meta, "o-", label="simulación")
ax[2].axhspan(meta_emp.mean() - meta_emp.std(), meta_emp.mean() + meta_emp.std(), color="gray", alpha=0.3, label="empírica ±1 de")
ax[2].set(xlabel="G", ylabel="metaestabilidad", title="Metaestabilidad")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fase1_barrido_no_lineal.png", dpi=130)

Gk = float(G_grid[jm])
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
ax[0].imshow(C, cmap="viridis"); ax[0].set_title("SC (grupo, normalizada)")
ax[1].imshow(FC_emp, vmin=-0.2, vmax=1, cmap="RdBu_r"); ax[1].set_title("FC empírica")
ax[2].imshow(check[Gk][0], vmin=-0.2, vmax=1, cmap="RdBu_r"); ax[2].set_title(f"FC analítica (G={Gk:.2f})")
ax[3].scatter(M.triu(check[Gk][0]), M.triu(check[Gk][1]), s=2, alpha=0.4)
ax[3].plot([-0.2, 1], [-0.2, 1], "k--", lw=0.8)
ax[3].set(xlabel="FC analítica filtrada", ylabel="FC simulada filtrada", title="Analítico vs simulación")
fig.tight_layout()
fig.savefig(FIG / "fase1_matrices.png", dpi=130)
print("figures saved")
