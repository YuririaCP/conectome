"""Fase 3: effective connectivity (directed, non-negative weights) for the Stuart-Landau model.

Starting from the anatomical SC, iterate (Deco et al. 2019 / Gilson et al. 2016 style)

    C_jk <- max(0, C_jk + eps * [(FC0_emp - FC0_mod) + (FCtau_emp - FCtau_mod)]_jk)   on mask

where FC0 is the filtered zero-lag correlation and FCtau the normalized lagged covariance
E[x_j(t+tau) x_k(t)] / sqrt(var_j var_k).  The lagged term breaks the symmetry and is what
lets the fit infer direction.  Model statistics are analytic (hopf.FilteredSpectrum.covs).

Validation:
  * split-half (fit on 4 subjects, test on 3) with learning curves -> early stopping
  * noise ceiling: how well group-A FC predicts group-B FC
  * two masks: all pairs vs. strongest 20% SC links (anatomical prior)
  * full fit -> nonlinear simulations: does FCD follow once FC is right?

Outputs: results/fase3_ec.npz, figures/fase3_*.png
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
FIG = ROOT / "figures"
BAND = (0.008, 0.08)
TAU = 2           # lag in TRs
A_HOM = -0.0696   # homogeneous a from Fase 2 (fit to FC + autocorrelations)
G0 = 3.3          # homogeneous G from Fase 2; EC is initialized at G0 * SC
EPS = 0.004
N_IT = 3000
REC = 50
BETA = 0.02

g = data.load_group()
SC, TR = g["C"], g["TR"]
N = SC.shape[0]
off = ~np.eye(N, dtype=bool)
iu = np.triu_indices(N, 1)
filt = [M.filter_ts(ts, TR, BAND) for ts in g["ts"]]
freqs_all = [M.peak_frequencies(ts, TR, BAND) for ts in g["ts"]]
FS = hopf.FilteredSpectrum(TR, band=BAND, lags_TR=[0, TAU])


def lagged_corr(x, k):
    x = x - x.mean(axis=1, keepdims=True)
    T = x.shape[1]
    S0 = x @ x.T / T
    Sk = x[:, k:] @ x[:, :T - k].T / (T - k)
    d = np.sqrt(np.diag(S0))
    return S0 / np.outer(d, d), Sk / np.outer(d, d)


def targets(idx):
    pairs = [lagged_corr(filt[i], TAU) for i in idx]
    fc0 = np.tanh(np.mean([np.arctanh(np.clip(p[0], -0.999999, 0.999999)) for p in pairs], axis=0))
    np.fill_diagonal(fc0, 1.0)
    fct = np.mean([p[1] for p in pairs], axis=0)
    om = 2 * np.pi * np.mean([freqs_all[i] for i in idx], axis=0)
    return fc0, fct, om


def model(Cm, om):
    S0, Sk = FS.covs(A_HOM, om, Cm, 1.0)
    d = np.sqrt(np.diag(S0))
    return S0 / np.outer(d, d), Sk / np.outer(d, d)


def score(fc_mod, fc_t):
    m, t = fc_mod[iu], fc_t[iu]
    return float(np.corrcoef(m, t)[0, 1]), float(np.mean((m - t) ** 2))


def fit_ec(fc0_t, fct_t, om, mask, tests=None, n_it=N_IT):
    Cm = G0 * SC.copy()
    curve = []
    t0 = time.time()
    for it in range(n_it + 1):
        f0, ft = model(Cm, om)
        if it % REC == 0:
            row = [it, *score(f0, fc0_t)]
            for (fc0_b, _) in (tests or []):
                row += list(score(f0, fc0_b))
            row.append(hopf.distance_to_criticality(A_HOM, om, Cm, 1.0))
            curve.append(row)
        if it == n_it:
            break
        dC = (fc0_t - f0) + (fct_t - ft)
        Cm = np.where(mask, np.maximum(0.0, Cm + EPS * dC), 0.0)
    print(f"   {n_it} it in {time.time() - t0:.0f}s")
    return Cm, np.array(curve)


masks = {"todos": off.copy()}
thr = np.quantile(SC[off], 0.80)
masks["SC top 20%"] = (SC >= thr) & off

# ---------------------------------------------------------------- split-half
A_idx, B_idx = [0, 2, 4, 6], [1, 3, 5]
fc0A, fctA, omA = targets(A_idx)
fc0B, fctB, omB = targets(B_idx)
ceil = score(fc0A, fc0B)
print(f"noise ceiling (FC grupo A -> grupo B): r={ceil[0]:.3f}  mse={ceil[1]:.4f}")
sc_test = score(model(G0 * SC, omB)[0], fc0B)
print(f"SC (G={G0}, a={A_HOM}) en B: r={sc_test[0]:.3f}  mse={sc_test[1]:.4f}")
curves = {}
for name, mask in masks.items():
    print(f"== split-half, máscara '{name}' ({mask.sum()} pesos libres)")
    _, cur = fit_ec(fc0A, fctA, omA, mask, tests=[(fc0B, fctB)])
    curves[name] = cur
    best = np.argmax(cur[:, 3])
    print(f"   final: train r={cur[-1, 1]:.3f}  test r={cur[-1, 3]:.3f} mse={cur[-1, 4]:.4f} | "
          f"mejor test r={cur[best, 3]:.3f} en it={int(cur[best, 0])} | delta final={cur[-1, -1]:.4f}")

# choose mask and iteration count by test performance
name_best = max(curves, key=lambda k: curves[k][:, 3].max())
it_best = int(curves[name_best][np.argmax(curves[name_best][:, 3]), 0])
print(f"elegido: máscara '{name_best}', {it_best} iteraciones")

# ---------------------------------------------------------------- full fit
fc0T, fctT, om = targets(range(len(filt)))
print("== ajuste completo (7 sujetos)")
EC, cur_full = fit_ec(fc0T, fctT, om, masks[name_best], n_it=max(it_best, REC))
f0, ft = model(EC, om)
print(f"   r(FC)={score(f0, fc0T)[0]:.3f}  r(FCtau)={np.corrcoef(ft[off], fctT[off])[0, 1]:.3f}")
m = masks[name_best]
print(f"   r(EC, SC) en la máscara = {np.corrcoef(EC[m], SC[m])[0, 1]:.3f};  "
      f"fracción de pesos en 0 = {np.mean(EC[m] == 0):.2f}")
asym = np.abs(EC - EC.T)[iu].sum() / (EC + EC.T)[iu].sum()
print(f"   asimetría |EC-EC^T|/(EC+EC^T) = {asym:.3f}")
delta = hopf.distance_to_criticality(A_HOM, om, EC, 1.0)
print(f"   delta = {delta:.4f}")

# ---------------------------------------------------------------- nonlinear check
FC_emp = fc0T
FCD_emp = np.concatenate([M.fcd(x, TR) for x in filt])
meta_emp = np.array([M.metastability(x) for x in filt])
n_TR = filt[0].shape[1]


def nonlinear(Cm, n_runs=7, seed0=300):
    fcs, fcds, metas = [], [], []
    for s in range(n_runs):
        x = M.filter_ts(hopf.simulate(A_HOM, om, Cm, 1.0, BETA, T=n_TR * TR, TR=TR, seed=seed0 + s), TR, BAND)
        fcs.append(M.fc(x)); fcds.append(M.fcd(x, TR)); metas.append(M.metastability(x))
    FCs = np.mean(fcs, axis=0)
    return {"r": score(FCs, FC_emp)[0], "meanfc": FCs[iu].mean(), "ks": M.ks(np.concatenate(fcds), FCD_emp),
            "meta": float(np.mean(metas)), "fcd": np.concatenate(fcds), "FC": FCs}


print("== simulaciones no lineales")
NL = {"SC": nonlinear(G0 * SC), "EC": nonlinear(EC)}
print(f"   empírico  FC media={FC_emp[iu].mean():.3f}  metaest.={meta_emp.mean():.3f}±{meta_emp.std():.3f}")
for k, v in NL.items():
    print(f"   {k:8s}  r(FC)={v['r']:.3f}  FC media={v['meanfc']:.3f}  KS(FCD)={v['ks']:.3f}  metaest.={v['meta']:.3f}")

np.savez(ROOT / "results" / "fase3_ec.npz", EC=EC, SC=SC, om=om, a=A_HOM, mask=masks[name_best],
         mask_name=name_best, it_best=it_best, ceil=np.array(ceil), sc_test=np.array(sc_test),
         curve_todos=curves["todos"], curve_top=curves["SC top 20%"], delta=delta, asym=asym,
         nl=np.array([[v["r"], v["meanfc"], v["ks"], v["meta"]] for v in NL.values()]), meta_emp=meta_emp)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for (k, cur), col in zip(curves.items(), ["C0", "C1"]):
    ax[0].plot(cur[:, 0], cur[:, 1], "--", color=col, label=f"{k} (train)")
    ax[0].plot(cur[:, 0], cur[:, 3], "-", color=col, label=f"{k} (test)")
ax[0].axhline(ceil[0], color="k", ls=":", label="techo de ruido A→B")
ax[0].axhline(sc_test[0], color="gray", ls="-.", label="SC (test)")
ax[0].set(xlabel="iteración", ylabel="r(FC)", title="Curvas de aprendizaje"); ax[0].legend(fontsize=7)
ax[1].scatter(SC[m], EC[m], s=3, alpha=0.4)
ax[1].set(xlabel="SC (normalizada)", ylabel="EC", title=f"EC vs SC (r={np.corrcoef(EC[m], SC[m])[0, 1]:.2f})")
bins = np.linspace(-0.2, 1, 60)
ax[2].hist(FCD_emp, bins, density=True, alpha=0.5, color="k", label="empírica")
for k, col in zip(NL, ["C0", "C3"]):
    ax[2].hist(NL[k]["fcd"], bins, density=True, histtype="step", lw=2, color=col, label=f"{k} (KS={NL[k]['ks']:.2f})")
ax[2].set(title="FCD (simulación no lineal)", xlabel="correlación entre ventanas"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fase3_ec_validacion.png", dpi=130)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].imshow(G0 * SC, cmap="viridis"); ax[0].set_title("SC × G")
ax[1].imshow(EC, cmap="viridis"); ax[1].set_title("EC (fila j recibe de columna k)")
ax[2].imshow(FC_emp, vmin=-0.3, vmax=1, cmap="RdBu_r"); ax[2].set_title("FC empírica")
ax[3].imshow(NL["EC"]["FC"], vmin=-0.3, vmax=1, cmap="RdBu_r"); ax[3].set_title("FC simulada con EC")
fig.tight_layout(); fig.savefig(FIG / "fase3_matrices.png", dpi=130)
print("figures saved")
