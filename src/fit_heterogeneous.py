"""Fase 2: region-specific bifurcation parameters a_j.

Each region's a_j sets its damping, hence the decay of its BOLD autocorrelation.
We fit (a_1..a_N, G) to the empirical filtered FC *and* regional autocorrelations
using the analytic statistics of the linearized model (hopf.FilteredSpectrum), with
L-BFGS on theta = (log(-a_j), log G).

    loss = MSE(FC upper triangle) + LAM * MSE(autocorrelations at lags) + MU * Var(log(-a))

Validation:
  * split-half: fit on 4 subjects, evaluate on the other 3; compare against the
    homogeneous model and against spatially shuffled a_j.
  * full fit on 7 subjects, then nonlinear simulations: FC, FCD (KS), metastability.

Outputs: results/fase2_heterogeneous.npz, figures/fase2_*.png
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
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
LAGS = [3, 7, 14, 28]          # in TRs (2.2, 5.0, 10.1, 20.2 s)
LAM, MU = 1.0, 1e-3
BETA = 0.02
KIND = "diffusive"

g = data.load_group()
C, TR = g["C"], g["TR"]
N = C.shape[0]
iu = np.triu_indices(N, 1)
filt = [M.filter_ts(ts, TR, BAND) for ts in g["ts"]]
freqs_all = [M.peak_frequencies(ts, TR, BAND) for ts in g["ts"]]
FS = hopf.FilteredSpectrum(TR, band=BAND, lags_TR=[0] + LAGS)


def targets(idx):
    fc = np.tanh(np.mean([np.arctanh(np.clip(M.fc(filt[i]), -0.999999, 0.999999)) for i in idx], axis=0))
    ac = np.mean([M.autocorr(filt[i], LAGS) for i in idx], axis=0)
    om = 2 * np.pi * np.mean([freqs_all[i] for i in idx], axis=0)
    return fc[iu], ac, om


def model_stats(a, G, om):
    FC, AC = FS.stats(a, om, C, G, kind=KIND)
    return FC[iu], AC


def loss_parts(a, G, om, fc_t, ac_t):
    fc, ac = model_stats(a, G, om)
    return float(np.mean((fc - fc_t) ** 2)), float(np.mean((ac - ac_t) ** 2)), float(np.corrcoef(fc, fc_t)[0, 1])


def fit_homogeneous(om, fc_t, ac_t):
    best = None
    for la in np.linspace(np.log(0.002), np.log(0.5), 15):
        for lG in np.linspace(np.log(0.05), np.log(6), 25):
            a, G = -np.exp(la), np.exp(lG)
            f, c, _ = loss_parts(a, G, om, fc_t, ac_t)
            L = f + LAM * c
            if best is None or L < best[0]:
                best = (L, a, G)
    return best[1], best[2]


def fit_heterogeneous(om, fc_t, ac_t, a0, G0, maxiter=150):
    def obj(th):
        a, G = -np.exp(th[:N]), np.exp(th[N])
        f, c, _ = loss_parts(a, G, om, fc_t, ac_t)
        return f + LAM * c + MU * np.var(th[:N])

    th0 = np.r_[np.full(N, np.log(-a0)), np.log(G0)]
    bounds = [(np.log(1e-4), np.log(1.0))] * N + [(np.log(0.01), np.log(10.0))]
    t0 = time.time()
    res = minimize(obj, th0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "eps": 1e-4})
    print(f"   L-BFGS: {res.nit} it, {res.nfev} evals, {time.time() - t0:.0f}s, loss {obj(th0):.4f} -> {res.fun:.4f}")
    return -np.exp(res.x[:N]), np.exp(res.x[N])


def report(tag, a, G, om, fc_t, ac_t):
    f, c, r = loss_parts(a, G, om, fc_t, ac_t)
    print(f"   {tag:28s} MSE_FC={f:.4f}  MSE_AC={c:.4f}  r(FC)={r:.3f}")
    return f, c, r


# ---------------------------------------------------------------- split-half validation
rng = np.random.default_rng(0)
A_idx, B_idx = [0, 2, 4, 6], [1, 3, 5]
fcA, acA, omA = targets(A_idx)
fcB, acB, omB = targets(B_idx)
print("== split-half: fit on", [g["sids"][i] for i in A_idx], "test on", [g["sids"][i] for i in B_idx])
a_h, G_h = fit_homogeneous(omA, fcA, acA)
print(f"   homogeneous fit: a={a_h:.4f}  G={G_h:.3f}")
a_het, G_het = fit_heterogeneous(omA, fcA, acA, a_h, G_h)
cv = {}
print(" train (A):")
cv["hom_train"] = report("homogéneo", a_h, G_h, omA, fcA, acA)
cv["het_train"] = report("heterogéneo", a_het, G_het, omA, fcA, acA)
print(" test (B):")
cv["hom_test"] = report("homogéneo", a_h, G_h, omB, fcB, acB)
cv["het_test"] = report("heterogéneo", a_het, G_het, omB, fcB, acB)
shuf = np.array([report(f"heterog. barajado #{k}", rng.permutation(a_het), G_het, omB, fcB, acB) for k in range(20)])
cv["shuf_test_mean"] = shuf.mean(0)
print(f"   barajado (media de 20)       MSE_FC={shuf[:, 0].mean():.4f}  MSE_AC={shuf[:, 1].mean():.4f}  r(FC)={shuf[:, 2].mean():.3f}")
# reliability of the a_j map: fit on B too and correlate log(-a)
a_hB, G_hB = fit_homogeneous(omB, fcB, acB)
a_hetB, G_hetB = fit_heterogeneous(omB, fcB, acB, a_hB, G_hB)
rel = np.corrcoef(np.log(-a_het), np.log(-a_hetB))[0, 1]
print(f"   reproducibilidad del mapa a_j entre mitades: r(log|a| A, log|a| B) = {rel:.3f}")

# ---------------------------------------------------------------- full fit
print("== full fit (7 subjects)")
all_idx = list(range(len(filt)))
fcT, acT, om = targets(all_idx)
a_h7, G_h7 = fit_homogeneous(om, fcT, acT)
print(f"   homogeneous fit: a={a_h7:.4f}  G={G_h7:.3f}")
a_f, G_f = fit_heterogeneous(om, fcT, acT, a_h7, G_h7, maxiter=250)
report("homogéneo", a_h7, G_h7, om, fcT, acT)
report("heterogéneo", a_f, G_f, om, fcT, acT)
print(f"   a_j: min={a_f.min():.4f} median={np.median(a_f):.4f} max={a_f.max():.4f};  G={G_f:.3f}")
delta_h = hopf.distance_to_criticality(a_h7, om, C, G_h7, KIND)
delta_f = hopf.distance_to_criticality(a_f, om, C, G_f, KIND)
print(f"   delta: homogéneo={delta_h:.4f} (tau={1/delta_h:.0f}s)  heterogéneo={delta_f:.4f} (tau={1/delta_f:.0f}s)")

# ---------------------------------------------------------------- nonlinear validation
FC_emp = np.eye(N); FC_emp[iu] = fcT; FC_emp = FC_emp + np.triu(FC_emp, 1).T
FCD_emp = np.concatenate([M.fcd(x, TR) for x in filt])
meta_emp = np.array([M.metastability(x) for x in filt])
n_TR = filt[0].shape[1]


def nonlinear(a, G, n_runs=7, seed0=200):
    fcs, fcds, metas, acs = [], [], [], []
    for s in range(n_runs):
        x = M.filter_ts(hopf.simulate(a, om, C, G, BETA, T=n_TR * TR, TR=TR, kind=KIND, seed=seed0 + s), TR, BAND)
        fcs.append(M.fc(x)); fcds.append(M.fcd(x, TR)); metas.append(M.metastability(x))
        acs.append(M.autocorr(x, LAGS))
    FCs = np.mean(fcs, axis=0)
    return {"r": M.fc_fit(FCs, FC_emp)["r"], "meanfc": M.triu(FCs).mean(),
            "ks": M.ks(np.concatenate(fcds), FCD_emp), "meta": float(np.mean(metas)),
            "fcd": np.concatenate(fcds), "ac": np.mean(acs, axis=0), "FC": FCs}


print("== nonlinear simulations (7 runs each)")
NL = {"homogéneo": nonlinear(a_h7, G_h7), "heterogéneo": nonlinear(a_f, G_f)}
print(f"   empírico                    meanFC={M.triu(FC_emp).mean():.3f}  metast={meta_emp.mean():.3f}±{meta_emp.std():.3f}")
for k, v in NL.items():
    print(f"   {k:27s} r(FC)={v['r']:.3f}  meanFC={v['meanfc']:.3f}  KS(FCD)={v['ks']:.3f}  metast={v['meta']:.3f}")

np.savez(ROOT / "results" / "fase2_heterogeneous.npz", a_f=a_f, G_f=G_f, a_h7=a_h7, G_h7=G_h7,
         a_het_A=a_het, a_het_B=a_hetB, rel=rel, LAGS=LAGS, acT=acT, fcT=fcT, om=om,
         cv=np.array([cv[k] for k in ["hom_train", "het_train", "hom_test", "het_test", "shuf_test_mean"]]),
         nl=np.array([[v["r"], v["meanfc"], v["ks"], v["meta"]] for v in NL.values()]),
         meta_emp=meta_emp, delta=np.array([delta_h, delta_f]))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
labels = ["homog.\ntrain", "heterog.\ntrain", "homog.\ntest", "heterog.\ntest", "barajado\ntest"]
vals = np.array([cv[k] for k in ["hom_train", "het_train", "hom_test", "het_test", "shuf_test_mean"]])
ax[0].bar(labels, vals[:, 0], color=["C0", "C1", "C0", "C1", "gray"]); ax[0].set_title("MSE(FC)")
ax[1].bar(labels, vals[:, 1], color=["C0", "C1", "C0", "C1", "gray"]); ax[1].set_title("MSE(autocorrelación)")
ax[2].scatter(np.log10(-a_het), np.log10(-a_hetB), s=12)
ax[2].set(xlabel=r"$\log_{10}|a_j|$ (mitad A)", ylabel=r"$\log_{10}|a_j|$ (mitad B)", title=f"Reproducibilidad del mapa, r={rel:.2f}")
strength = C.sum(1)
ax[3].scatter(strength, np.log10(-a_f), s=12)
ax[3].set(xlabel="fuerza estructural del nodo", ylabel=r"$\log_{10}|a_j|$ (ajuste completo)",
          title=f"r={np.corrcoef(strength, np.log10(-a_f))[0, 1]:.2f}")
fig.tight_layout(); fig.savefig(FIG / "fase2_validacion_cruzada.png", dpi=130)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
bins = np.linspace(-0.2, 1, 60)
ax[0].hist(FCD_emp, bins, density=True, alpha=0.5, label="empírica", color="k")
for k, col in zip(NL, ["C0", "C1"]):
    ax[0].hist(NL[k]["fcd"], bins, density=True, histtype="step", lw=2, color=col, label=f"{k} (KS={NL[k]['ks']:.2f})")
ax[0].set(title="Distribución de FCD", xlabel="correlación entre ventanas"); ax[0].legend(fontsize=8)
lag_s = np.array(LAGS) * TR
ax[1].plot(lag_s, acT.mean(1), "ko-", label="empírica")
for k, col in zip(NL, ["C0", "C1"]):
    ax[1].plot(lag_s, NL[k]["ac"].mean(1), "o-", color=col, label=k)
ax[1].set(xlabel="retardo (s)", ylabel="autocorrelación media", title="Autocorrelación (simulación)"); ax[1].legend(fontsize=8)
k1 = 2  # lag index 14 TR
ax[2].scatter(acT[k1], NL["homogéneo"]["ac"][k1], s=10, label="homogéneo")
ax[2].scatter(acT[k1], NL["heterogéneo"]["ac"][k1], s=10, label="heterogéneo")
lo, hi = ax[2].get_xlim(); ax[2].plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax[2].set(xlabel=f"AC empírica ({lag_s[k1]:.0f} s)", ylabel="AC simulada", title="Autocorrelación por región"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fase2_no_lineal.png", dpi=130)
print("figures saved")
