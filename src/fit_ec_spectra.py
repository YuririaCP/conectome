"""Fase 4: joint fit of effective connectivity and regional spectra.

Each iteration:
  * EC step (as in Fase 3) on zero-lag and lagged filtered correlations;
  * spectral step per region, comparing the analytic band-passed PSD of x_j with the
    empirical Welch PSD (both normalized to unit area on 0-0.12 Hz):
        omega_j <- omega_j + 2 pi * ETA_W * (centroid_emp_j - centroid_mod_j)
        log|a_j| <- log|a_j| + ETA_A * (width_emp_j - width_mod_j) / width_emp_j
    (a larger |a_j| = stronger damping = broader peak).

Validation: split-half (fit on 4, test on 3) for FC and regional spectra; then a full
fit and nonlinear simulations: FCD and regional autocorrelations.

Outputs: results/fase4_ec_spectra.npz, figures/fase4_*.png
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import hopf  # noqa: E402
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
MOU = "--mou" in sys.argv  # omega_j = 0: non-oscillatory (Ornstein-Uhlenbeck) nodes
TAG = "_mou" if MOU else ""
BAND = (0.008, 0.08)
TAU = 2
A0, G0 = -0.0696, 3.3
EPS = 0.004
ETA = 0.3
SPEC_EVERY = 10
A_BOUNDS = (1e-3, 1.0)
OM_BOUNDS = (0.005, 0.12)  # Hz
N_IT = 3000
REC = 100
BETA = 0.02
AC_LAGS = [3, 7, 14, 28]

g = data.load_group()
SC, TR = g["C"], g["TR"]
N = SC.shape[0]
off = ~np.eye(N, dtype=bool)
iu = np.triu_indices(N, 1)
filt = [M.filter_ts(ts, TR, BAND) for ts in g["ts"]]
FS = hopf.FilteredSpectrum(TR, band=BAND, lags_TR=[0, TAU])
fw, _ = signal.welch(np.zeros(filt[0].shape[1]), fs=1 / TR, nperseg=512)
SEL = fw <= 0.12
F = fw[SEL]
PSD_subj = []
for x in filt:
    _, p = signal.welch(x, fs=1 / TR, nperseg=512, axis=1)
    PSD_subj.append(p[:, SEL] / p[:, SEL].sum(1, keepdims=True))
PSD_subj = np.array(PSD_subj)


def moments(P):
    P = P / P.sum(1, keepdims=True)
    c = (P * F).sum(1)
    w = np.sqrt((P * (F[None, :] - c[:, None]) ** 2).sum(1))
    return c, w


def lagged_corr(x, k):
    x = x - x.mean(axis=1, keepdims=True)
    T = x.shape[1]
    S0 = x @ x.T / T
    Sk = x[:, k:] @ x[:, :T - k].T / (T - k)
    d = np.sqrt(np.diag(S0))
    return S0 / np.outer(d, d), Sk / np.outer(d, d)


def targets(idx):
    idx = list(idx)
    pairs = [lagged_corr(filt[i], TAU) for i in idx]
    fc0 = np.tanh(np.mean([np.arctanh(np.clip(p[0], -0.999999, 0.999999)) for p in pairs], axis=0))
    np.fill_diagonal(fc0, 1.0)
    fct = np.mean([p[1] for p in pairs], axis=0)
    P = PSD_subj[idx].mean(0)
    return fc0, fct, P


def model(Cm, a, om):
    (S0, Sk), P = FS.covs_and_psd(a, om, Cm, 1.0, F)
    d = np.sqrt(np.diag(S0))
    return S0 / np.outer(d, d), Sk / np.outer(d, d), P / P.sum(1, keepdims=True)


def fc_score(f0, t):
    return float(np.corrcoef(f0[iu], t[iu])[0, 1]), float(np.mean((f0[iu] - t[iu]) ** 2))


def spec_score(P, Pt):
    cm, wm = moments(P)
    ct, wt = moments(Pt)
    return (float(np.corrcoef(cm, ct)[0, 1]), float(np.corrcoef(wm, wt)[0, 1]),
            float(np.mean(np.abs(P - Pt).sum(1)) / 2))  # last: mean total-variation distance


def spectral_step(Cm, a, om, P, ct, wt, h=0.02):
    """Damped Newton step per region on (omega_j, log|a_j|) to match (centroid, width).
    Diagonal sensitivities from two global perturbations; steps clipped; parameters bounded."""
    c0, w0 = moments(P)
    la = np.log(-a)
    if MOU:
        _, _, Pa = model(Cm, -np.exp(la + h), om)
        _, wa = moments(Pa)
        J = (wa - w0) / h
        d_la = np.where(np.abs(J) > 1e-12, ETA * (wt - w0) / np.where(np.abs(J) > 1e-12, J, 1), 0.0)
        la = np.clip(la + np.clip(d_la, -0.2, 0.2), *np.log(A_BOUNDS))
        return om, -np.exp(la)
    _, _, Pw = model(Cm, a, om * (1 + h))
    _, _, Pa = model(Cm, -np.exp(la + h), om)
    cw, ww = moments(Pw)
    ca, wa = moments(Pa)
    J11, J21 = (cw - c0) / (h * om), (ww - w0) / (h * om)
    J12, J22 = (ca - c0) / h, (wa - w0) / h
    det = J11 * J22 - J12 * J21
    ok = np.abs(det) > 1e-12
    rc, rw = ETA * (ct - c0), ETA * (wt - w0)
    d_om = np.where(ok, (J22 * rc - J12 * rw) / np.where(ok, det, 1), 0.0)
    d_la = np.where(ok, (-J21 * rc + J11 * rw) / np.where(ok, det, 1), 0.0)
    d_om = np.clip(d_om, -0.1 * om, 0.1 * om)
    d_la = np.clip(d_la, -0.2, 0.2)
    om = np.clip(om + d_om, 2 * np.pi * OM_BOUNDS[0], 2 * np.pi * OM_BOUNDS[1])
    la = np.clip(la + d_la, *np.log(A_BOUNDS))
    return om, -np.exp(la)


def fit(fc0_t, fct_t, P_t, om0, spectra=True, tests=None, n_it=N_IT):
    Cm = G0 * SC.copy()
    a = np.full(N, A0)
    om = om0.copy()
    ct, wt = moments(P_t)
    curve = []
    t0 = time.time()
    for it in range(n_it + 1):
        f0, ft, P = model(Cm, a, om)
        if it % REC == 0:
            row = [it, *fc_score(f0, fc0_t), *spec_score(P, P_t)]
            for (fb, Pb) in (tests or []):
                row += [*fc_score(f0, fb), *spec_score(P, Pb)]
            curve.append(row)
        if it == n_it:
            break
        Cm = np.where(off, np.maximum(0.0, Cm + EPS * ((fc0_t - f0) + (fct_t - ft))), 0.0)
        if spectra and it % SPEC_EVERY == 0:
            om, a = spectral_step(Cm, a, om, P, ct, wt)
    print(f"   {n_it} it en {time.time() - t0:.0f}s")
    return Cm, a, om, np.array(curve)


om_init = 2 * np.pi * np.mean([M.peak_frequencies(ts, TR, BAND) for ts in g["ts"]], axis=0)
if MOU:
    om_init = np.zeros(N)

# ---------------------------------------------------------------- split-half
A_idx, B_idx = [0, 2, 4, 6], [1, 3, 5]
fc0A, fctA, PA = targets(A_idx)
fc0B, fctB, PB = targets(B_idx)
ceil_fc = fc_score(fc0A, fc0B)
ceil_sp = spec_score(PA, PB)
print(f"techo de ruido A->B: FC r={ceil_fc[0]:.3f} | centroide r={ceil_sp[0]:.3f} ancho r={ceil_sp[1]:.3f} TV={ceil_sp[2]:.3f}")
curves = {}
for name, sp in [("solo EC", False), ("EC + espectros", True)]:
    print(f"== split-half: {name}")
    _, a_, om_, cur = fit(fc0A, fctA, PA, om_init, spectra=sp, tests=[(fc0B, PB)])
    curves[name] = cur
    l = cur[-1]
    print(f"   train: FC r={l[1]:.3f} | centr r={l[3]:.3f} ancho r={l[4]:.3f} TV={l[5]:.3f}")
    print(f"   test : FC r={l[6]:.3f} | centr r={l[8]:.3f} ancho r={l[9]:.3f} TV={l[10]:.3f}")

# ---------------------------------------------------------------- full fit
print("== ajuste completo")
fc0T, fctT, PT = targets(range(len(filt)))
res = {}
for name, sp in [("solo EC", False), ("EC + espectros", True)]:
    Cm, a_, om_, cur = fit(fc0T, fctT, PT, om_init, spectra=sp)
    res[name] = (Cm, a_, om_)
    l = cur[-1]
    print(f"   {name:15s} FC r={l[1]:.3f} | centr r={l[3]:.3f} ancho r={l[4]:.3f} TV={l[5]:.3f}")
EC, a_f, om_f = res["EC + espectros"]
print(f"   a_j: mediana={np.median(a_f):.4f} rango=[{a_f.min():.4f}, {a_f.max():.4f}];  "
      f"f_j: {om_f.min() / 2 / np.pi:.4f}-{om_f.max() / 2 / np.pi:.4f} Hz")
print(f"   en límites: a_j {np.sum(-a_f <= A_BOUNDS[0] * 1.01) + np.sum(-a_f >= A_BOUNDS[1] * 0.99)}/{N};  "
      f"f_j {np.sum(om_f <= 2 * np.pi * OM_BOUNDS[0] * 1.01) + np.sum(om_f >= 2 * np.pi * OM_BOUNDS[1] * 0.99)}/{N}")
print(f"   delta = {hopf.distance_to_criticality(a_f, om_f, EC, 1.0):.4f}")

# ---------------------------------------------------------------- nonlinear validation
FCD_emp = np.concatenate([M.fcd(x, TR) for x in filt])
AC_emp = np.mean([M.autocorr(x, AC_LAGS) for x in filt], axis=0)
meta_emp = np.array([M.metastability(x) for x in filt])
n_TR = filt[0].shape[1]


def nonlinear(Cm, a, om, n_runs=7, seed0=400):
    fcs, fcds, metas, acs = [], [], [], []
    for s in range(n_runs):
        x = M.filter_ts(hopf.simulate(a, om, Cm, 1.0, BETA, T=n_TR * TR, TR=TR, seed=seed0 + s), TR, BAND)
        fcs.append(M.fc(x)); fcds.append(M.fcd(x, TR)); metas.append(M.metastability(x))
        acs.append(M.autocorr(x, AC_LAGS))
    FCs = np.mean(fcs, axis=0)
    ac = np.mean(acs, axis=0)
    return {"r": fc_score(FCs, fc0T)[0], "meanfc": FCs[iu].mean(), "ks": M.ks(np.concatenate(fcds), FCD_emp),
            "meta": float(np.mean(metas)), "fcd": np.concatenate(fcds),
            "r_ac": [float(np.corrcoef(ac[k], AC_emp[k])[0, 1]) for k in range(len(AC_LAGS))]}


print("== simulaciones no lineales")
NL = {k: nonlinear(*v) for k, v in res.items()}
print(f"   empírico        FC media={fc0T[iu].mean():.3f}  metaest.={meta_emp.mean():.3f}")
for k, v in NL.items():
    print(f"   {k:15s} r(FC)={v['r']:.3f}  FC media={v['meanfc']:.3f}  KS(FCD)={v['ks']:.3f}  "
          f"metaest.={v['meta']:.3f}  r(AC regional, lags {AC_LAGS})={np.round(v['r_ac'], 2)}")

np.savez(ROOT / "results" / f"fase4_ec_spectra{TAG}.npz", EC=EC, a=a_f, om=om_f, F=F, PT=PT,
         ceil_fc=np.array(ceil_fc), ceil_sp=np.array(ceil_sp),
         curve_ec=curves["solo EC"], curve_sp=curves["EC + espectros"],
         nl=np.array([[v["r"], v["meanfc"], v["ks"], v["meta"], *v["r_ac"]] for v in NL.values()]),
         meta_emp=meta_emp)

# ---------------------------------------------------------------- figures
_, _, Pm = model(EC, a_f, om_f)
_, _, Pe0 = model(*res["solo EC"])
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
ce, we = moments(PT)
for P, lab, col in [(Pe0, "solo EC", "C0"), (Pm, "EC + espectros", "C3")]:
    cm, wm = moments(P)
    ax[0].scatter(ce, cm, s=10, color=col, label=f"{lab} (r={np.corrcoef(ce, cm)[0, 1]:.2f})")
    ax[1].scatter(we, wm, s=10, color=col, label=f"{lab} (r={np.corrcoef(we, wm)[0, 1]:.2f})")
for k, t in [(0, "Centroide espectral (Hz)"), (1, "Ancho espectral (Hz)")]:
    lo, hi = ax[k].get_xlim(); ax[k].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[k].set(xlabel="empírico", ylabel="modelo", title=t); ax[k].legend(fontsize=8)
for j, col in zip([int(np.argmin(ce)), int(np.argmax(ce))], ["C2", "C4"]):
    ax[2].plot(F, PT[j], color=col, lw=2, label=f"región {j} emp.")
    ax[2].plot(F, Pm[j], "--", color=col, label=f"región {j} modelo")
ax[2].set(xlabel="frecuencia (Hz)", title="Espectros: regiones extremas"); ax[2].legend(fontsize=7)
bins = np.linspace(-0.2, 1, 60)
ax[3].hist(FCD_emp, bins, density=True, alpha=0.5, color="k", label="empírica")
for k, col in zip(NL, ["C0", "C3"]):
    ax[3].hist(NL[k]["fcd"], bins, density=True, histtype="step", lw=2, color=col, label=f"{k} (KS={NL[k]['ks']:.2f})")
ax[3].set(title="FCD (simulación no lineal)", xlabel="correlación entre ventanas"); ax[3].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / f"fase4_espectros_fcd{TAG}.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (k, cur), col in zip(curves.items(), ["C0", "C3"]):
    ax[0].plot(cur[:, 0], cur[:, 6], color=col, label=f"{k} (test)")
    ax[1].plot(cur[:, 0], cur[:, 10], color=col, label=f"{k} (test)")
ax[0].axhline(ceil_fc[0], color="k", ls=":", label="techo A→B"); ax[0].set(xlabel="iteración", ylabel="r(FC)", title="FC en sujetos de prueba")
ax[1].axhline(ceil_sp[2], color="k", ls=":", label="techo A→B"); ax[1].set(xlabel="iteración", ylabel="distancia TV media", title="Espectros regionales en prueba")
for x in ax: x.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / f"fase4_curvas{TAG}.png", dpi=130)
print("figures saved")
