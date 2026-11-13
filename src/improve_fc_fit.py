"""Fase 6: improve the FC fit of the linear diffusive model, then recompute kappa.

Model (Fase 5): FC = corr[(diag(theta) + L(C))^{-1}].  Equivalently the precision matrix of the
signals is diag(theta) + L(C): the model is a Gaussian graphical model whose precision is a
(shifted) graph Laplacian, so symmetric effective connectivity = minus the off-diagonal of a
Laplacian-constrained precision matrix (C >= 0  <=>  non-negative partial correlations).

Steps:
  0. Ceilings: individual test-retest (session 1+2 vs 3+4 of the same subject) and the
     group-template ceiling (group FC as predictor of an individual's FC).
  1. SC transformations: C = SC^alpha, extra weight on interhemispheric and homotopic
     (i, i+50) links (tractography underestimates callosal fibres); theta refit for each.
  2. Symmetric EC from the group FC of sessions 1-2 (all subjects), evaluated on sessions 3-4
     and on held-out subjects; Deco-style update C <- max(0, C + eps (FC_emp - FC_mod)).
  3. Individual kappa with the best group model: fit theta_s per subject/session; fit quality
     vs ceilings; reliability (ICC) and redundancy with mean FC.

Fit/evaluation use GSR (applied identically to model and data) unless stated.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcdata import gsr, prep_fc, laplacian, icc_31, iu, N, FC_all, SC_all, S_, K_  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
H = N // 2
inter = np.zeros((N, N), bool); inter[:H, H:] = True; inter |= inter.T
homo = np.zeros((N, N), bool)
for i in range(H):
    homo[i, i + H] = homo[i + H, i] = True


def fisher_mean(Fs):
    F = np.tanh(np.mean(np.arctanh(np.clip(np.asarray(Fs), -0.999999, 0.999999)), axis=0))
    np.fill_diagonal(F, 1.0)
    return F


def r_of(Fm, Ft, use_gsr=True):
    a, b = (gsr(Fm), gsr(Ft)) if use_gsr else (Fm, Ft)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


def model_fc(theta, C):
    P = np.diag(np.broadcast_to(theta, (N,))) + laplacian(C)
    Sig = np.linalg.inv(P)
    s = np.sqrt(np.diag(Sig))
    return Sig / np.outer(s, s)


def fit_theta(C, Ft, use_gsr=True):
    T = gsr(Ft)[iu] if use_gsr else Ft[iu]
    lam, U = np.linalg.eigh(laplacian(C))
    lam = np.clip(lam, 0, None)

    def fc(th):
        Sig = (U / (th + lam)) @ U.T
        s = np.sqrt(np.diag(Sig))
        return Sig / np.outer(s, s)

    def loss(lt):
        m = fc(np.exp(lt))
        m = gsr(m)[iu] if use_gsr else m[iu]
        return np.mean((m - T) ** 2)

    r = minimize_scalar(loss, bounds=(np.log(1e-5), np.log(1e3)), method="bounded")
    th = float(np.exp(r.x))
    return th, r_of(fc(th), Ft, use_gsr)


def kappa(theta, C):
    mu = np.linalg.eigvalsh(np.diag(np.broadcast_to(theta, (N,))) + laplacian(C))
    return mu.min() / mu.mean()


def norm(C):
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    return C / C.sum(1).mean()


# subjects split for held-out evaluation; sessions split for test-retest
SUB_A, SUB_B = np.arange(0, 100), np.arange(100, 200)
F12 = fisher_mean([prep_fc(FC_all[s, k]) for s in range(S_) for k in (0, 1)])
F34 = fisher_mean([prep_fc(FC_all[s, k]) for s in range(S_) for k in (2, 3)])
F12A = fisher_mean([prep_fc(FC_all[s, k]) for s in SUB_A for k in (0, 1)])
FB = fisher_mean([prep_fc(FC_all[s, k]) for s in SUB_B for k in range(4)])
Find = np.array([[fisher_mean([prep_fc(FC_all[s, k]) for k in ks]) for ks in [(0, 1), (2, 3)]] for s in range(S_)])

# ---------------------------------------------------------------- 0. ceilings
retest = np.array([r_of(Find[s, 0], Find[s, 1]) for s in range(S_)])
templ = np.array([r_of(F12, Find[s, 1]) for s in range(S_)])
print("== techos (con GSR)")
print(f"   grupo: FC ses 1-2 vs 3-4 r={r_of(F12, F34):.4f}")
print(f"   individual test-retest (ses 1-2 vs 3-4, mismo sujeto): mediana r={np.median(retest):.3f}")
print(f"   plantilla de grupo -> individuo (ses 3-4):              mediana r={np.median(templ):.3f}")

# ---------------------------------------------------------------- 1. SC transformations
SCg = SC_all[SUB_A].mean(0)
print("== transformaciones de SC (ajuste en ses 1-2 de sujetos A; prueba: sujetos B, todas las sesiones)")
rows = []
for alpha in [1.0, 0.5, 0.25, "log"]:
    base = np.log1p(SCg) if alpha == "log" else SCg ** alpha
    base = norm(base)
    for w_inter in [1.0, 2.0, 4.0]:
        for w_homo in [1.0, 3.0, 10.0]:
            C = base.copy()
            C[inter] *= w_inter
            C[homo] *= w_homo
            C = norm(C)
            th, r_tr = fit_theta(C, F12A)
            r_te = r_of(model_fc(th, C), FB)
            rows.append((alpha, w_inter, w_homo, th, r_tr, r_te))
rows.sort(key=lambda x: -x[4])
for row in rows[:6] + [r for r in rows if r[0] == 1.0 and r[1] == 1.0 and r[2] == 1.0]:
    print(f"   alpha={str(row[0]):4s} inter x{row[1]:<3} homot x{row[2]:<4} theta={row[3]:.3f}  r train={row[4]:.3f}  r test(B)={row[5]:.3f}")
best_t = rows[0]
base = np.log1p(SCg) if best_t[0] == "log" else SCg ** best_t[0]
C_t = norm(base); C_t[inter] *= best_t[1]; C_t[homo] *= best_t[2]; C_t = norm(C_t)

# ---------------------------------------------------------------- 2. symmetric EC
print("== conectividad efectiva simétrica (ajuste: sujetos A, ses 1-2, SIN GSR; prueba: sujetos B)")
th0 = best_t[3]


def fit_ec(Ft, C0, theta, eps=0.02, n_it=4000, rec=200, tests=()):
    C = C0.copy()
    curve = []
    t0 = time.time()
    for it in range(n_it + 1):
        Fm = model_fc(theta, C)
        if it % rec == 0:
            curve.append([it, r_of(Fm, Ft), r_of(Fm, Ft, False)] + [r_of(Fm, T) for T in tests])
        if it == n_it:
            break
        D = Ft - Fm
        np.fill_diagonal(D, 0)
        C = np.maximum(0.0, C + eps * D)
        C = (C + C.T) / 2
    print(f"   {n_it} it en {time.time() - t0:.0f}s")
    return C, np.array(curve)


EC, curve = fit_ec(F12A, C_t, th0, tests=(FB,))
best_it = int(curve[np.argmax(curve[:, 3]), 0])
print(f"   final: train r(GSR)={curve[-1, 1]:.3f}  r(sin GSR)={curve[-1, 2]:.3f}  test B r(GSR)={curve[-1, 3]:.3f}"
      f"  | mejor test B={curve[:, 3].max():.3f} en it={best_it}")
th_ec, r_ec = fit_theta(EC, F12A)
print(f"   theta reajustado={th_ec:.4f} (inicial {th0:.4f});  r train={r_ec:.3f}  r test B={r_of(model_fc(th_ec, EC), FB):.3f}")
print(f"   EC: r(EC, SC transformada)={np.corrcoef(EC[iu], C_t[iu])[0, 1]:.3f}  fracción en 0={np.mean(EC[iu] == 0):.2f}")

# ---------------------------------------------------------------- 3. individual kappa
print("== kappa individual (theta_s por sujeto y sesión; modelo de grupo fijo)")
models = {"SC cruda": norm(SC_all.mean(0)), "SC transformada": C_t, "EC simétrica": EC}
meanfc = np.array([[prep_fc(FC_all[s, k])[iu].mean() for k in range(K_)] for s in range(S_)])
day = lambda X: np.stack([X[:, :2].mean(1), X[:, 2:].mean(1)], 1)
out = {}
for name, C in models.items():
    TH = np.zeros((S_, K_)); RR = np.zeros_like(TH)
    for s in range(S_):
        for k in range(K_):
            TH[s, k], RR[s, k] = fit_theta(C, prep_fc(FC_all[s, k]))
    KA = np.array([[kappa(t, C) for t in row] for row in TH])
    lk = np.log(KA)
    b = np.polyfit(meanfc.ravel(), lk.ravel(), 1)
    res_ = lk - np.polyval(b, meanfc)
    out[name] = dict(TH=TH, RR=RR, KA=KA, icc=icc_31(lk), iccd=icc_31(day(lk)), rho=np.corrcoef(lk.ravel(), meanfc.ravel())[0, 1],
                     iccr=icc_31(res_))
    o = out[name]
    print(f"   {name:16s} r(FC) mediana={np.median(RR):.3f}  kappa mediana={np.median(KA):.4f}  ICC4={o['icc']:.3f}  "
          f"ICCdía={o['iccd']:.3f}  r(FC media)={o['rho']:+.3f}  ICC sin FC media={o['iccr']:.3f}")
print(f"   (referencia: ses-a-ses del mismo sujeto, mediana r={np.median([r_of(prep_fc(FC_all[s,0]), prep_fc(FC_all[s,2])) for s in range(S_)]):.3f}; "
      f"plantilla de grupo -> sesión, mediana r={np.median([r_of(F12, prep_fc(FC_all[s,2])) for s in range(S_)]):.3f})")
# agreement of kappa across models (is the ranking of subjects model-dependent?)
lkm = {k: np.log(v["KA"]).mean(1) for k, v in out.items()}
print(f"   r entre sujetos de log kappa: cruda~transf={np.corrcoef(lkm['SC cruda'], lkm['SC transformada'])[0,1]:.3f}  "
      f"cruda~EC={np.corrcoef(lkm['SC cruda'], lkm['EC simétrica'])[0,1]:.3f}  transf~EC={np.corrcoef(lkm['SC transformada'], lkm['EC simétrica'])[0,1]:.3f}")

np.savez(ROOT / "results" / "fase6_fc_fit.npz", EC=EC, C_t=C_t, curve=curve, best_t=np.array(best_t[1:], float),
         best_alpha=str(best_t[0]), retest=retest, templ=templ,
         **{f"{k}_{m}": v[m] for k, v in out.items() for m in ["TH", "RR", "KA"]},
         stats=np.array([[out[k][m] for m in ["icc", "iccd", "rho", "iccr"]] for k in models]), names=np.array(list(models)))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
ax[0].plot(curve[:, 0], curve[:, 1], "--", label="train (sujetos A, ses 1-2)")
ax[0].plot(curve[:, 0], curve[:, 3], "-", label="test (sujetos B)")
ax[0].axhline(r_of(F12, F34), color="k", ls=":", label="techo de grupo")
ax[0].set(xlabel="iteración", ylabel="r(FC) con GSR", title="EC simétrica: aprendizaje"); ax[0].legend(fontsize=8)
for name, col in zip(models, ["gray", "C0", "C3"]):
    ax[1].hist(np.median(out[name]["RR"], 1), bins=30, alpha=0.5, color=col, label=name)
ax[1].axvline(np.median(templ), color="k", ls=":", label="plantilla de grupo")
ax[1].axvline(np.median(retest), color="k", ls="--", label="test-retest")
ax[1].set(xlabel="r(FC) individual (mediana por sujeto)", title="Ajuste individual"); ax[1].legend(fontsize=7)
lk = np.log10(day(out["EC simétrica"]["KA"]))
ax[2].scatter(lk[:, 0], lk[:, 1], s=8)
lo, hi = ax[2].get_xlim(); ax[2].plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax[2].set(xlabel=r"$\log_{10}\kappa$ día 1", ylabel="día 2", title=f"κ con EC: ICC día={out['EC simétrica']['iccd']:.2f}")
x = np.arange(len(models)); w = 0.25
for j, (m, lab) in enumerate([("icc", "ICC 4 ses."), ("iccd", "ICC día"), ("iccr", "ICC sin FC media")]):
    ax[3].bar(x + (j - 1) * w, [out[k][m] for k in models], w, label=lab)
ax[3].set_xticks(x); ax[3].set_xticklabels(list(models), fontsize=8); ax[3].set(ylim=(0, 1), title="Reproducibilidad de κ"); ax[3].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fase6_ajuste_fc.png", dpi=130)
print("figures saved")
