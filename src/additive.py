"""Fase 7: additive vs diffusive coupling for zero-lag FC.

Closed forms (homogeneous a, homogeneous omega drops out; OU limit):
  diffusive: FC = corr[(theta I + L(C))^{-1}]       critical mode = uniform vector (L 1 = 0)
  additive : FC = corr[(I - rho Chat)^{-1}]          Chat = C / lambda_max(C),  rho = G lambda_max / |a| in (0, 1)
                                                     critical mode = Perron vector of C;  kappa = 1 - rho
Both precisions are diag(q) - C.  FC = corr(P^{-1}) is invariant under P -> D P D, so only the
partial correlations C_jk / sqrt(q_j q_k) are identifiable: with *free* EC the two coupling types
should be nearly indistinguishable; they differ only when C is tied to the SC.

1. SC-anchored comparison: C = SC^alpha (with callosal boosts), one dynamic parameter each
   (theta or rho); train on subjects 0-99 sessions 1-2, test on subjects 100-199.
2. Free symmetric EC for both models (same update rule) -> do fits differ?
3. Individual rho_s (additive, SC-anchored best and EC): reliability, mean-FC redundancy, GSR.
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


def corr(S):
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def r_of(Fm, Ft, use_gsr):
    a, b = (gsr(Fm), gsr(Ft)) if use_gsr else (Fm, Ft)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


class Model:
    """One-parameter family on a fixed C, via one eigendecomposition."""

    def __init__(self, C, kind):
        self.kind = kind
        if kind == "diffusive":
            M = laplacian(C / C.sum(1).mean())
            self.lam, self.U = np.linalg.eigh(M)
            self.lam = np.clip(self.lam, 0, None)
        else:
            lam, U = np.linalg.eigh(C)
            self.lam, self.U = lam / lam.max(), U

    def fc(self, p):  # p = theta (diffusive) or rho (additive)
        w = 1 / (p + self.lam) if self.kind == "diffusive" else 1 / (1 - p * self.lam)
        return corr((self.U * w) @ self.U.T)

    def kappa(self, p):
        mu = p + self.lam if self.kind == "diffusive" else 1 - p * self.lam
        return mu.min() / mu.mean()

    def fit(self, Ft, use_gsr):
        T = gsr(Ft)[iu] if use_gsr else Ft[iu]

        def loss(x):
            m = self.fc(np.exp(x) if self.kind == "diffusive" else 1 - np.exp(x))
            m = gsr(m)[iu] if use_gsr else m[iu]
            return np.mean((m - T) ** 2)

        r = minimize_scalar(loss, bounds=(np.log(1e-5), np.log(1e3) if self.kind == "diffusive" else np.log(0.9999)),
                            method="bounded")
        p = float(np.exp(r.x)) if self.kind == "diffusive" else float(1 - np.exp(r.x))
        return p, r_of(self.fc(p), Ft, use_gsr)


def build_C(SC, alpha, w_inter, w_homo):
    C = np.log1p(SC) if alpha == "log" else SC ** alpha
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    C[inter] *= w_inter
    C[homo] *= w_homo
    return C


SUB_A, SUB_B = np.arange(0, 100), np.arange(100, 200)
F12A = fisher_mean([prep_fc(FC_all[s, k]) for s in SUB_A for k in (0, 1)])
FB = fisher_mean([prep_fc(FC_all[s, k]) for s in SUB_B for k in range(4)])
SCg = SC_all[SUB_A].mean(0)

# ---------------------------------------------------------------- 1. SC-anchored comparison
print("== 1. C anclada a la SC (1 parámetro dinámico); entrenamiento: sujetos A ses 1-2; prueba: sujetos B")
grid = [(al, wi, wh) for al in [1.0, 0.5, 0.25, "log"] for wi in [1.0, 2.0, 4.0] for wh in [1.0, 3.0, 10.0]]
best = {}
for kind in ["diffusive", "additive"]:
    for use_gsr in [False, True]:
        rows = []
        for al, wi, wh in grid:
            m = Model(build_C(SCg, al, wi, wh), kind)
            p, rtr = m.fit(F12A, use_gsr)
            rows.append((rtr, r_of(m.fc(p), FB, use_gsr), p, m.kappa(p), (al, wi, wh)))
        rows.sort(key=lambda x: -x[0])
        b = rows[0]
        raw = [r for r in rows if r[4] == (1.0, 1.0, 1.0)][0]
        best[(kind, use_gsr)] = b
        par = "theta" if kind == "diffusive" else "rho"
        print(f"   {kind:9s} GSR={'sí' if use_gsr else 'no'}: mejor {b[4]}  r train={b[0]:.3f} test={b[1]:.3f}  {par}={b[2]:.4f} kappa={b[3]:.4f}"
              f"   | SC cruda: test={raw[1]:.3f} {par}={raw[2]:.4f}")

# ---------------------------------------------------------------- 2. free EC for both models
print("== 2. EC simétrica libre (misma regla), objetivo sin GSR; evaluación con y sin GSR en sujetos B")


def model_fc_free(C, kind, p):
    if kind == "diffusive":
        return corr(np.linalg.inv(p * np.eye(N) + laplacian(C)))
    return corr(np.linalg.inv(np.eye(N) - C))  # additive: rho absorbed into C, lambda_max(C) < 1


def fit_ec(Ft, C0, kind, p=None, eps=0.02, n_it=4000):
    C = C0.copy()
    t0 = time.time()
    for _ in range(n_it):
        D = Ft - model_fc_free(C, kind, p)
        np.fill_diagonal(D, 0)
        C = np.maximum(0.0, C + eps * D)
        C = (C + C.T) / 2
        if kind == "additive":
            lm = np.linalg.eigvalsh(C).max()
            if lm > 0.999:
                C *= 0.999 / lm
    return C, time.time() - t0


EC = {}
for kind in ["diffusive", "additive"]:
    b = best[(kind, False)]
    C0 = build_C(SCg, *b[4])
    if kind == "diffusive":
        C0 = C0 / C0.sum(1).mean()
        C, t = fit_ec(F12A, C0, kind, p=b[2])
        Fm = model_fc_free(C, kind, b[2])
        lmax = None
    else:
        C0 = C0 / np.linalg.eigvalsh(C0).max() * b[2]
        C, t = fit_ec(F12A, C0, kind)
        Fm = model_fc_free(C, kind, None)
        lmax = np.linalg.eigvalsh(C).max()
    EC[kind] = C
    from scipy.sparse.csgraph import connected_components
    ncomp = connected_components(C > 0, directed=False)[0]
    print(f"   {kind:9s} ({t:.0f}s): train r sin GSR={r_of(Fm, F12A, False):.3f} con GSR={r_of(Fm, F12A, True):.3f} | "
          f"test B sin GSR={r_of(Fm, FB, False):.3f} con GSR={r_of(Fm, FB, True):.3f} | ceros={np.mean(C[iu] == 0):.2f} "
          f"componentes={ncomp}" + (f"  rho=lambda_max={lmax:.4f}" if lmax is not None else ""))
pc = lambda C, kind: (lambda P: -P[iu] / np.sqrt(np.outer(np.diag(P), np.diag(P)))[iu])(
    (EC["diffusive"] * 0 + (best[("diffusive", False)][2] * np.eye(N) + laplacian(C))) if kind == "diffusive" else (np.eye(N) - C))
print(f"   correlaciones parciales implicadas: r(difusivo, aditivo) = {np.corrcoef(pc(EC['diffusive'], 'diffusive'), pc(EC['additive'], 'additive'))[0, 1]:.3f}")

# ---------------------------------------------------------------- 3. individual rho
print("== 3. rho individual (modelo aditivo de grupo fijo; 1 parámetro por sesión)")
meanfc = np.array([[prep_fc(FC_all[s, k])[iu].mean() for k in range(K_)] for s in range(S_)])
day = lambda X: np.stack([X[:, :2].mean(1), X[:, 2:].mean(1)], 1)
C_sc = build_C(SC_all.mean(0), *best[("additive", True)][4])
cases = {"aditivo SC": Model(C_sc, "additive"), "aditivo EC": Model(EC["additive"], "additive"),
         "difusivo SC": Model(build_C(SC_all.mean(0), *best[("diffusive", True)][4]), "diffusive")}
v = np.abs(np.linalg.eigh(C_sc)[1][:, -1])
print(f"   vector de Perron (aditivo SC): cos con uniforme = {v.sum() / np.sqrt(N) / np.linalg.norm(v):.3f}")
out = {}
for name, m in cases.items():
    for use_gsr in [False, True]:
        P = np.zeros((S_, K_)); R = np.zeros_like(P)
        for s in range(S_):
            for k in range(K_):
                P[s, k], R[s, k] = m.fit(prep_fc(FC_all[s, k]), use_gsr)
        KA = np.array([[m.kappa(p) for p in row] for row in P])
        lk = np.log(KA)
        bfit = np.polyfit(meanfc.ravel(), lk.ravel(), 1)
        o = dict(P=P, R=R, KA=KA, icc=icc_31(lk), iccd=icc_31(day(lk)), rho=np.corrcoef(lk.ravel(), meanfc.ravel())[0, 1],
                 iccr=icc_31(lk - np.polyval(bfit, meanfc)), atbound=np.mean(KA < 2e-4))
        out[(name, use_gsr)] = o
        print(f"   {name:11s} GSR={'sí' if use_gsr else 'no'}: r(FC) med={np.median(R):.3f}  kappa med={np.median(KA):.4f} "
              f"(en límite {o['atbound']:.0%})  ICC4={o['icc']:.3f} ICCdía={o['iccd']:.3f}  r(FC media)={o['rho']:+.3f}  ICC sin FC media={o['iccr']:.3f}")
lk = {k: np.log(v["KA"]).mean(1) for k, v in out.items()}
print(f"   concordancia entre sujetos (GSR): aditivo SC~EC r={np.corrcoef(lk[('aditivo SC', True)], lk[('aditivo EC', True)])[0, 1]:.3f}  "
      f"aditivo SC~difusivo SC r={np.corrcoef(lk[('aditivo SC', True)], lk[('difusivo SC', True)])[0, 1]:.3f}")

np.savez(ROOT / "results" / "fase7_additive.npz", EC_add=EC["additive"], EC_dif=EC["diffusive"],
         best=np.array([[*k, *b[:4]] for k, b in best.items()], dtype=object),
         **{f"{k[0]}|{int(k[1])}|{m}": v[m] for k, v in out.items() for m in ["P", "R", "KA"]},
         stats=np.array([[k[0], int(k[1]), v["icc"], v["iccd"], v["rho"], v["iccr"], v["atbound"]] for k, v in out.items()], dtype=object))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
names = list(out)
x = np.arange(len(names))
ax[0].bar(x - 0.2, [out[k]["iccd"] for k in names], 0.4, label="ICC día")
ax[0].bar(x + 0.2, [out[k]["iccr"] for k in names], 0.4, label="ICC sin FC media")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{k[0]}\n{'GSR' if k[1] else 'sin GSR'}" for k in names], fontsize=7)
ax[0].set(ylim=(0, 1), title="Reproducibilidad de κ"); ax[0].legend(fontsize=8)
for k, col in [(("aditivo SC", False), "C0"), (("aditivo SC", True), "C3")]:
    ax[1].scatter(meanfc.ravel(), np.log10(out[k]["KA"].ravel()), s=4, alpha=0.4, color=col, label=f"GSR={'sí' if k[1] else 'no'}")
ax[1].set(xlabel="FC media de la sesión", ylabel=r"$\log_{10}(1-\rho)$", title="Aditivo: κ vs FC media"); ax[1].legend()
ax[2].bar(["dif.\nSC", "adit.\nSC", "dif.\nEC", "adit.\nEC"],
          [best[("diffusive", True)][1], best[("additive", True)][1],
           r_of(model_fc_free(EC["diffusive"], "diffusive", best[("diffusive", False)][2]), FB, True),
           r_of(model_fc_free(EC["additive"], "additive", None), FB, True)], color=["C0", "C1", "C0", "C1"])
ax[2].set(ylim=(0, 1), ylabel="r(FC) prueba, con GSR", title="Difusivo vs aditivo")
fig.tight_layout(); fig.savefig(FIG / "fase7_aditivo.png", dpi=130)
print("figures saved")
