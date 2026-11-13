"""Fase 5: what zero-lag FC can identify — a dimensionless criticality index.

Theory (linearized Stuart-Landau, diffusive coupling, symmetric SC, Laplacian L):
  * Zero-lag FC is invariant under time rescaling A -> cA, so delta (in 1/s) is not identifiable.
  * A homogeneous frequency omega drops out exactly (rotating frame); realistic heterogeneous
    frequencies change FC by ~0.01.  Hence, to excellent approximation,
        FC = corr[(diag(theta) + L)^{-1}],   theta_j = |a_j| / G.
  * Relaxation rates (in units of G) are the eigenvalues mu_k of diag(theta) + L.  The
    dimensionless index
        kappa = mu_min / mean(mu)          (0 < kappa <= 1)
    measures how close the slowest mode is to the bifurcation relative to the typical mode
    (kappa -> 0: critical slowing of the slowest mode).  For homogeneous theta and
    L1 = 0: mu_min = theta, kappa = theta / (theta + mean(lambda_L)).

Data: Domhof et al. (EBRAINS), 200 HCP subjects, Schaefer-100, FC of 4 sessions
(REST1-LR, REST1-RL, REST2-LR, REST2-RL; ICA-FIX, detrended, z-scored, no band-pass, no GSR).

Analyses:
  A. per subject and session: 1-parameter fit of theta (own SC and group SC), with and
     without global signal regression (GSR, applied identically to model and data);
  B. reliability: ICC across days, and whether kappa carries information beyond mean FC;
  C. group level: heterogeneous theta_j map fitted on subjects 0-99 and 100-199 separately.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
d = np.load(ROOT / "data" / "raw" / "domhof_hcp" / "schaefer100.npz")
FC_all, SC_all = d["fc"], d["sc"]           # (200, 4, N, N), (200, N, N)
S_, K_, N = FC_all.shape[0], FC_all.shape[1], FC_all.shape[-1]
iu = np.triu_indices(N, 1)


def prep_fc(F):
    F = (F + F.T) / 2
    np.fill_diagonal(F, 1.0)
    return F


def gsr(F):
    """Global signal regression on the covariance of z-scored signals, then renormalize."""
    c = F.mean(axis=1)
    R = F - np.outer(c, c) / c.mean()
    s = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    return R / np.outer(s, s)


def norm_sc(S):
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 0.0)
    return S / S.sum(axis=1).mean()          # mean node strength = 1 -> theta comparable


def laplacian(S):
    return np.diag(S.sum(axis=1)) - S


class Homog:
    """FC(theta) = corr[(theta I + L)^{-1}] via one eigendecomposition of L."""

    def __init__(self, S):
        self.lam, self.U = np.linalg.eigh(laplacian(S))
        self.lam = np.clip(self.lam, 0, None)

    def fc(self, theta):
        Sig = (self.U / (theta + self.lam)) @ self.U.T
        s = np.sqrt(np.diag(Sig))
        return Sig / np.outer(s, s)

    def kappa(self, theta):
        mu = theta + self.lam
        return mu.min() / mu.mean()


def fit_theta(model, F, use_gsr):
    t = gsr(F)[iu] if use_gsr else F[iu]

    def loss(lt):
        m = model.fc(np.exp(lt))
        m = gsr(m)[iu] if use_gsr else m[iu]
        return np.mean((m - t) ** 2)

    r = minimize_scalar(loss, bounds=(np.log(1e-4), np.log(1e3)), method="bounded")
    th = float(np.exp(r.x))
    m = model.fc(th)
    m = gsr(m)[iu] if use_gsr else m[iu]
    return th, float(np.corrcoef(m, t)[0, 1]), float(r.fun)


def icc_31(X):
    """ICC(3,1) consistency; X: subjects x sessions."""
    n, k = X.shape
    gm = X.mean()
    ssr = k * ((X.mean(1) - gm) ** 2).sum()
    ssc = n * ((X.mean(0) - gm) ** 2).sum()
    sse = ((X - gm) ** 2).sum() - ssr - ssc
    msr, mse = ssr / (n - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


# ---------------------------------------------------------------- A. per-subject fits
S_group = norm_sc(SC_all.mean(0))
M_group = Homog(S_group)
lam_mean = M_group.lam.mean()
print(f"N={N}, {S_} sujetos x {K_} sesiones.  Laplaciano de grupo: lambda_2={np.sort(M_group.lam)[1]:.3f}, "
      f"media={lam_mean:.3f}, max={M_group.lam.max():.3f}")
res = {}
for sc_kind in ["propia", "grupo"]:
    for use_gsr in [False, True]:
        TH = np.zeros((S_, K_)); KA = np.zeros_like(TH); RR = np.zeros_like(TH)
        for s in range(S_):
            model = Homog(norm_sc(SC_all[s])) if sc_kind == "propia" else M_group
            for k in range(K_):
                th, r, _ = fit_theta(model, prep_fc(FC_all[s, k]), use_gsr)
                TH[s, k], RR[s, k], KA[s, k] = th, r, model.kappa(th)
        res[(sc_kind, use_gsr)] = (TH, KA, RR)
        print(f"SC {sc_kind:6s} GSR={'sí' if use_gsr else 'no'}: r(FC) mediana={np.median(RR):.3f}  "
              f"theta mediana={np.median(TH):.3f}  kappa mediana={np.median(KA):.4f}")

# ---------------------------------------------------------------- B. reliability and redundancy
meanfc = np.array([[prep_fc(FC_all[s, k])[iu].mean() for k in range(K_)] for s in range(S_)])
print("\nReproducibilidad entre sesiones, ICC(3,1) sobre 4 sesiones (y día 1 vs día 2 con promedios):")
day = lambda X: np.stack([X[:, :2].mean(1), X[:, 2:].mean(1)], 1)
print(f"   FC media (sin GSR)                 ICC4={icc_31(meanfc):.3f}  ICCdía={icc_31(day(meanfc)):.3f}")
summary = {}
for key, (TH, KA, RR) in res.items():
    lk = np.log(KA)
    rho = np.corrcoef(lk.ravel(), meanfc.ravel())[0, 1]
    # information beyond mean FC: residualize log kappa on mean FC (pooled), then ICC
    b = np.polyfit(meanfc.ravel(), lk.ravel(), 1)
    resid = lk - np.polyval(b, meanfc)
    summary[key] = (icc_31(lk), icc_31(day(lk)), rho, icc_31(resid), icc_31(RR))
    print(f"   log kappa SC {key[0]:6s} GSR={'sí' if key[1] else 'no'}  ICC4={summary[key][0]:.3f}  "
          f"ICCdía={summary[key][1]:.3f}  r(con FC media)={rho:+.3f}  ICC residual={summary[key][3]:.3f}  "
          f"ICC del ajuste r(FC)={summary[key][4]:.3f}")

# own SC vs group SC: does the subject's anatomy matter? compare fit quality on the same FC
d_r = res[("propia", True)][2] - res[("grupo", True)][2]
print(f"\nCon GSR, r(FC) con SC propia - SC de grupo: media={d_r.mean():+.4f} (fracción > 0: {np.mean(d_r > 0):.2f})")

# ---------------------------------------------------------------- C. heterogeneous theta_j map (group level)
def het_fc(theta, L):
    Sig = np.linalg.inv(np.diag(theta) + L)
    s = np.sqrt(np.diag(Sig))
    return Sig / np.outer(s, s)


def fit_het(F_target, L, th0, use_gsr=True, mu=1e-3):
    t = gsr(F_target)[iu] if use_gsr else F_target[iu]

    def obj(lt):
        m = het_fc(np.exp(lt), L)
        m = gsr(m)[iu] if use_gsr else m[iu]
        return np.mean((m - t) ** 2) + mu * np.var(lt)

    r = minimize(obj, np.full(N, np.log(th0)), method="L-BFGS-B",
                 bounds=[(np.log(1e-4), np.log(1e3))] * N, options={"maxiter": 300})
    return np.exp(r.x), r


def group_fc(subjects, sessions):
    Z = np.arctanh(np.clip(np.array([prep_fc(FC_all[s, k]) for s in subjects for k in sessions]), -0.999999, 0.999999))
    F = np.tanh(Z.mean(0)); np.fill_diagonal(F, 1.0)
    return F


L_g = laplacian(S_group)
A_sub, B_sub = range(0, 100), range(100, 200)
FA, FB = group_fc(A_sub, range(4)), group_fc(B_sub, range(4))
ceil = np.corrcoef(gsr(FA)[iu], gsr(FB)[iu])[0, 1]
th_h = fit_theta(Homog(S_group), FA, True)[0]
print(f"\nNivel de grupo (GSR): techo A->B r={ceil:.4f};  theta homogéneo (A)={th_h:.3f}")
thA, rA = fit_het(FA, L_g, th_h)
thB, rB = fit_het(FB, L_g, th_h)
sc_ = lambda th, F: np.corrcoef(gsr(het_fc(th, L_g))[iu], gsr(F)[iu])[0, 1]
r_hom_B = np.corrcoef(gsr(Homog(S_group).fc(th_h))[iu], gsr(FB)[iu])[0, 1]
shuf = np.mean([sc_(np.random.default_rng(i).permutation(thA), FB) for i in range(50)])
print(f"   test en B: homogéneo r={r_hom_B:.3f}  heterogéneo(A) r={sc_(thA, FB):.3f}  barajado r={shuf:.3f}")
rep = np.corrcoef(np.log(thA), np.log(thB))[0, 1]
print(f"   reproducibilidad del mapa log theta_j entre mitades: r={rep:.3f};  "
      f"en límites: A {np.sum((thA < 1.01e-4) | (thA > 0.99e3))}, B {np.sum((thB < 1.01e-4) | (thB > 0.99e3))}")
strength = S_group.sum(1)
print(f"   r(log theta_j, fuerza del nodo) = {np.corrcoef(np.log(thA), strength)[0, 1]:+.3f}")
muA = np.linalg.eigvalsh(np.diag(thA) + L_g)
print(f"   kappa heterogéneo (A) = {muA.min() / muA.mean():.4f}  vs homogéneo = {Homog(S_group).kappa(th_h):.4f}")

np.savez(ROOT / "results" / "fase5_fc_only.npz",
         **{f"TH_{k[0]}_{int(k[1])}": v[0] for k, v in res.items()},
         **{f"KA_{k[0]}_{int(k[1])}": v[1] for k, v in res.items()},
         **{f"RR_{k[0]}_{int(k[1])}": v[2] for k, v in res.items()},
         meanfc=meanfc, thA=thA, thB=thB, ceil=ceil, rep=rep, th_h=th_h,
         summary=np.array([summary[k] for k in res]), keys=np.array([f"{k[0]}|{int(k[1])}" for k in res]))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
for use_gsr, col in [(False, "C0"), (True, "C3")]:
    KA = res[("propia", use_gsr)][1]
    ax[0].scatter(meanfc.ravel(), np.log10(KA.ravel()), s=4, alpha=0.4, color=col, label=f"GSR={'sí' if use_gsr else 'no'}")
ax[0].set(xlabel="FC media de la sesión", ylabel=r"$\log_{10}\kappa$", title="Índice vs FC media"); ax[0].legend()
for use_gsr, col in [(False, "C0"), (True, "C3")]:
    lk = np.log10(day(res[("propia", use_gsr)][1]))
    ax[1].scatter(lk[:, 0], lk[:, 1], s=8, alpha=0.6, color=col, label=f"GSR={'sí' if use_gsr else 'no'} (ICC={icc_31(lk):.2f})")
lo, hi = ax[1].get_xlim(); ax[1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax[1].set(xlabel=r"$\log_{10}\kappa$ día 1", ylabel=r"$\log_{10}\kappa$ día 2", title="Reproducibilidad entre días"); ax[1].legend(fontsize=8)
ax[2].scatter(np.log10(thA), np.log10(thB), s=12)
ax[2].set(xlabel=r"$\log_{10}\theta_j$ (sujetos 0–99)", ylabel=r"$\log_{10}\theta_j$ (100–199)", title=f"Mapa regional, r={rep:.2f}")
labels = ["FC media", "κ no-GSR", "κ GSR", "κ resid.\nno-GSR", "κ resid.\nGSR"]
vals = [icc_31(meanfc), summary[("propia", False)][0], summary[("propia", True)][0], summary[("propia", False)][3], summary[("propia", True)][3]]
ax[3].bar(labels, vals, color=["gray", "C0", "C3", "C0", "C3"]); ax[3].set(ylabel="ICC(3,1)", title="Reproducibilidad (4 sesiones)", ylim=(0, 1))
fig.tight_layout(); fig.savefig(FIG / "fase5_indice_relativo.png", dpi=130)
print("figures saved")

# ---------------------------------------------------------------- D. control: is kappa's reliability driven by SC alone?
# Fit each subject's FC with ANOTHER subject's SC (fixed across that subject's sessions).
# If ICC stays high, the reliability comes from the SC, not from the subject's FC.
print("\nControl de SC: FC del sujeto s ajustada con la SC del sujeto pi(s)")
perm = np.random.default_rng(7).permutation(S_)
KA_x = np.zeros((S_, K_))
for s in range(S_):
    model = Homog(norm_sc(SC_all[perm[s]]))
    for k in range(K_):
        KA_x[s, k] = model.kappa(fit_theta(model, prep_fc(FC_all[s, k]), True)[0])
own = np.log(res[("propia", True)][1])
print(f"   ICC4 log kappa: SC propia={icc_31(own):.3f}  SC de otro sujeto={icc_31(np.log(KA_x)):.3f}  SC de grupo={icc_31(np.log(res[('grupo', True)][1])):.3f}")
# variance decomposition: how much of kappa (own SC, GSR) is predicted by SC-only features?
lam2 = np.array([np.sort(np.linalg.eigvalsh(laplacian(norm_sc(SC_all[s]))))[1] for s in range(S_)])
tot = np.array([SC_all[s][iu].sum() for s in range(S_)])
m_own = own.mean(1)
print(f"   r(log kappa medio, lambda_2 de su SC)={np.corrcoef(m_own, lam2)[0, 1]:+.3f}  r(log kappa, log total de streamlines)={np.corrcoef(m_own, np.log(tot))[0, 1]:+.3f}")
grp = np.log(res[("grupo", True)][1]).mean(1)
print(f"   r(log kappa con SC propia, log kappa con SC de grupo) entre sujetos = {np.corrcoef(m_own, grp)[0, 1]:.3f}")
np.savez(ROOT / "results" / "fase5_control_sc.npz", KA_x=KA_x, perm=perm, lam2=lam2, tot=tot)
