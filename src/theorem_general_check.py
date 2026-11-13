"""Numerical check of the general global-mode theorem (any symmetric excitatory coupling).

S(eps) = eps^{-1} v v^T + R(eps), v > 0 unit, R v = 0.
  (a) FC = 1 1^T - (eps/2) Omega_v + O(eps^2), Omega_v from Rt = D_v^{-1} R(0) D_v^{-1};
  (b) for global-signal weights w with w(0)^T v != 0, GSR(FC) -> corr(Pi R0 Pi^T),
      Pi = I - v w0^T / (w0^T v); z-scored GSR has w0 = D_v^{-1} 1, raw-signal GSR w0 = 1;
  (c) an artefact with loading v is absorbed exactly: eps_eff^{-1} = eps^{-1} + sigma^2.
Case: additive coupling on the Schaefer-100 group connectome, eps = 1 - rho = kappa.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcdata import SC_all, gsr, iu  # noqa: E402

C = SC_all.mean(0)
C = (C + C.T) / 2
np.fill_diagonal(C, 0)
lam, U = np.linalg.eigh(C)
lam = lam / lam[-1]
v = U[:, -1] * np.sign(U[:, -1].sum())
N = len(v)
one = np.ones(N)
print(f"Perron vector positive: {np.all(v > 0)}; cos(v,1) = {v.sum() / np.sqrt(N):.3f}; gap 1 - lambda_2 = {1 - lam[-2]:.3f}")


def corr(S):
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def S_of(eps):                      # (I - rho C_hat)^{-1}, rho = 1 - eps
    return (U / (1 - (1 - eps) * lam)) @ U.T


R0 = (U[:, :-1] / (1 - lam[:-1])) @ U[:, :-1].T
Rt = R0 / np.outer(v, v)
Om = np.diag(Rt)[:, None] + np.diag(Rt)[None, :] - 2 * Rt


def gsr_w(S, w):                    # regression of x on w^T x, then correlation
    Sw = S @ w
    return corr(S - np.outer(Sw, Sw) / (w @ Sw))


def limit(w0):
    Pi = np.eye(N) - np.outer(v, w0) / (w0 @ v)
    return corr(Pi @ R0 @ Pi.T)


Lz, L1 = limit(one / v), limit(one)
print("eps      |FC-(11-eps/2 Om)|/eps^2   |GSRz-lim|   |GSRraw-lim|   ratio dlogeps (z-scored GSR)")
for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
    S = S_of(eps)
    F = corr(S)
    ea = np.abs((F - (1 - eps / 2 * Om))[iu]).max() / eps ** 2
    ez = np.abs((gsr(F) - Lz)[iu]).max()                  # gsr() = z-scored GSR on the correlation matrix
    e1 = np.abs((gsr_w(S, one) - L1)[iu]).max()
    h = 0.01
    dF = (corr(S_of(eps * np.exp(h))) - corr(S_of(eps * np.exp(-h))))[iu]
    dG = (gsr(corr(S_of(eps * np.exp(h)))) - gsr(corr(S_of(eps * np.exp(-h)))))[iu]
    print(f"{eps:7.0e}  {ea:10.3g}                 {ez:.2e}     {e1:.2e}       {np.linalg.norm(dG) / np.linalg.norm(dF):.4f}")
eps, s2 = 1e-2, 50.0
Fa = corr(S_of(eps) + s2 * np.outer(v, v))
print(f"artefact along v: max|FC - FC(eps_eff)| = {np.abs((Fa - corr(S_of(1 / (1 / eps + s2))))[iu]).max():.2e} "
      f"(R(eps) vs R(eps_eff) differ only at O(eps))")
