"""Why theta (and kappa) is poorly identified from FC in the diffusive model.

With homogeneous theta the slowest mode of theta*I + L is the uniform vector (L 1 = 0), with
rate theta: the critical mode IS the global signal.  We measure how much FC changes with
log(theta), with and without GSR, and how much of that change lies in the global mode.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcdata import gsr, laplacian, iu, N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
r = np.load(ROOT / "results" / "fase6_fc_fit.npz")
for name, C in [("SC cruda", None), ("EC simétrica", r["EC"])]:
    if C is None:
        from fcdata import SC_all
        C = SC_all.mean(0); C = (C + C.T) / 2; np.fill_diagonal(C, 0); C = C / C.sum(1).mean()
    L = laplacian(C)
    lam, U = np.linalg.eigh(L)
    print(f"{name}: eigvec lento de L uniforme? |u0 - 1/sqrt(N)| max = {np.abs(np.abs(U[:, 0]) - 1 / np.sqrt(N)).max():.1e}; "
          f"lambda_2 = {lam[1]:.4f}")

    def fc(th):
        S = np.linalg.inv(th * np.eye(N) + L); s = np.sqrt(np.diag(S)); return S / np.outer(s, s)

    for th in [0.3, 0.06, 0.01]:
        h = 0.05
        d_raw = (fc(th * np.exp(h)) - fc(th * np.exp(-h)))[iu] / (2 * h)
        d_gsr = (gsr(fc(th * np.exp(h))) - gsr(fc(th * np.exp(-h))))[iu] / (2 * h)
        print(f"   theta={th:<5}: |dFC/dlog theta| sin GSR={np.sqrt(np.mean(d_raw**2)):.4f}   con GSR={np.sqrt(np.mean(d_gsr**2)):.4f}"
              f"   cociente={np.sqrt(np.mean(d_gsr**2)) / np.sqrt(np.mean(d_raw**2)):.2f}")
