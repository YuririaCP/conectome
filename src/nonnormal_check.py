"""Numerical check of Corollary (directed coupling) and Proposition (left critical mode is a clock),
on the directed effective connectivity fitted in Phase 3 (neurolib HCP, AAL2, N = 94).

Directed diffusive coupling A = -(theta I + L_out), L_out = diag(C 1) - C: right critical vector 1,
left critical vector = stationary distribution of the reversed random walk.
Directed additive coupling A = -I + (1 - eps) C / lambda_max: right/left Perron vectors of C.
"""
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
C = np.load(ROOT / "results" / "fase3_ec.npz")["EC"]
np.fill_diagonal(C, 0)
N = len(C)
ncomp, lab = connected_components(C > 0, directed=True, connection="strong")
keep = lab == np.bincount(lab).argmax()
C = C[np.ix_(keep, keep)]
N = len(C)
print(f"EC: {ncomp} strongly connected components; largest has {N} nodes; "
      f"asymmetry ||C-C^T||/||C|| = {np.linalg.norm(C - C.T) / np.linalg.norm(C):.2f}")
iu = np.triu_indices(N, 1)


def corr(S):
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def gsr_w(S, w):
    Sw = S @ w
    return corr(S - np.outer(Sw, Sw) / (w @ Sw))


def perron(M):
    lam, V = np.linalg.eig(M)
    k = np.argmax(lam.real)
    return lam[k].real, np.abs(V[:, k].real)


def cos(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


rows = {}
for kind in ["diffusive", "additive"]:
    if kind == "diffusive":
        Lout = np.diag(C.sum(1)) - C
        Lout = Lout / np.mean(C.sum(1))
        A_of = lambda e: -(e * np.eye(N) + Lout)
        _, l = perron(-Lout.T)
        r = np.ones(N)
    else:
        lmax, r = perron(C)
        _, l = perron(C.T)
        A_of = lambda e: -np.eye(N) + (1 - e) * C / lmax
    l = l / (l @ r)
    print(f"\n{kind}: cos(r,1) = {cos(r, np.ones(N)):.3f}, cos(l,1) = {cos(l, np.ones(N)):.3f}, cos(r,l) = {cos(r, l):.3f}")
    prev = None
    for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
        A = A_of(eps)
        S = solve_continuous_lyapunov(A, -np.eye(N))
        u = np.abs(np.linalg.eigh(S)[1][:, -1])
        Gz = gsr_w(S, 1 / np.sqrt(np.diag(S)))
        d = np.abs(Gz - prev)[iu].max() if prev is not None else np.nan
        prev = Gz
        taus = np.array([5.0, 20.0, 50.0])
        ac = lambda q: np.array([q @ expm(A * t) @ S @ q / (q @ S @ q) for t in taus])
        rate = lambda q: -np.log(ac(q)) / taus
        print(f"  eps={eps:.0e}: cos(top PC of S, r)={cos(u, r):.4f}  1-meanFC={1 - corr(S)[iu].mean():.2e}  "
              f"max|dGSR| vs previous eps={d:.1e}  rate(l^T x)/eps={np.round(rate(l) / eps, 4)}  "
              f"rate(global)/eps={np.round(rate(np.ones(N)) / eps, 2)}")
