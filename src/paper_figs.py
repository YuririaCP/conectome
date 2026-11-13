"""Figures 1 and 2 of the paper (English labels, vector PDF).

Fig. 1  Ring of N nodes: (a) time rescaling J -> cJ changes the autocorrelation but not FC;
        (b) spectrum of theta*I + L and the uniform slow mode.
Fig. 2  Proposition 5 on the group structural connectome (Schaefer-100):
        (a) 1 - FC and the GSR'd FC distance to corr(L^+) versus theta;
        (b) sensitivity ratio with/without GSR converging to rho_inf;
        (c) a global artefact sigma^2 is indistinguishable from a change of theta.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_continuous_lyapunov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcdata import gsr, laplacian, SC_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # reference categorical slots 1-3, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "font.family": "serif",
})


def corr(S):
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def tag(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)


# ---------------------------------------------------------------- Fig. 1
N = 16
A = np.zeros((N, N))
for i in range(N):
    A[i, (i + 1) % N] = A[(i + 1) % N, i] = 1.0
A[0, N // 2] = A[N // 2, 0] = 1.0           # one chord, so FC is not translation invariant
L = laplacian(A)
theta = 0.05
J = -(theta * np.eye(N) + L)                 # G = 1, time in units of 1/G
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)

lags = np.linspace(0, 40, 200)
for c, col, lab in [(1.0, C1, r"$J$"), (4.0, C2, r"$4J$")]:
    S = solve_continuous_lyapunov(c * J, -np.eye(N))
    ac = [(expm(c * J * t) @ S)[0, 0] / S[0, 0] for t in lags]
    axs[0].plot(lags, ac, color=col, label=lab)
axs[0].set_xlabel("lag (units of $1/G$)")
axs[0].set_ylabel("autocorrelation, node 1")
axs[0].legend(frameon=False)
tag(axs[0], "(a)")

S1 = solve_continuous_lyapunov(J, -np.eye(N))
S4 = solve_continuous_lyapunov(4 * J, -np.eye(N))
iu = np.triu_indices(N, 1)
axs[1].plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
axs[1].scatter(corr(S1)[iu], corr(S4)[iu], s=10, color=C1, edgecolor="white", linewidth=0.5)
axs[1].set_xlabel(r"FC, dynamics $J$")
axs[1].set_ylabel(r"FC, dynamics $4J$")
axs[1].set_aspect("equal")
tag(axs[1], "(b)")

lam, U = np.linalg.eigh(theta * np.eye(N) + L)
for k, col, lab in [(0, C1, rf"slowest mode, rate $\theta={lam[0]:.2f}$"),
                    (1, C2, rf"next mode, rate {lam[1]:.2f}")]:
    v = U[:, k] * np.sign(U[:, k].sum() if k == 0 else U[0, k])
    axs[2].plot(np.arange(1, N + 1), v, "o-", ms=3.5, color=col, label=lab)
axs[2].axhline(0, color=GRID, lw=1, zorder=0)
axs[2].set_xlabel("node")
axs[2].set_ylabel("eigenvector entry")
axs[2].legend(frameon=False, loc="lower left")
tag(axs[2], "(c)")
fig.savefig(OUT / "fig1_ring.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2
C = SC_all.mean(0)
C = (C + C.T) / 2
np.fill_diagonal(C, 0)
C = C / C.sum(1).mean()
N = len(C)
iu = np.triu_indices(N, 1)
L = laplacian(C)
Lp = np.linalg.pinv(L)
one = np.ones(N)
ell = np.diag(Lp)
Om = ell[:, None] + ell[None, :] - 2 * Lp
h = Lp @ ell
dS = -(Lp @ Lp - 0.5 * (np.outer(one, h) + np.outer(h, one)))
s = np.sqrt(ell)
ds = np.diag(dS) / (2 * s)
G1 = dS / np.outer(s, s) - Lp * (np.outer(ds, s) + np.outer(s, ds)) / np.outer(s, s) ** 2
rho_inf = 2 * np.linalg.norm(G1[iu]) / (N * np.linalg.norm(Om[iu]))
G0 = corr(Lp)


def F(th, s2=0.0):
    return corr(np.linalg.inv(th * np.eye(N) + L) + s2 * np.outer(one, one))


ths = np.logspace(-4, 0, 25)
raw = [np.mean(1 - F(t)[iu]) for t in ths]
gs = [np.sqrt(np.mean((gsr(F(t)) - G0)[iu] ** 2)) for t in ths]
ratio = []
for t in ths:
    e = 0.01
    dF = (F(t * np.exp(e)) - F(t * np.exp(-e)))[iu]
    dG = (gsr(F(t * np.exp(e))) - gsr(F(t * np.exp(-e))))[iu]
    ratio.append(np.linalg.norm(dG) / np.linalg.norm(dF))

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
ax = axs[0]
ax.loglog(ths, raw, color=C1, label=r"$1-\overline{\mathrm{FC}}$ (raw)")
ax.loglog(ths, gs, color=C2, label=r"$\|\mathrm{GSR(FC)}-\mathrm{corr}(L^+)\|_{\rm rms}$")
lin = (N / 2) * np.mean(Om[iu]) * ths
ax.loglog(ths[lin < 0.6], lin[lin < 0.6], color=MUTED, lw=1, ls="--", label=r"$\frac{N\theta}{2}\,\overline{\Omega}$")
ax.set_xlabel(r"$\theta=|a|/G$")
ax.legend(frameon=False, loc="lower right")
tag(ax, "(a)")

ax = axs[1]
ax.semilogx(ths, ratio, color=C1)
ax.axhline(rho_inf, color=MUTED, lw=1, ls="--")
ax.text(ths[1], rho_inf * 1.6, rf"$\rho_\infty={rho_inf:.4f}$", color=INK, fontsize=7)
ax.set_yscale("log")
ax.set_xlabel(r"$\theta=|a|/G$")
ax.set_ylabel(r"$\|\partial_{\log\theta}\mathrm{GSR(FC)}\|/\|\partial_{\log\theta}\mathrm{FC}\|$")
tag(ax, "(b)")

ax = axs[2]
th, s2 = 0.01, 1.0 / (N * 0.01)
th_eff = 1 / (1 / th + N * s2)
ax.plot([-0.1, 1], [-0.1, 1], color=GRID, lw=1, zorder=0)
ax.scatter(F(th_eff)[iu], F(th, s2)[iu], s=4, color=C1, alpha=0.5, lw=0, label=r"vs. $\theta_{\rm eff}$")
ax.scatter(F(th)[iu], F(th, s2)[iu], s=4, color=C2, alpha=0.5, lw=0, label=r"vs. $\theta$")
ax.set_xlabel("FC without artefact")
ax.set_ylabel(r"FC with global artefact $\sigma^2$")
ax.legend(frameon=False, loc="upper left", markerscale=2.5)
tag(ax, "(c)")
fig.savefig(OUT / "fig2_gsr.pdf")
plt.close(fig)
print(f"rho_inf = {rho_inf:.4f}; figures in {OUT}")
