"""Analysis and figure for the regime sweep (results/nonlinear_regimes.npz) -> Fig. 6 (revised).

For each (G, s) and each interval of neighbouring a: noise-corrected rms change of raw FC and of FC
after GSR, in units of the sampling noise of the change; synchronization measures along a.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT / "results" / "nonlinear_regimes.npz"] + sorted((ROOT / "results").glob("nonlinear_regimes_seeds*.npz"))
ds = [np.load(f) for f in files]
d = ds[0]
A, GS, DISP = d["A"], d["GS"], d["DISP"]
Fr = np.concatenate([x["Fr"].astype(float) for x in ds], axis=3)       # (G, s, a, seeds, pairs)
Fg = np.concatenate([x["Fg"].astype(float) for x in ds], axis=3)
SY = np.concatenate([x["SY"] for x in ds], axis=3)
print(f"{Fr.shape[3]} realizations per point from {len(files)} files")
mid = (A[:-1] + A[1:]) / 2
da = np.diff(A)


def change(F):
    """noise-corrected rms change between neighbouring a and its noise level (K realizations)."""
    m = F.mean(-2)
    raw = np.sqrt(np.mean(np.diff(m, axis=-2) ** 2, -1))
    K = F.shape[-2]
    var = np.mean(F.var(-2, ddof=1), -1) / K
    noise = np.sqrt(var[..., :-1] + var[..., 1:])
    return np.sqrt(np.clip(raw ** 2 - noise ** 2, 0, None)), noise


s_raw, n_raw = change(Fr)
s_gsr, n_gsr = change(Fg)
print("noise-corrected change of FC after GSR, in units of its noise, per interval (a midpoints):")
print("      " + " ".join(f"{m:+.4f}" for m in mid))
for i, G in enumerate(GS):
    for j, s in enumerate(DISP):
        print(f"G={G} s={s}: " + " ".join(f"{v:7.1f}" for v in s_gsr[i, j] / n_gsr[i, j]))
print("\nraw FC change / noise:")
for i, G in enumerate(GS):
    for j, s in enumerate(DISP):
        print(f"G={G} s={s}: " + " ".join(f"{v:7.1f}" for v in s_raw[i, j] / n_raw[i, j]))
SYm = SY.mean(-2)
print("\nnon-global phase coherence (PLVrel) at a=-0.05 / max over a>0, and R at a=0.05:")
for i, G in enumerate(GS):
    for j, s in enumerate(DISP):
        print(f"G={G} s={s}: PLVrel {SYm[i, j, 0, 2]:.3f} -> {SYm[i, j, A > 0, 2].max():.3f}   R(a=0.05) {SYm[i, j, -1, 0]:.3f}")
# split-half replication: the same statistic from two disjoint sets of realizations
K = Fr.shape[3]
h1, h2 = slice(0, K // 2), slice(K // 2, K)
z1 = change(Fg[:, :, :, h1])[0] / change(Fg[:, :, :, h1])[1]
z2 = change(Fg[:, :, :, h2])[0] / change(Fg[:, :, :, h2])[1]
print("\nsplit-half replication of (change after GSR / noise) on a in [0.02, 0.05] and [0.01, 0.02]:")
for i, G in enumerate(GS):
    for j, s_ in enumerate(DISP):
        print(f"G={G} s={s_}: halves {z1[i, j, -1]:5.1f} / {z2[i, j, -1]:5.1f}   and {z1[i, j, -2]:5.1f} / {z2[i, j, -2]:5.1f}")
below = (s_gsr / n_gsr)[:, :, mid < 0]
print(f"max over damped-side intervals of (change after GSR / noise): {below.max():.2f}")
np.savez(ROOT / "results" / "regimes_summary.npz", z1=z1, z2=z2, mid=mid, s_raw=s_raw, n_raw=n_raw, s_gsr=s_gsr, n_gsr=n_gsr, SYm=SYm)

# ---------------------------------------------------------------- figure
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRIDC = "#1f1f1e", "#6b6a63", "#e4e2dc"
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "pdf.fonttype": 42,
    "font.family": "serif",
})
iG = list(GS).index(0.2)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
ax = axs[0]
for j, s_ in enumerate(DISP):
    ax.semilogy(A, 1 - SYm[iG, j, :, 0], "o-", ms=3, color=C[j], label=f"dispersion {s_:g}")
ax.axvline(0, color=GRIDC, lw=1, zorder=0)
ax.set_xlabel("bifurcation parameter $a$")
ax.set_ylabel("incoherence $1-R$ (Kuramoto)")
ax.legend(frameon=False, loc="lower left")
ax.text(-0.2, 1.04, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
ax = axs[1]
for j, s in enumerate(DISP):
    y = np.maximum(s_gsr[iG, j] / n_gsr[iG, j], 0.05)
    ax.semilogy(mid, y, "-", lw=1, color=C[j], alpha=0.6)
    ax.semilogy(mid[y >= 1], y[y >= 1], "o", ms=3.5, color=C[j], label=f"dispersion {s:g}")
    ax.semilogy(mid[y < 1], y[y < 1], "o", ms=3.5, color=C[j], mfc="white")
ax.axhspan(0.03, 1, color=GRIDC, alpha=0.5, lw=0, zorder=0)
ax.axhline(1, color=MUTED, lw=1, ls="--")
ax.text(mid[-1], 0.037, "below sampling noise", color=MUTED, fontsize=6.5, ha="right")
ax.axvline(0, color=GRIDC, lw=1, zorder=0)
ax.set_xlabel("bifurcation parameter $a$")
ax.set_ylabel("change of FC after GSR / noise")
ax.legend(frameon=False, loc="upper left")
ax.text(-0.2, 1.04, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
ax = axs[2]
k = len(mid) - 1                         # interval [0.02, 0.05]
x = np.arange(len(GS))
wbar = 0.25
for j, s in enumerate(DISP):
    ax.bar(x + (j - 1) * wbar, s_gsr[:, j, k] / n_gsr[:, j, k], width=wbar * 0.9, color=C[j], label=f"dispersion {s:g}")
ax.axhline(1, color=MUTED, lw=1, ls="--")
ax.set_xticks(x, [f"$G={G:g}$" for G in GS])
ax.set_ylabel("change of FC after GSR / noise")
ax.set_title(r"$a\in[0.02,0.05]$; colours as in (a,b)", fontsize=7.5, color=INK)
ax.text(-0.2, 1.04, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
fig.savefig(ROOT / "paper" / "figures" / "fig6b_regimes.pdf")
print("figure written")
