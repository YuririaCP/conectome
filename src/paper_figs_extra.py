"""Figures 6 and 7: nonlinear check of the confound, and measuring delta from the global signal.
Requires results/nonlinear_confound.npz (nonlinear_confound.py) and results/temporal_demo.npz
(temporal_demo.py)."""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "figures"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "pdf.fonttype": 42,
    "font.family": "serif",
})


def tag(ax, s):
    ax.text(-0.2, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)


# ---------------------------------------------------------------- Fig. 6
d = np.load(RES / "nonlinear_confound.npz")
A = d["A"]
mid = (A[:-1] + A[1:]) / 2
da = np.diff(A)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
ax = axs[0]
ax.plot(A, d["F_raw"].mean(axis=(1, 2)), "o-", ms=3, color=C1)
ax.axvline(0, color=GRID, lw=1, zorder=0)
ax.set_xlabel("bifurcation parameter $a$")
ax.set_ylabel("mean FC (raw)")
tag(ax, "(a)")
ax = axs[1]
ax.semilogy(mid, d["unf_sraw"] / da, "o-", ms=3, color=C1, label="raw FC")
sg = d["unf_sgsr"] / da
det = d["unf_sgsr"] > d["unf_ngsr"]           # change after GSR resolved above sampling noise
ax.semilogy(mid, sg, "-", color=C2, lw=1, alpha=0.5)
ax.semilogy(mid[det], sg[det], "o", ms=3, color=C2, label="FC after GSR")
ax.semilogy(mid[~det], sg[~det], "o", ms=3, color=C2, mfc="none")
ax.semilogy(mid, d["unf_ngsr"] / da, color=MUTED, lw=1, ls="--", label="sampling noise (GSR)")
ax.axvline(0, color=GRID, lw=1, zorder=0)
ax.set_xlabel("bifurcation parameter $a$")
ax.set_ylabel(r"rms change of FC per unit $a$")
ax.legend(frameon=False, loc="upper left")
tag(ax, "(b)")
ax = axs[2]
ratio = np.where(d["unf_sraw"] > 0, d["unf_sgsr"] / np.maximum(d["unf_sraw"], 1e-12), np.nan)
up = d["unf_ngsr"] / np.maximum(d["unf_sraw"], 1e-12)
ax.semilogy(mid[det], ratio[det], "o", ms=4, color=C2, label="nonlinear model")
ax.semilogy(mid[~det], ratio[~det], "o", ms=4, color=C2, mfc="none", label="below sampling noise")
lin = d["d_ling"] / d["d_lin"]
ok = np.isfinite(lin)
ax.semilogy(mid[ok], lin[ok], "-", color=C1, label="linear theory")
ax.axvline(0, color=GRID, lw=1, zorder=0)
ax.set_xlabel("bifurcation parameter $a$")
ax.set_ylabel("sensitivity after GSR / raw")
ax.set_ylim(0.008, 2.5)
ax.legend(frameon=False, loc="lower left", fontsize=6.5)
tag(ax, "(c)")
fig.savefig(OUT / "fig6_nonlinear.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 7
t = np.load(RES / "temporal_demo.npz")
rows = t["rows"]          # c, true delta, mean est (gs, reg, reg_gsr), sd est (gs, reg, reg_gsr)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
ax = axs[0]
tr = rows[:, 1]
ax.plot([tr.min() * 0.7, 3], [tr.min() * 0.7, 3], color=GRID, lw=1, zorder=0)
ax.text(0.9, 0.6, "identity", color=MUTED, fontsize=7, rotation=38)
for j, (col, lab) in enumerate([(C1, "global signal"), (C2, "regions (median)"), (C3, "regions after GSR")]):
    ax.errorbar(tr, rows[:, 2 + j], yerr=rows[:, 5 + j], fmt="o-", ms=3, color=col, capsize=2, label=lab)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"true $\delta$ (s$^{-1}$)")
ax.set_ylabel(r"estimated $\delta$ (s$^{-1}$)")
ax.legend(frameon=False, loc="upper left")
tag(ax, "(a)")
ax = axs[1]
lag = t["LAG_D"] * 0.72
for arr, col, lab in [(t["ac_gs"], C1, "global signal"), (t["ac_reg"], C2, "regions (median)"),
                      (t["ac_regg"], C3, "regions after GSR")]:
    m, s = arr.mean(0), arr.std(0)
    ax.plot(lag, m, color=col, label=lab)
    ax.fill_between(lag, m - s, m + s, color=col, alpha=0.15, lw=0)
ax.axhline(0, color=GRID, lw=1, zorder=0)
ax.set_xlabel("lag (s)")
ax.set_ylabel("autocorrelation (7 subjects)")
ax.legend(frameon=False, loc="upper right")
tag(ax, "(b)")
ax = axs[2]
for k in range(len(t["tg"])):
    ax.plot([0, 1, 2], [t["tg"][k], t["tr"][k], t["trg"][k]], "-", color=GRID, lw=1, zorder=0)
for j, (arr, col) in enumerate([(t["tg"], C1), (t["tr"], C2), (t["trg"], C3)]):
    ax.scatter(np.full(len(arr), j), arr, s=12, color=col, zorder=2)
ax.set_xticks([0, 1, 2], ["global\nsignal", "regions", "regions\nafter GSR"])
ax.set_ylabel("integrated AC time (s)")
ax.set_xlim(-0.4, 2.4)
tag(ax, "(c)")
fig.savefig(OUT / "fig7_temporal.pdf")
plt.close(fig)
print("figures 6-7 written")
