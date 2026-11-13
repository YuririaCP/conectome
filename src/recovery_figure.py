"""Fig. 8: recovery of delta from synthetic fMRI-like data (results/recovery_delta.npz)."""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "results" / "recovery_delta.npz")
E, D = d["EST"], np.array(d["DELTAS"])           # E: kind, delta, len, prep, proj, aware, rep
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRIDC = "#1f1f1e", "#6b6a63", "#e4e2dc"
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "pdf.fonttype": 42,
    "font.family": "serif",
})
STRAT = [("no correction, global", 0, 0), ("physiological regression, global", 1, 0),
         ("physiological regression, left mode $l$", 1, 2), ("GSR, global", 2, 0)]
TITLES = ["symmetric diffusive ($l=r=\\mathbf{1}$)", "directed diffusive ($r=\\mathbf{1}\\neq l$)",
          "symmetric additive ($l=r\\neq\\mathbf{1}$)"]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), layout="constrained", sharey=True)
for ik, ax in enumerate(axs):
    for js, (lab, ip, ij) in enumerate(STRAT):
        rel = E[ik, :, 1, ip, ij, 1, :] / D[:, None] - 1          # 4 sessions, observation-aware fit
        q = np.percentile(rel, [25, 50, 75], axis=1)
        x = D * (1 + 0.08 * (js - 1.5))
        top = 2.5
        ax.errorbar(x, np.minimum(q[1], top), yerr=[np.minimum(q[1], top) - np.minimum(q[0], top), np.minimum(q[2], top) - np.minimum(q[1], top)],
                    fmt="o-", ms=3, capsize=2, color=COL[js], label=lab if ik == 0 else None)
        for xi, m in zip(x, q[1]):
            if m > top:
                ax.annotate(f"{m:+.1f}", (xi, top), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6, color=COL[js])
    ax.axhline(0, color=GRIDC, lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, 2.8)
    ax.set_xticks(D, [f"{v:g}" for v in D])
    ax.set_xlabel(r"true $\delta$ (s$^{-1}$)")
    ax.set_title(TITLES[ik], fontsize=7.5, color=INK)
    ax.text(-0.12 if ik else -0.25, 1.06, "(" + "abc"[ik] + ")", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
axs[0].set_ylabel(r"relative error of $\hat\delta$ (median, IQR)")
fig.legend(frameon=False, loc="outside lower center", ncol=4, fontsize=6.5)
fig.savefig(ROOT / "paper" / "figures" / "fig8_recovery.pdf")
print("figure written")
