"""Figures 3-5 of the paper (English labels, vector PDF), from saved results and raw data.

Fig. 3  FCD of empirical data vs multivariate phase-randomized surrogates (neurolib HCP, 7 subjects),
        and the nonlinear Hopf sweep: FCD fit improves with G while mean FC overshoots.
Fig. 4  Relative index kappa vs mean FC over 800 sessions (Domhof HCP), without and with GSR,
        and four-session reliability, including the other-subject connectome control.
Fig. 5  Additive coupling: Perron-vector alignment with the uniform vector, and model dependence
        of the individual index after GSR.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import metrics as M  # noqa: E402
from fcdata import SC_all, FC_all, prep_fc, iu, icc_31, N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "font.family": "serif",
})


def tag(ax, s):
    ax.text(-0.2, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)


# ---------------------------------------------------------------- Fig. 3
g = data.load_group()
TR = g["TR"]
filt = [M.filter_ts(ts, TR) for ts in g["ts"]]
rng = np.random.default_rng(0)


def phase_randomize(x):
    F = np.fft.rfft(x, axis=1)
    ph = np.exp(2j * np.pi * rng.random(F.shape[1]))
    ph[0] = 1
    if x.shape[1] % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(F * ph[None, :], n=x.shape[1], axis=1)


def gsr_ts(x):
    gs = x.mean(0)
    return x - np.outer(x @ gs / (gs @ gs), gs)


emp = np.concatenate([M.fcd(x, TR) for x in filt])
sur = np.concatenate([M.fcd(phase_randomize(x), TR) for x in filt for _ in range(3)])
emp_g = np.concatenate([M.fcd(gsr_ts(x), TR) for x in filt])
sur_g = np.concatenate([M.fcd(phase_randomize(gsr_ts(x)), TR) for x in filt for _ in range(3)])

r1 = np.load(RES / "fase1_group_fit.npz")
Gs, ks, mfc, rfc = r1["G_sim"], r1["sim_ks"], r1["sim_meanfc"], r1["sim_r"]
o = np.argsort(Gs)
Gs, ks, mfc, rfc = Gs[o], ks[o], mfc[o], rfc[o]
mfc_emp = M.triu(r1["FC_emp"]).mean()

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
bins = np.linspace(-0.2, 1, 49)
for ax, e, s, lab, t in [(axs[0], emp, sur, "(a)", "without GSR"), (axs[1], emp_g, sur_g, "(b)", "with GSR")]:
    ax.hist(e, bins, density=True, color=C1, alpha=0.55, lw=0, label="empirical")
    ax.hist(s, bins, density=True, histtype="step", color=C2, lw=1.5, label="phase surrogate")
    ax.text(0.03, 0.95, f"{t}\nKS = {M.ks(e, s):.3f}", transform=ax.transAxes, va="top", color=INK, fontsize=7)
    ax.set_xlabel("FCD entry (window correlation)")
    ax.set_ylabel("density")
    tag(ax, lab)
axs[1].legend(frameon=False, loc="center right", bbox_to_anchor=(1.1, 0.7), handlelength=0.9, handletextpad=0.4)
ax = axs[2]
ax.plot(Gs, ks, color=C1, label="KS distance of FCD")
ax.plot(Gs, mfc, color=C2, label="model mean FC")
ax.axhline(mfc_emp, color=C2, lw=1, ls="--")
ax.text(Gs[-1], mfc_emp - 0.07, f"empirical mean FC = {mfc_emp:.2f}", color=INK, fontsize=7, ha="right")
ax.set_xlabel("coupling $G$ (nonlinear model)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=1)
tag(ax, "(c)")
fig.savefig(OUT / "fig3_fcd.pdf")
plt.close(fig)
print(f"Fig3: KS {M.ks(emp, sur):.3f} / GSR {M.ks(emp_g, sur_g):.3f}; empirical mean FC {mfc_emp:.3f}")

# ---------------------------------------------------------------- Fig. 4
r5 = np.load(RES / "fase5_fc_only.npz")
rc = np.load(RES / "fase5_control_sc.npz")
meanfc = r5["meanfc"]
lk0, lk1 = np.log(r5["KA_grupo_0"]), np.log(r5["KA_grupo_1"])
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
for ax, lk, lab, t in [(axs[0], lk0, "(a)", "without GSR"), (axs[1], lk1, "(b)", "with GSR")]:
    ax.scatter(meanfc.ravel(), lk.ravel(), s=4, color=C1, alpha=0.45, lw=0)
    r = np.corrcoef(meanfc.ravel(), lk.ravel())[0, 1]
    ax.text(0.97, 0.95 if lab == "(a)" else 0.05, f"{t}\n$r = {r:+.3f}$", transform=ax.transAxes, ha="right", va="top" if lab == "(a)" else "bottom", color=INK, fontsize=7)
    ax.set_xlabel("mean FC of the session")
    ax.set_ylabel(r"$\log\kappa$")
    tag(ax, lab)
ax = axs[2]
vals = [("group SC\nno GSR", icc_31(lk0)), ("group SC\nGSR", icc_31(lk1)),
        ("own SC\nGSR", icc_31(np.log(r5["KA_propia_1"]))), ("other's SC\nGSR", icc_31(np.log(rc["KA_x"])))]
cols = [MUTED, C1, C1, C2]
x = np.arange(len(vals))
ax.bar(x, [v for _, v in vals], width=0.6, color=cols)
for xi, (_, v) in zip(x, vals):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=7)
ax.set_xticks(x, [n for n, _ in vals], fontsize=6.5)
ax.set_ylabel(r"ICC(3,1) of $\log\kappa$, 4 sessions")
ax.set_ylim(0, 0.85)
tag(ax, "(c)")
fig.savefig(OUT / "fig4_kappa.pdf")
plt.close(fig)
print("Fig4 ICC:", [f"{n.split()[0]} {v:.3f}" for n, v in vals])

# ---------------------------------------------------------------- Fig. 5
def perron_cos(C):
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    v = np.abs(np.linalg.eigh(C)[1][:, -1])
    return v.sum() / (np.sqrt(len(v)) * np.linalg.norm(v)), 2 * np.sqrt(v.min() * v.max()) / (v.min() + v.max())


SCg = SC_all.mean(0)
alphas = np.linspace(0.1, 1.0, 19)
cg = np.array([perron_cos(SCg ** a) for a in alphas])
ind = np.array([perron_cos(SC_all[s])[0] for s in range(SC_all.shape[0])])
r7 = np.load(RES / "fase7_additive.npz", allow_pickle=True)
lkm = {k: np.log(r7[f"{k}|1|KA"]).mean(1) for k in ["aditivo SC", "aditivo EC", "difusivo SC"]}

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
ax = axs[0]
ax.plot(alphas, cg[:, 0], color=C1, label=r"$\cos\angle(v,\mathbf{1})$, group SC$^{\alpha}$")
ax.plot(alphas, cg[:, 1], color=MUTED, ls="--", lw=1, label="Pólya–Szegő lower bound")
ax.scatter(np.full(len(ind), 1.03), ind, s=4, color=C2, alpha=0.4, lw=0, label=r"individual SC ($\alpha=1$)")
ax.set_xlabel(r"weight exponent $\alpha$")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower left", markerscale=2)
tag(ax, "(a)")
for ax, (ka, kb), lab, xl, yl in [
        (axs[1], ("aditivo SC", "difusivo SC"), "(b)", "additive, SC", "diffusive, SC"),
        (axs[2], ("aditivo SC", "aditivo EC"), "(c)", "additive, SC", "additive, EC")]:
    ax.scatter(lkm[ka], lkm[kb], s=6, color=C1, alpha=0.6, lw=0)
    r = np.corrcoef(lkm[ka], lkm[kb])[0, 1]
    ax.text(0.03, 0.95 if lab == "(b)" else 0.05, f"$r = {r:+.2f}$", transform=ax.transAxes, ha="left", va="top" if lab == "(b)" else "bottom", color=INK, fontsize=7)
    ax.set_xlabel(rf"$\log\kappa$, {xl} (GSR)")
    ax.set_ylabel(rf"$\log\kappa$, {yl} (GSR)")
    tag(ax, lab)
fig.savefig(OUT / "fig5_additive.pdf")
plt.close(fig)
print(f"Fig5: group raw cos {cg[-1, 0]:.3f}; individual {ind.min():.2f}-{ind.max():.2f}; "
      f"r(add SC, dif SC) {np.corrcoef(lkm['aditivo SC'], lkm['difusivo SC'])[0, 1]:.3f}; "
      f"r(add SC, add EC) {np.corrcoef(lkm['aditivo SC'], lkm['aditivo EC'])[0, 1]:.3f}")
