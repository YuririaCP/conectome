"""Diagnóstico: ¿la FCD empírica es compatible con un proceso lineal gaussiano estacionario?

Surrogados por aleatorización de fase multivariada (misma fase aleatoria para todas las
regiones en cada frecuencia): preservan espectros y FC, destruyen no estacionariedad y
no linealidad.  También: efecto de regresión de señal global (GSR) y de las regiones que el
ajuste heterogéneo 'apaga' (a_j en el límite).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data, metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
g = data.load_group(); TR = g["TR"]
filt = [M.filter_ts(ts, TR) for ts in g["ts"]]
rng = np.random.default_rng(0)


def phase_randomize(x):
    F = np.fft.rfft(x, axis=1)
    ph = np.exp(2j * np.pi * rng.random(F.shape[1])); ph[0] = 1
    if x.shape[1] % 2 == 0: ph[-1] = 1
    return np.fft.irfft(F * ph[None, :], n=x.shape[1], axis=1)


def gsr(x):
    gs = x.mean(0)
    beta = x @ gs / (gs @ gs)
    return x - np.outer(beta, gs)


emp = np.concatenate([M.fcd(x, TR) for x in filt])
sur = np.concatenate([M.fcd(phase_randomize(x), TR) for x in filt for _ in range(3)])
emp_g = np.concatenate([M.fcd(gsr(x), TR) for x in filt])
sur_g = np.concatenate([M.fcd(phase_randomize(gsr(x)), TR) for x in filt for _ in range(3)])
meta = lambda xs: np.mean([M.metastability(x) for x in xs])
print(f"FCD media/DE  empírica={emp.mean():.3f}/{emp.std():.3f}  surrogado={sur.mean():.3f}/{sur.std():.3f}  KS={M.ks(emp, sur):.3f}")
print(f"con GSR       empírica={emp_g.mean():.3f}/{emp_g.std():.3f}  surrogado={sur_g.mean():.3f}/{sur_g.std():.3f}  KS={M.ks(emp_g, sur_g):.3f}")
print(f"metaestabilidad empírica={meta(filt):.3f}  surrogado={meta([phase_randomize(x) for x in filt]):.3f}")
print(f"FC media: sin GSR={np.mean([M.triu(M.fc(x)).mean() for x in filt]):.3f}  con GSR={np.mean([M.triu(M.fc(gsr(x))).mean() for x in filt]):.3f}")

r = np.load(ROOT / "results" / "fase2_heterogeneous.npz")
a = r["a_f"]; fcT = r["fcT"]; N = len(a)
FC = np.eye(N); FC[np.triu_indices(N, 1)] = fcT; FC = FC + np.triu(FC, 1).T
mfc = (FC.sum(1) - 1) / (N - 1)
off = np.where(-a > 0.1)[0]
print(f"regiones con |a_j|>0.1: {list(off)}")
print(f"  FC media de esas regiones: {mfc[off].round(2)}  vs resto: {np.median(np.delete(mfc, off)):.2f}")
print(f"  r(log|a_j|, FC media de la región) = {np.corrcoef(np.log(-a), mfc)[0, 1]:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
b = np.linspace(-0.2, 1, 60)
for axx, e, s, t in [(ax[0], emp, sur, "sin GSR"), (ax[1], emp_g, sur_g, "con GSR")]:
    axx.hist(e, b, density=True, alpha=0.5, color="k", label="empírica")
    axx.hist(s, b, density=True, histtype="step", lw=2, color="C3", label=f"surrogado de fase (KS={M.ks(e, s):.2f})")
    axx.set(title=f"FCD {t}", xlabel="correlación entre ventanas"); axx.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fase2_surrogados_fcd.png", dpi=130)
