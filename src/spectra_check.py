"""Diagnóstico: con FC correcta (EC), ¿qué separa la FCD simulada de la empírica? -> espectros.
Compara espectros de potencia y 'número efectivo de muestras' por ventana, y prueba un
surrogado híbrido: espectros del modelo + FC empírica."""
import sys
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data, hopf, metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
g = data.load_group(); TR = g["TR"]
r = np.load(ROOT / "results" / "fase3_ec.npz")
EC, om, a = r["EC"], r["om"], float(r["a"])
filt = [M.filter_ts(ts, TR) for ts in g["ts"]]
n = filt[0].shape[1]
sims = [M.filter_ts(hopf.simulate(a, om, EC, 1.0, 0.02, T=n * TR, TR=TR, seed=300 + s), TR) for s in range(7)]

def psd(xs):
    f, P = signal.welch(np.concatenate(xs, 0), fs=1 / TR, nperseg=256, axis=1)
    P = P / P.sum(1, keepdims=True)
    return f, P.mean(0)

def n_eff(xs, w=83):
    # effective independent samples in a window (Bartlett): w / (1 + 2 sum rho_k^2)
    ac = np.mean([M.autocorr(x, range(1, w)) for x in xs], axis=(0, 2))
    return w / (1 + 2 * np.sum(ac ** 2))

f, Pe = psd(filt); _, Ps = psd(sims)
cent = lambda P: np.sum(f * P) / P.sum()
print(f"frecuencia centroide: empírica={cent(Pe):.4f} Hz  modelo EC={cent(Ps):.4f} Hz")
print(f"muestras efectivas por ventana de 60 s: empírica={n_eff(filt):.1f}  modelo EC={n_eff(sims):.1f}")

# hybrid surrogate: whiten each simulated signal's spectrum to the empirical one? Instead impose
# empirical FC on model spectra: take model signals, decorrelate, recolor with empirical FC (Cholesky)
rng = np.random.default_rng(0)
FCe = np.mean([M.fc(x) for x in filt], 0)
L = np.linalg.cholesky(FCe + 1e-6 * np.eye(len(FCe)))
hyb = []
for x in sims:
    z = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    W = np.linalg.cholesky(np.corrcoef(z) + 1e-6 * np.eye(len(z)))
    hyb.append(L @ np.linalg.solve(W, z))
FCD_e = np.concatenate([M.fcd(x, TR) for x in filt])
for name, xs in [("modelo EC", sims), ("híbrido: espectros modelo + FC empírica", hyb)]:
    fcd = np.concatenate([M.fcd(x, TR) for x in xs])
    print(f"{name:42s} FCD media={fcd.mean():.3f}  KS={M.ks(fcd, FCD_e):.3f}")
print(f"{'empírica':42s} FCD media={FCD_e.mean():.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(f, Pe, "k", label="empírica"); ax.semilogy(f, Ps, "C3", label="modelo EC")
ax.set(xlim=(0, 0.12), xlabel="frecuencia (Hz)", ylabel="potencia normalizada", title="Espectro medio (BOLD filtrado)")
ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "figures" / "fase3_espectros.png", dpi=130)

# control on empirical data: keep each region's spectrum and the zero-lag FC exactly,
# destroy frequency-dependent coherence (independent phase randomization + Cholesky recoloring)
def indep_phase(x):
    F = np.fft.rfft(x, axis=1)
    ph = np.exp(2j * np.pi * rng.random(F.shape)); ph[:, 0] = 1
    return np.fft.irfft(F * ph, n=x.shape[1], axis=1)

ctrl = []
for x in filt:
    for _ in range(3):
        y = indep_phase(x)
        y = (y - y.mean(1, keepdims=True)) / y.std(1, keepdims=True)
        W = np.linalg.cholesky(np.corrcoef(y) + 1e-6 * np.eye(len(y)))
        Lx = np.linalg.cholesky(M.fc(x) + 1e-6 * np.eye(len(x)))
        ctrl.append(Lx @ np.linalg.solve(W, y))
fcd = np.concatenate([M.fcd(x, TR) for x in ctrl])
print(f"{'control: espectros+FC empíricos, sin coherencia':42s} FCD media={fcd.mean():.3f}  KS={M.ks(fcd, FCD_e):.3f}")
