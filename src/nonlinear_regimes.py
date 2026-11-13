"""Two regimes of the global-mode confound in the full Stuart-Landau model.

Sweep a across the bifurcation for several global couplings G and frequency dispersions s
(omega_j = mean + s (omega_emp_j - mean); s = 0 is a common frequency).  For each condition:
  * FC sensitivity to a, raw and after GSR, as noise-corrected rms change between neighbouring a
    (2 independent realizations per point give the sampling noise);
  * synchronization: time-averaged Kuramoto order parameter of the phases arg z_j, mean pairwise
    phase-locking value (PLV), and the PLV after removing the global phase (phases relative to the
    phase of the mean field), which isolates non-global locking;
  * mean amplitude |z|.
Usage: python nonlinear_regimes.py [seeds, e.g. 3,4,5,6]
Output: results/nonlinear_regimes.npz (seeds 1,2) or results/nonlinear_regimes_seedsA-B.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import metrics as M  # noqa: E402
from fcdata import gsr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
g = data.load_group()
TR = g["TR"]
C = (g["C"] + g["C"].T) / 2
np.fill_diagonal(C, 0)
C = C / C.sum(1).mean()
N = len(C)
iu = np.triu_indices(N, 1)
L = np.diag(C.sum(1)) - C
om_emp = 2 * np.pi * np.mean([M.peak_frequencies(ts, TR) for ts in g["ts"]], axis=0)
om_bar = om_emp.mean()
BETA, T = 0.02, 20000.0
SEEDS = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (1, 2)
OUT = "nonlinear_regimes.npz" if SEEDS == (1, 2) else f"nonlinear_regimes_seeds{SEEDS[0]}-{SEEDS[-1]}.npz"
A = np.array([-0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05])
GS = [0.1, 0.2, 0.4]
DISP = [0.0, 0.5, 1.0]


def simulate_z(a, omega, G, seed, dt=TR / 8, transient=500.0):
    rng = np.random.default_rng(seed)
    K = -G * L
    lin = a + 1j * omega
    sub = int(round(TR / dt))
    n_out, n_tr = int(round(T / TR)), int(round(transient / TR))
    z = np.zeros(N, complex)
    out = np.empty((N, n_out), complex)
    sq = BETA * np.sqrt(dt)
    for t in range(n_tr + n_out):
        for _ in range(sub):
            z = z + dt * (lin * z - (z.real ** 2 + z.imag ** 2) * z + K @ z) + sq * (
                rng.standard_normal(N) + 1j * rng.standard_normal(N))
        if t >= n_tr:
            out[:, t - n_tr] = z
    return out


def sync_metrics(z):
    ph = np.exp(1j * np.angle(z))
    R = np.abs(ph.mean(0)).mean()
    plv = np.abs(ph @ ph.conj().T / ph.shape[1])[iu].mean()
    mf = ph.mean(0)
    rel = ph * np.exp(-1j * np.angle(mf))[None, :]          # phases relative to the mean-field phase
    rel = rel - rel.mean(1, keepdims=True)
    plv_rel = np.abs(rel @ rel.conj().T / rel.shape[1])     # coherence of residual phase fluctuations
    s = np.sqrt(np.diag(plv_rel).real)
    plv_rel = (plv_rel / np.outer(s, s))[iu].mean()
    return R, plv, plv_rel, np.abs(z).mean()


shape = (len(GS), len(DISP), len(A), len(SEEDS))
Fr = np.zeros(shape + (len(iu[0]),), np.float32)
Fg = np.zeros_like(Fr)
SY = np.zeros(shape + (4,))
for i, G in enumerate(GS):
    for j, s in enumerate(DISP):
        om = om_bar + s * (om_emp - om_bar)
        for k, a in enumerate(A):
            for m, sd in enumerate(SEEDS):
                z = simulate_z(a, om, G, seed=10000 * i + 1000 * j + 10 * k + sd)
                F = np.corrcoef(z.real)
                Fr[i, j, k, m], Fg[i, j, k, m] = F[iu], gsr(F)[iu]
                SY[i, j, k, m] = sync_metrics(z)
            print(f"G={G} s={s} a={a:+.3f}  meanFC={Fr[i, j, k].mean():.3f}  R={SY[i, j, k, :, 0].mean():.3f} "
                  f"PLV={SY[i, j, k, :, 1].mean():.3f} PLVrel={SY[i, j, k, :, 2].mean():.3f}", flush=True)
np.savez(ROOT / "results" / OUT, SEEDS=SEEDS, A=A, GS=GS, DISP=DISP, Fr=Fr, Fg=Fg, SY=SY)
print("saved")
