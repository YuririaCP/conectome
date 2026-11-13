"""Check of Proposition 1 in the full nonlinear model: FC along (ca, c omega, cG) with beta rescaled
(c beta, exact invariance) versus beta held fixed (not invariant).  AAL2 group connectome, G = 0.2,
empirical frequencies, c = 4.  The rescaled run uses the rescaled time grid (TR/c, dt/c) so that the
Euler discretization is identical in rescaled time.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import hopf  # noqa: E402
import metrics as M  # noqa: E402

g = data.load_group()
TR = g["TR"]
C = (g["C"] + g["C"].T) / 2
np.fill_diagonal(C, 0)
C = C / C.sum(1).mean()
iu = np.triu_indices(len(C), 1)
om = 2 * np.pi * np.mean([M.peak_frequencies(ts, TR) for ts in g["ts"]], axis=0)
c, T = 4.0, 30000.0
rms = lambda u, v: np.sqrt(np.mean((u - v) ** 2))
for a in [-0.01, 0.005]:
    f = lambda x: np.corrcoef(x)[iu]
    b1 = f(hopf.simulate(a, om, C, 0.2, 0.02, T=T, TR=TR, dt=TR / 16, transient=500, seed=1))
    b2 = f(hopf.simulate(a, om, C, 0.2, 0.02, T=T, TR=TR, dt=TR / 16, transient=500, seed=2))
    sc = f(hopf.simulate(c * a, c * om, C, c * 0.2, c * 0.02, T=T / c, TR=TR / c, dt=TR / (16 * c), transient=500 / c, seed=3))
    fx = f(hopf.simulate(c * a, c * om, C, c * 0.2, 0.02, T=T / c, TR=TR / c, dt=TR / (16 * c), transient=500 / c, seed=4))
    print(f"a={a:+.3f}: seed noise {rms(b1, b2):.3f} | beta rescaled {rms(b1, sc):.3f} | beta fixed {rms(b1, fx):.3f}")
