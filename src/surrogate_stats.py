"""Per-subject surrogate tests for FCD and metastability (neurolib HCP, 7 subjects).

For each subject we draw K multivariate phase-randomized surrogates (one random phase per
frequency, shared by all regions: preserves every auto- and cross-spectrum of the band-passed
signals, hence FC and lagged covariances, and is stationary, linear and Gaussian).
Statistics compared against the surrogate null, per subject:
  * metastability (std over time of the Kuramoto order parameter of Hilbert phases);
  * FCD spread (std of the upper-triangular FCD entries), a standard non-stationarity index;
  * KS distance between the subject's FCD distribution and the pooled surrogate FCD, compared
    with the leave-one-out KS of each surrogate against the remaining ones.
Two-sided Monte Carlo p-values, p = (1 + #{null at least as extreme}) / (K + 1).
Output: results/surrogate_stats.npz and a printed table.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = 39
g = data.load_group()
TR = g["TR"]
filt = [M.filter_ts(ts, TR) for ts in g["ts"]]
rng = np.random.default_rng(12)


def phase_randomize(x):
    F = np.fft.rfft(x, axis=1)
    ph = np.exp(2j * np.pi * rng.random(F.shape[1]))
    ph[0] = 1
    if x.shape[1] % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(F * ph[None, :], n=x.shape[1], axis=1)


def p_two(obs, null):
    null = np.asarray(null)
    d = np.abs(null - null.mean())
    return (1 + np.sum(d >= abs(obs - null.mean()))) / (len(null) + 1)


rows = []
for s, x in enumerate(filt):
    sur = [phase_randomize(x) for _ in range(K)]
    fcd_e = M.fcd(x, TR)
    fcd_s = [M.fcd(y, TR) for y in sur]
    meta_e, meta_s = M.metastability(x), [M.metastability(y) for y in sur]
    sd_e, sd_s = fcd_e.std(), [f.std() for f in fcd_s]
    ks_e = M.ks(fcd_e, np.concatenate(fcd_s))
    ks_null = [M.ks(fcd_s[k], np.concatenate(fcd_s[:k] + fcd_s[k + 1:])) for k in range(K)]
    p_ks = (1 + np.sum(np.array(ks_null) >= ks_e)) / (K + 1)
    rows.append([meta_e, np.mean(meta_s), p_two(meta_e, meta_s), sd_e, np.mean(sd_s), p_two(sd_e, sd_s),
                 ks_e, np.median(ks_null), p_ks])
    r = rows[-1]
    print(f"subject {g['sids'][s]}: metastab {r[0]:.3f} vs {r[1]:.3f} (p={r[2]:.2f}) | "
          f"FCD sd {r[3]:.3f} vs {r[4]:.3f} (p={r[5]:.2f}) | KS {r[6]:.3f} vs null median {r[7]:.3f} (p={r[8]:.2f})")
rows = np.array(rows)
print(f"subjects with p<0.05: metastability {np.sum(rows[:, 2] < 0.05)}/7, FCD sd {np.sum(rows[:, 5] < 0.05)}/7, "
      f"KS {np.sum(rows[:, 8] < 0.05)}/7  (K={K} surrogates each; min attainable p={1 / (K + 1):.3f})")
np.savez(ROOT / "results" / "surrogate_stats.npz", rows=rows, sids=np.array(g["sids"]), K=K)
