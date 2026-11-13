"""Recovery study for the physical distance delta from temporal structure (synthetic fMRI-like data).

Ground truth: linearized Stuart-Landau networks with heterogeneous frequencies (complex OU, exact
discretization at TR = 0.72 s), for three couplings:
  M1 symmetric diffusive  (Schaefer-100 group SC)          l = r = 1
  M2 directed diffusive   (Phase-3 effective connectivity) r = 1, l != 1
  M3 symmetric additive   (Schaefer-100 group SC)          l = r = Perron vector (non-uniform)
delta = -max Re spec(J) is set to 0.01, 0.03 or 0.1 s^-1 by a homogeneous damping shift.
Observation model: canonical HRF convolution, a global physiological component (slow arousal-like
AR(1), tau = 20 s, plus a respiration-like narrow-band term near 0.3 Hz) with heterogeneous loading
and the same variance as the neural global signal, white measurement noise, high-pass 0.008 Hz.
A recording of the physiological component (correlation 0.9 with the truth) is available.
Projections: global mean, right critical vector r, left critical vector l (from the real part of the
model Jacobian, i.e. from structure and damping, not from the truth with frequencies).
Preprocessing: none, physiological regression, or GSR.
Estimators: naive fit of exp(-delta tau) cos(omega tau) to the autocorrelation (lags <= 60 s), and an
observation-aware fit whose model autocorrelation includes HRF, high-pass filter and white noise.
Output: results/recovery_delta.npz and a printed table of median relative error and IQR.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import brentq, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import metrics as M  # noqa: E402
from fcdata import SC_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TR = 0.72
FS = 1 / TR
N_VOL = 4 * 1200
REPS = 10
DELTAS = [0.01, 0.03, 0.1]
G = 0.5                                   # overall coupling rate scale, s^-1
LAGS = np.arange(0, int(60 / TR) + 1)
rng = np.random.default_rng(2026)

# ---------------------------------------------------------------- structures
g = data.load_group()
om_aal = 2 * np.pi * np.mean([M.peak_frequencies(ts, TR) for ts in g["ts"]], axis=0)
Sg = SC_all.mean(0)
Sg = (Sg + Sg.T) / 2
np.fill_diagonal(Sg, 0)
EC = np.load(ROOT / "results" / "fase3_ec.npz")["EC"]
np.fill_diagonal(EC, 0)


def perron(Mx):
    lam, V = np.linalg.eig(Mx)
    k = np.argmax(lam.real)
    return lam[k].real, np.abs(V[:, k].real)


def make(kind):
    if kind == "M1":
        C = Sg / Sg.sum(1).mean()
        B = -(np.diag(C.sum(1)) - C)
        r = l = np.ones(len(C))
    elif kind == "M2":
        C = EC / EC.sum(1).mean()
        B = -(np.diag(C.sum(1)) - C)
        r = np.ones(len(C))
        _, l = perron(B.T)
    else:
        lmax, r = perron(Sg)
        B = Sg / lmax - np.eye(len(Sg))
        l = r
    n = len(B)
    om = om_aal if n == len(om_aal) else rng.permutation(np.resize(om_aal, n))
    return G * B, om, r / np.linalg.norm(r), l / np.linalg.norm(l)


def jac(B, om, shift):
    n = len(B)
    A = B - shift * np.eye(n)
    return np.block([[A, -np.diag(om)], [np.diag(om), A]])


def shift_for(B, om, delta):
    f = lambda s: -np.max(np.linalg.eigvals(jac(B, om, s)).real) - delta
    return brentq(f, -1.0, 5.0)


def simulate(J, n, seed):
    m = len(J)
    Phi = expm(J * TR)
    Sig = solve_continuous_lyapunov(J, -np.eye(m))
    Qd = Sig - Phi @ Sig @ Phi.T
    Lq = np.linalg.cholesky((Qd + Qd.T) / 2 + 1e-12 * np.eye(m))
    r_ = np.random.default_rng(seed)
    u = np.linalg.cholesky(Sig + 1e-12 * np.eye(m)) @ r_.standard_normal(m)
    out = np.empty((m // 2, n))
    for t in range(n):
        u = Phi @ u + Lq @ r_.standard_normal(m)
        out[:, t] = u[: m // 2]
    return out


# ---------------------------------------------------------------- observation model
def hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    h = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6
    return h / h.sum()


H = hrf(TR)
b_hp, a_hp = signal.butter(2, 0.008, btype="high", fs=FS)


def physio(n, seed):
    r_ = np.random.default_rng(seed)
    phi = np.exp(-TR / 20.0)
    slow = signal.lfilter([1.0], [1.0, -phi], r_.standard_normal(n))
    bb, aa = signal.butter(2, [0.25, 0.35], btype="band", fs=FS)
    resp = signal.lfilter(bb, aa, r_.standard_normal(n))
    p = slow / slow.std() + 0.7 * resp / resp.std()
    return p - p.mean()


def observe(x, seed):
    r_ = np.random.default_rng(seed)
    bold = signal.lfilter(H, [1.0], x, axis=1)
    bold = bold - bold.mean(1, keepdims=True)
    n = bold.shape[1]
    p = physio(n, seed + 1)
    load = 1 + 0.3 * r_.standard_normal(bold.shape[0])
    gs_var = bold.mean(0).var()
    y = bold + np.sqrt(gs_var) * np.outer(load, p) / np.sqrt(np.mean(load) ** 2)
    y = y + np.sqrt(0.5 * bold.var(1, keepdims=True)) * r_.standard_normal(bold.shape)
    y = signal.filtfilt(b_hp, a_hp, y, axis=1)
    p_rec = 0.9 * p / p.std() + np.sqrt(1 - 0.81) * r_.standard_normal(n)
    p_rec = signal.filtfilt(b_hp, a_hp, p_rec)
    return y, p_rec


def regress_out(y, reg):
    reg = reg - reg.mean()
    return y - np.outer(y @ reg / (reg @ reg), reg)


# ---------------------------------------------------------------- estimators
def acf(s):
    s = s - s.mean()
    f = np.fft.rfft(s, 2 * len(s))
    a = np.fft.irfft(f * np.conj(f))[: LAGS[-1] + 1]
    return a[LAGS] / a[0]


NF = 8192
freqs = np.fft.rfftfreq(NF, d=TR)
Hf = np.abs(np.fft.rfft(H, NF)) ** 2
_, hp = signal.freqz(b_hp, a_hp, worN=freqs, fs=FS)
HP = np.abs(hp) ** 4                     # forward-backward pass


def model_ac(delta, om, white, aware):
    w = 2 * np.pi * freqs
    lor = delta / (delta ** 2 + (w - om) ** 2) + delta / (delta ** 2 + (w + om) ** 2)
    if not aware:
        tau = LAGS * TR
        return np.exp(-delta * tau) * np.cos(om * tau)
    spec = lor * Hf * HP
    spec = spec / spec.sum()
    ac = np.fft.irfft(spec, NF)[: LAGS[-1] + 1]
    ac = ac / ac[0]
    ac = (1 - white) * ac
    ac[0] = 1.0
    return ac


def fit(ac, aware):
    def res(p):
        return model_ac(np.exp(p[0]), p[1], p[2], aware) - ac
    best = None
    for d0 in [0.01, 0.05, 0.2]:
        try:
            s = least_squares(res, x0=[np.log(d0), 0.15, 0.2], bounds=([np.log(1e-3), 0.0, 0.0], [np.log(2.0), 1.0, 0.95]))
            if best is None or s.cost < best.cost:
                best = s
        except ValueError:
            pass
    return np.exp(best.x[0]) if best is not None else np.nan


# ---------------------------------------------------------------- run
KINDS = ["M1", "M2", "M3"]
PROJ = ["global", "r", "l"]
PREP = ["none", "physio", "GSR"]
LENS = [1200, 4800]
EST = np.full((len(KINDS), len(DELTAS), len(LENS), len(PREP), len(PROJ), 2, REPS), np.nan)
for ik, kind in enumerate(KINDS):
    B, om, r, l = make(kind)
    print(f"{kind}: cos(r,1)={r.sum() / np.sqrt(len(r)):.2f} cos(l,1)={l.sum() / np.sqrt(len(l)):.2f}", flush=True)
    for idl, delta in enumerate(DELTAS):
        J = jac(B, om, shift_for(B, om, delta))
        for rep in range(REPS):
            seed = 100000 * ik + 1000 * idl + rep
            x = simulate(J, N_VOL, seed)
            for il, n in enumerate(LENS):
                y, p_rec = observe(x[:, :n], seed + 7)
                ys = {"none": y, "physio": regress_out(y, p_rec), "GSR": regress_out(y, y.mean(0))}
                for ipp, pp in enumerate(PREP):
                    for ipj, q in enumerate([np.ones(len(r)) / len(r), r, l]):
                        ac = acf(q @ ys[pp])
                        for ia, aware in enumerate([False, True]):
                            EST[ik, idl, il, ipp, ipj, ia, rep] = fit(ac, aware)
        rel = EST[ik, idl] / delta - 1
        print(f"  delta={delta}: median rel. error (4 sessions, aware fit)  "
              + "  ".join(f"{pp}/{pj}={np.nanmedian(rel[1, ipp, ipj, 1]):+.2f}"
                          for ipp, pp in enumerate(PREP) for ipj, pj in enumerate(PROJ)), flush=True)
np.savez(ROOT / "results" / "recovery_delta.npz", EST=EST, DELTAS=DELTAS, LENS=LENS, KINDS=KINDS, PROJ=PROJ, PREP=PREP)
print("saved")
