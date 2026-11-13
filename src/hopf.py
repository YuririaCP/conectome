"""
Whole-brain Stuart-Landau (Hopf normal form) model.

    dz_j = [(a_j + i w_j) z_j - |z_j|^2 z_j + G * coupling_j(z)] dt + beta (dW^x_j + i dW^y_j)

coupling = "diffusive":  sum_k C_jk (z_k - z_j)      (Deco et al. 2017)
coupling = "additive":   sum_k C_jk z_k

Re(z_j) is compared with band-passed BOLD of region j.

Two ways of computing statistics:
  * `simulate`: Euler-Maruyama integration of the full nonlinear SDE.
  * `linear_*`: exact statistics of the linearization around z = 0 (a multivariate
    Ornstein-Uhlenbeck process), valid when the origin is stable and the noise is
    small relative to the distance to the bifurcation.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, signal


# ----------------------------------------------------------------------------
# Linearization
# ----------------------------------------------------------------------------

def coupling_matrix(C: np.ndarray, kind: str = "diffusive") -> np.ndarray:
    """Real N x N matrix K such that coupling(z) = K z."""
    if kind == "diffusive":
        return C - np.diag(C.sum(axis=1))  # = -Laplacian
    if kind == "additive":
        return C.copy()
    raise ValueError(f"unknown coupling kind: {kind}")


def jacobian(a, omega, C, G, kind="diffusive") -> np.ndarray:
    """Real 2N x 2N Jacobian at z = 0 acting on u = [x; y].

    Complex form: J = diag(a + i w) + G K.  With J = Ar + i Ai,
    the real form is [[Ar, -Ai], [Ai, Ar]].
    """
    N = C.shape[0]
    a = np.broadcast_to(np.asarray(a, float), (N,))
    omega = np.broadcast_to(np.asarray(omega, float), (N,))
    Ar = np.diag(a) + G * coupling_matrix(C, kind)
    Ai = np.diag(omega)
    return np.block([[Ar, -Ai], [Ai, Ar]])


def distance_to_criticality(a, omega, C, G, kind="diffusive") -> float:
    """delta = -max Re(eig(J)).  delta > 0: origin stable; delta -> 0: Hopf bifurcation.
    1/delta is the slowest relaxation time (critical slowing down)."""
    ev = np.linalg.eigvals(jacobian(a, omega, C, G, kind))
    return -float(ev.real.max())


def critical_coupling_additive(a: float, C: np.ndarray) -> float:
    """For additive coupling, homogeneous a < 0 and identical frequencies, the origin
    loses stability when a + G * lambda_max(C) = 0  ->  G_c = -a / lambda_max(C)."""
    lam = np.linalg.eigvalsh((C + C.T) / 2).max()
    return -a / lam


def linear_covariance(a, omega, C, G, beta, kind="diffusive") -> np.ndarray:
    """Stationary covariance of u = [x; y] (unfiltered), solving A S + S A^T + Q = 0."""
    A = jacobian(a, omega, C, G, kind)
    Q = beta**2 * np.eye(A.shape[0])
    return linalg.solve_continuous_lyapunov(A, -Q)


def bandpass(TR: float, low: float = 0.008, high: float = 0.08, order: int = 2):
    fs = 1.0 / TR
    return signal.butter(order, [low, high], btype="band", fs=fs)


def linear_covariance_filtered(a, omega, C, G, beta, TR, kind="diffusive",
                               band=(0.008, 0.08), order=2, nf=600, fmax=None) -> np.ndarray:
    """Covariance of x after zero-phase Butterworth band-pass (filtfilt), computed
    analytically from the cross-spectral density of the OU process:

        S(f) = (2 pi i f - A)^-1 Q (2 pi i f - A)^-H
        Cov  = 2 Re int_0^fmax |H(f)|^4 S(f) df        (filtfilt -> |H|^2 amplitude)

    Uses the eigendecomposition A = V diag(lam) V^-1 so the integral reduces to a
    matrix of scalar integrals.  Returns the N x N covariance of the x block.
    """
    A = jacobian(a, omega, C, G, kind)
    n = A.shape[0]
    N = n // 2
    lam, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    M = beta**2 * (Vinv @ Vinv.conj().T)

    b, aa = bandpass(TR, *band, order=order)
    fN = 0.5 / TR
    fmax = fN if fmax is None else min(fmax, fN)
    f = np.linspace(0.0, fmax, nf)
    _, H = signal.freqz(b, aa, worN=f, fs=1.0 / TR)
    w = np.abs(H) ** 4
    df = f[1] - f[0]
    wts = w * df
    wts[0] *= 0.5
    wts[-1] *= 0.5

    D = 1.0 / (2j * np.pi * f[:, None] - lam[None, :])  # (nf, n)
    K = (D * wts[:, None]).T @ D.conj()                  # (n, n)
    S = V @ (M * K) @ V.conj().T
    S = 2.0 * S.real
    return S[:N, :N]


class FilteredSpectrum:
    """Precomputes the filter weights on a frequency grid so that filtered lagged
    covariances of the linearized model can be evaluated quickly for many parameters.

        Cov_x(tau) = 2 Re int_0^fmax |H(f)|^4 S(f) exp(2 pi i f tau) df

    gives E[x(t + tau) x(t)^T] of the band-passed (filtfilt) signals.
    """

    def __init__(self, TR, band=(0.008, 0.08), order=2, nf=400, fmax=0.2, lags_TR=(0,)):
        b, aa = bandpass(TR, *band, order=order)
        self._ba, self._TR = (b, aa), TR
        fmax = min(fmax, 0.5 / TR)
        self.f = np.linspace(0.0, fmax, nf)
        _, H = signal.freqz(b, aa, worN=self.f, fs=1.0 / TR)
        w = np.abs(H) ** 4 * (self.f[1] - self.f[0])
        w[0] *= 0.5
        w[-1] *= 0.5
        self.lags = np.asarray(lags_TR) * TR
        # (n_lags, nf) complex weights
        self.W = w[None, :] * np.exp(2j * np.pi * self.f[None, :] * self.lags[:, None])

    def covs(self, a, omega, C, G, beta=1.0, kind="diffusive"):
        """Filtered covariance matrices of x at each lag: list of N x N arrays,
        entry [i, j] = E[x_i(t + tau) x_j(t)].  C may be asymmetric (C[j, k]: k -> j)."""
        A = jacobian(a, omega, C, G, kind)
        N = A.shape[0] // 2
        lam, V = np.linalg.eig(A)
        Vinv = np.linalg.inv(V)
        M = beta**2 * (Vinv @ Vinv.conj().T)
        D = 1.0 / (2j * np.pi * self.f[:, None] - lam[None, :])
        Vx = V[:N]
        return [2.0 * (Vx @ (M * ((D * wl[:, None]).T @ D.conj())) @ Vx.conj().T).real for wl in self.W]

    def covs_and_psd(self, a, omega, C, G, f_psd, beta=1.0, kind="diffusive"):
        """Lagged filtered covariances (as in `covs`) plus the band-passed power spectrum
        of each x_j on the frequency grid f_psd: P_j(f) = |H(f)|^4 S_jj(f), shape (N, len(f_psd)).
        """
        A = jacobian(a, omega, C, G, kind)
        N = A.shape[0] // 2
        lam, V = np.linalg.eig(A)
        Vinv = np.linalg.inv(V)
        M = beta**2 * (Vinv @ Vinv.conj().T)
        D = 1.0 / (2j * np.pi * self.f[:, None] - lam[None, :])
        Vx = V[:N]
        covs = [2.0 * (Vx @ (M * ((D * wl[:, None]).T @ D.conj())) @ Vx.conj().T).real for wl in self.W]
        Dp = 1.0 / (2j * np.pi * np.asarray(f_psd)[:, None] - lam[None, :])     # (nf, n)
        B = Vx[None, :, :] * Dp[:, None, :]                                     # (nf, N, n)
        P = np.einsum("fjk,kl,fjl->jf", B, M, B.conj(), optimize=True).real
        _, H = signal.freqz(*self._ba, worN=np.asarray(f_psd), fs=1.0 / self._TR)
        return covs, P * np.abs(H)[None, :] ** 4

    def stats(self, a, omega, C, G, beta=1.0, kind="diffusive"):
        """Returns (FC, AC): filtered correlation matrix of x (N x N) and the
        normalized autocorrelation of each x_j at the requested lags (n_lags x N).
        beta only scales covariances, so FC and AC do not depend on it."""
        A = jacobian(a, omega, C, G, kind)
        n = A.shape[0]
        N = n // 2
        lam, V = np.linalg.eig(A)
        Vinv = np.linalg.inv(V)
        M = beta**2 * (Vinv @ Vinv.conj().T)
        D = 1.0 / (2j * np.pi * self.f[:, None] - lam[None, :])
        Vx = V[:N]
        out = []
        for wl in self.W:
            K = (D * wl[:, None]).T @ D.conj()
            out.append(2.0 * (Vx @ (M * K) @ Vx.conj().T).real)
        S0 = out[0] if self.lags[0] == 0 else None
        if S0 is None:
            raise ValueError("lags_TR must start with 0")
        var = np.diag(S0)
        FC = S0 / np.sqrt(np.outer(var, var))
        AC = np.array([np.diag(S) / var for S in out[1:]])
        return FC, AC


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


# ----------------------------------------------------------------------------
# Nonlinear simulation
# ----------------------------------------------------------------------------

def simulate(a, omega, C, G, beta, T: float, TR: float, dt: float | None = None,
             kind="diffusive", transient: float = 60.0, seed=None, z0=None) -> np.ndarray:
    """Euler-Maruyama integration. Returns x = Re(z) sampled every TR, shape (N, n_TR)."""
    rng = np.random.default_rng(seed)
    N = C.shape[0]
    a = np.broadcast_to(np.asarray(a, float), (N,))
    omega = np.broadcast_to(np.asarray(omega, float), (N,))
    if dt is None:
        k = max(1, int(np.ceil(TR / 0.1)))
        dt = TR / k
    sub = int(round(TR / dt))
    K = G * coupling_matrix(C, kind)
    lin = a + 1j * omega
    sq = beta * np.sqrt(dt)

    n_out = int(round(T / TR))
    n_trans = int(round(transient / TR))
    z = (np.zeros(N, complex) if z0 is None else np.asarray(z0, complex).copy())
    out = np.empty((N, n_out))
    for t in range(n_trans + n_out):
        for _ in range(sub):
            noise = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            z = z + dt * (lin * z - (z.real**2 + z.imag**2) * z + K @ z) + sq * noise
        if t >= n_trans:
            out[:, t - n_trans] = z.real
    return out
