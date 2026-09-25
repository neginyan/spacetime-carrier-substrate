"""Section 3 -- Gaussian dissipative duality and the Krein embedding.

Two finite-dimensional models are used (Assumption 1 works in a
finite-dimensional regularization):

(a) a random real Hurwitz generator A with real and complex-conjugate
    eigenvalues, small enough for a brute-force null-space computation;
(b) a phase-space grid model of Eq. (3.8): a harmonic oscillator with a
    skew-symmetric discretised Liouville operator L, a periodic phase-space
    Laplacian, and K_sigma = P L P^{-1} + D Delta with P = exp(sigma^2 Delta/2).

Checks:
* Eqs. (3.4), (3.6): Theta L Theta^{-1} = -L, also after coarse-graining.
* Eq. (3.16): Theta K Theta^{-1} = -L_sigma + D Delta has the SAME
  (contractive) spectrum as K, so the Theta image is not the mirror sector.
* Assumption 1 in model (b): simple zero mode (normalisation) and strictly
  negative real parts on the normalisation-preserving subspace.
* Theorem 1 (Eqs. 3.17-3.22): every symmetric solution eta of
  eta K + K^T eta = 0 has P = R = 0 (brute-force null space).
* Eqs. (3.23)-(3.24): the explicit Q for an oscillatory Jordan block.
* Eq. (3.25): eta = [[0,Q],[Q,0]] has split signature (N,N) and
  U(t)^T eta U(t) = eta (pseudo-unitarity).
* Eqs. (3.26)-(3.36): {C, K} = 0 and Tr(Pi_causal)/N_flow = 1/2.
* Remark 2 (Eq. 3.14): for V = 0, P L P^{-1} = L + sigma^2 d_q d_p, the
  entropy term Sigma = sigma^2 int d_q rho d_p rho / rho, and the bound
  |Sigma| <= (sigma^2/2) I_Fisher.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import linear_sum_assignment

from common import Report

RNG = np.random.default_rng(20260926)


# ---------------------------------------------------------------------------
# Model (a): random Hurwitz generator
# ---------------------------------------------------------------------------
def random_hurwitz(n_real=2, n_pairs=2):
    blocks = [np.array([[-g]]) for g in RNG.uniform(0.3, 2.0, n_real)]
    for g, w in zip(RNG.uniform(0.2, 1.5, n_pairs), RNG.uniform(0.5, 3.0, n_pairs)):
        blocks.append(np.array([[-g, w], [-w, -g]]))
    n = sum(b.shape[0] for b in blocks)
    B = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        B[i:i + k, i:i + k] = b
        i += k
    S = RNG.normal(size=(n, n)) + 2 * np.eye(n)
    return S @ B @ np.linalg.inv(S)


def spectral_Q(A):
    """Q = S^{-T} S^{-1} (Theorem 1 with D = I); real symmetric for real A."""
    _, S = np.linalg.eig(A)
    Si = np.linalg.inv(S)
    Q = Si.T @ Si
    return np.real_if_close(Q, tol=1e6).real


def krein_solutions(K):
    """All symmetric eta with eta K + K^T eta = 0 (brute-force null space)."""
    n = K.shape[0]
    idx = [(i, j) for i in range(n) for j in range(i, n)]
    cols = []
    for (i, j) in idx:
        E = np.zeros((n, n))
        E[i, j] = E[j, i] = 1.0
        cols.append((E @ K + K.T @ E).ravel())
    M = np.array(cols).T
    ns = null_space(M)
    sols = []
    for v in ns.T:
        E = np.zeros((n, n))
        for c, (i, j) in zip(v, idx):
            E[i, j] = E[j, i] = c
        sols.append(E)
    return sols


# ---------------------------------------------------------------------------
# Model (b): phase-space grid
# ---------------------------------------------------------------------------
def grid_model(n=10, L=6.0, sigma=0.4, D=0.3):
    z = np.linspace(-L / 2, L / 2, n, endpoint=False) + L / (2 * n)
    h = z[1] - z[0]
    I = np.eye(n)
    Dz = (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1)) / (2 * h)       # skew
    Lap1 = (np.roll(I, 1, axis=1) + np.roll(I, -1, axis=1) - 2 * I) / h**2
    q = p = z
    # index (i_q, j_p) -> i*n + j ; L = p d_q - V'(q) d_p with V = q^2/2
    Lv = np.kron(Dz, np.diag(p)) - np.kron(np.diag(q), Dz)
    Lap = np.kron(Lap1, I) + np.kron(I, Lap1)
    P = expm(sigma**2 / 2 * Lap)
    Ls = P @ Lv @ np.linalg.inv(P)
    K = Ls + D * Lap
    R = np.fliplr(I)                                                     # p -> -p
    Theta = np.kron(I, R)
    return dict(L=Lv, Lap=Lap, P=P, Ls=Ls, K=K, Theta=Theta, n=n)


def run() -> Report:
    rep = Report("Section 3  Dissipative duality and Krein embedding (Eqs. 3.4-3.36)")

    # ---------------- Model (b): time reversal and Assumption 1 ---------------
    m = grid_model()
    Th, L, Ls, K, Lap = m["Theta"], m["L"], m["Ls"], m["K"], m["Lap"]
    rep.close("Eq. (3.4) Theta L Theta^{-1} = -L (grid model)", Th @ L @ Th, -L, atol=1e-12)
    rep.close("Eq. (3.6) Theta L_sigma Theta^{-1} = -L_sigma", Th @ Ls @ Th, -Ls, atol=1e-9)
    ev_K = np.linalg.eigvals(K)
    ev_ThK = np.linalg.eigvals(Th @ K @ Th)
    # match the two spectra element by element (optimal assignment)
    rows, cols = linear_sum_assignment(abs(ev_K[:, None] - ev_ThK[None, :]))
    rep.close("Eq. (3.16) Theta K Theta^{-1} has the same spectrum as K (still contractive)",
              ev_ThK[cols], ev_K[rows], atol=1e-8)
    rep.check("... whereas the mirror generator -K is expansive (max Re > 0)",
              np.max((-ev_K).real) > 1e-3,
              f"max Re spec(K) = {np.max(ev_K.real):.1e},  max Re spec(-K) = {np.max((-ev_K).real):.3f}")

    ones = np.ones(K.shape[0]) / np.sqrt(K.shape[0])
    rep.close("Assumption 1: probability conservation, 1^T K = 0 (left zero mode)",
              ones @ K, np.zeros(K.shape[0]), atol=1e-10)
    B = null_space(ones[None, :])                  # normalisation-preserving subspace
    A_b = B.T @ K @ B
    ev_A = np.linalg.eigvals(A_b)
    rep.check("Assumption 1: K restricted to the normalisation-preserving subspace is Hurwitz",
              np.max(ev_A.real) < 0,
              f"max Re lambda = {np.max(ev_A.real):.4f} (N = {A_b.shape[0]})")

    # ---------------- Theorem 1 on model (a): brute force ---------------------
    A = random_hurwitz()
    N = A.shape[0]
    Z = np.zeros((N, N))
    K2 = np.block([[A, Z], [Z, -A]])
    sols = krein_solutions(K2)
    diag_norm = max(np.linalg.norm(E[:N, :N]) + np.linalg.norm(E[N:, N:]) for E in sols)
    rep.check("Theorem 1: every symmetric eta with eta K + K^T eta = 0 has P = R = 0 (brute-force null space)",
              diag_norm < 1e-10,
              f"{len(sols)} independent solutions, max |P|+|R| = {diag_norm:.1e}")
    lam = np.linalg.eigvals(A)
    pair_min = np.min(abs(lam[:, None] + lam[None, :]))
    rep.check("Eq. (3.21) no pairwise sum lambda_i + lambda_j vanishes (Lyapunov operator invertible)",
              pair_min > 1e-3, f"min |lambda_i + lambda_j| = {pair_min:.3f}")

    Q = spectral_Q(A)
    rep.close("Eq. (3.20) spectral solution Q = S^{-T}S^{-1} satisfies A^T Q = Q A", A.T @ Q, Q @ A, atol=1e-9)
    rep.close("... and Q is real symmetric", Q, Q.T, atol=1e-10)
    Ablk = np.array([[-0.7, 1.9], [-1.9, -0.7]])
    Qblk = np.diag([1.0, -1.0])
    rep.close("Eqs. (3.23)-(3.24) Jordan block: A_block^T Q_block = Q_block A_block",
              Ablk.T @ Qblk, Qblk @ Ablk, atol=1e-15)

    eta = np.block([[Z, Q], [Q, Z]])
    rep.close("Eq. (3.17) eta = [[0,Q],[Q,0]] satisfies eta K + K^T eta = 0",
              eta @ K2 + K2.T @ eta, np.zeros_like(eta), atol=1e-9)
    w = np.linalg.eigvalsh(eta)
    rep.check("Eq. (3.25) signature of eta is split (N, N)",
              (np.sum(w > 1e-9), np.sum(w < -1e-9)) == (N, N),
              f"(n+, n-) = ({np.sum(w > 1e-9)}, {np.sum(w < -1e-9)}), N = {N}")
    errs = [np.max(abs(expm(t * K2).T @ eta @ expm(t * K2) - eta)) / np.max(abs(eta)) for t in (0.3, 1.0, 2.5)]
    rep.check("Pseudo-unitarity U(t)^T eta U(t) = eta for t = 0.3, 1, 2.5",
              max(errs) < 1e-9, f"max rel. deviation = {max(errs):.1e}")

    # Same construction on the grid model's generator
    N_b = A_b.shape[0]
    Q_b = spectral_Q(A_b)
    Zb = np.zeros((N_b, N_b))
    K2b = np.block([[A_b, Zb], [Zb, -A_b]])
    eta_b = np.block([[Zb, Q_b], [Q_b, Zb]])
    res = np.max(abs(eta_b @ K2b + K2b.T @ eta_b)) / (np.max(abs(eta_b)) * np.max(abs(K2b)))
    rep.check(f"Theorem 1 construction on the grid model (N = {N_b}): eta K + K^T eta = 0",
              res < 1e-8, f"relative residual = {res:.1e}")

    # ---------------- Chiral grading and the 1/2 capacity ---------------------
    I = np.eye(N)
    Cs = np.block([[Z, I], [I, Z]])
    rep.close("Eq. (3.27) C^2 = I", Cs @ Cs, np.eye(2 * N), atol=0)
    rep.close("Eq. (3.29) {C, K} = 0", Cs @ K2 + K2 @ Cs, np.zeros_like(K2), atol=1e-12)
    for label, KK in (("random model", K2), ("grid model", K2b)):
        ev = np.linalg.eigvals(KK)
        ratio = np.sum(ev.real < 0) / KK.shape[0]
        rep.close(f"Eq. (3.31) Tr(Pi_causal)/N_flow = 1/2 ({label})", ratio, 0.5, rtol=0)

    # ---------------- Remark 2 (free particle) --------------------------------
    rep.rows.extend(remark2_checks().rows)
    return rep


def remark2_checks() -> Report:
    """Free particle: P L P^{-1} = L + sigma^2 d_q d_p and the Sigma bound."""
    rep = Report("")
    n, Lbox = 256, 24.0
    z = np.linspace(-Lbox / 2, Lbox / 2, n, endpoint=False)
    h = z[1] - z[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    Q, Pm = np.meshgrid(z, z, indexing="ij")
    KQ, KP = np.meshgrid(k, k, indexing="ij")

    def dq(f):
        return np.fft.ifft2(1j * KQ * np.fft.fft2(f)).real

    def dp(f):
        return np.fft.ifft2(1j * KP * np.fft.fft2(f)).real

    def heat(f, s2):  # exp(s2/2 * Laplacian), s2 >= 0
        return np.fft.ifft2(np.exp(-s2 / 2 * (KQ**2 + KP**2)) * np.fft.fft2(f)).real

    def gauss(C, mu=(0.0, 0.0), grad=False):
        Ci = np.linalg.inv(C)
        dq_, dp_ = Q - mu[0], Pm - mu[1]
        e = Ci[0, 0] * dq_**2 + 2 * Ci[0, 1] * dq_ * dp_ + Ci[1, 1] * dp_**2
        g = np.exp(-e / 2) / (2 * np.pi * np.sqrt(np.linalg.det(C)))
        if not grad:
            return g
        # analytic gradient: -C^{-1}(z - mu) g
        gq = -(Ci[0, 0] * dq_ + Ci[0, 1] * dp_) * g
        gp = -(Ci[1, 0] * dq_ + Ci[1, 1] * dp_) * g
        return g, gq, gp

    sigma = 0.6
    C = np.array([[1.8, 0.7], [0.7, 1.3]])
    rho = gauss(C, (0.4, -0.3))
    pre = gauss(C - sigma**2 * np.eye(2), (0.4, -0.3))      # = P^{-1} rho, exactly
    lhs = heat(Pm * dq(pre), sigma**2)                        # P L P^{-1} rho, L = p d_q
    rhs = Pm * dq(rho) + sigma**2 * dq(dp(rho))
    rep.close("Remark 2: P L P^{-1} = L + sigma^2 d_q d_p for V = 0 (spectral test)",
              lhs, rhs, rtol=0, atol=1e-10 * np.max(abs(rhs)))

    # Sigma = -int (L_sigma rho) ln rho  vs  sigma^2 int d_q rho d_p rho / rho
    # Gaussian mixture with analytic gradients (avoids dividing numerical
    # noise by the tiny density in the tails of the box)
    g1, g1q, g1p = gauss(C, (0.4, -0.3), grad=True)
    g2, g2q, g2p = gauss(np.array([[0.9, -0.5], [-0.5, 1.1]]), (-1.2, 1.0), grad=True)
    mix, mq, mp = 0.6 * g1 + 0.4 * g2, 0.6 * g1q + 0.4 * g2q, 0.6 * g1p + 0.4 * g2p
    Ls_mix = Pm * dq(mix) + sigma**2 * dq(dp(mix))
    sig_def = -np.sum(Ls_mix * np.log(mix)) * h * h
    sig_eq = sigma**2 * np.sum(mq * mp / mix) * h * h
    fisher = np.sum((mq**2 + mp**2) / mix) * h * h
    rep.close("Eq. (3.14) Sigma = sigma^2 int d_q rho d_p rho / rho (from its definition in Eq. 3.13)",
              sig_def, sig_eq, rtol=1e-6)
    rep.check("Eq. (3.14) |Sigma| <= (sigma^2/2) I_Fisher",
              abs(sig_eq) <= sigma**2 / 2 * fisher,
              f"|Sigma| = {abs(sig_eq):.4f} <= {sigma**2 / 2 * fisher:.4f}")
    return rep


if __name__ == "__main__":
    run().print()
