"""Results added in the revised manuscript (v6).

Every check below re-derives a statement that is new in the revision and
compares it with the equation or table printed in the paper. Equation
numbers refer to the revised manuscript.

Checks:
* Theorem 1 (Section 3.4), parts (a)-(e):
  - (a) the Lyapunov operator X -> A^T X + X A is invertible for Hurwitz A;
  - (b)-(c) the solutions of the intertwining equation A^T Q = Q A form a
    space whose dimension equals that of the commutant of A: N for a
    non-derogatory A, larger for a derogatory one (so eta is NOT unique);
  - (c) the Taussky-Zassenhaus parametrization Q = S C, C in the commutant;
  - (c) the explicit form Q = S_A^{-T} D S_A^{-1} for diagonalizable A;
  - (d) eta = [[0,Q],[Q^T,0]] has eigenvalues +/- (singular values of Q),
    hence signature (N,N), and U(t) = exp(t K) is eta-unitary;
  - (e) H_out is a maximal eta-neutral subspace.
* Example 1, Eqs. (3.24)-(3.26): Bateman's damped oscillator. Q1 and Q2 span
  the solution space; both bilinear invariants are conserved along a
  numerically integrated trajectory; det(a Q1 + b Q2) = -a^2 w^2 + g a b - b^2,
  negative for every (a, b) in the underdamped case.
* Corollary 1, Eqs. (3.29)-(3.30): the Riesz projector (computed with the
  matrix sign function) has trace N = dim(H_flow)/2 and range H_out + {0};
  initial data in its range stay bounded, any unstable component grows.
* Section 3.3: de Bruijn's identity dS/dt = D I_Fisher for pure heat flow.
* Section 5.1: Cramer-Rao bound for the Gaussian location family, and
  Eq. (5.6): the (mean, width) Gaussian family has Fisher metric
  diag(1/s^2, 2/s^2) and constant curvature -1/2.
* Eq. (5.18): the normalization kappa = nu hbar c / sigma0^4 gives
  sigma0^2 = (4 pi nu / 9) l_P^2, and sigma0 = l_P for nu = 9 / (4 pi).
* Section 5.3: the de Sitter check of the causal-diamond volume (5.19);
  a symmetric tensor vanishing on all timelike u vanishes (proof of
  Proposition 1); Einstein's equation (5.22) satisfies Postulate P2 (5.21)
  with lambda = (Lambda - 2R/3)/30 for every observer.
* Eq. (7.15): Delta S(xi)/hbar = (2Gm^2/hbar c) F(xi/sigma_eff) from
  Eqs. (7.9) and (7.13), its limits, and F(1), F(3), F(10).
* Section 7.7: the overlap and kinetic cross matrix element (7.31) of two
  3D Gaussian branches; the reduced matter energy and the bound (7.30) from
  an explicit density matrix; P_sigma(|p|^2/2) = |p|^2/2 + n sigma^2/2 and
  the bound (7.32).
* Table 4 and Section 8.2.1: visibility (8.3), Diosi-Penrose times and the
  quoted percentages, for silica spheres.
* Section 8.2.2: echo delays for 10, 30 and 60 solar masses.
"""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm, null_space, signm

from common import Report, G, HBAR, C, M_P, L_P, M_SUN, MICROGRAM

RNG = np.random.default_rng(20261020)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def random_hurwitz(n_real=2, n_pairs=2):
    """Real Hurwitz matrix with distinct real and complex-conjugate eigenvalues."""
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


def intertwiner_space(A):
    """Basis of {Q : A^T Q - Q A = 0} (column-major vec)."""
    n = A.shape[0]
    I = np.eye(n)
    M = np.kron(I, A.T) - np.kron(A.T, I)
    return [v.reshape(n, n, order="F") for v in null_space(M, rcond=1e-10).T]


def commutant_dim(A):
    n = A.shape[0]
    I = np.eye(n)
    M = np.kron(I, A) - np.kron(A.T, I)          # vec(A C - C A)
    return null_space(M, rcond=1e-10).shape[1]


def F(x):
    return 1 - np.sqrt(np.pi) / 2 * special.erf(x) / x


# ---------------------------------------------------------------------------
def theorem1_checks(rep: Report) -> None:
    A = random_hurwitz()
    N = A.shape[0]
    I = np.eye(N)

    # (a) Lyapunov operator invertible
    T = np.kron(I, A.T) + np.kron(A.T, I)
    smin = np.min(np.linalg.svd(T, compute_uv=False))
    rep.check("Theorem 1(a): Lyapunov operator X -> A^T X + X A is invertible (P = R = 0)",
              smin > 1e-6, f"smallest singular value = {smin:.3e} (N = {N})")

    # (c) dimension of the solution space = dim commutant = N (non-derogatory)
    sols = intertwiner_space(A)
    rep.check("Theorem 1(c): dim{Q : A^T Q = Q A} = dim commutant = N (eta not unique)",
              len(sols) == commutant_dim(A) == N,
              f"dim solutions = {len(sols)}, dim commutant = {commutant_dim(A)}, N = {N}")

    # (c) derogatory case: repeated eigenvalue with two Jordan-free blocks
    Ad = np.diag([-1.0, -1.0, -2.0, -0.5])
    Sd = RNG.normal(size=(4, 4)) + 2 * np.eye(4)
    Ad = Sd @ Ad @ np.linalg.inv(Sd)
    rep.check("Theorem 1(c): derogatory A (eigenvalue -1 twice) -> dimension 4 + 1 + 1 = 6 > N",
              len(intertwiner_space(Ad)) == commutant_dim(Ad) == 6,
              f"dim solutions = {len(intertwiner_space(Ad))}, dim commutant = {commutant_dim(Ad)}")

    # (c) Taussky-Zassenhaus: one symmetric nonsingular S; Q = S C spans everything
    S_tz = sum(RNG.normal() * Q for Q in sols)
    S_tz = (S_tz + S_tz.T) / 2
    rep.check("Theorem 1(c): a generic solution is symmetric and nonsingular (Taussky-Zassenhaus S)",
              np.linalg.norm(A.T @ S_tz - S_tz @ A) < 1e-8 * np.linalg.norm(S_tz)
              and abs(np.linalg.det(S_tz)) > 1e-8,
              f"residual = {np.linalg.norm(A.T @ S_tz - S_tz @ A):.1e}, det = {np.linalg.det(S_tz):.3e}")
    comm = [v.reshape(N, N, order="F") for v in
            null_space(np.kron(I, A) - np.kron(A.T, I), rcond=1e-10).T]
    span1 = np.array([Q.ravel() for Q in sols])
    span2 = np.array([(S_tz @ Cc).ravel() for Cc in comm])
    rank = np.linalg.matrix_rank(np.vstack([span1, span2]), tol=1e-8)
    rep.check("Theorem 1(c): {S C : C in commutant(A)} equals the solution space",
              rank == N, f"rank of the joint span = {rank} (expected {N})")

    # explicit spectral form Q = S_A^{-T} D S_A^{-1}
    lam, SA = np.linalg.eig(A)
    SAi = np.linalg.inv(SA)
    d = RNG.normal(size=N) + 1j * RNG.normal(size=N)
    # make D conjugation-symmetric so that Q is real
    for i in range(N):
        j = int(np.argmin(abs(lam - np.conj(lam[i]))))
        if j > i:
            d[j] = np.conj(d[i])
        elif j == i:
            d[i] = d[i].real
    Qs = SAi.T @ np.diag(d) @ SAi
    # eigenvector normalisation of conjugate pairs: rescale to make Qs real
    Qs = np.real_if_close(Qs, tol=1e8)
    rep.check("Theorem 1(c): Q = S_A^{-T} D S_A^{-1} solves A^T Q = Q A for diagonal D",
              np.linalg.norm(A.T @ Qs - Qs @ A) < 1e-8 * np.linalg.norm(Qs),
              f"residual = {np.linalg.norm(A.T @ Qs - Qs @ A):.1e}")

    # (d) signature (N,N) and eigenvalues +/- singular values, eta-unitarity
    Qn = S_tz
    Z = np.zeros((N, N))
    eta = np.block([[Z, Qn], [Qn.T, Z]])
    ev = np.sort(np.linalg.eigvalsh(eta))
    sv = np.sort(np.linalg.svd(Qn, compute_uv=False))
    rep.close("Theorem 1(d): eigenvalues of eta are +/- the singular values of Q",
              ev, np.sort(np.concatenate([-sv, sv])), rtol=1e-9)
    rep.check("Theorem 1(d): signature (N, N)", (ev > 0).sum() == N and (ev < 0).sum() == N,
              f"(+, -) = ({(ev > 0).sum()}, {(ev < 0).sum()})")
    K = np.block([[A, Z], [Z, -A]])
    U = expm(0.8 * K)
    rep.close("Theorem 1(d): U(t)^T eta U(t) = eta", U.T @ eta @ U, eta, atol=1e-9)

    # (e) H_out neutral and maximal
    X = np.vstack([RNG.normal(size=(N, 3)), np.zeros((N, 3))])
    rep.close("Theorem 1(e): H_out is eta-neutral (x^T eta y = 0 for x, y in H_out)",
              X.T @ eta @ X, np.zeros((3, 3)), atol=1e-12)
    w = np.concatenate([np.zeros(N), RNG.normal(size=N)])
    pairing = np.max(abs(np.vstack([np.eye(N), Z]).T @ eta @ w))
    rep.check("Theorem 1(e): adding any vector of H_in destroys neutrality (H_out is maximal)",
              pairing > 1e-6, f"max |<e_i, w>_eta| = {pairing:.3f}")


def bateman_checks(rep: Report) -> None:
    for g, w in ((0.3, 1.7), (1.1, 0.9)):
        A = np.array([[0.0, 1.0], [-w**2, -g]])
        Q1 = np.diag([w**2, -1.0])
        Q2 = np.array([[-g, -1.0], [-1.0, 0.0]])
        rep.close(f"Example 1, Eq. (3.25): A^T Q1 = Q1 A and A^T Q2 = Q2 A (gamma = {g}, omega = {w})",
                  [A.T @ Q1 - Q1 @ A, A.T @ Q2 - Q2 @ A], np.zeros((2, 2, 2)), atol=1e-14)
        sols = intertwiner_space(A)
        rank = np.linalg.matrix_rank(np.array([Q.ravel() for Q in sols] + [Q1.ravel(), Q2.ravel()]), tol=1e-9)
        rep.check(f"Example 1: solution space is two-dimensional and spanned by Q1, Q2 (gamma = {g})",
                  len(sols) == 2 and rank == 2, f"dim = {len(sols)}, joint rank = {rank}")

    g, w = 0.3, 1.7

    def rhs(t, y):
        x, v, yy, u = y
        return [v, -w**2 * x - g * v, u, -w**2 * yy + g * u]

    sol = solve_ivp(rhs, [0, 12], [1.0, 0.2, 0.4, -0.3], rtol=1e-12, atol=1e-13, dense_output=True)
    ts = np.linspace(0, 12, 7)
    x, v, yy, u = sol.sol(ts)
    I1 = v * u + w**2 * x * yy
    I2 = x * u - v * yy - g * x * yy
    rep.close("Example 1, Eq. (3.26): Bateman invariant xdot ydot + w^2 x y is conserved", I1, I1[0] * np.ones(7), rtol=1e-8)
    rep.close("Example 1, Eq. (3.26): Bateman invariant x ydot - xdot y - g x y is conserved", I2, I2[0] * np.ones(7), rtol=1e-8)
    # the same invariants are z^T Q_k w with z = (x, xdot), w = (y, -ydot)
    z = np.array([x, v])
    ww = np.array([yy, -u])
    Q1 = np.diag([w**2, -1.0])
    Q2 = np.array([[-g, -1.0], [-1.0, 0.0]])
    rep.close("Example 1: the invariants equal z^T Q_k w with w = (y, -ydot)",
              [np.einsum("it,ij,jt->t", z, Q1, ww), np.einsum("it,ij,jt->t", z, Q2, ww)], [I1, I2], rtol=1e-12)
    ab = RNG.normal(size=(200, 2))
    dets = np.array([np.linalg.det(a * Q1 + b * Q2) for a, b in ab])
    formula = -ab[:, 0]**2 * w**2 + g * ab[:, 0] * ab[:, 1] - ab[:, 1]**2
    rep.close("Example 1: det(a Q1 + b Q2) = -a^2 w^2 + g a b - b^2", dets, formula, rtol=1e-10)
    rep.check("Example 1: underdamped (g < 2w) -> every solution indefinite (det < 0)",
              np.all(dets < 0), f"max det over 200 random (a, b) = {dets.max():.3f}")
    go, wo = 3.0, 1.0                                    # overdamped, g > 2w
    Qdef = np.diag([wo**2, -1.0]) + np.array([[-go, -1.0], [-1.0, 0.0]])   # a = b = 1
    rep.check("Example 1 (contrast): for g > 2w a definite solution exists (a = b = 1, g = 3, w = 1)",
              np.all(np.linalg.eigvalsh(Qdef) < 0), f"eigenvalues = {np.linalg.eigvalsh(Qdef)}")


def corollary1_checks(rep: Report) -> None:
    A = random_hurwitz(n_real=3, n_pairs=2)
    N = A.shape[0]
    Z = np.zeros((N, N))
    K = np.block([[A, Z], [Z, -A]])
    Pi = (np.eye(2 * N) - signm(K).real) / 2            # Riesz projector, Eq. (3.29)
    rep.close("Corollary 1, Eq. (3.30): Tr(Pi_causal) / dim H_flow = 1/2 (matrix sign function)",
              np.trace(Pi) / (2 * N), 0.5, rtol=1e-9)
    rep.close("Corollary 1: Pi_causal is a projector onto H_out + {0}",
              Pi, np.block([[np.eye(N), Z], [Z, Z]]), atol=1e-8)
    v_in = np.concatenate([RNG.normal(size=N), np.zeros(N)])      # exactly in H_out + {0}
    rep.close("Corollary 1: Pi_causal leaves H_out + {0} invariant (Pi v = v)", Pi @ v_in, v_in, atol=1e-8)
    v_mix = v_in + 1e-6 * np.concatenate([np.zeros(N), RNG.normal(size=N)])
    norms_in = [np.linalg.norm(expm(t * K) @ v_in) for t in (0, 5, 20)]
    norms_mix = [np.linalg.norm(expm(t * K) @ v_mix) for t in (0, 5, 20)]
    rep.check("Corollary 1: data in the range of Pi_causal stay bounded (decay) ...",
              norms_in[-1] < norms_in[0], f"|U(t)v| at t = 0, 5, 20: {', '.join(f'{x:.2e}' for x in norms_in)}")
    rep.check("... while an unstable component of size 1e-6 grows without bound",
              norms_mix[-1] > 1e3 * norms_mix[0], f"|U(t)v| at t = 0, 5, 20: {', '.join(f'{x:.2e}' for x in norms_mix)}")


def entropy_and_fisher_checks(rep: Report) -> None:
    # de Bruijn: S(rho * N(0, s)) ; dS/d(s^2/2) = I_Fisher  (pure heat flow, L = 0)
    x = np.linspace(-30, 30, 60001)
    dx = x[1] - x[0]

    def mix(var):
        return 0.6 * np.exp(-(x + 2)**2 / (2 * (0.5 + var))) / np.sqrt(2 * np.pi * (0.5 + var)) + \
               0.4 * np.exp(-(x - 3)**2 / (2 * (1.2 + var))) / np.sqrt(2 * np.pi * (1.2 + var))

    def S(var):
        r = mix(var)
        return -np.sum(r * np.log(r + 1e-300)) * dx

    var, h = 0.7, 1e-4
    dS = (S(var + h) - S(var - h)) / (2 * h)               # dS / d(var); var = sigma^2
    r = mix(var)
    Ifish = np.sum(np.gradient(r, dx)**2 / (r + 1e-300)) * dx
    rep.close("Sec. 3.3 (de Bruijn): dS/d(sigma^2/2) = I_Fisher for Gaussian smoothing",
              2 * dS, Ifish, rtol=1e-5)

    # Cramer-Rao for the location family: sample mean attains sigma0^2 / M
    sigma0, M, trials = 1.3, 25, 40000
    est = RNG.normal(0.0, sigma0, size=(trials, M)).mean(axis=1)
    rep.close("Sec. 5.1: Cramer-Rao bound sigma0^2/M attained by the sample mean (Monte Carlo)",
              est.var(), sigma0**2 / M, rtol=0.03)

    # Eq. (5.6): Fisher metric of N(mu, s^2) in (mu, s)
    mu, s = 0.4, 1.7
    def score_integral(i, j):
        f = lambda t: np.exp(-(t - mu)**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)
        dmu = lambda t: (t - mu) / s**2
        ds = lambda t: -1 / s + (t - mu)**2 / s**3
        d = (dmu, ds)
        return quad(lambda t: f(t) * d[i](t) * d[j](t), -np.inf, np.inf)[0]
    g = np.array([[score_integral(i, j) for j in range(2)] for i in range(2)])
    rep.close("Eq. (5.6): Fisher metric of the (mean, width) Gaussian family = diag(1/s^2, 2/s^2)",
              g, np.diag([1 / s**2, 2 / s**2]), rtol=1e-8, atol=1e-12)

    # Gaussian curvature of ds^2 = (dmu^2 + 2 ds^2)/s^2 via Brioschi (E=1/s^2, F=0, Gm=2/s^2)
    def curvature(s0, h=1e-4):
        E = lambda s_: 1 / s_**2
        Gm = lambda s_: 2 / s_**2
        # K = -1/(2 sqrt(EG)) * d/ds( E_s / sqrt(EG) )  (metric depends on s only)
        sq = lambda s_: np.sqrt(E(s_) * Gm(s_))
        Es = lambda s_: (E(s_ + h) - E(s_ - h)) / (2 * h)
        inner = lambda s_: Es(s_) / sq(s_)
        return -1 / (2 * sq(s0)) * (inner(s0 + h) - inner(s0 - h)) / (2 * h)
    rep.close("Eq. (5.6) example: curvature of the (mean, width) family is -1/2 (s = 0.5, 1, 3)",
              [curvature(0.5), curvature(1.0), curvature(3.0)], [-0.5, -0.5, -0.5], rtol=1e-5)


def gravity_checks(rep: Report) -> None:
    # Eq. (5.18)
    nu = 9 / (4 * np.pi)
    sigma0 = np.sqrt(4 * np.pi * nu / 9) * L_P
    rep.close("Eq. (5.18): nu = 9/(4 pi) gives sigma0 = l_P", sigma0, L_P, rtol=1e-14)
    kappa = nu * HBAR * C / sigma0**4
    rep.close("Eq. (5.18): kappa sigma0^2 / 36 = c^4 / (16 pi G) with kappa = nu hbar c / sigma0^4",
              kappa * sigma0**2 / 36, C**4 / (16 * np.pi * G), rtol=1e-12)
    rep.close("Eq. (5.18): nu = 9/(4 pi) = 0.716 (an O(1) normalization)", nu, 0.7162, rtol=1e-4)

    # Eq. (5.19) de Sitter check: 1 - tau^2 R/180 + tau^2 R_uu/30 with R = 12, R_uu = -3
    rep.close("Eq. (5.19) de Sitter check: -R/180 + R_uu/30 = -1/6 for R = 12, R_uu = -3",
              -12 / 180 + (-3) / 30, -1 / 6, rtol=1e-14)

    # Proof of Proposition 1: S_ab u^a u^b = 0 for all timelike u  =>  S = 0
    eta = np.diag([-1.0, 1, 1, 1])
    us = []
    for _ in range(60):
        vv = RNG.normal(size=3) * 0.9 / np.sqrt(3)
        vv = vv * RNG.uniform(0, 0.95) / max(np.linalg.norm(vv), 1e-12)
        gam = 1 / np.sqrt(1 - vv @ vv)
        us.append(gam * np.concatenate([[1.0], vv]))
    idx = [(a, b) for a in range(4) for b in range(a, 4)]
    M = np.array([[u[a] * u[b] * (1 if a == b else 2) for (a, b) in idx] for u in us])
    rep.check("Proposition 1 (proof): S_ab u^a u^b = 0 on many timelike u forces S_ab = 0 (rank 10)",
              np.linalg.matrix_rank(M) == 10, f"rank = {np.linalg.matrix_rank(M)} of 10 components")

    # Einstein equation => Postulate P2 for every observer with lambda = (2R/3 - Lam)/30
    Ric = RNG.normal(size=(4, 4)); Ric = (Ric + Ric.T) / 2      # lower-index R_ab
    ginv = np.linalg.inv(eta)
    Rs = np.trace(ginv @ Ric)
    Lam, beta = 0.37, 0.05
    T = (Ric - 0.5 * Rs * eta + Lam * eta) / (30 * beta)          # from Eq. (5.22)
    lam = (Lam - 2 * Rs / 3) / 30
    resid = []
    for u in us[:20]:
        lhs = -(1 / 30) * u @ (Ric + Rs / 6 * eta) @ u
        rhs = -beta * u @ T @ u + lam * (u @ eta @ u)
        resid.append(lhs - rhs)
    rep.close("Proposition 1: Einstein eq. (5.22) satisfies Postulate P2 (5.21) for every observer",
              resid, np.zeros(20), atol=1e-12)


def decoherence_checks(rep: Report) -> None:
    m = 1.0e-8                                   # kg
    sig = 2.0e-4                                 # sigma_eff [m]
    for x in (0.3, 1.0, 4.0):
        xi = x * sig
        dEG = 2 * G * m**2 * (2 / (np.sqrt(np.pi) * sig) - special.erf(xi / sig) / xi)   # Eq. (7.9)
        tau = np.sqrt(np.pi) / 2 * sig / C                                                # Eq. (7.13)
        rep.close(f"Eq. (7.15): Delta E_G tau / hbar = (2Gm^2/hbar c) F(xi/sigma_eff), xi/sigma_eff = {x}",
                  dEG * tau / HBAR, 2 * G * m**2 / (HBAR * C) * F(x), rtol=1e-12)
    rep.close("Eq. (7.15): F(x) -> x^2/3 for x << 1", F(1e-3) / (1e-3**2 / 3), 1.0, rtol=1e-5)
    rep.close("Eq. (7.15): F(x) -> 1 - sqrt(pi)/(2x) for x >> 1", F(50.0), 1 - np.sqrt(np.pi) / 100, rtol=1e-12)
    rep.close("Sec. 7.3: F(1), F(3), F(10) = 0.25, 0.70, 0.91 (as quoted)",
              np.round([F(1.0), F(3.0), F(10.0)], 2), [0.25, 0.70, 0.91], rtol=0)


def energy_checks(rep: Report) -> None:
    # Two 3D Gaussian branches, |psi|^2 of variance sigma^2/2 per axis (hbar = m = 1)
    sigma = 1.0
    x = np.linspace(-25, 25, 40001)
    dx = x[1] - x[0]

    def psi1d(c):
        return (np.pi * sigma**2) ** (-0.25) * np.exp(-(x - c)**2 / (2 * sigma**2))

    for xi in (1.0, 3.0, 6.0):
        a, b = psi1d(+xi / 2), psi1d(-xi / 2)
        ov = np.sum(a * b) * dx
        kin_sep = -np.sum(b * np.gradient(np.gradient(a, dx), dx)) * dx    # <p^2> along xi
        a0 = psi1d(0.0)
        kin_perp = -np.sum(a0 * np.gradient(np.gradient(a0, dx), dx)) * dx  # = 1/(2 sigma^2)
        cross = 0.5 * (kin_sep + 2 * ov * kin_perp)                          # <2|p^2/2|1>, transverse overlap = 1
        rep.close(f"Sec. 7.7: overlap <psi2|psi1> = exp(-xi^2/4 sigma^2), xi = {xi} sigma",
                  ov, np.exp(-xi**2 / (4 * sigma**2)), rtol=1e-6)
        rep.close(f"Eq. (7.31): <psi2|p^2/2m|psi1> = (hbar^2/2m)(3/2s^2 - xi^2/4s^4) e^(-xi^2/4s^2), xi = {xi} sigma",
                  cross, 0.5 * (3 / (2 * sigma**2) - xi**2 / (4 * sigma**4)) * np.exp(-xi**2 / (4 * sigma**2)),
                  rtol=1e-5, atol=1e-10)

    # Reduced matter energy from an explicit 1D density matrix (Eq. 7.30)
    xi = 3.0
    a, b = psi1d(+xi / 2), psi1d(-xi / 2)
    c1, c2 = 0.6, 0.8 * np.exp(0.7j)
    lap = lambda f: np.gradient(np.gradient(f, dx), dx)
    H = lambda bra, ket: -0.5 * np.sum(np.conj(bra) * lap(ket)) * dx
    for dS in (0.0, 0.5, 3.0):
        e = np.exp(-dS)
        Em = (abs(c1)**2 * H(a, a) + abs(c2)**2 * H(b, b) +
              e * (c1 * np.conj(c2) * H(b, a) + np.conj(c1) * c2 * H(a, b))).real
        Em0 = (abs(c1)**2 * H(a, a) + abs(c2)**2 * H(b, b) +
               (c1 * np.conj(c2) * H(b, a) + np.conj(c1) * c2 * H(a, b))).real
        dE = Em - Em0
        dE_formula = -((1 - e) * (c1 * np.conj(c2) * H(b, a) + np.conj(c1) * c2 * H(a, b))).real
        bound = 2 * abs(c1 * c2) * abs(H(b, a))
        rep.check(f"Eq. (7.30): Delta E_m formula and |Delta E_m| <= sum |c_i c_j||<j|H|i>| (Delta S/hbar = {dS})",
                  abs(dE - dE_formula) < 1e-12 and abs(dE) <= bound + 1e-15,
                  f"Delta E_m = {dE:.4e}, formula = {dE_formula:.4e}, bound = {bound:.4e}")

    # (E3): P_sigma(|p|^2/2) = |p|^2/2 + n sigma^2 / 2  and the bound (7.32) for V = cos q
    s = 0.6
    p = np.linspace(-15, 15, 30001)
    dp = p[1] - p[0]
    kern = np.exp(-p**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)
    for p0 in (0.0, 1.3):
        smoothed = np.sum(kern * (p0 - p)**2 / 2) * dp           # 1 dimension
        rep.close(f"Sec. 7.7 (E3): P_sigma(p^2/2) = p^2/2 + sigma^2/2 per dimension (p = {p0})",
                  smoothed, p0**2 / 2 + s**2 / 2, rtol=1e-10)
    q = p
    PV = np.array([np.sum(kern * np.cos(q0 - q)) * dp for q0 in (0.0, 0.8, 2.0)])
    rep.close("Sec. 7.7 (E3): P_sigma cos q = exp(-sigma^2/2) cos q", PV,
              np.exp(-s**2 / 2) * np.cos([0.0, 0.8, 2.0]), rtol=1e-10)
    rep.check("Eq. (7.32): |(P_sigma - 1) V| <= (sigma^2/2) sup|Delta_q V| for V = cos q",
              np.max(abs((np.exp(-s**2 / 2) - 1) * np.cos(q))) <= s**2 / 2 + 1e-15,
              f"max |(P-1)V| = {np.max(abs((np.exp(-s**2 / 2) - 1) * np.cos(q))):.4f} <= {s**2 / 2:.4f}")


def observational_checks(rep: Report) -> None:
    mH = M_P / np.sqrt(2)
    rho = 2200.0
    # Table 4 rows: m [kg], R [um], (m/mH)^2, V(1), V(3), V(inf), tau_DP [s]  (as printed)
    table = [
        (1e-18, 0.048, 4.2e-21, 1, 1, 1, 3e4),
        (1e-12, 4.8, 4.2e-9, 1, 1, 1, 3e-6),
        (1e-9, 48, 4.2e-3, 0.9989, 0.997, 0.996, 3e-11),
        (1e-8, 103, 0.42, 0.90, 0.74, 0.66, 6e-13),
        (mH, 119, 1.0, 0.78, 0.49, 0.37, 3e-13),
        (3e-8, 148, 3.8, 0.38, 0.069, 0.022, 1e-13),
    ]
    ok_all, worst = True, ""
    for m, R_um, q_t, V1, V3, Vinf, tdp in table:
        R = (3 * m / (4 * np.pi * rho)) ** (1 / 3)
        seff = 2 * R / np.sqrt(5)
        qq = (m / mH) ** 2
        V = [np.exp(-qq * F(1.0)), np.exp(-qq * F(3.0)), np.exp(-qq)]
        tau = HBAR / (4 * G * m**2 / (np.sqrt(np.pi) * seff))
        def agrees(v, t):
            txt = f"{t}"
            if t == 1:                                   # printed as "1"
                return v > 0.9995
            dec = len(txt.split(".")[1]) if "." in txt else 0
            return abs(v - t) <= 0.5 * 10 ** (-dec) + 1e-12
        checks = [
            abs(R * 1e6 / R_um - 1) < 0.02,
            abs(qq / q_t - 1) < 0.03,
            all(agrees(v, t) for v, t in zip(V, (V1, V3, Vinf))),
            abs(np.log10(tau / tdp)) < np.log10(2.0),
        ]
        if not all(checks):
            ok_all = False
            worst += f" m={m:.2e}:{checks}"
    rep.check("Table 4: radii, (m/m_H)^2, visibilities (8.3) and tau_DP agree with the printed values",
              ok_all, "all six rows within rounding" if ok_all else worst)
    rep.close("Sec. 8.2.1: 1 - V <= 0.4% at 1 microgram, 34% at 10 micrograms (xi -> infinity)",
              np.round([100 * (1 - np.exp(-(MICROGRAM / mH)**2)), 100 * (1 - np.exp(-(10 * MICROGRAM / mH)**2))]),
              [0.0, 34.0], atol=0.5)
    rep.close("Sec. 8.2.1: V(30 micrograms, xi = 3 sigma_eff) = 0.07",
              np.exp(-(30 * MICROGRAM / mH)**2 * F(3.0)), 0.069, rtol=0.02)
    seff15 = 2 * (3 * mH / (4 * np.pi * rho)) ** (1 / 3) / np.sqrt(5)
    rep.check("Sec. 8.2.1: light-crossing time sigma_eff/c <~ 1e-12 s at the Heisenberg-cut mass",
              seff15 / C < 1e-12, f"sigma_eff/c = {seff15 / C:.2e} s")

    delays = [8 * G * Ms * M_SUN / C**3 * np.log(Ms * M_SUN / M_P) for Ms in (10, 30, 60)]
    rep.close("Sec. 8.2.2, Eq. (8.4): echo delays 0.035, 0.11, 0.22 s for 10, 30, 60 M_sun",
              delays, [0.035, 0.107, 0.216], rtol=0.02)


def run() -> Report:
    rep = Report("Revision  Results added in the revised manuscript (v6)")
    theorem1_checks(rep)
    bateman_checks(rep)
    corollary1_checks(rep)
    entropy_and_fisher_checks(rep)
    gravity_checks(rep)
    decoherence_checks(rep)
    energy_checks(rep)
    observational_checks(rep)
    return rep


if __name__ == "__main__":
    r = run()
    r.print()
    raise SystemExit(0 if r.all_passed else 1)
