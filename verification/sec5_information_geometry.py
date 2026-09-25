"""Section 5 -- Information geometry, volume deficit and the area-law counting.

Checks:
* Eq. (5.5): the Fisher metric of the 4D Gaussian location family is
  delta_mu_nu / sigma0^2 (Gauss-Hermite quadrature).
* Eq. (5.7): Bhattacharyya coefficient exp(-|dtheta|^2 / 8 sigma0^2).
* Eq. (5.9)-(5.10): geodesic-ball volume deficit R eps^2 / (6(n+2)), i.e.
  R/36 in four dimensions, tested on spheres and hyperbolic spaces of
  dimensions 2, 3, 4 (exact ball volumes).
* Eqs. (5.18)-(5.24): lambda_time = 1/2 (spectral projector of the doubled
  generator), lambda_space = 1/2 (chiral projector (1 + gamma5)/2 with the
  Dirac matrices), lambda_joint = 1/4 on the tensor product.
* Remark 5: projecting the full N-cell Hilbert space by Pi_joint lowers the
  entropy only by ln 4, whereas assumption (A3) (per-cell capacity) gives
  (1/4) N; the check makes the content of (A3) explicit.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import quad

from common import Report

SIGMA0 = 0.7


def fisher_metric_4d(sigma0=SIGMA0, n=20):
    """g_mu_nu = E[d_mu ln P d_nu ln P] for the Gaussian of Eq. (5.2)."""
    x, w = hermegauss(n)                  # probabilists' Hermite: weight exp(-x^2/2)
    w = w / w.sum()
    X = np.stack(np.meshgrid(x, x, x, x, indexing="ij"), axis=-1).reshape(-1, 4) * sigma0
    Wt = np.einsum("i,j,k,l->ijkl", w, w, w, w).ravel()
    score = X / sigma0**2                 # Eq. (5.3)
    return np.einsum("n,ni,nj->ij", Wt, score, score)


def bhattacharyya(dtheta, sigma0=SIGMA0):
    """int sqrt(P(x; theta1) P(x; theta2)) d^4x, as a product of 1D integrals."""
    out = 1.0
    for d in dtheta:
        p1 = lambda x: np.exp(-x**2 / (2 * sigma0**2)) / np.sqrt(2 * np.pi * sigma0**2)
        p2 = lambda x: np.exp(-(x - d) ** 2 / (2 * sigma0**2)) / np.sqrt(2 * np.pi * sigma0**2)
        out *= quad(lambda x: np.sqrt(p1(x) * p2(x)), -np.inf, np.inf)[0]
    return out


def ball_volume_deficit(n, eps, curvature_sign, a=1.0):
    """1 - Vol(B_eps)/Vol_flat(B_eps) on S^n (sign +1) or H^n (sign -1).

    The deficit is integrated directly (r^{n-1} - J(r)^{n-1}) to avoid
    cancellation between two nearly equal volumes.
    """
    if curvature_sign > 0:
        radial = lambda r: r ** (n - 1) - (a * np.sin(r / a)) ** (n - 1)
    else:
        radial = lambda r: r ** (n - 1) - (a * np.sinh(r / a)) ** (n - 1)
    num = quad(radial, 0, eps, epsabs=0, epsrel=1e-12)[0]
    return num / (eps**n / n)


def dirac_gamma5():
    s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    I2, Z2 = np.eye(2), np.zeros((2, 2))
    g0 = np.block([[I2, Z2], [Z2, -I2]])
    gs = [np.block([[Z2, si], [-si, Z2]]) for si in s]
    return 1j * g0 @ gs[0] @ gs[1] @ gs[2]


def run() -> Report:
    rep = Report("Section 5  Information geometry, volume deficit and counting")

    g = fisher_metric_4d()
    rep.close("Eq. (5.5) Fisher metric = delta_mu_nu / sigma0^2", g, np.eye(4) / SIGMA0**2, rtol=1e-10, atol=1e-12)
    for dth in ([0.3, 0, 0, 0], [0.5, -0.2, 0.9, 0.1]):
        dth = np.array(dth)
        rep.close(f"Eq. (5.7) Bhattacharyya = exp(-|dtheta|^2/8 sigma0^2), |dtheta| = {np.linalg.norm(dth):.3f}",
                  bhattacharyya(dth), np.exp(-dth @ dth / (8 * SIGMA0**2)), rtol=1e-10)

    # Volume deficit coefficient: (1 - V/V_flat)/eps^2 -> R/(6(n+2))
    for n in (2, 3, 4):
        for sgn, name in ((+1, "S"), (-1, "H")):
            R = sgn * n * (n - 1)                     # scalar curvature for a = 1
            eps = np.array([0.04, 0.02])
            coef = np.array([ball_volume_deficit(n, e, sgn) for e in eps]) / eps**2
            coef0 = (4 * coef[1] - coef[0]) / 3       # remove the O(eps^2) term
            rep.close(f"Eq. (5.9) deficit coefficient R/(6(n+2)) on {name}^{n}",
                      coef0, R / (6 * (n + 2)), rtol=1e-6)
    rep.close("Eq. (5.10) four-dimensional coefficient 1/(6(4+2)) = 1/36", 1 / (6 * (4 + 2)), 1 / 36, rtol=0)

    # lambda_time, lambda_space, lambda_joint
    rng = np.random.default_rng(7)
    N = 5
    A = -np.diag(rng.uniform(0.2, 2.0, N)) + 0.3 * rng.normal(size=(N, N))
    A -= (np.max(np.linalg.eigvals(A).real) + 0.1) * np.eye(N)          # ensure Hurwitz
    Kd = np.block([[A, np.zeros((N, N))], [np.zeros((N, N)), -A]])
    lam, V = np.linalg.eig(Kd)
    Pi_time = (V @ np.diag((lam.real < 0).astype(float)) @ np.linalg.inv(V)).real   # (1 - sgn Re K)/2
    rep.close("Eq. (5.19) lambda_time = Tr(Pi_time)/dim = 1/2", np.trace(Pi_time) / (2 * N), 0.5, rtol=1e-12)
    g5 = dirac_gamma5()
    rep.close("gamma5^2 = 1 (Dirac representation)", g5 @ g5, np.eye(4), atol=1e-14)
    Pi_space = (np.eye(4) + g5) / 2
    rep.close("Eq. (5.22) lambda_space = Tr((1 + gamma5)/2)/4 = 1/2", np.trace(Pi_space).real / 4, 0.5, rtol=1e-14)
    Pi_joint = np.kron(Pi_time, Pi_space)
    rep.close("Eq. (5.24) lambda_joint = Tr(Pi_time x Pi_space)/dim = 1/4",
              np.trace(Pi_joint).real / Pi_joint.shape[0], 0.25, rtol=1e-12)

    # Remark 5: the content of assumption (A3)
    d = 2
    Ns = np.array([10, 100, 1000])
    drop_full = Ns * np.log(d) - np.log(d**Ns.astype(float) / 4)   # projection on the full space
    rep.close("Remark 5: projecting the full N-cell space lowers ln(#states) by ln 4 only (independent of N)",
              drop_full, np.log(4) * np.ones(3), rtol=1e-9)
    # The extensive factor 1/4 therefore requires assumption (A3); it is not
    # a consequence of state counting.
    return rep


if __name__ == "__main__":
    run().print()
