"""Section 2 -- Carrier window sampling: kinematic framework.

Checks, by direct numerical integration of Eqs. (2.2)-(2.8):

* Eq. (2.13): the sampled matrix of a pure state is rank one, so window
  sampling cannot turn a superposition into a mixture.
* Eq. (2.17): the closed form of the interference component, including the
  continuum limit (property 1).
* Property 2: the inter-branch coherence at the branch nodes does not depend
  on the separation d0 and saturates |rho12|^2 = rho11 * rho22.
* Property 3: the only sigma-dependent attenuation is the momentum filter
  exp[-2 w^2 sigma^2 k0^2 / (2 w^2 + sigma^2)].
* The factor exp[-d0^2 / 4(sigma^2 + 2 w^2)] is the value of the interference
  term at coinciding nodes (Delta x_nm = 0). It is already present in the
  continuum (sigma -> 0), so it is an overlap of standard quantum mechanics
  and must not be read as sampling-induced decoherence.
* Eq. (2.19): partition of unity and branch weights |c_k|^2.

Units: lengths in units of the packet width w (w = 1), hbar = 1.
"""

from __future__ import annotations

import numpy as np

from common import Report

W = 1.0          # packet width
SIGMA = 0.35     # window width (plays the role of sigma_0)
A = SIGMA        # lattice spacing, a ~ sigma_0
K0 = 0.8         # carrier wavenumber
X = np.linspace(-40.0, 40.0, 16001)
DX = X[1] - X[0]


def psi1(x, d0, k0=K0, w=W):
    """Eq. (2.3)."""
    return (2 * np.pi * w**2) ** -0.25 * np.exp(-(x + d0 / 2) ** 2 / (4 * w**2) + 1j * k0 * x)


def psi2(x, d0, k0=K0, w=W):
    """Eq. (2.4)."""
    return (2 * np.pi * w**2) ** -0.25 * np.exp(-(x - d0 / 2) ** 2 / (4 * w**2) - 1j * k0 * x)


def windows(nodes, sigma):
    """Eq. (2.7), one row per lattice node."""
    return np.exp(-(X[None, :] - nodes[:, None]) ** 2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def sampled_amplitudes(psi, nodes, sigma):
    """Eq. (2.12): phi_n = int dx W(x - x_n) psi(x)  (trapezoidal rule)."""
    return np.trapezoid(windows(nodes, sigma) * psi[None, :], dx=DX, axis=1)


def rho12_closed_form(xn, xm, d0, sigma, k0=K0, w=W):
    """Eq. (2.17)."""
    Xnm = (xn + xm) / 2
    dx = xn - xm
    s = 2 * w**2 + sigma**2
    return (np.sqrt(2 / np.pi) * w / s
            * np.exp(-Xnm**2 / s)
            * np.exp(-(dx + d0) ** 2 / (4 * (sigma**2 + 2 * w**2)))
            * np.exp(-2 * w**2 * sigma**2 * k0**2 / s)
            * np.exp(4j * k0 * w**2 * Xnm / s))


def run() -> Report:
    rep = Report("Section 2  Carrier window sampling (Eqs. 2.2-2.19)")
    nodes = A * np.arange(-80, 81)
    d0 = 16.0
    c1, c2 = 0.6, 0.8j

    # -- Eq. (2.13): rank one ------------------------------------------------
    phi = sampled_amplitudes(c1 * psi1(X, d0) + c2 * psi2(X, d0), nodes, SIGMA)
    rho = np.outer(phi, phi.conj())
    sv = np.linalg.svd(rho, compute_uv=False)
    rep.check("Eq. (2.13) sampled matrix of a pure state is rank one",
              sv[1] / sv[0] < 1e-12,
              f"second/first singular value = {sv[1] / sv[0]:.1e}")
    rep.close("Eq. (2.13) |rho_nm|^2 = rho_nn rho_mm for all n, m",
              np.abs(rho) ** 2, np.outer(np.diag(rho).real, np.diag(rho).real),
              rtol=1e-10, atol=1e-14)

    # -- Eq. (2.17): closed form ----------------------------------------------
    phi1 = sampled_amplitudes(psi1(X, d0), nodes, SIGMA)
    phi2 = sampled_amplitudes(psi2(X, d0), nodes, SIGMA)
    rho12_num = np.outer(phi1, phi2.conj())
    xn, xm = np.meshgrid(nodes, nodes, indexing="ij")
    rho12_cf = rho12_closed_form(xn, xm, d0, SIGMA)
    rep.close("Eq. (2.17) closed form matches direct integration of Eq. (2.8)",
              rho12_num, rho12_cf, rtol=1e-8, atol=1e-12)

    # -- Property 1: continuum limit --------------------------------------------
    s_small = 1e-4
    xs = np.array([-8.3, -7.9, 0.4])
    xps = np.array([7.7, 8.2, -0.6])
    exact = psi1(xs, d0) * psi2(xps, d0).conj()
    rep.close("Property 1: sigma -> 0 limit of Eq. (2.17) equals rho12(x, x') of Eq. (2.6)",
              rho12_closed_form(xs, xps, d0, s_small), exact, rtol=1e-6)

    # -- Property 2: no suppression of inter-branch coherence -------------------
    peaks = []
    for d in (8.0, 16.0, 32.0, 64.0):
        n_left = -d / 2
        n_right = d / 2
        peaks.append(abs(rho12_closed_form(n_left, n_right, d, SIGMA)))
    rep.close("Property 2: |rho12| at the branch nodes is independent of d0 (d0 = 8..64 w)",
              np.array(peaks), np.full(4, peaks[0]), rtol=1e-12)
    i_l = np.argmin(abs(nodes + d0 / 2))
    i_r = np.argmin(abs(nodes - d0 / 2))
    rho11 = abs(phi1[i_l]) ** 2
    rho22 = abs(phi2[i_r]) ** 2
    rep.close("Property 2: |rho12_nm|^2 = rho11_nn * rho22_mm at the branch nodes (closed form)",
              abs(rho12_closed_form(nodes[i_l], nodes[i_r], d0, SIGMA)) ** 2, rho11 * rho22,
              rtol=1e-8)

    # -- Property 3: momentum filtering -----------------------------------------
    ratios, targets = [], []
    for k0 in (0.0, 0.5, 1.0, 2.0, 3.0):
        p1 = sampled_amplitudes(psi1(X, d0, k0), nodes, SIGMA)
        p2 = sampled_amplitudes(psi2(X, d0, k0), nodes, SIGMA)
        p1_0 = sampled_amplitudes(psi1(X, d0, 0.0), nodes, SIGMA)
        p2_0 = sampled_amplitudes(psi2(X, d0, 0.0), nodes, SIGMA)
        ratios.append(np.max(abs(np.outer(p1, p2.conj()))) / np.max(abs(np.outer(p1_0, p2_0.conj()))))
        targets.append(np.exp(-2 * W**2 * SIGMA**2 * k0**2 / (2 * W**2 + SIGMA**2)))
    rep.close("Property 3: sigma-dependent attenuation is the momentum filter exp[-2w^2 s^2 k0^2/(2w^2+s^2)]",
              np.array(ratios), np.array(targets), rtol=2e-3)

    # -- exp[-d0^2/4(s^2+2w^2)] is not a sampling effect ------------------------
    # Value of the interference term at a single node (Delta x_nm = 0) relative
    # to its value at the branch nodes, with and without sampling.
    d = 6.0
    with_sampling = abs(rho12_closed_form(0.0, 0.0, d, SIGMA)) / abs(rho12_closed_form(-d / 2, d / 2, d, SIGMA))
    continuum = abs(psi1(0.0, d) * psi2(0.0, d).conj()) / abs(psi1(-d / 2, d) * psi2(d / 2, d).conj())
    old_factor = np.exp(-d**2 / (4 * (SIGMA**2 + 2 * W**2)))
    rep.close("exp[-d0^2/4(s^2+2w^2)] is the coherence at Delta x_nm = 0, not at the branch nodes",
              with_sampling, old_factor, rtol=1e-12)
    rep.check("... and the same factor exists without any sampling (continuum value exp(-d0^2/8w^2))",
              np.isclose(continuum, np.exp(-d**2 / (8 * W**2)), rtol=1e-12)
              and abs(np.log(with_sampling) / np.log(continuum) - 1) < 0.1,
              f"continuum={continuum:.3e}, sampled={with_sampling:.3e}: "
              "an overlap of standard QM, not decoherence")

    # -- Eq. (2.19): partition of unity and branch weights ---------------------
    xa = np.linspace(-3, 3, 7)
    lhs = np.array([[A * np.sum(np.exp(-(x - nodes) ** 2 / (2 * SIGMA**2)) * np.exp(-(xp - nodes) ** 2 / (2 * SIGMA**2)))
                     / (2 * np.pi * SIGMA**2) for xp in xa] for x in xa])
    rhs = np.exp(-(xa[:, None] - xa[None, :]) ** 2 / (4 * SIGMA**2)) / np.sqrt(4 * np.pi * SIGMA**2)
    rel = np.max(abs(lhs - rhs)) / np.max(rhs)
    rep.check("Eq. (2.19) partition of unity a*sum_n W W = W*W, error ~ exp(-pi^2 s^2/a^2)",
              rel < 3 * np.exp(-np.pi**2 * SIGMA**2 / A**2),
              f"max rel. error = {rel:.1e}, bound 3*exp(-pi^2 s^2/a^2) = {3 * np.exp(-np.pi**2):.1e}")

    pops = abs(phi) ** 2
    pop1 = A * pops[nodes < 0].sum()
    pop2 = A * pops[nodes > 0].sum()
    s2 = 1 / (4 * W**2)                                   # momentum variance of each branch
    filt = np.exp(-SIGMA**2 * K0**2 / (1 + 2 * SIGMA**2 * s2)) / np.sqrt(1 + 2 * SIGMA**2 * s2)
    rep.close("Eq. (2.19) branch population = |c_k|^2 <psi_k| exp(-s^2 p^2) |psi_k>",
              np.array([pop1, pop2]), np.array([abs(c1) ** 2, abs(c2) ** 2]) * filt, rtol=1e-3)
    rep.close("Eq. (2.19) normalised weights are exactly |c_k|^2",
              pop1 / (pop1 + pop2), abs(c1) ** 2, rtol=1e-10)
    return rep


if __name__ == "__main__":
    run().print()
