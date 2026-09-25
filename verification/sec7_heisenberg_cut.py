"""Section 7 -- Gravitational back-reaction and the Heisenberg cut.

Checks:
* Eqs. (7.6)-(7.8): the interaction energy of two Gaussian branches with the
  erf-regularised kernel equals -G m^2 erf(xi/sigma_eff)/xi with
  sigma_eff^2 = 2 sigma^2 + sigma0^2.  Verified twice: by a 1D radial
  quadrature and by a 6D Monte Carlo integral over both branch densities.
* Eq. (7.15): the same for extended bodies, sigma_eff^2 = 2 sigma^2 + 4 s_m^2 + sigma0^2.
* Eqs. (7.9)-(7.12): Delta E_G >= 0, E_int(0), saturation value.
* Eq. (7.13): tau = sqrt(pi) sigma_eff / (2c).
* Eq. (7.14): Delta S = 2 G m^2 / c, independent of sigma_eff.
* Eq. (7.16): small-separation limit Delta S = (2 G m^2 / 3c)(xi/sigma_eff)^2.
* Eqs. (7.17)-(7.19): m_H = M_P / sqrt(2) = 15.39 micrograms.
* The alpha-rescaling statements after Eq. (7.13) and Eq. (7.19).
* Eqs. (7.20)-(7.23): -ln B12 = Delta S / hbar under the contact condition.
* Sec. 4.3: m_H < M_hor (no horizon at the cut).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from common import Report, G, HBAR, C, M_P, L_P, MICROGRAM

RNG = np.random.default_rng(314159)


def E_int_closed(xi, m, sigma_eff):
    """Eq. (7.8) in units G = 1."""
    return -m**2 * erf(xi / sigma_eff) / xi


def E_int_radial(xi, m, rel_var, sigma0):
    """-G m^2 E[erf(|xi + u|/sigma0)/|xi + u|], u ~ N(0, rel_var * I_3).

    The distribution of s = |xi e + u| is
    p(s) = s / (xi sqrt(2 pi v)) [exp(-(s-xi)^2/2v) - exp(-(s+xi)^2/2v)].
    """
    v = rel_var
    p = lambda s: s / (xi * np.sqrt(2 * np.pi * v)) * (np.exp(-(s - xi) ** 2 / (2 * v)) - np.exp(-(s + xi) ** 2 / (2 * v)))
    F = lambda s: erf(s / sigma0) / s if s > 0 else 2 / (np.sqrt(np.pi) * sigma0)
    val = quad(lambda s: F(s) * p(s), 0, xi + 12 * np.sqrt(v), points=[xi], limit=200, epsabs=0, epsrel=1e-12)[0]
    return -m**2 * val


def E_int_montecarlo(xi, m, sigma, sigma0, s_m=0.0, n=2_000_000):
    """6D Monte Carlo over the two branch densities rho_i = m |psi_i|^2 (Eq. 7.6).

    |psi_i|^2 = (pi sigma^2)^{-3/2} exp(-|x - x_i|^2 / sigma^2): per-axis variance
    sigma^2/2; an extended body adds its internal Gaussian of variance s_m^2.
    """
    sd = np.sqrt(sigma**2 / 2 + s_m**2)
    x1 = RNG.normal(0.0, sd, size=(n, 3))
    x2 = RNG.normal(0.0, sd, size=(n, 3))
    x2[:, 0] += xi
    d = np.linalg.norm(x1 - x2, axis=1)
    vals = erf(d / sigma0) / d
    return -m**2 * vals.mean(), m**2 * vals.std() / np.sqrt(n)


def delta_E(xi, m, sigma_eff):
    """Eq. (7.9), with a series for small xi/sigma_eff to avoid cancellation."""
    x = xi / sigma_eff
    if x < 1e-3:
        # erf(x)/x = (2/sqrt(pi)) (1 - x^2/3 + x^4/10 - ...)
        return 2 * m**2 * 2 / (np.sqrt(np.pi) * sigma_eff) * (x**2 / 3 - x**4 / 10)
    return 2 * m**2 * (2 / (np.sqrt(np.pi) * sigma_eff) - erf(x) / xi)


def run() -> Report:
    rep = Report("Section 7  Gravitational back-reaction and the Heisenberg cut")
    m, sigma, sigma0 = 1.0, 0.8, 0.5
    s_eff = np.sqrt(2 * sigma**2 + sigma0**2)

    # ---------------- Eqs. (7.6)-(7.8) ------------------------------------------
    xis = np.array([0.2, 0.9, 2.0, 5.0])
    rad = np.array([E_int_radial(x, m, sigma**2, sigma0) for x in xis])
    rep.close("Eq. (7.8) E_int = -G m^2 erf(xi/sigma_eff)/xi, sigma_eff^2 = 2 sigma^2 + sigma0^2 (radial quadrature)",
              rad, E_int_closed(xis, m, s_eff), rtol=1e-9)
    mc = [E_int_montecarlo(x, m, sigma, sigma0) for x in (0.9, 2.0)]
    ok = all(abs(v - E_int_closed(x, m, s_eff)) < 5 * e for (v, e), x in zip(mc, (0.9, 2.0)))
    rep.check("Eq. (7.6) direct 6D Monte Carlo over both branch densities agrees with Eq. (7.8) (within 5 s.e.)",
              ok, "; ".join(f"xi={x}: MC={v:.5f}+-{e:.5f}, closed={E_int_closed(x, m, s_eff):.5f}"
                             for (v, e), x in zip(mc, (0.9, 2.0))))

    # ---------------- Eq. (7.15): extended bodies ---------------------------------
    s_m = 1.1
    s_eff_ext = np.sqrt(2 * sigma**2 + 4 * s_m**2 + sigma0**2)
    rad_ext = np.array([E_int_radial(x, m, sigma**2 + 2 * s_m**2, sigma0) for x in xis])
    rep.close("Eq. (7.15) extended body: sigma_eff^2 = 2 sigma^2 + 4 s_m^2 + sigma0^2 (radial quadrature)",
              rad_ext, E_int_closed(xis, m, s_eff_ext), rtol=1e-9)
    v, e = E_int_montecarlo(2.0, m, sigma, sigma0, s_m=s_m)
    rep.check("Eq. (7.15) extended body, 6D Monte Carlo (within 5 s.e.)",
              abs(v - E_int_closed(2.0, m, s_eff_ext)) < 5 * e,
              f"MC={v:.5f}+-{e:.5f}, closed={E_int_closed(2.0, m, s_eff_ext):.5f}")

    # ---------------- Eqs. (7.9)-(7.12) -------------------------------------------
    grid = np.logspace(-4, 3, 400) * s_eff
    dE = np.array([delta_E(x, m, s_eff) for x in grid])
    rep.check("Eq. (7.9) Delta E_G(xi) >= 0 for all xi", np.all(dE >= 0), f"min = {dE.min():.2e}")
    rep.check("Eq. (7.9) Delta E_G(xi) is monotonically increasing", np.all(np.diff(dE) > 0))
    rep.close("Eq. (7.10) E_int(0) = -2 G m^2 / (sqrt(pi) sigma_eff)",
              E_int_closed(1e-7 * s_eff, m, s_eff), -2 * m**2 / (np.sqrt(np.pi) * s_eff), rtol=1e-10)
    rep.close("Eq. (7.12) saturation Delta E_G -> 4 G m^2 / (sqrt(pi) sigma_eff)",
              delta_E(1e8 * s_eff, m, s_eff), 4 * m**2 / (np.sqrt(np.pi) * s_eff), rtol=1e-7)

    # ---------------- Eqs. (7.13)-(7.14) -------------------------------------------
    c = 1.0
    tau = quad(lambda t: np.exp(-(c * t / s_eff) ** 2), 0, np.inf)[0]
    rep.close("Eq. (7.13) tau = int exp[-(ct/sigma_eff)^2] dt = sqrt(pi) sigma_eff / (2c)",
              tau, np.sqrt(np.pi) * s_eff / (2 * c), rtol=1e-10)
    dS = []
    for se in (1e-3, 0.37, 5.0, 1e4):
        tau_se = np.sqrt(np.pi) * se / (2 * c)
        dS.append(4 * m**2 / (np.sqrt(np.pi) * se) * tau_se)
    rep.close("Eq. (7.14) Delta S = Delta E_sat * tau = 2 G m^2 / c for any sigma_eff (1e-3 ... 1e4)",
              np.array(dS), 2 * m**2 / c * np.ones(4), rtol=1e-12)
    x = np.array([1e-2, 3e-3, 1e-3])
    ratio = np.array([delta_E(xx * s_eff, m, s_eff) * np.sqrt(np.pi) * s_eff / 2 for xx in x]) / (2 * m**2 / 3 * x**2)
    rep.close("Eq. (7.16) small separations: Delta S = (2 G m^2 / 3c)(xi/sigma_eff)^2", ratio, np.ones(3), rtol=1e-4)

    # ---------------- Eqs. (7.17)-(7.19) in SI -------------------------------------
    mH = np.sqrt(HBAR * C / (2 * G))
    rep.close("Eq. (7.19) m_H = M_P / sqrt(2)", mH, M_P / np.sqrt(2), rtol=1e-14)
    rep.check("Eq. (7.19) m_H = 15.39 micrograms", abs(mH / MICROGRAM - 15.39) < 0.005,
              f"m_H = {mH / MICROGRAM:.4f} ug  (M_P = {M_P / MICROGRAM:.4f} ug)")
    rep.close("Eq. (7.17) Delta S(m_H) = hbar", 2 * G * mH**2 / C, HBAR, rtol=1e-12)

    alpha = 1.7
    tau_a = alpha * s_eff / c
    rep.close("After Eq. (7.13): tau = alpha sigma_eff/c rescales Delta S by 2 alpha / sqrt(pi)",
              4 * m**2 / (np.sqrt(np.pi) * s_eff) * tau_a / (2 * m**2 / c), 2 * alpha / np.sqrt(np.pi), rtol=1e-12)
    mH_a = np.sqrt(HBAR * C / (2 * G) * np.sqrt(np.pi) / (2 * alpha))
    rep.close("After Eq. (7.19): m_H rescales as (sqrt(pi) / 2 alpha)^{1/2}",
              mH_a / mH, (np.sqrt(np.pi) / (2 * alpha)) ** 0.5, rtol=1e-12)

    # ---------------- Eqs. (7.20)-(7.23) ------------------------------------------
    for mm in (0.5 * mH, mH, 2.0 * mH):
        rg = 2 * G * mm / C**2
        dF2 = (2 * rg) ** 2 / L_P**2                      # Eq. (7.21), sigma0 = l_P
        B12 = np.exp(-dF2 / 8)                             # Eq. (7.22)
        rep.close(f"Eq. (7.23) -ln B12 = Delta S/hbar = 2Gm^2/(hbar c) at m = {mm / mH:.1f} m_H",
                  -np.log(B12), 2 * G * mm**2 / (HBAR * C), rtol=1e-12)

    # ---------------- Sec. 4.3: m_H < M_hor ------------------------------------------
    M_hor = 3 * np.sqrt(3) / 4 * L_P * C**2 / G
    rep.check("Sec. 4.3: m_H ~ 0.71 M_P < M_hor ~ 1.30 M_P (no horizon at the cut)",
              mH < M_hor, f"m_H/M_P = {mH / M_P:.4f}, M_hor/M_P = {M_hor / M_P:.4f}")
    return rep


if __name__ == "__main__":
    run().print()
