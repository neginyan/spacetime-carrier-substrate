"""Section 8 (and Eq. 2.20) -- Observational status.

Checks:
* Item 1: the Bild et al. (2023) 16.2-microgram acoustic cat state lies deep
  in the non-saturated regime of Eq. (7.16): Delta S/hbar ~ 1e-27.
* Item 2, Eq. (8.4): echo delay 8 G M/c^3 ln(M/M_P) + O(G M/c^3), about
  0.1 s for 30 solar masses (exact Schwarzschild tortoise coordinate), and
  the proper-distance relation used to place the reflecting boundary.
* Item 3, Eqs. (8.5)-(8.6): photon sphere and critical impact parameter of the
  regular metric versus the perturbative coefficients -10/9 and -2/3.
* Item 4: deviation from the inverse-square law at 1 micrometre.
* Eq. (2.20): dimensional consistency of the CSL heating rate.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from common import Report, G, HBAR, C, M_P, L_P, M_SUN, MICROGRAM


def f_reg(r, rg, b0):
    return 1 - rg * r**2 / (r**2 + b0**2) ** 1.5


def photon_sphere(eps):
    """Units r_g = 1, b0 = eps. Returns (r_ph, b_c)."""
    fr = lambda r: f_reg(r, 1.0, eps)
    dfr = lambda r: -(2 * r * (r**2 + eps**2) ** 1.5 - r**2 * 3 * r * (r**2 + eps**2) ** 0.5) / (r**2 + eps**2) ** 3
    cond = lambda r: 2 * fr(r) - r * dfr(r)
    r_ph = brentq(cond, 1.2, 1.8, xtol=1e-15, rtol=1e-15)
    return r_ph, r_ph / np.sqrt(fr(r_ph))


def run() -> Report:
    rep = Report("Section 8  Observational status (and Eq. 2.20)")

    # ---------------- Item 1: Bild et al. (2023) --------------------------------
    m = 16.2 * MICROGRAM           # effective (rms) mass of the mode
    xi = 2.1e-18                   # cat-state delocalisation [m]
    values = []
    for s_m in (10e-6, 27e-6, 100e-6):              # internal extent of the mode (waist 27 um)
        s_eff = np.sqrt(4 * s_m**2 + L_P**2)         # Eq. (7.15); wavefunction width negligible
        values.append(2 * G * m**2 / (3 * HBAR * C) * (xi / s_eff) ** 2)   # Eq. (7.16)
    rep.check("Item 1: Bild et al. (16.2 ug, xi = 2.1e-18 m): Delta S/hbar ~ 1e-27 (negligible suppression)",
              all(1e-29 < v < 1e-25 for v in values),
              "Delta S/hbar = " + ", ".join(f"{v:.1e}" for v in values) + "  for s_m = 10, 27, 100 um")
    rep.check("Item 1: that experiment is non-saturated, xi / R ~ 1e-13", 1e-14 < xi / 30e-6 < 1e-12,
              f"xi/R = {xi / 30e-6:.1e}")
    rep.check("Item 1: 16.2 ug is within a factor 1.1 of m_H", abs(m / (M_P / np.sqrt(2)) - 1) < 0.1,
              f"m / m_H = {m / (M_P / np.sqrt(2)):.3f}")

    # ---------------- Item 2: echo delay ------------------------------------------
    for Msun in (10.0, 30.0, 100.0):
        M = Msun * M_SUN
        rg = 2 * G * M / C**2
        x_wall = L_P**2 / (4 * rg**2)                 # (r_wall - r_g)/r_g
        # exact Schwarzschild tortoise coordinate r* = r + r_g ln(r/r_g - 1)
        rstar_ph = 1.5 * rg + rg * np.log(0.5)
        rstar_w = rg * (1 + x_wall) + rg * np.log(x_wall)
        dt = 2 / C * (rstar_ph - rstar_w)
        lead = 8 * G * M / C**3 * np.log(M / M_P)
        resid = (dt - lead) / (G * M / C**3)
        ok = abs(resid) < 20
        if Msun == 30.0:
            rep.check("Eq. (8.4) 30 M_sun: Delta t_echo ~ 0.1 s", abs(dt - 0.1) < 0.02, f"Delta t = {dt:.4f} s")
        rep.check(f"Eq. (8.4) {Msun:.0f} M_sun: Delta t - 8GM/c^3 ln(M/M_P) = O(GM/c^3)", ok,
                  f"residual = {resid:.2f} GM/c^3 (leading term = {lead / (G * M / C**3):.1f} GM/c^3)")
    # proper distance near the horizon, units r_g = 1
    for delta in (1e-4, 1e-6):
        ell = quad(lambda r: 1 / np.sqrt(1 - 1 / r), 1, 1 + delta, epsrel=1e-12)[0]
        rep.close(f"Item 2: proper distance l = 2 sqrt(r_g (r - r_g)) near the horizon (r - r_g = {delta:g} r_g)",
                  ell, 2 * np.sqrt(delta), rtol=5 * delta)

    # ---------------- Item 3: photon sphere and shadow -------------------------------
    eps = np.array([0.02, 0.01, 0.005])
    rb = np.array([photon_sphere(e) for e in eps])
    c_r = (rb[:, 0] / 1.5 - 1) / eps**2
    c_b = (rb[:, 1] / (1.5 * np.sqrt(3)) - 1) / eps**2
    rep.close("Eq. (8.5) r_ph = (3GM/c^2)[1 - (10/9) b0^2/r_g^2 + O(b0^4/r_g^4)]",
              c_r, -10 / 9 * np.ones(3), rtol=2e-3)
    rep.close("Eq. (8.6) b_c = (3 sqrt3 GM/c^2)[1 - (2/3) b0^2/r_g^2 + O(b0^4/r_g^4)]",
              c_b, -2 / 3 * np.ones(3), rtol=2e-3)
    for name, Mbh in (("Sgr A*", 4.3e6 * M_SUN), ("M87*", 6.5e9 * M_SUN)):
        corr = (L_P / (2 * G * Mbh / C**2)) ** 2
        rep.check(f"Item 3: fractional shadow correction for {name} is unobservable", corr < 1e-85,
                  f"(l_P/r_g)^2 = {corr:.1e}")

    # ---------------- Item 4: sub-millimetre gravity ----------------------------------
    r = 1e-6
    dev = -np.expm1(-1.5 * np.log1p(L_P**2 / r**2))  # 1 - (1 + b0^2/r^2)^{-3/2}, from Eq. (4.10)
    rep.check("Item 4: relative deviation of g(r) at r = 1 um is (3/2)(l_P/r)^2 ~ 4e-58 (undetectable)",
              np.isclose(dev, 1.5 * (L_P / r) ** 2, rtol=1e-12) and dev < 1e-56,
              f"deviation = {dev:.2e}")

    # ---------------- Eq. (2.20): dimensions -----------------------------------------
    # exponents of (M, L, T)
    dims = {"lambda": (0, 0, -1), "hbar": (1, 2, -1), "m": (1, 0, 0), "m0": (1, 0, 0), "rc": (0, 1, 0)}
    tot = np.array(dims["lambda"]) + 2 * np.array(dims["hbar"]) + np.array(dims["m"]) \
        - 2 * np.array(dims["m0"]) - 2 * np.array(dims["rc"])
    rep.check("Eq. (2.20) (3/4) lambda hbar^2 m / (m0^2 r_c^2) has units of power (kg m^2 s^-3)",
              tuple(tot) == (1, 2, -3), f"(M, L, T) exponents = {tuple(int(t) for t in tot)}")
    return rep


if __name__ == "__main__":
    run().print()
