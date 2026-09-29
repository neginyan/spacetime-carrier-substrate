"""Step 5 -- mean retardation delay of the regularised self-interaction.

Idea
----
Instead of truncating a symmetric Gaussian time profile to t >= 0 by hand,
use the retarded Green function itself. Let the relative-coordinate mass
distribution of the paper, P(r) ~ exp(-r^2 / sigma_eff^2), act for an
instant. The retarded potential at the centre is

    phi(t) = int d^3x P(x) delta(t - |x|/c) / (4 pi |x|)
           = c^2 t P(ct) / c        (t >= 0, and zero for t < 0),

which is causal automatically. Its time integral is the static Newtonian
potential, and its mean delay

    <t> = int t phi(t) dt / int phi(t) dt

is independent of the normalisation of P. Equivalently, <t> is the light
travel time r/c averaged over pair separations with the Newtonian weight
P(r)/r that defines the self-energy. For the Gaussian distribution of the
paper,

    <t> = (sqrt(pi)/2) sigma_eff / c ,

which is exactly the coherence time of Eq. (7.13).

What this does and does not show
--------------------------------
* It removes two ad hoc steps of the "half-line" reading of Eq. (7.13):
  no truncation of a non-causal kernel, and no peak normalisation.
* The identification Delta S = Delta E_G * <t> (the action accumulates over
  the mean retardation delay) is still an assumption.
* The field-overlap model of Step 3 weights the same physics differently
  (mean inverse mode frequency) and gives 1/(2 pi) of this value at small
  separations. Which average is physical is the open question.
* The coefficient depends on the kernel: the Plummer kernel of Section 4 of
  the paper gives <t> = b0 / c.

Units: sigma_eff = c = G = m = 1 unless stated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from model import Report, TAU_PAPER


def P_gauss(r: float, s: float = 1.0) -> float:
    """Relative-coordinate distribution of the paper, normalised to 1."""
    return np.exp(-r**2 / s**2) / (np.pi**1.5 * s**3)


def phi_ret(t: float, P=P_gauss) -> float:
    """Retarded potential at the centre of an instantaneous source (c = 1)."""
    return t * P(t) if t > 0 else 0.0


def phi_ret_direct(t: float, eps: float, P=P_gauss) -> float:
    """Same quantity from the retarded integral with a narrow pulse of width eps."""
    f = lambda r: r * P(r) * np.exp(-(t - r) ** 2 / (2 * eps**2)) / np.sqrt(2 * np.pi * eps**2)
    return quad(f, 0, 12, points=[t - 5 * eps, t, t + 5 * eps], limit=4000, epsabs=1e-14)[0]


def mean_delay(P=P_gauss) -> float:
    num = quad(lambda t: t * phi_ret(t, P), 0, np.inf, limit=400)[0]
    den = quad(lambda t: phi_ret(t, P), 0, np.inf, limit=400)[0]
    return num / den


def run() -> Report:
    rep = Report("Step 5  Mean retardation delay of the regularised self-interaction")

    ts = (0.3, 0.8, 1.5)
    direct = [phi_ret_direct(t, 1e-4) for t in ts]
    rep.check("(a) retarded potential of an instantaneous Gaussian source: phi(t) = t P(t), zero for t < 0",
              np.allclose(direct, [phi_ret(t) for t in ts], rtol=1e-4),
              "direct retarded integral vs t P(t) at t = 0.3, 0.8, 1.5 sigma_eff/c")

    static = quad(lambda r: 4 * np.pi * r**2 * P_gauss(r) / (4 * np.pi * r), 0, np.inf)[0]
    rep.check("(b) int phi dt equals the static Newtonian potential (Newtonian limit recovered)",
              abs(quad(phi_ret, 0, np.inf)[0] / static - 1) < 1e-12,
              f"static potential = {static:.10f}")

    md = mean_delay()
    rep.check("(c) mean retardation delay <t> = (sqrt(pi)/2) sigma_eff/c, the value of Eq. (7.13)",
              abs(md / TAU_PAPER - 1) < 1e-12, f"<t> = {md:.12f}, sqrt(pi)/2 = {TAU_PAPER:.12f}")

    scaled = mean_delay(lambda r: 7.3 * P_gauss(r))
    rep.check("(d) independent of the normalisation of the source (no peak normalisation needed)",
              abs(scaled / md - 1) < 1e-12, f"<t> with P -> 7.3 P: {scaled:.12f}")

    widths = (0.01, 1.0, 50.0)
    ratios = [mean_delay(lambda r, s=s: P_gauss(r, s)) / s for s in widths]
    rep.check("(e) <t> scales exactly as sigma_eff / c",
              np.allclose(ratios, TAU_PAPER, rtol=1e-10),
              "<t>/(sigma_eff/c) for sigma_eff = 0.01, 1, 50: " + ", ".join(f"{r:.10f}" for r in ratios))

    # pair-separation form: <r/c> weighted by P(r)/r over d^3r
    pair = (quad(lambda r: 4 * np.pi * r**2 * P_gauss(r) / r * r, 0, np.inf)[0]
            / quad(lambda r: 4 * np.pi * r**2 * P_gauss(r) / r, 0, np.inf)[0])
    rep.check("(f) same value as the light travel time r/c averaged with the self-energy weight P(r)/r",
              abs(pair / md - 1) < 1e-12, f"<r/c>_(P/r) = {pair:.12f}")

    dE_sat = 4 / np.sqrt(np.pi)                    # Eq. (7.12), G = m = sigma_eff = 1
    rep.check("(g) Delta E_G^sat * <t> = 2 G m^2 / c, i.e. Eq. (7.14) (given the identification)",
              abs(dE_sat * md - 2) < 1e-12, f"Delta E_sat * <t> = {dE_sat * md:.12f}")

    rho_plummer = lambda r: (1 + r**2) ** -2.5
    md_pl = mean_delay(rho_plummer)
    rep.check("(h) kernel dependence: the Plummer kernel of Section 4 gives <t> = b0/c",
              abs(md_pl - 1) < 1e-10, f"<t>_Plummer = {md_pl:.10f} b0/c")

    tau_step3_small = 1 / (4 * np.sqrt(np.pi))     # Step 3, small separations
    rep.check("(i) Step 3 (mean inverse mode frequency) gives exactly 1/(2 pi) of this value",
              abs(tau_step3_small / md * 2 * np.pi - 1) < 1e-12,
              f"ratio = {tau_step3_small / md:.6f}, 1/(2 pi) = {1 / (2 * np.pi):.6f}  -> which average is physical is open")
    return rep


if __name__ == "__main__":
    run().print()
