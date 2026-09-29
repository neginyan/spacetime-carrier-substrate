"""Step 4 (alternative) -- retarded interaction with a Gaussian time profile.

Model
-----
The branch difference is switched on with the Gaussian time profile
s(t) = exp(-t^2 / 2 delta_t^2), and the gravitational interaction propagates
with the retarded Green function delta(t - t' - r/c) / (4 pi r). The action
difference in the saturated regime is then

    Delta S = Delta E_G^sat * sqrt(pi) delta_t * R(u) ,   u = c delta_t / sigma_eff,

where the time correlation gives sqrt(pi) delta_t exp(-r^2 / 4 c^2 delta_t^2)
and the retardation factor is the spatial average

    R(u) = int d^3r P(r) exp(-r^2/(4 c^2 delta_t^2)) / r  /  int d^3r P(r) / r
         = 1 / (1 + 1/(4 u^2)),       P(r) ~ exp(-r^2 / sigma_eff^2).

Writing Delta S = C(u) G m^2 / c gives C(u) = 4 u R(u).

Checks
------
(a) the time correlation of s(t) is sqrt(pi) delta_t exp(-Delta^2 / 4 delta_t^2);
(b) the retardation factor R(u) = 1/(1 + 1/(4u^2)) (quadrature);
(c) with the width delta_t = sigma_eff/(2c) implied by Eq. (7.13), C = 1, not 2;
(d) C = 2 requires u = 0.7328 (solved backwards, not derived);
(e) C(u) -> 4u for u >> 1: the action grows with the duration, so the
    protocol alone does not fix a universal coefficient.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from model import Report


def R_numeric(u: float) -> float:
    num = quad(lambda r: r * np.exp(-r**2) * np.exp(-r**2 / (4 * u**2)), 0, np.inf)[0]
    den = quad(lambda r: r * np.exp(-r**2), 0, np.inf)[0]
    return num / den


def R_closed(u: float) -> float:
    return 1 / (1 + 1 / (4 * u**2))


def C(u: float) -> float:
    return 4 * u * R_closed(u)


def run() -> Report:
    rep = Report("Step 4 (alternative)  Retarded interaction, Gaussian time profile")

    d = 0.7
    lags = (0.0, 0.5, 1.3)
    corr = [quad(lambda t: np.exp(-t**2 / (2 * d**2)) * np.exp(-(t - L) ** 2 / (2 * d**2)), -np.inf, np.inf)[0]
            for L in lags]
    target = [np.sqrt(np.pi) * d * np.exp(-L**2 / (4 * d**2)) for L in lags]
    rep.check("(a) time correlation of s(t) = sqrt(pi) delta_t exp(-Delta^2/(4 delta_t^2))",
              np.allclose(corr, target, rtol=1e-10))

    us = (0.3, 0.5, 1.0, 3.0)
    rep.check("(b) retardation factor R(u) = 1/(1 + 1/(4u^2))",
              np.allclose([R_numeric(u) for u in us], [R_closed(u) for u in us], rtol=1e-10),
              ", ".join(f"u={u}: {R_closed(u):.4f}" for u in us))

    rep.check("(c) width implied by Eq. (7.13), delta_t = sigma_eff/(2c): C = 1, not 2",
              abs(C(0.5) - 1) < 1e-12, f"C(1/2) = {C(0.5):.6f}")

    u2 = brentq(lambda u: C(u) - 2, 0.1, 5, xtol=1e-14)
    rep.check("(d) C = 2 requires u = c delta_t / sigma_eff = 0.7328 (solved backwards)",
              abs(u2 - 0.732786) < 1e-5, f"u = {u2:.6f}")

    rep.check("(e) C(u) -> 4u for long durations: no universal coefficient from the protocol alone",
              abs(C(100.0) / 400 - 1) < 1e-4, f"C(100) = {C(100.0):.3f}")
    return rep


if __name__ == "__main__":
    run().print()
