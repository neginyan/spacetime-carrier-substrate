"""Step 6 -- the mean retardation delay as a lag of the accumulated action.

Idea
----
Let the branch difference be switched on suddenly at t = 0 and held. With
the normalised retarded response of Step 5,

    q(t) = phi_ret(t) / int phi_ret dt ,        F(t) = int_0^t q(u) du ,

the interaction energy that has arrived by time t is Delta E * F(t), and the
accumulated retarded action is

    S_R(T) = Delta E * int_0^T F(t) dt .

An instantaneous (non-retarded) reference would give S_0(T) = Delta E * T.
For the Gaussian distribution of the paper F(t) = 1 - exp(-(ct/sigma_eff)^2),
and

    lim_{T -> inf} [S_0(T) - S_R(T)] = Delta E * <t> = Delta E * (sqrt(pi)/2) sigma_eff / c ,

so that with Delta E_G^sat this is 2 G m^2 / c.

What this does and does not show
--------------------------------
* The result is the survival-function identity
      int_0^inf [1 - F(t)] dt = int_0^inf t q(t) dt = <t> ,
  valid for every normalised causal response. It holds for the Gaussian, the
  Plummer and an exponential kernel alike (checks b, e, f); it is a restatement
  of the definition of <t>, not an independent result, and it fixes no
  coefficient. The value sqrt(pi)/2 enters only through the Gaussian kernel of
  Step 5.
* What it does give is a clean interpretation: the retarded action follows
  the instantaneous one along the same line, delayed by exactly <t>,
      S_R(T) = Delta E (T - <t>) + o(1)      (check c).
  <t> is the time lag of the gravitational interaction, and Delta E <t> is
  the action that retardation has not yet accumulated.
* The action actually accumulated between the branches, S_R(T), still grows
  linearly with T (check d), as in Diosi-Penrose and in Step 4. S_0 is a
  non-relativistic reference, not a physical branch. Identifying the physical
  Delta S of the Heisenberg cut with the deficit S_0 - S_R, rather than with
  S_R(T), requires an additional physical principle. That identification
  remains open.

Units: sigma_eff = c = Delta E = 1 unless stated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from model import Report, TAU_PAPER
from step5_mean_retardation import P_gauss, phi_ret, mean_delay


def make_response(P):
    """Normalised causal response q(t) and its CDF F(t) from a radial kernel P(r)."""
    norm = quad(lambda t: phi_ret(t, P), 0, np.inf, limit=400)[0]
    q = lambda t: phi_ret(t, P) / norm
    F = lambda t: quad(q, 0, t, limit=400)[0] if t > 0 else 0.0
    return q, F


def survival_integral(F, T=np.inf) -> float:
    """int_0^T [1 - F(t)] dt."""
    return quad(lambda t: 1.0 - F(t), 0, T, limit=400)[0]


def S_R(F, T: float, dE: float = 1.0) -> float:
    """Accumulated retarded action after sudden switch-on at t = 0."""
    return dE * quad(F, 0, T, limit=400)[0]


def run() -> Report:
    rep = Report("Step 6  Mean retardation delay as a lag of the accumulated action")

    q, F = make_response(P_gauss)
    ts = (0.3, 1.0, 2.0)
    F_exact = [1 - np.exp(-t**2) for t in ts]
    rep.check("(a) Gaussian: arrived fraction F(t) = 1 - exp(-(ct/sigma_eff)^2)",
              np.allclose([F(t) for t in ts], F_exact, rtol=1e-10),
              "numerical CDF of q(t) vs closed form at t = 0.3, 1, 2 sigma_eff/c")

    deficit = survival_integral(F)
    rep.check("(b) lim [S_0 - S_R] = Delta E <t> = (sqrt(pi)/2) Delta E sigma_eff/c",
              abs(deficit / TAU_PAPER - 1) < 1e-10,
              f"int (1-F) dt = {deficit:.12f}, sqrt(pi)/2 = {TAU_PAPER:.12f}")

    Ts = (3.0, 5.0, 8.0)
    offsets = [S_R(F, T) - (T - TAU_PAPER) for T in Ts]
    rep.check("(c) S_R(T) = Delta E (T - <t>) + o(1): the retarded action lags by exactly <t>",
              max(abs(o) for o in offsets[1:]) < 1e-10,
              "S_R - (T - <t>) at T = 3, 5, 8: " + ", ".join(f"{o:.1e}" for o in offsets))

    slope = (S_R(F, 10.0) - S_R(F, 6.0)) / 4.0
    rep.check("(d) the accumulated branch action S_R(T) still grows linearly in T (slope Delta E)",
              abs(slope - 1) < 1e-10,
              f"dS_R/dT for T in [6, 10] = {slope:.12f}  -> S_R itself is protocol dependent")

    rho_plummer = lambda r: (1 + r**2) ** -2.5
    _, F_pl = make_response(rho_plummer)
    d_pl, md_pl = survival_integral(F_pl), mean_delay(rho_plummer)
    rep.check("(e) Plummer kernel: deficit = <t> = b0/c (same identity, different coefficient)",
              abs(d_pl - md_pl) < 1e-8 and abs(md_pl - 1) < 1e-8,
              f"int (1-F) dt = {d_pl:.10f}, <t> = {md_pl:.10f} b0/c")

    q_exp = lambda t: np.exp(-t)                 # a response not derived from Step 5
    F_exp = lambda t: 1 - np.exp(-t)
    d_exp = survival_integral(F_exp)
    t_exp = quad(lambda t: t * q_exp(t), 0, np.inf)[0]
    rep.check("(f) exponential response: deficit = <t> = 1 -> a kernel-independent identity",
              abs(d_exp - t_exp) < 1e-12 and abs(t_exp - 1) < 1e-12,
              f"int (1-F) dt = {d_exp:.12f}, int t q dt = {t_exp:.12f}  -> fixes no coefficient")

    dE_sat = 4 / np.sqrt(np.pi)                   # Eq. (7.12), G = m = sigma_eff = 1
    rep.check("(g) with Delta E_G^sat the deficit equals 2 G m^2 / c, i.e. Eq. (7.14)",
              abs(dE_sat * deficit - 2) < 1e-10,
              f"Delta E_sat * int (1-F) dt = {dE_sat * deficit:.12f}  (given Delta S = deficit: open)")
    return rep


if __name__ == "__main__":
    run().print()
