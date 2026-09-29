"""Shared definitions for the coherence-time programme.

Context
-------
In T. Namba, "The Spacetime Carrier Substrate: Bandlimited Information,
Regular Geometry, and a Gravitational Heisenberg Cut" (CQG-117584), the
action difference between two branches of a spatial superposition is
written as

    Delta S = Delta E_G * tau ,   tau = sqrt(pi) sigma_eff / (2c)    (Eq. 7.13)

where tau is stated to be the central physical assumption. The scripts in
this repository test whether tau (its scale, its coefficient and its
independence of the separation) can come out of models in which it is not
put in by hand.

Scalar-field toy model
----------------------
The Newtonian field is replaced by a massless quantised scalar field coupled
to the carrier-regularised mass density (g^2 = 4 pi G). A static source
displaces each mode into a coherent state with amplitude
alpha_k = -g rho_k / (sqrt(2) omega_k^{3/2}), omega_k = c k. With the Gaussian
regularisation of the paper, |rho_k|^2 -> m^2 exp(-k^2 sigma_eff^2 / 4), and
after the angular integral (lengths in units of sigma_eff, x = xi/sigma_eff):

    Delta E_G(x) = (4 G m^2 / (pi sigma_eff)) * I0(x)
    I0(x) = int_0^inf [1 - j0(k x)] exp(-k^2/4) dk
          = (pi/2) [2/sqrt(pi) - erf(x)/x]            (reproduces Eq. 7.9)

    Gamma_static(x) = (G m^2 / (pi hbar c)) * I1(x)
    I1(x) = int_0^inf [1 - j0(k x)] exp(-k^2/4) dk / k

If the source difference exists only for a time T (sudden switch-on and
switch-off), each mode acquires the factor |1 - exp(i omega T)|^2 = 4 sin^2(omega T/2):

    Gamma_T(x, y) = (G m^2 / (pi hbar c)) * J(x, y),   y = cT / sigma_eff
    J(x, y) = int_0^inf [1 - j0(k x)] exp(-k^2/4) 4 sin^2(k y / 2) dk / k

In every case the emergent coherence time is defined by
Gamma = Delta E_G tau_eff / hbar, i.e.

    tau_eff / (sigma_eff / c) = I1(x) / (4 I0(x))      (static)
                              = J(x, y) / (4 I0(x))    (finite time)

Units: G = hbar = c = 1 unless stated; G m^2/(hbar c) = (m / M_P)^2.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import erf, spherical_jn

warnings.simplefilter("ignore", IntegrationWarning)   # oscillatory tails; accuracy is checked

EULER_GAMMA = 0.5772156649015329
TAU_PAPER = np.sqrt(np.pi) / 2          # Eq. (7.13): tau in units of sigma_eff / c
K_MAX = 14.0                            # exp(-k^2/4) < 1e-21 beyond this


def _one_minus_j0(z):
    z = np.asarray(z, dtype=float)
    small = z < 1e-4
    out = np.empty_like(z)
    out[small] = z[small] ** 2 / 6 - z[small] ** 4 / 120
    out[~small] = 1 - spherical_jn(0, z[~small])
    return out


def _breakpoints(*scales: float) -> list[float]:
    """Zeros of the oscillatory factors inside (0, K_MAX), for quad."""
    pts = set()
    for s in scales:
        if s > 0:
            pts |= {j * np.pi / s for j in range(1, 800) if j * np.pi / s < K_MAX}
    return sorted(pts)


# ---------------------------------------------------------------------------
# Static model (Step 3)
# ---------------------------------------------------------------------------
def I0(x: float) -> float:
    """Closed form of the energy integral."""
    if x < 1e-4:
        return np.sqrt(np.pi) * x**2 / 3
    return np.pi / 2 * (2 / np.sqrt(np.pi) - erf(x) / x)


def I0_numeric(x: float) -> float:
    f = lambda k: _one_minus_j0(k * x) * np.exp(-k**2 / 4)
    return quad(f, 0, K_MAX, points=_breakpoints(x) or None, limit=20000,
                epsabs=1e-13, epsrel=1e-11)[0]


def I1(x: float) -> float:
    """Static decoherence integral."""
    if x < 1e-3:
        return x**2 / 3 - x**4 / 20
    f = lambda k: _one_minus_j0(k * x) * np.exp(-k**2 / 4) / k if k > 0 else 0.0
    return quad(f, 0, K_MAX, points=_breakpoints(x), limit=20000,
                epsabs=1e-13, epsrel=1e-11)[0]


def I1_asymptotic(x: float) -> float:
    """x >> 1:  I1 = ln x + gamma_E/2 + ln 2 - 1."""
    return np.log(x) + EULER_GAMMA / 2 + np.log(2) - 1


def tau_static(x: float) -> float:
    return I1(x) / (4 * I0(x))


# ---------------------------------------------------------------------------
# Finite-time model (Step 4)
# ---------------------------------------------------------------------------
def J(x: float, y: float) -> float:
    """Finite-time decoherence integral, x = xi/sigma_eff, y = cT/sigma_eff."""
    f = lambda k: (_one_minus_j0(k * x) * np.exp(-k**2 / 4) * 4 * np.sin(k * y / 2) ** 2 / k
                   if k > 0 else 0.0)
    return quad(f, 0, K_MAX, points=_breakpoints(x, y), limit=40000,
                epsabs=1e-12, epsrel=1e-10)[0]


def J_far(y: float) -> float:
    """J for xi -> infinity (xi >> cT):  2 int (1 - cos ky) exp(-k^2/4) dk / k."""
    f = lambda k: 2 * (1 - np.cos(k * y)) * np.exp(-k**2 / 4) / k if k > 0 else 0.0
    return quad(f, 0, K_MAX, points=_breakpoints(y), limit=40000,
                epsabs=1e-12, epsrel=1e-10)[0]


def J_far_asymptotic(y: float) -> float:
    """y >> 1:  J_far = 2 ln y + gamma_E + 2 ln 2."""
    return 2 * np.log(y) + EULER_GAMMA + 2 * np.log(2)


def tau_finite(x: float, y: float) -> float:
    return J(x, y) / (4 * I0(x))


# ---------------------------------------------------------------------------
# Implied cut mass: Gamma = 1
# ---------------------------------------------------------------------------
def cut_mass_ratio(integral: float) -> float:
    """m_cut / m_H with Gamma = (m/M_P)^2 * integral / pi and m_H = M_P/sqrt(2)."""
    return np.sqrt(2 * np.pi / integral)


# ---------------------------------------------------------------------------
# Minimal check recorder
# ---------------------------------------------------------------------------
class Report:
    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        self.rows.append((label, bool(ok), detail))

    @property
    def n_pass(self) -> int:
        return sum(ok for _, ok, _ in self.rows)

    def print(self) -> None:
        print()
        print("=" * 78)
        print(self.title)
        print("=" * 78)
        for label, ok, detail in self.rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
            if detail:
                print(f"         {detail}")
        print("-" * 78)
        print(f"{self.n_pass}/{len(self.rows)} checks passed")
