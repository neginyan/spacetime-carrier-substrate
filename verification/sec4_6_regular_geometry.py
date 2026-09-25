"""Sections 4 and 6 -- Regular geometry, horizons, curvature, energy conditions.

The curvature is NOT taken from the closed-form expressions of the paper.
It is computed independently from the metric (4.22) by building the
Christoffel symbols and the Riemann tensor numerically (high-order finite
differences), and then compared with the paper's closed forms.

Geometric units G = c = 1; lengths in units of b0 (b0 = 1) unless stated.

Checks:
* Eqs. (4.2)-(4.7): normalisation and enclosed mass M(r).
* Eqs. (4.9)-(4.12): potential, field and maximum field strength.
* Eq. (4.21): U_self = -(3 pi / 32) G M^2 / b0.
* Eqs. (4.25)-(4.30): extremum at sqrt(2) b0, horizon threshold M_hor,
  horizon counts, and the asymptotics of r+ and r-.
* Eq. (4.38): R(r) = 6 G M b0^2 (4 b0^2 - r^2) / [c^2 (r^2 + b0^2)^{7/2}].
* Eq. (4.44): K(0) = 96 G^2 M^2 / (c^4 b0^6); Eq. (6.21) Gauss-Bonnet at r=0.
* Eqs. (6.2)-(6.5), (6.11)-(6.13): effective source from the Einstein
  tensor, NEC everywhere, SEC violated for r < sqrt(2/3) b0, de Sitter core.
* Eq. (6.16): Raychaudhuri source for static observers.
* Eqs. (4.45)/(6.22): core curvature and density in Planck units.
* Section 4.5: Gaussian-kernel horizon threshold ~ 0.95 sigma0 c^2/G.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammainc

from common import Report, M_P, L_P, M_SUN

B0 = 1.0


# ---------------------------------------------------------------------------
# Metric function
# ---------------------------------------------------------------------------
def f(r, M, b0=B0):
    return 1 - 2 * M * r**2 / (r**2 + b0**2) ** 1.5


def metric(x, M):
    _, r, th, _ = x
    fr = f(r, M)
    return np.diag([-fr, 1 / fr, r**2, (r * np.sin(th)) ** 2])


# ---------------------------------------------------------------------------
# Generic numerical curvature (independent of the paper's formulas)
# ---------------------------------------------------------------------------
_STENCIL = (np.array([1, -8, 8, -1]) / 12.0, np.array([-2, -1, 1, 2]))


def _d(fun, x, mu, h):
    w, s = _STENCIL
    out = 0.0
    for wi, si in zip(w, s):
        xp = np.array(x, dtype=float)
        xp[mu] += si * h
        out = out + wi * fun(xp)
    return out / h


def christoffel(x, M, h=1e-3):
    g = metric(x, M)
    gi = np.linalg.inv(g)
    dg = np.array([_d(lambda y: metric(y, M), x, mu, h) for mu in range(4)])  # dg[c,a,b]
    G = np.einsum("ad,bdc->abc", gi, dg) + np.einsum("ad,cdb->abc", gi, dg) \
        - np.einsum("ad,dbc->abc", gi, dg)
    return 0.5 * G      # Gamma^a_{bc}


def riemann(x, M, h=1e-3):
    Gm = christoffel(x, M, h)
    dG = np.array([_d(lambda y: christoffel(y, M, h), x, mu, h) for mu in range(4)])  # dG[m,a,b,c]
    # R^a_{bcd} = d_c Gamma^a_{db} - d_d Gamma^a_{cb} + Gamma^a_{ce} Gamma^e_{db} - Gamma^a_{de} Gamma^e_{cb}
    R = (np.einsum("cadb->abcd", dG) - np.einsum("dacb->abcd", dG)
         + np.einsum("ace,edb->abcd", Gm, Gm) - np.einsum("ade,ecb->abcd", Gm, Gm))
    return R


def curvature(r, M, th=1.1):
    x = np.array([0.0, r, th, 0.3])
    g = metric(x, M)
    gi = np.linalg.inv(g)
    Rud = riemann(x, M)
    Ric = np.einsum("abad->bd", Rud)
    Rs = np.einsum("bd,bd->", gi, Ric)
    R_low = np.einsum("ae,ebcd->abcd", g, Rud)
    R_up = np.einsum("ae,bf,cg,dh,efgh->abcd", gi, gi, gi, gi, R_low)
    Kr = np.einsum("abcd,abcd->", R_low, R_up)
    Ein_mixed = np.einsum("ab,bc->ac", gi, Ric) - 0.5 * Rs * np.eye(4)   # G^a_c
    Ric_mixed = np.einsum("ab,bc->ac", gi, Ric)
    RicSq = np.einsum("ab,ba->", Ric_mixed, Ric_mixed)
    return dict(R=Rs, K=Kr, G=Ein_mixed, Ric=Ric, RicSq=RicSq, g=g)


# ---------------------------------------------------------------------------
# Closed forms from the paper (G = c = 1)
# ---------------------------------------------------------------------------
def R_paper(r, M, b0=B0):                      # Eq. (4.38)
    return 6 * M * b0**2 * (4 * b0**2 - r**2) / (r**2 + b0**2) ** 3.5


def rho_paper(r, M, b0=B0):                    # Eq. (6.5)
    return 3 * M * b0**2 / (4 * np.pi * (r**2 + b0**2) ** 2.5)


def pt_paper(r, M, b0=B0):                     # Eq. (6.11)
    return -M / (8 * np.pi) * (6 * b0**4 - 9 * b0**2 * r**2) / (r**2 + b0**2) ** 3.5


def sec_paper(r, M, b0=B0):                    # Eq. (6.12)
    return 3 * M * b0**2 * (3 * r**2 - 2 * b0**2) / (4 * np.pi * (r**2 + b0**2) ** 3.5)


def frep_paper(r, M, b0=B0):                   # Eq. (6.16)
    return 3 * M * b0**2 * (2 * b0**2 - 3 * r**2) / (r**2 + b0**2) ** 3.5


def horizons(M, b0=B0):
    rs = np.linspace(1e-4, 10 * max(M, 1.0) + 10, 400001)
    fs = f(rs, M, b0)
    idx = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0]
    return [brentq(f, rs[i], rs[i + 1], args=(M, b0), xtol=1e-14) for i in idx]


def run() -> Report:
    rep = Report("Sections 4 & 6  Regular geometry, curvature and energy conditions")
    M = 1.0

    # ---------------- Mass distribution and Newtonian field ------------------
    rho = lambda r: 3 * M / (4 * np.pi * B0**3) * (1 + r**2 / B0**2) ** -2.5      # Eq. (4.2)
    tot = quad(lambda r: 4 * np.pi * r**2 * rho(r), 0, np.inf)[0]
    rep.close("Eq. (4.7) total mass int rho d^3x = M", tot, M, rtol=1e-10)
    rr = np.array([0.2, 1.0, 3.0, 20.0])
    Mr = [quad(lambda s: 4 * np.pi * s**2 * rho(s), 0, r)[0] for r in rr]
    rep.close("Eq. (4.6) M(r) = M r^3 / (r^2 + b0^2)^{3/2}", Mr, M * rr**3 / (rr**2 + B0**2) ** 1.5, rtol=1e-10)
    # shell theorem: Phi(r) = -G [M(r)/r + int_r^inf 4 pi s rho(s) ds]
    Phi = [-(quad(lambda s: 4 * np.pi * s**2 * rho(s), 0, r)[0] / r
             + quad(lambda s: 4 * np.pi * s * rho(s), r, np.inf)[0]) for r in rr]
    rep.close("Eq. (4.9) Phi(r) = -G M / sqrt(r^2 + b0^2)", Phi, -M / np.sqrt(rr**2 + B0**2), rtol=1e-8)
    gabs = lambda r: M * r / (r**2 + B0**2) ** 1.5
    opt = minimize_scalar(lambda r: -gabs(r), bounds=(0, 5), method="bounded", options={"xatol": 1e-12})
    rep.close("Eq. (4.11) maximum field at r = b0/sqrt(2)", opt.x, B0 / np.sqrt(2), rtol=1e-6)
    rep.close("Eq. (4.12) |g|_max = 2 G M / (3 sqrt(3) b0^2)", gabs(opt.x), 2 * M / (3 * np.sqrt(3) * B0**2), rtol=1e-10)
    U = -quad(lambda r: gabs(r) ** 2 * r**2, 0, np.inf)[0] / 2      # -(1/8piG) int |g|^2 d^3x
    rep.close("Eq. (4.21) U_self = -(3 pi/32) G M^2 / b0", U, -3 * np.pi / 32 * M**2 / B0, rtol=1e-10)

    # ---------------- Horizons ------------------------------------------------
    hfun = lambda r: r**2 / (r**2 + B0**2) ** 1.5
    opt = minimize_scalar(lambda r: -hfun(r), bounds=(0, 5), method="bounded", options={"xatol": 1e-12})
    rep.close("Eq. (4.27) extremum of h(r) at r* = sqrt(2) b0", opt.x, np.sqrt(2) * B0, rtol=1e-6)
    M_hor = 3 * np.sqrt(3) / 4 * B0
    rep.close("Eq. (4.29) M_hor = (3 sqrt3 / 4) b0 c^2/G  (= 1/(2 h_max))", 1 / (2 * hfun(opt.x)), M_hor, rtol=1e-10)
    n_sub = len(horizons(M_hor * (1 - 1e-4)))
    n_sup = len(horizons(M_hor * (1 + 1e-4)))
    rep.check("Horizon count: 0 roots for M < M_hor, 2 roots for M > M_hor",
              (n_sub, n_sup) == (0, 2), f"(below, above) = ({n_sub}, {n_sup})")
    rp, rm = horizons(M_hor * (1 + 1e-8))[::-1]
    rep.close("Extremal limit: r+ = r- -> sqrt(2) b0", np.array([rp, rm]), np.sqrt(2) * B0 * np.ones(2), rtol=2e-4)
    errs_p, errs_m = [], []
    for Mbig in (1e2, 1e3, 1e4):
        rm_, rp_ = horizons(Mbig)
        rg = 2 * Mbig
        errs_p.append(abs(rp_ / (rg * (1 - 1.5 * B0**2 / rg**2)) - 1) * rg**4)
        errs_m.append(abs(rm_ / (B0 * np.sqrt(B0 / rg)) - 1))
    rep.check("Eq. (4.30) r+ = r_g [1 - (3/2) b0^2/r_g^2 + O(b0^4/r_g^4)]",
              max(errs_p) < 20, f"|r+/approx - 1| * r_g^4 = {[f'{e:.2f}' for e in errs_p]} (bounded)")
    rep.check("r- -> b0 sqrt(b0/r_g) for M >> M_hor (relative error decreasing)",
              errs_m[-1] < errs_m[0] and errs_m[-1] < 1e-3, f"rel. errors = {[f'{e:.1e}' for e in errs_m]}")

    # ---------------- Curvature from the metric --------------------------------
    Mc = 0.5                           # subcritical: f > 0 everywhere, static observers exist
    radii = np.array([0.15, 0.5, 0.9, 1.6, 3.0, 6.0])
    cv = [curvature(r, Mc) for r in radii]
    R_num = np.array([c["R"] for c in cv])
    rep.close("Eq. (4.38) Ricci scalar from the metric = 6GMb0^2(4b0^2 - r^2)/(r^2+b0^2)^{7/2}",
              R_num, R_paper(radii, Mc), rtol=1e-6, atol=1e-9)
    # Core values: curvature invariants are even in r, so Richardson
    # extrapolation from r = 0.1 b0 and 0.05 b0 removes the O(r^2) term.
    c1, c2 = curvature(0.1, Mc), curvature(0.05, Mc)
    core = lambda key: (4 * c2[key] - c1[key]) / 3
    rep.close("Eq. (4.39) R(0) = 24 G M / (c^2 b0^3)", core("R"), 24 * Mc / B0**3, rtol=2e-3)
    rep.close("Eq. (4.44) K(0) = 96 G^2 M^2 / (c^4 b0^6)", core("K"), 96 * Mc**2 / B0**6, rtol=2e-3)
    GB0 = core("R") ** 2 - 4 * core("RicSq") + core("K")
    rep.close("Eq. (6.21) Gauss-Bonnet density at the core = 96 G^2 M^2/(c^4 b0^6)", GB0, 96 * Mc**2, rtol=5e-3)
    rep.close("Eq. (6.19) R_mn R^mn (0) = R(0)^2 / 4", core("RicSq"), core("R") ** 2 / 4, rtol=2e-3)
    ratios = [curvature(r, Mc)["K"] / (48 * Mc**2 / r**6) for r in (20.0, 40.0)]
    rep.check("K(r) -> Schwarzschild 48 G^2M^2/(c^4 r^6) far outside the core, deviation ~ b0^2/r^2",
              abs(ratios[1] - 1) < 0.01 and abs((1 - ratios[0]) / (1 - ratios[1]) - 4) < 0.2,
              f"K/K_Schw at r = 20, 40 b0: {ratios[0]:.5f}, {ratios[1]:.5f}")

    # ---------------- Effective source (Einstein tensor) ---------------------
    rho_num = np.array([-c["G"][0, 0] for c in cv]) / (8 * np.pi)
    pr_num = np.array([c["G"][1, 1] for c in cv]) / (8 * np.pi)
    pt_num = np.array([c["G"][2, 2] for c in cv]) / (8 * np.pi)
    rep.close("Eq. (6.5) rho_eff c^2 = 3 M c^2 b0^2 / [4 pi (r^2+b0^2)^{5/2}] (from G^t_t)",
              rho_num, rho_paper(radii, Mc), rtol=1e-6, atol=1e-10)
    rep.close("Eq. (6.3) p_r = -rho_eff c^2 (from G^r_r)", pr_num, -rho_paper(radii, Mc), rtol=1e-6, atol=1e-10)
    rep.close("Eq. (6.11) p_t from G^theta_theta", pt_num, pt_paper(radii, Mc), rtol=1e-6, atol=1e-10)
    rep.close("Trace check: R = 16 pi G/c^4 (rho c^2 - p_t)", R_num,
              16 * np.pi * (rho_paper(radii, Mc) - pt_paper(radii, Mc)), rtol=1e-6, atol=1e-9)

    rs = np.linspace(0, 20, 20001)
    nec = rho_paper(rs, 1.0) + pt_paper(rs, 1.0)
    rep.check("Eq. (6.10) NEC: rho c^2 + p_t >= 0 for all r (and rho c^2 + p_r = 0)",
              np.all(nec >= -1e-15), f"min(rho + p_t) = {nec.min():.2e}")
    root = brentq(lambda r: sec_paper(r, 1.0), 0.1, 2.0)
    rep.close("Eq. (6.13) SEC violated exactly for r < r0 = sqrt(2/3) b0", root, np.sqrt(2 / 3) * B0, rtol=1e-12)
    rep.close("Eq. (6.12) rho c^2 + p_r + 2 p_t = 2 p_t", sec_paper(radii, Mc), 2 * pt_paper(radii, Mc), rtol=1e-12)
    rep.close("de Sitter core: p_t(0) = -rho_eff(0) c^2", pt_paper(0.0, 1.0), -rho_paper(0.0, 1.0), rtol=1e-12)

    # Raychaudhuri source for static observers, u = (1/sqrt f, 0, 0, 0)
    Frep_num = np.array([-c["Ric"][0, 0] / f(r, Mc) for c, r in zip(cv, radii)])
    rep.close("Eq. (6.16) F_rep = -R_mn u^m u^n (static observers, from the metric)",
              Frep_num, frep_paper(radii, Mc), rtol=1e-6, atol=1e-9)

    # ---------------- Planck-unit scaling (Eqs. 4.45, 6.22) -------------------
    m_ratio = M_SUN / M_P
    eps = 24 * m_ratio               # l_P^2 R(0) for b0 = l_P
    rep.check("Eq. (4.45) solar-mass core: l_P^2 R(0) = 24 M/M_P ~ 10^39 (outside domain of validity)",
              38.5 < np.log10(eps) < 39.9, f"l_P^2 R(0) = {eps:.2e}")

    # ---------------- Section 4.5: Gaussian kernel -----------------------------
    # f = 1 - 2 M P(3/2, r^2/sigma0^2)/r  ->  M_min = min_r r / (2 P)
    s0 = 1.0
    res = minimize_scalar(lambda r: r / (2 * gammainc(1.5, r**2 / s0**2)), bounds=(0.1, 10), method="bounded")
    rep.check("Sec. 4.5 Gaussian-kernel (Nicolini et al.) horizon threshold ~ 0.95 sigma0 c^2/G",
              abs(res.fun - 0.95) < 0.01, f"M_min = {res.fun:.4f} sigma0 c^2/G  (= {2 * res.fun:.3f} sqrt(theta), theta = sigma0^2/4)")
    rho_g = lambda s: np.exp(-s**2 / s0**2) / (np.pi**1.5 * s0**3)
    phi_g = [-(quad(lambda s: 4 * np.pi * s**2 * rho_g(s), 0, r)[0] / r
               + quad(lambda s: 4 * np.pi * s * rho_g(s), r, np.inf)[0]) for r in rr]
    from scipy.special import erf
    rep.close("Sec. 4.5 / Eq. (7.5) Gaussian source gives Phi = -G M erf(r/sigma0)/r",
              phi_g, -erf(rr / s0) / rr, rtol=1e-8)
    return rep


if __name__ == "__main__":
    run().print()
