"""Figures for the README (written to ./figures)."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from model import I0, I1, I1_asymptotic, J_far, J_far_asymptotic, TAU_PAPER
from step4_retarded_gaussian import C
from step5_mean_retardation import P_gauss, phi_ret

OUT = pathlib.Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False,
})


def fig_static():
    x = np.logspace(-2, 3, 120)
    tau = np.array([I1(v) / (4 * I0(v)) for v in x])
    xb = np.logspace(1.5, 6, 60)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, tau, color=BLUE, label="Emergent $\\tau_{\\rm eff}$, static superposition (Step 3)")
    ax.plot(xb, I1_asymptotic(xb) / (4 * np.sqrt(np.pi)), color=BLUE, linestyle=":", linewidth=1.5,
            label="Asymptote $[\\ln(\\xi/\\sigma_{\\rm eff})+\\gamma_E/2+\\ln 2-1]/4\\sqrt{\\pi}$")
    ax.axhline(TAU_PAPER, color=ORANGE, linestyle="--", label="Paper, Eq. (7.13): $\\sqrt{\\pi}/2$")
    ax.set_xscale("log")
    ax.set_xlabel("Branch separation $\\xi$ / $\\sigma_{\\rm eff}$")
    ax.set_ylabel("$\\tau_{\\rm eff}$ in units of $\\sigma_{\\rm eff}/c$")
    ax.set_title("Step 3: the $\\sigma_{\\rm eff}/c$ scale emerges; the coefficient grows with $\\xi$",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_step3_static.png", dpi=160)
    plt.close(fig)


def fig_finite_time():
    y = np.logspace(-1, 4, 120)
    tau = np.array([J_far(v) for v in y]) / (4 * np.sqrt(np.pi))
    yb = np.logspace(0.5, 4, 60)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(y, tau, color=BLUE, label="Emergent $\\tau_{\\rm eff}$ for $\\xi \\gg cT$ (Step 4)")
    ax.plot(yb, J_far_asymptotic(yb) / (4 * np.sqrt(np.pi)), color=BLUE, linestyle=":", linewidth=1.5,
            label="Asymptote $[\\ln(cT/\\sigma_{\\rm eff})+\\gamma_E/2+\\ln 2]/2\\sqrt{\\pi}$")
    ax.axhline(TAU_PAPER, color=ORANGE, linestyle="--", label="Paper, Eq. (7.13): $\\sqrt{\\pi}/2$")
    ax.axvline(8.70, color=INK2, linewidth=1, linestyle=":")
    ax.text(10, 0.05, "$cT = 8.70\\,\\sigma_{\\rm eff}$\n(solved backwards)", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Duration $cT$ / $\\sigma_{\\rm eff}$")
    ax.set_ylabel("$\\tau_{\\rm eff}$ in units of $\\sigma_{\\rm eff}/c$")
    ax.set_title("Step 4: a finite duration replaces $\\xi$ as the cut-off (only if $cT < \\xi$)",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_step4_finite_time.png", dpi=160)
    plt.close(fig)


def fig_retarded():
    u = np.linspace(0.05, 2.0, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(u, [C(v) for v in u], color=BLUE, label="$C(u) = 4u/(1+1/4u^2)$, retarded Gaussian model")
    ax.axhline(2, color=ORANGE, linestyle="--", label="Paper: $\\Delta S = 2Gm^2/c$")
    for uu, cc, txt in ((0.5, C(0.5), "width of Eq. (7.13):  C = 1"), (0.7328, 2.0, "C = 2 at u = 0.733")):
        ax.plot([uu], [cc], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(txt, (uu, cc), xytext=(uu + 0.12, cc - 0.55), fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlabel("$u = c\\,\\delta_t / \\sigma_{\\rm eff}$")
    ax.set_ylabel("$\\Delta S$ in units of $Gm^2/c$")
    ax.set_title("Retarded model: the coefficient depends on the time profile", loc="left",
                 fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_retarded_gaussian.png", dpi=160)
    plt.close(fig)


def fig_mean_retardation():
    t = np.linspace(0, 3.2, 400)
    phi = np.array([phi_ret(v) for v in t])
    phi /= phi.max()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(t, phi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(t, phi, color=BLUE, label="Retarded potential of an instantaneous Gaussian source")
    ax.axvline(TAU_PAPER, color=ORANGE, linestyle="--",
               label="Mean delay $\\langle t\\rangle = (\\sqrt{\\pi}/2)\\,\\sigma_{\\rm eff}/c$ = Eq. (7.13)")
    ax.set_xlabel("Time $t$ after the source acts, in units of $\\sigma_{\\rm eff}/c$")
    ax.set_ylabel("Potential at the centre (normalised)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Step 5: causal by construction, no truncation, no peak normalisation",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_step5_mean_retardation.png", dpi=160)
    plt.close(fig)


def fig_action_lag():
    T = np.linspace(0, 3.5, 400)
    S0 = T
    SR = T - TAU_PAPER + np.sqrt(np.pi) / 2 * (1 - erf(T))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(T, S0, color=INK2, linestyle=":", label="Instantaneous reference $S_0 = \\Delta E\\,T$")
    ax.plot(T, SR, color=BLUE, label="Retarded action $S_R(T) = \\Delta E\\int_0^T F\\,dt$")
    ax.plot(T, T - TAU_PAPER, color=BLUE, linestyle="--", lw=1.2, alpha=0.6,
            label="Asymptote $\\Delta E\\,(T - \\langle t\\rangle)$")
    x0 = 3.0
    ax.annotate("", xy=(x0, x0 - TAU_PAPER), xytext=(x0, x0),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8))
    ax.text(x0 + 0.08, x0 - TAU_PAPER / 2, "$\\Delta E\\,\\langle t\\rangle$\n(lag)",
            color=ORANGE, va="center", fontsize=10)
    ax.set_xlabel("Hold time $T$, in units of $\\sigma_{\\rm eff}/c$")
    ax.set_ylabel("Action, in units of $\\Delta E\\,\\sigma_{\\rm eff}/c$")
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 3.6)
    ax.set_title("Step 6: the retarded action lags by $\\langle t\\rangle$ but still grows with $T$",
                 loc="left", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_step6_action_lag.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_static()
    fig_finite_time()
    fig_retarded()
    fig_mean_retardation()
    fig_action_lag()
    print(f"figures written to {OUT}")
