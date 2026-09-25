"""Figures for the README (written to ../figures).

Fig. 1  Window sampling preserves inter-branch coherence (Section 2).
Fig. 2  Saturation of the gravitational action difference (Eqs. 7.9-7.16),
        with the Bild et al. (2023) operating point.
Fig. 3  Coherence factor exp(-2Gm^2/hbar c) in the saturated regime (Eq. 7.25).
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from common import G, HBAR, C, M_P, L_P, MICROGRAM
from sec2_window_sampling import rho12_closed_form, SIGMA, W

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False,
})


def fig_sampling():
    d0 = np.linspace(2, 40, 300)
    peak = np.array([abs(rho12_closed_form(-d / 2, d / 2, d, SIGMA)) for d in d0])
    peak /= peak[0]
    old = np.exp(-d0**2 / (4 * (SIGMA**2 + 2 * W**2)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(d0, peak, color=BLUE, label="Carrier readout at the branch nodes, Eq. (2.17)")
    ax.plot(d0, old, color=ORANGE, linestyle="--",
            label="Value at a single node ($\\Delta x_{nm}=0$): an overlap, not decoherence")
    ax.set_yscale("log")
    ax.set_ylim(1e-40, 5)
    ax.set_xlabel("Branch separation $d_0$ / packet width $w$")
    ax.set_ylabel("Inter-branch coherence (normalised)")
    ax.set_title("Window sampling does not suppress coherence", loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_sampling_preserves_coherence.png", dpi=160)
    plt.close(fig)


def fig_saturation():
    x = np.logspace(-15, 3, 800)
    small = x < 1e-3
    ratio = np.empty_like(x)
    ratio[small] = x[small] ** 2 / 3 - x[small] ** 4 / 10
    xs = x[~small]
    ratio[~small] = 1 - np.sqrt(np.pi) / 2 * erf(xs) / xs
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, ratio, color=BLUE, label="$\\Delta S(\\xi)/\\Delta S_{\\rm sat}$, Eqs. (7.9), (7.13)")
    ax.plot(x[x < 1], x[x < 1] ** 2 / 3, color=INK2, linestyle=":", linewidth=1.5,
            label="Small-separation limit $(\\xi/\\sigma_{\\rm eff})^2/3$, Eq. (7.16)")
    x_bild = 2.1e-18 / (2 * 27e-6)
    ax.plot([x_bild], [x_bild**2 / 3], "o", color=BLUE, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate("Bild et al. (2023)\n16.2 µg acoustic cat state", (x_bild, x_bild**2 / 3),
                xytext=(x_bild * 30, 1e-24), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.axvspan(1, 1e3, color=GRID, alpha=0.6, lw=0)
    ax.text(3, 1e-20, "saturated\nregime", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-31, 3)
    ax.set_xlabel("Branch separation $\\xi$ / effective width $\\sigma_{\\rm eff}$")
    ax.set_ylabel("Fraction of the saturated action")
    ax.set_title("The universal threshold applies only for $\\xi \\gg \\sigma_{\\rm eff} \\gtrsim R$",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_action_saturation.png", dpi=160)
    plt.close(fig)


def fig_threshold():
    m = np.linspace(0.1, 60, 600) * MICROGRAM
    coh = np.exp(-2 * G * m**2 / (HBAR * C))
    mH = M_P / np.sqrt(2)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(m / MICROGRAM, coh, color=BLUE)
    ax.axvline(mH / MICROGRAM, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(mH / MICROGRAM + 1, 0.8, f"$m_H = M_P/\\sqrt{{2}}$ = {mH / MICROGRAM:.2f} µg", fontsize=10)
    ax.plot([mH / MICROGRAM], [np.exp(-1)], "o", color=BLUE, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(mH / MICROGRAM + 1, np.exp(-1) + 0.03, "$e^{-1}$", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mass $m$ (µg)")
    ax.set_ylabel("Coherence factor $\\exp(-2Gm^2/\\hbar c)$")
    ax.set_title("Heisenberg cut in the saturated regime, Eqs. (7.19), (7.25)", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_heisenberg_cut.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_sampling()
    fig_saturation()
    fig_threshold()
    print(f"figures written to {OUT}")
