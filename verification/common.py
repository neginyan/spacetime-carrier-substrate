"""Shared helpers for the verification scripts.

Every script builds a :class:`Report`, records individual checks against the
equations of the paper, and prints a PASS/FAIL table. ``run_all.py`` collects
the reports of all sections.
"""

from __future__ import annotations

import numpy as np
from scipy import constants as _k

# ---------------------------------------------------------------------------
# Physical constants (CODATA values shipped with SciPy)
# ---------------------------------------------------------------------------
G = _k.G
HBAR = _k.hbar
C = _k.c
M_SUN = 1.98847e30                      # kg (IAU nominal)
M_P = np.sqrt(HBAR * C / G)             # Planck mass [kg]
L_P = np.sqrt(HBAR * G / C**3)          # Planck length [m]
MICROGRAM = 1e-9                        # kg


def _fmt(x) -> str:
    x = np.asarray(x)
    if x.size == 1:
        v = complex(x.item()) if np.iscomplexobj(x) else float(x.item())
        if isinstance(v, complex):
            return f"{v.real:.6g}{v.imag:+.3g}j"
        return f"{v:.10g}"
    return f"array(shape={x.shape})"


class Report:
    """Collects named checks for one section of the paper."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool, str]] = []

    # -- recording --------------------------------------------------------
    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((label, bool(ok), detail))
        return bool(ok)

    def close(self, label: str, value, target, rtol: float = 1e-9,
              atol: float = 0.0) -> bool:
        value = np.asarray(value)
        target = np.asarray(target)
        ok = bool(np.allclose(value, target, rtol=rtol, atol=atol))
        diff = float(np.max(np.abs(value - target)))
        scale = float(np.max(np.abs(target))) or 1.0
        detail = (f"value={_fmt(value)}  target={_fmt(target)}  "
                  f"rel.err={diff / scale:.1e}")
        return self.check(label, ok, detail)

    # -- output -----------------------------------------------------------
    @property
    def n_pass(self) -> int:
        return sum(ok for _, ok, _ in self.rows)

    @property
    def all_passed(self) -> bool:
        return self.n_pass == len(self.rows)

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
