"""Run every verification script and print a summary.

Usage:  python run_all.py            (exit code 0 if every check passes)
"""

from __future__ import annotations

import sys
import time

import sec2_window_sampling
import sec3_krein_duality
import sec4_6_regular_geometry
import sec5_information_geometry
import sec7_heisenberg_cut
import sec8_observables

MODULES = [
    sec2_window_sampling,
    sec3_krein_duality,
    sec4_6_regular_geometry,
    sec5_information_geometry,
    sec7_heisenberg_cut,
    sec8_observables,
]


def main() -> int:
    t0 = time.time()
    reports = []
    for mod in MODULES:
        rep = mod.run()
        rep.print()
        reports.append(rep)
    print()
    print("#" * 78)
    print("SUMMARY")
    print("#" * 78)
    total = passed = 0
    for rep in reports:
        total += len(rep.rows)
        passed += rep.n_pass
        print(f"  {'OK  ' if rep.all_passed else 'FAIL'}  {rep.n_pass:3d}/{len(rep.rows):<3d}  {rep.title}")
    print(f"\n  {passed}/{total} checks passed in {time.time() - t0:.1f} s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
