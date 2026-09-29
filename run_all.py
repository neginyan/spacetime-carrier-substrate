"""Run every check and print a summary.  Exit code 0 if all checks pass."""

from __future__ import annotations

import sys
import time

import step3_static
import step4_finite_time
import step4_retarded_gaussian
import step5_mean_retardation
import step6_action_lag


def main() -> int:
    t0 = time.time()
    reports = []
    for mod in (step3_static, step4_finite_time, step4_retarded_gaussian, step5_mean_retardation,
                step6_action_lag):
        rep = mod.run()
        rep.print()
        if hasattr(mod, "table"):
            mod.table()
        reports.append(rep)
    total = sum(len(r.rows) for r in reports)
    passed = sum(r.n_pass for r in reports)
    print()
    print("#" * 78)
    for r in reports:
        print(f"  {'OK  ' if r.n_pass == len(r.rows) else 'FAIL'}  {r.n_pass:2d}/{len(r.rows):<2d}  {r.title}")
    print(f"\n  {passed}/{total} checks passed in {time.time() - t0:.1f} s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
