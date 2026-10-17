#!/usr/bin/env python3
"""tl_analyze.py — analyze_pools.py with the two-level target of tl_target.py (env PF_C1SQ).

Only the exact rebuild's ε changes; decomposition and costing are analyze_pools.py's. Phase copies DVD† are
not used (--copies-n 0): for a non-diagonal target they approximate a different gate.
Usage: PF_C1SQ=... python3 tl_analyze.py POOL_DIR --procs 12 --copies-n 0 --out OUT.csv
"""
import sys
from pathlib import Path


sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / "prescribed_best")]
import analyze_pools as ap  # noqa: E402


def target_hp(c, th):
    import exact_check as ec
    return ec.target_tl(th)


ap.TARGET = target_hp

if __name__ == "__main__":
    ap.main()
