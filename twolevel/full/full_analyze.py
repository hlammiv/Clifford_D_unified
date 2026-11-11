#!/usr/bin/env python3
"""full_analyze.py — analyze_pools.py with the full Householder target (env PF_C1SQ, PF_C3SQ).

Only the exact rebuild's ε changes (target from each record's θ, η); decomposition and costing are
analyze_pools.py's. Use --copies-n 0: phase copies DVD† approximate a different gate for these targets.
Usage: PF_C1SQ=... PF_C3SQ=... python3 full_analyze.py POOL_DIR --procs 14 --copies-n 0 --out OUT.csv
"""
import sys
from pathlib import Path


H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(H.parent.parent / "prescribed_best")]
import analyze_pools as ap  # noqa: E402


def target_hp(c, th):
    import exact_check as ec
    return ec.target_full(th, c["eta"])


ap.TARGET = target_hp

if __name__ == "__main__":
    ap.main()
