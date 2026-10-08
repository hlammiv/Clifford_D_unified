#!/usr/bin/env python3
"""fit_generic.py — N_T = a + b·log₃(1/ε) at PRESCRIBED (generic) θ.

Points, one per (θ, level), taking the best fit at that level:
  levels 2, 4, 6 : costs_l{2,4,6}.csv (analyze_pools.py; same angles as the Oct-3 sweep)
  levels 8, 10   : ../nick_fits25/fits25_results.csv (Nick, 25 θ each)
Two selections per model:
  as-is      : the best-ε approximant itself
  best copy  : min over its 324 ±ζ symmetric copies (only θ with copies computed)
Compared with the special-θ headline (results_bundle/README.md).
"""
import collections
import csv
import gzip
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SPECIAL = {"T_unit": (2.5, 9.71), "T_merged": (3.3, 7.21), "T_meas": (1.8, 6.42)}
NAMES = {"T_unit": "unitary 7-T", "T_merged": "merged", "T_meas": "measurement 4-T"}


def load():
    pts = []                                   # (level, eps, {model: (asis, copy|None)}, N_D)
    for l in (2, 4, 6):
        p = H / f"costs_l{l}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            m = {k: (float(r[k]), float(r[k + "_copy"]) if r.get(k + "_copy") else None) for k in SPECIAL}
            pts.append((l, float(r["eps"]), m, float(r["N_D"])))
    for r in csv.DictReader(open(H.parent / "nick_fits25" / "fits25_results.csv")):
        if r["unitary_exact"] != "1":
            continue
        m = {k: (float(r[k + "_asis"]), float(r[k + "_best"])) for k in SPECIAL}
        pts.append((int(r["f"]), float(r["eps"]), m, float(r["N_D_asis"])))
    return pts


def rd(p):
    p = Path(p)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    return csv.DictReader(gzip.open(p, "rt") if str(p).endswith(".gz") else open(p))


def special_by_level():
    """Nick's special-θ matrices (the headline set): per f, median ε and mean T (as-is / best copy)."""
    U = H.parent
    key = lambda f, t: (str(f), f"{float(t):.6g}")  # noqa: E731
    eps = {key(r["f"], r["theta"]): float(r["epsilon"])
           for r in rd(U / "nick_request/topk_run/fast_all30_candidates.csv") if r["rank"] == "0"}
    by = collections.defaultdict(list)
    for r in rd(U / "symmetry_variants/stack_full30_rows.csv"):
        if r["ok"] == "1":
            by[key(r["f"], r["theta"])].append(r)
    out = {}
    for f in sorted({k[0] for k in by}, key=int):
        ks = [k for k in by if k[0] == f]
        row = {"eps": float(np.median([eps[k] for k in ks])), "n": len(ks)}
        for c in SPECIAL:
            row[c] = (np.mean([float([r for r in by[k] if r["variant"] == "0"][0][c]) for k in ks]),
                      np.mean([min(float(r[c]) for r in by[k]) for k in ks]))
        out[int(f)] = row
    return out


def fit(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def main():
    pts = load()
    levels = sorted({p[0] for p in pts})
    print("points per level:", {l: sum(p[0] == l for p in pts) for l in levels})
    print("\nper-level means (best fit at each θ)")
    print(f"{'level':>5} {'n':>4} {'med eps':>9} {'log3(1/e)':>9} {'N_D':>6} " +
          " ".join(f"{NAMES[k]:>22}" for k in SPECIAL))
    for l in levels:
        q = [p for p in pts if p[0] == l]
        e = np.median([p[1] for p in q])
        cells = []
        for k in SPECIAL:
            a = np.mean([p[2][k][0] for p in q])
            c = [p[2][k][1] for p in q if p[2][k][1] is not None]
            cells.append(f"{a:8.1f} / {np.mean(c):6.1f} (n={len(c):3d})" if c else f"{a:8.1f}")
        print(f"{l:>5} {len(q):>4} {e:9.2e} {np.log(1 / e) / np.log(3):9.2f} {np.mean([p[3] for p in q]):6.1f} " + " ".join(f"{c:>22}" for c in cells))
    print("  (cells: as-is / best symmetric copy)")

    x_all = np.array([np.log(1 / p[1]) / np.log(3) for p in pts])
    print("\nfits N = a + b·log₃(1/ε) over all (θ, level) points")
    a, b = fit(x_all, [p[3] for p in pts])
    print(f"  N_D               : {a:5.1f} + {b:5.2f}·log₃   (special-θ headline 3.94 + 5.16)")
    for k in SPECIAL:
        a1, b1 = fit(x_all, [p[2][k][0] for p in pts])
        sel = [p for p in pts if p[2][k][1] is not None]
        a2, b2 = fit([np.log(1 / p[1]) / np.log(3) for p in sel], [p[2][k][1] for p in sel])
        sa, sb = SPECIAL[k]
        print(f"  {NAMES[k]:18s}: as-is {a1:5.1f} + {b1:5.2f}·log₃ (@1e-10: {a1 + b1 * 20.96:5.0f}) | "
              f"best copy {a2:5.1f} + {b2:5.2f}·log₃ (@1e-10: {a2 + b2 * 20.96:5.0f}) | "
              f"special-θ best copy {sa} + {sb}·log₃ (@1e-10: {sa + sb * 20.96:4.0f})")
    compare_levels(pts)


def compare_levels(pts):
    sp = special_by_level()
    print("\nsame level, generic vs special θ (best copy, mean T; ε = median)")
    print(f"{'level':>5} {'eps generic':>11} {'eps special':>11} {'ratio':>6}  " +
          "  ".join(f"{NAMES[k]:>17}" for k in SPECIAL))
    for l in sorted({p[0] for p in pts}):
        if l not in sp:
            continue
        q = [p for p in pts if p[0] == l]
        eg = np.median([p[1] for p in q])
        cells = []
        for k in SPECIAL:
            c = [p[2][k][1] for p in q if p[2][k][1] is not None]
            cells.append(f"{np.mean(c):6.1f} vs {sp[l][k][1]:6.1f}")
        print(f"{l:>5} {eg:11.2e} {sp[l]['eps']:11.2e} {eg / sp[l]['eps']:6.1f}  " + "  ".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
