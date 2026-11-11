#!/usr/bin/env python3
"""compare_full.py — cost of full Householder targets vs R_z(θ) (review C34).

Settings (settings.txt) vs the control R_z(θ) = the paper's phase-free pools, costed as found
(prescribed_best/pfcosts_l*.csv). Per-level means ± standard error, and fits y = a + b·log3(1/ε) over the
individual approximants at levels 4-12 with delete-one jackknife errors, evaluated at ε = 1e-10.
Usage: python3 compare_full.py
"""
import glob
import re

import numpy as np
import pandas as pd

Q = ["N_D", "T_unit", "T_meas"]
rows = []
for p in glob.glob("costs_*_l*.csv"):
    m = re.match(r"costs_([a-z0-9]+)_l(\d+)[ab]?\.csv", p)
    if not m or "xchk" in p:
        continue
    d = pd.read_csv(p)
    d["set"], d["L"] = m.group(1), int(m.group(2))
    rows.append(d)
for p in glob.glob("../../prescribed_best/pfcosts_l*.csv"):
    L = int(re.search(r"_l(\d+)\.csv", p).group(1))
    if L > 12:
        continue
    d = pd.read_csv(p)
    d["set"], d["L"] = "Rz", L
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
D["x"] = np.log(1 / D.eps) / np.log(3)
order = ["Rz", "bal", "s2", "s3", "skew"]

print("per level: mean ± s.e. (n)")
for q in Q:
    print(q)
    for L, g in D.groupby("L"):
        cells = []
        for s in order:
            h = g[g.set == s]
            if len(h):
                cells.append(f"{s}: {h[q].mean():6.1f}±{h[q].std() / np.sqrt(len(h)):.1f} ({len(h)})")
        print(f"  L{L:2d} " + "  ".join(cells))


def fit(d, y):
    X = np.c_[np.ones(len(d)), d.x.values]
    Y = d[y].values.astype(float)
    full = np.linalg.lstsq(X, Y, rcond=None)[0]
    n = len(d)
    jk = np.array([np.linalg.lstsq(np.delete(X, i, 0), np.delete(Y, i), rcond=None)[0] for i in range(n)])
    cov = (n - 1) / n * (jk - jk.mean(0)).T @ (jk - jk.mean(0))
    return full, cov


x10 = 10 / np.log10(3)
print("\nfits over levels 4-12, y = a + b·log3(1/ε); y(1e-10) with jackknife error:")
F = D[(D.L >= 4) & (D.L <= 12)]
for q in Q:
    for s in order:
        d = F[F.set == s]
        if not len(d):
            continue
        (a, b), C = fit(d, q)
        v = np.array([1, x10])
        print(f"  {q:7s} {s:5s}: a = {a:6.2f}({np.sqrt(C[0, 0]):.2f})  b = {b:6.3f}({np.sqrt(C[1, 1]):.3f})"
              f"  y(1e-10) = {a + b * x10:6.1f}({np.sqrt(v @ C @ v):.1f})  levels {sorted(d.L.unique())}")
