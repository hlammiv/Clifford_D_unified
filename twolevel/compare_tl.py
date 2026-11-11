#!/usr/bin/env python3
"""compare_tl.py — cost of two-level Householder targets vs modulus c1² (review C34).

Per c1²: per-level means, and linear fits y = a + b·log3(1/ε) over the individual approximants (levels 4-12),
with delete-one jackknife errors. c1² = 1 is R_z(θ). Also a pooled fit with per-c1² offsets in intercept and slope.
Usage: python3 compare_tl.py
"""
import glob
import re

import numpy as np
import pandas as pd

rows = []
for p in glob.glob("costs_c*_l*.csv"):
    c, L = re.match(r"costs_c([\d.]+)_l(\d+)\.csv", p).groups()
    d = pd.read_csv(p)
    d["c1sq"], d["L"] = float(c), int(L)
    rows.append(d)
# level-12 control (c1² = 1): the paper's phase-free pool costs (as found) on the same 50 θ
th50 = {round(float(x), 9) for x in open("thetas_50.txt").read().split()}
c12 = pd.read_csv("../prescribed_best/pfcosts_l12.csv")
c12 = c12[c12.theta.round(9).isin(th50)].copy()
c12["c1sq"], c12["L"] = 1.0, 12
rows.append(c12)
D = pd.concat(rows)
D["x"] = np.log(1 / D.eps) / np.log(3)
Q = ["N_D", "T_unit", "T_meas"]


def fit(d, y):
    X = np.c_[np.ones(len(d)), d.x.values]
    Y = d[y].values.astype(float)
    full = np.linalg.lstsq(X, Y, rcond=None)[0]
    n = len(d)
    jk = np.array([np.linalg.lstsq(np.delete(X, i, 0), np.delete(Y, i), rcond=None)[0] for i in range(n)])
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    return full, err


print("per level (100 θ at levels 2-10, 50 at 12):")
g = D.groupby(["L", "c1sq"])
t = g.agg(n=("eps", "size"), med_eps=("eps", "median"), x=("x", "mean"),
          **{q: (q, "mean") for q in Q}, nT3=("n_T3", "mean"), nL4=("n_L4", "mean"), nR=("n_R", "mean"))
print(t.round(3).to_string())

print("\nfits over levels 4-12, y = a + b·log3(1/ε); value at ε = 1e-10 (x = 20.96):")
x10 = 10 / np.log10(3)
F = D[D.L >= 4]
for q in Q:
    for c, d in F.groupby("c1sq"):
        (a, b), (ea, eb) = fit(d, q)
        print(f"  {q:7s} c1²={c:4.2f}: a = {a:6.2f}({ea:.2f})  b = {b:5.3f}({eb:.3f})  y(1e-10) = {a + b * x10:6.1f}")
