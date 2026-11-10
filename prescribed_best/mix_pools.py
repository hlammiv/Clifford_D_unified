#!/usr/bin/env python3
"""mix_pools.py -- mixed-unitary (Campbell/Hastings) cost of R_z(theta) from our phase-free candidate pools.

For each theta: candidates within TOL x best eps are rebuilt exactly (analyze_pools.rebuild) and each gives the error
generator H = -i log(V T^dag), traceless part, T = diag(e^{-i th/2}, e^{i th/2}, 1).  The 324 diagonal phase copies
D V D^dag approximate the same T (D commutes with T), with H -> D H D^dag, so they cost nothing to add.
Weights p >= 0, sum 1, minimise ||sum p_k h_k||_2 (NNLS with a stiff sum row).  Normalised diamond distance of the
mixture, to second order:  eps_mix <= ||sum p_k H_k||_op + sum p_k ||H_k||_op^2.
Cost: the T counts of the components that carry weight, decomposed exactly (analyze_pools.costs), averaged with p.
Usage: python3 mix_pools.py LEVEL NTHETA --procs 4 --out mix_lLEVEL.csv"""
import argparse, csv, glob, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.linalg import logm

import analyze_pools as ap

TOL = 3.0
_Z = np.exp(2j * np.pi * np.arange(6) / 9)


def to_float(V):
    return np.array([[np.dot([int(x) for x in V[i][j].num.coefs], _Z) / 3 ** V[i][j].denom_pow3 for j in range(3)] for i in range(3)])


def herm8(H):
    return np.array([H[0, 0].real - H[2, 2].real, H[1, 1].real - H[2, 2].real,
                     H[0, 1].real, H[0, 1].imag, H[0, 2].real, H[0, 2].imag, H[1, 2].real, H[1, 2].imag])


def work(item):
    path, th_idx = item
    import fast_decompose
    fast_decompose.install()
    from variant_search import variant_groups, variant_list
    from ingest_decompose import build_ring
    cands = [json.loads(l) for l in open(path)]
    if not cands:
        return None
    th = cands[0]["theta"]
    T = np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1])
    cands.sort(key=lambda c: c["eps"])
    near = [c for c in cands if c["eps"] <= TOL * cands[0]["eps"]][:30]
    Vs = []
    for c in near:
        V, e = ap.rebuild(c, th)
        if V is not None:
            Vs.append((V, e))
    if not Vs:
        return None
    vl = variant_list("conj")
    Hs, comps = [], []                      # comps: (cand index, variant index)
    for ci, (V, e) in enumerate(Vs):
        M = to_float(V)
        H0 = -1j * logm(M @ T.conj().T)
        H0 = (H0 + H0.conj().T) / 2
        H0 -= np.trace(H0) / 3 * np.eye(3)
        for vi, (d, _t) in enumerate(vl):
            D = np.diag([s * np.exp(2j * np.pi * k / 9) for s, k in d])
            Hs.append(D @ H0 @ D.conj().T); comps.append((ci, vi))
    A = np.array([herm8(H) for H in Hs]).T                 # 8 x K
    scale = np.abs(A).max()
    w = 1e3
    Aw = np.vstack([A / scale, w * np.ones(A.shape[1])])
    b = np.zeros(9); b[-1] = w
    pw, _ = nnls(Aw, b, maxiter=20 * A.shape[1])
    pw /= pw.sum()
    keep = np.where(pw > 1e-9)[0]
    Hm = sum(pw[k] * Hs[k] for k in keep)
    first = np.linalg.norm(Hm, 2)
    second = sum(pw[k] * np.linalg.norm(Hs[k], 2) ** 2 for k in keep)
    # cost: decompose each component that carries weight
    tot = {m: 0.0 for m in ("T_unit", "T_merged", "T_meas", "N_D")}
    for k in keep:
        ci, vi = comps[k]
        V, _ = Vs[ci]
        g = [list(map(int, V[i][j].num.coefs)) for i in range(3) for j in range(3)]
        f = V[0][0].denom_pow3
        kk = ap.costs(build_ring(variant_groups(g, vl[vi][0], False), f))
        if kk is None:
            return dict(theta=th, fail=1)
        for m in tot:
            tot[m] += pw[k] * kk[m]
    return dict(theta=th, level=2 * near[0]["f"], eps_best=Vs[0][1], n_cand=len(Vs), n_comp=len(keep),
                first=first, second=second, eps_mix=first + second, **{m + "_mix": v for m, v in tot.items()}, fail=0)


def main():
    a = argparse.ArgumentParser(); a.add_argument("level", type=int); a.add_argument("ntheta", type=int)
    a.add_argument("--procs", type=int, default=4); a.add_argument("--out", required=True)
    o = a.parse_args()
    files = sorted(glob.glob(f"pfp_l{o.level}/pool_t*.jsonl"))[:o.ntheta]
    with Pool(o.procs) as P:
        recs = [r for r in P.imap_unordered(work, [(f, i) for i, f in enumerate(files)]) if r]
    keys = sorted({k for r in recs for k in r})
    with open(o.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(sorted(recs, key=lambda r: r["theta"]))
    print("wrote", o.out, len(recs))


if __name__ == "__main__":
    main()
