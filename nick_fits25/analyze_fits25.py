#!/usr/bin/env python3
"""analyze_fits25.py — Nick's best fits at 25 *generic* θ (f = 8, 10), vs his special-θ data.

His earlier fits_f=*.txt are special-θ matrices: each matrix defines its own θ, so the error
there is the best case for that f. These 25 θ were chosen by us, so they test what an
arbitrary rotation costs at a given f.

For each matrix: exact unitarity, Frobenius ε to R_z(θ), largest diagonal / off-diagonal
entry errors (Nick's metric), fast decomposition (as given, and best of 324 ±ζ conjugations)
in the unitary / merged / measurement T models.
Files have no '#f=' header; rows are 'θ , 9 groups of 6 ints' in column-major order (Dump()).
Usage: python3 analyze_fits25.py fits25_f=8.txt.gz fits25_f=10.txt.gz --procs 8
"""
import argparse
import csv
import gzip
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
U = H.parent
sys.path[:0] = [str(U / "symmetry_variants"), str(U / "decomp_speed"), str(U / "depth_optimality"),
                str(U / "compiler"), str(U / "nick_test"), str(U / "nick_request"), str(U), str(U / "hrsa")]


def read(path):
    f = int(re.search(r"f=(\d+)", Path(path).name).group(1))
    rows = []
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            g = [[int(x) for x in p.split()] for p in parts[1:10]]
            g = [g[3 * (k % 3) + k // 3] for k in range(9)]       # column-major -> row-major
            rows.append((f, float(parts[0]), g))
    return rows


def work(item):
    import fast_decompose
    fast_decompose.install()
    import canonical_reducer as cr
    from ingest_decompose import build_ring, build_complex, unitarity_exact
    from variant_search import variant_groups, variant_list
    from emit_variants import emit
    from tmerge import merge_symbolic
    f, th, g = item
    V = build_ring(g, f)
    exact, _ = unitarity_exact(V)
    M = build_complex(g, f)
    T = np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1])
    D = M - T
    rec = dict(f=f, theta=th, unitary_exact=int(exact), eps=float(np.linalg.norm(D)),
               diag_max=float(np.max(np.abs(np.diag(D)))),
               offdiag_max=float(np.max(np.abs(D - np.diag(np.diag(D))))))
    best = None
    t0 = time.time()
    for vid, (d, _t) in enumerate(variant_list("conj")):
        r = cr.decompose_canonical(build_ring(variant_groups(g, d, False), f))
        if not r["success"]:
            continue
        gates, c = emit(r["syllables"], r["trailing_clifford"], "shared", np.random.default_rng(0))
        t3, l4, nr = c["n_T3"], c["n_L4"], c["n_R"]
        cost = dict(N_D=r["D_count"], n_T3=t3, n_L4=l4, n_R=nr, T_unit=t3 + 7 * l4 + 7 * nr,
                    T_merged=merge_symbolic(gates), T_meas=t3 + 4 * l4 + 4 * nr)
        if vid == 0:
            rec.update({k + "_asis": v for k, v in cost.items()})
        if best is None:
            best = {k: v for k, v in cost.items()}
        else:
            for k in ("T_unit", "T_merged", "T_meas", "N_D"):
                best[k] = min(best[k], cost[k])
    rec.update({k + "_best": v for k, v in best.items() if k in ("T_unit", "T_merged", "T_meas", "N_D")})
    rec["wall"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=str(H / "fits25_results.csv"))
    a = ap.parse_args()
    items = [it for p in a.inputs for it in read(p)]
    with Pool(a.procs) as pool:
        recs = list(pool.imap_unordered(work, items))
    recs.sort(key=lambda r: (r["f"], r["theta"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
