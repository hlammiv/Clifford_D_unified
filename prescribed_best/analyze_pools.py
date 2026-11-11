#!/usr/bin/env python3
"""analyze_pools.py — gate costs of prescribed-θ candidate pools (run_pool.py output).

Per θ:
  best-ε candidate (= best fit at that level): N_D, n_T3, n_L4, n_R, T_unit, T_merged, T_meas
  best symmetric copy of it (324 ±ζ conjugations; only for the first --copies-n θ): min T per model, and the
  class content (N_D, n_T3, n_L4, n_R) of that cheapest copy as *_copy_{unit,merged,meas}
  best-of-K: min T over candidates with ε <= tol·best_ε (tol 1.25), and pool counts
Rebuild: exact unitarity, ε at 40 digits (exact_check.py). Decomposition: fast exact reducer; merged T: shared-ancilla emit + symbolic merge.
Usage: python3 analyze_pools.py POOL_DIR --procs 12 --copies-n 100 --out OUT.csv
"""
import argparse
import csv
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
U = H.parent
sys.path[:0] = [str(U / "symmetry_variants"), str(U / "decomp_speed"), str(U / "depth_optimality"), str(U / "compiler"),
                str(U / "r_count_study"), str(U / "nick_test"), str(U), str(U / "hrsa")]


def costs(V):
    import canonical_reducer as cr
    from emit_variants import emit
    from tmerge import merge_symbolic
    r = cr.decompose_canonical(V)
    if not r["success"]:
        return None
    gates, c = emit(r["syllables"], r["trailing_clifford"], "shared", np.random.default_rng(0))
    t3, l4, nr = c["n_T3"], c["n_L4"], c["n_R"]
    return dict(N_D=r["D_count"], n_T3=t3, n_L4=l4, n_R=nr, T_unit=t3 + 7 * l4 + 7 * nr,
                T_merged=merge_symbolic(gates), T_meas=t3 + 4 * l4 + 4 * nr)

_Z = np.exp(2j * np.pi * np.arange(6) / 9)


def TARGET(c, th):
    """Target at DPS digits for record c; tl_analyze / full_analyze replace this."""
    import exact_check as ec
    return ec.target_rz(th)


def rebuild(c, th):
    """Exact V for a pool candidate: unitarity checked exactly, ε evaluated at 40 digits.

    CANDDUMP prints numerators at each element's own denominator exponent. Pools written
    before 2026-10-08 evening lack these exponents ("exps"), so they are recovered: try
    every assignment in {0..f}^3 (all-f first) and accept it only if exactly one gives an
    exactly unitary V whose ε matches the search's value to 1e-4 relative.
    """
    from analyze_candidates import build_V_exps
    import exact_check as ec
    f = c["f"]
    combos = [c["exps"]] if c.get("exps") else \
        [[f, f, f]] + [[a, b, d] for a in range(f + 1) for b in range(f + 1) for d in range(f + 1) if (a, b, d) != (f, f, f)]
    T = TARGET(c, th)
    hits = []
    for ex in combos:
        V = build_V_exps(*c["nums"], ex)
        if not ec.exact_unitary(V):
            continue
        e = ec.eps_hp(V, T)
        if abs(e - c["eps"]) <= 1e-4 * e:
            hits.append((V, float(e)))
            if len(hits) > 1:
                break
    return hits[0] if len(hits) == 1 else (None, None)


def work(item):
    import fast_decompose
    fast_decompose.install()
    from analyze_candidates import build_V
    from ingest_decompose import build_complex
    from variant_search import variant_groups, variant_list
    from ingest_decompose import build_ring
    path, do_copies, tol = item
    cands = [json.loads(l) for l in open(path)]
    if not cands:
        return None
    th = cands[0]["theta"]
    for i, c in enumerate(cands):
        c["_idx"] = i                     # search order: breaks exact ties in ε (e.g. diagonal conjugates)
    cands.sort(key=lambda c: c["eps"])
    best = cands[0]
    near = [c for c in cands if c["eps"] <= tol * best["eps"]]
    rows = []
    nbad = nfail = 0
    relerr = 0.0                          # max |search ε − 40-digit ε| / ε over rebuilt candidates
    for c in near:
        V, e_exact = rebuild(c, th)
        if V is None:
            nbad += 1                     # no exponent assignment reproduces HRSA's candidate: skip
            continue
        relerr = max(relerr, abs(c["eps"] - e_exact) / e_exact)
        c["eps"] = e_exact
        k = costs(V)
        if k:
            rows.append((c["eps"], c["_idx"], k, V))
        else:
            nfail += 1                    # decomposition failed (double prefix did not lower sde): counted, not hidden
    if not rows:
        return None
    rows.sort(key=lambda r: r[:2])
    e0, _, k0, V0 = rows[0]
    rec = dict(theta=th, level=2 * best["f"], n_pool=len(cands), n_near=len(near), n_rebuild_bad=nbad, eps=e0,
               **{x: k0[x] for x in ("N_D", "n_T3", "n_L4", "n_R", "T_unit", "T_merged", "T_meas")})
    ndec = len(rows)
    for m in ("T_unit", "T_merged", "T_meas"):
        rec[m + "_bestK"] = min(k[m] for _, _, k, _ in rows)
    if do_copies:
        # V0 -> 9 coefficient groups at its denominator for variant_groups
        g = [list(map(int, V0[i][j].num.coefs)) for i in range(3) for j in range(3)]
        f = V0[0][0].denom_pow3
        bestc = {m: k0 for m in ("T_unit", "T_merged", "T_meas")}   # cheapest copy per model (k dict)
        for d, _t in variant_list("conj"):
            gv = variant_groups(g, d, False)
            k = costs(build_ring(gv, f))
            if k:
                ndec += 1
                for m in bestc:
                    if k[m] < bestc[m][m]:
                        bestc[m] = k
            else:
                nfail += 1
        for m, k in bestc.items():
            rec[m + "_copy"] = k[m]
            for c in ("N_D", "n_T3", "n_L4", "n_R"):          # class content of that cheapest copy
                rec[f"{c}_copy_{m[2:]}"] = k[c]
    rec["n_decomposed"] = ndec
    rec["n_decomp_fail"] = nfail
    rec["eps_relerr_max"] = relerr
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--copies-n", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(str(Path(a.pool_dir) / "pool_t*.jsonl")))
    items = [(p, i < a.copies_n, a.tol) for i, p in enumerate(files)]
    with Pool(a.procs) as p:
        recs = [r for r in p.imap_unordered(work, items, chunksize=2) if r]
    keys = sorted({k for r in recs for k in r}, key=lambda k: (k not in ("theta", "level", "eps"), k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(recs, key=lambda r: r["theta"]))
    print("wrote", a.out, len(recs))


if __name__ == "__main__":
    main()
