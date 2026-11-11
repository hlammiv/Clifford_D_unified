#!/usr/bin/env python3
"""hrsa_best_tcost.py — T-costs of HRSA best-ε fits at generic θ (sweep_hrsa_best_*_bisect JSONs).
Usage: python3 hrsa_best_tcost.py SWEEP_DIR --procs 8 --out OUT.csv"""
import argparse, csv, glob, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
U = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(U / "decomp_speed"), str(U / "depth_optimality"), str(U / "compiler"), str(U / "nick_test"), str(U), str(U / "hrsa")]

def work(path):
    import fast_decompose; fast_decompose.install()
    import canonical_reducer as cr
    from ingest_decompose import build_ring, build_complex, unitarity_exact
    from emit_variants import emit
    from tmerge import merge_symbolic
    j = json.load(open(path))
    if not j["achieved"]["success"]:
        return None
    f, th = j["unitary"]["f"], j["inputs"]["theta"]
    g = [list(e) for row in j["unitary"]["V"] for e in row]           # row-major 9 groups
    ex, _ = unitarity_exact(build_ring(g, f))
    T = np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1])
    eps = float(np.linalg.norm(build_complex(g, f) - T))
    r = cr.decompose_canonical(build_ring(g, f))
    gates, c = emit(r["syllables"], r["trailing_clifford"], "shared", np.random.default_rng(0))
    t3, l4, nr = c["n_T3"], c["n_L4"], c["n_R"]
    return dict(file=Path(path).name, f=f, theta=th, eps=eps, eps_json=j["achieved"]["achieved_frob"], unitary_exact=int(ex),
                N_D=r["D_count"], n_T3=t3, n_L4=l4, n_R=nr, T_unit=t3 + 7*l4 + 7*nr, T_merged=merge_symbolic(gates), T_meas=t3 + 4*l4 + 4*nr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("sweep"); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(str(Path(a.sweep) / "best_t*.json")))
    with Pool(a.procs) as p:
        recs = [r for r in p.map(work, files) if r]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    print("wrote", a.out, len(recs))
