#!/usr/bin/env python3
"""run_pool.py — candidate pools of exact approximants at PRESCRIBED (generic) θ.

Why: bisecting HRSA's target ε (clifford_d_paper/analysis/prescribed_theta/best_eps.py)
under-reports the best fit. HRSA's search region depends on the target, so a tighter
target can prune a better solution (θ=3.9508 at max_f=2: bisection 0.0100, true ≥ 0.0054
found at a looser target). Instead, run HRSA once per θ at a loose target with a huge
--max-solns and keep every candidate (CANDDUMP). min ε over the pool = best fit; the pool
also gives best-of-K for T-cost. --min-f max_f makes HRSA search ONLY level max_f
(HRSA otherwise stops at the first level with any candidate under the target).
--no-decompose (a real flag since 2026-10-08) skips HRSA's own C++ decompositions, ~1000x
faster; candidates are decomposed afterwards with the fast Python reducer.

Angles: rng(20261003).uniform(0, 2π, n), the same as best_eps.py / run_sde4.py.
Output: OUT/pool_t{i:04d}.jsonl, one candidate per line (θ, f, numerators, ε), plus OUT/index.csv.
Usage: python3 run_pool.py --hrsa PATH --out DIR --max-f 2 --target 0.05 --n-theta 500 --workers 26
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candpool  # noqa: E402


def one(args):
    i, th, hrsa, out, max_f, target, max_solns = args
    t0 = time.time()
    p = subprocess.run([hrsa, f"{th:.12f}", repr(target), str(max_f), "--no-direct",
                        "--min-f", str(max_f), "--max-solns", str(max_solns), "--no-decompose"], capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=6 * 3600)
    cands = candpool.parse(p.stdout)
    with open(os.path.join(out, f"pool_t{i:04d}.jsonl"), "w") as fh:
        for c in cands:
            fh.write(json.dumps(dict(i=i, theta=th, f=c["f"], nums=c["nums"], exps=c["exps"], D=c["D"],
                                     eps=candpool.eps_of(c, th))) + "\n")
    eps = [candpool.eps_of(c, th) for c in cands]
    return dict(i=i, theta=th, n_cand=len(cands), best_eps=min(eps) if eps else float("nan"),
                f_levels=" ".join(sorted({str(c["f"]) for c in cands})), wall=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrsa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-f", type=int, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--n-theta", type=int, default=100)
    ap.add_argument("--max-solns", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    thetas = np.random.default_rng(20261003).uniform(0, 2 * math.pi, a.n_theta)
    rows = []
    with Pool(a.workers) as p, open(os.path.join(a.out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["i", "theta", "n_cand", "best_eps", "f_levels", "wall"])
        w.writeheader()
        for k, r in enumerate(p.imap_unordered(one, [(i, float(t), a.hrsa, a.out, a.max_f, a.target, a.max_solns)
                                                    for i, t in enumerate(thetas)])):
            w.writerow(r)
            fh.flush()
            rows.append(r)
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{a.n_theta}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
