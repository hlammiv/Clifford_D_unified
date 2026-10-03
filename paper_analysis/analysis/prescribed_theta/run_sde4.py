#!/usr/bin/env python3
"""run_sde4.py -- uniform prescribed-angle HRSA runs at sde_3 = 2*max_f (default max_f = 2, sde_3 = 4).

Every cell: HRSA_tester theta eps max_f --no-direct --no-decompose --json (max_solns = 1, no candidate
selection, one OpenMP thread), so the counts carry no selection bias and are reproducible.  --no-decompose skips
HRSA's own decomposition (identical V; we re-decompose with the fixed reducer).  Angles are uniform in [0, 2pi) from a
fixed seed; the job list is split into shards so several machines can run disjoint parts.

Usage: run_sde4.py --hrsa PATH --out DIR --shard K --nshards N --workers W [--n-theta 500]
                   [--max-f 2] [--targets 0.01,0.02,...]
"""
import argparse, json, math, os, subprocess
from multiprocessing import Pool

import numpy as np

TARGETS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2)


def jobs(n_theta, targets=TARGETS, seed=20261003):
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * math.pi, n_theta)
    return [(i, j, float(th), eps) for j, eps in enumerate(targets) for i, th in enumerate(thetas)]


def run(args):
    (i, j, th, eps), hrsa, out, max_f = args
    path = os.path.join(out, f"hrsa_f{max_f}_e{eps:g}_t{i:04d}.json" if max_f != 2 else f"hrsa_e{j}_t{i:04d}.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            json.load(open(path))
            return path, "skip"
        except ValueError:
            pass                         # truncated by an interrupted run: redo
    try:
        # one OpenMP thread: HRSA keeps the first candidate it finds, so a single thread makes
        # the result deterministic and identical across machines
        subprocess.run([hrsa, f"{th:.12f}", str(eps), str(max_f), "--no-direct", "--no-decompose", "--json", path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600, check=False,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
        return path, "ok" if os.path.exists(path) else "nojson"
    except subprocess.TimeoutExpired:
        return path, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrsa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--weights", default="", help="comma-separated shard weights, e.g. 1,3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-theta", type=int, default=500)
    ap.add_argument("--max-f", type=int, default=2, help="HRSA u-denominator cap; sde_3 = 2 * max_f")
    ap.add_argument("--targets", default="", help="comma-separated error targets (default: the sde-4 list)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    targets = tuple(float(x) for x in a.targets.split(",")) if a.targets else TARGETS
    js = jobs(a.n_theta, targets)
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * a.nshards
    cut = np.cumsum([0] + w) / sum(w) * len(js)
    lo, hi = int(round(cut[a.shard])), int(round(cut[a.shard + 1]))
    mine = js[lo:hi]
    print(f"shard {a.shard}/{a.nshards}: jobs {lo}..{hi} ({len(mine)})", flush=True)
    done = 0
    with Pool(a.workers) as p:
        for path, st in p.imap_unordered(run, [(x, a.hrsa, a.out, a.max_f) for x in mine]):
            done += 1
            if done % 50 == 0 or st not in ("ok", "skip"):
                print(f"{done}/{len(mine)} {st} {os.path.basename(path)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
