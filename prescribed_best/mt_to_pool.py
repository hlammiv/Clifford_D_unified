#!/usr/bin/env python3
"""mt_to_pool.py — multi_theta output (DIR/t{i:04d}.txt) -> run_pool.py format (OUT/pool_t*.jsonl + index.csv).
Usage: python3 mt_to_pool.py MT_DIR THETA_FILE OUT_DIR
"""
import csv
import json
import os
import sys

import candpool

mt, tfile, out = sys.argv[1:4]
thetas = [float(x) for x in open(tfile).read().split()]
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["i", "theta", "n_cand", "best_eps", "f_levels", "wall"])
    w.writeheader()
    for i, th in enumerate(thetas):
        cands = candpool.parse(open(os.path.join(mt, f"t{i:04d}.txt")).read())
        eps = [candpool.eps_of(c, th) for c in cands]
        with open(os.path.join(out, f"pool_t{i:04d}.jsonl"), "w") as fo:
            for c, e in zip(cands, eps):
                fo.write(json.dumps(dict(i=i, theta=th, f=c["f"], nums=c["nums"], exps=c["exps"], D=0, eps=e)) + "\n")
        w.writerow(dict(i=i, theta=th, n_cand=len(cands), best_eps=min(eps) if eps else float("nan"),
                        f_levels=" ".join(sorted({str(c["f"]) for c in cands})), wall=""))
print("wrote", out)
