#!/usr/bin/env python3
"""tl_to_pool.py — phase_free_tl output -> run_pool.py pool format, ε against the two-level target (PF_C1SQ).
Usage: PF_C1SQ=... python3 tl_to_pool.py PF_DIR THETA_FILE OUT_DIR
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prescribed_best"))
import candpool  # noqa: E402
from tl_target import P01, c1sq, target  # noqa: E402

pf, tfile, out = sys.argv[1:4]
thetas = [float(x) for x in open(tfile).read().split()]
os.makedirs(out, exist_ok=True)
for i, th in enumerate(thetas):
    T = target(th)
    with open(os.path.join(out, f"pool_t{i:04d}.jsonl"), "w") as fo:
        for c in candpool.parse(open(os.path.join(pf, f"t{i:04d}.txt")).read()):
            u = np.array(c["x"])
            e = float(np.linalg.norm(P01 @ (np.eye(3) - np.outer(u, u.conj())) - T))
            fo.write(json.dumps(dict(i=i, theta=th, c1sq=c1sq(), f=c["f"], nums=c["nums"], exps=c["exps"], D=0, eps=e)) + "\n")
print("wrote", out)
