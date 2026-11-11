#!/usr/bin/env python3
"""full_to_pool.py — phase_free_full output -> run_pool.py pool format, ε against the full target (PF_C1SQ, PF_C3SQ).
Usage: PF_C1SQ=... PF_C3SQ=... python3 full_to_pool.py PF_DIR TARGET_FILE OUT_DIR
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(H.parent.parent / "prescribed_best")]
import candpool  # noqa: E402
from full_target import P01, moduli, read_targets, target  # noqa: E402

pf, tfile, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for i, (th, et) in enumerate(read_targets(tfile)):
    T = target(th, et)
    with open(os.path.join(out, f"pool_t{i:04d}.jsonl"), "w") as fo:
        for c in candpool.parse(open(os.path.join(pf, f"t{i:04d}.txt")).read()):
            u = np.array(c["x"])
            e = float(np.linalg.norm(P01 @ (np.eye(3) - np.outer(u, u.conj())) - T))
            fo.write(json.dumps(dict(i=i, theta=th, eta=et, csq=moduli(), f=c["f"], nums=c["nums"], exps=c["exps"],
                                     D=0, eps=e)) + "\n")
print("wrote", out)
