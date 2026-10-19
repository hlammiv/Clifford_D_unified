#!/usr/bin/env python3
"""c35_build_pools.py — pools (pool_t*.jsonl + index.csv) from the C35 rerun's multi_theta dumps.

Each level is a list of (dump_dir, theta_file) pairs in angle order; indices run consecutively across
the pairs so the numbering matches the pre-rerun pools (same angle files, same order).
Usage: python3 c35_build_pools.py OUT_SUFFIX   (writes pfp_l{L}{OUT_SUFFIX})
"""
import csv, json, os, sys
import candpool

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, "..", "c35_rerun")
LOC, LEN = os.path.join(R, "out"), os.path.join(R, "out_lenore")
LEVELS = {L: [(f"{LOC}/pf_l{L}", f"{H}/thetas_500.txt")] for L in (2, 4, 6, 8, 10)}
LEVELS[12] = [(f"{LOC}/L12_{s:03d}", f"{LOC}/L12_{s:03d}.th") for s in (0, 25, 50, 75)]
LEVELS[14] = [(f"{LEN}/L14_{s:03d}", f"{LEN}/L14_{s:03d}.th") for s in range(0, 200, 10)]
LEVELS[16] = [(f"{LEN}/L16_all", f"{LEN}/L16_all.th")]
suf = sys.argv[1]
for L, parts in LEVELS.items():
    out = os.path.join(H, f"pfp_l{L}{suf}")
    os.makedirs(out, exist_ok=True)
    i = 0
    with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["i", "theta", "n_cand", "best_eps", "f_levels", "wall"])
        w.writeheader()
        for mt, tfile in parts:
            thetas = [float(x) for x in open(tfile).read().split()]
            for j, th in enumerate(thetas):
                cands = candpool.parse(open(os.path.join(mt, f"t{j:04d}.txt")).read())
                eps = [candpool.eps_of(c, th) for c in cands]
                with open(os.path.join(out, f"pool_t{i:04d}.jsonl"), "w") as fo:
                    for c, e in zip(cands, eps):
                        fo.write(json.dumps(dict(i=i, theta=th, f=c["f"], nums=c["nums"], exps=c["exps"], D=0, eps=e)) + "\n")
                w.writerow(dict(i=i, theta=th, n_cand=len(cands), best_eps=min(eps) if eps else float("nan"),
                                f_levels=" ".join(sorted({str(c["f"]) for c in cands})), wall=""))
                i += 1
    print(f"level {L}: {i} angles -> {out}")
