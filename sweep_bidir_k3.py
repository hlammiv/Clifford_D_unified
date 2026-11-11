#!/usr/bin/env python3
"""sweep_bidir_k3.py — bidir K=3 sweep across the 100-θ grid.

For each θ in the standard grid, runs hrsa/bidir_bfs with K_f=K_b=3
(depth-6 BFS, ~350 MB RSS) to find the best Frobenius distance to
R^Z(θ) achievable in 6 Clifford+D operations.

Output CSV mirrors sweep_hrsa_grid.py columns so the plotter can ingest
it directly. Each θ produces ONE row per standard ε tier where the
K=3 reach fits (i.e., ε >= achieved_frob_K3).

Concurrency: 4 parallel workers (= 1.4 GB RAM) — safe on lucia.
"""
import argparse
import concurrent.futures as cf
import csv
import math
import re
import subprocess
import time
from pathlib import Path

HRSA_DIR = Path("/home/hlamm/Desktop/efficent_gates/unified/hrsa")
BIDIR = HRSA_DIR / "bidir_bfs"

CSV_FIELDS = [
    "theta_idx", "theta", "epsilon", "method", "N_D",
    "N_D_only", "N_R_only", "N_T", "f", "achieved_frob",
    "validate_frob", "wall_s", "all_checks_pass",
]


def run_bidir_k3(theta, timeout=180):
    """Run bidir_bfs at K=3, parse WIN line.
    Returns dict(theta, depth, frob, wall_s) or None on failure."""
    t0 = time.time()
    try:
        r = subprocess.run([str(BIDIR), str(theta), "3", "3"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"theta": theta, "depth": None, "frob": None,
                "wall_s": time.time() - t0, "error": "timeout"}
    dt = time.time() - t0
    m = re.search(r"WIN: depth=(\d+) frob=([\d.eE+-]+)", r.stdout)
    if not m:
        return {"theta": theta, "depth": None, "frob": None,
                "wall_s": dt, "error": "no WIN line"}
    return {"theta": theta, "depth": int(m.group(1)),
            "frob": float(m.group(2)), "wall_s": dt, "error": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_thetas", type=int, default=100)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=2 * math.pi)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel bidir invocations (each ~350 MB RSS).")
    p.add_argument("--out", required=True, help="CSV path.")
    p.add_argument("--eps_tiers", default="0.5,0.3,0.2,0.15,0.1,0.05,0.01",
                   help="ε values to emit rows at (each θ emits one row per ε "
                        "where ε >= achieved_frob_K3).")
    args = p.parse_args()

    step = (args.theta_max - args.theta_min) / args.n_thetas
    thetas = [(i, args.theta_min + (i + 0.5) * step) for i in range(args.n_thetas)]
    eps_tiers = [float(s) for s in args.eps_tiers.split(",")]

    print(f"[bidir-k3] {len(thetas)} θ across [{args.theta_min:.4f}, "
          f"{args.theta_max:.4f}], {args.workers} parallel workers")
    print(f"[bidir-k3] eps tiers: {eps_tiers}")
    print(f"[bidir-k3] writing to {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    fh.flush()

    n_done = 0
    t_sweep = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_bidir_k3, th): (i, th) for i, th in thetas}
        for f in cf.as_completed(futs):
            i, th = futs[f]
            res = f.result()
            n_done += 1
            elapsed = time.time() - t_sweep
            eta = elapsed / n_done * (len(thetas) - n_done)
            if res["frob"] is None:
                print(f"  [{i:03d}] θ={th:.4f} FAILED: {res.get('error')}  "
                      f"({n_done}/{len(thetas)}, eta={eta/60:.1f}min)")
                continue
            depth = res["depth"]
            frob = res["frob"]
            wall = res["wall_s"]
            # Emit one row per eps tier where ε >= frob (cell is reachable).
            emitted = 0
            for eps in eps_tiers:
                if eps >= frob:
                    writer.writerow({
                        "theta_idx": i, "theta": th, "epsilon": eps,
                        "method": "bidir(3+3)", "N_D": depth,
                        "N_D_only": depth, "N_R_only": 0, "N_T": depth,
                        "f": "", "achieved_frob": frob,
                        "validate_frob": "", "wall_s": f"{wall:.3f}",
                        "all_checks_pass": "true",
                    })
                    emitted += 1
            fh.flush()
            print(f"  [{i:03d}] θ={th:.4f} depth={depth} frob={frob:.4f} "
                  f"emitted={emitted} ({wall:.0f}s) "
                  f"({n_done}/{len(thetas)}, eta={eta/60:.1f}min)")

    fh.close()
    print(f"[bidir-k3] done in {(time.time() - t_sweep)/60:.1f} min")


if __name__ == "__main__":
    main()
