"""Containment: every OLD (phase-fixed) pool candidate with exact eps < new cap must be in the NEW (phase-free) pool.
Match by V = P01(I - uu†), phase-invariant key (u1*conj(u2), |u1|^2, |u3|^2)."""
import json, glob, sys, numpy as np, os
H = "/home/hlamm/Desktop/efficent_gates/unified/prescribed_best"
def key(nums, f, exps):
    Z = np.exp(2j*np.pi*np.arange(6)/9)
    u = [np.dot(n, Z)/3**e for n, e in zip(nums, exps or [f]*3)]
    w = u[0]*np.conj(u[1]); return (round(w.real, 10), round(w.imag, 10), round(abs(u[0])**2, 10), round(abs(u[2])**2, 10))
for L, old, cap in [(4, "pool_l4", 0.012), (6, "pool_l6_500", 4e-4), (8, "pool_l8", 1.5e-5), (10, "pool_l10", 5.5e-7)]:
    nold = nin = miss = nth = 0; worst = []
    for p in sorted(glob.glob(f"{H}/{old}/pool_t*.jsonl")):
        q = f"{H}/pfp_l{L}/" + os.path.basename(p)
        if not os.path.exists(q): continue
        nth += 1
        new = {key(c["nums"], c["f"], c.get("exps")) for c in map(json.loads, open(q))}
        for c in map(json.loads, open(p)):
            if c["eps"] >= cap * 0.999: continue
            nold += 1
            if key(c["nums"], c["f"], c.get("exps")) in new: nin += 1
            else: miss += 1; worst.append((c["eps"], os.path.basename(p)))
    print(f"level {L}: {nth} θ, old candidates below cap {nold}, found in new {nin}, MISSING {miss}", worst[:3], flush=True)
