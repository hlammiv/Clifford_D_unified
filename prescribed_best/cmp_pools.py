"""cmp_pools.py NEW_DIR POOL_DIR THETA_FILE — per θ, compare the candidate set (nums, exps) of phase_free raw
output NEW_DIR/t%04d.txt with the pool POOL_DIR/pool_t%04d.jsonl; prints mismatches and a summary."""
import sys, json
sys.path.insert(0, "/home/hlamm/Desktop/efficent_gates/unified/prescribed_best")
import candpool
new, pool, tfile = sys.argv[1:4]
th = [float(x) for x in open(tfile).read().split()]
bad = 0; tot = 0
for i, t in enumerate(th):
    N = candpool.parse(open(f"{new}/t{i:04d}.txt").read())
    kn = {(tuple(map(tuple, c["nums"])), tuple(c["exps"])) for c in N}
    ko = set()
    for line in open(f"{pool}/pool_t{i:04d}.jsonl"):
        c = json.loads(line); ko.add((tuple(map(tuple, c["nums"])), tuple(c["exps"])))
    tot += len(ko)
    if kn != ko or len(kn) != len(N):
        bad += 1; print(f"θ#{i} MISMATCH new {len(N)} distinct {len(kn)} pool {len(ko)} new-pool {len(kn-ko)} pool-new {len(ko-kn)}")
print(f"{len(th)} θ, {tot} pool candidates, {bad} θ mismatched -> " + ("IDENTICAL" if bad == 0 else "DIFF"))
