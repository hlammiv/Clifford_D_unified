"""Decomposer round-trip on phase-free candidates: canonical_reducer.decompose_canonical must succeed with
verification.matches_input and max_coef_residual == 0. Sample: all candidates of the first N θ per level."""
import sys, json, glob
from multiprocessing import Pool
sys.path[:0] = ["/home/hlamm/Desktop/efficent_gates/unified/hrsa", "/home/hlamm/Desktop/efficent_gates/unified/prescribed_best",
                "/home/hlamm/Desktop/efficent_gates/unified/r_count_study", "/home/hlamm/Desktop/efficent_gates/unified/symmetry_variants",
                "/home/hlamm/Desktop/efficent_gates/unified/decomp_speed", "/home/hlamm/Desktop/efficent_gates/unified/compiler",
                "/home/hlamm/Desktop/efficent_gates/unified/depth_optimality", "/home/hlamm/Desktop/efficent_gates/unified/nick_test",
                "/home/hlamm/Desktop/efficent_gates/unified"]
def work(path):
    import canonical_reducer as cr
    from analyze_candidates import build_V_exps
    n = ok = exact = 0; worst = 0
    for line in open(path):
        c = json.loads(line); n += 1
        V = build_V_exps(*c["nums"], c["exps"])
        r = cr.decompose_canonical(V)
        if r["success"]:
            ok += 1
            v = cr.verify_decomposition(V, r["syllables"], r["trailing_clifford"])
            if v["matches"] and v["max_coef_residual"] == 0: exact += 1
            worst = max(worst, int(v["max_coef_residual"]))
    return n, ok, exact, worst
if __name__ == "__main__":
    N = int(sys.argv[1])
    for L in sys.argv[2:]:
        files = sorted(glob.glob(f"pfp_l{L}/pool_t*.jsonl"))[:N]
        with Pool(8) as p: res = p.map(work, files, chunksize=1)
        print(f"level {L}: {len(files)} θ, {sum(r[0] for r in res)} candidates, decomposed {sum(r[1] for r in res)}, exact round-trip {sum(r[2] for r in res)}, max residual {max(r[3] for r in res)}", flush=True)
