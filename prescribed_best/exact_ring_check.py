"""Exact check of every phase-free candidate, in Z[zeta_9] with Python integers (no floats):
 (1) sum_i N(x_i) * 9^(E - e_i) == 2 * 9^E as a ring element (all 6 reduced coefficients), E = max e_i;
 (2) V = P01 (I - u u^dagger) is exactly unitary: with M = 3^E u (ring vector), check 9^E*I - ... i.e.
     W = 9^E*I - M M^dagger satisfies W W^dagger == 9^(2E) * I  (P01 is a permutation, irrelevant for unitarity);
     entries of V are then in Z[zeta_9, 1/3] by construction.
Ring: basis 1..zeta^5, reduce with Phi_9: zeta^6 = -zeta^3 - 1."""
import json, glob, sys
from multiprocessing import Pool
def mul(a, b):
    p = [0] * 11
    for i in range(6):
        if a[i]:
            for j in range(6): p[i + j] += a[i] * b[j]
    for k in (10, 9, 8, 7, 6):   # zeta^k = zeta^(k-3) - zeta^(k-6)
        c = p[k]
        if c: p[k - 3] -= c; p[k - 6] -= c; p[k] = 0
    return p[:6]
def conj(a):                      # zeta^k -> zeta^(9-k): 1->1, z->z^8, z^2->z^7, z^3->z^6, z^4->z^5, z^5->z^4
    p = [a[0], 0, 0, 0, a[5], a[4], a[3], a[2], a[1], 0, 0]
    for k in (8, 7, 6):
        c = p[k]
        if c: p[k - 3] -= c; p[k - 6] -= c; p[k] = 0
    return p[:6]
def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]
def scal(a, s): return [x * s for x in a]
ZERO = [0] * 6
def check_file(path):
    bad_norm = bad_unit = n = 0
    for line in open(path):
        c = json.loads(line); n += 1
        ex = c["exps"]; E = max(ex)
        M = [scal(x, 3 ** (E - e)) for x, e in zip(c["nums"], ex)]        # 3^E u, exact
        tot = ZERO
        for m in M: tot = add(tot, mul(m, conj(m)))
        if tot != [2 * 9 ** E, 0, 0, 0, 0, 0]: bad_norm += 1; continue
        # W = 9^E I - M M^dagger ; check W W^dagger = 9^(2E) I
        W = [[sub([9 ** E, 0, 0, 0, 0, 0] if i == j else ZERO, mul(M[i], conj(M[j]))) for j in range(3)] for i in range(3)]
        ok = True
        for i in range(3):
            for j in range(3):
                s = ZERO
                for k in range(3): s = add(s, mul(W[i][k], conj(W[j][k])))
                if s != ([9 ** (2 * E), 0, 0, 0, 0, 0] if i == j else ZERO): ok = False
        if not ok: bad_unit += 1
    return n, bad_norm, bad_unit
if __name__ == "__main__":
    for L in sys.argv[1:]:
        files = sorted(glob.glob(f"pfp_l{L}/pool_t*.jsonl"))
        with Pool(10) as p: res = p.map(check_file, files, chunksize=4)
        print(f"level {L}: {len(files)} θ, {sum(r[0] for r in res)} candidates, norm != 2: {sum(r[1] for r in res)}, not exactly unitary: {sum(r[2] for r in res)}", flush=True)
