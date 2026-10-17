"""exact_cert.py -- rounding-free certificate that no T-count <= 6 circuit on the system and two clean
ancillas implements a target (review C18).  Usage: python3 exact_cert.py TARGET[,TARGET...] NPROC

States are isometries M = N / 3^d, N exact over Z[zeta_9] (exact_ring), built from the orbit
representatives stored by orbits3.py (out/fwd_3.pkl, out/bwd_<T>_3.pkl).  The float search is used only
to *propose*; every decision below is exact:

 1. Completeness up to depth 2.  For every representative R of depth <= 1 and every rotation f(P), the child
    f(P) R is shown equivalent to a representative (C child G = lambda * R', checked exactly over Z[zeta_9]
    with C, G exact Cliffords), or is added as a new representative.  Since the orbit of f(P) J depends only
    on the orbit of J, every depth-<=2 state is then equivalent to a representative.
 2. Depth 3 is not quotiented: all children of all depth-<=2 representatives are kept.
 3. Meet test.  key(M) is a digest of the label multisets {|Tr(A^dag M^dag P M)|^2} (per P over the 9
    one-qutrit Paulis A, and per A over the 729 Paulis P), evaluated through two reductions modulo primes
    (exact_ring.Red).  It is invariant under M -> lambda C M G, so a forward and a backward state with
    different keys are inequivalent.  For every pair with equal keys we either verify an equivalence exactly
    (a circuit) or refute it with the finer invariant: for each of the 216 one-qutrit Cliffords G, the
    multiset over P of the omega-classes of G^dag Y_P G, Y_P = M^dag P M, differs from the backward one.
"""
import hashlib, pickle, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
import exact_ring as er
import c3core as cc

OUT = '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla/out'
Z = np.exp(2j * np.pi / 9)
REDS = [er.Red(p) for p in er.primes_1mod9(2)]
A_ = er.LABS[:, 0::2]; B_ = er.LABS[:, 1::2]
SRC = cc.SRC                                     # (P M)[y] = w^{EXP[P,y]} M[SRC[P,y]]
EXP = np.array([((er.DIG - A_[k]) % 3 * B_[k]).sum(1) % 3 for k in range(729)])
_G = {}


def tofloat(N, d):
    return (N * Z ** np.arange(6)).sum(-1) / 3 ** d


# ---------------- exact Cliffords from float witnesses ----------------
def cliff_exact(U):
    """A Clifford unitary (float, any global phase) -> exact monomial-entry array proportional to it."""
    nz = np.abs(U) > 1e-6
    r = np.abs(U[nz]); assert np.allclose(r, r[0], atol=1e-8), "non-flat Clifford"
    V = U / U[nz][0]
    out = np.zeros(U.shape + (6,), np.int64)
    for idx in zip(*np.nonzero(nz)):
        ang = np.angle(V[idx]) / (2 * np.pi / 18)
        k = int(np.rint(ang)); assert abs(ang - k) < 1e-6, "phase not an 18th root of unity"
        out[idx] = er.mono(k // 2) if k % 2 == 0 else er.mono((k + 9) // 2, -1)   # zeta_18^k
    assert np.allclose(tofloat(out, 0), V, atol=1e-9)
    return out


def proportional(X, Y):
    """Exact test X = lambda Y for a scalar lambda (both nonzero)."""
    i = tuple(np.argwhere(np.any(Y != 0, -1))[0])
    lhs = er.rmatmul(X.reshape(-1, 1, 6), Y[i].reshape(1, 1, 6)).reshape(X.shape)
    rhs = er.rmatmul(Y.reshape(-1, 1, 6), X[i].reshape(1, 1, 6)).reshape(Y.shape)
    return np.any(X != 0) and np.array_equal(lhs, rhs)


def witness(N1, d1, N2, d2):
    """Exact proof that N1/3^d1 ~ N2/3^d2 (C M1 G = lambda M2), or None."""
    r = cc.equiv(tofloat(N1, d1), tofloat(N2, d2))
    if r is None:
        return None
    C, G, _ = r
    X = er.rmatmul(er.rmatmul(cliff_exact(C), N1), cliff_exact(G))
    return True if proportional(X, N2) else None


# ---------------- exact (mod-p) invariants ----------------
def _pm(F, red, minus):
    """(729, 27, 3) images of P M under one reduction."""
    g3 = pow(red.g, -3 if minus else 3, red.p)
    ph = np.array([1, g3, g3 * g3 % red.p], np.int64)[EXP]          # (729, 27)
    return F[SRC] * ph[:, :, None] % red.p


def _Y(N, d, red):
    """Y_P images under phi_g and phi_g^-1: (2, 729, 3, 3)."""
    s = pow(red.inv3, d, red.p)
    Fp = red(N) * s % red.p; Fm = red(N, True) * s % red.p
    Yp = np.einsum('ya,pyb->pab', Fm, _pm(Fp, red, False)) % red.p
    Ym = np.einsum('ya,pyb->pab', Fp, _pm(Fm, red, True)) % red.p
    return Yp, Ym


def key(N, d):
    h = hashlib.sha256()
    for red in REDS:
        Yp, Ym = _Y(N, d, red)
        p = red.p
        # c[P, A] = Tr(A^dag Y_P), A = X^a Z^b: A_{xy} = w^{b y}[x = y + a]
        cp = np.zeros((729, 9), np.int64); cm = np.zeros((729, 9), np.int64)
        for ai in range(9):
            a, b = divmod(ai, 3)
            for y in range(3):
                x = (y + a) % 3
                cp[:, ai] += Yp[:, x, y] * pow(red.g, (-3 * b * y) % 9, p)
                cm[:, ai] += Ym[:, x, y] * pow(red.g, (3 * b * y) % 9, p)
        L = cp % p * (cm % p) % p
        h.update(np.sort(np.sort(L, 1).view([('', np.int64)] * 9).ravel()).tobytes())
        h.update(np.sort(np.sort(L, 0).T.copy().view([('', np.int64)] * 729).ravel()).tobytes())
    return h.hexdigest()


# ---------------- finer invariant for equal-key states ----------------
def _cl1_exact():
    return [cliff_exact(G) for G in cc.CL1]


RW = np.random.default_rng(12345).integers(1, (1 << 31) - 1, size=9)       # fixed row-hash weights
M61 = (1 << 61) - 1


def fine_key(N, d, CL1X):
    """min over the 216 one-qutrit Cliffords G of a digest of the multiset over P of the omega-classes of
    G^dag Y_P G (Y_P = M^dag P M), one reduction.  Invariant under M -> lambda C M G0, and a function of the
    exact values only (fixed hash weights, min over the three omega multiples)."""
    red = REDS[0]; p = red.p
    s = pow(red.inv3, d, p)
    g3 = pow(red.g, 3, p)
    best = None
    for Gx in CL1X:
        M = er.rmatmul(N, Gx)
        nrm = er.rmatmul(er.conj(Gx).transpose(1, 0, 2), Gx)[0, 0]
        inv = pow(int(red(nrm.reshape(1, 6))[0]), -1, p)
        F = red(M) * s % p; Fc = red(M, True) * s % p
        Y = (np.einsum('ya,pyb->pab', Fc, _pm(F, red, False)) % p * inv % p).reshape(729, 9)
        hs = np.stack([(Y * pow(g3, k, p) % p * RW % M61).sum(1) % M61 for k in range(3)])
        h = hashlib.sha256(np.sort(hs.min(0)).tobytes()).hexdigest()
        best = h if best is None or h < best else best
    return best


def _fine_job(args):
    N, d = args
    return fine_key(N, d, _G['cl1'])


# ---------------- driver ----------------
def _child_job(args):
    N, d, r = args
    return er.rmatmul(_G['rot'][r], N), d + 1


def _key_job(args):
    N, d = args
    return key(N, d)


def _wit_job(args):
    Nc, dc, cands = args
    for j, (Nr, dr) in cands:
        if witness(Nc, dc, Nr, dr):
            return j
    return -1


def build_side(J0, label, pool):
    """Representatives of depth <= 2 grown from J0 with exact witnesses, then every depth-3 child."""
    reps = [(J0, 0)]; keys = [key(J0, 0)]
    t0 = time.time(); nw = 0
    i = 0
    while i < len(reps):
        if reps[i][1] <= 1:
            kids = pool.map(_child_job, [reps[i] + (r,) for r in range(728)], chunksize=16)
            kk = pool.map(_key_job, kids, chunksize=8)
            # first pass in parallel against the current representatives
            res = pool.map(_wit_job, [(Nc, dc, [(j, reps[j]) for j, kj in enumerate(keys) if kj == kc])
                                      for (Nc, dc), kc in zip(kids, kk)], chunksize=4)
            for (Nc, dc), kc, j in zip(kids, kk, res):
                if j >= 0:
                    nw += 1; continue
                # sequentially against representatives added in this pass
                if _wit_job((Nc, dc, [(j2, reps[j2]) for j2 in range(len(reps)) if keys[j2] == kc])) >= 0:
                    nw += 1; continue
                reps.append((Nc, dc)); keys.append(kc)
        i += 1
    nd = [sum(1 for r in reps if r[1] == d) for d in range(3)]
    print(f"[{label}] representatives by depth {nd}; {nw} children matched by exact witnesses  [{time.time()-t0:.0f}s]",
          flush=True)
    allst = list(reps)
    for i, (N, d) in enumerate(reps):
        if d == 2:
            allst += pool.map(_child_job, [reps[i] + (r,) for r in range(728)], chunksize=16)
    allk = keys + pool.map(_key_job, allst[len(reps):], chunksize=8)
    print(f"[{label}] {len(allst)} states of depth <= 3, {len(set(allk))} distinct keys  [{time.time()-t0:.0f}s]", flush=True)
    return allst, allk


if __name__ == "__main__":
    names = sys.argv[1].split(','); NP = int(sys.argv[2])
    _G['rot'] = np.array([er.rot3(k) for k in range(1, 729)])
    TGT = {}
    for name, dg in (("L4k1", [er.mono(0), er.mono(0), er.mono(1)]), ("R", [er.mono(0), er.mono(0), er.mono(0, -1)]),
                     ("T", [er.mono(0), er.mono(1), er.mono(-1)]), ("T2", [er.mono(0), er.mono(2), er.mono(-2)]),
                     ("TL4", [er.mono(0), er.mono(1), er.mono(0)])):
        Tm = np.zeros((3, 3, 6), np.int64)
        for x in range(3):
            Tm[x, x] = dg[x]
        TGT[name] = Tm
    CL1X = _cl1_exact(); _G['cl1'] = CL1X
    with Pool(NP) as pool:
        Fst, Fk = build_side(er.E00.copy(), "fwd", pool)
        fidx = {}
        for i, k in enumerate(Fk):
            fidx.setdefault(k, []).append(i)
        for name in names:
            J = er.rmatmul(er.E00, TGT[name])
            Bst, Bk = build_side(J, f"bwd {name}", pool)
            shared = set(Fk) & set(Bk)
            fi = [i for i, k in enumerate(Fk) if k in shared]; bi = [j for j, k in enumerate(Bk) if k in shared]
            ff = pool.map(_fine_job, [Fst[i] for i in fi], chunksize=4)
            bf = pool.map(_fine_job, [Bst[j] for j in bi], chunksize=4)
            fmap = {}
            for i, h in zip(fi, ff):
                fmap.setdefault(h, i)
            both = {h: (fmap[h], j) for j, h in zip(bi, bf) if h in fmap}
            meets = 0; open_ = []
            for h, (i, j) in both.items():
                if witness(Fst[i][0], Fst[i][1], Bst[j][0], Bst[j][1]):
                    meets += 1
                else:
                    open_.append(h)
            print(f"== {name}: {len(shared)} shared coarse keys ({len(fi)} fwd, {len(bi)} bwd states); "
                  f"{len(set(ff))} fwd / {len(set(bf))} bwd fine keys, {len(both)} shared; exact meets {meets}; "
                  f"unresolved {len(open_)}", flush=True)
