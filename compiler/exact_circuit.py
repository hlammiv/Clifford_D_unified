"""exact_circuit.py -- exact validation of emitted Clifford+T circuits (review C24).

Every gate of the emitted gate set has entries in Q(zeta_9): Paulis, S and the zeta-diagonal Cliffords
are monomials in zeta_18 = -zeta_9^5, and H = [w^{jk}]/sqrt(-3) with sqrt(-3) = 1 + 2w (w = zeta_9^3),
so 1/sqrt(-3) = -(1 + 2w)/3.  A matrix is held as (N, d) = N / 3^d with N over Z[zeta_9] (Python ints),
and nothing is rounded or normalised up to phase.  For a circuit W and the exact approximant V we check

    W (I (x) |0>) = c (V (x) |0>)     exactly, and identify c as a root of unity,

i.e. the ancilla returns to |0> exactly and the implementation error is zero; c is the circuit's global
phase.  The merged circuit (depth_optimality/tmerge.py rule) is rebuilt exactly as B * prod_k M_k, each M_k
a product of rotations about dependent axes, checked equal to W, and each M_k is classified exactly from
its eigenvalues on the eigenspaces of its axis (Clifford or T-type), which gives the merged T-count.
Measured gadgets are checked exactly for every outcome.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

U_ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(U_ / "compiler"), str(U_ / "depth_optimality"), str(U_ / "two_ancilla"), str(U_ / "prescribed_best"),
                str(U_ / "symmetry_variants"), str(U_ / "decomp_speed"), str(U_ / "r_count_study"), str(U_ / "nick_test"),
                str(U_), str(U_ / "hrsa")]
import exact_ring as er          # noqa: E402

Z18 = np.exp(1j * np.pi / 9)


def mono(k, c=1):
    return er.mono(k, c).astype(object)


SQ3INV = (mono(0) + 2 * mono(3)) * -1          # -(1 + 2w); divide by 3 separately


# ---------------------------------------------------------------- exact objects
class Q:
    """N / 3^d, N an object array (..., 6) over Z[zeta_9]."""
    def __init__(self, N, d=0):
        self.N, self.d = N, d
        self._reduce()

    def _reduce(self):
        while self.d > 0 and all(int(x) % 3 == 0 for x in self.N.ravel()):
            self.N = self.N // 3; self.d -= 1

    def __matmul__(self, o):
        return Q(rmatmul(self.N, o.N), self.d + o.d)

    def dag(self):
        return Q(conj(self.N).transpose(1, 0, 2), self.d)

    def __eq__(self, o):
        d = max(self.d, o.d)
        return np.array_equal(self.N * 3 ** (d - self.d), o.N * 3 ** (d - o.d))


def _fold(c):
    out = c[..., :6].copy()
    for k in range(6, 11):
        out = out + c[..., k:k + 1] * mono(k)
    return out


def rmatmul(A, B):
    n, m, _ = A.shape; l = B.shape[1]
    c = np.zeros((n, l, 11), dtype=object)
    for x in range(6):
        for y in range(6):
            c[:, :, x + y] = c[:, :, x + y] + A[:, :, x].dot(B[:, :, y])
    return _fold(c)


def conj(A):
    out = np.zeros_like(A)
    for k in range(6):
        out = out + A[..., k:k + 1] * mono(-k)
    return out


def scal(A, k, sign=1):
    """A * sign * zeta_9^k (elementwise)."""
    out = np.zeros_like(A)
    for j in range(6):
        out = out + A[..., j:j + 1] * mono(j + k, sign)
    return out


def exact_matrix(U):
    """Float gate matrix -> exact Q.  Entries must be 0, zeta_18^k, or zeta_18^k / sqrt(3)."""
    n = U.shape[0]
    N = np.zeros((n, n, 6), dtype=object); d = 0
    mags = np.abs(U[np.abs(U) > 1e-9])
    if np.allclose(mags, 1):
        scaled, isH = U, False
    elif np.allclose(mags, 3 ** -0.5):
        scaled, isH = U * np.sqrt(-3 + 0j), True       # = U * i sqrt 3: entries zeta_18^k
    else:
        raise ValueError("unsupported gate entries")
    for i in range(n):
        for j in range(n):
            x = scaled[i, j]
            if abs(x) < 1e-9:
                continue
            k = int(np.rint(np.angle(x) / (np.pi / 9))) % 18
            assert abs(x - Z18 ** k) < 1e-9
            N[i, j] = mono(k // 2) if k % 2 == 0 else mono((k + 9) // 2, -1)
    if isH:                                             # U = scaled / sqrt(-3) = scaled * (-(1+2w)) / 3
        N = rmatmul(N, np.broadcast_to(SQ3INV, (n, n, 6)) * np.eye(n, dtype=object)[:, :, None]); d = 1
    out = Q(N, d)
    assert np.allclose(to_complex(out), U, atol=1e-9)
    return out


def to_complex(q):
    return (q.N.astype(complex) * np.exp(2j * np.pi * np.arange(6) / 9)).sum(-1) / 3 ** q.d


def identity(n):
    N = np.zeros((n, n, 6), dtype=object)
    for i in range(n):
        N[i, i] = mono(0)
    return Q(N)


def kron(A, B):
    n, m = A.N.shape[0], B.N.shape[0]
    N = np.zeros((n * m, n * m, 6), dtype=object)
    for i in range(n):
        for j in range(n):
            if any(A.N[i, j]):
                N[i * m:(i + 1) * m, j * m:(j + 1) * m] = _elmul(A.N[i, j], B.N)
    return Q(N, A.d + B.d)


def _elmul(a, B):
    c = np.zeros(B.shape[:-1] + (11,), dtype=object)
    for x in range(6):
        if a[x]:
            c[..., x:x + 6] = c[..., x:x + 6] + a[x] * B
    return _fold(c)


_I3 = identity(3)
SWAP = np.zeros((9, 9))
for _x in range(3):
    for _y in range(3):
        SWAP[3 * _y + _x, 3 * _x + _y] = 1
_SW = None


def embed(g):
    """Gate (depth_optimality.tdepth.Gate) on qutrits {0: system, else: ancilla} -> exact 9x9."""
    global _SW
    if _SW is None:
        _SW = exact_matrix(SWAP)
    A = exact_matrix(g.U)
    qs = tuple(0 if q == 0 else 1 for q in g.qubits)
    if len(qs) == 1:
        return kron(A, _I3) if qs[0] == 0 else kron(_I3, A)
    return A if qs == (0, 1) else _SW @ A @ _SW


def unit_ratio(X, Y):
    """If X = c Y exactly for c = +-zeta_9^k, return (sign, k); else None."""
    for k in range(9):
        for s in (1, -1):
            if Q(scal(Y.N, k, s), Y.d) == X:
                return s, k
    return None


# ---------------------------------------------------------------- circuit checks
def run_unitary(gates):
    W = identity(9)
    for g in gates:
        W = embed(g) @ W
    return W


def check_unitary(gates, V):
    """Exact: W (I (x) |0>) = c V (x) |0>; returns (sign, k) for c = sign * zeta_9^k, or None."""
    W = run_unitary(gates)
    E0 = np.zeros((9, 3, 6), dtype=object)
    for x in range(3):
        E0[3 * x, x] = mono(0)
    lhs = W @ Q(E0)
    rhs = Q(rmatmul(E0, V.N), V.d)
    return unit_ratio(lhs, rhs), W


def _proj(P):
    """Eigenprojectors of a Pauli P (exact): Pi_k = sum_s w^{-ks} P^s / 3."""
    P2 = P @ P; I = identity(9)
    out = []
    for k in range(3):
        acc = [I, Q(scal(P.N, -3 * k), P.d), Q(scal(P2.N, -6 * k), P2.d)]
        dmax = max(a.d for a in acc)
        tot = sum(a.N * 3 ** (dmax - a.d) for a in acc)
        out.append(Q(tot, dmax + 1))
    return out


def merged_exact(gates, W):
    """tmerge.merge with exact matrices.  Returns (T_merged, rebuilt == W)."""
    from tdepth import local_symplectic_of_dagger, symp, _dependent
    gs = [g for g in gates if g.kind in ("C", "T")]
    n = 2
    Nsym = np.eye(4, dtype=np.int64)
    B = identity(9)
    Zq = {q: embed(type(gs[0])("C", (q,), np.diag([1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)])))
          for q in (0, 1)}
    els = []
    for g in gs:
        Gm = embed(g)
        if g.kind == "C":
            loc = list(g.qubits); loc = [0 if q == 0 else 1 for q in loc]
            S = local_symplectic_of_dagger(g.U)
            cols = loc + [n + i for i in loc]
            Nsym[:, cols] = (Nsym[:, cols] @ S) % 3
            B = Gm @ B
            continue
        q = 0 if g.qubits[0] == 0 else 1
        Bd = B.dag()
        R = Bd @ Gm @ B
        P = Bd @ Zq[q] @ B
        ax = Nsym[:, n + q].copy()
        k = len(els) - 1; placed = False
        while k >= 0:
            if _dependent(els[k][2], ax):
                els[k][0] = R @ els[k][0]; placed = True; break
            if symp(els[k][2], ax, n):
                break
            k -= 1
        if not placed:
            els.append([R, P, ax])
    prod = identity(9)
    for M, _, _ in els:
        prod = M @ prod
    same = (B @ prod) == W
    tc = 0
    for M, P, _ in els:
        Pis = _proj(P)
        exps = []
        for Pi in Pis:
            MP = M @ Pi
            r = unit_ratio(MP, Pi)
            assert r is not None, "merged element is not a phase on an eigenspace"
            exps.append(r)
        # relative phases: Clifford iff all zeta-exponents agree mod 3 with equal signs
        s0, k0 = exps[0]
        rel = [((k - k0) % 9, s * s0) for s, k in exps]
        assert all(sg == 1 for _, sg in rel), "sign phase in merged element"
        if any(k % 3 for k, _ in rel):
            assert sum(k for k, _ in rel) % 3 == 0, "merged element is not T-type"
            tc += 1
    return tc, same


def check_measured_gadget(path, sign=1):
    """All outcomes of a 4-T measured gadget (system 0, ancilla 1): exact.  Returns list of (m, ok, phase)."""
    from tdepth import parse
    d = json.load(open(path))
    gl = list(d["gates"])
    if sign == -1:
        gl[-1] = "Tdg1"
    W = run_unitary([parse(x) for x in gl])
    out = []
    tgt = [mono(0), mono(0), mono(0)]
    for i, e in enumerate(d["target_exponents"]):
        tgt[i] = mono(0, -1) if e == "neg" else mono(sign * e)
    for m in range(3):
        E0 = np.zeros((9, 3, 6), dtype=object)
        for x in range(3):
            E0[3 * x, x] = mono(0)
        st = W @ Q(E0)                                                   # 9 x 3
        if d.get("measure_basis", "z").startswith("f"):                          # <f_m| = sum_a w^{-ma} <a| / sqrt3
            br = np.zeros((3, 9, 6), dtype=object)
            for x in range(3):
                for a in range(3):
                    br[x, 3 * x + a] = mono(-3 * m * a)
            proj = Q(br) @ st
        else:
            br = np.zeros((3, 9, 6), dtype=object)
            for x in range(3):
                br[x, 3 * x + m] = mono(0)
            proj = Q(br) @ st
        ce = d["correction_exponents"][str(m)]
        Cm = np.zeros((3, 3, 6), dtype=object)
        for x in range(3):
            Cm[x, x] = mono(3 * ce[x])
        res = Q(Cm) @ proj
        Tg = np.zeros((3, 3, 6), dtype=object)
        for x in range(3):
            Tg[x, x] = tgt[x]
        out.append((m, unit_ratio_scaled(res, Q(Tg))))
    return out


def unit_ratio_scaled(X, Y):
    """X = c Y with |c| fixed by the measurement normalisation: return c up to the positive scale, as (sign,k)."""
    # compare X against Y times +-zeta^k times any power of sqrt(3) scale: use proportionality, then the phase
    i = tuple(np.argwhere(np.array([[any(Y.N[a, b]) for b in range(Y.N.shape[1])] for a in range(Y.N.shape[0])]))[0])
    lhs = rmatmul(X.N, np.broadcast_to(Y.N[i], (X.N.shape[1], X.N.shape[1], 6)) * np.eye(X.N.shape[1], dtype=object)[:, :, None])
    rhs = rmatmul(Y.N, np.broadcast_to(X.N[i], (Y.N.shape[1], Y.N.shape[1], 6)) * np.eye(Y.N.shape[1], dtype=object)[:, :, None])
    return bool(np.array_equal(lhs, rhs)) and any(any(x) for x in X.N.reshape(-1, 6))


def ring_to_Q(V):
    """canonical_reducer ring matrix -> Q."""
    d = max(V[i][j].denom_pow3 for i in range(3) for j in range(3))
    N = np.zeros((3, 3, 6), dtype=object)
    for i in range(3):
        for j in range(3):
            c = [int(x) for x in V[i][j].num.coefs]
            N[i, j] = np.array(c + [0] * (6 - len(c)), dtype=object) * 3 ** (d - V[i][j].denom_pow3)
    return Q(N, d)


def branch_exact(gates, V):
    """One measured branch (outcomes drawn by the emitter), system + one re-prepared ancilla: X = c V (x)|0>?"""
    E0 = np.zeros((9, 3, 6), dtype=object)
    for x in range(3):
        E0[3 * x, x] = mono(0)
    st = Q(E0)
    for g in gates:
        if g.kind == "F":
            continue
        if g.kind == "M":
            br = np.zeros((3, 9, 6), dtype=object)
            for x in range(3):
                for a in range(3):
                    if g.basis == "F":
                        br[x, 3 * x + a] = mono(-3 * g.m * a)
                    elif a == g.m:
                        br[x, 3 * x + a] = mono(0)
            v = Q(br) @ st
            st = Q(rmatmul(E0, v.N), v.d)
            continue
        st = embed(g) @ st
    return unit_ratio_scaled(st, Q(rmatmul(E0, V.N), V.d))


def validate(item):
    """item = (pool jsonl path, seed).  Best-eps candidate of one angle: exact unitary, merged and one measured branch."""
    import canonical_reducer as cr
    import fast_decompose
    from analyze_pools import rebuild
    from emit_variants import emit
    from tmerge import merge_symbolic, merge
    path, seed = item
    fast_decompose.install()
    cands = sorted((json.loads(l) for l in open(path)), key=lambda c: c["eps"])
    c = cands[0]
    Vr, eps = rebuild(c, c["theta"])
    r = cr.decompose_canonical(Vr)
    V = ring_to_Q(Vr)
    rng = np.random.default_rng(seed)
    gates, cnt = emit(r["syllables"], r["trailing_clifford"], "shared", rng)
    ph, W = check_unitary(gates, V)
    tm_exact, same = merged_exact(gates, W)
    fl = merge(gates)
    mg, _ = emit(r["syllables"], r["trailing_clifford"], "meas", rng)
    br = branch_exact(mg, V)
    return dict(path=Path(path).name, f=c["f"], eps=eps, n_gates=len(gates), T_unit=cnt["n_T3"] + 7 * (cnt["n_L4"] + cnt["n_R"]),
                phase=ph, T_merged_exact=tm_exact, merged_equal=same, T_merged_symbolic=merge_symbolic(gates),
                float_merge_err=fl["merge_err"], T_merged_float=fl["T_merged"], meas_branch_ok=br, n_meas_gates=len(mg))


if __name__ == "__main__":
    from multiprocessing import Pool
    import glob
    PB = U_ / "prescribed_best"
    items = []
    for lvl, n in (("pfp_l16", 350), ("pfp_l14", 50), ("pfp_l12", 50)):
        fs = sorted(glob.glob(str(PB / lvl / "pool_t*.jsonl")))
        items += [(fs[i], i) for i in sorted(set(np.linspace(0, len(fs) - 1, min(n, len(fs))).astype(int)))]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for row in pool.imap_unordered(validate, items):
            print(json.dumps(row, default=str), flush=True)
