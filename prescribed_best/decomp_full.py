"""decomp_full.py — complete exact decomposition when the canonical reducer stalls.

canonical_reducer.decompose_canonical peels syllables until sde_χ(V[0][0]) = 0 and then
expects the residual to be a monomial (Clifford-like). For some matrices (often near-Clifford
targets whose approximant acts on a 2×2 block) column 0 becomes a unit vector while the other
columns still carry denominators, so the reducer reports
    "residual V at sde_chi=0 is not monomial".
(The C++ decompose_tool has the same limitation.)

Fix (wrapper; reducer untouched): right-multiply the residual by a column permutation σ
(a Clifford) that brings a non-reduced column to position 0 and peel again:
    V = M1† · R1,   R1·σ = M2† · R2   =>   V = (M2 M1)† · R2 σ⁻¹
so the syllable lists concatenate and the trailing Clifford becomes R2·σ⁻¹ (still monomial).
Repeats up to 3 rounds. Every result should be checked by compiler/cd_to_ct.verify_circuit.
"""
import canonical_reducer as cr

_ONE = cr._FastZ9Frac.one
_ZERO = cr._FastZ9Frac.zero


def _perm_matrix(p):
    """Matrix with ones at (p[j], j): columns permuted, V·P moves column p[j] to j."""
    return [[_ONE() if i == p[j] else _ZERO() for j in range(3)] for i in range(3)]


def _mul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), _ZERO()) for j in range(3)] for i in range(3)]


def _is_monomial(M):
    return all(sum(1 for j in range(3) if not M[i][j].is_zero()) == 1 for i in range(3))


def decompose_full(V, max_rounds=3):
    r = cr.decompose_canonical(V)
    if r["success"]:
        return r
    syl = []
    perm_total = [0, 1, 2]
    for _ in range(max_rounds):
        err = r.get("error", "")
        if "not monomial" not in err:
            return r                                    # a different failure: give up
        syl += r["syllables"]
        R = r["trailing_clifford"]
        # choose a column that is not yet a unit vector
        cols = [j for j in range(3) if sum(1 for i in range(3) if not R[i][j].is_zero()) > 1]
        if not cols:
            break
        j = cols[0]
        p = [j] + [k for k in range(3) if k != j]       # bring column j to position 0
        P = _perm_matrix(p)
        perm_total = [perm_total[k] for k in p]
        r = cr.decompose_canonical(_mul(R, P))
        if r["success"]:
            break
    if not r["success"]:
        return r
    # trailing = R_final · σ_total⁻¹  (σ⁻¹ = σᵀ for a permutation matrix)
    Ptot = _perm_matrix(perm_total)
    PinvT = [[Ptot[j][i] for j in range(3)] for i in range(3)]
    trailing = _mul(r["trailing_clifford"], PinvT)
    if not _is_monomial(trailing):
        return dict(success=False, error="combined trailing not monomial")
    _, md, _ = cr.classify_monomial_and_d_cost(trailing)
    d_count = sum(cr.count_d_from_syllable(s["a0"], s["a1"], s["a2"], s["eps"]) for s in syl + r["syllables"]) + md
    return dict(success=True, syllables=syl + r["syllables"], trailing_clifford=trailing,
                D_count=d_count, rounds=len(perm_total), note="column-permutation completion")
