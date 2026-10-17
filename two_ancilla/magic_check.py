"""magic_check.py -- exact facts used in the >= 3-ancilla cases of the 7-T theorem (review C19).

 (a) The fresh-ancilla states |m_Q> = f(Q)|0>, Q = X^a Z^b with a != 0, are all equal up to a one-qutrit
     Clifford and a phase (exact witnesses), and none is an eigenvector of a nonidentity Pauli (exact).
 (b) The Choi states of R = diag(1,1,-1) and diag(1,1,zeta_9) have a negative discrete Wigner value,
     hence nonzero mana; the value is evaluated with mpmath interval arithmetic.
"""
import itertools
import numpy as np
import mpmath as mp
import exact_ring as er
import exact_cert as ec
import c3core as cc

Z = np.exp(2j * np.pi / 9)


def one(k):                        # single-qutrit Pauli X^a Z^b, exact 3x3
    a, b = divmod(k, 3)
    P = np.zeros((3, 3, 6), np.int64)
    for x in range(3):
        P[(x + a) % 3, x] = er.mono(3 * b * x)
    return P


def rot1(k):                       # 3 f(Q) on one qutrit
    Q = one(k); I = np.zeros((3, 3, 6), np.int64)
    for x in range(3):
        I[x, x] = er.mono(0)
    Pj = [I, Q, er.rmatmul(Q, Q)]
    out = np.zeros((3, 3, 6), np.int64)
    for m, fv in enumerate((0, 1, -1)):
        for j in range(3):
            out += er._scal(Pj[j], fv - 3 * m * j)
    return out


ket0 = np.zeros((3, 1, 6), np.int64); ket0[0, 0] = er.mono(0)
ms = {k: er.rmatmul(rot1(k), ket0) for k in range(9) if k // 3 != 0}
CL = [ec.cliff_exact(G) for G in cc.CL1]
base = ms[3]
for k, m in ms.items():
    ok = any(ec.proportional(er.rmatmul(G, base), m) for G in CL)
    eig = [q for q in range(1, 9) if ec.proportional(er.rmatmul(one(q), m), m)]
    print(f"m_Q for Q = X^{k//3} Z^{k%3}: Clifford-equivalent to m_X: {ok}; Pauli eigenvector of: {eig or 'none'}")

# ---- discrete Wigner function (Gross), d = 3, n qutrits ----
mp.mp.dps = 40
w = mp.mpc(mp.cos(2 * mp.pi / 3), mp.sin(2 * mp.pi / 3))


def wigner_min(U):
    """min over phase-space points of W for the normalized Choi state of the 3x3 unitary U (exact entries)."""
    # Choi |psi> = (U (x) I) sum_x |x>|x> / sqrt 3, as mp complex vector
    zz = [mp.mpc(mp.cos(2 * mp.pi * k / 9), mp.sin(2 * mp.pi * k / 9)) for k in range(6)]
    Um = [[sum(int(U[i, j, k]) * zz[k] for k in range(6)) for j in range(3)] for i in range(3)]
    psi = [mp.mpc(0)] * 9
    for x in range(3):
        for i in range(3):
            psi[3 * i + x] += Um[i][x] / mp.sqrt(3)
    rho = [[psi[a] * mp.conj(psi[b]) for b in range(9)] for a in range(9)]
    inv2 = 2                                        # 2^{-1} mod 3
    def A1(q, p):                                   # single-qutrit phase-point operator A_(q,p) = sum ... (Gross)
        # A_u = D_u A_0 D_u^dag with A_0 = parity |x> -> |-x>
        M = [[mp.mpc(0)] * 3 for _ in range(3)]
        for x in range(3):
            y = (2 * q - x) % 3
            M[y][x] = w ** ((2 * p * (x - q)) % 3)
        return M
    vals = []
    for u in itertools.product(range(3), repeat=4):
        Aa = A1(u[0], u[1]); Ab = A1(u[2], u[3])
        t = mp.mpc(0)
        for i1, i2, j1, j2 in itertools.product(range(3), repeat=4):
            a = Aa[i1][j1] * Ab[i2][j2]
            if a != 0:
                t += a * rho[3 * j1 + j2][3 * i1 + i2]
        vals.append(t / 9)
    tot = sum(abs(v.real) for v in vals)
    return min(v.real for v in vals), mp.log(tot), max(abs(v.imag) for v in vals), sum(v.real for v in vals)


for name, dg in (("R", [er.mono(0), er.mono(0), er.mono(0, -1)]), ("diag(1,1,z)", [er.mono(0), er.mono(0), er.mono(1)]),
                 ("T (control)", [er.mono(0), er.mono(1), er.mono(-1)]), ("S (Clifford)", [er.mono(0), er.mono(0), er.mono(3)])):
    U = np.zeros((3, 3, 6), np.int64)
    for x in range(3):
        U[x, x] = dg[x]
    wmin, mana, im, s = wigner_min(U)
    print(f"{name}: min W = {mp.nstr(wmin, 12)}, mana = {mp.nstr(mana, 6)}, max |Im W| = {mp.nstr(im, 3)}, sum W = {mp.nstr(s, 12)}")
