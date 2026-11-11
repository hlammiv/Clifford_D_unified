"""exact_ring.py -- exact arithmetic in Z[zeta_9] and its reductions modulo primes, for the 7-T certificate.

An element of Z[zeta_9] is an int64 array of 6 coefficients in the basis 1, z, ..., z^5 (z = zeta_9),
reduced modulo Phi_9(z) = z^6 + z^3 + 1.  Arrays of elements carry the coefficient axis last.

Reductions.  For a prime p = 1 (mod 9) and g of multiplicative order 9 in F_p, phi_g(x) = x(g) mod p is a
ring homomorphism Z[zeta_9] -> F_p, and phi_{g^-1}(x) = phi_g(conj x).  Every invariant used for a
non-equivalence decision is computed through such maps, so it is a function of the exact value: equal exact
values give equal images, and different images prove different exact values (no rounding anywhere).
"""
import itertools
import numpy as np

D = 27
ZERO = np.zeros(6, np.int64)


def mono(k, c=1):
    """c * z^k as an element."""
    k %= 9
    e = np.zeros(6, np.int64)
    if k < 6:
        e[k] = c
    else:                           # z^6 = -z^3 - 1, z^7 = -z^4 - z, z^8 = -z^5 - z^2
        e[k - 6] -= c; e[k - 3] -= c
    return e


def _fold(c):
    """(..., 11) coefficients of z^0..z^10 -> (..., 6) reduced."""
    out = c[..., :6].copy()
    for k in range(6, 11):
        r = mono(k)
        out += c[..., k:k + 1] * r
    return out


def rmatmul(A, B):
    """A (n,m,6) @ B (m,l,6) -> (n,l,6), exact."""
    t = np.einsum('ikx,kjy->ijxy', A, B)
    c = np.zeros(t.shape[:2] + (11,), np.int64)
    for x in range(6):
        c[..., x:x + 6] += t[:, :, x, :]
    return _fold(c)


def conj(A):
    """Complex conjugation z -> z^-1 = z^8."""
    out = np.zeros_like(A)
    for k in range(6):
        out += A[..., k:k + 1] * mono(-k)
    return out


# ---- Paulis, T-rotations, and the input isometry on 3 qutrits ----
DIG = np.array(list(itertools.product(range(3), repeat=3)))
PW = 3 ** np.arange(3)[::-1]
LABS = np.array(list(itertools.product(range(3), repeat=6)))     # (a1,b1,a2,b2,a3,b3), base-3 index


def pauli(k):
    """Exact 27x27 Pauli X^a Z^b (label index k): P|x> = w^{b.x}|x+a>, w = z^3."""
    lab = LABS[k]; a = lab[0::2]; b = lab[1::2]
    P = np.zeros((D, D, 6), np.int64)
    for xi, xd in enumerate(DIG):
        y = (xd + a) % 3
        P[y @ PW, xi] = mono(3 * int((b * xd).sum() % 3))
    return P


def rot3(k):
    """3 f(P_k): f(P) = sum_m fv[m] Pi_m(P), Pi_m = sum_j w^{-mj} P^j / 3, fv = (1, z, z^-1) (T = diag(1,z,z^-1))."""
    P = pauli(k); Pj = [np.broadcast_to(np.eye(D, dtype=np.int64)[:, :, None] * mono(0), (D, D, 6)).copy(), P,
                        rmatmul(P, P)]
    out = np.zeros((D, D, 6), np.int64)
    for m, fv in enumerate((0, 1, -1)):
        for j in range(3):
            out += _scal(Pj[j], fv - 3 * m * j)
    return out


def _scal(A, k):
    """A * z^k."""
    t = np.zeros(A.shape[:-1] + (11,), np.int64)
    k %= 9
    if k <= 5:
        t[..., k:k + 6] = A
        return _fold(t)
    return _scal(_scal(A, 5), k - 5)


E00 = np.zeros((D, 3, 6), np.int64)
for _x in range(3):
    E00[9 * _x, _x] = mono(0)

# ---- reduction modulo primes ----
def primes_1mod9(n, start=1 << 20):
    out = []; p = start + (1 - start) % 9
    while len(out) < n:
        if all(p % q for q in range(2, int(p ** 0.5) + 1)):
            out.append(p)
        p += 9
    return out


def order9(p):
    for h in range(2, p):
        g = pow(h, (p - 1) // 9, p)
        if g != 1 and pow(g, 3, p) != 1:
            return g


class Red:
    """phi_g and phi_{g^-1} for one prime."""
    def __init__(self, p):
        self.p = p; g = order9(p); self.g = g
        self.gp = np.array([pow(g, k, p) for k in range(6)], np.int64)
        gi = pow(g, -1, p)
        self.gm = np.array([pow(gi, k, p) for k in range(6)], np.int64)
        self.inv3 = pow(3, -1, p)

    def __call__(self, A, minus=False):
        v = self.gm if minus else self.gp
        return (np.mod(A, self.p) * v).sum(-1) % self.p
