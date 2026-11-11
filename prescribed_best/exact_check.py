"""exact_check.py — exact unitarity and high-precision error of a pool candidate (review C39).

V has entries in Z[zeta_9, 1/3], so V V^dag = I is checked exactly in the ring.  The target is in general
transcendental, so eps = ||V - U_target||_F is evaluated from V's integer coefficients in DPS-digit
arithmetic (mpmath); the result is exact to far below any tolerance used downstream.
Targets: Rz(theta) (phase_free), the two-level target (tl_target, PF_C1SQ) and the full Householder
target (full_target, PF_C1SQ, PF_C3SQ); the moduli are read from the same environment strings.
"""
import os

import mpmath as mp

DPS = 40
mp.mp.dps = DPS
_Z = [mp.expjpi(mp.mpf(2 * k) / 9) for k in range(6)]


def exact_unitary(V):
    from canonical_reducer import _FastZ9Frac
    one, zero = _FastZ9Frac.one(), _FastZ9Frac.zero()
    for i in range(3):
        for j in range(i, 3):
            s = zero
            for k in range(3):
                s = s + V[i][k] * V[j][k].conjugate()
            if not s == (one if i == j else zero):
                return False
    return True


def _mp(zf):
    return mp.fsum(int(c) * z for c, z in zip(zf.num.coefs, _Z)) / mp.mpf(3) ** zf.denom_pow3


def _householder(t):
    """P01 (I - t t^dag)."""
    H = mp.eye(3) - mp.matrix([[t[i] * mp.conj(t[j]) for j in range(3)] for i in range(3)])
    return mp.matrix([[H[1, j] for j in range(3)], [H[0, j] for j in range(3)], [H[2, j] for j in range(3)]])


def target_rz(theta):
    h = mp.mpf(theta) / 2
    return mp.diag([mp.expj(-h), mp.expj(h), 1])


def target_tl(theta):
    s = mp.mpf(os.environ.get("PF_C1SQ", "1"))
    return _householder([mp.sqrt(s) * mp.expj(mp.mpf(theta) / 2), -mp.sqrt(2 - s), 0])


def target_full(theta, eta):
    s1, s3 = mp.mpf(os.environ["PF_C1SQ"]), mp.mpf(os.environ["PF_C3SQ"])
    return _householder([mp.sqrt(s1) * mp.expj(mp.mpf(theta) / 2), -mp.sqrt(2 - s1 - s3),
                         mp.sqrt(s3) * mp.expj(mp.mpf(eta))])


def eps_hp(V, T):
    return mp.sqrt(mp.fsum(abs(_mp(V[i][j]) - T[i, j]) ** 2 for i in range(3) for j in range(3)))
