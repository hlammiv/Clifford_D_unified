"""Two-level Householder target P01(I − t t†), t = (c1 e^{iθ/2}, −c2, 0), c1² + c2² = 2, c1² from env PF_C1SQ.

c1 = c2 = 1 is R_z(θ) = diag(e^{−iθ/2}, e^{iθ/2}, 1), the target of phase_free; see hrsa/phase_free_tl.cpp.
"""
import os

import numpy as np

P01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)


def c1sq():
    return float(os.environ.get("PF_C1SQ", "1"))


def target(theta, s=None):
    s = c1sq() if s is None else s
    t = np.array([np.sqrt(s) * np.exp(0.5j * theta), -np.sqrt(2 - s), 0])
    return P01 @ (np.eye(3) - np.outer(t, t.conj()))
