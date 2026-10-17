"""Full Householder target P01(I − t t†), t = (c1 e^{iθ/2}, −c2, c3 e^{iη}), c1² + c2² + c3² = 2 (env PF_C1SQ, PF_C3SQ).

See hrsa/phase_free_full.cpp. Targets are "θ η" pairs, one per line.
"""
import os

import numpy as np

P01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)


def moduli():
    s1, s3 = float(os.environ["PF_C1SQ"]), float(os.environ["PF_C3SQ"])
    return s1, 2 - s1 - s3, s3


def target(theta, eta):
    s1, s2, s3 = moduli()
    t = np.array([np.sqrt(s1) * np.exp(0.5j * theta), -np.sqrt(s2), np.sqrt(s3) * np.exp(1j * eta)])
    return P01 @ (np.eye(3) - np.outer(t, t.conj()))


def read_targets(path):
    return [tuple(map(float, l.split())) for l in open(path) if l.strip()]
