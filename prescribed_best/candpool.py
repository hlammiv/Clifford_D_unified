"""candpool.py — parse HRSA_tester CANDDUMP lines into candidate pools with exact ε.
CANDDUMP s D x1re x1im x2re x2im x3re x3im f a1_0..a1_5 a2_0..a2_5 a3_0..a3_5
V = P01 (I − u u†), u_i = a_i / 3^{f/2-ish}; the integer numerators give u exactly (see
r_count_study/analyze_candidates.build_V); ε = ‖V − R_z(θ)‖_F."""
import numpy as np
P01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)

def parse(stdout):
    out = []
    for line in stdout.splitlines():
        if not line.startswith("CANDDUMP"):
            continue
        p = line.split()
        x = [complex(float(p[3]), float(p[4])), complex(float(p[5]), float(p[6])), complex(float(p[7]), float(p[8]))]
        f = int(p[9])
        nums = [list(map(int, p[10 + 6 * k: 16 + 6 * k])) for k in range(3)]
        exps = [int(e) for e in p[p.index("E") + 1: p.index("E") + 4]] if "E" in p else None
        out.append(dict(s=int(p[1]), D=int(p[2]), x=x, f=f, nums=nums, exps=exps))
    return out

def eps_of(c, theta):
    u = np.array(c["x"])
    V = P01 @ (np.eye(3) - np.outer(u, u.conj()))
    T = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta), 1])
    return float(np.linalg.norm(V - T))
