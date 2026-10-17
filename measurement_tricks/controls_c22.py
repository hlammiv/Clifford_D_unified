"""controls_c22.py -- positive controls for the three measured-gadget searches (review C22).

Targets with known minimal T-count in every measured / adaptive / catalyst model:
  T1 = T                      1 T (and not Clifford)
  T2 = T . H T H^dag          2 T; mana of its Choi state is exactly 2 M(T|+>), so t >= 2 (mana_bounds.py)
  T3 = T . H T H^dag . T      3 T if its mana ratio is > 2 (printed below)
Each search must report its first deterministic solution at exactly the known T-count.
usage: python3 controls_c22.py NPROC
"""
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
sys.argv = [sys.argv[0], 'R', str(NPROC)]
import search1 as s1                                        # noqa: E402  (one ancilla, final measurement)
import search2 as s2                                        # noqa: E402  (two ancillas)
import search_adaptive as sa                                # noqa: E402  (one adaptive mid-circuit measurement)
from meascore import sig                                    # noqa: E402

z = np.exp(2j*np.pi/9); w = z**3
T = np.diag([1, z, z**8]); H = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(3)
TGT = {'T1': T, 'T2': T @ H @ T @ H.conj().T, 'T3': T @ H @ T @ H.conj().T @ T}

src = open(os.path.join(HERE, 'mana_bounds.py')).read().split("for name, G in")[0]
ns = {}; exec(src.replace('print(', 'str('), ns)
for k, G in TGT.items():
    ch = np.array([G[i, x] for i in range(3) for x in range(3)])
    print(f'{k}: mana bound t >= {ns["mana"](ch, 2)/ns["MT"]:.4f}', flush=True)


def run_search1(G):
    s1.G = G; s1.sG = sig(G); s1.sI = sig(np.eye(3))
    from multiprocessing import Pool
    with Pool(NPROC) as pool:
        hits = [h for _, _, out in pool.imap_unordered(s1.job, range(len(s1.REPS))) for h in out]
    det = [h[0] for h in hits if h[1] > 1 - 1e-9]
    return np.bincount(det, minlength=7).tolist()


def run_search2(G):
    sG = sig(G); cnt = [0] * 4
    for i, M in enumerate(s2.fw['M']):
        b, _ = s2.analyse(np.asarray(M), G, sG)
        if b:
            cnt[s2.fw['depth'][i]] += 1
    return cnt


def run_adaptive(G, budget=3):
    """as search_adaptive.__main__, recording the total T-count t1 + max_m t2(m) of each feasible (orbit, Q)."""
    sI, sG = sig(np.eye(3)), sig(G); cnt = {}
    for i, M in enumerate(sa.fw['M']):
        t1 = sa.fw['depth'][i]
        if t1 not in (1, 2):
            continue
        V = np.asarray(M)
        for k in range(1, 729):
            if sa.DBL[k] < k:
                continue
            Mq = V.conj().T @ sa.PM[k] @ V
            if np.abs(Mq - Mq[0, 0]*np.eye(3)).max() > 1e-7:
                continue
            if k not in sa.KCACHE:
                sa.KCACHE[k] = sa.to_z2(k)
            Y = sa.KCACHE[k] @ V; res = []
            for m in range(3):
                Wm = np.array([[Y[9*x0+3*x1+m, j] for j in range(3)] for x0 in range(3) for x1 in range(3)])
                p = np.linalg.norm(Wm)**2/3
                if p < 1e-10:
                    continue
                res.append(sa.completable(Wm/np.sqrt(p), G, sI, sG, budget - t1))
            if all(r is not None for r in res):
                key = (t1, max(res)); cnt[key] = cnt.get(key, 0) + 1
    return dict(sorted(cnt.items()))


if __name__ == '__main__':
    for name, G in TGT.items():
        t0 = time.time()
        print(f'{name}: search1 (1 ancilla, final meas) deterministic hits by T-count 0..6: {run_search1(G)}  '
              f'[{time.time()-t0:.0f}s]', flush=True)
        t0 = time.time()
        print(f'{name}: search2 (2 ancillas) deterministic orbits by T-count 0..3: {run_search2(G)}  '
              f'[{time.time()-t0:.0f}s]', flush=True)
        t0 = time.time()
        print(f'{name}: adaptive feasible (orbit, Q) by (t1, t2): {run_adaptive(G)}  [{time.time()-t0:.0f}s]', flush=True)
