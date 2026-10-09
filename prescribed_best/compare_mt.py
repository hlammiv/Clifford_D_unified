"""compare_mt.py A_DIR B_DIR N — compare multi_theta outputs per θ by (x1, x2) numerator sets."""
import sys
import candpool
a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
same = 0; na = nb = 0; bad = []
for i in range(n):
    key = lambda c: (tuple(c["nums"][0]), tuple(c["nums"][1]))
    A = {key(c) for c in candpool.parse(open(f"{a}/t{i:04d}.txt").read())}
    B = {key(c) for c in candpool.parse(open(f"{b}/t{i:04d}.txt").read())}
    na += len(A); nb += len(B)
    if A == B: same += 1
    else: bad.append((i, len(A), len(B), len(A - B), len(B - A)))
print(f"identical (x1,x2) sets: {same}/{n}; candidates {na} vs {nb}; differing (i, nA, nB, A-B, B-A): {bad[:5]}")
