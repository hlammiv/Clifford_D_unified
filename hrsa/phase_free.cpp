// phase_free.cpp — PHASE-FREE best-fit candidate pools for prescribed θ (2026-10-09).
//
// HRSA (and multi_theta) accept a Householder vector u = (x1, x2, x3)/3^f with |u|² = 2 when
//     |u1 − e^{iθ/2}|² + |u2 + 1|² + |u3|² < EPS²/8,
// which fixes the global phase of u. But V = P01(I − u u†) is invariant under u → e^{iφ}u, so
// the right acceptance is the minimum over φ:
//     min_φ |a − e^{iφ}|² + |b − e^{iφ}|² + |u3|² = |a|² + |b|² + 2 − 2|a + b| + |u3|²,
// with a = u1 e^{−iθ/2}, b = −u2, attained at φ* = arg(a + b). Multiplying u by a power of
// −ζ_9 (order 18) maps ring vectors to ring vectors, so every V has a representative with
// φ* ∈ [−π/18, π/18); we search exactly that window (each V found once, up to the window edge).
//
// Search. x2 lies in the thin arc {| |b| − 1 | < r, arg b ∈ window ± r}, r = sqrt(EPS²/8), and x1 in
// the matching arc around e^{iθ/2}, with |arg a − arg b| ≤ 2 asin r. The arc is cut into chunks;
// per chunk the x2 points (partitioned by arg b, so pairs are never produced twice) and the x1
// points (chunk ± 2 asin r) are enumerated with Fincke–Pohst on an ellipse fitted to the arc piece
// (σ_2, σ_4 bounded by 2·9^f as in multi_theta), then joined through the exact norm equation
// N(x1) + N(x2) + N(x3) = 2·9^f (σ_1-window on |x2|², hash lookup of the x3 norm class; x3 ranges
// over the disk of radius r around 0, first member per class as HRSA's k3 = 1).
// Accepted candidates are ranked by the TRUE error ||V − diag(e^{−iθ/2}, e^{iθ/2}, 1)||_F.
//
// θ's whose half-angles lie within PF_GROUP_SPAN (env; default one chunk width) share the x1 enumeration and
// the norm join (only the phase test is per θ), so a cluster of θ costs about as much as one θ.
//
// Speed (2026-10-10, ~4× at level 12, output identical): exact per-disk intervals at the enumeration leaf,
// double-double d1 instead of __float128, running integer coordinates, thread-local reused buffers, and a
// streaming sort–merge join on a compact d1 array instead of a random-access bucket lookup per x1 point.
// Second pass (same day, a further ~1.8× at level 12, output identical): the leaf also cuts the inner annulus
// radius and the two angular half-planes exactly (visits ≈ points), the level above the leaf cuts each disk's
// v_1-extent (and prunes nodes whose two last basis vectors are parallel in an embedding), and the x1
// enumeration probes a Bloom filter of the chunk's x2 d1 cells before storing a point, so only the ~5% of x1
// points that can have a join partner are materialised and sorted (PF_BLOOM_BITS=0 disables it). The per-θ
// "x1" count in the log is now the number of stored (filter-surviving) x1 points.
//
// Usage: phase_free THETA_FILE EPS F MAX_SOLNS OUT_DIR [FROB_CAP [CHUNK_PTS]]
//   OUT_DIR/t{i:04d}.txt: CANDDUMP lines (best true error first), as multi_theta (mt_to_pool.py).
#include "cyclotomic_int9.h"
#include "Z9chi.h"
#include <omp.h>
#include <quadmath.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using R9 = ringZ9Base<long long>;
using R9c = ringZ9chiBase<long long>;
typedef complex<double> cd;

static const double PI = 3.14159265358979323846;
static const double TOL = 1e-6;
static double pow3d(int n) { double p = 1; while (n-- > 0) p *= 3; return p; }

// ζ^i in the three embeddings σ_1, σ_2, σ_4.
static cd Z1[6], Z2[6], Z4[6];

// Exact real-subring key of N(x) = x·x̄ from the autocorrelation r_n = Σ_{i−j ≡ n (9)} a_i a_j:
// reduced coefficients (r0 − r3, r1 − r2, r2 − r1, r3 − r3, r4 − r2, r4 − r1), key (c0, c4, c5).
struct K3 { long long a, b, c; };
static inline K3 nkey(const int* x) {
	// r_n over ordered pairs (i, j), i − j ≡ n (mod 9), i, j ∈ 0..5: i − j = n (n ≤ 5) or i − j = n − 9 (n ≥ 4).
	const long long a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], a4 = x[4], a5 = x[5];
	const long long r0 = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5;
	const long long r1 = a1 * a0 + a2 * a1 + a3 * a2 + a4 * a3 + a5 * a4;
	const long long r2 = a2 * a0 + a3 * a1 + a4 * a2 + a5 * a3;
	const long long r3 = a3 * a0 + a4 * a1 + a5 * a2;
	const long long r4 = a4 * a0 + a5 * a1 + a0 * a5;          // i − j = 4, and i − j = −5
	return K3{r0 - r3, r4 - r2, r4 - r1};
}
static inline uint64_t k3hash(long long a, long long b, long long c) {
	uint64_t h = (uint64_t)a * 0x9E3779B97F4A7C15ULL;
	h ^= (uint64_t)b * 0xC2B2AE3D27D4EB4FULL; h = (h << 31) | (h >> 33);
	h ^= (uint64_t)c * 0x165667B19E3779F9ULL;
	return h ^ (h >> 29);
}
struct FlatK3 {
	vector<K3> keys; vector<int> vals; uint64_t mask = 0;
	void build(const vector<K3>& ks) {
		size_t cap = 16; while (cap < 2 * ks.size() + 16) cap <<= 1;
		keys.assign(cap, K3{0, 0, 0}); vals.assign(cap, -1); mask = cap - 1;
		for (int i = 0; i < (int)ks.size(); ++i) {
			uint64_t h = k3hash(ks[i].a, ks[i].b, ks[i].c) & mask;
			bool dup = false;
			while (vals[h] >= 0) { const K3& k = keys[h]; if (k.a == ks[i].a && k.b == ks[i].b && k.c == ks[i].c) { dup = true; break; } h = (h + 1) & mask; }
			if (!dup) { keys[h] = ks[i]; vals[h] = i; }      // keep the first member of each class
		}
	}
	inline int find(long long a, long long b, long long c) const {
		uint64_t h = k3hash(a, b, c) & mask;
		while (vals[h] >= 0) { const K3& k = keys[h]; if (k.a == a && k.b == b && k.c == c) return vals[h]; h = (h + 1) & mask; }
		return -1;
	}
};

static atomic<long long> g_visits(0), g_leaves(0), g_leaves0(0), g_pass(0);
static double g_lamscale = 1.0;   // env PF_LAMSCALE > 1 enlarges the ellipsoids (completeness check)
// d1 = |σ_1(x)|² − 9^f from the exact norm key (|σ_1(x)|² = σ_1(N(x)) is a fixed real combination of the
// integer key). N(x) reduced coefficients: (c0, c4 − c5, c5 − c4, 0, c4, c5) with key (c0, c4, c5), so
//     d1 = (c0 − 9^f) + (c4 − c5)(cos 2π/9 − cos 4π/9) + c4 cos 8π/9 + c5 cos 10π/9.
// The terms are ~9^f and cancel to ~r·9^f, so d1 is evaluated in double-double (fma-exact products, TwoSum
// accumulation): absolute error ~9^f·2^-104 ≈ 1e-24 at level 16, far below the join slack; the reference
// __float128 evaluation (d1_ref) is checked against it at startup. Contraction is disabled so the compiler
// cannot fuse the product and the sum of the error-free transformations.
static __float128 g_c1, g_c2, g_c4, g_c5, g_nine_f;
struct DD { double hi, lo; };
static DD g_k1, g_k4, g_k5; static double g_nine_fd;
static inline double d1_ref(const K3& k) {
	const __float128 c0 = k.a, c4 = k.b, c5 = k.c;
	return (double)((c0 - g_nine_f) + (c4 - c5) * g_c1 + (c5 - c4) * g_c2 + c4 * g_c4 + c5 * g_c5);
}
__attribute__((optimize("-ffp-contract=off"), noinline)) static double d1_of(const K3& k) {
	double s = (double)k.a - g_nine_fd, err = 0;          // exact: integers below 2^53
	const double a[3] = {(double)(k.b - k.c), (double)k.b, (double)k.c};
	const DD* c[3] = {&g_k1, &g_k4, &g_k5};
	for (int i = 0; i < 3; ++i) {
		const double p = a[i] * c[i]->hi, e = fma(a[i], c[i]->hi, -p) + a[i] * c[i]->lo;
		const double t = s + p, bv = t - s, et = (s - (t - bv)) + (p - bv);
		s = t; err += et + e;
	}
	return s + err;
}
struct Pt { int a[6]; cd z; double e1, d1; K3 k; };   // z = σ_1(x)/3^f, e1 = |σ_1(x)|² (unscaled, double), d1 = e1 − 9^f

// Leaf filter: |σ_2/3^f|², |σ_4/3^f|² <= s2max, and for σ_1 (z = σ_1/3^f) either the disk |z|² < rhi2
// (arc = false) or the arc piece |z|² ∈ (rlo2, rhi2), arg(z·cb) ∈ [lo, hi). Margins are applied by the caller.
struct Filt { bool arc; cd clo, chi; double rlo2, rhi2, s2max; };   // clo = cb·e^{−i·lo}, chi = cb·e^{−i·hi}

// Partner filter for the x1 enumeration (item "existence filter"): a Bloom filter over the d1 cells of the
// chunk's x2 points. The join needs an x2 with d1(x2) ∈ [elo, elo + win], elo = −d1(x1) − winA − slack; with
// cells of width cw ≥ win·(1 + 1e-3) that window meets at most the cells q(elo) and q(elo) + 1, so each x2
// marks q(d1) and q(d1) − 1 and an x1 probes q(elo) only. q(x) = floor((x − dref)·inv) is monotone in x, and the
// 1e-3 widening of cw covers the rounding of q (≲ 1e-5 at level 16), so no x1 with a partner is ever dropped;
// the survivors go through the unchanged exact join. One hash, ~16 bits per mark: ~12% false positives.
struct Bloom {
	vector<unsigned long long> bits; int shift = 64;
	static inline unsigned long long h(long long q) { return (unsigned long long)q * 0x9E3779B97F4A7C15ULL; }
	void build(const vector<double>& D, double dref, double inv, int bits_per_mark) {
		unsigned long long m = 1ULL << 16;
		while (m < (unsigned long long)D.size() * 2 * bits_per_mark) m <<= 1;
		shift = 64 - __builtin_ctzll(m);
		bits.assign(m / 64, 0);
		for (double d : D) {
			const long long q = (long long)floor((d - dref) * inv);
			for (long long qq = q - 1; qq <= q; ++qq) { const unsigned long long x = h(qq) >> shift; bits[x >> 6] |= 1ULL << (x & 63); }
		}
	}
};

// Fincke–Pohst over a ∈ Z^6 with  q(a) = [(Im w/ht)² + ((Re w − c0)/hr)²]/λ1 + (|σ_2/3^f|² + |σ_4/3^f|²)/λ2 <= R2,
// w = σ_1(a)·conj(d)/3^f (d a unit direction). LLL-reduced basis (6-D, redone per call). The three
// embeddings are carried as running sums down the recursion, so a visited point costs a few flops;
// integer coordinates are formed only for points passing the leaf filter.
struct FP {
	double R[6][6], pp[6], R2; long long U[6][6], y[6];   // y: offsets at levels 1..5 (used by the centre recurrence)
	cd E[6][3];                      // embeddings (σ_1, σ_2, σ_4)/3^f of reduced basis vector i
	const Filt* F; vector<Pt>* out; vector<double>* dout; double s3f; long long nvis = 0, nleaf = 0, nleaf0 = 0, npass = 0;
	long long a0[6];                 // y are offsets from the Babai point y0; a = a0 + Σ y_i U_i, a0 = y0·U
	double Aq[3], invA[3], bnd[3];   // |E_t|², 1/|E_t|² and the widened disk bounds for the leaf intervals
	double bndin;                    // narrowed inner σ_1 bound (arc only): the excluded middle of the annulus
	cd cu[3]; double pe[3], sqb[3];  // level-1 cut: cu = conj(E_0)/|E_0|, pe = Im(E_1·cu), sqb = sqrt(bnd)
	const unsigned long long* bfbits = nullptr; int bfshift = 64; double bfA = 0, bfB = 0, bfdref = 0, bfinv = 0;
	// Along the last coordinate the three embeddings are lines z_t(v) = S_t + v·E_t, so each disk constraint
	// (|z_1|² < rhi2, |z_2|², |z_4|² ≤ s2max) is a quadratic in v and cuts an exact interval; the ellipsoid
	// interval [cj − w, cj + w] is only an outer bound (the product of a rectangle and two disks fills ~1/7
	// of its ellipsoid). The σ_1 bound is widened by 1e-6 of the annulus thickness plus 1e-13 absolute (|z|² ≈ 1
	// carries ~1e-16 rounding, while the thickness is ~1e-10 at level 14 — a relative widening of the radius
	// would be either useless or unsafe), the σ_2/σ_4 bounds by 1e-9, and the interval by 1e-6 in v, so rounding
	// can never drop a point that passes the exact per-point tests below, which are kept unchanged.
	// The per-point tests are the exact ones; everything before them only narrows the visited range and is
	// widened so that rounding can never drop a point the per-point tests would accept.
	inline void visit(const cd* S, const long long* A, long long v) {
		const cd z1 = S[0] + (double)v * E[0][0];
		const double n1 = norm(z1);
		if (n1 >= F->rhi2) return;
		if (F->arc && n1 <= F->rlo2) return;
		if (norm(S[1] + (double)v * E[0][1]) > F->s2max) return;
		if (norm(S[2] + (double)v * E[0][2]) > F->s2max) return;
		if (F->arc) {   // arg(z·cb) ∈ [lo, hi) for a window of width < π around the real axis
			const cd zl = z1 * F->clo, zh = z1 * F->chi;
			if (zl.imag() < 0 || zh.imag() >= 0) return;
		}
		Pt p;
		for (int c = 0; c < 6; ++c) p.a[c] = (int)(A[c] + v * U[0][c]);   // integer coordinates, carried as running sums
		p.k = nkey(p.a); p.d1 = d1_of(p.k);
		++npass;
		if (bfbits) {   // partner filter: same elo expression as the join (tgt − winA − slack)
			const double tgt = -p.d1;
			const double elo = tgt - bfA - bfB;
			const long long q = (long long)floor((elo - bfdref) * bfinv);
			const unsigned long long x = Bloom::h(q) >> bfshift;
			if (!((bfbits[x >> 6] >> (x & 63)) & 1)) return;
		}
		cd z = 0; for (int i = 0; i < 6; ++i) z += Z1[i] * (double)p.a[i];
		p.z = z * s3f; p.e1 = norm(z);
		out->push_back(p); dout->push_back(p.d1);   // d1 also in a compact array (the join sorts on it)
	}
	void leaf(const cd* S, const long long* A, double cj, double w) {
		double vlo = cj - w, vhi = cj + w, B0 = 0;
		++nleaf;
		for (int t = 0; t < 3; ++t) {
			const double B = real(S[t] * conj(E[0][t])), C = norm(S[t]) - bnd[t];
			const double disc = B * B - Aq[t] * C;
			if (disc < 0) { ++nleaf0; return; }
			const double sq = sqrt(disc);
			vlo = max(vlo, (-B - sq) * invA[t] - 1e-6); vhi = min(vhi, (-B + sq) * invA[t] + 1e-6);
			if (t == 0) B0 = B;
		}
		long long h1, l2;   // visited: [lo, h1] ∪ [l2, hi] (the gap is the inside of the inner σ_1 disk)
		if (F->arc) {
			// Angular half-planes: Im((S_0 + v·E_0)·c) = al + v·be is linear in v. Skipped when the line is
			// nearly parallel to the boundary (|be| < 1e-8: the root would carry ~1e-16/|be| of rounding).
			for (int s = 0; s < 2; ++s) {
				const cd c = s ? F->chi : F->clo;
				const double al = imag(S[0] * c), be = imag(E[0][0] * c);
				if (fabs(be) < 1e-8) continue;
				const double vb = -al / be, wd = 1e-6 + 1e-9 * fabs(vb);
				// clo: al + v·be ≥ 0;  chi: al + v·be < 0
				if ((be > 0) == (s == 0)) vlo = max(vlo, vb - wd); else vhi = min(vhi, vb + wd);
			}
		}
		const long long lo = (long long)ceil(vlo), hi = (long long)floor(vhi);
		if (hi < lo) { ++nleaf0; return; }
		h1 = hi; l2 = hi + 1;
		if (F->arc) {   // exclude the open interval where |z_1|² < bndin (narrowed inner radius, shrunk by 1e-6 in v)
			const double C = norm(S[0]) - bndin, disc = B0 * B0 - Aq[0] * C;
			if (disc > 0) {
				const double sq = sqrt(disc);
				const double a = (-B0 - sq) * invA[0] + 1e-6, b = (-B0 + sq) * invA[0] - 1e-6;
				if (a < b) { h1 = min(hi, (long long)floor(a)); l2 = max(lo, (long long)ceil(b)); }   // floor(a) < ceil(b): disjoint
			}
		}
		nvis += max(0LL, h1 - lo + 1) + max(0LL, hi - l2 + 1);
		for (long long v = lo; v <= h1; ++v) visit(S, A, v);
		for (long long v = l2; v <= hi; ++v) visit(S, A, v);
	}
	void rec(int j, double dist, const cd* S, const long long* A) {
		double cj = pp[j];
		for (int i = j + 1; i < 6; ++i) cj -= R[j][i] * y[i];
		cj /= R[j][j];
		double rem = (R2 - dist) / (R[j][j] * R[j][j]);
		if (rem < 0) return;
		double w = sqrt(rem);
		if (j == 0) { leaf(S, A, cj, w); return; }
		double vlo = cj - w, vhi = cj + w;
		if (j == 1) {
			// Level-1 cut: in the (v_1, v_0) plane each disk is an ellipse; its v_1-extent is where the component of
			// S_t + v_1·E_1t orthogonal to E_0t (a line ps + v_1·pe in that scalar) stays within sqrt(bnd_t).
			// Adjacent reduced vectors are often real-subring multiples of each other, so E_1t ∥ E_0t exactly and
			// the orthogonal component is the constant ps: then either every v_1 passes or the whole node is empty
			// (|v_1·pe| ≤ 1e-9·vmax is allowed for).
			const double vmax = max(fabs(vlo), fabs(vhi));
			for (int t = 0; t < 3; ++t) {
				const double ps = imag(S[t] * cu[t]);
				if (fabs(pe[t]) < 1e-9) { if (fabs(ps) > sqb[t] + 1e-6 + 1e-9 * vmax) return; continue; }
				double a = (-sqb[t] - ps) / pe[t], b = (sqb[t] - ps) / pe[t];
				if (a > b) swap(a, b);
				vlo = max(vlo, a - 1e-6 - 1e-9 * fabs(a)); vhi = min(vhi, b + 1e-6 + 1e-9 * fabs(b));
			}
		}
		long long lo = (long long)ceil(vlo), hi = (long long)floor(vhi);
		for (long long v = lo; v <= hi; ++v) {
			y[j] = v;
			double d = R[j][j] * (v - cj), nd = dist + d * d;
			if (nd > R2) continue;
			cd S2[3] = {S[0] + (double)v * E[j][0], S[1] + (double)v * E[j][1], S[2] + (double)v * E[j][2]};
			long long A2[6]; for (int c = 0; c < 6; ++c) A2[c] = A[c] + v * U[j][c];
			rec(j - 1, nd, S2, A2);
		}
	}
};
// The weighted basis is extremely ill-conditioned (radial weight 1/hr ~ 1/r ≫ the O(1) σ_2/σ_4 weights:
// Gram entries span 1e18 at level 14, 1e22 at 16), so Gram–Schmidt and LLL are done in __float128
// (1e-34): once per chunk, negligible cost. Double LLL silently lost 99% of the points at level 14.
typedef __float128 qd;
struct PFilt { const Bloom* bf; double A, B, dref, inv; };   // partner filter parameters (x1 enumeration only)
static void fp_enum(cd d, double c0, double ht, double hr, double lam1, double lam2, double R2, int f, const Filt& flt, vector<Pt>& out, vector<double>& dout, const PFilt* pf = nullptr) {
	const double s3f = 1.0 / pow3d(f), w2 = 1.0 / sqrt(lam2);
	ht *= sqrt(lam1); hr *= sqrt(lam1);
	qd L[6][6];
	for (int i = 0; i < 6; ++i) {
		cd w = Z1[i] * conj(d) * s3f;
		L[i][0] = (qd)w.imag() / ht; L[i][1] = (qd)w.real() / hr;
		L[i][2] = (qd)(Z2[i].real() * s3f * w2); L[i][3] = (qd)(Z2[i].imag() * s3f * w2);
		L[i][4] = (qd)(Z4[i].real() * s3f * w2); L[i][5] = (qd)(Z4[i].imag() * s3f * w2);
	}
	qd p[6] = {0, (qd)c0 / hr, 0, 0, 0, 0};
	qd L0[6][6];
	for (int i = 0; i < 6; ++i) for (int c = 0; c < 6; ++c) L0[i][c] = L[i][c];
	FP fp;
	for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) fp.U[i][j] = (i == j);
	qd Bs[6][6], mu[6][6], bn[6];
	auto gs = [&]() {
		for (int i = 0; i < 6; ++i) {
			for (int c = 0; c < 6; ++c) Bs[i][c] = L[i][c];
			for (int j = 0; j < i; ++j) {
				qd s = 0; for (int c = 0; c < 6; ++c) s += L[i][c] * Bs[j][c];
				mu[i][j] = s / bn[j];
				for (int c = 0; c < 6; ++c) Bs[i][c] -= mu[i][j] * Bs[j][c];
			}
			bn[i] = 0; for (int c = 0; c < 6; ++c) bn[i] += Bs[i][c] * Bs[i][c];
		}
	};
	gs();
	int k = 1, guard = 0;
	while (k < 6 && guard++ < 100000) {
		for (int j = k - 1; j >= 0; --j) {
			qd q = nearbyintq(mu[k][j]);
			if (q != 0) { for (int c = 0; c < 6; ++c) { L[k][c] -= q * L[j][c]; fp.U[k][c] -= (long long)q * fp.U[j][c]; } gs(); }
		}
		if (bn[k] >= ((qd)0.99 - mu[k][k - 1] * mu[k][k - 1]) * bn[k - 1]) ++k;
		else { for (int c = 0; c < 6; ++c) { swap(L[k][c], L[k - 1][c]); swap(fp.U[k][c], fp.U[k - 1][c]); } gs(); k = max(k - 1, 1); }
	}
	// Rebuild L = U·L0 from the exact U (the in-place updates drift), then the QR used by the enumeration.
	for (int i = 0; i < 6; ++i) for (int c = 0; c < 6; ++c) { qd t = 0; for (int j = 0; j < 6; ++j) t += (qd)fp.U[i][j] * L0[j][c]; L[i][c] = t; }
	gs();
	for (int j = 0; j < 6; ++j) {
		qd nj = sqrtq(bn[j]);
		for (int i = 0; i < 6; ++i) fp.R[j][i] = (i == j) ? (double)nj : (i > j ? (double)(mu[i][j] * nj) : 0.0);
		qd s = 0; for (int c = 0; c < 6; ++c) s += Bs[j][c] * p[c];
		fp.pp[j] = (double)(s / nj);
	}
	for (int i = 0; i < 6; ++i) {
		cd e[3] = {0, 0, 0};
		for (int c = 0; c < 6; ++c) { e[0] += Z1[c] * (double)fp.U[i][c]; e[1] += Z2[c] * (double)fp.U[i][c]; e[2] += Z4[c] * (double)fp.U[i][c]; }
		for (int t = 0; t < 3; ++t) fp.E[i][t] = e[t] * s3f;
	}
	// Shift the origin to the Babai point y0 of the centre: the region lies ~1/r lattice units from 0, and
	// running sums started at 0 cancel catastrophically (lost boundary points at f = 5). The offsets y are
	// then O(1) and the running embedding sums start from the embedding of the integer point a0 = y0·U.
	long long y0[6];
	for (int j = 5; j >= 0; --j) {
		long double cj = fp.pp[j];
		for (int i = j + 1; i < 6; ++i) cj -= (long double)fp.R[j][i] * y0[i];
		y0[j] = llroundl(cj / fp.R[j][j]);
	}
	for (int j = 0; j < 6; ++j) {
		long double t = fp.pp[j];
		for (int i = j; i < 6; ++i) t -= (long double)fp.R[j][i] * y0[i];
		fp.pp[j] = (double)t;
	}
	for (int c = 0; c < 6; ++c) { long long t = 0; for (int i = 0; i < 6; ++i) t += y0[i] * fp.U[i][c]; fp.a0[c] = t; }
	cd S0[3] = {0, 0, 0};
	for (int c = 0; c < 6; ++c) { S0[0] += Z1[c] * (double)fp.a0[c]; S0[1] += Z2[c] * (double)fp.a0[c]; S0[2] += Z4[c] * (double)fp.a0[c]; }
	for (int t = 0; t < 3; ++t) S0[t] *= s3f;
	fp.R2 = R2; fp.F = &flt; fp.out = &out; fp.dout = &dout; fp.s3f = s3f;
	const double bq[3] = {flt.rhi2 + 1e-6 * (flt.rhi2 - flt.rlo2) + 1e-13, flt.s2max * (1 + 1e-9), flt.s2max * (1 + 1e-9)};
	for (int t = 0; t < 3; ++t) {
		fp.Aq[t] = norm(fp.E[0][t]); fp.invA[t] = 1.0 / fp.Aq[t]; fp.bnd[t] = bq[t];
		fp.cu[t] = conj(fp.E[0][t]) / sqrt(fp.Aq[t]); fp.pe[t] = imag(fp.E[1][t] * fp.cu[t]); fp.sqb[t] = sqrt(bq[t]);
	}
	fp.bndin = flt.rlo2 - 1e-6 * (flt.rhi2 - flt.rlo2) - 1e-13;
	if (pf && pf->bf) { fp.bfbits = pf->bf->bits.data(); fp.bfshift = pf->bf->shift; fp.bfA = pf->A; fp.bfB = pf->B; fp.bfdref = pf->dref; fp.bfinv = pf->inv; }
	fp.rec(5, 0.0, S0, fp.a0);
	g_visits += fp.nvis; g_leaves += fp.nleaf; g_leaves0 += fp.nleaf0; g_pass += fp.npass;
}


// (d1, index) of a compact d1 array in increasing d1. Two-level MSD bucket sort on the position of d1 in the
// a-priori range [dmin, dmax] (256 × 256 buckets, each scatter into 256 streams; values outside clamp to the end
// buckets), then std::sort inside each final bucket (a few elements, in L1). The first pass streams the 8-byte
// array once; the second works segment by segment in cache. Reading d1 out of the 80-byte Pt records cost a
// cache line per element per pass (~80 ns/element, DRAM-bound), as did an LSD radix sort and std::sort.
typedef pair<double, int> DI;
static void sort_by_d1(const vector<double>& D, double dmin, double dmax, vector<DI>& out, vector<DI>& tmp) {
	const size_t n = D.size();
	out.resize(n); tmp.resize(n);
	if (n == 0) return;
	const double scale = 65535.0 / (dmax - dmin);
	auto q16 = [&](double d) { int q = (int)((d - dmin) * scale); return q < 0 ? 0 : (q > 65535 ? 65535 : q); };
	size_t c1[257]; memset(c1, 0, sizeof c1);
	for (size_t i = 0; i < n; ++i) ++c1[(q16(D[i]) >> 8) + 1];
	for (int b = 0; b < 256; ++b) c1[b + 1] += c1[b];
	{ size_t pos[256]; memcpy(pos, c1, sizeof pos);
	  for (size_t i = 0; i < n; ++i) tmp[pos[q16(D[i]) >> 8]++] = {D[i], (int)i}; }
	for (int b = 0; b < 256; ++b) {
		const size_t s0 = c1[b], s1 = c1[b + 1];
		if (s1 == s0) continue;
		size_t c2[257]; memset(c2, 0, sizeof c2);
		for (size_t i = s0; i < s1; ++i) ++c2[(q16(tmp[i].first) & 0xFF) + 1];
		for (int d = 0; d < 256; ++d) c2[d + 1] += c2[d];
		{ size_t pos[256]; memcpy(pos, c2, sizeof pos);
		  for (size_t i = s0; i < s1; ++i) out[s0 + pos[q16(tmp[i].first) & 0xFF]++] = tmp[i]; }
		for (int d = 0; d < 256; ++d)
			if (c2[d + 1] - c2[d] > 1) sort(out.begin() + s0 + c2[d], out.begin() + s0 + c2[d + 1], [](const DI& A, const DI& B) { return A.first < B.first; });
	}
}

int main(int argc, char** argv) {
	if (argc < 6 || argc > 8) { cerr << "usage: phase_free THETA_FILE EPS F MAX_SOLNS OUT_DIR [FROB_CAP [CHUNK_PTS]]" << endl; return 1; }
	vector<double> thetas;
	{ ifstream in(argv[1]); double t; while (in >> t) thetas.push_back(t); }
	const double EPS = atof(argv[2]);
	const int f = atoi(argv[3]);
	const int max_solns = atoi(argv[4]);
	const string out_dir = argv[5];
	const double frob_cap = argc >= 7 ? atof(argv[6]) : 1e300;
	const double chunk_pts = argc >= 8 ? atof(argv[7]) : 2e5;
	if (f > 9) { cerr << "f > 9 not supported" << endl; return 1; }
	for (int i = 0; i < 6; ++i) { Z1[i] = polar(1.0, 2 * PI * i / 9); Z2[i] = polar(1.0, 2 * PI * 2 * i / 9); Z4[i] = polar(1.0, 2 * PI * 4 * i / 9); }
	if (getenv("PF_LAMSCALE")) g_lamscale = atof(getenv("PF_LAMSCALE"));
	const int K = (int)thetas.size();
	const double nine_f = pow3d(2 * f);
	g_nine_f = nine_f;
	{ const __float128 P = acosq(-1);
	  g_c1 = cosq(2 * P / 9); g_c2 = cosq(4 * P / 9); g_c4 = cosq(8 * P / 9); g_c5 = cosq(10 * P / 9);
	  auto split = [](__float128 x) { DD d; d.hi = (double)x; d.lo = (double)(x - (__float128)d.hi); return d; };
	  g_k1 = split(g_c1 - g_c2); g_k4 = split(g_c4); g_k5 = split(g_c5); g_nine_fd = nine_f; }
	const double eps_cond = EPS * EPS / 8.0;
	const double thr = eps_cond * (1 + TOL), r = sqrt(thr);
	if (r >= 0.5) { cerr << "EPS too large for the arc geometry (r = " << r << ")" << endl; return 1; }
	const double PHI = PI / 18;
	const double m = asin(r) * (1 + 1e-6) + 1e-12;           // |arg − φ*| bound for a point within r of e^{iφ*}
	const double s2max = 2.0 * (1 + TOL);
	// d1 values in the join are O(2r·9^f) (|x|² ≈ 9^f(1 ± r)²); their double rounding is 1e-16 of that,
	// far below the window width thr·9^f = r²·9^f as long as r > 1e-16. Slack = 1e-13 of the d1 range.
	const double e1_slack = 1e-13 * (2 * r * nine_f) + 1e-300;
	// x3: disk of radius r around 0, one representative per norm class.
	vector<Pt> X3;
	{
		Filt flt{false, cd(1, 0), cd(1, 0), 0, thr + 1e-12, s2max * (1 + 1e-9)};
		vector<Pt> pts; vector<double> dd;
		fp_enum(cd(1, 0), 0.0, r, r, 3.0, 6.0, 1.0 + 1e-6, f, flt, pts, dd);   // disk ≤ 1 → 1/3, |σ_k|² ≤ 2 → 1/3 each
		for (auto& p : pts) if (norm(p.z) < thr) X3.push_back(p);
	}
	sort(X3.begin(), X3.end(), [](const Pt& A, const Pt& B) { return A.e1 < B.e1; });
	{ double dev = 0; for (auto& p : X3) dev = max(dev, fabs(p.d1 + nine_f - p.e1) / nine_f);
	  cerr << "[pf] d1 key check on x3: max |d1 + 9^f − |σ1|²| / 9^f = " << dev << " | join window thr·9^f = " << thr * nine_f << ", slack " << e1_slack << endl;
	  if (dev > 1e-12) { cerr << "d1 key check FAILED" << endl; return 1; }
	  // double-double d1 against the __float128 reference: the x3 keys plus random keys of the join's size.
	  double dd = 0; unsigned long long st = 0x9E3779B97F4A7C15ULL;
	  auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double)(st >> 11) / 9007199254740992.0; };
	  for (int i = 0; i < 200000; ++i) {
		K3 k; if (i < (int)X3.size()) k = X3[i].k;
		else { k.a = llround(nine_f * (1 + 1e-3 * (rnd() - 0.5))); k.b = llround(2 * nine_f * (rnd() - 0.5)); k.c = llround(2 * nine_f * (rnd() - 0.5)); }
		dd = max(dd, fabs(d1_of(k) - d1_ref(k)));
	  }
	  cerr << "[pf] d1 double-double vs __float128: max diff " << dd << " (slack " << e1_slack << ")" << endl;
	  if (dd > 0.01 * e1_slack) { cerr << "d1 double-double check FAILED" << endl; return 1; } }
	FlatK3 t3; { vector<K3> ks; for (auto& p : X3) ks.push_back(p.k); t3.build(ks); }

	// Chunking of the x2 arc arg b ∈ [−Φ − m, Φ + m).
	const double D = nine_f * pow(2 * PI * nine_f, 2) / (pow(3.0, 4.5) / 8.0);   // lattice points per unit σ_1 area (z units)
	const double A0 = -PHI - m, A1 = PHI + m;
	double W = chunk_pts / (2 * r * D);
	W = min(W, 4 * sqrt(r));            // keep the sagitta ~ r
	W = max(W, 4 * m);                  // x1 overhang (2m each side) at most ~2×
	W = min(W, A1 - A0);
	const int NC = (int)ceil((A1 - A0) / W);
	W = (A1 - A0) / NC;
	cerr << "[pf] K=" << K << " f=" << f << " EPS=" << EPS << " r=" << r << " x3 points " << X3.size()
	     << " | chunks/θ " << NC << " width " << W << " | predicted x2 arc points " << (A1 - A0) * 2 * r * D
	     << " threads " << omp_get_max_threads() << endl;


	const long long C0i = 2LL * (long long)llround(nine_f);
	auto frob_err = [&](const cd u[3], double th) {
		cd H[3][3], V[3][3];
		for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) H[i][j] = (i == j ? 1.0 : 0.0) - u[i] * conj(u[j]);
		for (int j = 0; j < 3; ++j) { V[0][j] = H[1][j]; V[1][j] = H[0][j]; V[2][j] = H[2][j]; }
		cd T[3] = {polar(1.0, -0.5 * th), polar(1.0, 0.5 * th), 1.0};
		double s = 0;
		for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) s += norm(V[i][j] - (i == j ? T[i] : 0.0));
		return sqrt(s);
	};
	struct Cand { double frob; int i1[6], i2[6], i3; };
	vector<vector<Cand>> res(K);
	vector<long long> n1v(K, 0), n2v(K, 0), npair(K, 0), nacc(K, 0);
	double t_x2 = 0, t_x1 = 0, t_join = 0;   // thread-seconds

	// Points of an arc piece: |z| ∈ (1−r, 1+r), arg(z·conj(base)) ∈ [lo, hi), σ_2/σ_4 bounds.
	// The leaf filter is slightly widened; the exact test is redone on z recomputed from the integers,
	// so the x2 chunks partition the arc exactly.
	// out is cleared and filled in place (buffers are thread-local and reused across chunks: the per-chunk
	// allocation and page-faulting of ~100 MB vectors cost ~15% system time at level 14).
	auto arc_points = [&](cd base, double lo, double hi, vector<Pt>& out, vector<double>& dout, const PFilt* pf) {
		const double ph = 0.5 * (hi - lo), mid = 0.5 * (hi + lo);
		const cd d = base * polar(1.0, mid);
		const double rin = (1 - r) * cos(ph), rout = 1 + r;
		const double c0 = 0.5 * (rin + rout), hr = 0.5 * (rout - rin) * (1 + 1e-9) + 1e-15, ht = (1 + r) * sin(ph) * (1 + 1e-9) + 1e-15;
		Filt flt{true, conj(base) * polar(1.0, -(lo - 1e-12)), conj(base) * polar(1.0, -(hi + 1e-12)),
		         (1 - r) * (1 - r) - 1e-12, (1 + r) * (1 + r) + 1e-12, s2max * (1 + 1e-9)};
		out.clear(); dout.clear();
		fp_enum(d, c0, ht, hr, 6.0 * g_lamscale, 6.0 * g_lamscale, 1.0 + 1e-6, f, flt, out, dout, pf);   // rectangle ≤ 2 → 1/3, |σ_k|² ≤ 2 → 1/3 each (min-volume weights)
		size_t n = 0;
		for (size_t i = 0; i < out.size(); ++i) {
			const Pt& p = out[i];
			double az = abs(p.z);
			if (az <= 1 - r || az >= 1 + r) continue;
			double ang = arg(p.z * conj(base));
			if (ang < lo || ang >= hi) continue;
			if (n != i) { out[n] = p; dout[n] = dout[i]; }
			++n;
		}
		out.resize(n); dout.resize(n);
	};

	// θ groups: θ's whose half-angles lie within GSPAN of each other share one x1 enumeration (the union of
	// their x1 arcs) and one norm join; only the final phase test is per θ. Spread-out θ give singleton groups.
	const double GSPAN = getenv("PF_GROUP_SPAN") ? atof(getenv("PF_GROUP_SPAN")) : W;
	vector<int> ord(K); for (int k = 0; k < K; ++k) ord[k] = k;
	auto half = [&](int k) { double h = fmod(0.5 * thetas[k], 2 * PI); return h < 0 ? h + 2 * PI : h; };
	sort(ord.begin(), ord.end(), [&](int x, int y) { return half(x) < half(y); });
	vector<vector<int>> groups;
	for (int i = 0; i < K; ++i) {
		if (groups.empty() || half(ord[i]) - half(groups.back().front()) > GSPAN) groups.push_back({});
		groups.back().push_back(ord[i]);
	}
	const int G = (int)groups.size();
	cerr << "[pf] θ groups: " << G << " (span " << GSPAN << ")" << endl;

	const double DMAX = ((1 + r) * (1 + r) - 1) * nine_f * (1 + 1e-9);   // arc points have |z|² − 1 ∈ ((1−r)² − 1, (1+r)² − 1)
	const double DMIN = ((1 - r) * (1 - r) - 1) * nine_f * (1 + 1e-9);
	const double winA = thr * nine_f * (1 + 1e-9);                    // join window: d1(x2) ∈ [−d1(x1) − winA − slack, −d1(x1) + slack]
	const double cw = (winA + 2 * e1_slack) * (1 + 1e-3);             // partner-filter cell width (see Bloom)
	const int bloom_bits = getenv("PF_BLOOM_BITS") ? atoi(getenv("PF_BLOOM_BITS")) : 16;   // 0 disables the filter
	// Work unit = (x2 chunk, block of groups): the x2 points of a chunk are θ-independent and shared by the block.
	const int nt = omp_get_max_threads();
	const int TB = max(1, min(G, (int)((long long)G * NC / (4 * nt))));
	const int NB = (G + TB - 1) / TB;
	#pragma omp parallel for schedule(dynamic, 1)
	for (long long unit = 0; unit < (long long)NC * NB; ++unit) {
		const int c = (int)(unit % NC), kb = (int)(unit / NC);
		const double lo = A0 + c * W, hi = (c == NC - 1) ? A1 : A0 + (c + 1) * W;
		double tA = omp_get_wtime();
		static thread_local vector<Pt> X2, X1;
		static thread_local vector<double> e2, D1, D2;   // D1/D2: compact d1 of X1/X2
		static thread_local vector<DI> k1, k1tmp;   // (d1, index) of X1 in increasing d1
		static thread_local vector<int> idx2;      // X2 is left in enumeration order; e2/idx2 give the increasing-d1 order
		static thread_local Bloom bloom;
		arc_points(cd(-1, 0), lo, hi, X2, D2, nullptr);
		sort_by_d1(D2, DMIN, DMAX, k1, k1tmp);
		e2.resize(X2.size()); idx2.resize(X2.size());
		for (size_t j = 0; j < X2.size(); ++j) { e2[j] = k1[j].first; idx2[j] = k1[j].second; }
		PFilt pf{nullptr, winA, e1_slack, DMIN, 1.0 / cw};
		if (bloom_bits > 0) { bloom.build(D2, DMIN, 1.0 / cw, bloom_bits); pf.bf = &bloom; }
		double tB = omp_get_wtime(), ux1 = 0, ujn = 0;
		for (int g = kb * TB; g < min(G, (kb + 1) * TB); ++g) {
		const vector<int>& grp = groups[g];
		const int ng = (int)grp.size();
		vector<cd> emth(ng);
		for (int q = 0; q < ng; ++q) emth[q] = polar(1.0, -0.5 * thetas[grp[q]]);
		const double span = half(grp.back()) - half(grp.front());
		double tC = omp_get_wtime();
		arc_points(conj(emth[0]), lo - 2 * m, hi + 2 * m + span, X1, D1, &pf);
		double tD = omp_get_wtime(); ux1 += tD - tC;
		vector<vector<Cand>> loc(ng); long long np = 0; vector<long long> na(ng, 0);
		// Sort–merge join on d1: x1 in decreasing d1 → the x2 window [−d1 − thr·9^f, −d1] moves up monotonically,
		// so the scan over e2 is sequential (the earlier bucket lookup per x1 was a cache miss per point);
		// the X1/X2 records themselves are touched only for the (rare) pairs.
		sort_by_d1(D1, DMIN, DMAX, k1, k1tmp);
		size_t j0 = 0;
		for (size_t i1 = k1.size(); i1-- > 0;) {     // decreasing d1(x1)
			const DI& kk = k1[i1];
			const double tgt = -kk.first;          // d1(x2) = −d1(x1) − |σ_1(x3)|², |σ_1(x3)|² ∈ [0, thr·9^f)
			const double elo = tgt - winA - e1_slack, ehi = tgt + e1_slack;
			while (j0 < e2.size() && e2[j0] < elo) ++j0;
			if (j0 >= e2.size()) break;
			if (e2[j0] > ehi) continue;
			const Pt& p1 = X1[kk.second];
			for (size_t j2 = j0; j2 < e2.size() && e2[j2] <= ehi; ++j2) {
				const Pt& p2 = X2[idx2[j2]];
				++np;
				const int c3 = t3.find(C0i - p1.k.a - p2.k.a, -p1.k.b - p2.k.b, -p1.k.c - p2.k.c);
				if (c3 < 0) continue;
				const cd b = -p2.z;
				for (int q = 0; q < ng; ++q) {
					const cd a = p1.z * emth[q];
					const cd s = a + b;
					const double e = norm(a) + norm(b) + 2 - 2 * abs(s) + norm(X3[c3].z);
					if (!(e < eps_cond)) continue;
					const double phis = arg(s);
					if (phis < -PHI || phis >= PHI) continue;
					++na[q];
					const cd u[3] = {p1.z, p2.z, X3[c3].z};
					const double fr = frob_err(u, thetas[grp[q]]);
					if (!(fr < frob_cap)) continue;
					Cand cc; cc.frob = fr; cc.i3 = c3;
					for (int i = 0; i < 6; ++i) { cc.i1[i] = p1.a[i]; cc.i2[i] = p2.a[i]; }
					loc[q].push_back(cc);
				}
			}
		}
		ujn += omp_get_wtime() - tD;
		#pragma omp critical
		{
			if (g == kb * TB) t_x2 += tB - tA;
			t_x1 += ux1; t_join += ujn; ux1 = ujn = 0;
			for (int q = 0; q < ng; ++q) {
				const int k = grp[q];
				n1v[k] += X1.size(); n2v[k] += X2.size(); npair[k] += np; nacc[k] += na[q];
				res[k].insert(res[k].end(), loc[q].begin(), loc[q].end());
			}
		}
		}
	}

	cerr << "[pf] thread-seconds: x2 enum " << t_x2 << ", x1 enum " << t_x1 << ", join " << t_join << " | leaf calls " << g_leaves.load()
	     << " (empty " << g_leaves0.load() << ") visits " << g_visits.load() << " passed " << g_pass.load() << endl;
	for (int k = 0; k < K; ++k) {
		auto& v = res[k];
		sort(v.begin(), v.end(), [](const Cand& A, const Cand& B) { return A.frob < B.frob; });
		if ((int)v.size() > max_solns) v.resize(max_solns);
		char path[4096];
		snprintf(path, sizeof path, "%s/t%04d.txt", out_dir.c_str(), k);
		FILE* fo = fopen(path, "w");
		if (!fo) { cerr << "cannot write " << path << endl; return 1; }
		for (size_t s = 0; s < v.size(); ++s) {
			int arr1[9] = {0}, arr2[9] = {0}, arr3[9] = {0};
			for (int i = 0; i < 6; ++i) { arr1[i] = v[s].i1[i]; arr2[i] = v[s].i2[i]; arr3[i] = X3[v[s].i3].a[i]; }
			R9c c1(R9(arr1), f), c2(R9(arr2), f), c3(R9(arr3), f);
			c1.reduce(); c2.reduce(); c3.reduce();
			auto z1 = c1.toComplexDouble(), z2 = c2.toComplexDouble(), z3 = c3.toComplexDouble();
			fprintf(fo, "CANDDUMP %zu 0 %.17g %.17g %.17g %.17g %.17g %.17g %d", s, z1.real(), z1.imag(),
			        z2.real(), z2.imag(), z3.real(), z3.imag(), f);
			for (const R9c* cc : {&c1, &c2, &c3}) { R9 n = cc->getNumerator(); for (int t = 0; t < 6; ++t) fprintf(fo, " %lld", n.getTerm(t)); }
			fprintf(fo, " E %d %d %d\n", c1.getExp(), c2.getExp(), c3.getExp());
		}
		fclose(fo);
		cerr << "[pf] θ#" << k << " x1 " << n1v[k] << " x2 " << n2v[k] << " pairs " << npair[k] << " accepted " << nacc[k]
		     << " kept " << v.size() << " best " << (v.empty() ? NAN : v[0].frob) << endl;
	}
	cerr << "[pf] DONE" << endl;
	return 0;
}
