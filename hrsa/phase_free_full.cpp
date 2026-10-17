// phase_free_full.cpp — phase-free candidate pools for FULL Householder targets (2026-10-14).
//
// Target t = (c1 e^{iθ/2}, −c2, c3 e^{iη}), c1² + c2² + c3² = 2 (c1² = PF_C1SQ, c3² = PF_C3SQ, env), and
// V ≈ P01(I − t t†): a reflection about any vector of C³ (4 real parameters), times a Clifford. Three of them
// give any single-qutrit gate up to phase. phase_free.cpp is the case c1 = c2 = 1, c3 = 0 (R_z(θ)), and
// phase_free_tl.cpp the case c3 = 0 with general c1.
//
// Acceptance, minimised over the global phase φ of u = (x1, x2, x3)/3^f, |u|² = 2:
//     |a|² + |b|² + |c|² + 2 − 2|c1 a + c2 b + c3 c| < EPS²/8,   a = u1 e^{−iθ/2}, b = −u2, c = u3 e^{−iη},
// at φ* = arg(c1 a + c2 b + c3 c) ∈ [−π/18, π/18) (−ζ_9 window, as phase_free).
//
// Search. With x3 on an arc (not the disk around 0 of phase_free), the narrow d1 window that joined x1 with x2
// is gone, so the join is geometric. Per x2 chunk (arg b ∈ [lo, hi), radius c2) the x1 arc (radius c1,
// arg a ∈ [lo − m1 − m2, hi + m1 + m2)) and the x3 arc (radius c3, arg c ∈ [lo − m2 − m3, hi + m2 + m3)) are
// enumerated per target, m_i = asin(r/c_i). x1 is sorted by arg a; for each x2 the x1 with
// |arg a − arg b| ≤ m1 + m2 are scanned, a pair survives if its two-component error
// |a|² + |b|² + c1² + c2² − 2|c1 a + c2 b| (a lower bound of the full one) is below EPS²/8, and x3 is then
// looked up by its exact norm key 2·9^f − N(x1) − N(x2) in a hash of the x3 arc (all members of a class).
//
// Usage: phase_free_full TARGET_FILE EPS F MAX_SOLNS OUT_DIR [FROB_CAP [CHUNK_PTS]]
//   TARGET_FILE: one "θ η" pair per line. OUT_DIR/t{i:04d}.txt: CANDDUMP lines as phase_free.
//
// Shared infrastructure below (ring keys, d1, Fincke–Pohst over arc pieces) is phase_free.cpp's:
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

// (phase_free.cpp infrastructure follows)
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

// All members of each norm class (x3 lookup: different members are different vectors here).
struct MultiK3 {
	vector<K3> keys; vector<int> head, nxt; uint64_t mask = 0;
	void build(const vector<Pt>& P) {
		size_t cap = 16; while (cap < 2 * P.size() + 16) cap <<= 1;
		keys.assign(cap, K3{0, 0, 0}); head.assign(cap, -1); nxt.assign(P.size(), -1); mask = cap - 1;
		for (int i = 0; i < (int)P.size(); ++i) {
			const K3& k = P[i].k;
			uint64_t h = k3hash(k.a, k.b, k.c) & mask;
			while (head[h] >= 0 && !(keys[h].a == k.a && keys[h].b == k.b && keys[h].c == k.c)) h = (h + 1) & mask;
			if (head[h] < 0) keys[h] = k;
			nxt[i] = head[h]; head[h] = i;
		}
	}
	inline int find(long long a, long long b, long long c) const {
		uint64_t h = k3hash(a, b, c) & mask;
		while (head[h] >= 0) { const K3& k = keys[h]; if (k.a == a && k.b == b && k.c == c) return head[h]; h = (h + 1) & mask; }
		return -1;
	}
};

int main(int argc, char** argv) {
	if (argc < 6 || argc > 8) { cerr << "usage: phase_free_full TARGET_FILE EPS F MAX_SOLNS OUT_DIR [FROB_CAP [CHUNK_PTS]]" << endl; return 1; }
	vector<double> thetas, etas;
	{ ifstream in(argv[1]); double t, e; while (in >> t >> e) { thetas.push_back(t); etas.push_back(e); } }
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
	// Ring points carry z = σ_1(x)/3^f with absolute error ~1e-15 (a sum of 3^f-sized terms), i.e. ~1e-4 relative
	// in e at f = 8; accept with 1e-3 slack and let the true-error cap decide (review C35).
	const double eps_acc = eps_cond * (1 + 1e-3);
	const double C1SQ = getenv("PF_C1SQ") ? atof(getenv("PF_C1SQ")) : 2.0 / 3;
	const double C3SQ = getenv("PF_C3SQ") ? atof(getenv("PF_C3SQ")) : 2.0 / 3;
	const double C2SQ = 2.0 - C1SQ - C3SQ;
	if (!(C1SQ > 0 && C3SQ > 0 && C2SQ > 0)) { cerr << "need c1², c3² > 0 and c1² + c3² < 2" << endl; return 1; }
	const double C1 = sqrt(C1SQ), C2 = sqrt(C2SQ), C3 = sqrt(C3SQ);
	if (r >= 0.5 * min(C1, min(C2, C3))) { cerr << "EPS too large for the arc radii" << endl; return 1; }
	const double PHI = PI / 18;
	auto mrad = [&](double c) { return asin(r / c) * (1 + 1e-6) + 1e-12; };   // |arg − φ*| bound within r of c e^{iφ*}
	const double m1 = mrad(C1), m2 = mrad(C2), m3 = mrad(C3), m12 = m1 + m2, m23 = m2 + m3;
	const double s2max = 2.0 * (1 + TOL);

	const double D = nine_f * pow(2 * PI * nine_f, 2) / (pow(3.0, 4.5) / 8.0);   // lattice points per unit σ_1 area
	const double A0 = -PHI - m2, A1 = PHI + m2;
	double W = chunk_pts / (2 * r * C2 * D);
	W = min(W, 4 * sqrt(r));
	W = max(W, 2 * max(m12, m23));
	W = min(W, A1 - A0);
	const int NC = (int)ceil((A1 - A0) / W);
	W = (A1 - A0) / NC;
	cerr << "[pff] c^2 = (" << C1SQ << ", " << C2SQ << ", " << C3SQ << ")" << endl;
	cerr << "[pff] K=" << K << " f=" << f << " EPS=" << EPS << " r=" << r << " | chunks/target " << NC << " width " << W
	     << " | predicted x2 arc points " << (A1 - A0) * 2 * r * C2 * D << " threads " << omp_get_max_threads() << endl;

	const long long C0i = 2LL * (long long)llround(nine_f);
	auto frob_err = [&](const cd u[3], double th, double et) {
		cd H[3][3], V[3][3], G[3][3], T[3][3];
		const cd t[3] = {C1 * polar(1.0, 0.5 * th), cd(-C2, 0), C3 * polar(1.0, et)};
		for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
			H[i][j] = (i == j ? 1.0 : 0.0) - u[i] * conj(u[j]);
			G[i][j] = (i == j ? 1.0 : 0.0) - t[i] * conj(t[j]);
		}
		for (int j = 0; j < 3; ++j) { V[0][j] = H[1][j]; V[1][j] = H[0][j]; V[2][j] = H[2][j]; T[0][j] = G[1][j]; T[1][j] = G[0][j]; T[2][j] = G[2][j]; }
		double s = 0;
		for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) s += norm(V[i][j] - T[i][j]);
		return sqrt(s);
	};
	struct Cand { double frob; int i1[6], i2[6], i3[6]; };
	vector<vector<Cand>> res(K);
	vector<long long> n1v(K, 0), n3v(K, 0), npair(K, 0), nlook(K, 0), nacc(K, 0);   // nlook: pairs past the Bloom filter
	long long n2tot = 0;
	double t_x2 = 0, t_x13 = 0, t_join = 0;

	// Points of an arc piece: |z| ∈ (ρ−r, ρ+r), arg(z·conj(base)) ∈ [lo, hi), σ_2/σ_4 bounds (exact final test).
	auto arc_points = [&](cd base, double rho, double lo, double hi, vector<Pt>& out, vector<double>& dout) {
		const double ph = 0.5 * (hi - lo), mid = 0.5 * (hi + lo);
		const cd d = base * polar(1.0, mid);
		const double rin = (rho - r) * cos(ph), rout = rho + r;
		const double c0 = 0.5 * (rin + rout), hr = 0.5 * (rout - rin) * (1 + 1e-9) + 1e-15, ht = (rho + r) * sin(ph) * (1 + 1e-9) + 1e-15;
		Filt flt{true, conj(base) * polar(1.0, -(lo - 1e-12)), conj(base) * polar(1.0, -(hi + 1e-12)),
		         (rho - r) * (rho - r) - 1e-12, (rho + r) * (rho + r) + 1e-12, s2max * (1 + 1e-9)};
		out.clear(); dout.clear();
		fp_enum(d, c0, ht, hr, 6.0 * g_lamscale, 6.0 * g_lamscale, 1.0 + 1e-6, f, flt, out, dout, nullptr);
		size_t n = 0;
		for (size_t i = 0; i < out.size(); ++i) {
			const Pt& p = out[i];
			const double az = abs(p.z);
			if (az <= rho - r || az >= rho + r) continue;
			const double ang = arg(p.z * conj(base));
			if (ang < lo || ang >= hi) continue;
			if (n != i) out[n] = p;
			++n;
		}
		out.resize(n);
	};

	const int nt = omp_get_max_threads();
	const int TB = max(1, min(K, (int)((long long)K * NC / (4 * nt))));
	const int NB = (K + TB - 1) / TB;
	#pragma omp parallel for schedule(dynamic, 1)
	for (long long unit = 0; unit < (long long)NC * NB; ++unit) {
		const int c = (int)(unit % NC), kb = (int)(unit / NC);
		const double lo = A0 + c * W, hi = (c == NC - 1) ? A1 : A0 + (c + 1) * W;
		double tA = omp_get_wtime();
		static thread_local vector<Pt> X2, X1, X3;
		static thread_local vector<double> dd;
		static thread_local vector<pair<double, int>> s1;
		static thread_local MultiK3 h3;
		// x1 in increasing arg a, structure of arrays (the pair scan reads these sequentially)
		struct P1 { double g, ar, ai, n; long long ka, kb, kc; int idx; };   // x1 record for the pair scan
		static thread_local vector<P1> q1;
		static thread_local vector<int> o2;   // x2 in increasing arg b: the x1 window then slides monotonically
		static thread_local vector<unsigned long long> bl3;   // Bloom filter of the x3 keys (one hash, ~16 bits/key)
		arc_points(cd(-1, 0), C2, lo, hi, X2, dd);
		vector<double> angb(X2.size());
		for (size_t j = 0; j < X2.size(); ++j) angb[j] = arg(-X2[j].z);
		o2.resize(X2.size()); for (size_t j = 0; j < X2.size(); ++j) o2[j] = (int)j;
		sort(o2.begin(), o2.end(), [&](int x, int y) { return angb[x] < angb[y]; });
		double tB = omp_get_wtime(), u13 = 0, ujn = 0;
		for (int k = kb * TB; k < min(K, (kb + 1) * TB); ++k) {
			const cd eth = polar(1.0, -0.5 * thetas[k]), eet = polar(1.0, -etas[k]);
			double tC = omp_get_wtime();
			arc_points(conj(eth), C1, lo - m12, hi + m12, X1, dd);
			arc_points(conj(eet), C3, lo - m23, hi + m23, X3, dd);
			s1.resize(X1.size());
			for (size_t i = 0; i < X1.size(); ++i) s1[i] = {arg(X1[i].z * eth), (int)i};
			sort(s1.begin(), s1.end());
			const size_t N1 = s1.size();
			q1.resize(N1);
			for (size_t i = 0; i < N1; ++i) {
				const Pt& p = X1[s1[i].second]; const cd a = p.z * eth;
				q1[i] = P1{s1[i].first, a.real(), a.imag(), norm(a), p.k.a, p.k.b, p.k.c, s1[i].second};
			}
			h3.build(X3);
			int bsh = 64; { unsigned long long mb = 1ULL << 12; while (mb < 16ULL * X3.size()) mb <<= 1; bsh = 64 - __builtin_ctzll(mb);
			  bl3.assign(mb / 64, 0);
			  for (const Pt& p : X3) { const unsigned long long x = k3hash(p.k.a, p.k.b, p.k.c) >> bsh; bl3[x >> 6] |= 1ULL << (x & 63); } }
			double tD = omp_get_wtime(); u13 += tD - tC;
			vector<Cand> loc; long long np = 0, nl = 0, na = 0;
			size_t i0 = 0;
			for (size_t jj = 0; jj < X2.size(); ++jj) {
				const int j = o2[jj];
				const Pt& p2 = X2[j];
				const cd b = -p2.z;
				const double nb = norm(b), br = C2 * b.real(), bi = C2 * b.imag();
				const double base = nb + C1SQ + C2SQ - eps_cond;      // pair test: 2|c1 a + c2 b| > n(a) + base
				const long long qa = C0i - p2.k.a, qb = -p2.k.b, qc = -p2.k.c;
				const double glo = angb[j] - m12, ghi = angb[j] + m12;
				while (i0 < N1 && q1[i0].g < glo) ++i0;
				for (size_t i = i0; i < N1 && q1[i].g <= ghi; ++i) {
					const P1& q = q1[i];
					++np;
					const double R = q.n + base - 1e-12;   // slack ≫ double rounding of these O(1) terms (review C35)
					const double sr = C1 * q.ar + br, si = C1 * q.ai + bi;
					if (R > 0 && 4 * (sr * sr + si * si) <= R * R) continue;
					const long long ka = qa - q.ka, kb = qb - q.kb, kc = qc - q.kc;
					{ const unsigned long long x = k3hash(ka, kb, kc) >> bsh; if (!((bl3[x >> 6] >> (x & 63)) & 1)) continue; }
					++nl;
					const Pt& p1 = X1[q.idx];
					const cd a(q.ar, q.ai); const double na2 = q.n;
					for (int i3 = h3.find(ka, kb, kc); i3 >= 0; i3 = h3.nxt[i3]) {
						const Pt& p3 = X3[i3];
						const cd cc = p3.z * eet;
						const cd s = C1 * a + C2 * b + C3 * cc;
						const cd w = s / abs(s);       // difference form, no cancellation (review C35)
						const double e = norm(a - C1 * w) + norm(b - C2 * w) + norm(cc - C3 * w);
						if (!(e < eps_acc)) continue;
						const double phis = arg(s);
						if (phis < -PHI || phis >= PHI) continue;
						++na;
						const cd u[3] = {p1.z, p2.z, p3.z};
						const double fr = frob_err(u, thetas[k], etas[k]);
						if (!(fr < frob_cap)) continue;
						Cand cd_; cd_.frob = fr;
						for (int i = 0; i < 6; ++i) { cd_.i1[i] = p1.a[i]; cd_.i2[i] = p2.a[i]; cd_.i3[i] = p3.a[i]; }
						loc.push_back(cd_);
					}
				}
			}
			ujn += omp_get_wtime() - tD;
			#pragma omp critical
			{
				n1v[k] += X1.size(); n3v[k] += X3.size(); npair[k] += np; nlook[k] += nl; nacc[k] += na;
				res[k].insert(res[k].end(), loc.begin(), loc.end());
			}
		}
		#pragma omp critical
		{ t_x2 += tB - tA; t_x13 += u13; t_join += ujn; n2tot += X2.size(); }
	}

	cerr << "[pff] thread-seconds: x2 enum " << t_x2 << ", x1+x3 enum " << t_x13 << ", join " << t_join
	     << " | x2 points (all chunks, per block) " << n2tot << " leaf visits " << g_visits.load() << endl;
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
			for (int i = 0; i < 6; ++i) { arr1[i] = v[s].i1[i]; arr2[i] = v[s].i2[i]; arr3[i] = v[s].i3[i]; }
			R9c c1(R9(arr1), f), c2(R9(arr2), f), c3(R9(arr3), f);
			c1.reduce(); c2.reduce(); c3.reduce();
			auto z1 = c1.toComplexDouble(), z2 = c2.toComplexDouble(), z3 = c3.toComplexDouble();
			fprintf(fo, "CANDDUMP %zu 0 %.17g %.17g %.17g %.17g %.17g %.17g %d", s, z1.real(), z1.imag(),
			        z2.real(), z2.imag(), z3.real(), z3.imag(), f);
			for (const R9c* cc : {&c1, &c2, &c3}) { R9 n = cc->getNumerator(); for (int t = 0; t < 6; ++t) fprintf(fo, " %lld", n.getTerm(t)); }
			fprintf(fo, " E %d %d %d\n", c1.getExp(), c2.getExp(), c3.getExp());
		}
		fclose(fo);
		cerr << "[pff] target#" << k << " x1 " << n1v[k] << " x3 " << n3v[k] << " pairs " << npair[k] << " lookups " << nlook[k]
		     << " accepted " << nacc[k] << " kept " << v.size() << " best " << (v.empty() ? NAN : v[0].frob) << endl;
	}
	cerr << "[pff] DONE" << endl;
	return 0;
}
