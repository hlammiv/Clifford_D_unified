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
// Usage: phase_free THETA_FILE EPS F MAX_SOLNS OUT_DIR [FROB_CAP [CHUNK_PTS]]
//   OUT_DIR/t{i:04d}.txt: CANDDUMP lines (best true error first), as multi_theta (mt_to_pool.py).
#include "cyclotomic_int9.h"
#include "Z9chi.h"
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
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
	long long r[9] = {0};
	for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) r[((i - j) % 9 + 9) % 9] += (long long)x[i] * x[j];
	return K3{r[0] - r[3], r[4] - r[2], r[4] - r[1]};
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

static atomic<long long> g_visits(0);
static double g_lamscale = 1.0;   // env PF_LAMSCALE > 1 enlarges the ellipsoids (completeness check)
// d1 = |σ_1(x)|² − 9^f, evaluated in long double from the exact norm key (|σ_1(x)|² = σ_1(N(x)) is a fixed
// real combination of the integer key), so the small difference keeps ~1e-18·9^f absolute accuracy.
static long double g_c1, g_c2, g_c4, g_c5, g_nine_f;
static inline double d1_of(const K3& k) {
	// N(x) reduced coefficients: (c0, c4 − c5, c5 − c4, 0, c4, c5) with key (c0, c4, c5); N is real.
	const long double c0 = k.a, c4 = k.b, c5 = k.c;
	return (double)((c0 - g_nine_f) + (c4 - c5) * g_c1 + (c5 - c4) * g_c2 + c4 * g_c4 + c5 * g_c5);
}
struct Pt { int a[6]; cd z; double e1, d1; K3 k; };   // z = σ_1(x)/3^f, e1 = |σ_1(x)|² (unscaled, double), d1 = e1 − 9^f

// Leaf filter: |σ_2/3^f|², |σ_4/3^f|² <= s2max, and for σ_1 (z = σ_1/3^f) either the disk |z|² < rhi2
// (arc = false) or the arc piece |z|² ∈ (rlo2, rhi2), arg(z·cb) ∈ [lo, hi). Margins are applied by the caller.
struct Filt { bool arc; cd clo, chi; double rlo2, rhi2, s2max; };   // clo = cb·e^{−i·lo}, chi = cb·e^{−i·hi}

// Fincke–Pohst over a ∈ Z^6 with  q(a) = [(Im w/ht)² + ((Re w − c0)/hr)²]/λ1 + (|σ_2/3^f|² + |σ_4/3^f|²)/λ2 <= R2,
// w = σ_1(a)·conj(d)/3^f (d a unit direction). LLL-reduced basis (6-D, redone per call). The three
// embeddings are carried as running sums down the recursion, so a visited point costs a few flops;
// integer coordinates are formed only for points passing the leaf filter.
struct FP {
	double R[6][6], pp[6], R2; long long U[6][6], y[6];
	cd E[6][3];                      // embeddings (σ_1, σ_2, σ_4)/3^f of reduced basis vector i
	const Filt* F; vector<Pt>* out; double s3f; long long nvis = 0;
	long long a0[6];                 // y are offsets from the Babai point y0; a = a0 + Σ y_i U_i, a0 = y0·U
	void leaf(const cd* S, double cj, double w) {
		long long lo = (long long)ceil(cj - w), hi = (long long)floor(cj + w);
		if (hi >= lo) nvis += hi - lo + 1;
		for (long long v = lo; v <= hi; ++v) {
			const cd z1 = S[0] + (double)v * E[0][0];
			const double n1 = norm(z1);
			if (n1 >= F->rhi2) continue;
			if (F->arc && n1 <= F->rlo2) continue;
			if (norm(S[1] + (double)v * E[0][1]) > F->s2max) continue;
			if (norm(S[2] + (double)v * E[0][2]) > F->s2max) continue;
			if (F->arc) {   // arg(z·cb) ∈ [lo, hi) for a window of width < π around the real axis
				const cd zl = z1 * F->clo, zh = z1 * F->chi;
				if (zl.imag() < 0 || zh.imag() >= 0) continue;
			}
			Pt p; y[0] = v;
			for (int c = 0; c < 6; ++c) { long long s = a0[c]; for (int i = 0; i < 6; ++i) s += y[i] * U[i][c]; p.a[c] = (int)s; }
			cd z = 0; for (int i = 0; i < 6; ++i) z += Z1[i] * (double)p.a[i];
			p.z = z * s3f; p.e1 = norm(z); p.k = nkey(p.a); p.d1 = d1_of(p.k);
			out->push_back(p);
		}
	}
	void rec(int j, double dist, const cd* S) {
		double cj = pp[j];
		for (int i = j + 1; i < 6; ++i) cj -= R[j][i] * y[i];
		cj /= R[j][j];
		double rem = (R2 - dist) / (R[j][j] * R[j][j]);
		if (rem < 0) return;
		double w = sqrt(rem);
		if (j == 0) { leaf(S, cj, w); return; }
		long long lo = (long long)ceil(cj - w), hi = (long long)floor(cj + w);
		for (long long v = lo; v <= hi; ++v) {
			y[j] = v;
			double d = R[j][j] * (v - cj), nd = dist + d * d;
			if (nd > R2) continue;
			cd S2[3] = {S[0] + (double)v * E[j][0], S[1] + (double)v * E[j][1], S[2] + (double)v * E[j][2]};
			rec(j - 1, nd, S2);
		}
	}
};
static void fp_enum(cd d, double c0, double ht, double hr, double lam1, double lam2, double R2, int f, const Filt& flt, vector<Pt>& out) {
	const double s3f = 1.0 / pow3d(f), w2 = 1.0 / sqrt(lam2);
	ht *= sqrt(lam1); hr *= sqrt(lam1);
	double L[6][6];
	for (int i = 0; i < 6; ++i) {
		cd w = Z1[i] * conj(d) * s3f;
		L[i][0] = w.imag() / ht; L[i][1] = w.real() / hr;
		L[i][2] = Z2[i].real() * s3f * w2; L[i][3] = Z2[i].imag() * s3f * w2;
		L[i][4] = Z4[i].real() * s3f * w2; L[i][5] = Z4[i].imag() * s3f * w2;
	}
	double p[6] = {0, c0 / hr, 0, 0, 0, 0};
	double L0[6][6];
	for (int i = 0; i < 6; ++i) for (int c = 0; c < 6; ++c) L0[i][c] = L[i][c];
	FP fp;
	for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) fp.U[i][j] = (i == j);
	double Bs[6][6], mu[6][6], bn[6];
	auto gs = [&]() {
		for (int i = 0; i < 6; ++i) {
			for (int c = 0; c < 6; ++c) Bs[i][c] = L[i][c];
			for (int j = 0; j < i; ++j) {
				double s = 0; for (int c = 0; c < 6; ++c) s += L[i][c] * Bs[j][c];
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
			double q = nearbyint(mu[k][j]);
			if (q != 0) { for (int c = 0; c < 6; ++c) { L[k][c] -= q * L[j][c]; fp.U[k][c] -= (long long)q * fp.U[j][c]; } gs(); }
		}
		if (bn[k] >= (0.99 - mu[k][k - 1] * mu[k][k - 1]) * bn[k - 1]) ++k;
		else { for (int c = 0; c < 6; ++c) { swap(L[k][c], L[k - 1][c]); swap(fp.U[k][c], fp.U[k - 1][c]); } gs(); k = max(k - 1, 1); }
	}
	// The in-place float updates of L drift from U·L0 (U is exact). Rebuild L = U·L0 so the QR used by the
	// enumeration describes the true lattice (drift lost boundary points at f = 5).
	for (int i = 0; i < 6; ++i) for (int c = 0; c < 6; ++c) { double t = 0; for (int j = 0; j < 6; ++j) t += (double)fp.U[i][j] * L0[j][c]; L[i][c] = t; }
	gs();
	for (int j = 0; j < 6; ++j) {
		double nj = sqrt(bn[j]);
		for (int i = 0; i < 6; ++i) fp.R[j][i] = (i == j) ? nj : (i > j ? mu[i][j] * nj : 0.0);
		double s = 0; for (int c = 0; c < 6; ++c) s += Bs[j][c] * p[c];
		fp.pp[j] = s / nj;
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
	fp.R2 = R2; fp.F = &flt; fp.out = &out; fp.s3f = s3f;
	fp.rec(5, 0.0, S0);
	g_visits += fp.nvis;
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
	{ const long double P = 3.14159265358979323846264338327950288L;
	  g_c1 = cosl(2 * P / 9); g_c2 = cosl(4 * P / 9); g_c4 = cosl(8 * P / 9); g_c5 = cosl(10 * P / 9); }
	const double eps_cond = EPS * EPS / 8.0;
	const double thr = eps_cond * (1 + TOL), r = sqrt(thr);
	if (r >= 0.5) { cerr << "EPS too large for the arc geometry (r = " << r << ")" << endl; return 1; }
	const double PHI = PI / 18;
	const double m = asin(r) * (1 + 1e-6) + 1e-12;           // |arg − φ*| bound for a point within r of e^{iφ*}
	const double s2max = 2.0 * (1 + TOL);
	// x3: disk of radius r around 0, one representative per norm class.
	vector<Pt> X3;
	{
		Filt flt{false, cd(1, 0), cd(1, 0), 0, thr * (1 + 1e-9), s2max * (1 + 1e-9)};
		vector<Pt> pts;
		fp_enum(cd(1, 0), 0.0, r, r, 3.0, 6.0, 1.0 + 1e-6, f, flt, pts);   // disk ≤ 1 → 1/3, |σ_k|² ≤ 2 → 1/3 each
		for (auto& p : pts) if (norm(p.z) < thr) X3.push_back(p);
	}
	sort(X3.begin(), X3.end(), [](const Pt& A, const Pt& B) { return A.e1 < B.e1; });
	{ double dev = 0; for (auto& p : X3) dev = max(dev, fabs(p.d1 + nine_f - p.e1) / nine_f);
	  cerr << "[pf] d1 key check on x3: max |d1 + 9^f − |σ1|²| / 9^f = " << dev << endl;
	  if (dev > 1e-12) { cerr << "d1 key check FAILED" << endl; return 1; } }
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
	const double e1_slack = 1e-16 * nine_f + 1e-15;   // ≥ 100× the long-double error of d1 (~1e-18·9^f)
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
	auto arc_points = [&](cd base, double lo, double hi, vector<Pt>& out) {
		const double ph = 0.5 * (hi - lo), mid = 0.5 * (hi + lo);
		const cd d = base * polar(1.0, mid);
		const double rin = (1 - r) * cos(ph), rout = 1 + r;
		const double c0 = 0.5 * (rin + rout), hr = 0.5 * (rout - rin) * (1 + 1e-9) + 1e-15, ht = (1 + r) * sin(ph) * (1 + 1e-9) + 1e-15;
		Filt flt{true, conj(base) * polar(1.0, -(lo - 1e-9)), conj(base) * polar(1.0, -(hi + 1e-9)),
		         (1 - r) * (1 - r) - 1e-9, (1 + r) * (1 + r) + 1e-9, s2max * (1 + 1e-9)};
		vector<Pt> pts;
		fp_enum(d, c0, ht, hr, 6.0 * g_lamscale, 6.0 * g_lamscale, 1.0 + 1e-6, f, flt, pts);   // rectangle ≤ 2 → 1/3, |σ_k|² ≤ 2 → 1/3 each (min-volume weights)
		for (auto& p : pts) {
			double az = abs(p.z);
			if (az <= 1 - r || az >= 1 + r) continue;
			double ang = arg(p.z * conj(base));
			if (ang < lo || ang >= hi) continue;
			out.push_back(p);
		}
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

	// Work unit = (x2 chunk, block of groups): the x2 points of a chunk are θ-independent and shared by the block.
	const int nt = omp_get_max_threads();
	const int TB = max(1, min(G, (int)((long long)G * NC / (4 * nt))));
	const int NB = (G + TB - 1) / TB;
	#pragma omp parallel for schedule(dynamic, 1)
	for (long long unit = 0; unit < (long long)NC * NB; ++unit) {
		const int c = (int)(unit % NC), kb = (int)(unit / NC);
		const double lo = A0 + c * W, hi = (c == NC - 1) ? A1 : A0 + (c + 1) * W;
		double tA = omp_get_wtime();
		vector<Pt> X2;
		arc_points(cd(-1, 0), lo, hi, X2);
		sort(X2.begin(), X2.end(), [](const Pt& A, const Pt& B) { return A.d1 < B.d1; });
		vector<double> e2(X2.size());
		for (size_t j = 0; j < X2.size(); ++j) e2[j] = X2[j].d1;
		// Bucket index over e2 (O(1) window start instead of a binary search): bk[i] = first j with e2[j] >= emin + i·de.
		const int NBK = max(1, (int)X2.size());
		const double emin = e2.empty() ? 0 : e2.front(), de = e2.empty() ? 1 : max((e2.back() - emin) / NBK, 1e-300);
		vector<int> bk(NBK + 2);
		{ size_t j = 0; for (int i = 0; i <= NBK + 1; ++i) { const double v = emin + i * de; while (j < e2.size() && e2[j] < v) ++j; bk[i] = (int)j; } }
		auto first_ge = [&](double v) -> size_t {
			if (e2.empty() || v <= emin) return 0;
			long long i = (long long)((v - emin) / de);
			if (i > NBK) return e2.size();
			size_t j = bk[i];
			while (j < e2.size() && e2[j] < v) ++j;
			return j;
		};
		double tB = omp_get_wtime(), ux1 = 0, ujn = 0;
		for (int g = kb * TB; g < min(G, (kb + 1) * TB); ++g) {
		const vector<int>& grp = groups[g];
		const int ng = (int)grp.size();
		vector<cd> emth(ng);
		for (int q = 0; q < ng; ++q) emth[q] = polar(1.0, -0.5 * thetas[grp[q]]);
		const double span = half(grp.back()) - half(grp.front());
		double tC = omp_get_wtime();
		vector<Pt> X1;
		arc_points(conj(emth[0]), lo - 2 * m, hi + 2 * m + span, X1);
		double tD = omp_get_wtime(); ux1 += tD - tC;
		vector<vector<Cand>> loc(ng); long long np = 0; vector<long long> na(ng, 0);
		for (const Pt& p1 : X1) {
			const double tgt = -p1.d1;          // d1(x2) = −d1(x1) − |σ_1(x3)|², |σ_1(x3)|² ∈ [0, thr·9^f)
			const double elo = tgt - thr * nine_f * (1 + 1e-9) - e1_slack, ehi = tgt + e1_slack;
			for (size_t j2 = first_ge(elo); j2 < e2.size() && e2[j2] <= ehi; ++j2) {
				const Pt& p2 = X2[j2];
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

	cerr << "[pf] thread-seconds: x2 enum " << t_x2 << ", x1 enum " << t_x1 << ", join " << t_join << " | leaf visits " << g_visits.load() << endl;
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
