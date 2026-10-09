// multi_theta.cpp — HRSA candidate pools for MANY θ from ONE lattice enumeration.
//
// HRSA_bestD at level f enumerates the whole ball A = 8·3^{2f} of Z[ζ_9] numerators
// (entryEnumeration). Only the x_1 test depends on θ (|σ_1(x)/3^f − e^{iθ/2}|² < eps_cond);
// the x_2 (≈ −1) and x_3 (≈ 0) lists are θ-independent. At f = 4 the enumeration takes
// ~7 h on 8 threads per θ, so pools for N angles cost N× that. This tool enumerates once,
// keeps an x_1 list per θ, and then runs HRSA_bestD's pair search for each θ.
//
// Semantics match `HRSA_tester θ eps f --no-direct --min-f f --max-solns M --no-decompose`
// (k3 = 1, sequential x_1 order sorted by distance to target, x_2 sorted by distance to −1,
// x_1 mod-3 filter off). Validated against HRSA_tester pools at f = 3.
//
// Usage: multi_theta THETA_FILE EPS F MAX_SOLNS OUT_DIR
//   THETA_FILE: one θ per line (canonical convention, as HRSA_tester).
//   OUT_DIR/t{i:04d}.txt gets the CANDDUMP lines for θ_i (D field = 0, as --no-decompose),
//   read with prescribed_best/candpool.parse.
#include "cyclotomic_int9.h"
#include "Z9chi.h"
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

static const double FILTER_REL_TOL = 1e-6;   // as householder_search.cpp
static const double cos_vals[6] = {1.0, 0.766044443118978, 0.17364817766693041, -0.5, -0.9396926207859083, -0.9396926207859084};
static const double sin_vals[6] = {0.0, 0.6427876096865393, 0.984807753012208, 0.8660254037844387, 0.3420201433256689, -0.34202014332566866};

static int pow3(int n) { int p = 1; while (n-- > 0) p *= 3; return p; }

// ---------------------------------------------------------------------------------------
// Fincke–Pohst mode (2026-10-09). The ball enumeration above visits every point of the
// HRSA ball (~1e14 at f = 4) but only points with σ_1(x)/3^f near a target are ever used.
// Usable points also satisfy |σ_2(x)|², |σ_4(x)|² <= 2·9^f: for x_1, x_2 because the pair
// search requires s2(x_1)+s2(x_2) <= 2·9^f(1+tol) (likewise σ_4), and for x_3 because the
// exact norm identity gives |σ_k(x_3)|² = 2·9^f − |σ_k(x_1)|² − |σ_k(x_2)|². So we enumerate
// the integer points a ∈ Z^6 of the ellipsoid
//     |σ_1(a)/3^f − T|²/thr + |σ_2(a)/3^f|²/2 + |σ_4(a)/3^f|²/2 <= 3(1+tol)
// (a superset of the product of the three disks) with an LLL-reduced basis, then apply
// exactly HRSA's float filters. Same valid pairs as the ball enumeration; work ∝ output.
// ---------------------------------------------------------------------------------------
static const double PI = 3.14159265358979323846;
typedef array<int, 6> NKey;
// Real-subring element a + b(ζ+ζ⁻¹) + c(ζ²+ζ⁻²) has reduced power-basis coefficients
// (a, b−c, c−b, 0, −c, −b), so (coef0, coef4, coef5) determines it: a 3-int key.
struct K3 { int a, b, c; };
static inline uint64_t k3hash(int a, int b, int c) {
	uint64_t h = (uint64_t)(uint32_t)a * 0x9E3779B97F4A7C15ULL;
	h ^= (uint64_t)(uint32_t)b * 0xC2B2AE3D27D4EB4FULL; h = (h << 31) | (h >> 33);
	h ^= (uint64_t)(uint32_t)c * 0x165667B19E3779F9ULL;
	return h ^ (h >> 29);
}
struct FlatK3 {                       // open addressing, linear probing; value = class index
	vector<K3> keys; vector<int> vals; uint64_t mask = 0;
	void build(const vector<K3>& ks) {
		size_t cap = 16; while (cap < 2 * ks.size() + 16) cap <<= 1;
		keys.assign(cap, K3{0, 0, 0}); vals.assign(cap, -1); mask = cap - 1;
		for (int i = 0; i < (int)ks.size(); ++i) {
			uint64_t h = k3hash(ks[i].a, ks[i].b, ks[i].c) & mask;
			while (vals[h] >= 0) h = (h + 1) & mask;
			keys[h] = ks[i]; vals[h] = i;
		}
	}
	inline int find(int a, int b, int c) const {
		uint64_t h = k3hash(a, b, c) & mask;
		while (vals[h] >= 0) {
			const K3& k = keys[h];
			if (k.a == a && k.b == b && k.c == c) return vals[h];
			h = (h + 1) & mask;
		}
		return -1;
	}
};
struct NKeyHash { size_t operator()(const NKey& k) const { size_t h = 1469598103934665603ULL;
	for (int v : k) { h ^= (size_t)(unsigned)v; h *= 1099511628211ULL; } return h; } };

struct FP {
	double R[6][6];      // upper-triangular factor of the reduced basis (R[j][i], i >= j)
	double pp[6];        // target in the orthonormal frame
	long long U[6][6];   // reduced basis vector i = sum_j U[i][j] e_j (coefficient space)
	double R2;
	vector<array<int, 6>>* out;
	long long y[6];
	void rec(int j, double dist) {
		double cj = pp[j];
		for (int i = j + 1; i < 6; ++i) cj -= R[j][i] * y[i];
		cj /= R[j][j];
		double rem = (R2 - dist) / (R[j][j] * R[j][j]);
		if (rem < 0) return;
		double w = sqrt(rem);
		long long lo = (long long)ceil(cj - w), hi = (long long)floor(cj + w);
		for (long long v = lo; v <= hi; ++v) {
			y[j] = v;
			double d = R[j][j] * (v - cj);
			double nd = dist + d * d;
			if (nd > R2) continue;
			if (j == 0) {
				array<int, 6> a;
				for (int c = 0; c < 6; ++c) {
					long long s = 0;
					for (int i = 0; i < 6; ++i) s += y[i] * U[i][c];
					a[c] = (int)s;
				}
				out->push_back(a);
			} else rec(j - 1, nd);
		}
	}
};

// All a ∈ Z^6 in the ellipsoid around σ_1 target (tre, tim) (units of 3^f).
static void fp_enumerate(double tre, double tim, double thr, int f, vector<array<int, 6>>& out) {
	const double s3f = 1.0 / pow3(f);
	double L[6][6];      // L[i] = column i = embedding of ζ^i, weighted
	const double w1 = 1.0 / sqrt(thr), w2 = 1.0 / sqrt(2.0);
	for (int i = 0; i < 6; ++i) {
		L[i][0] = cos(2 * PI * i / 9) * s3f * w1; L[i][1] = sin(2 * PI * i / 9) * s3f * w1;
		L[i][2] = cos(2 * PI * 2 * i / 9) * s3f * w2; L[i][3] = sin(2 * PI * 2 * i / 9) * s3f * w2;
		L[i][4] = cos(2 * PI * 4 * i / 9) * s3f * w2; L[i][5] = sin(2 * PI * 4 * i / 9) * s3f * w2;
	}
	double p[6] = {tre * w1, tim * w1, 0, 0, 0, 0};
	FP fp;
	for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) fp.U[i][j] = (i == j);
	// LLL (δ = 0.99) on the columns, Gram–Schmidt recomputed after each change (6-D: cheap).
	double Bs[6][6], mu[6][6], bn[6];
	auto gs = [&]() {
		for (int i = 0; i < 6; ++i) {
			for (int c = 0; c < 6; ++c) Bs[i][c] = L[i][c];
			for (int j = 0; j < i; ++j) {
				double d = 0; for (int c = 0; c < 6; ++c) d += L[i][c] * Bs[j][c];
				mu[i][j] = d / bn[j];
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
			if (q != 0) {
				for (int c = 0; c < 6; ++c) { L[k][c] -= q * L[j][c]; fp.U[k][c] -= (long long)q * fp.U[j][c]; }
				gs();
			}
		}
		if (bn[k] >= (0.99 - mu[k][k - 1] * mu[k][k - 1]) * bn[k - 1]) ++k;
		else {
			for (int c = 0; c < 6; ++c) { swap(L[k][c], L[k - 1][c]); swap(fp.U[k][c], fp.U[k - 1][c]); }
			gs();
			k = max(k - 1, 1);
		}
	}
	// QR: R[j][j] = |b*_j|, R[j][i] = mu[i][j]·|b*_j|; pp_j = <b*_j/|b*_j|, p>.
	for (int j = 0; j < 6; ++j) {
		double nj = sqrt(bn[j]);
		for (int i = 0; i < 6; ++i) fp.R[j][i] = (i == j) ? nj : (i > j ? mu[i][j] * nj : 0.0);
		double d = 0; for (int c = 0; c < 6; ++c) d += Bs[j][c] * p[c];
		fp.pp[j] = d / nj;
	}
	fp.R2 = 3.0 * (1.0 + FILTER_REL_TOL) * (1.0 + 1e-9) + 1e-9;
	fp.out = &out;
	fp.rec(5, 0.0);
}

int main(int argc, char** argv) {
	if (argc != 6 && argc != 7) {
		cerr << "usage: multi_theta THETA_FILE EPS F MAX_SOLNS OUT_DIR [fp]" << endl;
		return 1;
	}
	vector<double> thetas;
	{ ifstream in(argv[1]); double t; while (in >> t) thetas.push_back(t); }
	const double epsilon = atof(argv[2]);
	const int f = atoi(argv[3]);
	const int max_solns = atoi(argv[4]);
	const string out_dir = argv[5];
	const bool use_fp = (argc == 7 && string(argv[6]) == "fp");
	if (f > 7) { cerr << "f > 7 would overflow the int ring arithmetic" << endl; return 1; }
	const int K = (int)thetas.size();
	const double c = 1.0;                       // HRSA_tester default
	const double eps_cond = epsilon * epsilon / (8.0 * c * c);
	const double thr = eps_cond * (1.0 + FILTER_REL_TOL);
	const double inv_3f = 1.0 / pow3(f);
	const int f_pow_sq = pow3(f) * pow3(f);
	const int A = 8 * f_pow_sq;
	vector<double> Tr(K), Ti(K);
	for (int k = 0; k < K; ++k) { Tr[k] = cos(thetas[k] / 2); Ti[k] = sin(thetas[k] / 2); }
	// x_1 annulus pre-test: |z| within sqrt(thr) of 1 is necessary for any θ.
	const double r_lo = 1.0 - sqrt(thr), r_hi = 1.0 + sqrt(thr);

	const int nt = omp_get_max_threads();
	vector<vector<vector<ringZ9>>> tl_x1(nt, vector<vector<ringZ9>>(K));
	vector<vector<ringZ9>> tl_x2(nt);
	vector<unordered_map<int, vector<ringZ9>>> tl_lookup(nt);

	const int max_a3 = (int)ceil(sqrt(A / 3.0));
	const int total_a3 = 2 * max_a3 + 1;
	atomic<int> progress(0);
	cerr << "[multi] K=" << K << " f=" << f << " eps=" << epsilon << " threads=" << nt << endl;

	if (use_fp) {
		// Same per-point float filters as the ball loop below, plus the σ_2/σ_4 bound.
		const double s2max = 2.0 * (1.0 + FILTER_REL_TOL);   // |σ_k/3^f|² bound
		auto sig = [&](const array<int, 6>& a, int m, double& re, double& im) {
			re = 0; im = 0;
			for (int i = 0; i < 6; ++i) { re += cos(2 * PI * m * i / 9) * a[i]; im += sin(2 * PI * m * i / 9) * a[i]; }
			re *= inv_3f; im *= inv_3f;
		};
		auto conj_ok = [&](const array<int, 6>& a) {
			double r, i;
			sig(a, 2, r, i); if (r * r + i * i > s2max) return false;
			sig(a, 4, r, i); if (r * r + i * i > s2max) return false;
			return true;
		};
		auto h_re_im = [&](const array<int, 6>& a, double& re, double& im) {   // HRSA's table form
			re = 0; im = 0;
			for (int k2 = 0; k2 < 6; ++k2) { re += cos_vals[k2] * a[k2]; im += sin_vals[k2] * a[k2]; }
			re *= inv_3f; im *= inv_3f;
		};
		auto to_ring = [](const array<int, 6>& a) { int arr[9] = {a[0], a[1], a[2], a[3], a[4], a[5], 0, 0, 0}; return ringZ9(arr); };
		size_t visited = 0;
		{   // x_2 (≈ −1) and x_3 (≈ 0)
			vector<array<int, 6>> pts;
			fp_enumerate(-1.0, 0.0, thr, f, pts); visited += pts.size();
			for (auto& a : pts) { double re, im; h_re_im(a, re, im);
				if ((re + 1.0) * (re + 1.0) + im * im < thr && conj_ok(a)) tl_x2[0].push_back(to_ring(a)); }
			pts.clear();
			fp_enumerate(0.0, 0.0, thr, f, pts); visited += pts.size();
			for (auto& a : pts) { double re, im; h_re_im(a, re, im);
				if (re * re + im * im < thr && conj_ok(a)) { ringZ9 x = to_ring(a); tl_lookup[0][x.quad()].push_back(x); } }
		}
		#pragma omp parallel for schedule(dynamic) reduction(+:visited)
		for (int k = 0; k < K; ++k) {
			vector<array<int, 6>> pts;
			fp_enumerate(Tr[k], Ti[k], thr, f, pts); visited += pts.size();
			for (auto& a : pts) { double re, im; h_re_im(a, re, im);
				double dx = re - Tr[k], dy = im - Ti[k];
				if (dx * dx + dy * dy < thr && conj_ok(a)) tl_x1[omp_get_thread_num()][k].push_back(to_ring(a)); }
		}
		cerr << "[fp] ellipsoid points visited: " << visited << endl;
	}

	#pragma omp parallel for schedule(dynamic)
	for (int a3 = -max_a3; a3 <= max_a3; ++a3) {
		if (use_fp) continue;
		int p = ++progress;
		if (p % max(1, total_a3 / 40) == 0) cerr << "[enum] a3-progress=" << p << "/" << total_a3 << endl;
		const int tid = omp_get_thread_num();
		int budget_a3 = A - 3 * a3 * a3;
		if (budget_a3 < 0) continue;
		int max_a4 = (int)ceil(sqrt(budget_a3));
		for (int a4 = -max_a4; a4 <= max_a4; ++a4) {
			int budget_a4 = budget_a3 - 3 * a4 * a4;
			if (budget_a4 < 0) continue;
			int max_a5 = (int)ceil(sqrt(budget_a4));
			for (int a5 = -max_a5; a5 <= max_a5; ++a5) {
				int budget_a5 = budget_a4 - 3 * a5 * a5;
				if (budget_a5 < 0) continue;
				// Triangle-inequality pruning, as entryEnumeration, over all K x_1 cones.
				{
					double sqrt_bu = sqrt((double)budget_a5);
					double m0 = 0.5 * (sqrt_bu + abs((double)a3));
					double m1 = 0.5 * (sqrt_bu + abs((double)a4));
					double m2 = 0.5 * (sqrt_bu + abs((double)a5));
					double re_p = (cos_vals[3] * a3 + cos_vals[4] * a4 + cos_vals[5] * a5) * inv_3f;
					double im_p = (sin_vals[3] * a3 + sin_vals[4] * a4 + sin_vals[5] * a5) * inv_3f;
					double mre = m0 + abs(cos_vals[1]) * m1 + abs(cos_vals[2]) * m2;
					double mim = abs(sin_vals[1]) * m1 + abs(sin_vals[2]) * m2;
					double max_rem = sqrt(mre * mre + mim * mim) * inv_3f;
					auto ok = [&](double tr, double ti) {
						double d = sqrt((re_p - tr) * (re_p - tr) + (im_p - ti) * (im_p - ti)) - max_rem;
						return d <= 0.0 || d * d <= thr;
					};
					bool any = ok(-1.0, 0.0) || ok(0.0, 0.0);
					for (int k = 0; k < K && !any; ++k) any = ok(Tr[k], Ti[k]);
					if (!any) continue;
				}
				int max_b = (int)ceil(sqrt(budget_a5));
				// Plain parity-stride loops (visit order is irrelevant: everything is collected).
				int b0s = -max_b; if (((b0s + a3) % 2 + 2) % 2) ++b0s;
				for (int b0 = b0s; b0 <= max_b; b0 += 2) {
					int budget_b0 = budget_a5 - b0 * b0;
					if (budget_b0 < 0) continue;
					int a0 = (b0 + a3) / 2;
					int max_b1 = (int)ceil(sqrt(budget_b0));
					int b1s = -max_b1; if (((b1s + a4) % 2 + 2) % 2) ++b1s;
					for (int b1 = b1s; b1 <= max_b1; b1 += 2) {
						int budget_b1 = budget_b0 - b1 * b1;
						if (budget_b1 < 0) continue;
						int a1 = (b1 + a4) / 2;
						int max_b2 = (int)ceil(sqrt(budget_b1));
						int b2s = -max_b2; if (((b2s + a5) % 2 + 2) % 2) ++b2s;
						for (int b2 = b2s; b2 <= max_b2; b2 += 2) {
							int a2 = (b2 + a5) / 2;
							int arr[9] = {a0, a1, a2, a3, a4, a5, 0, 0, 0};
							double re = 0.0, im = 0.0;
							for (int k = 0; k < 6; ++k) { re += cos_vals[k] * arr[k]; im += sin_vals[k] * arr[k]; }
							re *= inv_3f; im *= inv_3f;
							double abs_sq = re * re + im * im;
							if (abs_sq < thr) tl_lookup[tid][ringZ9(arr).quad()].push_back(ringZ9(arr));
							if ((re + 1.0) * (re + 1.0) + im * im < thr) tl_x2[tid].push_back(ringZ9(arr));
							if (abs_sq > r_lo * r_lo && abs_sq < r_hi * r_hi) {
								for (int k = 0; k < K; ++k) {
									double dx = re - Tr[k], dy = im - Ti[k];
									if (dx * dx + dy * dy < thr) tl_x1[tid][k].push_back(ringZ9(arr));
								}
							}
						}
					}
				}
			}
		}
	}

	// Merge.
	vector<vector<ringZ9>> x1(K);
	vector<ringZ9> x2;
	unordered_map<int, vector<ringZ9>> lookup;
	for (int t = 0; t < nt; ++t) {
		for (int k = 0; k < K; ++k) x1[k].insert(x1[k].end(), tl_x1[t][k].begin(), tl_x1[t][k].end());
		x2.insert(x2.end(), tl_x2[t].begin(), tl_x2[t].end());
		for (auto& kv : tl_lookup[t]) { auto& d = lookup[kv.first]; d.insert(d.end(), kv.second.begin(), kv.second.end()); }
	}
	// Same orderings as entryEnumeration (stable across thread merge order).
	auto lexless = [](const ringZ9& a, const ringZ9& b) {
		for (int i = 0; i < 6; ++i) if (a.getTerm(i) != b.getTerm(i)) return a.getTerm(i) < b.getTerm(i);
		return false;
	};
	complex<double> neg_one(-1.0, 0.0);
	stable_sort(x2.begin(), x2.end(), lexless);
	stable_sort(x2.begin(), x2.end(), [&](const ringZ9& a, const ringZ9& b) {
		return abs(a.toComplexDouble() - neg_one) < abs(b.toComplexDouble() - neg_one); });
	size_t n1 = 0; for (auto& v : x1) n1 += v.size();
	cerr << "[multi] enumeration done: x1 total " << n1 << ", x2 " << x2.size() << ", lookup buckets " << lookup.size() << endl;

	const int N2 = (int)x2.size();
	vector<double> x2_s2(N2), x2_s4(N2);
	for (int j = 0; j < N2; ++j) { x2_s2[j] = x2[j].GaloisAut(2).abs_val_sq(); x2_s4[j] = x2[j].GaloisAut(4).abs_val_sq(); }
	const double conj_thr = 2.0 * (double)f_pow_sq * (1.0 + FILTER_REL_TOL);
	const ringZ9 two_fsq(2 * f_pow_sq);
	// Norm classes for the fp-mode join: N(x) = x·x̄ (reduced, 6 coefficients) as hash key.
	auto nkey = [](const ringZ9& x) { ringZ9 n = x.complexConj() * x; n.reduce(); NKey k; for (int i = 0; i < 6; ++i) k[i] = n.getTerm(i); return k; };
	unordered_map<NKey, vector<int>, NKeyHash> g2, g3;
	vector<ringZ9> x3_all;
	FlatK3 t2; vector<vector<int>> c2_members;
	vector<K3> c3_keys; vector<vector<int>> c3_members; vector<double> c3_d;
	if (use_fp) {
		for (int j = 0; j < N2; ++j) g2[nkey(x2[j])].push_back(j);
		for (auto& kv : lookup) for (auto& x : kv.second) { g3[nkey(x)].push_back((int)x3_all.size()); x3_all.push_back(x); }
		cerr << "[fp] norm classes: x2 " << g2.size() << ", x3 " << g3.size() << endl;
		vector<K3> ks;
		for (auto& kv : g2) { ks.push_back(K3{kv.first[0], kv.first[4], kv.first[5]}); c2_members.push_back(kv.second); }
		t2.build(ks);
		vector<pair<double, int>> order;                      // x_3 classes by d3 = |x3/3^f|²
		vector<K3> k3s; vector<vector<int>> m3;
		for (auto& kv : g3) {
			k3s.push_back(K3{kv.first[0], kv.first[4], kv.first[5]}); m3.push_back(kv.second);
			order.push_back({norm(x3_all[kv.second[0]].toComplexDouble() * inv_3f), (int)k3s.size() - 1});
		}
		sort(order.begin(), order.end());
		for (auto& o : order) { c3_keys.push_back(k3s[o.second]); c3_members.push_back(m3[o.second]); c3_d.push_back(o.first * (1.0 - 1e-9)); }
	}

	#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < K; ++k) {
		complex<double> ang(Tr[k], Ti[k]);
		vector<ringZ9>& X1 = x1[k];
		stable_sort(X1.begin(), X1.end(), lexless);
		stable_sort(X1.begin(), X1.end(), [&](const ringZ9& a, const ringZ9& b) {
			return abs(a.toComplexDouble() - ang) < abs(b.toComplexDouble() - ang); });
		vector<array<ringZ9chi, 3>> sols;
		if (use_fp) {
			// Norm join (fp mode). A triple is valid iff N(x1)+N(x2)+N(x3) = 2·9^f exactly, with
			// N(x) = x·x̄ in the real subring; the q- and σ_2/σ_4-sum pre-filters of the loop below
			// are implied by it. All x_3 with the same N give the same ε (|x3|² = σ_1 of N(x3)),
			// so HRSA's "first passing x_3 per pair" (k3 = 1) selects the same (x_1, x_2) pairs.
			unordered_map<NKey, vector<int>, NKeyHash> g1;
			for (int i = 0; i < (int)X1.size(); ++i) g1[nkey(X1[i])].push_back(i);
			const int C0 = 2 * f_pow_sq;
			for (auto& e1 : g1) {
				if ((int)sols.size() >= max_solns) break;
				const NKey& n1 = e1.first;
				// ε >= d1 + d3 (+ d2 >= 0): x_3 classes sorted by d3, stop once d3 >= eps_cond − min d1.
				double d1min = 1e300;
				for (int i : e1.second) d1min = min(d1min, norm(X1[i].toComplexDouble() * inv_3f - ang));
				const double d3max = eps_cond - d1min;
				for (int c3 = 0; c3 < (int)c3_keys.size() && c3_d[c3] < d3max; ++c3) {
					const K3& n3 = c3_keys[c3];
					int c2 = t2.find(C0 - n1[0] - n3.a, -n1[4] - n3.b, -n1[5] - n3.c);
					if (c2 < 0) continue;
					for (int i : e1.second) {
						const ringZ9& x_1 = X1[i];
						for (int j : c2_members[c2]) {
							const ringZ9& x_2 = x2[j];
							for (int t : c3_members[c3]) {
								const ringZ9& x_3 = x3_all[t];
								array<ringZ9chi, 3> cand = {ringZ9chi(x_1, f), ringZ9chi(x_2, f), ringZ9chi(x_3, f)};
								complex<double> z = cand[0].toComplexDouble();
								double e = norm(z - ang) + (cand[1] + ringZ9chi(ringZ9(1), 0)).abs_val_sq() + cand[2].abs_val_sq();
								if (e < eps_cond) { sols.push_back(cand); break; }
							}
						}
					}
				}
			}
		}
		for (size_t i = 0; i < X1.size() && !use_fp && (int)sols.size() < max_solns; ++i) {
			const ringZ9& x_1 = X1[i];
			int q1 = x_1.quad();
			double s2 = x_1.GaloisAut(2).abs_val_sq(), s4 = x_1.GaloisAut(4).abs_val_sq();
			for (int j = 0; j < N2 && (int)sols.size() < max_solns; ++j) {
				const ringZ9& x_2 = x2[j];
				int q2 = x_2.quad();
				if (q1 + q2 > 2 * f_pow_sq) continue;
				if (s2 + x2_s2[j] > conj_thr) continue;
				if (s4 + x2_s4[j] > conj_thr) continue;
				auto it = lookup.find(2 * f_pow_sq - q1 - q2);
				if (it == lookup.end()) continue;
				ringZ9 x3sq = two_fsq - x_1.complexConj() * x_1 - x_2.complexConj() * x_2;
				for (const ringZ9& x_3 : it->second) {          // k3 = 1: first valid x_3 per pair
					if (!(x_3.complexConj() * x_3 == x3sq)) continue;
					array<ringZ9chi, 3> cand = {ringZ9chi(x_1, f), ringZ9chi(x_2, f), ringZ9chi(x_3, f)};
					complex<double> z = cand[0].toComplexDouble();
					double e = norm(z - ang) + (cand[1] + ringZ9chi(ringZ9(1), 0)).abs_val_sq() + cand[2].abs_val_sq();
					if (e < eps_cond) { sols.push_back(cand); break; }
				}
			}
		}
		char path[4096];
		snprintf(path, sizeof path, "%s/t%04d.txt", out_dir.c_str(), k);
		FILE* fo = fopen(path, "w");
		for (size_t s = 0; s < sols.size(); ++s) {
			ringZ9chi c1 = sols[s][0], c2 = sols[s][1], c3 = sols[s][2];
			c1.reduce(); c2.reduce(); c3.reduce();
			auto z1 = c1.toComplexDouble(), z2 = c2.toComplexDouble(), z3 = c3.toComplexDouble();
			fprintf(fo, "CANDDUMP %zu 0 %.17g %.17g %.17g %.17g %.17g %.17g %d", s, z1.real(), z1.imag(),
			        z2.real(), z2.imag(), z3.real(), z3.imag(), f);
			for (const ringZ9chi* cc : {&c1, &c2, &c3}) {
				ringZ9 n = cc->getNumerator();
				for (int t = 0; t < 6; ++t) fprintf(fo, " %d", n.getTerm(t));
			}
			fprintf(fo, " E %d %d %d\n", c1.getExp(), c2.getExp(), c3.getExp());
		}
		fclose(fo);
	}
	cerr << "[multi] DONE" << endl;
	return 0;
}
