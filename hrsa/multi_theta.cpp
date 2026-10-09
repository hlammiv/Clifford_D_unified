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

int main(int argc, char** argv) {
	if (argc != 6) {
		cerr << "usage: multi_theta THETA_FILE EPS F MAX_SOLNS OUT_DIR" << endl;
		return 1;
	}
	vector<double> thetas;
	{ ifstream in(argv[1]); double t; while (in >> t) thetas.push_back(t); }
	const double epsilon = atof(argv[2]);
	const int f = atoi(argv[3]);
	const int max_solns = atoi(argv[4]);
	const string out_dir = argv[5];
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

	#pragma omp parallel for schedule(dynamic)
	for (int a3 = -max_a3; a3 <= max_a3; ++a3) {
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

	#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < K; ++k) {
		complex<double> ang(Tr[k], Ti[k]);
		vector<ringZ9>& X1 = x1[k];
		stable_sort(X1.begin(), X1.end(), lexless);
		stable_sort(X1.begin(), X1.end(), [&](const ringZ9& a, const ringZ9& b) {
			return abs(a.toComplexDouble() - ang) < abs(b.toComplexDouble() - ang); });
		vector<array<ringZ9chi, 3>> sols;
		for (size_t i = 0; i < X1.size() && (int)sols.size() < max_solns; ++i) {
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
