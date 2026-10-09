# prescribed_best/ — best fits at prescribed (generic) θ, levels 2, 4, 6

Goal: put the T-cost headline on **generic angles**. Gustafson's C+R fits use generic angles. The C+D headline uses Nick's special-θ matrices, where each matrix picks its own best angle.

## Method
- **Angles:** `rng(20261003).uniform(0, 2π, n)`, the same as the Oct-3 sweep. n = 500 at levels 2 and 4, 150 at level 6.
- **`run_pool.py`:** runs HRSA once per θ, at a loose target, keeping every candidate:
  `HRSA_tester θ target max_f --no-direct --min-f max_f --max-solns 200000 --no-decompose`
  - The best fit at that level is the candidate with the smallest ε in the pool.
  - Targets: 0.6 (level 2), 0.05 (level 4), 0.003 (level 6).
  - This replaces target-ε bisection, which under-reports the best fit, because HRSA's search region depends on the target.
- **`analyze_pools.py`:**
  - rebuilds every candidate exactly (exact unitarity check; ε recomputed exactly);
  - decomposes with the fast reducer;
  - records T in three models: unitary 7-T, merged, measurement 4-T;
  - computes best-of-K over candidates within 1.25× of the best ε;
  - computes the best of the 324 ±ζ symmetric copies (first 100 θ at levels 2 and 4; all 150 at level 6).
- **`fit_generic.py`:** combines these with Nick's generic f=8,10 fits (`../nick_fits25/`) and compares with the special-θ data level by level.
- Pools are in `pools_l*.tar.gz`, per-θ summaries in `index_l*.csv`, costs in `costs_l*.csv`, and the fit output in `fit_generic_output.txt`.

## multi_theta (hrsa/multi_theta.cpp): one enumeration for all θ
HRSA's level-f enumeration covers the whole lattice ball and is θ-independent except for the x_1 test, so
`multi_theta THETA_FILE EPS F MAX_SOLNS OUT_DIR` enumerates once and runs the pair search per θ
(same semantics as `HRSA_tester ... --no-direct --min-f F --no-decompose`, k3 = 1).
Validated: identical (x_1, x_2) candidate sets for all 150 level-6 angles (12,677 candidates),
in 34 s on 8 threads vs ~45 min on 26 HRSA workers. `mt_to_pool.py` converts its output to pool format.
Level 6 is now 500 θ via multi_theta (costs_l6.csv; the original 150-θ HRSA run is costs_l6_hrsa150.csv).
Level 8 (500 θ, target 3e-4) runs on Lenore: ~7 h on 8 threads; `overnight_l8.sh` pulls and analyzes.

## HRSA changes made for this (2026-10-08)
Default behaviour is unchanged.
1. **`--no-decompose`** is now implemented. Before, it was silently ignored, as is any unknown flag. Skipping decomposition is ~1000× faster for pool collection.
2. **`--min-f N`** searches only level N. Otherwise HRSA stops at the first level with any candidate.
3. **Phase 0.5** (the signed-Clifford shortcut) is skipped when `--min-f > 0`. At a loose target it ended the run before the requested level for about 80% of angles.
4. **CANDDUMP prints reduced numerators**, and appends each element's denominator exponent `E e1 e2 e3`.
   - Numerators are printed at each element's own exponent, which can differ from f.
   - The level-6 pools predate the `E` field. For those, `analyze_pools.rebuild` recovers the exponents by requiring exact unitarity and the printed ε. **0 of ~10⁵ candidates were unrecoverable.**

## Results (best symmetric copy, mean T per θ)

| Level | n | median ε, generic | median ε, special (Nick) | ratio | unitary T, gen / spec | merged | measurement |
|---|---|---|---|---|---|---|---|
| 2 | 500 | 9.2e-2 | – | – | 20.6 / – | 15.8 | 13.0 |
| 4 | 500 | 6.2e-3 | 5.2e-3 | 1.2 | 56.2 / 51.8 | 42.4 / 40.2 | 36.2 / 34.8 |
| 6 | 150 | 4.8e-4 | 1.4e-4 | 3.5 | 80.0 / 79.1 | 61.2 / 61.2 | 53.3 / 52.9 |
| 8 (Nick) | 25 | 3.9e-5 | 2.8e-6 | 13.7 | 118.1 / 111.2 | 87.6 / 83.5 | 76.6 / 73.7 |
| 10 (Nick) | 25 | 3.8e-6 | 2.2e-7 | 17.2 | 142.2 / 139.9 | 105.0 / 104.1 | 93.2 / 91.2 |

- **At a fixed level, generic and special θ cost the same T,** within 1–6%.
- **The difference is entirely in ε.** The gap widens from 1.2× at level 4 to 17× at level 10.

**Pooled fit** (all (θ, level) points, best copy, N_T = a + b·log₃(1/ε)):

| Model | Generic θ | At 1e-10 | Special-θ headline | At 1e-10 |
|---|---|---|---|---|
| Unitary | −7.3 + 12.8·log₃ | 261 | 2.5 + 9.71·log₃ | 206 |
| Merged | −4.5 + 9.50·log₃ | 195 | 3.3 + 7.21·log₃ | 154 |
| Measurement | −5.4 + 8.50·log₃ | 173 | 1.8 + 6.42·log₃ | 136 |

## Caveats
- **Mixed sources.** Levels 2–6 are HRSA pools; levels 8–10 are Nick's search, at 25 θ each.
- **Low levels dominate the point count,** so the slope extrapolated to 1e-10 (about level 20) is uncertain.
- **The ε gap is still growing at level 10.** Whether it saturates decides the true slope. That needs generic data at levels 12–16 from Nick, and our own level 8 to cross-check his.
