# Nick's best fits at 25 generic θ (f = 8, 10), received 2026-10-08

- **Files:** `../fits25_f=8.txt.gz`, `../fits25_f=10.txt.gz`. No `#f=` header; one row per θ; groups stored column-major.
- **Analysis:** `analyze_fits25.py` → `fits25_results.csv` (exact unitarity, ε, Nick's diag/off-diag max errors, T-costs as given and best of 324 ±ζ copies).

## Finding: the special-θ headline is optimistic
- **The headline data are special-θ.** `nick_test/fits_f=*.txt` are matrices whose θ is the matrix's own best-fit angle, which selects for small ε.
- **Generic θ** (these 25) have **~15× larger ε at the same f**: median 3.9e-5 vs 2.8e-6 at f = 8, and 3.8e-6 vs 2.2e-7 at f = 10. The T-cost is about the same.

  | f | θ set | median ε | T_unit | T_merged | T_meas | best-copy unit / merged / meas |
  |---|---|---|---|---|---|---|
  | 8 | generic 25 | 3.86e-5 | 132.6 | 97.5 | 84.9 | 118.1 / 87.6 / 76.6 |
  | 10 | generic 25 | 3.78e-6 | 158.8 | 116.3 | 102.5 | 142.2 / 105.0 / 93.2 |
  | 8 | special | 2.83e-6 | 125.9 | | | |
  | 10 | special | 2.20e-7 | 148.8 | | | |

- **Same ε comparison:** at the same ε, generic θ costs ~22–28 T more (unitary) than the special-θ fit predicts.
- **Two-point slope estimates** (f = 8 → 10; uncertain): unitary ~12.4, merged ~8.9, meas ~8.3 per log₃. Special θ gives 10.0 / 7.2 / 6.4.
- **Why this matters:** Gustafson's C+R fits use uniformly sampled (generic) θ, so the like-for-like C+D numbers must use generic θ too.
- **Needed:** generic-θ fits at f = 12, 14, 16 (ideally ~100 uniform θ), with Nick's ε window large enough that every fit is confirmed.
- **Nick's caveat:** θ = 0.549968259619878 is "unconfirmed" at f = 10. Its diag error is 9.6e-6, above his search ε of 5e-6.
