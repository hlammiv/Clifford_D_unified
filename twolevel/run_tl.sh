#!/bin/bash
# Two-level Householder targets (review C34): phase_free_tl at c1² ∈ {1, 0.7, 0.4, 0.1}, levels 2-10,
# 100 θ (first 100 Oct-3 angles), caps as prescribed_best/run_pf_local.sh; costs from tl_analyze.py (as found).
# Build: g++ -O3 -march=native -fopenmp ../hrsa/phase_free_tl.cpp ../hrsa/cyclotomic_int9.cpp ../hrsa/Z9chi.cpp -lquadmath -o ../hrsa/phase_free_tl
set -e
cd "$(dirname "$0")"
B=../hrsa/phase_free_tl
for c in 1.0 0.7 0.4 0.1; do
  export PF_C1SQ=$c
  for spec in "2 1 0.15" "4 2 0.012" "6 3 4e-4" "8 4 1.5e-5" "10 5 5.5e-7"; do
    set -- $spec; L=$1; f=$2; cap=$3
    eps=$(python3 -c "print('%.6g' % ($cap * 2 ** 0.5 * 1.0005))")
    d=pf_c${c}_l$L; rm -rf $d && mkdir $d
    OMP_NUM_THREADS=16 /usr/bin/time -f "c1sq $c level $L: %e s %M KB" $B thetas_100.txt $eps $f 1000 $d $cap > $d.log 2>&1
    tail -1 $d.log
    python3 tl_to_pool.py $d thetas_100.txt pp_c${c}_l$L > /dev/null
    python3 tl_analyze.py pp_c${c}_l$L --procs 16 --copies-n 0 --out costs_c${c}_l$L.csv > /dev/null
  done
done
echo ALL_DONE
