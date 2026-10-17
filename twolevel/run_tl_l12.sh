#!/bin/bash
# Level 12 for the two-level study: 50 θ (first 50 of thetas_100), c1² ∈ {0.1, 0.4, 0.7}; cap as prescribed_best
# run_pf_lenore.sh. Control c1² = 1 is prescribed_best/pfcosts_l12.csv (same θ).
set -e
cd "$(dirname "$0")"
B=../hrsa/phase_free_tl
for c in 0.4 0.1 0.7; do
  export PF_C1SQ=$c
  d=pf_c${c}_l12; rm -rf $d && mkdir $d
  OMP_NUM_THREADS=16 /usr/bin/time -f "c1sq $c level 12: %e s %M KB" $B thetas_50.txt 2.5468e-8 6 1000 $d 1.8e-8 > $d.log 2>&1
  tail -1 $d.log
  python3 tl_to_pool.py $d thetas_50.txt pp_c${c}_l12 > /dev/null
  python3 tl_analyze.py pp_c${c}_l12 --procs 16 --copies-n 0 --out costs_c${c}_l12.csv > /dev/null
done
echo ALL_DONE
