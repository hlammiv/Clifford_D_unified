#!/bin/bash
# Phase-free pools, levels 2-10, 500 generic θ (thetas_500.txt = Oct-3 angles). EPS = √2·cap (exact for
# the phase-optimal acceptance), cap ≈ 1.7× the expected median best (no empties expected).
set -e
cd "$(dirname "$0")"
B=../hrsa/phase_free
for spec in "2 1 0.15" "4 2 0.012" "6 3 4e-4" "8 4 1.5e-5" "10 5 5.5e-7"; do
  set -- $spec; L=$1; f=$2; cap=$3
  eps=$(python3 -c "print('%.6g' % ($cap * 2 ** 0.5 * 1.0005))")
  rm -rf pf_l$L && mkdir pf_l$L
  OMP_NUM_THREADS=16 /usr/bin/time -f "level $L: %e s %M KB" $B thetas_500.txt $eps $f 1000 pf_l$L $cap > pf_l$L.log 2>&1
  tail -1 pf_l$L.log
  python3 mt_to_pool.py pf_l$L thetas_500.txt pfp_l$L > /dev/null
done
echo ALL_DONE
