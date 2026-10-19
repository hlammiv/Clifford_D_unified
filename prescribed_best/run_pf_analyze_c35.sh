#!/bin/bash
# Re-analysis of all phase-free pools after the C35 rerun (stabilized acceptance filter):
# 324 symmetric copies for the first 100 θ at f ≤ 12 and for every θ at f ≥ 14 (as in the paper).
set -e
cd "$(dirname "$0")"
for L in 2 4 6 8 10 12 14 16; do
  case $L in 14) C=200;; 16) C=350;; *) C=100;; esac
  /usr/bin/time -f "analyze level $L: %e s" python3 analyze_pools.py pfp_l$L --procs 16 --copies-n $C --out pfcosts_l$L.csv
done
echo ANALYZE_C35_DONE
