#!/bin/bash
# Gate costs of the phase-free pools (best fit, best-of-K at 1.25×, 324 symmetric copies for the first 100 θ).
set -e
cd "$(dirname "$0")"
for L in "$@"; do
  /usr/bin/time -f "analyze level $L: %e s" python3 analyze_pools.py pfp_l$L --procs 16 --copies-n 100 --out pfcosts_l$L.csv
done
echo ANALYZE_DONE
