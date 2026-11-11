#!/bin/bash
# Pools + costs for the C35 rerun of the full-Householder cells (run_full.sh with PFF_NOANALYZE=1 skipped these).
# Writes pp_{tag}_c35/ and c35_costs/costs_{tag}.csv (kept out of the paper's costs_{fam}_l*.csv glob until compared).
set -e
cd "$(dirname "$0")"
mkdir -p c35_costs
NT=${NT:-16}
for d in pf_*_c35; do
  tag=${d#pf_}; tag=${tag%_c35}                      # e.g. bal_l8, skew_l12b
  fam=${tag%%_l*}; lev=${tag#*_l}                     # bal, 8 / skew, 12b
  case $lev in *a|*b) tf=targets_${fam}_l${lev}.txt;; *) tf=targets_${fam}.txt;; esac
  read -r _ c1 c3 < <(grep "^$fam " settings.txt)
  [ -s c35_costs/costs_${tag}.csv ] && { echo "skip $tag"; continue; }
  PF_C1SQ=$c1 PF_C3SQ=$c3 python3 full_to_pool.py $d $tf pp_${tag}_c35 > /dev/null
  PF_C1SQ=$c1 PF_C3SQ=$c3 /usr/bin/time -f "$tag: %e s" python3 full_analyze.py pp_${tag}_c35 --procs $NT --copies-n 0 --out c35_costs/costs_${tag}.csv > /dev/null
done
echo FULL_C35_DONE
