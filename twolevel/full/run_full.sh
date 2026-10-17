#!/bin/bash
# Full Householder targets (review C34): one (setting, level, target file) cell of phase_free_full + pool + costs.
# Usage: run_full.sh SETTING LEVEL TARGET_FILE TAG THREADS     (settings.txt: name c1² c3²; caps as run_pf_local.sh)
# Build: g++ -O3 -march=native -fopenmp ../../hrsa/phase_free_full.cpp ../../hrsa/cyclotomic_int9.cpp ../../hrsa/Z9chi.cpp -lquadmath -o ../../hrsa/phase_free_full
set -e
cd "$(dirname "$0")"
S=$1; L=$2; TF=$3; TAG=$4; NT=$5
B=${PFF_BIN:-../../hrsa/phase_free_full}
PY=${PFF_PY:-python3}
read -r _ c1 c3 < <(grep "^$S " settings.txt)
export PF_C1SQ=$c1 PF_C3SQ=$c3
case $L in 2) f=1; cap=0.15;; 4) f=2; cap=0.012;; 6) f=3; cap=4e-4;; 8) f=4; cap=1.5e-5;; 10) f=5; cap=5.5e-7;; 12) f=6; cap=1.8e-8;; esac
eps=$($PY -c "print('%.6g' % ($cap * 2 ** 0.5 * 1.0005))")
d=pf_${TAG}; rm -rf $d && mkdir $d
OMP_NUM_THREADS=$NT /usr/bin/time -f "$TAG: %e s %M KB" $B $TF $eps $f 1000 $d $cap > $d.log 2>&1
tail -1 $d.log
[ "${PFF_NOANALYZE:-0}" = 1 ] && exit 0
$PY full_to_pool.py $d $TF pp_${TAG} > /dev/null
$PY full_analyze.py pp_${TAG} --procs $NT --copies-n 0 --out costs_${TAG}.csv > /dev/null
