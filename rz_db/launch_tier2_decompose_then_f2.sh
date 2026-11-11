#!/bin/bash
# launch_tier2_decompose_then_f2.sh — chained pipeline:
#   1. Run parallel decompose on tier-2 stage-5 outputs (32 workers, ~5 hr)
#   2. After it finishes, run f=2 sweep (32 workers, ~10-30 min)
#
# Replaces the original launch_hrsa_f2_sweep.sh which waited on the (now-killed)
# sequential decompose. Both chained sweeps log to data dir, summary.csv is
# pulled by the regen daemon on lucia.

set -u
export OMP_NUM_THREADS=1
ROOT=/home/hlamm/Desktop/efficent_gates/unified
TIER2_OUT=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/sweep_zeta9_tier2_eps1e-4_2026-05-23
F2_OUT=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/sweep_hrsa_f2_2026-05-23
F2_LOG=$F2_OUT/sweep.log

mkdir -p $F2_OUT

# --- Phase A: parallel decompose on tier-2 ---
echo "[chain] $(date +%H:%M:%S) — phase A: parallel decompose (32 workers)"
python3 $ROOT/decompose_parallel.py \
    --out_dir $TIER2_OUT \
    --n_thetas 10000 \
    --theta_min 0.0 \
    --theta_max 6.283185307179586 \
    --eps 1e-4 \
    --eps_pre 5e-5 \
    --max_f 2 \
    --workers 32 \
    > $TIER2_OUT/decompose_parallel.log 2>&1
echo "[chain] $(date +%H:%M:%S) — decompose done"

# --- Phase B: f=2 HRSA sweep ---
echo "[chain] $(date +%H:%M:%S) — phase B: f=2 HRSA sweep (32 worker shards)"
N_THETAS=100
N_WORKERS=32
EPS_LIST="0.5,0.1,0.05,0.01,0.005,0.001"
PI2=6.283185307179586

for w in $(seq 0 $((N_WORKERS - 1))); do
    lo=$(( w * N_THETAS / N_WORKERS ))
    hi=$(( (w+1) * N_THETAS / N_WORKERS ))
    [ $hi -le $lo ] && continue
    nth=$((hi - lo))
    th_min=$(awk "BEGIN {printf \"%.10f\", $lo * $PI2 / $N_THETAS}")
    th_max=$(awk "BEGIN {printf \"%.10f\", $hi * $PI2 / $N_THETAS}")
    wdir=$F2_OUT/worker_$(printf "%02d" $w)
    mkdir -p $wdir
    nohup python3 $ROOT/sweep_hrsa_grid.py \
        --n_thetas $nth \
        --theta_min $th_min \
        --theta_max $th_max \
        --max_f 2 \
        --eps $EPS_LIST \
        --timeout 300 \
        --out_dir $wdir \
        > $wdir/sweep.log 2>&1 &
done
wait
echo "[chain] $(date +%H:%M:%S) — f=2 workers exited; merging CSVs"
cat $F2_OUT/worker_*/summary.csv | awk 'NR==1 || !/^theta_idx,/' > $F2_OUT/summary.csv
total=$(($(wc -l < $F2_OUT/summary.csv) - 1))
pass=$(awk -F, 'NR>1 && $NF~/true/' $F2_OUT/summary.csv | wc -l)
echo "[chain] DONE: f=2 sweep has $total cells, $pass passes. tier-2 decompose in $TIER2_OUT/summary.csv"
