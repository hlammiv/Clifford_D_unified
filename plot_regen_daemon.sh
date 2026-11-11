#!/bin/bash
# plot_regen_daemon.sh — persistent plot regenerator.
#
# Polls all known data sources every 5 min. Pulls Lenore CSVs (tier-2 + f=2)
# to local mirrors. Regenerates plot whenever any source mtime is newer than
# the current plot. Idempotent.
#
# Run with `nohup ... &`; stop with `kill -TERM <PID>`.

set -u
ROOT=/home/hlamm/Desktop/efficent_gates/unified
PLOT=$ROOT/nd_vs_eps_v3_2026-05-23.png
POLL_SEC=300

LENORE_TIER2=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/sweep_zeta9_tier2_eps1e-4_2026-05-23
LENORE_F2=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/sweep_hrsa_f2_2026-05-23

mkdir -p $ROOT/sweep_zeta9_tier2_eps1e-4_2026-05-23
mkdir -p $ROOT/sweep_hrsa_f2_2026-05-23

# Local sources (some materialise later).
# sweep_bidir_k3_2026-05-23 was a standalone K=3 probe — superseded by
# sweep_hrsa_cascade_2026-05-23 which runs the full Direct→SignExt→bidir 2..4→HRSA
# cascade and emits the correct method label for each cell. Bidir_k3 is dropped
# to avoid double-counting bidir(3+3) cells in the plot scatter.
SOURCES=(
  $ROOT/hrsa_sweep_v4_2026-05-10.csv
  $ROOT/sweep_hrsa_grid_2026-05-22/summary.csv
  $ROOT/sweep_hrsa_dense_eps1e-3_2026-05-23/summary.csv
  $ROOT/sweep_hrsa_cascade_2026-05-23/summary.csv
  /tmp/lenore_f4_eps1e-4.csv
  /tmp/lenore_f4_eps1e-3.csv
  $ROOT/sweep_zeta9_tier2_eps1e-4_2026-05-23/summary.csv
  $ROOT/sweep_hrsa_f2_2026-05-23/summary.csv
)

trap 'echo "[$(date +%H:%M:%S)] daemon exiting"; exit 0' TERM INT

echo "[$(date +%H:%M:%S)] regen daemon started (poll=${POLL_SEC}s, plot=$PLOT)"
while true; do
  # 0. Merge local worker_*/summary.csv shards for parallel cascade sweep
  if [ -d $ROOT/sweep_hrsa_cascade_2026-05-23 ] && \
     ls $ROOT/sweep_hrsa_cascade_2026-05-23/worker_*/summary.csv >/dev/null 2>&1; then
    cat $ROOT/sweep_hrsa_cascade_2026-05-23/worker_*/summary.csv 2>/dev/null \
        | awk 'NR==1 || !/^theta_idx,/' \
        > $ROOT/sweep_hrsa_cascade_2026-05-23/summary.csv 2>/dev/null
  fi

  # 1. Refresh Lenore mirrors (silent failures are OK — files may not exist yet)
  scp -q -P 60022 -o ConnectTimeout=10 \
      lenore_remote:$LENORE_TIER2/summary.csv \
      $ROOT/sweep_zeta9_tier2_eps1e-4_2026-05-23/summary.csv 2>/dev/null || true

  # f=2 sweep: workers write to worker_NN/summary.csv. Merge on Lenore (cheap) + scp.
  ssh -p 60022 -o ConnectTimeout=10 lenore_remote "
    [ -d $LENORE_F2 ] && cd $LENORE_F2 && \
    ls worker_*/summary.csv >/dev/null 2>&1 && \
    cat worker_*/summary.csv | awk 'NR==1 || !/^theta_idx,/' > summary.csv 2>/dev/null
  " 2>/dev/null || true
  scp -q -P 60022 -o ConnectTimeout=10 \
      lenore_remote:$LENORE_F2/summary.csv \
      $ROOT/sweep_hrsa_f2_2026-05-23/summary.csv 2>/dev/null || true

  # 2. Compute newest source mtime
  newest=0
  for s in "${SOURCES[@]}"; do
    if [ -f "$s" ]; then
      mt=$(stat -c %Y "$s" 2>/dev/null || echo 0)
      [ $mt -gt $newest ] && newest=$mt
    fi
  done
  plot_mtime=0
  [ -f "$PLOT" ] && plot_mtime=$(stat -c %Y "$PLOT")

  # 3. Regen iff something changed
  if [ $newest -gt $plot_mtime ]; then
    diff_s=$((newest - plot_mtime))
    existing=()
    for s in "${SOURCES[@]}"; do
      [ -f "$s" ] && existing+=("$s")
    done
    echo "[$(date +%H:%M:%S)] regenerating (newest ${diff_s}s newer than plot, ${#existing[@]} sources)"
    cd $ROOT
    python3 plot_nd_vs_eps_v2.py $PLOT "${existing[@]}" 2>&1 | tail -2
  fi

  sleep $POLL_SEC
done
