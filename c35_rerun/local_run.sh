#!/bin/bash
# Local share of the C35 rerun (14 threads): R_z levels 2-10 (500 θ, as run_pf_local.sh) and reflection cells at
# levels 2-8, then the shared queue (walked backward).
cd "$(dirname "$0")"
U=/home/hlamm/Desktop/efficent_gates/unified
export SIDE=local NT=14 PF_BIN=$U/hrsa/phase_free PFF_BIN=$U/hrsa/phase_free_full PB=$U/prescribed_best FULL=$U/twolevel/full OUT=$PWD/out PY=python3
mkdir -p out
for spec in "2 1 0.15" "4 2 0.012" "6 3 4e-4" "8 4 1.5e-5" "10 5 5.5e-7"; do
  set -- $spec; L=$1; f=$2; cap=$3
  eps=$(python3 -c "print('%.6g' % ($cap * 2 ** 0.5 * 1.0005))")
  rm -rf out/pf_l$L && mkdir out/pf_l$L
  OMP_NUM_THREADS=$NT /usr/bin/time -f "level $L: %e s %M KB" $PF_BIN $PB/thetas_500.txt $eps $f 1000 out/pf_l$L $cap > out/pf_l$L.log 2>&1
  echo "$(date '+%F %T') R_z level $L done" >> out/worker_local.log
done
for k in bal s2 s3 skew; do for L in 2 4 6 8; do
  (cd $FULL && PFF_BIN=$PFF_BIN PFF_NOANALYZE=1 ./run_full.sh $k $L targets_$k.txt ${k}_l${L}_c35 $NT) > out/F_${k}_l$L.log 2>&1
done; done
echo "$(date '+%F %T') reflections 2-8 done" >> out/worker_local.log
./c35_queue.sh
