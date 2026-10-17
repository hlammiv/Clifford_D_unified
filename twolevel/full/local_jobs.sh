#!/bin/bash
# Local share (14 threads): levels 2-8 (100 targets) for every setting, level 10 targets 50-99, level 12 targets 15-22 (bal, skew).
cd "$(dirname "$0")"
for k in bal s2 s3 skew; do for L in 2 4 6 8; do ./run_full.sh $k $L targets_$k.txt ${k}_l$L 14; done; done
for k in bal s2 s3 skew; do ./run_full.sh $k 10 targets_${k}_l10b.txt ${k}_l10b 14; done
for k in bal skew; do ./run_full.sh $k 12 targets_${k}_l12b.txt ${k}_l12b 14; done
echo LOCAL_DONE
