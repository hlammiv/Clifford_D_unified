#!/bin/bash
# Lenore share (32 threads): cross-check cell, level 10 targets 0-49 of every setting, level 12 targets 0-14 (bal, skew).
cd "$(dirname "$0")"
export PFF_NOANALYZE=1
./run_full.sh bal 6 targets_bal.txt xchk_bal_l6 32
for k in bal s2 s3 skew; do ./run_full.sh $k 10 targets_${k}_l10a.txt ${k}_l10a 32; done
for k in bal skew; do ./run_full.sh $k 12 targets_${k}_l12a.txt ${k}_l12a 32; done
echo LENORE_DONE
