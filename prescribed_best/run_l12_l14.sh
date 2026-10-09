#!/bin/bash
# Levels 12 (500 θ) and 14 (100 θ) with multi_theta fp mode, 8 threads.
cd ~/cdft_topk/prescribed
rm -rf nj_l12 nj_l14; mkdir nj_l12 nj_l14
OMP_NUM_THREADS=8 /usr/bin/time -f "%e s wall %M KB" ~/cdft_topk/unified/hrsa/multi_theta thetas_500.txt 1.2e-6 6 200000 nj_l12 fp > nj_l12.log 2>&1
OMP_NUM_THREADS=8 /usr/bin/time -f "%e s wall %M KB" ~/cdft_topk/unified/hrsa/multi_theta thetas_100.txt 1e-7 7 200000 nj_l14 fp > nj_l14.log 2>&1
touch l12_l14.done
