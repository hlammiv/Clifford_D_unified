#!/bin/bash
# Lenore: level-16 batch c (10 clusters × 10 θ, thetas_l16c_clu.txt), SAME cap/eps as batches 1-2 so the pools
# merge into pfp_l16. 30 threads.
set -e; cd ~/cdft_topk/prescribed; B=~/cdft_topk/unified/hrsa/phase_free; PY=~/cdft_topk/venv/bin/python
cap=1.868e-11; eps=2.64307e-11; rm -rf pf_l16c && mkdir pf_l16c
echo "level 16c: cap $cap eps $eps  $(date)" > pf_l16c.log
PF_GROUP_SPAN=3e-8 OMP_NUM_THREADS=30 /usr/bin/time -f "level 16c: %e s %M KB" $B thetas_l16c_clu.txt $eps 8 1000 pf_l16c $cap 500000 >> pf_l16c.log 2>&1
$PY mt_to_pool.py pf_l16c thetas_l16c_clu.txt pfp_l16c > /dev/null; echo L16C_DONE >> pf_l16c.log
