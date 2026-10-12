#!/bin/bash
# Lenore: second level-16 batch (10 clusters × 10 θ, thetas_l16b_clu.txt) with the SAME cap/eps as batch 1
# (pf_l16.log: cap 1.868e-11, eps 2.64307e-11) so the pools can be merged into pfp_l16. 30 threads.
set -e
cd ~/cdft_topk/prescribed
B=~/cdft_topk/unified/hrsa/phase_free
PY=~/cdft_topk/venv/bin/python
cap=1.868e-11
eps=2.64307e-11
rm -rf pf_l16b && mkdir pf_l16b
echo "level 16b: cap $cap eps $eps  $(date)" > pf_l16b.log
PF_GROUP_SPAN=3e-8 OMP_NUM_THREADS=30 /usr/bin/time -f "level 16b: %e s %M KB" $B thetas_l16b_clu.txt $eps 8 1000 pf_l16b $cap 500000 >> pf_l16b.log 2>&1
$PY mt_to_pool.py pf_l16b thetas_l16b_clu.txt pfp_l16b > /dev/null
echo L16B_DONE >> pf_l16b.log
