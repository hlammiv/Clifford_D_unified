#!/bin/bash
# Lenore: second level-14 batch (10 clusters × 10 θ, thetas_l14b_clu.txt), SAME eps/cap/f' as run_pf_l14.sh so
# the pools merge into pfp_l14. 30 threads.
set -e; cd ~/cdft_topk/prescribed; B=~/cdft_topk/unified/hrsa/phase_free; PY=~/cdft_topk/venv/bin/python
rm -rf pf_l14b && mkdir pf_l14b
echo "level 14b: cap 7e-10 eps 9.9045e-10  $(date)" > pf_l14b.log
PF_GROUP_SPAN=1.5e-6 OMP_NUM_THREADS=30 /usr/bin/time -f "level 14b: %e s %M KB" $B thetas_l14b_clu.txt 9.9045e-10 7 1000 pf_l14b 7e-10 1000000 >> pf_l14b.log 2>&1
$PY mt_to_pool.py pf_l14b thetas_l14b_clu.txt pfp_l14b > /dev/null; echo L14B_DONE >> pf_l14b.log
