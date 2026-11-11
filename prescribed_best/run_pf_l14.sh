#!/bin/bash
# Lenore: level 14 phase-free pools, 10 clusters × 10 θ (θ spread 2e-6 ≪ chunk width 5.8e-6, so the x1
# arcs are widened by only ~25%). Rerun after the __float128 LLL fix and the join-slack fix, 2026-10-10.
set -e
cd ~/cdft_topk/prescribed
B=~/cdft_topk/unified/hrsa/phase_free
PY=~/cdft_topk/venv/bin/python
rm -rf pf_l14 && mkdir pf_l14
PF_GROUP_SPAN=1.5e-6 OMP_NUM_THREADS=24 /usr/bin/time -f "level 14: %e s %M KB" $B thetas_l14_clu.txt 9.9045e-10 7 1000 pf_l14 7e-10 1000000 > pf_l14.log 2>&1
$PY mt_to_pool.py pf_l14 thetas_l14_clu.txt pfp_l14 > /dev/null
echo L14_DONE > run_pf_l14.out
