#!/bin/bash
# Lenore: level 14 phase-free pools, 10 clusters × 10 θ (rerun after the __float128 LLL fix, 2026-10-10).
set -e
cd ~/cdft_topk/prescribed
B=~/cdft_topk/unified/hrsa/phase_free
PY=~/cdft_topk/venv/bin/python
rm -rf pf_l14 && mkdir pf_l14
PF_GROUP_SPAN=1.2e-5 OMP_NUM_THREADS=24 /usr/bin/time -f "level 14: %e s %M KB" $B thetas_l14_clu.txt 9.9045e-10 7 1000 pf_l14 7e-10 500000 > pf_l14.log 2>&1
$PY mt_to_pool.py pf_l14 thetas_l14_clu.txt pfp_l14 > /dev/null
echo L14_DONE > run_pf_l14.out
