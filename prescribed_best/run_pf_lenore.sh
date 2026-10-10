#!/bin/bash
# Lenore: phase-free pools at level 12 (100 random θ) and level 14 (10 clusters × 10 θ, θ-span 2e-5 each,
# centres = first 10 Oct-3 angles; clustered θ share one enumeration, see phase_free.cpp).
set -e
cd ~/cdft_topk/prescribed
B=~/cdft_topk/unified/hrsa/phase_free
PY=~/cdft_topk/venv/bin/python
rm -rf pf_l12 && mkdir pf_l12
OMP_NUM_THREADS=24 /usr/bin/time -f "level 12: %e s %M KB" $B thetas_100.txt 2.5468e-8 6 1000 pf_l12 1.8e-8 > pf_l12.log 2>&1
$PY mt_to_pool.py pf_l12 thetas_100.txt pfp_l12 > /dev/null
rm -rf pf_l14 && mkdir pf_l14
PF_GROUP_SPAN=1.2e-5 OMP_NUM_THREADS=24 /usr/bin/time -f "level 14: %e s %M KB" $B thetas_l14_clu.txt 9.9045e-10 7 1000 pf_l14 7e-10 2000000 > pf_l14.log 2>&1
$PY mt_to_pool.py pf_l14 thetas_l14_clu.txt pfp_l14 > /dev/null
echo LENORE_DONE
