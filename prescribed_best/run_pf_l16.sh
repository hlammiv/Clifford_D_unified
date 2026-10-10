#!/bin/bash
# Lenore: level 16 phase-free pools, 5 clusters × 10 θ (centres = Oct-3 angles 10–14, spread 4e-8).
# Queued behind run_pf_lenore.sh (waits for LENORE_DONE). 20 threads, small chunks (RAM ≲ 1 GB).
# Cap self-calibrates from the finished levels: cap16 = median14 / (median12 / median14) × 1.8.
set -e
cd ~/cdft_topk/prescribed
B=~/cdft_topk/unified/hrsa/phase_free
PY=~/cdft_topk/venv/bin/python
until grep -q LENORE_DONE run_pf_lenore.out 2>/dev/null; do sleep 300; done
cap=$($PY - <<'PY'
import csv, statistics, math
med = lambda d: statistics.median(float(r["best_eps"]) for r in csv.DictReader(open(f"{d}/index.csv")) if not math.isnan(float(r["best_eps"])))
m12, m14 = med("pfp_l12"), med("pfp_l14")
print("%.4g" % (m14 / (m12 / m14) * 1.8))
PY
)
eps=$($PY -c "print('%.6g' % ($cap * 2 ** 0.5 * 1.0005))")
echo "level 16: cap $cap eps $eps  $(date)" > pf_l16.log
rm -rf pf_l16 && mkdir pf_l16
PF_GROUP_SPAN=3e-8 OMP_NUM_THREADS=20 /usr/bin/time -f "level 16: %e s %M KB" $B thetas_l16_clu.txt $eps 8 1000 pf_l16 $cap 500000 >> pf_l16.log 2>&1
$PY mt_to_pool.py pf_l16 thetas_l16_clu.txt pfp_l16 > /dev/null
echo L16_DONE >> pf_l16.log
