#!/bin/bash
# Wait for Lenore's multi_theta level-8 run, then pull, convert, analyze, refit.
cd "$(dirname "$0")"
SSH="ssh -p 60022 -o BatchMode=yes"
until timeout 30 $SSH lenore_remote 'grep -q "\[multi\] DONE" ~/cdft_topk/prescribed/mt_l8.log' 2>/dev/null; do
  if timeout 30 $SSH lenore_remote 'grep -q "Command terminated\|rror" ~/cdft_topk/prescribed/mt_l8.log' 2>/dev/null; then echo "L8 FAILED"; exit 1; fi
  sleep 600
done
timeout 30 $SSH lenore_remote 'tail -3 ~/cdft_topk/prescribed/mt_l8.log'
rsync -az -e "$SSH" lenore_remote:cdft_topk/prescribed/mt_l8 . || exit 1
python3 mt_to_pool.py mt_l8 thetas_500.txt pool_l8 || exit 1
python3 -c "
import csv, numpy as np
r=list(csv.DictReader(open('pool_l8/index.csv'))); ok=[x for x in r if x['n_cand']!='0']
print(len(r),'angles,',len(ok),'with candidates; median best eps %.3g; median pool %d'%(np.median([float(x['best_eps']) for x in ok]), np.median([int(x['n_cand']) for x in ok])))"
OMP_NUM_THREADS=1 python3 analyze_pools.py pool_l8 --procs 12 --copies-n 100 --out costs_l8.csv > costs_l8.log 2>&1
tail -1 costs_l8.log
python3 fit_generic.py > fit_generic_output.txt 2>&1
cat fit_generic_output.txt
echo OVERNIGHT_DONE
