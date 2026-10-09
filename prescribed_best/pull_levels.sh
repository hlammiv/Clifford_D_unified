#!/bin/bash
# Pull multi_theta results for levels 12 and 14 from Lenore as they finish; convert, analyze, refit.
cd "$(dirname "$0")"
SSH="ssh -p 60022 -o BatchMode=yes"
for spec in "12 6 thetas_500.txt 100" "14 7 thetas_100.txt 100"; do
  set -- $spec; L=$1; TF=$3; CN=$4
  until timeout 30 $SSH lenore_remote "grep -q 'DONE' ~/cdft_topk/prescribed/nj_l$L.log" 2>/dev/null; do
    if timeout 30 $SSH lenore_remote "grep -q 'Command terminated\|rror\|Killed' ~/cdft_topk/prescribed/nj_l$L.log" 2>/dev/null; then echo "L$L FAILED"; continue 2; fi
    sleep 300
  done
  timeout 30 $SSH lenore_remote "grep -v a3-prog ~/cdft_topk/prescribed/nj_l$L.log"
  rsync -az -e "$SSH" lenore_remote:cdft_topk/prescribed/nj_l$L . || continue
  python3 mt_to_pool.py nj_l$L $TF pool_l$L
  python3 -c "
import csv, numpy as np
r=list(csv.DictReader(open('pool_l$L/index.csv'))); ok=[x for x in r if x['n_cand']!='0']
print('L$L:',len(r),'angles,',len(ok),'with candidates; median best eps %.3g (max %.3g); median pool %d'%(np.median([float(x['best_eps']) for x in ok]), max(float(x['best_eps']) for x in ok), np.median([int(x['n_cand']) for x in ok])))"
  OMP_NUM_THREADS=1 python3 analyze_pools.py pool_l$L --procs 12 --copies-n $CN --out costs_l$L.csv > costs_l$L.log 2>&1
  tail -1 costs_l$L.log
  python3 fit_generic.py > fit_generic_output.txt 2>&1
  echo "L$L ANALYZED"
done
cat fit_generic_output.txt
echo PULL_DONE
