#!/bin/bash
# Level-14 completeness test, 1 θ, cap 1e-10: point count vs analytic prediction, and invariance under a 2× ellipsoid.
cd ~/cdft_topk/prescribed
head -1 thetas_l14_clu.txt > th14_1.txt
rm -rf t14a t14b; mkdir t14a t14b
for ls in 1 2; do
  d=t14a; [ $ls = 2 ] && d=t14b
  PF_LAMSCALE=$ls OMP_NUM_THREADS=24 /usr/bin/time -f "lamscale $ls: %e s %M KB" ~/cdft_topk/unified/hrsa/phase_free th14_1.txt 1.4149e-10 7 1000 $d 1e-10 500000 2>&1 | grep -E "predicted|θ#|lamscale" | cut -c1-220
done
diff <(cut -d" " -f3- t14a/t0000.txt | sort) <(cut -d" " -f3- t14b/t0000.txt | sort) > /dev/null && echo "lamscale 1 vs 2: IDENTICAL" || echo "lamscale DIFF"
echo T14_TEST_DONE
