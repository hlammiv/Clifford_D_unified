#!/bin/bash
# Lenore: level-14 batch 2, then level-16 batches c and d, sequentially on 30 threads (~10 h + 17 h + 17 h).
cd ~/cdft_topk/prescribed
./run_pf_l14b.sh && ./run_pf_l16c.sh && ./run_pf_l16d.sh
echo CHAIN_B_EXIT $? >> chain_b.out
