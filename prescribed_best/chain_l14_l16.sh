#!/bin/bash
# Lenore: after t14_test.sh finishes (and only if lamscale 1 and 2 agree), run level 14 then level 16.
cd ~/cdft_topk/prescribed
until grep -q T14_TEST_DONE t14_test.out 2>/dev/null; do sleep 60; done
grep -q "IDENTICAL" t14_test.out || { echo "t14 test failed; not starting level 14" > chain.out; exit 1; }
./run_pf_l14.sh > run_pf_l14.log 2>&1
./run_pf_l16.sh > run_pf_l16.out 2>&1
echo CHAIN_DONE >> chain.out
