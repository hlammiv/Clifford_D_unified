#!/bin/bash
# Stop the level-14/16 chain and any phase_free production run (not tests).
for p in $(pgrep -f "chain_l14_l16.sh") $(pgrep -f "run_pf_l1[46].sh") $(pgrep -f "phase_free thetas_l1[46]"); do kill $p 2>/dev/null; done
sleep 2
pgrep -fa "phase_free|chain_l14|run_pf_l1" | grep -v kill_chain | cut -c1-60
echo KILLED
