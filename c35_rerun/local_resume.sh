#!/bin/bash
# C35 local worker: the queue without the level-14 jobs (5–7 GB each; Lenore takes them).
cd "$(dirname "$0")"
SIDE=local NT=14 SKIP=" L14_" PF_BIN=$PWD/../hrsa/phase_free PFF_BIN=$PWD/../hrsa/phase_free_full PB=$PWD/../prescribed_best FULL=$PWD/../twolevel/full OUT=$PWD/out PY=python3 ./c35_queue.sh
