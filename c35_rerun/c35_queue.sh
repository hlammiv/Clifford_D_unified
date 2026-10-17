#!/bin/bash
# C35 rerun: worker over jobs.txt shared by Lenore and the local machine. A job is claimed by an atomic mkdir in
# Lenore's claims/ directory; Lenore walks the list forward, the local machine backward.
# Env: SIDE=lenore|local NT PF_BIN PFF_BIN PB (dir with theta files) FULL (twolevel/full dir) OUT PY
cd "$(dirname "$0")"
Q=cdft_topk/c35
claim() { if [ "$SIDE" = lenore ]; then mkdir ~/$Q/claims/$1 2>/dev/null; else ssh -n -p 60022 -o BatchMode=yes lenore_remote "mkdir ~/$Q/claims/$1" 2>/dev/null; fi; }
mkdir -p "$OUT"
if [ "$SIDE" = lenore ]; then LIST=$(grep -v "${SKIP:-^$}" "${JOBS:-jobs.txt}"); else LIST=$(grep -v "${SKIP:-^$}" "${JOBS:-jobs.txt}" | tac); fi
while read -r kind id rest; do
  [ -z "$kind" ] && continue
  claim "$id" || continue
  echo "$(date '+%F %T') start $id" >> "$OUT/worker_$SIDE.log"
  if [ "$kind" = pf ]; then
    read -r L f eps cap span chunk tf st n <<< "$rest"
    sed -n "$((st + 1)),$((st + n))p" "$PB/$tf" > "$OUT/$id.th"
    rm -rf "$OUT/$id"; mkdir "$OUT/$id"
    args=("$OUT/$id.th" "$eps" "$f" 1000 "$OUT/$id" "$cap"); [ "$chunk" != - ] && args+=("$chunk")
    if [ "$span" != - ]; then export PF_GROUP_SPAN=$span; else unset PF_GROUP_SPAN; fi
    OMP_NUM_THREADS=$NT /usr/bin/time -f "$id: %e s %M KB" "$PF_BIN" "${args[@]}" < /dev/null > "$OUT/$id.log" 2>&1
  else
    read -r s L tf tag <<< "$rest"
    (cd "$FULL" && PFF_BIN=$PFF_BIN PFF_PY=$PY PFF_NOANALYZE=1 ./run_full.sh "$s" "$L" "$tf" "${tag}_c35" "$NT") < /dev/null > "$OUT/$id.log" 2>&1
  fi
  echo "$(date '+%F %T') done $id rc=$?" >> "$OUT/worker_$SIDE.log"
done <<< "$LIST"
echo "$(date '+%F %T') WORKER_EXIT" >> "$OUT/worker_$SIDE.log"
