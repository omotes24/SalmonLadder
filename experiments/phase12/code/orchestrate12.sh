#!/bin/bash
# Phase 12 on the server: (a) selection on the development splits, (b) draws of the labelled images on the benchmarks,
# (c) arrival-time mode on the 37 streams, (d) recurrence x mixing-rate grid. Progress in $P12/STATUS, logs in $P12/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P12=/home/omote/reprise_p12_20261011
CODE=$P12/code; LOG=$P12/logs
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P12/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
status "P12 START gpus=${GPUS[*]} steps=${STEPS:-sel,draws,online,rec}"
run_split() {  # run_split <tag> <script> <args...> : one worker per GPU, worker index / count appended
  local tag=$1; shift
  local pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY "$@" --worker $i --nworkers $NG > "$LOG/${tag}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "$tag (logs/${tag}_*.log)"; done
}
STEPS=${STEPS:-sel,draws,online,rec}
if [[ ",$STEPS," == *",sel,"* ]]; then
  # two development splits in parallel: half of the GPUs each
  H=$(( (NG + 1) / 2 ))
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    dev=dev1; w=$i; nw=$H
    if [ "$i" -ge "$H" ]; then dev=dev2; w=$((i - H)); nw=$((NG - H)); fi
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY sel12.py --dev $dev --worker $w --nworkers $nw > "$LOG/sel_${dev}_$w.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "sel12 (logs/sel_*.log)"; done
  $PY sel12.py --aggregate > "$LOG/sel12_aggregate.log" 2>&1 || fail "sel12 aggregate"
  status "SEL DONE: $(tail -n 1 $P12/results/sel/sel12_table.txt | head -c 200)"
fi
if [[ ",$STEPS," == *",draws,"* ]]; then
  for part in openood fourood; do
    run_split "draws_$part" draws12.py --part $part
    status "DRAWS $part done ($(ls $P12/results/draws/$part/*/*.csv | wc -l) files)"
  done
fi
if [[ ",$STEPS," == *",online,"* ]]; then
  for part in openood fourood; do
    run_split "online_$part" online12.py --part $part
    status "ONLINE $part done ($(ls $P12/results/online/$part/*.csv | wc -l) streams)"
  done
fi
if [[ ",$STEPS," == *",rec,"* ]]; then
  run_split "rec" rec12.py
  status "REC done ($(ls $P12/results/rec/*/*.csv | wc -l) conditions)"
fi
status "P12 DONE"
