#!/bin/bash
# Phase 18 on the server (descriptive, not registered; see p18common.py). One pipeline on the free GPUs:
#   main -> views -> valrc -> timing -> ana18. STEPS selects the steps (default: all). Progress in $P18/STATUS, logs in $P18/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P18=/home/omote/reprise_p18_20261012
CODE=$P18/code; LOG=$P18/logs
mkdir -p "$LOG" "$P18/results"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P18/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
if [ -n "${GPUS_OVERRIDE:-}" ]; then read -r -a GPUS <<< "$GPUS_OVERRIDE"; else
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
fi
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
STEPS=${STEPS:-main,views,valrc,timing,ana}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
status "P18 START gpus=${GPUS[*]} steps=$STEPS code_sha256=$(cat p18common.py core18.py run18.py ana18.py | sha256sum | cut -c1-16)"
workers() {  # workers <tag> <script> [args]: one worker per GPU
  local tag=$1; shift
  local pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY "$@" --worker $i --nworkers $NG > "$LOG/${tag}_$i.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
for step in main views valrc; do
  if has $step; then status "$step start"; workers p18_$step run18.py --step $step || fail "run18 $step"; status "$step done"; fi
done
if has timing; then
  status "timing start"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY run18.py --step timing > "$LOG/p18_timing.log" 2>&1 || fail "run18 timing"
  status "timing done"
fi
if has ana; then
  status "ana start"
  CUDA_VISIBLE_DEVICES= $PY ana18.py > "$LOG/ana18.log" 2>&1 || fail "ana18"
  grep -q ANA18_DONE "$LOG/ana18.log" || fail "ana18 did not finish"
  status "ana done"
fi
status "P18 DONE steps=$STEPS"
