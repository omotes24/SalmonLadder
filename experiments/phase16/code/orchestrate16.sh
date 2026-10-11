#!/bin/bash
# Phase 16 on the server (prereg_p16.json). One pipeline on all free GPUs:
#   smoke -> dev1 (+ ana16 --dev1-select) -> dev2 (+ ana16 --dev2-confirm) -> test -> in200 -> cifar -> rec -> ana16 --summary
# STEPS selects the steps (default: all). The dev2 step and every later step are launched only after the registration has been
# committed publicly (STATUS records the launch time). Progress in $P16/STATUS, logs in $P16/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P16=/home/omote/reprise_p16_20261011
CODE=$P16/code; LOG=$P16/logs
mkdir -p "$LOG" "$P16/results"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P16/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
if [ -n "${GPUS_OVERRIDE:-}" ]; then read -r -a GPUS <<< "$GPUS_OVERRIDE"; else
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
fi
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
STEPS=${STEPS:-smoke,dev1,dev2,test,in200,cifar,rec,ana}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
status "P16 START gpus=${GPUS[*]} steps=$STEPS prereg_sha256=$(sha256sum "$CODE/prereg_p16.json" | cut -c1-16)"
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
if has smoke; then
  status "smoke start"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY smoke16.py > "$LOG/smoke16.log" 2>&1 || fail "smoke16"
  grep -q SMOKE16_OK "$LOG/smoke16.log" || fail "smoke16 not ok"
  status "smoke done"
fi
if has dev1; then
  status "dev1 start"
  workers h16_dev1 h16.py --step dev --devs dev1 || fail "h16 dev1"
  CUDA_VISIBLE_DEVICES= $PY ana16.py --dev1-select > "$LOG/ana16_dev1.log" 2>&1 || fail "ana16 dev1"
  status "dev1 done: base combination $(grep -o '"chosen": "[^"]*"' "$P16/results/h16_dev/dev1_selection.json")"
fi
if has dev2; then
  status "dev2 start"
  workers h16_dev2 h16.py --step dev --devs dev2 || fail "h16 dev2"
  CUDA_VISIBLE_DEVICES= $PY ana16.py --dev2-confirm > "$LOG/ana16_dev2.log" 2>&1 || fail "ana16 dev2"
  status "dev2 done: adopt=$(grep -o '"adopt_h16_as_main": [a-z]*' "$P16/results/h16_dev/dev2_confirmation.json")"
fi
for step in test in200 cifar rec; do
  if has $step; then status "$step start"; workers h16_$step h16.py --step $step || fail "h16 $step"; status "$step done"; fi
done
if has ana; then
  status "ana start"
  CUDA_VISIBLE_DEVICES= $PY ana16.py --summary > "$LOG/ana16_summary.log" 2>&1 || fail "ana16 summary"
  grep -q ANA16_DONE "$LOG/ana16_summary.log" || fail "ana16 did not finish"
  status "ana done"
fi
status "P16 DONE steps=$STEPS"
