#!/bin/bash
# Phase 17 on the server (prereg_p17.json). One pipeline on all free GPUs:
#   dev (dev1, dev2) -> ana17 --dev2-adopt (the registered adoption rule) -> test -> in200 -> cifar -> rec -> csid -> ana17 --summary
# Launched only after the registration has been committed publicly (STATUS records the launch time). smoke17 was run before the
# registration was placed. STEPS selects the steps (default: all). Progress in $P17/STATUS, logs in $P17/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P17=/home/omote/reprise_p17_20261011
CODE=$P17/code; LOG=$P17/logs
mkdir -p "$LOG" "$P17/results"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P17/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
if [ -n "${GPUS_OVERRIDE:-}" ]; then read -r -a GPUS <<< "$GPUS_OVERRIDE"; else
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
fi
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
STEPS=${STEPS:-dev,test,in200,cifar,rec,csid,ana}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
status "P17 START gpus=${GPUS[*]} steps=$STEPS prereg_sha256=$(sha256sum "$CODE/prereg_p17.json" | cut -c1-16)"
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
if has dev; then
  status "dev start"
  workers p17_dev run17.py --step dev --devs dev1,dev2 || fail "run17 dev"
  CUDA_VISIBLE_DEVICES= $PY ana17.py --dev2-adopt > "$LOG/ana17_adopt.log" 2>&1 || fail "ana17 adopt"
  status "dev done: adopt=$(grep -o '"adopt_sigma_ref_as_main": [a-z]*' "$P17/results/p17_dev/adoption.json")"
fi
for step in test in200 cifar rec csid; do
  if has $step; then status "$step start"; workers p17_$step run17.py --step $step || fail "run17 $step"; status "$step done"; fi
done
if has ana; then
  status "ana start"
  CUDA_VISIBLE_DEVICES= $PY ana17.py --summary > "$LOG/ana17_summary.log" 2>&1 || fail "ana17 summary"
  grep -q ANA17_DONE "$LOG/ana17_summary.log" || fail "ana17 did not finish"
  status "ana done"
fi
status "P17 DONE steps=$STEPS"
