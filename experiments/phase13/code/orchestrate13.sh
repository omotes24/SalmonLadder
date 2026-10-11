#!/bin/bash
# Phase 13 on the server: (b) exchangeable calibration, (c1) products on the benchmarks, (d) ImageNet-200, (c2) the grid with
# products, (a) arrival-time mode with converged propagation, then the analysis. Progress in $P13/STATUS, logs in $P13/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P13=/home/omote/reprise_p13_20261011
CODE=$P13/code; LOG=$P13/logs
mkdir -p "$LOG" "$P13/results"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P13/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
STEPS=${STEPS:-fdpcal,product,in200,rec,online,ana}
status "P13 START gpus=${GPUS[*]} steps=$STEPS"
run_two_parts() {  # run_two_parts <tag> <script>: openood on the first half of the GPUs, fourood on the second half (or all GPUs sequentially)
  local tag=$1 script=$2
  local H=$(( (NG + 1) / 2 ))
  local pids=()
  for i in $(seq 0 $((NG - 1))); do
    if [ "$i" -lt "$H" ]; then part=openood; w=$i; nw=$H; else part=fourood; w=$((i - H)); nw=$((NG - H)); fi
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY "$script" --part $part --worker $w --nworkers $nw > "$LOG/${tag}_${part}_$w.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "$tag (logs/${tag}_*.log)"; done
  if [ "$NG" -eq 1 ]; then CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY "$script" --part fourood --worker 0 --nworkers 1 > "$LOG/${tag}_fourood_0.log" 2>&1 || fail "$tag fourood"; fi
}
run_workers() {  # run_workers <tag> <script> [args]: one worker per GPU
  local tag=$1; shift
  local pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY "$@" --worker $i --nworkers $NG > "$LOG/${tag}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "$tag (logs/${tag}_*.log)"; done
}
if [[ ",$STEPS," == *",fdpcal,"* ]]; then status "fdpcal start"; run_two_parts fdpcal fdpcal13.py; status "fdpcal done"; fi
if [[ ",$STEPS," == *",product,"* ]]; then status "product start"; run_two_parts product product13.py; status "product done"; fi
if [[ ",$STEPS," == *",in200,"* ]]; then status "in200 start"; run_workers in200 in200_13.py; status "in200 done"; fi
if [[ ",$STEPS," == *",rec,"* ]]; then status "rec start"; run_workers rec rec13.py; status "rec done"; fi
if [[ ",$STEPS," == *",online,"* ]]; then status "online start"; run_two_parts online online13.py; status "online done"; fi
if [[ ",$STEPS," == *",ana,"* ]]; then status "ana start"; CUDA_VISIBLE_DEVICES= $PY ana13.py > "$LOG/ana13.log" 2>&1 || fail "ana13"; status "ana done"; fi
status "P13 DONE"
