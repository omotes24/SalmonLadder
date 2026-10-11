#!/bin/bash
# Phase 14 on the server. Stage A (STEPS=smoke,sel): smoke test, dev-split runs, aggregation = the registered selection.
# Stage B (STEPS=test_train,test_val,in200,rec,views,cifar,online,ana): the test-set runs with the selected configuration, then
# the analysis. Progress in $P14/STATUS, logs in $P14/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P14=/home/omote/reprise_p14_20261011
CODE=$P14/code; LOG=$P14/logs
mkdir -p "$LOG" "$P14/results"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P14/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
STEPS=${STEPS:-smoke,sel}
status "P14 START gpus=${GPUS[*]} steps=$STEPS"
run_two_parts() {  # run_two_parts <tag> <script> [args]: openood on the first half of the GPUs, fourood on the second half
  local tag=$1 script=$2; shift 2
  local H=$(( (NG + 1) / 2 ))
  local pids=()
  for i in $(seq 0 $((NG - 1))); do
    if [ "$i" -lt "$H" ]; then part=openood; w=$i; nw=$H; else part=fourood; w=$((i - H)); nw=$((NG - H)); fi
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY "$script" --part $part --worker $w --nworkers $nw "$@" > "$LOG/${tag}_${part}_$w.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "$tag (logs/${tag}_*.log)"; done
  if [ "$NG" -eq 1 ]; then CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY "$script" --part fourood --worker 0 --nworkers 1 "$@" > "$LOG/${tag}_fourood_0.log" 2>&1 || fail "$tag fourood"; fi
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
has() { [[ ",$STEPS," == *",$1,"* ]]; }
if has smoke; then
  status "smoke start"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY smoke14.py > "$LOG/smoke14.log" 2>&1 || fail "smoke14 (logs/smoke14.log)"
  grep -q SMOKE14_OK "$LOG/smoke14.log" || fail "smoke14 did not finish"
  status "smoke done"
fi
if has sel; then
  status "sel start"
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    dev=$([ $((i % 2)) -eq 0 ] && echo dev1 || echo dev2); nw=$(( (NG + (1 - i % 2)) / 2 )); w=$((i / 2))
    [ "$NG" -eq 1 ] && { dev=dev1; nw=1; w=0; }
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY sel14.py --dev $dev --worker $w --nworkers $nw > "$LOG/sel_${dev}_$w.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "sel (logs/sel_*.log)"; done
  if [ "$NG" -eq 1 ]; then CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY sel14.py --dev dev2 --worker 0 --nworkers 1 > "$LOG/sel_dev2_0.log" 2>&1 || fail "sel dev2"; fi
  CUDA_VISIBLE_DEVICES= $PY sel14.py --aggregate > "$LOG/sel14_aggregate.log" 2>&1 || fail "sel14 aggregate"
  grep -q SEL14_DONE "$LOG/sel14_aggregate.log" || fail "sel14 aggregate did not finish"
  status "sel done: $(grep SELECTED "$LOG/sel14_aggregate.log")"
fi
if has test_train; then status "test_train start"; run_two_parts test_train test14.py --cal train; status "test_train done"; fi
if has test_val; then status "test_val start"; run_two_parts test_val test14.py --cal val; status "test_val done"; fi
if has in200; then status "in200 start"; run_workers in200 in200_14.py; status "in200 done"; fi
if has rec; then status "rec start"; run_workers rec rec14.py; status "rec done"; fi
if has views; then status "views start"; run_workers views views14.py; status "views done"; fi
if has cifar; then
  status "cifar start"
  if [ ! -f "$P14/cifar/table.parquet" ]; then CUDA_VISIBLE_DEVICES= $PY extract14.py --table > "$LOG/cifar_table.log" 2>&1 || fail "cifar table (logs/cifar_table.log)"; fi
  if [ ! -f "$P14/cifar/features/D3L.npy" ]; then
    pids=()
    for i in $(seq 0 $((NG - 1))); do
      CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY extract14.py --shard $i --nshards $NG > "$LOG/cifar_extract_$i.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail "cifar extract (logs/cifar_extract_*.log)"; done
    CUDA_VISIBLE_DEVICES= $PY extract14.py --merge > "$LOG/cifar_merge.log" 2>&1 || fail "cifar merge"
  fi
  run_workers cifar cifar14.py
  status "cifar done"
fi
if has online; then status "online start"; run_two_parts online online14.py; status "online done"; fi
if has ana; then status "ana start"; CUDA_VISIBLE_DEVICES= $PY ana14.py > "$LOG/ana14.log" 2>&1 || fail "ana14"; grep -q ANA14_DONE "$LOG/ana14.log" || fail "ana14 did not finish"; status "ana done"; fi
status "P14 DONE steps=$STEPS"
