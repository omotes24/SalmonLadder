#!/bin/bash
# Phase 11 on hades: transductive read-outs (trans11.py) for both parts over the free GPUs, then an11.py.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P11=/home/omote/reprise_p11_20261010
CODE=$P11/code; LOG=$P11/logs
mkdir -p "$LOG" "$P11/results"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P11/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
status "START gpus=${GPUS[*]}"
for part in openood fourood; do
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY trans11.py --part $part --worker $i --nworkers $NG > "$LOG/trans_${part}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "trans11 $part (logs/trans_${part}_*.log)"; done
  status "TRANS $part done ($(ls "$P11/results/$part"/*.npz | wc -l) streams)"
done
$PY an11.py > "$LOG/an11.log" 2>&1 || fail "an11 (logs/an11.log)"
status "DONE"
