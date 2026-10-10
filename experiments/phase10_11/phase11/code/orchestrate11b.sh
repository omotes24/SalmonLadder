#!/bin/bash
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P11=/home/omote/reprise_p11_20261010
CODE=$P11/code; LOG=$P11/logs
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P11/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}; [ "$NG" -ge 1 ] || fail "no free GPU"
status "SWEEP START gpus=${GPUS[*]}"
for part in openood fourood; do
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY sweep11.py --part $part --worker $i --nworkers $NG > "$LOG/sweep_${part}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "sweep11 $part (logs/sweep_${part}_*.log)"; done
  status "SWEEP $part done ($(ls "$P11/results/sweep/$part"/*.npz | wc -l) streams)"
done
$PY an11b.py > "$LOG/an11b.log" 2>&1 || fail "an11b (logs/an11b.log)"
status "SWEEP DONE"
