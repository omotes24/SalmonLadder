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
status "FINAL START gpus=${GPUS[*]}"
for part in openood fourood; do
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY final11.py --part $part --worker $i --nworkers $NG > "$LOG/final_${part}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "final11 $part (logs/final_${part}_*.log)"; done
  status "FINAL $part done ($(ls "$P11/results/final/$part"/*.csv | wc -l) streams)"
done
$PY final11.py --aggregate > "$LOG/an11f.log" 2>&1 || fail "an11f (logs/an11f.log)"
cd "$P11/results/final" && tar czf "$P11/results/final_results.tgz" p11_summary.json p11_table.txt p11_rows.csv openood/*.csv openood/*.meta.json fourood/*.csv fourood/*.meta.json
status "FINAL DONE"
