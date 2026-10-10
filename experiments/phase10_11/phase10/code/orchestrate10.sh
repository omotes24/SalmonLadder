#!/bin/bash
# Salmon Ladder Phase 10 on hades (runs detached under nohup; progress in $P10/STATUS, logs in $P10/logs).
#  0. reproduction check of Phase 5 (frozen views, first OpenOOD stream)      final10.py --check
#  1. DINOv3 B/16 and L/16 features of the OpenOOD and Four-OOD images         extract10.py (sharded over the free GPUs)
#  2. official VLM post-processors (TANL, AdaNeg, AdaNeg_TA, NegLabel)          vlm10.py (text once, then per stream)
#  3. frozen read-outs of the new views on every stream                        final10.py --views D3B,D3L
#  4. analysis of every views x read-out x base configuration                  analysis10.py
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P10=/home/omote/reprise_p10_20261010
CODE=$P10/code; LOG=$P10/logs
mkdir -p "$LOG" "$P10/results" "$P10/features" "$P10/cache"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P10/STATUS"; }
fail() { status "FAILED $*"; exit 1; }

# GPUs with less than 1.5 GB in use by others
GPUS=()
while IFS=, read -r idx used; do
  used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx")
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}
[ "$NG" -ge 1 ] || fail "no free GPU"
status "START gpus=${GPUS[*]}"

# 0. reproduction check in the background on the last free GPU (compared with results_final of Phase 5); the feature
#    extraction meanwhile uses the other free GPUs (all of them when only one is free)
CUDA_VISIBLE_DEVICES=${GPUS[$((NG - 1))]} $PY final10.py --part openood --check > "$LOG/check_openood.log" 2>&1 &
CHECK_PID=$!
if [ "$NG" -ge 2 ]; then XG=("${GPUS[@]:0:$((NG - 1))}"); else XG=("${GPUS[@]}"); fi
NX=${#XG[@]}

# 1. features (sharded over the extraction GPUs), then merge
for part in openood fourood; do
  pids=()
  for i in $(seq 0 $((NX - 1))); do
    CUDA_VISIBLE_DEVICES=${XG[$i]} $PY extract10.py --part $part --models D3B,D3L --shard $i --nshards $NX \
      > "$LOG/extract_${part}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "extract $part (see logs/extract_${part}_*.log)"; done
  $PY extract10.py --part $part --models D3B,D3L --nshards $NX --merge > "$LOG/merge_${part}.log" 2>&1 || fail "merge $part"
  status "FEATURES $part done"
done
wait $CHECK_PID; status "CHECK exit=$? $(tail -n 1 "$LOG/check_openood.log")"

# 2. official VLM post-processors: text side once, then the streams over the free GPUs
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY vlm10.py --text-only > "$LOG/vlm_text.log" 2>&1 || fail "vlm text (logs/vlm_text.log)"
status "VLM text ready"
for part in openood fourood; do
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY vlm10.py --part $part --worker $i --nworkers $NG > "$LOG/vlm_${part}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "vlm $part (logs/vlm_${part}_*.log)"; done
  status "VLM $part done ($(ls "$P10/results/vlm/$part"/*.npz | wc -l) streams)"
done

# 3. frozen read-outs of the new views (one worker per free GPU; a worker holds the static views of both new views)
for part in openood fourood; do
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY final10.py --part $part --views D3B,D3L --worker $i --nworkers $NG \
      > "$LOG/final_${part}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "final10 $part (logs/final_${part}_*.log)"; done
  status "READOUTS $part done ($(ls "$P10/results/$part/D3B+D3L"/*.npz | wc -l) streams)"
done

# 4. analysis
$PY analysis10.py > "$LOG/analysis10.log" 2>&1 || fail "analysis10 (logs/analysis10.log)"
status "DONE"
