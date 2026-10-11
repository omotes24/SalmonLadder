#!/bin/bash
# Phase 15 on the server. Two pipelines run concurrently on two GPUs each:
#   A (features and baselines): tables -> extraction (in200_train, cifar_train, cifar_clip) -> merge -> text sides -> base15
#   B (ladder runs):            smoke -> cap_dev -> cap_test -> cap_in200 -> cap_cifar -> cap_rec -> kg -> views_c
# then ana15. STEPS selects the steps (default: all). Progress in $P15/STATUS, logs in $P15/logs.
set -uo pipefail
PY=/home/omote/granood_ke/.venv/bin/python
P15=/home/omote/reprise_p15_20261011
CODE=$P15/code; LOG=$P15/logs
mkdir -p "$LOG" "$P15/results" "$P15/tables" "$P15/features"
cd "$CODE" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P15/STATUS"; }
fail() { status "FAILED $*"; exit 1; }
GPUS=()
while IFS=, read -r idx used; do used=${used// /}; [ "${used%MiB}" -lt 1500 ] && GPUS+=("$idx"); done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed 's/, /,/')
NG=${#GPUS[@]}; [ "$NG" -ge 2 ] || fail "need at least two free GPUs"
HA=$(( NG / 2 )); GA=("${GPUS[@]:0:$HA}"); GB=("${GPUS[@]:$HA}"); NA=${#GA[@]}; NB=${#GB[@]}
STEPS=${STEPS:-tables,extract,text,base,smoke,cap_dev,cap_test,cap_in200,cap_cifar,cap_rec,kg,views,ana}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
status "P15 START gpus A=${GA[*]} B=${GB[*]} steps=$STEPS"
workers() {  # workers <gpu list name> <tag> <script> [args]: one worker per GPU of the list
  local -n G=$1; local tag=$2; shift 2
  local n=${#G[@]} pids=()
  for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$i]} $PY "$@" --worker $i --nworkers $n > "$LOG/${tag}_$i.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
pipeline_a() {
  if has tables; then
    status "A tables start"
    for t in in200_train cifar_train cifar_clip; do [ -f "$P15/tables/$t.parquet" ] || { CUDA_VISIBLE_DEVICES= $PY extra15.py --table $t > "$LOG/table_$t.log" 2>&1 || return 1; }; done
    status "A tables done"
  fi
  if has extract; then
    status "A extract start"
    for spec in "in200_train L14,D3L D3L" "cifar_train L14,D3L D3L" "cifar_clip CLIP CLIP"; do
      set -- $spec; t=$1; m=$2; last=$3
      if [ ! -f "$P15/features/$t/$last.npy" ]; then
        pids=()
        for i in $(seq 0 $((NA - 1))); do
          CUDA_VISIBLE_DEVICES=${GA[$i]} $PY extra15.py --extract $t --models $m --shard $i --nshards $NA > "$LOG/extract_${t}_$i.log" 2>&1 &
          pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || return 1; done
        CUDA_VISIBLE_DEVICES= $PY extra15.py --merge $t --models $m > "$LOG/merge_$t.log" 2>&1 || return 1
      fi
      status "A extract $t done"
    done
  fi
  if has text; then
    status "A text start"
    CUDA_VISIBLE_DEVICES=${GA[0]} $PY base15.py --text imagenet200 > "$LOG/text_imagenet200.log" 2>&1 || return 1
    CUDA_VISIBLE_DEVICES=${GA[0]} $PY base15.py --text cifar100 > "$LOG/text_cifar100.log" 2>&1 || return 1
    status "A text done"
  fi
  if has base; then
    status "A base start"
    CUDA_VISIBLE_DEVICES=${GA[0]} $PY base15.py --bench imagenet200 > "$LOG/base_imagenet200.log" 2>&1 &
    pa=$!
    CUDA_VISIBLE_DEVICES=${GA[$((NA - 1))]} $PY base15.py --bench cifar100 > "$LOG/base_cifar100.log" 2>&1 &
    pb=$!
    wait $pa || return 1; wait $pb || return 1
    status "A base done"
  fi
  return 0
}
pipeline_b() {
  if has smoke; then
    status "B smoke start"
    CUDA_VISIBLE_DEVICES=${GB[0]} $PY smoke15.py > "$LOG/smoke15.log" 2>&1 || return 1
    grep -q SMOKE15_OK "$LOG/smoke15.log" || return 1
    status "B smoke done"
  fi
  for step in dev test in200 cifar rec; do
    if has cap_$step; then status "B cap_$step start"; workers GB cap_$step cap15.py --step $step || return 1; status "B cap_$step done"; fi
  done
  if has kg; then status "B kg start"; workers GB kg kg15.py || return 1; status "B kg done"; fi
  if has views; then status "B views start"; workers GB views_c views15.py || return 1; status "B views done"; fi
  return 0
}
pipeline_a > "$LOG/pipeline_a.log" 2>&1 &
PA=$!
pipeline_b > "$LOG/pipeline_b.log" 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
[ $RA -eq 0 ] || status "FAILED pipeline A (see logs/pipeline_a.log and the step logs)"
[ $RB -eq 0 ] || status "FAILED pipeline B (see logs/pipeline_b.log and the step logs)"
if has ana; then status "ana start"; CUDA_VISIBLE_DEVICES= $PY ana15.py > "$LOG/ana15.log" 2>&1 || fail "ana15"; grep -q ANA15_DONE "$LOG/ana15.log" || fail "ana15 did not finish"; status "ana done"; fi
[ $RA -eq 0 ] && [ $RB -eq 0 ] && status "P15 DONE steps=$STEPS" || status "P15 ENDED WITH FAILURES steps=$STEPS"
