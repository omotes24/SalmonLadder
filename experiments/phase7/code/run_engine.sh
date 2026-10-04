#!/bin/bash
# Phase 7 engine runs. usage: run_engine.sh <tag> "<gpu list>" <workers per gpu> <step> [<step> ...]
# steps: dev1 dev2 u1std u4std u2std u1alone x4 e1 e2 e3 ds  (optionally step:VIEWS, e.g. u1std:S14,G14)
P7=/home/omote/reprise_p7_20261003
PY=/home/omote/granood_ke/.venv/bin/python
cd $P7/code
export PYTHONPATH=$P7/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; GPUS=($2); WPG=$3; shift 3
L=$P7/logs; S=$L/engine_$TAG.status
st() { echo "$(date -u +%FT%TZ) $*" >> $S; }
NW=$(( ${#GPUS[@]} * WPG ))
st "start: gpus ${GPUS[*]}, $NW workers, steps $*"
for step in "$@"; do
  name=${step%%:*}; views=B14,L14
  [[ "$step" == *:* ]] && views=${step#*:}
  w=0
  for g in "${GPUS[@]}"; do
    for j in $(seq 1 $WPG); do
      case $name in
        dev1|dev2) cmd="dev7.py --dev $name --views $views --worker $w --nworkers $NW" ;;
        dev1r|dev2r) cmd="dev7.py --dev ${name%r} --retro --streams near --views $views --worker $w --nworkers $NW" ;;
        dev1w|dev2w) cmd="dev7.py --dev ${name%w} --within --views $views --worker $w --nworkers $NW" ;;
        x4)        cmd="x4.py --worker $w --nworkers $NW" ;;
        *)         cmd="run7.py --exp $name --views $views --worker $w --nworkers $NW $EXTRA" ;;
      esac
      CUDA_VISIBLE_DEVICES=$g $PY $cmd >> $L/eng_${TAG}_${name}_$w.log 2>&1 &
      w=$((w+1))
    done
  done
  wait
  n=$(grep -h '"seconds"' $L/eng_${TAG}_${name}_*.log 2>/dev/null | wc -l)
  e=$(grep -l "Traceback" $L/eng_${TAG}_${name}_*.log 2>/dev/null | wc -l)
  st "$step done: $n tasks logged, $e worker logs with errors"
done
st "ENGINE_${TAG}_DONE"
