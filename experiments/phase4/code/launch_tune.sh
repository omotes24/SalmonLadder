#!/bin/bash
# dev1 tuning workers: 2 per listed GPU
cd /home/omote/reprise_p4_20260928/code
P=/home/omote/granood_ke/.venv/bin/python
mkdir -p ../logs
N=$(( $# * 2 )); w=0
for g in "$@"; do
  for j in 0 1; do
    CUDA_VISIBLE_DEVICES=$g nohup nice -n 5 $P dev1_tune.py --worker $w --nworkers $N > ../logs/tune_w$w.log 2>&1 &
    echo "worker $w gpu $g pid $!"
    w=$((w+1))
  done
done
