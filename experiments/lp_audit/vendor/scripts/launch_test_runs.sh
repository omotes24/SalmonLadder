#!/usr/bin/env bash
# TINS on the OpenOOD test streams: 5 order seeds, the permute-first variant, and combined streams (4 GPUs).
set -uo pipefail
W=$HOME/vins_gonogo_20260925; cd $W
PY=/home/omote/granood_ke/.venv/bin/python
export PYTHONPATH=$W PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 VINS_WORK=$W
mkdir -p test_runs/logs test_runs/tins_cache
for sub in class_prototypes negative_bank inversion_init; do
  [ -d test_runs/tins_cache/$sub ] || cp -r $HOME/ood_large_best_20260923/tins_20260925/cache/$sub test_runs/tins_cache/
done
echo "[$(date -u +%FT%TZ)] launch gpu0" >> test_runs/logs/launcher.log
( CUDA_VISIBLE_DEVICES=0 $PY scripts/run_tins_test.py --variant default --seeds 123 > test_runs/logs/g0_default_123.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_tins_test.py --variant permfirst --seeds 123 > test_runs/logs/g0_permfirst_123.log 2>&1 ) &
for i in $(seq 1 180); do [ -f test_runs/cal_clip.pt ] && break; sleep 10; done
sleep 20
echo "[$(date -u +%FT%TZ)] cal ready: $(ls -la test_runs/cal_clip.pt 2>&1); launch gpu1-3" >> test_runs/logs/launcher.log
CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/run_tins_test.py --variant default --seeds 124 125 > test_runs/logs/g1_default_124_125.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY scripts/run_tins_test.py --variant default --seeds 126 127 > test_runs/logs/g2_default_126_127.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY scripts/run_tins_test.py --variant default --seeds 123 124 125 --streams near_all far_all > test_runs/logs/g3_combined.log 2>&1 &
wait
echo "[$(date -u +%FT%TZ)] all test runs finished" >> test_runs/logs/launcher.log
