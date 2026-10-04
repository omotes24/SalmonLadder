#!/usr/bin/env bash
# One read pass over the OpenOOD test images (CLIP L/14 + OpenOOD-style B/16), then TINS with the latter (seed 123).
set -uo pipefail
W=$HOME/vins_gonogo_20260925; cd $W
PY=/home/omote/granood_ke/.venv/bin/python
export PYTHONPATH=$W PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 VINS_WORK=$W
CUDA_VISIBLE_DEVICES=2 $PY scripts/extract_oo_dual.py > extra_feats/logs/g2_dual.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY scripts/run_tins_test.py --variant default --seeds 123 \
  --feature-file extra_feats/oo_test.clipb16_oodpre.pt --tag preproc_oodpre > test_runs/logs/g2_preproc.log 2>&1
echo "[$(date -u +%FT%TZ)] dual + preproc finished" >> extra_feats/logs/launcher.log
