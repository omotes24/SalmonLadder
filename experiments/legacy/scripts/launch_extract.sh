#!/usr/bin/env bash
# Feature extraction for the additional evaluation sets on GPUs 0-2 (GPU 3 is finishing the combined TINS runs).
set -uo pipefail
W=$HOME/vins_gonogo_20260925; cd $W
PY=/home/omote/granood_ke/.venv/bin/python
export PYTHONPATH=$W PYTHONUNBUFFERED=1 VINS_WORK=$W
mkdir -p extra_feats/logs
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/extract_extra.py --datasets in_val inat sun places dtd --models dino clipb16 > extra_feats/logs/g0.log 2>&1 &
( for i in $(seq 1 360); do grep -q "imagenet-sketch ok" $HOME/datasets/extra_ood/logs/download.log && break; sleep 10; done
  CUDA_VISIBLE_DEVICES=1 $PY scripts/extract_extra.py --datasets in_v2 in_r in_sketch --models dino clipb16 > extra_feats/logs/g1.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=2 nohup $PY scripts/extract_extra.py --datasets oo_test --models clipl14 > extra_feats/logs/g2.log 2>&1 &
wait
echo "[$(date -u +%FT%TZ)] extraction finished" >> extra_feats/logs/launcher.log
