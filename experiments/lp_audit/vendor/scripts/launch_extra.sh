#!/usr/bin/env bash
# TINS on Four-OOD / across-ID / ViT-L/14, chained after the feature extraction it needs.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; cd $W
PY=/home/omote/granood_ke/.venv/bin/python
export PYTHONPATH=$W PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 VINS_WORK=$W
mkdir -p extra_runs/logs
waitfor() { for i in $(seq 1 720); do ok=1; for f in "$@"; do [ -f "extra_feats/$f" ] || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
FOURB="in_val.clipb16.pt inat.clipb16.pt sun.clipb16.pt places.clipb16.pt dtd.clipb16.pt"
( CUDA_VISIBLE_DEVICES=3 $PY scripts/extract_extra.py --datasets in_val inat sun places dtd --models clipb16 > extra_feats/logs/g3_clipb16.log 2>&1
  CUDA_VISIBLE_DEVICES=3 $PY scripts/run_tins_extra.py --task fourood --seeds 123 124 > extra_runs/logs/g3_fourood.log 2>&1 ) &
( waitfor $FOURB in_v2.clipb16.pt in_r.clipb16.pt in_sketch.clipb16.pt
  for id in in_v2 in_r in_sketch; do CUDA_VISIBLE_DEVICES=1 $PY scripts/run_tins_extra.py --task acrossid --id $id --seeds 123 > extra_runs/logs/g1_acrossid_$id.log 2>&1; done ) &
( waitfor oo_test.clipl14.pt
  CUDA_VISIBLE_DEVICES=2 $PY scripts/run_tins_extra.py --task l14 --seeds 123 > extra_runs/logs/g2_l14.log 2>&1 ) &
( waitfor $FOURB in_val.dino.pt dtd.dino.pt
  while pgrep -f "extract_extra.py --datasets in_val inat sun places dtd --models dino clipb16" > /dev/null; do sleep 20; done
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_tins_extra.py --task fourood --seeds 125 > extra_runs/logs/g0_fourood.log 2>&1 ) &
wait
echo "[$(date -u +%FT%TZ)] extra runs finished" >> extra_runs/logs/launcher.log
