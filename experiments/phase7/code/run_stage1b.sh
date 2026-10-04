#!/bin/bash
# Phase 7 stage 1 (continued after the OOM of shard 3): GPUs 0-2 only; GPU 3 is left to the engine runs.
P7=/home/omote/reprise_p7_20261003
PY=/home/omote/granood_ke/.venv/bin/python
cd $P7/code
export PYTHONPATH=$P7/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=$P7/logs; S=$L/stage1.status
st() { echo "$(date -u +%FT%TZ) $*" >> $S; }
st "stage1b start (shard 3 of pass 1 failed with CUDA OOM next to the regression test; re-split over GPUs 0-2)"
X1=S14,DINO1,MAE,D3B,D3L,CLIPL,SIG2L
T=$P7/features/u1x/table.parquet; O=$P7/features/u1x
until [ -f $O/S14-DINO1-MAE-D3B-D3L-CLIPL-SIG2L.shard0of4.npz ] && [ -f $O/S14-DINO1-MAE-D3B-D3L-CLIPL-SIG2L.shard1of4.npz ] && [ -f $O/S14-DINO1-MAE-D3B-D3L-CLIPL-SIG2L.shard2of4.npz ]; do
  pgrep -f "extract7.py --table $T" > /dev/null || { st "FAILED pass1: shards 0-2 stopped without output"; exit 1; }
  sleep 20
done
st "pass1 shards 0-2 done"
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models $X1 --parent 3,4 --shard $g --nshards 3 > $L/x_u1x_p1_3_$g.log 2>&1 &
done
wait
$PY extract7.py --table $T --out $O --models $X1 --merge > $L/x_u1x_p1_merge.log 2>&1 || { st "FAILED merge pass1"; exit 1; }
st "u1x pass1 merged"
until grep -q "banks built\|FAILED build7" $S; do sleep 10; done
grep -q "banks built" $S || { st "banks missing: stop"; exit 1; }
for ds in cub cifar100 places365 inr insk; do
  T=$P7/banks/$ds/images.parquet; O=$P7/features/$ds
  for g in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models B14,L14,CLIP --shard $g --nshards 3 --batch 128 > $L/x_${ds}_$g.log 2>&1 &
  done
  wait
  $PY extract7.py --table $T --out $O --models B14,L14,CLIP --merge > $L/x_${ds}_merge.log 2>&1 || { st "FAILED merge $ds"; exit 1; }
  st "$ds merged"
done
CUDA_VISIBLE_DEVICES=0 $PY text7.py cub cifar100 places365 inr insk > $L/text7.log 2>&1 || { st "FAILED text7"; exit 1; }
st "datasets done"
T=$P7/features/u1x/table.parquet; O=$P7/features/u1x
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models G14 --shard $g --nshards 3 --batch 64 --sub 32 > $L/x_u1x_g14_$g.log 2>&1 &
done
wait
$PY extract7.py --table $T --out $O --models G14 --merge > $L/x_u1x_g14_merge.log 2>&1 || { st "FAILED merge G14"; exit 1; }
st "STAGE1_DONE"
