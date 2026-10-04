#!/bin/bash
# Phase 7 stage 1: item tables, dataset banks, feature extraction (cached weights only). Status: logs/stage1.status
P7=/home/omote/reprise_p7_20261003
PY=/home/omote/granood_ke/.venv/bin/python
cd $P7/code
export PYTHONPATH=$P7/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=$P7/logs; S=$L/stage1.status
st() { echo "$(date -u +%FT%TZ) $*" >> $S; }
st start
$PY u1x_table.py > $L/u1x_table.log 2>&1 || { st "FAILED u1x_table"; exit 1; }
( $PY build7.py > $L/build7.log 2>&1 && st "banks built" || st "FAILED build7" ) &
BUILD=$!
X1=S14,DINO1,MAE,D3B,D3L,CLIPL,SIG2L
T=$P7/features/u1x/table.parquet; O=$P7/features/u1x
CUDA_VISIBLE_DEVICES=0 $PY extract7.py --table $T --out $O --models $X1 --shard 0 --nshards 4 --limit 96 > $L/x_smoke.log 2>&1 || { st "FAILED smoke"; exit 1; }
st "smoke ok: $(tail -1 $L/x_smoke.log | cut -c1-300)"
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models $X1 --shard $g --nshards 4 > $L/x_u1x_p1_$g.log 2>&1 &
done
wait %2 %3 %4 %5 2>/dev/null; wait
$PY extract7.py --table $T --out $O --models $X1 --nshards 4 --merge > $L/x_u1x_p1_merge.log 2>&1 || { st "FAILED merge pass1"; exit 1; }
st "u1x pass1 merged"
wait $BUILD
grep -q "banks built" $S || { st "banks missing: stop"; exit 1; }
for ds in cub cifar100 places365 inr insk; do
  T=$P7/banks/$ds/images.parquet; O=$P7/features/$ds
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models B14,L14,CLIP --shard $g --nshards 4 --batch 128 > $L/x_${ds}_$g.log 2>&1 &
  done
  wait
  $PY extract7.py --table $T --out $O --models B14,L14,CLIP --nshards 4 --merge > $L/x_${ds}_merge.log 2>&1 || { st "FAILED merge $ds"; exit 1; }
  st "$ds merged"
done
CUDA_VISIBLE_DEVICES=0 $PY text7.py cub cifar100 places365 inr insk > $L/text7.log 2>&1 || { st "FAILED text7"; exit 1; }
st "datasets done"
T=$P7/features/u1x/table.parquet; O=$P7/features/u1x
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models G14 --shard $g --nshards 4 --batch 64 --sub 32 > $L/x_u1x_g14_$g.log 2>&1 &
done
wait
$PY extract7.py --table $T --out $O --models G14 --nshards 4 --merge > $L/x_u1x_g14_merge.log 2>&1 || { st "FAILED merge G14"; exit 1; }
st "STAGE1_DONE"
