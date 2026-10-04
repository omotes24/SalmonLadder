#!/bin/bash
# R5 Stage A pipeline (dev only). Launched with nohup; progress in r5/logs/pipeline.status
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/pipeline.status; }
fail() { st "FAILED: $*"; exit 1; }
st "start"
$PY -m pytest -q tests/test_r5.py > $L/pytest.log 2>&1 || fail "pytest"
st "pytest ok"
[ -f r5/shots/draws.parquet ] || { $PY scripts/r5_shots.py > $L/shots.log 2>&1 || fail "shots"; }
st "shots ok"
( CUDA_VISIBLE_DEVICES=0 $PY scripts/r5_features.py --model clip > $L/feat_clip.log 2>&1 || st "FAILED feat clip" ) &
( CUDA_VISIBLE_DEVICES=1 $PY scripts/r5_features.py --model dino > $L/feat_dino.log 2>&1 || st "FAILED feat dino" ) &
( CUDA_VISIBLE_DEVICES=2 $PY scripts/r5_tins.py --stage run --draws orig > $L/tins_dev1_orig.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=2 $PY scripts/r5_tins.py --stage setup --draws 0 1 2 3 4 > $L/setup_dev1.log 2>&1 || st "FAILED dev1 orig/setup" ) &
( CUDA_VISIBLE_DEVICES=3 VINS_WORK=$W/dev2 $PY scripts/r5_tins.py --stage run --draws orig > $L/tins_dev2_orig.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=3 VINS_WORK=$W/dev2 $PY scripts/r5_tins.py --stage setup --draws 0 1 2 3 4 > $L/setup_dev2.log 2>&1 || st "FAILED dev2 orig/setup" ) &
wait
[ -f r5/features/shots.clip.pt ] && [ -f r5/features/shots.dino_vitl14.pt ] || fail "features missing"
[ -f r5/dev1/tins_setup/draw4.pt ] && [ -f r5/dev2/tins_setup/draw4.pt ] || fail "setups missing"
st "features + setups + orig TINS ok"
( VINS_WORK=$W $PY scripts/r5_eval.py --draw orig --check-v4 > $L/eval_dev1_orig.log 2>&1 || st "FAILED eval dev1 orig" ) &
( CUDA_VISIBLE_DEVICES=0 $PY scripts/r5_tins.py --stage run --draws 0 1 > $L/tins_dev1_a.log 2>&1 || st "FAILED tins dev1 a" ) &
( CUDA_VISIBLE_DEVICES=1 $PY scripts/r5_tins.py --stage run --draws 2 3 4 > $L/tins_dev1_b.log 2>&1 || st "FAILED tins dev1 b" ) &
( CUDA_VISIBLE_DEVICES=2 VINS_WORK=$W/dev2 $PY scripts/r5_tins.py --stage run --draws 0 1 > $L/tins_dev2_a.log 2>&1 || st "FAILED tins dev2 a" ) &
( CUDA_VISIBLE_DEVICES=3 VINS_WORK=$W/dev2 $PY scripts/r5_tins.py --stage run --draws 2 3 4 > $L/tins_dev2_b.log 2>&1 || st "FAILED tins dev2 b" ) &
wait
st "tins runs done"
( VINS_WORK=$W/dev2 $PY scripts/r5_eval.py --draw orig > $L/eval_dev2_orig.log 2>&1 || st "FAILED eval dev2 orig" ) &
for d in 0 1 2; do ( VINS_WORK=$W $PY scripts/r5_eval.py --draw $d > $L/eval_dev1_$d.log 2>&1 || st "FAILED eval dev1 $d" ) & done
wait
for d in 3 4; do ( VINS_WORK=$W $PY scripts/r5_eval.py --draw $d > $L/eval_dev1_$d.log 2>&1 || st "FAILED eval dev1 $d" ) & done
for d in 0 1; do ( VINS_WORK=$W/dev2 $PY scripts/r5_eval.py --draw $d > $L/eval_dev2_$d.log 2>&1 || st "FAILED eval dev2 $d" ) & done
wait
for d in 2 3 4; do ( VINS_WORK=$W/dev2 $PY scripts/r5_eval.py --draw $d > $L/eval_dev2_$d.log 2>&1 || st "FAILED eval dev2 $d" ) & done
wait
$PY scripts/r5_summary.py > $L/summary.log 2>&1 || fail "summary"
st "DONE"
