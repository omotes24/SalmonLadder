#!/bin/bash
# Phase 3 final evaluation (run once). Requires the frozen r5/phase3/prereg_phase3.json.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
GR="$PY scripts/gpu_run.py"
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/p3.status; }
H=$(cat r5/phase3/prereg_phase3.sha256)
[ "$(sha256sum r5/phase3/prereg_phase3.json | cut -d' ' -f1)" = "$H" ] || { st "prereg hash mismatch"; exit 1; }
st "start (prereg $H)"
( $GR --mem 3500 --max-per-gpu 2 --log $L/p3_d1_openood.log -- $PY scripts/p3_extract.py --what test --part openood --unseal $H || st "FAILED dino1 openood" ) &
sleep 5
( $GR --mem 3500 --max-per-gpu 2 --log $L/p3_d1_fourood.log -- $PY scripts/p3_extract.py --what test --part fourood --unseal $H || st "FAILED dino1 fourood" ) &
$PY scripts/p3_m2b.py --unseal $H > $L/p3_m2b.log 2>&1 || st "FAILED m2b"
$PY scripts/p3_streams.py --stage build > $L/p3_streams_build.log 2>&1 || st "FAILED streams build"
wait
st "dino1 + m2b + stream build done"
( $GR --mem 6000 --max-per-gpu 1 --log $L/p3_static_openood.log -- $PY scripts/p3_static.py --part openood || st "FAILED static openood"
  $GR --mem 6000 --max-per-gpu 1 --log $L/p3_static_fourood.log -- $PY scripts/p3_static.py --part fourood || st "FAILED static fourood" ) &
sleep 5
( $GR --mem 4000 --max-per-gpu 2 --log $L/p3_locoop.log -- $PY scripts/p3_locoop_run.py --unseal $H || st "FAILED locoop" ) &
sleep 5
( $GR --mem 4000 --max-per-gpu 2 --log $L/p3_streams_aug.log -- $PY scripts/p3_streams.py --stage augfeat --unseal $H || st "FAILED streams aug"
  for g in "E2_ratio1 E2_ratio2 E2_ratio5 E2_ratio10 E2_ratio25" "E2_ratio50 E2_cls980x1 E2_cls196x5 E2_cls49x20 E2_cls20x49" "E3_idburst E3_idfirst E3_pseudo E3_mixall E4_delayed"; do
    ( $GR --mem 3000 --max-per-gpu 2 --log $L/p3_streams_tins_${g%% *}.log -- $PY scripts/p3_streams.py --stage tins --names $g || st "FAILED streams tins ${g%% *}" ) &
    sleep 5
  done; wait ) &
wait
st "static + locoop + stream TINS done"
( $PY scripts/p3_eval.py --part openood --workers 12 > $L/p3_eval_openood.log 2>&1 || st "FAILED eval openood" ) &
( $PY scripts/p3_eval.py --part fourood --workers 8 > $L/p3_eval_fourood.log 2>&1 || st "FAILED eval fourood" ) &
wait
st "F1 done"
$PY scripts/p3_streams.py --stage eval --workers 16 > $L/p3_streams_eval.log 2>&1 || st "FAILED streams eval"
$PY scripts/p3_cub.py --stage eval --workers 9 > $L/p3_cub_eval.log 2>&1 || st "FAILED cub eval"
$PY scripts/p3_report.py > $L/p3_report.log 2>&1 || st "FAILED report"
st "DONE"
