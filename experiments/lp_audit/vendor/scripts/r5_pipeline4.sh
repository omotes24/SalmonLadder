#!/bin/bash
# R5 Phase 2 / M3 on scene-type far OOD: Places365 val (non-Four-OOD, non-ImageNet classes) as a dev far source.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/pipeline4.status; }
GR="$PY scripts/gpu_run.py"
SPECS=$W/r5/stream_specs.json
st "waiting for the Places365 download"
until grep -q "DONE\|FAILED" /home/omote/datasets/places365_val256/download.status 2>/dev/null; do sleep 30; done
grep -q FAILED /home/omote/datasets/places365_val256/download.status && { st "download failed"; exit 1; }
[ -f r5/extra/places_meta.parquet ] || $GR --mem 3500 --max-per-gpu 3 --log $L/places_feats.log -- nice -n 5 $PY scripts/r5_places.py || { st "FAILED places features"; exit 1; }
st "places features ok"
$PY scripts/r5_streams.py --stage build --specs $SPECS --names M3_places > $L/places_build_dev1.log 2>&1 || { st "FAILED build dev1"; exit 1; }
VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage build --specs $SPECS --names M3_places > $L/places_build_dev2.log 2>&1 || { st "FAILED build dev2"; exit 1; }
st "streams built"
( $GR --mem 2000 --max-per-gpu 3 --log $L/places_tins_dev1.log -- nice -n 5 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names M3_places --record-cal || st "FAILED tins dev1" ) &
sleep 5
( $GR --mem 2000 --max-per-gpu 3 --log $L/places_tins_dev2.log -- env VINS_WORK=$W/dev2 nice -n 5 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names M3_places --record-cal || st "FAILED tins dev2" ) &
wait
st "tins done"
nice -n 5 $PY scripts/r5_streams.py --stage eval --specs $SPECS --names M3_places --workers 5 > $L/places_eval_dev1.log 2>&1 || st "FAILED eval dev1"
VINS_WORK=$W/dev2 nice -n 5 $PY scripts/r5_streams.py --stage eval --specs $SPECS --names M3_places --workers 5 > $L/places_eval_dev2.log 2>&1 || st "FAILED eval dev2"
st "DONE"
