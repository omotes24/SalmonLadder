#!/bin/bash
# R5 roadmap Phase 1/2 on dev: custom streams (E2-E5, batch sensitivity), E1, M1/M3/E5, sensitivity.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/pipeline3.status; }
SPECS=$W/r5/stream_specs.json
st "waiting for stage A"
until grep -q "DONE\|FAILED" $L/pipeline.status 2>/dev/null; do sleep 60; done
grep -q "FAILED" $L/pipeline.status && { st "stage A failed; stop"; exit 1; }
st "start (GPU part)"
$PY scripts/r5_streams.py --stage build --specs $SPECS > $L/streams_build_dev1.log 2>&1 || st "FAILED build dev1"
VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage build --specs $SPECS > $L/streams_build_dev2.log 2>&1 || st "FAILED build dev2"
CUDA_VISIBLE_DEVICES=0 $PY scripts/r5_streams.py --stage augfeat --specs $SPECS > $L/streams_aug_dev1.log 2>&1 || st "FAILED aug dev1"
CUDA_VISIBLE_DEVICES=0 VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage augfeat --specs $SPECS > $L/streams_aug_dev2.log 2>&1 || st "FAILED aug dev2"
st "streams built"
N1=$($PY -c "import json;print(' '.join(s['name'] for s in json.load(open('$SPECS')) if s['dev']=='dev1' and not s['name'].startswith('S_batch')))")
N2=$($PY -c "import json;print(' '.join(s['name'] for s in json.load(open('$SPECS')) if s['dev']=='dev2'))")
( CUDA_VISIBLE_DEVICES=0 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $N1 --draws 0 1 2 > $L/streams_tins_dev1_a.log 2>&1 || st "FAILED tins dev1 a" ) &
( CUDA_VISIBLE_DEVICES=1 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $N1 --draws 3 4 > $L/streams_tins_dev1_b.log 2>&1 || st "FAILED tins dev1 b" ) &
( CUDA_VISIBLE_DEVICES=2 VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $N2 --draws 0 1 2 > $L/streams_tins_dev2_a.log 2>&1 || st "FAILED tins dev2 a" ) &
( CUDA_VISIBLE_DEVICES=3 VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $N2 --draws 3 4 > $L/streams_tins_dev2_b.log 2>&1 || st "FAILED tins dev2 b" ) &
wait
( CUDA_VISIBLE_DEVICES=0 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names S_batch256 S_batch1024 --draws 0 1 2 > $L/streams_tins_batch_a.log 2>&1 || st "FAILED tins batch a" ) &
( CUDA_VISIBLE_DEVICES=1 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names S_batch64 --draws 0 1 2 > $L/streams_tins_batch_b.log 2>&1 || st "FAILED tins batch b" ) &
( CUDA_VISIBLE_DEVICES=2 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names S_batch16 --draws 0 > $L/streams_tins_batch_c.log 2>&1 || st "FAILED tins batch c" ) &
( CUDA_VISIBLE_DEVICES=3 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names S_batch16 --draws 1 2 > $L/streams_tins_batch_d.log 2>&1 || st "FAILED tins batch d" ) &
wait
st "stream TINS done; waiting for pipeline2 (CPU)"
until grep -q "DONE" $L/pipeline2.status 2>/dev/null; do sleep 60; done
st "start (CPU part)"
for d in 0 1 2 3 4; do ( $PY scripts/r5_e1.py --draw $d --workers 4 > $L/e1_dev1_$d.log 2>&1 || st "FAILED e1 dev1 $d" ) & done
wait
for d in 0 1 2 3 4; do ( VINS_WORK=$W/dev2 $PY scripts/r5_e1.py --draw $d --workers 4 > $L/e1_dev2_$d.log 2>&1 || st "FAILED e1 dev2 $d" ) & done
wait
st "e1 done"
for d in 0 1 2 3 4; do ( $PY scripts/r5_mods.py --what m1m3e5 --draw $d --workers 4 > $L/mods_dev1_$d.log 2>&1 || st "FAILED mods dev1 $d" ) & done
wait
for d in 0 1 2 3 4; do ( VINS_WORK=$W/dev2 $PY scripts/r5_mods.py --what m1m3e5 --draw $d --workers 4 > $L/mods_dev2_$d.log 2>&1 || st "FAILED mods dev2 $d" ) & done
wait
st "mods done"
$PY scripts/r5_streams.py --stage eval --specs $SPECS --names $N1 --workers 20 > $L/streams_eval_dev1.log 2>&1 || st "FAILED streams eval dev1"
VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage eval --specs $SPECS --names $N2 --workers 20 > $L/streams_eval_dev2.log 2>&1 || st "FAILED streams eval dev2"
$PY scripts/r5_streams.py --stage eval --specs $SPECS --names S_batch16 S_batch64 S_batch256 S_batch1024 --draws 0 1 2 --workers 12 > $L/streams_eval_batch.log 2>&1 || st "FAILED streams eval batch"
st "streams eval done"
for d in 0 1 2 3 4; do ( $PY scripts/r5_mods.py --what sens --draw $d --workers 4 > $L/sens_dev1_$d.log 2>&1 || st "FAILED sens $d" ) & done
wait
st "DONE"
