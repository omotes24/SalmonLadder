#!/bin/bash
# R5 roadmap, CPU part started as soon as the stream TINS runs were done (v3 of pipeline3; nice 5 so that pipeline2's
# audit keeps priority): E1, M1/M2a/M3/M4/E5, stream evaluation (E2-E5, batch size), sensitivity, full-train E1.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/pipeline3.status; }
GR="$PY scripts/gpu_run.py"
SPECS=$W/r5/stream_specs.json
N1=$($PY -c "import json;print(' '.join(s['name'] for s in json.load(open('$SPECS')) if s['dev']=='dev1' and not s['name'].startswith(('S_batch','M3_'))))")
N2=$($PY -c "import json;print(' '.join(s['name'] for s in json.load(open('$SPECS')) if s['dev']=='dev2' and not s['name'].startswith('M3_')))")
st "start (CPU part, v3, nice 5)"
for d in 0 1 2 3 4; do ( nice -n 5 $PY scripts/r5_e1.py --draw $d --workers 3 > $L/e1_dev1_$d.log 2>&1 || st "FAILED e1 dev1 $d" ) & done
wait
for d in 0 1 2 3 4; do ( VINS_WORK=$W/dev2 nice -n 5 $PY scripts/r5_e1.py --draw $d --workers 3 > $L/e1_dev2_$d.log 2>&1 || st "FAILED e1 dev2 $d" ) & done
wait
st "e1 done"
for d in 0 1 2 3 4; do ( nice -n 5 $PY scripts/r5_mods.py --what m1m3e5 --draw $d --workers 3 > $L/mods_dev1_$d.log 2>&1 || st "FAILED mods dev1 $d" ) & done
wait
for d in 0 1 2 3 4; do ( VINS_WORK=$W/dev2 nice -n 5 $PY scripts/r5_mods.py --what m1m3e5 --draw $d --workers 3 > $L/mods_dev2_$d.log 2>&1 || st "FAILED mods dev2 $d" ) & done
wait
st "mods done"
nice -n 5 $PY scripts/r5_streams.py --stage eval --specs $SPECS --names $N1 --workers 15 > $L/streams_eval_dev1.log 2>&1 || st "FAILED streams eval dev1"
VINS_WORK=$W/dev2 nice -n 5 $PY scripts/r5_streams.py --stage eval --specs $SPECS --names $N2 --workers 15 > $L/streams_eval_dev2.log 2>&1 || st "FAILED streams eval dev2"
nice -n 5 $PY scripts/r5_streams.py --stage eval --specs $SPECS --names S_batch16 S_batch64 S_batch256 S_batch1024 --draws 0 1 2 --workers 12 > $L/streams_eval_batch.log 2>&1 || st "FAILED streams eval batch"
st "streams eval done"
for d in 0 1 2 3 4; do ( nice -n 5 $PY scripts/r5_mods.py --what sens --draw $d --workers 3 > $L/sens_dev1_$d.log 2>&1 || st "FAILED sens $d" ) & done
wait
st "sens done"
until [ -f r5/features/in1k_train_l14/shard0of2.pt ] && [ -f r5/features/in1k_train_l14/shard1of2.pt ]; do sleep 120; done
$GR --mem 4000 --max-per-gpu 1 --log $L/e1full_dev1.log -- $PY scripts/r5_e1_full.py || st "FAILED e1full dev1"
$GR --mem 4000 --max-per-gpu 1 --log $L/e1full_dev2.log -- env VINS_WORK=$W/dev2 $PY scripts/r5_e1_full.py || st "FAILED e1full dev2"
st "DONE"
