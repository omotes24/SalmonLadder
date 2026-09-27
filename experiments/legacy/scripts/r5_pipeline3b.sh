#!/bin/bash
# R5 roadmap Phase 1/2 on dev: custom streams (E2-E5, batch sensitivity), E1 (+ full-train kNN/Maha++), M1/M3/E5,
# sensitivity. v2: every GPU step goes through scripts/gpu_run.py (shared machine; picks a GPU with free memory,
# retries after CUDA OOM); TINS on custom streams is split per (dev, draw) and per batch-size run.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/pipeline3.status; }
GR="$PY scripts/gpu_run.py"
SPECS=$W/r5/stream_specs.json
st "waiting for stage A (v2)"
until grep -q "DONE\|FAILED" $L/pipeline.status 2>/dev/null; do sleep 30; done
grep -q "FAILED" $L/pipeline.status && { st "stage A failed; stop"; exit 1; }
st "start (GPU part)"
$PY scripts/r5_streams.py --stage build --specs $SPECS > $L/streams_build_dev1.log 2>&1 || st "FAILED build dev1"
VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage build --specs $SPECS > $L/streams_build_dev2.log 2>&1 || st "FAILED build dev2"
$GR --mem 3500 --log $L/streams_aug_dev1.log -- $PY scripts/r5_streams.py --stage augfeat --specs $SPECS || st "FAILED aug dev1"
$GR --mem 3500 --log $L/streams_aug_dev2.log -- env VINS_WORK=$W/dev2 $PY scripts/r5_streams.py --stage augfeat --specs $SPECS || st "FAILED aug dev2"
st "streams built"
N1=$($PY -c "import json;print(' '.join(s['name'] for s in json.load(open('$SPECS')) if s['dev']=='dev1' and not s['name'].startswith('S_batch')))")
N2=$($PY -c "import json;print(' '.join(s['name'] for s in json.load(open('$SPECS')) if s['dev']=='dev2'))")
( $GR --mem 3000 --max-per-gpu 1 --log $L/dino1.log -- nice -n 10 $PY scripts/r5_extra_feats.py --what dino1 --workers 6 || st "FAILED dino1" ) &
for d in 0 1 2 3 4; do
  ( $GR --mem 2000 --max-per-gpu 3 --log $L/streams_tins_dev1_d$d.log -- nice -n 5 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $N1 --draws $d || st "FAILED tins dev1 d$d" ) &
  sleep 5
  ( $GR --mem 2000 --max-per-gpu 3 --log $L/streams_tins_dev2_d$d.log -- env VINS_WORK=$W/dev2 nice -n 5 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $N2 --draws $d || st "FAILED tins dev2 d$d" ) &
  sleep 5
done
for n in S_batch16 S_batch64 S_batch256 S_batch1024; do for d in 0 1 2; do
  ( $GR --mem 2000 --max-per-gpu 3 --log $L/streams_tins_${n}_d$d.log -- nice -n 5 $PY scripts/r5_streams.py --stage tins --specs $SPECS --names $n --draws $d || st "FAILED tins $n d$d" ) &
  sleep 5
done; done
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
st "sens done"
until [ -f r5/features/in1k_train_l14/shard0of2.pt ] && [ -f r5/features/in1k_train_l14/shard1of2.pt ]; do sleep 120; done
$GR --mem 4000 --max-per-gpu 1 --log $L/e1full_dev1.log -- $PY scripts/r5_e1_full.py || st "FAILED e1full dev1"
$GR --mem 4000 --max-per-gpu 1 --log $L/e1full_dev2.log -- env VINS_WORK=$W/dev2 $PY scripts/r5_e1_full.py || st "FAILED e1full dev2"
st "DONE"
