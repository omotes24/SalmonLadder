#!/bin/bash
# R5 (2) entrance at matched ID admission and (3) conformality audit (dev only). Waits for the Stage A pipeline.
set -uo pipefail
W=$HOME/vins_gonogo_20260925; PY=/home/omote/granood_ke/.venv/bin/python; cd $W
L=$W/r5/logs; mkdir -p $L
st() { echo "$(date -u +%H:%M:%S) $*" >> $L/pipeline2.status; }
st "waiting for stage A"
until grep -q "DONE\|FAILED" $L/pipeline.status 2>/dev/null; do sleep 60; done
grep -q "FAILED" $L/pipeline.status && { st "stage A failed; stop"; exit 1; }
st "start"
$PY -m pytest -q tests/test_r5.py > $L/pytest2.log 2>&1 || { st "FAILED pytest"; exit 1; }
[ -f extra_feats/in_v2.dinol14.pt ] || CUDA_VISIBLE_DEVICES=0 $PY scripts/extract_extra.py --datasets in_v2 --models dinol14 > $L/extract_v2l14.log 2>&1
st "v2 l14 ok"
for d in 0 1 2 3 4; do ( $PY scripts/r5_entrance.py --phase tune --draws $d --workers 4 > $L/ent_tune_$d.log 2>&1 || st "FAILED tune $d" ) & done
wait
$PY scripts/r5_entrance.py --phase select > $L/ent_select.log 2>&1 || { st "FAILED select"; exit 1; }
st "entrance selected"
for d in 0 1 2 3 4; do ( $PY scripts/r5_entrance.py --phase eval --devs dev2 --draws $d --workers 4 > $L/ent_eval2_$d.log 2>&1 || st "FAILED eval2 $d" ) & done
wait
for d in 0 1 2 3 4; do ( $PY scripts/r5_entrance.py --phase eval --devs dev1 --draws $d --workers 4 > $L/ent_eval1_$d.log 2>&1 || st "FAILED eval1 $d" ) & done
wait
st "entrance eval done"
for d in 0 1 2 3 4; do ( $PY scripts/r5_entrance.py --phase inject --draws $d --workers 1 > $L/ent_inject_$d.log 2>&1 || st "FAILED inject $d" ) & done
wait
st "inject done"
$PY scripts/r5_audit.py --phase run --workers 20 > $L/audit.log 2>&1 || st "FAILED audit"
st "DONE"
