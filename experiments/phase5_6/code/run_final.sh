#!/bin/bash
# Phase 5 final evaluation (after the lock and a successful dev2 confirmation). Every stage records OK/FAIL in
# logs/final.status; a failed stage stops the chain. Order (pre-registered): U4 first, then the OpenOOD test (6th use)
# and Four-OOD, each scored once with the locked method.
P5=/home/omote/reprise_p5_20261002
P=/home/omote/granood_ke/.venv/bin/python
L=$P5/logs
ST=$L/final.status
cd $P5/code
note() { echo "$1 $(date -u +%FT%TZ)" >> $ST; }
note "FINAL_START"
# A: bank
if [ ! -f $P5/banks/imagenet_pool.parquet ]; then
  $P u4_build.py > $L/u4_build.log 2>&1 && note U4_BUILD_OK || { note U4_BUILD_FAIL; exit 1; }
fi
# B: features (frozen encoders and transforms)
if [ ! -f $P5/features_u4/CLIP.npy ]; then
  for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $P u4_extract.py --shard $g --nshards 4 > $L/u4_extract_$g.log 2>&1 & done
  wait
  $P u4_extract.py --merge --nshards 4 > $L/u4_merge.log 2>&1 && note U4_FEATURES_OK || { note U4_FEATURES_FAIL; exit 1; }
fi
# C: TINS per stream
J0="1:0,2:1,3:2,5:0"; J1="1:1,2:2,4:0,5:1"; J2="1:2,3:0,4:1,5:2"; J3="2:0,3:1,4:2"
i=0
for J in $J0 $J1 $J2 $J3; do
  CUDA_VISIBLE_DEVICES=$i $P u4_tins.py --jobs $J > $L/u4_tins_g$i.log 2>&1 < /dev/null &
  i=$((i+1))
done
wait
n=$(ls $P5/tins_u4/U4_s*_seed*.npz 2>/dev/null | wc -l)
if [ "$n" -ge 45 ]; then note "U4_TINS_OK $n"; else note "U4_TINS_FAIL $n"; exit 1; fi
# D: U4, scored once
for w in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$w $P final5.py --part u4 --worker $w --nworkers 4 > $L/final_u4_w$w.log 2>&1 & done
wait
n=$(ls $P5/results_final/u4/*.npz 2>/dev/null | wc -l)
if [ "$n" -ge 45 ]; then note "U4_EVAL_OK $n"; else note "U4_EVAL_FAIL $n"; exit 1; fi
$P final_analysis.py --part u4 > $L/final_analysis_u4.log 2>&1 && note U4_ANALYSIS_OK || note U4_ANALYSIS_FAIL
# E: OpenOOD test (6th use) and Four-OOD, scored once
for w in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$w $P final5.py --part openood --worker $w --nworkers 4 > $L/final_openood_w$w.log 2>&1 & done
wait
n=$(ls $P5/results_final/openood/*.npz 2>/dev/null | wc -l)
if [ "$n" -ge 25 ]; then note "OPENOOD_EVAL_OK $n"; else note "OPENOOD_EVAL_FAIL $n"; exit 1; fi
$P final_analysis.py --part openood > $L/final_analysis_openood.log 2>&1 && note OPENOOD_ANALYSIS_OK || note OPENOOD_ANALYSIS_FAIL
for w in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$w $P final5.py --part fourood --worker $w --nworkers 4 > $L/final_fourood_w$w.log 2>&1 & done
wait
n=$(ls $P5/results_final/fourood/*.npz 2>/dev/null | wc -l)
if [ "$n" -ge 12 ]; then note "FOUROOD_EVAL_OK $n"; else note "FOUROOD_EVAL_FAIL $n"; exit 1; fi
$P final_analysis.py --part fourood > $L/final_analysis_fourood.log 2>&1 && note FOUROOD_ANALYSIS_OK || note FOUROOD_ANALYSIS_FAIL
note "FINAL_ALL_DONE"
