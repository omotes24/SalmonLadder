#!/bin/bash
# Orchestration after features/*.npy exist: Exp4 design + augmentation features, TINS on U1-U3 (4 GPUs),
# confirmatory evaluation (4 GPUs), analysis. Each stage waits for the previous one.
cd /home/omote/reprise_p4_20260928/code
P=/home/omote/granood_ke/.venv/bin/python
L=../logs
until grep -q CHAIN_DONE $L/extract_merge.log 2>/dev/null; do sleep 60; done
$P exp4_design.py > $L/exp4_design.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P extract_aug.py > $L/extract_aug.log 2>&1 &
J0="U1:1:0,U1:1:1,U1:1:2,U1:2:0,U2:0:0,U2:0:1,U3:0:0"
J1="U1:2:1,U1:2:2,U1:3:0,U1:3:1,U2:0:2,U3:0:1,U3:0:2"
J2="U1:3:2,U1:4:0,U1:4:1,U1:4:2,U2:0:3,U3:0:3"
J3="U1:5:0,U1:5:1,U1:5:2,U2:0:4,U3:0:4"
i=0
for J in $J0 $J1 $J2 $J3; do
  CUDA_VISIBLE_DEVICES=$i $P tins_bank.py --jobs $J > $L/tins_g$i.log 2>&1 &
  i=$((i+1))
done
wait
echo TINS_DONE >> $L/tins_g0.log
for w in 0 1 2 3 4 5 6 7; do
  g=$((w % 4))
  CUDA_VISIBLE_DEVICES=$g $P evaluate.py --worker $w --nworkers 8 > $L/eval_w$w.log 2>&1 &
done
wait
$P analyze_eval.py > $L/analyze_eval.log 2>&1
echo EVAL_DONE >> $L/analyze_eval.log
