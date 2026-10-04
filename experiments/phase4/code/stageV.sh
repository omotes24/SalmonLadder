#!/bin/bash
cd /home/omote/reprise_p4_20260928/code
P=/home/omote/granood_ke/.venv/bin/python
L=../logs; ST=$L/phase4.status
note() { echo "$1 $(date -u +%FT%TZ)" >> $ST; }
CUDA_VISIBLE_DEVICES=0 $P vlm_tta.py --jobs U1:1,U1:2 > $L/vlm_tta_g0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 $P vlm_tta.py --jobs U1:3,U1:4 > $L/vlm_tta_g1.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2 $P vlm_tta.py --jobs U1:5,U2:0 > $L/vlm_tta_g2.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 $P vlm_tta.py --jobs U3:0 > $L/vlm_tta_g3.log 2>&1 < /dev/null &
wait
n=$(ls ../results/vlm_tta/*_seed*.npz 2>/dev/null | wc -l)
[ "$n" -ge 21 ] && note "VLM_TTA2_OK $n" || { note "VLM_TTA2_FAIL $n"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $P analyze_baselines.py > $L/analyze_baselines.log 2>&1 && note BASELINES2_OK || note BASELINES2_FAIL
