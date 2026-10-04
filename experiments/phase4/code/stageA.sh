#!/bin/bash
cd /home/omote/reprise_p4_20260928/code
P=/home/omote/granood_ke/.venv/bin/python
L=../logs
echo "STAGE_A_START $(date -u +%FT%TZ)" >> $L/phase4.status
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s $P extract.py --shard $s --nshards 4 > $L/extract$s.log 2>&1 & done
wait
for s in 0 1 2 3; do [ -f ../features/shard${s}of4.npz ] || { echo "SHARD_FAIL $s $(date -u +%FT%TZ)" >> $L/phase4.status; exit 1; }; done
$P extract.py --merge --nshards 4 > $L/extract_merge.log 2>&1 || { echo "MERGE_FAIL $(date -u +%FT%TZ)" >> $L/phase4.status; exit 1; }
echo "FEATURES_OK $(date -u +%FT%TZ)" >> $L/phase4.status
