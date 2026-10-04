#!/bin/bash
cd /home/omote/reprise_p4_20260928/code
P=/home/omote/granood_ke/.venv/bin/python
$P build_banks.py > ../logs/build_banks.log 2>&1 || { echo BUILD_FAILED >> ../logs/build_banks.log; exit 1; }
( CUDA_VISIBLE_DEVICES=2 $P extract.py --shard 0 --nshards 4 > ../logs/extract0.log 2>&1 && CUDA_VISIBLE_DEVICES=2 $P extract.py --shard 1 --nshards 4 > ../logs/extract1.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 $P extract.py --shard 2 --nshards 4 > ../logs/extract2.log 2>&1 && CUDA_VISIBLE_DEVICES=3 $P extract.py --shard 3 --nshards 4 > ../logs/extract3.log 2>&1 ) &
wait
$P extract.py --merge --nshards 4 > ../logs/extract_merge.log 2>&1
echo CHAIN_DONE >> ../logs/extract_merge.log
