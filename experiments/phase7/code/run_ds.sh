#!/bin/bash
# D: dataset runs (with the '@0' read-out for the locked extension), static references and feature-quality indicator
P7=/home/omote/reprise_p7_20261003
PY=/home/omote/granood_ke/.venv/bin/python
cd $P7/code
export PYTHONPATH=$P7/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$P7/logs
bash run_engine.sh F "$1" 3 dsw
for w in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$1 $PY base7.py $w 4 > $L/base7_$w.log 2>&1 & done; wait
CUDA_VISIBLE_DEVICES=$1 $PY ncm7.py ds > $L/ncm_ds.log 2>&1
echo "$(date -u +%FT%TZ) DS_DONE" >> $L/engine_F.status
