#!/bin/bash
# After the whole extraction: ViT-g/14 views, a second queue that meets queue X from the other end, the
# feature-quality indicator of every view and the encoding cost (on an idle GPU, at the very end).
P7=/home/omote/reprise_p7_20261003
PY=/home/omote/granood_ke/.venv/bin/python
cd $P7/code
export PYTHONPATH=$P7/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$P7/logs
V1=S14,DINO1,MAE,D3B,D3L,CLIPL,SIG2L,CLIP
V2=B14,L14,CLIP,S14,DINO1,MAE,D3B,D3L,CLIPL,SIG2L
bash run_engine.sh G "0 1 2" 2 u1w:G14 u2w:G14
EXTRA="--reverse" bash run_engine.sh Z "0 1 2" 2 u2w:$V2 u1w:$V1 e3w e1w e2w
CUDA_VISIBLE_DEVICES=0 $PY ncm7.py u1 S14,B14,L14,G14,D3B,D3L,DINO1,MAE,CLIP,CLIPL,SIG2L > $L/ncm_u1.log 2>&1
until [ -z "$(pgrep -f 'run7.py|dev7.py|x4.py|extract7.py|base7.py')" ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 $PY bench7.py > $L/bench7.log 2>&1
echo "$(date -u +%FT%TZ) G14_BENCH_DONE" >> $L/engine_G.status
