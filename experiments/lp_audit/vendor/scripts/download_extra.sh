#!/usr/bin/env bash
# Extra public data for the paper: CLIP ViT-L/14, ImageNet-V2/-R/-Sketch, Four-OOD (MOS iNaturalist/SUN/Places, DTD).
set -uo pipefail
D=$HOME/datasets/extra_ood; mkdir -p $D/logs $HOME/vins_gonogo_20260925/weights; cd $D
get() { # url out
  for k in 1 2 3; do wget -q -c --tries=5 --timeout=60 -O "$2" "$1" && return 0; sleep 30; done; return 1; }
log() { echo "[$(date -u +%FT%TZ)] $*" >> $D/logs/download.log; }
log start
W=$HOME/vins_gonogo_20260925/weights/ViT-L-14.pt
get https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt $W && log "ViT-L-14 $(sha256sum $W | cut -c1-64)"
get https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz imagenetv2-mf.tar.gz && tar xzf imagenetv2-mf.tar.gz && log "imagenet-v2 ok $(find imagenetv2-matched-frequency* -type f | wc -l)"
get https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar imagenet-r.tar && tar xf imagenet-r.tar && log "imagenet-r ok $(find imagenet-r -type f | wc -l)"
for n in Places SUN iNaturalist; do
  get http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/$n.tar.gz $n.tar.gz && tar xzf $n.tar.gz && log "$n ok $(find $n -type f | wc -l)"
done
get https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz dtd.tar.gz && tar xzf dtd.tar.gz && log "dtd ok $(find dtd -type f -name '*.jpg' | wc -l)"
get https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip imagenet-sketch.zip && unzip -q -o imagenet-sketch.zip -d imagenet-sketch && log "imagenet-sketch ok $(find imagenet-sketch -type f | wc -l)"
log done
