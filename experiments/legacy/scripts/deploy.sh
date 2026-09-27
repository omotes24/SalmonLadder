#!/usr/bin/env bash
# One-time setup on hades: pinned upstream clone + hook commit, dataset symlink, criteria hash.
set -euo pipefail
WORK=${VINS_WORK:-$HOME/vins_gonogo_20260925}
PIN=194759d716b27534bf2e8eeb0d71f5c4f0dabc40
cd "$WORK"
if [ ! -d tins/.git ]; then
  git clone -q "$HOME/ood_large_best_20260923/tins_20260925/repo" tins
  git -C tins checkout -q -B vins-gonogo "$PIN"
  git -C tins -c user.name="Claude (VINS go/no-go)" -c user.email="noreply@anthropic.com" am -q "$WORK"/patches/0001-*.patch
fi
echo "--- tins HEAD"; git -C tins log --oneline -2
echo "--- diff vs upstream"; git -C tins diff --stat "$PIN" HEAD
test -z "$(git -C tins status --porcelain --untracked-files=no)"
mkdir -p datasets
ln -sfn /home/omote/datasets/openood_official/images_largescale/imagenet_1k datasets/ImageNet
ls -la datasets
echo "--- criteria"; sha256sum criteria.json
