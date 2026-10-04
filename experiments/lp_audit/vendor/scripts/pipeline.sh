#!/usr/bin/env bash
# Usage: bash scripts/pipeline.sh smoke [first_stage]   |   bash scripts/pipeline.sh main [first_stage]
set -euo pipefail
TAG=${1:?tag (smoke|main)}
FROM=${2:-}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)        # code lives here; data/results live in WORK
WORK=${VINS_WORK:-$HOME/vins_gonogo_20260925}
PY=${VINS_PY:-/home/omote/granood_ke/.venv/bin/python}
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30 VINS_WORK=$WORK
export CUDA_VISIBLE_DEVICES=${VINS_GPU:-1}
cd "$CODE"
mkdir -p "$WORK/runs/$TAG"
STAGES="$WORK/runs/$TAG/stages.json"
[ -f "$STAGES" ] || echo '{}' > "$STAGES"

RUN_TAG=$TAG
if [ "$TAG" = smoke ]; then
  ORDER=(splits tests_unit prepare_tins_clip dino dview tests_regression tins shadow report)
  SEEDS="123"
elif [ "$TAG" = dev2 ]; then     # confirmation split: run with VINS_WORK=<work>/dev2 and the VINS_* exclusions
  ORDER=(splits tests_unit prepare_tins_clip dino dview tins report confirm)
  SEEDS="123 124 125"
  RUN_TAG=main
else
  ORDER=(tins shadow report)
  SEEDS="123 124 125"
fi

record() { "$PY" - "$STAGES" "$1" "$2" <<'EOF'
import json, sys
path, name, secs = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(path)); data[name] = secs; json.dump(data, open(path, "w"), indent=1)
EOF
}

run_stage() {
  case "$1" in
    splits)            "$PY" -m vins.splits ;;
    tests_unit)        "$PY" -m pytest -q -p no:cacheprovider tests/test_dview.py tests/test_quantile.py tests/test_splits.py tests/test_sealing.py tests/test_stream_order.py ;;
    prepare_tins_clip) "$PY" scripts/prepare_tins_clip.py ;;
    dino)              "$PY" scripts/extract_dino.py ;;
    dview)             "$PY" scripts/compute_dview.py ;;
    tests_regression)  "$PY" -m pytest -q -p no:cacheprovider tests/test_regression.py tests/test_sealing.py ;;
    tins)              "$PY" scripts/run_tins.py --tag "$RUN_TAG" --seeds $SEEDS ;;
    shadow)            "$PY" scripts/shadow.py --tag "$RUN_TAG" --seed 123 ;;
    report)            "$PY" scripts/make_report.py --tag "$RUN_TAG" ;;
    confirm)           "$PY" scripts/confirm_dev2.py ;;
    *) echo "unknown stage $1"; exit 2 ;;
  esac
}

started=0
[ -z "$FROM" ] && started=1
echo "=== pipeline $TAG on GPU $CUDA_VISIBLE_DEVICES ($(date -u +%FT%TZ))"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for stage in "${ORDER[@]}"; do
  if [ $started -eq 0 ]; then
    [ "$stage" = "$FROM" ] && started=1 || continue
  fi
  t0=$(date +%s)
  echo "=== [$(date -u +%FT%TZ)] stage $stage"
  run_stage "$stage"
  t1=$(date +%s)
  record "$stage" $((t1 - t0))
  echo "=== stage $stage done in $((t1 - t0))s"
done
echo "=== pipeline $TAG finished ($(date -u +%FT%TZ))"
