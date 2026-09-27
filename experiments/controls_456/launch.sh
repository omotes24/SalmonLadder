#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$PWD"
PYTHON_BIN=/home/omote/granood_ke/.venv/bin/python
mkdir -p pids logs status tests
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q src/test_controls.py > tests/pytest.log 2>&1
start() {
  local name="$1"; shift
  if [[ -f "pids/$name.pid" ]] && kill -0 "$(cat "pids/$name.pid")" 2>/dev/null; then
    echo "$name already running: $(cat "pids/$name.pid")"
    return
  fi
  nohup nice -n 10 "$@" > "logs/daemon_$name.log" 2>&1 < /dev/null &
  echo "$!" > "pids/$name.pid"
  echo "$name started: $!"
}
start prepare env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u "$ROOT_DIR/src/prepare.py"
start cpu_probe env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u "$ROOT_DIR/src/scheduler.py" --cpu-probe
start cpu_baselines env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u "$ROOT_DIR/src/scheduler.py" --cpu-baselines
for gpu in 0 1 2 3; do
  start "gpu$gpu" "$PYTHON_BIN" -u "$ROOT_DIR/src/scheduler.py" --gpu "$gpu"
done
start reporter env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u "$ROOT_DIR/src/report.py" --watch
"$PYTHON_BIN" - <<'PY'
import json,time,os
from pathlib import Path
root=Path.cwd()
pids={p.stem:int(p.read_text()) for p in (root/'pids').glob('*.pid')}
out={'launched_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'root':str(root),'pids':pids,'gpu_policy':'wait for prior R5 GPU phase and actual idle card'}
(root/'launch.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
PY
