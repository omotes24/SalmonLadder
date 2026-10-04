#!/bin/bash
# Amendment 03 (three additional encoders in B): features on the u1x table, the u1w / u2w runs, the feature-quality
# indicator, the encoding cost (idle GPU, at the end), the B summary and the independent re-computation.
# Cached weights only (probe7b.py fetched them before). Status: logs/p7b.status
P7=/home/omote/reprise_p7_20261003
PY=/home/omote/granood_ke/.venv/bin/python
cd $P7/code
export PYTHONPATH=$P7/code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$P7/logs; S=$L/p7b.status
st() { echo "$(date -u +%FT%TZ) $*" >> $S; }
V=D3S,D3SP,RN50
T=$P7/features/u1x/table.parquet; O=$P7/features/u1x
st "start"
grep -q PROBE7B_DONE $L/probe7b.log || { st "FAILED: the probe did not finish"; exit 1; }
sha256sum -c amendment_03.sha256 > /dev/null 2>&1 || { st "FAILED: amendment_03.json hash"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY extract7.py --table $T --out $O --models $V --shard 0 --nshards 4 --limit 96 > $L/x7b_smoke.log 2>&1 || { st "FAILED smoke"; exit 1; }
st "smoke ok: $(tail -1 $L/x7b_smoke.log | cut -c1-300)"
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY extract7.py --table $T --out $O --models $V --shard $g --nshards 4 > $L/x7b_$g.log 2>&1 &
done
wait
$PY extract7.py --table $T --out $O --models $V --nshards 4 --merge > $L/x7b_merge.log 2>&1 || { st "FAILED merge"; exit 1; }
st "features merged: $(tail -1 $L/x7b_merge.log | cut -c1-200)"
bash run_engine.sh P7B "0 1 2 3" 2 u1w:$V u2w:$V
grep -q ENGINE_P7B_DONE $L/engine_P7B.status || { st "FAILED engine"; exit 1; }
st "engine: $(grep done $L/engine_P7B.status | cut -d' ' -f2- | tr '\n' ';' | cut -c1-300)"
for v in D3S D3SP RN50; do
  n1=$(ls $P7/results/u1w/$v 2>/dev/null | wc -l); n2=$(ls $P7/results/u2w/$v 2>/dev/null | wc -l)
  [ "$n1" = 45 ] && [ "$n2" = 15 ] || st "INCOMPLETE $v: u1w $n1 / 45, u2w $n2 / 15"
done
CUDA_VISIBLE_DEVICES=0 $PY ncm7.py u1 $V > $L/ncm_u1_p7b.log 2>&1 || st "FAILED ncm"
until [ -z "$(pgrep -u omote -f 'run7.py|dev7.py|x4.py|extract7.py|base7.py|ncm7.py')" ]; do sleep 10; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $L/bench7b.gpu 2>&1
CUDA_VISIBLE_DEVICES=1 $PY bench7.py S14,D3S,D3SP,RN50 bench7b.json > $L/bench7b.log 2>&1 || st "FAILED bench"
$PY - <<'PYEOF' >> $L/bench7b.log 2>&1
import json
from p7common import RESULTS
new, f = json.loads((RESULTS / "bench7b.json").read_text()), RESULTS / "bench7.json"
old = json.loads(f.read_text())
print("S14 timed again:", new["S14"], "earlier:", old["S14"])
for k in ("D3S", "D3SP", "RN50"):                      # the reference S14 keeps its first measurement
    old[k] = new[k]
f.write_text(json.dumps(old, indent=1) + "\n")
print("bench7.json:", sorted(old))
PYEOF
[ -f $P7/results/an_b.before_p7b.json ] || cp $P7/results/an_b.json $P7/results/an_b.before_p7b.json
$PY an_b.py > $L/an_b_p7b.log 2>&1 || { st "FAILED an_b"; exit 1; }
$PY - <<'PYEOF' > $L/an_b_p7b.cmp 2>&1
import json
from p7common import RESULTS
a, b = json.loads((RESULTS / "an_b.before_p7b.json").read_text()), json.loads((RESULTS / "an_b.json").read_text())
worst, n = 0.0, 0
def walk(x, y, path):
    global worst, n
    if isinstance(x, dict):
        for k in x:
            assert k in y, path + [k]
            walk(x[k], y[k], path + [k])
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        worst, n = max(worst, abs(x - y)), n + 1
    else:
        assert x == y, (path, x, y)
for U in ("U1", "U2"):
    assert set(a[U]["sets"]) <= set(b[U]["sets"])
    walk(a[U]["sets"], b[U]["sets"], [U])
    walk(a[U]["H_B"], b[U]["H_B"], [U, "H_B"])
print("registered sets:", len(a["U1"]["sets"]), "now:", len(b["U1"]["sets"]), "; values compared:", n, "; largest change:", worst)
print("AN_B_UNCHANGED" if worst == 0.0 else "AN_B_CHANGED")
PYEOF
st "an_b: $(tail -1 $L/an_b_p7b.cmp)"
$PY verify7.py > $L/verify7_p7b.log 2>&1
st "P7B_DONE $(tail -1 $L/verify7_p7b.log | cut -c1-200)"
