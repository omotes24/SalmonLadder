#!/bin/bash
# Phase 6 orchestrator (encoder swap, no tuning): weights -> features -> merge -> sanity gate -> dev runs -> analysis.
# Progress: $P5/logs/d3.status   (prereg_p6.json must be in place; its sha256 is checked)
P5=/home/omote/reprise_p5_20261002
PY=/home/omote/granood_ke/.venv/bin/python
cd $P5/code || exit 1
export P5_RESULTS=$P5/results
st() { echo "$(date -u +%FT%TZ) $*" >> $P5/logs/d3.status; }
[ -f $P5/prereg_p6.json ] && [ "$(sha256sum $P5/prereg_p6.json | cut -d' ' -f1)" = "$(cut -d' ' -f1 $P5/prereg_p6.sha256)" ] || { st "ABORT prereg missing or changed"; exit 1; }
st "START pid $$"

# 1. weights (the two approved files) and the read-out check; no image is opened
for i in $(seq 1 360); do pgrep -f "probe6.py" > /dev/null || break; sleep 5; done      # a probe started by an earlier run
if ! grep -q PROBE_DONE $P5/logs/probe6.log 2>/dev/null; then
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $PY -u probe6.py > $P5/logs/probe6.log 2>&1 && break
  done
fi
grep -q PROBE_DONE $P5/logs/probe6.log || { st "ABORT probe failed"; exit 1; }
st "PROBE_DONE"

# 2. features: one chain per GPU (shots -> dev1 -> dev2), four shards
for g in 0 1 2 3; do
  ( for s in shots dev1 dev2; do
      CUDA_VISIBLE_DEVICES=$g $PY -u extract6.py --set $s --shard $g --nshards 4 >> $P5/logs/extract6_g$g.log 2>&1 || st "extract failed: $s shard $g"
    done ) &
done
wait
st "EXTRACT_DONE"
$PY merge6.py > $P5/logs/merge6.log 2>&1
if ! grep -q MERGE_OK $P5/logs/merge6.log; then
  st "merge incomplete; second extraction pass, one shard at a time, on the first GPU that works"
  for s in shots dev1 dev2; do for sh in 0 1 2 3; do for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $PY -u extract6.py --set $s --shard $sh --nshards 4 --batch 32 >> $P5/logs/extract6_pass2.log 2>&1 && break
  done; done; done
  $PY merge6.py > $P5/logs/merge6.log 2>&1
fi
grep -q MERGE_OK $P5/logs/merge6.log || { st "ABORT merge"; exit 1; }
st "MERGE_OK"

# 3. sanity gate (train shots only)
CUDA_VISIBLE_DEVICES=0 $PY ncm6.py dev1 > $P5/logs/ncm6.log 2>&1
$PY - <<'PY' >> $P5/logs/ncm6.log 2>&1
import json, sys
r = json.load(open("/home/omote/reprise_p5_20261002/results/ncm6_dev1.json"))
ok = all(r[v]["mean"] >= 50.0 for v in ("D3B", "D3L", "D3Bs", "D3Ls"))
print("NCM_GATE_OK" if ok else "NCM_GATE_FAILED")
PY
grep -q NCM_GATE_OK $P5/logs/ncm6.log || { st "ABORT sanity gate (see logs/ncm6.log)"; exit 1; }
st "NCM_GATE_OK"

# 4. dev runs (dev2 first: it holds the primary comparisons); two workers per GPU, then a one-worker pass for leftovers
count() { ls $P5/results/$1 2>/dev/null | grep -c "^$2_.*npz$"; }
for dev in dev2 dev1; do
  for w in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY dev5.py --dev $dev --cfgs lock,d3,d3s --worker $w --nworkers 8 > $P5/logs/${dev}_d3_w$w.log 2>&1 &
  done
  wait
  if [ "$(count lock $dev)" -lt 30 ] || [ "$(count d3 $dev)" -lt 30 ] || [ "$(count d3s $dev)" -lt 30 ]; then
    st "$dev incomplete after the first pass (lock $(count lock $dev), d3 $(count d3 $dev), d3s $(count d3s $dev)); second pass"
    for w in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$w $PY dev5.py --dev $dev --cfgs lock,d3,d3s --worker $w --nworkers 4 > $P5/logs/${dev}_d3b_w$w.log 2>&1 &
    done
    wait
  fi
  for g in 0 1 2 3; do                                  # last resort: a single worker on the first GPU that works
    [ "$(count lock $dev)" -ge 30 ] && [ "$(count d3 $dev)" -ge 30 ] && [ "$(count d3s $dev)" -ge 30 ] && break
    st "$dev still incomplete; single worker on GPU $g"
    CUDA_VISIBLE_DEVICES=$g $PY dev5.py --dev $dev --cfgs lock,d3,d3s > $P5/logs/${dev}_d3c_g$g.log 2>&1
  done
  st "RUNS_DONE $dev: lock $(count lock $dev), d3 $(count d3 $dev), d3s $(count d3s $dev)"
done

# 5. analysis
$PY d3_analysis.py dev2 dev1 > $P5/logs/d3_analysis.log 2>&1
st "ANALYSIS_DONE rc $?"
st "D3_ALL_DONE"
