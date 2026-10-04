#!/bin/bash
# Phase 6, post hoc (after the pre-registered dev comparison): single-view and mixed-view variants on dev2 / dev1.
P5=/home/omote/reprise_p5_20261002
PY=/home/omote/granood_ke/.venv/bin/python
cd $P5/code || exit 1
export P5_RESULTS=$P5/results
st() { echo "$(date -u +%FT%TZ) $*" >> $P5/logs/d3_posthoc.status; }
CFGS=l14,d3L,d3mixB,d3mixL
count() { ls $P5/results/$1 2>/dev/null | grep -c "^$2_.*npz$"; }
done_all() { for c in l14 d3L d3mixB d3mixL; do [ "$(count $c $1)" -ge 30 ] || return 1; done; return 0; }
st "START post hoc (variants chosen after the Phase 6 dev scores were seen): $CFGS"
for dev in dev2 dev1; do
  for w in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$((w % 4)) $PY dev5.py --dev $dev --cfgs $CFGS --worker $w --nworkers 8 > $P5/logs/${dev}_ph_w$w.log 2>&1 &
  done
  wait
  for g in 0 1 2 3; do
    done_all $dev && break
    st "$dev incomplete; single worker on GPU $g"
    CUDA_VISIBLE_DEVICES=$g $PY dev5.py --dev $dev --cfgs $CFGS > $P5/logs/${dev}_phc_g$g.log 2>&1
  done
  st "RUNS_DONE $dev: l14 $(count l14 $dev), d3L $(count d3L $dev), d3mixB $(count d3mixB $dev), d3mixL $(count d3mixL $dev)"
done
$PY d3_posthoc.py dev2 dev1 > $P5/logs/d3_posthoc.log 2>&1
st "ANALYSIS_DONE rc $?"
st "POSTHOC_ALL_DONE"
