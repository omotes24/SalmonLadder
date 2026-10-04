#!/bin/bash
# Phase 4 orchestrator (after stageA.sh writes FEATURES_OK). Every stage records OK/FAIL in logs/phase4.status;
# stages whose inputs failed are skipped. Registered order: confirmatory evaluation first, then descriptive studies.
cd /home/omote/reprise_p4_20260928/code
R=/home/omote/reprise_p4_20260928
P=/home/omote/granood_ke/.venv/bin/python
L=$R/logs
ST=$L/phase4.status
note() { echo "$1 $(date -u +%FT%TZ)" >> $ST; }
until grep -q "FEATURES_OK" $ST 2>/dev/null; do
  grep -q "SHARD_FAIL\|MERGE_FAIL" $ST 2>/dev/null && { note "ORCH_ABORT_NO_FEATURES"; exit 1; }
  sleep 60
done
note "ORCH_START"

# B: Exp 4 design (features only), augmentation features, text side of the VLM baselines for U3
$P exp4_design.py > $L/exp4_design.log 2>&1 && note EXP4_DESIGN_OK || note EXP4_DESIGN_FAIL
( CUDA_VISIBLE_DEVICES=0 $P extract_aug.py > $L/extract_aug.log 2>&1 && note AUG_OK || note AUG_FAIL ) &
( CUDA_VISIBLE_DEVICES=1 $P vlm_tta.py --text-only --jobs U3:0 > $L/vlm_text_u3.log 2>&1 < /dev/null && note VLM_TEXT_OK || note VLM_TEXT_FAIL ) &

# C: TINS (pinned upstream, fixed hyper-parameters) per stream, 4 GPUs
J0="U1:1:0,U1:1:1,U1:1:2,U1:2:0,U2:0:0,U2:0:1,U3:0:0"
J1="U1:2:1,U1:2:2,U1:3:0,U1:3:1,U2:0:2,U3:0:1,U3:0:2"
J2="U1:3:2,U1:4:0,U1:4:1,U1:4:2,U2:0:3,U3:0:3"
J3="U1:5:0,U1:5:1,U1:5:2,U2:0:4,U3:0:4"
i=0
for J in $J0 $J1 $J2 $J3; do
  CUDA_VISIBLE_DEVICES=$i $P tins_bank.py --jobs $J > $L/tins_g$i.log 2>&1 &
  i=$((i+1))
done
wait
n=$(ls $R/tins/*_seed*.npz 2>/dev/null | wc -l)
if [ "$n" -ge 75 ]; then note "TINS_OK $n"; else note "TINS_FAIL $n"; exit 1; fi

# D: confirmatory evaluation (locked selections) and registered analysis
for w in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$((w % 4)) $P evaluate.py --worker $w --nworkers 8 > $L/eval_w$w.log 2>&1 &
done
wait
n=$(ls $R/results/eval/*_seed*.parquet 2>/dev/null | wc -l)
if [ "$n" -ge 75 ]; then note "EVAL_OK $n"; else note "EVAL_FAIL $n"; exit 1; fi
$P analyze_eval.py > $L/analyze_eval.log 2>&1 && note ANALYZE_EVAL_OK || note ANALYZE_EVAL_FAIL

# E: descriptive studies (post-lock, no selection), two lanes per GPU
lane() { local name=$1; shift; ( "$@" && note "${name}_OK" || note "${name}_FAIL" ); }
( lane VLM_TTA env CUDA_VISIBLE_DEVICES=0 $P vlm_tta.py > $L/vlm_tta.log 2>&1 < /dev/null
  lane BASELINES env CUDA_VISIBLE_DEVICES=0 $P analyze_baselines.py > $L/analyze_baselines.log 2>&1
  lane EXP5 env CUDA_VISIBLE_DEVICES=0 $P operating.py --exp exp5 > $L/exp5.log 2>&1 ) &
( lane EXP6 env CUDA_VISIBLE_DEVICES=0 $P operating.py --exp exp6 > $L/exp6.log 2>&1 ) &
if grep -q AUG_OK $ST && grep -q EXP4_DESIGN_OK $ST; then
  for w in 0 1 2 3 4 5; do
    g=$((1 + w / 2))
    if [ $w -ge 4 ]; then
      ( lane EXP4_W$w env CUDA_VISIBLE_DEVICES=$g $P exp4_run.py --worker $w --nworkers 6 > $L/exp4_w$w.log 2>&1
        lane EXP8_W$((w - 4)) env CUDA_VISIBLE_DEVICES=$g $P exp8.py --worker $((w - 4)) --nworkers 2 > $L/exp8_w$((w - 4)).log 2>&1 ) &
    else
      ( lane EXP4_W$w env CUDA_VISIBLE_DEVICES=$g $P exp4_run.py --worker $w --nworkers 6 > $L/exp4_w$w.log 2>&1 ) &
    fi
  done
else
  ( lane EXP8_W0 env CUDA_VISIBLE_DEVICES=3 $P exp8.py --worker 0 --nworkers 2 > $L/exp8_w0.log 2>&1 ) &
  ( lane EXP8_W1 env CUDA_VISIBLE_DEVICES=3 $P exp8.py --worker 1 --nworkers 2 > $L/exp8_w1.log 2>&1 ) &
fi
wait
$P analyze_exp4.py > $L/analyze_exp4.log 2>&1 && note ANALYZE_EXP4_OK || note ANALYZE_EXP4_FAIL
$P theory_checks.py --t3 > $L/theory_t3.log 2>&1 && note THEORY_T3_OK || note THEORY_T3_FAIL

# F: timing-sensitive runs, one process per GPU
( lane BOUNDED_A env CUDA_VISIBLE_DEVICES=1 $P bounded_p4.py --settings 0:fifo,1000:fifo,1000:random,1000:diversity,8192:fifo > $L/bounded_a.log 2>&1 ) &
( lane BOUNDED_B env CUDA_VISIBLE_DEVICES=2 $P bounded_p4.py --settings 8192:random,8192:diversity,32768:fifo,32768:random,32768:diversity > $L/bounded_b.log 2>&1 ) &
( lane RESOURCES env CUDA_VISIBLE_DEVICES=0 $P resources.py > $L/resources.log 2>&1 ) &
wait
note "PHASE4_ALL_DONE"
