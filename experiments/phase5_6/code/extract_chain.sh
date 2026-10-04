#!/bin/bash
# usage: extract_chain.sh <gpu> <shard> <nshards>
PY=/home/omote/granood_ke/.venv/bin/python
cd /home/omote/reprise_p5_20261002/code
for job in "shots dino" "dev1 dino" "shots vlm" "dev1 vlm" "dev2 dino" "dev2 vlm"; do
  set -- $job $1 $2 $3 2>/dev/null
done
