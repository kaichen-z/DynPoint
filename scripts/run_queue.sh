#!/bin/bash
# usage: run_queue.sh GPU scene1 scene2 ...   -- waits until the GPU is free, then runs scenes sequentially
cd "$(dirname "$0")/.."
G=$1; shift
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -lt 2000 ]; do sleep 30; done
for S in "$@"; do bash scripts/run_scene.sh data/$S $G > runs/q_${S}.log 2>&1; done
