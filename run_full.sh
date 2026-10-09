#!/bin/bash
# Full setting (numbers in the README): 20k render iterations, ~19 min per Nvidia scene on one GPU.
# usage: bash run_full.sh data/nsff_Umbrella [GPU] [extra flags, e.g. --sf_wsfmin 0]
cd "$(dirname "$0")"
S=$1; G=${2:-0}; shift; shift 2>/dev/null
NR_ITERS=20000 bash scripts/run_scene.sh "$S" "$G" "$@"
