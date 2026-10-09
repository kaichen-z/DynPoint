#!/bin/bash
# Fast setting: 4k render iterations, ~7 min per Nvidia scene (about -0.3 dB PSNR / +0.05 LPIPS vs full).
# usage: bash run_fast.sh data/nsff_Umbrella [GPU] [extra flags]
cd "$(dirname "$0")"
S=$1; G=${2:-0}; shift; shift 2>/dev/null
NR_ITERS=4000 bash scripts/run_scene.sh "$S" "$G" "$@"
