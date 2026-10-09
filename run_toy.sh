#!/bin/bash
# End-to-end smoke test on a synthetic scene with exact depth / cameras / masks (~2 min on one GPU).
set -e
cd "$(dirname "$0")"
P=${PYTHON:-python}
S=${1:-data/toy}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
$P -m dynpoint toy --scene $S --toy_frames 20
$P -m dynpoint all --scene $S --depth gt --K 3 --sf_iters 800 --nr_iters 1500 --rays 1024
