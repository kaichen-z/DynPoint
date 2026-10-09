#!/bin/bash
# usage: run_scene.sh SCENE_DIR GPU [extra flags...]   -- full pipeline with the refined config
set -e
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
S=$1; G=$2; shift 2
export CUDA_VISIBLE_DEVICES=$G
C="--K 6 --attn --p_loo 0.7 --radius_px ${RADIUS_PX:-6} --mv_thresh ${MV:-3}"
V="--sharp ${SHARP:-2} --conf_min ${CONF_MIN:-0.2} ${EDGE:+--edge_thresh $EDGE}"   # inference-time cleanup for eval/video (set EDGE=0.12 for benchmark scenes, leave unset for noisy wild depth)
$P -m dynpoint preprocess --scene $S --depth dav2 --K 6 "$@"
$P -m dynpoint train_flow --scene $S --K 6 --refine_depth ${REFINE:-all} --sf_wsfmin ${SFMIN:-1} "$@"
$P -m dynpoint train_render --scene $S $C --nr_iters ${NR_ITERS:-20000} "$@"
$P -m dynpoint eval --scene $S $C $V ${EVAL_VIEWS:+--max_eval_views $EVAL_VIEWS} "$@"
$P -m dynpoint video --scene $S $C $V --cam_path interp "$@"
$P -m dynpoint.visualize_flow --scene $S --gap 3        # stage-1 figure: warp with depth only vs depth + scene flow
