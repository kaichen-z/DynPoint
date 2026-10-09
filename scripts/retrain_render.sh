#!/bin/bash
# usage: retrain_render.sh SCENE_DIR GPU [extra flags]  -- stage 2 only (priors.pt + sceneflow.pt must exist)
set -e
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
S=$1; G=$2; shift 2
export CUDA_VISIBLE_DEVICES=$G
C="--K 6 --attn --p_loo 0.7 --radius_px ${RADIUS_PX:-6} --mv_thresh ${MV:-3}"
V="--sharp ${SHARP:-2} --conf_min ${CONF_MIN:-0.2} ${EDGE:+--edge_thresh $EDGE}"
$P -m dynpoint train_render --scene $S $C --nr_iters ${NR_ITERS:-20000} "$@"
$P -m dynpoint eval --scene $S $C $V ${EVAL_VIEWS:+--max_eval_views $EVAL_VIEWS} "$@"
$P -m dynpoint video --scene $S $C $V --cam_path interp "$@"
