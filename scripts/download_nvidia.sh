#!/bin/bash
# Nvidia Dynamic Scene dataset, NSFF release with multi-view ground truth (3.4 GB), then convert.
# Needs `pip install gdown`.
set -e
cd "$(dirname "$0")/.."
mkdir -p data && cd data
[ -f nvidia_data_full.zip ] || gdown 16Pp40VAr3_PuyFmCBpLLeSjuyhTHcu0T -O nvidia_data_full.zip
[ -d nvfull_tmp ] || unzip -q nvidia_data_full.zip -d nvfull_tmp
cd ..
${PYTHON:-python} - <<'PY'
import cv2, glob, os
from dynpoint.data import convert_nvidia_scene, add_nvidia_test_views
for S in ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "Playground", "Skating-2", "Truck-2", "Umbrella"]:
    D, dst = f"data/nvfull_tmp/nvidia_data_full/{S}/dense", f"data/nsff_{S}"
    if os.path.isdir(os.path.join(dst, "test")):
        continue
    m = cv2.imread(sorted(glob.glob(D + "/motion_masks/*.png"))[0], 0)
    convert_nvidia_scene(D, dst, mask_dir="motion_masks", mask_dynamic_is_white=(m > 127).mean() < 0.5, max_width=540)
    add_nvidia_test_views(D, dst)
    print("converted", S)
PY
