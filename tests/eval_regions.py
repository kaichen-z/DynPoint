"""Split novel-view error into static / dynamic regions using NSFF mv_masks.
Run: python tests/eval_regions.py EVAL_DIR NSFF_DENSE_DIR"""
import glob
import os
import sys

import cv2
import numpy as np

ev, D = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(ev, "pred_*.png"))):
    stem = os.path.basename(f)[5:-4]
    t, c = (int(x) for x in stem.split("_"))
    p = cv2.imread(f).astype(np.float64) / 255
    g = cv2.imread(os.path.join(ev, f"gt_{stem}.png")).astype(np.float64) / 255
    m = cv2.imread(os.path.join(D, "mv_masks", f"{t:05d}", f"cam{c + 1:02d}.png"), 0)
    m = cv2.resize(m, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    se = ((p - g) ** 2).mean(-1)
    rows.append((se.mean(), se[m].mean() if m.any() else np.nan, se[~m].mean(), m.mean()))
r = np.array(rows)
psnr = lambda x: -10 * np.log10(np.nanmean(x))
print(f"{len(r)} views: PSNR all {psnr(r[:, 0]):.2f} | dynamic region {psnr(r[:, 1]):.2f} | static region {psnr(r[:, 2]):.2f} | dynamic area {r[:, 3].mean() * 100:.1f}%")
print("per-view mean PSNR (all):", np.round(-10 * np.log10(r[:, 0]), 1).tolist()[:22])
# headroom check: PSNR after a per-view per-channel affine colour fit of pred to GT (exposure / white-balance differences between rig cameras)
cc = []
for f in sorted(glob.glob(os.path.join(ev, "pred_*.png"))):
    stem = os.path.basename(f)[5:-4]
    p = cv2.imread(f).astype(np.float64) / 255
    g = cv2.imread(os.path.join(ev, f"gt_{stem}.png")).astype(np.float64) / 255
    q = p.copy()
    for ch in range(3):
        a, b = np.polyfit(p[..., ch].ravel(), g[..., ch].ravel(), 1)
        q[..., ch] = a * p[..., ch] + b
    cc.append(((q - g) ** 2).mean())
print(f"PSNR after per-view colour affine fit: {psnr(np.array(cc)):.2f}")
