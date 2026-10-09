"""Make a DynPoint scene from an in-the-wild monocular video with no camera information.

Camera poses and intrinsics come from Depth-Anything-3 (feed-forward, no COLMAP); optional
segmentation masks (e.g. DAVIS annotations) become the dynamic mask.

python -m dynpoint.prepare_wild --frames DIR/*.jpg --out data/wild_camel [--masks DIR/*.png] [--max_frames 40] [--width 540]
Needs the DA3 source on PYTHONPATH and its checkpoint (`--da3 depth-anything/DA3NESTED-GIANT-LARGE` or a local dir).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from .data import write_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="glob of frames in temporal order")
    p.add_argument("--masks", default=None, help="glob of dynamic masks (non-zero = moving), same order")
    p.add_argument("--out", required=True)
    p.add_argument("--max_frames", type=int, default=40)
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--da3", default="depth-anything/DA3NESTED-GIANT-LARGE")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    frames = sorted(glob.glob(a.frames))
    masks = sorted(glob.glob(a.masks)) if a.masks else None
    step = max(1, len(frames) // a.max_frames)
    idx = list(range(0, len(frames), step))[: a.max_frames]
    frames = [frames[i] for i in idx]
    masks = [masks[i] for i in idx] if masks else None
    print(f"[wild] {len(frames)} frames (every {step}-th of {len(idx) * step})")

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(a.da3).to(a.device)
    with torch.no_grad():
        pred = model.inference(image=frames)
    ext = np.asarray(pred.extrinsics, dtype=np.float64)          # (N,4,4) world-to-camera (OpenCV)
    if ext.shape[-2:] == (3, 4):
        ext = np.concatenate([ext, np.tile(np.array([[[0, 0, 0, 1.0]]]), (len(ext), 1, 1))], 1)
    K = np.asarray(pred.intrinsics, dtype=np.float64)             # (N,3,3) at processed resolution
    dep = np.asarray(pred.depth)                                   # (N,h,w)
    ph, pw = dep.shape[-2:]
    print(f"[wild] DA3 processed {pw}x{ph}, median depth {np.median(dep):.3f}")

    for i, f in enumerate(frames):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        H0, W0 = im.shape[:2]
        W, H = a.width, int(round(H0 * a.width / W0))
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        Ki = K[i].copy()
        Ki[0] *= W / pw
        Ki[1] *= H / ph
        c2w = np.linalg.inv(ext[i])
        mo = None
        if masks:
            m = cv2.imread(masks[i], cv2.IMREAD_UNCHANGED)
            if m.ndim == 3:
                m = m.max(-1)
            mo = cv2.resize((m > 0).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
            mo = cv2.dilate(mo, np.ones((9, 9), np.uint8))
        d = cv2.resize(dep[i], (W, H), interpolation=cv2.INTER_LINEAR)
        write_frame(a.out, i, im, Ki, c2w, motion=mo, depth=d)      # DA3 depth kept as optional prior (--depth gt)
    t = np.stack([np.linalg.inv(e)[:3, 3] for e in ext])
    print(f"[wild] wrote {len(frames)} frames to {a.out}; camera path length {np.linalg.norm(np.diff(t, axis=0), axis=1).sum():.3f}, "
          f"median depth {np.median(dep):.3f} (ratio {np.linalg.norm(np.diff(t, axis=0), axis=1).sum() / np.median(dep):.3f})")


if __name__ == "__main__":
    main()
