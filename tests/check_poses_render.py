"""Render frame t from (a) its own camera, (b) the same rig camera 12 frames later, (c) the next camera,
and compare with the training image / held-out GT. Separates pose problems from rendering problems.
Run: python tests/check_poses_render.py SCENE_DIR [extra Stage2 kwargs as k=v]"""
import os
import sys

import torch

sys.path.insert(0, ".")
from dynpoint.data import Scene
from dynpoint.train_render import Stage2

scene_dir = sys.argv[1]
kw = dict(a.split("=") for a in sys.argv[2:])
dev = "cuda"
sc = Scene(scene_dir, dev)
pri = torch.load(os.path.join(scene_dir, "work", "priors.pt"), map_location="cpu", weights_only=False)
s2 = Stage2(sc, pri, os.path.join(scene_dir, "work"), device=dev, K_nb=6, attention=True, log=lambda *a: None, **{k: float(v) for k, v in kw.items()})
s2.load()
tests = {(t, c): im for t, c, im, K, c2w in sc.test_views()}


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


for t in (1, 2):
    rgb, _, _ = s2.render(t, sc.K[t], sc.c2w[t])
    print(f"t={t} own camera           : PSNR vs training frame {psnr(rgb.permute(2, 0, 1), sc.images[t]):.2f}")
    rgb, _, _ = s2.render(t, sc.K[t + 12], sc.c2w[t + 12])
    print(f"t={t} same cam, frame t+12 : PSNR vs training frame {psnr(rgb.permute(2, 0, 1), sc.images[t]):.2f}")
    for c in (t + 1, t + 2):
        rgb, _, _ = s2.render(t, sc.K[c], sc.c2w[c])
        gt = tests[(t, c)]
        print(f"t={t} camera {c}            : PSNR vs held-out GT {psnr(rgb.permute(2, 0, 1), gt):.2f}   (vs training frame {c}: {psnr(rgb.permute(2, 0, 1), sc.images[c]):.2f}, GT vs frame {c}: {psnr(gt, sc.images[c]):.2f})")
