"""How good is the stage-1 depth? Re-project static pixels of frame t into frame s with D_hat_t and
compare with RAFT flow (median px error per gap). Run: python tests/check_static_warp.py SCENE [WORK]"""
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, ".")
from dynpoint import geometry as G
from dynpoint.data import Scene

scene_dir = sys.argv[1]
work = sys.argv[2] if len(sys.argv) > 2 else os.path.join(scene_dir, "work")
dev = "cuda"
sc = Scene(scene_dir, dev)
P = torch.load(os.path.join(work, "priors.pt"), weights_only=False)
err, mag = defaultdict(list), defaultdict(list)
for (t, s), fl in P["flows"].items():
    fl = fl.to(dev).float().permute(1, 2, 0)
    pts = G.backproject(P["depth"][t].to(dev), sc.K[t], sc.c2w[t])
    pred = G.flow_from_points(pts, sc.K[s], sc.c2w[s], sc.h, sc.w)
    m = P["corr"][(t, s)].to(dev) & P["edge"][t].to(dev) & ~P["dyn"][t].to(dev)
    e = (pred - fl).norm(dim=-1)[m]
    err[abs(s - t)].append(e.median().item())
    mag[abs(s - t)].append(fl.norm(dim=-1)[m].median().item())
for g in sorted(err):
    print(f"gap {g}: static re-projection error median {sum(err[g]) / len(err[g]):.2f} px  (|flow| median {sum(mag[g]) / len(mag[g]):.1f} px)")
