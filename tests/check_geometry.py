"""Geometry sanity checks on the toy scene (exact depth/cameras):
1. triangulation with *ground-truth* flow must reproduce GT depth,
2. RAFT flow error vs GT flow on static pixels,
3. affine fit on GT depth must give alpha~1, beta~0.
Run:  python tests/check_geometry.py data/toy
"""
import sys

import torch

sys.path.insert(0, ".")
from dynpoint import geometry as G
from dynpoint.data import Scene
from dynpoint.priors import FlowNet

dev = "cuda"
sc = Scene(sys.argv[1], dev)
t, s = 8, 10
d = sc.gt_depth[t]
H, W = d.shape
pts = G.backproject(d, sc.K[t], sc.c2w[t])
gt_flow = G.flow_from_points(pts, sc.K[s], sc.c2w[s], H, W)
static = ~sc.motion[t] & (d > 0)
tri, ok = G.triangulate_midpoint(gt_flow, sc.K[t], sc.c2w[t], sc.c2w[s])
m = ok & static
print(f"[1] GT-flow triangulation: median |tri-gt| = {(tri - d).abs()[m].median():.5f}  (gt median {d[m].median():.3f}), valid {m.float().mean():.3f}")
fn = FlowNet(dev)
raft = fn(sc.images[t][None], sc.images[s][None])[0].permute(1, 2, 0)
err = (raft - gt_flow).norm(dim=-1)
print(f"[2] RAFT vs GT flow (px) static: median {err[static].median():.3f} mean {err[static].mean():.3f}; dynamic: median {err[sc.motion[t]].median():.3f}; |gt flow| median {gt_flow.norm(dim=-1)[static].median():.2f}")
tri2, ok2 = G.triangulate_midpoint(raft, sc.K[t], sc.c2w[t], sc.c2w[s])
m2 = ok2 & static & G.correspondence_mask(raft, fn(sc.images[s][None], sc.images[t][None])[0].permute(1, 2, 0)) & G.geometric_edge_mask(d)
print(f"[2b] RAFT-flow triangulation: median |tri-gt| = {(tri2 - d).abs()[m2].median():.4f}, rel {((tri2 - d).abs() / d)[m2].median():.4f}, n={int(m2.sum())}")
a, b, inv = G.fit_affine_depth(d[None].expand(1, H, W), tri2[None], m2[None])
print(f"[3] affine fit on GT depth vs RAFT triangulation: alpha {a:.4f} beta {b:.4f} inv {inv}")
a, b, inv = G.fit_affine_depth(d[None], tri[None], m[None])
print(f"[3b] affine fit on GT depth vs GT triangulation: alpha {a:.4f} beta {b:.4f} inv {inv}")
