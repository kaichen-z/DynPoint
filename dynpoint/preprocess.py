"""Stage 1a (Sec 3.2): optical flow, monocular depth, masks and consistent (affine-corrected) depth.

Produces `work/priors.pt` with everything stage 1b / stage 2 need:
    depth      (N,H,W)   consistent depth  D_hat_t
    depth_raw  (N,H,W)   raw network prediction
    affine     (N,3)     alpha, beta, use_inverse
    flows      {(t,t'): (2,H,W) float16}   RAFT flow t -> t'
    corr       {(t,t'): (H,W) bool}        correspondence mask  M_c
    edge       (N,H,W) bool                geometric-edge mask  M_g (True = reliable)
    dyn        (N,H,W) bool                dynamic-object mask  (True = dynamic)
    nbrs       {t: [t', ...]}              the 2K temporal neighbours of t
    center (3), scale (float)              world normalisation for the MLPs
"""
import os
import time

import numpy as np
import torch

from . import geometry as G
from .data import Scene
from .priors import DepthNet, FlowNet


def neighbours(t, n, K):
    """The 2K temporally nearest frames (asymmetric at sequence ends so every frame has 2K references)."""
    cands = sorted((abs(s - t), s) for s in range(n) if s != t)
    return sorted(s for _, s in cands[:2 * K])


@torch.no_grad()
def run(scene_dir, work_dir, K=4, depth_backend="midas", depth_ckpt=None, device="cuda", corr_eps=3.0,
        dyn_mode="auto", sampson_thresh=2.0, max_frames=None, log=print, trust_depth=False):
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    scene = Scene(scene_dir, device, max_frames=max_frames)
    N, H, W = scene.n, scene.h, scene.w
    log(f"[pre] scene {scene_dir}: {N} frames {W}x{H}, motion masks: {scene.motion is not None}")

    # ---- flow for every (t, t') pair ----------------------------------------------------
    flownet = FlowNet(device)
    nbrs = {t: neighbours(t, N, K) for t in range(N)}
    pairs = sorted({(t, s) for t in range(N) for s in nbrs[t]} | {(s, t) for t in range(N) for s in nbrs[t]})
    flows = {}
    bs = 8
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        a = torch.stack([scene.images[t] for t, _ in chunk])
        b = torch.stack([scene.images[s] for _, s in chunk])
        f = flownet(a, b)
        for (t, s), fl in zip(chunk, f):
            flows[(t, s)] = fl.half()
    del flownet
    log(f"[pre] flow: {len(flows)} pairs in {time.time() - t0:.0f}s")

    # ---- raw depth ----------------------------------------------------------------------
    if depth_backend == "gt":
        assert scene.gt_depth is not None, "depth backend 'gt' needs scene/depth/*.npy"
        depth_raw = scene.gt_depth.clone()
    else:
        dnet = DepthNet(device, depth_backend, depth_ckpt)
        depth_raw = torch.stack([dnet(scene.images[t]) for t in range(N)])
        del dnet
    log(f"[pre] depth ({depth_backend}) done {time.time() - t0:.0f}s; raw range {depth_raw.min():.3f}..{depth_raw.max():.3f}")

    # ---- masks --------------------------------------------------------------------------
    corr = {}
    for (t, s) in flows:
        corr[(t, s)] = G.correspondence_mask(flows[(t, s)].float().permute(1, 2, 0), flows[(s, t)].float().permute(1, 2, 0), corr_eps)
    edge = torch.stack([G.geometric_edge_mask(depth_raw[t]) for t in range(N)])
    if scene.motion is not None and dyn_mode != "sampson":
        dyn = scene.motion.clone()
    else:
        dyn = torch.zeros(N, H, W, dtype=torch.bool, device=device)
        for t in range(N):
            for s in nbrs[t]:
                dyn[t] |= G.sampson_dynamic_mask(flows[(t, s)].float().permute(1, 2, 0), scene.K[t], scene.c2w[t], scene.c2w[s], sampson_thresh)
        log(f"[pre] sampson dynamic masks: mean dynamic fraction {dyn.float().mean():.3f}")
    dyn_dil = torch.stack([~G.erode_bool(~dyn[t], 8) for t in range(N)])   # grow dynamic region a little

    # ---- triangulation + affine depth fit (Eq 2) ------------------------------------------
    # one parameterisation per scene: disparity networks (Depth-Anything) are aligned in inverse-depth space
    fit_mode = {"dav2": "disp", "midas": "depth", "gt": "depth"}[depth_backend]
    depth = torch.zeros_like(depth_raw)
    affine = torch.zeros(N, 3)
    stats = []
    for t in range(N):
        tri, msk, par = [], [], []
        for s in nbrs[t]:
            fl = flows[(t, s)].float().permute(1, 2, 0)
            td, ok = G.triangulate_midpoint(fl, scene.K[t], scene.c2w[t], scene.c2w[s])
            m = ok & corr[(t, s)] & edge[t] & ~dyn_dil[t] & (td > 0)
            tri.append(td)
            msk.append(m)
            par.append(fl.norm(dim=-1).clamp(max=30.0))       # parallax: larger baseline -> more reliable
        tri, msk, par = torch.stack(tri), torch.stack(msk), torch.stack(par)
        inter = msk.all(0)
        if inter.sum() < 500:            # fall back to per-pair masks when the intersection is too small
            inter = None
        m_all = msk if inter is None else inter[None].expand_as(msk)
        # remove outliers in triangulated depth (beyond 3 MAD of the masked median)
        vals = tri[m_all]
        med = vals.median()
        mad = (vals - med).abs().median() + 1e-6
        m_all = m_all & ((tri - med).abs() < 6 * mad)
        pred_rep = depth_raw[t][None].expand_as(tri)
        if trust_depth:                  # depth prior already metric-consistent with the poses (e.g. Depth-Anything-3)
            a, b, md = 1.0, 0.0, "depth"
        else:
            a, b, md = G.fit_affine_depth(pred_rep, tri, m_all, weight=par, mode=fit_mode)
        depth[t] = G.apply_affine(depth_raw[t], a, b, md).clamp(min=1e-3)
        affine[t] = torch.tensor([a, b, float(md == "disp")])
        err = ((depth[t][None].expand_as(tri) - tri).abs()[m_all]).mean()
        stats.append((int(m_all.sum()), float(err), float(med)))
    log("[pre] affine fit per frame (n_px, L1 vs triangulated, median tri depth): " +
        ", ".join(f"{t}:{s[0]}/{s[1]:.3f}/{s[2]:.2f}" for t, s in enumerate(stats) if t % max(1, N // 8) == 0))
    log(f"[pre] fit mode {fit_mode}: alpha median {affine[:, 0].median():.5f}, beta median {affine[:, 1].median():.4f}, disparity-mode frames: {int(affine[:, 2].sum())}/{N}")

    # ---- world normalisation (static points) -----------------------------------------------
    pts = []
    for t in range(0, N, max(1, N // 16)):
        p = G.backproject(depth[t], scene.K[t], scene.c2w[t])[~dyn[t] & edge[t]]
        pts.append(p[torch.randperm(len(p), device=device)[:20000]])
    pts = torch.cat(pts)
    center = pts.median(0).values
    scale = (pts - center).norm(dim=-1).quantile(0.9).item()
    log(f"[pre] world center {center.cpu().numpy().round(3)}, scale {scale:.3f}")

    out = dict(depth=depth.cpu(), depth_raw=depth_raw.cpu(), affine=affine, flows={k: v.cpu() for k, v in flows.items()},
               corr={k: v.cpu() for k, v in corr.items()}, edge=edge.cpu(), dyn=dyn.cpu(), nbrs=nbrs,
               center=center.cpu(), scale=scale, K=K, n=N, h=H, w=W)
    torch.save(out, os.path.join(work_dir, "priors.pt"))
    log(f"[pre] saved {work_dir}/priors.pt in {time.time() - t0:.0f}s")
    return out
