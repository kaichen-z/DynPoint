"""Stage 2 (Sec 3.4 + Appendix Fig 8): hierarchical neural point clouds and point-based rendering.

Cloud for target frame t = points of t itself plus points of its 2K neighbours
moved to time t by the scene-flow MLP, at H=3 pyramid levels (Eq 7). Every
point remembers (source frame, level, pixel) so its feature is gathered from
cached per-frame CNN feature maps at render time, and its confidence gamma is a
learnable per-source-pixel map (as in Point-NeRF).

Neighbour search: points are projected into the query camera and bucketed per
pixel; a ray only looks at the points of its 3x3 pixel neighbourhood, so the
cost is O(rays x candidates) instead of O(rays x points).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as G

LEVELS = (1, 2, 4)


# ----------------------------------------------------------------------------- clouds
@torch.no_grad()
def mv_valid_map(fm, s, t, thresh_static=2.0, thresh_dyn=4.0, color_tol=None):
    """Which pixels of reference frame s, lifted with its depth and moved to time t by the scene flow,
    land where the optical flow s->t says they should (full-res bool map). Removes flying points."""
    P = fm.P
    if (s, t) not in P["flows"]:
        return None
    H, W = P["h"], P["w"]
    p_s, dyn = fm.points(s, 1)
    p_t = fm.displace(p_s, s, t - s, dyn)
    uv, z = G.project(p_t.permute(1, 2, 0), fm.scene.K[t], fm.scene.c2w[t])
    flow = P["flows"][(s, t)].to(fm.device).float().permute(1, 2, 0)
    grid = G.pixel_grid(H, W, fm.device)[..., :2]
    err = (uv - grid - flow).norm(dim=-1)
    tgt = grid + flow
    inb = (tgt[..., 0] >= 0) & (tgt[..., 0] <= W - 1) & (tgt[..., 1] >= 0) & (tgt[..., 1] <= H - 1)
    thr = torch.where(dyn, torch.full_like(err, thresh_dyn), torch.full_like(err, thresh_static))
    valid = ((err < thr) | ~inb) & (z > 0)
    if color_tol is not None:
        # photometric check: the point, seen from the target camera, must show the colour the target image has there
        # (flying points at silhouettes land on the background and fail this)
        uvn = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1)
        c_t = F.grid_sample(fm.scene.images[t][None], uvn[None], align_corners=True, padding_mode="border")[0].permute(1, 2, 0)
        c_s = fm.scene.images[s].permute(1, 2, 0)
        inb2 = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
        valid = valid & (((c_t - c_s).abs().mean(-1) < color_tol) | ~inb2)
    # points whose flow target leaves the image cannot be checked and are kept (they fill the novel view's borders)
    return valid


@torch.no_grad()
def build_cloud(fm, t, K_nb, exclude_self=False, k_dyn=None, edge_thresh=None, mv_thresh=None, dyn_band=0, color_tol=None):
    """fm: sceneflow.FlowModel. Returns dict of (N,...) tensors on fm.device.
    exclude_self=True drops frame t's own points (leave-one-out training: the target camera then
    behaves like a novel view that must be synthesised from the warped neighbours only)."""
    xyz, src, lvl, pix = [], [], [], []
    frames = ([] if exclude_self else [t]) + [s for s in fm.P["nbrs"][t] if abs(s - t) <= K_nb]
    for s in frames:
        mv = mv_valid_map(fm, s, t, mv_thresh if mv_thresh is not None else 1e9, 2 * mv_thresh if mv_thresh is not None else 1e9,
                          color_tol) if ((mv_thresh is not None or color_tol is not None) and s != t) else None
        for li, L in enumerate(LEVELS):
            p, dyn = fm.points(s, L)
            p = fm.displace(p, s, t - s, dyn)                          # (3,h,w) at time t
            h, w = p.shape[-2:]
            d = fm.depth[s]
            if L > 1:
                d = F.interpolate(d[None, None], scale_factor=1.0 / L, mode="area")[0, 0]
            ok = (d > 1e-3).flatten()
            if edge_thresh is not None and s != t:            # drop "flying" points of references at depth discontinuities
                gx = torch.zeros_like(d)
                gy = torch.zeros_like(d)
                gx[:, 1:-1] = (d[:, 2:] - d[:, :-2]).abs() / 2
                gy[1:-1] = (d[2:] - d[:-2]).abs() / 2
                rel = torch.maximum(gx, gy) / d.clamp(min=1e-6)
                rel = F.max_pool2d(rel[None, None], 3, 1, 1)[0, 0]
                ok = ok & (rel < edge_thresh).flatten()
            if k_dyn is not None and abs(s - t) > k_dyn:      # dynamic points only from the nearest references
                ok = ok & ~dyn.flatten()
            if dyn_band and s != t:                            # drop reference points in a band around the dynamic-mask boundary
                k = 2 * int(round(dyn_band / L)) + 1
                if k >= 3:
                    dil = F.max_pool2d(dyn[None, None].float(), k, 1, k // 2)[0, 0] > 0.5
                    ero = 1 - F.max_pool2d(1 - dyn[None, None].float(), k, 1, k // 2)[0, 0] > 0.5
                    ok = ok & ~(dil & ~ero).flatten()
            if mv is not None:                                 # multi-view consistency: keep only well-tracked points
                m = mv.float()[None, None]
                if L > 1:
                    m = F.interpolate(m, scale_factor=1.0 / L, mode="area")
                ok = ok & (m[0, 0] > 0.5).flatten()
            xyz.append(p.flatten(1).T[ok])
            src.append(torch.full((int(ok.sum()),), s, device=fm.device, dtype=torch.long))
            lvl.append(torch.full((int(ok.sum()),), li, device=fm.device, dtype=torch.long))
            pix.append(torch.arange(h * w, device=fm.device)[ok])
    return dict(xyz=torch.cat(xyz), src=torch.cat(src), lvl=torch.cat(lvl), pix=torch.cat(pix))


class FrameFeatures(nn.Module):
    """Cached frozen backbone features per frame and level, plus learnable confidences."""

    def __init__(self, featnet, scene, device, dim=32):
        super().__init__()
        with torch.no_grad():
            full = torch.cat([featnet.backbone(scene.images[i:i + 1]) for i in range(scene.n)])   # (N,C,H,W)
        self.maps = []
        self.conf = nn.ParameterList()
        self.resid = nn.ParameterList()          # Point-NeRF style per-point optimisable feature residual
        for L in LEVELS:
            m = full if L == 1 else F.interpolate(full, scale_factor=1.0 / L, mode="area")
            self.maps.append(m.half().flatten(2))                    # (N,C,h*w)
            self.conf.append(nn.Parameter(torch.zeros(scene.n, m.shape[-2] * m.shape[-1], device=device)))
            self.resid.append(nn.Parameter(torch.zeros(scene.n, dim, m.shape[-2] * m.shape[-1], device=device)))
        self.pruned = [torch.zeros(scene.n, m_.shape[-1], dtype=torch.bool, device=device) for m_ in self.maps]
        self.C, self.dim = full.shape[1], dim

    @torch.no_grad()
    def prune(self, conf_thresh=0.1):
        """Point-NeRF style pruning: points whose learned confidence fell below `conf_thresh` are removed."""
        n_new = 0
        for li in range(len(LEVELS)):
            dead = torch.sigmoid(self.conf[li]) < conf_thresh
            n_new += int((dead & ~self.pruned[li]).sum())
            self.pruned[li] |= dead
        return n_new, sum(int(p.sum()) for p in self.pruned), sum(p.numel() for p in self.pruned)

    def pruned_mask(self, src, lvl, pix):
        out = torch.zeros(len(src), dtype=torch.bool, device=src.device)
        for li in range(len(LEVELS)):
            m = lvl == li
            if m.any():
                out[m] = self.pruned[li][src[m], pix[m]]
        return out

    def gather(self, src, lvl, pix):
        """-> frozen features (M,C) float, confidence logits (M,), feature residuals (M,dim)"""
        f = torch.empty(len(src), self.C, device=src.device)
        c = torch.empty(len(src), device=src.device)
        r = torch.empty(len(src), self.dim, device=src.device)
        for li in range(len(LEVELS)):
            m = lvl == li
            if m.any():
                f[m] = self.maps[li][src[m], :, pix[m]].float()
                c[m] = self.conf[li][src[m], pix[m]]
                r[m] = self.resid[li][src[m], :, pix[m]]
        return f, c, r


# ----------------------------------------------------------------------------- renderer
class MLP(nn.Sequential):
    def __init__(self, i, h, o, n=3, act_out=None):
        layers = []
        for k in range(n):
            layers += [nn.Linear(i if k == 0 else h, h if k < n - 1 else o)]
            if k < n - 1:
                layers.append(nn.ReLU(inplace=True))
        if act_out is not None:
            layers.append(act_out)
        super().__init__(*layers)


class PointRenderer(nn.Module):
    def __init__(self, feat_in, dim=32, hidden=128, n_neighbors=8, radius=0.03, max_per_cell=24, n_samples=24,
                 n_freq_dir=4, attention=False, radius_px=None, sharp=1.0, conf_min=0.0):
        super().__init__()
        self.attention = attention
        self.sharp, self.conf_min = sharp, conf_min   # weight exponent (>1 = closer to nearest-point selection); prune low-confidence points
        self.radius_px = radius_px          # if set: neighbour radius = radius_px * depth / focal (pixel footprint at that depth)
        self.proj = nn.Linear(feat_in, dim)
        self.dir_freq = 2.0 ** torch.arange(n_freq_dir) * math.pi
        self.feat_net = MLP(dim + 3 + 1, hidden, dim)                       # F_hat = Feature Net(F_m, q - p_m, level)
        self.density_net = MLP(dim, hidden, 1, n=3)
        self.weight_net = MLP(dim, hidden // 2, 1, n=2)                     # learned per-neighbour attention logit
        self.color_net = MLP(dim + 3 * (1 + 2 * n_freq_dir), hidden, 3, n=3, act_out=nn.Sigmoid())
        self.Kn, self.R, self.C, self.S = n_neighbors, radius, max_per_cell, n_samples

    # ---- per (cloud, camera) preparation ------------------------------------------------
    @torch.no_grad()
    def prepare(self, cloud, K, c2w, h, w, frame_feats=None):
        uv, z = G.project(cloud["xyz"], K, c2w)
        u, v = uv[:, 0], uv[:, 1]
        ok = (z > 1e-4) & (u >= -0.5) & (u < w - 0.5) & (v >= -0.5) & (v < h - 0.5)
        if frame_feats is not None:
            ok = ok & ~frame_feats.pruned_mask(cloud["src"], cloud["lvl"], cloud["pix"])
        idx = torch.nonzero(ok)[:, 0]
        pix = (v[idx].round().long().clamp(0, h - 1) * w + u[idx].round().long().clamp(0, w - 1))
        order = torch.argsort(z[idx], stable=True)
        order = order[torch.argsort(pix[order], stable=True)]
        idx, pix = idx[order], pix[order]
        counts = torch.bincount(pix, minlength=h * w)
        starts = torch.cumsum(counts, 0) - counts
        return dict(idx=idx, z=z[idx], counts=counts, starts=starts, h=h, w=w, K=K, c2w=c2w, n=len(idx))

    def candidates(self, view, ray_pix):
        """ray_pix (R,) pixel ids -> candidate cloud indices (R, 9C) and validity."""
        h, w, C = view["h"], view["w"], self.C
        r, c = ray_pix // w, ray_pix % w
        dr = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=ray_pix.device)
        dc = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=ray_pix.device)
        rr, cc = r[:, None] + dr, c[:, None] + dc
        inb = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        cell = (rr.clamp(0, h - 1) * w + cc.clamp(0, w - 1))
        st, cnt = view["starts"][cell], view["counts"][cell].clamp(max=C)
        ar = torch.arange(C, device=ray_pix.device)
        pos = st[..., None] + ar                                               # (R,9,C) positions in sorted list
        valid = (ar < cnt[..., None]) & inb[..., None]
        pos = pos.clamp(max=max(view["n"] - 1, 0)).flatten(1)
        return view["idx"][pos], view["z"][pos], valid.flatten(1)

    # ---- rendering ------------------------------------------------------------------------
    def forward(self, cloud, frame_feats, view, ray_pix):
        """Render rays through pixels `ray_pix` of the camera in `view`. Returns rgb (R,3), depth (R,), acc (R,)."""
        dev = ray_pix.device
        R_, S, Kn = len(ray_pix), self.S, self.Kn
        cand, cz, cvalid = self.candidates(view, ray_pix)                     # (R,M)
        M = cand.shape[1]
        feats, conf_logit, resid = frame_feats.gather(cloud["src"][cand.flatten()], cloud["lvl"][cand.flatten()], cloud["pix"][cand.flatten()])
        f_m = (self.proj(feats) + resid).view(R_, M, -1)
        conf = torch.sigmoid(conf_logit).view(R_, M)
        p_m = cloud["xyz"][cand]                                              # (R,M,3)
        lvl_m = cloud["lvl"][cand].float()

        # shading samples: up to S depths taken from the sorted candidate depths
        zs = torch.where(cvalid, cz, torch.full_like(cz, float("inf")))
        zs, zorder = zs.sort(dim=1)
        n = cvalid.sum(1)                                                     # (R,)
        rank = (torch.linspace(0, 1, S, device=dev)[None] * (n[:, None] - 1).clamp(min=0)).floor().long()  # (R,S)
        svalid = (n[:, None] > 0) & torch.cat([torch.ones(R_, 1, dtype=torch.bool, device=dev), rank[:, 1:] > rank[:, :-1]], 1)
        z_s = torch.gather(zs, 1, rank)                                       # (R,S)
        z_s = torch.where(svalid, z_s, torch.zeros_like(z_s))

        # ray geometry (camera at view["c2w"], pixel centres)
        h, w = view["h"], view["w"]
        uv1 = torch.stack([(ray_pix % w).float(), (ray_pix // w).float(), torch.ones(R_, device=dev)], -1)
        d_cam = uv1 @ torch.linalg.inv(view["K"]).T                            # z = 1
        d_w = d_cam @ view["c2w"][:3, :3].T
        o_w = view["c2w"][:3, 3]
        q = o_w + z_s[..., None] * d_w[:, None]                               # (R,S,3)
        dirs = F.normalize(d_w, dim=-1)                                       # (R,3)

        # K nearest candidates within radius for every shading sample
        dist = (q[:, :, None] - p_m[:, None]).norm(dim=-1)                   # (R,S,M)
        dist = torch.where(cvalid[:, None], dist, torch.full_like(dist, float("inf")))
        dist, nn_idx = dist.topk(min(Kn, M), dim=2, largest=False)            # (R,S,K)
        if self.radius_px is not None:
            Rs = (self.radius_px * z_s / float(view["K"][0, 0])).clamp(min=1e-6)[..., None]   # (R,S,1)
        else:
            Rs = torch.full((R_, S, 1), self.R, device=dev)
        nvalid = (dist < Rs) & svalid[..., None]
        f_k = torch.gather(f_m, 1, nn_idx.flatten(1)[..., None].expand(-1, -1, f_m.shape[-1])).view(R_, S, -1, f_m.shape[-1])
        p_k = torch.gather(p_m, 1, nn_idx.flatten(1)[..., None].expand(-1, -1, 3)).view(R_, S, -1, 3)
        c_k = torch.gather(conf, 1, nn_idx.flatten(1)).view(R_, S, -1)
        l_k = torch.gather(lvl_m, 1, nn_idx.flatten(1)).view(R_, S, -1, 1)
        wgt = c_k / (dist + 1e-4 * Rs)
        if self.sharp != 1.0:
            wgt = wgt.clamp(min=1e-12) ** self.sharp
        if self.conf_min > 0:
            nvalid = nvalid & (c_k > self.conf_min)
        wgt = torch.where(nvalid, wgt, torch.zeros_like(wgt))
        wsum = wgt.sum(-1, keepdim=True)
        wgt = wgt / wsum.clamp(min=1e-8)

        rel = (q[:, :, None] - p_k) / Rs[..., None]
        f_hat = self.feat_net(torch.cat([f_k, rel, l_k], -1))                 # (R,S,K,dim)
        if self.attention:
            wgt = wgt * torch.exp(self.weight_net(f_hat)[..., 0].clamp(-6, 6))
            wgt = torch.where(nvalid, wgt, torch.zeros_like(wgt))
            wgt = wgt / wgt.sum(-1, keepdim=True).clamp(min=1e-8)
        sigma_k = F.softplus(self.density_net(f_hat)[..., 0] - 1.0)           # (R,S,K)
        sigma = (wgt * sigma_k).sum(-1) * (wsum[..., 0] > 0).float()          # (R,S)
        f_agg = (wgt[..., None] * f_hat).sum(2)                               # (R,S,dim)
        dpe = dirs[:, :, None] * self.dir_freq.to(dev)[None, None]
        dir_enc = torch.cat([dirs, torch.sin(dpe).flatten(1), torch.cos(dpe).flatten(1)], -1)
        rgb_s = self.color_net(torch.cat([f_agg, dir_enc[:, None].expand(-1, S, -1)], -1))   # (R,S,3)

        # volume rendering (Eq 9) along the sorted shading depths
        delta = torch.cat([z_s[:, 1:] - z_s[:, :-1], 2 * Rs[:, -1]], 1)
        delta = torch.where(svalid, delta.clamp(min=0), torch.zeros_like(delta))
        delta = torch.minimum(delta, 4 * Rs[..., 0])
        alpha = 1 - torch.exp(-sigma * delta)
        T = torch.cumprod(torch.cat([torch.ones(R_, 1, device=dev), 1 - alpha[:, :-1] + 1e-10], 1), 1)
        wts = T * alpha                                                       # (R,S)
        rgb = (wts[..., None] * rgb_s).sum(1)
        depth = (wts * z_s).sum(1)
        acc = wts.sum(1)
        return rgb, depth, acc


@torch.no_grad()
def render_image(model, cloud, frame_feats, K, c2w, h, w, chunk=4096, bg=0.0):  # bg must match training (no background term)
    view = model.prepare(cloud, K, c2w, h, w, frame_feats)
    pix = torch.arange(h * w, device=K.device)
    rgb, dep, acc = [], [], []
    for i in range(0, h * w, chunk):
        r, d, a = model(cloud, frame_feats, view, pix[i:i + chunk])
        rgb.append(r), dep.append(d), acc.append(a)
    rgb, dep, acc = torch.cat(rgb), torch.cat(dep), torch.cat(acc)
    rgb = rgb + (1 - acc[:, None]) * bg
    return rgb.view(h, w, 3), dep.view(h, w), acc.view(h, w)
