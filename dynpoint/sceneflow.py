"""Stage 1b (Sec 3.3 + Appendix Fig 7): surface scene-flow MLP and its training.

The MLP is written with 1x1 convolutions so a whole frame of surface points
(3,H,W) is processed in one call. It predicts forward (t -> t+1) and backward
(t -> t-1) displacements in *normalised* world units; longer gaps are obtained
by chaining (Eq 6).
"""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as G


class PE(nn.Module):
    def __init__(self, n_freq, in_dim):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n_freq) * math.pi)
        self.out_dim = in_dim * (1 + 2 * n_freq)

    def forward(self, x):                       # x (B,C,H,W) -> (B, C*(1+2F), H, W)
        xs = x[:, :, None] * self.freq[None, None, :, None, None]
        return torch.cat([x, torch.sin(xs).flatten(1, 2), torch.cos(xs).flatten(1, 2)], 1)


class SceneFlowNet(nn.Module):
    def __init__(self, n_freq_xyz=10, n_freq_t=6, width=256, n_layers=4, out_scale=0.01):
        super().__init__()
        self.pe_x, self.pe_t = PE(n_freq_xyz, 3), PE(n_freq_t, 1)
        layers, c = [], self.pe_x.out_dim + self.pe_t.out_dim
        for _ in range(n_layers + 1):
            layers += [nn.Conv2d(c, width, 1), nn.LeakyReLU(0.2, inplace=True)]
            c = width
        self.body = nn.Sequential(*layers)
        self.head_f, self.head_b = nn.Conv2d(width, 3, 1), nn.Conv2d(width, 3, 1)
        self.out_scale = out_scale
        nn.init.zeros_(self.head_f.weight), nn.init.zeros_(self.head_f.bias)
        nn.init.zeros_(self.head_b.weight), nn.init.zeros_(self.head_b.bias)

    def forward(self, x, t):
        """x (B,3,H,W) normalised points, t (B,) in [0,1] -> (fwd, bwd) each (B,3,H,W)."""
        tm = t.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[-2:])
        h = self.body(torch.cat([self.pe_x(x), self.pe_t(tm)], 1))
        return self.head_f(h) * self.out_scale, self.head_b(h) * self.out_scale


def chain(net, x, t_idx, steps, n_frames):
    """Eq 6: displacement of points x (B,3,H,W) at frame t_idx to frame t_idx+steps (signed)."""
    total = torch.zeros_like(x)
    t = torch.full((x.shape[0],), t_idx / n_frames, device=x.device)
    dt = 1.0 / n_frames
    for _ in range(abs(steps)):
        f, b = net(x + total, t)
        d = f if steps > 0 else b
        total = total + d
        t = t + (dt if steps > 0 else -dt)
    return total


class FlowModel:
    """Convenience wrapper: world <-> normalised, frame points, chained displacement."""

    def __init__(self, net, priors, scene, device):
        self.net, self.P, self.scene, self.device = net, priors, scene, device
        self.center, self.scale, self.n = priors["center"].to(device), priors["scale"], priors["n"]
        self.depth = priors["depth"].to(device)
        self.dyn = priors["dyn"].to(device)
        self.dlog = None                     # optional (N,H,W) log-depth correction
        self.dlog_all = False                # False: dynamic pixels only; True: every pixel

    def depth_t(self, t):
        if self.dlog is None:
            return self.depth[t]
        m = 1.0 if self.dlog_all else self.dyn[t].float()
        return self.depth[t] * torch.exp(self.dlog[t] * m)

    def norm(self, p):
        return (p - self.center) / self.scale

    def denorm(self, p):
        return p * self.scale + self.center

    def points(self, t, level=1):
        """World points of frame t at pyramid level (1,2,4): (3,h,w), plus dyn mask (h,w)."""
        d, K = G.downsample_depth(self.depth_t(t), self.scene.K[t], level)
        p = G.backproject(d, K, self.scene.c2w[t]).permute(2, 0, 1)
        dyn = self.dyn[t]
        if level > 1:
            dyn = F.interpolate(dyn[None, None].float(), size=d.shape, mode="area")[0, 0] > 0.5
        return p, dyn

    def displace(self, p_world, t, steps, dyn):
        """World points of frame t -> world points at frame t+steps (static points stay put)."""
        if steps == 0:
            return p_world
        x = self.norm(p_world.permute(1, 2, 0)).permute(2, 0, 1)[None]
        d = chain(self.net, x, t, steps, self.n)[0] * self.scale
        return p_world + d * dyn[None].float()


def train(scene, priors, work_dir, iters=3000, lr=1e-3, device="cuda", log=print, w_depth=1.0, w_cycle=1.0,
          w_static=0.1, refine_depth=False, w_dreg=1.0, w_sfmin=0.0):
    """Optimise the scene-flow MLP with Eq 4 (flow re-projection + depth agreement) and Eq 5 (cycle)."""
    t0 = time.time()
    net = SceneFlowNet().to(device)
    fm = FlowModel(net, priors, scene, device)
    N, H, W, nbrs = priors["n"], priors["h"], priors["w"], priors["nbrs"]
    flows = {k: v.to(device).float() for k, v in priors["flows"].items()}
    corr = {k: v.to(device) for k, v in priors["corr"].items()}
    edge = priors["edge"].to(device)
    dyn_frames = [t for t in range(N) if fm.dyn[t].sum() > 50 and nbrs[t]]
    # per-step output scale from the observed one-frame dynamic parallax (normalised units),
    # so the zero-initialised heads only have to produce O(1) values for typical motions
    res = []
    with torch.no_grad():
        for t in dyn_frames[:: max(1, len(dyn_frames) // 8)]:
            for s in (t + 1, t - 1):
                if (t, s) not in flows:
                    continue
                p_t, dyn = fm.points(t)
                uv0, _ = G.project(p_t.permute(1, 2, 0), scene.K[s], scene.c2w[s])
                r_px = (uv0 - G.pixel_grid(H, W, device)[..., :2] - flows[(t, s)].permute(1, 2, 0)).norm(dim=-1)
                m = dyn & corr[(t, s)]
                if m.sum() > 50:
                    f_px = float(scene.K[t][0, 0])
                    res.append((r_px * fm.depth[t] / f_px / fm.scale)[m].median().item())
    if res:
        net.out_scale = float(min(max(sorted(res)[len(res) // 2], 0.005), 0.5))
    log(f"[sf] {len(dyn_frames)} frames with dynamic pixels; MLP step scale {net.out_scale:.4f} (normalised units)")
    if not dyn_frames:
        log("[sf] no dynamic pixels in any frame -> scene flow fixed to zero")
        torch.save(net.state_dict(), os.path.join(work_dir, "sceneflow.pt"))
        return net
    params = [{"params": net.parameters(), "lr": lr}]
    if refine_depth:
        fm.dlog = torch.zeros(N, H, W, device=device, requires_grad=True)
        fm.dlog_all = refine_depth == "all"
        params.append({"params": [fm.dlog], "lr": 1e-2})
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    g = torch.Generator(device="cpu").manual_seed(0)

    def pair_loss(t, s):
        p_t, dyn = fm.points(t)
        mask = dyn & corr[(t, s)] & edge[t]
        if mask.sum() < 20 and refine_depth != "all":
            return None, None
        if mask.sum() < 20:
            mask = corr[(t, s)] & edge[t]
        disp = chain(net, fm.norm(p_t.permute(1, 2, 0)).permute(2, 0, 1)[None], t, s - t, N)[0] * fm.scale
        p_s = (p_t + disp).permute(1, 2, 0)                                 # (H,W,3) world
        uv, z = G.project(p_s, scene.K[s], scene.c2w[s])
        flow_pred = uv - G.pixel_grid(H, W, device)[..., :2]
        l_flow = (flow_pred - flows[(t, s)].permute(1, 2, 0)).abs().sum(-1)[mask].mean()
        # depth agreement: z of the moved point vs D_hat_s sampled at the flow target
        d_s = G.warp_backward(fm.depth_t(s)[None], flows[(t, s)].permute(1, 2, 0))[0]
        l_depth = ((z - d_s).abs() / fm.scale)[mask].mean()
        # static pixels should not move
        static = (~dyn) & edge[t]
        l_static = (disp.norm(dim=0) / fm.scale)[static].mean() if static.any() else disp.sum() * 0
        loss = l_flow + w_depth * l_depth + w_static * l_static
        if w_sfmin > 0:                      # prefer small scene flow: lets depth refinement absorb baseline-aligned "motion"
            px_per_unit = float(scene.K[t][0, 0]) / fm.depth_t(t)[mask].mean().detach()
            loss = loss + w_sfmin * (disp.norm(dim=0)[mask] * px_per_unit).mean()
        if refine_depth == "all":            # dense depth refinement: static re-projection must match the flow
            uv_s, _ = G.project(p_t.permute(1, 2, 0), scene.K[s], scene.c2w[s])
            ms = (~dyn) & corr[(t, s)]
            loss = loss + (uv_s - G.pixel_grid(H, W, device)[..., :2] - flows[(t, s)].permute(1, 2, 0)).abs().sum(-1)[ms].mean()
        return loss, (l_flow.item(), l_depth.item())

    def cycle_loss(t):
        p_t, dyn = fm.points(t)
        if dyn.sum() == 0:
            return p_t.sum() * 0
        x = fm.norm(p_t.permute(1, 2, 0)).permute(2, 0, 1)[None]
        tt = torch.tensor([t / N], device=device)
        f, b = net(x, tt)
        # forward then backward from t+1
        f2, b2 = net(x + f, tt + 1.0 / N)
        lc = (f + b2).norm(dim=1)[0][dyn].mean()
        if t > 0:
            f3, b3 = net(x + b, tt - 1.0 / N)
            lc = lc + (b + f3).norm(dim=1)[0][dyn].mean()
        return lc

    K = priors["K"]
    for it in range(1, iters + 1):
        t = dyn_frames[torch.randint(len(dyn_frames), (1,), generator=g).item()]
        # curriculum: short gaps first, the full 2K neighbourhood after the first third of training
        max_gap = min(K, 1 + int(3 * K * (it - 1) / iters))
        cands = [s for s in nbrs[t] if abs(s - t) <= max_gap]
        s = cands[torch.randint(len(cands), (1,), generator=g).item()]
        loss, parts = pair_loss(t, s)
        if loss is None:
            continue
        lc = cycle_loss(t)
        loss = loss + w_cycle * lc
        if refine_depth:
            dl = fm.dlog[t]
            tv = (dl[:, 1:] - dl[:, :-1]).abs().mean() + (dl[1:] - dl[:-1]).abs().mean()
            loss = loss + w_dreg * (tv * 10 + dl.abs().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 10) == 0 or it == 1:
            log(f"[sf] it {it}/{iters} loss {loss.item():.4f} flow_px {parts[0]:.3f} depth {parts[1]:.4f} cycle {lc.item():.5f} ({time.time() - t0:.0f}s)")
    torch.save(net.state_dict(), os.path.join(work_dir, "sceneflow.pt"))
    if refine_depth:
        with torch.no_grad():
            ref = torch.stack([fm.depth_t(t) for t in range(N)]).cpu()
            dyn = priors["dyn"]
            if refine_depth == "all":
                dyn = torch.ones_like(dyn)
            log(f"[sf] refined depth ({refine_depth}): median scale change {float(torch.exp(fm.dlog.detach().cpu()[dyn]).median()):.3f}, "
                f"range {float(torch.exp(fm.dlog.detach().cpu()[dyn]).quantile(0.05)):.3f}..{float(torch.exp(fm.dlog.detach().cpu()[dyn]).quantile(0.95)):.3f}")
            priors["depth"] = ref
            torch.save(priors, os.path.join(work_dir, "priors.pt"))
            fm.depth, fm.dlog = ref.to(device), None
    return net


@torch.no_grad()
def evaluate(scene, priors, net, device="cuda"):
    """Mean |predicted flow - RAFT flow| (px) on dynamic pixels, with and without scene flow."""
    fm = FlowModel(net, priors, scene, device)
    N, H, W = priors["n"], priors["h"], priors["w"]
    e_sf, e_depth, n = 0.0, 0.0, 0
    for t in range(N):
        for s in priors["nbrs"][t]:
            p_t, dyn = fm.points(t)
            mask = dyn & priors["corr"][(t, s)].to(device) & priors["edge"][t].to(device)
            if mask.sum() < 20:
                continue
            fl = priors["flows"][(t, s)].to(device).float().permute(1, 2, 0)
            grid = G.pixel_grid(H, W, device)[..., :2]
            uv0, _ = G.project(p_t.permute(1, 2, 0), scene.K[s], scene.c2w[s])
            uv1, _ = G.project(fm.displace(p_t, t, s - t, dyn).permute(1, 2, 0), scene.K[s], scene.c2w[s])
            e_depth += (uv0 - grid - fl).norm(dim=-1)[mask].sum().item()
            e_sf += (uv1 - grid - fl).norm(dim=-1)[mask].sum().item()
            n += mask.sum().item()
    return (e_depth / max(n, 1), e_sf / max(n, 1), n)
