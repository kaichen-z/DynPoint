"""Stage 2 training (photometric L2 on sampled rays of the training views) plus evaluation / video export."""
import json
import os
import time

import cv2
import numpy as np
import torch

from . import geometry as G
from .metrics import Metrics
from .priors import FeatureNet
from .render import FrameFeatures, PointRenderer, build_cloud, render_image
from .sceneflow import FlowModel, SceneFlowNet


class Stage2:
    def __init__(self, scene, priors, work_dir, device="cuda", K_nb=4, radius=0.03, n_neighbors=8, n_samples=24,
                 max_per_cell=24, log=print, attention=False, k_dyn=None, radius_px=None, edge_thresh=None,
                 sharp=1.0, conf_min=0.0, mv_thresh=None, dyn_band=0, color_tol=None):
        self.scene, self.P, self.work, self.device, self.log = scene, priors, work_dir, device, log
        net = SceneFlowNet().to(device)
        net.load_state_dict(torch.load(os.path.join(work_dir, "sceneflow.pt"), map_location=device))
        self.fm = FlowModel(net.eval(), priors, scene, device)
        self.K_nb, self.k_dyn, self.edge_thresh, self.mv_thresh, self.dyn_band, self.color_tol = K_nb, k_dyn, edge_thresh, mv_thresh, dyn_band, color_tol
        self.eval_loo = False               # render novel views without the target frame's own points
        self.featnet = FeatureNet().to(device)
        self.frames = FrameFeatures(self.featnet, scene, device).to(device)
        self.model = PointRenderer(self.frames.C, radius=radius * priors["scale"], n_neighbors=n_neighbors,
                                   n_samples=n_samples, max_per_cell=max_per_cell, attention=attention,
                                   radius_px=radius_px, sharp=sharp, conf_min=conf_min).to(device)
        self.clouds = {}

    def cloud(self, t, loo=False):
        key = (t, loo)
        if key not in self.clouds:
            self.clouds[key] = build_cloud(self.fm, t, self.K_nb, exclude_self=loo, k_dyn=self.k_dyn, edge_thresh=self.edge_thresh,
                                           mv_thresh=self.mv_thresh, dyn_band=self.dyn_band, color_tol=self.color_tol)
        return self.clouds[key]

    def train(self, iters=4000, rays=2048, lr=1e-3, lr_conf=1e-2, lr_resid=5e-3, p_loo=0.0, w_lpips=0.0, patch=32,
              prune_every=0, prune_conf=0.1, prune_start=4000):
        t0 = time.time()
        params = [{"params": self.model.parameters(), "lr": lr}, {"params": self.frames.conf.parameters(), "lr": lr_conf},
                  {"params": self.frames.resid.parameters(), "lr": lr_resid}]
        opt = torch.optim.Adam(params)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
        N, H, W = self.scene.n, self.scene.h, self.scene.w
        g = torch.Generator(device="cpu").manual_seed(0)
        views = {}
        lp = None
        if w_lpips > 0:
            import lpips
            lp = lpips.LPIPS(net="vgg", verbose=False).to(self.device).eval()
            for q in lp.parameters():
                q.requires_grad_(False)
        for it in range(1, iters + 1):
            t = torch.randint(N, (1,), generator=g).item()
            loo = bool(torch.rand(1, generator=g).item() < p_loo)
            cloud = self.cloud(t, loo)
            if prune_every and it > prune_start and it % prune_every == 0:
                n_new, n_tot, n_all = self.frames.prune(prune_conf)
                views = {}
                self.log(f"[nr] prune @ {it}: +{n_new} points removed, {n_tot}/{n_all} total ({100 * n_tot / n_all:.1f}%)")
            if (t, loo) not in views:
                views[(t, loo)] = self.model.prepare(cloud, self.scene.K[t], self.scene.c2w[t], H, W, self.frames)
            if lp is not None:                                   # square patches so a perceptual loss can be applied
                n_p = max(1, rays // (patch * patch))
                y0 = torch.randint(H - patch, (n_p,), generator=g)
                x0 = torch.randint(W - patch, (n_p,), generator=g)
                yy, xx = torch.meshgrid(torch.arange(patch), torch.arange(patch), indexing="ij")
                pix = ((y0[:, None, None] + yy) * W + (x0[:, None, None] + xx)).flatten().to(self.device)
            else:
                pix = torch.randint(H * W, (rays,), generator=g).to(self.device)
            rgb, _, acc = self.model(cloud, self.frames, views[(t, loo)], pix)
            gt = self.scene.images[t].flatten(1).T[pix]
            loss = ((rgb - gt) ** 2).mean()
            if lp is not None:
                pr = rgb.view(n_p, patch, patch, 3).permute(0, 3, 1, 2)
                gp = gt.view(n_p, patch, patch, 3).permute(0, 3, 1, 2)
                loss = loss + w_lpips * lp(pr * 2 - 1, gp * 2 - 1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if it % max(1, iters // 10) == 0 or it == 1:
                psnr = -10 * torch.log10(loss.detach())
                self.log(f"[nr] it {it}/{iters} loss {loss.item():.5f} psnr {psnr:.2f} acc {acc.mean():.3f} clouds {len(self.clouds)} ({time.time() - t0:.0f}s)")
        self.save()

    def save(self):
        torch.save({"model": self.model.state_dict(), "conf": self.frames.conf.state_dict(), "resid": self.frames.resid.state_dict(),
                    "pruned": [p.cpu() for p in self.frames.pruned]}, os.path.join(self.work, "render.pt"))

    def load(self):
        ck = torch.load(os.path.join(self.work, "render.pt"), map_location=self.device)
        self.model.load_state_dict(ck["model"], strict=False)
        self.frames.conf.load_state_dict(ck["conf"])
        if "resid" in ck:
            self.frames.resid.load_state_dict(ck["resid"])
        if "pruned" in ck:
            self.frames.pruned = [p.to(self.device) for p in ck["pruned"]]

    @torch.no_grad()
    def render(self, t, K, c2w):
        return render_image(self.model, self.cloud(t, self.eval_loo), self.frames, K, c2w, self.scene.h, self.scene.w)

    # ---- evaluation ------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, out_dir, max_views=None):
        tests = self.scene.test_views()
        os.makedirs(out_dir, exist_ok=True)
        if not tests:
            self.log("[eval] no held-out views in scene/test -> reporting training-view re-rendering only")
            tests = [(t, -1, self.scene.images[t], self.scene.K[t], self.scene.c2w[t]) for t in range(self.scene.n)]
        if max_views:
            tests = tests[:max_views]
        met = Metrics(self.device)
        rows = []
        for t, cid, gt, K, c2w in tests:
            rgb, dep, acc = self.render(t, K, c2w)
            if gt.shape[-2:] != rgb.shape[:2]:          # GT may be a centre crop of the render (Nvidia mv_images)
                oy, ox = (rgb.shape[0] - gt.shape[-2]) // 2, (rgb.shape[1] - gt.shape[-1]) // 2
                rgb = rgb[oy:oy + gt.shape[-2], ox:ox + gt.shape[-1]]
                dep, acc = dep[oy:oy + gt.shape[-2], ox:ox + gt.shape[-1]], acc[oy:oy + gt.shape[-2], ox:ox + gt.shape[-1]]
            m = met(rgb.permute(2, 0, 1), gt)
            rows.append(dict(t=t, cam=cid, **m))
            save_png(os.path.join(out_dir, f"pred_{t:05d}_{cid:02d}.png"), rgb)
            save_png(os.path.join(out_dir, f"gt_{t:05d}_{cid:02d}.png"), gt.permute(1, 2, 0))
            save_png(os.path.join(out_dir, f"depth_{t:05d}_{cid:02d}.png"), colorize_depth(dep, acc))
        avg = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
        json.dump(dict(mean=avg, views=rows), open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
        self.log(f"[eval] {len(rows)} views: PSNR {avg['psnr']:.2f}  SSIM {avg['ssim']:.3f}  LPIPS {avg['lpips']:.3f}")
        return avg

    # ---- demo video ----------------------------------------------------------------------
    @torch.no_grad()
    def video(self, path, orbit=0.06, fps=10, n_rounds=1.0, cam_path="interp", span=2, times=None, sweep=24):
        """cam_path="interp": at time t the camera slides between the training cameras of frames t-span..t+span
        (stays on the observed trajectory, like the benchmark views); "orbit": circle around camera t.
        times: list of frames to synthesise (each gets a full `sweep`-step camera sweep); None = whole video."""
        if cam_path == "interp":
            return self.video_interp(path, fps=fps, span=span, times=times, sweep=sweep)
        return self.video_orbit(path, orbit=orbit, fps=fps, n_rounds=n_rounds)

    @torch.no_grad()
    def video_interp(self, path, fps=10, span=2, steps_per_frame=3, times=None, sweep=24):
        N, H, W = self.scene.n, self.scene.h, self.scene.w
        frames = []
        if times is not None:                                              # a few frozen moments, each swept once
            plan = [(t, np.sin(2 * np.pi * k / sweep) * span) for t in times for k in range(sweep)]
        else:
            total = N * steps_per_frame
            plan = [(i // steps_per_frame, np.sin(2 * np.pi * i / total * 3) * span) for i in range(total)]
        for t, a in plan:
            j = int(np.clip(t + np.floor(a), 0, N - 1)); k = int(np.clip(j + 1, 0, N - 1))
            w = float(a - np.floor(a)) if k != j else 0.0
            # translation interpolated between cameras j and k, rotation of the nearer one
            c2w = self.scene.c2w[j].clone() if w < 0.5 else self.scene.c2w[k].clone()
            c2w[:3, 3] = (1 - w) * self.scene.c2w[j][:3, 3] + w * self.scene.c2w[k][:3, 3]
            rgb, dep, acc = self.render(t, self.scene.K[t], c2w)
            inp = self.scene.images[t].permute(1, 2, 0)
            frames.append(np.concatenate([to_u8(inp), to_u8(rgb), to_u8(colorize_depth(dep, acc))], 1))
        write_mp4(path, frames, fps * (steps_per_frame if times is None else 2))
        self.log(f"[video] wrote {path} ({len(frames)} frames, interpolated camera path, span {span}, times {times})")

    @torch.no_grad()
    def video_orbit(self, path, orbit=0.06, fps=10, n_rounds=1.0):
        """Novel-view video: at each time t the camera circles around the training camera of frame t
        (radius = `orbit` x scene scale), looking at the scene centre. Also writes a side-by-side with the input."""
        N, H, W = self.scene.n, self.scene.h, self.scene.w
        frames = []
        for t in range(N):
            ang = 2 * np.pi * n_rounds * t / max(N - 1, 1)
            c2w = self.scene.c2w[t].clone()
            off = c2w[:3, :3] @ torch.tensor([np.cos(ang), np.sin(ang), 0.0], device=self.device, dtype=torch.float32) * orbit * self.P["scale"]
            c2w[:3, 3] += off
            rgb, dep, acc = self.render(t, self.scene.K[t], c2w)
            inp = self.scene.images[t].permute(1, 2, 0)
            frames.append(np.concatenate([to_u8(inp), to_u8(rgb), to_u8(colorize_depth(dep, acc))], 1))
        write_mp4(path, frames, fps)
        self.log(f"[video] wrote {path} ({len(frames)} frames)")


# ----------------------------------------------------------------------------- helpers
def to_u8(x):
    return (x.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_png(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(to_u8(rgb), cv2.COLOR_RGB2BGR))


def colorize_depth(dep, acc):
    d = dep.clone()
    m = acc > 0.5
    if m.any():
        lo, hi = d[m].quantile(0.02), d[m].quantile(0.98)
        d = ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    u8 = (d.cpu().numpy() * 255).astype(np.uint8)
    col = cv2.applyColorMap(255 - u8, cv2.COLORMAP_TURBO)[..., ::-1].copy()
    col[~m.cpu().numpy()] = 0
    return torch.from_numpy(col).float() / 255


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
