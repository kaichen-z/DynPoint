"""Export one frame of a scene for the interactive web viewer: coloured 3D point cloud (consistent depth),
scene-flow vectors of the dynamic points (t -> t+gap) and the camera frustum.

python -m dynpoint.export_viewer --scene DIR [--work DIR] --t 8 --gap 3 --out viewer/scene.bin
Binary layout (little endian): header JSON (uint32 length + utf8) then float32/uint8 arrays as described in the header.
"""
import argparse
import json
import os
import struct

import cv2
import numpy as np
import torch

from . import geometry as G
from .data import Scene
from .sceneflow import FlowModel, SceneFlowNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--work", default=None)
    p.add_argument("--t", type=int, default=None)
    p.add_argument("--gap", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--max_static", type=int, default=150000)
    p.add_argument("--max_flow", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    work = a.work or os.path.join(a.scene, "work")
    dev = a.device
    sc = Scene(a.scene, dev)
    pri = torch.load(os.path.join(work, "priors.pt"), map_location="cpu", weights_only=False)
    net = SceneFlowNet().to(dev)
    net.load_state_dict(torch.load(os.path.join(work, "sceneflow.pt"), map_location=dev))
    fm = FlowModel(net.eval(), pri, sc, dev)
    t = a.t if a.t is not None else sc.n // 2
    s = min(t + a.gap, sc.n - 1)
    with torch.no_grad():
        p_t, dyn = fm.points(t, 1)                                         # (3,H,W) world
        d = fm.depth_t(t)
        disp = fm.displace(p_t, t, s - t, dyn) - p_t                         # scene flow t -> s (world units)
        # drop points at depth discontinuities for a cleaner cloud
        gx = torch.zeros_like(d); gy = torch.zeros_like(d)
        gx[:, 1:-1] = (d[:, 2:] - d[:, :-2]).abs() / 2
        gy[1:-1] = (d[2:] - d[:-2]).abs() / 2
        ok = ((torch.maximum(gx, gy) / d.clamp(min=1e-6)) < 0.08) & (d > 1e-3)
        xyz = fm.norm(p_t.permute(1, 2, 0))                                  # normalised world coords
        rgb = sc.images[t].permute(1, 2, 0)
        # colour by depth (turbo) for the depth view
        dn = d[ok]
        lo, hi = dn.quantile(0.02), dn.quantile(0.98)
        dcol = cv2.applyColorMap((255 - ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        dcol = torch.from_numpy(dcol.copy()).to(dev)
        P = xyz[ok]; C = (rgb[ok] * 255).byte(); D = dcol[ok]; isdyn = dyn[ok]
        g = torch.Generator(device="cpu").manual_seed(0)
        if len(P) > a.max_static:
            idx = torch.randperm(len(P), generator=g)[: a.max_static].to(dev)
            P, C, D, isdyn = P[idx], C[idx], D[idx], isdyn[idx]
        # flow vectors on dynamic points (subsampled)
        fm_ok = ok & dyn
        Pf = xyz[fm_ok]; Vf = (disp.permute(1, 2, 0)[fm_ok]) / fm.scale
        if len(Pf) > a.max_flow:
            idx = torch.randperm(len(Pf), generator=g)[: a.max_flow].to(dev)
            Pf, Vf = Pf[idx], Vf[idx]
        # dynamic points of the reference frame s at their own time (shows where the object actually is at t+gap)
        p_s, dyn_s = fm.points(s, 1)
        d_s = fm.depth_t(s)
        ok_s = dyn_s & (d_s > 1e-3)
        P2 = fm.norm(p_s.permute(1, 2, 0))[ok_s]; C2 = (sc.images[s].permute(1, 2, 0)[ok_s] * 255).byte()
        if len(P2) > 60000:
            idx = torch.randperm(len(P2), generator=g)[:60000].to(dev)
            P2, C2 = P2[idx], C2[idx]
        # camera frustum (normalised coords)
        K, c2w = sc.K[t], sc.c2w[t]
        corners = torch.tensor([[0, 0, 1], [sc.w, 0, 1], [sc.w, sc.h, 1], [0, sc.h, 1]], dtype=torch.float32, device=dev)
        rays = (corners @ torch.linalg.inv(K).T) * 0.15 * fm.scale
        cam = torch.cat([c2w[:3, 3][None], rays @ c2w[:3, :3].T + c2w[:3, 3]])
        cam = fm.norm(cam)
    header = dict(scene=os.path.basename(a.scene.rstrip("/")), t=t, ref=s, n_points=int(len(P)), n_flow=int(len(Pf)), n_ref=int(len(P2)),
                  scale=float(fm.scale), flow_scale_hint=float(Vf.norm(dim=-1).median()) if len(Vf) else 0.0,
                  arrays=["xyz f32 (n_points,3)", "rgb u8 (n_points,3)", "depth_rgb u8 (n_points,3)", "dyn u8 (n_points,)",
                          "flow_xyz f32 (n_flow,3)", "flow_vec f32 (n_flow,3)", "cam f32 (5,3)", "ref_xyz f32 (n_ref,3)", "ref_rgb u8 (n_ref,3)"])
    hb = json.dumps(header).encode()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<I", len(hb))); f.write(hb)
        for arr, dt in ((P, np.float32), (C, np.uint8), (D, np.uint8), (isdyn, np.uint8), (Pf, np.float32), (Vf, np.float32), (cam, np.float32), (P2, np.float32), (C2, np.uint8)):
            f.write(np.ascontiguousarray(arr.cpu().numpy().astype(dt)).tobytes())
    print(f"[viewer] {a.out}: {len(P)} points ({int(isdyn.sum())} dynamic), {len(Pf)} flow vectors, frame {t} -> {s}, median |flow| {header['flow_scale_hint']:.4f}")


if __name__ == "__main__":
    main()
