"""Stage-1 visualisation (paper Fig. 5 / Fig. 10): warp a reference frame to a target frame using
(a) consistent depth only and (b) depth + scene flow, and show the re-projection error against the
RAFT flow as a red error map. Also writes the aggregated coloured point cloud as PLY.

python -m dynpoint.visualize_flow --scene DIR [--work DIR] --t 8 --gap 3 --out DIR/work/flow_vis
"""
import argparse
import os

import cv2
import numpy as np
import torch

from . import geometry as G
from .data import Scene
from .sceneflow import FlowModel, SceneFlowNet


def to_u8(x):
    return (x.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def error_map(err_px, vmax=8.0):
    e = (err_px / vmax).clamp(0, 1).cpu().numpy()
    col = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_HOT)[..., ::-1]
    return col


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--work", default=None)
    p.add_argument("--t", type=int, default=None, help="target frame (default: middle)")
    p.add_argument("--gap", type=int, default=3, help="reference = t + gap")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    work = a.work or os.path.join(a.scene, "work")
    out = a.out or os.path.join(work, "flow_vis")
    os.makedirs(out, exist_ok=True)
    dev = a.device
    sc = Scene(a.scene, dev)
    pri = torch.load(os.path.join(work, "priors.pt"), map_location="cpu", weights_only=False)
    net = SceneFlowNet().to(dev)
    net.load_state_dict(torch.load(os.path.join(work, "sceneflow.pt"), map_location=dev))
    fm = FlowModel(net.eval(), pri, sc, dev)
    t = a.t if a.t is not None else sc.n // 2
    s = t + a.gap
    assert 0 <= s < sc.n and (t, s) in pri["flows"], f"pair ({t},{s}) not in the 2K neighbourhood"
    H, W = sc.h, sc.w
    with torch.no_grad():
        p_t, dyn = fm.points(t)                                            # world points of target frame
        flow = pri["flows"][(t, s)].to(dev).float().permute(1, 2, 0)        # RAFT flow t->s
        grid = G.pixel_grid(H, W, dev)[..., :2]
        # (a) depth only
        uv0, _ = G.project(p_t.permute(1, 2, 0), sc.K[s], sc.c2w[s])
        # (b) depth + scene flow
        p_s = fm.displace(p_t, t, s - t, dyn)
        uv1, _ = G.project(p_s.permute(1, 2, 0), sc.K[s], sc.c2w[s])
        warp0 = G.warp_backward(sc.images[s], uv0 - grid).permute(1, 2, 0)   # reference sampled at predicted positions
        warp1 = G.warp_backward(sc.images[s], uv1 - grid).permute(1, 2, 0)
        err0 = (uv0 - grid - flow).norm(dim=-1)
        err1 = (uv1 - grid - flow).norm(dim=-1)
        valid = pri["corr"][(t, s)].to(dev)
        tgt = sc.images[t].permute(1, 2, 0)
        ref = sc.images[s].permute(1, 2, 0)
        m_dyn = dyn & valid
        print(f"target {t}, reference {s}: dynamic-pixel flow error depth-only {err0[m_dyn].mean():.2f} px -> depth+scene flow {err1[m_dyn].mean():.2f} px; "
              f"static {err0[~dyn & valid].mean():.2f} -> {err1[~dyn & valid].mean():.2f} px")
        # figure: row 1 target | reference | dynamic mask ; row 2 warp depth-only | error ; row 3 warp depth+flow | error
        def strip(ims):
            return np.concatenate([np.asarray(i) for i in ims], 1)
        mask_img = np.repeat((dyn.float().cpu().numpy()[..., None] * 255).astype(np.uint8), 3, 2)
        row1 = strip([to_u8(tgt), to_u8(ref), mask_img])
        row2 = strip([to_u8(warp0), error_map(err0 * valid), to_u8((warp0 - tgt).abs().clamp(0, 1))])
        row3 = strip([to_u8(warp1), error_map(err1 * valid), to_u8((warp1 - tgt).abs().clamp(0, 1))])
        fig = np.concatenate([row1, row2, row3], 0)
        cv2.imwrite(os.path.join(out, f"sceneflow_t{t:02d}_ref{s:02d}.png"), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
        for name, im in (("target", tgt), ("reference", ref), ("warp_depth_only", warp0), ("warp_depth_sceneflow", warp1)):
            cv2.imwrite(os.path.join(out, f"{name}.png"), cv2.cvtColor(to_u8(im), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, "error_depth_only.png"), cv2.cvtColor(error_map(err0 * valid), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, "error_depth_sceneflow.png"), cv2.cvtColor(error_map(err1 * valid), cv2.COLOR_RGB2BGR))
        # point cloud: target frame + references warped to t (coloured), as PLY
        pts, cols = [p_t.flatten(1).T], [tgt.reshape(-1, 3)]
        for r in pri["nbrs"][t]:
            p_r, dyn_r = fm.points(r)
            pts.append(fm.displace(p_r, r, t - r, dyn_r).flatten(1).T)
            cols.append(sc.images[r].permute(1, 2, 0).reshape(-1, 3))
        P = torch.cat(pts).cpu().numpy()
        C = (torch.cat(cols).cpu().numpy() * 255).astype(np.uint8)
        keep = np.random.RandomState(0).rand(len(P)) < min(1.0, 400000 / len(P))
        with open(os.path.join(out, f"pointcloud_t{t:02d}.ply"), "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {int(keep.sum())}\nproperty float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for q, c in zip(P[keep], C[keep]):
                f.write(f"{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
