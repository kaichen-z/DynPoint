"""CLI:  python -m dynpoint <toy|preprocess|train_flow|train_render|eval|video|all> --scene DIR [--work DIR] ...

`all` runs preprocess -> train_flow -> train_render -> eval -> video.
"""
import argparse
import os
import sys
import time

import torch


def get_log(work):
    os.makedirs(work, exist_ok=True)
    fh = open(os.path.join(work, "log.txt"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["toy", "preprocess", "train_flow", "train_render", "eval", "video", "all"])
    p.add_argument("--scene", required=True)
    p.add_argument("--work", default=None, help="output dir (default: <scene>/work)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    # stage 1
    p.add_argument("--K", type=int, default=4, help="temporal neighbours on each side (paper K=6 for Nvidia, 4 for DAVIS)")
    p.add_argument("--depth", default="midas", choices=["midas", "dav2", "gt"])
    p.add_argument("--depth_ckpt", default=None, help="authors' midas_cpkt.pt")
    p.add_argument("--trust_depth", action="store_true", help="skip the affine fit: depth prior is already consistent with the poses (wild scenes from Depth-Anything-3)")
    p.add_argument("--dyn", default="auto", choices=["auto", "sampson"], help="auto = dataset masks if present")
    p.add_argument("--sf_iters", type=int, default=3000)
    p.add_argument("--sf_lr", type=float, default=1e-3)
    p.add_argument("--refine_depth", default=None, choices=[None, "dyn", "all"], help="jointly optimise a per-frame log-depth correction (dynamic pixels or all)")
    p.add_argument("--sf_wsfmin", type=float, default=0.0, help="scene-flow magnitude prior (px units) during joint depth refinement")
    p.add_argument("--w_lpips", type=float, default=0.0, help="perceptual (LPIPS-vgg) loss weight on 32x32 patches")
    p.add_argument("--sf_wdepth", type=float, default=1.0, help="weight of the depth-agreement term in Eq 4")
    # stage 2
    p.add_argument("--nr_iters", type=int, default=4000)
    p.add_argument("--rays", type=int, default=2048)
    p.add_argument("--k_dyn", type=int, default=None, help="keep dynamic points only from references within this time gap")
    p.add_argument("--eval_loo", action="store_true", help="evaluate/render without the target frame's own points")
    p.add_argument("--attn", action="store_true", help="learned per-neighbour attention weights in the renderer")
    p.add_argument("--p_loo", type=float, default=0.0, help="prob. of leave-one-out training (target frame points excluded)")
    p.add_argument("--radius", type=float, default=0.03, help="neighbour radius in units of scene scale")
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--radius_px", type=float, default=None, help="depth-relative neighbour radius in pixels (overrides --radius)")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--max_eval_views", type=int, default=None)
    p.add_argument("--toy_frames", type=int, default=20)
    p.add_argument("--color_tol", type=float, default=None, help="photometric consistency culling of reference points (mean abs RGB error, e.g. 0.15)")
    p.add_argument("--dyn_band", type=int, default=0, help="drop reference points within this many px of the dynamic-mask boundary")
    p.add_argument("--mv_thresh", type=float, default=None, help="multi-view consistency culling: max re-projection error (px) of a reference point vs the optical flow")
    p.add_argument("--prune_every", type=int, default=0, help="prune low-confidence points every N render-training steps (0 = off)")
    p.add_argument("--prune_conf", type=float, default=0.1)
    p.add_argument("--edge_thresh", type=float, default=None, help="drop points at depth discontinuities (relative depth gradient per pixel, e.g. 0.03)")
    p.add_argument("--sharp", type=float, default=1.0, help="neighbour-weight exponent (>1 sharper selection)")
    p.add_argument("--conf_min", type=float, default=0.0, help="prune neighbours with confidence below this")
    p.add_argument("--cam_path", default="interp", choices=["interp", "orbit"], help="demo-video camera path")
    p.add_argument("--video_times", default=None, help="comma-separated frames to synthesise in the demo video (each swept once); default: all frames")
    p.add_argument("--sweep", type=int, default=24, help="camera-sweep steps per synthesised frame when --video_times is set")
    p.add_argument("--span", type=int, default=2, help="interp path: how many training cameras to slide across")
    p.add_argument("--orbit", type=float, default=0.06, help="demo-video camera orbit radius as a fraction of the scene scale")
    a = p.parse_args(argv)
    work = a.work or os.path.join(a.scene, "work")
    log = get_log(work)
    log(f"[cli] {' '.join(sys.argv[1:])}")
    torch.manual_seed(0)

    if a.cmd == "toy":
        from .toy import make
        make(a.scene, n=a.toy_frames)
        return

    from .data import Scene
    scene = Scene(a.scene, a.device, max_frames=a.max_frames)
    pri_path = os.path.join(work, "priors.pt")

    if a.cmd in ("preprocess", "all"):
        from .preprocess import run
        run(a.scene, work, K=a.K, depth_backend=a.depth, depth_ckpt=a.depth_ckpt, device=a.device, dyn_mode=a.dyn,
            max_frames=a.max_frames, log=log, trust_depth=a.trust_depth)
    priors = torch.load(pri_path, map_location="cpu", weights_only=False)

    if a.cmd in ("train_flow", "all"):
        from . import sceneflow
        net = sceneflow.train(scene, priors, work, iters=a.sf_iters, lr=a.sf_lr, device=a.device, log=log, w_depth=a.sf_wdepth, refine_depth=a.refine_depth, w_sfmin=a.sf_wsfmin)
        if a.refine_depth:
            priors = torch.load(pri_path, map_location="cpu", weights_only=False)
        e0, e1, n = sceneflow.evaluate(scene, priors, net, a.device)
        log(f"[sf] dynamic-pixel flow error (px): depth-only warp {e0:.3f} -> depth+scene-flow {e1:.3f}  ({n} px)")

    if a.cmd in ("train_render", "eval", "video", "all"):
        from .train_render import Stage2
        s2 = Stage2(scene, priors, work, device=a.device, K_nb=a.K, radius=a.radius, n_neighbors=a.neighbors,
                    n_samples=a.samples, log=log, attention=a.attn, k_dyn=a.k_dyn, radius_px=a.radius_px,
                    edge_thresh=a.edge_thresh, sharp=a.sharp, conf_min=a.conf_min, mv_thresh=a.mv_thresh, dyn_band=a.dyn_band, color_tol=a.color_tol)
        if a.cmd in ("train_render", "all"):
            s2.train(iters=a.nr_iters, rays=a.rays, p_loo=a.p_loo, w_lpips=a.w_lpips, prune_every=a.prune_every, prune_conf=a.prune_conf)
        else:
            s2.load()
        s2.eval_loo = a.eval_loo
        if a.cmd in ("eval", "all"):
            s2.evaluate(os.path.join(work, "eval"), max_views=a.max_eval_views)
        if a.cmd in ("video", "all"):
            s2.video(os.path.join(work, "demo.mp4"), orbit=a.orbit, cam_path=a.cam_path, span=a.span,
                     times=[int(x) for x in a.video_times.split(",")] if a.video_times else None, sweep=a.sweep)


if __name__ == "__main__":
    main()
