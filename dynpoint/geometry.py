"""Camera geometry, triangulation and the three masks of DynPoint Sec. 3.2.

Conventions: OpenCV pinhole (x right, y down, z forward), pixel centre of
pixel (i, j) is (u, v) = (j, i). `c2w` is a 4x4 camera-to-world matrix.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- basic ops
def pixel_grid(h, w, device=None):
    """(H,W,3) homogeneous pixel coordinates [u, v, 1]."""
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                          torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
    return torch.stack([u, v, torch.ones_like(u)], -1)


def backproject(depth, K, c2w):
    """depth (...,H,W) or (H,W); K (3,3); c2w (4,4) -> world points (...,H,W,3)."""
    h, w = depth.shape[-2:]
    g = pixel_grid(h, w, depth.device)                         # H,W,3
    cam = (g @ torch.linalg.inv(K).T) * depth[..., None]       # ...,H,W,3
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def project(pts, K, c2w):
    """world pts (...,3) -> pixel uv (...,2) and camera depth z (...)."""
    w2c = torch.linalg.inv(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[..., 2]
    uvw = cam @ K.T
    uv = uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-6)
    return uv, z


def flow_from_points(pts_world, K, c2w_target, h, w):
    """Optical flow (H,W,2) induced by moving each pixel's 3D point into target cam."""
    uv, _ = project(pts_world, K, c2w_target)
    return uv - pixel_grid(h, w, pts_world.device)[..., :2]


def warp_backward(img, flow):
    """Sample img (C,H,W) at pixel + flow (H,W,2). Returns (C,H,W)."""
    h, w = flow.shape[:2]
    g = pixel_grid(h, w, flow.device)[..., :2] + flow
    g = torch.stack([g[..., 0] / (w - 1) * 2 - 1, g[..., 1] / (h - 1) * 2 - 1], -1)
    return F.grid_sample(img[None], g[None], align_corners=True, padding_mode="border")[0]


def downsample_depth(depth, K, factor):
    """Area-downsample a depth map (H,W) and scale K accordingly (paper Sec 3.4)."""
    if factor == 1:
        return depth, K
    d = F.interpolate(depth[None, None], scale_factor=1.0 / factor, mode="area")[0, 0]
    K2 = K.clone()
    K2[0] = K[0] / factor
    K2[1] = K[1] / factor
    # pixel-centre convention: u_small = (u_big + 0.5)/f - 0.5
    K2[0, 2] = (K[0, 2] + 0.5) / factor - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) / factor - 0.5
    return d, K2


# ----------------------------------------------------------------------------- triangulation
def triangulate_midpoint(flow, K, c2w_a, c2w_b):
    """Mid-point triangulation of every pixel of frame a with its flow match in b.

    Returns depth in camera a (H,W) and a validity mask (positive depth in both
    cameras, non-degenerate rays).
    """
    h, w = flow.shape[:2]
    Kinv = torch.linalg.inv(K)
    ga = pixel_grid(h, w, flow.device)
    gb = ga.clone()
    gb[..., :2] += flow
    da = (ga @ Kinv.T) @ c2w_a[:3, :3].T                       # ray dirs (world)
    db = (gb @ Kinv.T) @ c2w_b[:3, :3].T
    oa, ob = c2w_a[:3, 3], c2w_b[:3, 3]
    # closest points between lines oa + s*da and ob + t*db (Eberly's formulation)
    w0 = oa - ob
    a = (da * da).sum(-1)
    b = (da * db).sum(-1)
    c = (db * db).sum(-1)
    d = (da * w0).sum(-1)
    e = (db * w0).sum(-1)
    den = a * c - b * b
    valid = den > 1e-8 * a * c
    den = torch.where(valid, den, torch.ones_like(den))
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    valid = valid & (s > 0) & (t > 0)
    # da = R_a K^-1 p has unit camera-z, so the camera-a depth of the mid-point is s.
    return s, valid


# ----------------------------------------------------------------------------- masks (Sec 3.2)
def correspondence_mask(flow_fw, flow_bw, eps=3.0, erode=8):
    """Eq (1): forward-backward consistency + out-of-bounds, then erosion. (H,W) bool."""
    h, w = flow_fw.shape[:2]
    bw_at_fw = warp_backward(flow_bw.permute(2, 0, 1), flow_fw).permute(1, 2, 0)
    err = (flow_fw + bw_at_fw).norm(dim=-1)
    tgt = pixel_grid(h, w, flow_fw.device)[..., :2] + flow_fw
    oob = (tgt[..., 0] < 0) | (tgt[..., 0] > w - 1) | (tgt[..., 1] < 0) | (tgt[..., 1] > h - 1)
    m = ((err <= eps) & ~oob).cpu().numpy().astype(np.uint8)
    if erode:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode)))
    return torch.from_numpy(m.astype(bool)).to(flow_fw.device)


def geometric_edge_mask(depth, dilate=8, lo=100, hi=150):
    """Canny on the normal map of `depth`; True = NOT an edge (reliable)."""
    d = depth.detach().cpu().numpy().astype(np.float64)[..., None]
    zx = cv2.Sobel(d, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(d, cv2.CV_64F, 0, 1, ksize=5)
    n = np.dstack((-zx, -zy, np.ones_like(d)))
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    n = ((n + 1) / 2 * 255)[:, :, ::-1].astype(np.uint8)
    edges = cv2.Canny(cv2.cvtColor(n, cv2.COLOR_BGR2GRAY), lo, hi)
    if dilate:
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))
    return torch.from_numpy(edges == 0).to(depth.device)


def erode_bool(mask, k=8):
    m = mask.cpu().numpy().astype(np.uint8)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return torch.from_numpy(m.astype(bool)).to(mask.device)


def sampson_dynamic_mask(flow, K, c2w_a, c2w_b, thresh=1.0, dilate=8):
    """Dynamic-object mask from the Sampson epipolar error of the optical flow.

    True = dynamic. Used when the dataset provides no motion masks.
    """
    h, w = flow.shape[:2]
    w2c_b = torch.linalg.inv(c2w_b)
    T = w2c_b @ c2w_a                                     # a -> b
    R, t = T[:3, :3], T[:3, 3]
    tx = torch.tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], device=flow.device, dtype=flow.dtype)
    Kinv = torch.linalg.inv(K)
    Fm = Kinv.T @ (tx @ R) @ Kinv
    pa = pixel_grid(h, w, flow.device)
    pb = pa.clone()
    pb[..., :2] += flow
    Fp = pa @ Fm.T
    Ftq = pb @ Fm
    num = (pb * Fp).sum(-1) ** 2
    den = Fp[..., 0] ** 2 + Fp[..., 1] ** 2 + Ftq[..., 0] ** 2 + Ftq[..., 1] ** 2 + 1e-8
    err = num / den
    # normalise by translation magnitude so the threshold is scene-agnostic
    err = err / (t.norm() ** 2 + 1e-8)
    m = (err > thresh).cpu().numpy().astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))
    return torch.from_numpy(m.astype(bool)).to(flow.device)


# ----------------------------------------------------------------------------- affine depth fit (Eq 2)
def fit_affine_depth(pred, target, mask, iters=10, weight=None, mode="depth"):
    """Robust (IRLS-L1) affine fit of the network prediction to triangulated depth over `mask`.

    mode="depth": target ~ alpha * pred + beta                 (networks that output depth, e.g. the authors' MiDaS ckpt)
    mode="disp" : 1/target ~ alpha * pred + beta               (affine-invariant disparity networks, e.g. Depth-Anything)
    mode="auto" : try both, keep the lower weighted L1 residual in depth space.
    `weight` multiplies residuals (e.g. pixel parallax). Returns (alpha, beta, mode_used).
    """
    w0 = torch.ones_like(pred) if weight is None else weight
    best = None
    for md in (("depth", "disp") if mode == "auto" else (mode,)):
        x, y, w0m = pred[mask].double(), target[mask].double(), w0[mask].double()
        if x.numel() < 50:
            continue
        if md == "disp":
            y = 1.0 / y.clamp(min=1e-6)
        wgt = w0m.clone()
        for _ in range(iters):
            A = torch.stack([x, torch.ones_like(x)], 1) * wgt[:, None].sqrt()
            sol = torch.linalg.lstsq(A, (y * wgt.sqrt())[:, None]).solution[:, 0]
            a, b = sol[0], sol[1]
            res = (y - (a * x + b)).abs()
            wgt = w0m / res.clamp(min=1e-3 if md == "depth" else 1e-5)
        d_fit = apply_affine(x, float(a), float(b), md)
        l1 = ((d_fit - target[mask].double()).abs() * w0m).sum() / w0m.sum()
        if best is None or l1 < best[0]:
            best = (l1, float(a), float(b), md)
    if best is None:
        return 1.0, 0.0, mode if mode != "auto" else "depth"
    return best[1], best[2], best[3]


def apply_affine(pred, alpha, beta, mode):
    if mode == "disp" or mode is True:
        return 1.0 / (alpha * pred + beta).clamp(min=1e-6)
    return alpha * pred + beta
