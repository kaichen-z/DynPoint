"""Check the LLFF->OpenCV camera conversion of a DynamicNeRF Nvidia scene against its COLMAP model,
and compare the stage-1 depth with COLMAP sparse depth.
Run: python tests/check_nvidia.py data/nvidia_tmp/data/Balloon1 data/nv_Balloon1
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from dynpoint.data import Scene


def read_images_bin(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cam = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            pts = np.frombuffer(f.read(24 * n2), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            out[name.decode()] = (np.array(q), np.array(t), cam, pts)
    return out


def read_points_bin(path):
    xyz, ids = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            p = struct.unpack("<3d", f.read(24))
            f.read(3)
            f.read(8)
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tl)
            xyz.append(p)
            ids.append(pid)
    return np.array(xyz), np.array(ids)


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


src, dst = sys.argv[1], sys.argv[2]
sp = os.path.join(src, "sparse/0") if os.path.isdir(os.path.join(src, "sparse/0")) else os.path.join(src, "sparse")
imgs = read_images_bin(os.path.join(sp, "images.bin"))
xyz, pids = read_points_bin(os.path.join(sp, "points3D.bin"))
sc = Scene(dst, "cpu")
name0 = sorted(imgs)[0]
q, t, _, pts2d = imgs[name0]
w2c = np.eye(4)
w2c[:3, :3], w2c[:3, 3] = qvec2rot(q), t
c2w_colmap = np.linalg.inv(w2c)
c2w_mine = sc.c2w[0].numpy()
print("[cam] colmap image", name0)
print("[cam] rotation agreement (should be ~identity):\n", np.round(c2w_mine[:3, :3].T @ c2w_colmap[:3, :3], 3))
print("[cam] translation mine", np.round(c2w_mine[:3, 3], 3), "colmap", np.round(c2w_colmap[:3, 3], 3))
# sparse depth in frame 0 vs stage-1 depth
cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
z = cam[:, 2]
K = sc.K[0].numpy()
uv = (cam @ K.T)
uv = uv[:, :2] / uv[:, 2:3]
inb = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < sc.w) & (uv[:, 1] >= 0) & (uv[:, 1] < sc.h)
print(f"[pts] {len(xyz)} colmap points, {inb.mean():.3f} project inside frame 0 with z>0; z range {np.percentile(z[inb], [5, 50, 95]).round(2)}")
pri = torch.load(os.path.join(dst, "work", "priors.pt"), weights_only=False)
d = pri["depth"][0].numpy()
raw = pri["depth_raw"][0].numpy()
u, v = uv[inb, 0].astype(int), uv[inb, 1].astype(int)
zz = z[inb]
print(f"[depth] stage-1 depth vs colmap sparse: median ratio {np.median(d[v, u] / zz):.3f}, median |rel err| {np.median(np.abs(d[v, u] - zz) / zz):.3f}")
for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 1e9)):
    m = (zz >= lo) & (zz < hi)
    if m.sum() > 20:
        print(f"[depth]   z in [{lo},{hi}): n={int(m.sum())} median ratio {np.median(d[v[m], u[m]] / zz[m]):.3f}  |rel err| {np.median(np.abs(d[v[m], u[m]] - zz[m]) / zz[m]):.3f}")
print(f"[depth] raw midas vs colmap: corr(raw, z) {np.corrcoef(raw[v, u], zz)[0, 1]:.3f}, corr(1/raw, z) {np.corrcoef(1 / raw[v, u], zz)[0, 1]:.3f}")
a = np.polyfit(raw[v, u], zz, 1)
print(f"[depth] least-squares depth = {a[0]:.3f} * raw + {a[1]:.3f}; stage-1 affine {pri['affine'][0].numpy().round(3)}")
for g in (1, 3, 6):
    if (0, g) in pri["flows"]:
        fl = pri["flows"][(0, g)].float().norm(dim=0)
        st = ~pri["dyn"][0]
        print(f"[flow] gap {g}: median |flow| static {fl[st].median():.2f} px, dynamic {fl[~st].median():.2f} px")
