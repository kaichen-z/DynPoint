"""Synthetic toy scene (Law-4): textured floor + back wall + a sphere flying across, hand-held camera.

Rendered by analytic ray casting so every pixel has exact depth, the camera is
exact and the motion mask is exact. Writes the standard scene layout plus
`test/` views from a second camera for each frame and `depth/` ground truth.
"""
import os

import numpy as np

from .data import write_frame, write_test_view


def _tex(p, scale, seed):
    """Smooth pseudo-random colour texture of 3D points p (...,3)."""
    rng = np.random.RandomState(seed)
    col = np.zeros(p.shape[:-1] + (3,))
    for k in range(6):
        f = rng.randn(3) * scale * (k + 1)
        ph = rng.rand(3) * 6.28
        for c in range(3):
            col[..., c] += np.sin((p * f).sum(-1) + ph[c]) / (k + 1)
    return (col - col.min()) / (col.max() - col.min() + 1e-6) * 0.8 + 0.1


def _render(K, c2w, h, w, sphere_c, sphere_r=0.35):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d = d_cam @ c2w[:3, :3].T
    o = c2w[:3, 3]
    depth = np.full((h, w), np.inf)
    col = np.zeros((h, w, 3))
    hit = np.zeros((h, w), bool)
    # floor y = 1.0 (y down), wall z = 6.0
    for plane_n, plane_d, seed, scale in (([0, 1, 0], 1.0, 1, 3.0), ([0, 0, 1], 6.0, 2, 2.0)):
        n = np.array(plane_n)
        denom = d @ n
        s = (plane_d - o @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        p = o + s[..., None] * d
        ok = (s > 0) & (s * d_cam[..., 2] < depth)
        z = s * d_cam[..., 2] / 1.0
        z = s  # d_cam has unit z so camera depth = s
        depth = np.where(ok, z, depth)
        c = _tex(p, scale, seed)
        col = np.where(ok[..., None], c, col)
        hit |= ok
    # sphere
    oc = o - sphere_c
    b = (d * oc).sum(-1)
    c_ = oc @ oc - sphere_r ** 2
    disc = b * b - (d * d).sum(-1) * c_
    s = (-b - np.sqrt(np.maximum(disc, 0))) / (d * d).sum(-1)
    ok = (disc > 0) & (s > 0) & (s < depth)
    p = o + s[..., None] * d
    sph = _tex((p - sphere_c) * 3, 4.0, 3) * np.array([1.0, 0.7, 0.5])
    col = np.where(ok[..., None], sph, col)
    depth = np.where(ok, s, depth)
    motion = ok
    depth[~hit & ~ok] = 0
    return col, depth, motion


def make(root, n=20, h=144, w=256, seed=0):
    rng = np.random.RandomState(seed)
    f = 0.9 * w
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    for t in range(n):
        a = t / (n - 1)
        # hand-held camera: slow lateral + slight rotation
        cx = -1.0 + 2.0 * a + 0.05 * np.sin(6 * a)
        cy = -0.2 + 0.1 * np.sin(4 * a)
        yaw = 0.15 * np.sin(2 * np.pi * a) - 0.1 * (a - 0.5)
        c2w = np.eye(4)
        c2w[:3, :3] = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        c2w[:3, 3] = [cx, cy, 0.0]
        sphere_c = np.array([1.2 - 2.4 * a, 0.2 - 0.6 * np.sin(np.pi * a), 3.0 + 0.5 * a])
        col, dep, mot = _render(K, c2w, h, w, sphere_c)
        write_frame(root, t, (col * 255).astype(np.uint8), K, c2w, motion=mot, depth=dep)
        # held-out camera: shifted sideways/up, same time
        c2w2 = c2w.copy()
        c2w2[:3, 3] += c2w[:3, :3] @ np.array([0.25, -0.1, 0.0])
        col2, _, _ = _render(K, c2w2, h, w, sphere_c)
        write_test_view(root, t, 0, (col2 * 255).astype(np.uint8), K, c2w2)
    print(f"[toy] wrote {n} frames {w}x{h} + {n} test views to {root}")
