"""Scene I/O.

Scene directory layout (same as the authors' DAVIS template, plus optional test views):
    scene/
      image/image_%05d.png            RGB frames of the monocular video
      camera/camera_%05d.npz          {intrinsic: 3x3, extrinsic: 4x4 camera-to-world (OpenCV)}
      motion/mask_%05d.png            optional, 255 = dynamic object
      depth/depth_%05d.npy            optional ground-truth depth (toy data only)
      test/image_%05d_%02d.png        optional held-out views (time, camera id)
      test/camera_%05d_%02d.npz
"""
import glob
import os

import cv2
import numpy as np
import torch


def _imread(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, path
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


class Scene:
    def __init__(self, root, device="cpu", max_frames=None):
        self.root = root
        files = sorted(glob.glob(os.path.join(root, "image", "image_*.png")))
        assert files, f"no frames under {root}/image"
        if max_frames:
            files = files[:max_frames]
        self.n = len(files)
        self.images = torch.from_numpy(np.stack([_imread(f) for f in files])).float().div_(255).permute(0, 3, 1, 2)
        self.h, self.w = self.images.shape[-2:]
        Ks, Cs = [], []
        for i in range(self.n):
            c = np.load(os.path.join(root, "camera", f"camera_{i:05d}.npz"))
            Ks.append(c["intrinsic"][:3, :3])
            Cs.append(c["extrinsic"])
        self.K = torch.from_numpy(np.stack(Ks)).float()
        self.c2w = torch.from_numpy(np.stack(Cs)).float()
        self.motion = None
        if os.path.isdir(os.path.join(root, "motion")):
            ms = [cv2.imread(os.path.join(root, "motion", f"mask_{i:05d}.png"), 0) for i in range(self.n)]
            if all(m is not None for m in ms):
                self.motion = torch.from_numpy(np.stack(ms) > 127)
        self.gt_depth = None
        if os.path.isdir(os.path.join(root, "depth")):
            ds = [np.load(os.path.join(root, "depth", f"depth_{i:05d}.npy")) for i in range(self.n)]
            self.gt_depth = torch.from_numpy(np.stack(ds)).float()
        self.to(device)

    def to(self, device):
        for k in ("images", "K", "c2w", "motion", "gt_depth"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))
        self.device = device
        return self

    def test_views(self):
        """List of (time, cam_id, image(3,H,W), K, c2w) held-out views, or []."""
        out = []
        for f in sorted(glob.glob(os.path.join(self.root, "test", "image_*.png"))):
            stem = os.path.basename(f)[6:-4]
            t, cid = (int(x) for x in stem.split("_"))
            c = np.load(os.path.join(self.root, "test", f"camera_{t:05d}_{cid:02d}.npz"))
            im = torch.from_numpy(_imread(f)).float().div_(255).permute(2, 0, 1)
            out.append((t, cid, im.to(self.device), torch.from_numpy(c["intrinsic"][:3, :3]).float().to(self.device),
                        torch.from_numpy(c["extrinsic"]).float().to(self.device)))
        return out


# ----------------------------------------------------------------------------- writers / converters
def write_frame(root, i, image_rgb_uint8, K, c2w, motion=None, depth=None):
    os.makedirs(os.path.join(root, "image"), exist_ok=True)
    os.makedirs(os.path.join(root, "camera"), exist_ok=True)
    cv2.imwrite(os.path.join(root, "image", f"image_{i:05d}.png"), cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR))
    np.savez(os.path.join(root, "camera", f"camera_{i:05d}.npz"), intrinsic=np.asarray(K, np.float32),
             extrinsic=np.asarray(c2w, np.float32))
    if motion is not None:
        os.makedirs(os.path.join(root, "motion"), exist_ok=True)
        cv2.imwrite(os.path.join(root, "motion", f"mask_{i:05d}.png"), (motion > 0).astype(np.uint8) * 255)
    if depth is not None:
        os.makedirs(os.path.join(root, "depth"), exist_ok=True)
        np.save(os.path.join(root, "depth", f"depth_{i:05d}.npy"), depth.astype(np.float32))


def write_test_view(root, t, cid, image_rgb_uint8, K, c2w):
    os.makedirs(os.path.join(root, "test"), exist_ok=True)
    cv2.imwrite(os.path.join(root, "test", f"image_{t:05d}_{cid:02d}.png"), cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR))
    np.savez(os.path.join(root, "test", f"camera_{t:05d}_{cid:02d}.npz"), intrinsic=np.asarray(K, np.float32),
             extrinsic=np.asarray(c2w, np.float32))


def add_nvidia_test_views(nsff_dense_dir, scene_dir, skip_train_cam=True):
    """Held-out views for a converted Nvidia scene from NSFF's `mv_images/{t:05d}/cam{c:02d}.jpg`.

    The 12 rig cameras are static and training frame i is camera i at time i, so
    the test view (t, c) uses the camera of training frame c. Verifies the frame
    correspondence first (training frame t must equal mv_images/t/cam{t+1}).
    """
    sc = Scene(scene_dir)
    n_mv = len(glob.glob(os.path.join(nsff_dense_dir, "mv_images", "*")))
    n = min(sc.n, n_mv)
    psnrs = []

    def load(f):
        # NSFF mv_images are a vertical centre crop (1920x1024 of 1920x1080); keep the same
        # aspect so the GT is a centre crop of the 16:9 rendering.
        im = _imread(f)
        h = int(round(sc.w * im.shape[0] / im.shape[1]))
        return cv2.resize(im, (sc.w, h), interpolation=cv2.INTER_AREA)

    for t in range(n):
        for c in range(12):
            if skip_train_cam and c == t % 12:
                continue
            f = os.path.join(nsff_dense_dir, "mv_images", f"{t:05d}", f"cam{c + 1:02d}.jpg")
            if not os.path.exists(f):
                continue
            write_test_view(scene_dir, t, c, load(f), sc.K[c].numpy(), sc.c2w[c].numpy())
        f = os.path.join(nsff_dense_dir, "mv_images", f"{t:05d}", f"cam{t % 12 + 1:02d}.jpg")
        if os.path.exists(f):
            im = load(f) / 255.0
            tr = sc.images[t].permute(1, 2, 0).numpy()
            o = (tr.shape[0] - im.shape[0]) // 2
            tr = tr[o:o + im.shape[0]]
            psnrs.append(-10 * np.log10(((im - tr) ** 2).mean()))
    print(f"[nvidia-test] {n} time steps; training-frame vs mv_images/t/cam(t+1) PSNR: {np.round(psnrs, 1)} (should be > 30 dB)")
    return psnrs


def llff_to_opencv(pose_3x5):
    """LLFF [down, right, back | t | hwf] -> OpenCV c2w (4x4) and (h, w, f)."""
    m = np.asarray(pose_3x5, np.float64)
    h, w, f = m[:, 4]
    r = m[:, :4]
    c2w = np.eye(4)
    c2w[:3, 0] = r[:, 1]      # x = right
    c2w[:3, 1] = r[:, 0]      # y = down
    c2w[:3, 2] = -r[:, 2]     # z = forward
    c2w[:3, 3] = r[:, 3]
    return c2w, (h, w, f)


def convert_nvidia_scene(src, dst, factor=2, images_dir=None, mask_dir=None, mask_dynamic_is_white=True, max_width=960):
    """DynamicNeRF-style Nvidia Dynamic Scene folder -> our layout.

    Expects `src/poses_bounds.npy`, `src/images_{factor}/*.png` and optional
    `src/motion_masks/1/*.png`. Held-out multi-view frames (NSFF nvidia_data_full
    `mv_images/{t:05d}/cam{c:02d}.jpg`) are added by `add_nvidia_test_views`.
    """
    pb = np.load(os.path.join(src, "poses_bounds.npy"))
    poses = pb[:, :15].reshape(-1, 3, 5)
    imgs = sorted(glob.glob(os.path.join(src, images_dir, "*"))) if images_dir else []
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(src, f"images_{factor}", "*")))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(src, "images", "*")))
    assert len(imgs) == len(poses), (len(imgs), len(poses))
    # DynamicNeRF data.zip: motion_masks/1 = dynamic (foreground), motion_masks/0 = its complement
    masks = sorted(glob.glob(os.path.join(src, mask_dir, "*.png"))) if mask_dir else []
    masks = masks or sorted(glob.glob(os.path.join(src, "motion_masks", "1", "*.png"))) or \
        sorted(glob.glob(os.path.join(src, f"motion_masks_{factor}", "*.png")))
    for i, f in enumerate(imgs):
        im = _imread(f)
        if im.shape[1] > max_width:
            im = cv2.resize(im, (max_width, int(round(im.shape[0] * max_width / im.shape[1]))), interpolation=cv2.INTER_AREA)
        c2w, (h, w, fl) = llff_to_opencv(poses[i])
        sx, sy = im.shape[1] / w, im.shape[0] / h
        K = np.array([[fl * sx, 0, (im.shape[1] - 1) / 2], [0, fl * sy, (im.shape[0] - 1) / 2], [0, 0, 1]])
        mo = None
        if masks:
            mo = cv2.imread(masks[i], 0)
            if mo.shape != im.shape[:2]:
                mo = cv2.resize(mo, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            if not mask_dynamic_is_white:
                mo = 255 - mo
        write_frame(dst, i, im, K, c2w, motion=mo)
    return len(imgs)
