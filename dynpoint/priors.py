"""Pre-trained priors: optical flow (RAFT), monocular depth (MiDaS / Depth-Anything), 2D features."""
import os

import torch
import torch.nn.functional as F
import torchvision


class FlowNet:
    """torchvision RAFT-large; weights download automatically on first use."""

    def __init__(self, device):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a, b):
        """a, b: (B,3,H,W) in [0,1] -> flow a->b (B,2,H,W) in pixels."""
        h, w = a.shape[-2:]
        s = max(1.0, 128 / h, 128 / w)                       # RAFT needs >= 128 px on each side
        hs, ws = int(round(h * s / 8) * 8), int(round(w * s / 8) * 8)
        a = F.interpolate(a * 2 - 1, size=(hs, ws), mode="bilinear", align_corners=False)
        b = F.interpolate(b * 2 - 1, size=(hs, ws), mode="bilinear", align_corners=False)
        flow = self.model(a.to(self.device), b.to(self.device))[-1]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / ws
        flow[:, 1] *= h / hs
        return flow


class DepthNet:
    """backend 'midas': authors' MiDaS checkpoint (depth). backend 'dav2': Depth-Anything-V2 (relative inverse depth)."""

    def __init__(self, device, backend="midas", ckpt=None):
        self.device, self.backend = device, backend
        if backend == "midas":
            from .midas import MidasNet
            self.model = MidasNet(ckpt).to(device).eval()
        elif backend == "dav2":
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            name = "depth-anything/Depth-Anything-V2-Large-hf"
            self.proc = AutoImageProcessor.from_pretrained(name)
            self.model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
        else:
            raise ValueError(backend)

    @torch.no_grad()
    def __call__(self, img):
        """img (3,H,W) in [0,1] -> raw prediction (H,W). Affine fit (and inverse check) happens downstream."""
        if self.backend == "midas":
            return self.model(img[None].to(self.device))[0, 0]
        inp = self.proc(images=(img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"), return_tensors="pt").to(self.device)
        out = self.model(**inp).predicted_depth[None]
        return F.interpolate(out, size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0].clamp(min=0)


class FeatureNet(torch.nn.Module):
    """Frozen ImageNet ResNet-18 stem (64ch @1/4) + layer2 (128ch @1/8), upsampled to full res,
    concatenated with RGB -> trainable 1x1 projection to `dim` channels (point feature F in Eq 7)."""

    def __init__(self, dim=32):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1).eval()
        self.layer2 = r.layer2.eval()
        for p in list(self.stem.parameters()) + list(self.layer2.parameters()):
            p.requires_grad_(False)
        self.proj = torch.nn.Conv2d(64 + 128 + 3, dim, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def backbone(self, img):
        """img (B,3,H,W) in [0,1] -> frozen features (B,195,H,W)."""
        h, w = img.shape[-2:]
        x = (img - self.mean) / self.std
        f1 = self.stem(x)
        f2 = self.layer2(f1)
        f1 = F.interpolate(f1, size=(h, w), mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=False)
        return torch.cat([img, f1, f2], 1)

    def forward(self, feats):
        return self.proj(feats)
