"""MiDaS v2.1 (ResNeXt-101 backbone) as used by the DynPoint authors.

Mirrors upstream `core_depth/MiDaS.py` so the released `midas_cpkt.pt` loads
unchanged, but builds the backbone from torchvision instead of torch.hub
(no network access needed). Output is metric-like *depth* (10000 / disparity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(x))
        out = self.conv2(self.relu(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        out = xs[0]
        if len(xs) == 2:
            out = out + self.resConfUnit1(xs[1])
        out = self.resConfUnit2(out)
        return F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor, self.mode = scale_factor, mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class MidasNet(nn.Module):
    def __init__(self, ckpt=None, features=256, resize=(256, 512)):
        super().__init__()
        r = torchvision.models.resnext101_32x8d(weights=None)
        self.pretrained = nn.Module()
        self.pretrained.layer1 = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.pretrained.layer2, self.pretrained.layer3, self.pretrained.layer4 = r.layer2, r.layer3, r.layer4
        self.scratch = nn.Module()
        for i, c in enumerate([256, 512, 1024, 2048]):
            setattr(self.scratch, f"layer{i + 1}_rn", nn.Conv2d(c, features, 3, 1, 1, bias=False))
        for i in range(1, 5):
            setattr(self.scratch, f"refinenet{i}", FeatureFusionBlock(features))
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, 3, 1, 1), Interpolate(2, "bilinear"),
            nn.Conv2d(128, 32, 3, 1, 1), nn.ReLU(True), nn.Conv2d(32, 1, 1), nn.ReLU(True))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize
        if ckpt is not None:
            sd = torch.load(ckpt, map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            missing = [k for k in missing if k not in ("mean", "std")]
            assert not missing, f"MiDaS checkpoint missing keys: {missing[:5]}"

    @torch.no_grad()
    def forward(self, x):
        """x: (B,3,H,W) RGB in [0,1] -> depth (B,1,H,W)."""
        h, w = x.shape[-2:]
        x = (x - self.mean) / self.std
        if self.resize is not None:
            x = F.interpolate(x, size=self.resize, mode="bicubic", align_corners=True)
        l1 = self.pretrained.layer1(x)
        l2 = self.pretrained.layer2(l1)
        l3 = self.pretrained.layer3(l2)
        l4 = self.pretrained.layer4(l3)
        p4 = self.scratch.refinenet4(self.scratch.layer4_rn(l4))
        p3 = self.scratch.refinenet3(p4, self.scratch.layer3_rn(l3))
        p2 = self.scratch.refinenet2(p3, self.scratch.layer2_rn(l2))
        p1 = self.scratch.refinenet1(p2, self.scratch.layer1_rn(l1))
        out = 10000.0 / torch.clamp(self.scratch.output_conv(p1), min=1e-2)
        if self.resize is not None:
            out = F.interpolate(out, size=(h, w), mode="bicubic", align_corners=True)
        return out
