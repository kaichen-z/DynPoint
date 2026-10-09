import numpy as np
import torch
from skimage.metrics import structural_similarity


class Metrics:
    def __init__(self, device):
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, gt):
        """pred, gt: (3,H,W) in [0,1] -> dict psnr/ssim/lpips."""
        pred, gt = pred.clamp(0, 1).to(self.device), gt.clamp(0, 1).to(self.device)
        mse = ((pred - gt) ** 2).mean()
        psnr = float(-10 * torch.log10(mse))
        p, g = pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy()
        ssim = float(structural_similarity(p, g, channel_axis=2, data_range=1.0))
        lp = float(self.lpips(pred[None] * 2 - 1, gt[None] * 2 - 1))
        return dict(psnr=psnr, ssim=ssim, lpips=lp)
