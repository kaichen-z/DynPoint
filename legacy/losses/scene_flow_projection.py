import torch
from torch import nn
import torch.nn.functional as F

class flow_by_depth_fn(nn.Module):
    def __init__(self, B, H, W, device):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
        self.coord = torch.ones([1, H, W, 1, 3])
        self.coord[0, ..., 0, 0] = xx
        self.coord[0, ..., 0, 1] = yy
        self.coord = self.coord.to(device)
        self.coord = self.coord.expand([B, H, W, 1, 3])
    def backward_warp(self, depth_2, flow_1_2):
        # flow[...,0]: dh
        # flow[...,0]: dw
        B, _, H, W = depth_2.shape
        coord = self.coord[..., :2].view(1, H, W, 2).expand([B, H, W, 2])
        sample_grids = coord + flow_1_2
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        return F.grid_sample(depth_2, sample_grids, align_corners=True, padding_mode='border')
    def forward(self, depth_1, depth_2, flow_1_2, R_1, R_2, R_1_T, R_2_T, t_1, t_2, K, K_inv):
        B, H, W = depth_1.shape
        coord = self.coord.clone()
        depth_1 = depth_1.view([B, H, W, 1, 1])
        depth_2 = depth_2.view([B, H, W, 1, 1])
        p1_camera_1 = depth_1 * torch.matmul(self.coord, K_inv)  
        p2_camera_2 = depth_2 * torch.matmul(self.coord, K_inv)
        global_p1 = torch.matmul(p1_camera_1, R_1) + t_1  # BHW13; Global Point 1
        global_p2 = torch.matmul(p2_camera_2, R_2) + t_2  # BHW13; Global Point 2
        global_p2 = global_p2.squeeze(3).permute([0, 3, 1, 2])  # B3HW
        return {'global_p1': global_p1, 'global_p2': global_p2}

class scene_flow_projection_fn(nn.Module):
    # tested
    def __init__(self, B, H, W, device):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
        self.coord = torch.ones([1, H, W, 1, 3])
        self.coord[0, ..., 0, 0] = xx
        self.coord[0, ..., 0, 1] = yy
        self.coord = self.coord.to(device)
        self.coord = self.coord.expand([B, H, W, 1, 3])

    def backward_warp(self, depth_2, flow_1_2):
        B, _, H, W = depth_2.shape
        coord = self.coord[..., :2].view(1, H, W, 2).expand([B, H, W, 2])
        sample_grids = coord + flow_1_2
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        return F.grid_sample(depth_2, sample_grids, align_corners=True, padding_mode='border')

    def forward(self, depth_1, depth_2, flow_1_2, R_1, R_2, R_1_T, R_2_T, t_1, t_2, K, K_inv, sflow_1_2):
        B, H, W = depth_1.shape
        coord = self.coord.clone()
        depth_1 = depth_1.view([B, H, W, 1, 1])
        depth_2 = depth_2.view([B, H, W, 1, 1])
        p1_camera_1 = depth_1 * torch.matmul(self.coord, K_inv)
        p2_camera_2 = depth_2 * torch.matmul(self.coord, K_inv)
        global_p1 = torch.matmul(p1_camera_1, R_1) + t_1
        # Point based on the optical flow.
        p2_camera_2_w = p2_camera_2.squeeze(3).permute([0, 3, 1, 2])  # B3HW
        warped_p2_camera_2 = self.backward_warp(p2_camera_2_w, flow_1_2)
        warped_p2_camera_2 = warped_p2_camera_2.permute([0, 2, 3, 1])[..., None, :]  # BHW13
        # Point based on the scene flow.
        p1_camera_2 = torch.matmul(global_p1 + sflow_1_2 - t_2, R_2_T) # HomoGh With Scene Flow
        p1_image_2 = torch.matmul(p1_camera_2, K) 
        coord_image_2 = (p1_image_2 / (p1_image_2[..., -1:] + 1e-8))[..., : -1]
        # Computing the optic flow based on scene flow and depth.
        depth_1 = depth_1.view(B, 1, H, W)
        depth_2 = depth_2.view(B, 1, H, W)
        depth_flow_1_2 = (coord_image_2 - coord[..., :-1])[..., 0, :] 
        return {'dflow_1_2': depth_flow_1_2, 'p1_camera_2': p1_camera_2, \
                'depth_1': depth_1, 'depth_2': depth_2, \
                'warped_p2_camera_2': warped_p2_camera_2}

# class unproject_ptcld()
class unproject_ptcld(nn.Module):
    # tested
    def __init__(self, is_one_way=True):
        super().__init__()
        self.coord = None

    def forward(self, depth_1, R_1, t_1, K_inv):
        B, _, H, W = depth_1.shape
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([1, H, W, 1, 3])
            self.coord[0, ..., 0, 0] = xx
            self.coord[0, ..., 0, 1] = yy
            self.coord = self.coord.to(depth_1.device)

        depth_1 = depth_1.view([B, H, W, 1, 1])
        p1_camera_1 = depth_1 * torch.matmul(self.coord, K_inv)
        global_p1 = torch.matmul(p1_camera_1, R_1) + t_1

        return global_p1

