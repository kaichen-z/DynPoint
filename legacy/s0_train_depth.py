import sys
sys.path.append('core_flow')
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from core_flow.things_eval import get_cfg as get_things_cfg
from core_flow.FlowFormer import build_flowformer
from core_depth.MiDaS import MidasNet

def visualize_depth(depth_np, add, index):
    depth1_vis = np.clip(depth_np, a_min=0, a_max=100)
    vmax = np.percentile(depth1_vis, 100)
    plt.imsave('%s/%s_depth.png'%(add, index), depth1_vis, cmap='turbo', vmax=15)
    plt.close('all')

def visualize_error(error, add, index):
    plt.imsave('%s/%s_error.png'%(add, index), error, cmap='Reds', vmax=0.8)
    plt.close('all')

def backward_flow_warp(im2, flow_1_2):
    # im2: H, W, C
    # flow_1_2: H, W, 2
    H, W, _ = im2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, H, W, 2])
    coord[0, ..., 0] = ww
    coord[0, ..., 1] = hh
    sample_grids = coord + flow_1_2[None, ...]
    sample_grids[..., 0] /= (W - 1) / 2
    sample_grids[..., 1] /= (H - 1) / 2
    sample_grids -= 1
    im = torch.from_numpy(im2).float().permute(2, 0, 1)[None, ...]
    out = torch.nn.functional.grid_sample(im, sample_grids, align_corners=True)
    o = out[0, ...].permute(1, 2, 0).numpy()
    return o

def get_oob_mask(flow_1_2):
    H, W, _ = flow_1_2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([H, W, 2])
    coord[..., 0] = ww
    coord[..., 1] = hh
    target_range = coord + flow_1_2
    m1 = (target_range[..., 0] < 0) + (target_range[..., 0] > W - 1)
    m2 = (target_range[..., 1] < 0) + (target_range[..., 1] > H - 1)
    return (m1 + m2).float().numpy()

def get_pixelgrid(h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, w).expand(1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, h, 1).expand(1, h, w)    
    ones = torch.ones_like(grid_h)
    pixelgrid = torch.cat((grid_h, grid_v, ones), dim=0)
    return pixelgrid

def compute_epipolar_distance(T_21, K, p_1, p_2):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]
    #E_mat = np.dot(skew(t_21), R_21)
    E_mat = skew(t_21) * R_21
    # compute bearing vector
    inv_K = np.linalg.inv(K)
    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)
    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :]**2 + l_2[1, :]**2) + 1e-8
    geometric_e_distance = algebric_e_distance/n_term
    geometric_e_distance = np.abs(geometric_e_distance)
    return geometric_e_distance

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def correspondence_mask(ref_flow, tar_flow):
    # former_1_2 [2, H, W] tensor
    # former_2_1 [2, H, W] tensor
    warp_flow_1_2 = backward_flow_warp(ref_flow.permute(1,2,0).numpy(), tar_flow.permute(1,2,0).numpy())  # using latter to sample former
    err_1 = np.linalg.norm(warp_flow_1_2 + tar_flow.permute(1,2,0).numpy(), axis=-1)
    mask_1 = np.where(err_1 > 3., 1, 0) # Backward and Forward Mask.
    oob_mask_1 = get_oob_mask(tar_flow.permute(1,2,0).numpy()) # Boundary Mask.
    mask_1 = np.clip(mask_1 + oob_mask_1, a_min=0, a_max=1)
    mask_1 = (1 - mask_1).astype(np.uint8)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    eroded_mask = cv2.erode(mask_1, structuring_element, iterations=1)
    return eroded_mask

def normal_edge(depth):
    # depth (H, W) tensor
    d_im = depth.numpy()[..., None]
    zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
    zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0:3] /= n[:, :, None]
    normal = (normal + 1) / 2 * 255
    normal = normal[:, :, ::-1]
    gray = cv2.cvtColor(normal.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    edges_dilation = cv2.dilate(edges, structuring_element, iterations=1)
    return normal, edges, edges_dilation

def triangulate_depth(flow, mask, K4, M10, check=False):
    # flow: 2, H, W tensor
    # mask: H, W numpy array
    H, W = flow.size(1), flow.size(2)
    grid = get_pixelgrid(H, W).permute(1, 2, 0)
    grid_nxt = torch.clone(grid)
    grid_nxt[...,:2] = (grid[...,:2] + flow.permute(1,2,0)[...,:2]).int()
    grid, grid_nxt = grid.reshape(-1, 3), grid_nxt.reshape(-1, 3)
    pts1, pts2 = grid[...,:2], grid_nxt[...,:2] 
    pts1, pts2 = np.ascontiguousarray(pts1.numpy(), np.float32), np.ascontiguousarray(pts2.numpy(), np.float32)
    MASK = mask.reshape(-1).astype(bool)
    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K4[:3,:3].numpy(), distCoeffs=None)
    pts_l_norm = pts_l_norm[MASK,0]
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K4[:3,:3].numpy(), distCoeffs=None)
    pts_r_norm = pts_r_norm[MASK,0]
    pc_pre = cv2.triangulatePoints(torch.eye(4)[:3].numpy(), M10[:3].numpy(), pts_l_norm.T, pts_r_norm.T).T
    pc_pre = pc_pre[:,:3]/pc_pre[:,3:4]
    tri_depth_correct = pc_pre[:, 2] 
    triangulate_dep = torch.zeros(H, W).view(-1)
    triangulate_dep[MASK] = torch.from_numpy(tri_depth_correct)
    triangulate_dep = triangulate_dep.view(H, W)
    if check:
        error = compute_epipolar_distance(M10.numpy(), torch.eye(3).numpy(), np.concatenate((pts_l_norm, np.ones((pts_l_norm.shape[0], 1))), axis=1).T,
            np.concatenate((pts_r_norm, np.ones((pts_r_norm.shape[0], 1))), axis=1).T)
        Error = error.mean()
        return triangulate_dep, Error
    else:
        return triangulate_dep

class SCALE(nn.Module):
    # ---------- Model For Scale Parameters ----------
    def __init__(self, length = 1):
        super(SCALE, self).__init__()
        dtype = torch.FloatTensor
        alpha = 1. ; beta = 0.
        self.scales = torch.nn.Parameter(torch.Tensor([alpha, beta]).type(dtype).repeat(length, 1, 1), requires_grad=True)
    def forward(self, depth):
        # depth (B, H, W)
        depth = self.scales[..., 0] * depth + self.scales[..., 1]
        return depth

def process_data(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img_np = np.array(img)
    if img_np.shape[2] == 4:
        input_mask = img_np[:,:,3] != 0
    else:
        input_mask = None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def process_optical_flow(model_flow, test_image1, test_image2, H, W, H_flow, W_flow):
    img1 = np.array(cv2.imread(test_image1)).astype(np.uint8)[..., :3]
    img2 = np.array(cv2.imread(test_image2)).astype(np.uint8)[..., :3]
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1_f = cv2.resize(img1.permute(1,2,0).numpy(), (W_flow, H_flow))
    img2_f = cv2.resize(img2.permute(1,2,0).numpy(), (W_flow, H_flow))
    with torch.no_grad():
        former_1_2, _, _  = model_flow(torch.from_numpy(img1_f[None]).permute(0, 3, 1, 2).to(f'cuda:{model_flow.device_ids[0]}'), \
                    torch.from_numpy(img2_f[None]).permute(0, 3, 1, 2).to(f'cuda:{model_flow.device_ids[0]}')) 
        former_2_1, _, _  = model_flow(torch.from_numpy(img2_f[None]).permute(0, 3, 1, 2).to(f'cuda:{model_flow.device_ids[0]}'), \
                    torch.from_numpy(img1_f[None]).permute(0, 3, 1, 2).to(f'cuda:{model_flow.device_ids[0]}')) 
    if W_flow == W and H_flow == H:
        return former_1_2, former_2_1
    else:
        former_1_2 = torch.nn.functional.interpolate(former_1_2[0], size=(H, W), mode="bilinear")[0]
        former_1_2[0] *= W / W_flow
        former_1_2[1] *= H / H_flow
        former_2_1 = torch.nn.functional.interpolate(former_2_1[0], size=(H, W), mode="bilinear")[0]
        former_2_1[0] *= W / W_flow
        former_2_1[1] *= H / H_flow
        return former_1_2[None], former_2_1[None]

def optimizer_depth(depth_pred, tri_depths, final_masks):
    lr = 1e-2
    number = 3
    total_iteration = 128 * number
    B = tri_depths.size(0)

    scale_model = SCALE(length = 1)
    scale_model.cuda()
    optim = torch.optim.Adam
    optim_params = dict()
    param_list = [{'params': [scale_model.scales]}]
    optimizer = optim(param_list, lr=lr)

    for iteration in range(total_iteration):
        scale_model.zero_grad()
        Depth_Pred = scale_model(depth_pred[None].repeat(B, 1, 1))
        loss = ((tri_depths - Depth_Pred).abs() * final_masks).mean()
        loss.backward()
        optimizer.step()
        if iteration == 0:
            print('Init_Error:', ((tri_depths.view(B, -1) - Depth_Pred.view(B, -1)).abs() * final_masks.view(B, -1)).mean(-1))
        elif (iteration+1) % total_iteration == 0:
            print('End_Error:', ((tri_depths.view(B, -1) - Depth_Pred.view(B, -1)).abs() * final_masks.view(B, -1)).mean(-1))
    return scale_model.scales

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)
    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        _pix_coords_ = torch.clone(pix_coords)
        _pix_coords_[..., 0] /= self.width - 1
        _pix_coords_[..., 1] /= self.height - 1
        _pix_coords_ = (_pix_coords_ - 0.5) * 2
        return _pix_coords_, pix_coords

def main(address, scene_name):
    # ===================== Loading Flow Model =====================
    cfg = get_things_cfg()
    model_flow = torch.nn.DataParallel(build_flowformer(cfg))
    model_flow.load_state_dict(torch.load(cfg.model))
    model_flow = model_flow.cuda()
    model_flow.eval()
    
    # ===================== Loading Depth Model =====================
    midas_pretrain_path = 'pretrained_weights/midas_cpkt.pt'
    depth_model = MidasNet(midas_pretrain_path, non_negative=True, resize=[256, 512], normalize_input=True)
    depth_model = depth_model.cuda() # we are going to use this.

    # ===================== Loading Data =====================
    Visualize_flag = True
    
    save_prefix = f'{address}/{scene_name}_git'
    if scene_name == 'train':
        cur_max = 39
    elif scene_name == 'dog':
        cur_max = 58
    
    if not os.path.exists(os.path.join(save_prefix, 'pair_file_init')):
        os.makedirs(os.path.join(save_prefix, 'pair_file_init'))
    if not os.path.exists(os.path.join(save_prefix, 'pair_file')):
        os.makedirs(os.path.join(save_prefix, 'pair_file'))
    data_type = 'davis_depth'
    address = 'test_img/%s'%data_type
    if not os.path.exists(address):
        os.makedirs(address)
    gaps = [-4, -3, -2, -1, 1, 2, 3, 4]
    for cur in range(cur_max+1):
        triangulate_depths = []
        tensor_dict = {}
        for item, gap in enumerate(gaps):
            # ========================================== Data Loader ==========================================
            img_prefix1 = os.path.join(save_prefix, 'image', f'image_{cur:05d}.png')
            cam_prefix1 = os.path.join(save_prefix, 'camera', f'camera_{cur:05d}.npz')
            motion_prefix1 = os.path.join(save_prefix, 'motion', f'mask_{cur:05d}.png')

            ref = cur+gap
            if ref < 0 or ref > cur_max:
                ref = cur - gap
            img_prefix2 = os.path.join(save_prefix, 'image', f'image_{ref:05d}.png')
            cam_prefix2 = os.path.join(save_prefix, 'camera', f'camera_{ref:05d}.npz')
            motion_prefix2 = os.path.join(save_prefix, 'motion', f'mask_{ref:05d}.png')


            mask1 = np.array(cv2.imread(motion_prefix1, cv2.IMREAD_GRAYSCALE)).astype(np.uint8)/255.
            mask2 = np.array(cv2.imread(motion_prefix2, cv2.IMREAD_GRAYSCALE)).astype(np.uint8)/255.
            H, W = mask1.shape[0], mask1.shape[1]
            camera1 = np.load(cam_prefix1)
            camera2 = np.load(cam_prefix2)
            K4 = torch.from_numpy(camera1['intrinsic'])
            M1 = torch.from_numpy(camera1['extrinsic'])
            M1[:3, :3] = M1[:3, :3]
            M2 = torch.from_numpy(camera2['extrinsic'])
            M2[:3, :3] = M2[:3, :3]
            M12 = torch.matmul(torch.linalg.inv(M2), M1)
            # ===================== Check =====================
            backproj = BackprojectDepth(1, H, W)
            projection = Project3D(1, H, W)
            # ===================== Flow Prediction =====================
            former_1_2, former_2_1 = process_optical_flow(model_flow, img_prefix1, img_prefix2, H, W, H, W)
            eroded_mask1 = correspondence_mask(former_2_1[0][0].cpu().detach(), former_1_2[0][0].cpu().detach())
            eroded_mask2 = correspondence_mask(former_1_2[0][0].cpu().detach(), former_2_1[0][0].cpu().detach())
            # ===================== Process Depth Map =====================
            if item == 0:
                test_image1 = os.path.join(img_prefix1)
                image1 = torch.from_numpy(np.array(process_data(test_image1)))[None]
                with torch.no_grad():
                    image1 = image1.permute(0, 3, 1, 2).cuda()
                    depth_pred1 = depth_model(image1/255.)
                    depth_pred1 = depth_pred1[0, 0]
                normal1, edges1, edges_dilation1 = normal_edge(depth_pred1.cpu())
                # ===================== Motion Mask =====================
                structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
                inv_motion_mask = cv2.erode((1.-mask1)*255., structuring_element, iterations=1)
                motion_mask = 1 - inv_motion_mask / 255.
            # ===================== Final Mask =====================
            final_mask_init = (1 - motion_mask + 1 - edges_dilation1/255. + eroded_mask1)
            final_mask_init = final_mask_init >= 3.
            triangulate_dep, pose_error = triangulate_depth(former_1_2[0][0].cpu().detach(), final_mask_init, K4, M12, check=True)
            final_mask = final_mask_init * (triangulate_dep.numpy() >= 0.) * (triangulate_dep.numpy() <= 8.)
            MASK = torch.from_numpy(final_mask).bool()
            triangulate_depths.append(triangulate_dep)
            if item == 0:
                final_masks = MASK[None].cuda()
            final_masks = final_masks * MASK[None].cuda()
            # ===================== Check =====================
            intrinsic = torch.eye(4)
            intrinsic[:3, :3] = K4
            cam_point1 = backproj(triangulate_dep[None], torch.linalg.inv(intrinsic)[None])
            _, cam_point12 = projection(cam_point1, intrinsic[None], M12[None])
            cam_point1 = backproj.pix_coords
            cam_point1 = cam_point1.view(1, 3, H, W)
            cam_point1 = cam_point1.permute(0, 2, 3, 1)[..., :2]
            depth_flow12_ = (cam_point12 - cam_point1)[0]
            error_img0 = (former_1_2[0][0].permute(1,2,0).cpu() - depth_flow12_).numpy()*final_mask[:,:,None]
            error_img0 = np.mean(np.abs(error_img0), axis=2)
            print(np.mean(error_img0), '------Error Triangulation1', cur, gap)

            cam_point1 = backproj(depth_pred1[None].cpu(), torch.linalg.inv(intrinsic)[None])
            _, cam_point12 = projection(cam_point1, intrinsic[None], M12[None])
            cam_point1 = backproj.pix_coords
            cam_point1 = cam_point1.view(1, 3, H, W)
            cam_point1 = cam_point1.permute(0, 2, 3, 1)[..., :2]
            depth_flow12_ = (cam_point12 - cam_point1)[0]
            error_img0 = (former_1_2[0][0].permute(1,2,0).cpu() - depth_flow12_).numpy()*final_mask[:,:,None]
            error_img0 = np.mean(np.abs(error_img0), axis=2)
            # ===================== Saving Dict =====================
            tensor_dict[f'flow_{cur:02d}_{gap:02d}'] = former_1_2[0][0].permute(1,2,0).cpu()
            tensor_dict[f'flow_{gap:02d}_{cur:02d}'] = former_2_1[0][0].permute(1,2,0).cpu()
            tensor_dict[f'R_1'] = M1[:3, :3]
            tensor_dict[f't_1'] = M1[:3, 3:]
            tensor_dict[f'K'] = K4
            tensor_dict[f'motion_1'] = torch.from_numpy(1 - motion_mask).bool()
            tensor_dict[f'corr_{cur:02d}_{gap:02d}'] = torch.from_numpy(eroded_mask1).bool()
            tensor_dict[f'corr_{gap:02d}_{cur:02d}'] = torch.from_numpy(eroded_mask2).bool()
            tensor_dict[f'edge_1'] = torch.from_numpy(1 - edges_dilation1/255.).bool()
            tensor_dict[f'img_1'] = image1/255.

            if Visualize_flag:
                # Visualize flow
                flow_imgs = flow_to_image(former_1_2[0])
                plt.imsave('%s/%s_flowformer.png'%(address, f'ref_{cur:02d}_{gap:02d}'), flow_imgs[0].float().permute(1,2,0).cpu().numpy()/255.)
                plt.close("all")                
                # Visualize depth and normal
                visualize_depth(depth_pred1.cpu().numpy(), address, f'tgt_{cur:02d}_midas')
                cv2.imwrite(os.path.join(address, f'tgt_{cur:02d}_normal.png'), normal1)
                plt.imsave(os.path.join(address, f'tgt_{cur:02d}_edge_dilate.png'), edges_dilation1, cmap='gray')
                plt.close("all")
                # Visualize motion
                plt.imsave(os.path.join(address, f'tgt_{cur:02d}_motion.png'), motion_mask*255., cmap='gray')
                plt.close("all")
                # Visualize final mask
                plt.imsave('%s/%s_mask_final.png'%(address, f'ref_{cur:02d}_{gap:02d}'), final_mask, cmap='gray')
                plt.close("all")
                # Visualize final depth
                depth = final_mask * triangulate_dep.numpy()
                visualize_depth(depth, address, f'ref_{cur:02d}_{gap:02d}_triangulate')

        tri_depths = torch.stack(triangulate_depths, dim=0).cuda()
        B = tri_depths.size(0)
        final_masks = final_masks.repeat(B, 1, 1)
        scale_param = optimizer_depth(depth_pred1, tri_depths, final_masks)

        test_image1 = os.path.join(img_prefix1)
        torch_image1 = torch.from_numpy(np.array(process_data(test_image1)))
        test_image2 = os.path.join(img_prefix2)
        torch_image2 = torch.from_numpy(np.array(process_data(test_image2)))

        tensor_dict['scale_1a'] = scale_param[0, 0, 0].detach().cpu()
        tensor_dict['scale_1b'] = scale_param[0, 0, 1].detach().cpu()
        tensor_dict['depth_pred_1'] = depth_pred1.cpu()

        file_name = f'cur_{cur:05d}.pt'
        save_total_file = os.path.join(save_prefix, 'pair_file_init', file_name)
        torch.save(tensor_dict, save_total_file)

    for cur in range(cur_max+1):
        for item, gap in enumerate(gaps):
            ref = cur+gap
            if ref < 0 or ref > cur_max:
                pass
            else:                
                file_name1 = f'cur_{cur:05d}.pt'
                save_total_file1 = os.path.join(save_prefix, 'pair_file_init', file_name1)
                tensor_dict1 = torch.load(save_total_file1)

                file_name2 = f'cur_{ref:05d}.pt'
                save_total_file2 = os.path.join(save_prefix, 'pair_file_init', file_name2)
                tensor_dict2 = torch.load(save_total_file2)

                tensor_dict3 = {}
                tensor_dict3['R_1'] = tensor_dict1['R_1']
                tensor_dict3['t_1'] = tensor_dict1['t_1']
                tensor_dict3['R_2'] = tensor_dict2['R_1']
                tensor_dict3['t_2'] = tensor_dict2['t_1']
                tensor_dict3['K'] = tensor_dict1['K']
                tensor_dict3['img_1'] = tensor_dict1['img_1'][0].permute(1,2,0).cpu()
                tensor_dict3['img_2'] = tensor_dict2['img_1'][0].permute(1,2,0).cpu()
                tensor_dict3['flow_1_2'] = tensor_dict1[f'flow_{cur:02d}_{gap:02d}']
                tensor_dict3['flow_2_1'] = tensor_dict1[f'flow_{gap:02d}_{cur:02d}']

                tensor_dict3['depth_pred_1'] = tensor_dict1['depth_pred_1']
                tensor_dict3['scale_1a'] = tensor_dict1['scale_1a']
                tensor_dict3['scale_1b'] = tensor_dict1['scale_1b']

                tensor_dict3['depth_pred_2'] = tensor_dict2['depth_pred_1']
                tensor_dict3['scale_2a'] = tensor_dict2['scale_1a']
                tensor_dict3['scale_2b'] = tensor_dict2['scale_1b']

                tensor_dict3[f'motion_1'] = tensor_dict1[f'motion_1']
                tensor_dict3[f'motion_2'] = tensor_dict2[f'motion_1']
                tensor_dict3[f'corr_1_2'] = tensor_dict1[f'corr_{cur:02d}_{gap:02d}']
                tensor_dict3[f'corr_2_1'] = tensor_dict1[f'corr_{gap:02d}_{cur:02d}']
                tensor_dict3[f'edge_1'] = tensor_dict1[f'edge_1'] 
                tensor_dict3[f'edge_2'] = tensor_dict2[f'edge_1']

                tensor_dict3[f'fid_1'] = torch.Tensor([cur])[0]
                tensor_dict3[f'fid_2'] = torch.Tensor([ref])[0]

                file_name = f'gap_{gap:02d}_cur_{cur:05d}.pt'
                save_total_file = os.path.join(save_prefix, 'pair_file', file_name)
                torch.save(tensor_dict3, save_total_file)

if __name__ == '__main__':
    address = '/root/autodl-tmp/kaichen/data/davis'
    # give the directory of dataset.
    main(address=address, scene_name = 'train')
    main(address=address, scene_name = 'dog')