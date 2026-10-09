import torch
import torch.nn.functional as F
import inspect
from functools import partial
from losses.scene_flow_projection import flow_by_depth_fn, scene_flow_projection_fn, unproject_ptcld
from .sceneflow_net import SceneFlowFieldNet
import numpy as np
from os.path import join
from os import makedirs
import pdb
import torch.nn as nn
from core_depth.MiDaS import MidasNet
from torch.nn import init

class Model(nn.Module):
    def __init__(self, opt, _device_):
        super(Model, self).__init__()
        self.opt = opt
        self.net_sceneflow = SceneFlowFieldNet(net_width=256, n_layers=4, \
            N_freq_xyz=opt.n_freq_xyz, N_freq_t=opt.n_freq_t).to(_device_)
        self.init_weight(self.net_sceneflow, 'kaiming', 0.01, a=0.2)
        # Function to be used        
        self.depth_flow = flow_by_depth_fn(self.opt.batch_size, self.opt.H, self.opt.W, _device_)
        flow_arg_list = inspect.getargspec(self.depth_flow.forward).args[1:]
        self._input_name_ori = ['R_1', 't_1', 'R_2', 't_2', 'K', 'img_1', 'img_2', 'flow_1_2', 'flow_2_1', \
        'depth_pred_1', 'scale_1a', 'scale_1b', 'depth_pred_2', 'scale_2a', 'scale_2b', 'motion_1', \
        'motion_2', 'corr_1_2', 'corr_2_1', 'edge_1', 'edge_2', 'fid_1', 'fid_2']
        self._input_name_pre = ['time_step', 'time_stamp_1', 'time_stamp_2', 'frame_id_1', 'frame_id_2', 'R_1_T', 'R_2_T', 'K_inv']
        self._input_name = self._input_name_ori + self._input_name_pre
        self.flow_args = []
        for x in flow_arg_list:
            if x in self._input_name:
                self.flow_args.append(x)
        self.warp = scene_flow_projection_fn(self.opt.batch_size, self.opt.H, self.opt.W, _device_)
        self.unproject_points = unproject_ptcld()

        midas_pretrain_path = 'pretrained_weights/midas_cpkt.pt'
        self.depth_model = MidasNet(midas_pretrain_path, non_negative=True, resize=[256, 512], normalize_input=True)
        self.depth_model = self.depth_model.to(_device_)

    def predict_on_batch_train(self, data_batch, _device_, epoch):
        for k in data_batch.keys():
            if type(data_batch[k]) is not list:
                data_batch[k] = data_batch[k].to(_device_)
        self._input = data_batch
        #depth_1 = self._input['depth_pred_1'] * self._input['scale_1a'] + self._input['scale_1b']
        #depth_2 = self._input['depth_pred_2'] * self._input['scale_2a'] + self._input['scale_2b']
        _depth_1_ = self.depth_model(self._input['img_1'])
        depth_1 = (_depth_1_ * self._input['scale_1a'] + self._input['scale_1b']).squeeze(1)
        _depth_2_ = self.depth_model(self._input['img_2'])
        depth_2 = (_depth_2_ * self._input['scale_2a'] + self._input['scale_2b']).squeeze(1)
        #depth_1 = self._input['depth_pred_1']
        #depth_2 = self._input['depth_pred_2']
        flow_data_input = {'depth_1': depth_1, 'depth_2': depth_2}
        for k in self.flow_args:
            flow_data_input[k] = self._input[k]
        dflow = self.depth_flow(**flow_data_input) 
        global_p1 = dflow['global_p1'].squeeze(3).permute(0, 3, 1, 2)  # .detach()  # B3HW
        global_p2 = dflow['global_p2'].squeeze(3).permute(0, 3, 1, 2)  # .detach()  # B3HW

        time_step = self._input['time_step'].squeeze().item()
        time_gap = torch.mean(self._input['time_stamp_2'] - self._input['time_stamp_1'])
        steps = (time_gap / time_step).round().long().item()
        sf_1_2 = self.forward_sf_net_multi_step(global_p1, self._input['time_stamp_1'], time_step=time_step, steps=steps)
        
        flow_data_input['sflow_1_2'] = sf_1_2.permute(0, 2, 3, 1)[..., None, :]  
        flow_data_input['flow_1_2'] = self._input['flow_1_2']
        result = self.warp(**flow_data_input)
        # -- 'dflow_1_2': Optic Flow - Camera + Movement (Involve Scene Flow) # Supervised the correctness of X,Y
        # -- 'p1_camera_2': Homography Point in 2 (Without Intrinsic Matrix - Involving Scene Flow) # Supervised the correctness of X,Y
        # -- 'warped_p2_camera_2': Depth of 2 in 1 (Based On Depth 2 - Involving Optical Flow)
        result['sf_1_2'] = sf_1_2
        result['global_p1'] = global_p1
        result['global_p2'] = global_p2
        return result

    def cal_loss(self, pred):
        # depth mask for wired prediction
        mask = (pred['depth_1'].detach() < 100).float().squeeze(1) * self._input['corr_1_2'] 
        mask = (pred['warped_p2_camera_2'][..., 2].detach() < 100).float().squeeze(3) * mask
        #crit = partial(F.mse_loss, reduction='none') if self.warm else partial(F.l1_loss, reduction='none')
        crit = partial(F.l1_loss, reduction='none')
        occ_mask = mask[:, :, :, None]
        # =========== 2D optic flow error (Optic Flow And Scene Flow Projection) ===========
        scene_flow_loss_1_2 = crit(pred['dflow_1_2'], self._input['flow_1_2'])
        error_scene_img = (torch.sum(torch.abs(scene_flow_loss_1_2), dim=-1)).detach()
        flow_loss_1_2 = torch.sum(occ_mask * scene_flow_loss_1_2) / (torch.sum(occ_mask) + 1e-8)
        # =========== Disparity Error (Optic Flow And Scene Flow Projection) ===========
        disp_loss_1_2 = self.disp_loss(pred['p1_camera_2'][..., -1], pred['warped_p2_camera_2'][..., -1]).permute([0, 3, 1, 2])
        disp_loss_1_2 = torch.sum(occ_mask[:, None, ..., 0] * disp_loss_1_2[:, 0:1, ...]) / (torch.sum(occ_mask) + 1e-8)
        loss = flow_loss_1_2 * self.opt.flow_mul + disp_loss_1_2 * self.opt.disp_mul
        loss_data = {'total_loss': loss.item(), 
                    'loss': loss.item(),
                    'flow_loss_1_2': flow_loss_1_2.item(),
                    'disp_loss_1_2': disp_loss_1_2.item()}
        return loss, loss_data, error_scene_img

    def disp_loss(self, d1, d2):
        if self.opt.use_disp:
            t1 = torch.clamp(d1, min=1e-3)
            t2 = torch.clamp(d2, min=1e-3)
            return 100 * torch.abs((1 / t1) - (1 / t2))
        if self.opt.use_disp_ratio:
            t1 = torch.clamp(d1, min=1e-3)
            t2 = torch.clamp(d2, min=1e-3)
            return torch.max(t1, t2) / torch.min(t1, t2) - 1
        else:
            return torch.abs(d1 - d2)

    def opt_cycle(self, pred):
        # replace this one with cycle:
        global_p1 = pred['global_p1']
        sf_forward, sf_backward = self.forward_sf_net(global_p1, self._input['time_stamp_1'])
        mseg = torch.ones_like(sf_forward)
        time_step = self._input['time_step'].squeeze().item()
        time_stamp = (self._input['time_stamp_1'] + time_step)
        global_p1_interp = (global_p1 + sf_forward)
        sf_forward_t1, sf_backward_t1 = self.forward_sf_net(global_p1_interp, time_stamp)
        acc_loss = (mseg * torch.abs(sf_forward + sf_backward_t1)).sum() / (mseg.sum() + 1e-6)
        loss = acc_loss * self.opt.acc_mul
        loss.backward()
        return loss.item()

    def forward_sf_net(self, global_p1, ts):
        sf_forward, sf_backward = self.net_sceneflow(global_p1, ts)
        sf_forward /= self.opt.sf_mag_div  # s1000
        sf_backward /= self.opt.sf_mag_div  # s1000
        return sf_forward, sf_backward

    def forward_sf_net_multi_step(self, global_p1, time_stamp, time_step, steps):
        sf_acc = 0
        if steps > 0:
            for i in range(steps):
                sf_forward, sf_backward = self.forward_sf_net(global_p1, time_stamp)
                sf_acc = sf_acc + sf_forward
                global_p1 = global_p1 + sf_forward
                time_stamp = time_stamp + time_step
        if steps < 0:
            for i in range(-steps):
                sf_forward, sf_backward = self.forward_sf_net(global_p1, time_stamp)
                sf_acc = sf_acc + sf_backward
                global_p1 = global_p1 + sf_backward
                time_stamp = time_stamp + time_step
        return sf_acc

    def init_weight(self, net=None, init_type='kaiming', init_param=0.02, a=0, turnoff_tracking=False):
        def init_func(m, init_type=init_type):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=a, mode='fan_in')
                elif init_type == 'orth':
                    init.orthogonal_(m.weight.data, gain=init_param)
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                if m.affine:
                    init.normal_(m.weight.data, 1.0, init_param)
                    init.constant_(m.bias.data, 0.0)
                if turnoff_tracking:
                    m.track_running_stats = False
        init_func(net)
    
    def predict_on_batch_val(self, data_batch, _device_):
        for k in data_batch.keys():
            if type(data_batch[k]) is not list:
                data_batch[k] = data_batch[k].to(_device_)
        self._input = data_batch
        _depth_1_ = self.depth_model(self._input['img_1'])
        depth_1 = (_depth_1_ * self._input['scale_1a'] + self._input['scale_1b']).squeeze(1)
        _depth_2_ = self.depth_model(self._input['img_2'])
        depth_2 = (_depth_2_ * self._input['scale_2a'] + self._input['scale_2b']).squeeze(1)

        flow_data_input = {'depth_1': depth_1, 'depth_2': depth_2}
        for k in self.flow_args:
            flow_data_input[k] = self._input[k]
        dflow = self.depth_flow(**flow_data_input) 
        global_p1 = dflow['global_p1'].squeeze(3).permute(0, 3, 1, 2)  # .detach()  # B3HW
        global_p2 = dflow['global_p2'].squeeze(3).permute(0, 3, 1, 2)  # .detach()  # B3HW
        
        time_step = self._input['time_step'].squeeze().item()
        time_gap = torch.mean(self._input['time_stamp_2'] - self._input['time_stamp_1'])
        steps = (time_gap / time_step).round().long().item()
        sf_1_2 = self.forward_sf_net_multi_step(global_p1, self._input['time_stamp_1'], time_step=time_step, steps=steps)
        
        flow_data_input['sflow_1_2'] = sf_1_2.permute(0, 2, 3, 1)[..., None, :]  
        flow_data_input['flow_1_2'] = self._input['flow_1_2']
        result = self.warp(**flow_data_input)

        mask = (result['depth_1'].detach() < 100).float().squeeze(1) * self._input['corr_1_2'] 
        mask = (result['warped_p2_camera_2'][..., 2].detach() < 100).float().squeeze(3) * mask
        return global_p1, sf_1_2, mask