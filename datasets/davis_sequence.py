import numpy as np
from glob import glob
from os.path import join
import torch
import torch.utils.data as data

"""
['R_1', 'R_2', 'R_1_T', 'R_2_T', 't_1', 't_2', 'K', 
'K_inv', 'img_1', 'img_2', 'depth_1', 'flow_1_2', 'flow_2_1', 
'mask_1', 'mask_2', 'motion_seg_1', 'depth_pred_1', 'fid_1', 'fid_2']
"""

"""
img, pose_c2w, depth_mvs, intrinsics, depth_pred, img_orig, motion_seg
"""
class Dataset(data.Dataset):
    def __init__(self, opt, mode='train'):
        super().__init__()
        self.opt = opt
        self.mode = mode
        assert mode in ('train', 'vali')
        data_root = opt.data_root
        track_name = opt.track_id  
        data_path = join(data_root, track_name)
        _gaps_ = opt.gaps.split(',')
        gaps = [int(x) for x in _gaps_]
        gaps = [-int(x) for x in _gaps_[::-1]] + gaps 
        self.file_list = []
        for g in gaps:
            file_list = sorted(glob(join(data_path, 'pair_file', f'gap_{g:02d}_*.pt')))
            self.file_list += file_list
        self.n_frames = int(self.file_list[-1].split("/")[-1].split(".")[0].split("_")[-1]) + 1

    def __len__(self):
        if self.mode != 'train':
            return len(self.file_list)
        else:
            return len(self.file_list) * self.opt.repeat

    def __getitem__(self, idx):
        sample_loaded = {}
        unit = 1.0
        dataset = torch.load(self.file_list[idx])
        H, W, _ = dataset['img_1'].size()
        for k in dataset.keys():
            sample_loaded[k] = dataset[k].float()
        # To facilitate following computing in training process. 
        sample_loaded['img_1'] = dataset['img_1'].permute([2, 0, 1])
        sample_loaded['img_2'] = dataset['img_2'].permute([2, 0, 1])
        sample_loaded['time_step'] = unit / self.n_frames
        sample_loaded['time_stamp_1'] = (dataset['fid_1'].reshape([-1, 1, 1]).expand(-1, H, W) / self.n_frames).float()
        sample_loaded['time_stamp_2'] = (dataset['fid_2'].reshape([-1, 1, 1]).expand(-1, H, W) / self.n_frames).float()
        sample_loaded['frame_id_1'] = np.asarray(dataset['fid_1'])
        sample_loaded['frame_id_2'] = np.asarray(dataset['fid_2'])
        sample_loaded['t_1'] = (dataset['t_1'].T)[None, None, :, :]
        sample_loaded['t_2'] = (dataset['t_2'].T)[None, None, :, :]
        sample_loaded['R_1'] = (dataset['R_1'].T)[None, None, :, :]
        sample_loaded['R_2'] = (dataset['R_2'].T)[None, None, :, :]
        sample_loaded['R_1_T'] = (dataset['R_1'])[None, None, :, :]
        sample_loaded['R_2_T'] = (dataset['R_2'])[None, None, :, :]
        sample_loaded['K'] = (dataset['K'].T)[None, None, :, :]
        sample_loaded['K_inv'] = (torch.linalg.inv(dataset['K'].T))[None, None, :, :]
        sample_loaded['pair_path'] = self.file_list[idx]
        self.convert_to_float32(sample_loaded)
        return sample_loaded

    def convert_to_float32(self, sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                sample_loaded[k] = torch.from_numpy(v).float()