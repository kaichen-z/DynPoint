import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import datasets
import models_point as models
from options import options_train
from util import util_loadlib as loadlib
from tqdm import tqdm
import pdb
from os.path import join

class Validation(object):
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        Model = models.get_model(opt.net)
        self.model = Model(opt, device)
        self.model.load_state_dict(torch.load(self.opt.resume))
        data_root = opt.data_root
        track_name = opt.track_id  
        data_path = join(data_root, track_name)
        self.save_path = join(data_path, 'pair_file_sf')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def val(self, dataloader):
        self.model.eval()
        for i, data in enumerate(tqdm(dataloader)):
            tensor_dict = {}
            with torch.no_grad():
                global_p1, sf_1_2, mask = self.model.predict_on_batch_val(data, self.device)
            tensor_dict['mask'] = mask[0].detach().cpu()
            tensor_dict['scene_flow_1_2'] = sf_1_2[0].detach().cpu().permute(1,2,0)
            tensor_dict['global_p1'] = global_p1[0].detach().cpu().permute(1,2,0)
            tensor_dict['fid_1'] = data['fid_1'].cpu()
            tensor_dict['fid_2'] = data['fid_2'].cpu()
            tensor_dict['img_1'] = data['img_1'][0].detach().cpu().permute(1,2,0)
            cur = int(tensor_dict['fid_1'].numpy()[0])
            gap = int((tensor_dict['fid_2']-tensor_dict['fid_1']).numpy()[0])
            file_name = f'gap_{gap:02d}_cur_{cur:05d}.pt'
            save_total_file = os.path.join(self.save_path, file_name)
            torch.save(tensor_dict, save_total_file)

def main_worker(opt):
    # gpu setting
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
    # seed setting
    loadlib.set_manual_seed(opt.manual_seed)
    # loading dataset
    dataset = datasets.get_dataset(opt.dataset)
    dataset_vali = dataset(opt, mode='vali')
    dataloader_vali = torch.utils.data.DataLoader(
        dataset_vali,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False)
    val_class = Validation(opt, device)
    val_class.val(dataloader_vali)

def main():
    opt = options_train.add_general_arguments()
    opt = opt.parse_args()
    main_worker(opt=opt)

if __name__ == '__main__':
    main()
