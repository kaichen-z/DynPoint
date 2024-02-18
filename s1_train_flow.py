import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import datasets
import models_point as models
from options import options_train
from util import util_loadlib as loadlib
from util.util_logger import Logger
from tqdm import tqdm

"""
bash ./experiments/davis/train_sequence.sh --track_id train_git --checkpoint exp_train --dataset davis_sequence
bash ./experiments/davis/train_sequence.sh --track_id dog_git --checkpoint exp_dog --dataset davis_sequence
"""

class Training(object):
    def __init__(self, opt, device):
        if opt.optim == 'adam':
            optim = torch.optim.Adam
            optim_params = dict()
            optim_params['betas'] = (opt.adam_beta1, opt.adam_beta2)
        elif opt.optim == 'sgd':
            optim = torch.optim.SGD
            optim_params = dict()
            optim_params['momentum'] = opt.sgd_momentum
            optim_params['dampening'] = opt.sgd_dampening
            optim_params['weight_decay'] = opt.wdecay
        else:
            raise NotImplementedError(
                'optimizer %s not added yet.' % opt.optim)
        self.opt = opt
        self.device = device
        Model = models.get_model(opt.net)
        self.model = Model(opt, device)
        self.model.net_sceneflow.to(device)

        self._nets = [self.model.net_sceneflow, self.model.depth_model]
        self.optimizer_scene = optim(self.model.net_sceneflow.parameters(), lr=opt.lr * opt.scene_lr_mul, **optim_params)
        #self.optimizer_depth = optim(self.model.depth_model.scratch.refinenet1.parameters(), lr=opt.lr, **optim_params)
        self.optimizer_depth = optim(list(self.model.depth_model.parameters()), \
                                     lr=opt.lr, **optim_params)
        self._optimizers = [self.optimizer_scene, self.optimizer_depth]

        self.logger = Logger(opt)

    def train(self, dataloader, initial_epoch, epochs):
        for epoch in (range(initial_epoch, initial_epoch + epochs)):
            for i, data in enumerate(tqdm(dataloader)):
                for n in self._nets:
                    n.zero_grad()
                pred = self.model.predict_on_batch_train(data, self.device, epoch)
                loss, loss_data, error_scene_img = self.model.cal_loss(pred)
                self.logger.visualize_error(epoch, i, error_scene_img[0].cpu())
                self.logger.visualize_depth(epoch, i, pred['depth_1'][0,0])
                self.logger.record_log(epoch, i, loss_data)
                loss.backward(retain_graph=True) 
                reg_loss = self.model.opt_cycle(pred) 
                loss_data['acc_reg'] = reg_loss
                for optimizer in self._optimizers:
                    optimizer.step()
                torch.cuda.empty_cache()
            self.logger.save_ckpt(epoch, i, self.model)

def main_worker(opt):
    # gpu setting
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
    # seed setting
    loadlib.set_manual_seed(opt.manual_seed)
    # loading dataset
    dataset = datasets.get_dataset(opt.dataset)
    dataset_train = dataset(opt, mode='train')
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True)
    # Training
    train_class = Training(opt, device)
    train_class.train(dataloader_train, initial_epoch=0, epochs=opt.epoch)

def main():
    opt = options_train.add_general_arguments()
    opt = opt.parse_args()
    main_worker(opt=opt)

if __name__ == '__main__':
    main()
