import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, opt):
        self.opt = opt
        check_dir = os.path.join(opt.log_file, opt.checkpoint)
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
        self.error_dir =  os.path.join(check_dir, 'error')  
        if not os.path.exists(self.error_dir):
            os.makedirs(self.error_dir)
        self.depth_dir =  os.path.join(check_dir, 'depth')  
        if not os.path.exists(self.depth_dir):
            os.makedirs(self.depth_dir)
        self.ckpt_dir = os.path.join(check_dir, 'ckpt')  
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.log_dir = os.path.join(check_dir, 'log.txt')
        self.init_log()

    def init_log(self,):
        self.epoch = -1
        self.loss = {}

    def visualize_error(self, epoch, batch, error_map):
        try:
            if batch % self.opt.frequency == 0:
                plt.imsave(f'{self.error_dir:}/{epoch:02d}_{batch:04d}_error.png', error_map, cmap='Reds', vmax=10)
                plt.close('all')
        except:
            print('----------ERROR LARGER THAN MAX----------')
    def visualize_depth(self, epoch, batch, depth):
        if batch % self.opt.frequency == 0:
            depth1_vis = np.clip(depth.detach().cpu().numpy(), a_min=0, a_max=100)
            vmax = np.percentile(depth1_vis, 100)
            plt.imsave(f'{self.depth_dir:}/{epoch:02d}_{batch:04d}_depth.png', depth1_vis, cmap='turbo', vmax=15)
            plt.close('all')

    def save_ckpt(self, epoch, batch, model):
        weight_dir = os.path.join(self.ckpt_dir, f'{epoch:02d}_{batch:04d}.pth')
        torch.save(model.state_dict(), weight_dir)

    def record_log(self, epoch, batch, loss):
        if self.epoch != epoch:
            self.epoch = epoch
            self.loss = loss
        elif self.epoch == epoch:
            for k in loss.keys():
                self.loss[k] += loss[k]
            if batch % self.opt.frequency == 0 and batch != 0:
                loss_rep = {}
                for k in self.loss.keys():
                    loss_rep[k] = self.loss[k]/(batch + 1)
                print(f'{epoch:02d}_{batch:04d}: ' + str(loss_rep) + '\n')
                with open(self.log_dir, 'a') as file:
                    file.write(f'{epoch:02d}_{batch:04d}: ' + str(loss_rep) + '\n')