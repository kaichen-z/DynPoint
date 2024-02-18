import sys
import argparse
import torch

def add_general_arguments():
    # Parameters that will NOT be overwritten when resuming
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('--manual_seed', type=int, default=1314,
                        help='manual seed for randomness')
    parser.add_argument('--epoch', type=int, default=0,
                        help='number of epochs to train')
    parser.add_argument('--resume', type=str, default='07_0255.pth',
                        help='directory of checkpoint')
    
    # Log Dir
    parser.add_argument('--prefix', type=str, default=None,
                        help='prefx for saving')
    parser.add_argument('--log_file', type=str, default="checkpoints",
                        help='prefx for saving')
    parser.add_argument('--checkpoint', type=str, default="exp1",
                        help='prefx for saving')
    parser.add_argument('--frequency', type=int, default=150)
    
    # Dataset IO 
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/kaichen/data/davis/',
                        help='dataset directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset to use')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--no_batching', action='store_true', help='do not use batching.')
    parser.add_argument('--vali_batches', default=None,
                        type=int, help='max number of batches used for validation per epoch')
    parser.add_argument('--vali_at_start', action='store_true',
                        help='run validation before starting to train')
    parser.add_argument('--log_time', action='store_true',
                        help='adding time log')
    parser.add_argument('--print_net', action='store_true',
                        help="print network")
    parser.add_argument('--H', default=192, type=int,
                        help='Height of Image')
    parser.add_argument('--W', default=384, type=int,
                        help="Width of Image")

    # Optimizer
    parser.add_argument('--gpu', type=int, default=0)    
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--adam_beta1', type=float, default=0.5,
                        help='beta1 of adam')
    parser.add_argument('--adam_beta2', type=float, default=0.9,
                        help='beta2 of adam')
    parser.add_argument('--sgd_momentum', type=float, default=0.9,
                        help="momentum factor of SGD")
    parser.add_argument('--sgd_dampening', type=float, default=0,
                        help="dampening for momentum of SGD")
    parser.add_argument('--wdecay', type=float, default=0.0,
                        help='weight decay')

    # initialization
    parser.add_argument('--init_type', type=str, default='normal', help='type of initialziation to use')

    # Mixed precision training
    parser.add_argument('--mixed_precision_training', action='store_true',
                        help='use mixed precision for training.')
    parser.add_argument('--loss_scaling', type=float, default=255,
                        help='the loss scale factor for mixed precision training. Set to -1 for dynamic scaling.')
                        
    # Logging and visualization
    parser.add_argument('--logdir', type=str, default=None,
                        help='Root directory for logging. Actual dir is [logdir]/[net_classes_dataset]/[expr_id]')
    parser.add_argument('--save_net', type=int, default=1,
                        help='Period of saving network weights')
    parser.add_argument('--expr_id', type=int, default=0,
                        help='Experiment index. non-positive ones are overwritten by default. Use 0 for code test. ')
    
    # Model Parameters
    parser.add_argument('--net', type=str, default='scene_flow_motion_field')
    parser.add_argument('--l1_mul', type=float, default=1e-4, help='L1 multiplier')
    parser.add_argument('--disp_mul', type=float, default=10, help='disparity multiplier')
    parser.add_argument('--one_way', action='store_true', help='use only losses on 1 to 2')
    parser.add_argument('--loss_type', type=str, default='l2', help='use l2 on sceneflow')
    parser.add_argument('--scene_lr_mul', type=float, default=1, help='lr multiplier for scene flow network')
    parser.add_argument('--n_down', type=int, default=3, help='sf net size')
    parser.add_argument('--sf_min_mul', type=float, default=0, help='minimize sf')
    parser.add_argument('--sf_quantile', type=float, default=0.5, help='minimize sf for 50% pixels')
    parser.add_argument('--static', action='store_true', help='optimize static regions with skip frames')
    parser.add_argument('--static_mul', type=float, default=1, help='multiplier for static large baseline losses')
    parser.add_argument('--flow_mul', type=float, default=10, help='multiplier for flow losses')
    parser.add_argument('--acc_mul', type=float, default=100, help='multiplier for acceleration regularization losses')
    parser.add_argument('--si_mul', type=float, default=0, help='multiplier for scale invariant losses')
    parser.add_argument('--cos_mul', type=float, default=0, help='multiplier for cosine angle losses for optical flow')
    parser.add_argument('--motion_seg_hard', action='store_true', help='flag for using hard motion segmentations')
    parser.add_argument('--warm_mul', type=float, default=1, help='multiplier for warm up state training')
    parser.add_argument('--interp_steps', type=int, default=2, help='steps for interpolation')
    parser.add_argument('--warm_static', action='store_true', help='only use static loss for warm up')
    parser.add_argument('--use_disp', action='store_true', help='flag for using disp losses')
    parser.add_argument('--use_disp_ratio', action='store_true', help='use  disp ratio losses')
    parser.add_argument('--time_dependent', action='store_true', help='flag for time dependent scene flow model')
    parser.add_argument('--use_cnn', action='store_true', help='flag for using CNN for scene flow model')
    parser.add_argument('--use_embedding', action='store_true', help='flag for using optimizable embedding for each frame')
    parser.add_argument('--use_motion_seg', action='store_true', help='flag for using motion seg')
    parser.add_argument('--warm_reg', action='store_true', help='use reg for warm up as well')
    parser.add_argument('--warm_sf', type=int, default=0, help='warm up flow network for k epochs')
    parser.add_argument('--n_freq_xyz', type=int, default=16, help='xyz_embeddings')
    parser.add_argument('--n_freq_t', type=int, default=16, help='time embeddings')
    parser.add_argument('--sf_mag_div', type=float, default=100, help='divident for sceneflow network output, making it easier to optimize')
    parser.add_argument('--midas', action='store_true', help='use midas for depth prediction')

    # Dataset parameter
    parser.add_argument('--cache', action='store_true', help='cache the data into ram')
    parser.add_argument('--subsample', action='store_true', help='subsample the video in time')
    parser.add_argument('--track_id', default='train', type=str, help='the track id to load')
    parser.add_argument('--overfit', action='store_true', help='overfit and see if things works')
    parser.add_argument('--gaps', type=str, default='1,2,3,4', help='gaps for sequences')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeatition')
    parser.add_argument('--select', action='store_true', help='pred')
    return parser
