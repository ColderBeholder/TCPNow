import os.path
import numpy as np
import argparse
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import json
import torch

def mse_evaluation(a, b):
    return np.mean(np.square((a - b)))

def mae_evaluation(a, b):
    return np.mean(np.abs((a - b)))

def tp(pre, gt):
    return np.sum(pre * gt)

def fn(pre, gt):
    a = pre + gt
    flag = (gt == 1) & (a == 1)
    return np.sum(flag)

def fp(pre, gt):
    a = pre + gt
    flag = (pre == 1) & (a == 1)
    return np.sum(flag)

def tn(pre, gt):
    a = pre + gt
    flag = a == 0
    return np.sum(flag)

def csi_single(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    return TP / (TP + FN + FP + eps)

def ets_single(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    N = TP + FN + FP + TN

    TPr = (TP + FP) * (TP + FN) / N

    return (TP - TPr) / (TP + FN + FP + eps - TPr)

def hss_single(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)

    n = TP + FN + FP + TN
    aref = (TP + FN) / n * (TP + FP)
    gss = (TP - aref) / (TP + FN + FP - aref + eps)
    hss = 2 * gss / (gss + 1)

    return hss

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='', type=str)
    parser.add_argument('--save', default='', type=str)
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--test_epoch', default=33, type=int)
    parser.add_argument('--load_checkpoint',action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', action='store_true')


    #data
    parser.add_argument('--input_frames', default=4, type=int)
    parser.add_argument('--output_frames', default=4, type=int)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--multi_modals', default='', type=str)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--env_frame', default=4, type=int)
    parser.add_argument('--input_transform_key', default='loge', type=str)
    parser.add_argument('--new_split', action='store_true')
    # train_batch_size

    #model
    parser.add_argument('--cond_dim', default=256, type=int)
    parser.add_argument('--timesteps', default=200, type=int)
    parser.add_argument('--loss_type', default='l2', type=str)
    parser.add_argument('--obs_channels', default=1, type=int)
    parser.add_argument('--training_steps', default=600000, type=int)

    #optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_rate", type=float, default=0.995)
    parser.add_argument("--grad_acc_step", type=int, default=2)
    parser.add_argument("--pre_milestone", type=int, default=0)

    parser.add_argument("--wandb_state", type=str, default='disabled')

    save_parser(parser.parse_args())

    return parser

def save_parser(args):
    save_path = args.save
    args_dict = vars(args)

    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path,args.log+'args.json'), 'w') as file:
        json.dump(args_dict, file, indent=4)

def colormap():
    colors = ['#fefffe', '#b2d0dd', '#97bcd6', '#799ec2', '#6ca45e',
              '#89b95e','#a7c17f','#e5f35b','#e7bd60','#e07066',
              '#e16a94','#dc5f9b','#b65dbf','#5743ec','#1c125b']

    bounds = [0, 1, 2, 3, 5,
              7, 10, 15, 20, 25,
              30,40,50,70,100,150]

    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap,norm

def diff_test_unnormalize(ds, samples_torch, data_target, data_obs_real):

    pre_diff_real = ds.un_normalize_diff(samples_torch)

    try:
        pre_abs_real = ds.un_normalize(samples_torch)
    except Exception as e:
        print("pre_abs_real failed:", repr(e))

    target_batch = ds.un_normalize(data_target)
    obs_real = ds.un_normalize(data_obs_real)

    np_samples = torch.zeros_like(data_target)
    for i in range(np_samples.shape[2]):
        np_samples[:, :, i] = torch.sum(pre_diff_real[:, :, :i + 1], dim=2) + obs_real[:, :, -1]

    np_samples_ut = ds.data_untransform(np_samples)
    target_ut = ds.data_untransform(target_batch)

    # b,c,f,h,w -> mean per frame
    drift = (np_samples_ut.mean(dim=(-1, -2))).squeeze()  # shape ~ (B, C, F) or (B, F)

    thr = 0.1  # adjust based on units (mm/hr etc.)
    frac = (np_samples_ut > thr).float().mean(dim=(-1, -2)).squeeze()

    np_samples_ut = torch.clamp(np_samples_ut, min=0)

    return target_ut, np_samples_ut