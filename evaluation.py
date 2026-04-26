import torch
from torch.utils.data import DataLoader

from video_diffusion_pytorch.rainfall_diffusion_wavelet import Unet3D, GaussianDiffusion, Trainer
from video_diffusion_pytorch.rainfall_dataset_eval import rainfall_data_multi,load_data_once
from utils import *

import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

parser = get_parser()
args = parser.parse_args()

file_path = './modal_txt/modal.txt'
sc_file_path = './modal_txt/sc.txt'
env_list = []
multi_modal = []
modal_channle = 0 
ifs_channle = 0
ifs_out_channel=3

with open(sc_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        env_list.extend(str.split(line, ' '))

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        for M in str.split(line,' '):
            if 'ifs' in M:
                ifs_channle = ifs_channle+1
            else:
                modal_channle = modal_channle+1
            multi_modal.append(M)

channel_nums = modal_channle + ifs_channle*ifs_out_channel

unet = Unet3D(
    dim=64,
    channels=2+channel_nums+1, #(data_obs_diff, data_obs_real) + ERA5_obs + img 
    out_dim=1,
    dim_mults=(1, 2, 4, 8),
    cond_dim=args.cond_dim,
    ifs_channels=channel_nums,
    multi_sc = env_list,
).cuda()
# x -> b c f h w


diffusion = GaussianDiffusion(
    unet,
    image_size = args.img_size,
    num_frames = args.output_frames+args.input_frames,
    input_frames=args.input_frames,
    output_frames=args.output_frames,
    channels =1,
    timesteps = args.timesteps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    obs_channels = args.obs_channels,
    use_wavelet_domain = True,
    wavelet_detail_weight = 2.0,
).cuda()

Alldata = load_data_once(multi_modal, root_path = './data_val')
rainfall = Alldata['rainfall']
modals = Alldata['modals']
env_data = Alldata['scalar']

ds = rainfall_data_multi(
    rainfall=rainfall,
    modals=modals,
    env_data=env_data,
    multi_modal=multi_modal, 
    multi_sc=env_list,
    img_size=args.img_size,
    pre_num=args.output_frames,
    input_transform_key=args.input_transform_key,
    data_augmentation=False,
)

trainer = Trainer(
    diffusion,
    train_batch_size = args.train_batch_size,
    train_num_steps = 600000,
    gradient_accumulate_every = args.grad_acc_step,
    ema_decay = args.ema_rate,
    amp = False,
    results_folder = args.save,
    dataset_train = ds,
    use_accelerate = False,
    pre_milestone = args.pre_milestone, 
)

def save_sample(root_path,data_p,data_t,count_i,epoch):
    cmap, norm = colormap()
    save_path = os.path.join(root_path,epoch_name+str(epoch),'predictions_'+str(epoch))
    os.makedirs(save_path,exist_ok=True)
    batch_shape = data_p.shape
    batch = batch_shape[0]
    if len(batch_shape) == 4:
        c = batch_shape[1]
    else:
        c = 1
    for batch_i in range(batch):
        for c_i in range(c):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            if len(batch_shape) == 4:
                img_p = data_p[batch_i, c_i]
                img_t = data_t[batch_i, c_i]
            else:
                img_p = data_p[batch_i]
                img_t = data_t[batch_i]

            img1 = ax1.imshow(img_t, cmap=cmap, norm=norm)
            ax1.set_title('groud truth')
            plt.colorbar(img1, ax=ax1)

            img2 = ax2.imshow(img_p, cmap=cmap, norm=norm)
            plt.colorbar(img2, ax=ax2)
            ax2.set_title('prediction')

            plt.savefig(save_path + '/' + str(batch_i+count_i)+'_'+str(c_i)+'.png')
            plt.close()

def save_visualize_tensor(root_path, tensor_data, folder_name, count_i, epoch):
    """
    Save a 5D tensor [B, C, F, H, W] as per-frame PNGs in a dedicated folder.
    """
    save_path = os.path.join(root_path, epoch_name + str(epoch), 'visualize_' + str(epoch), folder_name)
    os.makedirs(save_path, exist_ok=True)

    global_min = tensor_data.min().item()
    global_max = tensor_data.max().item()
    global_mean = tensor_data.mean().item()

    tensor_np = tensor_data.detach().cpu().numpy()
    if tensor_np.ndim == 4:
        tensor_np = tensor_np[:, np.newaxis, ...]

    batch_size, channels, frames, _, _ = tensor_np.shape
    for b_idx in range(batch_size):
        for c_idx in range(channels):
            for f_idx in range(frames):
                file_name = f'b{b_idx + count_i:05d}_c{c_idx:03d}_f{f_idx:03d}.png'
                plt.imsave(
                    os.path.join(save_path, file_name),
                    tensor_np[b_idx, c_idx, f_idx],
                    cmap='gray'
                )

def save_result(CSI, HSS,ETS,preds,target,mse,mae,threshold,step,epoch):
    print('Saving some results on step ->'+step)
    ps = np.concatenate(preds, axis=0)

    ts = np.concatenate(target, axis=0)

    np.save(os.path.join(args.save, epoch_name + str(epoch), 'prediction_all.npy'),ps)
    np.save(os.path.join(args.save, epoch_name + str(epoch), 'gt_all.npy'), ts)

    for i in range(3):
        CSI[i] = np.array(CSI[i]).mean()
        HSS[i] = np.array(HSS[i]).mean()
        ETS[i] = np.array(ETS[i]).mean()
    mse = np.array(mse).mean()
    mae = np.array(mae).mean()
    f = open(os.path.join(args.save, epoch_name + str(epoch), 'result_' + str(epoch) + '.txt'), 'a+')

    f.write('CSI: ')
    print('CSI: ')
    for i in range(len(threshold)):
        f.write('r >= ' + str(threshold[i]) + ':' + str(CSI[i]) + ' ')
        print('r >=', threshold[i], ':', CSI[i], end=' ')
    print()

    f.write('HSS:')
    print('HSS:')
    for i in range(len(threshold)):
        f.write('r >= ' + str(threshold[i]) + ':' + str(HSS[i]) + ' ')
        print('r >=', threshold[i], ':', HSS[i], end=' ')
    print()

    f.write('ETS:')
    print('ETS:')
    for i in range(len(threshold)):
        f.write('r >= ' + str(threshold[i]) + ':' + str(ETS[i]) + ' ')
        print('r >=', threshold[i], ':', ETS[i], end=' ')
    print()

    f.write('MSE:' + str(mse) + 'MAE:' + str(mae))
    print('MSE:', mse, 'MAE:', mae)


def sample_test(epoch=100,pre_len=1,output_dirpath=''):

    preds=[]
    target=[]

    epoch = trainer.load(epoch)
    save_path = os.path.join(output_dirpath, epoch_name + str(epoch), 'predictions_' + str(epoch))
    os.makedirs(save_path, exist_ok=True)

    dl = DataLoader(ds, batch_size = args.train_batch_size, shuffle=False, pin_memory=True, collate_fn=ds.collate_data)
    CSI, HSS,ETS, mse, mae = [], [], [], [],[]
    for i in range(3):
        CSI.append([])
        HSS.append([])
        ETS.append([])
    threshold = [6, 24, 60]
    dl_t = tqdm(dl, desc='Inference')
    obs_min, obs_max = float('inf'), float('-inf')
    target_min, target_max = float('inf'), float('-inf')

    for i,batch in enumerate(dl_t):
        data_obs_real = batch['obs_rain'].cuda()
        data_traget = batch['pre_rain'].cuda()
        obs_min = min(obs_min, data_obs_real.min().item())
        obs_max = max(obs_max, data_obs_real.max().item())
        target_min = min(target_min, data_traget.min().item())
        target_max = max(target_max, data_traget.max().item())
        data_obs_diff = batch['obs_diff'].cuda()
        data_traget_diff = batch['pre_diff'].cuda()

        modal_env = batch['modal_env']
        ERA5_obs = modal_env['obs'].cuda()
        ERA5_pre = modal_env['pre'].cuda()

        Env_obs = batch['env_data']
        for key in Env_obs:
            Env_obs[key] = Env_obs[key].cuda()

        data_obs = torch.cat([data_obs_diff, data_obs_real, ERA5_obs, ERA5_pre], dim=1)


        data_condition = Env_obs

        samples_torch = trainer.model.sample(
            batch_size=args.train_batch_size,
            data_condition=data_condition,
            obs_data=data_obs
        )

        target_batch,np_samples = diff_test_unnormalize(
            ds=ds,
            samples_torch=samples_torch,
            data_target=data_traget,
            data_obs_real=data_obs_real
        )

        count_i = args.train_batch_size*i

        if i < 10:
            save_visualize_tensor(
                output_dirpath,
                np_samples,
                'predicted_rainfall_outputs',
                count_i,
                epoch
            )

        target_batch = target_batch.squeeze().cpu().numpy()
        np_samples = np_samples.squeeze().cpu().numpy()

        if i < 10:
            save_visualize_tensor(
                output_dirpath,
                data_obs_real,
                'observed_rainfall_inputs',
                count_i,
                epoch
            )
            save_visualize_tensor(
                output_dirpath,
                data_traget,
                'target_rainfall_ground_truth',
                count_i,
                epoch
            )
            
            save_sample(output_dirpath, np_samples, target_batch, count_i,epoch)

        mse.append(mse_evaluation(target_batch, np_samples))
        mae.append(mae_evaluation(target_batch, np_samples))

        for t in range(3):
            a = np_samples.copy()
            b = target_batch.copy()
            thre = threshold[t]
            a[a < thre] = 0
            a[a >= thre] = 1
            b[b < thre] = 0
            b[b >= thre] = 1
            CSI[t].append(csi_single(a, b))
            HSS[t].append(hss_single(a, b))
            ETS[t].append(ets_single(a, b))


        preds.append(np_samples)
        target.append(target_batch)

        if i % 30 == 29:
            save_result(CSI.copy(), HSS.copy(), ETS.copy(),
                        preds.copy(), target.copy(),
                        mse.copy(), mae.copy(), threshold.copy(),step=str(i),epoch=epoch)


    save_result(CSI.copy(), HSS.copy(), ETS.copy(),
                preds.copy(), target.copy(),
                mse.copy(), mae.copy(), threshold.copy(),step='last',epoch=epoch)

# Run inference and store outputs

epoch_name = 'test'
epoch = args.test_epoch
sample_test(epoch=epoch,pre_len=args.output_frames,output_dirpath=args.save)
