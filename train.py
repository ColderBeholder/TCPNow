import argparse
from utils import *
from datetime import timedelta

from video_diffusion_pytorch.rainfall_diffusion_ultimate import Unet3D, GaussianDiffusion, Trainer
from video_diffusion_pytorch.rainfall_dataset import rainfall_data_multi,load_data_once

from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from accelerate import Accelerator

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    file_path = './modal_txt/modal.txt'
    multi_sc = ['intensity', 'month', 'lon', 'lat', 'wind', 'move_velocity'] # should call .txt
    multi_modal = []
    modal_channle = 0 
    ifs_channle = 0
    ifs_out_channel=3

    channel_nums = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            for M in str.split(line,' '):
                multi_modal.append(M)
                channel_nums += 1


    unet = Unet3D(
        dim = 64,
        channels = 2 + channel_nums + 1, #(data_obs_diff, data_obs_real) + ERA5_obs + img 
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        cond_dim = args.cond_dim,
        ifs_channels = channel_nums,
        multi_sc = multi_sc,
    ).cuda()
    # x -> b c f h w

    diffusion = GaussianDiffusion(
        unet,
        image_size = args.img_size,
        num_frames = args.output_frames+args.input_frames,
        input_frames = args.input_frames,
        output_frames = args.output_frames,
        channels = 1,
        timesteps = args.timesteps,   # number of steps
        loss_type = args.loss_type,    # L1 or L2
        obs_channels = args.obs_channels,
        use_wavelet_domain = True,
        wavelet_detail_weight = 2.0,
    ).cuda()

    Alldata = load_data_once(multi_modal, root_path = './dataset')
    rainfall = Alldata['rainfall']
    modals = Alldata['modals']
    env_sc = Alldata['scalar']
    ds = rainfall_data_multi(rainfall=rainfall, modals=modals,env_data=env_sc, multi_modal=multi_modal, 
                                    multi_sc=multi_sc,
                                    img_size=args.img_size, pre_num=args.output_frames,
                                    input_transform_key=args.input_transform_key,data_augmentation=False)

    project_config = ProjectConfiguration(
        project_dir = './Exps',
        logging_dir='./Exps/log'
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    set_seed(1)
    accelerator = Accelerator(
        project_config  =   project_config,
        kwargs_handlers =   [ddp_kwargs, process_kwargs],
        mixed_precision =   args.mixed_precision,
        log_with        =   'wandb'
    )



    # Config log tracker 'wandb' from accelerate
    accelerator.init_trackers(
        project_name='training',
        init_kwargs={"wandb":{"mode": args.wandb_state,}}
    )

    trainer = Trainer(
        diffusion,
        train_batch_size = args.train_batch_size,
        train_lr = args.lr,
        save_and_sample_every = 5000,
        train_num_steps = args.training_steps,
        gradient_accumulate_every = args.grad_acc_step,
        ema_decay = args.ema_rate,
        amp = False,
        results_folder = args.save,
        dataset_train = ds,
        use_accelerate = True,
        accelerator = accelerator,
        pre_milestone = args.pre_milestone, 
    )
    trainer.train()
