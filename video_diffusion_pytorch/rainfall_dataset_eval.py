import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import cv2
from datetime import datetime,timezone
import netCDF4 as nc
from tqdm import tqdm
import torch.nn.functional as F
# from .train import load_data_once
from einops import rearrange
import torch
import random
torch.random.seed()
np.random.seed(0)
import matplotlib.pyplot as plt
import h5py
import sys



def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds

class rainfall_data_multi(Dataset):
    def __init__(self, rainfall=None,modals=None,env_data=None,
                 multi_modal='', multi_sc = '',
                 obs_num = 4,pre_num = 2,img_size=100,input_transform_key='01',
                 data_augmentation=False,env_frame=4, 
                 ):

        self.rain_fall=rainfall
        self.env_data = env_data
        self.ERA5 = modals

        self.multi_modal = multi_modal #only list of multi modal's name
        self.multi_sc = multi_sc
        self.data_list = list(self.env_data.keys())#["train"]["id"][()]

        # self.data_path = data_path
        self.obs_num = obs_num
        self.pre_num = pre_num
        self.img_size = img_size
    
        
        self.data_augmentation = data_augmentation
        self.env_frame = env_frame #???

        assert input_transform_key in ['01','loge','sqrt'], 'check the normalization method!'
        self.input_transform_key = input_transform_key


        print('loading the multi_modal data')
        self.dataset_roots = modals


        self.dataset_normalize = {
            'sst_sf': (246.9372599, 310.29638671875),
            't2m_sf': (246.9372599, 319.1227845),
            'msl_sf': (93137.3759, 103474.4268),
            'z_200':  (114269.5519, 123746.875),
            'z_600':  (37223.24629, 44474.36399),
            'z_850':  (7864.913365, 16159.62581),
            'z_925':  (528.1218262, 9061.944304),
            't_200':  (205.4924104, 236.9169874),
            't_600':  (249.9107088, 292.5276622),
            't_850':  (265.3704041, 307.8244362),
            't_925':  (266.5474339, 313.981372),
            'q_200':  (-0.000155115, 0.000702596),
            'q_600':  (3.01590769944543E-07, 0.014554153),
            'q_850':  (9.12206086884151E-07, 0.0223369046048584),
            'q_925':  (2.28928862341937E-06, 0.024116571599734),
            'u_200':  (-50.3288295, 73.14137885),
            'u_600':  (-63.26103189, 65.79474437),
            'u_850':  (-70.79315186, 69.08036606),
            'u_925':  (-68.18152412, 64.11427186),
            'v_200':  (-68.61412644, 74.65857924),
            'v_600':  (-62.02254669, 67.49864796),
            'v_850':  (-68.00113911, 76.64611816),
            'v_925':  (-66.44315571, 68.26896275),
            'z_sf':   (-7906, 4500),

            'tp_ifs': (0, 0.06988716125488281),
            't2m_ifs': (242.53342598392996, 319.82794291554035),
            'msl_ifs': (94521.02333366396, 104166.30707048357),
            't_200_ifs': (204.02873631445442, 235.28852520565587),
            't_850_ifs': (260.0148344765363, 308.69361966130657),
            'q_200_ifs': (-4.0282640838995576e-05, 0.000435580859391594),
            'q_850_ifs': (1.5564412766243957e-06, 0.02220068623410043),
            'u_200_ifs': (-46.498877842765054, 83.27519730501616),
            'u_850_ifs': (-66.13909912109375, 53.6465625766141),
            'v_200_ifs': (-73.85488265328311, 83.86069117690691),
            'v_850_ifs': (-59.1883486895494, 62.60627350332703),

            'diff_loge':(-5.370121099,	5.3104655),
            'diff_sqrt': (-24.36050237,	20.13841541),
            # 'diff_original':(-693.875,	580.25),
            'diff_original': (-100, 100),
            'loge':(0,	6.549829553),
            'sqrt': (0, 26.42205518),
            'original': (0, 698.125),
            'tp_ifs':(0,69.88716125488281)


        }

    def _get_rain_seq(self, split: str, date_id: int):
        # 1) read from H5 -> numpy
        x = self.rain_fall[split][date_id][()]  # ndarray

        # 2) sanitize (optional but recommended)
        x = x.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) transform immediately (before any downstream logic sees it)
        k = self.input_transform_key
        if k == "01":
            return x
        elif k == "loge":
            # rainfall should be >=0; clip to avoid log of negative from bad data
            x = np.clip(x, 0.0, None)
            return np.log(x + 1.0)
        elif k == "sqrt":
            x = np.clip(x, 0.0, None)
            return np.sqrt(x)
        else:
            raise ValueError(f"Unknown input_transform_key: {k}")

    def data_transform(self):
        if self.input_transform_key == '01':
            self.rain_fall = self.rain_fall
        elif self.input_transform_key == 'loge':
            self.rain_fall = np.log(self.rain_fall+1)
        elif self.input_transform_key == 'sqrt':
            self.rain_fall = np.sqrt(self.rain_fall)

    def data_untransform(self,data):
        if isinstance(data, np.ndarray):
            if self.input_transform_key == '01':
                data_real = data
            elif self.input_transform_key == 'loge':
                data_real = np.exp(data) - 1
            elif self.input_transform_key == 'sqrt':
                data_real = data ** 2
        elif isinstance(data, torch.Tensor):
            if self.input_transform_key == '01':
                data_real = data
            elif self.input_transform_key == 'loge':
                data_real = torch.exp(data) - 1
            elif self.input_transform_key == 'sqrt':
                data_real = data ** 2
        return data_real


    def normalize(self,data):
        if self.input_transform_key == '01':
            min_data, max_data = (0,100)
        elif self.input_transform_key == 'loge':
            min_data, max_data = self.dataset_normalize['loge']
        elif self.input_transform_key == 'sqrt':
            min_data, max_data = self.dataset_normalize['sqrt']
        data_normal = (data - min_data) / (max_data - min_data)

        return data_normal

    def un_normalize(self,data):
        if self.input_transform_key == '01':
            min_data, max_data = (0,100)
        elif self.input_transform_key == 'loge':
            min_data, max_data = self.dataset_normalize['loge']
        elif self.input_transform_key == 'sqrt':
            min_data, max_data = self.dataset_normalize['sqrt']
        data_unnormal = data * (max_data - min_data) + min_data
        return data_unnormal

    def normalize_diff(self,diff):
        if self.input_transform_key == '01':
            min_diff, max_diff = self.dataset_normalize['diff_original']
        elif self.input_transform_key == 'loge':
            min_diff, max_diff = self.dataset_normalize['diff_loge']
        elif self.input_transform_key == 'sqrt':
            min_diff, max_diff = self.dataset_normalize['diff_sqrt']

        diff_normal = (diff - min_diff)/(max_diff - min_diff)
        return diff_normal

    def un_normalize_diff(self,diff):
        if self.input_transform_key == '01':
            min_diff, max_diff = self.dataset_normalize['diff_original']
        elif self.input_transform_key == 'loge':
            min_diff, max_diff = self.dataset_normalize['diff_loge']
        elif self.input_transform_key == 'sqrt':
            min_diff, max_diff = self.dataset_normalize['diff_sqrt']
        diff_real = diff * (max_diff - min_diff) + min_diff
        return diff_real


    def get_multi_modal_data(self, modals_seq, index):


        tc_modals ={}
        for key in self.multi_modal:
            modal = modals_seq[key][index]

            if len(modal.shape) == 2:
                modal = np.flipud(modal)
            elif len(modal.shape) == 3:
                modal = np.fliplr(modal)
            else:
                print('ERA5 data error!')
                exit()

            tc_modals[key] = torch.tensor(modal.copy())

        return tc_modals

    def get_env_data(self,env_scs):

        env_dict = {}
        for key in self.multi_sc:
            env_dict[key] = torch.tensor(env_scs[key])

        return env_dict


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        """Return one sample consisting of rainfall, ERA5 modals, and environmental scalars."""

        # ---- Select sequence ID ----
        date_id = self.data_list[idx]
        date_id_str = str(date_id)  # Convert to string for HDF5 access

        # ---- Load rainfall, ERA5, and scalar data ----
        rain_seq = self._get_rain_seq('val', date_id_str)
        modals_seq = {m: self.ERA5[m]['val'][date_id_str][()] for m in self.multi_modal}     

        env_seq = self.env_data[date_id] if self.env_data is not None else None
        # ---- Initialize containers ----
        rain_tensors, rain_diff_tensors = [], []
        modals_tensor = {m: [] for m in self.multi_modal}
        envs_tensor = {s: [] for s in self.multi_sc} if env_seq is not None else {}

        # ---- Iterate over each timestep ----
        for t in range(len(rain_seq)):
            # --- Rainfall ---
            rain_frame = torch.tensor(rain_seq[t], dtype=torch.float32)
            rain_frame = F.interpolate(
                rain_frame.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            rain_tensors.append(rain_frame)

            if t == 0:
                rain_diff_tensors.append(torch.zeros_like(rain_frame))
            else:
                rain_diff_tensors.append(rain_frame - rain_tensors[t - 1])

            # --- ERA5 Modals ---
            tc_modals = self.get_multi_modal_data(modals_seq, t)
            for key in self.multi_modal:
                modals_tensor[key].append(tc_modals[key])

            # --- Scalar Environment ---
            if env_seq is not None:
                env_dict = self.get_env_data(env_seq[t])
                for key in self.multi_sc:
                    envs_tensor[key].append(env_dict[key])

        # ---- Stack and normalize rainfall ----
        rain_tensor = torch.nan_to_num(torch.stack(rain_tensors).unsqueeze(-1))
        rain_tensor = self.normalize(rain_tensor)
        rain_diff_tensor = torch.nan_to_num(torch.stack(rain_diff_tensors).unsqueeze(-1))
        rain_diff_tensor = self.normalize_diff(rain_diff_tensor)

        obs_rain, pre_rain = rain_tensor[:self.obs_num], rain_tensor[self.obs_num:]
        obs_diff, pre_diff = rain_diff_tensor[:self.obs_num], rain_diff_tensor[self.obs_num:]

        # ---- Process scalar environment ----
        sc_tensor = {}
        if env_seq is not None:
            for key in envs_tensor:
                tensor = torch.stack(envs_tensor[key]).float()
                # if len(tensor.shape) == 1:
                #     tensor = tensor.unsqueeze(1)

                # sc_tensor[key] = {
                #     'obs': tensor[:self.obs_num],#.unsqueeze(1),
                #     'pre': tensor[self.obs_num:]#.unsqueeze(1)
                # }

                sc_tensor[key] = tensor

        # ---- Process ERA5 modals ----
        modals_era5 = {'obs': [], 'pre': []}
        for key in self.multi_modal:
            tensor = torch.nan_to_num(torch.stack(modals_tensor[key]).unsqueeze(-1)).float()
            min_val, max_val = self.dataset_normalize[key]
            tensor = (tensor - min_val) / (max_val - min_val)
            tensor = rearrange(tensor, 'b h w c -> b c h w')
            tensor = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)

            # modals_era5[key] = {
            #     'obs': tensor[:self.obs_num],       
            #     'pre': tensor[self.obs_num:]
            # }
            
            modals_era5['obs'].append(rearrange(tensor[:self.obs_num], 'b c h w -> c b h w'))
            modals_era5['pre'].append(rearrange(tensor[self.obs_num:], 'b c h w -> c b h w'))

        modals_era5['obs'] = torch.cat(modals_era5['obs'],dim=0)
        modals_era5['pre'] = torch.cat(modals_era5['pre'],dim=0)

        return (
            obs_rain.permute(3, 0, 1, 2),
            pre_rain.permute(3, 0, 1, 2),
            obs_diff.permute(3, 0, 1, 2),
            pre_diff.permute(3, 0, 1, 2),
            modals_era5,
            sc_tensor
        )


    def random_crop(self,tensor):
        b, c, f, h, w = tensor.shape
        crop_size = 60
        top = torch.randint(0, tensor.shape[3] - crop_size + 1, (1,)).item()
        left = torch.randint(0, tensor.shape[4] - crop_size + 1, (1,)).item()

        cropped = tensor[:, :, :, top:top + crop_size, left:left + crop_size]
        cropped_orginal = F.interpolate(cropped, size=(64, 64), mode='bilinear', align_corners=False)

        return cropped_orginal

    def collate_data(self,data_list):
        
        obs_rain, pre_rain,obs_diff,pre_diff, modals_data_obs, modals_data_pre = [], [], [], [], [], []
        
        if self.env_data is None:
            env_data = 0
        else:
            env_data = {env_name: [] for env_name in data_list[0][-1]}

        for data_one in data_list:
            obs_rain.append(data_one[0])
            pre_rain.append(data_one[1])
            obs_diff.append(data_one[2])
            pre_diff.append(data_one[3])
            modals_data_pre.append(data_one[4]['pre'])
            modals_data_obs.append(data_one[4]['obs'])
            if self.env_data is not None:
                for env_one in data_one[5]:
                    env_data[env_one].append(data_one[5][env_one])

        obs_rain = torch.stack(obs_rain, dim=0)
        pre_rain = torch.stack(pre_rain, dim=0)
        obs_diff = torch.stack(obs_diff, dim=0)
        pre_diff = torch.stack(pre_diff, dim=0)
        modals_data_pre = torch.stack(modals_data_pre, dim=0)
        modals_data_obs = torch.stack(modals_data_obs, dim=0)
        if self.env_data is not None:
            for key_env_data in env_data:
                env_data[key_env_data] = torch.stack(env_data[key_env_data], dim=0)
                if len(env_data[key_env_data].shape) == 2:
                    env_data[key_env_data] = env_data[key_env_data].unsqueeze(-1)

        if self.data_augmentation:
            data_aug_list = [obs_rain, pre_rain, obs_diff, pre_diff, modals_data_pre, modals_data_obs]
            if random.random() < 0.5:
                # 随机选择旋转角度
                rotation_angle = torch.randint(0, 4, (1,)).item() * 90
                for data_aug_i,_ in enumerate(data_aug_list):
                    data_aug_list[data_aug_i] = torch.rot90(data_aug_list[data_aug_i], k=rotation_angle // 90,dims=(-2, -1))


            if random.random() < 0.5:
                for data_aug_i,_ in enumerate(data_aug_list):
                    # data_aug_list[data_aug_i] = torch.flip(data_aug_list[data_aug_i], [-1])
                    b, c, f, h, w = data_aug_list[data_aug_i].shape
                    crop_size = 60
                    top = torch.randint(0, data_aug_list[data_aug_i].shape[-2] - crop_size + 1, (1,)).item()
                    left = torch.randint(0, data_aug_list[data_aug_i].shape[-1] - crop_size + 1, (1,)).item()

                    cropped = data_aug_list[data_aug_i][:, :, :, top:top + crop_size, left:left + crop_size]
                    cropped = rearrange(cropped,'b c f h w -> b (c f) h w')
                    cropped_hw = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
                    cropped_hw = rearrange(cropped_hw, 'b (c f) h w -> b c f h w', c=c, f=f)
                    data_aug_list[data_aug_i] = cropped_hw

            obs_rain,pre_rain,obs_diff,pre_diff,modals_data_pre, modals_data_obs = tuple(data_aug_list)

        modals_data = {'obs':modals_data_obs, 'pre':modals_data_pre}
        # x -> b c f h w
        return {'obs_rain': obs_rain, 'pre_rain': pre_rain,'obs_diff': obs_diff, 'pre_diff': pre_diff,
                'modal_env':modals_data, 'env_data':env_data,}

def _load_scalar(root_path: str):
    """
    Load scalar.npy even if it was pickled with a different NumPy module path.
    Some pickles reference numpy._core.*, which no longer exists in modern NumPy,
    so we alias it to numpy.core before loading.
    """
    scalar_path = f'{root_path}/scalar/scalar.npy'
    try:
        return np.load(scalar_path, allow_pickle=True).item()
    except ModuleNotFoundError as exc:
        if getattr(exc, 'name', '') != 'numpy._core':
            raise
        import numpy.core as np_core
        sys.modules['numpy._core'] = np_core
        sys.modules.setdefault('numpy._core.multiarray', np_core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np_core.numeric)
        return np.load(scalar_path, allow_pickle=True).item()

def load_data_once(multi_modal ='', root_path = ''):
    # data_path = 'J:\BP1backup\ICML_subset2020'
    modals = {}
    for modal in multi_modal:
        # print(f'loading {modal} data')
        if 'sf' in modal:
            modals[modal] = h5py.File(f'{root_path}/surface/{modal}.h5','r')
        else:
            modals[modal] = h5py.File(f'{root_path}/pressure/{modal}.h5','r')

    rainfall = h5py.File(f'{root_path}/MSWEP/MSWEP.h5','r')
    # imgs = MSWEP[self.type][str(index)][()]
    scalar = _load_scalar(root_path)

    print(modals.keys())
    return {'rainfall': rainfall,'modals': modals,'scalar': scalar}