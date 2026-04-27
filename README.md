# TCPNow

Code implementation for my Master's thesis on tropical cyclone precipitation nowcasting with diffusion models and multimodal atmospheric inputs.

## Overview

TCPNow trains and evaluates video diffusion models for short-term rainfall prediction. The model conditions on recent rainfall observations, rainfall differences, ERA5/IFS-style gridded meteorological variables, and scalar storm/environment features.

The main workflow is:

1. Prepare rainfall, modal atmospheric fields, and scalar features in the expected HDF5/NumPy layout.
2. Configure the selected modal variables in `modal_txt/modal.txt` and scalar variables in `modal_txt/sc.txt`.
3. Train with `train.py` using Hugging Face Accelerate.
4. Evaluate checkpoints with one of the evaluation scripts and save metrics, arrays, and visualizations.

## Repository Structure

```text
.
├── train.py                         # Main training entry point
├── evaluation.py                    # Wavelet-domain evaluation entry point
├── evaluation_DiG.py                # DiG/GLA evaluation entry point
├── evaluation_simple.py             # Simple baseline evaluation entry point
├── evaluation_wavelet.py            # Wavelet evaluation variant
├── run_evaluation.sh                # Helper script for checkpoint evaluation
├── config.yaml                      # Accelerate multi-GPU config
├── training_script                  # Example SLURM training script
├── utils.py                         # Metrics, argument parser, plotting colormap
├── video_diffusion_pytorch/          # Model, attention, dataset, and transformer code
├── rotary_embedding_torch/           # Local rotary embedding implementation
└── environment.yaml                 # Conda environment snapshot
```

## Requirements

The project is designed for Python 3.8 with CUDA-capable GPUs. Create the Conda environment with:

```bash
conda env create -f environment.yaml
conda activate envir
```

If your environment does not already include the core deep-learning packages, install them as needed:

```bash
pip install torch accelerate einops matplotlib h5py netCDF4 opencv-python tqdm wandb
```

## Data Layout

Training expects data under `./dataset`. Evaluation expects validation data under `./data_val`.

The loader expects this structure:

```text
dataset/
├── MSWEP/
│   └── MSWEP.h5
├── surface/
│   └── <surface_modal>.h5
├── pressure/
│   └── <pressure_modal>.h5
└── scalar/
    └── scalar.npy

data_val/
├── MSWEP/
│   └── MSWEP.h5
├── surface/
│   └── <surface_modal>.h5
├── pressure/
│   └── <pressure_modal>.h5
└── scalar/
    └── scalar.npy
```

Modal and scalar feature lists are read from:

```text
modal_txt/modal.txt
modal_txt/sc.txt
```

These data directories are intentionally not tracked by git because they are large experiment artifacts.

## Training

Run the main training script with Accelerate:

```bash
accelerate launch --config_file config.yaml --main_process_port 29501 train.py \
  --log train \
  --save results/TCPNow \
  --train_batch_size 16 \
  --training_steps 600000 \
  --input_frames 4 \
  --output_frames 4 \
  --img_size 64 \
  --input_transform_key loge
```

For cluster jobs, adapt the included `training_script` SLURM file to your paths, partition, and GPU allocation.

## Evaluation

Evaluate a trained checkpoint by passing the checkpoint directory and epoch:

```bash
python evaluation.py \
  --save results/TCPNow \
  --test_epoch 100 \
  --pre_milestone 100 \
  --train_batch_size 16
```

Or use the helper script:

```bash
bash run_evaluation.sh TCPNow 100
```

Evaluation writes prediction arrays, ground-truth arrays, visualizations, and metrics to the selected `--save` directory. Reported metrics include CSI, HSS, ETS, MSE, and MAE.

## Common Arguments

```text
--save                 Output/checkpoint directory
--test_epoch           Checkpoint epoch used during evaluation
--input_frames         Number of observed frames
--output_frames        Number of predicted frames
--img_size             Spatial resolution used by the model
--train_batch_size     Batch size
--timesteps            Diffusion timesteps
--loss_type            Diffusion loss type, such as l1 or l2
--input_transform_key  Rainfall transform: 01, loge, or sqrt
--wandb_state          WandB mode, such as disabled or online
```

## Notes

- Keep datasets, checkpoints, generated figures, and experiment outputs out of git.
- `config.yaml` is currently configured for local multi-GPU training with 2 processes and fp16 mixed precision.
- Some evaluation scripts are model-variant entry points. Make sure the matching model module and checkpoint are available before running a variant.
