# EchoWDiff
Stage 1: Unpaired image-to-image translation using CycleGAN for generating clean-hazy training pairs for diffusion models
python stage1/train.py
Stage 2: Wavelet-domain diffusion model with near-field loss for echocardiography dehazing
python stage2/train.py
