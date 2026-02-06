# EchoWDiff
#### Stage 1: Unpaired image-to-image translation using CycleGAN for generating clean-hazy training pairs for diffusion models
```bash
python EchoWDiff/stage1/train.py
```
#### Stage 2: Wavelet-domain diffusion model with near-field loss for echocardiography dehazing
```bash
python EchoWDiff/stage2/train.py
```
