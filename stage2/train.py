'''
1. CycleGAN + WDDGAN
2. errG = errG_adv + errG_GT + errG_CR
   errG_GT = l1_loss
   errG_CR = pixel-wise (CR: contrastive regularization <= VGG) "before iwt"
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.distributed as dist

import shutil
import argparse
import time
import numpy as np
import pandas as pd
import random
import sys
from timm.utils import AverageMeter
from tqdm import tqdm
from glob import glob
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

from diffusion import sample_from_model, sample_posterior, \
    q_sample_pairs, get_time_schedule, \
    Posterior_Coefficients, Diffusion_Coefficients
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse

from utils.dataloader import get_dataloaders
from score_sde.models.discriminator import Discriminator_large, Discriminator_small
from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp
from utils.EMA import EMA
from utils.metrics import calculate_train_metrics, calculate_metrics
from utils.CR import ContrastLoss, WaveletContrastLoss
from utils.PatchNCELoss import cal_PatchNCELoss

device = torch.device('cuda')

def to_range_0_1(img):
    min_val = img.min()
    max_val = img.max()
    img_nor = (img - min_val)/(max_val - min_val)
    
    return img_nor

def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()


# %% test
def test(train_clean_path, test_noisy_path, test_dataloader, save_path, epoch, args):
    
    batch_size = args.batch_size
    exp_path = args.exp
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    
    # Generator-diffusion
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))
    
    netG = gen_net(args).to(device)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)  
    
    checkpoint = os.path.join(exp_path, 'weights', 'netG_{}.pth'.format(epoch))
    ckpt = torch.load(checkpoint, map_location=device) 
    
    # # loading weights from ddp in single gpu
    # for key in list(ckpt.keys()):
    #     ckpt[key[7:]] = ckpt.pop(key)
    
    netG.load_state_dict(ckpt, strict=False) 
    netG.eval()
    
    # Wavelet Pooling
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
        iwt = DWTInverse(mode='zero', wave='haar').to(device)
    
    # save results
    test_save_path = os.path.join(exp_path, save_path)
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(os.path.join(test_save_path, 'visualize', 'epoch_{}').format(epoch), exist_ok=True)
    os.makedirs(os.path.join(test_save_path, 'result', 'epoch_{}').format(epoch), exist_ok=True)
                
    for iteration, (x, img_name) in tqdm(enumerate(test_dataloader), leave=False):
        netG.zero_grad()
        # sample from p(x_0)
        x0 = x.to(device, non_blocking=True)
        
        if not args.use_pytorch_wavelet:
            xll, xlh, xhl, xhh = dwt(x0)
        else:
            xll, xh = dwt(x0)  # [b, 1, h, w], [b, 1, 3, h, w]
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
        
        x = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [b, 4, h, w]
        x = x / 2.0
        
        assert -1 <= x.min() < 0
        assert 0 < x.max() <=1
        
        # sample t
        x_tp1 = torch.randn_like(x)
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_tp1, x, args)
        
        if not args.use_pytorch_wavelet:
            fake_sample0 = iwt(fake_sample[:, 0].unsqueeze(1), fake_sample[:, 1].unsqueeze(1), fake_sample[:, 2].unsqueeze(1), fake_sample[:, 3].unsqueeze(1))
        else:
            fake_sample0 = iwt((fake_sample[:, 0].unsqueeze(1), [torch.stack((fake_sample[:, 1].unsqueeze(1), fake_sample[:, 2].unsqueeze(1), fake_sample[:, 3].unsqueeze(1)), dim=2)]))
        
        # generated images
        for item in range(fake_sample0.shape[0]):
            x_img = x0[item]
            fake_img = fake_sample0[item]
            x_img = to_range_0_1(x_img)
            fake_img = to_range_0_1(fake_img)
            torchvision.utils.save_image(torch.cat((x_img, fake_img), dim=2), os.path.join(test_save_path, 'visualize', 'epoch_{}', img_name[item]).format(epoch), normalize=True)
            torchvision.utils.save_image(fake_img, os.path.join(test_save_path, 'result', 'epoch_{}', img_name[item]).format(epoch), normalize=True)
    # calculate metrics
    fake_clean_path= os.path.join(test_save_path, 'result', 'epoch_{}').format(epoch)
    
    MI, CSSIM, SSIM, PSNR, FID = calculate_metrics(train_clean_path, test_noisy_path, fake_clean_path)

    test_results = {
        'Epoch': epoch,
        'MI': MI,
        'CSSIM': CSSIM,
        'SSIM': SSIM,
        'PSNR': PSNR,
        'FID': FID
    }
    
    return test_results


# %% train
def train(
    train_clean_path,
    train_noisy_path,
    test_noisy_path,
    args
):

    batch_size = args.batch_size
    nz = args.nz  # latent dimension
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    
    # DataLoader
    dataloaders = get_dataloaders(
        train_clean_path,
        train_noisy_path,
        test_noisy_path,
        args
    )
    
    # model
    exp_path = args.exp
    os.makedirs(exp_path, exist_ok=True)
    samples = os.path.join(exp_path, 'samples')
    os.makedirs(samples, exist_ok=True)
    weights = os.path.join(exp_path, 'weights')
    os.makedirs(weights, exist_ok=True) 
    
    # Generator and Discriminator
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = [Discriminator_small, Discriminator_large]
    print("GEN: {}, DISC: {}".format(gen_net, disc_net))
    netG = gen_net(args).to(device)
    netD = disc_net[1](nc=2 * args.num_channels, ngf=args.ngf,
                        t_emb_dim=args.t_emb_dim,
                        act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
    netCR = WaveletContrastLoss(device=device)

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # Wavelet Pooling
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
        iwt = DWTInverse(mode='zero', wave='haar').to(device)

    # ddp
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume or os.path.exists(os.path.join(weights, 'content.pth')):
        checkpoint_file = os.path.join(weights, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # ============ train ============
    train_results = pd.DataFrame(columns = ['Epoch', 'G_loss', 'G_adv', 'G_GT', 'G_perc', 'D_loss', 'D_real', 'D_fake', 'MI', 'CSSIM', 'SSIM', 'PSNR'])
    test_results = pd.DataFrame(columns = ['Epoch', 'MI', 'CSSIM', 'SSIM', 'PSNR', 'FID'])
    
    for epoch in range(init_epoch, args.num_epoch):

        for iteration, (x, y, img_name) in tqdm(enumerate(dataloaders['train']), leave=False):
            for p in netD.parameters():
                p.requires_grad = True
            optimizerD.zero_grad()

            for p in netG.parameters():
                p.requires_grad = False
            
            train_meters = {
                'G_loss': AverageMeter(),
                'G_adv': AverageMeter(),
                'G_GT': AverageMeter(),
                'G_perc': AverageMeter(),
                'D_loss': AverageMeter(),
                'D_real': AverageMeter(),
                'D_fake': AverageMeter(),
                'MI': AverageMeter(),
                'CSSIM': AverageMeter(),
                'SSIM': AverageMeter(),
                'PSNR': AverageMeter()
            }

            # sample from p(x_0)
            x0 = x.to(device, non_blocking=True) # clean image
            y0 = y.to(device, non_blocking=True) # noisy image

            if not args.use_pytorch_wavelet:
                xll, xlh, xhl, xhh = dwt(x0)
                yll, ylh, yhl, yhh = dwt(y0)
            else:
                xll, xh = dwt(x0)  # [b, 1, h, w], [b, 1, 3, h, w]
                xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
                yll, yh = dwt(y0)  # [b, 1, h, w], [b, 1, 3, h, w]
                ylh, yhl, yhh = torch.unbind(yh[0], dim=2)

            real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [b, 4, h, w]
            noisy_data = torch.cat([yll, ylh, yhl, yhh], dim=1)  # [b, 4, h, w]

            # normalize real_data
            real_data = real_data / 2.0  # [-1, 1]
            noisy_data = noisy_data / 2.0

            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1

            #### D part
            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            
            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_penalty_call(args, D_real, x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_penalty_call(args, D_real, x_t)

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            
            x_0_predict = netG(torch.cat((x_tp1.detach(), noisy_data.detach()), axis=1), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            
            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            
            # Update D
            optimizerD.step()

            #### G part
            # update G
            for p in netD.parameters():
                p.requires_grad = False

            for p in netG.parameters():
                p.requires_grad = True
            optimizerG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(batch_size, nz, device=device)
            
            x_0_predict = netG(torch.cat((x_tp1.detach(), noisy_data.detach()), axis=1), t, latent_z)
            
            if not args.use_pytorch_wavelet:
                x_0_predict0 = iwt(x_0_predict[:, 0].unsqueeze(1), x_0_predict[:, 1].unsqueeze(1), x_0_predict[:, 2].unsqueeze(1), x_0_predict[:, 3].unsqueeze(1))
            else:
                x_0_predict0 = iwt((x_0_predict[:, 0].unsqueeze(1), [torch.stack((x_0_predict[:, 1].unsqueeze(1), x_0_predict[:, 2].unsqueeze(1), x_0_predict[:, 3].unsqueeze(1)), dim=2)]))
            
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            
            errG_adv = F.softplus(-output)
            
            errG_GT = F.l1_loss(x_0_predict, real_data)
            errG_perc = netCR(x_0_predict, real_data, noisy_data)
            
            errG_adv = errG_adv.mean()
            errG_GT = errG_GT.mean()
            errG_perc = 0.05 * errG_perc.mean()
            
            errG = errG_adv + errG_GT + errG_perc

            errG.backward()
            optimizerG.step()

            global_step += 1
            
            MI, CSSIM, SSIM, PSNR = calculate_train_metrics(x, x_0_predict0)
            train_meters['G_loss'].update(errG.item(), n=1)
            train_meters['G_adv'].update(errG_adv.item(), n=1)
            train_meters['G_GT'].update(errG_GT.item(), n=1)
            train_meters['G_perc'].update(errG_perc.item(), n=1)
            train_meters['D_loss'].update(errD.item(), n=1)
            train_meters['D_real'].update(errD_real.item(), n=1)
            train_meters['D_fake'].update(errD_fake.item(), n=1)
            train_meters['MI'].update(MI, n=1)
            train_meters['CSSIM'].update(CSSIM, n=1)
            train_meters['SSIM'].update(SSIM, n=1)
            train_meters['PSNR'].update(PSNR, n=1)
            
            if iteration % 100 == 0:
                print('epoch {} iteration{}, G Loss: {:.3f}, G_adv: {:.3f}, G_GT: {:.3f}, G_perc: {:.3f}, D Loss: {:.3f}, D_real: {:.3f}, D_fake: {:.3f}'.format( \
                    epoch+1, iteration, errG.item(), errG_adv.item(), errG_GT.item(), errG_perc.item(), errD.item(), errD_real.item(), errD_fake.item()))

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()
        
        print('epoch {}, G Loss: {}, D Loss: {}'.format(epoch+1, train_meters['G_loss'].avg, train_meters['D_loss'].avg))

        # save results
        epoch_results = {
            'Epoch': epoch + 1,
            'G_loss': train_meters['G_loss'].avg,
            'G_adv': train_meters['G_adv'].avg,
            'G_GT': train_meters['G_GT'].avg,
            'G_perc': train_meters['G_perc'].avg,
            'D_loss': train_meters['D_loss'].avg,
            'D_real': train_meters['D_real'].avg,
            'D_fake': train_meters['D_fake'].avg,
            'MI': train_meters['MI'].avg,
            'CSSIM': train_meters['CSSIM'].avg,
            'SSIM': train_meters['SSIM'].avg,
            'PSNR': train_meters['PSNR'].avg
        }
        epoch_results = pd.DataFrame([epoch_results])
        train_results = pd.concat([train_results, epoch_results], ignore_index=True)
        train_results.to_excel(os.path.join(exp_path, 'train_results.xlsx'), index=False, engine='openpyxl')
        

        if (epoch+1) % 1 == 0:

            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, noisy_data, args)

            fake_sample *= 2
            if not args.use_pytorch_wavelet:
                fake_sample = iwt(fake_sample[:, 0].unsqueeze(1), fake_sample[:, 1].unsqueeze(1), fake_sample[:, 2].unsqueeze(1), fake_sample[:, 3].unsqueeze(1))
            else:
                fake_sample = iwt((fake_sample[:, 0].unsqueeze(1), [torch.stack((fake_sample[:, 1].unsqueeze(1), fake_sample[:, 2].unsqueeze(1), fake_sample[:, 3].unsqueeze(1)), dim=2)]))
            
            fake_sample = to_range_0_1(fake_sample)
        
            sample_image = torch.cat((x0, y0, fake_sample), dim=0)

            torchvision.utils.save_image(sample_image, os.path.join(samples, 'sample_discrete_epoch_{}.png'.format(epoch+1)), normalize=False)

        if (epoch+1) % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

            torch.save(netG.state_dict(), os.path.join(weights, 'netG_{}.pth'.format(epoch+1)))
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
        
        if (epoch+1) % args.save_content_every == 0:
            print('Saving content.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                        'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                        'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                        'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            torch.save(content, os.path.join(weights, 'content.pth'))
        
        ############ test ############
        if (epoch+1) % args.save_ckpt_every == 0:
            test_epoch_results = test(train_clean_path, test_noisy_path, dataloaders['test'], 'test_results', epoch+1, args)
            test_epoch_results = pd.DataFrame([test_epoch_results])
            test_results = pd.concat([test_results, test_epoch_results], ignore_index=True)
            test_results.to_excel(os.path.join(exp_path, 'test_results.xlsx'), index=False, engine='openpyxl')

        
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=256,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=4,
                        help='channel of wavelet subbands')
    parser.add_argument('--num_input_channels', type=int, default=8,
                        help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 2, 2],
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # generator and training
    parser.add_argument('--exp', default='Results/EchoWDiff-GT_0.05CR-bs4', help='name of experiment')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float,
                        default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float,
                        default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float,
                        default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    # wavelet GAN
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--net_type", default="wavelet") # normal or wavelet
    parser.add_argument("--num_disc_layers", default=6, type=int)
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10,
                        help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int,
                        default=5, help='save ckpt every x epochs') 

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6002',
                        help='port for master')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers')

    args = parser.parse_args()

    # dataset
    train_clean_path = '/storage/ImageDenoising/Dataset/cardiacUDC_Site_G/train/clean'
    train_noisy_path = '/storage/ImageDenoising/Dataset/cardiacUDC_Site_G/train/clean_noisy'
    test_noisy_path = '/storage/ImageDenoising/Dataset/cardiacUDC_Site_G/test/noisy'
    
    train(train_clean_path, train_noisy_path, test_noisy_path, args)