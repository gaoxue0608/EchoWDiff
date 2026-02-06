# 1. generate clean images and calculate metrics (test: noisy)
# 2. generate noisy images and calculate metrics (test: clean)
import time
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from tqdm import tqdm
import wandb
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from glob import glob
import numpy as np
import pandas as pd
from cal_metrics import calculate_metrics


class ImageDataset_test(Dataset):
    def __init__(self, img_path, transform):
        self.img_paths = glob(img_path + '/*.png')
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('L')
        img_name = os.path.basename(img_path)
        if self.transform:
            image = self.transform(image)
            
        return {'A': image, 'A_paths': img_name}


def load_test_datasets(
    test_path
):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    
    test_dataset = ImageDataset_test(test_path, transform)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=1
    )
    
    return test_dataloader


def tensor2image(image_tensor):
    img = image_tensor[0]
    min_val = img.min()
    max_val = img.max()
    img_nor = (img - min_val)/(max_val - min_val)
    
    return img_nor
    

if __name__ == '__main__':
    
    opt = TestOptions().parse() 
    opt.num_threads = 0  
    opt.batch_size = 1    
    opt.serial_batches = True  
    # modify according to requirements
    opt.name = 'CycleGAN-bs8-2'
    
    test_noisy_results = pd.DataFrame(columns = ['Epoch', 'MI', 'CSSIM', 'SSIM', 'PSNR', 'FID'])
    test_clean_results = pd.DataFrame(columns = ['Epoch', 'MI', 'CSSIM', 'SSIM', 'PSNR', 'FID'])
    for epoch in range(5, 101, 5):
        
        opt.epoch = epoch
        
        #### A2B: noisy -> clean
        opt.model_suffix = '_A'
        test_noisy_path = os.path.join(opt.dataroot, 'test', 'noisy')
        test_dataset = load_test_datasets(test_noisy_path)
        
        result_path = os.path.join(opt.results_dir, opt.name, 'noisy2clean', 'epoch_{}'.format(opt.epoch))
        fake_result_path = os.path.join(result_path, 'fake')
        contrast_result_path = os.path.join(result_path, 'contrast')
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(fake_result_path, exist_ok=True)
        os.makedirs(contrast_result_path, exist_ok=True)
        
        model = create_model(opt)
        model.setup(opt)
        
        if opt.eval:
            model.eval()
        
        for i, data in tqdm(enumerate(test_dataset), leave=False):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            real_noisy = tensor2image(visuals['real'])
            fake_clean = tensor2image(visuals['fake'])
            torchvision.utils.save_image(fake_clean, os.path.join(fake_result_path, img_path), normalize=True)
            torchvision.utils.save_image(torch.cat((real_noisy, fake_clean), dim=2), os.path.join(contrast_result_path, img_path), normalize=True)
        
        MI, CSSIM, SSIM, PSNR, FID = calculate_metrics(
            train_B_path = os.path.join(opt.dataroot, 'train', 'clean'),
            test_A_path = os.path.join(opt.dataroot, 'test', 'noisy'),
            generated_A_path = fake_result_path
        )
        test_noisy_epoch_results = {
            'Epoch': epoch,
            'MI': MI,
            'CSSIM': CSSIM,
            'SSIM': SSIM,
            'PSNR': PSNR,
            'FID': FID,
        }
        test_noisy_epoch_results = pd.DataFrame([test_noisy_epoch_results])
        test_noisy_results = pd.concat([test_noisy_results, test_noisy_epoch_results], ignore_index=True)
        test_noisy_results.to_excel(os.path.join(opt.results_dir, opt.name, 'noisy2clean', 'noisy2clean_results.xlsx'), index=False, engine='openpyxl')
        
        #### B2A: clean -> noisy
        opt.model_suffix = '_B'
        test_clean_path = os.path.join(opt.dataroot, 'test', 'clean')
        test_dataset = load_test_datasets(test_clean_path)
        
        result_path = os.path.join(opt.results_dir, opt.name, 'clean2noisy', 'epoch_{}'.format(opt.epoch))
        fake_result_path = os.path.join(result_path, 'fake')
        contrast_result_path = os.path.join(result_path, 'contrast')
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(fake_result_path, exist_ok=True)
        os.makedirs(contrast_result_path, exist_ok=True)
        
        model = create_model(opt)
        model.setup(opt)
        
        if opt.eval:
            model.eval()
        
        for i, data in tqdm(enumerate(test_dataset), leave=False):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            real_clean = tensor2image(visuals['real'])
            fake_noisy = tensor2image(visuals['fake'])
            torchvision.utils.save_image(fake_noisy, os.path.join(fake_result_path, img_path), normalize=True)
            torchvision.utils.save_image(torch.cat((real_clean, fake_noisy), dim=2), os.path.join(contrast_result_path, img_path), normalize=True)       
        
        MI, CSSIM, SSIM, PSNR, FID = calculate_metrics(
            train_B_path = os.path.join(opt.dataroot, 'train', 'noisy'),
            test_A_path = os.path.join(opt.dataroot, 'test', 'clean'),
            generated_A_path = fake_result_path
        )
        test_clean_epoch_results = {
            'Epoch': epoch,
            'MI': MI,
            'CSSIM': CSSIM,
            'SSIM': SSIM,
            'PSNR': PSNR,
            'FID': FID,
        }
        test_clean_epoch_results = pd.DataFrame([test_clean_epoch_results])
        test_clean_results = pd.concat([test_clean_results, test_clean_epoch_results], ignore_index=True)
        test_clean_results.to_excel(os.path.join(opt.results_dir, opt.name, 'clean2noisy', 'clean2noisy_results.xlsx'), index=False, engine='openpyxl')
        
        print("{} finished ...".format(epoch))  
