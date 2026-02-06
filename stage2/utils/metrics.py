import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import mutual_info_score as mi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score


def cssim(image1, image2):
    image1 = np.array(image1, dtype=np.float64)
    image2 = np.array(image2, dtype=np.float64)
    
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    covariance = np.cov(image1.ravel(), image2.ravel())[0, 1]
    C2 = (0.03)**2  # 0.03 is a small constant used in SSIM
    
    cssim_value = (2 * covariance + C2) / (sigma1_sq + sigma2_sq + C2)
    
    return cssim_value


def calculate_metrics(
    train_clean_path,
    test_noisy_path,
    fake_clean_path
):
    
    mi_values = []  
    cssim_values = []  
    ssim_values = []
    psnr_values = []
    
    for filename in os.listdir(fake_clean_path):  
        if filename.endswith('.png'): 
            image1_path = os.path.join(fake_clean_path, filename)  
            image2_path = os.path.join(test_noisy_path, filename)  

            if os.path.exists(image2_path):  
                image1 = Image.open(image1_path).convert('L')  
                image2 = Image.open(image2_path).convert('L')  

                hist1, _ = np.histogram(np.array(image1).flatten(), bins=256, range=(0, 256))  
                hist2, _ = np.histogram(np.array(image2).flatten(), bins=256, range=(0, 256))  
                mutual_info_value = mi(np.array(image1).flatten(), np.array(image2).flatten())  
                mi_values.append(mutual_info_value)  
                
                image1 = np.array(image1)
                image2 = np.array(image2)
                image1_min, image1_max = image1.min(), image1.max()  
                image1 = (image1 - image1_min) / (image1_max - image1_min)  
                image2_min, image2_max = image2.min(), image2.max()  
                image2 = (image2 - image2_min) / (image2_max - image2_min)  

                cssim_value = cssim(image1, image2)  
                cssim_values.append(cssim_value)
                
                ssim_value = ssim(image1, image2, data_range=1)
                ssim_values.append(ssim_value)
                psnr_value = psnr(image1, image2)
                psnr_values.append(psnr_value)
                

    average_mi = np.mean(mi_values) if mi_values else None  
    average_cssim = np.mean(cssim_values) if cssim_values else None 
    average_ssim = np.mean(ssim_values) if ssim_values else None 
    average_psnr = np.mean(psnr_values) if psnr_values else None 
    
    FID = fid_score.calculate_fid_given_paths([train_clean_path, fake_clean_path],
                                                    batch_size=64,
                                                    num_workers=0,
                                                    device='cuda', 
                                                    dims=2048)
    
    return average_mi, average_cssim, average_ssim, average_psnr, FID

def calculate_train_metrics(batch1, batch2):
    if batch1.shape != batch2.shape:
        raise ValueError("Input batches must have the same shape.")
    
    batch_size = batch1.shape[0]
    MI = []
    CSSIM = []
    SSIM = []
    PSNR = []
    
    for i in range(batch_size):
        image1 = batch1[i].detach().cpu().numpy()
        image2 = batch2[i].detach().cpu().numpy()
        image1 = image1.squeeze()
        image2 = image2.squeeze()
        
        image1_min, image1_max = image1.min(), image1.max()  
        image1 = (image1 - image1_min) / (image1_max - image1_min)  
        image2_min, image2_max = image2.min(), image2.max()  
        image2 = (image2 - image2_min) / (image2_max - image2_min)  

        cssim_value = cssim(image1, image2)
        ssim_value = ssim(image1, image2, data_range=1.0)
        psnr_value = psnr(image1, image2)
        
        image1 = (image1 * 255).astype(np.uint8)
        image2 = (image2 * 255).astype(np.uint8)
        hist1, _ = np.histogram(np.array(image1).flatten(), bins=256, range=(0, 256))  
        hist2, _ = np.histogram(np.array(image2).flatten(), bins=256, range=(0, 256))  
        mi_value = mi(np.array(image1).flatten(), np.array(image2).flatten())  
        
        MI.append(mi_value)
        CSSIM.append(cssim_value)
        SSIM.append(ssim_value)
        PSNR.append(psnr_value)
        
    return np.mean(MI), np.mean(CSSIM), np.mean(SSIM), np.mean(PSNR)