import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class ImageDataset_Train(Dataset):
    def __init__(self, img_path1, img_path2, transform):
        self.img_paths1 = glob(img_path1 + '/*.png')
        self.img_paths2 = glob(img_path2 + '/*.png')
        self.transform = transform
        
        self.img_paths1.sort()
        self.img_paths2.sort()
        
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)
    
    def __len__(self):
        return len(self.img_paths1)
    
    def __getitem__(self, idx):
        img_path1 = self.img_paths1[idx]
        img_path2 = self.img_paths2[idx]
        image1 = Image.open(img_path1).convert('L')
        image2 = Image.open(img_path2).convert('L')
        img_name = os.path.basename(img_path1)
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        if random.random() < 0.5:
            image1 = self.flip_transform(image1)
            image2 = self.flip_transform(image2)
            
        return image1, image2, img_name


class ImageDataset_Test(Dataset):
    def __init__(self, img_path, transform):
        self.img_paths = glob(img_path + '/*.png')
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('L')
        img_name = os.path.basename(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_name


def get_dataloaders(
    train_clean_path,
    train_noisy_path,
    test_noisy_path,
    args  
):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageDataset_Train(train_clean_path, train_noisy_path, transform)
    test_dataset = ImageDataset_Test(test_noisy_path, transform)
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=16
    )
    dataloaders['test'] = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=16
    )
    
    return dataloaders