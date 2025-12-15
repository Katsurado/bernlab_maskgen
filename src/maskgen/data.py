import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
import torch.nn.functional as F

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .utils import list_dir

class ImageData(Dataset):
    @staticmethod
    def create_transforms(img_size:int = 512):
        '''
        Create transforms for images
        '''
        geom = T.Compose([
            T.RandomCrop(img_size, pad_if_needed=True)
        ])

        norm = T.Compose([
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return geom, norm
    
    def __init__(self, root:str, partiton:str, transform:bool, config:dict) -> None:
        '''
        Custom dataset class for image and mask data

        Args:
        root(str): root dir of data
        partition(str): train/val/test data
        transform(bool): whether to perform transform
        config(dict): config dict
        '''
        self.crops_per_img = config['crop_per_img']
        self.geom, self.norm = self.create_transforms(config["img_size"])
        self.transform = transform
        self.img_path = '/'.join([root, partiton, 'images'])
        self.mask_path = '/'.join([root, partiton, 'masks'])

        self.images = []
        self.masks = []

        img_names = sorted(list_dir(self.img_path))
        mask_names = sorted(list_dir(self.mask_path))

        assert(len(img_names) == len(mask_names))
        
        self.num_raw_img = len(img_names)

        for i in tqdm(range(self.num_raw_img)):
            img_pth = '/'.join([self.img_path, img_names[i]])
            mask_pth = '/'.join([self.mask_path, mask_names[i]])

            image = tv_tensors.Image(torchvision.io.read_image(img_pth))
            mask = tv_tensors.Mask(torchvision.io.read_image(mask_pth))

            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return self.crops_per_img * self.num_raw_img
    
    def __getitem__(self, index):
        raw_idx = index // self.crops_per_img

        img = self.images[raw_idx][:3]
        mask = self.masks[raw_idx]

        if self.transform:
            out = self.geom({'image': img, 'mask': mask})
            img_crop, mask_crop = out["image"], out["mask"]
        else:
            img_crop, mask_crop = img, mask

        img_crop = T.ToDtype(torch.float32, scale=True)(img_crop)
        img_crop = self.norm(img_crop)

        # i do not fully understand how the lines below work but the appear to work

        # mask: RGBA/RGB -> 1ch binary (white dots = 1)
        mc = (mask_crop[:3] if mask_crop.ndim == 3 else mask_crop).amax(dim=0, keepdim=True)  # [1,H,W]
        mask_crop = (mc > 127).to(torch.float32)  # swap to <128 if your foreground is dark


        return img_crop, mask_crop     
