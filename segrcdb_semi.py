from dataset.transform import *
from copy import deepcopy
import numpy as np
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()
        if mode == 'train_l' and nsample is not None:
             self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        id = self.ids[item]
        
        img_path = os.path.join(self.root, id.split(' ')[0])
        img = Image.open(img_path).convert('RGB')
        
        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        mask = Image.fromarray(mask)

        # Validation
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        # --- Geometric Augmentation ---
        ignore_value = 255 
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        # --- Strong Augmentation ---
        img_s1, img_s2 = deepcopy(img), deepcopy(img)

        # Strong Aug 1
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.9)

        # Strong Aug 2
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.9)

        img_s1 = normalize(img_s1)
        img_s2 = normalize(img_s2)
        
        mask = torch.from_numpy(np.array(mask)).long()

        return img_s1, img_s2, mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)