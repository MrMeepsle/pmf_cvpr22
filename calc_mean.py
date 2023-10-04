# collapse-hide

####### PACKAGES

import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        for root, d_names, f_names in os.walk(data_dir):
            print(root)
            for f in f_names:
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.data.append(os.path.join(root, f))
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx], cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


####### PARAMS

device = torch.device('cpu')
num_workers = 8
image_size = 512
batch_size = 8
data_path = './data/PMF_dataset'

# collapse-show

augs = A.Compose([A.Resize(height=image_size,
                           width=image_size),
                  A.Normalize(mean=(0, 0, 0),
                              std=(1, 1, 1)),
                  ToTensorV2()])

# dataset
image_dataset = CustomDataset(data_dir=data_path,
                              transform=augs)

# data loader
image_loader = DataLoader(image_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

####### COMPUTE MEAN / STD

# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])
count = 0

# loop through images
print("starting loop")
for inputs in tqdm(image_loader):
    psum += inputs.sum(axis=[0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

####### FINAL CALCULATIONS

# pixel count
count = len(image_loader.dataset) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))
