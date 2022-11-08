import os
import torch
import cv2 # SİLİNECEK
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image

# Tuğçe Hoca Tez Damar Veri Seti

class CustomImageDataset(Dataset):
    def __init__(self, mode, img_dir, lbl_dir, transform=None, target_transform=None, de_train = False):
        self.mode = mode
        self.images = pd.read_csv(os.path.join(img_dir, f"{mode}_images.txt"))
        self.labels = pd.read_csv(os.path.join(lbl_dir, f"{mode}_images.txt")) 
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.de_train = de_train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.de_train == False:
            return len(self.images)
        else:
            return len(self.images) // 2

    def __getitem__(self, idx):

        # Read Image
        image = read_image(os.path.join(self.img_dir, self.mode, self.images.iloc[idx]['Name']))[0,:,:].reshape((1, 512, 512))

        # Read Label
        lbl_name = os.path.join(self.lbl_dir, self.mode, self.labels.iloc[idx]['Name'])
        label = read_image(lbl_name)[0,:,:].reshape((1, 512, 512))
    	
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image / 255, label / 255 # Min-max Normalization
