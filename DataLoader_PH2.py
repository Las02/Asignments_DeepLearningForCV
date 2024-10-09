import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time


import matplotlib.pyplot as plt
from IPython.display import clear_output
data_path = '/dtu/datasets1/02516/PH2_Dataset_images/'
class PH2(torch.utils.data.Dataset):
    def __init__(self, transform, data_path=data_path):
        'Initialization'
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        for folder in os.listdir(data_path):
            self.image_paths.append(data_path + f"{folder}/{folder}_Dermoscopic_Image/{folder}.bmp")
            self.label_paths.append(data_path + f"{folder}/{folder}_lesion/{folder}_lesion.bmp")
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
    
size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

Full_set = PH2(transform=train_transform)

train_val_set, test_set =  torch.utils.data.random_split(Full_set, [180, 20])
train_set, val_set = torch.utils.data.random_split(train_val_set, [150, 30])

test_loader = DataLoader(test_set,batch_size= 6 , shuffle=False, num_workers=3)
train_loader = DataLoader(train_set,batch_size= 20, shuffle=False, num_workers=3)
val_loader = DataLoader(val_set, shuffle=False, num_workers=3)
