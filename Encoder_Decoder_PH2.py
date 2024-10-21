#Model  for input 110x110 
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

import sys
import os
sys.path.append('/zhome/45/0/155089/Deeplearning_in_computer_vision/Segmentation_project/Asignments_DeepLearningForCV/')  
from Performance_Metrics import dice_coefficient, intersection_over_union, accuracy, sensitivity, specificity
#import dataset DRIVE 
from DataLoader_PH2 import train_loader , val_loader, test_loader
import time 
from time import time  # Correct import


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




class EncDec(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn_enc0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64              
        
        self.enc_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_enc1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32                         
        
        self.enc_conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_enc2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16                             
        
        self.enc_conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_enc3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8                               

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 512, 3, padding=1)

        # decoder (upsampling) with ConvTranspose2d
        self.upsample0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_dec0 = nn.BatchNorm2d(256)
        
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_dec1 = nn.BatchNorm2d(128)
        
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_dec2 = nn.BatchNorm2d(64)
        
        self.upsample3 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)     # 64 -> 128
        self.dec_conv3 = nn.Conv2d(1, 1, 3, padding=1)  # output skal være 1 kanal for segmentation 

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.bn_enc0(self.enc_conv0(x))))
        e1 = self.pool1(F.relu(self.bn_enc1(self.enc_conv1(e0))))
        e2 = self.pool2(F.relu(self.bn_enc2(self.enc_conv2(e1))))
        e3 = self.pool3(F.relu(self.bn_enc3(self.enc_conv3(e2))))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.bn_dec0(self.dec_conv0(self.upsample0(b))))
        d1 = F.relu(self.bn_dec1(self.dec_conv1(self.upsample1(d0))))
        d2 = F.relu(self.bn_dec2(self.dec_conv2(self.upsample2(d1))))
        
        # Final decoding layer - no batchnorm
        d3 = self.dec_conv3(self.upsample3(d2))  # final output

        return d3


model = EncDec().to(device)
summary(model, input_size=(3, 128,128))