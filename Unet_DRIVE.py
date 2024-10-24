import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool0 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)  # 110 -> 55
        self.enc_conv1 = nn.Conv2d(64, 128,kernel_size= 3, padding=1,)
        self.pool1 = nn.Conv2d(128, 128,kernel_size=3, stride=2,padding=1)  # 55 -> 27
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.Conv2d(256, 256,kernel_size=3, stride=2,padding=1)  # 27 -> 13
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.Conv2d(512, 512,kernel_size=3, stride=2,padding=1)  # 13 -> 6

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 1024, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024,1024,2,stride = 2, padding = 0)# 6 -> 12
        self.upsample0match = nn.Upsample(13)  # 12 -> 13  
        
        self.upsample1 = nn.ConvTranspose2d(512,512,2,stride = 2, padding = 0) # 13 -> 26
        self.upsample1match = nn.Upsample(27)  # 26 -> 27  
        
        self.upsample2 = nn.ConvTranspose2d(256,256,2,stride = 2, padding = 0)  # 27 -> 54
        self.upsample2match = nn.Upsample(55)  # 54 -> 55  
        
        
        self.upsample3 = nn.ConvTranspose2d(128,128,2,stride = 2, padding = 0) # 55 -> 110 

        self.dec_conv0 = nn.Conv2d(1024 + 512, 512, 3, padding=1)
        self.dec_conv1 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(128 + 64, 64, 3, padding=1)

        # final output layer
        self.final_conv = nn.Conv2d(64, 1, 1)  # 1x1 convolution for binary segmentation

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(self.pool0(e0)))
        e2 = F.relu(self.enc_conv2(self.pool1(e1)))
        e3 = F.relu(self.enc_conv3(self.pool2(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0match(self.upsample0(b)), e3], dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1match(self.upsample1(d0)), e2], dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2match(self.upsample2(d1)), e1], dim=1)))
        d3 = F.relu(self.dec_conv3(torch.cat([self.upsample3(d2), e0], dim=1)))


        # final output layer (logits)
        output = self.final_conv(d3)

        return output


model = UNet2().to(device)
summary(model, input_size=(3, 128,128))




def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
