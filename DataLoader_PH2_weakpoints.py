from DataLoader_PH2 import PH2
import torch 
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os 



data_path = '/dtu/datasets1/02516/PH2_Dataset_images/'
class PH2_weekpoints(torch.utils.data.Dataset):
    
    def __init__(self, transform, number_of_points, data_path=data_path):
        'Initialization'
        self.number_of_points = number_of_points
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
        
        ones_indices = (Y == 1).nonzero(as_tuple=True)
        zeros_indices = (Y == 0).nonzero(as_tuple=True)

        # Randomly select 10 indices from 1s and 10 from 0s to preserve
        preserve_ones = torch.randperm(len(ones_indices[0]))[:self.number_of_points]
        preserve_zeros = torch.randperm(len(zeros_indices[0]))[:self.number_of_points]

        # Create a mask to set everything to 0.5 except the preserved ones and zeros
        Y = torch.ones_like(Y) * 0.5

        # Set the preserved indices back to their original values
        Y[ones_indices[0][preserve_ones], ones_indices[1][preserve_ones], ones_indices[2][preserve_ones]] = 1
        Y[zeros_indices[0][preserve_zeros], zeros_indices[1][preserve_zeros], zeros_indices[2][preserve_zeros]] = 0

        
        return X, Y



