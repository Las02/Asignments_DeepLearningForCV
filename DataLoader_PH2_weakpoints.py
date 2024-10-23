from DataLoader_PH2 import PH2
import torch 
import PIL.Image as Image


class PH2_weekpoints(PH2):
    
    def __init__(self, *args, **kwargs):
        super(PH2, self).__init__(*args, **kwargs)
    
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
        preserve_ones = torch.randperm(len(ones_indices[0]))[:10]
        preserve_zeros = torch.randperm(len(zeros_indices[0]))[:10]

        # Create a mask to set everything to 0.5 except the preserved ones and zeros
        Y = torch.ones_like(Y) * 0.5

        # Set the preserved indices back to their original values
        Y[ones_indices[0][preserve_ones], ones_indices[1][preserve_ones], ones_indices[2][preserve_ones]] = 1
        Y[zeros_indices[0][preserve_zeros], zeros_indices[1][preserve_zeros], zeros_indices[2][preserve_zeros]] = 0

        
        return X, Y
    