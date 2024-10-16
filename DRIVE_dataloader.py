#USE THIS 

import os
import glob
from PIL import Image
import torch
from torchvision import transforms

class DRIVE_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, data_path='/dtu/datasets1/02516/DRIVE'):
        'Initialization'
        self.transform = transform

        # Find Images and mask paths:))
        self.image_dir = os.path.join(data_path, 'training/images')
        self.mask_dir = os.path.join(data_path, 'training/1st_manual')
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*_manual1.gif')))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(image_path).convert("RGB")  # Konverter til RGB for konsistens
        mask = Image.open(mask_path).convert("L")      # Konverter maske til grayscale (1 kanal)

        # Anvend transformationer
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transformations, deleted: #Image_sizetransforms.Resize((500, 500)),
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert til tensor
])

# Create dataloadere
dataset = DRIVE_Dataset(transform=transform)

# Using the 60% 20% 20% 
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size  # Rest to da test 

print(total_size)
print(train_size, val_size, test_size)

# Dividing the sets into 3 parts 
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# create dataloader for train, val and test 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False)
