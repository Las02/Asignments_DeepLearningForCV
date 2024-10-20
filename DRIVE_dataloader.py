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

# Define transformations
transform = transforms.Compose([
   # transforms.Resize((510, 510)),  #Image_size
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

#Patching the dataset 
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, original_loader, patch_size=110, stride=110):
        self.original_loader = original_loader
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        # Returner det samlede antal patches, der kan genereres
        total_patches = 0
        for images, masks in self.original_loader:
            for i in range(images.size(0)):
                image = images[i]  # Hent individuelt billede
                H, W = image.size(1), image.size(2)
                total_patches += ((H - self.patch_size) // self.stride + 1) * ((W - self.patch_size) // self.stride + 1)
        return total_patches

    def __getitem__(self, idx):
        # Find det passende billede og maske baseret på idx
        for images, masks in self.original_loader:
            for i in range(images.size(0)):
                image = images[i]
                mask = masks[i]
                H, W = image.size(1), image.size(2)

                # Generer patches
                for y in range(0, H - self.patch_size + 1, self.stride):
                    for x in range(0, W - self.patch_size + 1, self.stride):
                        if idx == 0:
                            return image[:, y:y + self.patch_size, x:x + self.patch_size], mask[:, y:y + self.patch_size, x:x + self.patch_size]
                        idx -= 1
        raise IndexError("Index out of range in PatchDataset")




# Opret PatchDatasets fra eksisterende dataloaders
train_patch_dataset = PatchDataset(train_loader, patch_size=110, stride=110)
val_patch_dataset = PatchDataset(val_loader, patch_size=110, stride=110)
test_patch_dataset = PatchDataset(test_loader, patch_size=110, stride=110)

# Opret nye dataloaders til patches
train_patch_loader = torch.utils.data.DataLoader(train_patch_dataset, batch_size=128, shuffle=True) # before 12 
val_patch_loader = torch.utils.data.DataLoader(val_patch_dataset, batch_size=128, shuffle=False)
test_patch_loader = torch.utils.data.DataLoader(test_patch_dataset, batch_size=128, shuffle=False)


#PLotting images and masks to see everything looks fine 
import matplotlib.pyplot as plt
import numpy as np 
plt.rcParams['figure.figsize'] = [18, 6]

images, labels = next(iter(train_loader))

for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))
    plt.axis('off')  


    plt.subplot(2, 6, i+7)
    plt.imshow(labels[i].squeeze())
    plt.axis('off')  
    plt.tight_layout()

plt.show()



#PLotting images and masks to see everything looks fine - now each 550x550 image is cropped to 25 x 110x110
plt.rcParams['figure.figsize'] = [18, 6]

images, labels = next(iter(train_patch_loader))

for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))
    plt.axis('off')  


    plt.subplot(2, 6, i+7)
    plt.imshow(labels[i].squeeze())
    plt.axis('off')  
    plt.tight_layout()

plt.show()