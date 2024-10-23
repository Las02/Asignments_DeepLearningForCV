
def joint_transform(image, label):
    """
    Applies the same transformation to both image and label
    """
    # Define the desired output size
    output_size = 128
    
    # Resize both image and label
    resize = transforms.Resize((output_size, output_size))
    image = resize(image)
    label = resize(label)
    
    # Define parameters for RandomResizedCrop
    crop_size = (128, 128)  # Desired output size after resizing
    scale = (0.2, 1.0)      # Scale range for the area of the crop
    ratio = (0.9, 1.1)      # Aspect ratio range for the crop

    # Obtain random parameters for cropping
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=scale, ratio=ratio)
    
    # Apply the same crop to both image and label
    image = TF.resized_crop(image, i, j, h, w, size=crop_size, interpolation=Image.BILINEAR)
    label = TF.resized_crop(label, i, j, h, w, size=crop_size, interpolation=Image.NEAREST)
    
    # Optional: If you want to perform random cropping, ensure the resized size is larger
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)
    
    # Random vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)
    
    # Convert to tensor
    image = TF.to_tensor(image)
    label = TF.to_tensor(label)
    
    return image, label

###### DATA LOADER
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
        
        # Open the image and label
        image = Image.open(image_path).convert("RGB")   # Ensure image is in RGB
        label = Image.open(label_path).convert("L")     # Ensure mask is in grayscale
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label
    
Full_set = PH2(transform=joint_transform)
