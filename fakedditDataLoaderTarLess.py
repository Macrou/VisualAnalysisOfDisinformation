import os
from skimage import io, transform
import pandas as pd
import tarfile
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image,UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None


class Fakeddit(Dataset):

    def __init__(self,annotations_file, img_dir="dataset/public_image_set", transform=None, target_transform=None):
        anotations = pd.read_csv(annotations_file,sep='\t')
        self.id = anotations['id'].values
        self.img_dir = img_dir
        self.img_labels = anotations['2_way_label'].values
        self.transform = transform
        self.target_transform = target_transform
 
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        name = self.id[idx] + '.jpg'
        path = os.path.join(self.img_dir,name)
        try:
            image  = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError, IOError) as e: 
            print(f"Error loading image {path}: {e}")
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))  # Adjust dimensions as needed
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
