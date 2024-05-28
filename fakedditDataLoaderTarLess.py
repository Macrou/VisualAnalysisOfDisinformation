import os
import io
import pandas as pd
import tarfile
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image


class Fakeddit(Dataset):

    def __init__(self,annotations_file, img_dir="dataset/public_images.tar.bz2", transform=None, target_transform=None):
        anotations = pd.read_csv(annotations_file,sep='\t')
        self.id = anotations['id'].values
        self.img_dir = img_dir
        self.img_labels = anotations['2_way_label'].values
        self.transform = transform
        self.target_transform = target_transform
 
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        name = name + '.jpg'
        path = os.path.join('public_image_set',name)
        image  = read_image(path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
