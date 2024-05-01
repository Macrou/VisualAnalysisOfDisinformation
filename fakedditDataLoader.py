import os
import pandas as pd
import tarfile
from torch.utils.data import Dataset
from torchvision.io import read_image
import io


class Fakeddit(Dataset):
    def __init__(self,annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file,sep='\t')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        with tarfile.open(self.img_dir,'r:bz2') as tar:
            img_dir_without_tar = tarfile.os.path.splitext(self.img_dir)[0]
            img_path = os.path.join(img_dir_without_tar, self.img_labels.iloc[idx, 5])
            img_data = tar.extractfile(img_path).read()
            image = read_image(io.BytesIO(img_data))
        label = self.img_labels.iloc[idx, 13]
        if self.transform:
            image = self.transform(image)            
        if self.target_transform:
            label = self.target_transform(label)
        return image,label 