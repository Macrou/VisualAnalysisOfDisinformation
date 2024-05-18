import os
import pandas as pd
import tarfile
from torch.utils.data import Dataset
from torchvision.io import read_image
import io


class Fakeddit(Dataset):

    def __init__(self,annotations_file, img_dir="dataset/public_images.tar.bz2", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file,sep='\t')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_dir_without_tar = self.img_dir[:-8]
        img_path = os.path.join(img_dir_without_tar, self.img_labels.iloc[idx, 5])
        with tarfile.open(self.img_dir, 'r:bz2') as tar:
            try:
                img_file = tar.extractfile(img_path)
                if img_file is None:
                    raise FileNotFoundError(f"{img_path} not found in the tar archive.")
                img_data = img_file.read()
                image = read_image(io.BytesIO(img_data))
            except KeyError:
                raise FileNotFoundError(f"{img_path} not found in the tar archive.")
        label = self.img_labels.iloc[idx, 13]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
