import os
import pandas as pd
import tarfile
from torch.utils.data import Dataset
from torchvision.io import read_image
import io


class Fakeddit(Dataset):

    def __init__(self,annotations_file, img_dir="dataset/public_images.tar.bz2", transform=None, target_transform=None):
        anotations = pd.read_csv(annotations_file,sep='\t')
        self.id = anotations['id'].values
        self.img_dir = img_dir
        self.img_labels = anotations['2_way_label'].values
        self.transform = transform
        self.target_transform = target_transform

    def get_image_from_tar(self,tar ,name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = tar.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        return image

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        with tarfile.open(self.img_dir,'r:bz2') as tar:
            image = self.get_image_from_tar(tar,self.id[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
