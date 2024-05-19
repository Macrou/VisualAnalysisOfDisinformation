import os
import pandas as pd
import tarfile
from torch.utils.data import Dataset
from torchvision.io import read_image
import io


class Fakeddit(Dataset):

    def __init__(self,annotations_file, img_dir="dataset/public_images.tar.bz2", transform=None, target_transform=None):
        self.id = pd.read_csv(annotations_file,sep='\t',index_col= 'id').index.values
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file,sep='\t',index_col= '2_way_label').index.values
        self.transform = transform
        self.target_transform = target_transform
        self.tf = tarfile.open(self.img_dir,'r:bz2')

    def get_image_from_tar(self, name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        image = self.tf.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        return image

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        print(self.img_labels[0])
        if idx == (self.__len__() - 1) :  # close tarfile opened in __init__
            self.tf.close()
        image = self.get_image_from_tar(self.id[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
